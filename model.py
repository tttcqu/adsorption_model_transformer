"""Equation-guided isotherms with a query-independent monotone pressure head."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from components import FeatureToken, MechanismIsothermTransformer


TYPES = ["henry", "langmuir", "dual_site_langmuir", "sips_hill", "toth"]


class EquationGuidedTransformer(nn.Module):
    def __init__(self, groups, hidden=192, heads=8, layers=2, decoder_layers=3,
                 dropout=0.1, correction_limit=3.0, grid_min_log_pa=-12.0,
                 grid_max_log_pa=8.0, grid_intervals=160):
        super().__init__()
        if grid_intervals < 8 or grid_min_log_pa >= grid_max_log_pa:
            raise ValueError("Invalid fixed pressure grid")
        self.names = list(groups)
        self.k = len(TYPES)
        d = hidden
        self.tokens = nn.ModuleDict({n: FeatureToken(len(groups[n]), d, dropout)
                                     for n in self.names})
        self.token_type = nn.Parameter(torch.randn(1, len(groups), d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout,
                                         activation="gelu", batch_first=True,
                                         norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers, norm=nn.LayerNorm(d))
        self.range_token = FeatureToken(2, d, dropout)
        self.pressure_token = FeatureToken(7, d, dropout)
        self.router = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(d, self.k))
        self.param_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(),
                                        nn.Linear(d, self.k * 4))
        self.correction_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(),
                                             nn.Linear(d, 1))
        self.shape_strength = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(),
                                            nn.Linear(d, 1))
        self.capacity = nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(d, 1))
        self.max_shape_strength = 2.0
        self.register_buffer("grid_log_pa", torch.linspace(
            grid_min_log_pa, grid_max_log_pa, grid_intervals + 1))
        reference_index = round((6.0 - grid_min_log_pa) * grid_intervals /
                                (grid_max_log_pa - grid_min_log_pa))
        if reference_index <= 0 or reference_index >= grid_intervals:
            raise ValueError("10-bar reference must lie inside the fixed pressure grid")
        if abs(float(self.grid_log_pa[reference_index]) - 6.0) > 1e-5:
            raise ValueError("The fixed grid must contain the 10-bar reference exactly")
        self.reference_knot_index = reference_index

    @staticmethod
    def _positive(raw, lo=1e-4, hi=1e4):
        return torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * raw.sigmoid())

    def equation_shapes(self, pbar, raw):
        p = pbar.unsqueeze(-1)
        b = self._positive(raw[:, :, 1]).unsqueeze(1)
        z = (b * p).clamp_min(1e-12)
        henry = p.expand(-1, -1, 1)
        lang = z[:, :, 1:2] / (1 + z[:, :, 1:2])
        w = raw[:, :, 0].sigmoid().unsqueeze(1)
        b2 = self._positive(raw[:, :, 3]).unsqueeze(1)
        z1 = z[:, :, 2:3]
        z2 = (b2[:, :, 2:3] * p).clamp_min(1e-12)
        dual = w[:, :, 2:3] * z1 / (1 + z1) + (1 - w[:, :, 2:3]) * z2 / (1 + z2)
        n = (0.2 + 4.8 * raw[:, :, 2].sigmoid()).unsqueeze(1)
        zs = z[:, :, 3:4].pow(n[:, :, 3:4])
        sips = zs / (1 + zs)
        zt = z[:, :, 4:5]
        nt = n[:, :, 4:5]
        toth = zt / (1 + zt.pow(nt)).pow(1 / nt)
        return torch.cat([henry, lang, dual, sips, toth], -1)

    def _interpolate(self, knot_values, pressure_log_pa):
        grid = self.grid_log_pa
        index = torch.searchsorted(grid.contiguous(), pressure_log_pa.contiguous(), right=True)
        left = index.clamp(1, len(grid) - 1) - 1
        right = left + 1
        fraction = (pressure_log_pa - grid[left]) / (grid[right] - grid[left])
        y0 = knot_values.gather(1, left)
        y1 = knot_values.gather(1, right)
        return y0 + fraction * (y1 - y0), left

    def forward(self, b, force_type=None, mask_group=None):
        pressure = b["pressure"]
        mask = b["mask"]
        if not torch.all(torch.isfinite(pressure[mask])):
            raise ValueError("Nonfinite pressure")
        if ((pressure[mask] < self.grid_log_pa[0] - 1e-5) |
                (pressure[mask] > self.grid_log_pa[-1] + 1e-5)).any():
            raise ValueError("Pressure lies outside the model pressure grid")
        tok = torch.stack([
            self.tokens[n](torch.zeros_like(b[n]) if n == mask_group else b[n])
            for n in self.names
        ], 1)
        memory = self.encoder(tok + self.token_type)
        context = memory.mean(1)
        fixed_extent = self.grid_log_pa[[0, -1]].to(context).expand(len(context), -1)
        extent = self.range_token(fixed_extent / 6)
        joint = torch.cat([context, extent], 1)
        logits = self.router(joint)
        prob = logits.softmax(-1)
        selected = (logits.argmax(-1) if force_type is None else
                    torch.full((len(context),), force_type, device=context.device, dtype=torch.long))
        hard = F.one_hot(selected, self.k).to(prob.dtype)
        gate = hard
        raw = self.param_head(joint).view(-1, self.k, 4)

        grid = self.grid_log_pa.to(context)
        pbar = torch.pow(10.0, grid) / 1e5
        base_all = self.equation_shapes(pbar.expand(len(context), -1), raw)
        base = (base_all * gate[:, None, :]).sum(-1)
        base_inc = (base[:, 1:] - base[:, :-1]).clamp_min(1e-12)
        mid = (grid[1:] + grid[:-1]) / 2
        pressure_tokens = self.pressure_token(
            MechanismIsothermTransformer.pressure_features(mid.expand(len(context), -1)))
        correction_input = torch.cat([
            pressure_tokens, context[:, None, :].expand_as(pressure_tokens)
        ], -1)
        adjustment = self.correction_head(correction_input).squeeze(-1)
        strength = self.max_shape_strength * torch.sigmoid(self.shape_strength(joint)).squeeze(-1)
        scores = adjustment + strength[:, None] * base_inc.log()
        log_reference_mass = torch.logsumexp(
            scores[:, :self.reference_knot_index], dim=1, keepdim=True)
        knot_increments = torch.exp((scores - log_reference_mass).clamp(max=30))
        knot_shape = F.pad(knot_increments.cumsum(1), (1, 0))
        shape, interval_index = self._interpolate(knot_shape, pressure)
        shape = shape * mask
        end = F.softplus(self.capacity(context)).squeeze(-1).clamp(max=20)
        q = torch.expm1(end)[:, None] * shape
        query_adjust = adjustment.gather(1, interval_index)
        query_base, _ = self._interpolate(base, pressure)
        query_inc = (shape - F.pad(shape[:, :-1], (1, 0))) * mask
        return {
            "pred_log1p": torch.log1p(q), "shape": shape, "increments": query_inc,
            "endpoint_log1p": end, "route_logits": logits, "prob": prob,
            "selected": selected, "base_shape": query_base * mask,
            "correction_factor": torch.exp(query_adjust.clamp(max=13.8)) * mask,
            "shape_adjustment": query_adjust * mask, "shape_strength": strength,
            "raw_params": raw, "knot_increments": knot_increments,
            "knot_shape": knot_shape, "knot_adjustment": adjustment,
        }

