"""Feature tokens and pressure representation for inference."""
import math
import torch
from torch import nn

class FeatureToken(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())

    def forward(self, x):
        return self.net(x)

class MechanismIsothermTransformer:

    @staticmethod
    def pressure_features(pressure_log10):
        x = pressure_log10 / 6.0
        features = [x]
        for frequency in (1.0, 2.0, 4.0):
            features.extend([torch.sin(math.pi * frequency * x), torch.cos(math.pi * frequency * x)])
        return torch.stack(features, dim=-1)
