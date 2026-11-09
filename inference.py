"""Predict complete isotherms from raw descriptors using a saved checkpoint."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import EquationGuidedTransformer, TYPES

HERE = Path(__file__).resolve().parent


def predict_frame(frame, checkpoint):
    cfg = checkpoint['config']
    groups = checkpoint['groups']
    required = list(dict.fromkeys(c for cols in groups.values() for c in cols))
    required += ['framework_id', 'gas_key', 'temperature_K', 'pressure_Pa']
    missing = sorted(set(required) - set(frame))
    if missing:
        raise ValueError(f'Missing input columns: {missing}')
    numeric = list(dict.fromkeys(c for cols in groups.values() for c in cols)) + ['pressure_Pa']
    if not np.isfinite(frame[numeric].to_numpy(float)).all():
        raise ValueError('Input descriptors and pressures must be finite')
    if frame.pressure_Pa.lt(0).any():
        raise ValueError('Pressure must be nonnegative and expressed in Pa')
    model = EquationGuidedTransformer(
        groups, hidden=cfg['hidden'], heads=cfg['heads'], layers=cfg['layers'],
        decoder_layers=cfg['decoder_layers'],
        grid_min_log_pa=cfg['fixed_pressure_log10_pa'][0],
        grid_max_log_pa=cfg['fixed_pressure_log10_pa'][1],
        grid_intervals=cfg['fixed_intervals'])
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    result = frame.copy().reset_index(drop=True)
    result['prediction_molkg'] = np.nan
    result['selected_model'] = ''
    for _, curve in result.groupby(['framework_id', 'gas_key', 'temperature_K'], sort=False):
        batch = {}
        for group, cols in groups.items():
            if curve[cols].nunique().gt(1).any():
                raise ValueError('Static descriptors must be constant within an isotherm')
            mean, std = checkpoint['stats'][group]
            raw = curve.iloc[0][cols].to_numpy(dtype=np.float32)
            batch[group] = torch.tensor(((raw - mean) / std)[None], dtype=torch.float32)
        pressure = curve.pressure_Pa.to_numpy(float)
        positive = pressure > 0
        predictions = np.zeros(len(curve))
        # A zero-only query still evaluates the router at a harmless reference pressure.
        query = pressure[positive] if positive.any() else np.array([1.0])
        batch['pressure'] = torch.tensor(np.log10(query)[None], dtype=torch.float32)
        batch['mask'] = torch.ones_like(batch['pressure'], dtype=torch.bool)
        with torch.inference_mode():
            output = model(batch)
        if positive.any():
            predictions[positive] = output['pred_log1p'][0].expm1().numpy()
        result.loc[curve.index, 'prediction_molkg'] = predictions
        result.loc[curve.index, 'selected_model'] = TYPES[int(output['selected'][0])]
    if not np.isfinite(result.prediction_molkg).all():
        raise ValueError('Nonfinite prediction')
    return result

