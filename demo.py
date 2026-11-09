"""Run the best adsorption model for a bundled MOF or COF."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from inference import predict_frame

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--material', default='cof:00229_JUC-551',
                        help='Material ID from data/structure_manifest.csv')
    parser.add_argument('--gas', default='H2', help='Gas key from data/gas_descriptors.csv')
    parser.add_argument('--temperature', type=float, default=298.0, help='Temperature in K')
    parser.add_argument('--pressures', type=float, nargs='+', default=[1, 5, 10, 25, 50, 75, 100],
                        help='Absolute pressures in bar')
    parser.add_argument('--output', type=Path, default=HERE / 'outputs/isotherm.csv')
    args = parser.parse_args()
    if not np.isfinite(args.temperature) or args.temperature <= 0:
        parser.error('Temperature must be finite and greater than zero')
    pressures = np.asarray(args.pressures, dtype=float)
    if not np.isfinite(pressures).all() or (pressures < 0).any():
        parser.error('Pressures must be finite and nonnegative')
    materials = pd.read_csv(HERE / 'data/material_descriptors.csv')
    gases = pd.read_csv(HERE / 'data/gas_descriptors.csv')
    material = materials.loc[materials.framework_id.eq(args.material)]
    if len(material) != 1:
        parser.error('Material has no complete descriptor record. Check structure_manifest.csv.')
    gas = gases.loc[gases.gas_key.eq(args.gas)]
    if len(gas) != 1:
        parser.error(f'Unknown gas. Available gas keys: {", ".join(gases.gas_key)}')
    torch.set_num_threads(4)
    checkpoint = torch.load(HERE / 'model/best_model.pt', map_location='cpu', weights_only=False)
    frame = pd.concat([material] * len(pressures), ignore_index=True)
    for name in checkpoint['groups']['gas']:
        frame[name] = gas.iloc[0][name]
    frame['gas_key'] = args.gas
    frame['temperature_K'] = args.temperature
    frame['pressure_Pa'] = pressures * 1e5
    predictions = predict_frame(frame, checkpoint)
    predictions['pressure_bar'] = pressures
    result = predictions[['framework_id', 'gas_key', 'temperature_K', 'pressure_bar',
                          'prediction_molkg', 'selected_model']]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f'\nSaved predictions to {args.output}')


if __name__ == '__main__':
    main()
