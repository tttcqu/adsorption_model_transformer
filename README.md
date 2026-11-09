# Gas adsorption isotherm prediction

This package contains the curated NIST adsorption data, the best adsorption model, MOF and COF crystal structures, and a runnable prediction demo.

## Files

- `data/nist_adsorption.csv` — 10,373 processed experimental adsorption records from the NIST/ARPA-E database, with source references, measurement conditions and material descriptors.
- `model/best_model.pt` — model weights, feature normalization parameters and architecture settings.
- `structures/MOF/` — 9,835 MOF CIF files.
- `structures/COF/` — all 517 source COF CIF files. 
- `data/structure_manifest.csv` — material identifiers, CIF paths and availability for the demo.
- `data/material_descriptors.csv` — precomputed inputs for 9,835 MOFs and 510 COFs.
- `data/gas_descriptors.csv` — gas identifiers and gas descriptors.
- `data/feature_groups.json` — input feature names and ordering.
- `demo.py`, `inference.py`, `model.py`, `components.py` — prediction code.

## Run the demo

Use Python 3.10 or newer. From this directory, run:

```bash
python -m pip install -r requirements.txt
python demo.py
```

The demo predicts an H2 isotherm at 298 K for JUC-551. Results are saved to `outputs/isotherm.csv`. To use another material or gas:

```bash
python demo.py --material "cof:00229_JUC-551" --gas CO2 --temperature 298 --pressures 0.1 0.5 1 5 10 --output outputs/co2_isotherm.csv
```

Use the exact material identifier in `data/structure_manifest.csv` and a gas key in `data/gas_descriptors.csv`. Temperature is in K, pressure arguments are absolute pressure in bar, and predicted uptake is in mol/kg. `selected_model` reports the base isotherm selected by the model.
