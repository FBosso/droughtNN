# droughtNN

Machine-learning pipelines for subseasonal/monthly drought (precipitation)
forecasting, combining local ERA5 variables with PCA-reduced global climate
fields (MSLP, SST, Z500) and teleconnection signals (EA, ENSO, NAO, SCA).

## Pipelines

- **`monthly_elm/`** - Extreme Learning Machines (skELM) for monthly
  precipitation forecasting, with Leave-One-Out cross-validation for joint
  model selection and hyperparameter tuning. See `monthly_elm/README.md`.
- **`daily_nn/`** - Feed-forward neural network (Keras) on a 30-day moving
  window of daily variables, plus a linear-regression baseline. See
  `daily_nn/README.md`.

Both pipelines are organized as a sequence of numbered stage scripts
(`01_...py`, `02_...py`, ...) that share a `common/` package (path
configuration in `common/config.py`, plus dataset/model/plotting helpers).
Stage scripts resolve all paths through `common/config.py`, so they should be
run from the repository root, e.g.:

```bash
uv run python monthly_elm/01_generate_datasets_and_tune.py
uv run python daily_nn/01_generate_datasets.py
```

## Installation

CPU-only:
```bash
uv sync
```

With NVIDIA GPU (requires CUDA 12 drivers):
```bash
uv sync --extra gpu
```

The `[gpu]` extra installs `nvidia-cudnn-cu12`, the only library missing from a
standard CUDA 12 toolkit installation. TensorFlow picks it up automatically from
the virtual environment — no system-level cuDNN installation needed.

## Data layout

`data/` (gitignored) holds the input datasets shared by both pipelines:

- `local_data/`, `local_data_daily/` - per-variable local ERA5 timeseries
  (monthly and daily resolution).
- `global_data/`, `raw_global_data/` - gridded global fields (MSLP, SST,
  Z500) and precomputed teleconnection-signal combinations.
- `climate_signals/` - EA, ENSO, NAO, SCA index timeseries.
- `ECMWF_benchmark/` - ECMWF subseasonal forecast benchmark used for
  comparison in both pipelines.

`monthly_elm/01_generate_datasets_and_tune.py` also reads a sibling project's
raw ERA5 archive via `DROUGHTNN_EXTERNAL_GLOBAL_DATA` (see
`monthly_elm/common/config.py` for the default path and override).

## Outputs

Each pipeline writes its generated datasets, tuned models, predictions and
plots to its own `results/` directory (`monthly_elm/results/`,
`daily_nn/results/`), which is gitignored for new runs. Cross-pipeline trend
comparison data is written to the top-level `results/tot_trend_comparison/`.

## Other directories

- `docs/` - non-code, non-generated material: manuscript figures
  (`docs/figures/`), the model-comparison spreadsheet
  (`docs/model_comparison.xlsx`), and archived legacy test tables
  (`docs/legacy_test_tables/`).

## License

This project is licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE).
You are free to use, modify, and redistribute this code for any purpose, including commercially, as long as you give appropriate credit to the original work.

## Citation

If you use this code in your research, please cite it using the metadata in [`CITATION.cff`](CITATION.cff), or use the "Cite this repository" button on GitHub.

```bibtex
@software{bosso2026droughtnn,
  author    = {Bosso, Francesco},
  title     = {droughtNN: Subseasonal drought forecasting with Extreme Learning Machines},
  year      = {2026},
  url       = {https://github.com/FBosso/droughtNN},
  license   = {CC-BY-4.0}
}
```
