# Daily feed-forward NN for subseasonal drought forecasting

## Brief description

This pipeline builds a feed-forward neural network (FFNN) that predicts
monthly-cumulative precipitation from a 30-day moving window of daily local
variables plus PCA-reduced global fields (MSLP, Z500). A linear-regression
baseline is also trained on the same data for comparison.

## Pipeline stages

1. `01_generate_datasets.py` - build the local + global (PCA-reduced) feature
   dataset for the full variable combination and split it into train/test sets.
2. `02_tune_hyperparameters.py` - run a Keras Hyperband search over a
   2-hidden-layer FFNN for each generated dataset and save the tuned model.
3. `03_generate_best_models_summary.py` - rank the tuned models by final
   validation loss.
4. `04_extract_best_predictions.py` - copy the best tuned dataset/model into
   `results/best_data/` and save train/test prediction-vs-truth tables.
5. `05_final_test.py` - evaluate the best model globally and month-by-month,
   saving summary plots.
6. `06_trend_comparison.py` - compare the best FFNN's predictions against the
   ECMWF subseasonal benchmark for every month and produce yearly trend data
   (shared with `monthly_elm` via `results/tot_trend_comparison/`).

`linear_baseline.py` trains/evaluates the linear-regression baseline on the
same data used by stage 6, for comparison against the FFNN.

## Layout

- `common/` - shared path configuration (`config.py`), dataset/PCA helpers
  (`datasets.py`), combo-string utilities (`combo_utils.py`), normalization
  helpers (`normalization.py`) and plotting helpers (`plotting.py`).
- `results/` - generated datasets, tuned models, predictions and plots
  (gitignored for new runs).
