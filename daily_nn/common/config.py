#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Central path configuration for the daily_nn pipeline.

All stage scripts resolve paths through this module instead of relying on
relative paths from the current working directory.
"""
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PIPELINE_DIR.parent

# ---- shared input data ------------------------------------------------
DATA_DIR = Path("/media/francesco/T7/Personale/NeuralNetworks_BACKUP/data") #REPO_ROOT / "data"
LOCAL_DATA_DAILY_DIR = DATA_DIR / "local_data_daily"
RAW_GLOBAL_DATA_DIR = DATA_DIR / "raw_global_data" / "lead_30_presaved"
ECMWF_BENCHMARK_DIR = DATA_DIR / "ECMWF_benchmark"

LOCAL_VARIABLES = ['MSSHF', 'SH', 't2m', 'TCC', 'tp', 'UW', 'VW']
GLOBAL_VARIABLES = ['MSLP', 'Z500']

# ---- pipeline outputs ---------------------------------------------------
RESULTS_DIR = PIPELINE_DIR / "results"

GENERATED_DATASETS_DIR = RESULTS_DIR / "generated_datasets"
TUNED_DATASETS_DIR = RESULTS_DIR / "tuned_datasets"
TUNER_TRIALS_DIR = RESULTS_DIR / "tuner_trials" / "best_hyperparams"
SUMMARY_CSV = RESULTS_DIR / "summary.csv"

BEST_DATA_DIR = RESULTS_DIR / "best_data"
BEST_PREDICTIONS_DIR = RESULTS_DIR / "best_predictions"
PREDICTION_VS_TRUTH_DIR = RESULTS_DIR / "predictionVStruth_datasets"
TARGETS_DIR = RESULTS_DIR / "targets"
PLOTS_DIR = RESULTS_DIR / "plots"

# Names of the best-performing feature combinations selected by stage 02,
# used by the downstream stages that load that specific model/dataset.
BEST_DATASET_NAME_NN = 'MSLP-Z500-MSSHF-SH-t2m-TCC-tp-UW-VW'
BEST_DATASET_NAME_TREND = 'MSSHF-SH-t2m-TCC-tp-UW-VW-MSLP-Z500'

# Cross-pipeline trend-comparison data, shared with monthly_elm.
TOT_TREND_DIR = REPO_ROOT / "results" / "tot_trend_comparison"
