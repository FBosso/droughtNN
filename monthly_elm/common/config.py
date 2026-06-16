#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Central path configuration for the monthly_elm pipeline.

All stage scripts resolve paths through this module instead of relying on
relative paths from the current working directory.
"""
import os
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PIPELINE_DIR.parent

# ---- shared input data ------------------------------------------------
DATA_DIR = REPO_ROOT / "data"
LOCAL_DATA_DIR = DATA_DIR / "local_data"
GLOBAL_DATA_DIR = DATA_DIR / "global_data"
ECMWF_BENCHMARK_DIR = DATA_DIR / "ECMWF_benchmark"

LOCAL_VARIABLES = ['MER', 'MSSHF', 'RH', 'SD', 'SH', 't2m', 'TCC', 'TCWV', 'tp', 'UW', 'VW']

# Global ERA5 data used to enumerate per-month variable combinations
# (lives outside this repo, in a sibling project).
EXTERNAL_GLOBAL_RAW_DIR = Path(os.environ.get(
    "DROUGHTNN_EXTERNAL_GLOBAL_DATA",
    "/media/francesco/fat1/projects/subseasonal/era_5/0,25x0,25_global",
))

# ---- pipeline outputs ---------------------------------------------------
RESULTS_DIR = PIPELINE_DIR / "results"

FEATURES_SCORES_DIR = RESULTS_DIR / "features_permutation_scores"
FEATURES_PREDICTIONS_DIR = RESULTS_DIR / "features_permutation_predictions"
FEATURES_MODELS_DIR = RESULTS_DIR / "features_permutation_models"

BEST_SUMMARY_CSV = RESULTS_DIR / "best_summary.csv"
PREDICTION_VS_TRUTH_DIR = RESULTS_DIR / "predictionVStruth_datasets"

PLOTS_DIR = RESULTS_DIR / "plots"
ELM_ECMWF_PLOTS_DIR = PLOTS_DIR / "ELM_ECMWF"
ECMWF_SCATTER_PLOTS_DIR = RESULTS_DIR / "ecmwf_scatter_comparison"

# Small reference dataset (ECMWF target values), checked into version control
# alongside the pipeline rather than the gitignored results/ tree.
TRUE_ECMWF_TARGETS_DIR = PIPELINE_DIR / "reference_data" / "true_ECMWF_targets"

# Cross-pipeline trend-comparison data, shared with daily_nn.
TOT_TREND_DIR = REPO_ROOT / "results" / "tot_trend_comparison"
