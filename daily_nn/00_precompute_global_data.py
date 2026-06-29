#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 0 (optional): precompute and cache processed global fields.

Run this once before 01_generate_datasets.py (or whenever raw data changes).
Saves two .nc files per global variable into RAW_GLOBAL_DATA_DIR:
  - {var}.nc          — two-step spatial template for lat/lon masking
  - {var}_adjusted.nc — anomaly field after 30-day rolling mean and lead-time shift

Uses xr.open_mfdataset + Dask chunks so the full time series is never held
in memory at once; data streams chunk-by-chunk to disk.
"""

import os
import xarray as xr

from common import config

startyr = 1979
endyr = 2021
lead = 30

presaved_dir = config.RAW_GLOBAL_DATA_DIR / f'lead_{lead}_presaved'
presaved_dir.mkdir(parents=True, exist_ok=True)

for var in config.GLOBAL_VARIABLES:
    path = str(config.RAW_GLOBAL_DATA_DIR / var)
    print(f"Processing {var}...")

    # Collect and sort files, filtered to [startyr, endyr]
    files = sorted(f for f in os.listdir(path) if not f.startswith('.'))
    file_paths = [
        f'{path}/{f}' for f in files
        if startyr <= int(f.split('-')[0]) <= endyr
    ]

    # Lazy-load all files with Dask — one month of days per chunk,
    # so only ~3.5 MB is in memory at a time instead of the full 2+ GB
    data = xr.open_mfdataset(
        file_paths, engine='netcdf4',
        combine='nested', concat_dim='time',
        chunks={'time': 31},
    )

    # Drop the last time step (matches global_timeseries_from_folder_full)
    data = data.isel(time=slice(None, -1))

    # Save original_dataset: 2 time steps is enough for the lat/lon structure
    # and shape[1]/shape[2] that filtering_conditions needs in stage 01
    data.isel(time=slice(0, 2)).load().to_netcdf(presaved_dir / f'{var}.nc')
    print(f"  Saved {presaved_dir / f'{var}.nc'}")

    # Unit conversion (lazy — no data loaded yet)
    var_name = list(data.keys())[0]  # 'msl' or 'z'
    field = data[var_name]
    if var_name == 'z':
        field = field / 9.80665
    elif var_name == 'msl':
        field = field / 100

    # Subtract temporal mean — Dask makes two streaming passes:
    # one to accumulate the spatial mean, one to subtract it
    field = field - field.mean(dim='time')

    # 30-day rolling mean; drop the first 29 incomplete windows
    rolling = field.rolling(time=30, min_periods=30).mean().isel(time=slice(29, None))

    # Lead-time shift: drop the last `lead` time steps
    n = rolling.sizes['time']
    data_shifted = rolling.isel(time=slice(0, n - lead))

    # Stream to disk chunk-by-chunk — peak memory stays at one chunk (~3.5 MB)
    data_shifted.to_dataset(name=var_name).to_netcdf(presaved_dir / f'{var}_adjusted.nc')
    print(f"  Saved {presaved_dir / f'{var}_adjusted.nc'}")

print("Done.")
