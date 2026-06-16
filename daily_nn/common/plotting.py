#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting/output helpers shared across the daily_nn pipeline.
"""
import numpy as np

from . import config


def save_trend_comparison(month_index, hat, true, ECMWF, true_years):
    '''
    Parameters
    ----------
    month_index : TYPE int
        DESCRIPTION. index corresponding to the month o the current iteration
    hat : TYPE array
        DESCRIPTION. predictions of the model
    true : TYPE array
        DESCRIPTION. ture target values
    ECMWF : TYPE array
        DESCRIPTION. value according to ECMWF model
    true_years : TYPE array
        DESCRIPTION. years of the true observations

    Returns
    -------
    None. It saves the data into the prefixed folder

    '''

    # month dict to translate numbers into names
    month_dict = {
        1: 'January',
        2: 'February',
        3: 'March',
        4: 'April',
        5: 'May',
        6: 'June',
        7: 'July',
        8: 'August',
        9: 'September',
        10: 'October',
        11: 'November',
        12: 'December'
    }

    month_dir = config.TOT_TREND_DIR / month_dict[month_index]
    month_dir.mkdir(parents=True, exist_ok=True)

    # save the vectors
    np.save(month_dir / 'FFNN', hat)
    np.save(month_dir / 'years', true_years)
    np.save(month_dir / 'ECMWF', ECMWF)
    np.save(month_dir / 'true', true)
