#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 16:33:04 2023

@author: ncozzi
"""

import pandas as pd
import numpy as np



##############################################################################
#               TIME MANIPULATION
##############################################################################




def shift_time(df, n):
    return pd.DataFrame(np.array([df.groupby('cfips')[col].shift(n) for col in df.columns]).transpose(),
                                  columns = df.columns)



def gwt_m(dataframe, variable, lag=0):
    def divide_if_zero(a,b):
        try:
            return a/b-1
        except ZeroDivisionError:
            return 0

    vect_div = np.vectorize(divide_if_zero)
    return vect_div(shift_time(dataframe, lag)[variable], shift_time(dataframe, lag+1)[variable])



def gwt_q(dataframe, variable, lag=0):
    def divide_if_zero(a,b):
        try:
            return a/b-1
        except ZeroDivisionError:
            return 0

    vect_div = np.vectorize(divide_if_zero)
    return vect_div(shift_time(dataframe, lag)[variable], shift_time(dataframe, lag+3)[variable])


def rolling_mean_df(data, column_rolled, date,
                    id_groups='cfips', date_column='first_day_of_month',
                    lag=1, rolling_n=5):
    def rolling_function(x): return x[column_rolled].shift(lag).rolling(rolling_n).mean()
    _rolling_mean = data.groupby(id_groups, observed=True).apply(rolling_function
                                                        ).reset_index().set_index(
                                                            'level_1').sort_index()[column_rolled]
    return _rolling_mean



##############################################################################
#               GENERATING TIME SERIES COLUMNS
##############################################################################


def gen_ts_columns(data, variable='microbusiness_density', var_abbrev = 'mbd', growth=False, lag_list=[1], max_neighbors=0,
                   mad_cutoff=50, id_field='cfips', date_field='first_day_of_month',
                   max_neigh_threshold=5):
    # asserting, checking
    assert isinstance(max_neigh_threshold, int)
    if var_abbrev is None:
        var_abbrev = variable
    assert isinstance(var_abbrev, str)
    if max_neighbors is not None: assert isinstance(max_neighbors, int)
    if max_neighbors is None: max_neighbors=0
    max_neighbors = max(0,min(max_neighbors, max_neigh_threshold))
    
    # transformations
    data['gwt_m'] = gwt_m(data, variable, lag=0)
    data['gwt_q'] = gwt_q(data, variable, lag=0)   
    if max_neighbors>=1:
        for i in range(1,max_neighbors+1):
            data[f'gwt_m_nearest_{i}']=gwt_m(data, var_abbrev+f'_nearest_{i}', lag=0)
            data[f'gwt_q_nearest_{i}']=gwt_q(data, var_abbrev+f'_nearest_{i}', lag=0)
    for lag in lag_list:
        data[var_abbrev+f'(t-{lag})'] = shift_time(data, lag)[variable]
        data[var_abbrev+f'_gwt_m(t-{lag})'] = gwt_m(data, variable, lag=lag)
        data[var_abbrev+f'_gwt_q(t-{lag})'] = gwt_q(data, variable, lag=lag)
        if max_neighbors>=1:
            for i in range(1,max_neighbors+1):
                data[var_abbrev+f'(t-{lag})_nearest_{i}']=shift_time(data, lag)[f'mbd_nearest_{i}']
                data[f'gwt_m(t-{lag})_nearest_{i}']=gwt_m(data, f'mbd_nearest_{i}', lag=lag)
                data[f'gwt_q(t-{lag})_nearest_{i}']=gwt_q(data, f'mbd_nearest_{i}', lag=lag)




##############################################################################
#               NON-TIME SERIES PREPROCESSING
##############################################################################

def dummyize_df(data, dummy=None):
    assert isinstance(dummy, str)
    if dummy is not None:
        data = pd.get_dummies(data, columns=[dummy])
    return data


def fourier_seasonality(data, date_field='first_day_of_month'):
    assert isinstance(date_field, str)
    for order in range(1,13):
        month = pd.DatetimeIndex(data[date_field]).month
        data[f'fourier_sin_order_{order}'] = np.sin(2*np.pi*order*month/12)
        data[f'fourier_cos_order_{order}'] = np.cos(2*np.pi*order*month/12)
    return data


def replace_outlier_mad(data, variable, mad_cutoff=50, id_field='cfips', date_field='first_day_of_month', warnings=True):
    from scipy.stats import median_abs_deviation
    compare_df = pd.DataFrame({'cfips': data[id_field], 'date': data[date_field], variable: data[variable]})
    median = np.median(compare_df[variable])
    MAD = median_abs_deviation(compare_df[variable])
    compare_df['x_minus_median'] = compare_df[variable] - median
    compare_df['x-m/mad'] = compare_df['x_minus_median']/MAD
    replace_max = np.repeat(median + mad_cutoff*MAD, len(compare_df))
    replace_min = np.repeat(median - mad_cutoff*MAD, len(compare_df))
    compare_df[variable] = np.where(compare_df['x-m/mad']>mad_cutoff,
                                   replace_max,
                                   compare_df[variable])
    compare_df[variable] = np.where(compare_df['x-m/mad']<-mad_cutoff,
                                   replace_min,
                                   compare_df[variable])
    # compare_df.loc[compare_df['x-m/mad']>mad_cutoff][variable] = median + mad_cutoff*MAD
    # compare_df.loc[compare_df['x-m/mad']<mad_cutoff][variable] = median - mad_cutoff*MAD
    data[variable] = compare_df[variable]
    return data

# try:
#     pd.options.mode.chained_assignment = None