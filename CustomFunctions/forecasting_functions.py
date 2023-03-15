#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 16:33:04 2023

@author: ncozzi
"""

# import pandas as pd
import numpy as np
from itertools import product
from .preprocessing_functions import shift_time, rolling_mean_df



##############################################################################
#               TIME MANIPULATION
##############################################################################


def rolling_where(data, abv_column_replaced, column_rolled, date, lag_list,
                    id_groups='cfips', date_column='first_day_of_month',
                    rolling_n=3):
    # rolling_mean = rolling_mean_df(data, column_rolled, date,
    #                     id_groups=id_groups, date_column=date_column,
    #                     lag=lag, rolling_n=rolling_n)
    # column_return = np.where(data[date_column]==date,
    #                          rolling_mean,
                             # data[column_to_replace])
    
    return [*zip(
        *[np.where(data.first_day_of_month==date,
                   rolling_mean_df(data, column_rolled, date,
                                       id_groups=id_groups, date_column=date_column,
                                       lag=lag, rolling_n=rolling_n),
                   data[f'{abv_column_replaced}_rolling_mean(t-{lag})'])
         for lag in lag_list]
    )]



def shift_where(data, abv_column_replaced, column_shifted, lag_list, date):
    return [*zip(
        *[np.where(data.first_day_of_month==date,
                   shift_time(data, lag)[column_shifted].astype('float32'),
                   data[f'{abv_column_replaced}(t-{lag})'])
         for lag in lag_list]
    )]

def shift_columns_where(data, lag, col_list, date):
    return [*zip(
        *[np.where(data.first_day_of_month==date,
                    shift_time(data, lag)[col].astype('float32'),
                    data[f'{col}(t-{lag})'.format])
         for col in col_list]
    )]
    
# def gwt_where(data, time_gwt, lag_list, date, abv_column_replaced=None, column_shifted=None):
    
#     return [*zip(
#         *[np.where(data.first_day_of_month==date,
#                    shift_time(data, lag)[column_shifted].astype('float32'),
#                    data[f'{abv_column_replaced}(t-{lag})'])
#          for lag in lag_list]
#     )]

##############################################################################
#               NEIGHBORS
##############################################################################


def shift_where_neighbors(data, abv_column_replaced, lag_list, neighb_list, date):
    lag_neigh_c = list(product(lag_list, neighb_list))
    return [*zip(
        *[np.where(
                data.first_day_of_month==date,
                shift_time(data, lag)[f'{abv_column_replaced}_nearest_{i}'].astype('float32'),
                data[f'{abv_column_replaced}(t-{lag})_nearest_{i}'])
         for lag, i in lag_neigh_c
        ]
    )]
