#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/11/7 AM11:58
# @Author  : shaoguang.csg
# @File    : mean_std.py

import numpy as np


def _var(a, axis=None, dtype=None, out=None, keepdims=np._NoValue):
    """
    A safe std helper that can not lead to NAN
    :param a:
    :param axis:
    :param dtype:
    :param out:
    :param keepdims:
    :return:
    """
    if not isinstance(a, np.ndarray):
        a = a.values
    return 0.0 if len(a) == 1 else np.var(a, axis=axis, dtype=dtype, ddof=0, out=out, keepdims=keepdims)



def cal_mean_var(df, columns=None, groupby_key=None, only_mean=False):
    """
    Cal the mean and std of df. When columns is None, all columns in df will be calculated, Otherwise, only column in
    @columns will be calcuated.
    :param df: pandas.DataFrame
    :param columns:
    :param groupby_key: a
    :return:
    """
    if columns is None:
        columns = df.columns.values
    if groupby_key is not None and groupby_key not in df.columns.values:
        raise Exception('%s not exists in df.columns' % groupby_key)

    if only_mean:
        ops = [np.mean]
    else:
        ops = [np.mean, _var]

    if groupby_key is None:
        return df[columns].astype('float64').groupby(by=lambda _:1).agg(ops)
    else:
        return df[columns].astype('float64').groupby(df[groupby_key]).agg(ops)


def apply_mean_var(df, mean_var, global_mean_var, columns=None, groupby_key=None):
    """
    apply mean and std calculated by function cal_mean_std to df.
    :param df:
    :param columns:
    :param groupby_key:
    :return:
    """
    if groupby_key is not None and groupby_key not in df.columns.values:
        raise Exception('%s not exists in df.columns' % groupby_key)
    if df is None or mean_var is None or global_mean_var is None:
        raise Exception('df or mean_var or global_mean_var is None')
    if columns is None:
        columns = df.columns.values

    if groupby_key is None:
        for column in columns:
            df[column] = (df[column] - global_mean_var[column]['mean'].values[0])/(np.sqrt(global_mean_var[column]['_var'].values[0])+1.0e-8)
    else:
        def func(_, column, groupby_key, mean_var_column):
            key = _[groupby_key].unique()[0]
            if key in mean_var_column:
                tmp = mean_var_column[key]
                return (_[column] - tmp['mean'].values[0]) / (np.sqrt(tmp['_var'].values[0]) + 1.0e-8)
            else:
                return (_[column] - global_mean_var[column]['mean'].values[0]) / (np.sqrt(global_mean_var[column]['_var'].values[0])+1.0e-8)

        for column in columns:
            mean_var_column = mean_var[column]
            mean_var_column = {index: mean_var_column[mean_var_column.index == index] for index in mean_var_column.index}
            df[column] = df[[groupby_key, column]].groupby(df[groupby_key]).apply(func, column, groupby_key, mean_var_column).sort_index(level=1).values
    return df


def reverse_mean_std(df, mean_var, global_mean_var, columns=None, use_columns=None, groupby_key=None):
    """
    The reverse process of apply_mean_std
    :param mean_var:
    :param global_mean_var:
    :param columns:
    :param use_columns:
    :param groupby_key:
    :return:
    """
    if groupby_key is not None and groupby_key not in df.columns.values:
        raise Exception('%s not exists in df.columns' % groupby_key)
    if df is None or mean_var is None or global_mean_var is None:
        raise Exception('df or mean_var or global_mean_var is None')
    if columns is not None and use_columns is None :
        raise Exception('when columns is not None, use_columns must not be None')
    if columns is None:
        columns = mean_var.columns.levels[0].values
        use_columns = mean_var.columns.levels[0].values
    if len(columns) != len(use_columns):
        raise Exception('columns and use_columns must have the same length')

    # TODO check elements in use_columns

    if groupby_key is None:
        for src_col, dst_col in zip(use_columns, columns):
            df[dst_col] = global_mean_var[src_col]['mean'].values[0]+df[dst_col]*(np.sqrt(global_mean_var[src_col]['_var'].values[0])+1e-8)
    else:
        def func(df, column, groupby_key, mean_var, global_mean_var):
            key = df[groupby_key].unique()[0]
            if key in mean_var:
                tmp = mean_var[key]
                return tmp['mean'].values[0] + df[column] * (np.sqrt(tmp['_var'].values[0]) + 1e-8)
            else:
                return global_mean_var['mean'].values[0] + df[column] * (np.sqrt(global_mean_var['_var'].values[0]) + 1e-8)

        for src_col, dst_col in zip(use_columns, columns):
            mean_var_column = mean_var[src_col]
            mean_var_column = {index: mean_var_column[mean_var_column.index == index] for index in mean_var_column.index}
            df[dst_col] = df[[groupby_key, dst_col]].groupby(df[groupby_key]).apply(func, dst_col, groupby_key, mean_var_column, global_mean_var[src_col]).sort_index(level=1).values
    return df


if __name__ == '__main__':
    import pandas as pd

    df = pd.DataFrame(np.random.randint(0, 10, size=(10,3)), columns=list('abc'))
    print("df: ", df)

    mean_var = cal_mean_var(df, columns=['b', 'c'], groupby_key='a')
    g_mean_var = cal_mean_var(df, columns=['b', 'c'])
    print("mean_var: ", mean_var.columns.levels[0].values)
    print("g_mean_var: ", g_mean_var.columns.values)

    df = apply_mean_var(df, mean_var, g_mean_var, ['b', 'c'], groupby_key='a')
    print("df after applying mean_var: ", df)

    df = reverse_mean_std(df, mean_var, g_mean_var, groupby_key='a')
    print("df after reversed from mean_var: ", df)