#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/11/9 AM10:53
# @Author  : shaoguang.csg
# @File    : pandas_helper.py


def get_unique_values(df, columns):
    """
    check unique value for each column in columns
    :param df:
    :param columns:
    :return:
    """
    category_unique_values = {}
    for column in columns:
        if column not in df.columns.values:
            raise Exception('%s must exists in df.columns' % column)
        category_unique_values[column] = df[column].unique()
    return category_unique_values


def cal_square_sum(df, columns=None, groupby_key=None):
    """

    :param df:
    :param columns:
    :param groupby_key:
    :return:
    """
    if columns is None:
        columns = df.columns.values
    if groupby_key is not None and groupby_key not in df.columns.values:
        raise Exception('%s not exists in chunk.columns' % groupby_key)

    if groupby_key is None:
        return df[columns].mul(df[columns], fill_value=1.0).astype('float64').groupby(by=lambda _:1).sum()
    else:
        return df[columns].mul(df[columns], fill_value=1.0).astype('float64').groupby(df[groupby_key]).sum()
