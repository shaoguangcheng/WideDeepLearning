#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/11/9 AM10:56
# @Author  : shaoguang.csg
# @File    : minmax_normalizer.py

from normalizer import Normalizer


class MinMaxNormalizer(Normalizer):

    def __init__(self, columns=None, groupby_key=None, big_data_mode=False, use_min=True):
        super(MinMaxNormalizer, self).__init__(columns, groupby_key, big_data_mode)

        self._min = None
        self._max = None
        self._use_min = use_min

    def fit(self, data, columns, groupby_key, category_columns=None):
        pass

    def transform(self, df):
        return df

    def reverse_transform(self, df):
        pass

    def save_to_file(self, filename):
        pass

    def load_from_file(self, filename):
        pass