#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/11/9 AM9:38
# @Author  : shaoguang.csg
# @File    : normalizer.py


from abc import ABCMeta, abstractmethod


class Normalizer(object):

    __metaclass__ = ABCMeta

    def __init__(self, columns=None, groupby_key=None, big_data_mode=False):
        """

        :param big_data_mode:
        """
        self._columns = columns
        self._groupby_key = groupby_key
        self._big_data_mode = big_data_mode

    @abstractmethod
    def fit(self, data, columns, groupby_key, category_columns=None):
        raise NotImplementedError()

    @abstractmethod
    def transform(self, df):
        raise NotImplementedError()

    @abstractmethod
    def reverse_transform(self, df):
        raise NotImplementedError()

    @abstractmethod
    def save_to_file(self, filename):
        raise NotImplementedError()

    @abstractmethod
    def load_from_file(self, filename):
        raise NotImplementedError()
