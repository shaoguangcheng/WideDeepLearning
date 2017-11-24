#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/11/10 PM5:20
# @Author  : shaoguang.csg
# @File    : make_batch_data.py

import tensorflow as tf
import numpy as np


class Dataset(object):

    def __init__(self, batch_size, data_conf, features, labels=None):
        assert features.shape[0] == labels.shape[0], ""

        self._batch_size = batch_size
        self._num_example = features.shape[0]
        self._data_conf = data_conf
        self._features = features
        self._labels = labels

        self._current_pos = 0

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    @property
    def current_pos(self):
        return self._current_pos

    def next_batch(self):
        start = self._current_pos
        self._current_pos += self._batch_size

        if self._current_pos > self._num_example:
            self._current_pos = 0

            perm = np.arange(self._num_example)

            np.random.shuffle(perm)
            self._features = self._features[perm]
            self._labels = self._labels[perm]

            start = 0
            self._current_pos = self._batch_size
        end = self._current_pos

        batch_features = self._features[start:end]
        batch_labels = None
        if self._labels is not None:
            batch_labels= self._labels[start:end]

        return make_tf_batch_columns(
            self._data_conf,
            batch_features=batch_features,
            batch_labels=batch_labels
        )


def make_tf_batch_columns(data_conf, batch_features, batch_labels=None):
    feature_cols = {}
    if data_conf.continuous_columns is not None:
        continuous_cols = {
            k: tf.constant(
                batch_features[k].values.astype(np.float64),
                shape=[batch_features[k].size, 1]
            )
            for k in data_conf.continuous_columns
        }
        feature_cols.update(continuous_cols)

    if data_conf.multi_category_columns is not None:
        multi_category_cols = {
            k: tf.SparseTensor(
                indices=[[i, 0] for i in range(batch_features[k].size)],
                values=batch_features[k].values.astype(str),
                dense_shape=[batch_features[k].size, 1]
            )
            for k in data_conf.multi_category_columns
        }
        feature_cols.update(multi_category_cols)

    if data_conf.multi_hot_columns is not None:
        multi_hot_columns = {
            k: tf.SparseTensor(
                indices=[[i, 0] for i in range(batch_features[k].size)],
                values=batch_features[k].values.astype(str),
                dense_shape=[batch_features[k].size, 1]
            )
            for k in data_conf.multi_hot_columns
        }
        feature_cols.update(multi_hot_columns)

    if batch_labels is not None:
        if batch_labels.dtype == 'int64':
            dtype = np.int32
        elif batch_labels.dtype == 'float64':
            dtype = np.float32
        else:
            dtype = batch_labels.dtype
        labels = tf.constant(batch_labels.values.astype(dtype), shape=[batch_labels.size, 1])
        return feature_cols, tf.reshape(labels, [-1, 1])
    return feature_cols
