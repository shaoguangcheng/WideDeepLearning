#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/11/7 AM10:50
# @Author  : shaoguang.csg
# @File    : parse_conf.py

import os.path as path
import yaml
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model_conf_file', 'conf/model_conf.yaml', 'Path to the model config yaml file')
tf.app.flags.DEFINE_string('data_conf_file', 'conf/data_conf.yaml', 'Path to the data config yaml file')


# singleton pattern
def singleton(cls):
    _instances = {}

    def _singleton(*args, **kwargs):
        if cls not in _instances:
            _instances[cls] = cls(*args, **kwargs)
        return _instances[cls]
    return _singleton


def check_file_exists(filename):
    if not path.exists(filename):
        raise Exception("%s does not exists" % filename)


@singleton
class DataConf(object):

    def __init__(self):
        with open(FLAGS.data_conf_file) as data_conf_file:
            data_conf = yaml.load(data_conf_file)

        self.train_file = data_conf.get('train_file', None)
        self.evaluate_file = data_conf.get('evaluate_file', None)
        self.test_file = data_conf.get('test_file', None)
        self.target_column = data_conf.get('target_column', None)
        self.multi_hot_columns = data_conf.get('multi_hot_columns', {})
        self.multi_category_columns = data_conf.get('multi_category_columns',None)
        self.continuous_columns = data_conf.get('continuous_columns', None)
        self.crossed_columns = data_conf.get('crossed_columns', None)
        self.bucketized_columns = data_conf.get('bucketized_columns', None)

        self._check_param()

    def _check_param(self):
        assert self.train_file is not None or self.test_file is not None, \
            "train_file and test_file can not be None at hte same time"

        if self.train_file is not None:
            check_file_exists(self.train_file)
        if self.evaluate_file is not None:
            check_file_exists(self.evaluate_file)
        if self.test_file is not None:
            check_file_exists(self.test_file)

        if self.target_column is None:
            raise Exception("Must have a target column")


@singleton
class ModelConf(object):

    def __init__(self):
        with open(FLAGS.model_conf_file) as model_conf_file:
            model_conf = yaml.load(model_conf_file)

        self.num_cpu_core = model_conf.get('num_cpu_core', 4)
        self.num_threads_per_core = model_conf.get('num_threads_per_core', 1)

        self.data_mode = model_conf.get('data_mode', 0)
        self.model_dir = model_conf.get('model_dir', '/tmp/')
        self.log_dir = model_conf.get('log_dir', '/tmp/')
        self.append_cols = model_conf.get('append_cols', None)

        self.model_type = model_conf.get('model_type', 0)
        self.problem_type = model_conf.get('problem_type', 0)

        self.n_classes = model_conf.get('n_classes', 2)
        self.max_iter = model_conf.get('max_iter', 100000)
        self.save_checkpoint_interval = model_conf.get('save_checkpoint_interval', 10000)
        self.batch_size = model_conf.get('batch_size', 256)

        self.norm_type = model_conf.get('norm_type', None)
        self.norm_columns = model_conf.get('norm_columns', [])
        self.groupby_key = model_conf.get('groupby_key', None)

        self.base_lr = model_conf.get('base_lr', 0.01)
        self.lr_policy = model_conf.get('lr_policy', 'step')
        self.step_size = model_conf.get('step_size', 10000)

        self.alpha = model_conf.get('alpha', 0.0)
        self.beta = model_conf.get('beta', 0.0)
        self.early_stopping_interval = model_conf.get('early_stopping_interval', 1000)
        self.evaluate_interval = model_conf.get('evaluate_interval', 500)

        self.hidden_units = model_conf.get('hidden_units', [10,10])
        self.embedding_dimension = model_conf.get('embedding_dimension', 16)
        self.dropout_ratio = model_conf.get('dropout_ratio', 0.5)

        self.wide_features = model_conf.get('wide_features', None)
        self.deed_features = model_conf.get('deed_features', None)

        self.metrics = model_conf.get('metrics', None)

        self._check_param()

    def _check_param(self):
        if self.model_dir is not None:
            check_file_exists(self.model_dir)
        if self.log_dir is not None:
            check_file_exists(self.log_dir)

        assert self.data_mode in (0, 1), "data_mode must be in {0: standard, 1: big data mode}"
        assert self.model_type in (0, 1, 2), "model_type must be in {0:wide, 1:deep, 2:wide_and_deep}"
        assert self.problem_type in (0, 1), "problem_type must be in {0: classification, 1:regression}"
        assert self.n_classes >= 2, "n_classes must be greater than 2"

        pass  # TODO need to check other params (some trivial but important work)


if __name__ == '__main__':
    x = DataConf()

