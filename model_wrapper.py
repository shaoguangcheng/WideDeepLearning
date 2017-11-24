#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/11/10 AM10:26
# @Author  : shaoguang.csg
# @File    : model_wrapper.py

import pandas as pd
import numpy as np
import copy
import os
from functools import partial
from tensorflow.contrib.learn import monitors

from utils.logger import logger
from utils.system_op import wcl
from utils.metrics import metric_func_mapping
from parse_conf import DataConf, ModelConf
from mean_std_normlizer import MeanStdNormalizer
from minmax_normalizer import MinMaxNormalizer
from model_core import ModelCore
from make_batch_data import Dataset, make_tf_batch_columns


class ModelWrapper(object):

    def __init__(self, data_conf, model_conf):
        self._data_conf = data_conf
        self._model_conf = model_conf
        self._normalizer = None

        self._train_data = None
        self._validation_data = None
        self._test_data = None

        self._feature_columns = self.get_feature_columns() # TODO check whether this columns is None

        if self._model_conf.norm_type is not None:
            self._normalizer = self._normalizer_mapping[self._model_conf.norm_type](
                self._model_conf.norm_columns,
                self._model_conf.groupby_key,
                self._model_conf.data_mode
            )

        self._model = ModelCore(model_conf, data_conf)

    def train(self):
        num_examples = wcl(self._data_conf.train_file)-1
        logger.info("{num_examples} examples in {filename}".format(num_examples=num_examples, filename=self._data_conf.train_file))

        if self._normalizer is not None:
            self._cal_normalizer()
            self._normalizer.save_to_file(self._model_conf.model_dir)

        logger.info("start to training the model ...")

        if self._model_conf.data_mode == 1 or self._train_data is None:
            self._train_data = self._load_data(
                self._data_conf.train_file,
                self._model_conf.data_mode,
                chunksize=10000000
            )

        self.add_validation_monitor()

        if self._model_conf.data_mode == 1: # big data mode
            num_epoch = max(1, int(self._model_conf.batch_size * self._model_conf.max_iter / num_examples))
            for chunk in self._train_data:
                if self._normalizer is not None:
                    chunk = self._normalizer.transform(chunk, self._model_conf.norm_columns)
                features, labels = chunk[self._feature_columns], chunk[self._data_conf.target_column]
                labels = self._check_label_type(labels)
                num_steps = chunk.shape[0] * num_epoch / self._model_conf.batch_size
                dataset = Dataset(self._model_conf.batch_size, self._data_conf , features, labels)
                self._model.partial_fit(
                    input_fn=lambda: dataset.next_batch(),
                    steps=num_steps
                )
        else:
            if self._normalizer is not None:
                self._train_data = self._normalizer.transform(self._train_data, self._model_conf.norm_columns)
            features, labels = self._train_data[self._feature_columns], self._train_data[self._data_conf.target_column]
            labels = self._check_label_type(labels)
            dataset = Dataset(self._model_conf.batch_size, self._data_conf, features, labels)
            self._model.fit(
                input_fn=lambda: dataset.next_batch(),
                steps=self._model_conf.max_iter
            )

    def evaluate(self):
        eval_result = self._model.evaluate(
             input_fn=lambda: self.build_validation_input_fn()
        )
        return eval_result

    def predict(self):
        num_examples = wcl(self._data_conf.test_file) - 1
        logger.info("{num_examples} examples in {filename}".format(num_examples=num_examples, filename=self._data_conf.test_file))

        if self._normalizer is not None:
            logger.info("load normalizer from disk ...")
            self._normalizer.load_from_file(self._model_conf.model_dir)

        test_data = self._load_data(
            filename=self._data_conf.test_file,
            big_data_mode=True,
            chunksize=500000
        )

        predict_norm_columns = copy.deepcopy(self._model_conf.norm_columns)
        if self._model_conf.norm_columns is not None and\
                        self._data_conf.target_column in self._model_conf.norm_columns:  # remove target if exists
            predict_norm_columns.remove(self._data_conf.target_column)

        result_columns = self._model_conf.append_cols + ['prediction_result'] \
            if self._model_conf.append_cols is not None else ['prediction_result']
        if self._model_conf.problem_type == 0:
            result_columns += ['class_id', 'class_proba']

        output_result = pd.DataFrame()
        for chunk in test_data:  # do prediction
            if self._normalizer is not None:
                chunk = self._normalizer.transform(chunk, predict_norm_columns)
            result = self._model.predict(
                input_fn=lambda: make_tf_batch_columns(
                    data_conf=self._data_conf,
                    batch_features=chunk[self._feature_columns]
                )
            )

            result = list(result)
            chunk['prediction_result'] = pd.Series(data=result, index=chunk.index)

            if self._model_conf.problem_type == 0:
                chunk['class_id'] = pd.Series(np.argmax(result, axis=1), index=chunk.index)
                chunk['class_proba'] = pd.Series(np.max(result, axis=1), index=chunk.index)

            if self._normalizer is not None \
                    and self._model_conf.problem_type == 1\
                    and self._data_conf.target_column in self._model_conf.norm_columns:
                chunk = self._normalizer.reverse_transform(chunk, columns=['prediction_result'], use_columns=[self._data_conf.target_column])
            output_result = pd.concat([output_result, chunk[result_columns]], axis=0)

        filename = os.path.join(self._model_conf.model_dir, 'prediction_result')
        output_result.to_csv(filename)
        logger.info("Save predicted result to {filename}".format(filename=filename))

        if self._model_conf.problem_type == 1 and \
                        self._data_conf.target_column in output_result.columns.values:\
            metric_result = self._cal_metrics(
                output_result[self._data_conf.target_column],
                output_result['prediction_result']
            )
        else:
            metric_result = None  # TODO add metrics for classification
        logger.info(metric_result)

        return output_result, metric_result

    def _cal_metrics(self, y_true, y_pred):
        result = {}
        for metric in self._model_conf.metrics:
            result[metric] = metric_func_mapping[metric](y_true, y_pred)
        return result

    def get_weight(self):
        return self._model.get_weight()

    def add_validation_monitor(self):
        if self._data_conf.evaluate_file is not None:
            valid_monitor = partial(
                monitors.ValidationMonitor,
                input_fn=lambda: self.build_validation_input_fn(),
                eval_steps=1,
                every_n_steps=self._model_conf.evaluate_interval,
                metrics=self._model.metrics
            )

            if self._model_conf.early_stopping_interval > 0:
                valid_monitor = valid_monitor(
                    early_stopping_rounds=self._model_conf.early_stopping_interval,
                    early_stopping_metric="loss",
                    early_stopping_metric_minimize=True
                )

            self._model.add_monitor(valid_monitor)

    def build_validation_input_fn(self):
        num_examples = wcl(self._data_conf.evaluate_file) - 1
        logger.info("{num_examples} examples in {filename}".format(num_examples=num_examples, filename=self._data_conf.evaluate_file))

        if self._normalizer is not None:
            logger.info("load normalizer from disk ...")
            self._normalizer.load_from_file(self._model_conf.model_dir)

        eval_data = self._load_data(
            filename=self._data_conf.evaluate_file,
            big_data_mode=False
        )

        if self._normalizer is not None:
            eval_data = self._normalizer.transform(eval_data, self._model_conf.norm_columns)

        labels = eval_data[self._data_conf.target_column]
        labels = self._check_label_type(labels)
        return make_tf_batch_columns(
                 data_conf=self._data_conf,
                 batch_features=eval_data[self._feature_columns],
                 batch_labels=labels
             )

    def get_feature_columns(self):
        feature_columns = []
        for columns in [self._data_conf.continuous_columns, self._data_conf.multi_category_columns]:
            feature_columns += columns if columns is not None else []
        if self._data_conf.multi_hot_columns is not None:
            feature_columns += list(self._data_conf.multi_hot_columns.keys())
        return feature_columns

    def _cal_normalizer(self):
        logger.info("start to cal normalizer ...")
        used_columns = None
        if self._model_conf.data_mode == 1:
            used_columns = self._model_conf.norm_columns
            used_columns += [self._model_conf.groupby_key] if self._model_conf.groupby_key is not None else []

        self._train_data = self._load_data(
            self._data_conf.train_file,
            self._model_conf.data_mode,
            columns=used_columns
        )

        self._normalizer.fit(self._train_data)

    def _check_label_type(self, labels):
        if self._model_conf.problem_type == 0:
            return labels.astype(np.int32)
        elif self._model_conf.problem_type == 1:
            return labels.astype(np.float32)
        else:
            logger.error("unsupport problem type")
            raise TypeError

    @staticmethod
    def _load_data(filename, big_data_mode, columns=None, chunksize=10000000):
        logger.info("loading data from disk ...")
        if filename is None:
            logger.error('Can not load the data, filename is None')
            raise FileExistsError

        if big_data_mode:
            data = pd.read_csv(filename, sep=',', usecols=columns, header='infer', engine='c', chunksize=chunksize)
        else:
            data = pd.read_csv(filename, sep=',', header='infer', engine='c')
        return data

    _normalizer_mapping = {
        0: MeanStdNormalizer,
        1: MinMaxNormalizer
    }
