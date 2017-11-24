#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/11/7 PM6:28
# @Author  : shaoguang.csg
# @File    : wdl_core.py

from parse_conf import (ModelConf, DataConf)
from utils.logger import logger
from utils.tensorflow_helper import get_available_devices
from utils.metrics import (
    streaming_classification_metrics,
    streaming_regression_metrics
)

from functools import partial
import tensorflow as tf

from tensorflow.python.feature_column.feature_column import (
    categorical_column_with_vocabulary_list,
    categorical_column_with_hash_bucket,
    numeric_column,
    embedding_column,
    crossed_column,
    bucketized_column,
    indicator_column
)

tf.logging.set_verbosity(tf.logging.INFO)


class ModelCore(object):
    """
    The core of wdl implementation. You do not need to use this class
    """
    def __init__(self, model_conf, data_conf):
        logger.info("build model ...")
        self._model_conf = model_conf
        self._data_conf = data_conf
        self._monitors = None

        self._build_feature_columns()

        self._run_config = tf.contrib.learn.RunConfig(
            save_summary_steps=100,
            save_checkpoints_steps=self._model_conf.save_checkpoint_interval,
            session_config=self._get_session_config(),
            keep_checkpoint_max=5
        )

        if self._model_conf.problem_type == 0:
            logger.info("Classification mode ...")
            self._create_classification_model()
            self.metrics = streaming_classification_metrics
        elif self._model_conf.problem_type == 1:
            logger.info("Regression mode ...")
            self._create_regression_model()
            self.metrics = streaming_regression_metrics
        else:
            logger.error("Unkown problem type")

        logger.info(self._model)

    def fit(self, input_fn, steps):
        logger.info("train the model ...") # TODO set monitors inner class
        self._model.fit(
            input_fn=input_fn,
            steps=steps,
            monitors=self._monitors
        )
        return self

    def partial_fit(self, input_fn, steps):
        logger.info("train the model ...")
        self._model.partial_fit(
            input_fn=input_fn,
            steps=steps,  #
            monitors=self._monitors
        )
        return self

    def evaluate(self, input_fn, name=None):
        logger.info("do the evaluation ...")
        return self._model.evaluate(
            input_fn=input_fn,
            steps=1,
            metrics=self.metrics,
            name=name
        )

    def predict(self, input_fn):
        logger.info("do the prediction ...")
        if self._model_conf.problem_type == 0:
            func = self._model.predict_proba
        elif self._model_conf.problem_type == 1:
            func = self._model.predict_scores
        else:
            logger.error("unsupport problem type") # TODO

        return func(
            input_fn=input_fn,
            as_iterable=True
        )

    def add_monitor(self, monitor):
        if self._monitors is None:
            self._monitors = []
        self._monitors.append(monitor)

    def export_saved_model(self):
        pass # TODO

    def load_model(self):
        pass # TODO

    def get_weight(self):
        weight = {}
        for var_name in self._model.get_variable_names():
            if var_name.endswith('weights'):
                weight[var_name] = self._model.get_variable_value(var_name)
        return weight    

    def _get_session_config(self):
        gpu_devices, _ = get_available_devices()
        if len(gpu_devices) == 0:
            logger.warning("No GPU found, using CPU")
            session_config = tf.ConfigProto(
                device_count={'CPU': self._model_conf.num_cpu_core},
                intra_op_parallelism_threads=self._model_conf.num_threads_per_core,
                inter_op_parallelism_threads=self._model_conf.num_threads_per_core,
                log_device_placement=False
            )
        else:
            logger.info("GPU: {} found".format(gpu_devices))
            session_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        return session_config

    def _create_classification_model(self):
        model_object = partial(
            self.classification_model_mapping[self._model_conf.model_type],
            model_dir=self._model_conf.model_dir,
            n_classes=self._model_conf.n_classes,
            gradient_clip_norm=10.0,
            config=self._run_config
        )

        if self._model_conf.model_type == 0:
            logger.info("Using linear model ...")
            self._model = model_object(
                feature_columns=self._wide_columns,
                optimizer=self._get_optimizer(optimizer_type='ftrl'),  # TODO
            )
        elif self._model_conf.model_type == 1:
            logger.info("Using dnn model ...")
            self._model = model_object(
                hidden_units=self._model_conf.hidden_units,
                feature_columns=self._deep_columns,
                dropout=self._model_conf.dropout_ratio,
                optimizer=self._get_optimizer(optimizer_type='momentum')
            )
        elif self._model_conf.model_type == 2:
            logger.info("Using wdl model ...")
            self._model = model_object(
                linear_feature_columns=self._wide_columns,
                linear_optimizer=self._get_optimizer(optimizer_type='ftrl'),
                dnn_feature_columns=self._deep_columns,
                dnn_optimizer=self._get_optimizer(optimizer_type='momentum'),
                dnn_hidden_units=self._model_conf.hidden_units,
                dnn_dropout=self._model_conf.dropout_ratio,
                fix_global_step_increment_bug=True
            )
        else:
            logger.error("Unkown model type")
            self._model = None

    def _create_regression_model(self):
        model_object = partial(
            self.regression_model_mapping[self._model_conf.model_type],
            model_dir=self._model_conf.model_dir,
            gradient_clip_norm=10.0,
            label_dimension=1,
            config=self._run_config
        )
        if self._model_conf.model_type == 0:
            logger.info("Using linear model ...")
            self._model = model_object(
                feature_columns=self._wide_columns,
                optimizer=self._get_optimizer(optimizer_type='ftrl'),
            )
        elif self._model_conf.model_type == 1:
            logger.info("Using dnn model ...")
            self._model = model_object(
                hidden_units=self._model_conf.hidden_units,
                feature_columns=self._deep_columns,
                dropout=self._model_conf.dropout_ratio,
                optimizer=self._get_optimizer(optimizer_type='momentum')
            )
        elif self._model_conf.model_type == 2:
            logger.info("Using wdl model ...")
            self._model = model_object(
                linear_feature_columns=self._wide_columns,
                linear_optimizer=self._get_optimizer(optimizer_type='ftrl'),
                dnn_feature_columns=self._deep_columns,
                dnn_optimizer=self._get_optimizer(optimizer_type='momentum'),
                dnn_hidden_units=self._model_conf.hidden_units,
                dnn_dropout=self._model_conf.dropout_ratio,
                fix_global_step_increment_bug=True
            )
        else:
            logger.error("Unkown model type")

    def _get_optimizer(self, optimizer_type='ftrl'):
        if optimizer_type == 'ftrl':
            optimizer = tf.train.FtrlOptimizer(
                learning_rate=self._model_conf.base_lr,
                l1_regularization_strength=self._model_conf.alpha,
                l2_regularization_strength=self._model_conf.beta
            )
        elif optimizer_type == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=self._model_conf.base_lr
            )
        elif optimizer_type == 'momentum':
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=self._model_conf.base_lr, # TODO change the learning rate policy from fixed to step decay
                momentum=0.9
            )
        elif optimizer_type == 'adam':
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self._model_conf.base_lr
            )
        else:
            logger.error("Unsupport optimizer type")
            optimizer = None
        return optimizer

    def _get_lr(self):
        self._global_step = tf.train.get_or_create_global_step()
        if self._model_conf.lr_policy == 'fixed':
            lr = self._model_conf.base_lr
        elif self._model_conf.lr_policy == 'step':
            lr = tf.train.exponential_decay(
                learning_rate=self._model_conf.base_lr,
                global_step=self._global_step,
                decay_steps=self._model_conf.step_size,
                decay_rate=0.95,
                staircase=True
            )
        else:
            logger.error("Unsupport lr policy")
            lr = self._model_conf.base_lr
        return lr

    def _build_feature_columns(self,):
        multi_hot_feature_columns = {}
        multi_hot_feature_columns_deep = {}
        multi_category_feature_columns = {}
        continuous_feature_columns = {}
        crossed_feature_columns = []
        bucketized_feature_columns = []
        embedding_feature_columns = []

        if self._data_conf.multi_hot_columns is not None :
            for column in self._data_conf.multi_hot_columns:
                multi_hot_feature_columns[column] = categorical_column_with_vocabulary_list(
                    column,
                    self._data_conf.multi_hot_columns[column],
                    dtype=tf.string
                )
                multi_hot_feature_columns_deep[column] = indicator_column(multi_hot_feature_columns[column])

        if self._data_conf.multi_category_columns is not None:
            multi_category_feature_columns = {column: categorical_column_with_hash_bucket(column, hash_bucket_size=1000)
                                      for column in self._data_conf.multi_category_columns}

        if self._data_conf.continuous_columns is not None:
            continuous_feature_columns = {column: numeric_column(column) for column in self._data_conf.continuous_columns}

        if self._data_conf.crossed_columns is not None:
            crossed_feature_columns = [crossed_column(_, hash_bucket_size=100000) for _ in self._data_conf.crossed_columns]

        if self._data_conf.bucketized_columns is not None:
            [bucketized_feature_columns.append(
                bucketized_column(continuous_feature_columns[column], boundaries=boundary))
             for column, boundary in self._data_conf.bucketized_columns.items]

        if len(multi_category_feature_columns) > 0:
            embedding_feature_columns = [embedding_column(_, dimension=self._model_conf.embedding_dimension)
                                         for _ in multi_category_feature_columns.values()]

        self._feature_mapping = {
            0: list(multi_hot_feature_columns.values()),
            1: list(multi_category_feature_columns.values()),
            2: list(continuous_feature_columns.values()),
            3: crossed_feature_columns,
            4: bucketized_feature_columns,
            5: embedding_feature_columns,
            6: list(multi_hot_feature_columns_deep.values())
        }

        self._build_feature_columns_for_model()

    def _build_feature_columns_for_model(self):
        if self._model_conf.model_type == 0:
            self._build_wide_feature_columns()
        elif self._model_conf.model_type == 1:
            self._build_deep_feature_columns()
        elif self._model_conf.model_type == 2:
            self._build_wide_feature_columns()
            self._build_deep_feature_columns()
        elif self._model_conf.model_type == 3:
            pass
        else:
            logger.error("Unsupport model type")

    def _build_wide_feature_columns(self):
        self._wide_columns = []
        for index in self._model_conf.wide_features:
            self._wide_columns += self._feature_mapping[index]

    def _build_deep_feature_columns(self):
        self._deep_columns = []
        for index in self._model_conf.deed_features:
            if index == 0:
                self._deep_columns += self._feature_mapping[6]
            else:
                self._deep_columns += self._feature_mapping[index]

    classification_model_mapping = {
        0: tf.contrib.learn.LinearClassifier,
        1: tf.contrib.learn.DNNClassifier,
        2: tf.contrib.learn.DNNLinearCombinedClassifier,
        3: tf.contrib.learn.LogisticRegressor
    }

    regression_model_mapping = {
        0: tf.contrib.learn.LinearRegressor,
        1: tf.contrib.learn.DNNRegressor,
        2: tf.contrib.learn.DNNLinearCombinedRegressor
    }


if __name__ == '__main__':
    model_conf = ModelConf()
    data_conf = DataConf()
    wdl = ModelCore(model_conf, data_conf)
    params = wdl._model.get_params(deep=True)['params']
    params['dropout'] = 0.9
    xx = {'params':params}
    wdl._model.set_params(**xx)

    logger.info(wdl._model.get_variable_names())
    for w in wdl._model.get_variable_names():
        logger.info(wdl._model.get_variable_value(w))