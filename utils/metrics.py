#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/11/10 AM8:33
# @Author  : shaoguang.csg
# @File    : metrics.py

from sklearn.metrics import *
from sklearn.metrics.regression import _check_reg_targets
from sklearn.externals.six import string_types
import numpy as np


from tensorflow.contrib.learn import (PredictionKey, MetricSpec)
from tensorflow.contrib.metrics import (
    streaming_accuracy,
    streaming_auc,
    streaming_mean_absolute_error,
    streaming_precision,
    streaming_recall,
    streaming_mean_squared_error,
    streaming_root_mean_squared_error
)
from utils.tensorflow_helper import streaming_mean_absolute_percentage_error


def mean_absolute_percentage_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average'):
    """
    mape = sum(abs(y_pred-y_true)/y_true)/N
    :param y_true:
    :param y_pred:
    :param sample_weight:
    :param multioutput:
    :return:
    """
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput)
    output_errors = np.average(np.abs((y_pred - y_true)/y_true), weights=sample_weight, axis=0)

    if isinstance(multioutput, string_types):
        if multioutput == 'raw_values':
            return output_errors
        elif multioutput == 'uniform_average':
            multioutput = None

    return np.average(output_errors, weights=multioutput)


metric_func_mapping = {
    'auc': auc,
    'accuracy': accuracy_score,
    'precision': precision_score,
    'recall': recall_score,
    'mae': mean_absolute_error,
    'mse': mean_squared_error,
    'cm': confusion_matrix,
    'mape': mean_absolute_percentage_error
}

# for evaluating and monitoring
streaming_classification_metrics = {
    'accuracy': MetricSpec(
        metric_fn=streaming_accuracy,
        prediction_key=PredictionKey.CLASSES
    ),
    'recall': MetricSpec(
        metric_fn=streaming_recall,
        prediction_key=PredictionKey.CLASSES,
    ),
    'precision': MetricSpec(
        metric_fn=streaming_precision,
        prediction_key=PredictionKey.CLASSES
    )
}

streaming_regression_metrics = {
    'mae': MetricSpec(
        metric_fn=streaming_mean_absolute_error,
        prediction_key=PredictionKey.SCORES
    ),
    'mse': MetricSpec(
        metric_fn=streaming_mean_squared_error,
        prediction_key=PredictionKey.SCORES
    ),
    'rmse': MetricSpec(
        metric_fn=streaming_root_mean_squared_error,
        prediction_key=PredictionKey.SCORES
    ),
    'mape': MetricSpec(
        metric_fn=streaming_mean_absolute_percentage_error,
        prediction_key=PredictionKey.SCORES
    )
}