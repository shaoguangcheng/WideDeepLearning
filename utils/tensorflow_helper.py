#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/11/9 PM12:46
# @Author  : shaoguang.csg
# @File    : tensorflow_helper.py

from tensorflow.python.client import device_lib
from tensorflow.python.ops.metrics_impl import _remove_squeezable_dimensions, mean
from tensorflow.python.ops import math_ops


def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    gpu_devices = [device_proto.name for device_proto in local_device_protos if device_proto.device_type == 'GPU']
    cpu_devices = [device_proto.name for device_proto in local_device_protos if device_proto.device_type == 'CPU']
    return gpu_devices, cpu_devices


def streaming_mean_absolute_percentage_error(
        predictions,
        labels,
        weights=None,
        metrics_collections=None,
        updates_collections=None,
        name=None):
    return mean_absolute_percentage_error(
        predictions=predictions, labels=labels, weights=weights,
        metrics_collections=metrics_collections,
        updates_collections=updates_collections, name=name)


def mean_absolute_percentage_error(
        labels,
        predictions,
        weights=None,
        metrics_collections=None,
        updates_collections=None,
        name=None):
    predictions, labels, weights = _remove_squeezable_dimensions(
        predictions=predictions, labels=labels, weights=weights)

    absolute_percentage_errors = math_ops.abs(math_ops.div(predictions-labels, labels))
    return mean(absolute_percentage_errors, weights, metrics_collections,
                updates_collections, name or 'mean_absolute_percentage_error')


