#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/11/7 AM11:59
# @Author  : shaoguang.csg
# @File    : logger.py

import logging


def _create_logger(log_level, log_format="", log_file=""):
    if log_format == "":
        log_format = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s'

    _logger = logging.getLogger("")
    _logger.setLevel(log_level)

    formatter = logging.Formatter(log_format)
    if log_file != "":
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file)
        file_handler.setFormatter(formatter)
        _logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    _logger.addHandler(stream_handler)
    return _logger

logger = _create_logger(logging.INFO)