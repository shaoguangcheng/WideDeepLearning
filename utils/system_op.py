#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/11/10 PM5:01
# @Author  : shaoguang.csg
# @File    : system_op.py

import os
from utils.logger import logger


def wcl(filename):
    """
    wc -l filename
    :param filename:
    :return:
    """
    if not os.path.exists(filename):
        logger.error('{} does not exists'.format(filename))
        raise FileNotFoundError
    cmd = 'wc -l ' + filename
    line = os.popen(cmd).readline().strip()
    return int(line.split()[0])


