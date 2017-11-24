#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/11/14 PM2:21
# @Author  : shaoguang.csg
# @File    : main.py

from parse_conf import DataConf, ModelConf
from model_wrapper import ModelWrapper
from utils.logger import logger
from time import time

start = time()

data_conf = DataConf()
model_conf = ModelConf()

model = ModelWrapper(data_conf=data_conf, model_conf=model_conf)
model.train()
logger.info(model.get_weight())
model.evaluate()
result = model.predict()

end = time()

logger.info('time: {}'.format(end-start))

# 2 core 1 threads 116
# 1 core  228