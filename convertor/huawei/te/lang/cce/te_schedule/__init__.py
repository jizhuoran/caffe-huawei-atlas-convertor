#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

cce schedule
"""

from .cce_schedule import cce_build_code
from .depthwise_conv2d_schedule import depthwise_conv2d_backprop_filter_d_schedule
from .depthwise_conv2d_schedule import depthwise_conv2d_backprop_input_d_schedule
from .depthwise_conv2d_schedule import depthwise_conv2d_schedule
from .pooling2d_schedule import pooling2d_schedule
from .reduce_mean_mid_reduce_high_performance_schedule import *
from .depthwise_conv2d_native_v200_schedule import depthwise_conv2d_native_v200_schedule
from .reduce_5hdc_intrin import *
