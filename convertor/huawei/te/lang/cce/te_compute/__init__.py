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

cce compute API:
In order to simplify the  procedure of writing schedule,TE provides a set of
TensorEngine APIs.
Using those API to develop operators, you can use the "Auto_schedule" create
schedule.
"""
# pylint: disable=redefined-builtin
from .broadcast_compute import broadcast
from .cast_compute import ceil, floor, round, trunc, round_d
from .common import round_to, cast_to, cast_to_round
from .concat_compute import concat
from .conv_compute import conv, ConvParam, check_conv_shape, conv_compress
from .dim_conv import compute_four2five, compute_five2four
from .elewise_compute import vmuls, vadds, vlog, vexp, vabs, vrec, vrelu, vnot, \
    vsqrt, vrsqrt, vdiv, vmul, vadd, vsub, vmin, vmax, vor, vand, vaxpy, vmla, \
    vmadd, \
    vmaddrelu, vmaxs, vmins, vcmp, vlogic, vsel, vcmpsel, vmod
from .reduction_compute import sum, reduce_min, reduce_max, reduce_prod
from .segment_compute import unsorted_segment_max, unsorted_segment_min, \
    unsorted_segment_sum, \
    unsorted_segment_mean, unsorted_segment_prod
from .depthwise_conv2d_compute import depthwise_conv2d_backprop_filter_d_compute
from .depthwise_conv2d_compute import depthwise_conv2d_backprop_input_d_compute
from .mmad_compute import matmul
from .gemm_compute import gemm
from .depthwise_conv2d_compute import depthwise_conv2d_compute
from .inplace_compute import inplace_add, inplace_sub, inplace_update
from .depthwise_conv2d_native_v200_compute import depthwise_conv2d_native_v200_compute
from .conv3d_compute import conv3d
from .conv3d_compute import Conv3DParam
from .conv3d_compute import check_conv3d_shape
from .conv3d_backprop_input_compute import conv3d_backprop_input_compute
