"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use 
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

comput schedule init
"""

from .te_compute.broadcast_compute import broadcast
from .te_compute.cast_compute import ceil, floor, round, trunc
from .te_compute.common import round_to, cast_to, cast_to_round, calculate_one_or_zero
from .te_compute.concat_compute import concat
from .te_compute.conv_compute import conv
from .te_compute.conv_compute import ConvParam
from .te_compute.conv_compute import check_conv_shape
from .te_compute.conv_compute import conv_compress
from .te_compute.max_pool_v200_compute import MaxPoolParam, max_pool_v200
from .te_compute.dim_conv import compute_four2five, compute_five2four
from .te_compute.elewise_compute import vmuls, vadds, vlog, vexp, vabs, vrec, \
    vrelu, vnot, vsqrt, vrsqrt, vdiv, vmul, vadd, vsub, vmin, vmax, vor, vand, vaxpy, \
    vmla, vmadd, vmaddrelu, vmaxs, vmins, vcmp, vlogic, vsel, vcmpsel, vmod
from .te_compute.reduction_compute import sum, reduce_min, reduce_max, \
    reduce_prod, tuple_sum
from .te_compute.segment_compute import unsorted_segment_max, \
    unsorted_segment_min, unsorted_segment_sum, unsorted_segment_mean, \
    unsorted_segment_prod
from .te_compute import util
from .te_schedule import cce_build_code
from .te_compute.mmad_compute import matmul
from .te_compute.gemm_compute import gemm
from .te_compute.depthwise_conv2d_compute import depthwise_conv2d_backprop_filter_d_compute
from .te_compute.depthwise_conv2d_compute import depthwise_conv2d_backprop_input_d_compute
from .te_compute.depthwise_conv2d_compute import depthwise_conv2d_compute
from .te_compute.pooling2d_compute import pooling2d
from .te_compute.conv2d_backprop_filter_compute import conv2d_backprop_filter_compute
from .te_compute.conv2d_backprop_input_compute import conv2d_backprop_input_compute
from .te_compute.split_compute import split_compute_com
from .te_schedule.split_schedule import split_schedule_com
from .te_compute.inplace_compute import inplace_add, inplace_sub, inplace_update
from .te_compute.four_2_five_computer import compute_four_2_five
from .te_compute.conv3d_backprop_input_compute import conv3d_backprop_input_compute
from .te_compute.conv3d_backprop_filter_compute import \
    conv3d_backprop_filter_compute
from .te_compute.util import dsl_check_support
