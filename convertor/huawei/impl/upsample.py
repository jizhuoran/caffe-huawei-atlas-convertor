"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0
upsample
"""
from te import platform as tbe_platform
from te import tvm
from te.platform.cce_build import build_config
from topi.cce import util

# size of 5HD format
DIM_5HD = 5
# size of c0 for fp16 fp32
C0 = 16
# size of c0 for uint8 int8
INT8_C0_SIZE = 32

#pylint: disable=locally-disabled,invalid-name
def check_shape_dtype_format(input_shape, input_dtype, input_format, stride_h, stride_w):
    """
    input_shape:input dic shape
    input_dtype: input dtype
    input_format: input format,NC1HWC0
    The common check rule for tensor shape, just for 5hd
    """
    util.check_shape_rule(input_shape)
    if len(input_shape) != DIM_5HD:
        raise RuntimeError(
            "The dim of tensor must be %d"
            ", actual dim is %d" % (DIM_5HD, len(input_shape)))
    n, c1, h, w, c0 = input_shape

    util.check_shape_rule([n, c1, h * stride_h, w *stride_w, c0])
    util.check_tensor_shape_size([n, c1, h * stride_h, w *stride_w, c0])
    util.check_tensor_shape_size(input_shape)
    product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
    product_list = ["Hi3796CV300ES"]
    if product in product_list:
        check_list = ["float16"]
    else:
        check_list = ["float16", "float32"]
    if input_dtype not in check_list:
        raise RuntimeError("clip only support while dtype is worng")
    shape_c0 = C0
    if input_shape[DIM_5HD - 1] != shape_c0:
        raise RuntimeError(
            "The value of C0 must be 16")

    if input_format != "NC1HWC0":
        raise RuntimeError(
            "The format must be NC1HWC0")


def upsample_check(dic, stride_h, stride_w, kernel_name="upsample"):
    """
    calculating data

    Parameters
    ----------
    dci : dict,include shape dtype and format
    stride : the shape change axis
    scale : the value of tensor change axis, default value is 1
    kernel_name: str  kernel_name

    Returns
    -------
    res: TVM tensor

    result of compute
    """
    input_shape = dic.get("shape")
    input_format = dic.get("format")
    input_dtype = dic.get("dtype").lower()
    if stride_h <= 0 or stride_w <= 0:
        raise RuntimeError(
            "The stride must be greater than 0")
    check_shape_dtype_format(input_shape, input_dtype, input_format, stride_h, stride_w)
    util.check_kernel_name(kernel_name)



def buffer_mapping(schedule, op_list):
    """
    buffer data
    Parameters
    ---------
    schedule : op schedule
    oplist: list of op
    Returns
    -------
    None
    """
    for i_op in op_list:
        schedule[i_op].set_scope(tbe_platform.scope_ubuf)


def cal_shape(dshape, stride_h, stride_w):
    """
    calcute outshape shape
    Parameters
    ---------
    dshape : input shape
    stride:
    Returns
    -------
    output shape
    """
    batch, channel1, height, width, shape_c0 = dshape
    #input_shape = (batch, channel1, height, width, shape_c0)
    output_shape = (batch, channel1, height * stride_h, width * stride_w, shape_c0)
    return output_shape




#pylint: disable=locally-disabled,unnecessary-lambda,too-many-arguments
def gen_upsample_nint(dshape, feature, scale, stride_h, stride_w, dtype):
    """
    gen upsample calclate produce and and tensor
    when float
    Parameters
    ---------
    dshape : input shape
    feature: input tensor
    scale:
    stride:
    dtype:data type
    Returns
    -------
    op_list:op list
    ins_list: instiction list
    tensor_dic:tensor dic
    """
    output_shape = cal_shape(dshape, stride_h, stride_w)
    tensor_dic = {}
    op_list = []
    ins_list = []
    tensor_in_ub = tvm.compute(dshape, lambda *i: feature(*i)
                               , name="tensor_in_ub")
    tensor_dic["tensor_in_ub"] = tensor_in_ub
    op_list += [tensor_in_ub]
    ins_list += ["dma_copy"]
    scale_value = tvm.const(scale, dtype)
    tensor_cal = tvm.compute(output_shape \
                             , lambda n, ci, h, w, c0 \
                                 : tensor_in_ub[n, ci \
            , h // stride_h, w // stride_w, c0] * scale_value, name="tensor_cal")
    tensor_dic["tensor_cal"] = tensor_cal
    op_list += [tensor_cal]
    ins_list += ["vector_muls"]
    res = tvm.compute(output_shape, lambda *i: tensor_cal(*i), name="res")
    tensor_dic["res"] = res
    op_list += [res]
    ins_list += ["dma_copy"]
    return op_list, ins_list, tensor_dic


def gen_upsample(input_x, dtype, scale, stride_h, stride_w):
    """
    gen upsample
    when int or uint
    Parameters
    ---------
    x:inout
    dshape : input shape
    isint:int or not
    scale:
    stride:
    dtype:data type
    Returns
    -------
    op_list:op list
    ins_list: instiction list
    tensor_dic:tensor dic
    feature:input
    y:output
    """
    dshape = input_x.get("shape")
    feature = tvm.placeholder(dshape, name="x", dtype=dtype)
    op_list, ins_list, tensor_dic \
        = gen_upsample_nint(dshape, feature, scale, stride_h, stride_w, dtype)
    res_y = op_list[-1]
    return op_list, ins_list, tensor_dic, feature, res_y


def cal_tilling(input_x, stride_h, stride_w, data_type):
    """
    cal tilling factor

    Parameters
    ---------
    x:inpput
    stride:
    Returns
    -------
    factor:
    axis :axis to cal
    """
    dshape = input_x.get("shape")
    if data_type == "float32":
        ub_limit = 16 * 1024
    else:
        ub_limit = 32 * 1024
    output_shape = cal_shape(dshape, stride_h, stride_w)
    axis = len(output_shape)
    tmp_size = 1
    size = 1
    for i in range(1, len(output_shape) + 1):
        axis = axis-1
        size = tmp_size
        tmp_size *= output_shape[-i]
        if tmp_size > ub_limit:
            break
    factor = ub_limit // size
    return factor, axis

def spilt_axis(schedule, tensor_dic, stride_h, stride_w):
    '''
    :param schedule:
    :param tensor_dic:
    :param stride:
    :return:
     tilling_spilt_axis_dic:spilt axis to tiling dtype is not int8 or uin 8
    '''
    tilling_spilt_axis_dic = {}
    res_o3, res_i3 = schedule[tensor_dic.get("res")] \
        .split(tensor_dic.get("res").op.axis[3], stride_w)
    tilling_spilt_axis_dic["res_o3"] = res_o3
    tilling_spilt_axis_dic["res_i3"] = res_i3
    res_o2, res_i2 = schedule[tensor_dic.get("res")] \
        .split(tensor_dic.get("res").op.axis[2], stride_h)
    tilling_spilt_axis_dic["res_i2"] = res_i2
    tilling_spilt_axis_dic["res_o2"] = res_o2
    tensor_cal_o3, tensor_cal_i3 = schedule[tensor_dic.get("tensor_cal")] \
        .split(tensor_dic.get("tensor_cal").op.axis[3], stride_w)
    tilling_spilt_axis_dic["tensor_cal_o3"] = tensor_cal_o3
    tilling_spilt_axis_dic["tensor_cal_i3"] = tensor_cal_i3
    tensor_cal_o2, tensor_cal_i2 = schedule[tensor_dic.get("tensor_cal")] \
        .split(tensor_dic.get("tensor_cal").op.axis[2], stride_h)
    tilling_spilt_axis_dic["tensor_cal_i2"] = tensor_cal_i2
    tilling_spilt_axis_dic["tensor_cal_o2"] = tensor_cal_o2
    return tilling_spilt_axis_dic


def tilling_spilt_axis(schedule, tensor_dic, stride_h, stride_w):
    """
    spilt axis for cal
    Parameters
    ---------
    isint:
    schedule:schedule
    tensor_dic:tensor dic
    stride
    Returns
    -------
    tilling_spilt_axis_dic:after tilling axis dic
    """
    tilling_spilt_axis_dic = {}
    tilling_spilt_axis_dic = spilt_axis(schedule, tensor_dic, stride_h, stride_w)
    return tilling_spilt_axis_dic


#pylint: disable=locally-disabled,unnecessary-lambda,too-many-arguments
def cal_axis_spilt(input_x, stride_h, stride_w, tilling_spilt_axis_dic, tensor_dic, schedule):
    """
   spilt  axis to cal
    Parameters
    ---------
    x:
    stride
    tilling_spilt_axis_dic
    schedule:schedule
    tensor_dic:tensor dic
    Returns
    -------
    cal_axis_dic:cal tilling axis dic
    """

    cal_axis_dic = {}
    data_type = input_x.get("dtype")
    factor, axis = cal_tilling(input_x, stride_h, stride_w, data_type)
    if axis == 2:
        if factor > stride_h:
            factor = factor // stride_h
            axis_xo, axis_xi = schedule[tensor_dic.get("res")] \
                .split(tilling_spilt_axis_dic.get("res_o2"), factor)
            cal_axis_dic["axis_xi"] = axis_xi
            cal_axis_dic["axis_xo"] = axis_xo
            _, axis_ei = schedule[tensor_dic.get("tensor_cal")] \
                .split(tilling_spilt_axis_dic.get("tensor_cal_o2"), factor)
            cal_axis_dic["axis_ei"] = axis_ei
            _, axis_di = schedule[tensor_dic.get("tensor_in_ub")] \
                .split(tensor_dic.get("tensor_in_ub").op.axis[axis], factor)
            cal_axis_dic["axis_di"] = axis_di
        else:
            cal_axis_dic["axis_xo"] = tilling_spilt_axis_dic.get("res_o3")
            cal_axis_dic["axis_xi"] = tilling_spilt_axis_dic.get("res_i3")
            cal_axis_dic["axis_ei"] = tilling_spilt_axis_dic.get("tensor_cal_i3")
            _, axis_di = schedule[tensor_dic.get("tensor_in_ub")] \
                .split(tensor_dic.get("tensor_in_ub").op.axis[3], stride_w)
            cal_axis_dic["axis_di"] = axis_di
    elif axis == 3:
        factor = factor // stride_w
        axis_xo, axis_xi = schedule[tensor_dic.get("res")] \
            .split(tilling_spilt_axis_dic.get("res_o3"), factor)
        cal_axis_dic["axis_xi"] = axis_xi
        cal_axis_dic["axis_xo"] = axis_xo
        _, axis_ei = schedule[tensor_dic.get("tensor_cal")] \
            .split(tilling_spilt_axis_dic.get("tensor_cal_o3"), factor)
        cal_axis_dic["axis_ei"] = axis_ei
        _, axis_di = schedule[tensor_dic.get("tensor_in_ub")] \
            .split(tensor_dic.get("tensor_in_ub").op.axis[3], factor)
        cal_axis_dic["axis_di"] = axis_di
    else:
        axis_xo, axis_xi = schedule[tensor_dic.get("res")] \
            .split(tensor_dic.get("res").op.axis[axis], factor)
        cal_axis_dic["axis_xi"] = axis_xi
        cal_axis_dic["axis_xo"] = axis_xo
        _, axis_ei = schedule[tensor_dic.get("tensor_cal")] \
            .split(tensor_dic.get("tensor_cal").op.axis[axis], factor)
        cal_axis_dic["axis_ei"] = axis_ei
        _, axis_di = schedule[tensor_dic.get("tensor_in_ub")] \
            .split(tensor_dic.get("tensor_in_ub").op.axis[axis], factor)
        cal_axis_dic["axis_di"] = axis_di
    return cal_axis_dic, axis


def upsample_compute(schedule, cal_axis_dic, tensor_dic):
    """
    cal upsample compute
    Parameters
    ---------
    isint:
    schedule
    cal_axis_dic
    schedule:schedule
    tensor_dic:tensor dic
    Returns
    -------
    axis_list:cal axis list
    """

    schedule[tensor_dic.get("tensor_in_ub")] \
        .compute_at(schedule[tensor_dic.get("res")] \
                    , cal_axis_dic.get("axis_xo"))
    schedule[tensor_dic.get("tensor_cal")] \
        .compute_at(schedule[tensor_dic.get("res")] \
                    , cal_axis_dic.get("axis_xo"))

    schedule[tensor_dic.get("tensor_in_ub")].double_buffer()
    schedule[tensor_dic.get("tensor_cal")].double_buffer()

    axis_list = [cal_axis_dic.get("axis_di") \
        , cal_axis_dic.get("axis_ei"), cal_axis_dic.get("axis_xi")]
    return axis_list


def ins_emit(schedule, op_list, axis_list, ins_list):
    """
    when int8 or uint8 spilt  axis to cal
    Parameters
    ---------

    schedule:schedule
    op_list:ins list
    axis_list:axis list
    ins_list:ins list

    Returns
    -------
    NOne
"""
    length = len(op_list)
    for i in range(0, length):
        schedule[op_list[i]].emit_insn(axis_list[i], ins_list[i])

def bind_multcore(axis, x, schedule, res_op):
    '''
    :param axis:axis to spilt
    :param x: input x
    :param schedule: schedule
    :param res_op: res tensor
    :return:
    res_out:axis to bind core
    res_in
    '''
    shape_x = x.get("shape")
    n = shape_x[0]
    c1 = shape_x[1]
    device_core_num = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
    batch_factor = 1

    if axis == 1:
        if int(device_core_num) >= n:
            batch_factor = 1
        else:
            batch_factor = n // int(device_core_num)
        res_out, res_in = schedule[res_op].split(res_op.op.axis[0], batch_factor)
    elif axis in (2, 3):

        if int(device_core_num) >= n:
            if n * c1 <= 65535:
                fused_axis = schedule[res_op].fuse(res_op.op.axis[0], res_op.op.axis[1])
                if n*c1 <= int(device_core_num):
                    res_out, res_in = schedule[res_op].split(fused_axis, batch_factor)
                else:
                    batch_factor = (n*c1)//device_core_num

                    res_out, res_in = schedule[res_op].split(fused_axis, batch_factor)
            else:
                res_out, res_in = schedule[res_op].split(res_op.op.axis[0], batch_factor)
        else:
            batch_factor = n//int(device_core_num)

            res_out, res_in = schedule[res_op].split(res_op.op.axis[0], batch_factor)

    else:
        res_out, res_in = schedule[res_op].split(res_op.op.axis[0], batch_factor)
    return res_out, res_in

# pylint: disable=locally-disabled,too-many-arguments,invalid-name,too-many-locals
@util.check_input_type(dict, dict, (int, float), int, int, str)
def upsample(x, y, scale=1, stride_h=2, stride_w=2, kernel_name="upsample"):
    """
    calculating data

    Parameters
    ---------
    x : dict
        include shape dtype and format
    stride_h : int
        the shape change axis h
    stride_w : int
        the shape change axis w
    scale : float
        the value of tensor change axis, default value is 1
    y :output
    kernel_name : str
        kernel name, default value is "upsample"

    Returns
    -------
    None
    """
    upsample_check(x, stride_h, stride_w, kernel_name)
    dtype = x.get("dtype")
    op_list, ins_list, tensor_dic, feature, y \
        = gen_upsample(x, dtype, scale, stride_h, stride_w)
    schedule = tvm.create_schedule(y.op)
    buffer_mapping(schedule, op_list)
    tilling_spilt_axis_dic \
        = tilling_spilt_axis(schedule, tensor_dic, stride_h, stride_w)
    cal_axis_dic, axis \
        = cal_axis_spilt(x, stride_h, stride_w
                         , tilling_spilt_axis_dic, tensor_dic, schedule)

    axis_list = upsample_compute(schedule, cal_axis_dic, tensor_dic)
    res_op = tensor_dic.get("res")
    ins_emit(schedule, op_list, axis_list, ins_list)
    if axis == 0:
        schedule[y].bind(cal_axis_dic.get("axis_xo"), tvm.thread_axis("blockIdx.x"))
    else:
        res_out, _ = bind_multcore(axis, x, schedule, res_op)
        schedule[y].bind(res_out, tvm.thread_axis("blockIdx.x"))

    with build_config:
        # print(tvm.lower(schedule, [feature, y], simple_mode=True))
        tvm.build(schedule, [feature, y], "cce", name=kernel_name)
