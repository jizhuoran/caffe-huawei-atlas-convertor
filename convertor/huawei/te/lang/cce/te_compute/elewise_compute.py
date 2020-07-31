#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# pylint: disable=too-many-lines, import-error
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


elewise compute
"""
from decorator import decorator
from te import platform as cce
from te import tvm
from te.platform import intrinsic_check_support
from te.platform import get_soc_spec

from .cast_compute import _cast
from .broadcast_compute import broadcast
from .util import is_cast_support
from .util import judge_var
from .util import shape_to_list
from .util import auto_cast_tensor
from .util import get_tvm_scalar
from .util import dtype_check_decorator


NAME_INDEX = [0]

# pylint: disable=too-many-locals, too-many-branches, too-many-statements
@decorator
def auto_cast_of_elewise(func, *args, **kwargs):
    '''
    auto cast dectorator.
    Before calling elewise api, check the input tensor is supported by the intr.
    If not supported, casting the input tensor to supported dtype.
    (On condition that the cast type is supported.
    If the cast type is not supported,raising a RuntimeError).
    '''
    intr = func.__name__
    intr = _intrinsic_check(intr)

    is_support_fp32 = intrinsic_check_support("Intrinsic_"+intr, "float32")
    if len(args) == 1:
        if not isinstance(args[0], tvm.tensor.Tensor):
            raise RuntimeError("The first input type must be tvm.tensor")

        temp_tensor = args[0]
        dtype = temp_tensor.dtype
        is_support_dtype = intrinsic_check_support("Intrinsic_"+intr, dtype)
        if not is_support_dtype:
            if is_support_fp32 and is_cast_support(dtype, "float32"):
                temp_tensor = _cast(temp_tensor, "float32")
            else:
                temp_tensor = _cast(temp_tensor, "float16")
        return func(temp_tensor)
    if len(args) == 2:
        if not isinstance(args[0], tvm.tensor.Tensor):
            raise RuntimeError("The first input type must be tvm.tensor")

        if isinstance(args[1], tvm.tensor.Tensor):
            lhs = args[0]
            rhs = args[1]
            dtype_l = lhs.dtype
            dtype_r = rhs.dtype

            lhs_t = lhs
            rhs_t = rhs
            is_support_ldtype = intrinsic_check_support("Intrinsic_"+intr,
                                                        dtype_l)
            is_support_rdtype = intrinsic_check_support("Intrinsic_"+intr,
                                                        dtype_r)
            if not is_support_ldtype \
                    or not is_support_rdtype or dtype_l != dtype_r:
                if is_support_fp32 \
                        and is_cast_support(dtype_l, "float32") \
                        and is_cast_support(dtype_r, "float32"):
                    lhs_t = _cast(lhs, "float32")
                    rhs_t = _cast(rhs, "float32")
                else:
                    lhs_t = _cast(lhs, "float16")
                    rhs_t = _cast(rhs, "float16")

            return func(lhs_t, rhs_t)
        temp_tensor = args[0]
        scalar = args[1]
        dtype = temp_tensor.dtype
        is_support_dtype = intrinsic_check_support("Intrinsic_"+intr, dtype)
        if not is_support_dtype:
            if is_support_fp32 \
                    and is_cast_support(dtype, "float32"):
                temp_tensor = _cast(temp_tensor, "float32")
                dtype = "float32"
            else:
                temp_tensor = _cast(temp_tensor, "float16")
                dtype = "float16"

        tmp_arg = scalar
        scalar_type = judge_var(scalar)
        if scalar_type == "tvm_const" and scalar.dtype != dtype:
            tmp_arg = tvm.const(scalar.value, dtype=dtype)

        if scalar_type == "python_const":
            tmp_arg = tvm.const(scalar, dtype=dtype)
        return func(temp_tensor, tmp_arg)
    if len(args) == 3:
        if not isinstance(args[0], tvm.tensor.Tensor):
            raise RuntimeError("The first input type must be tvm.tensor")

        if not isinstance(args[1], tvm.tensor.Tensor):
            raise RuntimeError("The second input type must be tvm.tensor")

        if isinstance(args[2], tvm.tensor.Tensor):
            tensor_0 = args[0]
            tensor_1 = args[1]
            tensor_2 = args[2]

            dtype_0 = tensor_0.dtype
            dtype_1 = tensor_1.dtype
            dtype_2 = tensor_2.dtype

            tensor_0_t = tensor_0
            tensor_1_t = tensor_1
            tensor_2_t = tensor_2

            if dtype_0 != dtype_1 or dtype_0 != dtype_2 or dtype_2 != dtype_1:
                raise RuntimeError("Input tensors must has same dtype!")

            is_support_dtype0 = intrinsic_check_support("Intrinsic_"+intr,
                                                        dtype_0)
            if not is_support_dtype0:
                if is_support_fp32 \
                        and is_cast_support(dtype_0, "float32"):
                    tensor_0_t = _cast(tensor_0, "float32")
                    tensor_1_t = _cast(tensor_1, "float32")
                    tensor_2_t = _cast(tensor_2, "float32")
                else:
                    tensor_0_t = _cast(tensor_0, "float16")
                    tensor_1_t = _cast(tensor_1, "float16")
                    tensor_2_t = _cast(tensor_2, "float16")

            return func(tensor_0_t, tensor_1_t, tensor_2_t)
        lhs = args[0]
        rhs = args[1]
        scalar = args[2]
        dtype_l = lhs.dtype
        dtype_r = rhs.dtype

        lhs_t = lhs
        rhs_t = rhs
        is_support_ldtype = intrinsic_check_support("Intrinsic_"+intr, dtype_l)
        is_support_rdtype = intrinsic_check_support("Intrinsic_"+intr, dtype_r)
        if not is_support_ldtype \
                or not is_support_rdtype or dtype_l != dtype_r:
            if is_support_fp32 \
                    and is_cast_support(dtype_l, "float32") \
                    and is_cast_support(dtype_r, "float32"):
                lhs_t = _cast(lhs, "float32")
                rhs_t = _cast(rhs, "float32")
                dtype_l = "float32"
            else:
                lhs_t = _cast(lhs, "float16")
                rhs_t = _cast(rhs, "float16")
                dtype_l = "float16"

        tmp_arg = scalar
        if not isinstance(tmp_arg, str):
            scalar_type = judge_var(scalar)
            if scalar_type == "tvm_const" and scalar.dtype != dtype_l:
                tmp_arg = tvm.const(scalar.value, dtype=dtype_l)

            if scalar_type == "python_const":
                tmp_arg = tvm.const(scalar, dtype=dtype_l)
        return func(lhs_t, rhs_t, tmp_arg)
    return func(*args, **kwargs)


def _intrinsic_check(intr):
    ret_intr = intr
    if not intrinsic_check_support("Intrinsic_" + intr):
        if intr == "vdiv":
            ret_intr = "vrec"
        elif intr == "vsqrt":
            ret_intr = "vrsqrt"
        elif intr == "vlog":
            ret_intr = "vln"
        elif intr == "vmaxs":
            ret_intr = "vmax"
        elif intr == "vmins":
            ret_intr = "vmin"

    return ret_intr


@auto_cast_of_elewise
def vmuls(raw_tensor, scalar):
    """
    multiply a tensor by a scalar, dtype of raw_tensor
    and scalar must be the same

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    scalar : float, int, or tvm const

    Returns
    -------
    wrapped_tensor : raw_tensor*scalar
    """
    dtype = raw_tensor.dtype

    if isinstance(scalar, tvm.tensor.Tensor):
        raise RuntimeError("The second input type must be scalar")

    return __single_elewise_op(raw_tensor, dtype, 'elewise_single_VS_mul',
                               args=[scalar])


@auto_cast_of_elewise
def vadds(raw_tensor, scalar):
    """
    add a tensor by a scalar, dtype of raw_tensor and scalar must be the same

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    scalar : float, int, or tvm const

    Returns
    -------
    wrapped_tensor : raw_tensor + scalar
    """
    dtype = raw_tensor.dtype

    if isinstance(scalar, tvm.tensor.Tensor):
        raise RuntimeError("The second input type must be scalar")

    return __single_elewise_op(raw_tensor, dtype, 'elewise_single_VS_add',
                               args=[scalar])


@auto_cast_of_elewise
def vmaxs(raw_tensor, scalar):
    """
    Calculate elewise compare, return the max one of scalar or tensor's element,
    dtype of raw_tensor and scalar must be the same

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    scalar : float, int, or tvm const

    Returns
    -------
    wrapped_tensor : max(raw_tensor, scalar)
    """

    dtype = raw_tensor.dtype

    if isinstance(scalar, tvm.tensor.Tensor):
        raise RuntimeError("The second input type must be scalar")

    return __single_elewise_op(raw_tensor, dtype, 'elewise_single_VS_max',
                               args=[scalar])


@auto_cast_of_elewise
def vmins(raw_tensor, scalar):
    """
    Calculate elewise compare, return the min one of scalar or tensor's element,
     dtype of raw_tensor and scalar must be the same

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    scalar : float, int, or tvm const

    Returns
    -------
    wrapped_tensor : min(raw_tensor, scalar)
    """

    dtype = raw_tensor.dtype

    if isinstance(scalar, tvm.tensor.Tensor):
        raise RuntimeError("The second input type must be scalar")

    return __single_elewise_op(raw_tensor, dtype, 'elewise_single_VS_min',
                               args=[scalar])

def __vlog_calculate_by_taylor(data_x):
    """
    calculate ln(raw_tensor), use taylor expansion to calculate log
    """
    # pylint: disable=too-many-locals, too-many-statements
    from .common import cast_to

    # Log threshold
    const_log_threshold_1 = 0.6666666666666667
    const_log_threshold_2 = 0.3333333333333333
    # Log value
    log_four_three = 0.28768207245178085
    log_five_three = 0.5108256237659907
    log_five_two = 0.916290731874155
    # const value
    const_neg_one = -1
    const_one = 1
    const_two = 2
    const_one_three = 0.3333333333333333
    const_half = 1.0 / 2
    const_three_four = 0.75
    const_one_five = 0.2
    const_one_four_neg = -0.25
    const_five_two = 0.4
    const_dot_six = 0.6
    float_16_max = 32768
    const_half_neg = -0.5

    const_one_nine = 0.111111111111111
    const_one_eight_neg = -0.125
    const_one_seven = 0.142857142857143
    const_one_six_neg = -0.166666666666667

    dtype = data_x.dtype
    shape = data_x.shape

    def _taylor_compute(data):
        # pylint: disable=too-many-locals
        taylor_nine = vmuls(data, tvm.const(const_one_nine, dtype))
        taylor_eight_1 = vadds(taylor_nine,
                               tvm.const(const_one_eight_neg, dtype))
        taylor_eight_2 = vmul(taylor_eight_1, data)
        taylor_seven_1 = vadds(taylor_eight_2,
                               tvm.const(const_one_seven, dtype))
        taylor_seven_2 = vmul(taylor_seven_1, data)
        taylor_six_1 = vadds(taylor_seven_2,
                             tvm.const(const_one_six_neg, dtype))
        taylor_six_2 = vmul(taylor_six_1, data)
        taylor_five_1 = vadds(taylor_six_2, tvm.const(const_one_five, dtype))
        taylor_five_2 = vmul(taylor_five_1, data)
        taylor_four_1 = vadds(taylor_five_2,
                              tvm.const(const_one_four_neg, dtype))
        taylor_four_2 = vmul(taylor_four_1, data)
        taylor_three_1 = vadds(taylor_four_2, tvm.const(const_one_three, dtype))
        taylor_three_2 = vmul(taylor_three_1, data)
        taylor_two_1 = vadds(taylor_three_2,
                             tvm.const(const_half_neg, dtype))
        taylor_two_2 = vmul(taylor_two_1, data)
        taylor_one = vadds(taylor_two_2, tvm.const(const_one, dtype))
        taylor = vmul(taylor_one, data)

        return taylor

    def _log_compute_block_gt_2(data_x, res, shape):
        """
        when data > 2, use vlog directly
        when data > 32768, float16 will overflow, use log(x/2.5)+log(2.5)

        Parameters
        ----------
        data: input tensor that we want to calculate log

        Returns
        -------
        res : return of log

        """
        # pylint: disable=too-many-locals
        # if data > 2, use vlog
        threshold_3 = broadcast(tvm.const(const_two, dtype), shape)
        index_3 = vcmp(data_x, threshold_3, 'ge')
        res = vsel(index_3, vlog(data_x), res)
        # if data > 32768, use log(x/2.5)+log(2.5)
        float_16_max_tensor = broadcast(tvm.const(float_16_max, dtype), shape)
        index_4 = vcmp(data_x, float_16_max_tensor, 'ge')
        overflow_value = vmuls(data_x, const_five_two)
        res_overflow = vadds(vlog(overflow_value), log_five_two)
        res = vsel(index_4, res_overflow, res)
        res = cast_to(res, dtype)

        return res

    def _log_compute_block_lt_2_gt_1(data_x, shape):
        # pylint: disable=too-many-locals
        # phase1: index_1:data>(5/3)&&data<2
        data = vadds(data_x, tvm.const(const_neg_one, dtype))
        threshold_1 = broadcast(tvm.const(const_log_threshold_1, dtype), shape)
        index_1 = vcmp(data, threshold_1, 'ge')
        data_1 = vadds(data,
                       tvm.const(const_neg_one * const_log_threshold_1, dtype))
        data_sel = vsel(index_1, vmuls(data_1, tvm.const(const_dot_six, dtype)),
                        data)
        data_sel = cast_to(data_sel, dtype)

        # phase2:index_2:data>(4/3)&&data<(5/3)
        threshold_2 = broadcast(tvm.const(const_log_threshold_2, dtype), shape)
        index_2 = vcmp(data_sel, threshold_2, 'ge')
        data_2 = vadds(data_sel,
                       tvm.const(const_neg_one * const_log_threshold_2, dtype))
        data_vmuls = vmuls(data_2, tvm.const(const_three_four, dtype))
        data_sel = vsel(index_2, data_vmuls, data_sel)
        data_sel = cast_to(data_sel, dtype)

        # phase3: Taylor
        taylor = _taylor_compute(data_sel)

        # phase4:return back to original data
        # add log(4/3)
        res = vsel(index_2, vadds(taylor, tvm.const(log_four_three, dtype)),
                   taylor)
        res = cast_to(res, dtype)
        # add log(5/3)
        res = vsel(index_1, vadds(taylor, tvm.const(log_five_three, dtype)),
                   res)
        res = _cast(res, dtype)
        # d: vlog:

        return res

    def _log_compute_block_gt_1(data_x, shape):
        res = _log_compute_block_lt_2_gt_1(data_x, shape)
        res = _log_compute_block_gt_2(data_x, res, shape)

        return res

    def _log_compute_block_gt_half_lt_1(data_x, res, shape):
        threshold_5 = broadcast(tvm.const(const_one, dtype), shape)
        index_6 = vcmp(data_x, threshold_5, 'le')
        data = vadds(data_x, tvm.const(const_neg_one, dtype))
        taylor = _taylor_compute(data)
        res = vsel(index_6, taylor, res)
        res = cast_to(res, dtype)

        return res

    def _log_compute_block_lt_half(data_x, res, shape):
        threshold_4 = broadcast(tvm.const(const_half, dtype), shape)
        index_5 = vcmp(data_x, threshold_4, 'le')
        res = vsel(index_5, vmuls(_log_compute_block_gt_1(vrec(data_x), shape),
                                  const_neg_one), res)
        res = cast_to(res, dtype)

        return res

    res = _log_compute_block_gt_1(data_x, shape)

    res = _log_compute_block_gt_half_lt_1(data_x, res, shape)

    res = _log_compute_block_lt_half(data_x, res, shape)

    return res


@auto_cast_of_elewise
def vlog(raw_tensor, priority_flag=0):
    """
    calculate ln(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    priority_flag : priority flag, only support 1(precision) and 0(performance)

    Returns
    -------
    wrapped_tensor : log(raw_tensor)
    """

    if not intrinsic_check_support("Intrinsic_vln", "float32") \
            and priority_flag.value == 1.0:
        return __vlog_calculate_by_taylor(raw_tensor)

    dtype = raw_tensor.dtype
    return __single_elewise_op(raw_tensor, dtype, 'elewise_single_log')


@auto_cast_of_elewise
def vexp(raw_tensor):
    """
    calculate exp(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : exp(raw_tensor)
    """
    dtype = raw_tensor.dtype

    return __single_elewise_op(raw_tensor, dtype, 'elewise_single_exp')


@auto_cast_of_elewise
def vabs(raw_tensor):
    """
    calculate abs(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : abs(raw_tensor)
    """
    dtype = raw_tensor.dtype

    return __single_elewise_op(raw_tensor, dtype, 'elewise_single_abs')


@auto_cast_of_elewise
def vrec(raw_tensor):
    """
    calculate vrec(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : vrec(raw_tensor)
    """
    dtype = raw_tensor.dtype

    return __single_elewise_op(raw_tensor, dtype, 'elewise_single_rec')


@auto_cast_of_elewise
def vrelu(raw_tensor):
    """
    calculate vrelu(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : vrelu(raw_tensor)
    """
    dtype = raw_tensor.dtype

    return __single_elewise_op(raw_tensor, dtype, 'elewise_single_relu')


@auto_cast_of_elewise
def vnot(raw_tensor):
    """
    calculate vnot(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : vnot(raw_tensor)
    """
    dtype = raw_tensor.dtype

    return __single_elewise_op(raw_tensor, dtype, 'elewise_single_not')


def __vsqrt_calculate_by_newton(raw_tensor):
    """
    calculate vsqrt(raw_tensor), use newton iteration to calcuate sqrt

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : vsqrt(raw_tensor)
    """

    const_half = 1.0 / 2
    sqrt_const_iter = 3

    dtype = raw_tensor.dtype

    init_res = vlog(raw_tensor)
    init_res = vmuls(init_res, tvm.const(const_half))
    init_res = vexp(init_res)

    for _ in range(sqrt_const_iter):
        res = vdiv(raw_tensor, init_res)
        res = vadd(res, init_res)
        res = vmuls(res, tvm.const(const_half, dtype))
        init_res = res
    return res


@auto_cast_of_elewise
def vsqrt(raw_tensor, priority_flag=0):
    """
    calculate vsqrt(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    priority_flag: priority flag, only support 1(precision), 0(performance)

    Returns
    -------
    wrapped_tensor : vsqrt(raw_tensor)
    """

    if not intrinsic_check_support("Intrinsic_vsqrt"):
        if priority_flag.value == 1.0:
            return __vsqrt_calculate_by_newton(raw_tensor)
        dtype = raw_tensor.dtype
        res = __single_elewise_op(raw_tensor, dtype, 'elewise_single_rsqrt')
        return vrec(res)
    dtype = raw_tensor.dtype
    return __single_elewise_op(raw_tensor, dtype, 'elewise_single_sqrt')


def __vrsqrt_calculate_by_newton(raw_tensor):
    """
    calculate vrsqrt(raw_tensor), use newton iteration to calcuate sqrt

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : vrsqrt(raw_tensor)
    """

    const_half = 1.0 / 2
    sqrt_const_iter = 3

    dtype = raw_tensor.dtype

    init_res = vlog(raw_tensor)
    init_res = vmuls(init_res, tvm.const(const_half))
    init_res = vexp(init_res)

    for _ in range(sqrt_const_iter):
        res = vdiv(raw_tensor, init_res)
        res = vadd(res, init_res)
        res = vmuls(res, tvm.const(const_half, dtype))
        init_res = res
    return vrec(res)


@auto_cast_of_elewise
def vrsqrt(raw_tensor, priority_flag=0):
    """
    calculate vrsqrt(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : vrsqrt(raw_tensor)
    """

    if not intrinsic_check_support("Intrinsic_vsqrt") \
            and priority_flag.value == 1.0:
        return __vrsqrt_calculate_by_newton(raw_tensor)
    dtype = raw_tensor.dtype
    return __single_elewise_op(raw_tensor, dtype, 'elewise_single_rsqrt')


def check_elewise_single_shape(input_tensor):
    """
    check the input_tensor's shape whether is positive integer
    :param input_tensor
    """
    for i in range(len(input_tensor.shape)):
        if input_tensor.shape[i].value <= 0 \
                or isinstance(input_tensor.shape[i].value, int) is False:
            raise RuntimeError("The input shape value \
                               must be a positive integer")

# pylint: disable=too-many-locals, too-many-branches, too-many-statements
# pylint: cell-var-from-loop
def __single_elewise_op(input_tensor, dtype, op_name, args=None):
    """
    factory method of single elewise operations
    """
    check_elewise_single_shape(input_tensor)
    shape = shape_to_list(input_tensor.shape)
    if op_name == "elewise_single_log":
        lambda_func = lambda *indice: tvm.log(input_tensor(*indice))
    elif op_name == "elewise_single_exp":
        lambda_func = lambda *indice: tvm.exp(input_tensor(*indice))
    elif op_name == "elewise_single_rec":
        lambda_func = lambda *indice: 1 / input_tensor(*indice)
    elif op_name == "elewise_single_VS_add":
        lambda_func = lambda *indice: input_tensor(*indice) + args[0].astype(dtype)
    elif op_name == "elewise_single_VS_mul":
        lambda_func = lambda *indice: input_tensor(*indice) * args[0].astype(dtype)
    elif op_name == "elewise_single_VS_max":
        lambda_func = lambda *indice: tvm.max(input_tensor(*indice), args[0].astype(dtype))
    elif op_name == "elewise_single_VS_min":
        lambda_func = lambda *indice: tvm.min(input_tensor(*indice), args[0].astype(dtype))
    elif op_name == "elewise_single_abs":
        lambda_func = lambda *indice: tvm.select(input_tensor(*indice) >= 0, input_tensor(*indice),
                                                 - input_tensor(*indice))
    elif op_name == "elewise_single_relu":
        lambda_func = lambda *indice: tvm.select(input_tensor(*indice) >= 0, input_tensor(*indice),
                                                 tvm.const(0, dtype=dtype))
    elif op_name == "elewise_single_not":
        lambda_func = lambda *indice: ~input_tensor(*indice)
    elif op_name == "elewise_single_sqrt":
        lambda_func = lambda *indice: tvm.sqrt(input_tensor(*indice))
    elif op_name == "elewise_single_rsqrt":
        lambda_func = lambda *indice: tvm.rsqrt(input_tensor(*indice))
    elif op_name == "elewise_single_VS_max":
        temp_scalar = args[0].astype(dtype)
        lambda_func = lambda *indice: tvm.max(input_tensor(*indice), temp_scalar)
    elif op_name == "elewise_single_VS_min":
        temp_scalar = args[0].astype(dtype)
        lambda_func = lambda *indice: tvm.min(input_tensor(*indice), temp_scalar)
    else:
        raise RuntimeError("operation %s not support yet" % op_name)

    name = op_name.split("_")[-1] + "_" + str(NAME_INDEX[0])
    NAME_INDEX[0] += 1

    with tvm.tag_scope(op_name):
        tmp = tvm.compute(shape, lambda_func, name=name)

    if op_name == "elewise_single_rec":
        def __get_newton_iter_num():
            newton_iter_num = 2
            soc_ver = get_soc_spec("SOC_VERSION")
            if soc_ver in ("Ascend310",):
                newton_iter_num = 1
            return newton_iter_num

        newton_iter_num = __get_newton_iter_num()
        name_pre = op_name.split("_")[-1] + "_"
        const_num_neg_one = tvm.const(-1, dtype=dtype)
        const_num_two = tvm.const(2, dtype=dtype)

        # newton iteration formula is x(n) = x(n-1)(2 - ax(n-1))
        for _ in range(newton_iter_num):
            # the name of each compute
            name = name_pre + str(NAME_INDEX[0])
            NAME_INDEX[0] += 1
            # compute tmp_mul = a*x(n-1)
            with tvm.tag_scope("elewise_binary_mul"):
                tmp_mul = tvm.compute(
                    shape,
                    lambda *indice: input_tensor(*indice) * tmp(*indice),
                    name=name)

            name = name_pre + str(NAME_INDEX[0])
            NAME_INDEX[0] += 1
            # compute tmp_negative = -1*temp_mul
            # pylint: disable=cell-var-from-loop
            with tvm.tag_scope("elewise_single_VS_mul"):
                tmp_negative = tvm.compute(
                    shape,
                    lambda *indice: tmp_mul(*indice) * const_num_neg_one,
                    name=name)

            name = name_pre + str(NAME_INDEX[0])
            NAME_INDEX[0] += 1
            # compute tmp_plus = 2 + tmp_negative
            # pylint: disable=cell-var-from-loop
            with tvm.tag_scope("elewise_single_VS_add"):
                tmp_plus = tvm.compute(
                    shape,
                    lambda *indice: tmp_negative(*indice) + const_num_two,
                    name=name)
            name = name_pre + str(NAME_INDEX[0])
            NAME_INDEX[0] += 1
            # compute tmp = x(n-1)*tmp_plus
            # pylint: disable=cell-var-from-loop
            with tvm.tag_scope("elewise_binary_mul"):
                tmp = tvm.compute(shape,
                                  lambda *indice: tmp_plus(*indice) * tmp(*indice),
                                  name=name)

    return tmp


@auto_cast_of_elewise
def vmul(lhs, rhs):
    """
    calculate elewise multiply

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        left hand tensor

    rhs : wrapped_tensor or tvm.tensor
        right hand tensor

    Returns
    -------
    wrapped_tensor : lhs*rhs
    """
    if not isinstance(rhs, tvm.tensor.Tensor):
        raise RuntimeError("The second input type must be tvm.tensor")

    return __binary_elewise_op(lhs, rhs, "elewise_binary_mul")


@auto_cast_of_elewise
def vdiv(lhs, rhs):
    """
    calculate elewise div

    Parameters
    -----
    lhs: wrapped_tensor or tvm.tensor
         divisor tensor
    rhs: wrapped_tensor or tvm.tensor
         divided tensor

    returns
    -----
    wrapped_tensor: lhs / rhs
    """
    if not isinstance(rhs, tvm.tensor.Tensor):
        raise RuntimeError("The second input type must be tvm.tensor")

    if not intrinsic_check_support("Intrinsic_vdiv"):
        dtype = rhs.dtype
        reciprocal_rhs = __single_elewise_op(rhs, dtype, 'elewise_single_rec')
        vdiv_value = __binary_elewise_op(lhs, reciprocal_rhs, "elewise_binary_mul")
        return vdiv_value

    return __binary_elewise_op(lhs, rhs, "elewise_binary_div")

def __vmod_small_hisi(lhs, rhs):
    from .cast_compute import floor
    # small hisi
    dtype = lhs.dtype
    res_div = vdiv(lhs, rhs)
    res_floor = floor(res_div)
    res_floor = _cast(res_floor, dtype)
    res_mul = vmul(rhs, res_floor)
    res = vsub(lhs, res_mul)

    return _cast(res, dtype)


def __vmod_cloud(lhs, rhs):
    from .cast_compute import floor
    # cloud
    dtype = lhs.dtype
    lhs = _cast(lhs, "float32")
    rhs = _cast(rhs, "float32")
    res_div = vdiv(lhs, rhs)
    res_floor = floor(res_div)
    res_floor = _cast(res_floor, "float32")
    res_mul = vmul(rhs, res_floor)
    res = vsub(lhs, res_mul)

    return _cast(res, dtype)

# pylint: disable=too-many-locals
def __vmod_mini(lhs, rhs):
    from .cast_compute import floor
    dtype = rhs.dtype
    rhs_f16 = rhs
    # 1. calculate result for testing, using float32 for better precision
    lhs = _cast(lhs, "float32")
    rhs = _cast(rhs, "float32")
    test_div = vmul(lhs, vrec(rhs))
    test_floor = _cast(floor(test_div), "float32")
    test_res = vsub(lhs, vmul(rhs, test_floor))

    # 2. correct the floor result, using float16
    test_res = _cast(test_res, dtype)
    test_floor = _cast(test_floor, dtype)
    zero = broadcast(0.0, lhs.shape, dtype)

    # rhs positive: 0 <= res < rhs
    prhs = vcmp(test_res, zero, 'lt', mode='bool')
    prhs_floor = vsel(prhs, vadds(test_floor, -1.0), test_floor)
    # rhs negative: rhs < res <= 0
    nrhs = vcmp(test_res, zero, 'gt', mode='bool')
    nrhs_floor = vsel(nrhs, vadds(test_floor, -1.0), test_floor)

    # according to positive and negative rhs to choose p_floor or n_floor
    rhs_f16_gt_zero = vcmp(rhs_f16, zero, 'gt', mode='bool')
    result_floor = vsel(rhs_f16_gt_zero, prhs_floor, nrhs_floor)

    # 3. calculate the final result, using float32 for better precision
    result_floor = _cast(result_floor, "float32")
    result = vsub(lhs, vmul(rhs, result_floor))

    return _cast(result, dtype)


@dtype_check_decorator
def vmod(lhs, rhs):
    """
    calculate element-wise remainder of division

    Parameters
    -----
    lhs : wrapped_tensor or tvm.tensor
          left hand tensor

    rhs : wrapped_tensor or tvm.tensor
          right hand tensor

    Returns
    -----
    wrapped_tensor : lhs - floor(lhs/rhs) * rhs
    """
    if not isinstance(lhs, tvm.tensor.Tensor):
        raise RuntimeError("The first input type must be tvm.tensor")
    if not isinstance(rhs, tvm.tensor.Tensor):
        raise RuntimeError("The second input type must be tvm.tensor")

    check_elewise_binary_shape(lhs, rhs)
    if lhs.dtype != rhs.dtype:
        raise RuntimeError("dtype must be the same while lhType is %s, "
                           "rhType is %s" % (lhs.dtype, rhs.dtype))

    # cloud using vdiv. mini using vrec for division calculation,
    # and mini should improve vmod calculation accuracy.
    if (not intrinsic_check_support("Intrinsic_vdiv")) and \
       (not intrinsic_check_support("Intrinsic_vconv", "f322s32f")):
        if lhs.dtype not in ("float16", ):
            raise RuntimeError("dtype must be float16.")
        res = __vmod_mini(lhs, rhs)
    elif not intrinsic_check_support("Intrinsic_vconv", "f322s32f"):
        if lhs.dtype not in ("float16", ):
            raise RuntimeError("dtype must be float16.")
        res = __vmod_small_hisi(lhs, rhs)
    else:
        if lhs.dtype not in ("float16", "float32"):
            raise RuntimeError("dtype must be float16 or float32.")
        res = __vmod_cloud(lhs, rhs)

    return res


@auto_cast_of_elewise
def vadd(lhs, rhs):
    """
    calculate elewise add

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        left hand tensor

    rhs : wrapped_tensor or tvm.tensor
        left hand tensor

    Returns
    -------
    wrapped_tensor : lhs + rhs
    """

    if not isinstance(rhs, tvm.tensor.Tensor):
        raise RuntimeError("The second input type must be tvm.tensor")

    def is_conv_oper(tensor):
        if hasattr(tensor.op, "reduce_axis") and len(tensor.op.reduce_axis) == 2 and \
                hasattr(tensor.op, "tag")  and "conv" in tensor.op.tag:
            return True
        if tensor.op.input_tensors:
            for input_tensor in tensor.op.input_tensors:
                return is_conv_oper(input_tensor)
        else:
            return False

    if is_conv_oper(rhs):
        return __binary_elewise_op(rhs, lhs, "elewise_binary_add")

    return __binary_elewise_op(lhs, rhs, "elewise_binary_add")


@auto_cast_of_elewise
def vsub(lhs, rhs):
    """
    calculate elewise sub

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        left hand tensor

    rhs : wrapped_tensor or tvm.tensor
        left hand tensor

    Returns
    -------
    wrapped_tensor : lhs - rhs
    """

    if not isinstance(rhs, tvm.tensor.Tensor):
        raise RuntimeError("The second input type must be tvm.tensor")

    return __binary_elewise_op(lhs, rhs, "elewise_binary_sub")


@auto_cast_of_elewise
def vmin(lhs, rhs):
    """
    calculate elewise compare, return the min one
    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        left hand tensor
    rhs : wrapped_tensor or tvm.tensor
        left hand tensor
    Returns
    -------
    wrapped_tensor : min(lhs , rhs)
    """
    if not isinstance(rhs, tvm.tensor.Tensor):
        raise RuntimeError("The second input type must be tvm.tensor")

    return __binary_elewise_op(lhs, rhs, "elewise_binary_min")


@auto_cast_of_elewise
def vmax(lhs, rhs):
    """
    calculate elewise compare, return the min one
    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        left hand tensor
    rhs : wrapped_tensor or tvm.tensor
        left hand tensor
    Returns
    -------
    wrapped_tensor : max(lhs , rhs)
    """
    if not isinstance(rhs, tvm.tensor.Tensor):
        raise RuntimeError("The second input type must be tvm.tensor")

    return __binary_elewise_op(lhs, rhs, "elewise_binary_max")


@auto_cast_of_elewise
def vor(lhs, rhs):
    """
    calculate bitwise or op, return the or value
    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        left hand tensor
    rhs : wrapped_tensor or tvm.tensor
        left hand tensor
    Returns
    -------
    wrapped_tensor : or(lhs , rhs)
    """
    if not isinstance(rhs, tvm.tensor.Tensor):
        raise RuntimeError("The second input type must be tvm.tensor")

    return __binary_elewise_op(lhs, rhs, "elewise_binary_or")


@auto_cast_of_elewise
def vand(lhs, rhs):
    """
    calculate bitwise and op, return the and value
    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        left hand tensor
    rhs : wrapped_tensor or tvm.tensor
        left hand tensor
    Returns
    -------
    wrapped_tensor : max(lhs , rhs)
    """
    if not isinstance(rhs, tvm.tensor.Tensor):
        raise RuntimeError("The second input type must be tvm.tensor")

    return __binary_elewise_op(lhs, rhs, "elewise_binary_and")

@auto_cast_of_elewise
def vaxpy(lhs, rhs, scalar):
    """
    calculate elewise scalar*lhs + rhs, return the min one
    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        left hand tensor
    rhs : wrapped_tensor or tvm.tensor
        left hand tensor
    Returns
    -------
    wrapped_tensor : max(lhs , rhs)
    """
    if isinstance(scalar, tvm.tensor.Tensor):
        raise RuntimeError("The third input type must be scalar")

    return __binary_elewise_op(lhs, rhs, "elewise_binary_scalar_axpy",
                               args=[scalar])

def _vcmp_supported_types(mode):
    supported_types = None
    # the get_cmpmask need 16b aligned, so if is float32, should cast to float16
    # bit model using vcmpv. v200 support float32. v100 only support float16
    if mode == 'bit':
        supported_types = ['float16']

    return supported_types

# pylint: disable=too-many-branches, too-many-statements
def vcmp(lhs, rhs, operation='lt', mode='bool'):
    """
    calculate elewise compare

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        left hand tensor

    rhs : wrapped_tensor or tvm.tensor
        right hand tensor

    operation : operator type, eq, ne, lt, gt, ge, le

    mode : bool, the dtype of return value is bool
           bit, the dtype of return value is uint8

    Returns
    -------
    wrapped_tensor
    """
    if not isinstance(lhs, tvm.tensor.Tensor):
        raise RuntimeError("The input type must be tvm.tensor")

    if operation not in ['eq', 'ne', 'lt', 'gt', 'ge', 'le']:
        raise RuntimeError("The op's value must be eq, ne, lt, gt, ge, le")

    if mode not in ['bool', 'bit']:
        raise RuntimeError("The op's mode must be bit and bool")

    shape = shape_to_list(lhs.shape)
    if mode == 'bit' and shape[-1] % 8 != 0:
        raise RuntimeError("in bit mode the last dim must be mutiply of 8")

    supported_types = _vcmp_supported_types(mode)

    # the output is bool or uint8, is not the same as input,
    # no need to cast to back in auto schedule
    if isinstance(rhs, tvm.tensor.Tensor):
        if lhs.dtype != rhs.dtype:
            raise RuntimeError("dtype must be the same while lhs "
                               "is %s, rhs is %s" % (lhs.dtype, rhs.dtype))
        lhs = auto_cast_tensor(lhs, 'vcmp', supported_types, is_auto_cast=False)
        rhs = auto_cast_tensor(rhs, 'vcmp', supported_types, is_auto_cast=False)
    else:
        lhs = auto_cast_tensor(lhs, 'vcmp', supported_types, is_auto_cast=False)
        rhs = get_tvm_scalar(rhs, lhs.dtype)

    cmp_op = "emit_insn_elewise_binary_cmp"
    if isinstance(rhs, tvm.tensor.Tensor):
        if operation == 'lt':
            lambda_func = lambda *indice: lhs(*indice) < rhs(*indice)
        elif operation == 'gt':
            lambda_func = lambda *indice: lhs(*indice) > rhs(*indice)
        elif operation == 'le':
            lambda_func = lambda *indice: lhs(*indice) <= rhs(*indice)
        elif operation == 'ge':
            lambda_func = lambda *indice: lhs(*indice) >= rhs(*indice)
        elif operation == 'eq':
            lambda_func = lambda *indice: \
                tvm.expr.EQ(lhs(*indice), rhs(*indice))
        elif operation == 'ne':
            lambda_func = lambda *indice: \
                tvm.expr.NE(lhs(*indice), rhs(*indice))
        else:
            raise RuntimeError("vcmp do not support the input op" % operation)
    else:
        if operation == 'lt':
            lambda_func = lambda *indice: lhs(*indice) < rhs
        elif operation == 'gt':
            lambda_func = lambda *indice: lhs(*indice) > rhs
        elif operation == 'le':
            lambda_func = lambda *indice: lhs(*indice) <= rhs
        elif operation == 'ge':
            lambda_func = lambda *indice: lhs(*indice) >= rhs
        elif operation == 'eq':
            lambda_func = lambda *indice: tvm.expr.EQ(lhs(*indice), rhs)
        elif operation == 'ne':
            lambda_func = lambda *indice: tvm.expr.NE(lhs(*indice), rhs)
        else:
            raise RuntimeError("vcmp do not support the input op" % operation)

    name = cmp_op.split("_")[-1] + "_" + str(NAME_INDEX[0])
    NAME_INDEX[0] += 1

    if mode == 'bit':
        shape = shape_to_list(lhs.shape)
        if shape[-1] % 8 != 0:
            raise RuntimeError("The input shape's "
                               "last axis must be multiple of 8")

        k = tvm.reduce_axis((0, 8), name='k')
        res_shape = shape
        res_shape[-1] = res_shape[-1] // 8

        def _compute(*index):
            res_index = []
            for i, value in enumerate(index):
                if i == len(index) - 1:
                    res_index.append(value * 8 + k)
                else:
                    res_index.append(value)
            tensor = tvm.bit(lambda_func(*res_index).astype('uint8'), axis=k)
            return tensor

        cmp_op = cmp_op + "|" + operation + "|" + mode

        with tvm.tag_scope(cmp_op):
            output = tvm.compute(res_shape, _compute, name='output')
        return output

    cmp_op = cmp_op + "|" + operation + "|" + mode

    with tvm.tag_scope(cmp_op):
        tmp = tvm.compute(shape, lambda_func, name=name)

    return tmp


@dtype_check_decorator
def vlogic(lhs, rhs=None, operation='logic_and'):
    """
    calculate elewise logic operation

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        left hand tensor

    rhs : wrapped_tensor or tvm.tensor
        right hand tensor

    operation : operator type, logic_and, logic_or, logic_not

    Returns
    -------
    wrapped_tensor
    """

    if operation not in ['logic_and', 'logic_or', 'logic_not']:
        raise RuntimeError("The op's value must be logic_and, \
            logic_or, logic_not, the op is %s" % operation)

    if not isinstance(lhs, tvm.tensor.Tensor):
        raise RuntimeError("The lhs input type must be Tensor")

    if operation != "logic_not" and not isinstance(rhs, tvm.tensor.Tensor):
        raise RuntimeError("The rhs input type must be Tensor")

    if operation == "logic_not":
        rhs = tvm.placeholder(lhs.shape, name="rhs", dtype=lhs.dtype)
    # the output is bool is not the same as input,
    # no need to cast to back in auto schedule
    return __binary_elewise_op(lhs, rhs, "elewise_binary_logic", args=[operation[6:]])

# pylint: disable=too-many-locals, too-many-branches, too-many-statements
# pylint: consider-using-in
def __binary_elewise_op(tensor_l, tensor_r, op_name, args=None):
    """
    factory method of binary elewise operations
    """
    check_elewise_binary_shape(tensor_l, tensor_r)
    if tensor_l.dtype != tensor_r.dtype and op_name != "elewise_binary_scalar_axpy":
        raise RuntimeError("dtype must be the same while lhType \
                is %s, rhType is %s" % (tensor_l.dtype, tensor_r.dtype))
    shape = tensor_l.shape
    if op_name == "elewise_binary_add":
        lambda_func = lambda *indice: tensor_l(*indice) + tensor_r(*indice)
    elif op_name == "elewise_binary_sub":
        lambda_func = lambda *indice: tensor_l(*indice) - tensor_r(*indice)
    elif op_name == "elewise_binary_div":
        lambda_func = lambda *indice: tensor_l(*indice) / tensor_r(*indice)
    elif op_name == "elewise_binary_mul":
        lambda_func = lambda *indice: tensor_l(*indice) * tensor_r(*indice)
    elif op_name == "elewise_binary_min":
        lambda_func = lambda *indice: \
            tvm.min(tensor_l(*indice), tensor_r(*indice))
    elif op_name == "elewise_binary_max":
        lambda_func = lambda *indice: \
            tvm.max(tensor_l(*indice), tensor_r(*indice))
    elif op_name == "elewise_binary_and":
        lambda_func = lambda *indice: tensor_l(*indice) & tensor_r(*indice)
    elif op_name == "elewise_binary_or":
        lambda_func = lambda *indice: tensor_l(*indice) | tensor_r(*indice)
    elif op_name == "elewise_binary_vcmpv_le":
        lambda_func = lambda *indice: tensor_l(*indice) <= tensor_r(*indice)
    elif op_name == "elewise_binary_vcmpv_lt":
        lambda_func = lambda *indice: tensor_l(*indice) < tensor_r(*indice)
    elif op_name == "elewise_binary_vcmpv_ge":
        lambda_func = lambda *indice: tensor_l(*indice) >= tensor_r(*indice)
    elif op_name == "elewise_binary_vcmpv_gt":
        lambda_func = lambda *indice: tensor_l(*indice) > tensor_r(*indice)
    elif op_name == "elewise_binary_vcmpv_ne":
        lambda_func = lambda *indice: tensor_l(*indice) != tensor_r(*indice)
    elif op_name == "elewise_binary_vcmpv_eq":
        lambda_func = lambda *indice: tensor_l(*indice) == tensor_r(*indice)
    elif op_name == "elewise_binary_scalar_axpy":
        intr = "v" + op_name.split("_")[-1]
        is_support_dtype = intrinsic_check_support("Intrinsic_"+intr,
                                                   tensor_l.dtype)
        if tensor_l.dtype != tensor_r.dtype:
            if tensor_l.dtype != "float16" or tensor_r.dtype != "float32":
                raise RuntimeError("dtype error, vaxpy not support mixed data type auto cast")
        elif not is_support_dtype:
            raise RuntimeError("dtype error, vaxpy not support mixed data type auto cast")
        rtype = tensor_r.dtype
        lambda_func = lambda *indice: \
            tvm.expr.Cast(rtype, tensor_l(*indice))*args[0].astype(rtype) + \
            tensor_r(*indice)
    elif op_name == "emit_insn_elewise_binary_cmp":
        operation = args[0]
        if operation == 'lt':
            lambda_func = lambda *indice: tensor_l(*indice) < tensor_r(*indice)
        elif operation == 'gt':
            lambda_func = lambda *indice: tensor_l(*indice) > tensor_r(*indice)
        elif operation == 'le':
            lambda_func = lambda *indice: tensor_l(*indice) <= tensor_r(*indice)
        elif operation == 'ge':
            lambda_func = lambda *indice: tensor_l(*indice) >= tensor_r(*indice)
        elif operation == 'eq':
            lambda_func = lambda *indice: \
                tvm.expr.EQ(tensor_l(*indice), tensor_r(*indice))
        elif operation == 'ne':
            lambda_func = lambda *indice: \
                tvm.expr.NE(tensor_l(*indice), tensor_r(*indice))
        else:
            raise RuntimeError("vcmp do not support the input op_name" % operation)
    elif op_name == "elewise_binary_logic":
        operation = args[0]
        if operation == 'and':
            lambda_func = lambda *indice: tensor_l(*indice) & tensor_r(*indice)
        elif operation == 'or':
            lambda_func = lambda *indice: tensor_l(*indice) | tensor_r(*indice)
        elif operation == 'not':
            lambda_func = lambda *indice: ~tensor_l(*indice)
        else:
            raise RuntimeError("vlogic do not support the input op_name")
    else:
        raise RuntimeError("operation %s not support yet" % op_name)

    name = op_name.split("_")[-1] + "_" + str(NAME_INDEX[0])
    NAME_INDEX[0] += 1

    if op_name == "emit_insn_elewise_binary_cmp" and args[1] == 'bit':
        shape = shape_to_list(shape)
        if shape[-1] % 8 != 0:
            raise RuntimeError("The input shape's \
                               last axis must be multiple of 8")

        k = tvm.reduce_axis((0, 8), name='k')
        res_shape = shape
        res_shape[-1] = res_shape[-1] // 8

        def _compute(*index):
            """
            elewise compare for bit
            """
            res_index = []
            for i, value in enumerate(index):
                if i == len(index) - 1:
                    res_index.append(value*8 + k)
                else:
                    res_index.append(value)
            tensor = tvm.bit(lambda_func(*res_index).astype('uint8'), axis=k)
            return tensor

        op_name = op_name + "|" + args[0] + "|" + args[1]

        with tvm.tag_scope(op_name):
            output = tvm.compute(res_shape, _compute, name='output')
        return output

    if op_name in ("emit_insn_elewise_binary_cmp",
                   "elewise_binary_logic"):
        for arg in args:
            op_name = op_name + "|" + arg

    with tvm.tag_scope(op_name):
        tmp = tvm.compute(shape, lambda_func, name=name)

    return tmp


def check_elewise_binary_shape(lhs, rhs):
    """
    check elewise binary shape
    :param lhs: left tensor
    :param rhs: right tensor
    :return:
    """
    if len(lhs.shape) != len(rhs.shape):
        raise RuntimeError("The lhs shape ndim %d must \
            be equal to the rhs %d" % (len(lhs.shape), len(rhs.shape)))

    for i in range(len(lhs.shape)):
        if lhs.shape[i].value != rhs.shape[i].value:
            raise RuntimeError("The lhs shape must be equal to the rhs")

    for sh_value in lhs.shape:
        if sh_value.value <= 0 \
                or not isinstance(sh_value.value, int):
            raise RuntimeError("The input shape value \
                                must be a positive integer")


def check_is_equal(lhs, rhs):
    """
    check lhs and rhs value is equal
    :param lhs: left tensor
    :param rhs: right tensor
    :return:
    """
    if lhs.value == rhs.value:
        raise RuntimeError("when lhs and rhs \
                           are all scalar, lhs should unequal to rhs")


@auto_cast_of_elewise
def vmla(tensor_0, tensor_1, tensor_2):
    """
    calculate x*tensor_1 + tensor_2,  only support float16, float32
    Parameters
    ----------
    x : wrapped_tensor or tvm.tensor
    tensor_1 : wrapped_tensor or tvm.tensor
    tensor_2 : wrapped_tensor or tvm.tensor
    Returns
    -------
    wrapped_tensor : X*tensor_1 + tensor_2
    """
    if not isinstance(tensor_2, tvm.tensor.Tensor):
        raise RuntimeError("The third input type must be tvm.tensor")

    return __multiple_elewise_op(tensor_0, tensor_1, tensor_2,
                                 "elewise_multiple_mla")

@auto_cast_of_elewise
def vmadd(tensor_0, tensor_1, tensor_2):
    """
    calculate tensor_0*tensor_2 + tensor_1,  only support  float16, float32
    Parameters
    ----------
    tensor_0 : wrapped_tensor or tvm.tensor
    tensor_1 : wrapped_tensor or tvm.tensor
    tensor_2 : wrapped_tensor or tvm.tensor
    Returns
    -------
    wrapped_tensor : tensor_0*tensor_2 + tensor_1
    """
    if not isinstance(tensor_2, tvm.tensor.Tensor):
        raise RuntimeError("The third input type must be tvm.tensor")

    return __multiple_elewise_op(tensor_0, tensor_1, tensor_2,
                                 "elewise_multiple_madd")

@auto_cast_of_elewise
def vmaddrelu(tensor_0, tensor_1, tensor_2):
    """
    calculate relu(tensor_0*tensor_2 + tensor_1), only support  float16, float32
    Parameters
    ----------
    tensor_0 : wrapped_tensor or tvm.tensor
    tensor_1 : wrapped_tensor or tvm.tensor
    tensor_2 : wrapped_tensor or tvm.tensor
    Returns
    -------
    wrapped_tensor : relu(tensor_0*tensor_2 + tensor_1)
    """
    if not isinstance(tensor_2, tvm.tensor.Tensor):
        raise RuntimeError("The third input type must be tvm.tensor")

    return __multiple_elewise_op(tensor_0, tensor_1, tensor_2,
                                 "elewise_multiple_maddrelu")


def __multiple_elewise_op(tensor_0, tensor_1, tensor_2, op_name):
    """
    factory method of binary multiple operations
    """
    intr = "v" + op_name.split("_")[-1]
    is_support_dtype = intrinsic_check_support("Intrinsic_"+intr,
                                               tensor_0.dtype)

    check_multiple_elewise_op_shape(tensor_0, tensor_1, tensor_2)
    if tensor_0.dtype != tensor_1.dtype or tensor_0.dtype != tensor_2.dtype \
            or tensor_2.dtype != tensor_1.dtype:
        if op_name != "elewise_multiple_mla" \
                or tensor_0.dtype != tensor_1.dtype \
                or tensor_0.dtype != "float16" \
                or tensor_2.dtype != "float32":
            raise RuntimeError("dtype error, vmla not support mixed data type auto cast")
    elif not is_support_dtype:
        raise RuntimeError("dtype error, vmla not support mixed data type auto cast")

    shape = tensor_0.shape
    if op_name == "elewise_multiple_mla":
        ztype = tensor_2.dtype
        lambda_func = lambda *indice: tvm.expr.Cast(ztype, \
            tensor_0(*indice) * tensor_1(*indice)) + tensor_2(*indice)
    elif op_name == "elewise_multiple_madd":
        lambda_func = lambda *indice: tensor_0(*indice) * tensor_2(*indice) + tensor_1(*indice)
    elif op_name == "elewise_multiple_maddrelu":
        lambda_func = lambda *indice: \
            tvm.relu(tensor_0(*indice) * tensor_2(*indice) + tensor_1(*indice))
    else:
        raise RuntimeError("operation %s not support yet" % op_name)

    name = op_name.split("_")[-1] + "_" + str(NAME_INDEX[0])
    NAME_INDEX[0] += 1

    # list cases of same input, "1"s in same_list means same input
    if tensor_0 == tensor_1 and tensor_0 == tensor_2:
        same_list = [1, 1, 1]
    elif tensor_0 == tensor_1:
        same_list = [1, 1, 0]
    elif tensor_0 == tensor_2:
        same_list = [1, 0, 1]
    elif tensor_1 == tensor_2:
        same_list = [0, 1, 1]
    else:
        same_list = [0, 0, 0]

    str_same_list = ",".join([str(i) for i in same_list])
    with tvm.tag_scope(op_name + '|' + str_same_list):
        tmp = tvm.compute(shape, lambda_func, name=name)

    return tmp


def check_multiple_elewise_op_shape(tensor_0, tensor_1, tensor_2):
    """
    check multiple elewise op's shape
    :param tensor_0:
    :param tensor_1:
    :param tensor_2:
    :return:
    """
    if len(tensor_0.shape) != len(tensor_1.shape) or len(tensor_0.shape) != len(tensor_2.shape) \
            or len(tensor_2.shape) != len(tensor_1.shape):
        raise RuntimeError("The input shape ndim \
                           must be equal to the each other")

    for i in range(len(tensor_0.shape)):
        if tensor_0.shape[i].value != tensor_1.shape[i].value or \
                        tensor_0.shape[i].value != tensor_2.shape[i].value or \
                        tensor_1.shape[i].value != tensor_2.shape[i].value:
            raise RuntimeError("The input shape \
                               must be equal to the each other")

    for i in range(len(tensor_0.shape)):
        if tensor_0.shape[i].value <= 0 or isinstance(tensor_0.shape[i].value, int) is False:
            raise RuntimeError("The input shape value \
                               must be a positive integer")


def vsel_bit_shape_check(condition, input_tensor):
    """
    check vsel_bit's shape
    :param condition:
    :param input_tensor:
    :return:
    """
    if len(condition.shape) != len(input_tensor.shape):
        raise RuntimeError("The condition shape ndim %d \
            must be equal to the input_tensor %d" %
                           (len(condition.shape), len(input_tensor.shape)))

    for i in range(len(condition.shape)):
        if i == len(condition.shape) - 1:
            if (input_tensor.shape[i].value % 8 != 0) \
                    or (input_tensor.shape[i].value // 8
                            != condition.shape[i].value):
                raise RuntimeError(
                    "the sel tensor's last dim must be multiple of 8 \
                    and div the last dim of condition shape is 8")
        else:
            if condition.shape[i].value != input_tensor.shape[i].value:
                raise RuntimeError("The lhs shape must be equal to the rhs")

    for i in range(len(input_tensor.shape)):
        if input_tensor.shape[i].value <= 0 \
                or isinstance(input_tensor.shape[i].value, int) is False:
            raise RuntimeError("The input shape value \
                               must be a positive integer")

# pylint: disable=too-many-branches, too-many-statements
def vsel(condition, lhs, rhs):
    """
    if condition = ture, the result is lhs,
        select

    Parameters
    ----------
    condition : wrapped_tensor or tvm.tensor, the dtype is bool or uint8

    lhs : wrapped_tensor or tvm.tensor or scalar

    rhs : wrapped_tensor or tvm.tensor or scalar

    Returns
    -------
    wrapped_tensor :
    """
    if not isinstance(condition, tvm.tensor.Tensor):
        raise RuntimeError("The condition type must be tvm.tensor")

    src_dtype = "float16"
    op_name = "emit_insn_elewise_multiple_sel"
    if condition.dtype == "bool":
        mode = 'bool'
        shape = condition.shape
        if isinstance(lhs, tvm.tensor.Tensor) \
                and isinstance(rhs, tvm.tensor.Tensor):
            src_dtype = lhs.dtype
            lhs = auto_cast_tensor(lhs, 'vsel')
            rhs = auto_cast_tensor(rhs, 'vsel')
            check_multiple_elewise_op_shape(condition, lhs, rhs)
            lambda_func = lambda *indice: \
                tvm.select(condition(*indice), lhs(*indice), rhs(*indice))
        elif not isinstance(lhs, tvm.tensor.Tensor) \
                and isinstance(rhs, tvm.tensor.Tensor):
            check_elewise_binary_shape(condition, rhs)
            src_dtype = rhs.dtype
            rhs = auto_cast_tensor(rhs, 'vsel')
            lhs = get_tvm_scalar(lhs, rhs.dtype)
            lambda_func = lambda *indice: \
                tvm.select(condition(*indice), lhs, rhs(*indice))
        elif isinstance(lhs, tvm.tensor.Tensor) \
                and not isinstance(rhs, tvm.tensor.Tensor):
            check_elewise_binary_shape(condition, lhs)
            src_dtype = lhs.dtype
            lhs = auto_cast_tensor(lhs, 'vsel')
            rhs = get_tvm_scalar(rhs, lhs.dtype)
            lambda_func = lambda *indice: \
                tvm.select(condition(*indice), lhs(*indice), rhs)
        else:
            # if lhs,rhs are all scalar, only support float16
            if judge_var(lhs) == "tvm_const" and lhs.dtype != "float16":
                raise RuntimeError("when lhs and rhs \
                                   are all scalar, only support float16")

            if judge_var(rhs) == "tvm_const" and rhs.dtype != "float16":
                raise RuntimeError("when lhs and rhs \
                                   are all scalar, only support float16")

            lhs = get_tvm_scalar(lhs, "float16")
            rhs = get_tvm_scalar(rhs, "float16")
            check_is_equal(lhs, rhs)

            lambda_func = lambda *indice: tvm.select(condition(*indice), lhs, rhs)

        name = "sel" + "_" + str(NAME_INDEX[0])
        NAME_INDEX[0] += 1
        op_name = op_name + '|' + mode
        with tvm.tag_scope(op_name):
            tmp = tvm.compute(shape, lambda_func, name=name)
    elif condition.dtype == "uint8":
        mode = 'bit'
        shape_condition = shape_to_list(condition.shape)
        shape = shape_condition
        shape[-1] = shape[-1] * 8

        supported_type = ["float16"]

        def get_indice(indice):
            """
            get indice
            """
            res_index = []
            for i, value in enumerate(indice):
                if i == len(indice) - 1:
                    res_index.append(value // 8)
                else:
                    res_index.append(value)
            return res_index

        if isinstance(lhs, tvm.tensor.Tensor) \
                and isinstance(rhs, tvm.tensor.Tensor):
            check_elewise_binary_shape(lhs, rhs)
            vsel_bit_shape_check(condition, lhs)
            src_dtype = lhs.dtype
            lhs = auto_cast_tensor(lhs, 'vsel', supported_type)
            rhs = auto_cast_tensor(rhs, 'vsel', supported_type)

            def _compute(*indice):
                res_index = get_indice(indice)
                return tvm.select(condition(*res_index).astype('bool'),
                                  lhs(*indice), rhs(*indice))
        elif not isinstance(lhs, tvm.tensor.Tensor) \
                and isinstance(rhs, tvm.tensor.Tensor):
            vsel_bit_shape_check(condition, rhs)
            src_dtype = rhs.dtype
            rhs = auto_cast_tensor(rhs, 'vsel', supported_type)
            lhs = get_tvm_scalar(lhs, rhs.dtype)

            def _compute(*indice):
                res_index = get_indice(indice)
                return tvm.select(condition(*res_index).astype('bool'),
                                  lhs, rhs(*indice))
        elif isinstance(lhs, tvm.tensor.Tensor) \
                and not isinstance(rhs, tvm.tensor.Tensor):
            vsel_bit_shape_check(condition, lhs)
            src_dtype = lhs.dtype
            lhs = auto_cast_tensor(lhs, 'vsel', supported_type)
            rhs = get_tvm_scalar(rhs, lhs.dtype)

            def _compute(*indice):
                res_index = get_indice(indice)
                return tvm.select(condition(*res_index).astype('bool'), \
                                  lhs(*indice), rhs)
        else:
            # if lhs,rhs are all scalar, only support float16
            if judge_var(lhs) == "tvm_const" and lhs.dtype != "float16":
                raise RuntimeError("when lhs and rhs \
                                   are all scalar, only support float16")

            if judge_var(rhs) == "tvm_const" and rhs.dtype != "float16":
                raise RuntimeError("when lhs and rhs \
                                   are all scalar, only support float16")

            lhs = get_tvm_scalar(lhs, "float16")
            rhs = get_tvm_scalar(rhs, "float16")
            check_is_equal(lhs, rhs)
            def _compute(*indice):
                res_index = get_indice(indice)
                return tvm.select(condition(*res_index).astype('bool'), lhs, rhs)

        name = "sel" + "_" + str(NAME_INDEX[0])
        NAME_INDEX[0] += 1

        op_name = op_name + '|' + mode
        with tvm.tag_scope(op_name):
            tmp = tvm.compute(shape, _compute, name=name)
    else:
        raise RuntimeError("condition only support bool and uint8")

    if src_dtype in ("int8", "uint8"):
        tmp = _cast(tmp, src_dtype, is_auto_cast=False)
    return tmp


def vcmpsel_data_shape_check(*args):
    """
    check vcmpsel's data shape
    :param args:
    :return:
    """
    arg_temp = args[0]

    for sh_value in arg_temp.shape:
        if sh_value.value <= 0 \
                or not isinstance(sh_value.value, int):
            raise RuntimeError("The input shape value \
                               must be a positive integer!")

    for arg in args:
        if len(arg.shape) != len(arg_temp.shape):
            raise RuntimeError("The input shape ndim \
                               must be equal to the each other!")

    for i in range(len(arg_temp.shape)):
        for arg in args:
            if arg_temp.shape[i].value != arg.shape[i].value:
                raise RuntimeError("The input shape \
                                must be equal to the each other!")


def vcmpsel_data_dtype_check(*args):
    """
    check vcmpsel's data type
    :param args:
    :return:
    """
    arg_temp = args[0]

    for arg in args:
        if arg.dtype != arg_temp.dtype:
            raise RuntimeError("The input dtype \
                               must be the same to the each other!")

# pylint: disable=too-many-branches, too-many-statements
def vcmpsel(lhs, rhs=None, operation='lt', slhs=None, srhs=None):
    """
    calculate elewise compare

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        compare left hand tensor
    rhs : wrapped_tensor or tvm.tensor or scalar
        compare right hand tensor or scalar
    operation : operator type, eq, ne, lt, gt, ge, le
    slhs : wrapped_tensor or tvm.tensor or scalar
        select left hand tensor or scalar
    srhs : wrapped_tensor or tvm.tensor or scalar
        select right hand tensor or scalar

    Returns
    -------
    wrapped_tensor
    """

    if not isinstance(lhs, tvm.tensor.Tensor):
        raise RuntimeError("The first input type must be tvm.tensor!")

    if operation not in ['eq', 'ne', 'lt', 'gt', 'ge', 'le']:
        raise RuntimeError("The op's value must be eq, ne, lt, gt, ge, le!")

    if rhs is None:
        rhs = 2.0

    if slhs is None:
        slhs = lhs

    if srhs is None:
        if isinstance(rhs, tvm.tensor.Tensor):
            srhs = rhs
        else:
            srhs = 0.0

    shape = lhs.shape
    cmpsel_op = "elewise_binary_cmpsel"

    if not isinstance(rhs, tvm.tensor.Tensor) \
            and not isinstance(slhs, tvm.tensor.Tensor) \
            and not isinstance(srhs, tvm.tensor.Tensor):
        lhs = auto_cast_tensor(lhs, "vsel")
        rhs = get_tvm_scalar(rhs, lhs.dtype)
        slhs = get_tvm_scalar(slhs, lhs.dtype)
        srhs = get_tvm_scalar(srhs, lhs.dtype)
        if operation == 'lt':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) < rhs, slhs, srhs)
        elif operation == 'gt':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) > rhs, slhs, srhs)
        elif operation == 'le':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) <= rhs, slhs, srhs)
        elif operation == 'ge':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) >= rhs, slhs, srhs)
        elif operation == 'eq':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) == rhs, slhs, srhs)
        elif operation == 'ne':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) != rhs, slhs, srhs)
        else:
            raise RuntimeError("vcmpsel do not support the input op")
    elif isinstance(rhs, tvm.tensor.Tensor) \
            and not isinstance(slhs, tvm.tensor.Tensor) \
            and not isinstance(srhs, tvm.tensor.Tensor):
        vcmpsel_data_shape_check(lhs, rhs)
        vcmpsel_data_dtype_check(lhs, rhs)
        lhs = auto_cast_tensor(lhs, "vsel")
        rhs = auto_cast_tensor(rhs, "vsel")
        slhs = get_tvm_scalar(slhs, lhs.dtype)
        srhs = get_tvm_scalar(srhs, lhs.dtype)
        if operation == 'lt':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) < rhs(*indice), slhs, srhs)
        elif operation == 'gt':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) > rhs(*indice), slhs, srhs)
        elif operation == 'le':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) <= rhs(*indice), slhs, srhs)
        elif operation == 'ge':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) >= rhs(*indice), slhs, srhs)
        elif operation == 'eq':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) == rhs(*indice), slhs, srhs)
        elif operation == 'ne':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) != rhs(*indice), slhs, srhs)
        else:
            raise RuntimeError("vcmpsel do not support the input op")
    elif not isinstance(rhs, tvm.tensor.Tensor) \
            and isinstance(slhs, tvm.tensor.Tensor) \
            and not isinstance(srhs, tvm.tensor.Tensor):
        vcmpsel_data_shape_check(lhs, slhs)
        vcmpsel_data_dtype_check(lhs, slhs)
        lhs = auto_cast_tensor(lhs, "vsel")
        rhs = get_tvm_scalar(rhs, lhs.dtype)
        slhs = auto_cast_tensor(slhs, "vsel")
        srhs = get_tvm_scalar(srhs, lhs.dtype)
        if operation == 'lt':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) < rhs, slhs(*indice), srhs)
        elif operation == 'gt':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) > rhs, slhs(*indice), srhs)
        elif operation == 'le':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) <= rhs, slhs(*indice), srhs)
        elif operation == 'ge':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) >= rhs, slhs(*indice), srhs)
        elif operation == 'eq':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) == rhs, slhs(*indice), srhs)
        elif operation == 'ne':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) != rhs, slhs(*indice), srhs)
        else:
            raise RuntimeError("vcmpsel do not support the input op")
    elif not isinstance(rhs, tvm.tensor.Tensor) \
            and not isinstance(slhs, tvm.tensor.Tensor) \
            and isinstance(srhs, tvm.tensor.Tensor):
        vcmpsel_data_shape_check(lhs, srhs)
        vcmpsel_data_dtype_check(lhs, srhs)
        lhs = auto_cast_tensor(lhs, "vsel")
        rhs = get_tvm_scalar(rhs, lhs.dtype)
        slhs = get_tvm_scalar(slhs, lhs.dtype)
        srhs = auto_cast_tensor(srhs, "vsel")
        if operation == 'lt':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) < rhs, slhs, srhs(*indice))
        elif operation == 'gt':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) > rhs, slhs, srhs(*indice))
        elif operation == 'le':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) <= rhs, slhs, srhs(*indice))
        elif operation == 'ge':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) >= rhs, slhs, srhs(*indice))
        elif operation == 'eq':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) == rhs, slhs, srhs(*indice))
        elif operation == 'ne':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) != rhs, slhs, srhs(*indice))
        else:
            raise RuntimeError("vcmpsel do not support the input op")
    elif isinstance(rhs, tvm.tensor.Tensor) \
            and isinstance(slhs, tvm.tensor.Tensor) \
            and not isinstance(srhs, tvm.tensor.Tensor):
        vcmpsel_data_shape_check(lhs, rhs, slhs)
        vcmpsel_data_dtype_check(lhs, rhs, slhs)
        lhs = auto_cast_tensor(lhs, "vsel")
        rhs = auto_cast_tensor(rhs, "vsel")
        slhs = auto_cast_tensor(slhs, "vsel")
        srhs = get_tvm_scalar(srhs, lhs.dtype)
        if operation == 'lt':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) < rhs(*indice), slhs(*indice), srhs)
        elif operation == 'gt':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) > rhs(*indice), slhs(*indice), srhs)
        elif operation == 'le':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) <= rhs(*indice), slhs(*indice), srhs)
        elif operation == 'ge':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) >= rhs(*indice), slhs(*indice), srhs)
        elif operation == 'eq':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) == rhs(*indice), slhs(*indice), srhs)
        elif operation == 'ne':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) != rhs(*indice), slhs(*indice), srhs)
        else:
            raise RuntimeError("vcmpsel do not support the input op")
    elif isinstance(rhs, tvm.tensor.Tensor) \
            and not isinstance(slhs, tvm.tensor.Tensor) \
            and isinstance(srhs, tvm.tensor.Tensor):
        vcmpsel_data_shape_check(lhs, rhs, srhs)
        vcmpsel_data_dtype_check(lhs, rhs, srhs)
        lhs = auto_cast_tensor(lhs, "vsel")
        rhs = auto_cast_tensor(rhs, "vsel")
        slhs = get_tvm_scalar(slhs, lhs.dtype)
        srhs = auto_cast_tensor(srhs, "vsel")
        if operation == 'lt':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) < rhs(*indice), slhs, srhs(*indice))
        elif operation == 'gt':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) > rhs(*indice), slhs, srhs(*indice))
        elif operation == 'le':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) <= rhs(*indice), slhs, srhs(*indice))
        elif operation == 'ge':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) >= rhs(*indice), slhs, srhs(*indice))
        elif operation == 'eq':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) == rhs(*indice), slhs, srhs(*indice))
        elif operation == 'ne':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) != rhs(*indice), slhs, srhs(*indice))
        else:
            raise RuntimeError("vcmpsel do not support the input op")
    elif not isinstance(rhs, tvm.tensor.Tensor) \
            and isinstance(slhs, tvm.tensor.Tensor) \
            and isinstance(srhs, tvm.tensor.Tensor):
        vcmpsel_data_shape_check(lhs, slhs, srhs)
        vcmpsel_data_dtype_check(lhs, slhs, srhs)
        lhs = auto_cast_tensor(lhs, "vsel")
        rhs = get_tvm_scalar(rhs, lhs.dtype)
        slhs = auto_cast_tensor(slhs, "vsel")
        srhs = auto_cast_tensor(srhs, "vsel")
        if operation == 'lt':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) < rhs, slhs(*indice), srhs(*indice))
        elif operation == 'gt':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) > rhs, slhs(*indice), srhs(*indice))
        elif operation == 'le':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) <= rhs, slhs(*indice), srhs(*indice))
        elif operation == 'ge':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) >= rhs, slhs(*indice), srhs(*indice))
        elif operation == 'eq':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) == rhs, slhs(*indice), srhs(*indice))
        elif operation == 'ne':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) != rhs, slhs(*indice), srhs(*indice))
        else:
            raise RuntimeError("vcmpsel do not support the input op")
    else:
        vcmpsel_data_shape_check(lhs, rhs, slhs, srhs)
        vcmpsel_data_dtype_check(lhs, rhs, slhs, srhs)
        lhs = auto_cast_tensor(lhs, "vsel")
        rhs = auto_cast_tensor(rhs, "vsel")
        slhs = auto_cast_tensor(slhs, "vsel")
        srhs = auto_cast_tensor(srhs, "vsel")
        if operation == 'lt':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) < rhs(*indice),
                           slhs(*indice), srhs(*indice))
        elif operation == 'gt':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) > rhs(*indice),
                           slhs(*indice), srhs(*indice))
        elif operation == 'le':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) <= rhs(*indice),
                           slhs(*indice), srhs(*indice))
        elif operation == 'ge':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) >= rhs(*indice),
                           slhs(*indice), srhs(*indice))
        elif operation == 'eq':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) == rhs(*indice),
                           slhs(*indice), srhs(*indice))
        elif operation == 'ne':
            lambda_func = lambda *indice: \
                tvm.select(lhs(*indice) != rhs(*indice),
                           slhs(*indice), srhs(*indice))
        else:
            raise RuntimeError("vcmpsel do not support the input op")

    name = cmpsel_op.split("_")[-1] + "_" + str(NAME_INDEX[0])
    NAME_INDEX[0] += 1

    cmpsel_op = cmpsel_op + "|" + operation

    with tvm.tag_scope(cmpsel_op):
        tmp = tvm.compute(shape, lambda_func, name=name)

    return tmp
