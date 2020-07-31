#!/usr/bin/env python # pylint: disable=import-error, too-many-lines
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

mmad schedule
"""
from __future__ import absolute_import

import te.platform.cce_params as cce
import te.platform.cce_conf as conf
from te.platform.fusion_manager import fusion_manager
from te.platform import get_soc_spec
import tvm

DTYPE_WIDTH_MAP = {"float16": 1,
                   "float32": 2,
                   "int32": 2,
                   "int16": 1,
                   "uint16": 1,
                   "int8": 0.5,
                   "uint8": 0.5,
                   "bool": 0.5}

def gemm_para_check(gm_shape, l1_tiling_shape, l0_tiling_shape):
    """
    algorithm: gemm_para_check

    Parameters
    ----------
    gm_shape : the M,N,K shape

    l1_tiling_shape : L1 Tiling shape

    l0_tiling_shape : L0 Tiling shape

    Returns :
    -------
    None
    """
    gm_m = gm_shape[0]
    gm_k = gm_shape[1]
    gm_n = gm_shape[2]

    block_m = l1_tiling_shape[0]
    block_k = l1_tiling_shape[1]
    block_n = l1_tiling_shape[2]

    block_ml = l0_tiling_shape[0]
    block_kl = l0_tiling_shape[1]
    block_nl = l0_tiling_shape[2]

    # check the block
    if block_m <= 0 or block_n <= 0 or block_k <= 0:
        raise RuntimeError(
            "input block param should not be less than 0: actual (block_m, "
            "block_n, block_k) = (%d, %d, %d)" % (block_m, block_n, block_k))
    if block_m > gm_m or block_n > gm_n or block_k > gm_k:
        raise RuntimeError(
            "input block param should not be less than shape value: actual "
            "(block_m, block_n, block_k) = (%d, %d, %d)" %
            (block_m, block_n, block_k))
    if ((block_m % 16 != 0) or (block_n % 16 != 0) or (
            block_k % 16 != 0)) and block_m != 1 and block_n != 1:
        raise RuntimeError(
            "input shape block_m or block_k or block_n should be multiple of "
            "16: actual (block_m, block_k, block_n) = (%d, %d, %d)" %
            (block_m, block_k, block_n))

    # check the block L0
    if block_ml <= 0 or block_nl <= 0 or block_kl <= 0:
        raise RuntimeError(
            "input block param should not be less than 0: actual "
            "(block_ml, block_nl, block_kl) = (%d, %d, %d)" %
            (block_ml, block_nl, block_kl))
    if block_ml > block_m or block_nl > block_n or block_kl > block_k:
        raise RuntimeError(
            "input block param should not be less than blockL1 value: actual "
            "(block_ml, block_nl, block_kl) = (%d, %d, %d)" %
            (block_ml, block_nl, block_kl))
    if ((block_ml % 16 != 0) or (block_nl % 16 != 0) or (
            block_kl % 16 != 0)) and block_ml != 1 and block_nl != 1:
        raise RuntimeError(
            "input shape block_ml or block_kl or block_nl should be multiple "
            "of 16: actual (block_ml, block_kl, block_nl) = (%d, %d, %d)" %
            (block_ml, block_kl, block_nl))


def _shape_to_list(shape):
    """
    translate tvm.shape to list type in python
    """
    tmp = []
    for i in shape:
        tmp.append(i.value)
    return tmp


def get_special_l0_factor(src_shape, m_l0_shape, k_l0_shape, n_l0_shape):
    """
    get temp factors
    """
    m_shape = src_shape[0]
    k_shape = src_shape[1]
    n_shape = src_shape[2]
    if m_shape * n_shape * k_shape == m_l0_shape * n_l0_shape * k_l0_shape and \
            m_l0_shape != 1:
        m_l0_shape = int((m_l0_shape // 2))
        if int((m_l0_shape % 16)) != 0:
            m_l0_shape = int((m_l0_shape + 15) // 16 * 16)

    src_shape = [m_shape, k_shape, n_shape]
    if src_shape == [256, 64, 256]:
        m_l0_shape = 256
        k_l0_shape = 64
        n_l0_shape = 128
    elif src_shape == [256, 256, 64]:
        m_l0_shape = 64
        k_l0_shape = 256
        n_l0_shape = 64
    return m_l0_shape, k_l0_shape, n_l0_shape


def get_batch_factors(tensor_a_shape, # pylint: disable=too-many-arguments
                      tensor_a_l0a, tensor_b_l0b, m_var, n_var, is_gemv):
    """
    get batch vars
    """
    m_shape = m_var[0]
    m_factors = m_var[1]
    n_shape = n_var[0]
    n_factors = n_var[1]
    core_inner_m = m_shape
    core_inner_n = n_shape
    if len(tensor_a_shape) == 3 or len(tensor_a_shape) == 5:
        batch = tensor_a_l0a.shape[0].value
        if is_gemv:
            batch = tensor_b_l0b.shape[0].value
    else:
        batch = 0
    if batch in (0, 1):
        block_in = cce.BLOCK_IN
        block_out = cce.BLOCK_OUT
        if m_shape != 1:
            core_inner_m = (((m_shape + block_in - 1) // block_in + \
                (m_factors - 1)) // m_factors) * block_in
        core_inner_n = (((n_shape + block_out - 1) // block_out + \
            (n_factors - 1)) // n_factors) * block_out

    return batch, core_inner_m, core_inner_n


def get_perfect_core_num(m_shape, # pylint: disable=too-many-locals
                         n_shape, k_shape):
    """
    :param input_shape_1:the tensor_a shape
    :param input_shape_2:the tensor_b shape
    :return:core_num
    """
    frac_size = 16
    core_num = conf.getValue("Device_core_num")
    m_axis_outer = (m_shape + frac_size - 1) // frac_size
    if m_shape == 1:
        m_axis_outer = 1
        n_axis_outer = (n_shape + frac_size - 1) // frac_size
        if n_axis_outer > core_num:
            return 1, core_num
        return 1, 1

    m_axis_outer = m_shape // frac_size
    n_axis_outer = n_shape // frac_size
    if (m_axis_outer * n_axis_outer) <= core_num:
        return m_axis_outer, n_axis_outer
    tensor_a_size = m_shape * k_shape
    tensor_b_size = n_shape * k_shape
    min_copy_size = core_num * (tensor_a_size + tensor_b_size)
    m_factor = m_axis_outer
    n_factor = n_axis_outer

    exp = 1
    if core_num == 32:
        # the exp for 32, 2^(6-1)
        exp = 6
    elif core_num == 2:
        # the exp for 2, 2^(2-1)
        exp = 2
    for i in (2 ** e for e in range(0, exp)):
        # judge cur_factor
        cur_m_factor = i
        cur_n_factor = core_num // i
        if cur_m_factor > m_axis_outer or (m_axis_outer // cur_m_factor) == 0:
            continue
        if cur_n_factor > n_axis_outer or (n_axis_outer // cur_n_factor) == 0:
            continue

        cur_copy_size = cur_n_factor * tensor_a_size + cur_m_factor * \
                        tensor_b_size
        temp_m_shape = m_shape
        temp_n_shape = n_shape
        if m_axis_outer % m_factor != 0:
            temp_m_shape = (((m_axis_outer // cur_m_factor) + 1) *
                            cur_m_factor) * frac_size

        if n_shape % n_factor != 0:
            temp_n_shape = (((n_axis_outer // cur_n_factor) + 1) *
                            cur_n_factor) * frac_size

        cur_copy_size = cur_n_factor * (temp_m_shape * k_shape) + \
            cur_m_factor * (temp_n_shape * k_shape)
        if cur_copy_size < min_copy_size:
            min_copy_size = cur_copy_size
            m_factor = cur_m_factor
            n_factor = cur_n_factor

    return m_factor, n_factor


def check_mini_core_num():
    """
    check mini device or cloud device
    """
    target_core_num = get_soc_spec("CORE_NUM")
    if target_core_num == 2:
        return True
    return False

def update_op_pattern(fractal_a, fractal_b):
    """
    only support frac+frac for elementwise fusion
    """
    if not fractal_a or not fractal_b:
        fusion_manager.set_current_op_pattern("Opaque")

def mmad_schedule(res, sch_list):
    """
    algorithm: mmad_schedule

    Parameters
    ----------
    res : the output tensor

    sch_list : schedule list

    Returns : if it is true,then we can get the valid schedule

    """
    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    sch = sch_list[0]
    fusion_flag = False
    sqrt_flag = False
    gemv_flag = False
    placeholder_tensors = []  # to list placeholder type tensor
    compute_tensors = []  # to list compute type tensor
    compute_tensors_local = []
    def get_placeholder_tensor(tensor):
        """
        scan all the transient tensor during calculation
        tersor: target tensor which needs to find placeholder tensor
        """
        compute_tensors_local.append(tensor)
        # pylint: disable=too-many-locals, too-many-branches, too-many-statements
        tensor_list = tensor.op.input_tensors
        for one_tensor in tensor_list:
            # check which tensor has not been checked
            if one_tensor not in compute_tensors_local:
                if isinstance((one_tensor.op), tvm.tensor.PlaceholderOp):
                    placeholder_tensors.append(one_tensor)
                else:
                    compute_tensors_local.append(one_tensor)
                    get_placeholder_tensor(one_tensor)
        return compute_tensors_local

    compute_tensors = get_placeholder_tensor(res)

    def match_and_get_tensor(compute_tensors, tensor_name):
        """
        match and get tensor
        """
        for i in compute_tensors:
            if tensor_name == i.op.name:
                return i
        return None

    tensor_a_ub = match_and_get_tensor(compute_tensors, 'tensor_a_ub')
    tensor_a_l1 = match_and_get_tensor(compute_tensors, 'tensor_a_l1')
    tensor_b_ub = match_and_get_tensor(compute_tensors, 'tensor_b_ub')
    tensor_b_l1 = match_and_get_tensor(compute_tensors, 'tensor_b_l1')
    tensor_bias_ub = match_and_get_tensor(compute_tensors, 'tensor_bias_ub')
    tensor_a_ub_fract = match_and_get_tensor(compute_tensors,
                                             'tensor_a_ub_fract')
    tensor_b_ub_fract = match_and_get_tensor(compute_tensors,
                                             'tensor_b_ub_fract')
    tensor_c_ub_fract = match_and_get_tensor(compute_tensors,
                                             'tensor_c_ub_fract')
    tensor_c_gm = match_and_get_tensor(compute_tensors, 'tensor_c_gm')

    tensor_c_ub = None
    tensor_sqrt = None
    dequant_relu = None
    is_b_nz = False

    def __get_b_nz_flag(tensor):
        is_b_nz = False
        if tensor.op.name == "tensor_c_ub":
            is_b_nz = tensor.op.attrs['nz_b'].value
        return is_b_nz

    for i in compute_tensors:
        if 'tensor_c_ub' in i.op.name and not fusion_flag:
            tensor_c_ub = i
            tensor_c_ub_inner = tensor_c_ub
            is_b_nz = __get_b_nz_flag(i)
        if 'tensor_c_ub' in i.op.name and fusion_flag:
            tensor_c_ub_inner = i
        if 'dequant' in i.op.name:
            tensor_c_ub = i
            fusion_flag = True
        if "dequant_scale" in i.op.name:
            tensor_c_ub = i
            fusion_flag = True
        if 'dequant_sqrt' in i.op.name:
            tensor_sqrt = i
            sqrt_flag = True
        if 'dequant_relu' in i.op.name:
            dequant_relu = i
        if i.op.tag == 'matmul_gemv':
            gemv_flag = True

    if fusion_flag:
        tensor_c_ub_fract = sch.cache_write(res, cce.scope_ubuf)

    matmul_end_tensors = tensor_c_gm
    if tensor_c_gm is None:
        matmul_end_tensors = tensor_c_ub

    matmul_tensors = []
    compute_tensors_local = []
    matmul_tensors = get_placeholder_tensor(matmul_end_tensors)

    tensor_ele_map = []
    fusion_ele = False
    if tensor_c_gm != res and tensor_c_gm is not None:
        fusion_ele = True
        for ten_in in compute_tensors:
            if ten_in == res:
                continue
            if ten_in not in matmul_tensors:
                tensor_ele_map.append(ten_in)

    if tensor_a_ub is not None:
        tensor_a = tensor_a_ub.op.input_tensors[0]
    elif tensor_a_l1 is not None:
        tensor_a = tensor_a_l1.op.input_tensors[0]
    else:
        raise RuntimeError(
            "Lack of tensor_a_ub or tensor_a_l1.")
    tensor_a_shape = tensor_a.shape
    if tensor_b_ub is not None:
        tensor_b = tensor_b_ub.op.input_tensors[0]
    elif tensor_b_l1 is not None:
        tensor_b = tensor_b_l1.op.input_tensors[0]
    else:
        raise RuntimeError(
            "Lack of tensor_b_ub or tensor_b_l1.")
    tensor_b_shape = tensor_b.shape

    is_with_bias = tensor_bias_ub is not None

    is_fractal_a = len(tensor_a_shape) == 4 or len(tensor_a_shape) == 5
    is_fractal_b = len(tensor_b_shape) == 4 or len(tensor_b_shape) == 5

    # only support frac+frac for elementwise fusion
    if tensor_c_gm is not None:
        update_op_pattern(is_fractal_a, is_fractal_b)

    if is_with_bias:
        tensor_c_add_bias = tensor_c_ub_inner.op.input_tensors[0]
        tensor_c = tensor_c_add_bias.op.input_tensors[1]
        tensor_bias_l0c = tensor_c_add_bias.op.input_tensors[0]
    else:
        tensor_c = tensor_c_ub_inner.op.input_tensors[0]

    tensor_a_l0a = tensor_c.op.input_tensors[0]
    tensor_b_l0b = tensor_c.op.input_tensors[1]

    is_gemv = False
    if tensor_c.op.attrs['input_order'].value == "positive":
        tensor_a_l1 = tensor_a_l0a.op.input_tensors[0]
        tensor_b_l1 = tensor_b_l0b.op.input_tensors[0]
    else:
        tensor_a_l1 = tensor_b_l0b.op.input_tensors[0]
        tensor_b_l1 = tensor_a_l0a.op.input_tensors[0]
        is_gemv = True

    if matmul_end_tensors.op.tag == 'matmul_gemv' or gemv_flag:
        block_in = cce.BLOCK_VECTOR
        mad_pattern = cce.GEVM_MODE
    else:
        mad_pattern = cce.GEMM_MODE
        block_in = cce.BLOCK_IN

    l0_tensor_len_a = len(tensor_a_l0a.shape)
    l0_tensor_len_b = len(tensor_b_l0b.shape)
    tensor_len_a = len(tensor_a_l1.shape)
    tensor_len_b = len(tensor_b_l1.shape)
    tensor_len_c = len(tensor_c.shape)

    block_out = cce.BLOCK_OUT

    out_dtype = tensor_c.dtype

    def _get_block_reduce(out_dtype):
        if out_dtype in ("float16", "float32"):
            return cce.BLOCK_REDUCE
        return cce.BLOCK_REDUCE_INT8

    block_reduce = _get_block_reduce(out_dtype)

    # get matrix axis shapes
    m_shape = tensor_a_l0a.shape[l0_tensor_len_a - 4].value * block_in
    k_shape = tensor_a_l0a.shape[l0_tensor_len_a - 3].value * block_reduce
    n_shape = tensor_b_l0b.shape[l0_tensor_len_b - 3].value * block_out

    core_inner_m = m_shape
    core_inner_n = n_shape
    m_factors, n_factors = get_perfect_core_num(m_shape, n_shape, k_shape)
    m_var = [m_shape, m_factors]
    n_var = [n_shape, n_factors]
    batch, core_inner_m, core_inner_n = get_batch_factors(
        tensor_a_shape, tensor_a_l0a, tensor_b_l0b, m_var, n_var, is_gemv)

    l0a_byte = 2 if (tensor_a_l0a.dtype == "float16") else 1
    l0b_byte = 2 if (tensor_b_l0b.dtype == "float16") else 1
    l0c_byte = 2 if (tensor_c.dtype == "float16") else 4

    l1a_byte = 2 if (tensor_a_l1.dtype == "float16") else 1
    l1b_byte = 2 if (tensor_b_l1.dtype == "float16") else 1
    a_ub_byte = 2
    b_ub_byte = 2
    ub_res_byte = 2
    ub_reserve_buff = 0

    def get_scope_byte_size(tensor_ub, tensor_ub_fract):
        """
        get unit byte size for buffer scope
        """
        # Calculating tiling para need a_ub info
        ub_byte = 0
        if tensor_ub is not None:
            if tensor_ub.dtype == "float16":
                ub_byte = 2
            elif tensor_ub.dtype == "int8" or tensor_ub.dtype == "uint8":
                ub_byte = 1
            else:
                ub_byte = 4
            if tensor_ub_fract is not None:
                ub_byte = ub_byte * 2
        return ub_byte

    a_ub_byte = get_scope_byte_size(tensor_a_ub, tensor_a_ub_fract)
    b_ub_byte = get_scope_byte_size(tensor_b_ub, tensor_b_ub_fract)
    ub_res_byte = get_scope_byte_size(tensor_c_ub, tensor_c_ub_fract)

    if is_gemv:
        tmp = a_ub_byte # pylint: disable=R1712
        a_ub_byte = b_ub_byte
        b_ub_byte = tmp

    ub_reserve_buff = 0
    if fusion_flag or tensor_c_ub.op.attrs['scale_drq'].value == "ENABLE":
        # quant parameter is fixed float16, it's 2 bytes
        # just support scalar now, not support vector yet
        ub_reserve_buff = cce.BLOCK_OUT * 2

    get_tiling_shape = tvm.get_global_func("cce.matmul_tiling_gen")
    tiling_shape = get_tiling_shape(core_inner_m, k_shape, core_inner_n,
                                    a_ub_byte,
                                    b_ub_byte, l1a_byte,
                                    l1b_byte, l0a_byte, l0b_byte, l0c_byte,
                                    ub_res_byte, ub_reserve_buff, is_b_nz)
    if tiling_shape.find('_') == -1:
        raise RuntimeError(tiling_shape)

    tiled_shape = tiling_shape.split('_')
    m_l1_shape = int(tiled_shape[0])
    k_l1_shape = int(tiled_shape[1])
    n_l1_shape = int(tiled_shape[2])

    m_l0_shape = int(tiled_shape[3])
    k_l0_shape = int(tiled_shape[4])
    n_l0_shape = int(tiled_shape[5])

    src_shape = [m_shape, k_shape, n_shape]
    m_l0_shape, k_l0_shape, n_l0_shape = get_special_l0_factor(
        src_shape, m_l0_shape, k_l0_shape, n_l0_shape)

    m_l1_shape = m_l0_shape
    k_l1_shape = k_l0_shape
    n_l1_shape = n_l0_shape

    if m_l1_shape == m_shape and k_l1_shape == k_shape:
        m_l1_double_buffer = 1
    else:
        m_l1_double_buffer = 2

    if n_l1_shape == n_shape and k_l1_shape == k_shape:
        n_l1_double_buffer = 1
    else:
        n_l1_double_buffer = 2

    l1_tiling_shape = (m_l1_shape, k_l1_shape, n_l1_shape)
    l0_tiling_shape = (m_l0_shape, k_l0_shape, n_l0_shape)

    # tiling param check
    gemm_para_check([m_shape, k_shape, n_shape], l1_tiling_shape,
                    l0_tiling_shape)

    # compute L1->L0 tiling params
    m_l0_tile = m_l0_shape // block_in
    k_l0_tile = k_l0_shape // block_reduce
    n_l0_tile = n_l0_shape // block_out

    # compute GM to L1 tiling params
    m_l1_tile = (m_l1_shape // block_in)
    k_l1_tile = (k_l1_shape // block_reduce)
    n_l1_tile = (n_l1_shape // block_out)

    repeat_a = False
    repeat_b = False

    def set_scope_buffer_type():
        """
        set all tensors buffer scope
        """
        if tensor_a_ub is not None:
            sch[tensor_a_ub].set_scope(cce.scope_ubuf)
        if tensor_b_ub is not None:
            sch[tensor_b_ub].set_scope(cce.scope_ubuf)

        if tensor_a_ub_fract is not None:
            sch[tensor_a_ub_fract].set_scope(cce.scope_ubuf)
        if tensor_b_ub_fract is not None:
            sch[tensor_b_ub_fract].set_scope(cce.scope_ubuf)
        if tensor_c_ub_fract is not None:
            sch[tensor_c_ub_fract].set_scope(cce.scope_ubuf)
        sch[tensor_a_l1].set_scope(cce.scope_cbuf)
        sch[tensor_b_l1].set_scope(cce.scope_cbuf)
        sch[tensor_a_l0a].set_scope(cce.scope_ca)
        sch[tensor_b_l0b].set_scope(cce.scope_cb)
        sch[tensor_c].set_scope(cce.scope_cc)
        sch[tensor_c_ub].set_scope(cce.scope_ubuf)
        if fusion_flag:
            sch[tensor_c_ub_inner].set_scope(cce.scope_ubuf)
            if sqrt_flag:
                sch[tensor_sqrt].set_scope(cce.scope_ubuf)
            if dequant_relu is not None:
                sch[dequant_relu].set_scope(cce.scope_ubuf)
        for ten_i in tensor_ele_map:
            sch[ten_i].set_scope(cce.scope_ubuf)
        if is_with_bias:
            sch[tensor_bias_ub].set_scope(cce.scope_ubuf)
            sch[tensor_bias_l0c].set_scope(cce.scope_cc)
            sch[tensor_c_add_bias].set_scope(cce.scope_cc)

    set_scope_buffer_type()

    def set_tensor_buffer_align(gm_pattern):
        """
        set tensor_c and tensor_c_add_bias buffer align
        """
        unchanged = 1
        # gevm single batch
        if gm_pattern == cce.GEVM_MODE:
            if len(tensor_c.shape) == 4:
                sch[tensor_c].buffer_align((unchanged, unchanged),
                                       (unchanged, unchanged),
                                       (unchanged, block_out),
                                       (unchanged, block_out),
                                       (unchanged, block_reduce),
                                       (unchanged, block_reduce))
                if is_with_bias:
                    sch[tensor_c_add_bias].buffer_align((unchanged, unchanged),
                                                        (unchanged, unchanged),
                                                        (unchanged, block_out),
                                                        (unchanged, block_out))
            else:
                # gevm multi batch
                sch[tensor_c].buffer_align((unchanged, unchanged),
                                           (unchanged, unchanged),
                                           (unchanged, unchanged),
                                           (unchanged, block_out),
                                           (unchanged, block_out),
                                           (unchanged, block_reduce),
                                           (unchanged, block_reduce))
                if is_with_bias:
                    sch[tensor_c_add_bias].buffer_align((unchanged, unchanged),
                                                        (unchanged, unchanged),
                                                        (unchanged, unchanged),
                                                        (unchanged, block_out),
                                                        (unchanged, block_out))
        # single batch
        if block_in != cce.BLOCK_VECTOR:
            if len(tensor_c.shape) == 4:
                if tensor_a_ub is not None:
                    sch[tensor_a_ub].buffer_align((unchanged, block_in),
                                                  (unchanged, block_reduce))
                sch[tensor_a_l0a].buffer_align((unchanged, unchanged),
                                               (unchanged, unchanged),
                                               (unchanged, block_in),
                                               (unchanged, block_reduce))
                sch[tensor_c_ub].buffer_align((unchanged, unchanged),
                                              (unchanged, unchanged),
                                              (unchanged, block_in),
                                              (unchanged, block_out))
            else:
                # multi batch
                if tensor_a_ub is not None:
                    sch[tensor_a_ub].buffer_align((unchanged, unchanged),
                                                  (unchanged, block_in),
                                                  (unchanged, block_reduce))
                sch[tensor_a_l0a].buffer_align((unchanged, unchanged),
                                               (unchanged, unchanged),
                                               (unchanged, unchanged),
                                               (unchanged, block_in),
                                               (unchanged, block_reduce))
                sch[tensor_c_ub].buffer_align((unchanged, unchanged),
                                              (unchanged, unchanged),
                                              (unchanged, unchanged),
                                              (unchanged, block_in),
                                              (unchanged, block_out))

    # set tensor buffer align
    set_tensor_buffer_align(mad_pattern)

    core_num = conf.getValue("Device_core_num")
    core_thres = core_num // 2

    batch_outer = None
    # for multi batch use multi block
    if batch > 1:
        batch_factor = batch
        if batch > core_num:
            batch_factor = core_num

        batch_outer, _ = sch[res].split(res.op.axis[0],
                                        nparts=batch_factor)
        thread_block = tvm.thread_axis("blockIdx.x")
        sch[res].bind(batch_outer, thread_block)

    emit_insn_map = {"elewise_single_cast": "vector_conv",
                     "elewise_single_round_d": "vector_conv_round",
                     "elewise_single_VS_max": "vector_maxs",
                     "elewise_single_VS_min": "vector_mins",
                     "elewise_single_ceil": "elewise_single_ceil",
                     "elewise_single_log": "vector_ln",
                     "elewise_single_exp": "vector_exp",
                     "elewise_single_relu": "vector_relu",
                     "elewise_single_abs": "vector_abs",
                     "elewise_single_not": "vector_not",
                     "elewise_single_sqrt": "vector_sqrt",
                     "elewise_single_rsqrt": "vector_rsqrt",
                     "elewise_binary_mul": "vector_mul",
                     "elewise_single_rec": "vector_rec",
                     "elewise_single_VS_mul": "vector_muls",
                     "elewise_binary_div": "vector_div",
                     "elewise_binary_sub": "vector_sub",
                     "elewise_binary_add": "vector_add",
                     "elewise_single_VS_add": "vector_adds",
                     "elewise_binary_min": "vector_min",
                     "elewise_binary_max": "vector_max",
                     "elewise_binary_vcmpv_gt": "vector_gt",
                     "elewise_binary_vcmpv_ge": "vector_ge",
                     "elewise_binary_vcmpv_lt": "vector_lt",
                     "elewise_binary_vcmpv_le": "vector_le",
                     "elewise_binary_vcmpv_eq": "vector_eq",
                     "elewise_binary_vcmpv_ne": "vector_ne",
                     "elewise_binary_or": "vector_or",
                     "elewise_binary_and": "vector_and",
                     "elewise_multiple_mla": "vector_multiple",
                     "elewise_multiple_madd": "vector_multiple",
                     "elewise_multiple_maddrelu": "vector_multiple",
                     "elewise_binary_scalar_axpy": "vector_multiple",
                     "elewise_binary_cmpsel":"vector_cmpsel",
                     "broadcast": "vector_dup",
                     "emit_insn_elewise_multiple_sel": "elewise_multiple_sel",
                     "emit_insn_elewise_binary_cmp": "elewise_binary_cmp"
                     }

    def _map_apend(input_map, key, value):
        """
        map apend
        """
        if input_map.get(key):
            if isinstance(value, list):
                for sub_v in value:
                    if sub_v not in input_map[key]:
                        input_map[key].append(sub_v)
            else:
                if value not in input_map[key]:
                    input_map[key].append(value)
        else:
            if isinstance(value, list):
                input_map[key] = value
            else:
                input_map[key] = [value]

    def _gen_in_out_tensor_map(out_tensor, in_out_tensor_map):
        """
        traverse tensors by Depth-First-Search

        Parameters
        ----------
        out_tensor : tensor
            traverse tensors from this tensor,
            traversing its input tensors recursively.

        tensor_list : list
            record tensors in the order of Depth-First-Search.

        """
        if out_tensor is None:
            return
        stack = [out_tensor]
        visited_list = []
        while stack:
            cur_tensor = stack.pop()
            visited_list.append(cur_tensor)
            for in_tensor in cur_tensor.op.input_tensors:
                if in_tensor not in visited_list:
                    stack.append(in_tensor)
                _map_apend(in_out_tensor_map, in_tensor, cur_tensor)

    def _get_out_tensors_width(out_tensor):
        """
        get max width for tensors

        Parameters
        ----------
        out_tensor : tensor
            need to count all its input tensorss

        Return
        ------
            max width for tensors
        """
        # pylint: cell-var-from-loop
        in_out_tensor_map = {}
        _gen_in_out_tensor_map(out_tensor, in_out_tensor_map)
        stack = [out_tensor]
        width = len(stack)
        visited_list = []
        tmp_stack = stack
        while tmp_stack:
            for tens in tmp_stack:
                if tens in in_out_tensor_map:
                    def calc_width_mid(width):
                        """
                        get mid tesnor width
                        """
                        all_out = True
                        for out_ten in in_out_tensor_map[tens]: # pylint: disable=W0640
                            if out_ten not in visited_list:
                                all_out = False
                        if all_out and (tens not in visited_list): # pylint: disable=W0640
                            visited_list.append(tens) # pylint: disable=W0640
                            stack.remove(tens) # pylint: disable=W0640
                            for in_ten in tens.op.input_tensors: # pylint: disable=W0640
                                if in_ten not in stack and in_ten != tensor_c_gm:
                                    stack.append(in_ten)
                            width_local = 0
                            cast_flag = False
                            for ele in stack:
                                width_local = width_local + DTYPE_WIDTH_MAP[ele.dtype]
                                if DTYPE_WIDTH_MAP[ele.dtype] == 2:
                                    cast_flag = True
                            if width_local == 2 and cast_flag:
                                width_local = 3
                            if width_local > width:
                                width = width_local
                        return width
                    width = calc_width_mid(width)

                else:
                    def calc_width_tail(width):
                        # pylint: cell-var-from-loop
                        visited_list.append(tens) # pylint: disable=W0640
                        stack.remove(tens) # pylint: disable=W0640
                        for in_ten in tens.op.input_tensors: # pylint: disable=W0640
                            if in_ten not in stack and in_ten != tensor_c_gm:
                                stack.append(in_ten)
                        width_local = 0
                        cast_flag = False
                        for ele in stack:
                            width_local = width_local + DTYPE_WIDTH_MAP[ele.dtype]
                            if DTYPE_WIDTH_MAP[ele.dtype] == 2:
                                cast_flag = True
                        if width_local == 2 and cast_flag:
                            width_local = 3
                        if width_local > width:
                            width = width_local
                        return width

                    width = calc_width_tail(width)

            tmp_stack = []
            for ele in stack:
                tmp_stack.append(ele)
        return width

    def schedule_l1_mkn_l0_k_tiling():
        """
        CUT73 schedule method
        """
        # pylint: disable=too-many-locals, too-many-branches, too-many-statements
        if fusion_flag:
            sch[tensor_c_ub_inner].compute_inline()
        # tiling L1 and L0 on tensor_c
        l1_n_outer, l1_n_inner = sch[tensor_c].split(
            tensor_c.op.axis[tensor_len_c - 4], factor=n_l1_tile)
        l1_m_outer, l1_m_inner = sch[tensor_c].split(
            tensor_c.op.axis[tensor_len_c - 3], factor=m_l1_tile)
        l1_k_outer, l1_k_inner = sch[tensor_c].split(tensor_c.op.reduce_axis[0],
                                                     factor=k_l1_tile)

        l0_n_outer, l0_n_inner = sch[tensor_c].split(l1_n_inner,
                                                     factor=n_l0_tile)
        l0_m_outer, l0_m_inner = sch[tensor_c].split(l1_m_inner,
                                                     factor=m_l0_tile)
        l0_k_outer, l0_k_inner = sch[tensor_c].split(l1_k_inner,
                                                     factor=k_l0_tile)
        # |---gm_n---|     |--------gm_n-------|
        # nGM, mGM, kGM, n_l1_tile, m_l1_tile, k_l1_tile, n_l0_tile,
        # |------------------Nz--------------------|
        # m_l0_tile, mBurstSize, nBurstSize, k_l0_tile, kBurstSize
        sch[tensor_c].reorder(l1_n_outer, l1_m_outer, l1_k_outer,
                              l0_n_outer, l0_m_outer, l0_k_outer,
                              l0_n_inner, l0_m_inner,
                              tensor_c.op.axis[tensor_len_c - 2],
                              tensor_c.op.axis[tensor_len_c - 1],
                              l0_k_inner, tensor_c.op.reduce_axis[1])

        if tensor_a_ub is not None:
            sch[tensor_a_ub].compute_at(sch[tensor_c], l1_k_outer)
        if tensor_b_ub is not None:
            sch[tensor_b_ub].compute_at(sch[tensor_c], l1_k_outer)

        if tensor_a_ub_fract is not None:
            sch[tensor_a_ub_fract].compute_at(sch[tensor_c], l1_k_outer)
        if tensor_b_ub_fract is not None:
            sch[tensor_b_ub_fract].compute_at(sch[tensor_c], l1_k_outer)

        sch[tensor_a_l0a].compute_at(sch[tensor_c], l0_k_outer)
        sch[tensor_b_l0b].compute_at(sch[tensor_c], l0_k_outer)
        sch[tensor_a_l1].compute_at(sch[tensor_c], l1_k_outer)
        sch[tensor_b_l1].compute_at(sch[tensor_c], l1_k_outer)

        # tiling tensor_c_ub
        ub_n_outer, ub_n_inner = sch[tensor_c_ub].split(
            tensor_c_ub.op.axis[tensor_len_c - 4], factor=n_l1_tile)
        ub_m_outer, ub_m_inner = sch[tensor_c_ub].split(
            tensor_c_ub.op.axis[tensor_len_c - 3], factor=m_l1_tile)
        sch[tensor_c_ub].reorder(ub_n_outer, ub_m_outer,
                                 ub_n_inner, ub_m_inner,
                                 tensor_c_ub.op.axis[tensor_len_c - 2],
                                 tensor_c_ub.op.axis[tensor_len_c - 1])

        # tiling C
        axis_block = batch_outer
        res_ub = None
        if is_fractal_a and is_fractal_b:
            def sche_l1_mkn_l0_k_frac_frac():
                """
                fractail schedule config
                """
                axis_block = batch_outer
                factor_max = n_l1_tile
                inner_factor_m = m_l1_tile
                inner_factor_n = n_l1_tile
                res_ub = None
                if fusion_ele:
                    sch[tensor_c_gm].compute_inline()
                    res_ub = sch.cache_write(res, cce.scope_ubuf)

                    def get_inner_factor(res, tensor_c_gm, m_tile, n_tile, n_parts_value):
                        """
                        calcute L0C to UB split factor
                        """
                        # deal with width and cube size
                        width = _get_out_tensors_width(res)
                        un_max = get_soc_spec("UB_SIZE")
                        gm_width = DTYPE_WIDTH_MAP[tensor_c_gm.dtype] * 2
                        cube_size = block_in * block_out * gm_width
                        width = width // DTYPE_WIDTH_MAP[tensor_c_gm.dtype]

                        cube_size = cube_size * m_tile
                        factor_max = un_max // 2 // cube_size // width
                        factor_n = n_tile
                        factor_m = m_tile
                        if factor_max == 0:
                            factor_n = n_tile
                            factor_m = 1
                        elif factor_max < n_tile:
                            factor_n = 1
                            if (res.shape[tensor_len_c - 4].value) % n_parts_value == 0:
                                n_value = (res.shape[tensor_len_c - 4].value) // n_parts_value
                                if n_value % n_tile == 0:
                                    for idx in range(1, n_tile):
                                        if n_tile % idx == 0 and \
                                                factor_n < idx < (factor_max+1):
                                            factor_n = idx
                        return factor_max, factor_m, factor_n

                    factor_max, inner_factor_m, inner_factor_n = \
                        get_inner_factor(res, tensor_c_gm, m_l1_tile, n_l1_tile, n_factors)
                    support_split_tail = True
                    if inner_factor_m < m_l1_tile or inner_factor_n < n_l1_tile:
                        # set to False while need to disable fusion for split tail
                        support_split_tail = True
                    if not support_split_tail:
                        raise RuntimeError("Not support fusion while spliting tail.")
                n_outer_group = sch[res].split(res.op.axis[tensor_len_c - 4], nparts=n_factors)
                m_outer_group = sch[res].split(res.op.axis[tensor_len_c - 3], nparts=m_factors)

                n_outer, n_inner = sch[res].split(n_outer_group[1],
                                                  factor=n_l1_tile)
                m_outer, m_inner = sch[res].split(m_outer_group[1],
                                                  factor=m_l1_tile)

                if factor_max == 0:
                    # axit m is too large, need to split inner m
                    n_inner_outer, n_inner_inner = sch[res].split(n_inner, factor=inner_factor_n)
                    m_inner_outer, m_inner_inner = sch[res].split(m_inner, factor=inner_factor_m)
                    sch[res].reorder(n_outer_group[0], m_outer_group[0],
                                     n_outer, m_outer,
                                     m_inner_outer, n_inner_outer,
                                     n_inner_inner, m_inner_inner,
                                     res.op.axis[tensor_len_c - 2],
                                     res.op.axis[tensor_len_c - 1])
                else:
                    n_inner_outer, n_inner_inner = sch[res].split(n_inner, factor=inner_factor_n)
                    sch[res].reorder(n_outer_group[0], m_outer_group[0],
                                     n_outer, m_outer,
                                     n_inner_outer, n_inner_inner, m_inner,
                                     res.op.axis[tensor_len_c - 2],
                                     res.op.axis[tensor_len_c - 1])

                c_at_axis = m_outer
                c_ub_at_axis = n_inner_outer
                res_insn_axis = n_inner_inner

                sch[tensor_c].compute_at(sch[res], c_at_axis)
                if is_with_bias:
                    sch[tensor_bias_ub].compute_at(sch[res], c_at_axis)
                    sch[tensor_bias_l0c].compute_at(sch[res], c_at_axis)
                    sch[tensor_c_add_bias].compute_at(sch[res], c_at_axis)

                # multi kernel axis must be > 1
                if axis_block is None:
                    # modify there
                    if repeat_a:
                        sch[tensor_a_l1].allocate_at(sch[tensor_c], l1_m_outer,
                                                     run_once_axes=[n_outer])

                    if repeat_b:
                        sch[tensor_b_l1].allocate_at(sch[tensor_c], l1_n_outer,
                                                     run_once_axes=[m_outer])
                    fuse_list = []
                    fuse_list.append(n_outer_group[0])
                    fuse_list.append(m_outer_group[0])
                    axis_block = sch[res].fuse(*fuse_list)
                    # kkk
                    if axis_block is not None:
                        thread_block = tvm.thread_axis("blockIdx.x")
                        sch[res].bind(axis_block, thread_block)

                sch[tensor_c_ub].compute_at(sch[res], c_ub_at_axis)
                if res_ub is not None:
                    sch[res_ub].compute_at(sch[res], c_ub_at_axis)
                    sch[tensor_c_ub].reused_by(res_ub)
                for ten_in in tensor_ele_map:
                    sch[ten_in].compute_at(sch[res], c_ub_at_axis)
                return axis_block, res_ub, res_insn_axis

            axis_block, res_ub, res_insn_axis = sche_l1_mkn_l0_k_frac_frac()
        else:
            res_len = len(res.shape)
            c_ub_m_outer, c_ub_m_inner = sch[res].split(
                res.op.axis[res_len - 2], factor=block_in)
            c_ub_n_outer, c_ub_n_inner = sch[res].split(
                res.op.axis[res_len - 1], factor=block_out)

            if tensor_c_ub_fract is not None:
                c_ub_fract_m_outer, c_ub_fract_m_inner = \
                    sch[tensor_c_ub_fract].split(
                        tensor_c_ub_fract.op.axis[res_len - 2], factor=block_in)
                c_ub_fract_n_outer, c_ub_fract_n_inner = \
                    sch[tensor_c_ub_fract].split(
                        tensor_c_ub_fract.op.axis[res_len - 1],
                        factor=block_out)

            n_outer_group = sch[res].split(c_ub_n_outer, nparts=n_factors)

            m_outer, m_inner = sch[res].split(
                c_ub_m_outer, factor=m_l1_tile)
            n_outer, n_inner = sch[res].split(
                n_outer_group[1], factor=n_l1_tile)
            sch[res].reorder(m_outer, n_outer_group[0], n_outer,
                             m_inner, n_inner,
                             c_ub_m_inner, c_ub_n_inner)
            c_at_axis = n_outer
            c_ub_at_axis = n_outer
            res_insn_axis = m_inner

            sch[tensor_c].compute_at(sch[res], c_at_axis)
            if is_with_bias:
                sch[tensor_bias_ub].compute_at(sch[res], c_at_axis)
                sch[tensor_bias_l0c].compute_at(sch[res], c_at_axis)
                sch[tensor_c_add_bias].compute_at(sch[res], c_at_axis)

            if tensor_c_ub_fract is not None:
                fract_m_outer, fract_m_inner = sch[tensor_c_ub_fract].split(
                    c_ub_fract_m_outer, factor=m_l1_tile)
                fract_n_outer, fract_n_inner = sch[tensor_c_ub_fract].split(
                    c_ub_fract_n_outer, factor=n_l1_tile)
                sch[tensor_c_ub_fract].reorder(fract_m_outer, fract_n_outer,
                                               fract_m_inner, fract_n_inner,
                                               c_ub_fract_m_inner,
                                               c_ub_fract_n_inner)

            sch[tensor_c_ub].compute_at(sch[res], c_ub_at_axis)
            if fusion_flag and sqrt_flag:
                sch[tensor_sqrt].compute_at(sch[res], c_ub_at_axis)
            if fusion_flag and dequant_relu is not None:
                sch[dequant_relu].compute_at(sch[res], c_ub_at_axis)
            if tensor_c_ub_fract is not None:
                sch[tensor_c_ub_fract].compute_at(sch[res], c_ub_at_axis)

            # multi kernel axis must be > 1
            if axis_block is None:
                if res.shape[res_len - 2].value > (m_l1_tile * block_in):
                    axis_block = m_outer
                elif res.shape[res_len - 1].value > (n_l1_tile * block_out):
                    axis_block = n_outer_group[0]
                if axis_block is not None:
                    thread_block = tvm.thread_axis("blockIdx.x")
                    sch[res].bind(axis_block, thread_block)

        sch[tensor_a_l0a].emit_insn(tensor_a_l0a.op.axis[l0_tensor_len_a - 4],
                                    'dma_copy')
        sch[tensor_b_l0b].emit_insn(tensor_b_l0b.op.axis[l0_tensor_len_b - 4],
                                    'dma_copy')
        sch[tensor_a_l1].emit_insn(tensor_a_l1.op.axis[tensor_len_a - 4],
                                   'dma_copy')
        sch[tensor_b_l1].emit_insn(tensor_b_l1.op.axis[tensor_len_b - 4],
                                   'dma_copy')

        if tensor_a_ub is not None:
            sch[tensor_a_ub].emit_insn(tensor_a_ub.op.axis[tensor_len_a - 4],
                                       'dma_copy')
        if tensor_b_ub is not None:
            sch[tensor_b_ub].emit_insn(tensor_b_ub.op.axis[tensor_len_b - 4],
                                       'dma_copy')

        if tensor_a_ub_fract is not None:
            sch[tensor_a_ub_fract].emit_insn(
                tensor_a_ub_fract.op.axis[tensor_len_a - 4], 'vector_auto')
        if tensor_b_ub_fract is not None:
            sch[tensor_b_ub_fract].emit_insn(
                tensor_b_ub_fract.op.axis[tensor_len_b - 4], 'vector_auto')

        mad_dict = {"mad_pattern": mad_pattern,
                    "k_outer": [l1_k_outer, l0_k_outer]}
        if is_with_bias:
            sch[tensor_bias_ub].emit_insn(tensor_bias_ub.op.axis[0],
                                          'dma_copy')
            sch[tensor_bias_l0c].emit_insn(tensor_bias_l0c.op.axis[0],
                                           'dma_copy')
            sch[tensor_c_add_bias].emit_insn(tensor_c_add_bias.op.axis[0],
                                             'phony_insn')

            # reuse bias_l0c
            sch[tensor_bias_l0c].pragma(tensor_bias_l0c.op.axis[0],
                                        'reuse_output', 0)
            sch[tensor_c_add_bias].pragma(tensor_c_add_bias.op.axis[0],
                                          'replace_output', 0)
            sch[tensor_c].pragma(l0_n_inner, 'replace_output', 0)

            mad_dict["init_bias"] = 1
        # emit_insn

        # mad_pattern value: 0 for gemm, 1 for gemv
        # set pragma for k_outer
        sch[tensor_c].emit_insn(l0_n_inner, 'mad', mad_dict)
        # quantization config
        if not fusion_flag:
            if tensor_c_ub.op.attrs['scale_drq'].value == "ENABLE":
                # tensor_drq is second input for tensor_c_ub
                tensor_drq = tensor_c_ub.op.input_tensors[1]
                c_ub = sch.cache_read(tensor_drq, cce.scope_ubuf, [tensor_c_ub])
                sch[c_ub].emit_insn(c_ub.op.axis[0], 'dma_copy')
                if axis_block is not None:
                    sch[c_ub].compute_at(sch[res], axis_block)

                if tensor_c_ub.op.attrs['sqrt_out'].value == "SQRT":
                    # Sqrt Mode
                    sch[tensor_c_ub].pragma(ub_n_inner, 'deq_scale',
                                            'scalar_sqrt')
                else:
                    # No Sqrt Mode
                    sch[tensor_c_ub].pragma(ub_n_inner, 'deq_scale', 'scalar')
            else:
                sch[tensor_c_ub].emit_insn(ub_n_inner, 'dma_copy')
        else:
            # tensor_drq is second input for tensor_c_ub
            tensor_drq = tensor_c_ub.op.input_tensors[1]
            if sqrt_flag:
                c_ub = sch.cache_read(tensor_drq, cce.scope_ubuf,
                                      [tensor_c_ub, tensor_sqrt])
            else:
                c_ub = sch.cache_read(tensor_drq, cce.scope_ubuf, [tensor_c_ub])
            sch[c_ub].emit_insn(c_ub.op.axis[0], 'dma_copy')
            if axis_block is not None:
                sch[c_ub].compute_at(sch[res], axis_block)

            def _emit_dequant_insn():
                soc_ver = get_soc_spec("SOC_VERSION")
                if soc_ver in ("Ascend610", "Ascend620"):
                    sch[tensor_c_ub].emit_insn(ub_n_inner, 'dma_copy')
                else:
                    sch[tensor_c_ub].pragma(ub_n_inner, 'deq_scale', 'scalar')
                    if sqrt_flag:
                        sch[tensor_sqrt].emit_insn(tensor_sqrt.op.axis[0],
                                                   'vector_auto')

            _emit_dequant_insn()

        sch[res].emit_insn(res_insn_axis, 'dma_copy')
        if tensor_c_ub_fract is not None:
            sch[tensor_c_ub_fract].emit_insn(fract_m_outer, 'vector_auto')

        def emit_insn_simple():
            """
            emit insion base on simple axis
            """
            if fusion_flag and dequant_relu is not None:
                sch[dequant_relu].emit_insn(dequant_relu.op.axis[0],
                                            'vector_relu')
            if res_ub is not None:
                if res_ub.op.tag.find("|") != -1:
                    str_list = res_ub.op.tag.split("|")
                    insn = emit_insn_map.get(str_list[0])
                else:
                    insn = emit_insn_map.get(res_ub.op.tag)
                sch[res_ub].emit_insn(res_ub.op.axis[0], insn)
            for ten_in in tensor_ele_map:
                if ten_in.op.tag.find("|") != -1:
                    str_list = ten_in.op.tag.split("|")
                    insn = emit_insn_map.get(str_list[0])
                else:
                    insn = emit_insn_map.get(ten_in.op.tag)
                sch[ten_in].emit_insn(ten_in.op.axis[0], insn)

        emit_insn_simple()

        def open_double_buffer():
            # double buffer
            if m_l1_double_buffer == 2:
                sch[tensor_a_l1].double_buffer()
            sch[tensor_a_l0a].double_buffer()
            if n_l1_double_buffer == 2:
                sch[tensor_b_l1].double_buffer()
            sch[tensor_b_l0b].double_buffer()
            sch[tensor_c].double_buffer()

            if tensor_c_ub.op.tag != 'matmul' and \
                    tensor_c_ub.op.tag != 'matmul_gemv':
                sch[tensor_c_ub].double_buffer()

            if tensor_a_ub is not None:
                sch[tensor_a_ub].double_buffer()
            if tensor_b_ub is not None:
                sch[tensor_b_ub].double_buffer()
            if tensor_a_ub_fract is not None:
                sch[tensor_a_ub_fract].double_buffer()
            if tensor_b_ub_fract is not None:
                sch[tensor_b_ub_fract].double_buffer()
            if tensor_c_ub_fract is not None:
                sch[tensor_c_ub_fract].double_buffer()
            if res_ub is not None:
                sch[res_ub].double_buffer()

        open_double_buffer()

    def schedule_l1mn_l0_mkn_tiling():
        """
        CUT37 schedule method
        """
        # pylint: disable=too-many-locals, too-many-branches, too-many-statements
        is_fractal_out = is_fractal_a and is_fractal_b
        if fusion_flag:
            sch[tensor_c_ub_inner].compute_inline()
        # tiling tensor_c_ub
        ub_n_outer, ub_n_inner = sch[tensor_c_ub].split(
            tensor_c_ub.op.axis[tensor_len_c - 4], factor=n_l0_tile)
        ub_m_outer, ub_m_inner = sch[tensor_c_ub].split(
            tensor_c_ub.op.axis[tensor_len_c - 3], factor=m_l0_tile)
        sch[tensor_c_ub].reorder(ub_n_outer, ub_m_outer, ub_n_inner, ub_m_inner,
                                 tensor_c_ub.op.axis[tensor_len_c - 2],
                                 tensor_c_ub.op.axis[tensor_len_c - 1])

        # tiling L1 and L0 on tensor_c
        l0_n_outer, l0_n_inner = sch[tensor_c].split(
            tensor_c.op.axis[tensor_len_c - 4], factor=n_l0_tile)
        l0_m_outer, l0_m_inner = sch[tensor_c].split(
            tensor_c.op.axis[tensor_len_c - 3], factor=m_l0_tile)
        l0_k_outer, l0_k_inner = sch[tensor_c].split(tensor_c.op.reduce_axis[0],
                                                     factor=k_l0_tile)
        # |------------------Nz--------------------|
        # nL0TileOuter, mL0TileOuter, kL0TileOuter,
        # nL0TileInner, mL0TileInner, mBurstSize, nBurstSize, kL0TileInner, kBurstSize
        sch[tensor_c].reorder(l0_n_outer, l0_m_outer, l0_k_outer,
                              l0_n_inner, l0_m_inner,
                              tensor_c.op.axis[tensor_len_c - 2],
                              tensor_c.op.axis[tensor_len_c - 1],
                              l0_k_inner, tensor_c.op.reduce_axis[1])

        sch[tensor_a_l0a].compute_at(sch[tensor_c], l0_k_outer)
        sch[tensor_b_l0b].compute_at(sch[tensor_c], l0_k_outer)

        # tiling C
        no_tail = True
        axis_block = batch_outer
        res_ub = None
        fuse_list = []
        if is_fractal_out:
            def sche_l1mn_l0_mkn_frac_frac():
                """
                fractal schedule config
                """
                axis_block = batch_outer
                factor_max = n_l0_tile
                inner_factor_m = m_l0_tile
                inner_factor_n = n_l0_tile
                res_ub = None
                if fusion_ele:
                    res_ub = sch.cache_write(res, cce.scope_ubuf)
                    sch[tensor_c_gm].compute_inline()

                    def get_inner_factor(res, tensor_c_gm,
                                         n_tile_l1, m_tile_l0, n_tile_l0):
                        """
                        calcute L0C to UB split factor
                        """
                        # deal with width and cube size
                        width = _get_out_tensors_width(res)
                        un_max = get_soc_spec("UB_SIZE")
                        gm_width = DTYPE_WIDTH_MAP[tensor_c_gm.dtype] * 2
                        cube_size = block_in * block_out * gm_width
                        width = width // DTYPE_WIDTH_MAP[tensor_c_gm.dtype]
                        cube_size = cube_size * m_tile_l0
                        factor_max = un_max // 2 // cube_size // width

                        inner_factor_m = m_tile_l0
                        inner_factor_n = n_tile_l0
                        if factor_max == 0:
                            inner_factor_m = 1
                            inner_factor_n = n_tile_l0
                        elif factor_max < n_tile_l0:
                            inner_factor_n = 1
                            if (res.shape[tensor_len_c - 4].value) % n_tile_l1 == 0:
                                n_value = (res.shape[tensor_len_c - 4].value) // n_tile_l1
                                if n_value % n_tile_l0 == 0:
                                    for idx in range(1, n_tile_l0):
                                        if n_tile_l0 % idx == 0 and \
                                                inner_factor_n < idx < (factor_max+1):
                                            inner_factor_n = idx
                        return factor_max, inner_factor_m, inner_factor_n

                    factor_max, inner_factor_m, inner_factor_n = \
                        get_inner_factor(res, tensor_c_gm,
                                         n_l1_tile, m_l0_tile, n_l0_tile)

                    support_split_tail = True
                    if inner_factor_m < m_l1_tile or inner_factor_n < n_l1_tile:
                        # set to False while need to disable fusion for split tail
                        support_split_tail = True
                    if not support_split_tail:
                        raise RuntimeError("Not support fusion while spliting tail.")

                c_l1_n_outer, c_l1_n_inner = sch[res].split(
                    res.op.axis[tensor_len_c - 4], factor=n_l1_tile)
                c_l1_m_outer, c_l1_m_inner = sch[res].split(
                    res.op.axis[tensor_len_c - 3], factor=m_l1_tile)
                c_l0_n_outer, c_l0_n_inner = sch[res].split(c_l1_n_inner,
                                                            factor=n_l0_tile)
                c_l0_m_outer, c_l0_m_inner = sch[res].split(c_l1_m_inner,
                                                            factor=m_l0_tile)

                if factor_max == 0:
                    # axit m is too large, need to split inner m
                    c_n_inner_outer, c_n_inner_inner = \
                        sch[res].split(c_l0_n_inner, factor=inner_factor_n)
                    c_m_inner_outer, c_m_inner_inner = \
                        sch[res].split(c_l0_m_inner, factor=inner_factor_m)

                    sch[res].reorder(c_l1_n_outer, c_l1_m_outer,
                                     c_l0_n_outer, c_l0_m_outer,
                                     c_m_inner_outer, c_n_inner_outer,
                                     c_n_inner_inner, c_m_inner_inner,
                                     res.op.axis[tensor_len_c - 2],
                                     res.op.axis[tensor_len_c - 1])
                else:
                    c_n_inner_outer, c_n_inner_inner = \
                        sch[res].split(c_l0_n_inner, factor=inner_factor_n)

                    sch[res].reorder(c_l1_n_outer, c_l1_m_outer,
                                     c_l0_n_outer, c_l0_m_outer,
                                     c_n_inner_outer, c_n_inner_inner,
                                     c_l0_m_inner,
                                     res.op.axis[tensor_len_c - 2],
                                     res.op.axis[tensor_len_c - 1])

                c_at_axis = c_l0_m_outer
                c_ub_at_axis = c_n_inner_outer
                res_insn_axis = c_n_inner_inner

                def out_ub_compute_at_sch(
                        ten_c_ub, ten_ele_map, res_ub, comp_at_axis):
                    """
                    C_UB tensor compute at res
                    """
                    sch[ten_c_ub].compute_at(sch[res], comp_at_axis)
                    if res_ub is not None:
                        sch[res_ub].compute_at(sch[res], comp_at_axis)
                        sch[ten_c_ub].reused_by(res_ub)
                    for ten_in in ten_ele_map:
                        sch[ten_in].compute_at(sch[res], comp_at_axis)

                # multi kernel axis must be > 1
                if axis_block is None:
                    n_block_cnt = (res.shape[tensor_len_c - 4].value +
                                   n_l1_tile - 1) // n_l1_tile
                    m_block_cnt = (res.shape[tensor_len_c - 4].value +
                                   m_l1_tile - 1) // m_l1_tile
                    if n_block_cnt > core_thres:
                        sch[tensor_a_l1].compute_at(sch[res], c_l0_m_outer)
                        sch[tensor_b_l1].compute_at(sch[res], c_l0_n_outer)

                        out_ub_compute_at_sch(tensor_c_ub, tensor_ele_map,
                                              res_ub, c_ub_at_axis)

                        axis_block = c_l1_n_outer
                        if check_mini_core_num():
                            # just for mini config
                            temp_core = axis_block
                            axis_block, _ = sch[res].split(temp_core, nparts=2)

                    elif n_block_cnt > 1 or m_block_cnt > 1:
                        fuse_list.append(c_l1_n_outer)
                        fuse_list.append(c_l1_m_outer)
                        axis_block = sch[res].fuse(*fuse_list)
                        c_l1_n_outer = None
                        c_l1_m_outer = None

                        sch[tensor_a_l1].compute_at(sch[tensor_c], l0_m_outer)
                        sch[tensor_b_l1].compute_at(sch[tensor_c], l0_n_outer)

                        out_ub_compute_at_sch(tensor_c_ub, tensor_ele_map,
                                              res_ub, c_ub_at_axis)
                    else:
                        fuse_list.append(c_l1_n_outer)
                        fuse_list.append(c_l1_m_outer)
                        fuse_list.append(c_l0_n_outer)
                        fuse_list.append(c_l0_m_outer)
                        axis_block = sch[res].fuse(*fuse_list)
                        c_l1_n_outer = None
                        c_l1_m_outer = None
                        c_l0_n_outer = None
                        c_l0_m_outer = None
                        c_at_axis = axis_block

                        sch[tensor_a_l1].compute_at(sch[tensor_c], l0_k_outer)
                        sch[tensor_b_l1].compute_at(sch[tensor_c], l0_k_outer)

                        out_ub_compute_at_sch(tensor_c_ub, tensor_ele_map,
                                              res_ub, c_ub_at_axis)

                    thread_block = tvm.thread_axis("blockIdx.x")
                    sch[res].bind(axis_block, thread_block)

                else:
                    sch[tensor_a_l1].compute_at(sch[res], c_l0_m_outer)
                    sch[tensor_b_l1].compute_at(sch[res], c_l0_n_outer)

                    out_ub_compute_at_sch(tensor_c_ub, tensor_ele_map,
                                          res_ub, c_ub_at_axis)

                # config compute_at just by c_at_axis for fractal mode
                sch[tensor_c].compute_at(sch[res], c_at_axis)
                if is_with_bias:
                    sch[tensor_c_add_bias].compute_at(sch[res], c_at_axis)
                    sch[tensor_bias_l0c].compute_at(sch[res], c_at_axis)
                    sch[tensor_bias_ub].compute_at(sch[res], c_at_axis)

                return axis_block, c_l1_n_outer, c_l1_m_outer, res_ub, res_insn_axis

            axis_block, c_l1_n_outer, c_l1_m_outer, res_ub, res_insn_axis = \
                sche_l1mn_l0_mkn_frac_frac()
        else:
            res_len = len(res.shape)
            no_tail = (res.shape[res_len - 2].value % cce.BLOCK_IN) == 0
            if mad_pattern == cce.GEVM_MODE:
                no_tail = True
            if tensor_c_ub_fract is not None:
                c_ub_fract_m_outer, c_ub_fract_m_inner = \
                    sch[tensor_c_ub_fract].split(
                        tensor_c_ub_fract.op.axis[res_len - 2],
                        factor=block_in)
                c_ub_fract_n_outer, c_ub_fract_n_inner = \
                    sch[tensor_c_ub_fract].split(
                        tensor_c_ub_fract.op.axis[res_len - 1],
                        factor=block_out)

                c_l1_fract_m_outer, c_l1_fract_m_inner = \
                    sch[tensor_c_ub_fract].split(
                        c_ub_fract_m_outer, factor=m_l1_tile)
                c_l1_fract_n_outer, c_l1_fract_n_inner = \
                    sch[tensor_c_ub_fract].split(
                        c_ub_fract_n_outer, factor=n_l1_tile)
                c_l0_fract_m_outer, c_l0_fract_m_inner = \
                    sch[tensor_c_ub_fract].split(
                        c_l1_fract_m_inner, factor=m_l0_tile)
                c_l0_fract_n_outer, c_l0_fract_n_inner = \
                    sch[tensor_c_ub_fract].split(
                        c_l1_fract_n_inner, factor=n_l0_tile)

                sch[tensor_c_ub_fract].reorder(
                    c_l1_fract_n_outer, c_l1_fract_m_outer,
                    c_l0_fract_m_outer, c_l0_fract_n_outer,
                    c_l0_fract_n_inner, c_l0_fract_m_inner,
                    c_ub_fract_m_inner, c_ub_fract_n_inner)

            c_ub_m_outer, c_ub_m_inner = sch[res].split(
                res.op.axis[res_len - 2], factor=block_in)
            c_ub_n_outer, c_ub_n_inner = sch[res].split(
                res.op.axis[res_len - 1], factor=block_out)
            c_l1_m_outer, c_l1_m_inner = sch[res].split(
                c_ub_m_outer, factor=m_l1_tile)
            c_l1_n_outer, c_l1_n_inner = sch[res].split(
                c_ub_n_outer, factor=n_l1_tile)
            c_l0_m_outer, c_l0_m_inner = sch[res].split(c_l1_m_inner,
                                                        factor=m_l0_tile)
            c_l0_n_outer, c_l0_n_inner = sch[res].split(c_l1_n_inner,
                                                        factor=n_l0_tile)
            sch[res].reorder(c_l1_n_outer, c_l1_m_outer,
                             c_l0_m_outer, c_l0_n_outer,
                             c_l0_n_inner, c_l0_m_inner,
                             c_ub_m_inner, c_ub_n_inner)
            if no_tail:
                res_insn_axis = c_l0_n_inner
            else:
                res_insn_axis = c_ub_m_inner

            sch[tensor_c].compute_at(sch[tensor_c_ub], ub_m_outer)
            if is_with_bias:
                sch[tensor_c_add_bias].compute_at(sch[tensor_c_ub], ub_m_outer)
                sch[tensor_bias_l0c].compute_at(sch[tensor_c_ub], ub_m_outer)
                sch[tensor_bias_ub].compute_at(sch[tensor_c_ub], ub_m_outer)

            if axis_block is None:
                # multi kernel axis must be > 1
                if res.shape[res_len - 1].value > n_l1_tile * block_out:
                    axis_block = c_l1_n_outer
                    c_l1_n_outer = None
                elif res.shape[res_len - 2].value > m_l1_tile * block_in:
                    axis_block = c_l1_m_outer
                    c_l1_n_outer = None
                    c_l1_m_outer = None

                if axis_block is not None:
                    thread_block = tvm.thread_axis("blockIdx.x")
                    sch[res].bind(axis_block, thread_block)
        if c_l1_m_outer is not None:
            m_outer_axis = c_l1_m_outer
        else:
            m_outer_axis = axis_block
        if c_l1_n_outer is not None:
            n_outer_axis = c_l1_n_outer
        else:
            n_outer_axis = axis_block

        if tensor_a_ub is not None:
            sch[tensor_a_ub].compute_at(sch[res], m_outer_axis)
        if tensor_b_ub is not None:
            sch[tensor_b_ub].compute_at(sch[res], n_outer_axis)

        if tensor_a_ub_fract is not None:
            sch[tensor_a_ub_fract].compute_at(sch[res], m_outer_axis)
        if tensor_b_ub_fract is not None:
            sch[tensor_b_ub_fract].compute_at(sch[res], n_outer_axis)
        if not is_fractal_out:
            sch[tensor_c_ub].compute_at(sch[res], c_l0_m_outer)
            if fusion_flag and sqrt_flag:
                sch[tensor_sqrt].compute_at(sch[res], c_l0_m_outer)
            if fusion_flag and dequant_relu is not None:
                sch[dequant_relu].compute_at(sch[res], c_l0_m_outer)
            if tensor_c_ub_fract is not None:
                sch[tensor_c_ub_fract].compute_at(sch[res], c_l0_m_outer)

            sch[tensor_a_l1].compute_at(sch[res], m_outer_axis)
            sch[tensor_b_l1].compute_at(sch[res], n_outer_axis)
        # emit_insn
        sch[tensor_a_l1].emit_insn(tensor_a_l1.op.axis[tensor_len_a - 4],
                                   'dma_copy')
        sch[tensor_b_l1].emit_insn(tensor_b_l1.op.axis[tensor_len_b - 4],
                                   'dma_copy')

        sch[tensor_a_l0a].emit_insn(tensor_a_l0a.op.axis[l0_tensor_len_a - 4],
                                    'dma_copy')
        sch[tensor_b_l0b].emit_insn(tensor_b_l0b.op.axis[l0_tensor_len_b - 4],
                                    'dma_copy')
        if tensor_a_ub is not None:
            sch[tensor_a_ub].emit_insn(tensor_a_ub.op.axis[tensor_len_a - 4],
                                       'dma_copy')
        if tensor_b_ub is not None:
            sch[tensor_b_ub].emit_insn(tensor_b_ub.op.axis[tensor_len_b - 4],
                                       'dma_copy')
        if tensor_a_ub_fract is not None:
            sch[tensor_a_ub_fract].emit_insn(
                tensor_a_ub_fract.op.axis[tensor_len_a - 4], 'vector_auto')
        if tensor_b_ub_fract is not None:
            sch[tensor_b_ub_fract].emit_insn(
                tensor_b_ub_fract.op.axis[tensor_len_b - 4], 'vector_auto')

        if tensor_c_ub_fract is not None:
            sch[tensor_c_ub_fract].emit_insn(
                c_l0_fract_n_inner, 'vector_auto')

        mad_dict = {"mad_pattern": mad_pattern, "k_outer": l0_k_outer}
        if is_with_bias:
            sch[tensor_bias_ub].emit_insn(tensor_bias_ub.op.axis[0], 'dma_copy')
            sch[tensor_bias_l0c].emit_insn(tensor_bias_l0c.op.axis[0],
                                           'dma_copy')

            sch[tensor_c_add_bias].emit_insn(tensor_c_add_bias.op.axis[0],
                                             'phony_insn')
            sch[tensor_bias_l0c].pragma(tensor_bias_l0c.op.axis[0],
                                        'reuse_output', 0)
            sch[tensor_c_add_bias].pragma(tensor_c_add_bias.op.axis[0],
                                          'replace_output', 0)
            sch[tensor_c].pragma(l0_n_inner, 'replace_output', 0)

            mad_dict["init_bias"] = 1

        # mad_pattern value: 0 for gemm, 1 for gemv
        sch[tensor_c].emit_insn(l0_n_inner, 'mad', mad_dict)
        # quantization config
        if not fusion_flag:
            if tensor_c_ub.op.attrs['scale_drq'].value == "ENABLE":
                # tensor_drq is second input for tensor_c_ub
                tensor_drq = tensor_c_ub.op.input_tensors[1]
                c_ub = sch.cache_read(tensor_drq, cce.scope_ubuf, [tensor_c_ub])
                sch[c_ub].emit_insn(c_ub.op.axis[0], 'dma_copy')
                if axis_block is not None:
                    sch[c_ub].compute_at(sch[res], axis_block)
                if tensor_c_ub.op.attrs['sqrt_out'].value == "SQRT":
                    # Sqrt Mode
                    sch[tensor_c_ub].pragma(ub_n_inner, 'deq_scale',
                                            'scalar_sqrt')
                else:
                    # No Sqrt Mode
                    sch[tensor_c_ub].pragma(ub_n_inner, 'deq_scale', 'scalar')
            else:
                sch[tensor_c_ub].emit_insn(ub_n_inner, 'dma_copy')
        else:
            # tensor_drq is second input for tensor_c_ub
            tensor_drq = tensor_c_ub.op.input_tensors[1]
            if sqrt_flag:
                c_ub = sch.cache_read(tensor_drq, cce.scope_ubuf,
                                      [tensor_c_ub, tensor_sqrt])
            else:
                c_ub = sch.cache_read(tensor_drq, cce.scope_ubuf, [tensor_c_ub])
            sch[c_ub].emit_insn(c_ub.op.axis[0], 'dma_copy')
            if axis_block is not None:
                sch[c_ub].compute_at(sch[res], axis_block)

            def _emit_dequant_insn():
                soc_ver = get_soc_spec("SOC_VERSION")
                if soc_ver in ("Ascend610", "Ascend620"):
                    sch[tensor_c_ub].emit_insn(ub_n_inner, 'dma_copy')
                else:
                    sch[tensor_c_ub].pragma(ub_n_inner, 'deq_scale', 'scalar')
                    if sqrt_flag:
                        sch[tensor_sqrt].emit_insn(tensor_sqrt.op.axis[0],
                                                   'vector_auto')

            _emit_dequant_insn()

        sch[res].emit_insn(res_insn_axis, 'dma_copy')

        def emit_insn_simple():
            """
            emit insion base on simple axis
            """
            if fusion_flag and dequant_relu is not None:
                sch[dequant_relu].emit_insn(dequant_relu.op.axis[0],
                                            'vector_relu')
            if res_ub is not None:
                if res_ub.op.tag.find("|") != -1:
                    str_list = res_ub.op.tag.split("|")
                    insn = emit_insn_map.get(str_list[0])
                else:
                    insn = emit_insn_map.get(res_ub.op.tag)
                sch[res_ub].emit_insn(res_ub.op.axis[0], insn)
            for ten_in in tensor_ele_map:
                if ten_in.op.tag.find("|") != -1:
                    str_list = ten_in.op.tag.split("|")
                    insn = emit_insn_map.get(str_list[0])
                else:
                    insn = emit_insn_map.get(ten_in.op.tag)
                sch[ten_in].emit_insn(ten_in.op.axis[0], insn)

        emit_insn_simple()

        def open_double_buffer():
            """
            set all tensors double buffer
            """
            if m_l1_double_buffer == 2:
                sch[tensor_a_l1].double_buffer()
            sch[tensor_a_l0a].double_buffer()
            if n_l1_double_buffer == 2:
                sch[tensor_b_l1].double_buffer()
            sch[tensor_b_l0b].double_buffer()
            sch[tensor_c].double_buffer()

            if tensor_c_ub.op.tag != 'matmul' and \
                    tensor_c_ub.op.tag != 'matmul_gemv':
                sch[tensor_c_ub].double_buffer()

            if tensor_a_ub is not None:
                sch[tensor_a_ub].double_buffer()
            if tensor_b_ub is not None:
                sch[tensor_b_ub].double_buffer()
            if tensor_a_ub_fract is not None:
                sch[tensor_a_ub_fract].double_buffer()
            if tensor_b_ub_fract is not None:
                sch[tensor_b_ub_fract].double_buffer()
            if tensor_c_ub_fract is not None:
                sch[tensor_c_ub_fract].double_buffer()
            if res_ub is not None:
                sch[res_ub].double_buffer()

        # double_buffer
        open_double_buffer()

    if k_shape == k_l1_shape:
        schedule_l1mn_l0_mkn_tiling()
    elif m_l0_shape == m_l1_shape and n_l0_shape == n_l1_shape:
        schedule_l1_mkn_l0_k_tiling()
    else:
        raise RuntimeError("unhandled tiling")
    return True
