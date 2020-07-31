#!/usr/bin/env python
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

common function
"""
import math
import collections
from functools import reduce as functools_reduce
from te import tvm
from te import platform as cceconf
from . import pattern

# fake node label
FAKE_NODE_TAG = "elewise_empty_intrin"
SET_GM_SCOPE_TAG = "elewise_set_gm_scope"

REDUCE_OP_TAG_LABEL = "reduce_"
BROADCAST_TAG_LABEL = "broadcast_"

FAKE_NODE_PRAGMA = "phony_insn"
BROADCAST_ALIGN_PRAGMA = "vector_dup"
BROADCAST_TRANSPOSE = "vector_broadcast_transpose"
VECTOR_AUTO_PRAGMA = "vector_auto"
DMA_COPY_PRAGMA = "dma_copy"

REDUCE_MULTI_PRIME_KEY = "_unique_name_reduce_multi"

FAKE_NODE_FUSE_FAILED = -1

MULTI_CORE_UNIT = 1024
REDUCE_AXIS_SIZE = 1

VECTOR_ONE_REPEAT_UNIT = 128
VECTOR_ONE_BLOCK_UNIT = 16
VECTOR_ONE_REPEAT_BLOCK = 8
MAX_TYPE_SIZE_UNIT = 2
MIN_TYPE_SIZE_UNIT = 0.5

LOG_SOFTMAX_LIMIT = 80000
LOG_SOFTMAX_MATCH = 30528

MULTI_REDUCE = True
MULTI_ELEMWISE = True
MULTI_WORKSPACE = True
MULTI_WORKSPACE_ALL = False

PATTERN_OPTIMAZE = True
PATTERN_LIMIT = False

# the bit of dtype/16 map
DTYPE_WIDTH_MAP = {"float16": 1,
                   "float32": 2,
                   "int32": 2,
                   "int16": 1,
                   "uint16": 1,
                   "int8": 0.5,
                   "uint8": 0.5,
                   "bool": 0.5}

REDUCE_ATOMIC_SUPPORT = {"Ascend910": {"tag": "reduce_sum",
                                       "dtype": "float32"}, }

DEFAULT_INDEX = -1
INIT_COUNT = 0
INIT_SIZE = 1

TILING_RADICAL = 0
TILING_CONSERVATIVE = 1


def get_split_axis(shape, max_ub_count):
    """
    obtains the axis and factor of segmentation based on the number of data
    records that can be stored at a time in UB.

    Parameters
    ----------
    shape : tensor shape
    max_ub_count : data count

    Returns
    -------
    int : split rfactor
    int : split axis
    """
    # find the split axis, shape = (shape[0], ..., shape[split_axis], shape[-1])
    # so that shape[split_axis]*shape[split_axis + 1]*...*shape[-1] < max_ub_count
    # and shape[split_axis - 1]**shape[split_axis + 1]*...*shape[-1] > max_ub_count
    rfactor = max_ub_count
    axis = len(shape) - 1

    for i, it_n in enumerate(reversed(shape)):
        if max_ub_count < it_n:
            break
        rfactor = it_n
        max_ub_count = max_ub_count // it_n
        # exactly divided by the current axis
        if max_ub_count == 1 or i == len(shape) - 1:
            return rfactor, axis

        # obtains the currently traversed axis.
        if i != len(shape) - 1:
            axis -= 1

    # if the calculated value is less than the current axis.
    # calculate the factor corresponding to the maximum common value of the
    # current axis.
    for i in range(max_ub_count, 1, -1):
        if shape[axis] % i == 0:
            rfactor = i
            return rfactor, axis

    # last axis prime number
    if axis == len(shape) - 1:
        rfactor = 1
        return rfactor, axis

    # the segmentation factor is not found. The split axis is the next axis.
    return rfactor, axis + 1


def get_align_factor(dtype):
    """
    get_align_factor
    """
    # base on the diff data type, get the align_factor
    align_factor = 16
    dtype_bytes = 2
    if dtype in ('int8', 'uint8'):
        align_factor = 32
        dtype_bytes = 1
    elif dtype in ('float16', 'int16', 'uint16'):
        align_factor = 16
        dtype_bytes = 2
    else:
        align_factor = 8
        dtype_bytes = 4
    return align_factor, dtype_bytes


def get_dst_tensor_map(reslist, tensor_map):
    """
    get the dst_tensor list of the tensor with more than one dst_tensor
    tensor_map = {input: outputlist}
    """
    for out_tensor in reslist:
        for in_tensor in list(out_tensor.op.input_tensors):
            if in_tensor in tensor_map:
                if out_tensor not in tensor_map[in_tensor]:
                    tensor_map[in_tensor].append(out_tensor)
            else:
                tensor_map[in_tensor] = [out_tensor]
                get_dst_tensor_map([in_tensor], tensor_map)


def shape_to_list(shape):
    """
    translate tvm.shape to list type in python
    """
    tmp = []
    for i in shape:
        if isinstance(i, tvm.expr.Var):
            tmp.append(i)
        else:
            tmp.append(i.value)
    return tmp


def is_prime_number(value):
    """
    check the value is prime number or not.
    """
    if value < 2:
        return False
    for i in range(2, int(math.sqrt(value)) + 1):
        if value % i == 0:
            return False
    return True


def get_reduce_axis_num(reduce_tensor):
    """
    get reduce axis num
    """
    data_axis_var = reduce_tensor.op.body[0].source[0].args
    reduce_axis_var = []
    for i in reduce_tensor.op.reduce_axis:
        reduce_axis_var.append(i.var)

    axis_num = []
    for ax_var in reduce_axis_var:
        num = 0
        for i in data_axis_var:
            if i.same_as(ax_var):
                axis_num.append(num)
            num += 1

    return axis_num


def get_bits_of(dtype):
    """
    calculate bits of dtype of TVM
    Parameters
    ----------
    dtype : string
        dtype of TVM

    Returns
    -------
    ret : int
        bit length of dtype.
    """
    index = 0
    for i in dtype:
        if i.isdigit():
            break
        index += 1
    return int(dtype[index:])


def get_max_divisor(num, x_var=None):
    """
    find the maximum divisor of a number (not counting itself, and after x)
    """
    if x_var is None:
        divisor = num // 2
    else:
        divisor = min(num // 2, x_var)

    while divisor > 1:
        if num % divisor == 0:
            break
        else:
            divisor = divisor - 1
    return divisor


def get_greatest_common_divisor(m_var, n_var):
    """
    greatest common divisor
    """
    if m_var % n_var == 0:
        return n_var
    while m_var % n_var != 0:
        m_var, n_var = n_var, m_var % n_var
    return n_var


def get_least_common_multiple(m_var, n_var):
    """
    least common multiple
    product of two numbers = least common multiple * greatest common divisor
    lcm = m * n // gcd
    """
    return (m_var * n_var) // get_greatest_common_divisor(m_var, n_var)


def get_mod2_count(num):
    """
    The count of remainders for 2
    """
    count = 0
    while num % 2 == 0:
        num = num // 2
        count = count + 1
    return count


def get_shape_size_ext(shape, value=None):
    """
    shape size
    """
    if not shape:
        shape_size = 0
    else:
        shape_size = functools_reduce(lambda i, j: i * j, shape)

    if isinstance(value, int) and shape_size == 0:
        shape_size = value
    elif isinstance(value, int) and shape_size != 0:
        shape_size = shape_size * value

    return shape_size


# pylint: disable=too-many-branches, too-many-return-statements
def tiling_from_front_to_back(shape, max_size, align_factor=None,
                              is_divisible=False):
    """
    tiling_from_front_to_back, using for block_tiling
    shape:
    max_size:
    is_divisible: if true, do force divisibility.
    align_factor: if not None, try alignment, but do not force alignment.
                If the data does not match the alignment, it can also be misaligned.
    """
    bound_size = 1
    split_axis = 0
    # pylint: disable=consider-using-enumerate
    for i in range(len(shape)):
        bound_size = shape[i] * bound_size
        split_axis = i
        if bound_size >= max_size:
            break
    # 1. bound_size <= max_size
    if bound_size <= max_size:
        outer = shape[split_axis]
        inner = 1
        return outer, inner, split_axis

    # 2. bound_size > max_size
    outer = max_size * shape[split_axis] // bound_size
    inner = (shape[split_axis] + outer - 1) // outer

    # 2.1 do force divisibility.
    if is_divisible:
        outer = get_max_divisor(shape[split_axis], outer)
        inner = shape[split_axis] // outer

    # 2.2 try 256 or 32 alignment, but do not force alignment.
    # If the data does not match the alignment, it can also be misaligned.
    if align_factor and align_factor is not None:
        mod2count_align = get_mod2_count(align_factor)
        mod2count_reduceshape = get_mod2_count(bound_size // shape[split_axis])
        mod2count_splitaxis = get_mod2_count(shape[split_axis])
        mod2count_all = mod2count_reduceshape + mod2count_splitaxis

        # 2.2.1 if: mod2count_reduceshape + mod2count_splitaxis < mod2count_all
        if mod2count_align > mod2count_all:
            return outer, inner, split_axis
        # 2.2.2 elif: mod2count_reduceshape >= mod2count_all
        elif mod2count_align <= mod2count_reduceshape:
            return outer, inner, split_axis
        # 2.2.3 else: mod2count_reduceshape + mod2count_splitaxis >= mod2count_all,
        # and mod2count_reduceshape < mod2count_all
        else:
            factor = 2 ** (mod2count_align - mod2count_reduceshape)
            # 2.2.3.1 if: is_divisible
            if is_divisible:
                outer_update = outer
                while outer_update >= factor:
                    if outer_update % factor == 0:
                        outer = outer_update
                        inner = (shape[split_axis] + outer - 1) // outer
                        break
                    outer_update = get_max_divisor(shape[split_axis],
                                                   outer_update - 1)

                return outer, inner, split_axis

            # pylint: disable=no-else-return
            # 2.2.3.2 else: not is_divisible
            if outer < factor:
                return outer, inner, split_axis
            else:
                outer = outer // factor * factor
                inner = (shape[split_axis] + outer - 1) // outer
                return outer, inner, split_axis

    return outer, inner, split_axis


# pylint: disable=too-many-branches, too-many-return-statements
def tiling_from_back_to_front(shape, max_size, align_factor=None,
                              is_divisible=False):
    """
    tiling_from_back_to_front, using for ub_tiling
    shape:
    max_size:
    is_divisible: if true, do force divisibility.
    align_factor: if not None, try alignment, but do not force alignment.
                If the data does not match the alignment, it can also be misaligned.
    """
    bound_size = 1
    split_axis = len(shape) - 1
    for i in reversed(range(len(shape))):
        bound_size = shape[i] * bound_size
        split_axis = i
        if bound_size >= max_size:
            break
    # 1. bound_size <= max_size
    if bound_size <= max_size:
        inner = shape[split_axis]
        outer = 1
        return outer, inner, split_axis

    # 2. bound_size > max_size
    inner = max_size * shape[split_axis] // bound_size
    outer = (shape[split_axis] + inner - 1) // inner

    # 2.1 do force divisibility.
    if is_divisible:
        inner = get_max_divisor(shape[split_axis], inner)
        outer = shape[split_axis] // inner

    # 2.2 try 256 or 32 alignment, but do not force alignment.
    # If the data does not match the alignment, it can also be misaligned.
    if align_factor and align_factor is not None:
        mod2count_align = get_mod2_count(align_factor)
        mod2count_reduceshape = get_mod2_count(bound_size // shape[split_axis])
        mod2count_splitaxis = get_mod2_count(shape[split_axis])
        mod2count_all = mod2count_reduceshape + mod2count_splitaxis

        # pylint: disable=no-else-return
        # 2.2.1 if: mod2count_reduceshape + mod2count_splitaxis < mod2count_all
        if mod2count_align > mod2count_all:
            return outer, inner, split_axis
        # 2.2.2 elif: mod2count_reduceshape >= mod2count_all
        elif mod2count_align <= mod2count_reduceshape:
            return outer, inner, split_axis
        # 2.2.3 else: mod2count_reduceshape + mod2count_splitaxis >= mod2count_all,
        # and mod2count_reduceshape < mod2count_all
        else:
            factor = 2 ** (mod2count_align - mod2count_reduceshape)
            # 2.2.3.1 if: is_divisible
            if is_divisible:
                inner_update = inner
                while inner_update >= factor:
                    if inner_update % factor == 0:
                        inner = inner_update
                        outer = (shape[split_axis] + inner - 1) // inner
                        break
                    inner_update = get_max_divisor(shape[split_axis],
                                                   inner_update - 1)
                return outer, inner, split_axis

            # pylint: disable=no-else-return
            # 2.2.3.2 else: not is_divisible
            if inner < factor:
                return outer, inner, split_axis
            else:
                inner = inner // factor * factor
                outer = (shape[split_axis] + inner - 1) // inner
                return outer, inner, split_axis

    return outer, inner, split_axis


def fake_node_fuse_fun(tensors):
    """
    fuse tensors into a fake node by mul compute with 0.
    """
    dtype = tensors[0].dtype
    shape = shape_to_list(tensors[0].shape)
    dim = len(shape)

    temp_tensors = tensors[:]
    visited = []
    while temp_tensors:
        tensor = temp_tensors[0]
        temp_tensors.remove(tensor)
        if tensor not in visited:
            visited.append(tensor)
            temp_tensors = temp_tensors + list(tensor.op.input_tensors)
            if len(tensor.shape) != dim:
                return FAKE_NODE_FUSE_FAILED
            for i in range(dim):
                if shape[i] < tensor.shape[i].value:
                    shape[i] = tensor.shape[i].value

    def phony_insn_fuse(*indice):
        res = tvm.const(1, dtype)
        for tensor in tensors:
            # get full indice order
            cur_index = []
            for i in range(dim):
                if tensor.shape[i].value == shape[i]:
                    cur_index.append(indice[i])
                else:
                    cur_index.append(indice[i] % tensor.shape[i].value)
            res *= tvm.expr.Cast(dtype, tensor(*cur_index))
        return res

    with tvm.tag_scope(FAKE_NODE_TAG):
        res = tvm.compute(shape, phony_insn_fuse, name="fake_node")
    return res


def ceil(value_a, value_b):
    """
    get up multi value_b
    """
    return int((value_a + value_b - 1) // value_b)


def align(value_a, value_b):
    """
    get up align value_b
    """
    return ceil(value_a, value_b) * value_b


def gcd(value_a, value_b):
    """
    get gcd value
    """
    if value_a < value_b:
        value_a, value_b = value_b, value_a
    while value_a % value_b != 0:
        value_a, value_b = value_b, value_a % value_b
    return value_b


def get_limit_coef(value_a, value_b):
    """
    get limit coef value
    """
    coef = []
    cur_num = 1
    while cur_num <= value_b:
        if value_a % cur_num == 0:
            coef.append(cur_num)
        cur_num += 1
    return coef


def get_shape_size(shape):
    """
    get shape size
    """
    size = INIT_SIZE
    for dim_v in shape:
        size *= dim_v
    return size


def get_block_factor_conservative(shape, barrier, factor):
    """
    get block factor conservative
    """
    visit = [[], ]

    res_idx = []
    res_factor = []
    res_efficiency = DEFAULT_INDEX

    # updata result
    def compare_cut_result(cur_size, cur_idxs, cur_factor):
        """
        compare cut result
        """
        nonlocal res_idx, res_factor, res_efficiency
        complete_size = align(cur_size, factor)
        cur_efficiency = cur_size / complete_size
        # equal case, use front index
        priority1 = cur_efficiency > res_efficiency
        priority2 = cur_efficiency == res_efficiency and \
                    cur_idxs[0] < res_idx[0]
        priority3 = cur_efficiency == res_efficiency and \
                    cur_idxs[0] == res_idx[0] and cur_factor[0] > res_factor[0]
        if priority1 or priority2 or priority3:
            res_idx = cur_idxs[:]
            res_factor = cur_factor[:]
            res_efficiency = cur_efficiency

    # dsf for factor
    def calcu_factor_from_series_axes(shape, idxs):
        """
        calculate factor from series axes
        """
        if idxs in visit:
            return
        visit.append(idxs)

        ### single pattern
        if len(shape) == 1:
            coef = ceil(shape[0], factor)
            compare_cut_result(shape[0], idxs, [coef])
            return

        ### multi pattern
        inner_size = get_shape_size(shape[1:-1])
        # inner_size too large
        if factor < inner_size:
            calcu_factor_from_series_axes(shape[1:-1], idxs[1:-1])
        # inner_size and factor have no common side
        elif int(factor // inner_size) < 1:
            compare_cut_result(inner_size, shape[1:-1], idxs[1:-1])
        # inner_size less than factor and have space for coef
        else:
            pre_dim = shape[0]
            suf_dim = shape[-1]
            pre_coefss = get_limit_coef(pre_dim, factor)
            suf_coefss = get_limit_coef(suf_dim, factor)
            for coef in suf_coefss:
                coef = int(suf_dim / coef)
            for pre_coef in pre_coefss:
                for suf_coef in suf_coefss:
                    cur_size = inner_size * pre_coef * int(suf_dim / suf_coef)
                    if cur_size <= factor:
                        compare_cut_result(cur_size, idxs, [pre_coef, suf_coef])

        # dsf part
        calcu_factor_from_series_axes(shape[1:], idxs[1:])
        calcu_factor_from_series_axes(shape[:-1], idxs[:-1])

    barrier = list(barrier)
    barrier.append(len(shape))
    shape = list(shape)
    shape.append(DEFAULT_INDEX)
    temp_shape = []
    temp_idxs = []
    # pylint: disable=consider-using-enumerate
    for idx in range(len(shape)):
        if idx in barrier:
            calcu_factor_from_series_axes(temp_shape, temp_idxs)
            temp_shape = []
            temp_idxs = []
        else:
            temp_shape.append(shape[idx])
            temp_idxs.append(idx)

    return res_idx, res_factor


def get_block_factor_radical(shape, barrier, factor):
    '''
    make block dim axes, and the result axis are series
    :param shape: input shape
    :param barrier: axis that can't be cut
    :param factor: block dim size, axes outer
    :return: 1 : axes index
             2 : cut coef as fuse all axes in `1`
    '''

    def get_factor_from_series_axes(shape):
        total_size = get_shape_size(shape)
        align_size = align(total_size, factor)
        return align_size // factor, total_size / align_size

    def compare_cut_result(ori_res, new_res):
        return new_res > ori_res

    temp_shape = []
    temp_idx = []
    res_idx = []
    res_efficiency = DEFAULT_INDEX
    res_coef = INIT_SIZE
    # pylint: consider-using-enumerate
    for d_var in range(len(shape)):
        if d_var in barrier:
            temp_coef, temp_efficiency = get_factor_from_series_axes(temp_shape)
            if compare_cut_result(res_efficiency, temp_efficiency) and len(
                    temp_shape):
                res_idx = temp_idx[:]
                res_efficiency = temp_efficiency
                res_coef = temp_coef
            temp_shape = []
            temp_idx = []
        else:
            temp_shape.append(shape[d_var])
            temp_idx.append(d_var)

    return res_idx, res_coef


def get_ub_factor(shape, barrier, rest_size):
    '''
    make ub as full as possiable, axes are series
    :param shape: input shape
    :param barrier: axis that can't be cut
    :param factor: block dim size, axes inner
    :return: 1 : cut index, until to last, except barrier axes
             2 : cut factor for axes in `1`
    '''

    # modify ub factor
    # case like, shape is (10) and the rest size is 6
    # calculat cut by 6, modify to 5
    def modify_d(axis_size, rest_size):
        cur_res = ceil(axis_size, rest_size)
        if axis_size % cur_res == 0:
            return int(axis_size // cur_res)
        return int(rest_size)

    legal_dim = DEFAULT_INDEX
    for idx in reversed(range(len(shape))):
        if idx not in barrier:
            if rest_size < shape[idx]:
                return idx, modify_d(shape[idx], rest_size)
            rest_size = int(rest_size // shape[idx])
            legal_dim = idx
    return legal_dim, shape[legal_dim]


def get_atomic_reduce_info():
    """
    get atomic reduce info
    """
    version_code = cceconf.get_soc_spec("SOC_VERSION")
    if version_code in REDUCE_ATOMIC_SUPPORT.keys():
        return REDUCE_ATOMIC_SUPPORT[version_code]
    return {}


def is_support_atomic_reduce(tensor, info):
    """
    check support atomic reduce or not
    """
    if not info:
        return False
    if tensor.op.tag == info["tag"] and tensor.dtype == info["dtype"]:
        return True
    return False


def pattern_identify(tensor_list):
    """
    pattern identify
    """
    tag_list = []
    former_broadcast = False
    for tensor in tensor_list:
        if BROADCAST_TAG_LABEL in tensor.op.tag:
            if former_broadcast:
                continue
            else:
                former_broadcast = True
        else:
            former_broadcast = False
        tag_list.append(tensor.op.tag)
    for cur_pattern in pattern.data:
        if pattern.data[cur_pattern] == tag_list:
            return cur_pattern
    return pattern.P_NONE

def _map_apend(input_map, key, value):
    if input_map.get(key):
        if isinstance(value, list):
            for tmp_v in value:
                if not tmp_v in input_map[key]:
                    input_map[key].append(tmp_v)
        else:
            if not value in input_map[key]:
                input_map[key].append(value)
    else:
        if isinstance(value, list):
            input_map[key] = value
        else:
            input_map[key] = [value]


def gen_reversed_subgraph_list(out_tensor, tensor_list_map,
                               tensor_list_dst_tensor_map):
    """traverse tensors by Depth-First-Search

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
                tensor_list_map[in_tensor.name] = in_tensor
            _map_apend(tensor_list_dst_tensor_map, in_tensor, cur_tensor)


def get_emit_insn_map(tensor):
    """
    get tensor's emit_insn key
    """
    insn_map = {"elewise_single_cast": "vector_conv",
                "elewise_single_VS_max": "vector_maxs",
                "elewise_single_VS_min": "vector_mins",
                "elewise_single_log": "vector_ln",
                "elewise_single_exp": "vector_exp",
                "elewise_single_rec": "vector_rec",
                "elewise_single_relu": "vector_relu",
                "elewise_single_abs": "vector_abs",
                "elewise_single_not": "vector_not",
                "elewise_single_sqrt": "vector_sqrt",
                "elewise_single_rsqrt": "vector_rsqrt",
                "elewise_binary_mul": "vector_mul",
                "elewise_single_VS_mul": "vector_muls",
                "elewise_binary_div": "vector_div",
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
                "broadcast_for_tensor": "broadcast_for_tensor",
                "elewise_binary_sub": "vector_sub"}
    if tensor.op.tag.find("|") != -1:
        str_list = tensor.op.tag.split("|")
        insn = insn_map.get(str_list[0])
    else:
        insn = insn_map.get(tensor.op.tag)
    return insn

def is_bert_bn_target(tensor_list):
    """
    bert_bn_training_reduce_grad_fp32_1980 identify
    """
    tag_list = []
    for tensor in tensor_list:
        tag_list.append(tensor.op.tag)
    if tag_list in pattern.bn_update_bert_target:
        return True
    return False


def get_nearest_factor(dim, split_size):
    """
    find the exact division factor small than split_size as nearest_factor,
    if distance of nearest_factor and split_size is small, will use the
    nearest_factor as factor, otherwise use the split_size
    """
    nearest_factor = split_size
    while dim % nearest_factor != 0:
        nearest_factor -= 1
    if split_size / nearest_factor < 2:
        split_size = nearest_factor
    return split_size


def dfs_tensor_graph(tensor,  # pylint: disable=R0912, R0915, R0913
                     is_out=True, visited=None, input_tensors=None,
                     mid_tensors=None, tensor_map=None):
    """
    Based on the output tensor, use dfs(Depth First Search)
    algorithm to construct the graph.
    """
    # First trial, initialize results
    if visited is None:
        visited = []
        input_tensors = []
        mid_tensors = []
        tensor_map = collections.OrderedDict()
    # Record current tensor
    if tensor not in visited:
        visited.append(tensor)
    else:
        return visited, input_tensors, mid_tensors, tensor_map
    # Input tensor is a tensor without any input_tensor
    if not tuple(tensor.op.input_tensors) and \
            isinstance(tensor.op, tvm.tensor.PlaceholderOp):
        input_tensors.insert(0, tensor)
    elif not is_out:
        mid_tensors.append(tensor)
    # Iterate through all of current_tensor's input tensors
    for input_tensor in list(tensor.op.input_tensors):
        # Add input -> output mapping to tensor_map
        if input_tensor not in tensor_map:
            tensor_map[input_tensor] = [tensor]
        else:
            tensor_map[input_tensor].insert(0, tensor)
        dfs_tensor_graph(input_tensor, False, visited,
                         input_tensors, mid_tensors, tensor_map)
    return visited, input_tensors, mid_tensors, tensor_map


def gen_dfs_tensor_map(outs):
    """
    Based on the output tensor, use dfs(Depth First Search)
    algorithm to construct the graph.
    """
    visited, input_tensors, mid_tensors, tensor_map = dfs_tensor_graph(outs[0])
    for out in outs[1:]:
        dfs_results = dfs_tensor_graph(out, True,
                                       visited, input_tensors,
                                       mid_tensors, tensor_map)
        visited, input_tensors, mid_tensors, tensor_map = dfs_results
    return visited, input_tensors, mid_tensors, tensor_map


def _check_pattern_matched(dfs_tensor_list, expected_dfs_tag_list):
    """
    check pattern matched
    """
    exct_idx = 0
    for i, _ in enumerate(dfs_tensor_list):
        if exct_idx == len(expected_dfs_tag_list):
            return True
        if dfs_tensor_list[i].op.tag == expected_dfs_tag_list[exct_idx] or \
                (dfs_tensor_list[i].op.tag == "" and
                 expected_dfs_tag_list[exct_idx] == "placeholder"):
            exct_idx = exct_idx + 1
        elif dfs_tensor_list[i].op.tag.find("elewise") == -1 and \
                dfs_tensor_list[i].op.tag != "" and \
                dfs_tensor_list[i].op.tag != "broadcast_for_tensor":
            return False
    if exct_idx == len(expected_dfs_tag_list):
        return True
    return False


def get_reduce_axes(reduce_tensor):
    """Get reduce axes var"""
    reduce_tensor_body = reduce_tensor.op.body
    reduce_tensor_axes = list(reduce_tensor_body[0].axis)
    for idx, axis in enumerate(reduce_tensor_axes):
        reduce_tensor_axes[idx] = axis.var
    return reduce_tensor_axes


def get_all_axes(reduce_tensor):
    """Get all axes"""
    reduce_tensor_body = reduce_tensor.op.body
    return list(reduce_tensor_body[0].source[0].args)
