#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright 2019 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

util_apply_op_schedule
"""

import collections
from functools import reduce as function_reduce
import operator
import math

from te import tvm
from te.platform import cce_params as param
from te.platform.cce_build import build_config
from te.platform.cce_build import build_config_update
from te import platform as cceconf
from te.tvm import build_module
from topi.cce import util

# shape limit 2^31
SHAPE_SIZE_LIMIT = 2147483648
THREAD_AXIS_IDX = 0
OUTER_AXIS_IDX = 1
INNER_AXIS_IDX = 2


def _custom_build_config():
    """
    return build_config with double_buffer_non_reuse

    Parameters
    ----------

    Returns
    ----------
    build_config with double_buffer_non_reuse
    """
    config = build_config_update(build_config, "double_buffer_non_reuse", True)
    return config


def _get_custom_max_ub_count(compute_width):
    """
    get the max split num based on dtype

    Parameters
    ----------
    compute_width: the width of compute

    Returns
    ----------
    max_ub_count: the unit is float16
    """

    # converted to the number of float16
    total_size = cceconf.CceProductParams().getParams("Unified_Buffer") // 2
    # enable double buffer
    total_size = total_size // 2

    max_bound = compute_width * 128
    max_ub_count = int(total_size // max_bound * 128)

    return max_ub_count


def _tile_apply_ops(shape, input_dtype, compute_width):
    """
    according to shape and input_dtype, and the max useful ub buffer to tile the shape

    Parameters
    ----------
    shape: the input shape
    input_dtype: the input dtype
    compute_width: the width of compute

    Returns
    ----------
    the tile factor and split axis
    """

    max_ub_count = _get_custom_max_ub_count(compute_width)
    if input_dtype == 'float32':
        max_ub_count = max_ub_count // 2

    factor = shape
    axis = 0

    if shape > max_ub_count:
        factor = max_ub_count

    return factor, axis


def _cache_and_inline(sch,
                      cache_list,
                      inline_compute_list,
                      ir_tensor_list=None):
    """
    cache_read/cache_write/compute_inline

    Parameters
    ----------
    sch: schedule
    cache_list: cache buffer in global
    inline_compute_list: the tensor list which need compute_inline
    ir_tensor_list: tensor list want to print

    Returns
    ----------
    buffer_read_list: read_cache buffer in local
    buffer_write_list: write_cache buffer in global
    """

    # cache_read
    # store the ub read buffer list
    buffer_read_list = []
    for read_tensor in cache_list['read']:
        buffer_read_list.append(
            sch.cache_read(read_tensor[0], param.scope_ubuf, read_tensor[1]))

    # cache_write
    # store the ub write buffer list
    buffer_write_list = []
    for write_tensor in cache_list['write']:
        buffer_write_list.append(
            sch.cache_write(write_tensor, param.scope_ubuf))

    if ir_tensor_list is not None:
        print('----------- after cache_read/cache_write IR -----------')
        with _custom_build_config():
            print(tvm.lower(sch, ir_tensor_list, simple_mode=True))

    # compute_inline
    for inline_tensor in inline_compute_list:
        sch[inline_tensor].compute_inline()

    if ir_tensor_list is not None:
        print('----------- after compute_inline IR -----------')
        with _custom_build_config():
            print(tvm.lower(sch, ir_tensor_list, simple_mode=True))

    return {'read': buffer_read_list, 'write': buffer_write_list}


def _get_emit_insn_map(emit_insn_intrin):
    """
    use vector_xx to replace elewise_xxx in case warning of no intrisic rules

    Parameters
    ----------
    emit_insn_intrin: insn intrin

    Returns
    ----------
    emit_insn_pragma: insn pragma
    """

    # map table referenced to tensor_engine/python/te/lang/cce/te_schedule/elewise_schedule.py
    emit_insn_map = {
        "elewise_single_cast": "vector_conv",
        "elewise_single_VS_max": "vector_maxs",
        "elewise_single_VS_min": "vector_mins",
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
        "elewise_binary_cmpsel": "vector_cmpsel",
        # new
        "emit_insn_elewise_binary_cmp": "elewise_binary_cmp",
        "emit_insn_elewise_multiple_sel": "elewise_multiple_sel",
    }
    emit_insn_pragma = emit_insn_map.get(emit_insn_intrin)
    if emit_insn_pragma is None:
        emit_insn_pragma = emit_insn_intrin
    return emit_insn_pragma


def _split_tensor(tensor):
    """
    get the information from tensor

    Parameters
    ----------
    tensor: tensor

    Returns
    ----------
    the information about tensor
    """

    tmp_op = {}
    tensor_op = tensor.op
    tmp_op["op"] = tensor_op.tag
    tmp_op["dst_buffer"] = tensor
    tmp_op["src_buffer"] = list(tensor_op.input_tensors)

    if tmp_op["op"].find("|") != -1:
        str_list = tensor_op.tag.split("|")
        tmp_op["op"] = str_list[0]

    return tmp_op


def _get_info_from_gragh(outs):
    """
    record relate context imformations of operations

    Parameters
    ----------
    outs: the muti-outputs

    Returns
    ----------
    write_cache, read_cache, inline_compute, op_dict contains the tag
    """

    write_cache = []
    read_cache = []
    inline_compute = []
    op_dict = {}

    operation_list = []

    for tensor in outs:
        operation_list.append(tensor)

    def get_read_and_inline(cur_buf):
        if isinstance(cur_buf.op, tvm.tensor.PlaceholderOp):
            tmp = (cur_buf, tmp_op['dst_buffer'])
            if tmp not in read_cache:
                read_cache.append(tmp)
        else:
            if cur_buf not in tmp_operation_list:
                tmp_operation_list.append(i)
            if cur_buf not in inline_compute:
                inline_compute.append(i)

    while operation_list:
        for i in operation_list:
            # the output write_cache need write_cache in a group to keep muti output,
            # so the outs cache_write need deal alone
            if i not in write_cache and i not in outs:
                write_cache.append(i)
        tmp_operation_list = []
        for j in operation_list:
            tmp_op = _split_tensor(j)
            for i in tmp_op["src_buffer"]:
                get_read_and_inline(i)
            if tmp_op["dst_buffer"] not in op_dict.keys(
            ) and tmp_op["dst_buffer"] not in outs:
                op_dict[tmp_op["dst_buffer"]] = tmp_op
        operation_list = list(set(tmp_operation_list))
        operation_list.sort(key=tmp_operation_list.index)

    return {'write': write_cache, 'read': read_cache}, inline_compute, op_dict


def _split_tasks_by_strategy_two(num_core, num_task):
    """
    split tasks by strategy two

    Parameters
    ----------
    num_core: the maximum number of cores of the device
    num_task: the number of tasks

    Returns
    ----------
    the size of outer and inner axes
    """

    num = num_task
    if num <= num_core:
        return num, 1
    inner = 2
    while num > num_core:
        outer = math.ceil(num / inner)
        if outer <= num_core:
            return outer, inner
        inner += 1


def _split_core_strategy(num_core, num_task):
    """
    choose a better strategy for splitting the core

    Parameters
    ----------
    num_core: the maximum number of cores of the device
    num_task: the number of tasks

    Returns
    ----------
    True: do not split axis for splitting core
    False: split axis for splitting core
    """

    # strategy 1
    volume1 = num_task
    max_iter1 = math.ceil(num_task / num_core)
    time1 = max_iter1 / volume1

    # strategy 2
    outer, inner = _split_tasks_by_strategy_two(num_core, num_task)
    time2 = inner / (outer * inner)

    return time1 < time2


def _execute_split_core(sch, out, shape, split_axis, rfactor):
    """
    execute a split operation

    Parameters
    ----------
    sch: schedule
    out: out tensor
    shape: shape of tensor
    split_axis: splitting axis
    rfactor: splitting factor

    Returns
    ----------
    thread_axis: outermost axis
    outer_axis: outer axis after splitting
    inner_axis: inner axis after splitting
    """
    outer_axis, inner_axis = sch[out].split(sch[out].op.axis[split_axis],
                                            rfactor)

    # get split axis
    out_extent = math.ceil(shape / rfactor)

    thread_axis = outer_axis
    if out_extent > 1:
        block_index = tvm.thread_axis('blockIdx.x')
        max_core_num = cceconf.CceProductParams().getParams("Device_core_num")

        if _split_core_strategy(max_core_num,
                                out_extent) or out_extent <= max_core_num:
            sch[out].bind(thread_axis, block_index)
        else:
            outer, _ = _split_tasks_by_strategy_two(max_core_num, out_extent)
            thread_axis, outer_axis = sch[out].split(outer_axis, nparts=outer)
            sch[out].bind(thread_axis, block_index)

    return thread_axis, outer_axis, inner_axis


def _execute_compute_at(sch, out, buffer_list, axes, scalar_name):
    """
    execute compute at

    Parameters
    ----------
    sch: schedule
    out: out tensor
    buffer_list: read/write buffer list
    axes: all axis
    scalar_name: name of scalar

    Returns
    ----------
    None
    """
    for buffer_read in buffer_list['read']:
        end_idx = buffer_read.op.name.find('.')
        origin_name = buffer_read.op.name[:
                                          end_idx] if end_idx != -1 else buffer_read.op.name
        if origin_name not in scalar_name:
            sch[buffer_read].compute_at(sch[out], axes[OUTER_AXIS_IDX])
        else:
            sch[buffer_read].compute_at(sch[out], axes[THREAD_AXIS_IDX])

    list_op = ["lhs.local.UB", "lhs_beta1.local.UB", "rhs.local.UB", "rhs_beta2.local.UB",
               "lhs1.local.UB", "lhs_power.local.UB", "rhs2.local.UB", "rhs_power.local.UB",
               "alpha_lr.local.UB", "vlog_t.local.UB", "num_one.local.UB", "alpha.local.UB"]
    for buffer_write in buffer_list['write']:
        if buffer_write.op.name in list_op:
            sch[buffer_write].compute_at(sch[out], axes[THREAD_AXIS_IDX])
        else:
            sch[buffer_write].compute_at(sch[out], axes[OUTER_AXIS_IDX])


def _execute_insn_emit(sch, buffer_list, cache_list, op_dict):
    """
    execute insn emit

    Parameters
    ----------
    sch: schedule
    buffer_list: read/write buffer list
    cache_list: read/write cache list
    op_dict: dictionary of op information

    Returns
    ----------
    None
    """
    # intrin
    for i, buffer_cache in enumerate(buffer_list['write']):
        write_cache = cache_list['write'][i]
        emit_insn_intrin = op_dict[write_cache]["op"]
        emit_insn_pragma = _get_emit_insn_map(emit_insn_intrin)
        sch[buffer_cache].emit_insn(sch[buffer_cache].op.axis[0],
                                    emit_insn_pragma)

    # pragam
    for read_buffer in buffer_list['read']:
        sch[read_buffer].emit_insn(sch[read_buffer].op.axis[0],
                                   cceconf.dma_copy)


def _enable_double_buffer(sch, buffer_list):
    """
    enable double buffer

    Parameters
    ----------
    sch: schedule
    buffer_list: read/write buffer list

    Returns
    ----------
    None
    """
    for read_buffer in buffer_list['read']:
        sch[read_buffer].double_buffer()
    for write_buffer in buffer_list['write']:
        sch[write_buffer].double_buffer()


def _execute_rewrite_inputs(sch, buffer_list, rewrite_back_map, axes):
    """
    execute rewrite inputs

    Parameters
    ----------
    sch: schedule
    buffer_list: read/write buffer list
    rewrite_back_map: correspondence between placeholder and output tensor
    axes: all axis

    Returns
    ----------
    None
    """
    index = 0
    for rewrite_input in rewrite_back_map.keys():
        for rewrite_input_read_buffer in buffer_list['read']:
            if rewrite_input == rewrite_input_read_buffer.op.input_tensors[0]:
                replace_buffer = rewrite_back_map[rewrite_input]
                sch[rewrite_input_read_buffer].pragma(
                    sch[rewrite_input_read_buffer].op.axis[0], "reuse_input",
                    index)
                sch[replace_buffer].pragma(axes[INNER_AXIS_IDX],
                                           "replace_output", index)
                index += 1


def common_apply_op_schedule(outs,
                             rewrite_back_map,
                             compute_width,
                             scalar_name,
                             ir_tensor_list=None):
    """
    Parameters
    ----------
    outs: the muti-outs in the operator
    rewrite_back_map: the inputs wanted to be rewrited by outputs
    scalar_name: name of scalar tensor
    ir_tensor_list: tensor list want to print

    Returns
    ----------
    the schedule
    """
    out = outs[0]
    shape = int(out.shape[0])

    # parsing the graph to get the information
    cache_list, inline_compute_list, op_dict = _get_info_from_gragh(outs)

    # create origin schedule
    sch = tvm.create_schedule([out_tensor.op for out_tensor in outs])
    if ir_tensor_list is not None:
        print('----------- origin IR -----------')
        with _custom_build_config():
            print(tvm.lower(sch, ir_tensor_list, simple_mode=True))

    # cache and inline
    buffer_list = _cache_and_inline(sch, cache_list, inline_compute_list,
                                    ir_tensor_list)

    # tiling
    rfactor, split_axis = _tile_apply_ops(shape, out.dtype, compute_width)

    # split axes
    axes = _execute_split_core(sch, out, shape, split_axis, rfactor)

    _execute_compute_at(sch, out, buffer_list, axes, scalar_name)
    if ir_tensor_list is not None:
        print('----------- after compute_at IR -----------')
        with _custom_build_config():
            print(tvm.lower(sch, ir_tensor_list, simple_mode=True))

    _execute_insn_emit(sch, buffer_list, cache_list, op_dict)
    # for muti-outs, the outs is a group, the old dma coy can't deal it,
    # so we add a group_gm_to_ub to deal the situation
    sch[out].emit_insn(axes[INNER_AXIS_IDX], "group_gm_to_ub")
    if ir_tensor_list is not None:
        print('----------- after emit_insn IR -----------')
        with _custom_build_config():
            print(tvm.lower(sch, ir_tensor_list, simple_mode=True))

    _enable_double_buffer(sch, buffer_list)

    _execute_rewrite_inputs(sch, buffer_list, rewrite_back_map, axes)
    if ir_tensor_list is not None:
        print('----------- after reuse input IR -----------')
        with _custom_build_config():
            print(tvm.lower(sch, ir_tensor_list, simple_mode=True))

    return sch


def _check_shape_rule(input_tensors, dtype_check_list):
    """
    Parameters
    ----------
    input_tensors: input tensor dict
    dtype_check_list: dtype supported

    Returns
    ----------
    normalized_dtype_list: list of normalized data type
    broadcast_shape_list: list of broadcasted shape
    """

    normalized_dtype_list = [None] * len(input_tensors)
    broadcast_shape_list = [None] * len(input_tensors)
    for i, tensor_dict in enumerate(input_tensors):
        shape = tensor_dict.get('shape')
        dtype = tensor_dict.get('dtype')
        util.check_shape_rule(shape)
        util.check_shape_size(shape, SHAPE_SIZE_LIMIT)
        util.check_dtype_rule(dtype, dtype_check_list)
        normalized_dtype_list[i] = dtype.lower()
        broadcast_shape_list[i], _, _ = util.produce_shapes(
            shape, input_tensors[0].get('shape'))
    return normalized_dtype_list, broadcast_shape_list


def _check_scalar_shape(shape_list, name_idx_map, scalar_name):
    """
    check the shape of scalar

    Parameters
    ----------
    shape_list: list of shapes containing all tensors
    name_idx_map: correspondence between tensor name and indexing
    scalar_name: name of scalar

    Returns
    ----------
    True or False
    """
    need_to_check = [shape_list[name_idx_map[name]] for name in scalar_name]
    return all(util.is_scalar(item) for item in need_to_check)


# pylint: disable=locally-disabled, too-few-public-methods
class ApplyOpConfig:
    """
    configuration of apply op
    """

    TensorArgs = collections.namedtuple('TensorArgs',
                                        'input_vars compute output width')
    TensorName = collections.namedtuple('TensorName', 'all scalar reuse')
    TensorOptions = collections.namedtuple('TensorOptions',
                                           'attrs build dtype')
    TensorOptions.__new__.__defaults__ = (None, _custom_build_config(),
                                          ('float16', 'float32'))

    def __init__(self,
                 args,
                 name,
                 options=TensorOptions(None, _custom_build_config(),
                                       ('float16', 'float32'))):
        self.args = args
        self.name = name
        self.options = options


def check_shape_and_dtype(config, name_idx_map, same_flag):
    """
    check shape and dtype

    Parameters
    ----------
    config: configuration of apply ops
    name_idx_map: correspondence between tensor name and indexing

    Returns
    ----------
    shape_list: list of shapes that support broadcast operations
    dtype_list: list of data types after lower
    """
    def reduce_2_tuple(shape):
        return (function_reduce(operator.mul, shape), )

    shape_list = [i.get('shape') for i in config.args.input_vars]

    # check shape
    ## check shape of scalar
    if not _check_scalar_shape(shape_list, name_idx_map, config.name.scalar):
        raise RuntimeError("input must be scalar.")
    ## check shape rule
    dtype_list, shape_list = _check_shape_rule(config.args.input_vars,
                                               config.options.dtype)
    ## check shape of non scalar
    non_scalar_shape_list = [(name, shape_list[name_idx_map[name]])
                             for name in config.name.all
                             if name not in config.name.scalar]
    if any(elem[1] != non_scalar_shape_list[0][1]
           for elem in non_scalar_shape_list):
        raise RuntimeError("All input tensor shape must be the same.")
    ## reduce shape to one dimension
    shape_list = [reduce_2_tuple(shape) for shape in shape_list]

    # check dtype
    if same_flag:
        if any(elem != dtype_list[0] for elem in dtype_list):
            raise RuntimeError("All input data types must be the same.")

    return shape_list, dtype_list


def common_apply_op_process(config, kernel_name, same_flag=True):
    """
    Parameters
    ----------
    config: configuration of apply ops
    kernel_name: kernel name
    same_flag: check dtype same or not

    Returns
    ----------
    None
    """

    # check kernel name
    util.check_kernel_name(kernel_name)

    # correspondence between tensor name and indexing
    name_idx_map = {name: idx for idx, name in enumerate(config.name.all)}
    for i in config.args.input_vars:
        if len(i.get('shape')) == 0:
            i["shape"] = (1,)

    # add for ge/fe based on tensorflow
    for scalar in config.name.scalar:
        if config.args.input_vars[name_idx_map[scalar]]['shape'] == ():
            config.args.input_vars[name_idx_map[scalar]]['shape'] = (1,)

    # check shape and dtype
    shape_list, dtype_list = check_shape_and_dtype(config, name_idx_map, same_flag)

    # get placeholder of all tensors
    input_tensors = [None] * len(name_idx_map)
    for name, index in name_idx_map.items():
        input_tensors[index] = tvm.placeholder(shape_list[index],
                                               dtype_list[index],
                                               name=name)

    # call the compute function
    attrs = config.options.attrs
    if (not isinstance(attrs, collections.Iterable)) or isinstance(attrs, str):
        attrs = [attrs]

    if isinstance(config.args.output, list):
        outs = config.args.compute(*(input_tensors + config.args.output +
                                     list(attrs)))
    else:
        outs = config.args.compute(*(input_tensors + [config.args.output] +
                                     list(attrs)))

    # create a write back map
    # key: placeholder value: out tensor
    rewrite_back_map = collections.OrderedDict()
    for i, name in enumerate(config.name.reuse):
        rewrite_back_map[input_tensors[name_idx_map[name]]] = outs[i]

    # execute the schedule operation
    sch = common_apply_op_schedule(outs, rewrite_back_map, config.args.width,
                                   config.name.scalar)

    # build
    with config.options.build:
        tvm.build(sch, input_tensors + list(outs), "cce", name=kernel_name)
