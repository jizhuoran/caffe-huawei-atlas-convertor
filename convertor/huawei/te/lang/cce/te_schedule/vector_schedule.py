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

vector schedule
"""
import abc
import math
import copy
from te import platform as cceconf
from te import tvm
from te.platform import cce_emitinsn_params
from .cce_schedule_declarations import OpSpecTypes
from .util import get_align_factor

# the bit of dtype/16 map
DTYPE_WIDTH_MAP = {"float16": 1,
                   "float32": 2,
                   "int32": 2,
                   "int16": 1,
                   "uint16": 1,
                   "int8": 0.5,
                   "uint8": 0.5,
                   "bool": 0.5}

# For non-divisible multi-core splitting, block dim is at most a multiple of core num
BLOCK_DIM_MULTIPLE = 4  # Bytes

ENABLE_MULTI_CORE_THRESHOLD = 1024  # Bytes
# For non-divisible multi-core splitting, select factor lower bound threshold
BLOCK_DIM_LOWER_BOUND_THRESHOLD = 0.85


# pylint: disable=no-self-use, too-many-instance-attributes, too-few-public-methods, redefined-builtin, useless-object-inheritance
class VectorSchedule(object):
    """
    Base class of cce vector schedule

    Parameters
    ----------
    None

    Returns
    -------
    VectorSchedule_instance : instance of VectorSchedule
    """

    def __init__(self, need_multi_core=True):

        self._schedule = None
        self._schedule_valid = True
        self._need_db = None
        self._need_multi_core = need_multi_core
        self._spec_node_list = []
        self._multi_core_bind_tensor = None
        self._multi_core_fused_axis = None
        self._cancel_ub_tiling = False
        self._estimate_factor = None
        self._out_tensors = []
        self._buffer_tile_out = []

        # for super kernel, when only bind the batch axis, the axis is True
        self._batch_bind_only = False

        # cache read para map, e.g. schedule.cache_read(read_tensor, scope,
        # readers), {reader_tensor: readers}
        self._cache_read_tensors_and_readers_map = {}

        # cache read result map, e.g. buffer = schedule.cache_read(read_tensor,
        # scope, readers), {reader_tensor: buffer}
        self._cache_read_tensors_and_buffer_map = {}

        # cache write para list, e.g. schedule.cache_write(write_tensor, scope),
        # Example: write_tensor,...
        self._cache_write_tensors = []

        # cache write result map, e.g. buffer = schedule.cache_write(
        # write_tensor, scope), {write_tensor: buffer}
        self._cache_write_tensors_and_buffer_map = {}

        # compute inline para list, e.g. schedule[compute_inline_tensor].
        # Example: compute_inline(), [compute_inline_tensor,...]
        self._compute_inline_tensors = []

        # double buffer para list, e.g. schedule[double_buffer_tensor].
        # Example: double_buffer(), [double_buffer_tensor,...]
        self._double_buffer_tensors = []

        # record double buffer map[read] = write
        self._double_buffer_map = {}

        self._tiling_tensor = None

        self._insn_map = {}

        self._reg_insn_map = {}

        # Example:"block_tiling" {"axis" block_split_axis, "factor" block_split_inner_size}
        # Example:"ub_tiling" {"axis"  ub_split_axis, "factor"  ub_split_inner}
        self._tiling_para = {"block_tiling": {"axis": 0, "factor": 1},
                             "ub_tiling": {"axis": 0, "factor": 1}}

        # _after_tiling_para:
        # Example: "block_tiling":
        # Example: "axis":axis_index,
        # Example: "parent_itervar": parent_axis_itervar,
        # Example: "outer_itervar": outer_axis_itervar,
        # Example: "inner_itervar": inner_axis_itervar
        # Example: "ub_tiling":
        # Example: "axis":axis_index,
        # Example: "parent_itervar": parent_axis_itervar,
        # Example: "outer_itervar": outer_axis_itervar,
        # Example: "inner_itervar": inner_axis_itervar
        self._tiling_result = {}
        self.block_tiling_use_nparts_mode = False

        # compute at para map, e.g. schedule[stage].compute_at(parent_stage,
        # scope_iter_var),
        # Example: {
        # stage:{"parent": parent_stage, "scope:": scope_iter_var},
        # Example: ...}
        self._compute_at_map = {}

        # emit instruction para map, e.g. schedule[stage].emit_insn(
        # scope_iter_var, instruction),
        # Example: {
        # stage:{"scope": scope_iter_var, "instruction:": instruction},
        # Example: ...}
        self._emit_insn_map = {}

        # broadcast_axis_list e.g. [4, 3, 2]
        # broadcasting axis for each producers in place
        self.max_last_broadcast_axis_offset = {}

        self._mem_unique_enable = False

        self._scope = "local.UB"

        self._op_type = None
        self._op_subpattern = None

    def set_op_type(self, type, subpattern=None):
        """
        set operation type
        """
        self._op_type = type
        if subpattern is not None:
            self._op_subpattern = subpattern


    def do_schedule(self, out_tensors, sch_list, spec_node_list):
        """
        auto_schedule for cce AI-CORE

        Parameters
        ----------
        outTensors : the out tvm.tensor

        sch_list : schedule, the computation schedule for the op

        spec_node_list : special node list

        Returns
        -------
        Bool, now is true

        """

        self._spec_node_list = spec_node_list
        self._out_tensors = copy.copy(out_tensors)

        if sch_list[0] is not None:
            self._schedule = sch_list[0]

        is_success = self._construct_compute_graph(out_tensors, spec_node_list)
        if not is_success:
            return False

        self._calculate_cache_read()
        self._do_cache_read()

        self._calculate_cache_write()
        self._do_cache_write()

        self._calculate_tiling()
        self._do_tiling()

        self._do_buffer_tile()

        self._calculate_compute_inline()
        self._do_compute_inline()

        self._calculate_multi_core()
        self._do_multi_core()

        self._calculate_compute_at()
        self._do_compute_at()

        self._calculate_emit_insn()
        self._do_emit_insn()

        self._calculate_double_buffer()
        self._do_double_buffer()

        sch_list[0] = self._schedule

        for i in range(0, len(out_tensors)):
            out_tensors.pop()
        for i in self._out_tensors:
            out_tensors.append(i)

        return self._schedule_valid

    def _do_cache_read(self):
        """
        cache read operations

        Parameters
        ----------
        None

        Returns
        -------
        list : read buffers
        """

        for i in self._cache_read_tensors_and_readers_map:
            readers = self._cache_read_tensors_and_readers_map[i]
            read_buffer = self._schedule.cache_read(i, self._scope, readers)

            self._cache_read_tensors_and_buffer_map[i] = read_buffer

            self._double_buffer_tensors.append(read_buffer)

            if self._op_type == OpSpecTypes.RELU_GRAD_V2:
                self._schedule[read_buffer].preload()

            if self._mem_unique_enable:
                self._schedule[read_buffer].mem_unique()

    def _do_cache_write(self):
        """
        cache write operations

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for i in self._cache_write_tensors:
            write_buffer = self._schedule.cache_write(i, self._scope)
            self._cache_write_tensors_and_buffer_map[i] = write_buffer

    def calculate_need_cancel_ub_tiling(self):
        """
        to prevent multi-core trampling
        """
        shape = list(map(int, self._tiling_tensor.shape))
        block_tiling_para = self._tiling_para["block_tiling"]
        ub_tiling_para = self._tiling_para["ub_tiling"]
        ub_axis = ub_tiling_para["axis"]
        ub_factor = ub_tiling_para["factor"]
        block_axis = block_tiling_para["axis"]
        outer = block_tiling_para["factor"]
        if self.block_tiling_use_nparts_mode:
            inner = math.ceil(shape[block_axis]/outer)
        else:
            inner = outer
        tail = inner % ub_factor
        align_num, _ = get_align_factor(self._tiling_tensor.dtype)
        estimate_factor = math.ceil(inner / ub_factor)
        if tail < align_num and ub_axis == block_axis and tail != 0:
            return True, estimate_factor
        return False, None

    def _do_tiling(self):
        res = self._tiling_tensor
        block_tiling_para = self._tiling_para["block_tiling"]
        block_split_axis = block_tiling_para["axis"]
        block_split_factor_size = block_tiling_para["factor"]

        ub_tiling_para = self._tiling_para["ub_tiling"]
        ub_split_axis = ub_tiling_para["axis"]
        ub_split_inner = ub_tiling_para["factor"]

        if self.block_tiling_use_nparts_mode:
            res_block_outer, res_block_inner = self._schedule[res].split(
                res.op.axis[block_split_axis], nparts=block_split_factor_size)
        else:
            res_block_outer, res_block_inner = self._schedule[res].split(
                res.op.axis[block_split_axis], factor=block_split_factor_size)

        block_tiling_result = {"axis": block_split_axis,
                               "parent_itervar": res.op.axis[block_split_axis],
                               "outer_itervar": res_block_outer,
                               "inner_itervar": res_block_inner}
        self._cancel_ub_tiling, self._estimate_factor = \
            self.calculate_need_cancel_ub_tiling()

        if self._cancel_ub_tiling:
            res_ub_outer, res_ub_inner = self._schedule[res].split(
                res_block_inner, nparts=self._estimate_factor)
            ub_tiling_result = {"axis": ub_split_axis,
                                "parent_itervar": res_block_inner,
                                "outer_itervar": res_ub_outer,
                                "inner_itervar": res_ub_inner}
        elif block_split_axis == ub_split_axis:
            res_ub_outer, res_ub_inner = self._schedule[res].split(
                res_block_inner, factor=ub_split_inner)
            ub_tiling_result = {"axis": ub_split_axis,
                                "parent_itervar": res_block_inner,
                                "outer_itervar": res_ub_outer,
                                "inner_itervar": res_ub_inner}

        else:
            res_ub_outer, res_ub_inner = self._schedule[res].split(
                res.op.axis[ub_split_axis], factor=ub_split_inner)
            ub_tiling_result = {"axis": ub_split_axis,
                                "parent_itervar": res.op.axis[ub_split_axis],
                                "outer_itervar": res_ub_outer,
                                "inner_itervar": res_ub_inner}

        self._tiling_result = {"block_tiling": block_tiling_result,
                               "ub_tiling": ub_tiling_result}

    def _do_buffer_tile(self):
        return

    def _do_compute_inline(self):
        """
        compute inline operations
        """
        for i in self._compute_inline_tensors:
            self._schedule[i].compute_inline()

    def _do_multi_core(self):
        if self._need_multi_core:
            res = self._multi_core_bind_tensor
            block = tvm.thread_axis("blockIdx.x")
            self._schedule[res].bind(self._multi_core_fused_axis, block)

        if self._batch_bind_only:
            res = self._last_output_tensor # pylint: disable=no-member
            ub_tiling_result = self._tiling_result["ub_tiling"]
            res_ub_outer = ub_tiling_result["outer_itervar"]
            self._schedule[res].pragma(res_ub_outer,
                                       "json_info_batchBindOnly", 1)


    def _do_compute_at(self):
        for stage in self._compute_at_map:
            parent_stage = self._compute_at_map[stage]["parent"]
            scope_iter_var = self._compute_at_map[stage]["scope"]
            self._schedule[stage].compute_at(parent_stage, scope_iter_var)

    def _do_double_buffer(self):
        """
        double buffer operations
        read_buffer : the all read_cache for input in ub, type is list
        """
        temp_write_buffer = []

        if self._need_db:
            for i in self._double_buffer_tensors:
                self._schedule[i].double_buffer()

                # just for ternary instruction
                if i in self._double_buffer_map:
                    buffers = list(set(self._double_buffer_map[i]))
                    for buffer in buffers:
                        temp_write_buffer.append(buffer)
                        self._schedule[buffer].double_buffer()
            if temp_write_buffer:
                self._recursive_double_buffer(temp_write_buffer)

    def _recursive_double_buffer(self, write_buffer):
        """
        open cache write double buffer for ternary instruction by recursive
        """
        if not write_buffer:
            return

        temp_write_buffer = []
        for i in write_buffer:
            if i in self._double_buffer_map:
                buffers = list(set(self._double_buffer_map[i]))
                for buffer in buffers:
                    temp_write_buffer.append(buffer)
                    self._schedule[buffer].double_buffer()
        self._recursive_double_buffer(temp_write_buffer)

    def _do_emit_insn(self):
        for stage in self._emit_insn_map:
            scope_iter_var = self._emit_insn_map[stage]["scope"]
            instruction = self._emit_insn_map[stage]["instruction"]
            self._schedule[stage].emit_insn(scope_iter_var, instruction)
        if self.max_last_broadcast_axis_offset:
            cce_emitinsn_params.cceEmitParamsIns.del_param(
                "broadcast_axis_offset")
            cce_emitinsn_params.cceEmitParamsIns.insert_param(
                "broadcast_axis_offset",
                self.max_last_broadcast_axis_offset)

    @abc.abstractmethod
    def _construct_compute_graph(self, out_tensors, spec_node_list):
        return

    @abc.abstractmethod
    def _calculate_cache_read(self):
        return

    @abc.abstractmethod
    def _calculate_cache_write(self):
        return

    @abc.abstractmethod
    def _calculate_tiling(self):
        return

    @abc.abstractmethod
    def _calculate_compute_inline(self):
        return

    @abc.abstractmethod
    def _calculate_multi_core(self):
        return

    @abc.abstractmethod
    def _calculate_compute_at(self):
        return

    @abc.abstractmethod
    def _calculate_double_buffer(self):
        return

    @abc.abstractmethod
    def _calculate_emit_insn(self):
        return

    def _find_split_axis(self, shape, begin_axis, end_axis, bound_size):
        axis_num = len(shape)
        if begin_axis >= axis_num or begin_axis < 0 \
                or end_axis >= axis_num or end_axis < 0:
            return 0, 1
        if begin_axis < end_axis:
            step = 1
        else:
            step = -1
        split_axis = end_axis
        temp_size = 1
        need_split = False
        for i in range(begin_axis, end_axis + step, step):
            temp_size = temp_size * shape[i]
            if temp_size >= bound_size:
                split_axis = i
                temp_size = temp_size / shape[i]
                need_split = True
                break

        split_size = 1
        # split the split axis
        if need_split:
            for i in range(1, shape[split_axis] + 1, 1):
                if (temp_size * i) == bound_size:
                    split_size = i
                    break
                if (temp_size * i) > bound_size:
                    split_size = i - 1
                    break

        return split_axis, split_size

    def _get_block_num(self):
        return cceconf.get_soc_spec("CORE_NUM")

    def _get_block_tiling(self, shape, dtype,
                          multi_core_threshold=ENABLE_MULTI_CORE_THRESHOLD):
        data_size = DTYPE_WIDTH_MAP[dtype] * 2

        for i in range(0, len(shape), 1):
            data_size = data_size * shape[i]

        core_num = self._get_block_num()
        block_dim = core_num
        if data_size < core_num * multi_core_threshold:
            block_dim = (data_size + multi_core_threshold - 1) // \
                        multi_core_threshold

        if block_dim < 1:
            block_dim = 1

        block_split_axis, block_split_outer_size = self._find_split_axis(
            shape, 0, len(shape) - 1, block_dim)
        block_split_inner_size = shape[block_split_axis] // \
                                 block_split_outer_size

        block_split_axis, block_split_modified_inner_size, \
        block_split_outer_size = \
            self._modify_block_tiling(shape, data_size, block_split_axis,
                                      block_split_inner_size,
                                      multi_core_threshold,
                                      block_split_outer_size)

        return block_split_axis, block_split_modified_inner_size, \
               block_split_outer_size

    # The backend does not support non-divisible split+fuse,
    # so block tiling needs to be adjusted to divide the split for
    # non-divisible split
    # pylint: disable=too-many-arguments
    def _modify_block_tiling(self, shape, data_size, block_split_axis,
                             block_split_inner_size, multi_core_threshold,
                             block_split_outer_size):
        # There is no need to adjust the situation:
        # 1) split the first axis, 2) divide the split, 3) the axis before
        # the axis to be split is 1
        if block_split_axis == 0 or \
                shape[block_split_axis] % block_split_inner_size == 0 \
                or sum(shape[0:block_split_axis]) == block_split_axis:
            return block_split_axis, block_split_inner_size, \
                   block_split_outer_size

        core_num = self._get_block_num()
        sorted_factors = self._get_factors_of_positive_integer(
            shape[block_split_axis])
        bound_size = 1
        for i in range(0, block_split_axis, 1):
            bound_size = bound_size * shape[i]

        # find f_small and f_large that closest to core_num,
        # and f_small < core_num, f_large > core_num
        f_small = 1
        f_large = 1
        bound_size_temp = bound_size
        for i in range(0, len(sorted_factors), 1):
            bound_size_temp = bound_size * sorted_factors[i]
            if bound_size_temp > core_num:
                f_large = sorted_factors[i]
                if i > 0:
                    f_small = sorted_factors[i - 1]
                break

        if f_large * bound_size > core_num * BLOCK_DIM_MULTIPLE:
            block_split_outer_size = f_small
        elif data_size < f_large * bound_size * multi_core_threshold:
            block_split_outer_size = f_small
        elif f_small * bound_size > core_num * BLOCK_DIM_LOWER_BOUND_THRESHOLD:
            block_split_outer_size = f_small
        else:
            block_split_outer_size = f_large

        block_split_inner_size = shape[block_split_axis] // \
                                 block_split_outer_size

        return block_split_axis, block_split_inner_size, block_split_outer_size


    def _get_factors_of_positive_integer(self, axis):
        factors = []
        if axis <= 0:
            return factors
        sqrt_axis = int(math.sqrt(axis))
        for i in range(1, sqrt_axis + 1, 1):
            if axis % i == 0:
                rfactor = axis // i
                factors.append(i)
                if rfactor != i:
                    factors.append(rfactor)
        factors.sort()
        return factors


    def _get_ub_tiling(self, # pylint: disable=too-many-locals, too-many-branches
                       shape, block_tiling_axis, block_tiling_inner_loop,
                       max_ub_count, divisible_split=False):
        last_axis = len(shape) - 1
        ub_split_inner = 1
        ub_split_axis = 0
        if block_tiling_axis < 0 or block_tiling_axis > last_axis:
            return ub_split_axis, ub_split_inner

        bound_size = max_ub_count
        split_axis = block_tiling_axis
        step = -1
        temp_size = 1
        need_split = False
        for i in range(last_axis, block_tiling_axis + step, step):
            temp_size = temp_size * shape[i]
            if temp_size >= bound_size:
                split_axis = i
                temp_size = temp_size / shape[i]
                need_split = True
                break

        split_size = 1
        # split the split axis
        if need_split:
            for i in range(1, shape[split_axis] + 1, 1):
                if (temp_size * i) == bound_size:
                    split_size = i
                    break
                if (temp_size * i) > bound_size:
                    split_size = i - 1
                    break
            if divisible_split:
                if shape[split_axis] % split_size != 0:
                    while shape[split_axis] % split_size != 0:
                        split_size -= 1
        else:
            split_size = block_tiling_inner_loop

        split_size = self._modify_split_size(block_tiling_axis,
                                             block_tiling_inner_loop,
                                             bound_size, need_split,
                                             shape, split_axis, split_size)

        ub_split_inner = split_size
        ub_split_axis = split_axis

        # There is no need to adjust the situation:
        # 1:block_tiling_axis == 0
        # 2:block tiling is not the same axis as ub tiling
        # 3:ub tiling axis divisible the split factor
        if block_tiling_axis != 0:
            if block_tiling_axis == ub_split_axis and \
                    block_tiling_inner_loop % ub_split_inner != 0:
                ub_split_inner = self._modify_ub_tiling(
                    block_tiling_inner_loop, ub_split_inner)

        return ub_split_axis, ub_split_inner

    @staticmethod
    def _modify_split_size(*args):
        block_tiling_axis, block_tiling_inner_loop, bound_size, need_split, \
        shape, split_axis, split_size = args
        if split_axis == block_tiling_axis:
            if (shape[block_tiling_axis] % block_tiling_inner_loop == 1) and \
                    not need_split and split_size < bound_size:
                split_size = split_size + 1
            elif split_size > block_tiling_inner_loop:
                split_size = block_tiling_inner_loop
        return split_size

    # The backend does not support non-divisible split+fuse, if block tiling
    # and ub tiling are the same axis,
    # then it need to ensure that the ub tiling divisible split,
    # for the non-divisible split needs to be adjusted to divisible split
    def _modify_ub_tiling(self, block_split_inner, ub_split_inner):

        sorted_factors = self._get_factors_of_positive_integer(
            block_split_inner)

        # find ub_split_inner_factor, ub_split_inner_factor <= ub_split_inner
        ub_split_inner_factor = 1
        for i in range(0, len(sorted_factors), 1):
            if sorted_factors[i] > ub_split_inner:
                if i > 0:
                    ub_split_inner_factor = sorted_factors[i - 1]
                break
            else:
                ub_split_inner_factor = sorted_factors[i]

        return ub_split_inner_factor

    # pylint: disable=too-many-locals, too-many-branches, too-many-arguments, too-many-return-statements
    def _is_need_modify_block_and_ub_tiling(self, shape, dtype,
                                            block_split_axis,
                                            block_split_inner_size,
                                            ub_split_axis, ub_split_inner,
                                            max_ub_count):
        last_axis = len(shape) - 1

        # judge block dim is 1
        if block_split_axis == 0:
            if shape[block_split_axis] // block_split_inner_size == 1:
                return False
        else:
            block_dim = 1
            for i in range(0, block_split_axis + 1, 1):
                block_dim = block_dim * shape[i]
            if block_dim // block_split_inner_size == 1:
                return False

        if ub_split_axis == last_axis:
            size = DTYPE_WIDTH_MAP[dtype] * 2 * ub_split_inner
            if size < 32:
                return True
            if int(size) % max_ub_count != 0:
                if size % max_ub_count < 32:
                    return True
            # ub split, the tail process
            if block_split_axis == ub_split_axis:
                tail_count = block_split_inner_size % ub_split_inner
            else:
                tail_count = shape[ub_split_axis] % ub_split_inner
            tail_size = DTYPE_WIDTH_MAP[dtype] * 2 * tail_count
            if tail_count != 0:
                if tail_size < 32:
                    return True
                if int(tail_size) % max_ub_count != 0:
                    if tail_size % max_ub_count < 32:
                        return True

        else:
            data_size = 1
            for i in range(ub_split_axis + 1, len(shape), 1):
                data_size = data_size * shape[i]
            size = DTYPE_WIDTH_MAP[dtype] * 2 * ub_split_inner * data_size
            if size < 32:
                return True
            if int(size) % max_ub_count != 0:
                if size % max_ub_count < 32:
                    return True
            # ub split, the tail process
            if block_split_axis == ub_split_axis:
                tail_count = block_split_inner_size % ub_split_inner
            else:
                tail_count = shape[ub_split_axis] % ub_split_inner
            tail_size = DTYPE_WIDTH_MAP[dtype] * 2 * tail_count * data_size
            if tail_count != 0:
                if tail_size < 32:
                    return True
                if int(tail_size) % max_ub_count != 0:
                    if tail_size % max_ub_count < 32:
                        return True

        return False

    def _map_apend(self, input_map, key, value):
        if input_map.get(key):
            if isinstance(value, list):
                for tmp_value in value:
                    if tmp_value not in input_map[key]:
                        input_map[key].append(tmp_value)
            else:
                if value not in input_map[key]:
                    input_map[key].append(value)
        else:
            if isinstance(value, list):
                input_map[key] = value
            else:
                input_map[key] = [value]

    def _shape_to_list(self, shape):
        """
        translate tvm.shape to list type in python
        """
        tmp = []
        for i in shape:
            tmp.append(i.value)
        return tmp

    def _get_emit_insn_map(self):
        self._insn_map = {"elewise_single_cast": "vector_conv",
                          "elewise_single_round_d": "vector_conv_round",
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
                          "elewise_binary_sub": "vector_sub",
                          "elewise_binary_phony": "elewise_binary_phony"}

    def _get_reg_emit_insn_map(self):
        self._reg_insn_map = {
            "broadcast_for_tensor": "unified_broadcast",
            "elewise_binary_scalar_axpy": "vector_multiple",
            "emit_insn_elewise_multiple_sel":
                "elewise_multiple_sel",
            "emit_insn_elewise_binary_cmp":
                "elewise_binary_cmp",
            "elewise_binary_cmpsel": "vector_cmpsel",
            "elewise_single_VS_cond": "elewise_single_VS_cond",
            "elewise_binary_logic": "elewise_binary_logic",
            "broadcast": "broadcast",
            "elewise_single_round": "elewise_single_round",
            "elewise_single_rec": "vector_rec",
            "elewise_binary_sub": "elewise_binary_sub",
            "elewise_single_cast": "elewise_single_cast",
            "elewise_single_floor": "elewise_single_floor",
            "elewise_single_ceil": "elewise_single_ceil",
            "elewise_single_trunc": "elewise_single_trunc",
            "elewise_binary_compare_lt":
                "elewise_binary_compare_lt",
            "elewise_binary_compare_gt":
                "elewise_binary_compare_gt",
            "elewise_single_VS_mul_with_reg_in_quant":
                "elewise_single_VS_mul_with_reg_in_quant",
            "elewise_single_VS_adds_with_reg":
                "elewise_single_VS_adds_with_reg",
            "elewise_single_VS_mul_with_reg_sqrt_in_quant":
                "elewise_single_VS_mul_with_reg_sqrt_in_quant",
            "elewise_single_VS_mul_with_reg":
                "elewise_single_VS_mul_with_reg",
            "elewise_single_VS_add_with_reg":
                "elewise_single_VS_add_with_reg",
            "elewise_single_diagonal":
                "elewise_single_diagonal"}

