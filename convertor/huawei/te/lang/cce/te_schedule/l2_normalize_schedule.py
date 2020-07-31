#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# pylint: unused-import
"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

l2 normalize schedule
"""
# pylint: disable=unused-import
import math
from te import platform as cceconf
from te import tvm
from te.platform import cce_util
import te.platform.cce_params as cce
from te.platform import cce_emitinsn_params
from . import util
from .vector_schedule import VectorSchedule


DTYPE_WIDTH_MAP = {"float16": 1,
                   "float32": 2,
                   "int32": 2,
                   "int16": 1,
                   "uint16": 1,
                   "int8": 0.5,
                   "uint8": 0.5,
                   "bool": 0.5}


# pylint: disable=too-many-return-statements,too-few-public-methods,too-many-arguments,too-many-statements,no-self-use,too-many-lines,too-many-instance-attributes,too-many-branches,
class L2NormalizeSchedule(VectorSchedule):
    """
    Base class of cce API

    Parameters
    ----------
    scope : cal buffer like local.UB
    Returns
    -------
    """

    def __init__(self, need_multi_core=True):
        VectorSchedule.__init__(self, need_multi_core)
        # record ops
        self._op = []
        # record origin op
        self._origin_op = []
        self._is_muti_output = False
        self._have_reduce = False
        self._last_output_tensor = None
        self._input_tensors = []
        self._input_tensor_dst_tensor_map = {}
        self._mid_tensors = []  # exclude _input_tensors and last_output_tensor
        self._mid_tensor_dst_tensor_map = {}  # {mid_tensor->dst_tensor}
        self._mid_output_tensors = []
        self._mid_output_tensors_dst_tensor_map = {}
        self._broadcast_not_last_axis_tensors = []
        self._cache_write_exclude_tensors = []
        self._broadcast_last_axis_tensors = []
        self._broadcast_last_axis_tensor_dst_tensor_map = {}
        self._broadcast_scalars = []
        self._broadcast_scalar_dst_tensor_map = {}
        self._shape_before_reduce = None
        self._before_reduce_tensor_list = []
        self._after_reduce_tensor_list = []
        self._tensor_scaler_operator = ["elewise_binary_mul",
                                        "elewise_binary_add"]
        self._reduce_info = {"reduce_tensor": None,
                             # key:reduce_axis_index, value:reduce_axis_var
                             "reduce_axis_map": {},
                             "reduce_axis_index": [],
                             "shape_before_reduce": None,
                             "is_last_axis_reduce": False,
                             "dtype": None}

    # pylint: disable=arguments-differ
    def do_schedule(self, out_tensors, sch, spec_node_list):
        """
        auto_schedule for l2 normalize

        Parameters
        ----------
        outTensors : tvm.tensor, only support form like: out_1->..out_2->..out_n

        Returns
        -------
        sch: Schedule
            The computation schedule for the op.

        """

        self._schedule = sch
        self._construct_compute_graph(out_tensors, spec_node_list)

        schedule_valid = self._check_pattern_supported()
        if not schedule_valid:

            return False

        self._calculate_cache_read()
        self._do_cache_read()

        self._calculate_cache_write()
        self._do_cache_write()

        self._calculate_tiling()
        self._do_tiling()

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

        return True


    def _construct_compute_graph(self, out_tensors, spec_node_list):
        """
        record relate context imformations of operations

        outTensors only support form like: out_1->..out_2->..out_n

        """
        # find the last out tensor
        if hasattr(out_tensors, 'index'):
            if len(out_tensors) > 1:
                util.get_dst_tensor_map(out_tensors,
                                        self._mid_output_tensors_dst_tensor_map)
                for out in out_tensors:
                    if out not in self._mid_output_tensors_dst_tensor_map.keys():
                        self._last_output_tensor = out
            else:
                self._last_output_tensor = out_tensors[0]
        else:
            self._last_output_tensor = out_tensors

        self._mid_output_tensors = out_tensors

        after_reduce_tensor_list = []
        dst_tensor_map = {}
        self._spec_node_list = spec_node_list

        spec_node_list_temp = [each for each in spec_node_list]
        reduce_tensor = self._find_reduce_node(self._last_output_tensor,
                                               spec_node_list)
        if reduce_tensor is not None:
            self._record_reduce_info(reduce_tensor)

        # find tensors after reduce_tensor, include reduce_tensor
        self.__gen_reversed_subgraph_list(self._last_output_tensor,
                                          spec_node_list_temp,
                                          after_reduce_tensor_list,
                                          dst_tensor_map)
        after_reduce_tensor_list.append(self._last_output_tensor)

        self._record_tensor_info(spec_node_list, after_reduce_tensor_list,
                                 dst_tensor_map)

        self._after_reduce_tensor_list = after_reduce_tensor_list

        tensor_excluding_input_list = self._mid_tensors + [self._last_output_tensor]

        for tens in reversed(tensor_excluding_input_list):
            tmp_op = self.__split_tensor(tens)
            if tmp_op["effective_op"]:
                self._op.append(tmp_op)
            self._origin_op.append(tmp_op)

    # pylint: disable=too-many-nested-blocks
    def __gen_reversed_subgraph_list(self, out_tensor, spec_node_list,
                                     tensor_list, tensor_list_dst_tensor_map):
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
        # pylint: disable=too-many-nested-blocks
        while stack:
            cur_tensor = stack.pop()
            visited_list.append(cur_tensor)
            for in_tensor in cur_tensor.op.input_tensors:
                if in_tensor not in visited_list:
                    if in_tensor not in spec_node_list:
                        stack.append(in_tensor)
                    tensor_list.append(in_tensor)
                    self._map_apend(tensor_list_dst_tensor_map, in_tensor,
                                    cur_tensor)
                    if self._is_broadcast_last_axis_tensor(in_tensor):
                        if in_tensor not in \
                                self._broadcast_last_axis_tensor_dst_tensor_map.keys():
                            self._broadcast_last_axis_tensors.append(in_tensor)
                        self._map_apend(
                            self._broadcast_last_axis_tensor_dst_tensor_map,
                            in_tensor, cur_tensor)
                    else:
                        if self._is_broadcast_not_last_axis_tensor(in_tensor):
                            if in_tensor not in self._broadcast_not_last_axis_tensors:
                                self._broadcast_not_last_axis_tensors.append(
                                    in_tensor)

    def _record_tensor_info(self, spec_node_list, all_tensor_list,
                            all_tensor_list_dst_tensor_map):
        for tensor in reversed(all_tensor_list):
            if isinstance(tensor.op, tvm.tensor.PlaceholderOp) or \
                    tensor in spec_node_list:
                if tensor not in self._input_tensor_dst_tensor_map.keys():
                    self._input_tensors.append(tensor)
                self._map_apend(self._input_tensor_dst_tensor_map, tensor,
                                all_tensor_list_dst_tensor_map[tensor])
            else:
                if tensor != self._last_output_tensor:
                    if tensor not in self._mid_tensor_dst_tensor_map.keys():
                        self._mid_tensors.append(tensor)
                    self._map_apend(self._mid_tensor_dst_tensor_map, tensor,
                                    all_tensor_list_dst_tensor_map[tensor])

    def _find_reduce_node(self, out_tensor, spec_node_list):
        if out_tensor is None:
            return None
        stack = [out_tensor]
        visited_list = []
        while stack:
            cur_tensor = stack.pop()
            visited_list.append(cur_tensor)
            if self._is_reduce_tensor(cur_tensor):
                return cur_tensor
            for in_tensor in cur_tensor.op.input_tensors:
                if in_tensor not in visited_list and \
                        in_tensor not in spec_node_list:
                    stack.append(in_tensor)

        return None

    def _check_pattern_supported(self):
        shape_before_reduce = self._shape_before_reduce
        reduce_axis_index = self._reduce_info["reduce_axis_index"]
        dtype = self._reduce_info["dtype"]

        reduce_last_axis = self._is_reduce_last_axis(shape_before_reduce,
                                                     reduce_axis_index)
        if not reduce_last_axis:
            return False

        align_32_byte = self._is_reduce_axis_32byte_align(
            shape_before_reduce, reduce_axis_index, dtype)
        if not align_32_byte:
            return False

        large = self._is_reduce_axis_large_than_ub_max(
            shape_before_reduce, reduce_axis_index, dtype)
        if large:
            return False

        return True

    def _is_reduce_last_axis(self, shape_before_reduce, reduce_axis_index):

        if len(reduce_axis_index) > 1:
            has_last_reduce_axis = \
                ((len(shape_before_reduce) - 1) in reduce_axis_index)
            if has_last_reduce_axis:
                is_continuous_reduce = \
                    self._is_continuous_last_reduce(reduce_axis_index)
                if not is_continuous_reduce:
                    return self._is_all_one_between_reduce_axis(
                        shape_before_reduce, reduce_axis_index)
                return True
            return False
        has_last_reduce_axis = \
            ((len(shape_before_reduce) - 1) in reduce_axis_index)
        return has_last_reduce_axis


    def _is_reduce_axis_32byte_align(self, shape_before_reduce,
                                     reduce_axis_index, dtype):

        last_axis_size = shape_before_reduce[reduce_axis_index[-1]]

        if not self._is_last_axis_32byte_align(last_axis_size, dtype):
            return False

        return True

    # pylint: consider-using-enumerate
    def _is_continuous_last_reduce(self, reduce_axis_index):
        # pylint: disable=consider-using-enumerate
        for i in range(0, len(reduce_axis_index)):
            if i > 0:
                if reduce_axis_index[i] != reduce_axis_index[i-1] + 1:
                    return False
        return True

    def _is_all_one_between_reduce_axis(self, shape_before_reduce,
                                        reduce_axis_index):
        for i in range(reduce_axis_index[0] + 1, reduce_axis_index[-1]):
            if i not in reduce_axis_index:
                if shape_before_reduce[i] != 1:
                    return False
        return True

    def _is_last_axis_32byte_align(self, last_axis_size, dtype):
        # last_axis_size include continuous last axis
        data_size = DTYPE_WIDTH_MAP[dtype] * 2 * last_axis_size
        if not (data_size >= 32 and data_size % 32 == 0):
            return False

        return True

    def _is_reduce_axis_large_than_ub_max(self, shape_before_reduce,
                                          reduce_axis_index, dtype):
        last_axis_size = 1
        for i in range(reduce_axis_index[0], reduce_axis_index[-1] + 1):
            last_axis_size = last_axis_size * shape_before_reduce[i]

        data_size = DTYPE_WIDTH_MAP[dtype] * 2 * last_axis_size
        max_ub_count = self._get_max_ub_count()
        return data_size > max_ub_count

    def _calculate_cache_read(self):
        """
        cache read operations

        Parameters
        ----------
        None

        Returns
        -------
        list : read buffers
        """
        for i in self._input_tensors:
            self._map_apend(self._cache_read_tensors_and_readers_map, i,
                            self._input_tensor_dst_tensor_map[i])

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
            if len(readers) > 1:
                for reader in readers:
                    read_buffer = self._schedule.cache_read(i, self._scope, reader)
                    self._map_apend(self._cache_read_tensors_and_buffer_map, i, read_buffer)
                    self._double_buffer_tensors.append(read_buffer)
            else:
                read_buffer = self._schedule.cache_read(i, self._scope, readers)
                self._cache_read_tensors_and_buffer_map[i] = read_buffer
                self._double_buffer_tensors.append(read_buffer)

    def _calculate_cache_write(self):
        """
        cache read operations

        Parameters
        ----------
        None

        Returns
        -------
        list : read buffers
        """
        for i in self._mid_tensors:
            if i not in self._cache_write_exclude_tensors:
                self._cache_write_tensors.append(i)
        self._cache_write_tensors.append(self._last_output_tensor)

    # pylint: consider-using-enumerate
    def _calculate_tiling(self):
        shape_before_reduce = self._shape_before_reduce
        dtype = self._last_output_tensor.dtype.lower()
        reduce_axis_index = self._reduce_info["reduce_axis_index"]

        block_split_shape = []
        # pylint: disable=consider-using-enumerate
        for i in range(0, len(shape_before_reduce)):
            if i not in reduce_axis_index:
                block_split_shape.append(shape_before_reduce[i])
            else:
                break

        if self._need_multi_core:
            block_split_axis, block_split_inner_size, _ = \
                self._get_block_tiling(block_split_shape, dtype, 1024)
        else:
            block_split_axis = 0
            block_split_inner_size = shape_before_reduce[block_split_axis]

        block_tiling_para = {"tiling_tensor": self._last_output_tensor,
                             "axis": block_split_axis,
                             "factor": block_split_inner_size}

        max_ub_count = self._get_max_ub_count()

        ub_split_axis, ub_split_inner = \
            self._get_ub_tiling(shape_before_reduce, block_split_axis,
                                block_split_inner_size, max_ub_count)

        if self._need_multi_core and \
                self._is_need_modify_block_and_ub_tiling(
                        shape_before_reduce, dtype, block_split_axis,
                        block_split_inner_size, ub_split_axis,
                        ub_split_inner, max_ub_count):
            block_split_axis = 0
            block_split_inner_size = shape_before_reduce[block_split_axis]
            ub_split_axis, ub_split_inner = \
                self._get_ub_tiling(shape_before_reduce,
                                    block_split_axis,
                                    block_split_inner_size,
                                    max_ub_count)

        ub_tiling_para = [{"tiling_tensor": self._last_output_tensor,
                           "axis": ub_split_axis,
                           "axis_var": None,
                           "factor": ub_split_inner}]

        self._tiling_para["block_tiling"] = block_tiling_para
        self._tiling_para["ub_tiling"] = ub_tiling_para
        self._tiling_tensor = self._last_output_tensor

    # pylint: disable=too-many-locals
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

    # pylint: disable=too-many-locals
    def _do_tiling(self):
        block_tiling_para = self._tiling_para["block_tiling"]
        block_tiling_tensor = block_tiling_para["tiling_tensor"]
        block_split_axis = block_tiling_para["axis"]
        block_split_inner_size = block_tiling_para["factor"]
        if block_tiling_tensor is not None:
            # if axis_var is not empty, use axis_var as split parameter first
            # otherwise use split axis of tilting_tensor as the split parameter
            axis_var = block_tiling_tensor.op.axis[block_split_axis]
            if "axis_var" in block_tiling_para.keys() and \
                    block_tiling_para["axis_var"]:
                axis_var = block_tiling_para["axis_var"]
            res_block_outer, res_block_inner = \
                self._schedule[block_tiling_tensor].split(
                    axis_var, factor=block_split_inner_size)

            block_tiling_result = {"tiling_tensor": block_tiling_tensor,
                                   "axis": block_split_axis,
                                   "parent_itervar": axis_var,
                                   "outer_itervar": res_block_outer,
                                   "inner_itervar": res_block_inner}

            self._tiling_result["block_tiling"] = block_tiling_result

        ub_tiling_result_list = []
        ub_tiling_para_list = self._tiling_para["ub_tiling"]
        for ub_tiling_para in ub_tiling_para_list:
            ub_tiling_tensor = ub_tiling_para["tiling_tensor"]
            ub_split_axis = ub_tiling_para["axis"]
            ub_split_inner = ub_tiling_para["factor"]
            if ub_tiling_tensor is not None:
                if block_tiling_tensor is not None and \
                        block_split_axis == ub_split_axis and \
                        ub_tiling_tensor == block_tiling_tensor:
                    res_ub_outer, res_ub_inner = \
                        self._schedule[ub_tiling_tensor].split(res_block_inner,
                                                               factor=ub_split_inner)
                    ub_tiling_result = {"tiling_tensor": ub_tiling_tensor,
                                        "axis": ub_split_axis,
                                        "parent_itervar": res_block_inner,
                                        "outer_itervar": res_ub_outer,
                                        "inner_itervar": res_ub_inner}
                else:
                    # if axis_var is not empty, use axis_var as split parameter
                    # first, otherwise use split axis of tilting_tensor as the
                    # split parameter
                    axis_var = ub_tiling_tensor.op.axis[ub_split_axis]
                    if "axis_var" in ub_tiling_para.keys() and \
                            ub_tiling_para["axis_var"] is not None:
                        axis_var = ub_tiling_para["axis_var"]
                    res_ub_outer, res_ub_inner = \
                        self._schedule[ub_tiling_tensor].split(
                            axis_var, factor=ub_split_inner)
                    ub_tiling_result = {"tiling_tensor": ub_tiling_tensor,
                                        "axis": ub_split_axis,
                                        "parent_itervar": axis_var,
                                        "outer_itervar": res_ub_outer,
                                        "inner_itervar": res_ub_inner}
                ub_tiling_result_list.append(ub_tiling_result)

        self._tiling_result["ub_tiling"] = ub_tiling_result_list

    def _calculate_compute_inline(self):
        for i in self._mid_tensors:
            self._compute_inline_tensors.append(i)

    def _calculate_multi_core(self):
        if self._need_multi_core:
            res = self._last_output_tensor
            block_tiling_result = self._tiling_result["block_tiling"]
            if block_tiling_result:
                block_split_axis = block_tiling_result["axis"]
                res_block_outer = block_tiling_result["outer_itervar"]
                need_fuse_list = [res_block_outer]
                for i in range(block_split_axis - 1, -1, -1):
                    need_fuse_list.append(res.op.axis[i])
                fused_axis = need_fuse_list[0]
                for i in range(1, len(need_fuse_list)):
                    fused_axis = self._schedule[res].fuse(fused_axis,
                                                          need_fuse_list[i])

                self._multi_core_fused_axis = fused_axis
                self._multi_core_bind_tensor = res

    def _calculate_compute_at(self):
        ub_tiling_result_list = self._tiling_result["ub_tiling"]
        include_tensor_list = []

        for ub_tiling_result in ub_tiling_result_list:
            if "tiling_tensor" not in ub_tiling_result.keys() or \
                    "outer_itervar" not in ub_tiling_result.keys():
                continue

            ub_tiling_tensor = ub_tiling_result["tiling_tensor"]
            res_ub_outer = ub_tiling_result["outer_itervar"]
            res = ub_tiling_tensor
            reduce_buffer = self._cache_write_tensors_and_buffer_map[
                self._reduce_info["reduce_tensor"]]
            if ub_tiling_tensor in \
                    (self._reduce_info["reduce_tensor"], reduce_buffer):
                include_tensor_list = self._before_reduce_tensor_list
            elif ub_tiling_tensor == self._last_output_tensor:
                include_tensor_list = self._after_reduce_tensor_list

            for i in self._cache_read_tensors_and_buffer_map:
                if i in include_tensor_list:
                    read_buffer = self._cache_read_tensors_and_buffer_map[i]
                    para = {"parent": self._schedule[res],
                            "scope": res_ub_outer}
                    if isinstance(read_buffer, (list)):
                        for buffer_tensor in read_buffer:
                            self._compute_at_map[buffer_tensor] = para
                    else:
                        self._compute_at_map[read_buffer] = para

            for i in self._cache_write_tensors_and_buffer_map:
                if i in include_tensor_list:
                    write_buffer = self._cache_write_tensors_and_buffer_map[i]
                    para = {"parent": self._schedule[res], "scope": res_ub_outer}
                    self._compute_at_map[write_buffer] = para

    def _calculate_double_buffer(self):
        """
        double buffer operations
        read_buffer : the all read_cache for input in ub, type is list
        """
        shape = self._shape_before_reduce

        block_tiling_para = self._tiling_para["block_tiling"]
        block_split_axis = block_tiling_para["axis"]
        block_split_inner_size = block_tiling_para["factor"]

        ub_tiling_para_list = self._tiling_para["ub_tiling"]
        if not ub_tiling_para_list:
            return

        for ub_tiling_para in ub_tiling_para_list:
            ub_split_axis = ub_tiling_para["axis"]
            ub_split_inner = ub_tiling_para["factor"]

        self._need_db = self._need_double_buffer(shape, block_split_axis,
                                                 block_split_inner_size,
                                                 ub_split_axis,
                                                 ub_split_inner)

    # pylint: disable=too-many-locals
    def _calculate_emit_insn(self):

        self._get_emit_insn_map()
        self._get_reg_emit_insn_map()

        ub_tiling_result_list = self._tiling_result["ub_tiling"]
        include_tensor_list = []

        for ub_tiling_result in ub_tiling_result_list:

            if "tiling_tensor" not in ub_tiling_result.keys() or \
                    "inner_itervar" not in ub_tiling_result.keys() or \
                    "axis" not in ub_tiling_result.keys():
                continue

            ub_tiling_tensor = ub_tiling_result["tiling_tensor"]
            res_ub_inner = ub_tiling_result["inner_itervar"]
            ub_split_axis = ub_tiling_result["axis"]
            reduce_buffer = self._cache_write_tensors_and_buffer_map[
                self._reduce_info["reduce_tensor"]]

            if ub_tiling_tensor in \
                    (self._reduce_info["reduce_tensor"], reduce_buffer):
                # include reduce_tensor
                include_tensor_list = []
                include_tensor_list = \
                    [each for each in self._before_reduce_tensor_list]
                include_tensor_list.append(self._reduce_info["reduce_tensor"])
            elif ub_tiling_tensor == self._last_output_tensor:
                include_tensor_list = []
                # exclude reduce_tensor
                for each in self._after_reduce_tensor_list:
                    include_tensor_list.append(each)

            res = ub_tiling_tensor

            for i in self._cache_read_tensors_and_buffer_map:
                if i in include_tensor_list:
                    read_buffer = self._cache_read_tensors_and_buffer_map[i]
                    if isinstance(read_buffer, (list)):
                        for buffer_tensor in read_buffer:
                            para = {"scope": buffer_tensor.op.axis[ub_split_axis],
                                    "instruction": 'dma_copy'}
                            self._emit_insn_map[buffer_tensor] = para
                    else:
                        para = {"scope": read_buffer.op.axis[ub_split_axis],
                                "instruction": 'dma_copy'}
                        self._emit_insn_map[read_buffer] = para

            for i in self._cache_write_tensors_and_buffer_map:
                if i in include_tensor_list:
                    write_buffer = self._cache_write_tensors_and_buffer_map[i]
                    insn = self._calculate_emit_insn_map(write_buffer)

                    para = {"scope": write_buffer.op.axis[ub_split_axis],
                            "instruction": insn}

                    if self._is_reduce_tensor(write_buffer):
                        insn = write_buffer.op.tag.split("|")[0]
                        insn = "vector_" + insn
                        para = {"scope": write_buffer.op.reduce_axis[0],
                                "instruction": insn}

                    if insn == "unified_broadcast":
                        if not self._is_broadcast_last_axis_tensor(i):
                            continue
                        else:
                            max_last_broadcast_axis_offset = \
                                self._find_max_broadcast_last_axis_offset()
                            cce_emitinsn_params.cceEmitParamsIns.del_param(
                                "broadcast_axis_offset")
                            cce_emitinsn_params.cceEmitParamsIns.insert_param(
                                "broadcast_axis_offset",
                                max_last_broadcast_axis_offset)

                    self._emit_insn_map[write_buffer] = para

            for out_tensor in [self._last_output_tensor]:
                if out_tensor in include_tensor_list:
                    self._emit_insn_map[res] = {"scope": res_ub_inner,
                                                "instruction": 'dma_copy'}

    def _calculate_emit_insn_map(self, tensor):
        if tensor.op.tag.find("|") != -1:
            str_list = tensor.op.tag.split("|")
            insn = self._insn_map.get(str_list[0])
            if insn and self._check_cast_support(tensor):
                return insn
            insn = self._reg_insn_map.get(str_list[0])
        else:
            insn = self._insn_map.get(tensor.op.tag)
            if insn and self._check_cast_support(tensor):
                return insn
            insn = self._reg_insn_map.get(tensor.op.tag)

        return insn

    def _check_cast_support(self, tensor):
        cache_buffer = tensor
        read_buffer = tensor.op.input_tensors[0]
        if read_buffer.dtype == "int32":
            if cache_buffer.dtype == "float16" or \
                    cache_buffer.dtype == "float32":
                return False
        return True

    def _is_reduce_tensor(self, tensor):
        if tensor.op.tag.find("reduce") != -1:
            return True
        return False

    def _record_reduce_info(self, tensor):
        if self._is_reduce_tensor(tensor):
            self._reduce_info["reduce_tensor"] = tensor
            op_node = tensor.op
            reduce_axis_var = []

            for i in op_node.reduce_axis:
                reduce_axis_var.append(i)
            data_axis_var = op_node.body[0].source[0].args

            for ax_var in reduce_axis_var:
                for index in range(0, len(data_axis_var), 1):
                    if data_axis_var[index].same_as(ax_var.var):
                        self._reduce_info["reduce_axis_index"].append(index)
                        self._reduce_info["reduce_axis_map"][index] = ax_var

            self._reduce_info["dtype"] = tensor.dtype
            self._reduce_info["reduce_axis_index"].sort()
            tmp_reduce_axis_num = self._reduce_info["reduce_axis_index"]

            if tensor.op.input_tensors:
                shape_before_reduce = \
                    self._shape_to_list(tensor.op.input_tensors[0].shape)
                self._shape_before_reduce = shape_before_reduce
                self._reduce_info["shape_before_reduce"] = shape_before_reduce
                is_last_reduce = \
                    ((len(shape_before_reduce) - 1) in tmp_reduce_axis_num)
                self._reduce_info["is_last_axis_reduce"] = is_last_reduce

    def is_keepdims(self):
        """
        check whether keepdims
        """
        # if the dims of shape_before_reduce is the same as resTensor.shape,
        # the keepdims is true
        if len(self._shape_before_reduce) == len(self._last_output_tensor.shape):
            return True
        return False

    # (a_1,..,(a_ko,a_ki),...,(a_lo,a_li),...,a_n)
    def _need_double_buffer(self, shape, block_axis,
                            block_tiling_inner_loop,
                            ub_axis, ub_tiling_inner_loop):
        if ub_axis < block_axis or ub_axis < 0 or block_axis < 0:
            return False
        if ub_axis == block_axis:
            one_core_loop_number = block_tiling_inner_loop
        else:
            ub_tiling_outer_loop = shape[ub_axis] // ub_tiling_inner_loop
            one_core_loop_number = \
                block_tiling_inner_loop * ub_tiling_outer_loop

        for i in range(block_axis + 1, ub_axis, 1):
            one_core_loop_number = one_core_loop_number * shape[i]

        return one_core_loop_number > 1

    def _get_total_width(self):
        """
        caculate the max useable number based on op liveness
        :return: max useable number
        """
        res = self._last_output_tensor

        def _post_dfs_order(op_node, op_graph, visited, post_order):
            if op_node in visited:
                return
            visited[op_node] = True
            post_order.append(op_node)
            if op_node in op_graph:
                for src in op_graph[op_node]:
                    _post_dfs_order(src, op_graph, visited, post_order)

        def _op_width(op_node):
            num_type = op_node.dtype
            if num_type.lower() not in DTYPE_WIDTH_MAP.keys():
                raise RuntimeError("Can not calculate with no compute")

            tmp_width = 0
            if op_node.op.tag is not None:
                tag = op_node.op.tag
                # logic use 4 fp16 temp buffer
                if tag.find("logic") != -1:
                    tmp_width = 4 * DTYPE_WIDTH_MAP["float16"]
                # cond use 3 fp16 temp buffer
                elif tag.find("cond") != -1:
                    tmp_width = 3 * DTYPE_WIDTH_MAP["float16"]
                # vsel use 3 fp16 temp buffer
                elif tag.find("sel") != -1:
                    tmp_width = 3 * DTYPE_WIDTH_MAP["float16"]
                # vcompare use 2 temp buffer
                elif tag.find("compare") != -1:
                    tmp_width = 2 * DTYPE_WIDTH_MAP[num_type.lower()]

            return DTYPE_WIDTH_MAP[num_type.lower()] + tmp_width

        op_graph = {}
        for op_node in self._origin_op:
            src_op = list(op_node['src_buffer'])
            src_op.reverse()
            op_graph[op_node['dst_buffer']] = src_op
        visited = {}
        post_order = []
        _post_dfs_order(res, op_graph, visited, post_order)
        lives = [res]
        live_width = _op_width(lives[0])
        max_width = live_width
        visited = {lives[0]: True}
        for op_node in post_order:
            if op_node in op_graph:
                for src in op_graph[op_node]:
                    if src in visited:
                        continue
                    lives.append(src)
                    live_width += _op_width(src)
                    visited[src] = True
                if live_width > max_width:
                    max_width = live_width
            lives.remove(op_node)
            live_width -= _op_width(op_node)
        return max_width

    def _get_max_ub_count(self):
        """
        caculate the max element num loaded in UB buffer
        :return: max element num loaded in UB buffer
        """
        # div 2 for align to fp16
        total_size = cceconf.get_soc_spec("UB_SIZE") // 2
        total_size = total_size // 2  # div 2 for double buffer
        total_width = self._get_total_width()
        if not total_width:
            raise RuntimeError("Can not calculate with no compute")

        max_bound = total_width * 128
        max_ub_count = int(total_size // max_bound * 128)

        return max_ub_count

    def __split_tensor(self, tensor):
        tmp_op = {}
        op_stmt = tensor.op
        tmp_op["op"] = op_stmt.tag
        tmp_op["dst_buffer"] = tensor
        tmp_op["src_buffer"] = list(op_stmt.input_tensors)
        tmp_op["args"] = []
        tmp_op["effective_op"] = True

        if tmp_op["op"].find("elewise_single") != -1:
            if hasattr(op_stmt.body[0], 'b'):
                if isinstance(op_stmt.body[0].a, tvm.expr.Call):
                    tmp_op["args"] = [op_stmt.body[0].b]
                else:
                    tmp_op["args"] = [op_stmt.body[0].a]
        if tmp_op["op"].find("elewise_binary_compare") != -1:
            if hasattr(op_stmt.body[0], 'condition'):
                tmp_op["args"] = [op_stmt.body[0].condition.b]
            if tmp_op["op"].find("lt") != -1:
                tmp_op["args"].append("lt")
            elif tmp_op["op"].find("gt") != -1:
                tmp_op["args"].append("gt")
        if tmp_op["op"].find("elewise_binary_scalar") != -1:
            if hasattr(op_stmt.body[0], 'a'):
                if isinstance(op_stmt.body[0].a, tvm.expr.Call):
                    if hasattr(op_stmt.body[0].b, 'a'):
                        if isinstance(op_stmt.body[0].b.a, tvm.expr.Call):
                            tmp_op["args"] = [op_stmt.body[0].b.b]
                        else:
                            tmp_op["args"] = [op_stmt.body[0].b.a]
                else:
                    if hasattr(op_stmt.body[0].a, 'a'):
                        if isinstance(op_stmt.body[0].a.a, tvm.expr.Call):
                            tmp_op["args"] = [op_stmt.body[0].a.b]
                        else:
                            tmp_op["args"] = [op_stmt.body[0].a.a]
        elif tmp_op["op"].find("broadcast") != -1:
            if tmp_op["op"] == "broadcast_for_tensor":
                if self._shape_to_list(tmp_op["src_buffer"][0].shape)[-1] != 1:
                    tmp_op["effective_op"] = False
            else:
                tmp_op["args"] = [op_stmt.body[0]]
        elif tmp_op["op"].find("reduce") != -1:
            if self._have_reduce:
                raise RuntimeError("Only support one time reduce")
            self._have_reduce = True
            tmp_op["reduce_axis"] = list(op_stmt.reduce_axis)

            reduce_axis_var = []
            for i in op_stmt.reduce_axis:
                reduce_axis_var.append(i.var)
            data_axis_var = op_stmt.body[0].source[0].args
            tmp_op["reduce_axis_num"] = []
            for ax_var in reduce_axis_var:
                axis_num = 0
                for i in data_axis_var:
                    if i.same_as(ax_var):
                        tmp_op["reduce_axis_num"].append(axis_num)
                    axis_num += 1

        if tmp_op["op"].find("elewise_single_VS_cond") != -1 \
                or tmp_op["op"].find("elewise_binary_cmp") != -1 \
                or tmp_op["op"].find("elewise_binary_cmpsel") != -1 \
                or tmp_op["op"].find("elewise_binary_logic") != -1:
            str_list = op_stmt.tag.split("|")
            tmp_op["op"] = str_list[0]
            tmp_op["args"] = []
            for i in range(1, len(str_list)):
                tmp_op["args"].append(str_list[i])

        # split inputs sign and add into args for elewise_multiple op
        elif tmp_op["op"].find("elewise_multiple") != -1:
            str_list = op_stmt.tag.split("|")
            tmp_op["op"] = str_list[0]
            if len(str_list) >= 2:
                same_list_str = str_list[1].split(',')
                tmp_op["args"] = same_list_str

        if tmp_op["op"].find("|") != -1:
            str_list = op_stmt.tag.split("|")
            tmp_op["op"] = str_list[0]

        return tmp_op



    def _is_broadcast_last_axis_tensor(self, tensor):
        if tensor.op.tag == "broadcast_for_tensor":
            # broadcast not last axis
            if list(tensor.op.input_tensors):
                original_tensor = tensor.op.input_tensors[0]
                original_shape = self._shape_to_list(original_tensor.shape)
                broadcast_shape = self._shape_to_list(tensor.shape)
                if original_shape[-1] == 1 and broadcast_shape[-1] != 1:
                    return True
                # include (1,1,1,1,1,1,1)->(10, 10, 5, 2, 3, 9,1)
                if sum(original_shape[:]) == len(original_shape):
                    return True
        return False

    def _is_broadcast_not_last_axis_tensor(self, tensor):
        """
        Check if the non-last axis broadcast scene

        Parameters:
        ----------
        tensors :  tensor to be checked

        Returns
        -------
        Bool: True or False
        """
        if tensor.op.tag == "broadcast_for_tensor":
            # broadcast not last axis
            if list(tensor.op.input_tensors):
                original_tensor = tensor.op.input_tensors[0]
                original_shape = self._shape_to_list(original_tensor.shape)
                broadcast_shape = self._shape_to_list(tensor.shape)
                if original_shape[-1] == broadcast_shape[-1]:
                    # not include (1,1,1,1,1,1,1)->(10, 10, 5, 2, 3, 9,1)
                    if sum(original_shape[:]) != len(original_shape):
                        return True
        return False

    # pylint: missing-final-newline
    def _find_max_broadcast_last_axis_offset(self):
        """
        Find the largest broadcast offset of the last axis broadcast

        Parameters:
        ----------
        tensor : input tensor

        Returns
        -------
        the largest broadcast offset
        """
        max_broadcast_axis_offset = 0
        if self._broadcast_last_axis_tensors:
            for broadcast_tensor in self._broadcast_last_axis_tensors:
                if list(broadcast_tensor.op.input_tensors):
                    original_tensor = broadcast_tensor.op.input_tensors[0]
                    original_shape = self._shape_to_list(original_tensor.shape)
                    broadcast_shape = \
                        self._shape_to_list(broadcast_tensor.shape)
                    broadcast_axis_offset = 0
                    for i in range(len(original_shape) - 1, -1, -1):
                        if original_shape[i] == 1 and \
                                original_shape[i] != broadcast_shape[i]:
                            broadcast_axis_offset += 1
                            continue
                        elif original_shape[i] == 1 and \
                                original_shape[i] == broadcast_shape[i]:
                            continue
                        else:
                            break
                    max_broadcast_axis_offset = max(max_broadcast_axis_offset,
                                                    broadcast_axis_offset)

        return max_broadcast_axis_offset
