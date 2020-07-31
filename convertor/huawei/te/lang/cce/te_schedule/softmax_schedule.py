"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

softmax schedule, provide a schedule for softmax
"""
# pylint: disable=too-many-lines
from __future__ import absolute_import
from te import tvm
from te import platform as cce
from te.platform import cce_emitinsn_params
from te.platform import cce_util
from te.platform import intrinsic_check_support
from .vector_schedule import VectorSchedule
from .util import shape_to_list
from .util import get_align_factor
from .util import get_shape_size_ext
from .util import get_bits_of
from .util import get_reduce_axis_num
from .util import tiling_from_front_to_back
from .util import tiling_from_back_to_front
from .util import get_max_divisor

# the byte of dtype
DTYPE_BYTE_WIDTH_MAP = {"float16": 2,
                        "float32": 4,
                        "int32": 4,
                        "int16": 2,
                        "uint16": 2,
                        "int8": 1,
                        "uint8": 1,
                        "bool": 1}


# pylint: disable=abstract-method, too-many-instance-attributes, too-few-public-methods
class SoftmaxSchedule(VectorSchedule):
    """class of cce index op

    Parameters
    ----------
    scope : cal buffer like local.UB

    Returns
    -------
    CceOp_instance : instance of CceOp

    """
    # pylint: disable=super-init-not-called
    def __init__(self):
        self._scope = cce.scope_ubuf
        self._core_dim = cce.get_soc_spec("CORE_NUM")
        if self._scope.lower().find('.ub') != -1:
            self._total_size = cce.get_soc_spec("UB_SIZE")
        else:
            raise RuntimeError("only support UB buffer now")
        self._ub_tiling_max_size = self._total_size
        self._max_ub_count = 0
        self._schedule = None
        self._spec_node_list = []
        self._spec_mid_list = []
        self._tiling_para = {"block_tiling": {"axis": 0, "factor": 1, "nparts": 1},
                             "ub_tiling": {"axis": 0, "factor": 1, "nparts": 1}}
        self._tiling_result = {"block_tiling": {"axis": 0,
                                                "parent_itervar": None,
                                                "outer_itervar": None,
                                                "inner_itervar": None},
                               "ub_tiling": {"axis": 0,
                                             "parent_itervar": None,
                                             "outer_itervar": None,
                                             "inner_itervar": None}}

        self._cache_read_tensors_and_readers_map = {}
        self._cache_read_buffer_and_reader_map = {}
        self._cache_read_tensors_and_buffer_map = {}
        self._cache_write_tensors = []
        self._cache_write_tensors_and_input_map = {}
        self._cache_write_tensors_and_buffer_map = {}
        self._reduce_tensors = []
        self._broadcast_tensors = []
        self._after_reduce_bc_tensors = []
        self._compute_inline_buffer = []
        self._compute_at_map = {}
        self._compute_at_tensor_map = {}
        self._emit_insn_map = {}
        self._res = None
        self._is_last_reduce_axis = None
        self._is_32byte_align = None
        self._reduce_axis_num = None
        self._is_this_schedule_support = True
        self._origin_op = []
        self._double_buffer_tensors = []
        self._is_double_buffer = False
        self._is_multi_core = False
        self._is_block_ub_one_axis = False
        self._is_block_fuse = False
        self._is_reduce_last_axis_enhance_insn = False

    # pylint: disable=arguments-differ
    def do_schedule(self, out_tensors):
        """
        auto_schedule for cce AI-CORE

        Parameters
        ----------
        outTensors : the out tvm.tensor

        sch : schedule, the computation schedule for the op

        Returns
        -------
        Bool, now is true

        """
        self._construct_compute_graph(out_tensors)
        shape = shape_to_list(self._res.shape)
        align_factor_32byte, _ = get_align_factor(self._res.dtype)

        is_last_with_workspace = \
            self._is_last_reduce_axis and \
            self._max_ub_count < shape[self._reduce_axis_num]

        if is_last_with_workspace:
            # log_softmax_grad float16, data_input reader only cast_compute
            # not support
            self._is_this_schedule_support = False
            for i in self._cache_read_tensors_and_readers_map:
                input_readers_len = \
                    len(self._cache_read_tensors_and_readers_map[i])
                if input_readers_len > 1:
                    self._is_this_schedule_support = True
                    break

            if self._is_this_schedule_support:
                sch = self._do_schedule_last_with_workspace()
                return sch, self._spec_node_list

        if not self._is_this_schedule_support:
            return None, []

        if not self._is_last_reduce_axis:
            sch = self._do_schedule_nlst_axis()
        else:
            cce_emitinsn_params.cceEmitParamsIns.clear_param()
            cce_emitinsn_params.cceEmitParamsIns.insert_param("broadcast_axis_offset", 1)
            if len(shape) == 1:
                sch = self._do_schedule_last_dim1()
            elif self._is_32byte_align:
                sch = self._do_schedule_last_32align()
            elif shape[self._reduce_axis_num] < align_factor_32byte:
                self._is_reduce_last_axis_enhance_insn = True
                sch = self._do_schedule_last_noalign_lt32()
            elif shape[self._reduce_axis_num] < align_factor_32byte*2:
                self._is_reduce_last_axis_enhance_insn = True
                sch = self._do_schedule_last_noalign_ge32()
            else:
                sch = self._do_schedule_last_noalign_storagealign()

        return sch, []


    def _do_schedule_last_with_workspace(self):
        self._schedule = tvm.create_schedule(self._res.op)

        self._get_softmax_spec_node()
        cce_emitinsn_params.cceEmitParamsIns.clear_param()
        cce_emitinsn_params.cceEmitParamsIns.insert_param(
            "broadcast_axis_offset", 1)

        self._do_cache_rw_with_workspace()
        self._do_compute_inline_with_workspace()
        self._do_storage_align_last_with_workspace()

        self._calculate_compute_at_with_workspace()
        self._do_tiling_last_with_workspace()
        self._do_compute_at()
        self._do_emit_insn_last_with_workspace()

        self._do_double_buffer()
        sch = self._schedule
        return sch


    def _do_schedule_last_32align(self):
        self._schedule = tvm.create_schedule(self._res.op)
        self._do_cache_read()
        self._do_cache_write()

        self._calculate_compute_inline()
        self._do_compute_inline()

        self._do_tiling_last_32align()

        self._calculate_compute_at()
        self._do_compute_at()

        self._calculate_emit_insn_last()
        self._do_emit_insn()
        self._do_double_buffer()

        sch = self._schedule
        return sch


    def _do_schedule_last_noalign_storagealign(self):
        self._schedule = tvm.create_schedule(self._res.op)
        self._do_cache_read()
        self._do_cache_write()

        self._calculate_compute_inline()
        self._do_compute_inline()

        self._do_storage_align_last()
        self._do_tiling_last_noalign_storagealign()

        self._calculate_compute_at()
        self._do_compute_at()

        self._calculate_emit_insn_last()
        self._do_emit_insn()
        self._do_double_buffer()

        sch = self._schedule
        return sch


    def _do_schedule_last_dim1(self):
        self._schedule = tvm.create_schedule(self._res.op)
        self._do_cache_read()
        self._do_cache_write()

        self._calculate_compute_inline()
        self._do_compute_inline()

        self._do_tiling_last_dim1()

        self._calculate_compute_at()
        self._do_compute_at()

        self._calculate_emit_insn_last()
        self._do_emit_insn()
        self._do_double_buffer()

        sch = self._schedule
        return sch


    def _do_schedule_last_noalign_lt32(self):
        self._schedule = tvm.create_schedule(self._res.op)
        self._do_cache_read()
        self._do_cache_write()

        self._calculate_compute_inline()
        self._do_compute_inline()

        self._do_tiling_last_noalign_lt32()

        self._calculate_compute_at()
        self._do_compute_at()

        self._calculate_emit_insn_last()
        self._do_emit_insn()
        self._do_double_buffer()

        sch = self._schedule
        return sch


    def _do_schedule_last_noalign_ge32(self):
        self._schedule = tvm.create_schedule(self._res.op)
        self._do_cache_read()
        self._do_cache_write()

        self._calculate_compute_inline()
        self._do_compute_inline()

        self._do_tiling_last_noalign_ge32()

        self._calculate_compute_at()
        self._do_compute_at()

        self._calculate_emit_insn_last()
        self._do_emit_insn()
        self._do_double_buffer()

        sch = self._schedule
        return sch


    def _do_schedule_nlst_axis(self):
        self._schedule = tvm.create_schedule(self._res.op)

        self._do_cache_read()
        self._do_cache_write()

        self._calculate_compute_inline()
        self._do_compute_inline()

        self._calculate_tiling_nlst_axis()
        if not self._is_this_schedule_support:
            return None

        self._do_tiling()

        self._calculate_multi_core()
        self._do_multi_core()

        self._do_reorder()

        self._calculate_compute_at()
        self._do_compute_at()

        self._calculate_emit_insn_nlst()
        self._do_emit_insn()
        self._do_storage_align_nlst()

        self._do_double_buffer()

        sch = self._schedule

        return sch


    def _construct_compute_graph(self, out_tensors):
        """
        record relate context imformations of operations

        outTensors only support form like: out_1->..out_2->..out_n

        """
        # 1. find the last out tensor
        if hasattr(out_tensors, 'index'):
            if len(out_tensors) > 1:
                self._is_this_schedule_support = False
                return False  # not mutile output
            self._res = out_tensors[0]
        else:
            self._res = out_tensors

        # 2. traverse tensors of subgraph by Depth-First-Search
        tensor_list = []
        visited_list = []
        self.__gen_reversed_subgraph_list(self._res, tensor_list, visited_list)

        self._cache_write_tensors = list(reversed(tensor_list))
        for tensor in self._cache_write_tensors:
            tmp_op = self.__get_tensor_op(tensor)
            self._origin_op.append(tmp_op)

        # 3. check if the subgraph matches this schedule template
        shape = shape_to_list(self._res.shape)
        for i in self._broadcast_tensors:
            if shape_to_list(i.shape) != shape:
                self._is_this_schedule_support = False
                return False # broadcast must be as same as the shape of elewise res

        reduce_axis_num = get_reduce_axis_num(self._reduce_tensors[0])
        for i in self._reduce_tensors:
            if get_reduce_axis_num(i) != reduce_axis_num:
                self._is_this_schedule_support = False
                return False # all reduce must have the same reduce_axis

        if len(reduce_axis_num) != 1:
            self._is_this_schedule_support = False
            return False # must be only one reduce_axis

        self._is_last_reduce_axis = ((len(shape) - 1) in reduce_axis_num)
        self._reduce_axis_num = reduce_axis_num[0]

        align_factor_32byte, _ = \
            self._get_data_alignment(self._cache_write_tensors)
        self._is_32byte_align = (shape[-1] % align_factor_32byte == 0)

        self._get_max_ub_count_and_try_double_buffer()

        return True

    def _get_max_ub_count_and_try_double_buffer(self):
        shape = shape_to_list(self._res.shape)
        align_factor_32byte, _ = \
            self._get_data_alignment(self._cache_write_tensors)

        # calculate max_ub_count
        reduce_rows = 1 if self._is_last_reduce_axis else shape[self._reduce_axis_num]
        self._max_ub_count = self._get_max_ub_count(self._total_size,
                                                    align_factor_32byte,
                                                    reduce_rows)
        if (self._max_ub_count < align_factor_32byte) or \
           (self._is_last_reduce_axis and self._max_ub_count < shape[self._reduce_axis_num]):
            self._is_this_schedule_support = False
            return False # workspace

        # calculate max_ub_count: try double buffer
        self._is_double_buffer = False
        tmp_max_ub_count = self._get_max_ub_count(self._total_size // 2,
                                                  align_factor_32byte,
                                                  reduce_rows)
        if (tmp_max_ub_count < align_factor_32byte) or \
           (self._is_last_reduce_axis and tmp_max_ub_count < shape[self._reduce_axis_num]):
            self._is_double_buffer = False
        else:
            self._is_double_buffer = True
            self._total_size = self._total_size // 2
            self._max_ub_count = tmp_max_ub_count

        return True


    def __gen_reversed_subgraph_list(self, tensor, tensor_list, visited_list):
        """traverse tensors by Depth-First-Search

        Parameters
        ----------
        tensor : tensor
            traverse tensors from this tensor,
            traversing its input tensors recursively.

        tensor_list : list
            record tensors in the order of Depth-First-Search.

        visited_list : list
            record tensors which has been visited.
        """
        search_tensor_list = list(tensor.op.input_tensors)
        search_tensor_list.append(tensor)
        for in_tensor in search_tensor_list:
            # Mark special nodes, handle them separately
            if self._is_reduce_ops(in_tensor):
                if in_tensor not in self._reduce_tensors:
                    self._reduce_tensors.append(in_tensor)
            elif self._is_broadcast_ops(in_tensor):
                if in_tensor not in self._broadcast_tensors:
                    self._broadcast_tensors.append(in_tensor)
            elif self._is_after_reduce_bc_ops(in_tensor):
                if in_tensor not in self._after_reduce_bc_tensors:
                    self._after_reduce_bc_tensors.append(in_tensor)

            if in_tensor in self._spec_node_list or \
               isinstance(in_tensor.op, tvm.tensor.PlaceholderOp):
                if not in_tensor.same_as(tensor):
                    self._map_apend(self._cache_write_tensors_and_input_map, tensor, in_tensor)
                    self._map_apend(self._cache_read_tensors_and_readers_map, in_tensor, tensor)
                continue
            else:
                if not in_tensor.same_as(tensor):
                    self._map_apend(self._cache_write_tensors_and_input_map, tensor, in_tensor)

            if in_tensor in visited_list:
                continue

            visited_list.append(in_tensor)
            self.__gen_reversed_subgraph_list(in_tensor, tensor_list, visited_list)
            tensor_list.append(in_tensor)

    # pylint: disable=no-self-use
    def __get_tensor_op(self, tensor):
        """
        Split the tensor and construct map

        Parameters:
        ----------
        None

        Returns
        -------
        Dict: construct map
        """
        tmp_op = {"op": None,
                  "dst_buffer": [],
                  "src_buffer": [],
                  "args": [],
                  "effective_op": True}
        str_list = tensor.op.tag.split("|")
        tmp_op["op"] = str_list[0]
        tmp_op["dst_buffer"] = tensor
        tmp_op["src_buffer"] = list(tensor.op.input_tensors)

        return tmp_op

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

    # pylint: disable=no-self-use
    def _is_reduce_ops(self, tensor):
        """
        reduce_ops
        """
        if tensor.op.tag.find("reduce") != -1:
            return True
        return False

    # pylint: disable=no-self-use
    def _is_broadcast_ops(self, tensor):
        """
        broadcast
        """
        if tensor.op.tag.find("broadcast") != -1: # broadcast_for_tensor
            return True
        return False

    def _is_after_reduce_bc_ops(self, tensor):
        """
        broadcast
        """
        if self._is_reduce_ops(tensor) or self._is_broadcast_ops(tensor):
            return False
        for in_tensor in list(tensor.op.input_tensors):
            if self._is_reduce_ops(in_tensor) or self._is_broadcast_ops(in_tensor):
                return True
        return False


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

    def _calculate_compute_inline(self):
        """
        Calculate the tensor that needs compute inline

        Parameters:
        ----------
        None

        Returns
        -------
        None
        """
        # reduce_axis is not last axis, broadcast could compute_inline
        if not self._is_last_reduce_axis:
            for i in self._broadcast_tensors:
                write_buffer = self._cache_write_tensors_and_buffer_map[i]
                self._compute_inline_buffer.append(write_buffer)

    def _do_compute_inline(self):
        """
        compute inline operations
        """
        for i in self._cache_write_tensors[1:]:
            self._schedule[i].compute_inline()
        for i in self._compute_inline_buffer:
            self._schedule[i].compute_inline()


    def _get_total_width(self):
        """
        caculate the max useable number based on op liveness
        return: max useable number.
            example: Three float16 nodes, return 2B*3 = 6B
                     Three float32 nodes, return 4B*3 = 12B
        """
        def _op_width(op_node):
            num_type = op_node.dtype
            if num_type.lower() not in DTYPE_BYTE_WIDTH_MAP.keys():
                raise RuntimeError("Can not calculate with no compute")

            tmp_width = 0
            if op_node.op.tag is not None:
                tag = op_node.op.tag
                # logic use 4 fp16 temp buffer
                if tag.find("logic") != -1:
                    tmp_width = 4 * DTYPE_BYTE_WIDTH_MAP["float16"]
                # cond use 3 fp16 temp buffer
                elif tag.find("cond") != -1:
                    tmp_width = 3 * DTYPE_BYTE_WIDTH_MAP["float16"]
                # vsel use 3 fp16 temp buffer
                elif tag.find("sel") != -1:
                    tmp_width = 3 * DTYPE_BYTE_WIDTH_MAP["float16"]
                # vcompare use 2 temp buffer
                elif tag.find("compare") != -1:
                    tmp_width = 2 * DTYPE_BYTE_WIDTH_MAP[num_type.lower()]
                # vcomsel use 3 temp buffer
                elif tag.find("cmpsel") != -1:
                    tmp_width = 3 * DTYPE_BYTE_WIDTH_MAP[num_type.lower()]

            return DTYPE_BYTE_WIDTH_MAP[num_type.lower()] + tmp_width

        def _post_dfs_order(op_node, op_graph, visited, post_order):
            if op_node in visited:
                return
            visited[op_node] = True
            post_order.append(op_node)
            if op_node in op_graph:
                for src in op_graph[op_node]:
                    _post_dfs_order(src, op_graph, visited, post_order)

        tensors_and_input_map = {}
        for op_node in self._origin_op:
            src_op = list(op_node['src_buffer'])
            src_op.reverse()
            tensors_and_input_map[op_node['dst_buffer']] = src_op

        visited = {}
        post_order = []
        _post_dfs_order(self._res, tensors_and_input_map, visited, post_order)

        lives = [self._res]
        visited = {lives[0]: True}
        live_width = _op_width(lives[0])
        max_width = live_width
        for op_node in post_order:
            if op_node in tensors_and_input_map:
                for src in tensors_and_input_map[op_node]:
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

    def _get_max_ub_count(self, total_size, align_factor=32, reduce_rows=1):
        """
        caculate the max element num loaded in UB buffer
        :return: max element num loaded in UB buffer
        """
        total_width = self._get_total_width()
        if self._is_last_reduce_axis:
            total_width = total_width + DTYPE_BYTE_WIDTH_MAP["float32"]

        if not total_width:
            raise RuntimeError("Can not calculate with no compute")

        # example: Three float16 nodes, the number of reduce rows is 5
        # max_ub_count = 248KB // (2B*3*5) = 248KB // (2B*3) // 5
        max_ub_count = total_size // total_width // reduce_rows

        max_ub_count = max_ub_count // align_factor * align_factor

        return max_ub_count

    # pylint: disable=no-self-use
    def _get_data_alignment(self, tensor_list):
        """calculate the unified buffer data alignment
        1. enable multi-core needs to be larger than 32B,
        2. vector calculation should be 256B aligned as much as possible
        fp16: align_factor_32byte=16, align_factor_256byte=128
        fp32: align_factor_32byte=8,  align_factor_256byte=64

        therefore, using the smallest dtype to make a tiling
        """
        bits = 65535
        for i in tensor_list:
            bits_update = get_bits_of(i.dtype.lower())
            if bits > bits_update:
                bits = bits_update

        align_factor_32byte = 32*8//bits
        align_factor_256byte = 256*8//bits
        return align_factor_32byte, align_factor_256byte


    def _calculate_tiling_last_common(self):
        """
        calculate tiling strategy: last axis
        """
        # Tiling strategy: 1) 128 alignment reduction set_mask (not use)
        # 2) Enable 32 cores > 3) Maximize UB utilization
        shape = shape_to_list(self._res.shape)
        align_factor_32byte, align_factor_256byte = \
            self._get_data_alignment(self._cache_write_tensors)

        block_nparts, block_factor, block_axis = \
            tiling_from_front_to_back(shape[:self._reduce_axis_num],
                                      self._core_dim,
                                      self._core_dim,
                                      is_divisible=False)

        ub_nparts, ub_factor, ub_axis = \
            tiling_from_back_to_front(shape, self._max_ub_count,
                                      align_factor=align_factor_256byte,
                                      is_divisible=False)

        self._is_block_ub_one_axis = (block_axis >= ub_axis)
        self._is_block_fuse = (block_axis not in (0,))

        # like (7,2,42767,2,16) scene, if the block tiling split the 3 axis,
        # because 42767 is a prime number, then ub tiling should be divisible,
        # then lead to the dma copy is less efficient, so adjust the block tiling

        # 1.block tiling adjust, be divisible
        if self._is_block_ub_one_axis or self._is_block_fuse:
            block_nparts, block_factor, block_axis = \
                tiling_from_front_to_back(shape[:self._reduce_axis_num],
                                          self._core_dim,
                                          self._core_dim,
                                          is_divisible=True)
        self._is_block_ub_one_axis = (block_axis >= ub_axis)
        self._is_block_fuse = (block_axis not in (0,))

        # 2.ub tiling adjust
        if self._is_block_ub_one_axis:
            shape_adjust = shape_to_list(self._res.shape)
            shape_adjust[block_axis] = block_factor
            ub_nparts, ub_factor, ub_axis = \
                tiling_from_back_to_front(shape_adjust[block_axis:], self._max_ub_count,
                                          align_factor=align_factor_32byte,
                                          is_divisible=False)
            ub_axis = block_axis + ub_axis

        if block_axis > ub_axis:
            raise RuntimeError("block and ub tiling error: block_axis > ub_axis")
        self._is_block_ub_one_axis = (block_axis == ub_axis)

        self._tiling_para["block_tiling"]["nparts"] = block_nparts
        self._tiling_para["block_tiling"]["factor"] = block_factor
        self._tiling_para["block_tiling"]["axis"] = block_axis

        self._tiling_para["ub_tiling"]["nparts"] = ub_nparts
        self._tiling_para["ub_tiling"]["factor"] = ub_factor
        self._tiling_para["ub_tiling"]["axis"] = ub_axis


    # pylint: disable=too-many-locals
    def _calculate_last_axis_tiling(self):
        """
        calculate tiling strategy: last axis
        """
        shape = shape_to_list(self._res.shape)
        align_factor_32byte, align_factor_256byte = \
            self._get_data_alignment(self._cache_write_tensors)

        block_nparts, block_factor, block_axis = \
            calculate_last_axis_block_tiling(shape, self._reduce_axis_num,
                                             block_tiling_min_size=align_factor_32byte,
                                             core_num=self._core_dim,
                                             is_divisible=False)

        ub_nparts, ub_factor, ub_axis = \
            calculate_last_axis_ub_tiling(shape, self._reduce_axis_num,
                                          (block_nparts, block_factor, block_axis),
                                          self._max_ub_count,
                                          align_factor=align_factor_256byte,
                                          is_divisible=False)

        self._is_block_ub_one_axis = (block_axis == ub_axis)
        self._is_block_fuse = (block_axis not in (0, self._reduce_axis_num + 1))

        if self._is_block_ub_one_axis or self._is_block_fuse:
            block_nparts, block_factor, block_axis = \
                calculate_last_axis_block_tiling(shape, self._reduce_axis_num,
                                                 block_tiling_min_size=align_factor_32byte,
                                                 core_num=self._core_dim,
                                                 is_divisible=True)

        self._is_block_ub_one_axis = (block_axis == ub_axis)
        self._is_block_fuse = (block_axis not in (0, self._reduce_axis_num + 1))

        if self._is_block_ub_one_axis:
            ub_nparts, ub_factor, ub_axis = \
                calculate_last_axis_ub_tiling(shape, self._reduce_axis_num,
                                              (block_nparts, block_factor, block_axis),
                                              self._max_ub_count,
                                              align_factor=align_factor_32byte,
                                              is_divisible=True)

        self._is_block_ub_one_axis = (block_axis == ub_axis)
        if self._is_block_ub_one_axis and shape[ub_axis] != (block_nparts * ub_nparts * ub_factor):
            raise RuntimeError("block and ub tiling the same axis, must be divisible split")

        # 3. multi-core should >=32B
        self._is_multi_core = self._is_block_fuse or (block_nparts > 1)
        remaining_size = get_shape_size_ext(shape[ub_axis+1:], ub_factor)
        tail_size = get_shape_size_ext(shape[ub_axis+1:], shape[ub_axis] % ub_factor)
        align_factor_out, _ = get_align_factor(self._res.dtype)
        is_ub_tiling_less_32byte = (remaining_size < align_factor_32byte)
        is_tail_less_32byte = (tail_size != 0) and (tail_size < align_factor_out)
        if self._is_multi_core and (is_ub_tiling_less_32byte or is_tail_less_32byte):
            self._is_this_schedule_support = False
            return

        # 4. sigle-core ub_tiling is too little processing performance reduced,
        # therefore not optimized
        if not self._is_multi_core and is_ub_tiling_less_32byte:
            self._is_this_schedule_support = False
            return

        self._tiling_para["block_tiling"]["nparts"] = block_nparts
        self._tiling_para["block_tiling"]["factor"] = block_factor
        self._tiling_para["block_tiling"]["axis"] = block_axis

        self._tiling_para["ub_tiling"]["nparts"] = ub_nparts
        self._tiling_para["ub_tiling"]["factor"] = ub_factor
        self._tiling_para["ub_tiling"]["axis"] = ub_axis

    # pylint: disable=too-many-locals
    def _calculate_tiling_nlst_axis(self):
        """
        calculate tiling strategy: not last axis
        """
        shape = shape_to_list(self._res.shape)
        align_factor_32byte, align_factor_256byte = \
            self._get_data_alignment(self._cache_write_tensors)

        block_nparts, block_factor, block_axis = \
            calculate_nlst_axis_block_tiling(shape, self._reduce_axis_num,
                                             block_tiling_min_size=align_factor_32byte,
                                             core_num=self._core_dim,
                                             is_divisible=False)

        ub_nparts, ub_factor, ub_axis = \
            calculate_nlst_axis_ub_tiling(shape, self._reduce_axis_num,
                                          (block_nparts, block_factor, block_axis),
                                          self._max_ub_count,
                                          align_factor=align_factor_256byte,
                                          is_divisible=False)

        self._is_block_ub_one_axis = (block_axis == ub_axis)
        self._is_block_fuse = (block_axis not in (0, self._reduce_axis_num + 1))

        # 1. divisible block_tiling
        if self._is_block_fuse:
            block_nparts, block_factor, block_axis = \
                calculate_nlst_axis_block_tiling(shape, self._reduce_axis_num,
                                                 block_tiling_min_size=align_factor_32byte,
                                                 core_num=self._core_dim,
                                                 is_divisible=True)
            self._is_block_ub_one_axis = (block_axis == ub_axis)
            self._is_block_fuse = (block_axis not in (0, self._reduce_axis_num + 1))

        # 2. divisible ub_tiling
        if self._is_block_ub_one_axis:
            ub_nparts, ub_factor, ub_axis = \
                calculate_nlst_axis_ub_tiling(shape, self._reduce_axis_num,
                                              (0, 0, self._reduce_axis_num),
                                              self._max_ub_count,
                                              align_factor=align_factor_32byte,
                                              is_divisible=True)

            shape_bckup = shape[0:ub_axis]
            shape_bckup.append(ub_nparts)
            block_nparts, block_factor, block_axis = \
                calculate_nlst_axis_block_tiling(shape_bckup, self._reduce_axis_num,
                                                 block_tiling_min_size=1,
                                                 core_num=self._core_dim,
                                                 is_divisible=True)

            self._is_block_ub_one_axis = (block_axis == ub_axis)
            self._is_block_fuse = (block_axis not in (0, self._reduce_axis_num + 1))
            if self._is_block_ub_one_axis and \
               shape[ub_axis] != (block_nparts * block_factor * ub_factor):
                raise RuntimeError("block and ub tiling the same axis, must be divisible split")

        # 3. multi-core should >=32B
        self._is_multi_core = self._is_block_fuse or (block_nparts > 1)
        remaining_size = get_shape_size_ext(shape[ub_axis+1:], ub_factor)
        tail_size = get_shape_size_ext(shape[ub_axis+1:], shape[ub_axis] % ub_factor)
        align_factor_out, _ = get_align_factor(self._res.dtype)
        is_ub_tiling_less_32byte = (remaining_size < align_factor_32byte)
        is_tail_less_32byte = (tail_size != 0) and (tail_size < align_factor_out)
        if self._is_multi_core and (is_ub_tiling_less_32byte or is_tail_less_32byte):
            self._is_this_schedule_support = False
            return

        # 4. sigle-core ub_tiling is too little processing performance reduced,
        # therefore not optimized
        if not self._is_multi_core and is_ub_tiling_less_32byte:
            self._is_this_schedule_support = False
            return

        self._tiling_para["block_tiling"]["nparts"] = block_nparts
        self._tiling_para["block_tiling"]["factor"] = block_factor
        self._tiling_para["block_tiling"]["axis"] = block_axis

        self._tiling_para["ub_tiling"]["nparts"] = ub_nparts
        self._tiling_para["ub_tiling"]["factor"] = ub_factor
        self._tiling_para["ub_tiling"]["axis"] = ub_axis

    def _do_tiling(self):
        block_nparts = self._tiling_para["block_tiling"]["nparts"]
        block_axis = self._tiling_para["block_tiling"]["axis"]

        ub_factor = self._tiling_para["ub_tiling"]["factor"]
        ub_axis = self._tiling_para["ub_tiling"]["axis"]

        block_outer, block_inner = self._schedule[self._res].split(
            self._res.op.axis[block_axis], nparts=block_nparts)

        if self._is_block_ub_one_axis:
            block_inner, tobe_split = self._schedule[self._res].split(block_inner, nparts=1)
            ub_outer, ub_inner = self._schedule[self._res].split(tobe_split, factor=ub_factor)
        else:
            ub_outer, ub_inner = self._schedule[self._res].split(self._res.op.axis[ub_axis],
                                                                 factor=ub_factor)

        block_tiling_result = {"axis": block_axis,
                               "outer_itervar": block_outer,
                               "inner_itervar": block_inner}
        ub_tiling_result = {"axis": ub_axis,
                            "outer_itervar": ub_outer,
                            "inner_itervar": ub_inner}
        self._tiling_result = {"block_tiling": block_tiling_result,
                               "ub_tiling": ub_tiling_result}

    def _calculate_multi_core(self):
        block_axis = self._tiling_result["block_tiling"]["axis"]
        block_outer = self._tiling_result["block_tiling"]["outer_itervar"]

        if not self._is_block_fuse:
            self._multi_core_fused_axis = block_outer
        else:
            if block_axis > self._reduce_axis_num:
                fuse_from_axis = self._reduce_axis_num + 1
            else:
                fuse_from_axis = 0

            fuse_list = []
            for i in range(fuse_from_axis, block_axis):
                fuse_list.append(self._res.op.axis[i])
            fuse_list.append(block_outer)
            self._multi_core_fused_axis = self._schedule[self._res].fuse(*fuse_list)

    def _do_multi_core(self):
        block = tvm.thread_axis("blockIdx.x")
        self._schedule[self._res].bind(self._multi_core_fused_axis, block)


    def _do_reorder(self):
        if self._is_last_reduce_axis:
            return

        ub_axis = self._tiling_result["ub_tiling"]["axis"]
        ub_outer = self._tiling_result["ub_tiling"]["outer_itervar"]
        ub_inner = self._tiling_result["ub_tiling"]["inner_itervar"]

        block_axis = self._tiling_result["block_tiling"]["axis"]
        block_inner = self._tiling_result["block_tiling"]["inner_itervar"]
        # 1. res tensor reorder
        # block_outer,block_inner,ub_outer,reduce_axis,ub_inner
        reorder_list = []
        if block_axis > self._reduce_axis_num:
            # (reduce~block], reorder multi_core first
            reorder_list.append(self._multi_core_fused_axis)
            reorder_list.append(block_inner)

            # [0~reduce)
            for i in range(0, self._reduce_axis_num):
                reorder_list.append(self._res.op.axis[i])
            # (reduce~block] ==split==> (self._multi_core_fused_axis, block_inner)
            # (block_axis~ub_axis)
            for i in range(block_axis+1, ub_axis):
                reorder_list.append(self._res.op.axis[i])
            # ub_split, reduce axis
            reorder_list.append(ub_outer)
            reorder_list.append(self._res.op.axis[self._reduce_axis_num])
            reorder_list.append(ub_inner)
        else:
            # [0~block]
            reorder_list.append(self._multi_core_fused_axis)
            reorder_list.append(block_inner)
            # (block_axis~reduce~ub_axis)
            for i in range(block_axis+1, ub_axis):
                if i != self._reduce_axis_num:
                    reorder_list.append(self._res.op.axis[i])
            # ub_split, reduce axis
            reorder_list.append(ub_outer)
            reorder_list.append(self._res.op.axis[self._reduce_axis_num])
            reorder_list.append(ub_inner)

        self._schedule[self._res].reorder(*reorder_list)

        # 2. reduce_tensor reorder
        ub_factor = self._tiling_para["ub_tiling"]["factor"]
        for i in self._reduce_tensors:
            write_buffer = self._cache_write_tensors_and_buffer_map[i]
            outer, inner = self._schedule[write_buffer].split(
                write_buffer.op.axis[ub_axis],
                factor=ub_factor)
            self._schedule[write_buffer].reorder(outer, write_buffer.op.reduce_axis[0], inner)

    def _do_storage_align_nlst(self):
        if self._is_last_reduce_axis:
            return

        at_axis = self._reduce_axis_num

        for i in self._cache_read_tensors_and_buffer_map:
            read_buffer = self._cache_read_tensors_and_buffer_map[i]
            align_factor, _ = get_align_factor(read_buffer.dtype)
            self._schedule[read_buffer].storage_align(read_buffer.op.axis[at_axis], align_factor, 0)

        for i in self._cache_write_tensors_and_buffer_map:
            write_buffer = self._cache_write_tensors_and_buffer_map[i]
            align_factor, _ = get_align_factor(write_buffer.dtype)
            if write_buffer not in self._compute_inline_buffer:
                self._schedule[write_buffer].storage_align(write_buffer.op.axis[at_axis],
                                                           align_factor, 0)

    def _do_storage_align_last(self):
        if not self._is_last_reduce_axis:
            return

        at_axis = self._reduce_axis_num - 1  # axis: -2

        for i in self._cache_read_tensors_and_buffer_map:
            read_buffer = self._cache_read_tensors_and_buffer_map[i]
            align_factor, _ = get_align_factor(read_buffer.dtype)
            self._schedule[read_buffer].storage_align(read_buffer.op.axis[at_axis], align_factor, 0)

        for i in self._cache_write_tensors_and_buffer_map:
            write_buffer = self._cache_write_tensors_and_buffer_map[i]
            align_factor, _ = get_align_factor(write_buffer.dtype)
            if write_buffer not in self._compute_inline_buffer:
                self._schedule[write_buffer].storage_align(write_buffer.op.axis[at_axis],
                                                           align_factor, 0)


    def _do_tiling_last_noalign_lt32(self):
        # Tiling strategy: 1) 128 alignment reduction set_mask (not use)
        # 2) Enable 32 cores > 3) Maximize UB utilization
        self._calculate_tiling_last_common()
        ub_factor = self._tiling_para["ub_tiling"]["factor"]
        ub_axis = self._tiling_para["ub_tiling"]["axis"]
        shape = shape_to_list(self._res.shape)
        align_factor_32byte, _ = \
            self._get_data_alignment(self._cache_write_tensors)

        # multi-core should >=32B
        if get_shape_size_ext(shape[ub_axis+1:], ub_factor) >= align_factor_32byte:
            self._do_tiling()
            self._calculate_multi_core()
            self._do_multi_core()

        else:
            # Assume that the amount of data entering this branch
            # is already very small. After 32 cores are allocated,
            # ub> 32B is not satisfied. Therefore,
            # you can directly fuse the previous ones,
            # and then divide the factor into UB tiling.
            while ub_axis > 0:
                tmp_remain = get_shape_size_ext(shape[ub_axis:])
                if tmp_remain >= align_factor_32byte:
                    break
                ub_axis = ub_axis - 1

            ub_factor = shape[ub_axis]
            tmp_factor = ub_factor
            while tmp_factor > 1:
                if get_shape_size_ext(shape[ub_axis+1:], tmp_factor) < align_factor_32byte:
                    break
                ub_factor = tmp_factor
                tmp_factor = get_max_divisor(shape[ub_axis], tmp_factor-1)

            # ub tiling
            block_fuse, ub_inner = \
                self._schedule[self._res].split(self._res.op.axis[ub_axis],
                                                factor=ub_factor)
            block_fuse, ub_outer = \
                self._schedule[self._res].split(block_fuse, factor=1)

            # multi core
            fuse_list = []
            for i in range(0, ub_axis):
                fuse_list.append(self._res.op.axis[i])
            fuse_list.append(block_fuse)
            block_outer = self._schedule[self._res].fuse(*fuse_list)

            self._multi_core_fused_axis = block_outer
            block = tvm.thread_axis("blockIdx.x")
            self._schedule[self._res].bind(self._multi_core_fused_axis, block)

            block_tiling_result = {"outer_itervar": block_outer,
                                   "inner_itervar": None}
            ub_tiling_result = {"axis": ub_axis,
                                "outer_itervar": ub_outer,
                                "inner_itervar": ub_inner}
            self._tiling_result = {"block_tiling": block_tiling_result,
                                   "ub_tiling": ub_tiling_result}

    def _do_tiling_last_noalign_ge32(self):
        # Prerequisites:
        # 1. Eliminate Worskspace, and certainly will not cut the reduce axis.
        # 2. Definitely greater than 32B, multi-core could enabled
        # multi-core, already >=32B
        self._calculate_tiling_last_common()
        self._do_tiling()
        self._calculate_multi_core()
        self._do_multi_core()


    def _do_tiling_last_32align(self):
        # Prerequisites:
        # 1. Eliminate Worskspace, and certainly will not cut the reduce axis.
        # 2. Definitely greater than 32B, multi-core could enabled
        shape = shape_to_list(self._res.shape)
        remaining_shape_size = get_shape_size_ext(shape[:self._reduce_axis_num])
        remaining_ub_count = self._max_ub_count // shape[self._reduce_axis_num]
        if remaining_shape_size <= remaining_ub_count:
            fuse_list = []
            for i in range(0, self._reduce_axis_num):
                fuse_list.append(self._res.op.axis[i])
            tmp_fuse_axis = self._schedule[self._res].fuse(*fuse_list)

            # Tiling strategy: 1. Enable 32 cores
            # block tiling
            block_outer, block_inner = self._schedule[self._res].split(
                tmp_fuse_axis, nparts=self._core_dim)
            # ub tiling
            ub_outer, ub_inner = self._schedule[self._res].split(
                self._res.op.axis[self._reduce_axis_num], nparts=1)
            # multi core
            self._multi_core_fused_axis = block_outer
            block = tvm.thread_axis("blockIdx.x")
            self._schedule[self._res].bind(self._multi_core_fused_axis, block)

            block_tiling_result = {"outer_itervar": block_outer,
                                   "inner_itervar": block_inner}
            ub_tiling_result = {"axis": self._reduce_axis_num,
                                "outer_itervar": ub_outer,
                                "inner_itervar": ub_inner}
            self._tiling_result = {"block_tiling": block_tiling_result,
                                   "ub_tiling": ub_tiling_result}

        else:
            # 2. large
            # Tiling strategy: 1) 128 alignment reduction set_mask (not use)
            # 2) Enable 32 cores > 3) Maximize UB utilization
            # multi-core, already >=32B
            self._calculate_tiling_last_common()
            self._do_tiling()
            self._calculate_multi_core()
            self._do_multi_core()

    def _do_tiling_last_noalign_storagealign(self):
        # Prerequisites:
        # 1. Eliminate Worskspace, and certainly will not cut the reduce axis.
        # 2. Definitely greater than 32B, multi-core could enabled
        fuse_list = []
        for i in range(0, self._reduce_axis_num):
            fuse_list.append(self._res.op.axis[i])
        tmp_fuse_axis = self._schedule[self._res].fuse(*fuse_list)

        # Tiling strategy: 1. Enable 32 cores
        # block tiling
        block_outer, block_inner = self._schedule[self._res].split(
            tmp_fuse_axis, nparts=self._core_dim)
        # ub tiling
        ub_outer, ub_inner = self._schedule[self._res].split(
            self._res.op.axis[self._reduce_axis_num], nparts=1)
        # multi core
        self._multi_core_fused_axis = block_outer
        block = tvm.thread_axis("blockIdx.x")
        self._schedule[self._res].bind(self._multi_core_fused_axis, block)

        block_tiling_result = {"outer_itervar": block_outer,
                               "inner_itervar": block_inner}
        ub_tiling_result = {"axis": self._reduce_axis_num,
                            "outer_itervar": ub_outer,
                            "inner_itervar": ub_inner}
        self._tiling_result = {"block_tiling": block_tiling_result,
                               "ub_tiling": ub_tiling_result}

    def _do_tiling_last_dim1(self):
        # ub tiling
        ub_outer, ub_inner = self._schedule[self._res].split(
            self._res.op.axis[0], nparts=1)

        block_tiling_result = {"outer_itervar": None,
                               "inner_itervar": None}
        ub_tiling_result = {"axis": 0,
                            "outer_itervar": ub_outer,
                            "inner_itervar": ub_inner}
        self._tiling_result = {"block_tiling": block_tiling_result,
                               "ub_tiling": ub_tiling_result}

    def _calculate_compute_at(self):
        """
        Calculate the tensor that needs compute at
        """
        ub_outer = self._tiling_result["ub_tiling"]["outer_itervar"]

        for i in self._cache_read_tensors_and_buffer_map:
            read_buffer = self._cache_read_tensors_and_buffer_map[i]
            para = {"parent": self._schedule[self._res], "scope": ub_outer}
            self._compute_at_map[read_buffer] = para

        # write_buffer compute_at, except compute_inline_buffer
        for i in self._cache_write_tensors_and_buffer_map:
            write_buffer = self._cache_write_tensors_and_buffer_map[i]
            if write_buffer not in self._compute_inline_buffer:
                para = {"parent": self._schedule[self._res], "scope": ub_outer}
                self._compute_at_map[write_buffer] = para


    def _do_compute_at(self):
        for stage in self._compute_at_map:
            parent_stage = self._compute_at_map[stage]["parent"]
            scope_iter_var = self._compute_at_map[stage]["scope"]
            if scope_iter_var is not None:
                self._schedule[stage].compute_at(parent_stage, scope_iter_var)

    def _get_instruction(self, tensor):
        """
        Get the instruction map of tensor
        get insn for backend instruction map first (self._insn_map),
        then front-end instruction map (self._reg_insn_map)

        Parameters:
        ----------
        None

        Returns
        -------
        Instruction map string
        """
        str_list = tensor.op.tag.split("|")
        insn = self._insn_map.get(str_list[0])
        if insn and self._check_cast_support(tensor):
            return insn

        insn = self._reg_insn_map.get(str_list[0])
        return insn

    # pylint: disable=no-self-use
    def _check_cast_support(self, tensor):
        """
        Judge if tensor supports cast instruction operations,
        because of backend instruction bug of cast

        Parameters:
        ----------
        tensors :  input tensor

        Returns
        -------
        Bool: True or False
        """
        cache_buffer = tensor
        read_buffer = tensor.op.input_tensors[0]
        if read_buffer.dtype == "int32":
            if cache_buffer.dtype == "float16" or \
                    cache_buffer.dtype == "float32":
                return False
        return True

    def _calculate_emit_insn_nlst(self):
        """
        Calculate the instruction map of tensor

        Parameters:
        ----------
        None

        Returns
        -------
        None
        """
        # Backend instruction map
        self._get_emit_insn_map()
        # Front-end instruction map
        self._get_reg_emit_insn_map()

        # 1. res dma_copy
        # 2. input dma_copy
        for i in self._cache_read_tensors_and_buffer_map:
            read_buffer = self._cache_read_tensors_and_buffer_map[i]
            self._emit_insn_map[read_buffer] = {"scope": read_buffer.op.axis[0],
                                                "instruction": 'dma_copy'}

        # 3. ub compute
        for i in self._cache_write_tensors_and_buffer_map:
            write_buffer = self._cache_write_tensors_and_buffer_map[i]
            insn = self._get_instruction(write_buffer)
            if not insn and i not in self._reduce_tensors:
                raise RuntimeError("get_instruction is None")

            # 3.0 compute_inline, (not last reduce axis broadcast)
            if write_buffer in self._compute_inline_buffer:
                continue
            # 3.1 reduce
            elif i in self._reduce_tensors:
                # vector_reduce_max/vector_reduce_min/vector_reduce_sum
                insn = write_buffer.op.tag.split("|")[0]
                insn = "vector_" + insn
                insn_para = {"scope": write_buffer.op.reduce_axis[0], "instruction": insn}
            # 3.2 elewise using reduce result, compute_inline broadcast(not last axis)
            elif i in self._after_reduce_bc_tensors and (not self._is_last_reduce_axis):
                insn_para = {
                    "scope": write_buffer.op.axis[self._tiling_result["ub_tiling"]["axis"]],
                    "instruction": insn}
            # 3.3 last reduce axis broadcast
            elif i in self._broadcast_tensors and self._is_last_reduce_axis:
                insn_para = {"scope": write_buffer.op.axis[self._reduce_axis_num],
                             "instruction": insn}
            # 3.4 elewise
            else:
                insn_para = {"scope": write_buffer.op.axis[0], "instruction": insn}

            self._emit_insn_map[write_buffer] = insn_para


    def _calculate_emit_insn_last(self):
        """
        Calculate the instruction map of tensor

        Parameters:
        ----------
        None

        Returns
        -------
        None
        """
        # Backend instruction map
        self._get_emit_insn_map()
        # Front-end instruction map
        self._get_reg_emit_insn_map()

        # 1. res dma_copy
        # 2. input dma_copy
        for i in self._cache_read_tensors_and_buffer_map:
            read_buffer = self._cache_read_tensors_and_buffer_map[i]
            self._emit_insn_map[read_buffer] = {"scope": read_buffer.op.axis[0],
                                                "instruction": 'dma_copy'}

        # 3. ub compute
        for i in self._cache_write_tensors_and_buffer_map:
            write_buffer = self._cache_write_tensors_and_buffer_map[i]
            insn = self._get_instruction(write_buffer)
            if not insn and i not in self._reduce_tensors:
                raise RuntimeError("get_instruction is None")

            # 3.0 compute_inline, (not last reduce axis broadcast)
            if write_buffer in self._compute_inline_buffer:
                continue
            # 3.1 reduce
            elif i in self._reduce_tensors:
                # vector_reduce_max/vector_reduce_min/vector_reduce_sum
                insn = write_buffer.op.tag.split("|")[0]
                if self._is_reduce_last_axis_enhance_insn:
                    insn = "reduce_last_axis_enhance_" + insn
                    insn_scope = write_buffer.op.axis[self._tiling_result["ub_tiling"]["axis"]]
                else:
                    insn = "vector_" + insn
                    insn_scope = write_buffer.op.axis[0]
                insn_para = {"scope": insn_scope, "instruction": insn}
            # 3.2 elewise using reduce result, compute_inline broadcast(not last axis)
            elif i in self._after_reduce_bc_tensors and (not self._is_last_reduce_axis):
                insn_para = {
                    "scope": write_buffer.op.axis[self._tiling_result["ub_tiling"]["axis"]],
                    "instruction": insn}
            # 3.3 last reduce axis broadcast
            elif i in self._broadcast_tensors and self._is_last_reduce_axis:
                insn_scope = write_buffer.op.axis[self._tiling_result["ub_tiling"]["axis"]]
                insn_para = {"scope": insn_scope, "instruction": insn}
            # 3.4 elewise
            else:
                insn_para = {"scope": write_buffer.op.axis[0], "instruction": insn}

            self._emit_insn_map[write_buffer] = insn_para

    def _do_emit_insn(self):
        for stage in self._emit_insn_map:
            scope_iter_var = self._emit_insn_map[stage]["scope"]
            instruction = self._emit_insn_map[stage]["instruction"]
            self._schedule[stage].emit_insn(scope_iter_var, instruction)

        # res dma_copy, if self._is_multi_core and >=32B:
        ub_inner = self._tiling_result["ub_tiling"]["inner_itervar"]
        self._schedule[self._res].emit_insn(ub_inner, 'dma_copy', {"no_overlap": 1})

    def _do_double_buffer(self):
        """
        double buffer operations
        read_buffer : the all read_cache for input in ub, type is list
        """
        if self._is_double_buffer:
            for i in self._double_buffer_tensors:
                self._schedule[i].double_buffer()


    def _get_softmax_spec_node(self):
        self._spec_node_list = []
        self._spec_mid_list = []
        for i in self._reduce_tensors:
            tensors_before_reduce, spec_node, consumers = \
                self._get_tensors_before_reduce_and_spec_node(i)
            self._map_apend(self._compute_at_tensor_map, i,
                            tensors_before_reduce)
            self._map_apend(self._cache_read_tensors_and_readers_map,
                            spec_node, consumers)
            self._spec_mid_list.append(spec_node)
            if not isinstance((spec_node.op), tvm.tensor.PlaceholderOp):
                self._spec_node_list.append(spec_node)
        self._spec_mid_list.append(self._res)
        self._spec_mid_list.reverse()

        for i in range(0, len(self._spec_mid_list)-1):
            compute_at_tensor = \
                self._get_tensors_list(self._spec_mid_list[i],
                                       self._spec_mid_list[i+1])
            for rtensor in self._reduce_tensors:
                if rtensor in compute_at_tensor:
                    compute_at_tensor = compute_at_tensor - \
                        set(self._compute_at_tensor_map[rtensor])
                    self._map_apend(self._compute_at_tensor_map,
                                    self._spec_mid_list[i],
                                    list(compute_at_tensor))


    def _get_tensors_before_reduce_and_spec_node(self, reduce_tensor):
        tensors_before_reduce = set()

        def _get_tensor_list(tmp_tensor, tmp_end_tensor=None):
            # not contain reduce
            if tmp_tensor.op.tag.find("reduce") == -1:
                tensors_before_reduce.add(tmp_tensor)

            # sigle input
            if len(tmp_tensor.op.input_tensors) > 1:
                return
            if (tmp_end_tensor is not None) and \
                tmp_tensor.same_as(tmp_end_tensor):
                return

            for in_tensor in list(tmp_tensor.op.input_tensors):
                tensors_before_reduce.add(in_tensor)

            for in_tensor in list(tmp_tensor.op.input_tensors):
                if not isinstance((in_tensor.op), tvm.tensor.PlaceholderOp):
                    _get_tensor_list(in_tensor, tmp_end_tensor)

        tensors_before_reduce.clear()
        _get_tensor_list(reduce_tensor)

        spec_node = []
        consumers = []
        for i in self._cache_write_tensors_and_input_map:
            if len(self._cache_write_tensors_and_input_map[i]) == 2:
                for leaf_tensor in self._cache_write_tensors_and_input_map[i]:
                    if leaf_tensor in tensors_before_reduce:
                        spec_node.append(leaf_tensor)
                        consumers.append(i)

        # contain reduce tensor, to find readers
        tensors_before_reduce.add(reduce_tensor)
        for i in tensors_before_reduce:
            for in_tensor in list(i.op.input_tensors):
                if in_tensor.same_as(spec_node[0]):
                    consumers.append(i)

        tensors_before_reduce.clear()
        _get_tensor_list(reduce_tensor, spec_node[0])

        return list(tensors_before_reduce), spec_node[0], consumers

    def _get_tensors_list(self, begin_tensor, end_tensor=None):
        tensors_before_reduce = set()

        def _get_tensor_list(tmp_tensor, tmp_end_tensor=None):
            if (tmp_end_tensor is not None) and \
                tmp_tensor.same_as(tmp_end_tensor):
                return

            tensors_before_reduce.add(tmp_tensor)
            for in_tensor in list(tmp_tensor.op.input_tensors):
                tensors_before_reduce.add(in_tensor)

            for in_tensor in list(tmp_tensor.op.input_tensors):
                if not isinstance((in_tensor.op), tvm.tensor.PlaceholderOp):
                    _get_tensor_list(in_tensor, tmp_end_tensor)

        _get_tensor_list(begin_tensor, end_tensor)

        return tensors_before_reduce

    def _do_cache_rw_with_workspace(self):
        for i in self._cache_read_tensors_and_readers_map:
            readers = self._cache_read_tensors_and_readers_map[i]
            for j in readers:
                read_buffer = self._schedule.cache_read(i, self._scope, [j])
                self._double_buffer_tensors.append(read_buffer)
                self._map_apend(self._cache_read_tensors_and_buffer_map,
                                i, read_buffer)
                self._cache_read_buffer_and_reader_map[read_buffer] = j

        for i in self._cache_write_tensors:
            write_buffer = self._schedule.cache_write(i, self._scope)
            self._cache_write_tensors_and_buffer_map[i] = write_buffer

    def _do_compute_inline_with_workspace(self):
        for i in self._cache_write_tensors:
            # res(div) and workspace node (exp) as output, not compute_inline
            if i not in self._spec_mid_list:
                self._schedule[i].compute_inline()

    def _do_storage_align_last_with_workspace(self):
        if not self._is_last_reduce_axis:
            return

        # shape dim1, not need storage_align
        if len(shape_to_list(self._res.shape)) == 1:
            return

        at_axis = self._reduce_axis_num - 1  # axis: -2

        for i in self._cache_read_buffer_and_reader_map:
            align_factor, _ = get_align_factor(i.dtype)
            self._schedule[i].storage_align(i.op.axis[at_axis],
                                            align_factor, 0)

        for i in self._cache_write_tensors_and_buffer_map:
            write_buffer = self._cache_write_tensors_and_buffer_map[i]
            align_factor, _ = get_align_factor(write_buffer.dtype)
            if write_buffer not in self._compute_inline_buffer:
                self._schedule[write_buffer].storage_align(
                    write_buffer.op.axis[at_axis], align_factor, 0)

    def _calculate_compute_at_with_workspace(self):
        for at_tensor in self._compute_at_tensor_map:
            # 1. at to reduce node
            if at_tensor not in self._spec_mid_list: # reduce tensor
                new_at_tensor = \
                    self._cache_write_tensors_and_buffer_map[at_tensor]
                for tmp_tensor in self._compute_at_tensor_map[at_tensor]:
                    # 1.1 (data/exp) spec_node
                    # need to find the corresponding ub_buffer
                    # from _cache_read_buffer_and_reader_map
                    if tmp_tensor in self._spec_mid_list:
                        for ub_tensor in \
                            self._cache_read_buffer_and_reader_map:
                            gm_reader = \
                                self._cache_read_buffer_and_reader_map[
                                    ub_tensor]
                            find_list = \
                                list(self._compute_at_tensor_map[at_tensor])
                            find_list.append(at_tensor)
                            if gm_reader in find_list:
                                para = \
                                    {"parent": self._schedule[new_at_tensor],
                                     "scope": None}
                                self._compute_at_map[ub_tensor] = para
                    # 1.2 Normal output node
                    else:
                        ub_tensor = \
                            self._cache_write_tensors_and_buffer_map[
                                tmp_tensor]
                        para = {"parent": self._schedule[new_at_tensor],
                                "scope": None}
                        self._compute_at_map[ub_tensor] = para

            # 2. at to output spec_node (exp/div)
            else:
                # 2.1 Normal output node, not contain input_node PlaceholderOp
                for tmp_tensor in self._compute_at_tensor_map[at_tensor]:
                    # process for log_softmax_grad,
                    # not contain input_node PlaceholderOp
                    if isinstance((tmp_tensor.op), tvm.tensor.PlaceholderOp):
                        continue
                    ub_tensor = \
                        self._cache_write_tensors_and_buffer_map[tmp_tensor]
                    para = {"parent": self._schedule[at_tensor],
                            "scope": None}
                    self._compute_at_map[ub_tensor] = para

                # 2.2 spec_node (data/exp)
                for ub_tensor in self._cache_read_buffer_and_reader_map:
                    gm_reader = \
                        self._cache_read_buffer_and_reader_map[ub_tensor]
                    find_list = \
                        set(self._compute_at_tensor_map[at_tensor]) - \
                        set(self._reduce_tensors)
                    if gm_reader in find_list:
                        para = {"parent": self._schedule[at_tensor],
                                "scope": None}
                        self._compute_at_map[ub_tensor] = para

        # 3. workspace node, at to res
        for spec_node in self._spec_node_list:
            para = {"parent": self._schedule[self._res], "scope": None}
            self._compute_at_map[spec_node] = para

    def _do_tiling_last_with_workspace(self):
        shape = shape_to_list(self._res.shape)
        align_factor_32byte, align_factor_256byte = \
            self._get_data_alignment(self._cache_write_tensors)

        # double_buffer, and max_ub_count could adjust
        self._is_double_buffer = True

        # fix softmax float32 oom of mini
        self._max_ub_count = self._max_ub_count - 2048
        # ub tiling
        factor = min(self._max_ub_count, shape[self._reduce_axis_num])
        factor = factor // align_factor_256byte * align_factor_256byte
        tail = shape[self._reduce_axis_num] % factor
        if 0 < tail < align_factor_32byte:
            factor = factor - align_factor_32byte


        for at_tensor in self._compute_at_tensor_map:
            # 1. reduce_node
            if at_tensor not in self._spec_mid_list:
                reduce_tensor = \
                    self._cache_write_tensors_and_buffer_map[at_tensor]
                parent_stage = self._schedule[reduce_tensor]
                outer_o, outer_i = parent_stage.split(
                    parent_stage.op.reduce_axis[0], nparts=1)
                outer, inner = parent_stage.split(outer_i, factor=factor)
                parent_stage.reorder(
                    outer_o, parent_stage.op.axis[self._reduce_axis_num],
                    outer, inner)
                # reduce emit_insn at inner
                self._emit_insn_map[reduce_tensor] = \
                    {"scope": inner, "instruction": 'vector_auto'}
            # 2. exp/div spec_node
            else:
                parent_stage = self._schedule[at_tensor]
                outer_o, outer_i = parent_stage.split(
                    parent_stage.op.axis[self._reduce_axis_num], nparts=1)
                outer, inner = parent_stage.split(outer_i, factor=factor)
                parent_stage.reorder(outer_o, outer, inner)
                # spec_node, dma_copy output, emit_insn at inner
                self._emit_insn_map[at_tensor] = {"scope": inner,
                                                  "instruction": 'dma_copy'}

            for stage in self._compute_at_map:
                if self._compute_at_map[stage]["parent"] == parent_stage:
                    # 1. reduce node,
                    # at spec_node (exp/div), outer_outer axis
                    if stage.op.tag.find("reduce") != -1:
                        self._compute_at_map[stage]["scope"] = outer_o
                    # 2. workspace node,
                    # at res, outer_outer axis
                    elif stage in self._spec_node_list:
                        self._compute_at_map[stage]["scope"] = outer_o
                    # 3. normal ub_buffer node ,
                    # at spec_node (exp/div), outer axis
                    else:
                        self._compute_at_map[stage]["scope"] = outer

        ub_tiling_result = {"axis": self._reduce_axis_num,
                            "outer_itervar": None,
                            "inner_itervar": None}
        self._tiling_result = {"block_tiling": None,
                               "ub_tiling": ub_tiling_result}

        # multi-core
        if len(shape) > 1:
            fuse_list = []
            for i in range(0, self._reduce_axis_num):
                fuse_list.append(self._res.op.axis[i])
            fuse_block_axis = self._schedule[self._res].fuse(*fuse_list)

            # block tiling
            self._multi_core_fused_axis, _ = self._schedule[self._res].split(
                fuse_block_axis, nparts=self._core_dim)

            block = tvm.thread_axis("blockIdx.x")
            self._schedule[self._res].bind(self._multi_core_fused_axis, block)

    def _do_emit_insn_last_with_workspace(self):
        # Backend instruction map
        self._get_emit_insn_map()
        # Front-end instruction map
        self._get_reg_emit_insn_map()

        # 1. res dma_copy, reduce emit_insn in compute_at
        # 2. input dma_copy
        for i in self._cache_read_tensors_and_buffer_map:
            for j in self._cache_read_tensors_and_buffer_map[i]:
                self._emit_insn_map[j] = {"scope": j.op.axis[0],
                                          "instruction": 'dma_copy'}

        # 3. ub compute
        for i in self._cache_write_tensors_and_buffer_map:
            write_buffer = self._cache_write_tensors_and_buffer_map[i]
            insn = self._get_instruction(write_buffer)
            if not insn and i not in self._reduce_tensors:
                raise RuntimeError("get_instruction is None")

            if insn == "elewise_single_rec":
                insn = "vector_rec"

            # 3.0 compute_inline, (not last reduce axis broadcast)
            if write_buffer in self._compute_inline_buffer:
                continue
            # 3.1 reduce
            elif i in self._reduce_tensors:
                # vector_reduce_max/vector_reduce_min/vector_reduce_sum
                insn = write_buffer.op.tag.split("|")[0]
                insn = "vector_" + insn
                self._emit_insn_map[write_buffer]["instruction"] = insn
            # 3.2 elewise using reduce result,
            # compute_inline broadcast(not last axis)
            elif i in self._after_reduce_bc_tensors and \
                (not self._is_last_reduce_axis):
                at_axis = self._tiling_result["ub_tiling"]["axis"]
                insn_scope = write_buffer.op.axis[at_axis]
                insn_para = {"scope": insn_scope, "instruction": insn}
                self._emit_insn_map[write_buffer] = insn_para
            # 3.3 last reduce axis broadcast
            elif i in self._broadcast_tensors and self._is_last_reduce_axis:
                at_axis = self._tiling_result["ub_tiling"]["axis"]
                insn_scope = write_buffer.op.axis[at_axis]
                insn_para = {"scope": insn_scope, "instruction": insn}
                self._emit_insn_map[write_buffer] = insn_para
            # 3.4 elewise
            else:
                insn_para = {"scope": write_buffer.op.axis[0],
                             "instruction": insn}
                self._emit_insn_map[write_buffer] = insn_para

        for stage in self._emit_insn_map:
            scope_iter_var = self._emit_insn_map[stage]["scope"]
            instruction = self._emit_insn_map[stage]["instruction"]
            self._schedule[stage].emit_insn(scope_iter_var, instruction)



# pylint: disable=too-many-locals, invalid-name
def calculate_nlst_axis_block_tiling(shape,
                                     reduce_axis,
                                     block_tiling_min_size=16,
                                     core_num=32,
                                     is_divisible=False):
    """
    calculate not last axis block_tiling
    """
    shape_before_reduce_axis = shape[0:reduce_axis]
    shape_after_reduce_axis = shape[reduce_axis+1:]
    # 1. not last axis, shape_after_reduce_axis shoude not be None
    if not shape_after_reduce_axis:
        raise RuntimeError("the shape after reduce_axis must no be None")
    # 2. shape_before_reduce_axis is None, calculate block_tiling after reduce_axis
    outer, inner, split_axis = tiling_from_front_to_back(shape_after_reduce_axis,
                                                         core_num,
                                                         align_factor=core_num,
                                                         is_divisible=is_divisible)
    while split_axis > 0:
        remaining_size = get_shape_size_ext(shape_after_reduce_axis[split_axis+1:], inner)
        if remaining_size >= block_tiling_min_size:
            break
        outer, inner, split_axis = tiling_from_front_to_back(shape_after_reduce_axis[0:split_axis],
                                                             core_num,
                                                             align_factor=core_num,
                                                             is_divisible=is_divisible)
    # else split_axis == 0
    else:
        remaining_size = get_shape_size_ext(shape_after_reduce_axis[split_axis+1:], inner)
        if remaining_size < block_tiling_min_size:
            outer, inner, split_axis = 1, shape_after_reduce_axis[0], 0

    split_axis_ret = split_axis + reduce_axis + 1
    # 3. shape_before_reduce_axis and shape_after_reduce_axis are not None
    if shape_before_reduce_axis:
        mutilcore_after_reduce_axis = get_shape_size_ext(shape_after_reduce_axis[0:split_axis],
                                                         outer)

        if get_shape_size_ext(shape_after_reduce_axis) < block_tiling_min_size:
            outer, inner, split_axis_ret = 1, shape_before_reduce_axis[0], 0
        else:
            outer_tmp, inner_tmp, split_axis_tmp = tiling_from_front_to_back(
                shape_before_reduce_axis,
                core_num,
                align_factor=core_num,
                is_divisible=is_divisible)
            mutilcore_before_reduce_axis = get_shape_size_ext(
                shape_before_reduce_axis[0:split_axis_tmp],
                outer_tmp)

            if mutilcore_before_reduce_axis >= mutilcore_after_reduce_axis:
                outer, inner, split_axis_ret = outer_tmp, inner_tmp, split_axis_tmp

    return (outer, inner, split_axis_ret)


# pylint: disable=invalid-name
def calculate_last_axis_block_tiling(shape,
                                     reduce_axis,
                                     block_tiling_min_size=16,
                                     core_num=32,
                                     is_divisible=False):
    """
    calculate last axis block_tiling
    """
    shape_before_reduce_axis = shape[0:reduce_axis]
    shape_after_reduce_axis = shape[reduce_axis+1:]
    # 1. last axis, shape_after_reduce_axis shoude be None
    if shape_after_reduce_axis or not shape_before_reduce_axis:
        raise RuntimeError("the shape before reduce_axis must not be None,"
                           " and the shape after reduce_axis must be None")
    # 2. shape_before_reduce_axis calculate block_tiling
    outer, inner, split_axis = tiling_from_front_to_back(shape_before_reduce_axis,
                                                         core_num,
                                                         align_factor=core_num,
                                                         is_divisible=is_divisible)
    while split_axis > 0:
        remaining_size = get_shape_size_ext(shape[split_axis+1:], inner)
        if remaining_size >= block_tiling_min_size:
            break
        outer, inner, split_axis = tiling_from_front_to_back(
            shape_before_reduce_axis[0:split_axis],
            core_num,
            align_factor=core_num,
            is_divisible=is_divisible)
    # else split_axis == 0
    else:
        remaining_size = get_shape_size_ext(shape[split_axis+1:], inner)
        if remaining_size < block_tiling_min_size:
            outer, inner, split_axis = 1, shape_before_reduce_axis[0], 0

    return (outer, inner, split_axis)


# pylint: disable=too-many-arguments
def calculate_nlst_axis_ub_tiling(shape,
                                  reduce_axis,
                                  block_parm,
                                  ub_tiling_max_size,
                                  align_factor=128,
                                  is_divisible=False):
    """
    calculate not last axis ub_tiling
    """
    _, blcok_inner, blcok_split_axis = block_parm[0], block_parm[1], block_parm[2]
    if reduce_axis < blcok_split_axis:
        shape_tobe_tiling = shape[blcok_split_axis+1:]
        baseline_axis = blcok_split_axis + 1
        if is_divisible:
            shape_tobe_tiling.insert(0, blcok_inner)
            baseline_axis = baseline_axis - 1
        # shape_tobe_tiling is still []
        if not shape_tobe_tiling:
            shape_tobe_tiling = [blcok_inner]
            baseline_axis = baseline_axis - 1
    else:
        shape_tobe_tiling = shape[reduce_axis+1:]
        baseline_axis = reduce_axis + 1
    outer, inner, split_axis = tiling_from_back_to_front(shape_tobe_tiling,
                                                         ub_tiling_max_size,
                                                         align_factor=align_factor,
                                                         is_divisible=is_divisible)
    return (outer, inner, split_axis+baseline_axis)


# pylint: disable=too-many-arguments
def calculate_last_axis_ub_tiling(shape,
                                  reduce_axis,
                                  block_parm,
                                  ub_tiling_max_size,
                                  align_factor=128,
                                  is_divisible=False):
    """
    calculate last axis ub_tiling
    """
    _, blcok_inner, blcok_split_axis = block_parm[0], block_parm[1], block_parm[2]
    if blcok_split_axis >= reduce_axis:
        raise RuntimeError("reduce_last_axis: blcok_split_axis >= reduce_axis")

    shape_tobe_tiling = shape[blcok_split_axis+1:]
    baseline_axis = blcok_split_axis + 1
    if is_divisible:
        shape_tobe_tiling.insert(0, blcok_inner)
        baseline_axis = baseline_axis - 1
    # shape_tobe_tiling is still []
    if not shape_tobe_tiling:
        shape_tobe_tiling = [blcok_inner]
        baseline_axis = baseline_axis - 1

    outer, inner, split_axis = tiling_from_back_to_front(shape_tobe_tiling,
                                                         ub_tiling_max_size,
                                                         align_factor=align_factor,
                                                         is_divisible=is_divisible)

    ret_axis = split_axis + baseline_axis

    if ret_axis == reduce_axis and outer != 1:
        raise RuntimeError("error split last axis")
    return (outer, inner, ret_axis)



@tvm.register_func("tvm.intrin.cce.reduce_last_axis_enhance_reduce_sum")
def reduce_last_axis_enhance_reduce_sum(tensor_op):
    """
    reduce last axis reduce sum enhance
    """
    return reduce_last_axis_enhance(tensor_op, "vcadd")

@tvm.register_func("tvm.intrin.cce.reduce_last_axis_enhance_reduce_max")
def reduce_last_axis_enhance_reduce_max(tensor_op):
    """
    reduce last axis reduce max enhance
    """
    return reduce_last_axis_enhance(tensor_op, "vcmax")

@tvm.register_func("tvm.intrin.cce.reduce_last_axis_enhance_reduce_min")
def reduce_last_axis_enhance_reduce_min(tensor_op):
    """
    reduce last axis reduce min enhance
    """
    return reduce_last_axis_enhance(tensor_op, "vcmin")


def get_mask_fp16_skip_one(length):
    """
    calculate MASK in cce for skip one half
    """
    length = int(length)
    len1 = max(length - 32, 0)
    len2 = min(length, 32)
    mask1 = 0
    mask2 = 0
    for _ in range(len1):
        mask1 = mask1 * 4 + 1
    for _ in range(len2):
        mask2 = mask2 * 4 + 1
    return mask1, mask2


def reset_mask_insn(ib_expr, type_, bits=128, mask_func=None):
    """
    :describe: caculate the mask, and set vector mask
    :param ib_expr: ir builder
    :param type_: the type of mask dst
    :param bits: the bit of mask, default : 128
    """
    # argmin/argmax has his own set_mask func
    if mask_func is not None:
        mask1, mask2 = mask_func(bits)
    else:
        mask1, mask2 = cce_util.set_mask(bits)

    ib_expr.emit(tvm.call_extern(
        type_, "set_vector_mask", tvm.const(mask1, dtype="uint64"),
        tvm.const(mask2, dtype="uint64")))

def new_alloc(ib_expr, dtype, shape, name, scope):
    """
    new alloc
    """
    buf_var = ib_expr.allocate(dtype, shape, name=name, scope=scope)
    new_buffer = tvm.decl_buffer(shape, buf_var.dtype, name=name, scope=scope, data=buf_var)

    return new_buffer

def _get_for_vars(tensor_op):
    for_extent_vals = []
    for_vars = []
    def _post_order_for(tensor_op):
        """
        post order
        """
        if isinstance(tensor_op, tvm.stmt.For):
            for_extent_vals.append(tensor_op.extent.value)
            for_vars.append(tensor_op.loop_var)

    _ = tvm.ir_pass.IRTransform(tensor_op, None, _post_order_for, ["For"])

    return for_vars, for_extent_vals


def _insn_reduce_last_using_vcxx_dim1(ib_expr,
                                      intrin_cmd,
                                      res_buffer,
                                      src_buffer,
                                      reduce_axis_len,
                                      vector_inst_one_repeat_size,
                                      k1_size,
                                      repeat_stride):
    dtype = src_buffer.dtype
    repeat_time = reduce_axis_len // vector_inst_one_repeat_size
    remain_size = reduce_axis_len % vector_inst_one_repeat_size

    if repeat_time > 0:
        reset_mask_insn(ib_expr, dtype, bits=vector_inst_one_repeat_size)

        ib_expr.emit(tvm.call_extern(
            res_buffer.dtype, intrin_cmd,
            res_buffer.access_ptr("rw", offset=0),
            src_buffer.access_ptr("r", offset=-k1_size),
            repeat_time,
            *repeat_stride))

    if remain_size > 0:
        reset_mask_insn(ib_expr, dtype, bits=remain_size)

        src_offset = repeat_time * vector_inst_one_repeat_size - k1_size
        ib_expr.emit(tvm.call_extern(
            res_buffer.dtype, intrin_cmd,
            res_buffer.access_ptr("rw", offset=repeat_time),
            src_buffer.access_ptr("r", offset=src_offset),
            1,
            *repeat_stride))

    if repeat_time > 1 or (repeat_time == 1 and remain_size > 0):
        if remain_size > 0:
            reset_mask_insn(ib_expr, dtype, bits=repeat_time + 1)
        else:
            reset_mask_insn(ib_expr, dtype, bits=repeat_time)
        ib_expr.emit(tvm.call_extern(
            res_buffer.dtype, intrin_cmd,
            res_buffer.access_ptr("rw", offset=0),
            res_buffer.access_ptr("r", offset=0),
            1,
            *repeat_stride))

def __tail_vcadd_proc(ib_expr, intrin_cmd, src_buffer, repeat_time, remain_size):
    if repeat_time > 1 or (repeat_time == 1 and remain_size > 0):
        if remain_size > 0:
            reset_mask_insn(ib_expr, src_buffer.dtype, bits=repeat_time + 1)
        else:
            reset_mask_insn(ib_expr, src_buffer.dtype, bits=repeat_time)

        ib_expr.emit(tvm.call_extern(
            src_buffer.dtype, intrin_cmd,
            src_buffer.access_ptr("rw", offset=0),
            src_buffer.access_ptr("r", offset=0),
            1, 1, 1, 8))

def __tail_vcmax_proc(ib_expr, intrin_cmd, src_buffer, repeat_time, remain_size,
                      vector_inst_one_repeat_size, factor, repeat_stride):
    vcmax_repeat_size = (vector_inst_one_repeat_size // 2)
    total_time = repeat_time
    if remain_size > 0:
        total_time = total_time + 1
    sub_repeat_time = total_time // vcmax_repeat_size
    sub_remain_size = total_time % vcmax_repeat_size

    if sub_repeat_time > 0:
        reset_mask_insn(ib_expr, src_buffer.dtype, bits=vcmax_repeat_size,
                        mask_func=get_mask_fp16_skip_one)
        ib_expr.emit(tvm.call_extern(
            src_buffer.dtype, intrin_cmd,
            src_buffer.access_ptr("rw", offset=0),
            src_buffer.access_ptr("r", offset=0),
            sub_repeat_time,
            *repeat_stride))

    if total_time > 1 and sub_remain_size > 0:
        reset_mask_insn(ib_expr, src_buffer.dtype, bits=sub_remain_size,
                        mask_func=get_mask_fp16_skip_one)
        ib_expr.emit(tvm.call_extern(
            src_buffer.dtype, intrin_cmd,
            src_buffer.access_ptr("rw", offset=sub_repeat_time * factor),
            src_buffer.access_ptr("r",
                                  offset=sub_repeat_time
                                  * vector_inst_one_repeat_size),
            1, *repeat_stride))

    if sub_repeat_time > 1 or (sub_repeat_time == 1 and sub_remain_size > 0):
        if sub_remain_size > 0:
            reset_mask_insn(ib_expr, src_buffer.dtype, bits=sub_repeat_time + 1,
                            mask_func=get_mask_fp16_skip_one)
        else:
            reset_mask_insn(ib_expr, src_buffer.dtype, bits=sub_repeat_time,
                            mask_func=get_mask_fp16_skip_one)

        ib_expr.emit(tvm.call_extern(
            src_buffer.dtype, intrin_cmd,
            src_buffer.access_ptr("rw", offset=0),
            src_buffer.access_ptr("r", offset=0),
            1, *repeat_stride))


def _insn_reduce_last_using_vcxx(ib_expr,
                                 intrin_cmd,
                                 dst_buffer,
                                 src_buffer,
                                 reduce_axis_len,
                                 vector_inst_one_repeat_size,
                                 loop,
                                 repeat_stride):
    is_vcmax = (intrin_cmd in ("vcmax", "vcmin"))
    factor = 2 if is_vcmax else 1

    repeat_time = reduce_axis_len // vector_inst_one_repeat_size
    remain_size = reduce_axis_len % vector_inst_one_repeat_size

    if repeat_time > 0:
        reset_mask_insn(ib_expr, src_buffer.dtype, bits=vector_inst_one_repeat_size)
        ib_expr.emit(tvm.call_extern(
            src_buffer.dtype, intrin_cmd,
            src_buffer.access_ptr("rw", offset=0),
            src_buffer.access_ptr("r", offset=0),
            repeat_time,
            *repeat_stride))

    if remain_size > 0:
        reset_mask_insn(ib_expr, src_buffer.dtype, bits=remain_size)
        ib_expr.emit(tvm.call_extern(
            src_buffer.dtype, intrin_cmd,
            src_buffer.access_ptr("rw", offset=repeat_time * factor),
            src_buffer.access_ptr("r", offset=
                                  repeat_time * vector_inst_one_repeat_size),
            1,
            *repeat_stride))

    if is_vcmax:
        __tail_vcmax_proc(ib_expr, intrin_cmd, src_buffer, repeat_time, remain_size,
                          vector_inst_one_repeat_size, factor, repeat_stride)
    else:
        __tail_vcadd_proc(ib_expr, intrin_cmd, src_buffer, repeat_time, remain_size)

    ib_expr.emit(tvm.call_extern(
        dst_buffer.dtype, "reg_mov",
        dst_buffer.access_ptr("rw", offset=loop),
        src_buffer.access_ptr("rw"), ))


def _insn_reg_mov(ib_expr, dst_buffer, src_buffer, reg, reg_num,
                  reduce_axis_len, loop, k1_size):
    # reg[8]
    reg_mov_loop = reduce_axis_len//8
    if reg_mov_loop:
        remain_reg_mov_loop = reduce_axis_len - reg_mov_loop*8
    else:
        remain_reg_mov_loop = 1

    if remain_reg_mov_loop != 1:
        reamin_reg_num = remain_reg_mov_loop
    else:
        reamin_reg_num = reg_num

    if reg_mov_loop:
        with ib_expr.for_range(0, reg_mov_loop, name="lp_idx") as loop_idx:
            for i in range(8):
                ib_expr.emit(tvm.call_extern(
                    src_buffer.dtype, "reg_mov",
                    tvm.call_extern(reg.dtype, "reg", reg[i]),
                    src_buffer.access_ptr(
                        "rw", offset=loop*reduce_axis_len-k1_size + loop_idx*8+i)
                ))
            for j in range(8):
                ib_expr.emit(tvm.call_extern(
                    src_buffer.dtype, "reg_mov",
                    dst_buffer.access_ptr("rw", offset=loop_idx*8+j),
                    tvm.call_extern(reg.dtype, "reg", reg[j])
                ))

    src_dst_offset = reg_mov_loop*8

    if remain_reg_mov_loop:
        for i in range(reamin_reg_num):
            ib_expr.emit(tvm.call_extern(
                src_buffer.dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[i]),
                src_buffer.access_ptr(
                    "rw", offset=src_dst_offset+loop*reduce_axis_len-k1_size +i)
            ))
        for j in range(reamin_reg_num):
            ib_expr.emit(tvm.call_extern(
                src_buffer.dtype, "reg_mov",
                dst_buffer.access_ptr("rw", offset=src_dst_offset+j),
                tvm.call_extern(reg.dtype, "reg", reg[j])
            ))


def reduce_last_axis_enhance(tensor_op, intrin_cmd):
    """
    reduce last axis reduce sum and max
    """
    ins, outs = cce_util.get_buffer(tensor_op)
    ib_expr = tvm.ir_builder.create()

    for_vars, for_extent_vals = _get_for_vars(tensor_op)
    reduce_axis_len = for_extent_vals[0]
    k1_size = for_vars[0]

    src_buffer = ins[1]
    res_buffer = outs[0]
    dtype = src_buffer.dtype
    if dtype == "float16":
        vector_inst_one_repeat_size = 128
    else:
        vector_inst_one_repeat_size = 64

    is_vcmax = (intrin_cmd in ("vcmax", "vcmin"))
    is_vcmax_v200 = (dtype.lower() == "float32") and \
        intrinsic_check_support("Intrinsic_vcmax", "float32")
    # v200 vcmax/vcmin fp32, 7 input param
    if is_vcmax and is_vcmax_v200:
        repeat_stride = (1, 1, 8, 0, 0, 0)
    else:
        repeat_stride = (1, 1, 8)


    if len(for_extent_vals) == 1:
        _insn_reduce_last_using_vcxx_dim1(ib_expr,
                                          intrin_cmd,
                                          res_buffer,
                                          src_buffer,
                                          reduce_axis_len,
                                          vector_inst_one_repeat_size,
                                          k1_size,
                                          repeat_stride)

        reset_mask_insn(ib_expr, res_buffer.dtype)
        return ib_expr.get()

    # 32B align buffer
    tmp_buf_len = (reduce_axis_len + 15) // 16 * 16
    tmp_buf = new_alloc(ib_expr, src_buffer.dtype, (tmp_buf_len,), 'tmp_buf', scope=cce.scope_ubuf)
    # reg
    reg_num = reduce_axis_len if(reduce_axis_len < 8) else 8
    reg = ib_expr.allocate(outs[0].dtype, (reg_num,), name="reg_buf", scope=cce.scope_reg)

    loop_len = get_shape_size_ext(for_extent_vals[1:])

    with ib_expr.for_range(0, loop_len, name="idx") as loop:
        _insn_reg_mov(ib_expr, tmp_buf, src_buffer, reg, reg_num,
                      reduce_axis_len, loop, k1_size)

        _insn_reduce_last_using_vcxx(ib_expr,
                                     intrin_cmd,
                                     res_buffer,
                                     tmp_buf,
                                     reduce_axis_len,
                                     vector_inst_one_repeat_size,
                                     loop,
                                     repeat_stride)

    reset_mask_insn(ib_expr, res_buffer.dtype)

    return ib_expr.get()
