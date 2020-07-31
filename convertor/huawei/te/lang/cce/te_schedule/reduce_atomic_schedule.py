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

reduce atomic schedule
"""
from functools import reduce as reduceIns
from te import platform as cceconf
from te import tvm
from te.platform import cce_emitinsn_params
import te.platform.cce_params as cce
from te.platform import cce_conf
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


# pylint: disable=too-many-locals, too-many-return-statements,too-few-public-methods,too-many-arguments,too-many-statements,no-self-use,too-many-lines,too-many-instance-attributes,too-many-branches,
class ReduceAtomicSchedule(VectorSchedule):
    """
    class of cce elewise schedule

    Parameters
    ----------
    VectorSchedule: base class of reduce atomic schedule

    Returns
    -------
    ReduceAtomicSchedule_instance : instance of ReduceAtomicSchedule
    """

    def __init__(self, need_multi_core=True):
        VectorSchedule.__init__(self, need_multi_core)
        # record ops
        self._op = []
        # record origin op
        self._origin_op = []

        self._res_tensor = None
        self._last_output_tensors = []
        self._input_tensors = []
        self._input_tensor_dst_tensor_map = {}
        self._mid_tensors = []  # exclude _input_tensors and last_output_tensor
        self._mid_tensor_dst_tensor_map = {}  # {mid_tensor->dst_tensor}

        self._mid_output_tensors = []
        self._mid_output_tensors_dst_tensor_map = {}

        self._cache_write_exclude_tensors = []

        self._broadcast_last_axis_tensors = []
        self._broadcast_scalars = []
        self._broadcast_scalar_dst_tensor_map = {}
        self._broadcast_not_last_axis_tensors = []

        self._tuple_reduce_tensor_out_list = []

        self._reduce_tensors = []
        self._broadcast_tensors = []
        self._vector_dup_tensors = []  # broadcast scalar in ub
        self._tensor_dst_tensor_map = {}  # {tensor->dst_tensor(next_tensor)}

        self._tensor_scaler_operator = ["elewise_binary_mul",
                                        "elewise_binary_add"]

        # 0: reserved; 1: reduce all; 2: reduce nlast; 3: reduce last
        self._tiling_case = 0

        self._spec_node_list = []
        self._is_last_axis_broadcast = False
        self._total_size = 0
        self._is_muti_output = False
        self._have_reduce = False
        self._final_out_tensor_global = None
        self._final_out_tensor_global_emit_axis = 0
        self._is_multi_core_need_fused = False
        self._is_need_dichotomy_add = False
        self._is_32b_align = True
        self._last_axis_size = None
        self._final_out_tensor_ub_rf = None

        # reduce_axis_map: key:reduce_axis_index, value:reduce_axis_var
        # reduce_index_map: key:reduce_axis_index in original index,
        #                   value:reduce_axis_index in reduce axis
        self._reduce_info = {"reduce_tensor": None,
                             "reduce_axis_map": {},
                             "reduce_axis_index": [],
                             "reduce_index_map": [],
                             "shape_before_reduce": None,
                             "keep_dims": True,
                             "dtype": None}

        self._reduce_tiling_para = {
            "block_tiling": {"tiling_tensor": None, "axis": 0, "axis_var": None,
                             "factor": 1},
            "ub_tiling": [{"tiling_tensor": None, "axis": 0, "axis_var": None,
                           "factor": 1}]}

        self._reduce_tiling_result = {"block_tiling": {}, "ub_tiling": [{}]}

    # pylint: disable=too-many-locals
    def _construct_compute_graph(self, out_tensors, spec_node_list):
        """
        record relate context imformations of operations

        """
        # find the last out tensor
        mid_output_tensors_dst_tensor = {}
        last_output_tensor = None
        last_output_tensors = []
        if hasattr(out_tensors, 'index'):
            if len(out_tensors) > 1:
                self._is_muti_output = True
                util.get_dst_tensor_map(out_tensors,
                                        mid_output_tensors_dst_tensor)
                for out in out_tensors:
                    if out not in mid_output_tensors_dst_tensor.keys():
                        last_output_tensors.append(out)
                        if last_output_tensor is None:
                            last_output_tensor = out
            else:
                last_output_tensor = out_tensors[0]
                last_output_tensors.append(out_tensors[0])
        else:
            last_output_tensor = out_tensors
            last_output_tensors.append(out_tensors[0])

        # record tensor list and tensor->dst_tensor(next_tensor) map
        visited_list = []
        tensor_list = []

        visited_list.append(last_output_tensor)
        tensor_list.append(last_output_tensor)
        self.__gen_reversed_subgraph_list(last_output_tensor, tensor_list,
                                          visited_list)

        # tensor classification
        for tensor in tensor_list:
            if isinstance(tensor.op, tvm.tensor.PlaceholderOp) or tensor in self._spec_node_list:
                self._input_tensors.append(tensor)
                if tensor in self._tensor_dst_tensor_map.keys():
                    self._input_tensor_dst_tensor_map[tensor] = \
                        self._tensor_dst_tensor_map[tensor]
            else:
                if tensor.op.tag.find("reduce") != -1:
                    self._reduce_tensors.append(tensor)
                if tensor.op.tag.find("broadcast") != -1:
                    if tensor.op.tag == "broadcast_for_tensor":
                        self._broadcast_tensors.append(tensor)
                    else:
                        self._vector_dup_tensors.append(tensor)
                if tensor in out_tensors:
                    if tensor in self._tensor_dst_tensor_map.keys():
                        self._mid_output_tensors.append(tensor)
                        self._mid_output_tensors_dst_tensor_map[tensor] = \
                            self._tensor_dst_tensor_map[tensor]
                        self._mid_tensors.append(tensor)
                else:
                    self._mid_tensors.append(tensor)

        self._last_output_tensors = last_output_tensors

        is_supported = self._check_pattern_supported()
        if not is_supported:
            return False

        self._res_tensor = self._last_output_tensors[0]
        self._record_reduce_info(self._res_tensor)

        # record info in order to calculate ub tiling
        for tensor in reversed(tensor_list):
            tmp_op = self.__split_tensor(tensor)
            if tmp_op["effective_op"]:
                self._op.append(tmp_op)
            self._origin_op.append(tmp_op)

        is_supported = self._check_supported()
        if not is_supported:
            return False

        self._record_broadcast_info()
        # calculate cache_write_exclude_tensors
        for i in self._broadcast_not_last_axis_tensors:
            self._cache_write_exclude_tensors.append(i)

        for i in self._broadcast_scalars:
            dst_tensors = self._broadcast_scalar_dst_tensor_map[i]
            if self._support_tensor_scaler_operate(dst_tensors):
                self._cache_write_exclude_tensors.append(i)

        self._get_max_ub_count()
        self._get_total_width()

        return True

    def _check_supported(self):
        """
        :return: Bool
        """
        is_supported = self._check_broadcast_supported()
        if not is_supported:
            return False
        is_supported = self._is_supported_atomic_add()
        if not is_supported:
            return False
        is_success = self._select_tiling_case()
        if not is_success:
            return False

        return True

    def _check_pattern_supported(self):
        """
        :return: Bool
        """
        if len(self._last_output_tensors) > 1:
            for tensor in self._last_output_tensors:
                if tensor.op.tag.find("tuple_reduce_sum") == -1:
                    return False
        else:
            if self._last_output_tensors[0].op.tag.find("reduce") == -1:
                return False
        for tensor in self._reduce_tensors:
            if tensor not in self._last_output_tensors:
                return False
        return True

    def _check_broadcast_supported(self):
        """
        :return: Bool
        """
        reduce_axis_index = self._reduce_info["reduce_axis_index"]
        if not reduce_axis_index:
            return False
        broadcast_index = self._find_broadcast_axis_index()
        for ele in broadcast_index:
            if ele not in reduce_axis_index:
                return False

        return True

    def _is_supported_atomic_add(self):
        """
        :return: Bool
        """
        # cloud, fp32
        reduce_tensor = self._reduce_info["reduce_tensor"]
        if reduce_tensor is None:
            return False
        dtype = reduce_tensor.dtype
        if dtype != "float32":
            return False
        product = cce_conf.get_product()
        if not product.startswith("1.60"):
            return False
        tag = reduce_tensor.op.tag
        if tag.find("sum") != -1:
            return True
        return False

    # pylint: disable=too-many-locals
    def _select_tiling_case(self):

        shape_before_reduce = self._reduce_info["shape_before_reduce"]
        reduce_axis_index = self._reduce_info["reduce_axis_index"]
        dtype = self._reduce_info["dtype"]

        def _is_not_reduce_for_one(shape_before_reduce):
            # for case (a , b, 1, 1) axis[1]
            for i in range(reduce_axis_index[-1] + 1, len(shape_before_reduce)):
                if shape_before_reduce[i] != 1:
                    break
                if i == len(shape_before_reduce) - 1:
                    return False

            # for case (a ,b , 1, 1) axis[2, 3]
            if len(reduce_axis_index) == 1 and \
                    len(shape_before_reduce) - 1 in reduce_axis_index and \
                    shape_before_reduce[-1] == 1:
                return False
            if len(shape_before_reduce) - 1 == reduce_axis_index[-1] and \
                    shape_before_reduce[-1] == 1:
                for i in range(len(reduce_axis_index) - 2, -1, -1):
                    if reduce_axis_index[i + 1] - reduce_axis_index[i] == 1 and \
                            shape_before_reduce[reduce_axis_index[i]] == 1:
                        continue
                    else:
                        return False
            return True

        def __is_reduce_for_one_and_not_tuple_sum():
            if len(self._last_output_tensors) > 1:
                for tensor in self._last_output_tensors:
                    if tensor.op.tag.find("tuple_reduce_sum") == -1:
                        if not _is_not_reduce_for_one(shape_before_reduce):
                            return False
            else:
                if not _is_not_reduce_for_one(shape_before_reduce):
                    return False
            return True

        if not __is_reduce_for_one_and_not_tuple_sum():
            return False

        if self._is_reduce_all_axis(shape_before_reduce, reduce_axis_index):
            return True

        def _do_mix_reduce_nlast_and_nlast():
            # reorder (ak+1,rk,..,r2,a2,r1,a1) to (ak+1,ak,...,a2, rk,..,r2,,r1,a1)
            reordered_shape, _, orignal_to_reorder_axis_map = \
                self._reorder_reduce_nlast_shape(shape_before_reduce, reduce_axis_index)
            # if (ak+1)*ak*..*a2 > core_num and not tuple reduce sum,
            # do not need atomic
            first_reduce_axis = orignal_to_reorder_axis_map[reduce_axis_index[0]]
            size = 1
            for i in range(0, first_reduce_axis):
                size = size * reordered_shape[i]
            core_num = cceconf.get_soc_spec("CORE_NUM")
            # not tuple reduce sum
            if size >= core_num and len(self._last_output_tensors) < 2:
                return False
            # if (ak+1)*ak*..*a2 > rk*..*r2*r1, do not need atomic
            reduce_size = 1
            for i in reduce_axis_index:
                # pylint: disable=unsubscriptable-object
                reduce_size = reduce_size * shape_before_reduce[i]
            if size >= reduce_size:
                return False
            last_axis_size = self._get_nlast_axis_size(shape_before_reduce, reduce_axis_index)
            self._last_axis_size = last_axis_size
            if self._is_last_axis_large_than_ub_max(last_axis_size, dtype):
                return False
            # if a1 is not 32B align, check whether support storage align
            if (not self._is_last_axis_32b_align(last_axis_size, dtype)) and \
                    (not self._support_storage_align()):
                return False

            return True

        if self._is_mix_reduce_nlast_and_nlast(shape_before_reduce, reduce_axis_index):
            return _do_mix_reduce_nlast_and_nlast()


        def _do_mix_reduce_nlast_and_last():
            # block tiling do not split r1
            # (ak,rk,..,a2,r2,a1,r1)
            # use (rk,,..,r2,r1) to do block tiling
            to_do_block_tiling_shape = []
            for i in reduce_axis_index:
                # pylint: disable=unsubscriptable-object
                to_do_block_tiling_shape.append(shape_before_reduce[i])
            split_axis, _, _ = self._get_block_tiling(
                to_do_block_tiling_shape, dtype)
            block_split_axis = self._reduce_info["reduce_axis_index"][
                split_axis]
            r1_start_index, r1_end_index = self._find_last_reduce_axis(
                shape_before_reduce, reduce_axis_index)
            if block_split_axis >= r1_start_index:
                return False

            # if ak*..*a2*a1/core_num > rk*..*r2*r2, do not need atomic
            none_reduce_size = 1
            for i, _ in enumerate(shape_before_reduce):
                if i not in reduce_axis_index:
                    # pylint: disable=unsubscriptable-object
                    none_reduce_size = none_reduce_size * shape_before_reduce[i]
            size = 1
            for i in range(0, r1_start_index):
                if i in reduce_axis_index:
                    # pylint: disable=unsubscriptable-object
                    size = size * shape_before_reduce[i]
            core_num = cceconf.get_soc_spec("CORE_NUM")
            if none_reduce_size / core_num > size:
                return False

            # r1 large than ub_max or r1 is 32b align
            last_axis_size = 1
            for i in range(r1_start_index, r1_end_index + 1):
                # pylint: disable=unsubscriptable-object
                last_axis_size = last_axis_size * shape_before_reduce[i]
            if self._is_last_axis_large_than_ub_max(last_axis_size, dtype):
                return True
            if not self._is_last_axis_32b_align(last_axis_size, dtype):
                return False
            return True

        if self._is_mix_reduce_nlast_and_last(shape_before_reduce, reduce_axis_index):
            return _do_mix_reduce_nlast_and_last()

        return False

    def _get_nlast_axis_size(self, shape_before_reduce, reduce_axis_index):
        last_none_reduce_index = len(shape_before_reduce) - 1
        for i in range(len(shape_before_reduce) - 1, -1, -1):
            if shape_before_reduce[i] != 1:
                if i in reduce_axis_index:
                    last_none_reduce_index = i + 1
                    break
        size = 1
        for i in range(last_none_reduce_index, len(shape_before_reduce)):
            size = size * shape_before_reduce[i]

        return size

    def _record_broadcast_info(self):
        """
        :return:
        """
        for tensor in self._broadcast_tensors:
            if self._is_broadcast_orignal_scalar(tensor):
                self._broadcast_scalars.append(tensor)
                if tensor in self._tensor_dst_tensor_map.keys():
                    dst_tensor = self._tensor_dst_tensor_map[tensor]
                    self._map_apend(self._broadcast_scalar_dst_tensor_map,
                                    tensor, dst_tensor)
            elif self._is_broadcast_not_last_axis_tensor(tensor):
                self._broadcast_not_last_axis_tensors.append(tensor)

    def _support_tensor_scaler_operate(self, tensors):
        """
        Judge if tensor supports scalar instruction operations

        Parameters:
        ----------
        tensors :  input tensor

        Returns
        -------
        Bool: True or False
        """
        flag = True
        for tensor in tensors:
            tag = tensor.op.tag
            if tag.find("|") != -1:
                str_list = tag.split("|")
                tag = str_list[0]
            operator = tag
            if operator not in self._tensor_scaler_operator:
                flag = False

        return flag

    def _find_broadcast_axis_index(self):
        """
        :return:
        """
        index = []
        shape_before_reduce = self._reduce_info["shape_before_reduce"]
        for tensor in self._broadcast_tensors:
            shape = self._shape_to_list(tensor.op.input_tensors[0].shape)
            for i, ele in enumerate(shape):
                # pylint: disable=unsubscriptable-object
                if ele != shape_before_reduce[i]:
                    if ele not in index:
                        index.append(i)

        return index

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
                broadcast_before = self._shape_to_list(
                    tensor.op.input_tensors[0].shape)
                shape = self._shape_to_list(tensor.shape)
                for i in range(len(shape) - 1, -1, -1):
                    if shape[i] != 1:
                        return broadcast_before[i] == shape[i] and i != 0
        return False

    def _is_broadcast_orignal_scalar(self, tensor):
        """
        Check if the scaler broadcast scene

        Parameters:
        ----------
        tensors :  tensor to be checked

        Returns
        -------
        Bool: True or False
        """
        if tensor.op.tag == "broadcast_for_tensor":
            # broadcast scalar
            if list(tensor.op.input_tensors):
                shape = self._shape_to_list(tensor.op.input_tensors[0].shape)
                flag = True
                for i in range(0, len(shape), 1):
                    if shape[i] != 1:
                        flag = False
                        break
                return flag
        return False

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
        for in_tensor in list(tensor.op.input_tensors):
            self._map_apend(self._tensor_dst_tensor_map, in_tensor,
                            tensor)
            if in_tensor not in visited_list:
                visited_list.append(in_tensor)
                tensor_list.append(in_tensor)
            if in_tensor in self._spec_node_list:
                continue

            self.__gen_reversed_subgraph_list(in_tensor, tensor_list,
                                              visited_list)

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

        for i in self._mid_output_tensors:
            self._map_apend(self._cache_read_tensors_and_readers_map, i,
                            self._mid_output_tensors_dst_tensor_map[i])

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

    def _is_reduce_all_axis(self, shape_before_reduce, reduce_axis_index):
        """
        :return:
        """
        # (1,1..,r1..rk,1,1)
        for i, _ in enumerate(shape_before_reduce):
            if i not in reduce_axis_index:
                if shape_before_reduce[i] != 1:
                    return False
        return True

    def _is_reduce_last_axis(self, shape_before_reduce, reduce_axis_index):
        """
        :param shape_before_reduce:
        :param reduce_axis_index:
        :return:
        """
        # (a,r1), (a,r1..rk), (a,r1..rk,1,1)
        if len(reduce_axis_index) > 1:
            has_last_reduce_axis = \
                ((len(shape_before_reduce) - 1) in reduce_axis_index)
            if has_last_reduce_axis:
                is_continuous_reduce = self._is_continuous_reduce(reduce_axis_index)
                return is_continuous_reduce

            for i in range(reduce_axis_index[-1] + 1, len(shape_before_reduce)):
                if shape_before_reduce[i] != 1:
                    return False
            return True

        has_last_reduce_axis = ((len(shape_before_reduce) - 1) in reduce_axis_index)
        if has_last_reduce_axis:
            return True

        for i in range(reduce_axis_index[-1] + 1, len(shape_before_reduce)):
            if shape_before_reduce[i] != 1:
                return False
        return True

    def _is_mix_reduce_nlast_and_last(self, shape_before_reduce, reduce_axis_index):
        """
        :param shape_before_reduce:
        :param reduce_axis_index:
        :return:
        """
        if self._is_reduce_last_axis(shape_before_reduce, reduce_axis_index):
            return True
        has_last_reduce_axis = \
            ((len(shape_before_reduce) - 1) in reduce_axis_index)

        if not has_last_reduce_axis:
            has_last_reduce_axis = True
            for i in range(reduce_axis_index[-1] + 1, len(shape_before_reduce)):
                if shape_before_reduce[i] != 1:
                    has_last_reduce_axis = False
        has_not_last_reduce_axis = \
            (reduce_axis_index[0] in range(0, len(shape_before_reduce)-1))

        return has_last_reduce_axis and has_not_last_reduce_axis

    def _is_mix_reduce_nlast_and_nlast(self, shape_before_reduce, reduce_axis_index):
        """
        :param shape_before_reduce:
        :param reduce_axis_index:
        :return:
        """
        last_reduce_axis_index = -1
        for i in range(len(reduce_axis_index) - 1, -1, -1):
            index = reduce_axis_index[i]
            if shape_before_reduce[index] != 1:
                last_reduce_axis_index = index
                break
        if last_reduce_axis_index == -1:
            return False
        is_not_last_reduce_axis = False
        for i in range(last_reduce_axis_index, len(shape_before_reduce)):
            if i not in reduce_axis_index and shape_before_reduce[i] != 1:
                is_not_last_reduce_axis = True
                break
        return is_not_last_reduce_axis

    def _is_continuous_reduce(self, reduce_axis_index):
        """
        :param reduce_axis_index:
        :return:
        """
        for i, _ in enumerate(reduce_axis_index):
            if i > 0:
                if reduce_axis_index[i] != reduce_axis_index[i-1] + 1:
                    return False
        return True

    def _is_last_axis_32b_align(self, last_axis_size, dtype):
        """
        :param last_axis_size:
        :param dtype:
        :return:
        """
        # last_axis_size include continuous last axis
        float16_size = 2
        block_size = 32
        data_size = DTYPE_WIDTH_MAP[dtype] * float16_size * last_axis_size
        if not (data_size >= block_size and data_size % block_size == 0):
            return False

        return True

    def _is_last_axis_large_than_ub_max(self, last_axis_size, dtype):
        """
        :param last_axis_size:
        :param dtype:
        :return:
        """
        data_size_align_to_fp16 = DTYPE_WIDTH_MAP[dtype] * last_axis_size
        # max_ub_count align to fp16
        max_ub_count = self._get_max_ub_count()
        return data_size_align_to_fp16 > max_ub_count

    def _elewise_tiling(self, shape, dtype, max_ub_count, is_32b_align=True):
        """
        :param shape:
        :param dtype:
        :param max_ub_count:
        :return:
        """
        if self._need_multi_core:
            if is_32b_align:
                block_split_axis, block_split_inner, _ = self._get_block_tiling(
                    shape, dtype)
            else:
                multi_core_threshold = 256
                block_split_axis, block_split_inner, _ = self._get_block_tiling(
                    shape, dtype, multi_core_threshold)
        else:
            block_split_axis = 0
            block_split_inner = shape[block_split_axis]

        ub_split_axis, ub_split_inner = self._get_ub_tiling(
            shape, block_split_axis, block_split_inner,
            max_ub_count)

        if self._need_multi_core and self._is_need_modify_block_and_ub_tiling(
                shape, dtype, block_split_axis, block_split_inner,
                ub_split_axis, ub_split_inner, max_ub_count):
            block_split_axis = 0
            block_split_inner = shape[block_split_axis]
            ub_split_axis, ub_split_inner = self._get_ub_tiling(
                shape, block_split_axis, block_split_inner,
                max_ub_count)

        return block_split_axis, block_split_inner, ub_split_axis, ub_split_inner


    def _reorder_reduce_nlast_shape(self, shape_before_reduce, reduce_axis_index):
        """
        reorder shape (ak+1,rk,..,r2,a2,r1,a1) to (ak+1,ak,...,a2, rk,..,r2,,r1,a1)
        :param shape_before_reduce: (ak+1,rk,..,r2,a2,r1,a1)
        :param reduce_axis_index:
        :return:
        """
        # shape_before_reduce: (ak+1,rk,..,r2,a2,r1,a1)
        # find the last none-reduce axis a1

        a1_start_index, _ = self._find_last_none_reduce_axis(shape_before_reduce, reduce_axis_index)

        last_none_reduce_axis = a1_start_index

        orignal_to_reorder_axis_map = {}
        reorder_to_orignal_axis_map = {}
        #  (ak+1,ak,...,a2, rk,..,r2,,r1,a1)
        reordered_shape = list(shape_before_reduce)
        temp_axis = last_none_reduce_axis - 1
        for i in range(len(reduce_axis_index) - 1, -1, -1):
            reordered_shape[temp_axis] = shape_before_reduce[
                reduce_axis_index[i]]
            reorder_to_orignal_axis_map[temp_axis] = reduce_axis_index[i]
            orignal_to_reorder_axis_map[reduce_axis_index[i]] = temp_axis
            temp_axis = temp_axis - 1
        for i in range(last_none_reduce_axis - 1, -1, -1):
            if i not in reduce_axis_index:
                reordered_shape[temp_axis] = shape_before_reduce[i]
                reorder_to_orignal_axis_map[temp_axis] = i
                orignal_to_reorder_axis_map[i] = temp_axis
                temp_axis = temp_axis - 1

        for i in range(last_none_reduce_axis, len(shape_before_reduce)):
            reorder_to_orignal_axis_map[i] = i
            orignal_to_reorder_axis_map[i] = i

        return reordered_shape, reorder_to_orignal_axis_map, orignal_to_reorder_axis_map

    def _support_storage_align(self):
        # do not support mid output
        if self._mid_output_tensors:
            return False
        if not self._need_multi_core:
            return False
        if self._broadcast_tensors:
            return False
        return True

    def _do_storage_align(self, shape_before_reduce, reduce_axis_index):
        """
        :param hape_before_reduce:
        :param reduce_axis_index:
        :return:
        """
        a1_start_index, _ = self._find_last_none_reduce_axis(
            shape_before_reduce,
            reduce_axis_index)
        align_axis = a1_start_index - 1

        align_type = self._res_tensor.dtype
        # bool is represented by int8
        if align_type == 'bool':
            align_type = 'int8'
        align_factor, _ = util.get_align_factor(align_type)

        for key in self._cache_read_tensors_and_buffer_map:
            cache_read_buffer = self._cache_read_tensors_and_buffer_map[key]
            self._schedule[cache_read_buffer].storage_align(
                cache_read_buffer.op.axis[align_axis], align_factor, 0)

        for key in self._cache_write_tensors_and_buffer_map:
            if key != self._res_tensor:
                cache_write_buffer = self._cache_write_tensors_and_buffer_map[key]
                self._schedule[cache_write_buffer].storage_align(
                    cache_write_buffer.op.axis[align_axis], align_factor, 0)

    # pylint: disable=too-many-locals
    def _reduce_not_last_atomic_tiling(self, shape_before_reduce,
                                       reduce_axis_index, max_ub_count, dtype):
        """
        :param shape_before_reduce:
        :param reduce_axis_index:
        :param max_ub_count:
        :param dtype:
        :return:
        """
        def _shape_mul(shape):
            if not shape:
                return 1
            return reduceIns(lambda x, y: x*y, shape)

        def update_ub_max_by_align(max_ub_count):
            a1_start_index, _ = self._find_last_none_reduce_axis(
                shape_before_reduce,
                reduce_axis_index)

            align_type = self._res_tensor.dtype
            # bool is represented by int8
            if align_type == 'bool':
                align_type = 'int8'
            align_factor, _ = util.get_align_factor(align_type)

            total_size_of_reduce = _shape_mul(shape_before_reduce[a1_start_index:])

            max_ub_count = int(total_size_of_reduce / \
                               (total_size_of_reduce + align_factor - \
                                total_size_of_reduce % align_factor) * max_ub_count)

            return max_ub_count

        # reorder (ak+1,rk,..,r2,a2,r1,a1) to (ak+1,ak,...,a2, rk,..,r2,,r1,a1)
        reordered_shape, reorder_to_orignal_axis_map, orignal_to_reorder_axis_map = \
            self._reorder_reduce_nlast_shape(shape_before_reduce, reduce_axis_index)

        # rk
        first_reduce_axis = orignal_to_reorder_axis_map[reduce_axis_index[0]]
        # use (1,1,..,1,rk,..,r2,,r1,a1) to do tiling
        to_do_tiling_shape = [1] * first_reduce_axis + reordered_shape[first_reduce_axis:]

        last_axis_size = self._get_nlast_axis_size(shape_before_reduce, reduce_axis_index)
        is_32b_align = self._is_last_axis_32b_align(last_axis_size, dtype)
        self._is_32b_align = is_32b_align
        self._last_axis_size = last_axis_size
        if not is_32b_align:
            if self._support_storage_align():
                max_ub_count = update_ub_max_by_align(max_ub_count)
        reorder_block_tiling_axis, block_inner, reorder_ub_tiling_axis, ub_inner = \
            self._elewise_tiling(to_do_tiling_shape, dtype, max_ub_count, is_32b_align)

        self._is_multi_core_need_fused = \
            self._is_need_fused(to_do_tiling_shape, reorder_block_tiling_axis)

        if reorder_block_tiling_axis < first_reduce_axis:
            block_split_axis = reduce_axis_index[0]
            block_inner = shape_before_reduce[reduce_axis_index[0]]
            reorder_block_tiling_axis = first_reduce_axis

        else:
            block_split_axis = reorder_to_orignal_axis_map[
                reorder_block_tiling_axis]
        if reorder_ub_tiling_axis < first_reduce_axis:
            ub_split_axis = reduce_axis_index[0]
            ub_inner = shape_before_reduce[reduce_axis_index[0]]
            reorder_ub_tiling_axis = first_reduce_axis
        else:
            ub_split_axis = reorder_to_orignal_axis_map[reorder_ub_tiling_axis]

        a1_start_index, a1_end_index = self._find_last_none_reduce_axis(shape_before_reduce,
                                                                        reduce_axis_index)
        # if block split a1, adjust block to split r1
        if block_split_axis >= a1_start_index:
            block_split_axis = a1_start_index - 1
            block_inner = 1
            ub_split_axis = a1_start_index - 1
            ub_inner = 1

        if not is_32b_align:
            if self._support_storage_align():
                self._do_storage_align(shape_before_reduce, reduce_axis_index)

        self._need_db = self._is_need_double_buffer(reordered_shape,
                                                    first_reduce_axis,
                                                    reorder_block_tiling_axis,
                                                    block_inner,
                                                    reorder_ub_tiling_axis,
                                                    ub_inner)

        las_axis_size = 1
        for i in range(a1_start_index, a1_end_index + 1):
            las_axis_size *= shape_before_reduce[i]

        loop_size = 1
        for i in range(reorder_ub_tiling_axis + 1, len(to_do_tiling_shape)):
            loop_size *= to_do_tiling_shape[i]
        loop_size *= ub_inner

        self._is_need_dichotomy_add = self._need_dichotomy_add(loop_size, las_axis_size, dtype)


        return block_split_axis, block_inner, ub_split_axis, ub_inner

    def _need_dichotomy_add(self, loop_size, las_axis_size, dtype):
        if dtype == "float16":
            vector_inst_one_repeat_size = 128
            dtype_size = 2
        elif dtype == "float32":
            vector_inst_one_repeat_size = 64
            dtype_size = 4
        else:
            return False
        block_size = 32

        if las_axis_size > vector_inst_one_repeat_size:
            return False
        if vector_inst_one_repeat_size % las_axis_size != 0:
            return False
        if las_axis_size * dtype_size % block_size != 0:
            return False
        if loop_size // vector_inst_one_repeat_size < 2:
            return False

        return True

    def _is_need_double_buffer(self, shape, block_start_axis, block_end_axis, block_inner,
                               ub_split_axis, ub_inner):

        loop = 1
        for i in range(0, block_start_axis):
            loop *= shape[i]
        if loop > 2:
            return True
        if block_end_axis == ub_split_axis:
            loop *= block_inner // ub_inner
        else:
            for i in range(block_end_axis + 1, ub_split_axis):
                loop *= shape[i]
            loop *= shape[ub_split_axis] // ub_inner
        if loop > 2:
            self._need_db = True
            return True
        return False

    def _find_last_none_reduce_axis(self, shape_before_reduce, reduce_axis_index):
        """
        :param shape_before_reduce:
        :param reduce_axis_index:
        :return:
        """
        # shape_before_reduce:(ak+1,rk,..,r2,a2,r1,a1) or (ak,rk,..,r2,a1,r1),
        # find a1 position, a1 may contain continues axis

        def __find_special_last_none_reduce_axis(a1_start_index, a1_end_index):
            if self._tiling_case == 1:
                for i in range(len(shape_before_reduce) - 1, -1, -1):
                    if shape_before_reduce[i] == 1 and i not in reduce_axis_index:
                        a1_end_index = i
                        a1_start_index = a1_end_index
                        break
                for i in range(a1_end_index, -1, -1):
                    if shape_before_reduce[i] == 1 and i not in reduce_axis_index:
                        a1_start_index = i
                        break
            return a1_start_index, a1_end_index

        a1_end_index = 0
        for i in range(len(shape_before_reduce) - 1, -1, -1):
            if self._is_mix_reduce_nlast_and_last(shape_before_reduce, reduce_axis_index):
                if (i not in reduce_axis_index and i < reduce_axis_index[-1]) or \
                        (i in reduce_axis_index and shape_before_reduce[i] == 1):
                    a1_end_index = i
                    break
            else:
                if i not in reduce_axis_index or \
                        (i in reduce_axis_index and shape_before_reduce[i] == 1):
                    a1_end_index = i
                    break

        a1_start_index = a1_end_index
        for i in range(a1_end_index, -1, -1):
            if i in reduce_axis_index:
                a1_start_index = i + 1
                break

        a1_start_index, a1_end_index = \
            __find_special_last_none_reduce_axis(a1_start_index, a1_end_index)

        return a1_start_index, a1_end_index

    @staticmethod
    def _find_last_reduce_axis(shape_before_reduce, reduce_axis_index):
        """
        :param shape_before_reduce:
        :param reduce_axis_index:
        :return:
        """
        # shape_before_reduce:(ak+1,rk,..,r2,a2,r1,a1) or (ak,rk,..,r2,a1,r1),
        # find r1 position, r1 may contain continues axis
        r1_end_index = 0
        for i in range(len(shape_before_reduce) - 1, -1, -1):
            if shape_before_reduce[i] != 1:
                if i in reduce_axis_index:
                    r1_end_index = i
                    break
        r1_start_index = r1_end_index
        for i in range(r1_end_index, -1, -1):
            if i not in reduce_axis_index:
                r1_start_index = i + 1
                break

        return r1_start_index, r1_end_index

    # pylint: disable=too-many-locals
    @staticmethod
    def _reorder_reduce_last_shape(shape_before_reduce, reduce_axis_index):
        """
        reorder shape (ak,rk,..,r2,a1,r1) to (ak,...,a2,rk,..,r2,a1,r1)
        :param shape_before_reduce:
        :param reduce_axis_index:
        :return:
        """
        # shape_before_reduce: (ak,rk,..,r2,a1,r1)
        # find the last none-reduce axis a1
        last_none_reduce_axis = len(shape_before_reduce) - 1
        for i in range(len(shape_before_reduce) - 1, -1, -1):
            if shape_before_reduce[i] != 1:
                if i in reduce_axis_index:
                    last_none_reduce_axis = i + 1
                    break

        orignal_to_reorder_axis_map = {}
        reorder_to_orignal_axis_map = {}
        #  (ak,...,a2,rk,..,r2,a1,r1)
        reordered_shape = list(shape_before_reduce)
        temp_axis = last_none_reduce_axis - 1
        for i in range(len(reduce_axis_index) - 1, -1, -1):
            if shape_before_reduce[reduce_axis_index[i]] != 1:
                reordered_shape[temp_axis] = shape_before_reduce[
                    reduce_axis_index[i]]
                reorder_to_orignal_axis_map[temp_axis] = reduce_axis_index[i]
                orignal_to_reorder_axis_map[reduce_axis_index[i]] = temp_axis
                temp_axis = temp_axis - 1
            else:
                reorder_to_orignal_axis_map[i] = i
                orignal_to_reorder_axis_map[i] = i
        for i in range(last_none_reduce_axis - 1, -1, -1):
            if i not in reduce_axis_index:
                reordered_shape[temp_axis] = shape_before_reduce[i]
                reorder_to_orignal_axis_map[temp_axis] = i
                orignal_to_reorder_axis_map[i] = temp_axis
                temp_axis = temp_axis - 1

        for i in range(last_none_reduce_axis, len(shape_before_reduce)):
            reorder_to_orignal_axis_map[i] = i
            orignal_to_reorder_axis_map[i] = i

        return reordered_shape, reorder_to_orignal_axis_map, \
               orignal_to_reorder_axis_map

    # pylint: disable=too-many-locals
    def _reduce_last_axis_32b_align_atomic_tiling(self, shape_before_reduce,
                                                  reduce_axis_index,
                                                  max_ub_count, dtype):
        """
        :param shape_before_reduce: (ak,rk,..,r2,a1,r1)
        :param reduce_axis_index:
        :param max_ub_count:
        :param dtype:
        :return:
        """
        # use (rk..,r2,r1) to do block tiling
        to_do_block_tiling_shape = []
        for i in reduce_axis_index:
            to_do_block_tiling_shape.append(shape_before_reduce[i])

        if self._need_multi_core:
            split_axis, block_split_inner, _ = self._get_block_tiling(
                to_do_block_tiling_shape, dtype)
            block_split_axis = self._reduce_info["reduce_axis_index"][split_axis]
            self._is_multi_core_need_fused = \
                self._is_need_fused(to_do_block_tiling_shape, split_axis)
        else:
            split_axis = 0
            block_split_axis = 0
            block_split_inner = shape_before_reduce[block_split_axis]

        # shape_before_reduce:(ak,rk,..,r2,a1,r1), find a1 position,
        # a1 may contain continues axis
        a1_start_index, a1_end_index = \
            self._find_last_none_reduce_axis(shape_before_reduce, reduce_axis_index)

        # if block tiling do not spit r1
        if block_split_axis < a1_start_index:
            # block tiling split (rk,..,rb,..,r2,r1) to (rk,..,rbo,rbi,..,r2,r1)
            # use (rbi,..,r2,a1,r1) to do ub tiling
            reorder_to_orignal_map = {}
            to_do_ub_tiling_shape = []
            count = 0
            for i in range(0, split_axis):
                axis_size = shape_before_reduce[reduce_axis_index[i]]
                to_do_ub_tiling_shape.append(axis_size)
                reorder_to_orignal_map[count] = reduce_axis_index[i]
                count = count + 1

            for i in range(block_split_axis, a1_start_index):
                if i in reduce_axis_index:
                    to_do_ub_tiling_shape.append(shape_before_reduce[i])
                    reorder_to_orignal_map[count] = i
                    count = count + 1

            for i in range(a1_start_index, a1_end_index + 1):
                to_do_ub_tiling_shape.append(shape_before_reduce[i])
                reorder_to_orignal_map[count] = i
                count = count + 1

            for i in range(a1_end_index + 1, len(shape_before_reduce)):
                if i in reduce_axis_index:
                    to_do_ub_tiling_shape.append(shape_before_reduce[i])
                    reorder_to_orignal_map[count] = i
                    count = count + 1

            ub_split_temp_axis, ub_split_inner = self._get_ub_tiling(
                to_do_ub_tiling_shape, split_axis, block_split_inner,
                max_ub_count)

            self._need_db = self._is_need_double_buffer(to_do_ub_tiling_shape,
                                                        0,
                                                        split_axis,
                                                        block_split_inner,
                                                        ub_split_temp_axis,
                                                        ub_split_inner)
            if not self._need_db:
                # ak*..*k2
                loop = 1
                for i in range(0, a1_start_index):
                    if i not in reduce_axis_index:
                        loop *= shape_before_reduce[i]
                self._need_db = loop > 2

            ub_split_axis = reorder_to_orignal_map[ub_split_temp_axis]

            return block_split_axis, block_split_inner, ub_split_axis, ub_split_inner
        return None, None, None, None

    def _is_need_fused(self, shape, block_tiling_axis):
        if block_tiling_axis == 0:
            return False
        for i in range(0, block_tiling_axis):
            if shape[i] != 1:
                return True
        return False

    def _reduce_all_axis_tiling(self, max_ub_count):
        """
        :param max_ub_count:
        :return:
        """
        reduce_tensor = self._reduce_info["reduce_tensor"]
        dtype = reduce_tensor.dtype
        shape_before_reduce = self._reduce_info["shape_before_reduce"]
        reduce_axis_index = self._reduce_info["reduce_axis_index"]
        to_do_tiling_shape = []
        for i, _ in enumerate(reduce_axis_index):
            # pylint: disable=unsubscriptable-object
            to_do_tiling_shape.append(shape_before_reduce[reduce_axis_index[i]])

        if self._is_supported_atomic_add():
            block_split_axis, block_split_inner, ub_split_axis, ub_split_inner = \
                self._elewise_tiling(to_do_tiling_shape, dtype, max_ub_count)

            block_split_axis = reduce_axis_index[block_split_axis]
            ub_split_axis = reduce_axis_index[ub_split_axis]

        else:
            block_split_axis = 0
            # pylint: disable=unsubscriptable-object
            block_split_inner = shape_before_reduce[block_split_axis]
            ub_split_axis, ub_split_inner = self._get_ub_tiling(
                shape_before_reduce, block_split_axis, block_split_inner,
                max_ub_count)

        self._need_db = self._is_need_double_buffer(shape_before_reduce,
                                                    0,
                                                    block_split_axis,
                                                    block_split_inner,
                                                    ub_split_axis,
                                                    ub_split_inner)

        self._is_multi_core_need_fused = \
            self._is_need_fused(shape_before_reduce, block_split_axis)

        return block_split_axis, block_split_inner, ub_split_axis, ub_split_inner

    # pylint: disable=too-many-locals
    def _calculate_tiling(self):
        """
        calculate tiling strategy

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        max_ub_count = self._get_max_ub_count()
        # if the cmp mode is bit the res shape is 8 muti input shape, we should
        # tiling as the input shape
        for i in range(len(self._op)):
            if self._op[i]["op"] == 'emit_insn_elewise_binary_cmp' \
                    and self._op[i]["args"][1] == 'bit':
                max_ub_count = max_ub_count // 8
                break

        shape_before_reduce = self._reduce_info["shape_before_reduce"]
        reduce_axis_index = self._reduce_info["reduce_axis_index"]
        dtype = self._reduce_info["dtype"]

        if self._is_reduce_all_axis(shape_before_reduce, reduce_axis_index):
            block_split_axis, block_inner, ub_split_axis, ub_inner = \
                self._reduce_all_axis_tiling(max_ub_count)
            self._tiling_case = 1

        elif self._is_mix_reduce_nlast_and_nlast(shape_before_reduce, reduce_axis_index):
            block_split_axis, block_inner, ub_split_axis, ub_inner = \
                self._reduce_not_last_atomic_tiling(
                    shape_before_reduce,
                    reduce_axis_index, max_ub_count, dtype)
            self._tiling_case = 2

        elif self._is_mix_reduce_nlast_and_last(shape_before_reduce, reduce_axis_index):
            block_split_axis, block_inner, ub_split_axis, ub_inner = \
                self._reduce_last_axis_32b_align_atomic_tiling(
                    shape_before_reduce, reduce_axis_index, max_ub_count, dtype)
            self._tiling_case = 3
        else:
            block_split_axis = 0
            # pylint: disable=unsubscriptable-object
            block_inner = shape_before_reduce[0]
            ub_split_axis = -1
            # pylint: disable=unsubscriptable-object
            ub_inner = shape_before_reduce[-1]

        res_tensor = self._res_tensor
        reduce_axis_index = self._reduce_info["reduce_axis_index"]
        if block_split_axis in reduce_axis_index:
            reduce_axis = self._reduce_info["reduce_axis_map"]
            axis_var = reduce_axis[block_split_axis]
            block_tiling_para = {"tiling_tensor": res_tensor,
                                 "axis": block_split_axis, "axis_var": axis_var,
                                 "factor": block_inner}
        else:
            block_tiling_para = {"tiling_tensor": res_tensor,
                                 "axis": block_split_axis, "axis_var": None,
                                 "factor": block_inner}
        # if ub tiling is performed along a certain reduce axis,
        # need to pass the reduce axis as the split itervar parameter
        if ub_split_axis in reduce_axis_index:
            reduce_axis = self._reduce_info["reduce_axis_map"]
            axis_var = reduce_axis[ub_split_axis]
            ub_tiling_para = [
                {"tiling_tensor": res_tensor, "axis": ub_split_axis,
                 "axis_var": axis_var, "factor": ub_inner}]
        else:
            ub_tiling_para = [
                {"tiling_tensor": res_tensor, "axis": ub_split_axis,
                 "axis_var": None, "factor": ub_inner}]

        self._reduce_tiling_para["block_tiling"] = block_tiling_para
        self._reduce_tiling_para["ub_tiling"] = ub_tiling_para

    def _do_tiling(self):
        """
        :return:
        """
        self._do_block_tiling()

        block_tiling_result = self._reduce_tiling_result["block_tiling"]
        block_tiling_tensor = block_tiling_result["tiling_tensor"]
        block_split_axis = block_tiling_result["axis"]
        res_block_outer = block_tiling_result["outer_itervar"]

        # atomic additional schedule
        if self._tiling_case > 0:
            self._atomic_additonal_schedule(block_tiling_tensor,
                                            block_split_axis, res_block_outer)

        self._do_ub_tiling()

        if self._tiling_case > 0:
            final_out_tensor_global = self._final_out_tensor_global
            final_out_tensor_ub_rf = self._final_out_tensor_ub_rf

            if self._tiling_case == 1:
                self._reorder_atomic_reduce_all(final_out_tensor_ub_rf,
                                                final_out_tensor_global)
            if self._tiling_case == 2:
                # reorder (a1,a2,..ak,rbo,r1,.,rb-1,rb+1,..rn,rbi) to
                # (rbo,a1,a2,..ak,r1,.rb-1,rbi,rb+1,,.rn)
                self._reorder_atomic_reduce_not_last_axis(final_out_tensor_ub_rf,
                                                          final_out_tensor_global)
            if self._tiling_case == 3:
                # reorder (a1,a2,..ak,rbo,r1,.,rb-1,rb+1,..rn,rbi) to
                # (rbo,a1,a2,..ak-1,r1,.rb-1,rbi,rb+1,,.,ak,rn)
                self._reorder_atomic_reduce_last_axis(final_out_tensor_ub_rf,
                                                      final_out_tensor_global)

    def _do_block_tiling(self):
        """
        :return:
        """
        block_tiling_para = self._reduce_tiling_para["block_tiling"]
        block_tiling_tensor = block_tiling_para["tiling_tensor"]
        block_split_axis = block_tiling_para["axis"]
        block_split_inner = block_tiling_para["factor"]

        if "axis_var" in block_tiling_para.keys() and \
                block_tiling_para["axis_var"] is not None:
            axis_var = block_tiling_para["axis_var"]
        else:
            axis_var = block_tiling_tensor.op.axis[block_split_axis]

        res_block_outer, res_block_inner = \
            self._schedule[block_tiling_tensor].split(axis_var,
                                                      factor=block_split_inner)
        block_tiling_result = {"tiling_tensor": block_tiling_tensor,
                               "axis": block_split_axis,
                               "parent_itervar": axis_var,
                               "outer_itervar": res_block_outer,
                               "inner_itervar": res_block_inner}
        self._reduce_tiling_result["block_tiling"] = block_tiling_result

    # pylint: disable=too-many-locals
    def _do_ub_tiling(self):
        """
        :return:
        """
        block_tiling_result = self._reduce_tiling_result["block_tiling"]
        block_tiling_tensor = block_tiling_result["tiling_tensor"]
        block_split_axis = block_tiling_result["axis"]
        res_block_inner = block_tiling_result["inner_itervar"]

        ub_tiling_result_list = []
        ub_tiling_para_list = self._reduce_tiling_para["ub_tiling"]
        ub_tiling_para = ub_tiling_para_list[0]
        ub_tiling_tensor = ub_tiling_para["tiling_tensor"]
        ub_split_axis = ub_tiling_para["axis"]
        ub_split_inner = ub_tiling_para["factor"]

        if ub_tiling_tensor is not None:
            if block_tiling_tensor is not None and block_split_axis == ub_split_axis \
                    and ub_tiling_tensor == block_tiling_tensor:
                res_ub_outer, res_ub_inner = self._schedule[
                    ub_tiling_tensor].split(res_block_inner,
                                            factor=ub_split_inner)
                ub_tiling_result = {"tiling_tensor": ub_tiling_tensor,
                                    "axis": ub_split_axis,
                                    "parent_itervar": res_block_inner,
                                    "outer_itervar": res_ub_outer,
                                    "inner_itervar": res_ub_inner}
            else:
                # if the axis_var is not empty,
                # the axis_var is used as the split parameter first,
                # otherwise the split_axis of the tilting_tensor is used as
                # the split parameter
                if "axis_var" in ub_tiling_para.keys() and \
                        ub_tiling_para["axis_var"] is not None:
                    axis_var = ub_tiling_para["axis_var"]
                else:
                    if self._tiling_case > 0:
                        shape_before_reduce = self._reduce_info["shape_before_reduce"]
                        reduce_axis_index = self._reduce_info["reduce_axis_index"]
                        none_reduce_index_map = self._find_none_reduce_axis_map(
                            shape_before_reduce, reduce_axis_index)
                        axis = none_reduce_index_map[ub_split_axis]
                        axis_var = ub_tiling_tensor.op.axis[axis]
                    else:
                        axis_var = ub_tiling_tensor.op.axis[ub_split_axis]
                if self._tiling_case > 0 and block_split_axis == ub_split_axis:
                    res_ub_outer, res_ub_inner = self._schedule[
                        ub_tiling_tensor].split(ub_tiling_tensor.op.reduce_axis[-1],
                                                factor=ub_split_inner)
                else:
                    res_ub_outer, res_ub_inner = self._schedule[
                        ub_tiling_tensor].split(axis_var, factor=ub_split_inner)

                ub_tiling_result = {"tiling_tensor": ub_tiling_tensor,
                                    "axis": ub_split_axis,
                                    "parent_itervar": axis_var,
                                    "outer_itervar": res_ub_outer,
                                    "inner_itervar": res_ub_inner}
            ub_tiling_result_list.append(ub_tiling_result)

        self._reduce_tiling_result["ub_tiling"] = ub_tiling_result_list

    def _atomic_additonal_schedule(self, block_tiling_tensor, block_split_axis, block_outer_var):
        """
        :param block_tiling_tensor:
        :param block_split_axis:
        :param block_outer_var:
        :return:
        """
        if self._is_multi_core_need_fused:
            fused_list = []
            reduce_index_map = self._reduce_info["reduce_index_map"]
            reduce_block_axis = reduce_index_map[block_split_axis]
            for i in range(0, reduce_block_axis):
                fused_list.append(block_tiling_tensor.op.reduce_axis[i])

            fused_list.append(block_outer_var)
            fused = self._schedule[block_tiling_tensor].fuse(*fused_list)
            final_out_tensor_ub_rf = self._schedule.rfactor(block_tiling_tensor,
                                                            fused,
                                                            factor_axis=-1)
        else:
            final_out_tensor_ub_rf = self._schedule.rfactor(block_tiling_tensor,
                                                            block_outer_var,
                                                            factor_axis=-1)
        if len(self._last_output_tensors) > 1:
            final_out_tensor_ub_rf = final_out_tensor_ub_rf[0]
        self._schedule[final_out_tensor_ub_rf].set_scope(cce.scope_ubuf)
        self._reduce_tiling_para["ub_tiling"][0]["tiling_tensor"] = \
            final_out_tensor_ub_rf
        final_out_tensor_global_list = self._schedule.cache_write(
            self._last_output_tensors, "")
        final_out_tensor_global = final_out_tensor_global_list[0]
        self._final_out_tensor_global = final_out_tensor_global
        self._final_out_tensor_ub_rf = final_out_tensor_ub_rf
        self.__replace_out_tensors(final_out_tensor_global_list)

    def __replace_out_tensors(self, final_out_tensor_global_list):
        """
        :param final_out_tensor_global_list:
        :return:
        """
        final_out_tensor_list_index = []
        for tensor in self._last_output_tensors:
            for i in range(0, len(self._out_tensors)):
                if tensor == self._out_tensors[i]:
                    final_out_tensor_list_index.append(i)
                    break
        for i, _ in enumerate(final_out_tensor_global_list):
            self._out_tensors[final_out_tensor_list_index[i]] = \
                final_out_tensor_global_list[i]

    def __get_offset(self, a1_start_index, a1_end_index, is_keep_dim, out_tensor_ub_rf):
        offset = 0
        if self._tiling_case == 2:
            offset = a1_end_index - a1_start_index
        if is_keep_dim:
            offset = offset + len(out_tensor_ub_rf.op.reduce_axis)
        return offset

    def _reorder_atomic_reduce_all(self, out_tensor_ub_rf, out_tensor_global):
        """
        :param out_tensor_ub_rf:
        :param out_tensor_global:
        :return:
        """
        block_tiling_result = self._reduce_tiling_result["block_tiling"]
        block_split_axis = block_tiling_result["axis"]

        ub_tiling_result_list = self._reduce_tiling_result["ub_tiling"]
        ub_tiling_result = ub_tiling_result_list[0]
        ub_split_axis = ub_tiling_result["axis"]
        res_ub_outer = ub_tiling_result["outer_itervar"]
        res_ub_inner = ub_tiling_result["inner_itervar"]

        global_reordered_axis_list = [out_tensor_global.op.reduce_axis[0]]
        for i in range(0, len(out_tensor_global.op.axis)):
            axis = out_tensor_global.op.axis[i]
            global_reordered_axis_list.append(axis)

        self._schedule[out_tensor_global].reorder(*(global_reordered_axis_list))

        ub_rf_reordered_axis_list = [out_tensor_ub_rf.op.axis[-1]]

        reduce_index_map = self._reduce_info["reduce_index_map"]
        reduce_block_axis = reduce_index_map[block_split_axis]
        reduce_ub_axis = reduce_index_map[ub_split_axis]

        if self._is_multi_core_need_fused:
            if block_split_axis != ub_split_axis:
                # rbi
                ub_rf_reordered_axis_list.append(out_tensor_ub_rf.op.reduce_axis[-1])
            for i in range(reduce_block_axis, reduce_ub_axis - 1):
                reduce_axis = out_tensor_ub_rf.op.reduce_axis[i - reduce_block_axis]
                ub_rf_reordered_axis_list.append(reduce_axis)

            ub_rf_reordered_axis_list.append(res_ub_outer)
            ub_rf_reordered_axis_list.append(res_ub_inner)
            for i in range(reduce_ub_axis,
                           len(out_tensor_ub_rf.op.reduce_axis) + reduce_block_axis - 1):
                reduce_axis = out_tensor_ub_rf.op.reduce_axis[i - reduce_block_axis]
                ub_rf_reordered_axis_list.append(reduce_axis)

        else:
            for i in range(0, reduce_block_axis):
                reduce_axis = out_tensor_ub_rf.op.reduce_axis[i]
                ub_rf_reordered_axis_list.append(reduce_axis)

            if block_split_axis != ub_split_axis:
                # rbi
                ub_rf_reordered_axis_list.append(out_tensor_ub_rf.op.reduce_axis[-1])

            for i in range(reduce_block_axis, reduce_ub_axis - 1):
                reduce_axis = out_tensor_ub_rf.op.reduce_axis[i]
                ub_rf_reordered_axis_list.append(reduce_axis)

            ub_rf_reordered_axis_list.append(res_ub_outer)
            ub_rf_reordered_axis_list.append(res_ub_inner)
            for i in range(reduce_ub_axis, len(out_tensor_ub_rf.op.reduce_axis) - 1):
                reduce_axis = out_tensor_ub_rf.op.reduce_axis[i]
                ub_rf_reordered_axis_list.append(reduce_axis)

        self._schedule[out_tensor_ub_rf].reorder(*(ub_rf_reordered_axis_list))

    # pylint: disable=too-many-locals
    def _reorder_atomic_reduce_not_last_axis(self, out_tensor_ub_rf, out_tensor_global):
        """
        :param out_tensor_ub_rf:
        :param out_tensor_global:
        :return:
        """
        # reorder (ak+1,ak,..a2,a1,rbo,rk,.,rb+1,rb-1,..r1,rbi) to
        # (rbo, ak+1,ak,..a2,rk,.,rb-1,rbi,rb+1,..r2,r1,a1) or
        # (rbo_fused, ak,..a2,rbi,rb+1,..r2,r1,a1) if need fused
        block_tiling_result = self._reduce_tiling_result["block_tiling"]
        block_split_axis = block_tiling_result["axis"]

        ub_tiling_result_list = self._reduce_tiling_result["ub_tiling"]
        ub_tiling_result = ub_tiling_result_list[0]
        ub_split_axis = ub_tiling_result["axis"]
        res_ub_outer = ub_tiling_result["outer_itervar"]
        res_ub_inner = ub_tiling_result["inner_itervar"]

        global_reordered_axis_list = [out_tensor_global.op.reduce_axis[0]]
        for i in range(0, len(out_tensor_global.op.axis)):
            axis = out_tensor_global.op.axis[i]
            global_reordered_axis_list.append(axis)

        self._schedule[out_tensor_global].reorder(*(global_reordered_axis_list))

        shape_before_reduce = self._reduce_info["shape_before_reduce"]
        reduce_axis_index = self._reduce_info["reduce_axis_index"]
        a1_start_index, a1_end_index = self._find_last_none_reduce_axis(
            shape_before_reduce, reduce_axis_index)

        is_keep_dim = self._reduce_info["keep_dims"]

        ub_rf_reordered_axis_list = [out_tensor_ub_rf.op.axis[-1]]
        offset = self.__get_offset(a1_start_index, a1_end_index, is_keep_dim, out_tensor_ub_rf)
        if (len(out_tensor_ub_rf.op.axis) - offset) > 2:
            ub_rf_reordered_axis_list.append(out_tensor_ub_rf.op.axis[0])

        reduce_index_map = self._reduce_info["reduce_index_map"]
        reduce_block_axis = reduce_index_map[block_split_axis]
        reduce_ub_axis = reduce_index_map[ub_split_axis]

        if self._is_multi_core_need_fused:
            # reorder (ak+1,ak,..a2,a1,rbo,rk,.,rb+1,rb-1,..r1,rbi) to
            # (rbo_fused, ak,..a2,rbi,rb+1,..r2,r1,a1)
            if block_split_axis != ub_split_axis:
                # rbi
                ub_rf_reordered_axis_list.append(out_tensor_ub_rf.op.reduce_axis[-1])
            for i in range(reduce_block_axis, reduce_ub_axis - 1):
                reduce_axis = out_tensor_ub_rf.op.reduce_axis[i - reduce_block_axis]
                ub_rf_reordered_axis_list.append(reduce_axis)

            ub_rf_reordered_axis_list.append(res_ub_outer)
            ub_rf_reordered_axis_list.append(res_ub_inner)
            for i in range(reduce_ub_axis,
                           len(out_tensor_ub_rf.op.reduce_axis) + reduce_block_axis - 1):
                reduce_axis = out_tensor_ub_rf.op.reduce_axis[i - reduce_block_axis]
                ub_rf_reordered_axis_list.append(reduce_axis)

        else:
            # reorder (ak+1,ak,..a2,a1,rbo,rk,.,rb+1,rb-1,..r1,rbi) to
            # (rbo, ak+1,ak,..a2,rk,.,rb-1,rbi,rb+1,..r2,r1,a1)
            for i in range(0, reduce_block_axis):
                reduce_axis = out_tensor_ub_rf.op.reduce_axis[i]
                ub_rf_reordered_axis_list.append(reduce_axis)

            if block_split_axis != ub_split_axis:
                # rbi
                ub_rf_reordered_axis_list.append(out_tensor_ub_rf.op.reduce_axis[-1])

            for i in range(reduce_block_axis, reduce_ub_axis - 1):
                reduce_axis = out_tensor_ub_rf.op.reduce_axis[i]
                ub_rf_reordered_axis_list.append(reduce_axis)

            ub_rf_reordered_axis_list.append(res_ub_outer)
            ub_rf_reordered_axis_list.append(res_ub_inner)
            for i in range(reduce_ub_axis, len(out_tensor_ub_rf.op.reduce_axis) - 1):
                reduce_axis = out_tensor_ub_rf.op.reduce_axis[i]
                ub_rf_reordered_axis_list.append(reduce_axis)

        none_reduce_index_map = self._find_none_reduce_axis_map(
            shape_before_reduce, reduce_axis_index)
        for i in range(a1_start_index, a1_end_index + 1):
            if is_keep_dim:
                none_reduce_index = i
            else:
                none_reduce_index = none_reduce_index_map[i]

            ub_rf_reordered_axis_list.append(
                self._schedule[out_tensor_ub_rf].op.axis[none_reduce_index])
        self._schedule[out_tensor_ub_rf].reorder(*(ub_rf_reordered_axis_list))

    def _reorder_atomic_reduce_last_axis(self, out_tensor_ub_rf, out_tensor_global):
        """
        :param out_tensor_ub_rf:
        :param out_tensor_global:
        :return:
        """
        # reorder (ak,..a2,a1,rbo,rk,.,rb+1,rb-1,..r1,rbi) to
        # (rbo, ak,..a2,rk,.,rb-1,rbi,rb+1,..a1,r1) or
        # (rbo_fused, ak,..a2,rbi,rb+1,..a1,r1) if need fused
        block_tiling_result = self._reduce_tiling_result["block_tiling"]
        block_split_axis = block_tiling_result["axis"]

        ub_tiling_result_list = self._reduce_tiling_result["ub_tiling"]
        ub_tiling_result = ub_tiling_result_list[0]
        ub_split_axis = ub_tiling_result["axis"]
        res_ub_outer = ub_tiling_result["outer_itervar"]
        res_ub_inner = ub_tiling_result["inner_itervar"]

        shape_before_reduce = self._reduce_info["shape_before_reduce"]
        reduce_axis_index = self._reduce_info["reduce_axis_index"]
        is_keep_dim = self._reduce_info["keep_dims"]

        # reorder (ak,..a2,a1,rbo) to (rbo,ak,..a2,a1)
        global_reordered_axis_list = [out_tensor_global.op.reduce_axis[0]]
        for i in range(0, len(out_tensor_global.op.axis)):
            axis = out_tensor_global.op.axis[i]
            global_reordered_axis_list.append(axis)

        self._schedule[out_tensor_global].reorder(*global_reordered_axis_list)

        # shape_before_reduce:(ak,rk,..,a1,r1), find a1 position
        a1_start_index, a1_end_index = self._find_last_none_reduce_axis(
            shape_before_reduce, reduce_axis_index)

        none_reduce_index_map = self._find_none_reduce_axis_map(
            shape_before_reduce, reduce_axis_index)

        ub_rf_reordered_axis_list = []
        reduce_index_map = self._reduce_info["reduce_index_map"]
        # rbo
        ub_rf_reordered_axis_list.append(out_tensor_ub_rf.op.axis[-1])
        if len(out_tensor_ub_rf.op.axis) > 3:
            ub_rf_reordered_axis_list.append(out_tensor_ub_rf.op.axis[0])

        def __get_reorder_case_num():
            if block_split_axis < a1_start_index and ub_split_axis < a1_start_index:
                return 1
            if block_split_axis < a1_start_index and a1_end_index < ub_split_axis:
                return 2
            if ub_split_axis in range(a1_start_index, a1_end_index + 1):
                return 3
            return 0

        case_num = __get_reorder_case_num()
        if case_num == 1:
            block_reduce_index = reduce_index_map[block_split_axis]
            ub_reduce_index = reduce_index_map[ub_split_axis]
            if self._is_multi_core_need_fused:
                offset = block_reduce_index
            else:
                offset = 0

            for i in range(0, block_reduce_index - offset):
                reduce_axis = out_tensor_ub_rf.op.reduce_axis[i]
                ub_rf_reordered_axis_list.append(reduce_axis)

            if block_split_axis != ub_split_axis:
                # rbi
                ub_rf_reordered_axis_list.append(out_tensor_ub_rf.op.reduce_axis[-1])

            for i in range(block_reduce_index, ub_reduce_index - 1):
                reduce_axis = out_tensor_ub_rf.op.reduce_axis[i - offset]
                ub_rf_reordered_axis_list.append(reduce_axis)

            ub_rf_reordered_axis_list.append(res_ub_outer)
            ub_rf_reordered_axis_list.append(res_ub_inner)

            for i in range(ub_reduce_index, reduce_index_map[a1_start_index - 1]):
                reduce_axis = out_tensor_ub_rf.op.reduce_axis[i - offset]
                ub_rf_reordered_axis_list.append(reduce_axis)

            for i in range(a1_start_index, a1_end_index + 1):
                if is_keep_dim:
                    none_reduce_index = i
                else:
                    none_reduce_index = none_reduce_index_map[i]
                ak_axis = out_tensor_ub_rf.op.axis[none_reduce_index]
                ub_rf_reordered_axis_list.append(ak_axis)

            for i in range(reduce_index_map[a1_end_index + 1],
                           len(out_tensor_ub_rf.op.reduce_axis) + offset):
                reduce_axis = out_tensor_ub_rf.op.reduce_axis[i - 1 - offset]
                ub_rf_reordered_axis_list.append(reduce_axis)

        elif case_num == 2:
            block_reduce_index = reduce_index_map[block_split_axis]
            ub_reduce_index = reduce_index_map[ub_split_axis]
            if self._is_multi_core_need_fused:
                offset = block_reduce_index
            else:
                offset = 0

            for i in range(0, block_reduce_index - offset):
                reduce_axis = out_tensor_ub_rf.op.reduce_axis[i]
                ub_rf_reordered_axis_list.append(reduce_axis)

            if block_split_axis != ub_split_axis:
                # rbi
                ub_rf_reordered_axis_list.append(out_tensor_ub_rf.op.reduce_axis[-1])

            for i in range(block_reduce_index, reduce_index_map[a1_start_index - 1]):
                reduce_axis = out_tensor_ub_rf.op.reduce_axis[i - offset]
                ub_rf_reordered_axis_list.append(reduce_axis)

            for i in range(a1_start_index, a1_end_index + 1):
                if is_keep_dim:
                    none_reduce_index = i
                else:
                    none_reduce_index = none_reduce_index_map[i]
                ak_axis = out_tensor_ub_rf.op.axis[none_reduce_index]
                ub_rf_reordered_axis_list.append(ak_axis)

            for i in range(reduce_index_map[a1_end_index + 1], ub_reduce_index):
                reduce_axis = out_tensor_ub_rf.op.reduce_axis[i - 1 - offset]
                ub_rf_reordered_axis_list.append(reduce_axis)

            ub_rf_reordered_axis_list.append(res_ub_outer)
            ub_rf_reordered_axis_list.append(res_ub_inner)

            for i in range(ub_reduce_index + 1, len(out_tensor_ub_rf.op.reduce_axis) + offset):
                reduce_axis = out_tensor_ub_rf.op.reduce_axis[i - 1 - offset]
                ub_rf_reordered_axis_list.append(reduce_axis)

        elif case_num == 3:
            block_reduce_index = reduce_index_map[block_split_axis]
            if self._is_multi_core_need_fused:
                offset = block_reduce_index
            else:
                offset = 0
            for i in range(0, block_reduce_index - offset):
                reduce_axis = out_tensor_ub_rf.op.reduce_axis[i]
                ub_rf_reordered_axis_list.append(reduce_axis)

            if block_split_axis != ub_split_axis:
                # rbi
                ub_rf_reordered_axis_list.append(out_tensor_ub_rf.op.reduce_axis[-1])

            for i in range(block_reduce_index, reduce_index_map[a1_start_index - 1]):
                reduce_axis = out_tensor_ub_rf.op.reduce_axis[i - offset]
                ub_rf_reordered_axis_list.append(reduce_axis)

            if ub_split_axis == a1_start_index:
                ub_rf_reordered_axis_list.append(res_ub_outer)
                ub_rf_reordered_axis_list.append(res_ub_inner)

                for i in range(a1_start_index + 1, a1_end_index + 1):
                    if is_keep_dim:
                        none_reduce_index = i
                    else:
                        none_reduce_index = none_reduce_index_map[i]
                    ak_axis = out_tensor_ub_rf.op.axis[none_reduce_index]
                    ub_rf_reordered_axis_list.append(ak_axis)
                for i in range(reduce_index_map[a1_end_index + 1],
                               len(out_tensor_ub_rf.op.reduce_axis) + offset):
                    reduce_axis = out_tensor_ub_rf.op.reduce_axis[i - 1 - offset]
                    ub_rf_reordered_axis_list.append(reduce_axis)

        self._schedule[out_tensor_ub_rf].reorder(*ub_rf_reordered_axis_list)

    @staticmethod
    def _find_none_reduce_axis_map(shape_before_reduce, reduce_axis_index):
        """
        :param shape_before_reduce:
        :param reduce_axis_index:
        :return:
        """
        none_reduce_index_map = {}
        count = 0
        for i in range(0, len(shape_before_reduce)):
            if i not in reduce_axis_index:
                none_reduce_index_map[i] = count
                count += 1
        return none_reduce_index_map



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
        for i in self._mid_tensors:
            if i not in self._mid_output_tensors:
                self._compute_inline_tensors.append(i)

    def _calculate_multi_core(self):
        """
        Calculate fuse and bind axis of multicore

        Parameters:
        ----------
        None

        Returns
        -------
        None
        """
        if self._need_multi_core:
            self._multi_core_fused_axis = \
                self._final_out_tensor_global.op.reduce_axis[0]
            self._multi_core_bind_tensor = self._final_out_tensor_global

    def _calculate_compute_at(self):
        """
        Calculate the tensor that needs compute at

        Parameters:
        ----------
        None

        Returns
        -------
        None
        """

        ub_tiling_result_list = self._reduce_tiling_result["ub_tiling"]
        for ub_tiling_result in ub_tiling_result_list:
            if "tiling_tensor" not in ub_tiling_result.keys() or \
                    "outer_itervar" not in ub_tiling_result.keys():
                continue
            ub_tiling_tensor = ub_tiling_result["tiling_tensor"]
            res_ub_outer = ub_tiling_result["outer_itervar"]
            res = ub_tiling_tensor

            for i in self._cache_read_tensors_and_buffer_map:
                read_buffer = self._cache_read_tensors_and_buffer_map[i]
                para = {"parent": self._schedule[res], "scope": res_ub_outer}
                self._compute_at_map[read_buffer] = para
            for i in self._cache_write_tensors_and_buffer_map:
                write_buffer = self._cache_write_tensors_and_buffer_map[i]
                para = {"parent": self._schedule[res], "scope": res_ub_outer}
                self._compute_at_map[write_buffer] = para
            for i in self._mid_output_tensors:
                para = {"parent": self._schedule[res], "scope": res_ub_outer}
                self._compute_at_map[i] = para

        if self._tiling_case == 1 or self._tiling_case == 3:
            para = {"parent": self._schedule[self._final_out_tensor_global],
                    "scope": self._final_out_tensor_global.op.reduce_axis[0]}
            self._compute_at_map[self._final_out_tensor_ub_rf] = para

        if self._tiling_case == 2:
            none_redcuce_shape = self._shape_to_list(
                self._final_out_tensor_global.shape)
            max_ub_count = self._get_max_ub_count()
            if not self._is_32b_align and self._support_storage_align():
                max_ub_count = self._last_axis_size

            ub_split_axis, ub_split_inner = self._get_ub_tiling(
                none_redcuce_shape, 0, none_redcuce_shape[0],
                max_ub_count)
            compute_at_axis = ub_split_axis
            if ub_split_inner == none_redcuce_shape[ub_split_axis]:
                compute_at_axis = compute_at_axis - 1
            emit_axis = ub_split_axis
            if ub_split_inner != none_redcuce_shape[ub_split_axis]:
                emit_axis = emit_axis + 1
            self._final_out_tensor_global_emit_axis = emit_axis
            if compute_at_axis != -1:
                para = {"parent": self._schedule[self._final_out_tensor_global],
                        "scope": self._final_out_tensor_global.op.axis[
                            compute_at_axis]}
            else:
                para = {"parent": self._schedule[self._final_out_tensor_global],
                        "scope": self._final_out_tensor_global.op.reduce_axis[
                            0]}
            self._compute_at_map[self._final_out_tensor_ub_rf] = para

    def _calculate_double_buffer(self):
        """
        double buffer operations
        read_buffer : the all read_cache for input in ub, type is list
        """

    def _calculate_emit_insn(self):
        """
        Calculate the instruction map of tensor

        Parameters:
        ----------
        None

        Returns
        -------
        None
        """
        ub_tiling_result_list = self._reduce_tiling_result["ub_tiling"]
        ub_tiling_result = ub_tiling_result_list[0]
        ub_tiling_tensor = ub_tiling_result["tiling_tensor"]
        res_ub_inner = ub_tiling_result["inner_itervar"]

        self._get_emit_insn_map()
        self._get_reg_emit_insn_map()

        for i in self._cache_read_tensors_and_buffer_map:
            read_buffer = self._cache_read_tensors_and_buffer_map[i]
            para = {"scope": read_buffer.op.axis[0],
                    "instruction": 'dma_copy'}
            self._emit_insn_map[read_buffer] = para

        for i in self._cache_write_tensors_and_buffer_map:
            write_buffer = self._cache_write_tensors_and_buffer_map[i]
            insn = self._calculate_emit_insn_map(write_buffer)
            if insn == "unified_broadcast":
                if not self._is_broadcast_last_axis_tensor(i):
                    continue

                max_last_broadcast_axis_offset = self._find_max_broadcast_last_axis_offset(i)
                cce_emitinsn_params.cceEmitParamsIns.del_param("broadcast_axis_offset")
                cce_emitinsn_params.cceEmitParamsIns.insert_param("broadcast_axis_offset",
                                                                  max_last_broadcast_axis_offset)
            if insn == "vector_multiple":
                self._do_emit_multiple_insn(i, write_buffer)
                continue
            para = {"scope": write_buffer.op.axis[0],
                    "instruction": insn}
            self._emit_insn_map[write_buffer] = para

        for out_tensor in self._mid_output_tensors:
            para = {"scope": out_tensor.op.axis[0],
                    "instruction": 'dma_copy'}
            self._emit_insn_map[out_tensor] = para

        res_tensor = self._res_tensor

        if self._tiling_case == 1:
            self._emit_insn_map[ub_tiling_tensor] = {"scope": res_ub_inner,
                                                     "instruction": 'vector_reduce_sum'}
            self._emit_insn_map[self._final_out_tensor_global] = {
                "scope": self._final_out_tensor_global.op.axis[0],
                "instruction": 'dma_copy'}

            self._emit_insn_map[res_tensor] = {
                "scope": self._schedule[res_tensor].op.axis[0],
                "instruction": 'phony_insn'}
        elif self._tiling_case == 2:
            if self._is_need_dichotomy_add:
                if len(self._last_output_tensors) == 2:
                    self._emit_insn_map[ub_tiling_tensor] = {
                        "scope": res_ub_inner, "instruction": 'vector_dichotomy_add_for_bn_reduce'}
                elif len(self._last_output_tensors) == 1:
                    self._emit_insn_map[ub_tiling_tensor] = {"scope": res_ub_inner,
                                                             "instruction": 'vector_dichotomy_add'}
                else:
                    self._emit_insn_map[ub_tiling_tensor] = {"scope": res_ub_inner,
                                                             "instruction": 'vector_reduce_sum'}
            else:
                self._emit_insn_map[ub_tiling_tensor] = {"scope": res_ub_inner,
                                                         "instruction": 'vector_reduce_sum'}

            emit_axis = self._final_out_tensor_global_emit_axis
            self._emit_insn_map[self._final_out_tensor_global] = {
                "scope": self._final_out_tensor_global.op.axis[emit_axis],
                "instruction": 'dma_copy'}
            self._emit_insn_map[res_tensor] = {
                "scope": self._schedule[res_tensor].op.axis[0],
                "instruction": 'phony_insn'}
        elif self._tiling_case == 3:
            shape_before_reduce = self._reduce_info["shape_before_reduce"]
            reduce_axis_index = self._reduce_info["reduce_axis_index"]
            ub_split_axis = ub_tiling_result["axis"]
            # ub cut ak (none reduce axis),
            if ub_split_axis not in reduce_axis_index:
                _, ak_end_index = self._find_last_none_reduce_axis(
                    shape_before_reduce, reduce_axis_index)
                last_reduce_index = self._reduce_info["reduce_index_map"][
                    ak_end_index + 1]

                block_tiling_result = self._reduce_tiling_result["block_tiling"]
                block_split_axis = block_tiling_result["axis"]
                reduce_index_map = self._reduce_info["reduce_index_map"]
                block_reduce_index = reduce_index_map[block_split_axis]
                if self._is_multi_core_need_fused:
                    offset = block_reduce_index
                else:
                    offset = 0

                self._emit_insn_map[ub_tiling_tensor] = {
                    "scope": ub_tiling_tensor.op.reduce_axis[
                        last_reduce_index - 1 - offset],
                    "instruction": 'vector_reduce_sum'}
            else:
                self._emit_insn_map[ub_tiling_tensor] = {"scope": res_ub_inner,
                                                         "instruction": 'vector_reduce_sum'}
            self._emit_insn_map[self._final_out_tensor_global] = {
                "scope": self._final_out_tensor_global.op.axis[0],
                "instruction": 'dma_copy'}
            self._emit_insn_map[res_tensor] = {
                "scope": self._schedule[res_tensor].op.axis[0],
                "instruction": 'phony_insn'}
        else:
            self._emit_insn_map[res_tensor] = {
                "scope": self._schedule[res_tensor].op.axis[0],
                "instruction": 'dma_copy'}


    def _calculate_emit_insn_map(self, tensor):
        """
        Get the instruction map of tensor

        Parameters:
        ----------
        None

        Returns
        -------
        Instruction map string
        """
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
        """
        Judge if tensor supports cast instruction operations

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

    def _do_emit_multiple_insn(self, tensor, tensor_buffer):

        for lop in self._op:
            if tensor == lop["dst_buffer"]:
                self._emit_multiple_insn(tensor_buffer, lop)

    def _emit_multiple_insn(self, cache_buffer, lop):
        """
        Multiple output instruction map

        Parameters:
        ----------
        cache_buffer :  cache write buffer
        lop : tensor op

        Returns
        -------
        None
        """
        # pylint: disable=invalid-sequence-index
        ub_tiling_result = self._reduce_tiling_result["ub_tiling"]
        ub_split_axis = ub_tiling_result["axis"]

        op_cmd = lop["op"].split("_")
        src = []
        index = 0
        args_map = ""
        for i in lop["args"]:
            if args_map in src:
                src.append(args_map)
            else:
                if i == "1":
                    args_map = lop["src_buffer"][index]
                src.append(lop["src_buffer"][index])
                index += 1

        if op_cmd[-1].lower() == "maddrelu":
            src_buffer = src[1]
            pragma = "vector_maddrelu"
        elif op_cmd[-1].lower() == "madd":
            src_buffer = src[1]
            pragma = "vector_madd"
        elif op_cmd[-1].lower() == "mla":
            src_buffer = src[2]
            pragma = "vector_mla"
        elif op_cmd[-1].lower() == "axpy":
            index = 1 if len(lop["src_buffer"]) > 1 else 0
            src_buffer = lop["src_buffer"][index]
            pragma = "vector_axpy"

        if src_buffer in self._cache_read_tensors_and_buffer_map:
            reuse_buffer = self._cache_read_tensors_and_buffer_map[src_buffer]
        else:
            reuse_buffer = self._cache_write_tensors_and_buffer_map[src_buffer]

        self._schedule[cache_buffer].emit_insn(
            cache_buffer.op.axis[ub_split_axis], pragma)
        self._schedule[reuse_buffer].reused_by(cache_buffer)
        if reuse_buffer in self._double_buffer_map:
            self._double_buffer_map[reuse_buffer].append(cache_buffer)
        else:
            self._double_buffer_map[reuse_buffer] = [cache_buffer]

    def _is_reduce_tensor(self, tensor):
        """
        :param tensor:
        :return:
        """
        return tensor.op.tag.find("reduce") != -1

    def _record_reduce_info(self, tensor):
        if self._is_reduce_tensor(tensor):
            self._reduce_info["reduce_tensor"] = tensor
            tensor_op = tensor.op
            reduce_axis_var = []
            for i in tensor_op.reduce_axis:
                reduce_axis_var.append(i)
            data_axis_var = tensor_op.body[0].source[0].args
            for ax_item in reduce_axis_var:
                for index in range(0, len(data_axis_var), 1):
                    if data_axis_var[index].same_as(ax_item.var):
                        self._reduce_info["reduce_axis_index"].append(index)
                        self._reduce_info["reduce_axis_map"][index] = ax_item

            self._reduce_info["reduce_axis_index"].sort()
            tmp_reduce_axis_num = self._reduce_info["reduce_axis_index"]
            reduce_index_map = {}
            for i, ele in enumerate(tmp_reduce_axis_num):
                reduce_index_map[ele] = i

            self._reduce_info["reduce_index_map"] = reduce_index_map
            if tensor.op.input_tensors:
                shape_before_reduce = self._shape_to_list(tensor.op.input_tensors[0].shape)
                self._reduce_info["shape_before_reduce"] = shape_before_reduce
            self._reduce_info["dtype"] = tensor.dtype

            is_keep_dims = len(self._reduce_info["shape_before_reduce"]) == len(tensor.shape)
            self._reduce_info["keep_dims"] = is_keep_dims


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
                # include (3,1,1) -> (3,2,1)
                for i in reversed(range(len(original_shape))):
                    if original_shape[i] != 1 and broadcast_shape[i] != 1:
                        return False
                    if original_shape[i] == 1 and broadcast_shape[i] != 1:
                        return True
        return False

    def _find_max_broadcast_last_axis_offset(self, tensor):
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
                    broadcast_axis_offset = 0
                    for i in range(len(original_shape) - 1, -1, -1):
                        if original_shape[i] == 1:
                            broadcast_axis_offset += 1
                            continue
                        break
                    if broadcast_axis_offset > max_broadcast_axis_offset:
                        max_broadcast_axis_offset = broadcast_axis_offset
        # input -> broadcast output, like tile
        else:
            max_broadcast_axis_offset = self. \
                _find_max_broadcast_last_axis_offset_from_tensor(tensor)

        return max_broadcast_axis_offset

    def _find_max_broadcast_last_axis_offset_from_tensor(self, tensor):
        max_broadcast_axis_offset = 0
        if list(tensor.op.input_tensors):
            original_tensor = tensor.op.input_tensors[0]
            original_shape = self._shape_to_list(original_tensor.shape)
            broadcast_axis_offset = 0
            for i in range(len(original_shape) - 1, -1, -1):
                if original_shape[i] == 1:
                    broadcast_axis_offset += 1
                    continue
                break
            if broadcast_axis_offset > max_broadcast_axis_offset:
                max_broadcast_axis_offset = broadcast_axis_offset

        return max_broadcast_axis_offset

    def _get_total_width(self):
        """
        caculate the max useable number based on op liveness
        return: max useable number
        """
        res = self._res_tensor

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
                # vcomsel use 3 temp buffer
                elif tag.find("cmpsel") != -1:
                    tmp_width = 3 * DTYPE_WIDTH_MAP[num_type.lower()]

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
        # for tuple sum
        if len(self._last_output_tensors) > 1:
            max_width += _op_width(res)

        return max_width

    def _get_max_ub_count(self):
        """
        caculate the max element num loaded in UB buffer
        :return: max element num loaded in UB buffer
        """
        # div 2 for align to fp16
        self._total_size = cceconf.get_soc_spec("UB_SIZE") // 2
        self._total_size = self._total_size // 2  # div 2 for double buffer
        total_width = self._get_total_width()
        if not total_width:
            raise RuntimeError("Can not calculate with no compute")

        max_bound = total_width * 128
        max_ub_count = int(self._total_size // max_bound * 128)

        return max_ub_count

    def __split_tensor(self, tensor):
        """
        Split the tensor and construct map

        Parameters:
        ----------
        None

        Returns
        -------
        Dict: construct map
        """
        tmp_op = {}
        op_node = tensor.op
        tmp_op["op"] = op_node.tag
        tmp_op["dst_buffer"] = tensor
        tmp_op["src_buffer"] = list(op_node.input_tensors)
        tmp_op["args"] = []
        tmp_op["effective_op"] = True

        if tmp_op["op"].find("elewise_single") != -1:
            if hasattr(op_node.body[0], 'b'):
                if isinstance(op_node.body[0].a, tvm.expr.Call):
                    tmp_op["args"] = [op_node.body[0].b]
                else:
                    tmp_op["args"] = [op_node.body[0].a]
        if tmp_op["op"].find("elewise_binary_compare") != -1:
            if hasattr(op_node.body[0], 'condition'):
                tmp_op["args"] = [op_node.body[0].condition.b]
            if tmp_op["op"].find("lt") != -1:
                tmp_op["args"].append("lt")
            elif tmp_op["op"].find("gt") != -1:
                tmp_op["args"].append("gt")
        if tmp_op["op"].find("elewise_binary_scalar") != -1:
            if hasattr(op_node.body[0], 'a'):
                if isinstance(op_node.body[0].a, tvm.expr.Call):
                    if hasattr(op_node.body[0].b, 'a'):
                        if isinstance(op_node.body[0].b.a, tvm.expr.Call):
                            tmp_op["args"] = [op_node.body[0].b.b]
                        else:
                            tmp_op["args"] = [op_node.body[0].b.a]
                else:
                    if hasattr(op_node.body[0].a, 'a'):
                        if isinstance(op_node.body[0].a.a, tvm.expr.Call):
                            tmp_op["args"] = [op_node.body[0].a.b]
                        else:
                            tmp_op["args"] = [op_node.body[0].a.a]
        elif tmp_op["op"].find("broadcast") != -1:
            if tmp_op["op"] == "broadcast_for_tensor":
                # broadcast not last axis
                if self._shape_to_list(tmp_op["src_buffer"][0].shape)[-1] != 1:
                    tmp_op["effective_op"] = False
            else:
                tmp_op["args"] = [op_node.body[0]]
        elif tmp_op["op"].find("reduce") != -1:
            self._have_reduce = True
            tmp_op["reduce_axis"] = list(op_node.reduce_axis)
            reduce_axis_var = []
            for i in op_node.reduce_axis:
                reduce_axis_var.append(i.var)
            data_axis_var = op_node.body[0].source[0].args
            tmp_op["reduce_axis_num"] = []
            for axis in reduce_axis_var:
                axis_num = 0
                for i in data_axis_var:
                    if i.same_as(axis):
                        tmp_op["reduce_axis_num"].append(axis_num)
                    axis_num += 1

        if tmp_op["op"].find("elewise_single_VS_cond") != -1 \
                or tmp_op["op"].find("elewise_binary_cmp") != -1 \
                or tmp_op["op"].find("elewise_binary_cmpsel") != -1 \
                or tmp_op["op"].find("elewise_binary_logic") != -1:
            str_list = op_node.tag.split("|")
            tmp_op["op"] = str_list[0]
            tmp_op["args"] = []
            for i in range(1, len(str_list)):
                tmp_op["args"].append(str_list[i])

        # split inputs sign and add into args for elewise_multiple op
        elif tmp_op["op"].find("elewise_multiple") != -1:
            str_list = op_node.tag.split("|")
            tmp_op["op"] = str_list[0]
            if len(str_list) >= 2:
                same_list_str = str_list[1].split(',')
                tmp_op["args"] = same_list_str

        if tmp_op["op"].find("|") != -1:
            str_list = op_node.tag.split("|")
            tmp_op["op"] = str_list[0]

        return tmp_op
