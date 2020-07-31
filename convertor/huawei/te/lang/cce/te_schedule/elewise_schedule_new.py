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

elewise schedule
"""
# pylint: disable=import-error, unused-import
import math
from functools import reduce
from te import platform as cceconf
from te import tvm
from te.platform import cce_emitinsn_params
from . import util
from .vector_schedule import VectorSchedule
from .cce_schedule_mappings import OpSubPatterns
from .cce_schedule_mappings import OpSpecTypes
DTYPE_WIDTH_MAP = {"float16": 1,
                   "float32": 2,
                   "int32": 2,
                   "int16": 1,
                   "uint16": 1,
                   "int8": 0.5,
                   "uint8": 0.5,
                   "bool": 0.5}

ALIGN_FACTOR = 32

# The last axis of the broadcast is not in the 32B alignment scenario, if the
# last axis is greater than this threshold, then move the amount of data of the
# last axis each time. If the last axis is smaller than this threshold,
# then carry as much data as possible and use the scale to do the broadcast in UB.
BROADCAST_LAST_AXIS_THRESHOLD = 512

BLOCK_TILING_PRIME_THRESHOLD = 67

UB_UTILIZATION_RATIO_OPT_THRESHOLD = 0.005

NON_LAST_BROADCAST_UNIT_THRESHOLD = 20


# pylint: disable=too-many-return-statements,too-few-public-methods,too-many-arguments,too-many-statements,no-self-use,too-many-lines,too-many-instance-attributes,too-many-branches,
class ElewiseSchedule(VectorSchedule):
    """
    class of cce elewise schedule

    Parameters
    ----------
    VectorSchedule: base class of elewise schedule

    Returns
    -------
    ElewiseSchedule_instance : instance of ElewiseSchedule
    """

    def __init__(self, need_multi_core=True):
        VectorSchedule.__init__(self, need_multi_core)
        # record ops
        self._op = []
        # record origin op
        self._origin_op = []

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

        self._tensor_scaler_operator = ["elewise_binary_mul",
                                        "elewise_binary_add",
                                        "elewise_binary_div"]
        # default at least data processed by each core, unit is byte
        # this value is dynamically adjusted based on the type of the operator
        self._multi_core_threshold = 1024
        self._spec_node_list = []
        self._is_last_axis_broadcast = False
        self._total_size = 0
        self._max_ub_count = 0
        self._normalize_scale_enhance_opt = False
        self._is_muti_output = False
        self._have_reduce = False
        self._special_broadcast_with_transpose = False
        self._special_non_last_broadcast_scene = False
        self._special_mix_broadcast_scene = False
        self._special_non_32align_broadcast_scene = False
        self._special_32align_broadcast_scene = False
        self._special_non_last_broadcast_factor16_scene = False
        self._is_need_update_compute_at_axis = False
        self._is_preload_fused_axis_scene = False
        self._preload_fused_axis = None
        self._continue_broadcast_last_axis = []
        self._broadcast_enhance_insn_map = \
            {"vector_mul": "vector_mul_with_broadcast_enhance",
             "vector_div": "vector_div_with_broadcast_enhance",
             "vector_add": "vector_add_with_broadcast_enhance",
             "vector_sub": "vector_sub_with_broadcast_enhance",
             "vector_min": "vector_min_with_broadcast_enhance",
             "vector_max": "vector_max_with_broadcast_enhance"
            }
        self._non_32align_broadcast_insn_map = \
            {"vector_mul": "vector_mul_with_broadcast_non_32align",
             "vector_div": "vector_div_with_broadcast_non_32align",
             "vector_add": "vector_add_with_broadcast_non_32align",
             "vector_sub": "vector_sub_with_broadcast_non_32align",
            }

        # For (x <= 32, 100, 1, 4) to (x <= 32, 100, 2, 4) broadcasting
        self._less_32_core_middle_broadcast_scene = False
        # For zeros_like special output
        self._elewise_binary_phony_as_output = False

    def _construct_compute_graph(self, out_tensors, spec_node_list):
        """
        record relate context imformations of operations

        outTensors only support form like: out_1->..out_2->..out_n

        """
        # find the last out tensor
        if hasattr(out_tensors, 'index'):
            if len(out_tensors) > 1:
                self._is_muti_output = True
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
        self._mid_output_tensors.remove(self._last_output_tensor)

        tensor_list = []
        visited_list = []
        self.__gen_reversed_subgraph_list(self._last_output_tensor,
                                          tensor_list, visited_list)
        tensor_list.append(self._last_output_tensor)

        output_shape = self._shape_to_list(self._last_output_tensor.shape)
        self._is_optimize_network_shape(output_shape)

        self._is_need_enable_mem_unique(output_shape)

        if len(tensor_list) <= 5:
            self._multi_core_threshold = 2 * 1024

        # in order to process input -> broadcast -> output, like broadcast_to
        if self._is_broadcast_last_axis_tensor(self._last_output_tensor):
            if self._last_output_tensor not in self._broadcast_last_axis_tensors:
                self._broadcast_last_axis_tensors.append(
                    self._last_output_tensor)
        else:
            if self._is_broadcast_not_last_axis_tensor(self._last_output_tensor):
                if self._last_output_tensor not in \
                        self._broadcast_not_last_axis_tensors:
                    self._broadcast_not_last_axis_tensors.append(
                        self._last_output_tensor)

        # calculate cache_write_exclude_tensors
        for i in self._broadcast_not_last_axis_tensors:
            self._cache_write_exclude_tensors.append(i)
        shape = self._shape_to_list(self._last_output_tensor.shape)
        dtype = self._last_output_tensor.dtype.lower()

        if self._is_only_broadcast_last_axis():
            if not self._is_32b_align_of_broadcast_last_axis_tensors(
                    shape, dtype) and shape[-1] > BROADCAST_LAST_AXIS_THRESHOLD:
                for i in self._broadcast_last_axis_tensors:
                    # in order to avoid the last output tensor can not get
                    # the map value
                    if i in \
                            self._broadcast_last_axis_tensor_dst_tensor_map.keys():
                        dst_tensors = \
                            self._broadcast_last_axis_tensor_dst_tensor_map[i]
                        if self._support_tensor_scaler_operate(dst_tensors):
                            self._cache_write_exclude_tensors.append(i)

        for i in self._broadcast_scalars:
            dst_tensors = self._broadcast_scalar_dst_tensor_map[i]
            if self._support_tensor_scaler_operate(dst_tensors):
                self._cache_write_exclude_tensors.append(i)

        for tensor in reversed(tensor_list):
            tmp_op = self.__split_tensor(tensor)
            if tmp_op["effective_op"]:
                self._op.append(tmp_op)
            else:
                # broadcast not last axis no need do cache_write, include
                # output tensor
                self._cache_write_exclude_tensors.append(tensor)
            self._origin_op.append(tmp_op)

        return True

    # pylint: disable=unnecessary-pass
    def _is_optimize_network_shape(self, shape):
        """
        Judge if the shape need optimize operation

        Parameters:
        ----------
        shape :  output tensor shape

        Returns
        -------
        """
        pass

    def _is_need_enable_mem_unique(self, shape):
        """
        Judge if need enable memory unique

        Parameters:
        ----------

        Returns
        -------
        """
        if cceconf.get_soc_spec("SOC_VERSION") != "Ascend310":
            return

        if self._is_contain_broadcast_tensor():
            return

        if len(shape) != 1:
            return

        if len(self._input_tensors) > 2:
            return

        self._mem_unique_enable = True

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

            if not (in_tensor in self._spec_node_list \
                    or isinstance(in_tensor.op, tvm.tensor.PlaceholderOp)):
                if in_tensor not in self._mid_tensor_dst_tensor_map.keys():
                    self._mid_tensors.append(in_tensor)
                self._map_apend(self._mid_tensor_dst_tensor_map, in_tensor,
                                tensor)

            if self._is_broadcast_last_axis_tensor(in_tensor):
                if in_tensor not in \
                        self._broadcast_last_axis_tensor_dst_tensor_map.keys():
                    self._broadcast_last_axis_tensors.append(in_tensor)
                self._map_apend(self._broadcast_last_axis_tensor_dst_tensor_map,
                                in_tensor, tensor)
            else:
                self.apply_broadcast_not_last_axis_tensors(in_tensor)

            if self._is_broadcast_orignal_scalar(in_tensor):
                if in_tensor not in self._broadcast_scalar_dst_tensor_map.keys():
                    self._broadcast_scalars.append(in_tensor)
                self._map_apend(self._broadcast_scalar_dst_tensor_map,
                                in_tensor, tensor)

            if in_tensor in self._spec_node_list \
                    or isinstance(in_tensor.op, tvm.tensor.PlaceholderOp):
                if in_tensor not in self._input_tensor_dst_tensor_map.keys():
                    self._input_tensors.append(in_tensor)
                self._map_apend(self._input_tensor_dst_tensor_map, in_tensor,
                                tensor)
                continue

            if in_tensor in visited_list:
                continue

            visited_list.append(in_tensor)

            self.__gen_reversed_subgraph_list(in_tensor, tensor_list,
                                              visited_list)
            tensor_list.append(in_tensor)

    def apply_broadcast_not_last_axis_tensors(self, in_tensor):
        """Apply broadcast not last axis tensors"""
        is_cmpsel = self._op_subpattern == OpSubPatterns.CMPSEL_PATTERN
        is_mid_broadcast = self._is_broadcast_not_last_axis_tensor(in_tensor)
        is_mid_non32_broadcast = \
            self._is_non_32align_broadcast_not_last_axis(in_tensor)
        if (is_cmpsel and is_mid_non32_broadcast) or (
                not is_cmpsel and is_mid_broadcast):
            if in_tensor not in self._broadcast_not_last_axis_tensors:
                self._broadcast_not_last_axis_tensors.append(in_tensor)

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
            # elewise_binary_phony will prevent its second input tensor from being read
            # And, its first input tensor will become last_output
            if self._input_tensor_dst_tensor_map[i][0].op.tag == "elewise_binary_phony":
                if i == self._schedule[self._input_tensor_dst_tensor_map[i][0]].op.input_tensors[1]:
                    continue
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

        if self._last_output_tensor not in self._cache_write_exclude_tensors:
            # in order to avoid last axis broadcast tensor as output tensor
            # do cache_write
            if self._last_output_tensor.op.tag == "elewise_binary_phony":
                self._elewise_binary_phony_as_output = True
                return
            self._cache_write_tensors.append(self._last_output_tensor)

    def _do_emit_insn(self):
        # fix for bert bn_training_reduce_grad
        if util.shape_to_list(self._last_output_tensor.shape) == [1, 8192, 1, 1024] and \
                util.is_bert_bn_target(self._mid_tensors):
            for stage in self._emit_insn_map:
                if stage in self._cache_write_tensors:
                    continue
                if util.BROADCAST_TAG_LABEL in stage.op.tag:
                    self._schedule[stage].compute_inline()
                for input_stage in stage.op.input_tensors:
                    if util.BROADCAST_TAG_LABEL in input_stage.op.tag:
                        self._emit_insn_map[stage]["scope"] = stage.op.axis[-1]
        VectorSchedule._do_emit_insn(self)

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
        # pylint: disable=too-many-locals
        def __is_block_tiling_use_nparts_mode(shape, block_split_axis):
            """check block tiling whether to use nparts mode"""
            if block_split_axis == 0:
                return True
            if sum(shape[0:block_split_axis]) == block_split_axis:
                return True
            return False

        shape = self._shape_to_list(self._last_output_tensor.shape)
        dtype = self._last_output_tensor.dtype.lower()

        if self._need_multi_core:
            multi_core_threshold = self._get_multi_core_threshold(shape, dtype)
            block_split_axis, block_split_inner_size, block_split_outer_size = \
                self._get_block_tiling(shape, dtype, multi_core_threshold)
            if __is_block_tiling_use_nparts_mode(shape, block_split_axis):
                self.block_tiling_use_nparts_mode = True
                block_split_inner_size = shape[block_split_axis] // \
                                         block_split_outer_size
        else:
            block_split_axis = 0
            block_split_inner_size = shape[block_split_axis]
            block_split_outer_size = 1

        # like (7,2,42767,2,19) scene, if the block tiling split the 3 axis,
        # because 42767 is a prime number, then ub tiling should be divisible,
        # then lead to the dma copy is less efficient, so adjust the block tiling
        if block_split_axis != 0 and \
                sum(shape[0:block_split_axis]) != block_split_axis:
            if self._is_prime_number(shape[block_split_axis]) and \
                    shape[block_split_axis] > BLOCK_TILING_PRIME_THRESHOLD:
                if shape[block_split_axis] % block_split_inner_size == 0:
                    block_split_axis = block_split_axis - 1
                    block_split_inner_size = 1

        max_ub_count = self._get_max_ub_count()
        self._max_ub_count = max_ub_count

        # if the cmp mode is bit mode, the res shape is 1/8 input shape,
        # we should tiling as the input shape
        self._get_emit_insn_map()
        self._get_reg_emit_insn_map()
        max_ub_count = self._calculate_special_scene_max_ub_count(max_ub_count)

        ub_split_axis, ub_split_inner = self._calculate_tiling_core(
            shape, dtype, block_split_axis, block_split_inner_size,
            max_ub_count)

        if self._need_multi_core:
            if self._is_need_optimize_block_tiling(shape, block_split_axis,
                                                   ub_split_axis,
                                                   ub_split_inner,
                                                   max_ub_count):
                block_split_axis = block_split_axis - 1
                block_split_inner_size = 1

                ub_split_axis, ub_split_inner = self._calculate_tiling_core(
                    shape, dtype, block_split_axis, block_split_inner_size,
                    max_ub_count)

            if self._is_need_modify_block_and_ub_tiling(shape, dtype,
                                                        block_split_axis,
                                                        block_split_inner_size,
                                                        ub_split_axis,
                                                        ub_split_inner,
                                                        max_ub_count):
                self.block_tiling_use_nparts_mode = False
                block_split_axis = 0
                block_split_inner_size = shape[block_split_axis]
                ub_split_axis, ub_split_inner = self._calculate_tiling_core(
                    shape, dtype, block_split_axis, block_split_inner_size,
                    max_ub_count)

        block_split_factor_size = self._get_block_tiling_factor_size(
            block_split_inner_size, block_split_outer_size)

        block_tiling_para = {"axis": block_split_axis,
                             "factor": block_split_factor_size}
        ub_tiling_para = {"axis": ub_split_axis, "factor": ub_split_inner}

        self._tiling_para["block_tiling"] = block_tiling_para
        # If there are more than one tensors, check if ub_tiling_factor is smaller than align factor
        if len(self._mid_output_tensors) > 1:
            for tensor in self._mid_output_tensors:
                size = 1
                minimum_size = int(
                    ALIGN_FACTOR // (DTYPE_WIDTH_MAP[tensor.dtype] * 2))
                size *= ub_tiling_para["factor"]
                for axis in range(ub_tiling_para["axis"] + 1,
                                  len(tensor.shape)):
                    size *= tensor.shape[axis]
                if int(size) < minimum_size:
                    ub_tiling_para["factor"] = minimum_size

        self._tiling_para["ub_tiling"] = ub_tiling_para

        self._tiling_tensor = self._last_output_tensor

    def _get_multi_core_threshold(self, shape, dtype):
        """
        get multiply core threshold

        Parameters
        ----------
        shape: output tensor shape

        dtype: output tensor date type

        Returns
        -------
        multiply core threshold
        """
        multi_core_threshold = self._multi_core_threshold
        core_num = cceconf.get_soc_spec("CORE_NUM")
        if core_num == shape[0] and len(shape) > 1:
            data_size = DTYPE_WIDTH_MAP[dtype] * 2
            for i in range(1, len(shape)):
                data_size = data_size * shape[i]
            if data_size >= 512:
                multi_core_threshold = 512
        return multi_core_threshold

    def _get_block_tiling_factor_size(self, block_split_inner_size,
                                      block_split_outer_size):
        """
        get block tiling split factor size

        Parameters
        ----------
        block_split_inner_size: block tiling split inner size

        block_split_outer_size: block tiling split outer size

        Returns
        -------
        block tiling split factor size
        """
        if self.block_tiling_use_nparts_mode:
            return block_split_outer_size

        return block_split_inner_size

    def _calculate_special_scene_max_ub_count(self, max_ub_count):
        """
        calculate special scene max ub count

        Parameters
        ----------
        max_ub_count: maximum ub buffer available space

        Returns
        -------
        maximum ub buffer available space
        """
        output_tensor = self._last_output_tensor
        if output_tensor.op.tag.find("|") != -1:
            str_list = output_tensor.op.tag.split("|")
            insn = self._reg_insn_map.get(str_list[0])
            if insn == 'elewise_binary_cmp' and 'bit' in str_list:
                max_ub_count = max_ub_count // 8

        return max_ub_count

    def _calculate_ub_utilization_ratio(self, shape, ub_split_axis,
                                        ub_split_inner, max_ub_count):
        """
        calculate ub buffer utilization ratio

        Parameters
        ----------
        shape: output tensor shape

        ub_split_axis: ub split axis

        ub_split_inner: ub split inner factor

        max_ub_count: maximum ub buffer available space

        Returns
        -------
        ub utilization ratio value
        """
        count = 1
        if ub_split_axis == len(shape) - 1:
            count = ub_split_inner
        else:
            for i in range(ub_split_axis + 1, len(shape), 1):
                count = count * shape[i]
            count = count * ub_split_inner

        return count / max_ub_count

    def _is_need_optimize_block_tiling(self, shape, block_split_axis,
                                       ub_split_axis, ub_split_inner,
                                       max_ub_count):
        """
        judge if it is need optimize block tiling

        Parameters
        ----------
        shape: output tensor shape

        block_split_axis: block split axis

        ub_split_axis: ub split axis

        ub_split_inner: ub split inner factor

        max_ub_count: maximum ub buffer available space

        Returns
        -------
        True or False
        """
        if block_split_axis == 0:
            return False

        max_broadcast_axis = \
            self._find_max_broadcast_axis_of_broadcast_not_last_axis_tensors()
        dtype = self._last_output_tensor.dtype.lower()
        if self._is_contain_broadcast_tensor():
            if self._is_only_broadcast_not_last_axis():
                if not self._is_32b_align_of_broadcast_not_last_axis_tensors(shape,
                                                                             dtype):
                    if max_broadcast_axis == ub_split_axis:
                        return False

        utilization_ratio = self._calculate_ub_utilization_ratio(
            shape, ub_split_axis, ub_split_inner, max_ub_count)

        if utilization_ratio >= UB_UTILIZATION_RATIO_OPT_THRESHOLD:
            return False

        return True

    def _is_prime_number(self, axis):
        if axis < 2:
            return False
        for i in range(2, int(math.sqrt(axis)) + 1):
            if axis % i == 0:
                return False
        return True

    def _is_need_modify_block_and_ub_tiling(self, shape, dtype,
                                            block_split_axis,
                                            block_split_inner_size,
                                            ub_split_axis, ub_split_inner,
                                            max_ub_count):
        """
        check whether need modify block and ub tiling or not
        """
        # pylint: disable=too-many-locals
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


    def _calculate_tiling_core(self, # pylint: disable=too-many-locals
                               shape, dtype, block_split_axis,
                               block_split_inner_size, max_ub_count):
        """
        calculate tiling core strategy

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        last_axis = len(shape) - 1

        def __is_relu_v2_or_grad():
            if self._op_type == OpSpecTypes.RELU_V2 or self._op_type == OpSpecTypes.RELU_GRAD_V2:
                return True
            return False

        def __is_special_only_broadcast_scene(dtype, shape):
            # just support [x, 1] -> [x, 2] float32 scene

            if self._op_type != OpSpecTypes.ONLY_BROADCAST_TYPE:
                return

            if dtype != "float32":
                return

            if shape[-1] != 2:
                return

            if self._is_mix_broadcast_of_last_axis_broadcast():
                return

            count = 1
            for i in range(0, len(shape) - 1):
                count = count * shape[i]

            # if dim is small, use scalar directly
            opt_threshold = 1024*1024
            if count < opt_threshold:
                return

            self._special_broadcast_with_transpose = True

        divisible_split = __is_relu_v2_or_grad()

        def __is_special_non_last_broadcast_factor16_scene(ub_split_axis):
            if ub_split_axis != 0 and \
                self._is_special_factor16_broadcast_sence():
                return True
            return False

        def __is_elewise_multiple_sel_bit_mode():
            if self._op[index]["op"] == 'emit_insn_elewise_multiple_sel' \
                    and self._op[index]["args"][0] == 'bit':
                return True
            return False

        def __shape_mul(shape):
            if not shape:
                return 1
            return reduce(lambda x, y: x * y, shape)

        def __do_mix_broadcast_of_last_axis_broadcast_ub_tiling():
            if shape[-1] > BROADCAST_LAST_AXIS_THRESHOLD:
                # ub tiling along last axis
                if block_split_axis != last_axis:
                    ub_split_axis, ub_split_inner = \
                        self._get_ub_tiling(shape, last_axis - 1,
                                            1,
                                            max_ub_count)
                else:
                    ub_split_axis, ub_split_inner = \
                        self._get_ub_tiling(shape, last_axis,
                                            block_split_inner_size,
                                            max_ub_count)
            else:
                max_broadcast_axis = \
                    self._find_max_broadcast_axis_of_mix_broadcast()
                if max_broadcast_axis <= block_split_axis:
                    ub_split_axis, ub_split_inner = self._get_ub_tiling(
                        shape, block_split_axis, block_split_inner_size,
                        max_ub_count)
                else:
                    ub_split_axis, ub_split_inner = self._get_ub_tiling(
                        shape, max_broadcast_axis,
                        shape[max_broadcast_axis], max_ub_count)
            return ub_split_axis, ub_split_inner

        def __do_continue_broadcast_ub_tiling():
            if __shape_mul(shape[self._continue_broadcast_last_axis[0]:]) > \
                    BROADCAST_LAST_AXIS_THRESHOLD:
                if block_split_axis < self._continue_broadcast_last_axis[0]:
                    ub_split_axis, ub_split_inner = \
                        self._get_ub_tiling(shape, self._continue_broadcast_last_axis[0] - 1,
                                            1,
                                            max_ub_count)
                else:
                    ub_split_axis, ub_split_inner = \
                        self._get_ub_tiling(shape, block_split_axis,
                                            block_split_inner_size,
                                            max_ub_count)
            else:
                max_broadcast_axis = \
                    self._find_max_broadcast_axis_of_mix_broadcast()
                if max_broadcast_axis <= block_split_axis:
                    ub_split_axis, ub_split_inner = self._get_ub_tiling(
                        shape, block_split_axis, block_split_inner_size,
                        max_ub_count)
                else:
                    ub_split_axis, ub_split_inner = self._get_ub_tiling(
                        shape, max_broadcast_axis,
                        shape[max_broadcast_axis], max_ub_count)
            return ub_split_axis, ub_split_inner

        def __is_mix_non_32align_of_last_axis_broadcast(shape, dtype):
            if not self._is_mix_broadcast_of_last_axis_broadcast():
                return False
            if self._is_32align_of_mix_broadcast_last_axis(shape, dtype):
                return False

            if self._op_type == OpSpecTypes.NORMALIZE_SCALE:
                if not self._is_mix_broadcast_enable_32align_tiling():
                    return True

                if self._is_32align_of_mix_broadcast_last_axis(shape, dtype):
                    return False

                if self._is_32b_align_of_broadcast_last_axis_tensors(shape,
                                                                     dtype):
                    return False

            return True

        def __normalize_broadcast_opt_proc():
            if self._op_type != OpSpecTypes.NORMALIZE_SCALE:
                return
            # remove (1,1,1,1) -> (a,b,c,d) broadcast
            for tensor in self._broadcast_last_axis_tensors:
                if list(tensor.op.input_tensors):
                    input_shape = self._shape_to_list(
                        tensor.op.input_tensors[0].shape)
                    if sum(input_shape[:]) == len(input_shape):
                        self._broadcast_last_axis_tensors.remove(tensor)
        __normalize_broadcast_opt_proc()

        # pylint: disable=too-many-nested-blocks
        if self._is_contain_broadcast_tensor():
            if self._is_only_broadcast_last_axis():
                __is_special_only_broadcast_scene(dtype, shape)
                # process (1,1,1,1,1,1,1)->(10, 10, 5, 2, 3, 9,1) scene
                if self._is_only_scale_broadcast_of_last_axis_broadcast():
                    ub_split_axis, ub_split_inner = self._get_ub_tiling(
                        shape, block_split_axis, block_split_inner_size,
                        max_ub_count)
                # process (a,b,c,d,e), (a,b,c,1,1) scene
                elif self._is_continue_broadcast_of_last_axis_broadcast():
                    ub_split_axis, ub_split_inner = \
                        __do_continue_broadcast_ub_tiling()
                # process (a,b,c,d,e), (a,b,1,d,1) scene
                elif __is_mix_non_32align_of_last_axis_broadcast(shape, dtype):
                    ub_split_axis, ub_split_inner = \
                        __do_mix_broadcast_of_last_axis_broadcast_ub_tiling()
                else:
                    if self._is_32b_align_of_broadcast_last_axis_tensors(
                            shape, dtype):
                        ub_split_axis, ub_split_inner = self._get_ub_tiling(
                            shape, block_split_axis, block_split_inner_size,
                            max_ub_count)
                    else:
                        if shape[-1] > BROADCAST_LAST_AXIS_THRESHOLD:
                            # ub tiling along last axis
                            if block_split_axis != last_axis:
                                # Example 128, 139301
                                ub_split_axis, ub_split_inner = \
                                    self._get_ub_tiling(shape, last_axis-1,
                                                        1,
                                                        max_ub_count)
                            else:
                                ub_split_axis, ub_split_inner = \
                                    self._get_ub_tiling(shape, last_axis,
                                                        block_split_inner_size,
                                                        max_ub_count)
                        else:
                            ub_split_axis, ub_split_inner = self._get_ub_tiling(
                                shape, block_split_axis, block_split_inner_size,
                                max_ub_count)
            else:
                if self._is_32b_align_of_broadcast_not_last_axis_tensors(
                        shape, dtype):
                    ub_split_axis, ub_split_inner = self._get_ub_tiling(
                        shape, block_split_axis, block_split_inner_size,
                        max_ub_count)
                    self._special_non_last_broadcast_factor16_scene = \
                        __is_special_non_last_broadcast_factor16_scene(ub_split_axis)

                    # broadcast non-last axis, if broadcast axis is same as
                    # ub tiling axis, then broadcast tensor copy_gm_to_ub
                    # process can be mentioned outside the loop,
                    # avoid duplication dma
                    max_broadcast_axis = \
                        self._find_max_broadcast_axis_of_broadcast_not_last_axis_tensors()
                    self._update_compute_at_axis(block_split_axis,
                                                 ub_split_axis,
                                                 max_broadcast_axis)
                else:
                    if self._is_special_broadcast_sence(block_split_axis,
                                                        block_split_inner_size):
                        ub_split_axis, ub_split_inner = self._get_ub_tiling(
                            shape, block_split_axis, block_split_inner_size,
                            max_ub_count)
                        self._special_non_last_broadcast_scene = True
                    elif self._is_less_32_core_middle_broadcast_out_scene(block_split_axis,
                                                                          block_split_inner_size):
                        ub_split_axis, ub_split_inner = self._get_ub_tiling(
                            shape, block_split_axis, block_split_inner_size,
                            max_ub_count)
                        self._less_32_core_middle_broadcast_scene = True
                    elif self._is_mix_broadcast_out_scene(block_split_axis,
                                                          block_split_inner_size):
                        ub_split_axis = -2
                        ub_split_inner = shape[-2]
                        self._special_mix_broadcast_scene = True
                    elif self._is_non_32align_broadcast_out_scene(block_split_axis,
                                                                  block_split_inner_size,
                                                                  shape, max_ub_count):
                        ub_split_axis, ub_split_inner = self._get_ub_tiling(
                            shape, block_split_axis, block_split_inner_size,
                            max_ub_count)
                        self._special_non_32align_broadcast_scene = True
                    elif self._is_32align_broadcast_out_scene(block_split_axis,
                                                              block_split_inner_size):
                        ub_split_axis, ub_split_inner = self._get_ub_tiling(
                            shape, block_split_axis, block_split_inner_size,
                            max_ub_count)
                        self._special_32align_broadcast_scene = True
                    else:
                        # Example: (a,b,c,1,e), (a,b,1,d,e), max_broadcast_axis is 3,
                        # ub tiling along axis 4
                        max_broadcast_axis = \
                            self._get_max_broadcast_axis()
                        if max_broadcast_axis < block_split_axis:
                            ub_split_axis, ub_split_inner = self._get_ub_tiling(
                                shape, block_split_axis, block_split_inner_size,
                                max_ub_count)
                        else:
                            ub_split_axis, ub_split_inner = self._get_ub_tiling(
                                shape, max_broadcast_axis, 1, max_ub_count)
                            # broadcast non-last axis, if broadcast axis is same as
                            # ub tiling axis, then broadcast tensor copy_gm_to_ub
                            # process can be mentioned outside the loop,
                            # avoid duplication dma
                            self._update_compute_at_axis(block_split_axis,
                                                         ub_split_axis,
                                                         max_broadcast_axis)
        else:
            ub_split_axis, ub_split_inner = self._get_ub_tiling(
                shape, block_split_axis, block_split_inner_size, max_ub_count,
                divisible_split)

        # the vsel in bit mode, the condition shape is not same as input shape,
        # then input_shape[-1] equals condition_shape[-1] * 8
        # so when spilt the last axis, spilt factor should be the mutiply of 8
        for index in range(len(self._op)):
            if __is_elewise_multiple_sel_bit_mode():
                if ub_split_axis == len(shape) - 1:
                    for factor in range(ub_split_inner, 1, -1):
                        if shape[ub_split_axis] % factor == 0 and \
                                factor % 8 == 0:
                            ub_split_inner = factor
                            return ub_split_axis, ub_split_inner

        return ub_split_axis, ub_split_inner

    def _update_compute_at_axis(self, block_split_axis, ub_split_axis,
                                max_broadcast_axis):
        """
        Judge is need update compute at axis for broadcast non-last scene

        Parameters:
        ----------
        block_split_axis, ub_split_axis, max_broadcast_axis

        Returns
        -------
        None
        """

        self._is_need_update_compute_at_axis = False

        if ub_split_axis == 0:
            return

        if ub_split_axis > max_broadcast_axis:
            return

        # [1, 1, 1, 1, 32, 16, 3, 3] -> [32, 16, 3, 3, 32, 16, 3, 3],
        # ub tiling is 2, max_broadcast_axis is 3
        if ub_split_axis < max_broadcast_axis:
            if not self._is_all_one_before_max_broadcast_axis(
                    ub_split_axis, max_broadcast_axis):
                return

        self._is_need_update_compute_at_axis = True

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

    def _fused_compute_at_axis(self):
        """
        Calculate fuse and compute at axis

        Parameters:
        ----------
        None

        Returns
        -------
        None
        """

        # pylint: disable=too-many-locals
        if self._op_type != OpSpecTypes.RELU_GRAD_V2:
            return

        res = self._last_output_tensor

        block_tiling_result = self._tiling_result["block_tiling"]
        block_split_axis = block_tiling_result["axis"]
        res_block_inner = block_tiling_result["inner_itervar"]
        ub_tiling_result = self._tiling_result["ub_tiling"]
        ub_split_axis = ub_tiling_result["axis"]
        res_ub_outer = ub_tiling_result["outer_itervar"]

        if ub_split_axis == 0:
            return

        if ub_split_axis == block_split_axis:
            return

        need_fuse_list = [res_ub_outer]
        for i in range(ub_split_axis - 1, block_split_axis, -1):
            need_fuse_list.append(res.op.axis[i])
        need_fuse_list.append(res_block_inner)

        fused_axis = need_fuse_list[0]
        for i in range(1, len(need_fuse_list)):
            fused_axis = self._schedule[res].fuse(fused_axis, need_fuse_list[i])

        self._preload_fused_axis = fused_axis
        self._is_preload_fused_axis_scene = True

        if self._batch_bind_only:
            self._schedule[res].pragma(fused_axis,
                                       "json_info_batchBindOnly", 1)


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
        # pylint: disable=too-many-locals
        if self._need_multi_core:
            res = self._last_output_tensor
            block_tiling_result = self._tiling_result["block_tiling"]
            block_split_axis = block_tiling_result["axis"]
            res_block_outer = block_tiling_result["outer_itervar"]

            if block_split_axis == 0:
                self._batch_bind_only = True

            need_fuse_list = [res_block_outer]
            for i in range(block_split_axis - 1, -1, -1):
                need_fuse_list.append(res.op.axis[i])
            fused_axis = need_fuse_list[0]
            for i in range(1, len(need_fuse_list)):
                fused_axis = self._schedule[res].fuse(fused_axis,
                                                      need_fuse_list[i])

            self._multi_core_fused_axis = fused_axis
            self._multi_core_bind_tensor = res

    # pylint: too-many-nested-blocks
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
        # pylint: disable=too-many-locals
        def __get_compute_at_scope(tensor, compute_at_outer):
            if self._op_type != OpSpecTypes.NORMALIZE_SCALE:
                return compute_at_outer
            input_shape = self._shape_to_list(tensor.shape)
            if sum(input_shape[:]) == len(input_shape):
                compute_at_outer = self._multi_core_fused_axis
            return compute_at_outer

        res = self._last_output_tensor
        ub_tiling_result = self._tiling_result["ub_tiling"]
        res_ub_outer = ub_tiling_result["outer_itervar"]
        ub_split_axis = ub_tiling_result["axis"]

        block_tiling_result = self._tiling_result["block_tiling"]
        block_split_axis = block_tiling_result["axis"]
        res_block_inner = block_tiling_result["inner_itervar"]
        res_block_outer = block_tiling_result["outer_itervar"]

        self._fused_compute_at_axis()
        preload_fused_axis = self._preload_fused_axis

        # pylint: disable=too-many-nested-blocks
        for i in self._cache_read_tensors_and_buffer_map:
            if self._is_need_update_compute_at_axis:
                readers_tensor = self._cache_read_tensors_and_readers_map[i]
                read_buffer = self._cache_read_tensors_and_buffer_map[i]

                if self._is_broadcast_not_last_axis_tensor(readers_tensor[0]):
                    # like (8, 128, 1, 2, 1, 1) (1, 1, 1, 2, 3, 1) scene,
                    # both tensor need to do broadcast, but not all tensors can
                    # move the compute at axis to outward
                    max_broadcast_axis = \
                        self._find_max_broadcast_axis_of_tensor(
                            readers_tensor[0])
                    if max_broadcast_axis < ub_split_axis:
                        compute_at_outer = res_ub_outer
                    else:
                        if block_split_axis == ub_split_axis:
                            compute_at_outer = self._multi_core_fused_axis
                        elif block_split_axis == ub_split_axis - 1:
                            compute_at_outer = res_block_inner
                        else:
                            shape = self._shape_to_list(i.shape)
                            if self._is_all_one_before_broadcast_axis(
                                    shape, max_broadcast_axis):
                                if block_split_axis == 0:
                                    compute_at_outer = res_block_outer
                                else:
                                    compute_at_outer = \
                                        self._multi_core_fused_axis
                            else:
                                compute_at_outer = \
                                    self._schedule[res].op.axis[
                                        ub_split_axis - 1]
                    para = {"parent": self._schedule[res],
                            "scope": compute_at_outer}
                else:
                    compute_at_outer = __get_compute_at_scope(i, res_ub_outer)
                    para = {"parent": self._schedule[res],
                            "scope": compute_at_outer}

                self._compute_at_map[read_buffer] = para
            else:
                read_buffer = self._cache_read_tensors_and_buffer_map[i]
                if self._is_preload_fused_axis_scene:
                    para = {"parent": self._schedule[res],
                            "scope": preload_fused_axis}
                else:
                    compute_at_outer = __get_compute_at_scope(i, res_ub_outer)
                    para = {"parent": self._schedule[res],
                            "scope": compute_at_outer}
                self._compute_at_map[read_buffer] = para

        for i in self._cache_write_tensors_and_buffer_map:
            write_buffer = self._cache_write_tensors_and_buffer_map[i]
            if self._is_preload_fused_axis_scene:
                para = {"parent": self._schedule[res],
                        "scope": preload_fused_axis}
            else:
                compute_at_outer = __get_compute_at_scope(i, res_ub_outer)
                para = {"parent": self._schedule[res],
                        "scope": compute_at_outer}
            self._compute_at_map[write_buffer] = para

        for i in self._mid_output_tensors:
            para = {"parent": self._schedule[res], "scope": res_ub_outer}
            self._compute_at_map[i] = para

    def _calculate_double_buffer(self):
        """
        double buffer operations
        read_buffer : the all read_cache for input in ub, type is list
        """
        # Because transpose will apply for new buffer,
        # ping pong reuse the same buffer, the db flow will be interrupted,
        # and there may be a risk of ub cross-border, so the db is not enabled.
        if self._special_broadcast_with_transpose:
            return
        shape = self._shape_to_list(self._last_output_tensor.shape)

        block_tiling_para = self._tiling_para["block_tiling"]
        block_split_axis = block_tiling_para["axis"]
        block_split_inner_size = block_tiling_para["factor"]

        ub_tiling_para = self._tiling_para["ub_tiling"]
        ub_split_axis = ub_tiling_para["axis"]
        ub_split_inner = ub_tiling_para["factor"]
        if self._need_db is None:
            self._need_db = self._need_double_buffer(
                shape, block_split_axis, block_split_inner_size, ub_split_axis,
                ub_split_inner)

    def _calculate_emit_insn(self): # pylint: disable=too-many-locals
        """
        Calculate the instruction map of tensor

        Parameters:
        ----------
        None

        Returns
        -------
        None
        """
        res = self._last_output_tensor
        ub_tiling_result = self._tiling_result["ub_tiling"]
        ub_split_axis = ub_tiling_result["axis"]
        res_ub_inner = ub_tiling_result["inner_itervar"]

        special_broadcast_insn_map = {"vector_mul": "vector_mul_with_broadcast",
                                      "vector_div": "vector_div_with_broadcast",
                                      "vector_add": "vector_add_with_broadcast",
                                      "vector_sub": "vector_sub_with_broadcast",
                                      }

        self._get_emit_insn_map()
        self._get_reg_emit_insn_map()

        def __is_special_non_last_broadcast_scene(i):
            if insn in self._broadcast_enhance_insn_map.keys() and \
                    self._is_special_tensor_of_broadcast_not_last_axis(i):
                return True
            return False

        def __is_special_non_last_broadcast_factor16_scene(i):
            if insn in special_broadcast_insn_map.keys() and \
                    self._is_special_factor16_broadcast_tensor(i):
                return True
            return False

        def __is_special_mix_broadcast_scene(i):
            if insn in self._broadcast_enhance_insn_map.keys() and \
                    self._is_mix_broadcast_not_last_axis(i):
                return True
            return False

        def __is_special_non_32align_broadcast_scene(i):
            if insn in self._non_32align_broadcast_insn_map.keys() and \
                    self._is_non_32align_broadcast_not_last_axis(i):
                return True
            return False

        def __is_special_32align_broadcast_scene(i):
            if insn in self._broadcast_enhance_insn_map.keys() and \
                    self._is_32align_broadcast_not_last_axis(i):
                return True
            return False

        def __get_inst(insn, i):
            # special process for sub (1,1,1,32) (32,224,224,3)
            if self._special_non_last_broadcast_scene:
                if __is_special_non_last_broadcast_scene(i):
                    insn = self._broadcast_enhance_insn_map.get(insn)
            elif self._special_non_last_broadcast_factor16_scene:
                if __is_special_non_last_broadcast_factor16_scene(i):
                    insn = special_broadcast_insn_map.get(insn)
            elif self._special_mix_broadcast_scene:
                if __is_special_mix_broadcast_scene(i):
                    insn = self._broadcast_enhance_insn_map.get(insn)
            elif self._special_non_32align_broadcast_scene:
                if __is_special_non_32align_broadcast_scene(i):
                    insn = self._non_32align_broadcast_insn_map.get(insn)
            elif self._special_32align_broadcast_scene:
                if __is_special_32align_broadcast_scene(i):
                    insn = self._broadcast_enhance_insn_map.get(insn)
            elif self._special_broadcast_with_transpose:
                insn = "broadcast_with_transpose"
            else:
                pass
            return insn

        for i in self._cache_read_tensors_and_buffer_map:
            read_buffer = self._cache_read_tensors_and_buffer_map[i]
            para = {"scope": read_buffer.op.axis[ub_split_axis],
                    "instruction": 'dma_copy'}
            self._emit_insn_map[read_buffer] = para

        for i in self._cache_write_tensors_and_buffer_map:
            write_buffer = self._cache_write_tensors_and_buffer_map[i]
            insn = self._calculate_emit_insn_map(write_buffer)
            if insn in ["unified_broadcast", "broadcast_for_tensor"]:
                self.max_last_broadcast_axis_offset[
                    self._schedule[i].op.input_tensors[0].name] = \
                    self._find_max_broadcast_last_axis_offset(i)

            insn = __get_inst(insn, i)

            if insn == "vector_multiple":
                self._do_emit_multiple_insn(i, write_buffer)
                continue

            # Special compute for mul
            if "manual_mul_without_broadcast" in i.name:
                axis = 0
                small_buffer = i.op.input_tensors[1]
                for j in range(len(small_buffer.shape)):
                    if int(small_buffer.shape[j]) != int(i.shape[j]):
                        axis = j
                        break
                para = {"scope": write_buffer.op.axis[axis],
                        "instruction": insn}
                self._emit_insn_map[write_buffer] = para
                continue

            emit_insn_axis = ub_split_axis
            if self._special_32align_broadcast_scene:
                emit_insn_axis = -3
            para = {"scope": write_buffer.op.axis[emit_insn_axis],
                    "instruction": insn}
            self._emit_insn_map[write_buffer] = para

        for out_tensor in self._mid_output_tensors:
            para = {"scope": out_tensor.op.axis[ub_split_axis],
                    "instruction": 'dma_copy'}
            self._emit_insn_map[out_tensor] = para

        self._emit_insn_map[res] = {"scope": res_ub_inner,
                                    "instruction": 'dma_copy'}

        # Special process for (x, 100, 1, 4) to (x, 100, 2, 4) as result
        if self._less_32_core_middle_broadcast_scene:
            self._emit_insn_map[res] = {"scope": res_ub_inner,
                                        "instruction": 'broadcast_for_tensor_opt_mid_le32'}
        if self._elewise_binary_phony_as_output:
            self._emit_insn_map[res] = {"scope": res_ub_inner,
                                        "instruction": 'elewise_binary_phony'}

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
            if  cache_buffer.dtype == "float32":
                return False
        return True

    def _do_emit_multiple_insn(self, tensor, tensor_buffer):
        """
        do emit multiple insn
        """
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
        ub_tiling_result = self._tiling_result["ub_tiling"]
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

    def _check_is_last_axis_broadcast(self, tensor):
        """
        Check if the last axis broadcast scene

        Parameters:
        ----------
        tensors :  tensor to be checked

        Returns
        -------
        Bool: True or False
        """
        if tensor.op.tag == "broadcast_for_tensor":
            if list(tensor.op.input_tensors) and \
                    self._shape_to_list(tensor.op.input_tensors[0].shape)[-1] == 1:
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
                        if self.is_exceeded_mid_broadcast_threshold(tensor):
                            return True

        return False

    def is_exceeded_mid_broadcast_threshold(self, tensor):
        """Check if mid-broadcast tensor exceeded the size threshold"""
        block_size = util.get_align_factor(tensor.dtype)[0]
        unit_size = self.get_mid_broadcast_tensor_unit_size(tensor)
        if unit_size > math.ceil(block_size * NON_LAST_BROADCAST_UNIT_THRESHOLD) or \
                unit_size % block_size == 0:
            return True
        return False

    def get_mid_broadcast_tensor_unit_size(self, tensor):
        """Get mid-broadcast tensor size"""
        original_tensor = tensor.op.input_tensors[0]
        original_shape = list(reversed(self._shape_to_list(original_tensor.shape)))
        broadcast_shape = list(reversed(self._shape_to_list(tensor.shape)))
        size = 1
        for shape_idx, shape_size in enumerate(original_shape):
            if shape_size == broadcast_shape[shape_idx]:
                size *= shape_size
            else:
                break
        return size

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
        return False

    def _is_special_factor16_broadcast_sence(self):
        """
        Judge is special 32 byte align sence of the non-last broadcast

        Parameters:
        ----------
        None

        Returns
        -------
        True or False
        """
        if cceconf.get_soc_spec("SOC_VERSION") not in \
            ["Ascend910", "Ascend610"]:
            return False

        if not self._is_only_broadcast_not_last_axis():
            return False

        if not self._is_special_factor16_broadcast_not_last_axis_scene():
            return False

        return True

    def _is_special_factor16_broadcast_not_last_axis_scene(self):
        """
        Judge is special 32 byte align sence of the non-last axis broadcast

        Parameters:
        ----------
        None

        Returns
        -------
        True or False
        """
        # process like (1,16,1,1,16) (a,16,b,c,16) scene
        if not self._broadcast_not_last_axis_tensors:
            return False

        special_insn_map = {"vector_mul": "vector_mul_with_broadcast",
                            "vector_div": "vector_div_with_broadcast",
                            "vector_add": "vector_add_with_broadcast",
                            "vector_sub": "vector_sub_with_broadcast",
                            }

        for broadcast_tensor in self._broadcast_not_last_axis_tensors:
            if list(broadcast_tensor.op.input_tensors):
                original_tensor = broadcast_tensor.op.input_tensors[0]
                original_shape = self._shape_to_list(original_tensor.shape)
                broadcast_shape = self._shape_to_list(broadcast_tensor.shape)
                date_type = broadcast_tensor.dtype.lower()
                if date_type != "float32":
                    return False

                if len(original_shape) != 5:
                    return False
                if original_shape[-1] != 16:
                    return False
                if original_shape[0] != 1 or original_shape[2] != 1 \
                        or original_shape[3] != 1:
                    return False
                if broadcast_shape[2] == 1 or broadcast_shape[3] == 1:
                    return False

                if broadcast_tensor not in self._mid_tensor_dst_tensor_map.keys():
                    return False

                self._get_emit_insn_map()
                self._get_reg_emit_insn_map()

                dest_tensor = self._mid_tensor_dst_tensor_map[broadcast_tensor]
                # like xdiv operator, broadcast destination are vadd and vabs,
                # as long as one of the destination is not supported broadcast
                # enhance, disable broadcast enhance function
                for tensor in dest_tensor:
                    insn = self._calculate_emit_insn_map(tensor)
                    if insn not in special_insn_map.keys():
                        return False
            else:
                return False

        return True

    def _is_special_factor16_broadcast_tensor(self, tensor):
        """
        Judge is special sence of the non-last axis broadcast

        Parameters:
        ----------
        None

        Returns
        -------
        True or False
        """
        # process like (1,16,1,1,16) (a,16,b,c,16) scene
        if not tensor.op.input_tensors:
            return False

        broadcast_flag = False
        in_tensor = None
        for in_tensor in list(tensor.op.input_tensors):
            insn = self._reg_insn_map.get(in_tensor.op.tag)
            if insn == "unified_broadcast":
                broadcast_flag = True
                break

        if not broadcast_flag:
            return False

        broadcast_tensor = in_tensor

        date_type = broadcast_tensor.dtype.lower()
        if date_type != "float32":
            return False
        broadcast_shape = self._shape_to_list(broadcast_tensor.shape)
        if list(broadcast_tensor.op.input_tensors):
            original_tensor = broadcast_tensor.op.input_tensors[0]
            original_shape = self._shape_to_list(original_tensor.shape)

            if len(original_shape) != 5:
                return False
            if original_shape[-1] != 16:
                return False
            if original_shape[0] != 1 or original_shape[2] != 1 \
                    or original_shape[3] != 1:
                return False
            if broadcast_shape[2] == 1 or broadcast_shape[3] == 1:
                return False
        else:
            return False
        return True

    def _is_factor16_broadcast_input_tensor(self, tensor):
        """
        Judge is special input tensor

        Parameters:
        ----------
        None

        Returns
        -------
        True or False
        """
        # process like (1,c1,1,1,16) (n,c1,h,w,16) scene
        original_shape = self._shape_to_list(tensor.shape)

        if len(original_shape) != 5:
            return False
        if original_shape[-1] != 16:
            return False
        if original_shape[0] != 1 or original_shape[2] != 1 \
                or original_shape[3] != 1:
            return False

        return True

    def _is_special_broadcast_sence(self, block_split_axis,
                                    block_split_inner_size):
        def __check_support_version():
            product_version = cceconf.get_soc_spec("SOC_VERSION")
            if product_version != "Ascend910":
                if product_version == "Ascend310":
                    if self._op_type == OpSpecTypes.NORMALIZE_SCALE:
                        return True
                return False
            return True

        if not __check_support_version():
            return False

        if not self._is_only_broadcast_not_last_axis():
            return False

        if not self._is_special_sence_of_broadcast_not_last_axis(
                block_split_axis, block_split_inner_size):
            return False

        return True

    def _is_less_32_core_middle_broadcast_out_scene(self, block_split_axis,
                                                    block_split_inner_size):
        if cceconf.get_soc_spec("SOC_VERSION") != "Ascend910":
            return False

        if not self._is_only_broadcast_not_last_axis():
            return False

        if not self._is_less_32_core_middle_broadcast_out(
                block_split_axis, block_split_inner_size):
            return False

        return True

    def _is_less_32_core_middle_broadcast_out(self, block_split_axis,
                                              block_split_inner_size):
        # (a, b, 1, 4) -> (a, b, 2, 4) where a <= 32, 16 <= b <= 4080
        if not self._broadcast_not_last_axis_tensors:
            return False

        if block_split_axis != 0:
            return False

        if block_split_inner_size != 1:
            return False

        is_out = False
        # pylint: disable=too-many-nested-blocks
        for broadcast_tensor in self._broadcast_not_last_axis_tensors:
            if list(broadcast_tensor.op.input_tensors):
                original_tensor = broadcast_tensor.op.input_tensors[0]
                original_shape = self._shape_to_list(original_tensor.shape)
                broadcast_shape = self._shape_to_list(broadcast_tensor.shape)
                # (x <= 32, x, x, x) only
                if original_shape[0] // block_split_inner_size > 32:
                    continue
                # (x <= 32, x, x, 4) only
                if original_shape[-1] != 4:
                    continue
                # fp32 only
                if original_tensor.dtype != "float32":
                    continue
                # (x <= 32, x, 1, 4) to (x <= 32, x, 2, 4) only
                if original_shape[-2] != 1 or broadcast_shape[-2] != 2:
                    continue
                # data per core must be larger than MULTI_CORE_THRESHOLD
                if original_shape[-3] * original_shape[-1] * \
                        DTYPE_WIDTH_MAP[original_tensor.dtype] < \
                        BROADCAST_LAST_AXIS_THRESHOLD:
                    continue
                # data per core must be smaller than max_repeat
                if original_shape[-3] \
                        * original_shape[-1] \
                        * DTYPE_WIDTH_MAP[original_tensor.dtype] > 255 * 16:
                    continue
                if broadcast_tensor not in self._mid_tensor_dst_tensor_map.keys():
                    is_out = True
        return is_out

    def _is_mix_broadcast_out_scene(self, block_split_axis, block_split_inner_size):
        if cceconf.get_soc_spec("SOC_VERSION") != "Ascend910":
            return False

        if not self._is_only_broadcast_not_last_axis():
            return False

        if not self._is_mix_broadcast_out(
                block_split_axis, block_split_inner_size):
            return False

        return True

    def _is_mix_broadcast_out(self, block_split_axis, block_split_inner_size):
        if not self._broadcast_not_last_axis_tensors:
            return False

        if block_split_axis != 0:
            return False

        shape = self._shape_to_list(self._last_output_tensor.shape)
        if block_split_inner_size == shape[0] and shape[-1] != 4:
            if not (len(shape) == 3 and shape[0] == 32 and shape[1] >= 4):
                return False

        def __is_correct_shape(original_shape1, original_shape2, broadcast_shape):
            if len(original_shape1) != 6 or len(original_shape2) != 6:
                return False
            if original_shape1[-1] > 64 or \
                    original_shape1[-1] != original_shape2[-1] or \
                    original_shape1[-1] == 1:
                return False
            for i in range(1, len(broadcast_shape) - 2, 1):
                if broadcast_shape[i] == 1 or \
                        (original_shape1[i] != 1 and original_shape2[i] != 1):
                    return False
                if original_shape1[-2] == 1 and original_shape1[i] == 1:
                    return False
                if original_shape2[-2] == 1 and original_shape2[i] == 1:
                    return False
            return True

        def __is_shpae_len_6_case(original_shape1, original_shape2, broadcast_shape):
            if block_split_inner_size != 1:
                return False
            dype = self._broadcast_not_last_axis_tensors[0].dtype
            out_size = \
                broadcast_shape[-1] * broadcast_shape[-2] * DTYPE_WIDTH_MAP[dype] * 2
            if out_size < 16:
                return False
            if not __is_correct_shape(original_shape1, original_shape2, broadcast_shape):
                return False
            return True

        def __is_shpae_len_3_case(original_shape1, original_shape2, broadcast_shape):
            if len(original_shape1) != 3 or \
                    original_shape1[-1] != original_shape2[-1] or \
                    original_shape1[-1] > 64:
                return False
            if broadcast_shape[0] == 1 or (
                    original_shape1[0] != 1 and original_shape2[0] != 1) or \
                    (original_shape1[0] == 1 and original_shape2[0] == 1):
                return False
            if broadcast_shape[1] == 1 or (
                    original_shape1[1] != 1 and original_shape2[1] != 1) or \
                    (original_shape1[1] == 1 and original_shape2[1] == 1):
                return False
            if original_shape1[-1] != original_shape2[-1]:
                return False
            return True

        def __is_real_mix_broadcast_out():
            self._get_emit_insn_map()
            self._get_reg_emit_insn_map()

            if len(self._broadcast_not_last_axis_tensors) != 2:
                return False

            for broadcast_tensor in self._broadcast_not_last_axis_tensors:
                if list(broadcast_tensor.op.input_tensors):
                    if broadcast_tensor not in self._mid_tensor_dst_tensor_map.keys():
                        return False
                    dest_tensor = self._mid_tensor_dst_tensor_map[broadcast_tensor]
                    for tensor in dest_tensor:
                        insn = self._calculate_emit_insn_map(tensor)
                        if insn not in self._broadcast_enhance_insn_map.keys():
                            return False
                else:
                    return False

            original_tensor1 = \
                self._broadcast_not_last_axis_tensors[0].op.input_tensors[0]
            original_shape1 = self._shape_to_list(original_tensor1.shape)
            original_tensor2 = \
                self._broadcast_not_last_axis_tensors[1].op.input_tensors[0]
            original_shape2 = self._shape_to_list(original_tensor2.shape)
            broadcast_shape = \
                self._shape_to_list(self._broadcast_not_last_axis_tensors[0].shape)
            if len(broadcast_shape) == 6:
                return __is_shpae_len_6_case(original_shape1,
                                             original_shape2, broadcast_shape)
            if len(broadcast_shape) == 3:
                return __is_shpae_len_3_case(original_shape1,
                                             original_shape2, broadcast_shape)
            return False

        return __is_real_mix_broadcast_out()

    def _is_non_32align_broadcast_out_scene(self, block_split_axis,
                                            block_split_inner_size,
                                            shape, max_ub_count):
        if cceconf.get_soc_spec("SOC_VERSION") != "Ascend910":
            return False

        if not self._is_only_broadcast_not_last_axis():
            return False

        if not self._is_non_32align_broadcast_out(
                block_split_axis, block_split_inner_size):
            return False

        if not self._is_inner_32lign_broadcast_out(
                block_split_axis, block_split_inner_size,
                shape, max_ub_count):
            return False

        return True

    def _is_non_32align_broadcast_out(self, block_split_axis, block_split_inner_size):
        if not self._broadcast_not_last_axis_tensors:
            return False

        if block_split_axis != 0:
            return False

        def __is_shape_len_5_case(original_shape, broadcast_shape):
            if block_split_inner_size != 1:
                return False
            if original_shape[-1] > 64:
                return False
            if broadcast_shape[0] == 1 or original_shape[0] != 1:
                return False
            if broadcast_shape[-2] == 1 or original_shape[-2] != 1:
                return False
            for i in range(1, len(broadcast_shape) - 2, 1):
                if broadcast_shape[i] == 1 or broadcast_shape[i] != original_shape[i]:
                    return False
            return True

        def __is_shape_len_3_case(original_shape, broadcast_shape):
            if broadcast_shape[-2] == 1 or original_shape[-2] != 1:
                return False
            for i in range(1, len(broadcast_shape) - 2, 1):
                if broadcast_shape[i] == 1 or broadcast_shape[i] != original_shape[i]:
                    return False
            return True

        self._get_emit_insn_map()
        self._get_reg_emit_insn_map()

        if len(self._broadcast_not_last_axis_tensors) != 1:
            return False

        for broadcast_tensor in self._broadcast_not_last_axis_tensors:
            if list(broadcast_tensor.op.input_tensors):
                if broadcast_tensor not in self._mid_tensor_dst_tensor_map.keys():
                    return False
                dest_tensor = self._mid_tensor_dst_tensor_map[broadcast_tensor]
                for tensor in dest_tensor:
                    insn = self._calculate_emit_insn_map(tensor)
                    if insn not in self._non_32align_broadcast_insn_map.keys():
                        return False
            else:
                return False

        original_tensor = self._broadcast_not_last_axis_tensors[0].op.input_tensors[0]
        original_shape = self._shape_to_list(original_tensor.shape)
        broadcast_shape = self._shape_to_list(self._broadcast_not_last_axis_tensors[0].shape)
        if len(broadcast_shape) == 5:
            return __is_shape_len_5_case(original_shape, broadcast_shape)
        if len(broadcast_shape) == 3:
            return __is_shape_len_3_case(original_shape, broadcast_shape)
        return False


    def _is_inner_32lign_broadcast_out(self, block_split_axis, block_split_inner_size,
                                       shape, max_ub_count):
        def __maxis_mul(maxis):
            if not maxis:
                return 1
            return reduce(lambda x, y: x * y, maxis)

        def __get_loop_cout(ub_split_axis, ub_split_factor):
            loop_count = 1
            loop_count_out = 1
            for i in range(len(shape) - 3, ub_split_axis, -1):
                loop_count_temp = loop_count
                loop_count = loop_count * shape[i]
                if loop_count > 512:
                    loop_count_out = __maxis_mul(shape[ub_split_axis + 1:i + 1])
                    loop_count = loop_count_temp
                    break
            if len(shape) == 3:
                loop_count_out = ub_split_factor
            return loop_count, loop_count_out

        ub_split_axis, ub_split_factor = \
            self._get_ub_tiling(shape, block_split_axis, block_split_inner_size, max_ub_count)
        loop_count, loop_count_out = __get_loop_cout(ub_split_axis, ub_split_factor)
        dtype = self._last_output_tensor.dtype.lower()
        size = DTYPE_WIDTH_MAP[dtype] * 2 * shape[-1] * shape[-2] * loop_count
        ub_split_remain = shape[ub_split_axis] % ub_split_factor
        if block_split_axis == ub_split_axis:
            ub_split_remain = block_split_inner_size % ub_split_factor
        if ub_split_remain != 0 or (size % 32 != 0 and loop_count_out != 1):
            return False
        return True

    def _is_32align_broadcast_out_scene(self, block_split_axis, block_split_inner_size):
        if cceconf.get_soc_spec("SOC_VERSION") != "Ascend910":
            return False

        if not self._is_only_broadcast_not_last_axis():
            return False

        if not self._is_32align_broadcast_out(
                block_split_axis, block_split_inner_size):
            return False

        return True

    def _is_32align_broadcast_out(self, block_split_axis, block_split_inner_size):
        if not self._broadcast_not_last_axis_tensors:
            return False

        if block_split_axis != 0:
            return False

        if block_split_inner_size != 1:
            return False

        def __is_broadcast_correct_tensor():
            self._get_emit_insn_map()
            self._get_reg_emit_insn_map()
            if len(self._broadcast_not_last_axis_tensors) != 1:
                return False
            for broadcast_tensor in self._broadcast_not_last_axis_tensors:
                if list(broadcast_tensor.op.input_tensors):
                    if broadcast_tensor not in self._mid_tensor_dst_tensor_map.keys():
                        return False
                    dest_tensor = self._mid_tensor_dst_tensor_map[broadcast_tensor]
                    for tensor in dest_tensor:
                        insn = self._calculate_emit_insn_map(tensor)
                        if insn not in self._broadcast_enhance_insn_map.keys():
                            return False
                else:
                    return False
            return True


        def __is_real_32align_broadcast_out():
            if not __is_broadcast_correct_tensor():
                return False

            original_tensor = self._broadcast_not_last_axis_tensors[0].op.input_tensors[0]
            original_shape = self._shape_to_list(original_tensor.shape)
            broadcast_shape = \
                self._shape_to_list(self._broadcast_not_last_axis_tensors[0].shape)
            if len(original_shape) != 5 or original_shape[-1] > 64:
                return False
            if original_shape[-2] != 1 or broadcast_shape[-2] != 1:
                return False
            if original_shape[-3] != 1 or broadcast_shape[-3] == 1:
                return False
            dtype = self._broadcast_not_last_axis_tensors[0].dtype
            last_size = original_shape[-1] * DTYPE_WIDTH_MAP[dtype] * 2
            if original_shape[-1] != broadcast_shape[-1] or last_size % 32 != 0:
                return False
            for i in range(0, len(broadcast_shape) - 3, 1):
                if original_shape[i] == broadcast_shape[i]:
                    continue
                else:
                    return False
            return True

        return __is_real_32align_broadcast_out()

    def _get_block_split_factor(self, block_split_inner_size, block_split_axis,
                                core_num, broadcast_tensor):
        block_split_factor = block_split_inner_size
        if self.block_tiling_use_nparts_mode:
            tensor_shape = self._shape_to_list(broadcast_tensor.shape)
            if core_num != tensor_shape[block_split_axis]:
                block_split_factor = tensor_shape[block_split_axis] / core_num

        return block_split_factor

    def _get_special_broadcast_optimize_value(self):
        """
        Get the specail broadcast pattern optimize threshold value
        """
        product_version = cceconf.get_soc_spec("SOC_VERSION")
        if product_version == "Ascend910":
            return 64
        return 32

    def _normalize_scale_opt(self):
        """
        Check is the normalzie scale optimize pattern
        """
        if self._op_type != OpSpecTypes.NORMALIZE_SCALE:
            return

        shape = self._shape_to_list(self._last_output_tensor.shape)
        max_ub_count = self._max_ub_count

        # now just support NCHW NWHC
        if len(shape) != 4:
            return

        # C dimension needs to do broadcast
        for broadcast_tensor in self._broadcast_not_last_axis_tensors:
            if list(broadcast_tensor.op.input_tensors):
                original_tensor = broadcast_tensor.op.input_tensors[0]
                original_shape = self._shape_to_list(original_tensor.shape)
                broadcast_shape = self._shape_to_list(broadcast_tensor.shape)
                if original_shape[1] != 1 or broadcast_shape[1] == 1:
                    return

        # ensure the multi-core and ub do not split the same axis
        count = 1
        for i in range(1, len(shape)):
            count = count * shape[i]

        if count > max_ub_count:
            self._normalize_scale_enhance_opt = True

        return

    def _is_multicore_satisfy_pattern(self, axis, block_split_axis,
                                      block_split_factor):
        """
        Judge the multicore tiling whether satisfy special broadcast pattern
        """
        if block_split_axis != axis:
            return False
        if block_split_factor == 1:
            return True
        if self._normalize_scale_enhance_opt:
            return True
        return False

    def _is_last_two_dim_pattern(self, axis, original_shape, broadcast_shape):
        """
        Check the last two dim whether satisfy special broadcast pattern
        """
        if axis != len(original_shape) - 2:
            return False

        if original_shape[axis + 1] != 1:
            return True

        if self._op_type == OpSpecTypes.NORMALIZE_SCALE:
            if original_shape[axis + 1] == broadcast_shape[axis + 1]:
                return True

        return False

    def _is_special_sence_of_broadcast_not_last_axis(self, block_split_axis,
                                                     block_split_inner_size):
        """
        Judge is special sence of the non-last axis broadcast

        Parameters:
        ----------
        None

        Returns
        -------
        True or False
        """
        # (1,1,1,3)->(32,224,224,3)
        if not self._broadcast_not_last_axis_tensors:
            return False

        core_num = cceconf.get_soc_spec("CORE_NUM")

        threshold_value = self._get_special_broadcast_optimize_value()
        self._normalize_scale_opt()

        # pylint: disable=too-many-nested-blocks
        for broadcast_tensor in self._broadcast_not_last_axis_tensors:
            if list(broadcast_tensor.op.input_tensors):
                original_tensor = broadcast_tensor.op.input_tensors[0]
                original_shape = self._shape_to_list(original_tensor.shape)
                broadcast_shape = self._shape_to_list(broadcast_tensor.shape)
                block_split_factor = self._get_block_split_factor(
                    block_split_inner_size, block_split_axis, core_num,
                    broadcast_tensor)

                if original_shape[-1] > threshold_value:
                    return False
                for i in range(0, len(original_shape) - 1, 1):
                    if original_shape[i] == 1:
                        continue
                    elif original_shape[i] != 1:
                        # process like (32,1,4) (32,68,4) scene
                        if self._is_multicore_satisfy_pattern\
                                    (i, block_split_axis, block_split_factor):
                            continue
                        # process like (32,32,32,3,2) (1,1,1,3,2) scene
                        elif self._is_last_two_dim_pattern(i, original_shape,
                                                           broadcast_shape):
                            last_dim = original_shape[-1]
                            second_last_dim = original_shape[-2]
                            if last_dim * second_last_dim < 16:
                                continue
                            else:
                                return False
                        else:
                            return False
                    else:
                        return False

                if broadcast_tensor not in self._mid_tensor_dst_tensor_map.keys():
                    return False

                self._get_emit_insn_map()
                self._get_reg_emit_insn_map()

                dest_tensor = self._mid_tensor_dst_tensor_map[broadcast_tensor]
                # like xdiv operator, broadcast destination are vadd and vabs,
                # as long as one of the destination is not supported broadcast
                # enhance, disable broadcast enhance function
                for tensor in dest_tensor:
                    insn = self._calculate_emit_insn_map(tensor)
                    if insn not in self._broadcast_enhance_insn_map.keys():
                        return False

            else:
                return False

        return True

    def _get_block_split_factor_by_broadcast_tensor(
            self, block_split_inner_size, broadcast_tensor, core_num,
            block_split_axis):
        block_split_factor = block_split_inner_size
        tensor_shape = self._shape_to_list(broadcast_tensor.shape)
        if self.block_tiling_use_nparts_mode:
            if core_num == tensor_shape[block_split_axis]:
                block_split_factor = tensor_shape[block_split_axis] / core_num

        return block_split_factor

    # pylint: undefined-loop-variable
    def _is_special_tensor_of_broadcast_not_last_axis(self, tensor):
        """
        Judge is special sence of the non-last axis broadcast

        Parameters:
        ----------
        None

        Returns
        -------
        True or False
        """
        def __is_enable_broadcast_axis_multiply_flag(original_shape):
            if self._op_type != OpSpecTypes.NORMALIZE_SCALE:
                return True
            if original_shape[-1] == 1 or original_shape[-2] == 1:
                return False
            return True

        # pylint: disable=too-many-locals
        # (1,1,1,3)->(32,224,224,3)
        if not tensor.op.input_tensors:
            return False

        broadcast_flag = False
        in_tensor = None
        for in_tensor in list(tensor.op.input_tensors):
            insn = self._reg_insn_map.get(in_tensor.op.tag)
            if insn == "unified_broadcast":
                broadcast_flag = True
                break

        if not broadcast_flag:
            return False

        if in_tensor not in self._broadcast_not_last_axis_tensors:
            return False

        block_tiling_para = self._tiling_para["block_tiling"]
        block_split_axis = block_tiling_para["axis"]
        block_split_inner_size = block_tiling_para["factor"]

        core_num = cceconf.get_soc_spec("CORE_NUM")
        threshold_value = self._get_special_broadcast_optimize_value()

        broadcast_axis_multiply_flag = False
        cce_emitinsn_params.cceEmitParamsIns.del_param(
            "broadcast_axis_multiply_flag")
        cce_emitinsn_params.cceEmitParamsIns.insert_param(
            "broadcast_axis_multiply_flag", broadcast_axis_multiply_flag)

        # pylint: disable=undefined-loop-variable
        broadcast_tensor = in_tensor
        block_split_factor = self._get_block_split_factor_by_broadcast_tensor(
            block_split_inner_size, broadcast_tensor, core_num,
            block_split_axis)
        if list(broadcast_tensor.op.input_tensors):
            original_tensor = broadcast_tensor.op.input_tensors[0]
            original_shape = self._shape_to_list(original_tensor.shape)
            broadcast_shape = self._shape_to_list(broadcast_tensor.shape)
            if original_shape[-1] > threshold_value:
                return False
            for i in range(0, len(original_shape) - 1, 1):
                if original_shape[i] == 1:
                    continue
                elif original_shape[i] != 1:
                    # process like (32,1,4) (32,68,4) scene
                    if self._is_multicore_satisfy_pattern(i, block_split_axis,
                                                          block_split_factor):
                        continue
                    # process like (32,32,32,3,2) (1,1,1,3,2) scene
                    elif self._is_last_two_dim_pattern(i, original_shape,
                                                       broadcast_shape):
                        last_dim = original_shape[-1]
                        second_last_dim = original_shape[-2]
                        if last_dim * second_last_dim < 16:
                            broadcast_axis_multiply_flag = \
                                __is_enable_broadcast_axis_multiply_flag(
                                    original_shape)
                            continue
                        else:
                            return False
                    else:
                        return False
                else:
                    return False

        else:
            return False

        if broadcast_axis_multiply_flag:
            cce_emitinsn_params.cceEmitParamsIns.del_param(
                "broadcast_axis_multiply_flag")
            cce_emitinsn_params.cceEmitParamsIns.insert_param(
                "broadcast_axis_multiply_flag", broadcast_axis_multiply_flag)

        return True

    def _is_mix_broadcast_not_last_axis(self, tensor):
        """
        Judge is special sence of the non-last axis broadcast

        Parameters:
        ----------
        None

        Returns
        -------
        True or False
        """
        def __get_broadcast_flag():
            if not tensor.op.input_tensors:
                return False, None

            broadcast_flag = False
            boracast_tensor = None
            for in_tensor in list(tensor.op.input_tensors):
                insn = self._reg_insn_map.get(in_tensor.op.tag)
                if insn == "unified_broadcast":
                    original_tensor = in_tensor.op.input_tensors[0]
                    original_shape = self._shape_to_list(original_tensor.shape)
                    if original_shape[-2] == 1:
                        broadcast_flag = True
                        boracast_tensor = in_tensor
                        break

            if not broadcast_flag:
                return False, None
            return True, boracast_tensor

        def __is_shape_len_6_case(original_shape, broadcast_shape):
            dype = self._broadcast_not_last_axis_tensors[0].dtype
            out_size = broadcast_shape[-1] * broadcast_shape[-2] * \
                DTYPE_WIDTH_MAP[dype] * 2
            if out_size < 16:
                return False
            if len(original_shape) != 6:
                return False
            if original_shape[-1] > 64:
                return False
            for i in range(0, len(broadcast_shape) - 1, 1):
                if broadcast_shape[i] == 1:
                    return False
                if (i != len(broadcast_shape) - 2) and original_shape[i] == 1:
                    return False
            return True

        def __is_shape_len_3_case(original_shape, broadcast_shape):
            if len(original_shape) != 3 or original_shape[-1] > 64:
                return False
            if broadcast_shape[0] == 1 or original_shape[0] == 1:
                return False
            if broadcast_shape[1] == 1:
                return False
            if broadcast_shape[-1] == 1 or original_shape[-1] == 1:
                return False
            return True

        def __is_real_mix_broadcast_not_last_axis(input_tensor):
            broadcast_tensor = input_tensor
            if list(broadcast_tensor.op.input_tensors):
                original_tensor = broadcast_tensor.op.input_tensors[0]
                original_shape = self._shape_to_list(original_tensor.shape)
                broadcast_shape = \
                    self._shape_to_list(self._broadcast_not_last_axis_tensors[0].shape)
                if len(broadcast_shape) == 6:
                    return __is_shape_len_6_case(original_shape, broadcast_shape)
                if len(broadcast_shape) == 3:
                    return __is_shape_len_3_case(original_shape, broadcast_shape)
                return False
            return False

        broadcast_flag, input_tensor = __get_broadcast_flag()
        if not broadcast_flag:
            return False

        if not __is_real_mix_broadcast_not_last_axis(input_tensor):
            return False

        return True

    def _is_non_32align_broadcast_not_last_axis(self, tensor):
        """
        Judge is special sence of the non-last axis broadcast

        Parameters:
        ----------
        None

        Returns
        -------
        True or False
        """
        def __get_broadcast_flag():
            if not tensor.op.input_tensors:
                return False, None

            broadcast_flag = False
            boracast_tensor = None
            for in_tensor in list(tensor.op.input_tensors):
                insn = self._reg_insn_map.get(in_tensor.op.tag)
                if insn == "unified_broadcast":
                    broadcast_flag = True
                    boracast_tensor = in_tensor
                    break

            if not broadcast_flag:
                return False, None
            return True, boracast_tensor

        def __is_shape_len_5_case(original_shape, broadcast_shape):
            if original_shape[-1] > 64:
                return False
            if broadcast_shape[0] == 1 or original_shape[0] != 1:
                return False
            if broadcast_shape[-2] == 1 or original_shape[-2] != 1:
                return False
            for i in range(1, len(broadcast_shape) - 2, 1):
                if broadcast_shape[i] == 1 or broadcast_shape[i] != \
                        original_shape[i]:
                    return False
            return True

        def __is_shape_len_3_case(original_shape, broadcast_shape):
            if broadcast_shape[-2] == 1 or original_shape[-2] != 1:
                return False
            for i in range(1, len(broadcast_shape) - 2, 1):
                if broadcast_shape[i] == 1 or broadcast_shape[i] != original_shape[i]:
                    return False
            return True

        def __is_real_non_32align_broadcast_not_last_axis(in_tensor):
            broadcast_tensor = in_tensor
            if list(broadcast_tensor.op.input_tensors):
                original_tensor = broadcast_tensor.op.input_tensors[0]
                original_shape = self._shape_to_list(original_tensor.shape)
                broadcast_shape = self._shape_to_list(broadcast_tensor.shape)
                if len(broadcast_shape) == 5:
                    return __is_shape_len_5_case(original_shape, broadcast_shape)
                if len(broadcast_shape) == 3:
                    return __is_shape_len_3_case(original_shape, broadcast_shape)
                return False
            return False

        broadcast_flag, in_tensor = __get_broadcast_flag()
        if not broadcast_flag:
            return False

        return __is_real_non_32align_broadcast_not_last_axis(in_tensor)

    def _is_32align_broadcast_not_last_axis(self, tensor):
        def __get_broadcast_flag():
            if not tensor.op.input_tensors:
                return False, None

            broadcast_flag = False
            boracast_tensor = None
            for in_tensor in list(tensor.op.input_tensors):
                insn = self._reg_insn_map.get(in_tensor.op.tag)
                if insn == "unified_broadcast":
                    broadcast_flag = True
                    boracast_tensor = in_tensor
                    break

            if not broadcast_flag:
                return False, None
            return True, boracast_tensor

        def __is_real_32align_broadcast_not_last_axis(in_tensor):
            broadcast_tensor = in_tensor
            if list(broadcast_tensor.op.input_tensors):
                original_tensor = broadcast_tensor.op.input_tensors[0]
                original_shape = self._shape_to_list(original_tensor.shape)
                broadcast_shape = self._shape_to_list(broadcast_tensor.shape)
                if len(original_shape) != 5 or original_shape[-1] > 64:
                    return False
                if original_shape[-2] != 1 or broadcast_shape[-2] != 1:
                    return False
                if original_shape[-3] != 1 or broadcast_shape[-3] == 1:
                    return False
                dtype = self._broadcast_not_last_axis_tensors[0].dtype
                last_size = original_shape[-1] * DTYPE_WIDTH_MAP[dtype] * 2
                if original_shape[-1] != broadcast_shape[-1] or last_size % 32 != 0:
                    return False
                for i in range(0, len(broadcast_shape) - 3, 1):
                    if original_shape[i] == broadcast_shape[i]:
                        continue
                    else:
                        return False
                return True
            return False

        broadcast_flag, in_tensor = __get_broadcast_flag()
        if not broadcast_flag:
            return False

        return __is_real_32align_broadcast_not_last_axis(in_tensor)

    # fix line too long
    def _get_max_broadcast_axis(self):

        max_broadcast_axis = \
            self._find_max_broadcast_axis_of_broadcast_not_last_axis_tensors()

        return max_broadcast_axis

    # Example: (a,b,c,1,e), (a,b,1,d,e), max_broadcast_axis is 3
    def _find_max_broadcast_axis_of_broadcast_not_last_axis_tensors(self):
        """
        Find the largest broadcast axis of the non-last axis broadcast

        Parameters:
        ----------
        None

        Returns
        -------
        the largest broadcast axis
        """
        max_broadcast_axis = 0
        for broadcast_tensor in self._broadcast_not_last_axis_tensors:
            if list(broadcast_tensor.op.input_tensors):
                original_tensor = broadcast_tensor.op.input_tensors[0]
                original_shape = self._shape_to_list(original_tensor.shape)
                broadcast_shape = self._shape_to_list(broadcast_tensor.shape)
                broadcast_axis = 0
                for i in range(len(original_shape) - 1, -1, -1):
                    if original_shape[i] == 1 and \
                            original_shape[i] != broadcast_shape[i]:
                        broadcast_axis = i
                        break
                if broadcast_axis > max_broadcast_axis:
                    max_broadcast_axis = broadcast_axis

        return max_broadcast_axis

    def _is_mix_broadcast_enable_32align_tiling(self):
        """
        Check is the normalzie scale can enable common align tiling
        """
        if self._op_type != OpSpecTypes.NORMALIZE_SCALE:
            return False

        shape = self._shape_to_list(self._last_output_tensor.shape)
        max_ub_count = self._max_ub_count

        if len(shape) != 4:
            return False

        count = 1
        for i in range(1, len(shape)):
            count = count * shape[i]

        if count >= max_ub_count:
            return True

        # c dimension does not do broadcast, don't split batch axis
        # like 16,1,1,1->16,8,11,16, 1,8,1,1->16,8,11,16
        for broadcast_tensor in self._broadcast_last_axis_tensors:
            if list(broadcast_tensor.op.input_tensors):
                original_tensor = broadcast_tensor.op.input_tensors[0]
                original_shape = self._shape_to_list(original_tensor.shape)
                broadcast_shape = self._shape_to_list(broadcast_tensor.shape)
                if original_shape[1] == broadcast_shape[1]:
                    return False
        return True

    def _is_32align_of_mix_broadcast_last_axis(self, shape, dtype):
        """check is 32 byte align in mix broadcast scene"""
        max_axis = self._find_max_non_broadcast_axis_of_broadcast_last_tensors()
        count = 1
        if max_axis != len(shape):
            for i in range(max_axis + 1, len(shape)):
                count = count * shape[i]
        else:
            count = shape[-1]

        size = DTYPE_WIDTH_MAP[dtype] * 2 * count
        return size % 32 == 0

    def _find_max_non_broadcast_axis_of_broadcast_last_tensors(self):
        """
        Find the largest non broadcast axis of the axis broadcast

        Parameters:
        ----------
        None

        Returns
        -------
        the largest non broadcast axis
        """
        max_non_broadcast_axis = 0
        for broadcast_tensor in self._broadcast_last_axis_tensors:
            if list(broadcast_tensor.op.input_tensors):
                original_tensor = broadcast_tensor.op.input_tensors[0]
                original_shape = self._shape_to_list(original_tensor.shape)
                broadcast_shape = self._shape_to_list(broadcast_tensor.shape)
                non_broadcast_axis = 0
                for axis in range(len(original_shape) - 1, -1, -1):
                    if original_shape[axis] == broadcast_shape[axis]:
                        non_broadcast_axis = axis
                        break
                if non_broadcast_axis > max_non_broadcast_axis:
                    max_non_broadcast_axis = non_broadcast_axis

        return max_non_broadcast_axis

    def _is_all_one_before_max_broadcast_axis(self, ub_split_axis,
                                              max_broadcast_axis):
        """
        Judge is all the axis before max broadcast axis is 1

        Parameters:
        ----------
        max_broadcast_axis

        Return
        -------
        True or False
        """
        if not self._broadcast_not_last_axis_tensors:
            return False

        for broadcast_tensor in self._broadcast_not_last_axis_tensors:
            if list(broadcast_tensor.op.input_tensors):
                original_tensor = broadcast_tensor.op.input_tensors[0]
                original_shape = self._shape_to_list(original_tensor.shape)
                for i in range(ub_split_axis, max_broadcast_axis, 1):
                    if original_shape[i] != 1:
                        return False
            else:
                return False

        return True

    def _is_all_one_before_broadcast_axis(self, shape, broadcast_axis):
        """
        Judge is all the axis before broadcast axis is 1

        Parameters:
        ----------
        shape, broadcast_axis

        Return
        -------
        True or False
        """
        if broadcast_axis == 0:
            return False

        for i in range(0, broadcast_axis, 1):
            if shape[i] != 1:
                return False

        return True

    # Example: (a,b,c,1,e), (a,b,1,d,e), max_broadcast_axis is 3
    def _find_max_broadcast_axis_of_tensor(self, broadcast_tensor):
        """
        Find the largest broadcast axis of broadcast tensor

        Parameters:
        ----------
        None

        Returns
        -------
        the largest broadcast axis
        """
        max_broadcast_axis = 0

        if list(broadcast_tensor.op.input_tensors):
            original_tensor = broadcast_tensor.op.input_tensors[0]
            original_shape = self._shape_to_list(original_tensor.shape)
            broadcast_shape = self._shape_to_list(broadcast_tensor.shape)

            for i in range(len(original_shape) - 1, -1, -1):
                if original_shape[i] == 1 and \
                        original_shape[i] != broadcast_shape[i]:
                    max_broadcast_axis = i
                    return max_broadcast_axis

        return max_broadcast_axis

    def _is_only_scale_broadcast_of_last_axis_broadcast(self):
        """
        Judge is the only scale broadcast axis of last axis broadcast,
        like (1,1,1,1,1,1,1)->(10, 10, 5, 2, 3, 9,1)

        Parameters:
        ----------
        None

        Returns
        -------
        Bool: True or False
        """
        if not self._broadcast_last_axis_tensors:
            return False

        for broadcast_tensor in self._broadcast_last_axis_tensors:
            if list(broadcast_tensor.op.input_tensors):
                original_tensor = broadcast_tensor.op.input_tensors[0]
                original_shape = self._shape_to_list(original_tensor.shape)
                if sum(original_shape[:]) != len(original_shape):
                    return False
            else:
                return False

        return True

    def _is_mix_broadcast_of_last_axis_broadcast(self):
        """
        Judge is the mix broadcast axis of last axis broadcast,
        like (a,b,c,d,e), (a,b,1,d,1)

        Parameters:
        ----------
        None

        Returns
        -------
        Bool: True or False
        """
        for broadcast_tensor in self._broadcast_last_axis_tensors:
            if list(broadcast_tensor.op.input_tensors):
                original_tensor = broadcast_tensor.op.input_tensors[0]
                original_shape = self._shape_to_list(original_tensor.shape)
                broadcast_shape = self._shape_to_list(broadcast_tensor.shape)
                broadcast_axis_number = 0
                for i in range(len(original_shape) - 1, -1, -1):
                    if (original_shape[i] != broadcast_shape[i]) and \
                            (original_shape[i] == 1):
                        broadcast_axis_number = broadcast_axis_number + 1
                    if broadcast_axis_number >= 2:
                        return True
        return False

    def _is_continue_broadcast_of_last_axis_broadcast(self):
        """
        Judge is the continue broadcast axis of last axis broadcast,
        like (a,b,c,d,e), (a,b,c,1,1)

        Parameters:
        ----------
        None

        Returns
        -------
        Bool: True or False
        """
        for broadcast_tensor in self._broadcast_last_axis_tensors:
            if list(broadcast_tensor.op.input_tensors):
                original_tensor = broadcast_tensor.op.input_tensors[0]
                original_shape = self._shape_to_list(original_tensor.shape)
                broadcast_shape = self._shape_to_list(broadcast_tensor.shape)
                for i in range(0, len(original_shape), 1):
                    if (original_shape[i] != broadcast_shape[i]) and \
                            (original_shape[i] == 1):
                        self._continue_broadcast_last_axis.append(i)
                index_temp = self._continue_broadcast_last_axis[0]
                if len(self._continue_broadcast_last_axis) <= 1:
                    return False
                for i in range(1, len(self._continue_broadcast_last_axis), 1):
                    if self._continue_broadcast_last_axis[i] - index_temp == 1:
                        index_temp = self._continue_broadcast_last_axis[i]
                        continue
                    else:
                        return False
                return True
        return False


    # (a,b,c,d,e), (a,b,1,d,1), max_broadcast_axis is 3
    def _find_max_broadcast_axis_of_mix_broadcast(self):
        """
        Find the largest broadcast axis of the mix axis broadcast

        Parameters:
        ----------
        None

        Returns
        -------
        the largest broadcast axis
        """
        max_broadcast_axis = 0
        for broadcast_tensor in self._broadcast_last_axis_tensors:
            if list(broadcast_tensor.op.input_tensors):
                original_tensor = broadcast_tensor.op.input_tensors[0]
                original_shape = self._shape_to_list(original_tensor.shape)
                broadcast_shape = self._shape_to_list(broadcast_tensor.shape)
                broadcast_axis = 0
                for i in range(len(original_shape) - 1, -1, -1):
                    if (original_shape[i] == broadcast_shape[i]) and \
                            (original_shape[i] != 1):
                        broadcast_axis = i
                        break
                if broadcast_axis > max_broadcast_axis:
                    max_broadcast_axis = broadcast_axis
        return max_broadcast_axis

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
        max_broadcast_axis_offset = self. \
            _find_max_broadcast_last_axis_offset_from_tensor(tensor)

        return max_broadcast_axis_offset

    def _find_max_broadcast_last_axis_offset_from_tensor(self, tensor):
        max_broadcast_axis_offset = 0
        if list(tensor.op.input_tensors):
            original_tensor = tensor.op.input_tensors[0]
            original_shape = self._shape_to_list(original_tensor.shape)
            broadcast_shape = self._shape_to_list(tensor.shape)
            if len(original_shape) != len(broadcast_shape):
                difference = len(broadcast_shape) - len(original_shape)
                original_shape = difference * [1] + original_shape
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
            if broadcast_axis_offset > max_broadcast_axis_offset:
                max_broadcast_axis_offset = broadcast_axis_offset

        return max_broadcast_axis_offset

    def _is_32b_align_of_broadcast_not_last_axis_tensors(self, shape, dtype):
        """
        The non-last axis broadcast is 32B alignment or not

        Parameters:
        ----------
        shape :  tensor shape
        dtype :  tensor date type

        Returns
        -------
        Bool: True or False
        """
        if self._broadcast_not_last_axis_tensors:
            max_broadcast_axis = self. \
                _find_max_broadcast_axis_of_broadcast_not_last_axis_tensors()
            size = DTYPE_WIDTH_MAP[dtype] * 2
            for i in range(max_broadcast_axis + 1, len(shape), 1):
                size = size * shape[i]
            return size % 32 == 0

        return False

    def _is_32b_align_of_broadcast_last_axis_tensors(self, shape, dtype):
        """
        The last axis broadcast is 32B alignment or not

        Parameters:
        ----------
        shape :  tensor shape
        dtype :  tensor date type

        Returns
        -------
        Bool: True or False
        """
        if self._broadcast_last_axis_tensors:
            size = DTYPE_WIDTH_MAP[dtype] * 2 * shape[-1]
            return size % 32 == 0

        return False

    def _is_contain_broadcast_tensor(self):
        """
        Check the graph include broadcast tensor or not

        Parameters:
        ----------
        None

        Returns
        -------
        Bool: True or False
        """
        if self._broadcast_last_axis_tensors or \
                self._broadcast_not_last_axis_tensors:
            return True
        return False

    def _is_broadcast_last_axis(self):
        if self._broadcast_last_axis_tensors:
            return True
        return False

    def _is_only_broadcast_last_axis(self):
        """
        Check the graph is only include broadcast last axis tensor or not

        Parameters:
        ----------
        None

        Returns
        -------
        Bool: True or False
        """
        if self._broadcast_last_axis_tensors and not \
                self._broadcast_not_last_axis_tensors:
            return True
        return False

    def _is_only_broadcast_not_last_axis(self):
        """
        Check the graph is only include broadcast non-last axis tensor or not

        Parameters:
        ----------
        None

        Returns
        -------
        Bool: True or False
        """
        if self._broadcast_not_last_axis_tensors and not \
                self._broadcast_last_axis_tensors:
            return True
        return False

    def _is_broadcast_not_last_axis(self):
        if self._broadcast_not_last_axis_tensors:
            return True
        return False

    # (a_1,..,(a_ko,a_ki),...,(a_lo,a_li),...,a_n)
    def _need_double_buffer(self, shape, block_axis, block_tiling_inner_loop,
                            ub_axis, ub_tiling_inner_loop):
        """
        Check if tensor needs to enable double buffer

        Parameters:
        ----------
        shape :  tensor shape

        block_axis : Multicore split axis

        block_tiling_inner_loop : Multicore split axis factor

        ub_axis : UB split axis

        ub_tiling_inner_loop :  UB split axis factor

        Returns
        -------
        Bool: True or False
        """
        if ub_axis < block_axis or ub_axis < 0 or block_axis < 0:
            return False
        if ub_axis == block_axis:
            one_core_loop_number = block_tiling_inner_loop
        else:
            ub_tiling_outer_loop = shape[ub_axis] // ub_tiling_inner_loop
            one_core_loop_number = block_tiling_inner_loop * \
                                   ub_tiling_outer_loop

        for i in range(block_axis + 1, ub_axis, 1):
            one_core_loop_number = one_core_loop_number * shape[i]

        return one_core_loop_number > 1

    def _get_total_width(self):
        """
        caculate the max useable number based on op liveness
        return: max useable number
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
        return max_width

    def _get_max_ub_count(self):
        """
        caculate the max element num loaded in UB buffer
        :return: max element num loaded in UB buffer
        """
        def __update_total_width(total_width):
            if not self._mem_unique_enable:
                return total_width
            dtype = self._last_output_tensor.dtype.lower()
            update_width = 0.5
            if dtype == "float32":
                update_width = 1
            return total_width + update_width

        # div 2 for align to fp16
        self._total_size = cceconf.get_soc_spec("UB_SIZE") // 2
        self._total_size = self._total_size // 2  # div 2 for double buffer
        if self._op_type == OpSpecTypes.RELU_GRAD_V2:
            dtype = self._input_tensors[1].dtype.lower()
            if dtype == "float16":
                total_width = 3
            else:
                total_width = 8
        else:
            total_width = self._get_total_width()
            total_width = __update_total_width(total_width)

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
            self.apply_broadcast(op_node, tmp_op, tensor)
        elif tmp_op["op"].find("reduce") != -1:
            if self._have_reduce and not hasattr(self,
                                                 util.REDUCE_MULTI_PRIME_KEY):
                raise RuntimeError("Only support one time reduce")
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

    def apply_broadcast(self, op_node, tmp_op, tensor):
        """Apply broadcast for __split_tensor"""
        if tmp_op["op"] == "broadcast_for_tensor":
            # broadcast not last axis
            if self._shape_to_list(tmp_op["src_buffer"][0].shape)[-1] != 1 and \
                    self._op_subpattern != OpSubPatterns.CMPSEL_PATTERN and \
                    self.is_exceeded_mid_broadcast_threshold(tensor):
                tmp_op["effective_op"] = False
        else:
            tmp_op["args"] = [op_node.body[0]]
