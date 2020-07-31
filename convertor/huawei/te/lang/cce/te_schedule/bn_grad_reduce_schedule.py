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

batchnorm grad reduce schedule
"""
import copy
from te import platform as cceconf
from te import tvm
from te.platform import cce_emitinsn_params
from . import util
from .elewise_schedule_new import ElewiseSchedule
from .bn_update_grad_shedule import _get_factors_of_positive_integer
from .bn_update_grad_shedule import _find_closest_factor
from .util import get_nearest_factor
from .util import DTYPE_WIDTH_MAP

SPECIAL_BROADCAST_INSN_MAP = {"vector_mul": "vector_mul_with_broadcast",
                              "vector_div": "vector_div_with_broadcast",
                              "vector_add": "vector_add_with_broadcast",
                              "vector_sub": "vector_sub_with_broadcast",
                              }

RESNET_50_SHAPE_LIST = [
    [32, 64 // 16, 112, 112, 16],
    [32, 256 // 16, 56, 56, 16],
    [32, 512 // 16, 28, 28, 16],
    [32, 128 // 16, 56, 56, 16],
    [32, 64 // 16, 56, 56, 16],
    [32, 256 // 16, 28, 28, 16],
    [32, 1024 // 16, 14, 14, 16],
    [32, 1024 // 16, 7, 7, 16],
    [32, 2048 // 16, 7, 7, 16],
    [32, 128 // 16, 28, 28, 16],
    [32, 512 // 16, 14, 14, 16],
    [32, 256 // 16, 14, 14, 16],
    [32, 512 // 16, 7, 7, 16],

    [32, 1, 224, 224, 16],
    [32, 4, 57, 57, 16],
    [32, 4, 112, 112, 16],
    [32, 8, 29, 29, 16],
    [32, 8, 57, 57, 16],
    [32, 16, 15, 15, 16],
    [32, 16, 29, 29, 16],
    [32, 16, 57, 57, 16],
    [32, 32, 15, 15, 16],
    [32, 32, 29, 29, 16],
    [32, 32, 8, 8, 16],
    [32, 64, 15, 15, 16],
]

CDIM_OPTIMIZE_THRESHOLD = 2048

# 2048 * 256 * 256
CHWDIM_OPTIMIZE_THRESHOLD = 134217728

# pylint: disable=too-many-return-statements,too-few-public-methods,too-many-arguments,too-many-statements,no-self-use,too-many-lines,too-many-instance-attributes,too-many-branches,
class BnGradReduceSchedule(ElewiseSchedule):
    """
    class of cce batchnorm grad reduce schedule

    Parameters
    ----------
    ElewiseSchedule: class of elewise schedule

    Returns
    -------
    BnGradReduce instance : instance of BnGradReduceSchedule
    """

    def __init__(self, need_multi_core=True):
        ElewiseSchedule.__init__(self, need_multi_core)
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
                                        "elewise_binary_add"]
        # default at least data processed by each core, unit is byte
        # this value is dynamically adjusted based on the type of the operator
        self._multi_core_threshold = 1024
        self._spec_node_list = []
        self._is_last_axis_broadcast = False
        self._total_size = 0
        self._is_muti_output = False
        self._have_reduce = False
        self._special_non_last_broadcast_scene = False
        self._resnet50_shape_pattern = False
        self._optimize_shape_pattern = False
        self._broadcast_enhance_insn_map = \
            {"vector_mul": "vector_mul_with_broadcast_enhance",
             "vector_div": "vector_div_with_broadcast_enhance",
             "vector_add": "vector_add_with_broadcast_enhance",
             "vector_sub": "vector_sub_with_broadcast_enhance",
             "vector_min": "vector_min_with_broadcast_enhance"
            }

        # For (x <= 32, 100, 1, 4) to (x <= 32, 100, 2, 4) broadcasting
        self._less_32_core_middle_broadcast_scene = False
        # For zeros_like special output
        self._elewise_binary_phony_as_output = False

    def _is_optimize_network_shape(self, shape):
        """
        Judge if the shape need optimize operation

        Parameters:
        ----------
        shape :  output tensor shape

        Returns
        -------
        """

        if shape in RESNET_50_SHAPE_LIST:
            self._resnet50_shape_pattern = True

        # bn_grad_reduce only support 5HD format currently, NC1HWC0
        length = len(shape)
        if length == 5 and shape[length - 1] == 16:
            c_dim = shape[1]*shape[4]
            chw_dim = c_dim*shape[2]*shape[3]

            if c_dim <= CDIM_OPTIMIZE_THRESHOLD and \
                    chw_dim <= CHWDIM_OPTIMIZE_THRESHOLD:
                self._optimize_shape_pattern = True

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

            # only two large input do double buffer, ub buffer do not reused
            if not self._is_factor16_broadcast_input_tensor(i):
                self._double_buffer_tensors.append(read_buffer)
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

        # output ub buffer do not reused
        output = self._last_output_tensor
        write_buffer = self._cache_write_tensors_and_buffer_map[output]
        self._schedule[write_buffer].mem_unique()

    # pylint: disable=too-many-locals
    def _get_ub_tiling(self, shape, block_tiling_axis,
                       block_tiling_inner_loop,
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
                    split_size = get_nearest_factor(shape[split_axis],
                                                    split_size)
                    break
        else:
            split_size = block_tiling_inner_loop

        if split_axis == block_tiling_axis and \
                split_size > block_tiling_inner_loop:
            split_size = block_tiling_inner_loop

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

    def _is_special_optimize_pattern(self):
        """
        check is special optimize scene

        Parameters:
        ----------
        None

        Returns
        -------
        True or False
        """
        if self._resnet50_shape_pattern or self._optimize_shape_pattern:
            return True

        return False

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
        res = self._last_output_tensor
        ub_tiling_result = self._tiling_result["ub_tiling"]
        ub_split_axis = ub_tiling_result["axis"]
        res_ub_inner = ub_tiling_result["inner_itervar"]

        self._get_emit_insn_map()
        self._get_reg_emit_insn_map()

        for i in self._cache_read_tensors_and_buffer_map:
            read_buffer = self._cache_read_tensors_and_buffer_map[i]
            if self._is_special_optimize_pattern() and \
                    self._is_factor16_broadcast_input_tensor(i):
                para = {"scope": read_buffer.op.axis[1],
                        "instruction": 'dma_copy'}
            else:
                para = {"scope": read_buffer.op.axis[ub_split_axis],
                        "instruction": 'dma_copy'}
            self._emit_insn_map[read_buffer] = para

        for i in self._cache_write_tensors_and_buffer_map:
            write_buffer = self._cache_write_tensors_and_buffer_map[i]
            insn = self._calculate_emit_insn_map(write_buffer)

            if insn == "unified_broadcast":
                if not self._is_broadcast_last_axis_tensor(i):
                    pass
                else:
                    max_last_broadcast_axis_offset = \
                        self._find_max_broadcast_last_axis_offset(i)
                    cce_emitinsn_params.cceEmitParamsIns.del_param(
                        "broadcast_axis_offset")
                    cce_emitinsn_params.cceEmitParamsIns.insert_param(
                        "broadcast_axis_offset", max_last_broadcast_axis_offset)

            # special process for sub (1,1,1,32) (32,224,224,3)
            if self._special_non_last_broadcast_scene:
                if insn in self._broadcast_enhance_insn_map.keys():
                    if self._is_special_tensor_of_broadcast_not_last_axis(i):
                        insn = self._broadcast_enhance_insn_map.get(insn)
            elif self._special_non_last_broadcast_factor16_scene:
                if insn in SPECIAL_BROADCAST_INSN_MAP.keys():
                    if self._is_special_factor16_broadcast_tensor(i):
                        insn = SPECIAL_BROADCAST_INSN_MAP.get(insn)

            if insn == "vector_multiple":
                self._do_emit_multiple_insn(i, write_buffer)
                continue

            if self._is_special_optimize_pattern() and \
                    self._is_factor16_broadcast_input_tensor(i):
                para = {"scope": write_buffer.op.axis[1],
                        "instruction": insn}
            else:
                para = {"scope": write_buffer.op.axis[ub_split_axis],
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

        self._is_need_update_compute_at_axis = True

    def _is_resnet50_factor16_tensor(self, tensor):
        """
        check is resnet50 factor16 tensor

        Parameters:
        ----------
        None

        Returns
        -------
        True or False
        """
        if self._resnet50_shape_pattern and \
                self._is_factor16_broadcast_input_tensor(tensor):
            return True

        return False

    def _is_optimize_factor16_tensor(self, tensor):
        """
        check is optimize factor16 tensor

        Parameters:
        ----------
        None

        Returns
        -------
        True or False
        """
        if self._optimize_shape_pattern and \
                self._is_factor16_broadcast_input_tensor(tensor):
            return True

        return False

    def _calculate_optimize_compute_at_axis(self, res_block_outer):
        """
        calculate optimize factor16 tensor compute at axis

        Parameters:
        ----------
        res_block_outer: block split tensor

        Returns
        -------
        compute at axis
        """
        if self._multi_core_fused_axis is not None:
            compute_at_outer = self._multi_core_fused_axis
        else:
            compute_at_outer = res_block_outer

        return compute_at_outer

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
        res = self._last_output_tensor
        ub_tiling_result = self._tiling_result["ub_tiling"]
        res_ub_outer = ub_tiling_result["outer_itervar"]
        ub_split_axis = ub_tiling_result["axis"]

        block_tiling_result = self._tiling_result["block_tiling"]
        block_split_axis = block_tiling_result["axis"]
        res_block_inner = block_tiling_result["inner_itervar"]
        res_block_outer = block_tiling_result["outer_itervar"]

        # pylint: disable=too-many-nested-blocks
        for i in self._cache_read_tensors_and_buffer_map:
            if self._is_need_update_compute_at_axis:
                readers_tensor = self._cache_read_tensors_and_readers_map[i]
                read_buffer = self._cache_read_tensors_and_buffer_map[i]

                if self._is_broadcast_not_last_axis_tensor(readers_tensor[0]):
                    # like (8, 128, 1, 2, 1, 1) (1, 1, 1, 2, 3, 1) scene,
                    # both tensor need to do broadcast, but not all tensors can
                    # move the compute at axis to outward
                    if self._is_resnet50_factor16_tensor(i):
                        para = {"parent": self._schedule[res],
                                "scope": res_block_outer}
                    elif self._is_optimize_factor16_tensor(i):
                        compute_at_outer = \
                            self._calculate_optimize_compute_at_axis(
                                res_block_outer)
                        para = {"parent": self._schedule[res],
                                "scope": compute_at_outer}
                    else:
                        max_broadcast_axis = \
                            self._find_max_broadcast_axis_of_tensor(
                                readers_tensor[0])
                        if max_broadcast_axis < ub_split_axis:
                            compute_at_outer = res_ub_outer
                        else:
                            if block_split_axis == ub_split_axis:
                                compute_at_outer = res_ub_outer
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
                                        self._schedule[res].op.axis[ub_split_axis-1]
                        para = {"parent": self._schedule[res],
                                "scope": compute_at_outer}
                elif self._special_non_last_broadcast_factor16_scene:
                    if self._is_resnet50_factor16_tensor(i):
                        para = {"parent": self._schedule[res],
                                "scope": res_block_outer}
                    elif self._is_optimize_factor16_tensor(i):
                        compute_at_outer = \
                            self._calculate_optimize_compute_at_axis(
                                res_block_outer)
                        para = {"parent": self._schedule[res],
                                "scope": compute_at_outer}
                    elif self._is_factor16_broadcast_input_tensor(i):
                        if ub_split_axis in (0, 1):
                            para = {"parent": self._schedule[res],
                                    "scope": res_ub_outer}
                        else:
                            if block_split_axis == ub_split_axis:
                                compute_at_outer = res_ub_outer
                            elif block_split_axis in (1, 2, 3):
                                compute_at_outer = res_block_inner
                            else:
                                compute_at_outer = \
                                    self._schedule[res].op.axis[1]
                            para = {"parent": self._schedule[res],
                                    "scope": compute_at_outer}
                    else:
                        para = {"parent": self._schedule[res],
                                "scope": res_ub_outer}
                else:
                    para = {"parent": self._schedule[res],
                            "scope": res_ub_outer}

                self._compute_at_map[read_buffer] = para
            else:
                read_buffer = self._cache_read_tensors_and_buffer_map[i]
                para = {"parent": self._schedule[res], "scope": res_ub_outer}
                self._compute_at_map[read_buffer] = para

        for i in self._cache_write_tensors_and_buffer_map:
            write_buffer = self._cache_write_tensors_and_buffer_map[i]
            if i != self._last_output_tensor:
                if self._special_non_last_broadcast_factor16_scene:
                    if self._is_resnet50_factor16_tensor(i):
                        compute_at_outer = res_block_outer
                        para = {"parent": self._schedule[res],
                                "scope": compute_at_outer}
                    elif self._is_optimize_factor16_tensor(i):
                        compute_at_outer = \
                            self._calculate_optimize_compute_at_axis(
                                res_block_outer)
                        para = {"parent": self._schedule[res],
                                "scope": compute_at_outer}
                    elif self._is_factor16_broadcast_input_tensor(i):
                        if ub_split_axis in (0, 1):
                            para = {"parent": self._schedule[res],
                                    "scope": res_ub_outer}
                        else:
                            if block_split_axis == ub_split_axis:
                                compute_at_outer = res_ub_outer
                            elif block_split_axis in (1, 2, 3):
                                compute_at_outer = res_block_inner
                            else:
                                compute_at_outer = \
                                    self._schedule[res].op.axis[1]
                            para = {"parent": self._schedule[res],
                                    "scope": compute_at_outer}
                    else:
                        para = {"parent": self._schedule[res],
                                "scope": res_ub_outer}
            else:
                para = {"parent": self._schedule[res], "scope": res_ub_outer}

            self._compute_at_map[write_buffer] = para

        for i in self._mid_output_tensors:
            para = {"parent": self._schedule[res], "scope": res_ub_outer}
            self._compute_at_map[i] = para

    def _calculate_double_buffer(self):
        """
        double buffer operations
        read_buffer : the all read_cache for input in ub, type is list
        """
        self._need_db = False

    def _get_max_ub_count(self):
        """
        caculate the max element num loaded in UB buffer
        :return: max element num loaded in UB buffer
        """
        dtype = self._input_tensors[0].dtype.lower()
        # div 2 for align to fp16
        total_size = cceconf.get_soc_spec("UB_SIZE") // 2
        dtype_size = DTYPE_WIDTH_MAP.get(dtype)
        shape = util.shape_to_list(self._last_output_tensor.shape)
        total_size = total_size // dtype_size
        if not self._resnet50_shape_pattern:
            if dtype == "float16":
                total_width = 13
            else:
                total_width = 9
        else:
            if dtype == "float16":
                total_width = 8.5
                if shape in [[32, 4, 57, 57, 16], [32, 8, 57, 57, 16],
                             [32, 16, 57, 57, 16]]:
                    total_width = 10
            else:
                total_width = 5.5
                if shape in [[32, 4, 57, 57, 16], [32, 8, 57, 57, 16],
                             [32, 16, 57, 57, 16], [32, 32, 29, 29, 16]]:
                    total_width = 7.5

        align_to = 128

        max_bound = total_width * align_to
        max_ub_count = int(total_size / max_bound * align_to)

        return max_ub_count

    @staticmethod
    def _get_split_factor_of_small_c1(dtype):
        if dtype == "float16":
            batch_split_factor = 4
            c1_split_factor = 4
        else:
            batch_split_factor = 2
            c1_split_factor = 8
        return batch_split_factor, c1_split_factor

    def _do_schedule_for_model_parallel_small_c1(self, out_tensor, sch_list):
        """
        do schedule for model parallel of resnet50 case
        :return:
        """
        shape_out = util.shape_to_list(out_tensor.shape)
        dtype = out_tensor.dtype

        self._schedule = sch_list[0]

        self._calculate_cache_read()
        self._do_cache_read()

        self._calculate_cache_write()
        self._do_cache_write()

        self._calculate_compute_inline()
        self._do_compute_inline()

        batch_split_factor, c1_split_factor = \
            self._get_split_factor_of_small_c1(dtype)

        final_out_tensor = self._last_output_tensor
        out_batch_outer, out_batch_inner =\
            self._schedule[final_out_tensor].split(final_out_tensor.op.axis[0],
                                                   factor=batch_split_factor)

        out_c1_outer, out_c1_inner =\
            self._schedule[final_out_tensor].split(final_out_tensor.op.axis[1],
                                                   factor=c1_split_factor)

        self._schedule[final_out_tensor].reorder(
            out_batch_outer,
            out_c1_outer,
            out_c1_inner,
            out_batch_inner,
            final_out_tensor.op.axis[2],
            final_out_tensor.op.axis[3],
            final_out_tensor.op.axis[4]
        )

        out_fused_axis = self._schedule[final_out_tensor].fuse(
            out_batch_outer, out_c1_outer,)

        # _calculate_compute_at
        # pylint: disable=too-many-nested-blocks
        for tensor in self._cache_read_tensors_and_buffer_map:
            tensor_shape = util.shape_to_list(tensor.shape)
            read_buffer = self._cache_read_tensors_and_buffer_map[tensor]

            if tensor_shape == shape_out:
                para = {"parent": self._schedule[final_out_tensor],
                        "scope": out_c1_inner}

                self._compute_at_map[read_buffer] = para
            else:
                para = {"parent": self._schedule[final_out_tensor],
                        "scope": out_fused_axis}
                self._compute_at_map[read_buffer] = para

        for tensor in self._cache_write_tensors_and_buffer_map:
            tensor_shape = util.shape_to_list(tensor.shape)
            write_buffer = self._cache_write_tensors_and_buffer_map[tensor]
            if tensor_shape == shape_out:
                para = {"parent": self._schedule[final_out_tensor],
                        "scope": out_c1_inner}

                self._compute_at_map[write_buffer] = para
            else:
                para = {"parent": self._schedule[final_out_tensor],
                        "scope": out_fused_axis}
                self._compute_at_map[write_buffer] = para

        self._do_compute_at()

        block = tvm.thread_axis("blockIdx.x")
        self._schedule[final_out_tensor].bind(out_fused_axis, block)

        # _calculate_emit_insn
        ub_split_axis = 0
        for i in self._cache_read_tensors_and_buffer_map:
            read_buffer = self._cache_read_tensors_and_buffer_map[i]
            para = {"scope": read_buffer.op.axis[ub_split_axis],
                    "instruction": 'dma_copy'}
            self._emit_insn_map[read_buffer] = para

        self._get_emit_insn_map()
        self._get_reg_emit_insn_map()

        for tensor in self._cache_write_tensors_and_buffer_map:
            write_buffer = self._cache_write_tensors_and_buffer_map[tensor]
            insn = self._calculate_emit_insn_map(write_buffer)

            if insn == "unified_broadcast":
                if not self._is_broadcast_last_axis_tensor(tensor):
                    continue
                else:
                    max_last_broadcast_axis_offset = \
                        self._find_max_broadcast_last_axis_offset(tensor)
                    cce_emitinsn_params.cceEmitParamsIns.del_param(
                        "broadcast_axis_offset")
                    cce_emitinsn_params.cceEmitParamsIns.insert_param(
                        "broadcast_axis_offset", max_last_broadcast_axis_offset)


            if insn in SPECIAL_BROADCAST_INSN_MAP.keys():
                if self._is_special_factor16_broadcast_tensor(tensor):
                    insn = SPECIAL_BROADCAST_INSN_MAP.get(insn)

            para = {"scope": write_buffer.op.axis[ub_split_axis],
                    "instruction": insn}
            self._emit_insn_map[write_buffer] = para

        self._emit_insn_map[final_out_tensor] = \
            {"scope": out_batch_inner, "instruction": 'dma_copy'}

        self._do_emit_insn()

        return True

    def _do_schedule_for_model_parallel(self, out_tensor, sch_list):
        """
        do schedule for model parallel of resnet50 case
        :return:
        """
        core_num = cceconf.get_soc_spec("CORE_NUM")
        shape_out = util.shape_to_list(out_tensor.shape)

        if shape_out in ([32, 16, 14, 14, 16], [32, 16, 15, 15, 16]):
            return self._do_schedule_for_model_parallel_small_c1(
                out_tensor, sch_list)

        n_size = shape_out[0]
        c1_size = shape_out[1]
        h_size = shape_out[2]
        w_size = shape_out[3]
        c0_size = shape_out[4]
        max_ub_count = self._get_max_ub_count()

        # do tiling
        ub_split_axis = 0
        ub_split_inner = 1
        if c1_size >= core_num and c1_size % core_num == 0:
            n_inner = n_size
        else:
            n_inner = n_size // core_num

        for i in range(n_inner, 0, -1):
            if n_inner % i != 0:
                continue
            if h_size*w_size*c0_size*i > max_ub_count:
                continue

            ub_split_inner = i
            break

        split_factor = ub_split_inner

        self._schedule = sch_list[0]

        self._calculate_cache_read()
        self._do_cache_read()

        self._calculate_cache_write()
        self._do_cache_write()

        self._calculate_compute_inline()
        self._do_compute_inline()


        inner_loop = shape_out[ub_split_axis]

        factors = _get_factors_of_positive_integer(inner_loop)
        split_factor = _find_closest_factor(factors, split_factor)

        final_out_tensor = self._last_output_tensor
        out_block_outer, out_block_inner =\
            self._schedule[final_out_tensor].split(final_out_tensor.op.axis[1],
                                                   nparts=core_num)

        sum_x_ub_outer, sum_x_ub_inner = \
            self._schedule[final_out_tensor].split(
                final_out_tensor.op.axis[0],
                factor=split_factor)

        self._schedule[final_out_tensor].reorder(
            out_block_outer,
            out_block_inner,
            sum_x_ub_outer,
            sum_x_ub_inner,
            final_out_tensor.op.axis[2],
            final_out_tensor.op.axis[3],
            final_out_tensor.op.axis[4]
        )

        # _calculate_compute_at
        # pylint: disable=too-many-nested-blocks
        for tensor in self._cache_read_tensors_and_buffer_map:
            tensor_shape = util.shape_to_list(tensor.shape)
            read_buffer = self._cache_read_tensors_and_buffer_map[tensor]

            if tensor_shape == shape_out:
                para = {"parent": self._schedule[final_out_tensor],
                        "scope": sum_x_ub_outer}

                self._compute_at_map[read_buffer] = para
            else:
                para = {"parent": self._schedule[final_out_tensor],
                        "scope": out_block_outer}
                self._compute_at_map[read_buffer] = para

        for tensor in self._cache_write_tensors_and_buffer_map:
            tensor_shape = util.shape_to_list(tensor.shape)
            write_buffer = self._cache_write_tensors_and_buffer_map[tensor]
            if tensor_shape == shape_out:
                para = {"parent": self._schedule[final_out_tensor],
                        "scope": sum_x_ub_outer}

                self._compute_at_map[write_buffer] = para
            else:
                para = {"parent": self._schedule[final_out_tensor],
                        "scope": out_block_outer}
                self._compute_at_map[write_buffer] = para

        self._do_compute_at()

        block = tvm.thread_axis("blockIdx.x")
        self._schedule[final_out_tensor].bind(out_block_outer, block)

        # _calculate_emit_insn
        for i in self._cache_read_tensors_and_buffer_map:
            read_buffer = self._cache_read_tensors_and_buffer_map[i]
            para = {"scope": read_buffer.op.axis[ub_split_axis],
                    "instruction": 'dma_copy'}
            self._emit_insn_map[read_buffer] = para

        self._get_emit_insn_map()
        self._get_reg_emit_insn_map()

        for tensor in self._cache_write_tensors_and_buffer_map:
            write_buffer = self._cache_write_tensors_and_buffer_map[tensor]
            insn = self._calculate_emit_insn_map(write_buffer)

            if insn == "unified_broadcast":
                if not self._is_broadcast_last_axis_tensor(tensor):
                    pass
                else:
                    max_last_broadcast_axis_offset = \
                        self._find_max_broadcast_last_axis_offset(tensor)
                    cce_emitinsn_params.cceEmitParamsIns.del_param(
                        "broadcast_axis_offset")
                    cce_emitinsn_params.cceEmitParamsIns.insert_param(
                        "broadcast_axis_offset", max_last_broadcast_axis_offset)


            if insn in SPECIAL_BROADCAST_INSN_MAP.keys():
                if self._is_special_factor16_broadcast_tensor(tensor):
                    insn = SPECIAL_BROADCAST_INSN_MAP.get(insn)

            para = {"scope": write_buffer.op.axis[ub_split_axis],
                    "instruction": insn}
            self._emit_insn_map[write_buffer] = para

        self._emit_insn_map[final_out_tensor] = \
            {"scope": sum_x_ub_inner, "instruction": 'dma_copy'}

        self._do_emit_insn()

        return True

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
        core_num = cceconf.get_soc_spec("CORE_NUM")
        self._out_tensors = copy.copy(out_tensors)
        out_tensors_bak = copy.copy(out_tensors)
        out_tensor = out_tensors[0]
        is_success = self._construct_compute_graph(out_tensors, [])
        if not is_success:
            return False

        shape_out = util.shape_to_list(out_tensor.shape)
        c1_size = shape_out[1]
        h_size = shape_out[2]
        w_size = shape_out[3]
        c0_size = shape_out[4]
        max_ub_count = self._get_max_ub_count()
        if max_ub_count // (h_size * w_size * c0_size) >= 2 \
                and (c1_size >= core_num and c1_size % core_num == 0)\
                or shape_out in ([32, 16, 14, 14, 16], [32, 16, 15, 15, 16]):
            # ub utilization ratio is small, so use "model parallel"
            # c1_size axis as block_axis and n_size axis as ub split axis
            # can raise dma copy data size and dichotomy efficiency
            if self._do_schedule_for_model_parallel(out_tensor, sch_list):
                return True

        return super().do_schedule(out_tensors_bak, sch_list, spec_node_list)
