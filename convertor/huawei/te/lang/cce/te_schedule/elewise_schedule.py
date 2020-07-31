#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# pylint: disable=too-many-lines
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

elewise schedule
"""
# pylint: disable=unused-import
from functools import reduce as reduceIns
from math import ceil
import math

from te.platform import cce_conf
from te.platform import intrinsic_check_support
from te import platform as cceconf
from te import tvm
# pylint: disable=unused-import
from te.platform.cce_build import build_config
from te.platform import cce_emitinsn_params
import te.platform.cce_params as cce_params
from . import util

# pylint: disable=too-many-return-statements,too-few-public-methods,too-many-arguments
# pylint: too-many-statements,no-self-use,too-many-lines,too-many-instance-attributes
# pylint: too-many-branches, no-member, consider-using-enumerate

# the bit of dtype/16 map
DTYPE_WIDTH_MAP = {"float16": 1,
                   "float32": 2,
                   "int32": 2,
                   "int16": 1,
                   "uint16": 1,
                   "int8": 0.5,
                   "uint8": 0.5,
                   "bool": 0.5}
# dsl input type
DSL_REDUCE_TYPE_MAP = {"single_reduce_sum_float32": 0,
                       "cast_single_reduce_sum": 1,
                       "single_reduce_sum_float32_4d": 2,
                       "cast_single_reduce_sum_4d": 3,
                       "single_reduce_sum_float32_2d": 4,
                       "cast_single_reduce_sum_2d": 5,
                       "single_reduce_mean_float32": 6,
                       "cast_single_reduce_mean": 7,
                       "single_reduce_mean_float32_4d": 8,
                       "cast_single_reduce_mean_4d": 9,
                       "single_reduce_mean_float32_2d": 10,
                       "cast_single_reduce_mean_2d": 11}
ELEMENTS_VECTOR_OP_FP16 = cce_params.ELEMENTS_VECTOR_OP_FP16

# Multi-core splitting, so that the data processed into each core is greater
# than or equal to this threshold, too-many-public-methods
ENABLE_MULTI_CORE_THRESHOLD = 1 * 1024  # Bytes

# For non-divisible multi-core splitting, block dim is at most a multiple of core num
BLOCK_DIM_MULTIPLE = 4  # Bytes


# pylint: disable=too-many-instance-attributes, too-many-public-methods
class CceOp:
    """
    Base class of cce API

    Parameters
    ----------
    scope : cal buffer like local.UB

    need_tensorize : if need to doing tensorize when using calculate

    need_pragma : if need to doing pragma when using calculate

    need_double_buffer : if need to do double buffer,
    only support double buffer for the buffer in inner of for loop
    Returns
    -------
    CceOp_instance : instance of CceOp

    example:
        def sqrt_cce(data_shape, need_build):
            inp_dtype = "float16"
            power_num = tvm.const(0.5)
            data = tvm.placeholder(data_shape, name="data", dtype=inp_dtype)
            cce = CceOp(cceconf.scope_ubuf, need_pragma = True,
            need_tensorize = True)

            res = cce.vexp(data)

            cce.calculate(res, print_ir = True, need_build = need_build)

        sqrt_cce((32768, 32768), need_build = True)

    """
    BLOCK_WIDTH = 16
    BLOCK_LEN = 16
    REDUCE_BLOCK = 128
    MAX_BLOCK_DIM = 65535

    # pylint: disable=too-many-statements
    def __init__(self, scope, need_tensorize=True, need_pragma=True,
                 need_double_buffer=True, need_enable_muticore=True):
        # judge if need to tensorize in schedule
        self._need_tensorize = need_tensorize
        # judge if need to pragma in schedule
        self._need_pragma = need_pragma
        # judge if need double buffer
        self._need_double_buffer = need_double_buffer
        # judge if need enable muti core
        self._need_enable_muticore = need_enable_muticore
        # local scope in vector oprations, always be Unified buffer
        self._scope = scope
        # recorde relate buffers in cache_read
        self._read_cache_map = {}
        # recorde cache_read buffers
        self._read_cache = []
        # recorde cache_read for muti res, res:res_cache
        self._read_cache_muti_out = {}
        # recorde cache_write buffers
        self._write_cache = []
        # recorde buffers to compute_inline
        self._compute_inline = []
        # recorde buffers pass to lower or build
        self._origin_tensor = []
        # init tvm.schedule value
        self._schedule = None
        # if has reduce op
        self._have_reduce = False
        # record ops
        self._op = []
        # record origin op
        self._origin_op = []
        # dst_tensor_map
        self._dst_tensor_map = {}
        # the tensor:write_buffer map
        self._write_buffer_map = {}
        # record double buffer map[read] = write
        self._double_buffer_map = {}
        # record cache buffer eg. map[gm] = ub
        self._cache_buffer_map = {}

        self._compute_at_after_reduce_axis = None
        self._compute_at_after_reduce_buffer = None
        self._compute_at_before_reduce_axis = None
        self._compute_at_before_reduce_buffer = None
        self._need_compute_at_before = True
        self._need_compute_at_after = False
        self._read_dma_axis = None
        self._emit_before_reduce_axis = 0
        self._emit_after_reduce_axis = 0
        self._multi_core_bind_axis = None
        self._multi_core_buffer = None
        self._res_tensor = None
        self._res_tensor_list = []
        self._last_num = None
        self._split_axis = None
        self._res_dma_axis = None
        self._res_tensorize_axis = None
        self._shape_before_reduce = []
        self._is_last_reduce = None
        # if self._is_keepdims is True,the dim of res after reduce is the same
        # as the input,the reduce axis will be 1
        self._is_keepdims = False
        # the reduce axis number
        self._reduce_axis_num = []
        # reduce continuous tailing axis
        self._is_continuous_tailing_axis = False
        # extend API for further online supporting
        self._vars = []
        self._reduce_index = []
        self._operaters = {}
        self._res_tensor_axis_len = None
        self._write_buffer = None
        self._read_buffer = None
        self._is_muti_output = False
        self._reuse_buffer_index = 1
        self._is_elewise_single_and_broadcast = False
        self._is_last_axis_boradcast = False
        self._last_boradcast_axis = 0
        # the number of core
        self._spec_node_list = []
        # record the ub_tiling_axis
        self._ub_tiling_axis = 0

        self._reduce_axis_index = []
        # 0: old; 1: ; 2:
        self._reduce_schedule_algo = 0
        self._need_split_after = False

        self._need_storage_align_falg = False

        # optimal reduce_sum 5d, 4d
        self.device_core_num = cceconf.get_soc_spec("CORE_NUM")
        self.xouter = []
        self.xinner = []
        self.dsl_type = DSL_REDUCE_TYPE_MAP["cast_single_reduce_sum"]

        if self._scope.lower().find('.ub') != -1:
            self._ub_max_buff = \
                cceconf.get_soc_spec("UB_SIZE") * 224 // 256
            self._total_size = \
                cceconf.get_soc_spec("UB_SIZE") // 2
        else:
            raise RuntimeError("only support UB buffer now")

        self._emit_insn_map = {"elewise_single_cast": "vector_conv",
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
                               "elewise_binary_cmpsel": "vector_cmpsel"
                               }
        # pylint: disable=no-self-use
        self.init_opt_policy()

    def _is_continuous_reduce(self, reduce_axis_index):
        """
        _is_continuous_reduce
        """
        # pylint: disable=consider-using-enumerate, no-self-use
        for i in range(0, len(reduce_axis_index)):
            if i > 0:
                if reduce_axis_index[i] != reduce_axis_index[i-1] + 1:
                    return False
        return True

    def _shape_mul(self, shape):
        """
        _shape_mul
        """
        # pylint: disable=no-self-use
        if not shape:
            return 1
        return reduceIns(lambda x, y: x*y, shape)

    def _is_reduce_last_axis(self):
        """
        _is_reduce_last_axis
        """
        shape_before_reduce = self._shape_before_reduce
        reduce_axis_index = self._reduce_axis_index

        if len(reduce_axis_index) > 1:
            has_last_reduce_axis = \
                ((len(shape_before_reduce) - 1) in reduce_axis_index)
            if has_last_reduce_axis:
                # pylint: disable=no-self-use
                is_continuous_reduce = self._is_continuous_reduce(reduce_axis_index)
                return is_continuous_reduce
            for i in range(reduce_axis_index[-1] + 1, len(shape_before_reduce)):
                if shape_before_reduce[i] != 1:
                    return False
            return True

        has_last_reduce_axis = \
            ((len(shape_before_reduce) - 1) in reduce_axis_index)

        if has_last_reduce_axis:
            return True
        for i in range(reduce_axis_index[-1] + 1, len(shape_before_reduce)):
            if shape_before_reduce[i] != 1:
                return False
        return True

    def _is_mix_reduce_nlast_and_last(self):
        """
        _is_mix_reduce_nlast_and_last
        """
        shape_before_reduce = self._shape_before_reduce
        reduce_axis_index = self._reduce_axis_index
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

    def _is_mix_reduce_nlast_and_nlast(self):
        """
        _is_mix_reduce_nlast_and_nlast
        """
        shape_before_reduce = self._shape_before_reduce
        reduce_axis_index = self._reduce_axis_index
        if len(reduce_axis_index) > 1:
            for i in reduce_axis_index:
                if i not in range(0, len(shape_before_reduce)-1):
                    return False
            return True
        return False

    def _is_after_last_reduce_axis_litter_ub(self):
        """
        _is_after_last_reduce_axis_litter_ub
        """
        total_size = self._shape_mul(self._shape_before_reduce[(self._reduce_axis_index[-1]+1):])
        if total_size < self.get_max_ub_count():
            return True
        return False

    def _is_block_align(self):
        """
        _is_block_align
        """
        if isinstance(self._reduce_index, (list)):
            self._reduce_index = self._reduce_index[0]
        reduce_op = [self._op[self._reduce_index]]
        dtype = reduce_op[0]['src_buffer'][-1].dtype

        reduce_node_last_axis_size = DTYPE_WIDTH_MAP[dtype] * 2 * self._shape_mul(
            self._shape_before_reduce[self._reduce_axis_num[-1] + 1:])

        out_node_last_axis_size = DTYPE_WIDTH_MAP[
            self._res_tensor.dtype] * 2 * self._shape_mul(
                self._shape_before_reduce[self._reduce_axis_num[-1] + 1:])

        if not (reduce_node_last_axis_size >= 32 and reduce_node_last_axis_size % 32 == 0):
            return False

        if not (out_node_last_axis_size >= 32 and out_node_last_axis_size % 32 ==
                0):
            return False

        return True

    def _is_reduce_not_last_axis(self):
        """
        _is_reduce_not_last_axis
        """
        shape_before_reduce = self._shape_before_reduce
        reduce_axis_index = self._reduce_axis_index
        if len(reduce_axis_index) > 1:
            is_continuous_reduce = self._is_continuous_reduce(reduce_axis_index)
            return is_continuous_reduce
        is_not_last_reduce_axis = \
            (reduce_axis_index[0] in range(0, len(shape_before_reduce)-1))
        return is_not_last_reduce_axis

    def _is_nreduce_axis_not_bigger_than_block(self):
        """
        _is_nreduce_axis_not_bigger_than_block
        """
        reduce_op = [self._op[self._reduce_index[0]]]
        op_cmd = reduce_op[0]['op']
        dtype = self._res_tensor.dtype
        if op_cmd == "reduce_prod" and dtype == "float32":
            dtype = "float16"
        shape = []
        for i in range(len(self._shape_before_reduce)):
            if i not in self._reduce_axis_num:
                shape.append(self._shape_before_reduce[i])
        reduce_node_last_axis_size = DTYPE_WIDTH_MAP[dtype] * 2 * self._shape_mul(shape)

        if reduce_node_last_axis_size <= 32:
            return True

        return False

    def _is_vector_op_block_align(self):
        """
        _is_vector_op_block_align
        """
        def __is_vector_op_block_align_case(ub_split_axis):
            self._need_storage_align_falg = self._need_storage_align()
            if self._shape_before_reduce[self._split_axis] % self._last_num != 0:
                self._need_storage_align_falg = False
            # for (126,15,15) [-2] not 32B align, can not enable multi core
            if not self._need_storage_align_falg and ub_split_axis in self._reduce_axis_num and \
                    ub_split_axis - self._split_axis == 1 and \
                    DTYPE_WIDTH_MAP[dtype] * 2 * self._shape_mul(
                            self._shape_before_reduce[ub_split_axis + 1:]) < 32:
                return False
            return True


        vector_op_block_align = True
        dtype = self._res_tensor.dtype
        if (self._is_mix_reduce_nlast_and_nlast() or self._is_reduce_not_last_axis()) and \
                self._is_after_last_reduce_axis_litter_ub() and \
                (not (self._reduce_axis_num[0] == 0 and \
                      self._is_continuous_reduce(self._reduce_axis_num))):
            self._split_axis, self._last_num, ub_split_axis, _ = \
                self._mix_reduce_nlast_and_nlast_tiling(
                    self._shape_before_reduce, self._reduce_axis_num)
            # for (4,5,6,8,9,10,10), [-2] dst not 32B align,can not enable multi core
            # for (16, 16, 16, 8, 8, 8, 8), [-7, 5, 1] dst not 32B align,can not enable multi core
            if (ub_split_axis in self._reduce_axis_num) and (
                    ub_split_axis - self._split_axis >= 2) and (
                        ub_split_axis - 1 not in self._reduce_axis_num) and (
                            DTYPE_WIDTH_MAP[dtype] * 2 * self._shape_mul(
                                self._shape_before_reduce[ub_split_axis + 1:]) < 32):
                return False

            # for (8,5,6,1,8,6,5), [1,3,4,5] dst not 32B align,can not enable multi core
            if (ub_split_axis in self._reduce_axis_num) and (
                    ub_split_axis - self._split_axis >= 2) and (
                        not self._is_block_align()) and (
                            DTYPE_WIDTH_MAP[dtype] * 2 * self._shape_mul(
                                self._shape_before_reduce[ub_split_axis + 1:]) < 32):
                return False

            vector_op_block_align = __is_vector_op_block_align_case(ub_split_axis)
        return vector_op_block_align

    def reduce_need_enable_multicore(self):
        """
        reduce_need_enable_multicore
        """
        reduce_op = [self._op[self._reduce_index[0]]]
        self._shape_before_reduce = self._shape_to_list(reduce_op[0]['src_buffer'][-1].shape)
        self._reduce_axis_index = reduce_op[0]["reduce_axis_num"]
        self._reduce_axis_index.sort()

        index = self.arg_sort(self._reduce_axis_index)
        tmp_reduce_axis_num = self.reorder_list(self._reduce_axis_index, index)
        self._reduce_axis_num = tmp_reduce_axis_num

        if self._need_enable_muticore:
            # Mixed scene with last axis and non-last axis,
            # multicore enabled contision: last axis not 1; or last axis is 1
            # and last reduce axis not 1 and the length of reduce axis not 1.
            if self._is_reduce_last_axis() or self._is_mix_reduce_nlast_and_last():
                if self._shape_before_reduce[-1] != 1:
                    self._reduce_schedule_algo = 1
                else:
                    if self._shape_before_reduce[
                            self._reduce_axis_index[-1]] != 1 and \
                            reduce_op[0]['op'] == "reduce_sum":
                        self._reduce_schedule_algo = 1
                    else:
                        self._need_enable_muticore = False
                        self._reduce_schedule_algo = 0
            # Mixed scene with non-last axis and non-last axis,
            # multicore enabled contision:
            # finally the non-reduce axis is smaller than the ub size
            elif (self._is_mix_reduce_nlast_and_nlast() or \
                  self._is_reduce_not_last_axis()) and \
                    self._is_after_last_reduce_axis_litter_ub():
                if (self._is_nreduce_axis_not_bigger_than_block()) or \
                        (not self._is_vector_op_block_align()):
                    self._need_enable_muticore = False
                    self._reduce_schedule_algo = 0
                else:
                    self._reduce_schedule_algo = 2
            else:
                self._reduce_schedule_algo = 1

        else:
            self._reduce_schedule_algo = 0

    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    def core_schedule_reduce(self, res, spec_node_list, sch_list, tensor_map):
        """
        auto_schedule for cce AI-CORE. For now, only N's elewise operation +
        (0 - 1) reduction operations are supported.
        the last axis must be n*128, except the case of reducting last axis.
        the exception case requires the last axis and reduction axis is 16*n.

        Parameters
        ----------
        res : tvm.tensor

        Returns
        -------
        sch: Schedule
            The computation schedule for the op.

        _origin_tensor: list of tvm.Tensor
            return tensors for tvm.build and tvm.lower
        """
        # for pylint
        tensor_map = tensor_map
        self._res_tensor_list = res
        if hasattr(res, 'index'):
            if len(res) > 1:
                self._is_muti_output = True
                util.get_dst_tensor_map(res, self._dst_tensor_map)
                # get the last out,out1-->....-->out2
                for out in self._res_tensor_list:
                    if out not in self._dst_tensor_map.keys():
                        self._res_tensor = out
            else:
                self._res_tensor = res[0]
        else:
            self._res_tensor = res

        # out1-->.-->out2,self._res_tensor_list constains the all outs except out2
        self._res_tensor_list.remove(self._res_tensor)
        self._schedule = sch_list[0]
        self._spec_node_list = spec_node_list
        self.update(self._res_tensor)

        # optimal reduce_sum
        if self._check_optimal_condition() and not self._is_muti_output and \
                self._need_enable_muticore:
            self._optimal_reduce_sum_4d_5d_schedule()
            sch_list[0] = self._schedule
            return True

        read_buffer = self.local_cache_read()
        write_buffer = self.local_cache_write()

        muti_output_read_buffer = []
        if self._is_muti_output:
            # cache read again for muti-res
            muti_output_read_buffer = self.local_cache_read_for_muti_out()

        self._need_compute_at_before = True
        self._need_compute_at_after = False
        self._is_last_reduce = False

        def call_schedule_func():
            """
            special operations including data slice, data tiling
            """
            if self._have_reduce:
                # Scene judgment
                self.reduce_need_enable_multicore()
                if self._reduce_schedule_algo == 1:
                    # Original reduce multi-core process,
                    # including all reduce last axis and part non-last axis
                    self.reduce_schedule_muticore()
                elif self._reduce_schedule_algo == 2:
                    # New reduce multi-core process,
                    # including mixed scene of non-last axis and non-last axis
                    self.mix_reduce_nlast_schedule()
                else:
                    # reduce non-multi-core process
                    self.reduce_schedule()
            else:
                self.elewise_schedule()

        call_schedule_func()
        # compute_inline operations
        self.local_compute_inline()

        # enable muti core
        if self._need_enable_muticore:
            self.local_enable_muti_core()

        # compute_at operatinos
        self.local_compute_at(read_buffer)
        read_buffer = read_buffer + muti_output_read_buffer

        if not self.check_valid_schedule():
            return False

        # tensorize operations
        if self._need_tensorize:
            if self._need_enable_muticore:
                self.local_tensorize_reduce_muticore()
            else:
                self.local_tensorize()

        if self._is_elewise_single_and_broadcast:
            shape_res = self._shape_to_list(self._res_tensor.shape)
            shape_input = self._shape_to_list(self._origin_tensor[-1].shape)

            if len(shape_res) != len(shape_input):
                raise RuntimeError("shape length for res[%d] and input[%d] are diff."
                                   % (len(shape_res), len(shape_input)))
            broadcast_axis = len(shape_res) - 1
            for i in range(len(shape_res) - 1, 0, -1):
                if shape_res[i] != shape_input[i]:
                    broadcast_axis = i
                    break

            broadcast_prev_axis = broadcast_axis - 1 \
                if broadcast_axis != 0 else 0

            reader = read_buffer[0]
            self._schedule[reader].storage_align(
                self._schedule[reader].op.axis[broadcast_prev_axis], 16, 0)

            writer = write_buffer[0]
            self._schedule[writer].storage_align(
                self._schedule[writer].op.axis[broadcast_prev_axis], 16, 0)

        # pragma operations
        if self._need_pragma:
            self.local_pragma(read_buffer)

        # double buffer
        if not self._have_reduce and self._need_double_buffer:
            self.local_double_buffer(read_buffer)

        sch_list[0] = self._schedule

        return True, self.get_current_workspace_info()

    def get_current_workspace_info(self):
        '''
        try to enable multi core in workspace
        :return: tiling relative info
        '''
        info = {}
        if self._need_compute_at_after:
            info["compute_at_axis"] = self._compute_at_after_reduce_axis
        else:
            info["compute_at_axis"] = self._compute_at_before_reduce_axis
        info["split_axis_index"] = self._split_axis
        info["cache_buffer_map"] = self._cache_buffer_map
        return info

    def cpu_schedule(self, res):
        """
        auto_schedule for cce AI-CPU.
        When the last axis is a large prime number (like 99991),
        will build to ai-cpu automatically

        Parameters
        ----------
        res : tvm.tensor

        Returns
        -------
        sch: Schedule
            The computation schedule for the op.

        """
        mid_tensors = []

        def get_transient_tensor(tensor):
            """
            scan all the transient tensor during calculation
            """
            tmp_list = tensor.op.input_tensors
            for sub_list in tmp_list:
                if sub_list not in mid_tensors and not \
                        isinstance((sub_list.op), tvm.tensor.PlaceholderOp):
                    mid_tensors.append(sub_list)
                    get_transient_tensor(sub_list)

        get_transient_tensor(res)
        schedule = tvm.create_schedule(res.op)
        self.update(res)

        if self._have_reduce:
            for i in schedule.stages:
                if hasattr(i.op, "reduce_axis") and i.op.reduce_axis:
                    ko1, _ = i.split(i.op.reduce_axis[0], 300)
                    tmp = schedule.rfactor(i.op.output(0), ko1)
                    schedule[tmp].set_scope(cceconf.scope_aicpu)
                    for tensor in mid_tensors:
                        schedule[tensor].set_scope(cceconf.scope_aicpu)
                    for tensor in mid_tensors:
                        if not tensor.op.tag.startswith("reduce"):
                            schedule[tensor].compute_inline()
                    break
        else:
            for tensor in mid_tensors:
                schedule[tensor].set_scope(cceconf.scope_aicpu)
            for tensor in mid_tensors:
                schedule[tensor].compute_inline()
        return schedule

    def init_opt_policy(self):
        """
        you can init your optimization policy here, such as double buffer,
        double core and so on
        :return:
        """
        # less memory make more loose depend for double buffer
        if self._need_double_buffer:
            self._total_size = self._total_size // 2

        # According to the 2 member,
        # you can adjust the ration of going to speel schedule
        # the shape is split to (shape[0], outer*self._last_num, shape[1])
        # if (max_ub_count /(self._last_num*shape[1])) >= max_ub_occupancy_ratio
        # represents ub is not far full,
        # if (shape[0]*outer) >= max_outer_ele_num represents more element not
        # move to ub satisfy the two condition will go to speel schedule
        self._max_ub_occupancy_ratio = 3
        self._max_outer_ele_num = 10

        # get device_core_num
        device_core_num = \
            cceconf.get_soc_spec("CORE_NUM")
        if device_core_num == 1:
            self._need_enable_muticore = False
        else:
            self._block_dim = device_core_num

    def get_total_width(self):
        """
        caculate the max useable number based on op liveness
        :return: max useable number
        """

        def _post_dfs_order(tensor_op, op_graph, visited, post_order):
            if tensor_op in visited:
                return
            visited[tensor_op] = True
            post_order.append(tensor_op)
            if tensor_op in op_graph:
                for src in op_graph[tensor_op]:
                    _post_dfs_order(src, op_graph, visited, post_order)

        def _op_width(tensor_op):
            num_type = tensor_op.dtype
            if num_type.lower() not in DTYPE_WIDTH_MAP.keys():
                raise RuntimeError("Can not calculate with no compute")

            tmp_width = 0
            if tensor_op.op.tag is not None:
                tag = tensor_op.op.tag
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
                #vcomsel use 3 temp buffer
                elif tag.find("cmpsel") != -1:
                    tmp_width = 3 * DTYPE_WIDTH_MAP[num_type.lower()]

            return DTYPE_WIDTH_MAP[num_type.lower()] + tmp_width

        op_graph = {}
        for tensor_op in self._origin_op:
            src_op = list(tensor_op['src_buffer'])
            src_op.reverse()
            op_graph[tensor_op['dst_buffer']] = src_op
        visited = {}
        post_order = []
        _post_dfs_order(self._res_tensor, op_graph, visited, post_order)
        lives = [self._res_tensor]
        live_width = _op_width(lives[0])
        max_width = live_width
        visited = {lives[0]: True}
        for tensor_op in post_order:
            if tensor_op in op_graph:
                for src in op_graph[tensor_op]:
                    if src in visited:
                        continue
                    lives.append(src)
                    live_width += _op_width(src)
                    visited[src] = True
                if live_width > max_width:
                    max_width = live_width
            lives.remove(tensor_op)
            live_width -= _op_width(tensor_op)
        return max_width

    def get_max_ub_count(self):
        """
        caculate the max element num loaded in UB buffer
        :return: max element num loaded in UB buffer
        """
        total_width = self.get_total_width()
        if not total_width:
            raise RuntimeError("Can not calculate with no compute")

        max_bound = total_width*128
        max_ub_count = int(self._total_size // max_bound*128)

        return max_ub_count

    # pylint: disable=too-many-branches
    def elewise_tiling(self, shape):
        """
        simple auto tiling module for elewise op

        Parameters
        ----------
        shape : tensor shape

        Returns
        -------
        int : slice number
        """
        max_ub_count = self.get_max_ub_count()

        # if the cmp mode is bit the res shape is 8 muti input shape,
        # we should tiling as the input shape
        if self._op and self._op[-1]["op"] == 'emit_insn_elewise_binary_cmp' \
                and self._op[-1]["args"][1] == 'bit':
            max_ub_count = max_ub_count // 8

        rfactor = max_ub_count
        axis = len(shape) - 1
        # find the split axis, shape = (shape[0],, shape[split_axis], shape[-1])
        # shape[split_axis]*shape[split_axis + 1]*...*shape[-1] less than max_ub_count
        # shape[split_axis - 1]* shape[split_axis + 1].*shape[-1] greater than max_ub_count
        if not self.is_strict_last_axis():
            for num in reversed(shape):
                if max_ub_count == 1:
                    return rfactor, axis + 1
                if max_ub_count >= num:
                    rfactor = num
                    max_ub_count = max_ub_count // num
                else:
                    break
                axis -= 1

        # the vsel in bit mode, the condition shape is not same as input shape,
        # input_shape[-1] equal to condition_shape[-1] * 8,
        # so when spilt the last axis,
        # the spilt factor should be the mutiply of 8

        for i in range(len(self._op)):
            if self._op[i]["op"] == 'emit_insn_elewise_multiple_sel' \
                    and self._op[i]["args"][0] == 'bit':
                if axis == len(shape) - 1:
                    for num in range(max_ub_count, 1, -1):
                        if shape[axis] % num == 0 and num % 8 == 0:
                            rfactor = num
                            return rfactor, axis

        # split the split axis
        # find max(rfactor) st shape[axis] % rfactor == 0
        # and rfactor < max_ub_count
        if axis != -1:
            for num in range(max_ub_count, 1, -1):
                if shape[axis] % num == 0:
                    rfactor = num
                    return rfactor, axis
            if axis == (len(shape) - 1):  # prime number
                rfactor = 1
                return rfactor, axis
        return rfactor, axis + 1

    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    def _block_tiling_reduce_last_axis(self, shape, reduce_axis):
        """
        simple auto block tiling module for reduce op

        Parameters
        ----------
        shape : tensor shape

        Returns
        -------
        int : slice number

        example
        ----------
        [A1,A2...,B,C1,C2...,D]->[A1,A2...,(B1,B2),C1,C2...,D]
        {
            w = C1*C2*...
            h = A1*A1*...
            t = lcm(w,16) #get lcm value of w and 16
            t = t / w
            B2 = t
            B1 = B / B2
            while (B1 !=0 and h * B1 > 32):
                B2 = B2 + t
                B1 = B / B2

            if B1 == 0:
                offset to the axis before B, and do again
            if h*B1<=32
                return
        }
        """

        def gcd(value, factor):
            '''
             G.C.D: get the greatest common divisor for value and factor
            '''
            temp_value = value
            temp_factor = factor
            while temp_value != temp_factor:
                if temp_value > temp_factor:
                    temp_value = temp_value - temp_factor
                else:
                    temp_factor = temp_factor - temp_value
            return temp_value

        def lcm(value, factor):
            '''
             L.C.M: get the lowest common multiple for value and factor
            '''
            res = gcd(value, factor)
            return (value * factor) // res

        def get_origin_axis_by_res_axis(keep_dims_shape, res_axis):
            '''
             get_origin_axis_by_res_axis
            '''
            temp = res_axis
            # pylint: disable=consider-using-enumerate
            for i in range(len(keep_dims_shape)):
                if keep_dims_shape[i] == 0:
                    continue
                temp = temp - 1
                if temp < 0:
                    return i
            return len(keep_dims_shape) - 1

        # init
        origin_axis = 0
        res_axis = 0
        rfactor = shape[res_axis]

        aicore_count = self._block_dim

        # all reduce, no split
        if len(shape) == 1:
            return rfactor, res_axis, origin_axis

        # generate tensor shape after reduce when keep_dims is true or false.
        # res_tensor_shape is the last shape after normal reduce.
        # keep_dims_shape is the last shape after reduce, but keep_dims is true.
        res_tensor_shape = []
        keep_dims_shape = []
        # pylint: disable=consider-using-enumerate
        for i in range(len(shape)):
            if i in reduce_axis:
                if self._is_keepdims:
                    res_tensor_shape.append(1)
                    keep_dims_shape.append(1)
                else:
                    # set the value to 0, as the flag that the axis had reduced.
                    keep_dims_shape.append(0)
            else:
                keep_dims_shape.append(shape[i])
                res_tensor_shape.append(shape[i])

        if not res_tensor_shape:
            return rfactor, res_axis, origin_axis

        right_value = 1.0

        align_type = self._res_tensor.dtype
        #bool is represented by int8
        if align_type == 'bool':
            align_type = 'int8'
        align_factor, _ = util.get_align_factor(align_type)

        # lcm_value is the lowest common multiple of right_value and align_factor,
        # this is to make sure that the amount of data processed
        # by one core is a multiple of align_factor.
        lcm_value = lcm(right_value, align_factor)
        lcm_value = lcm_value // right_value
        # begin from the last non reduce axis to find the block bind axis.
        i = len(res_tensor_shape) - 1

        while i >= 0:
            left_value = 1.0
            if i != 0:
                left_value = self._shape_mul(res_tensor_shape[0:i])

            cur_value = res_tensor_shape[i]
            split_cur_value2 = lcm_value
            split_cur_value1 = cur_value / split_cur_value2
            temp_value = (left_value * ceil(split_cur_value1) - 1) * split_cur_value2
            # the amount of data processed by one core is a multiple of 32 bytes
            # the total core num should be little than aicore_count;
            # if the split axis is not the first axis,
            # the value of split axis should be multiple of rfactor.
            while (int(split_cur_value1) != 0 and
                   ((left_value * ceil(split_cur_value1)) > aicore_count or
                    temp_value >= (left_value * cur_value) and i != 0)):
                # the rfactor(split_cur_value2) increase by lcm_value
                split_cur_value2 = split_cur_value2 + 1
                split_cur_value1 = cur_value / split_cur_value2
                temp_value = (left_value * ceil(split_cur_value1) - 1) * split_cur_value2

            if (int(split_cur_value1) != 0 and (left_value * ceil(
                    split_cur_value1)) <= aicore_count and temp_value < (
                        left_value * cur_value) or (i == 0)):
                res_axis = i
                if split_cur_value2 > res_tensor_shape[res_axis]:
                    split_cur_value2 = res_tensor_shape[res_axis]
                rfactor = int(split_cur_value2)
                origin_axis = get_origin_axis_by_res_axis(keep_dims_shape, res_axis)
                return rfactor, res_axis, origin_axis
            # offset to next axis
            right_value = right_value * cur_value
            if right_value <= align_factor:
                lcm_value = align_factor // right_value + 1
            else:
                lcm_value = 1
            i = i - 1

        origin_axis = get_origin_axis_by_res_axis(keep_dims_shape, res_axis)
        return rfactor, res_axis, origin_axis

    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    def _reduce_before_tiling_last_axis(self, shape, reduce_axis,
                                        split_axis, last_num, use_storage_align=False):
        """
        reduce before ub tiling module for reduce op
        """
        max_ub_count = self.get_max_ub_count()

        def __is_justifiable_split_factor(no_need_multiple, split_axis,
                                          ub_tiling_axis,
                                          cur_axis_val, rfactor):
            """check is justifiable split factor"""
            if no_need_multiple:
                return True

            if split_axis != ub_tiling_axis:
                return True

            if cur_axis_val % rfactor == 0:
                return True

            return False

        align_type = self._res_tensor.dtype
        #bool is represented by int8
        if align_type == 'bool':
            align_type = 'int8'
        if align_type == 'float32':
            align_type = 'float16'
        align_factor, _ = util.get_align_factor(align_type)

        temp_reduce_axis = reduce_axis
        # if the reduce axis is not continuous, only to split the last axis
        if (reduce_axis[-1] - reduce_axis[0]) >= len(reduce_axis):
            num = len(reduce_axis)
            gap = 1
            while num > 0 and gap == 1:
                num = num - 1
                gap = reduce_axis[num] - reduce_axis[num - 1]
            temp_reduce_axis = temp_reduce_axis[num:]

        # get the total size of reduce axis
        total_size_of_reduce = self._shape_mul(shape[temp_reduce_axis[0]:])

        # if input shape dim is 1, and the dim value is prime number,
        # cur_axis_val no need be a multiple of rfactor.
        no_need_multiple = False
        if len(shape) == 1 and util.is_prime_number(shape[0]):
            no_need_multiple = True

        if total_size_of_reduce >= max_ub_count:
            # from the last axis to find the split axis
            i = len(temp_reduce_axis) - 1
            size_load_to_ub = self._shape_mul(shape[temp_reduce_axis[i]:])
            while (i > 0) and (size_load_to_ub < max_ub_count):
                i = i - 1
                size_load_to_ub = size_load_to_ub * shape[temp_reduce_axis[i]]

            cur_axis_val = shape[temp_reduce_axis[i]]
            rfactor = cur_axis_val
            if size_load_to_ub <= max_ub_count:
                return rfactor, temp_reduce_axis[i]

            size_load_to_ub = size_load_to_ub // cur_axis_val
            rfactor = int(cur_axis_val // 2)
            while rfactor > 1:
                if (rfactor * size_load_to_ub) <= max_ub_count \
                        and (__is_justifiable_split_factor(no_need_multiple,
                                                           split_axis,
                                                           temp_reduce_axis[i],
                                                           cur_axis_val,
                                                           rfactor)):
                    break
                rfactor = rfactor - 1
            return rfactor, temp_reduce_axis[i]

        if total_size_of_reduce % align_factor != 0 and not use_storage_align:
            # if total_size_of_reduce is not 32B align,
            # we load total_size_of_reduce data to ub once.
            return shape[temp_reduce_axis[0]], temp_reduce_axis[0]

        # use_storage_align scene, recalculate ub size,
        # max_ub_count equal to reduce_axis_size * max_ub_count // reduce_axis_storage_align_size
        if use_storage_align:
            max_ub_count = int(total_size_of_reduce / \
                               (total_size_of_reduce + align_factor - \
                                total_size_of_reduce % align_factor) * max_ub_count)
        size_load_to_ub = 1
        for index in reversed(reduce_axis):
            size_load_to_ub = size_load_to_ub * shape[index]
            if size_load_to_ub > max_ub_count:
                size_load_to_ub = size_load_to_ub // shape[index]
                rfactor = int(shape[index] // 2)
                while rfactor > 1:
                    if (rfactor * size_load_to_ub) <= max_ub_count:
                        break
                    rfactor = rfactor - 1
                return rfactor, index


        i = temp_reduce_axis[0] - 1
        while (i >= split_axis) and (size_load_to_ub < max_ub_count) and (i not in reduce_axis):
            if i == split_axis:
                size_load_to_ub = size_load_to_ub * last_num
            else:
                size_load_to_ub = size_load_to_ub * shape[i]
            i = i - 1
        i = i + 1

        cur_axis_val = shape[i]
        # if i is the block tiling axis, cur_axis_val equal to last_num
        if i == split_axis:
            cur_axis_val = last_num
        rfactor = cur_axis_val
        if size_load_to_ub <= max_ub_count:
            if i != temp_reduce_axis[0]:
                return rfactor, i
            return shape[reduce_axis[0]], reduce_axis[0]

        size_load_to_ub = size_load_to_ub // cur_axis_val
        rfactor = int(cur_axis_val // 2)
        while rfactor > 1:
            if (rfactor * size_load_to_ub) <= max_ub_count:
                break
            rfactor = rfactor - 1

        return rfactor, i

    def _reduce_after_tiling_last_axis(self, shape, reduce_axis,
                                       split_axis, last_num):
        """
        reduce after ub tiling module for reduce op
        """

        def update_ub_max_by_align(max_ub_count):
            if self._need_storage_align_falg:
                align_type = self._get_align_type()
                if align_type == 'bool':
                    align_type = 'int8'
                if align_type == 'float32':
                    align_type = 'float16'
                align_factor, _ = util.get_align_factor(align_type)
                if self._is_last_reduce:
                    if align_factor > shape[-1]:
                        if self._is_keepdims:
                            max_ub_count = max_ub_count*temp_shape[-1] // \
                                           align_factor
                        else:
                            max_ub_count = max_ub_count // align_factor
                else:
                    a1_start_index, _ = \
                        self._find_last_none_reduce_axis(
                            self._shape_before_reduce, self._reduce_axis_num)

                    non_align_count = self._shape_mul(
                        self._shape_before_reduce[a1_start_index:])
                    align_count = (self._shape_mul(
                        self._shape_before_reduce[a1_start_index:]) //
                                   align_factor + 1)*align_factor
                    max_ub_count = max_ub_count * non_align_count // \
                                   align_count

            return max_ub_count
        max_ub_count = self.get_max_ub_count()

        temp_shape = []
        # set block split axis and reduce axis value to 1
        # pylint: disable=consider-using-enumerate
        for i in range(0, len(shape)):
            if i in reduce_axis:
                if self._is_keepdims:
                    temp_shape.append(1)
            elif i < split_axis:
                temp_shape.append(1)
            elif i == split_axis:
                temp_shape.append(last_num)
            else:
                temp_shape.append(shape[i])

        # shape algined
        max_ub_count = update_ub_max_by_align(max_ub_count)

        total_size_of_non_reduce = self._shape_mul(temp_shape)
        # no need to split
        if total_size_of_non_reduce <= max_ub_count:
            return False, None, None

        i = len(temp_shape) - 1
        size_load_to_ub = temp_shape[i]
        while (i > 0) and (size_load_to_ub < max_ub_count):
            i = i - 1
            size_load_to_ub = size_load_to_ub * temp_shape[i]

        if size_load_to_ub == max_ub_count:
            return True, temp_shape[i], i

        cur_axis_val = temp_shape[i]
        size_load_to_ub = size_load_to_ub // cur_axis_val
        rfactor = int(cur_axis_val // 2)
        align_factor, _ = util.get_align_factor(self._res_tensor.dtype)
        last_axies_count = int(self._shape_mul(self._res_tensor.shape[i + 1:]))
        while rfactor > 1:
            remain_count = self._res_tensor.shape[i].value % rfactor
            if ((rfactor * size_load_to_ub) <= max_ub_count) and \
                    (remain_count == 0 or
                     (remain_count * last_axies_count) >= align_factor):
                break
            rfactor = rfactor - 1

        return True, rfactor, i

    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    def _reduce_tiling_nlst(self, shape, reduce_axis):
        """
        _reduce_tiling_nlst
        """
        def get_res_shape(shape, reduce_axis):
            """
            get_res_shape
            """
            reduce_res_shape = []
            # pylint: disable=consider-using-enumerate
            for i in range(len(shape)):
                if i not in reduce_axis:
                    reduce_res_shape.append(shape[i])
                elif self._is_keepdims:
                    reduce_res_shape.append(1)
            return reduce_res_shape

        def _get_factors_of_positive_integer(num):
            """
            _get_factors_of_positive_integer
            """
            factors = []
            if num <= 0:
                return factors
            sqrt_num = int(math.sqrt(num))
            for i in range(1, sqrt_num + 1, 1):
                if num % i == 0:
                    value = num // i
                    factors.append(i)
                    if value != i:
                        factors.append(value)
            factors.sort()
            return factors

        def get_max_divided_factor(num, value):
            """
            get_max_divided_factor
            """
            max_factor = int(1)
            if num < value:
                return max_factor
            if value == 1:
                return num
            for i in range(value, 0, -1):
                if num % i == 0:
                    max_factor = num // i
                    break
            return max_factor

        # pylint: disable=chained-comparison
        def get_max_factor(num, behand_cnt, align_limit_cnt, ub_limit_cnt):
            """
            get_max_factor
            """
            max_factor = int(1)
            sqrt_n = int(math.sqrt(num))
            for fac in range(num, sqrt_n, -1):
                quotient = num // fac
                remainder = num - (quotient) * fac
                fac_cnt = fac * behand_cnt
                remainder_cnt = remainder * behand_cnt
                if fac_cnt >= align_limit_cnt and fac_cnt <= ub_limit_cnt and \
                        ((remainder_cnt >= align_limit_cnt and \
                          remainder_cnt <= ub_limit_cnt) or remainder_cnt == 0):
                    max_factor = fac
                    break
            return max_factor

        def revers_travl_search(ub_factor_list, d_align_factor,
                                data_cnt_behand_split_axis, max_ub_count):
            """
            revers_travl_search
            """
            res = int(1)
            for factor in reversed(ub_factor_list):
                if d_align_factor <= (factor * data_cnt_behand_split_axis) <= max_ub_count:
                    res = factor
                    break
            return res

        def get_factor(max_devided_factor, d_align_factor,
                       data_cnt_behand_split_axis, max_ub_count):
            """
            revers_travl_search
            """
            if data_cnt_behand_split_axis > max_ub_count:
                return 1
            if data_cnt_behand_split_axis > d_align_factor:
                factor = 1
            else:
                factor = d_align_factor // data_cnt_behand_split_axis + 1
            while factor <= max_devided_factor:
                if (factor * data_cnt_behand_split_axis) <= max_ub_count:
                    factor = factor + 1
                else:
                    break
            res = factor - 1
            return res
        # bool&int8:32, fp16:16, fp32&int32:8
        align_type = self._res_tensor.dtype
        if align_type == 'bool':
            align_type = 'int8'
        d_align_factor, _ = util.get_align_factor(align_type)
        # get reduce_res_shape list
        reduce_res_shape = get_res_shape(shape, reduce_axis)

        # get first axis in reduce_res_shape.
        # need after last reduce axis in org shape
        # ok for keepdim or not
        beg_axis_no_behand_reduce_axis = \
            len(reduce_res_shape) - (len(shape) - reduce_axis[-1] - 1)

        # block split axis
        split_axis = len(reduce_res_shape) - 1
        # block split factor
        rfactor = reduce_res_shape[-1]
        # ub split axis
        ub_tiling_axis = split_axis
        # ub split axis
        ub_tiling_factor = rfactor

        aicore_count = self._block_dim
        max_ub_count = self.get_max_ub_count()

        reduce_res_dim_cnt = len(reduce_res_shape)
        data_cnt_behand_reduce_axis = \
            self._shape_mul(reduce_res_shape[beg_axis_no_behand_reduce_axis:])
        # start assum split beg_axis_no_behand_reduce_axis
        data_cnt_behand_split_axis = \
            self._shape_mul(reduce_res_shape[(beg_axis_no_behand_reduce_axis + 1):])
        dim_val_at_start_axis = \
            reduce_res_shape[beg_axis_no_behand_reduce_axis]
        min_factor_at_start_axis = \
            math.ceil(dim_val_at_start_axis / aicore_count)
        min_data_cnt_each_core = \
            min_factor_at_start_axis * data_cnt_behand_split_axis
        if data_cnt_behand_reduce_axis <= max_ub_count or \
                min_data_cnt_each_core < d_align_factor or aicore_count == 1:
            #scne 1: (cut_b:n, cut_u:n)
            split_axis = beg_axis_no_behand_reduce_axis
            rfactor = dim_val_at_start_axis
            # ubtiling
            ub_tiling_axis = split_axis
            ub_tiling_factor = rfactor
            return split_axis, rfactor, ub_tiling_axis, ub_tiling_factor
        if dim_val_at_start_axis >= aicore_count:
            split_axis = beg_axis_no_behand_reduce_axis
            rfactor = dim_val_at_start_axis

            data_cnt_behand_split_axis = self._shape_mul(
                reduce_res_shape[(beg_axis_no_behand_reduce_axis + 1):])
            # 1，there is just one axis behend last reduce axis
            if (len(reduce_res_shape) - 1) == (beg_axis_no_behand_reduce_axis):
                if min_factor_at_start_axis <= max_ub_count:
                    # scne 2.1.1:(cub_b:y, cut_u:n)
                    # ubtiling
                    ub_tiling_axis = split_axis
                    ub_tiling_factor = min_factor_at_start_axis
                    return split_axis, min_factor_at_start_axis, \
                           ub_tiling_axis, ub_tiling_factor

                max_devided_factor = \
                    get_max_divided_factor(dim_val_at_start_axis,
                                           aicore_count)
                ub_factor_list = \
                    _get_factors_of_positive_integer(max_devided_factor)
                ub_factor = revers_travl_search(ub_factor_list,
                                                d_align_factor,
                                                data_cnt_behand_split_axis,
                                                max_ub_count)
                if ub_factor != 1:
                    # scne 2.1.2:(cut_b:y, cut_u:y)
                    # ubtiling
                    ub_tiling_axis = split_axis
                    ub_tiling_factor = ub_factor
                    return split_axis, max_devided_factor, \
                           ub_tiling_axis, ub_tiling_factor

                # scne 2.1.3:(cut_b:n, cut_u:y)
                # ubtiling
                ub_tiling_axis = split_axis
                ub_tiling_factor = max_ub_count
                return split_axis, dim_val_at_start_axis, \
                       ub_tiling_axis, ub_tiling_factor

            # 2，there is multi axises behend last reduce axis
            if min_data_cnt_each_core <= max_ub_count:
                # scne 2.2.1:(cut_b:y, cut_u:n)
                # ubtiling
                ub_tiling_axis = split_axis
                ub_tiling_factor = min_factor_at_start_axis
                return split_axis, min_factor_at_start_axis, \
                       ub_tiling_axis, ub_tiling_factor
            max_devided_factor = \
                get_max_divided_factor(dim_val_at_start_axis, aicore_count)
            ub_factor = get_factor(max_devided_factor, d_align_factor,
                                   data_cnt_behand_split_axis,
                                   max_ub_count)
            if ub_factor != 1:
                # scne 2.2.2:(cut_b:y, cut_u:y)
                # ubtiling
                ub_tiling_axis = split_axis
                ub_tiling_factor = ub_factor
                return split_axis, max_devided_factor, \
                       ub_tiling_axis, ub_tiling_factor

            # start_axis block_inner is a prime
            # k constraint max block_inner,  avoid too much loop
            k = 5
            if min_factor_at_start_axis <= k:
                if data_cnt_behand_split_axis <= max_ub_count:
                    # scne 2.2.3_1
                    ub_tiling_axis = split_axis
                    ub_tiling_factor = 1
                    return split_axis, min_factor_at_start_axis, \
                           ub_tiling_axis, ub_tiling_factor

                # ub_tiling axis not eq block_tiling axis
                # scne 2.2.3_2
                one_loop_data_cnt = 1
                last_one_loop_data_cnt = 1
                ub_tiling_axis = \
                    beg_axis_no_behand_reduce_axis + 1
                ub_tiling_factor = reduce_res_shape[
                    beg_axis_no_behand_reduce_axis + 1]
                for i in reversed(range(
                        beg_axis_no_behand_reduce_axis + 1,
                        len(reduce_res_shape))):
                    ub_tiling_axis = i
                    one_loop_data_cnt *= reduce_res_shape[i]
                    if one_loop_data_cnt <= max_ub_count:
                        ub_tiling_factor = reduce_res_shape[i]
                        continue
                    else:
                        last_one_loop_data_cnt = \
                            one_loop_data_cnt // reduce_res_shape[i]
                        ub_tiling_factor = get_max_factor(
                            reduce_res_shape[i],
                            last_one_loop_data_cnt,
                            d_align_factor, max_ub_count)
                        break
                return split_axis, min_factor_at_start_axis, \
                       ub_tiling_axis, ub_tiling_factor

            #scne 2.2.4
            # ubtiling
            one_loop_data_cnt = 1
            ub_tiling_axis = beg_axis_no_behand_reduce_axis + 1
            ub_tiling_factor = reduce_res_shape[
                beg_axis_no_behand_reduce_axis + 1]
            for i in reversed(range(
                    beg_axis_no_behand_reduce_axis,
                    len(reduce_res_shape))):
                ub_tiling_axis = i
                one_loop_data_cnt *= reduce_res_shape[i]
                if one_loop_data_cnt <= max_ub_count:
                    ub_tiling_factor = reduce_res_shape[i]
                    continue
                else:
                    last_one_loop_data_cnt = \
                        one_loop_data_cnt // reduce_res_shape[i]
                    ub_tiling_factor = \
                        max_ub_count // last_one_loop_data_cnt
                    break
            return split_axis, dim_val_at_start_axis, \
                   ub_tiling_axis, ub_tiling_factor

        # shape: a1 a2 k1 a3 k2 a4 a5 a6 .. an
        # behand last reduce axis k2: a4*a5*..ak_o <= 32
        # eg:(4, 2, 33333333, 5) --> (4, 2, (3,111111111), 5).
        # -> (fused(4, 2, 3), 111111111, 5)
        max_o = 1
        for i in range(beg_axis_no_behand_reduce_axis,
                       reduce_res_dim_cnt):
            fused_dim_val = self._shape_mul(
                reduce_res_shape[beg_axis_no_behand_reduce_axis:(i + 1)])
            if fused_dim_val <= aicore_count:
                max_o = aicore_count // fused_dim_val
                split_axis = i
                rfactor = reduce_res_shape[i]
                ub_tiling_axis = split_axis
                ub_tiling_factor = reduce_res_shape[i]
                continue
            split_axis = i
            data_cnt_behand_split_axis = \
                self._shape_mul(reduce_res_shape[(i + 1):])
            max_devided_factor = \
                get_max_divided_factor(reduce_res_shape[i], max_o)
            ub_factor_list = \
                _get_factors_of_positive_integer(max_devided_factor)
            ub_factor = revers_travl_search(ub_factor_list,
                                            d_align_factor,
                                            data_cnt_behand_split_axis,
                                            max_ub_count)
            if ub_factor != 1:
                #scne 3.1
                # (cut_b:y, cut_u:n, fused:y)
                # ubtiling
                ub_tiling_axis = split_axis
                ub_tiling_factor = ub_factor
                return split_axis, max_devided_factor, \
                       ub_tiling_axis, ub_tiling_factor
            #scne 3.2
            # ubtiling
            one_loop_data_cnt = 1
            last_one_loop_data_cnt = 1
            ub_tiling_axis = i
            ub_tiling_factor = reduce_res_shape[i]
            for j in reversed(range(i, len(reduce_res_shape))):
                ub_tiling_axis = j
                one_loop_data_cnt *= reduce_res_shape[j]
                if one_loop_data_cnt <= max_ub_count:
                    ub_tiling_factor = reduce_res_shape[j]
                    continue
                else:
                    last_one_loop_data_cnt = \
                        one_loop_data_cnt // reduce_res_shape[j]
                    ub_tiling_factor = get_max_factor(
                        reduce_res_shape[j], last_one_loop_data_cnt,
                        d_align_factor, max_ub_count)
                    break
            return split_axis - 1, 1, ub_tiling_axis, \
                   ub_tiling_factor

        return split_axis, rfactor, ub_tiling_axis, ub_tiling_factor

    def _reduction_tiling(self, shape, reduce_axis):
        """
        simple auto tiling module for reduce op

        Parameters
        ----------
        shape : tensor shape

        Returns
        -------
        int : slice number
        """
        # for pylint, otherwise
        reduce_axis = reduce_axis
        # new version
        max_ub_count = self.get_max_ub_count()

        rfactor = max_ub_count

        # reduce continuous tailing axis
        if self._is_continuous_tailing_axis:
            rfactor, axis = util.get_split_axis(shape, max_ub_count)
            reduce_max_axis = reduce_axis[0]
            # if the index of split axis is smaller than the reduce axis
            # the first axis of the reduce is used for segmentation
            if axis < reduce_max_axis:
                rfactor, axis = shape[reduce_max_axis], reduce_max_axis

            return rfactor, axis

        axis = len(shape) - 1

        for num in range(max_ub_count, 1, -1):
            if shape[axis] % num == 0:
                rfactor = num
                return rfactor, self._reduction_split_axis(axis)
        if axis == len(shape) - 1:  # prime number
            rfactor = 1
            return rfactor, self._reduction_split_axis(axis)
        return rfactor, self._reduction_split_axis(axis)

    def _reduction_split_axis(self, axis):
        """
        according to the split_axis, keepdims,
        if keepdims is False,
        len(input_shape) - len(reduce_axis) = len(output_shape),
        so the split_axis = axis - len(reduce_axis)

        if keepdims is True ,
        len(input_shape) = len(output_shape),so the split_axis = axis
        """
        # if keepdims is False,len(input) - len(reduce_axis) = len(output)
        if not self._is_keepdims:
            return axis - len(self._reduce_axis_num)
        return axis

    def _check_is_elewise_single_and_broadcast(self):
        '''
        check the op is meet the following conditions:
            1. only contain elewise_single_* and broadcast
            2. broadcast is last compute
            3. only have one broadcast
            4. broadcast asix is not last axis
        if meet, return true, if not, return false.
        :return: bool
        '''
        shape_res = self._shape_to_list(self._res_tensor.shape)
        shape_input = self._shape_to_list(self._origin_tensor[-1].shape)

        if len(shape_res) != len(shape_input):
            raise RuntimeError("shape length for res[%d] and input[%d] are diff."
                               % (len(shape_res), len(shape_input)))
        broadcast_axis = len(shape_res) - 1
        for i in range(len(shape_res) - 1, 0, -1):
            if shape_res[i] != shape_input[i]:
                broadcast_axis = i
                break

        if broadcast_axis == len(shape_res) - 1:
            return False

        if self._res_tensor.op.tag != "broadcast_for_tensor":
            return False

        is_contain_elewise_single = False
        for i in range(len(self._op)):
            op_tag = self._op[i]["op"]
            if op_tag.find("elewise_single") == -1:
                return False
            is_contain_elewise_single = True

            if not op_tag in self._emit_insn_map:
                return False

        if not is_contain_elewise_single:
            return False

        return True

    # pylint: disable=too-many-locals, too-many-branches
    def _check_is_last_axis_broadcast(self):
        """
        checkout the broadcast whether is last axis broadcast;
        [2,2,1,1]->[2,2,2,2] is last axis broadcast;

        [1,a2,1,1]->[a1,a2,a3,a4] contains last axis broadcast
        and not last axis broadcast,
        If a3*a4 is aligned ,we think it is last axis broadcast,
        otherwise it is not last axis broadcast;

        [1,1,2,2]->[2,2,2,2] is not last axis broadcast;
        [1,2,1,2]->[2,2,2,2] is not last axis broadcast;
        :return: bool,
        """
        is_inc_broadcast_for_tensor = False
        last_broadcast_axis = 0
        last_broadcast_sizes = []
        is_contrain_nlst_axis_broadcast = False

        for tensor_op in self._origin_op:
            if tensor_op["op"] == "broadcast_for_tensor":
                last_broadcast_size = 1
                is_inc_broadcast_for_tensor = True
                src = tensor_op['src_buffer'][0]
                dst_shape = util.shape_to_list(tensor_op['dst_buffer'].shape)
                src_shape = util.shape_to_list(src.shape)

                broadcast_axis = []
                # pylint: disable=consider-using-enumerate
                for i in range(len(src_shape)):
                    if src_shape[i] != dst_shape[i]:
                        broadcast_axis.append(i)
                    elif src_shape[i] == 1 and dst_shape[i] == 1:
                        broadcast_axis.append(i)

                axis = len(src_shape) - 1
                for i in reversed(broadcast_axis):
                    if i != axis:
                        if axis == len(src_shape) - 1:
                            # the last broadcast axis is not last axis
                            return False, last_broadcast_axis
                        # contains last axis broadcast
                        # and not last axis broadcast
                        is_contrain_nlst_axis_broadcast = True
                        break
                    else:
                        last_broadcast_size *= dst_shape[i]
                    axis = axis - 1

                if broadcast_axis:
                    last_broadcast_axis = axis + 1

                last_broadcast_sizes.append(last_broadcast_size)

        if not is_inc_broadcast_for_tensor:
            return False, last_broadcast_axis

        align_type = self._res_tensor.dtype
        # bool is represented by int8
        if align_type == 'bool':
            align_type = 'int8'

        align_factor, _ = util.get_align_factor(align_type)

        # for [1,a2,1,1]->[a1,a2,a3,a4]
        if is_contrain_nlst_axis_broadcast:
            for size in last_broadcast_sizes:
                if size % align_factor != 0:
                    return False, last_broadcast_axis

        return True, last_broadcast_axis

    def elewise_schedule(self, read_buffer=None):
        """
        do the elewise pattern schedule
        """
        # for pylint, add read_buffer, otherwise
        # "Arguments number differs from overridden method"
        read_buffer = read_buffer
        self._reduce_index = len(self._op)
        self._is_elewise_single_and_broadcast = \
            self._check_is_elewise_single_and_broadcast()
        self._is_last_axis_boradcast, self._last_boradcast_axis = \
            self._check_is_last_axis_broadcast()
        self._need_compute_at_after = False
        shape = self._shape_to_list(self._res_tensor.shape)
        self._last_num, self._split_axis = self.elewise_tiling(shape)
        res_op = self._schedule[self._res_tensor]
        self._res_tensor_axis_len = len(self._res_tensor.op.axis)
        self._compute_at_before_reduce_buffer = self._res_tensor
        outer, inner = res_op.split(res_op.op.axis[self._split_axis], self._last_num)
        self._compute_at_before_reduce_axis = outer
        self._res_dma_axis = inner

        self._shape_before_reduce = self._shape_to_list(self._res_tensor.shape)
        self._read_dma_axis = self._split_axis

    def is_keepdims(self):
        """
        check whether keepdims
        """
        # if the dims of shape_before_reduce is the same as resTensor.shape,
        # the keepdims is true
        return len(self._shape_before_reduce) == len(self._res_tensor.shape)

    def is_continuous_tailing_axis(self, shape, reduce_axis):
        """
        check reduce continuous tailing axis
        """
        for i in self._origin_op:
            if i["op"] == 'broadcast_for_tensor':
                return False

        if len(reduce_axis) > 1:
            index_list = [index for index, _ in enumerate(shape)]

            if index_list[-len(reduce_axis):] == reduce_axis:
                return True

        return False

    def find_last_continue_axes_size(self):
        """
        find the size of the last continue axes
        """
        if self._is_last_reduce:
            #Find the last few reduce axes
            reduce_axis = self._reduce_axis_num
            temp_reduce_axis = self._reduce_axis_num
            if (reduce_axis[-1] - reduce_axis[0]) >= len(reduce_axis):
                num = len(reduce_axis)
                gap = 1
                while num > 0 and gap == 1:
                    num = num - 1
                    gap = reduce_axis[num] - reduce_axis[num - 1]
                temp_reduce_axis = temp_reduce_axis[num:]
            # get the total size of reduce axis
            total_size_of_reduce = self._shape_mul(self._shape_before_reduce[temp_reduce_axis[0]:])
            return total_size_of_reduce
        a1_start_index, _ = \
            self._find_last_none_reduce_axis(self._shape_before_reduce, self._reduce_axis_num)
        total_size_of_not_reduce = self._shape_mul(self._shape_before_reduce[a1_start_index:])
        return total_size_of_not_reduce

    def _is_last_axis_32b_align(self):
        """
        determine if the last continue axes is 32Bytes aligned
        """
        align_type = self._res_tensor.dtype
        align_factor, _ = util.get_align_factor(align_type)
        total_size_of_reduce = self.find_last_continue_axes_size()
        if total_size_of_reduce % align_factor == 0:
            return True
        return False

    def _is_axis_bigger_than_ub(self):
        """
        determine if the last continue axes is bigger than the ub max size
        """
        align_type = self._res_tensor.dtype
        align_factor, _ = util.get_align_factor(align_type)
        total_size_of_axis = self.find_last_continue_axes_size()
        max_ub_count = self.get_max_ub_count()
        max_ub_count = int(total_size_of_axis / (
            total_size_of_axis + align_factor - \
            total_size_of_axis % align_factor) * max_ub_count)
        if total_size_of_axis >= max_ub_count:
            return True
        return False

    def _need_storage_align(self):
        """
        determine if need storage align
        """
        for compute_op in self._op:
            if compute_op["op"] == "broadcast_for_tensor":
                return False

        if isinstance(self._reduce_index, (list)):
            self._reduce_index = self._reduce_index[0]
        reduce_op = [self._op[self._reduce_index]]
        if reduce_op[0]['op'] != "reduce_sum":
            return False
        if not self._need_enable_muticore:
            return False
        # reduce pre-node can not be broadcast
        src_buffer = reduce_op[0]["src_buffer"][0]
        if src_buffer.op.tag.find("broadcast") != -1:
            return False
        if self._is_last_axis_32b_align():
            return False

        if self._is_axis_bigger_than_ub():
            return False

        return True

    # pylint: disable=no-self-use
    def _find_last_none_reduce_axis(self, shape_before_reduce, reduce_axis_index):
        """
        :param shape_before_reduce:
        :param reduce_axis_index:
        :return:
        """
        # shape_before_reduce:(ak+1,rk,..,r2,a2,r1,a1) or (ak,rk,..,r2,a1,r1),
        # find a1 position, a1 may contain continues axis
        a1_end_index = -1
        for i in range(len(shape_before_reduce) - 1, -1, -1):
            if shape_before_reduce[i] != 1:
                if i not in reduce_axis_index:
                    a1_end_index = i
                    break
        a1_start_index = a1_end_index
        for i in range(a1_end_index, -1, -1):
            if i in reduce_axis_index:
                a1_start_index = i + 1
                break

        return a1_start_index, a1_end_index

    def _get_align_type(self):
        """
        get min align type to ensure all storage align is ok
        :return: min align type
        """

        return self._res_tensor.dtype

    def _do_storage_align(self, shape_before_reduce, reduce_axis_index):
        """
        :param hape_before_reduce:
        :param reduce_axis_index:
        :return:
        """

        align_type = self._get_align_type()
        # bool is represented by int8
        if align_type == 'bool':
            align_type = 'int8'
        align_factor, _ = util.get_align_factor(align_type)

        def __do_storage_align_for_last_reduce():
            _, a1_end_index = self._find_last_none_reduce_axis(
                shape_before_reduce,
                reduce_axis_index)
            align_axis = a1_end_index
            if align_axis < 0:
                return

            for i in self._read_cache:
                cache_read_buffer = self._cache_buffer_map[i]
                align_factor, _ = util.get_align_factor(
                    cache_read_buffer.dtype)
                self._schedule[cache_read_buffer].storage_align(
                    cache_read_buffer.op.axis[align_axis], align_factor, 0)
            for i in self._op[:self._reduce_index]:
                if "cache_buffer" in i.keys():
                    cache_write_buffer = i["cache_buffer"]
                    align_factor, _ = util.get_align_factor(
                        cache_write_buffer.dtype)
                    self._schedule[cache_write_buffer].storage_align(
                        cache_write_buffer.op.axis[align_axis], align_factor, 0)

                if i['dst_buffer'] in self._res_tensor_list:
                    before_reduce_mid_out_tensor = i['dst_buffer']
                    align_factor, _ = util.get_align_factor(
                        before_reduce_mid_out_tensor.dtype)
                    self._schedule[before_reduce_mid_out_tensor].storage_align(
                        before_reduce_mid_out_tensor.op.axis[align_axis],
                        align_factor, 0)

                if i['dst_buffer'] in self._read_cache_muti_out.keys():
                    # before reduce mid out read buffer
                    out_read_buffer = self._read_cache_muti_out[i['dst_buffer']]
                    align_factor, _ = util.get_align_factor(
                        out_read_buffer.dtype)
                    self._schedule[out_read_buffer].storage_align(
                        out_read_buffer.op.axis[align_axis],
                        align_factor, 0)

        def __do_storage_align_for_non_last_reduce():
            align_axis_before = self._reduce_axis_num[-1]
            for i in self._read_cache:
                cache_read_buffer = self._cache_buffer_map[i]
                align_factor, _ = util.get_align_factor(
                    cache_read_buffer.dtype)
                self._schedule[cache_read_buffer].storage_align(
                    cache_read_buffer.op.axis[align_axis_before], align_factor, 0)
            for i in self._op[:self._reduce_index]:
                if "cache_buffer" in i.keys():
                    cache_write_buffer = i["cache_buffer"]
                    align_factor, _ = util.get_align_factor(
                        cache_write_buffer.dtype)
                    self._schedule[cache_write_buffer].storage_align(
                        cache_write_buffer.op.axis[align_axis_before], align_factor, 0)
            if self._is_keepdims:
                align_axis_after = align_axis_before
            else:
                align_axis_after = self._reduce_axis_num[-1] - len(self._reduce_axis_num)
            for i in self._op[self._reduce_index:]:
                if "cache_buffer" in i.keys():
                    cache_write_buffer = i["cache_buffer"]
                    align_factor, _ = util.get_align_factor(
                        cache_write_buffer.dtype)
                    self._schedule[cache_write_buffer].storage_align(
                        cache_write_buffer.op.axis[align_axis_after], align_factor, 0)

        # for reduce last axis scene, only node that before reduce need to do storage align
        if self._is_last_reduce:
            __do_storage_align_for_last_reduce()
        else:
            __do_storage_align_for_non_last_reduce()

    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    def reduce_schedule_muticore(self, read_buffer=None):
        """
        do the reduction pattern schedule with last and nist axis reduce
        """
        # for pylint, add read_buffer, otherwise
        # "Arguments number differs from overridden method"
        read_buffer = read_buffer
        if isinstance(self._reduce_index, (list)):
            self._reduce_index = self._reduce_index[0]
        reduce_op = [self._op[self._reduce_index]]

        reduce_buffer = [reduce_op[0]["cache_buffer"]]
        reduce_sub_buffer = reduce_buffer[0]
        tmp_reduce_axis_num = reduce_op[0]["reduce_axis_num"]
        self._shape_before_reduce = \
            self._shape_to_list(reduce_op[0]['src_buffer'][-1].shape)
        reduce_axis = reduce_op[0]["reduce_axis"]
        self._is_last_reduce = self._is_reduce_last_axis() or self._is_mix_reduce_nlast_and_last()
        reduce_op[0]["self._is_last_reduce"] = self._is_last_reduce

        index = self.arg_sort(tmp_reduce_axis_num)
        tmp_reduce_axis_num = self.reorder_list(tmp_reduce_axis_num, index)
        reduce_axis = self.reorder_list(reduce_axis, index)
        self._schedule[reduce_sub_buffer].reorder(*(reduce_axis))

        self._is_keepdims = self.is_keepdims()
        self._reduce_axis_num = tmp_reduce_axis_num

        if self._is_last_reduce:
            src_buffer = reduce_op[0]["src_buffer"][0]
            reduce_sub_buffer = reduce_buffer[0]
            self._res_tensor_axis_len = len(self._res_tensor.op.axis)
            self._is_continuous_tailing_axis = \
                self.is_continuous_tailing_axis(self._shape_before_reduce,
                                                tmp_reduce_axis_num)
            # block tiling, find the block bind axis and
            # rfactor among non reduce axis;
            # the constraint is that the amount of data processed
            # by one core is a multiple of 32 bytes.
            self._last_num, res_axis, self._split_axis = \
                self._block_tiling_reduce_last_axis(self._shape_before_reduce,
                                                    tmp_reduce_axis_num)
            # ub tiling for reduce before, find the split axis and rfactor.
            self._need_storage_align_falg = self._need_storage_align()
            ub_rfactor_before, ub_split_axis_before = \
                self._reduce_before_tiling_last_axis(
                    self._shape_before_reduce, tmp_reduce_axis_num,
                    self._split_axis, self._last_num,
                    use_storage_align=self._need_storage_align_falg)

            if self._need_storage_align_falg:
                self._do_storage_align(self._shape_before_reduce, tmp_reduce_axis_num)

            if ub_split_axis_before >= tmp_reduce_axis_num[0] and \
                    ub_split_axis_before in tmp_reduce_axis_num:
                reduce_axis_index = \
                    tmp_reduce_axis_num.index(ub_split_axis_before)
                real_reduce_axis = reduce_axis[reduce_axis_index]
                split_no, split_ni = \
                    self._schedule[reduce_sub_buffer].split(
                        real_reduce_axis, int(ub_rfactor_before))
            else:
                spit_axis_index = ub_split_axis_before
                if not self._is_keepdims:
                    for i in range(0, ub_split_axis_before, 1):
                        if i in tmp_reduce_axis_num:
                            spit_axis_index = spit_axis_index - 1
                split_no, split_ni = self._schedule[reduce_sub_buffer].split(
                    reduce_sub_buffer.op.axis[spit_axis_index],
                    int(ub_rfactor_before))

            # compute_at_before_axis
            self._compute_at_before_reduce_axis = split_no
            self._compute_at_before_reduce_buffer = reduce_sub_buffer

            # _read_dma_axis
            self._read_dma_axis = ub_split_axis_before
            if (ub_split_axis_before not in tmp_reduce_axis_num) and \
                    (ub_split_axis_before > tmp_reduce_axis_num[0]):
                self._read_dma_axis = tmp_reduce_axis_num[0]

            # emit_insn_axis
            if ub_split_axis_before >= tmp_reduce_axis_num[0] and \
                    ub_split_axis_before in tmp_reduce_axis_num:
                reduce_op[0]["tensorize_axis"] = split_ni
            else:
                if self._need_storage_align():
                    reduce_op[0]["tensorize_axis"] = split_ni
                else:
                    reduce_op[0]["tensorize_axis"] = reduce_axis[0]

            # _res_dma_axis
            res_xo, res_xi = self._schedule[self._res_tensor].split(
                self._res_tensor.op.axis[res_axis], self._last_num)

            fuse_list = []
            for i in range(0, res_axis):
                fuse_list.append(self._res_tensor.op.axis[i])
            fuse_list.append(res_xo)
            fuse_axis = self._schedule[self._res_tensor].fuse(*fuse_list)

            self._multi_core_bind_axis = fuse_axis
            self._multi_core_buffer = self._res_tensor

            self._need_compute_at_after = True
            if len(src_buffer.shape) == 1:
                self._need_compute_at_after = False

            # ub tiling for reduce after, find the split axis and rfactor.
            # we put all the data after reduce into Ub and move it out once,
            # so if the data after reduce is too big, we need tiling.
            need_split_after, ub_rfactor_after, ub_split_axis_after = \
                self._reduce_after_tiling_last_axis(self._shape_before_reduce,
                                                    tmp_reduce_axis_num,
                                                    self._split_axis,
                                                    self._last_num)
            if need_split_after:
                if ub_split_axis_after == res_axis:
                    res_xio, res_xii = self._schedule[self._res_tensor].split(
                        res_xi, int(ub_rfactor_after))
                else:
                    res_xio, res_xii = self._schedule[self._res_tensor].split(
                        self._res_tensor.op.axis[ub_split_axis_after],
                        int(ub_rfactor_after))
                self._res_dma_axis = res_xii
                compute_at_after_reduce_axis = res_xio
                self._res_tensorize_axis = ub_split_axis_after
            else:
                self._res_dma_axis = res_xi
                compute_at_after_reduce_axis = fuse_axis
                self._res_tensorize_axis = res_axis

            if self._need_compute_at_after:
                self._compute_at_after_reduce_axis = \
                    compute_at_after_reduce_axis
                self._compute_at_after_reduce_buffer = self._res_tensor
        else:
            self._res_tensor_axis_len = len(self._res_tensor.op.axis)

            self._split_axis, self._last_num, ub_split_axis, ub_split_factor = \
                self._reduce_tiling_nlst(self._shape_before_reduce,
                                         tmp_reduce_axis_num)

            split_axis_no_in_before_shape = \
                len(self._shape_before_reduce) - \
                (len(self._schedule[reduce_sub_buffer].op.axis) -
                 self._split_axis)

            #reorder
            reduce_reorder_list = []
            reduce_res_axis_in_befor_shape = []
            for i in range(len(self._shape_before_reduce)):
                if i not in tmp_reduce_axis_num or self._is_keepdims:
                    reduce_res_axis_in_befor_shape.append(i)
            for i in range(len(self._schedule[reduce_sub_buffer].op.axis)):
                reduce_reorder_list.insert(
                    reduce_res_axis_in_befor_shape[i],
                    self._schedule[reduce_sub_buffer].op.axis[i])
            insert_point = split_axis_no_in_before_shape - len(reduce_axis)
            if self._is_keepdims:
                insert_point = split_axis_no_in_before_shape
            for k_var in reversed(reduce_axis):
                reduce_reorder_list.insert(insert_point, k_var)

            self._schedule[reduce_sub_buffer].reorder(*(reduce_reorder_list))

            # before reduce:just compute at res tensor
            #ub tiling split
            reduce_ub_split_axis = \
                self._schedule[reduce_sub_buffer].op.axis[ub_split_axis]
            _, reduce_ub_split_i = \
                self._schedule[reduce_sub_buffer].split(reduce_ub_split_axis,
                                                        int(ub_split_factor))
            reduce_op[0]["tensorize_axis"] = reduce_ub_split_i

            # after reduce
            # block tiling split
            res_block_split_o, res_block_split_i = \
                self._schedule[self._res_tensor].split(
                    self._schedule[self._res_tensor].op.axis[self._split_axis],
                    self._last_num)
            # ub tiling split
            res_ub_split_axis = self._res_tensor.op.axis[ub_split_axis]
            if ub_split_axis == self._split_axis:
                res_ub_split_axis = res_block_split_i
            res_ub_split_o, res_ub_split_i = \
                self._schedule[self._res_tensor].split(res_ub_split_axis,
                                                       int(ub_split_factor))
            self._res_dma_axis = res_ub_split_i
            self._emit_after_reduce_axis = ub_split_axis

            self._need_compute_at_after = True
            if self._need_compute_at_after:
                self._compute_at_after_reduce_axis = res_ub_split_o
                self._compute_at_after_reduce_buffer = self._res_tensor

            self._compute_at_before_reduce_axis = reduce_axis[-1]
            self._compute_at_before_reduce_buffer = reduce_sub_buffer

            #fused for bind
            self._emit_before_reduce_axis = split_axis_no_in_before_shape
            fuse_list_outer = []
            last_reduce_axis_no = tmp_reduce_axis_num[-1]
            # axis cnt between tmp_reduce_axis_num[-1] ~ splitaxis
            axis_cnt_btwn_lreduce_and_split = \
                split_axis_no_in_before_shape - last_reduce_axis_no - 1
            start_fuse_idx = \
                self._split_axis - axis_cnt_btwn_lreduce_and_split
            for i in range(start_fuse_idx, self._split_axis):
                fuse_list_outer.append(
                    self._schedule[self._res_tensor].op.axis[i])
            fuse_list_outer.append(res_block_split_o)
            self._multi_core_bind_axis = \
                self._schedule[self._res_tensor].fuse(*fuse_list_outer)
            self._multi_core_buffer = self._res_tensor

            dma_axis_before_reduce = \
                len(self._shape_before_reduce) - \
                (len(self._schedule[reduce_sub_buffer].op.axis) - ub_split_axis)
            self._read_dma_axis = dma_axis_before_reduce

    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    def _find_split_axis(self, shape, begin_axis, end_axis):
        """
        Find the block tiling axis

        Parameters:
        ----------
        shape :  input shape
        begin_axis : begin axis index
        end_axis : end axis index

        Returns
        -------
        res_axis : block tiling axis index
        rfactor : block tiling factor
        """
        if self._is_block_align():
            temp_size = 1
            for i in range(begin_axis, end_axis + 1, 1):
                temp_size = temp_size * shape[i]

            core_num = cceconf.get_soc_spec("CORE_NUM")
            block_dim = core_num
            if temp_size < core_num:
                block_dim = temp_size

            if block_dim < 1:
                block_dim = 1

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
                if temp_size >= block_dim:
                    split_axis = i
                    temp_size = temp_size / shape[i]
                    need_split = True
                    break

            split_size = 1
            # split the split axis
            if need_split:
                for i in range(1, shape[split_axis] + 1, 1):
                    if (temp_size * i) == block_dim:
                        split_size = i
                        break
                    if (temp_size * i) > block_dim:
                        split_size = i - 1
                        break

            return split_axis, split_size

        def gcd(value, factor):
            '''
             gcd
            '''
            temp_value = value
            temp_factor = factor
            while temp_value != temp_factor:
                if temp_value > temp_factor:
                    temp_value = temp_value - temp_factor
                else:
                    temp_factor = temp_factor - temp_value
            return temp_value

        def lcm(value, factor):
            '''
             L.C.M: get the lowest common multiple for value and factor
            '''
            res = gcd(value, factor)
            return (value * factor) // res

        res_axis = 0
        rfactor = shape[res_axis]

        if len(self._res_tensor.shape) == 1:
            return res_axis, rfactor

        core_num = cceconf.get_soc_spec("CORE_NUM")

        align_type = self._res_tensor.dtype
        # bool is represented by int8
        if align_type == 'bool':
            align_type = 'int8'
        align_factor, _ = util.get_align_factor(align_type)
        if self._shape_mul(shape[self._reduce_axis_num[-1] + 1:]) < align_factor:
            right_value = 1.0 * ((align_factor + shape[-1] - 1) // shape[-1])
            lcm_value = right_value
        else:
            right_value = 1.0
            lcm_value = 1.0

        for i in range(end_axis, begin_axis - 1, -1):
            left_value = 1.0
            if i != 0:
                left_value = self._shape_mul(shape[begin_axis:i])

            cur_value = shape[i]
            split_cur_value2 = lcm_value
            split_cur_value1 = cur_value / split_cur_value2
            temp_value = (left_value * ceil(
                split_cur_value1) - 1) * split_cur_value2
            while (int(split_cur_value1) != 0 and
                   ((left_value * ceil(split_cur_value1)) > core_num or
                    temp_value >= (left_value * cur_value) or
                    ((cur_value % split_cur_value2) != 0) and i != 0)):
                split_cur_value2 = split_cur_value2 + lcm_value
                split_cur_value1 = cur_value / split_cur_value2
                temp_value = (left_value * ceil(
                    split_cur_value1) - 1) * split_cur_value2

            if (int(split_cur_value1) != 0 and
                    (left_value * ceil(
                        split_cur_value1)) <= core_num and
                    temp_value < (left_value * cur_value) and
                    ((cur_value % split_cur_value2) == 0) or (i == 0)):
                res_axis = i
                if split_cur_value2 > shape[res_axis]:
                    split_cur_value2 = shape[res_axis]
                rfactor = int(split_cur_value2)
                return res_axis, rfactor
            # offset to next axis
            right_value = right_value * cur_value
            lcm_value = lcm(right_value, align_factor)
            lcm_value = lcm_value // right_value
        return res_axis, rfactor

    # The backend does not support non-divisible split+fuse,
    # so block tiling needs to be adjusted to divide the split for
    # non-divisible split
    def _modify_block_tiling(self, shape, data_size, block_split_axis,
                             block_split_inner_size):
        # There is no need to adjust the situation:
        # 1) split the first axis, 2) divide the split, 3) the axis before
        # the axis to be split is 1
        if block_split_axis == 0 or \
                shape[block_split_axis] % block_split_inner_size == 0 \
                or sum(shape[0:block_split_axis]) == block_split_axis:
            return block_split_axis, block_split_inner_size

        core_num = cceconf.get_soc_spec("CORE_NUM")
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
            bound_size_temp = bound_size_temp * sorted_factors[i]
            if bound_size_temp > core_num:
                f_large = sorted_factors[i]
                if i > 0:
                    f_small = sorted_factors[i-1]
                break

        if f_large * bound_size > core_num * BLOCK_DIM_MULTIPLE:
            block_split_outer_size = f_small
        elif data_size < f_large * bound_size * ENABLE_MULTI_CORE_THRESHOLD:
            block_split_outer_size = f_small
        else:
            block_split_outer_size = f_large

        block_split_inner_size = shape[block_split_axis] // \
                                 block_split_outer_size

        return block_split_axis, block_split_inner_size

    # pylint: disable=no-self-use
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
                    ub_split_inner_factor = sorted_factors[i-1]
                break
            else:
                ub_split_inner_factor = sorted_factors[i]

        return ub_split_inner_factor

    # pylint: disable=unused-argument
    def _get_block_tiling(self, shape, dtype, start_axis, end_axis):
        """
        nlast axis and nlast axis reduce block tiling

        Parameters:
        ----------
        shape :  input shape
        dtype : data type
        begin_axis : begin axis index
        end_axis : end axis index

        Returns
        -------
        res_axis : block tiling axis index
        rfactor : block tiling factor
        """
        if self._is_block_align():
            block_split_axis, block_split_outer_size = self._find_split_axis(
                shape, start_axis, end_axis)
            block_split_inner_size = \
                (shape[block_split_axis] + block_split_outer_size - 1) // \
                block_split_outer_size

            return block_split_axis, block_split_inner_size
        block_split_axis, block_split_inner_size = self._find_split_axis(
            shape, start_axis, end_axis)

        return block_split_axis, block_split_inner_size

    # pylint: disable=too-many-locals
    def _get_ub_tiling(self, shape, block_tiling_axis, block_tiling_inner_loop,
                       max_ub_count):
        """
        nlast axis and nlast axis reduce ub tiling

        Parameters:
        ----------
        shape :  input shape
        block_tiling_axis : block tiling axis index
        block_tiling_inner_loop : block tiling factor
        max_ub_count : max ub size

        Returns
        -------
        ub_split_axis : ub tiling axis index
        ub_split_inner : ub tiling factor
        """
        def __update_max_ub_cout(max_ub_count):
            if self._need_storage_align_falg:
                a1_start_index, _ = \
                    self._find_last_none_reduce_axis(self._shape_before_reduce,
                                                     self._reduce_axis_num)

                align_type = self._get_align_type()
                if align_type == 'bool':
                    align_type = 'int8'
                if align_type == 'float32':
                    align_type = 'float16'
                align_factor, _ = util.get_align_factor(align_type)

                non_align_count = self._shape_mul(self._shape_before_reduce[a1_start_index:])
                align_count = (self._shape_mul(self._shape_before_reduce[a1_start_index:]) //
                               align_factor + 1) * align_factor
                max_ub_count = max_ub_count * non_align_count // align_count
            return max_ub_count

        max_ub_count = __update_max_ub_cout(max_ub_count)

        def __is_align():
            return self._is_block_align() or self._need_storage_align_falg

        if __is_align():
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
        ub_split_axis = self._reduce_axis_num[-1]
        ub_split_inner = 1
        return ub_split_axis, ub_split_inner

    # pylint: disable=too-many-locals
    def _mix_reduce_nlast_and_nlast_tiling(self, shape_before_reduce,
                                           reduce_axis_index):
        # shape_before_reduce: (ak+1,rk,..,r2,a2,r1,a1)
        # find the last none-reduce axis
        # last_none_reduce_axis equal to len(shape_before_reduce) - 1
        last_none_reduce_axis = self._reduce_axis_num[-1] + 1

        orignal_to_reorder_axis_map = {}
        reorder_to_orignal_axis_map = {}
        #  (ak+1,ak,...,a2, rk,..,r2,,r1,a1)
        reordered_shape = [ele for ele in shape_before_reduce]
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

        start_axis = 0
        end_axis = orignal_to_reorder_axis_map[reduce_axis_index[0]] - 1

        max_ub_count = self.get_max_ub_count()
        reorder_block_tiling_axis, block_rfactor = self._get_block_tiling(
            reordered_shape, self._res_tensor.dtype, start_axis, end_axis)

        reorder_ub_tiling_axis, ub_tiling_rfactor = self._get_ub_tiling(
            reordered_shape, reorder_block_tiling_axis, block_rfactor,
            max_ub_count)
        self._ub_tiling_axis = reorder_ub_tiling_axis

        block_tiling_axis = reorder_to_orignal_axis_map[
            reorder_block_tiling_axis]
        ub_tiling_axis = reorder_to_orignal_axis_map[reorder_ub_tiling_axis]

        return block_tiling_axis, block_rfactor, ub_tiling_axis, ub_tiling_rfactor

    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    def _reduce_nlast_and_fisrt_axis_block_tiling(self, shape, begin_axis, end_axis):
        """
        simple auto block tiling module for reduce op

        Parameters
        ----------
        shape : tensor shape

        Returns
        -------
        int : slice number

        example
        ----------
        [A1,A2...,B,C1,C2...,D,G]->[A1,A2...,(B1,B2),C1,C2...,D,G]
        [D,A1,A2...,B,C1,C2...]->[D,A1,A2...,(B1,B2),C1,C2...]
        {
            w = C1*C2*...
            h = A1*A1*...
            t = lcm(w,16) #get lcm value of w and 16
            t = t / w
            B2 = t
            B1 = B / B2
            while (B1 !=0 and h * B1 > 32):
                B2 = B2 + t
                B1 = B / B2

            if B1 == 0:
                offset to the axis before B, and do again
            if h*B1<=32
                return
        }
        """
        def gcd(value, factor):
            """
            gcd
            """
            temp_value = value
            temp_factor = factor
            while temp_value != temp_factor:
                if temp_value > temp_factor:
                    temp_value = temp_value - temp_factor
                else:
                    temp_factor = temp_factor - temp_value
            return temp_value

        def lcm(value, factor):
            '''
             L.C.M: get the lowest common multiple for value and factor
            '''
            res = gcd(value, factor)
            return (value * factor) // res

        if self._is_block_align():
            res_axis = begin_axis
            rfactor = shape[begin_axis]

            core_num = cceconf.get_soc_spec("CORE_NUM")

            align_type = self._res_tensor.dtype
            # bool is represented by int8
            if align_type == 'bool':
                align_type = 'int8'
            align_factor, _ = util.get_align_factor(align_type)
            if self._shape_mul(shape[begin_axis:end_axis+1]) < align_factor:
                return begin_axis, shape[begin_axis]
            right_value = 1.0
            lcm_value = lcm(right_value, align_factor)

            for i in range(end_axis, begin_axis - 1, -1):
                left_value = 1.0
                if i != 0:
                    left_value = self._shape_mul(shape[begin_axis:i])

                cur_value = shape[i]
                split_cur_value2 = lcm_value
                split_cur_value1 = cur_value / split_cur_value2
                temp_value = (left_value * ceil(
                    split_cur_value1) - 1) * split_cur_value2
                while (int(split_cur_value1) != 0 and
                       ((left_value * ceil(split_cur_value1)) > core_num or
                        temp_value >= (left_value * cur_value) or
                        ((cur_value % split_cur_value2) != 0) and i != 0)):
                    split_cur_value2 = split_cur_value2 + lcm_value
                    split_cur_value1 = cur_value / split_cur_value2
                    temp_value = (left_value * ceil(
                        split_cur_value1) - 1) * split_cur_value2

                if (int(split_cur_value1) != 0 and
                        (left_value * ceil(
                            split_cur_value1)) <= core_num and
                        temp_value < (left_value * cur_value) and
                        ((cur_value % split_cur_value2) == 0) or (i == 0)):
                    res_axis = i
                    if split_cur_value2 > shape[res_axis]:
                        split_cur_value2 = shape[res_axis]
                    rfactor = int(split_cur_value2)
                    return res_axis, rfactor
                # offset to next axis
                right_value = right_value * cur_value
                lcm_value = lcm(right_value, align_factor)
                lcm_value = lcm_value // right_value
            return res_axis, rfactor
        res_axis = begin_axis
        rfactor = shape[begin_axis]

        core_num = cceconf.get_soc_spec("CORE_NUM")

        align_type = self._res_tensor.dtype
        # bool is represented by int8
        if align_type == 'bool':
            align_type = 'int8'
        align_factor, _ = util.get_align_factor(align_type)
        if self._shape_mul(shape[begin_axis:end_axis+1]) < align_factor:
            return begin_axis, shape[begin_axis]
        right_value = 1.0
        gcd_value = align_factor


        for i in range(end_axis, begin_axis - 1, -1):
            left_value = 1.0
            if i != 0:
                left_value = self._shape_mul(shape[begin_axis:i])

            cur_value = shape[i]
            split_cur_value2 = gcd_value
            split_cur_value1 = cur_value / split_cur_value2
            temp_value = (left_value * ceil(
                split_cur_value1) - 1) * split_cur_value2
            while (int(split_cur_value1) != 0 and
                   ((left_value * ceil(split_cur_value1)) > core_num or
                    temp_value >= (left_value * cur_value) and i != 0)):
                split_cur_value2 = split_cur_value2 + 1
                split_cur_value1 = cur_value / split_cur_value2
                temp_value = (left_value * ceil(
                    split_cur_value1) - 1) * split_cur_value2

            if (int(split_cur_value1) != 0 and
                    (left_value * ceil(
                        split_cur_value1)) <= core_num and
                    temp_value < (left_value * cur_value) or (i == 0)):
                res_axis = i
                if split_cur_value2 > shape[res_axis]:
                    split_cur_value2 = shape[res_axis]
                rfactor = int(split_cur_value2)
                return res_axis, rfactor
            # offset to next axis
            right_value = right_value * cur_value
            lcm_value = lcm(right_value, gcd_value)
            if lcm_value < align_factor:
                lcm_value = lcm(right_value, align_factor)
            gcd_value = lcm_value // right_value
        return res_axis, rfactor

    def _reduce_nlast_and_fisrt_axis_ub_tiling(self, shape, block_tiling_axis,
                                               block_tiling_inner_loop,
                                               max_ub_count):
        """
        _reduce_nlast_and_fisrt_axis_ub_tiling ub tiling

        Parameters:
        ----------
        shape :  input shape
        block_tiling_axis : block tiling axis index
        block_tiling_inner_loop : block tiling factor
        max_ub_count : max ub size

        Returns
        -------
        ub_split_axis : ub tiling axis index
        ub_split_inner : ub tiling factor
        """
        if self._is_block_align():
            last_axis = len(shape) - 1
            ub_split_inner = 1
            ub_split_axis = 0
            if block_tiling_axis < 0 or block_tiling_axis > last_axis:
                return ub_split_axis, ub_split_inner

            bound_size = max_ub_count
            split_axis = 0
            temp_size = block_tiling_inner_loop * self._shape_mul(shape[block_tiling_axis + 1:])
            need_split = False
            for i in range(block_tiling_axis - 1, -1, -1):
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
            else:
                split_size = shape[0]

            ub_split_inner = split_size
            ub_split_axis = split_axis

            return ub_split_axis, ub_split_inner
        ub_split_axis = self._reduce_axis_num[-1]
        ub_split_inner = 1
        return ub_split_axis, ub_split_inner

    def _reduce_nlast_and_fisrt_axis_tiling(self, shape_before_reduce,
                                            reduce_axis_index):
        """
        _reduce_nlast_and_fisrt_axis_tiling

        Parameters:
        ----------
        shape_before_reduce :  input shape
        reduce_axis_index : reduce axis index list

        Returns
        -------
        block_tiling_axis : block tiling axis index
        block_rfactor : block tiling factor
        ub_split_axis : ub tiling axis index
        ub_split_inner : ub tiling factor
        """
        start_axis = reduce_axis_index[-1] + 1
        end_axis = len(shape_before_reduce) - 1
        max_ub_count = self.get_max_ub_count()
        block_tiling_axis, block_rfactor = self._reduce_nlast_and_fisrt_axis_block_tiling(
            shape_before_reduce, start_axis, end_axis)

        ub_tiling_axis, ub_tiling_rfactor = self._reduce_nlast_and_fisrt_axis_ub_tiling(
            shape_before_reduce, block_tiling_axis, block_rfactor,
            max_ub_count)

        return block_tiling_axis, block_rfactor, ub_tiling_axis, ub_tiling_rfactor

    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    # pylint: disable=consider-using-enumerate, too-many-nested-blocks
    def mix_reduce_nlast_schedule(self, read_buffer=None):
        """
        mix reduce nlast and nlast schedule
        :param read_buffer:
        :return:
        """
        # for pylint, add read_buffer, otherwise
        # "Arguments number differs from overridden method"
        read_buffer = read_buffer
        if isinstance(self._reduce_index, (list)):
            self._reduce_index = self._reduce_index[0]
        reduce_op = [self._op[self._reduce_index]]

        reduce_buffer = [reduce_op[0]["cache_buffer"]]
        reduce_sub_buffer = reduce_buffer[0]
        tmp_reduce_axis_num = reduce_op[0]["reduce_axis_num"]
        self._shape_before_reduce = \
            self._shape_to_list(reduce_op[0]['src_buffer'][-1].shape)
        reduce_axis = reduce_op[0]["reduce_axis"]
        self._is_last_reduce = self._is_reduce_last_axis() or self._is_mix_reduce_nlast_and_last()
        reduce_op[0]["self._is_last_reduce"] = self._is_last_reduce

        index = self.arg_sort(tmp_reduce_axis_num)
        tmp_reduce_axis_num = self.reorder_list(tmp_reduce_axis_num, index)
        reduce_axis = self.reorder_list(reduce_axis, index)

        self._is_keepdims = self.is_keepdims()
        self._reduce_axis_num = tmp_reduce_axis_num

        self._res_tensor_axis_len = len(self._res_tensor.op.axis)

        def __do_storage_align():
            if self._need_storage_align_falg:
                self._do_storage_align(self._shape_before_reduce, tmp_reduce_axis_num)

        # the reduce axis is non-last axis and the first axis,
        # tiling with _reduce_nlast_and_fisrt_axis_tiling,
        # others with _mix_reduce_nlast_and_nlast_tiling
        if self._reduce_axis_num[0] == 0 and self._is_continuous_reduce(self._reduce_axis_num):
            self._split_axis, self._last_num, ub_split_axis, ub_split_factor = \
                self._reduce_nlast_and_fisrt_axis_tiling(
                    self._shape_before_reduce, tmp_reduce_axis_num)
        else:
            self._split_axis, self._last_num, ub_split_axis, ub_split_factor = \
                self._mix_reduce_nlast_and_nlast_tiling(
                    self._shape_before_reduce, tmp_reduce_axis_num)
            __do_storage_align()

        def __is_align():
            return self._is_block_align() or self._need_storage_align_falg

        if __is_align():
            reordered_axis_list = reduce_axis
            for i in range(self._reduce_axis_num[-1] + 1, len(self._shape_before_reduce)):
                if not self._is_keepdims:
                    op_axis = i - len(self._reduce_axis_num)
                else:
                    op_axis = i
                reordered_axis_list = reordered_axis_list + [
                    self._schedule[reduce_sub_buffer].op.axis[op_axis]]

            self._schedule[reduce_sub_buffer].reorder(*(reordered_axis_list))

            #ub tiling split
            is_ub_tiling_reduce_axis = True
            if ub_split_axis in tmp_reduce_axis_num:
                for i in range(0, len(tmp_reduce_axis_num)):
                    if ub_split_axis == tmp_reduce_axis_num[i]:
                        reduce_ub_split_axis = \
                            self._schedule[reduce_sub_buffer].op.reduce_axis[i]
            else:
                reduce_after_ub_split_axis = ub_split_axis
                if not self._is_keepdims:
                    for i in range(0, reduce_after_ub_split_axis):
                        if i in self._reduce_axis_num:
                            reduce_after_ub_split_axis = reduce_after_ub_split_axis - 1
                reduce_ub_split_axis = \
                    self._schedule[reduce_sub_buffer].op.axis[reduce_after_ub_split_axis]
                is_ub_tiling_reduce_axis = False

            reduce_ub_split_o, reduce_ub_split_i = \
                self._schedule[reduce_sub_buffer].split(reduce_ub_split_axis,
                                                        int(ub_split_factor))

            if is_ub_tiling_reduce_axis:
                reduce_op[0]["tensorize_axis"] = reduce_ub_split_i
            else:
                if len(self._reduce_axis_num) == 1 and self._reduce_axis_num[0] == 0:
                    reduce_op[0]["tensorize_axis"] = reduce_ub_split_i
                else:
                    reduce_op[0]["tensorize_axis"] = \
                        self._schedule[reduce_sub_buffer].op.reduce_axis[0]

            split_block_axis = self._split_axis
            if not self._is_keepdims:
                for i in range(0, len(self._shape_before_reduce)):
                    if i in self._reduce_axis_num and self._split_axis > i:
                        split_block_axis = split_block_axis - 1

            # after reduce
            # block tiling split
            res_block_split_o, res_block_split_i = \
                self._schedule[self._res_tensor].split(
                    self._schedule[self._res_tensor].op.axis[split_block_axis],
                    self._last_num)

            # self._res_dma_axis equal to res_block_split_i

            fuse_list_outer = []
            for i in range(0, split_block_axis):
                fuse_list_outer.append(
                    self._schedule[self._res_tensor].op.axis[i])
            fuse_list_outer.append(res_block_split_o)

            self._multi_core_bind_axis = \
                self._schedule[self._res_tensor].fuse(*fuse_list_outer)
            self._multi_core_buffer = self._res_tensor

            # ub tiling for reduce after, find the split axis and rfactor.
            # we put all the data after reduce into Ub and move it out once,
            # so if the data after reduce is too big, we need tiling.
            self._need_split_after, ub_rfactor_after, ub_split_axis_after = \
                self._reduce_after_tiling_last_axis(self._shape_before_reduce,
                                                    tmp_reduce_axis_num,
                                                    self._split_axis,
                                                    self._last_num)
            if self._need_split_after:
                if ub_split_axis_after == split_block_axis:
                    res_xio, res_xii = self._schedule[self._res_tensor].split(
                        res_block_split_i, int(ub_rfactor_after))
                else:
                    res_xio, res_xii = self._schedule[self._res_tensor].split(
                        self._res_tensor.op.axis[ub_split_axis_after],
                        int(ub_rfactor_after))
                self._res_dma_axis = res_xii
                compute_at_after_reduce_axis = res_xio
                self._res_tensorize_axis = ub_split_axis_after
            else:
                self._res_dma_axis = res_block_split_i
                compute_at_after_reduce_axis = self._multi_core_bind_axis
                self._res_tensorize_axis = split_block_axis

            self._need_compute_at_after = True
            if self._need_compute_at_after:
                # self._compute_at_after_reduce_axis eaual to self._multi_core_bind_axis
                self._compute_at_after_reduce_axis = compute_at_after_reduce_axis
                self._compute_at_after_reduce_buffer = self._res_tensor

            self._compute_at_before_reduce_axis = reduce_ub_split_o
            self._compute_at_before_reduce_buffer = reduce_sub_buffer

            self._read_dma_axis = ub_split_axis
        else:
            reordered_axis_list = reduce_axis
            for i in range(self._reduce_axis_num[-1] + 1, len(self._shape_before_reduce)):
                if not self._is_keepdims:
                    op_axis = i - len(self._reduce_axis_num)
                else:
                    op_axis = i
                reordered_axis_list = reordered_axis_list + [
                    self._schedule[reduce_sub_buffer].op.axis[op_axis]]

            self._schedule[reduce_sub_buffer].reorder(*(reordered_axis_list))
            #ub tiling split
            is_ub_tiling_reduce_axis = True
            reduce_after_ub_split_axis = ub_split_axis
            if ub_split_axis in tmp_reduce_axis_num:
                for i in range(0, len(tmp_reduce_axis_num)):
                    if ub_split_axis == tmp_reduce_axis_num[i]:
                        reduce_ub_split_axis = \
                            self._schedule[reduce_sub_buffer].op.reduce_axis[i]
            else:
                if not self._is_keepdims:
                    for i in range(0, reduce_after_ub_split_axis):
                        if i in self._reduce_axis_num:
                            reduce_after_ub_split_axis = reduce_after_ub_split_axis - 1
                reduce_ub_split_axis = \
                    self._schedule[reduce_sub_buffer].op.axis[reduce_after_ub_split_axis]
                is_ub_tiling_reduce_axis = False

            reduce_ub_split_o, reduce_ub_split_i = \
                self._schedule[reduce_sub_buffer].split(reduce_ub_split_axis,
                                                        int(ub_split_factor))

            if is_ub_tiling_reduce_axis:
                reduce_op[0]["tensorize_axis"] = reduce_ub_split_i
            else:
                reduce_op[0]["tensorize_axis"] = self._schedule[reduce_sub_buffer].op.reduce_axis[0]

            # after reduce
            # block tiling split
            split_block_axis = self._split_axis
            if not self._is_keepdims:
                for i in range(0, len(self._shape_before_reduce)):
                    if i in self._reduce_axis_num and self._split_axis > i:
                        split_block_axis = split_block_axis - 1
            res_block_split_o, res_block_split_i = \
                self._schedule[self._res_tensor].split(
                    self._schedule[self._res_tensor].op.axis[split_block_axis],
                    self._last_num)

            fuse_list_outer = []
            for i in range(0, split_block_axis):
                fuse_list_outer.append(
                    self._schedule[self._res_tensor].op.axis[i])
            fuse_list_outer.append(res_block_split_o)
            self._multi_core_bind_axis = \
                self._schedule[self._res_tensor].fuse(*fuse_list_outer)
            self._multi_core_buffer = self._res_tensor

            res_dam_axis = ub_split_axis + 1
            if not self._is_keepdims:
                for i in range(0, ub_split_axis + 1):
                    if i in self._reduce_axis_num:
                        res_dam_axis = res_dam_axis - 1

            if res_dam_axis in range(0, split_block_axis + 1):
                # for {"shape":(1601, 161, 63),"dtype":"float16"}
                self._res_dma_axis = res_block_split_i
            else:
                self._res_dma_axis = self._schedule[self._res_tensor].op.axis[res_dam_axis]

            self._need_compute_at_after = True
            if self._need_compute_at_after:
                if (tmp_reduce_axis_num[0] == 0 and \
                        self._is_continuous_reduce(tmp_reduce_axis_num) and \
                        not self._is_block_align()):
                    # for {"shape":(1601, 161, 63),"dtype":"float16"}
                    self._compute_at_after_reduce_axis = self._multi_core_bind_axis
                    self._compute_at_after_reduce_buffer = self._res_tensor
                else:
                    compute_at_after_reduce_axis = res_block_split_i
                    copy_size = \
                        DTYPE_WIDTH_MAP[self._res_tensor.dtype] * 2 * self._shape_mul(
                            self._shape_before_reduce[reduce_after_ub_split_axis + 1:])
                    for i in range(reduce_after_ub_split_axis, self._split_axis, -1):
                        if (i in self._reduce_axis_num) and (
                                i - 1 not in self._reduce_axis_num) and (
                                    i - 1 != self._split_axis) and (
                                        copy_size >= 32):
                            axis_index = i - 1
                            for j in range(0, i-1):
                                if j in self._reduce_axis_num and not self._is_keepdims:
                                    axis_index = axis_index - 1
                            compute_at_after_reduce_axis = \
                                self._schedule[self._res_tensor].op.axis[axis_index]
                            break
                        else:
                            copy_size = copy_size * self._shape_before_reduce[i - 1]
                    self._compute_at_after_reduce_axis = compute_at_after_reduce_axis
                    self._compute_at_after_reduce_buffer = self._res_tensor

            self._compute_at_before_reduce_axis = reduce_ub_split_o
            self._compute_at_before_reduce_buffer = reduce_sub_buffer

            self._read_dma_axis = ub_split_axis

    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    def reduce_schedule(self, read_buffer=None):
        """
        do the reduction pattern schedule with last and nist axis reduce
        """
        # for pylint, add read_buffer, otherwise
        # "Arguments number differs from overridden method"
        read_buffer = read_buffer
        if isinstance(self._reduce_index, (list)):
            self._reduce_index = self._reduce_index[0]
        reduce_op = [self._op[self._reduce_index]]

        reduce_buffer = [reduce_op[0]["cache_buffer"]]
        reduce_sub_buffer = reduce_buffer[0]
        tmp_reduce_axis_num = reduce_op[0]["reduce_axis_num"]
        self._shape_before_reduce = \
            self._shape_to_list(reduce_op[0]['src_buffer'][-1].shape)
        reduce_axis = reduce_op[0]["reduce_axis"]
        self._is_last_reduce = \
            ((len(self._shape_before_reduce) - 1) in tmp_reduce_axis_num)
        reduce_op[0]["self._is_last_reduce"] = self._is_last_reduce

        index = self.arg_sort(tmp_reduce_axis_num)
        tmp_reduce_axis_num = self.reorder_list(tmp_reduce_axis_num, index)
        reduce_axis = self.reorder_list(reduce_axis, index)
        self._schedule[reduce_sub_buffer].reorder(*(reduce_axis))

        self._is_keepdims = self.is_keepdims()
        self._reduce_axis_num = tmp_reduce_axis_num

        if self._is_last_reduce:
            real_reduce_axis = reduce_axis[-1]
            src_buffer = reduce_op[0]["src_buffer"][0]
            reduce_sub_buffer = reduce_buffer[0]
            self._res_tensor_axis_len = len(self._res_tensor.op.axis)
            self._is_continuous_tailing_axis = self.is_continuous_tailing_axis(
                self._shape_before_reduce,
                tmp_reduce_axis_num)
            self._last_num, self._split_axis = \
                self._reduction_tiling(self._shape_before_reduce,
                                       tmp_reduce_axis_num)
            ndim = len(self._schedule[self._res_tensor].op.axis)
            self._need_compute_at_after = True
            if len(src_buffer.shape) == 1:
                self._need_compute_at_after = False

            # reduce continuous tailing axis
            if self._is_continuous_tailing_axis:
                reduce_axis_index = tmp_reduce_axis_num.index(self._split_axis)
                real_reduce_axis = reduce_axis[reduce_axis_index]

            if self._is_continuous_tailing_axis or \
                    self._last_num < self._shape_before_reduce[-1]:
                xouter, xinner = self._schedule[reduce_sub_buffer].split(
                    real_reduce_axis, self._last_num)
                self._compute_at_before_reduce_axis = xouter
                reduce_op[0]["tensorize_axis"] = xinner
            else:
                if len(tmp_reduce_axis_num) == 1:
                    self._compute_at_before_reduce_axis = \
                        self._schedule[reduce_sub_buffer].op.axis[ndim - 1]
                else:
                    self._compute_at_before_reduce_axis = reduce_axis[-2]

                reduce_op[0]["tensorize_axis"] = real_reduce_axis

            self._res_dma_axis = self._schedule[self._res_tensor].op.axis[
                self._res_tensor_axis_len - 1]
            if self._need_compute_at_after:
                if self._is_keepdims:
                    self._compute_at_after_reduce_axis = \
                        self._res_tensor.op.axis[
                            len(self._res_tensor.op.axis) - 2]
                else:
                    # if not keepdims, the input shape [2,3] with reduce axis 1
                    # , the res shape will be [2]
                    # when after computing res at input,
                    # we want to get a dma copy axis,
                    # so we should split the the axis of res from 2 to 2 *1
                    # the 2 is the compute at axis, the 1 is the dma copy axis.
                    xouter, xinner = self._schedule[self._res_tensor].split(
                        self._res_tensor.op.axis[len(
                            self._res_tensor.op.axis) - 1], 1)
                    self._compute_at_after_reduce_axis = xouter
                    self._res_dma_axis = xinner
                self._compute_at_after_reduce_buffer = self._res_tensor
            self._compute_at_before_reduce_buffer = reduce_sub_buffer

        else:
            self._res_tensor_axis_len = len(self._res_tensor.op.axis)
            self._last_num, self._split_axis = \
                self._reduction_tiling(self._shape_before_reduce,
                                       tmp_reduce_axis_num)
            ndim = len(self._schedule[self._res_tensor].op.axis)
            self._need_compute_at_after = True
            xouter, xinner = self._schedule[self._res_tensor].split(
                self._schedule[self._res_tensor].op.axis[self._split_axis],
                self._last_num)
            self._res_dma_axis = xinner

            if self._need_compute_at_after:
                self._compute_at_after_reduce_axis = xouter
                self._compute_at_after_reduce_buffer = self._res_tensor

            self._compute_at_before_reduce_buffer = reduce_sub_buffer
            self._compute_at_before_reduce_axis = reduce_axis[-1]
            reduce_op[0]["tensorize_axis"] = \
                self._schedule[reduce_op[0]["cache_buffer"]].op.axis[
                    self._split_axis]
            ndim = len(
                self._schedule[self._compute_at_before_reduce_buffer].op.axis)

            self._schedule[self._compute_at_before_reduce_buffer].reorder( \
                *(reduce_axis + list(self._schedule[ \
                    self._compute_at_before_reduce_buffer].op.axis)[self._split_axis:ndim]))

        # if keepdims the split_axis has plused the len(tmp_reduce_axis_num)
        if not self._is_keepdims and not self._is_continuous_tailing_axis:
            self._read_dma_axis = self._split_axis + len(tmp_reduce_axis_num)
        else:
            self._read_dma_axis = self._split_axis

    def local_compute_inline(self):
        """
        compute inline operations
        """
        for i in set(self._compute_inline):
            self._schedule[i].compute_inline()

    def local_cache_read(self):
        """
        cache read operations

        Parameters
        ----------
        None

        Returns
        -------
        list : read buffers
        """
        read_buffer = []
        for i in self._read_cache:
            cache_read_buffer = \
                self._schedule.cache_read(i, self._scope,
                                          list(set(self._read_cache_map[i])))
            read_buffer.append(cache_read_buffer)
            self._cache_buffer_map[i] = cache_read_buffer
        return read_buffer

    def local_cache_write(self):
        """
        cache write operations

        Parameters
        ----------
        None

        Returns
        -------
        list : write buffers
        """
        write_buffer = []
        for i in self._write_cache:
            cache_write_buffer = self._schedule.cache_write(i, self._scope)
            write_buffer.append(cache_write_buffer)
            self._cache_buffer_map[i] = cache_write_buffer
            self._operaters[write_buffer[-1]] = self._operaters.get(i)
            self._write_buffer_map[i] = write_buffer[-1]

        for i in range(len(self._op)):
            self._op[i]["cache_buffer"] = write_buffer[i]
        return write_buffer

    def local_cache_read_for_muti_out(self):
        """
           cache read operations for muti-res

           Parameters
           ----------
           None

           Returns
           -------
           list : read buffers
        """
        read_buffer = []
        for tensor in self._res_tensor_list:
            tensor_out = self._dst_tensor_map[tensor]
            tensor_out_cache = []
            for out in tensor_out:
                tensor_out_cache.append(self._write_buffer_map[out])
            read_buffer.append(
                self._schedule.cache_read(tensor, self._scope,
                                          list(set(tensor_out_cache))))
            # record the read_cache of out, it will be emit_insn empty
            self._read_cache_muti_out[tensor] = read_buffer[-1]

        return read_buffer

    # pylint: disable=too-many-branches
    def local_tensorize(self):
        """
        tensorize operations
        """

        split_shape = self._last_num
        # cmp in bit mode, the tiling shape is the cmp res ,
        # the input shape's last dim is 8 muti res shape
        # if split the last dim, the split shape should mutipy 8,
        # so the tensorize shape is right
        if self._op and \
                self._op[-1]["op"] == 'emit_insn_elewise_binary_cmp' \
                and self._op[-1]["args"][1] == 'bit' \
                and self._split_axis == \
                len(self._op[-1]["dst_buffer"].shape) - 1:
            split_shape = self._last_num*8

        for lop in self._op[self._reduce_index + 1:]:
            cache_buffer = lop["cache_buffer"]
            tmp_shape = self._shape_to_list(lop["dst_buffer"].shape)
            # such as (101, 10241)-->(101,), the tmp_shape should append 1,
            # or tensorize_shape will be wrong
            if not self._is_keepdims and self._is_last_reduce:
                tmp_shape.append(1)
            if (lop["op"].lower().find("elewise") != -1) or \
                    (lop["op"].lower().find("vector") != -1) or \
                    (lop["op"].lower().find("broadcast") != -1):

                lop["tensorize_axis"] = self._schedule[cache_buffer].op.axis[
                    len(self._schedule[cache_buffer].op.axis) - 1]
                lop["tensorize_shape"] = [min(tmp_shape[-1], split_shape)]

                # the muti res has cache write and cache_read,
                # this for remove the reducant cache read by emit an empyt stmt
                # A_UB equal to compute_xxx()
                # A equal to A_UB
                # A_UB_TEMP equal to A
                # B_UB eaual to compute(A_UB_TEMP)
                # we should remove A_UB_TEMP = A
                if lop["dst_buffer"] in self._read_cache_muti_out.keys():
                    lop["cache_read_for_res"] = \
                        self._read_cache_muti_out[lop["dst_buffer"]]
                    lop["tensorize_axis_for_res"] = \
                        self._schedule[lop["cache_read_for_res"]].op.axis[
                            len(self._schedule[lop["cache_read_for_res"]].op.axis) - 1]

        for lop in self._op[:self._reduce_index]:
            cache_buffer = lop["cache_buffer"]
            tmp_shape = self._shape_to_list(lop["dst_buffer"].shape)
            if (lop["op"].lower().find("elewise") != -1) or \
                    (lop["op"].lower().find("vector") != -1) or \
                    (lop["op"].lower().find("broadcast") != -1):
                lop["tensorize_axis"] = \
                    self._schedule[cache_buffer].op.axis[self._read_dma_axis]
                lop["tensorize_shape"] = (min(tmp_shape[self._read_dma_axis], split_shape),) + \
                                         tuple(tmp_shape[self._read_dma_axis + 1:])
                # the muti res has cache write and cache_read,
                # this for remove the reducant cache read by emit an empyt stmt
                # A_UB equal to compute_xxx()
                # A equal to A_UB
                # A_UB_TEMP equal to A
                # B_UB equal to compute(A_UB_TEMP)
                # we should remove A_UB_TEMP = A,
                if lop["dst_buffer"] in self._read_cache_muti_out.keys():
                    lop["cache_read_for_res"] = \
                        self._read_cache_muti_out[lop["dst_buffer"]]
                    lop["tensorize_axis_for_res"] = \
                        self._schedule[lop["cache_read_for_res"]].op.axis[
                            self._read_dma_axis]

        if self._have_reduce:
            reduce_op = self._op[self._reduce_index]
            tmp_shape = self._shape_to_list(reduce_op["src_buffer"][0].shape)
            reduce_tensorize_shape = \
                (min(tmp_shape[self._read_dma_axis], split_shape),) + \
                tuple(tmp_shape[self._read_dma_axis + 1:])
            if reduce_op["self._is_last_reduce"]:
                reduce_op["tensorize_shape"] = reduce_tensorize_shape
            else:
                reduce_op["tensorize_shape"] = \
                    [1]*(self._read_dma_axis - max(
                        reduce_op["reduce_axis_num"])) \
                    + list(reduce_tensorize_shape)

            # A as reduce(input), B as elewise(A),the A,B are output
            # the reduce op A is a mid tensor and out,
            # it should be cache read again from gm
            # the operation sholud be emit an empty intrinc
            if reduce_op["dst_buffer"] in self._read_cache_muti_out.keys():
                reduce_op["cache_read_for_res"] = \
                    self._read_cache_muti_out[reduce_op["dst_buffer"]]
                cache_res = reduce_op["cache_read_for_res"]
                reduce_op["tensorize_axis_for_res"] = \
                    self._schedule[cache_res].op.axis[
                        len(self._schedule[cache_res].op.axis) - 1]

        for lop in self._op:
            self.tensorize_for_op(lop)

    # pylint: too-many-nested-blocks, too-many-branches, too-many-statements
    def local_tensorize_reduce_muticore(self):
        """
        tensorize operations
        """
        split_shape = self._last_num
        if self._op and \
                self._op[-1]["op"] == 'emit_insn_elewise_binary_cmp' \
                and self._op[-1]["args"][1] == 'bit' \
                and self._split_axis == \
                len(self._op[-1]["dst_buffer"].shape) - 1:
            split_shape = self._last_num*8
        for lop in self._op[self._reduce_index + 1:]:
            cache_buffer = lop["cache_buffer"]
            tmp_shape = self._shape_to_list(lop["dst_buffer"].shape)
            if not self._is_keepdims and self._is_last_reduce:
                tmp_shape.append(1)
            if (lop["op"].lower().find("elewise") != -1) or \
                    (lop["op"].lower().find("vector") != -1) or \
                    (lop["op"].lower().find("broadcast") != -1):
                if self._is_last_reduce:
                    lop["tensorize_axis"] = \
                        self._schedule[cache_buffer].op.axis[self._res_tensorize_axis]
                else:
                    block_axis = self._split_axis
                    if not self._is_keepdims:
                        for i in range(0, block_axis + 1):
                            if i in self._reduce_axis_num:
                                block_axis = block_axis - 1
                    if self._need_split_after:
                        lop["tensorize_axis"] = \
                            self._schedule[cache_buffer].op.axis[self._res_tensorize_axis]
                    else:
                        lop["tensorize_axis"] = self._schedule[cache_buffer].op.axis[block_axis]
                lop["tensorize_shape"] = [min(tmp_shape[-1], split_shape)]
                if lop["dst_buffer"] in self._read_cache_muti_out.keys():
                    lop["cache_read_for_res"] = self._read_cache_muti_out[lop["dst_buffer"]]
                    if self._is_last_reduce:
                        lop["tensorize_axis_for_res"] = \
                            self._schedule[lop["cache_read_for_res"]].op.axis[ \
                                self._res_tensorize_axis]
                    else:
                        if self._need_split_after:
                            lop["tensorize_axis_for_res"] = \
                                self._schedule[
                                    lop["cache_read_for_res"]].op.axis[
                                        self._res_tensorize_axis]
                        else:
                            lop["tensorize_axis_for_res"] = \
                                self._schedule[lop["cache_read_for_res"]].op.axis[self._split_axis]
        for lop in self._op[:self._reduce_index]:
            cache_buffer = lop["cache_buffer"]
            tmp_shape = self._shape_to_list(lop["dst_buffer"].shape)
            if (lop["op"].lower().find("elewise") != -1) or \
                    (lop["op"].lower().find("vector") != -1) or \
                    (lop["op"].lower().find("broadcast") != -1):
                lop["tensorize_axis"] = \
                    self._schedule[cache_buffer].op.axis[self._read_dma_axis]
                lop["tensorize_shape"] = (min(tmp_shape[self._read_dma_axis], split_shape),) + \
                                         tuple(tmp_shape[self._read_dma_axis + 1:])
                if lop["dst_buffer"] in self._read_cache_muti_out.keys():
                    lop["cache_read_for_res"] = \
                        self._read_cache_muti_out[lop["dst_buffer"]]
                    lop["tensorize_axis_for_res"] = \
                        self._schedule[lop["cache_read_for_res"]].op.axis[
                            self._read_dma_axis]
        if self._have_reduce:
            reduce_op = self._op[self._reduce_index]
            tmp_shape = self._shape_to_list(reduce_op["src_buffer"][0].shape)
            reduce_tensorize_shape = (min(tmp_shape[self._read_dma_axis], split_shape),) + tuple(
                tmp_shape[self._read_dma_axis + 1:])
            if reduce_op["self._is_last_reduce"]:
                reduce_op["tensorize_shape"] = reduce_tensorize_shape
            else:
                reduce_op["tensorize_shape"] = \
                    [1]*(self._read_dma_axis - max(
                        reduce_op["reduce_axis_num"])) + \
                    list(reduce_tensorize_shape)

            # A as reduce(input), B as elewise(A),the A,B are output
            # the reduce op A is a mid tensor and out,
            # it should be cache read again from gm
            # the operation sholud be emit an empty intrinc
            if reduce_op["dst_buffer"] in self._read_cache_muti_out.keys():
                reduce_op["cache_read_for_res"] = \
                    self._read_cache_muti_out[reduce_op["dst_buffer"]]
                cache_res = reduce_op["cache_read_for_res"]
                reduce_op["tensorize_axis_for_res"] = \
                    self._schedule[cache_res].op.axis[len(
                        self._schedule[cache_res].op.axis) - 1]
        for lop in self._op:
            self.tensorize_for_op(lop)

    def local_pragma(self, read_buffer):
        """
        pragma operations
        """
        for i in read_buffer:
            self._schedule[i].emit_insn(
                self._schedule[i].op.axis[self._read_dma_axis],
                cceconf.dma_copy)

        align_type = self._res_tensor.dtype
        align_factor, _ = util.get_align_factor(align_type)
        total_size_of_axis = self.find_last_continue_axes_size()
        if self._need_storage_align_falg and \
                total_size_of_axis < align_factor and \
                not self._is_last_reduce:
            self._schedule[self._res_tensor].emit_insn(
                self._res_dma_axis, "dma_copy_for_non_32_align")
        else:
            self._schedule[self._res_tensor].emit_insn(
                self._res_dma_axis, cceconf.dma_copy)
        # dma copy for muti res
        for lop in self._op[self._reduce_index:]:
            # the muti res has cache write and cache_read, this for cache read
            write_buffer = lop["dst_buffer"]
            # the muti-output will produce redundant gm_to_ub ,
            # will will use emit_insn to delete it
            if write_buffer in self._read_cache_muti_out.keys():
                self._schedule[write_buffer].emit_insn(
                    self._schedule[write_buffer].op.axis[
                        len(self._schedule[write_buffer].op.axis) - 1],
                    cceconf.dma_copy)

                # A_UB equal to compute(XXX)
                # A equal to A_UB
                # A_UB_TEMP equal to A
                # B equal to compute(A_UB_TEMP)
                # we want to replace A_UB_TEMP by A_UB,
                # and delete the A_UB_TEMP = A
                self._schedule[write_buffer].pragma(
                    self._schedule[write_buffer].op.axis[len(
                        self._schedule[write_buffer].op.axis) - 1],
                    'reuse_input', self._reuse_buffer_index)
                self._schedule[self._read_cache_muti_out[write_buffer]].pragma(
                    self._read_cache_muti_out[write_buffer].op.axis[len(
                        self._schedule[write_buffer].op.axis) - 1],
                    'replace_output', self._reuse_buffer_index)
                self._reuse_buffer_index = self._reuse_buffer_index + 1

        # dma copy for muti res
        for lop in self._op[:self._reduce_index]:
            write_buffer = lop["dst_buffer"]
            # the muti-output will produce redundant gm_to_ub ,
            # will will use emit_insn to delete it
            if write_buffer in self._read_cache_muti_out.keys():
                self._schedule[write_buffer].emit_insn(
                    self._schedule[write_buffer].op.axis[self._read_dma_axis],
                    cceconf.dma_copy)

                # A_UB equal to compute(XXX)
                # A equal to A_UB
                # A_UB_TEMP equal to A
                # B equal to compute(A_UB_TEMP)
                # we want to replace A_UB_TEMP by A_UB,
                # and delete the A_UB_TEMP = A
                self._schedule[write_buffer].pragma(
                    self._schedule[write_buffer].op.axis[self._read_dma_axis],
                    'reuse_input', self._reuse_buffer_index)
                self._schedule[self._read_cache_muti_out[write_buffer]].pragma(
                    self._read_cache_muti_out[write_buffer].op.axis[
                        self._read_dma_axis],
                    'replace_output', self._reuse_buffer_index)
                self._reuse_buffer_index = self._reuse_buffer_index + 1


    def local_double_buffer(self, read_buffer):
        """
        double buffer operations
        read_buffer : the all read_cache for input in ub, type is list
        """
        temp_write_buffer = []
        for i in read_buffer:
            self._schedule[i].double_buffer()
            # just for ternary instruction
            if i in self._double_buffer_map:
                buffers = list(set(self._double_buffer_map[i]))
                for buffer_tmp in buffers:
                    temp_write_buffer.append(buffer_tmp)
                    self._schedule[buffer_tmp].double_buffer()
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
                for buffer_tmp in buffers:
                    temp_write_buffer.append(buffer_tmp)
                    self._schedule[buffer_tmp].double_buffer()
        self._recursive_double_buffer(temp_write_buffer)

    def local_enable_muti_core(self):
        """
        local_enable_muti_core
        """
        # reduce do not support muti-core
        if not self._have_reduce:
            self.elewise_muti_core()
        else:
            block = tvm.thread_axis("blockIdx.x")
            self._schedule[self._multi_core_buffer].bind(
                self._multi_core_bind_axis, block)

    # pylint: disable=too-many-locals
    def elewise_muti_core(self):
        """
        elewise muti core
        :return:
        """
        shape = self._shape_to_list(self._res_tensor.shape)

        # if all number can load ub once, do not enable muti core
        if self._split_axis == 0 and self._last_num == shape[0]:
            return

        # the shape is split to (shape[0], outer*self._last_num, shape[1])
        dma_copy_count = self._last_num
        for axis in range(self._split_axis + 1, len(shape)):
            dma_copy_count = dma_copy_count*shape[axis]

        align_type = self._res_tensor.dtype
        # bool is represented by int8
        if align_type == 'bool':
            align_type = 'int8'

        align_factor, _ = util.get_align_factor(align_type)

        # if the dma_copy_count in one core less than align_factor,
        # do not enable muti-core
        if dma_copy_count <= align_factor:
            return

        # if the dma_copy_count can not div aligin,
        # the muti core write to gm may conflict, do not enable muti-core

        # the shape is split to (shape[0], outer*self._last_num, shape[1])
        fuse_axis_length = 1
        for axis in range(0, self._split_axis + 1):
            fuse_axis_length = fuse_axis_length*shape[axis]

        fuse_axis_length = fuse_axis_length // self._last_num

        # if the outloop is 1, no need to muti core
        if fuse_axis_length == 1:
            return

        # we find minimum factor from [core_number,
        # fuse_axis_length) which can be dived by fuse_axis_length
        # then we spilt fuse_axis_length to factor and fuse_axis_length/factor,
        # we will take the factor task to muti_core
        factor = self._block_dim if self._block_dim < fuse_axis_length else \
            fuse_axis_length

        while factor < fuse_axis_length:
            if fuse_axis_length % factor == 0:
                break
            factor += 1

        # if blockdim is so big, no need to enbale muti-core
        if factor > self.MAX_BLOCK_DIM:
            return

        # the shape is split to (shape[0], outer*self._last_num, shape[1])
        need_fuse_axis = []
        for i in range(0, self._split_axis):
            need_fuse_axis.append(
                self._compute_at_before_reduce_buffer.op.axis[i])
        # the out axis
        need_fuse_axis.append(self._compute_at_before_reduce_axis)

        fuse_axis = need_fuse_axis[0]
        for i in range(1, len(need_fuse_axis)):
            fuse_axis = \
                self._schedule[self._compute_at_before_reduce_buffer].fuse(
                    fuse_axis, need_fuse_axis[i])

        xouter, xiner = \
            self._schedule[self._compute_at_before_reduce_buffer].split(
                fuse_axis, nparts=factor)
        self._compute_at_before_reduce_axis = xiner
        block = tvm.thread_axis("blockIdx.x")
        self._schedule[self._compute_at_before_reduce_buffer].bind(xouter, block)

    def local_compute_at(self, read_buffer):
        """
        tvm.compute_at operations
        """
        if self._need_compute_at_after and \
                self._compute_at_after_reduce_axis is not None:
            for i in self._op[self._reduce_index:]:
                if i['dst_buffer'] in self._res_tensor_list:
                    self._schedule[i['dst_buffer']].compute_at(
                        self._schedule[self._compute_at_after_reduce_buffer],
                        self._compute_at_after_reduce_axis)
                if "cache_buffer" in i.keys():
                    self._schedule[i["cache_buffer"]].compute_at(
                        self._schedule[self._compute_at_after_reduce_buffer],
                        self._compute_at_after_reduce_axis)

                # the cache read again from mid res should
                # compute at the same with res
                if i['dst_buffer'] in self._read_cache_muti_out.keys():
                    self._schedule[
                        self._read_cache_muti_out[i['dst_buffer']]].compute_at(
                        self._schedule[self._compute_at_after_reduce_buffer],
                        self._compute_at_after_reduce_axis)

        if self._need_compute_at_before and \
                self._compute_at_before_reduce_axis is not None:
            for i in self._op[:self._reduce_index]:
                # the res is not compute_inlined so it need compute_at too
                if i['dst_buffer'] in self._res_tensor_list:
                    self._schedule[i['dst_buffer']].compute_at(
                        self._schedule[self._compute_at_before_reduce_buffer],
                        self._compute_at_before_reduce_axis)

                if "cache_buffer" in i.keys():
                    self._schedule[i["cache_buffer"]].compute_at(
                        self._schedule[self._compute_at_before_reduce_buffer],
                        self._compute_at_before_reduce_axis)

                # the cache read again from mid res
                # should compute at the same with res
                if i['dst_buffer'] in self._read_cache_muti_out.keys():
                    self._schedule[
                        self._read_cache_muti_out[i['dst_buffer']]].compute_at(
                        self._schedule[self._compute_at_before_reduce_buffer],
                        self._compute_at_before_reduce_axis)

            for i in read_buffer:
                self._schedule[i].compute_at(
                    self._schedule[self._compute_at_before_reduce_buffer],
                    self._compute_at_before_reduce_axis)

    def check_valid_schedule(self):
        """
        check_valid_schedule
        """
        # the speel schedule can not deal more than one spec_node
        if len(self._spec_node_list) > 1:
            return True

        # in bit mode, the shape is changed , the realize pass can not deal
        for i in self._origin_op:
            if (i["op"] == 'emit_insn_elewise_multiple_sel' and
                    i["args"][0] == 'bit') \
                    or (i["op"] == 'emit_insn_elewise_binary_cmp' and
                        i["args"][1] == 'bit'):
                return True

        if self._have_reduce:
            return self.check_valid_reduction_schedule()
        return self.check_valid_elewise_schedule()

    def check_valid_elewise_schedule(self):
        """
        check_valid_elewise_schedule
        """
        # if return False, it will try speel schedule
        # big prime number,shape likely (2,99991)
        if self._last_num == 1:
            return False

        # if there are more than one independent intermediary tensor,
        # the speel schedule not support
        if len(self._spec_node_list) == 1:
            shape = self._shape_to_list(self._res_tensor.shape)

            # the shape is split to (shape[0], outer*self._last_num, shape[1])
            # if (max_ub_count /(self._last_num*shape[1])) >= 5
            # represents ub is not far full
            # if (shape[0]*outer) >= 10 represents more element not move to ub
            outer = shape[self._split_axis] // self._last_num

            max_ub_count = self.get_max_ub_count()
            ele_not_in_ub_num = outer
            for axis in range(0, self._split_axis):
                ele_not_in_ub_num = ele_not_in_ub_num*shape[axis]

            ele_in_ub_num = self._last_num
            for axis in range(self._split_axis + 1, len(shape)):
                ele_in_ub_num = ele_in_ub_num*shape[axis]

            if (max_ub_count // ele_in_ub_num >= self._max_ub_occupancy_ratio) \
                    and ele_not_in_ub_num >= self._max_outer_ele_num:
                return False

        return True

    # pylint: disable=no-self-use
    def check_valid_reduction_schedule(self):
        """
        check_valid_reduction_schedule
        """
        # self._last_num is the max common divisor of length of last axis
        # and max number of points calculate once
        self.__clean_orig_tensor()
        self._origin_tensor = \
            list(set(self._vars)) + list(set(self._origin_tensor))
        is_last_reduce_num = 1 if self._is_last_reduce else 0

        # if keepdims = True, the shape of result is same as input,
        # cann't sub is_last_reduce_num
        if self._is_keepdims:
            tmp_len = len(self._res_tensor.op.axis) - 1
        else:
            tmp_len = len(self._res_tensor.op.axis) - 1 - is_last_reduce_num
        if not self._need_enable_muticore:
            if is_last_reduce_num == 1 and \
                    (self._last_num == 1) and (self._shape_before_reduce[-1] != 1) \
                    and (self._split_axis == tmp_len):
                return False
        return True

    def _is_cast_support(self, lop):
        """
        _is_cast_support
        """
        cache_buffer = lop["dst_buffer"]
        read_buffer = lop["src_buffer"][0]
        if read_buffer.dtype == "int32":
            if cache_buffer.dtype == "float32":
                return False
        return True

    def _check_dich_add_reduce_dim(self):
        """
        check reduce dim of dich add
        """
        if len(self._shape_before_reduce) == 4:
            # for (N//16,M//16,block_in,block_out), batch is M//16 * block_in
            if self._reduce_axis_num == [1, 2]:
                return True
        elif len(self._shape_before_reduce) == 2:
            if self._reduce_axis_num == [0] and \
                    self._shape_before_reduce[-1] % 16 == 0:
                return True
        return False

    def _check_fractal_dich_add(self):
        """
        check dich_add fractal case
        """
        if self._check_dich_add_reduce_dim() and self._ub_tiling_axis == 1:
            if self._shape_before_reduce[-1] == 16 and \
                self._shape_before_reduce[-2] == 16:
                return True
        return False

    def _check_nd_dich_add(self):
        """
        check dich add ND case
        """
        dich_add_rep_size = 64
        dich_add_type_size = 4
        dich_last_shape = self._last_num

        if self._check_dich_add_reduce_dim() and \
                dich_last_shape <= dich_add_rep_size and \
                dich_add_rep_size % dich_last_shape == 0 and \
                dich_last_shape * dich_add_type_size % 32 == 0:
            return True
        return False

    def _check_dich_add_emit_insn(self, lop):
        if self._res_tensor.dtype.lower() == "float16" and \
                lop["op"] == "reduce_sum":
            if self._check_fractal_dich_add() or \
                    self._check_nd_dich_add():
                return True
        return False

    def tensorize_for_op_reduce_nlast(self, lop, vec_intrin):
        """
        tensorize for reduce_nlast single_op
        """
        if self._need_enable_muticore:
            if lop["op"] == "reduce_prod":
                self._schedule[lop["cache_buffer"]].emit_insn(
                    lop["tensorize_axis"], "vector_mul")
            # for dichotomy add optimize
            elif self._check_dich_add_emit_insn(lop):
                self._schedule[lop["cache_buffer"]].emit_insn(
                    lop["tensorize_axis"], "vector_dichotomy_add")
            else:
                self._schedule[lop["cache_buffer"]].emit_insn(
                    # tensorize_axis, "reduce_nlst_axis_" + lop["op"])
                    lop["tensorize_axis"], "vector_" + lop["op"])
        else:
            reduce_func = vec_intrin("reduce_nist_axis")(
                lop["tensorize_shape"], lop["op"], 0, lop["cache_buffer"].dtype)
            self._schedule[lop["cache_buffer"]].tensorize(lop["tensorize_axis"], reduce_func)

    def _get_backend_reduce_last_insn(self):
        # v100 not support fp32 vcmax/vcmin,
        #     front-end reduce_max/min using vmax/vmin.
        # 1951 v200 support fp32 vcmax/vcmin, using backend emit_insn.
        vc_intr_support_fp32 = intrinsic_check_support("Intrinsic_vcmax",
                                                       "float32")
        backend_reduce_insn = ["reduce_sum"]
        if vc_intr_support_fp32:
            backend_reduce_insn = ["reduce_sum", "reduce_min", "reduce_max"]

        return backend_reduce_insn

    # pylint: disable=too-many-branches, too-many-statements
    def tensorize_for_op(self, lop):
        """
        tensorize for single_op
        """
        vec_intrin = cceconf.cce_intrin.intrin_factor(self._scope)
        op_cmd = lop["op"].split("_")
        cache_buffer = lop["cache_buffer"]
        tensorize_shape = lop["tensorize_shape"]
        tensorize_axis = lop["tensorize_axis"]
        if op_cmd[0] == "emit":
            elewise_func = lop["op"].split("emit_insn_")[-1]
            self._schedule[cache_buffer].emit_insn(tensorize_axis, elewise_func)
        elif op_cmd[0].lower() == "elewise":
            emit_insn_pragma = self._emit_insn_map.get(lop["op"])
            if emit_insn_pragma:
                if emit_insn_pragma == "vector_multiple":
                    self.emit_multiple(cache_buffer, lop, op_cmd)
                else:
                    self._schedule[cache_buffer].emit_insn(
                        cache_buffer.op.axis[0], emit_insn_pragma)

            else:
                if op_cmd[1] == "multiple":
                    elewise_func = vec_intrin("elewise_multiple_intrin_cce")(
                        tensorize_shape,
                        lop["op"],
                        lop["dst_buffer"].dtype,
                        lop["src_buffer"][-1].dtype, lop["args"])
                elif op_cmd[1] == "binary":
                    is_same = (len(set(self._operaters[cache_buffer])) == 1)
                    elewise_func = vec_intrin("elewise_binary_intrin_cce")(tensorize_shape,
                                                                           lop["op"],
                                                                           lop["dst_buffer"].dtype,
                                                                           lop["src_buffer"][
                                                                               -1].dtype,
                                                                           is_same=is_same,
                                                                           args=lop["args"])
                else:
                    elewise_func = vec_intrin("elewise_single_intrin_cce")(tensorize_shape,
                                                                           lop["op"],
                                                                           lop["dst_buffer"].dtype,
                                                                           lop["src_buffer"][
                                                                               -1].dtype,
                                                                           lop["args"])
                self._schedule[cache_buffer].tensorize(tensorize_axis,
                                                       elewise_func)
        elif op_cmd[0].lower() == "vector":
            self._schedule[cache_buffer].emit_insn(tensorize_axis, lop["op"])
        elif op_cmd[0].lower() == "broadcast":
            self.emit_for_broadcast(cache_buffer, lop, tensorize_axis)

        elif op_cmd[0].lower() == "reduce":
            # ====== reduction tensorize ============
            self._is_last_reduce = lop["self._is_last_reduce"]
            if self._is_last_reduce:
                if self._need_enable_muticore:
                    def __is_need_vector_emit_insn():
                        backend_reduce_insn = self._get_backend_reduce_last_insn()
                        if lop["op"] in backend_reduce_insn or \
                                (cache_buffer.dtype == "float16" and
                                 lop["op"] in ["reduce_min", "reduce_max"]):
                            return True
                        return False

                    if __is_need_vector_emit_insn():
                        self._schedule[cache_buffer].emit_insn(
                            tensorize_axis, "vector_" + lop["op"])
                    else:
                        self._schedule[cache_buffer].emit_insn(
                            tensorize_axis, "reduce_last_axis_" + lop["op"])

                else:
                    reduce_func = vec_intrin("reduce_last_axis")(
                        tensorize_shape, lop["op"], cache_buffer.dtype)
                    self._schedule[cache_buffer].tensorize(tensorize_axis,
                                                           reduce_func)
            else:
                self.tensorize_for_op_reduce_nlast(lop, vec_intrin)
        else:
            raise RuntimeError("%s not support" % lop["op"])

        # by emit_insn phony_insn to rm the redundant copy_gm_ub op for muti-out
        if 'cache_read_for_res' in lop.keys():
            cache_buffer_for_res = lop['cache_read_for_res']
            self._schedule[cache_buffer_for_res].emit_insn(
                lop["tensorize_axis_for_res"], 'phony_insn')

    def emit_for_broadcast(self, cache_buffer, lop, tensorize_axis):
        """Emit insn for broadcast in tensorize"""
        if lop["op"] == "broadcast_for_tensor":
            lop["op"] = "unified_broadcast"
        self._schedule[cache_buffer].emit_insn(tensorize_axis, lop["op"])

    # pylint: disable=too-many-branches, too-many-statements
    def emit_multiple(self, cache_buffer, lop, op_cmd):
        """
        emit insn for ternary instruction
        """
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
            reuse_buffer = src[1]
            pragma = "vector_maddrelu"
        elif op_cmd[-1].lower() == "madd":
            reuse_buffer = src[1]
            pragma = "vector_madd"
        elif op_cmd[-1].lower() == "mla":
            reuse_buffer = src[2]
            pragma = "vector_mla"
        elif op_cmd[-1].lower() == "axpy":
            index = 1 if len(lop["src_buffer"]) > 1 else 0
            reuse_buffer = lop["src_buffer"][index]
            pragma = "vector_axpy"

        reuse_buffer_scope = self._cache_buffer_map[reuse_buffer]
        self._schedule[cache_buffer].emit_insn(cache_buffer.op.axis[0], pragma)
        self._schedule[reuse_buffer_scope].reused_by(cache_buffer)
        if reuse_buffer_scope in self._double_buffer_map:
            self._double_buffer_map[reuse_buffer_scope].append(cache_buffer)
        else:
            self._double_buffer_map[reuse_buffer_scope] = [cache_buffer]

    def __split_tensor(self, tensor):
        """
        __split_tensor
        """
        tmp_op = {}
        tensor_op = tensor.op
        tmp_op["op"] = tensor_op.tag
        tmp_op["dst_buffer"] = tensor
        tmp_op["src_buffer"] = list(tensor_op.input_tensors)
        tmp_op["args"] = []
        tmp_op["effective_op"] = True
        if tmp_op["op"].find("elewise_single") != -1:
            if hasattr(tensor_op.body[0], 'b'):
                if isinstance(tensor_op.body[0].a, tvm.expr.Call):
                    tmp_op["args"] = [tensor_op.body[0].b]
                else:
                    tmp_op["args"] = [tensor_op.body[0].a]
        if tmp_op["op"].find("elewise_binary_scalar") != -1:
            if hasattr(tensor_op.body[0], 'a'):
                if isinstance(tensor_op.body[0].a, tvm.expr.Call):
                    if hasattr(tensor_op.body[0].b, 'a'):
                        if isinstance(tensor_op.body[0].b.a, tvm.expr.Call):
                            tmp_op["args"] = [tensor_op.body[0].b.b]
                        else:
                            tmp_op["args"] = [tensor_op.body[0].b.a]
                else:
                    if hasattr(tensor_op.body[0].a, 'a'):
                        if isinstance(tensor_op.body[0].a.a, tvm.expr.Call):
                            tmp_op["args"] = [tensor_op.body[0].a.b]
                        else:
                            tmp_op["args"] = [tensor_op.body[0].a.a]
        elif tmp_op["op"].find("broadcast") != -1:
            if tmp_op["op"] == "broadcast_for_tensor":
                if self._shape_to_list(tmp_op["src_buffer"][0].shape)[-1] != 1:
                    tmp_op["effective_op"] = False
            else:
                tmp_op["args"] = [tensor_op.body[0]]
        elif tmp_op["op"].find("reduce") != -1:
            if self._have_reduce:
                raise RuntimeError("Only support one time reduce")
            self._have_reduce = True
            tmp_op["reduce_axis"] = list(tensor_op.reduce_axis)

            reduce_axis_var = []
            for i in tensor_op.reduce_axis:
                reduce_axis_var.append(i.var)
            data_axis_var = tensor_op.body[0].source[0].args
            tmp_op["reduce_axis_num"] = []
            for ax_var in reduce_axis_var:
                axis_num = 0
                for i in data_axis_var:
                    if i.same_as(ax_var):
                        tmp_op["reduce_axis_num"].append(axis_num)
                    axis_num += 1

        if tmp_op["op"].find("elewise_binary_cmp") != -1 \
                or tmp_op["op"].find("elewise_binary_cmpsel") != -1 \
                or tmp_op["op"].find("elewise_binary_logic") != -1:
            str_list = tensor_op.tag.split("|")
            tmp_op["op"] = str_list[0]
            tmp_op["args"] = []
            for i in range(1, len(str_list)):
                tmp_op["args"].append(str_list[i])

        # split inputs sign and add into args for elewise_multiple op
        elif tmp_op["op"].find("elewise_multiple") != -1:
            str_list = tensor_op.tag.split("|")
            tmp_op["op"] = str_list[0]
            if len(str_list) >= 2:
                same_list_str = str_list[1].split(',')
                tmp_op["args"] = same_list_str

        if tmp_op["op"].find("|") != -1:
            str_list = tensor_op.tag.split("|")
            tmp_op["op"] = str_list[0]

        return tmp_op

    def __clean_orig_tensor(self):
        """
        __clean_orig_tensor
        """
        tmp_tensor = []
        for i in self._origin_tensor[::-1]:
            if_duplicate_tensor = False
            for j in tmp_tensor:
                if i.same_as(j):
                    if_duplicate_tensor = True
                    break
            if not if_duplicate_tensor:
                tmp_tensor.append(i)
        self._origin_tensor = tmp_tensor

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
            if in_tensor in visited_list:
                continue
            if in_tensor in self._spec_node_list \
                    or isinstance(in_tensor.op, tvm.tensor.PlaceholderOp):
                continue
            visited_list.append(in_tensor)
            self.__gen_reversed_subgraph_list(in_tensor, tensor_list,
                                              visited_list)
            tensor_list.append(in_tensor)

    def update(self, tensor):
        """
        record relate context imformations of operations
        """
        self._origin_tensor = [tensor]

        tensor_list = []
        visited_list = []
        self.__gen_reversed_subgraph_list(tensor, tensor_list, visited_list)
        tensor_list.append(tensor)

        for idx in reversed(tensor_list):
            have_reduce_before = self._have_reduce
            tmp_op = self.__split_tensor(idx)
            if (not have_reduce_before) and self._have_reduce:
                self._reduce_index.append(len(self._op))
            self._operaters[tmp_op["dst_buffer"]] = tmp_op["src_buffer"]
            for i in tmp_op["src_buffer"]:
                if isinstance(i.op, tvm.tensor.PlaceholderOp) or \
                        i in self._spec_node_list:
                    self.__cache_read(i, tmp_op["dst_buffer"])
                else:
                    if i not in self._res_tensor_list:
                        self.__inline_compute(i)
            if tmp_op["effective_op"]:
                self.__cache_write(idx)
                self._op.append(tmp_op)
            self._origin_op.append(tmp_op)

        self._origin_tensor += self._read_cache
        self._op = self._op[::-1]
        self._reduce_index = \
            [(len(self._op) - i - 1) for i in self._reduce_index]
        # the read_cache for placeholder
        self._read_cache = self._read_cache[::-1]
        self._write_cache = self._write_cache[::-1]

    # pylint: disable=no-self-use
    def _shape_to_list(self, shape):
        """
        translate tvm.shape to list type in python
        """
        tmp = []
        for i in shape:
            tmp.append(i.value)
        return tmp

    def __cache_read(self, data, res):
        """
        record cache read buffers
        """
        if self._read_cache_map.get(data):
            self._read_cache_map[data].append(res)
        else:
            self._read_cache_map[data] = [res]
            self._read_cache.append(data)

    def __cache_write(self, res):
        """
        record cache write buffers
        """
        self._write_cache.append(res)

    def __inline_compute(self, res):
        """
        record inline compute buffers
        """
        self._compute_inline.append(res)

    def is_strict_last_axis(self):
        """
        is_strict_last_axis
        """
        strict_op = ["reduce_sum", "reduce_prod", "reduce_max", "reduce_min"]

        if not self._is_elewise_single_and_broadcast and not \
                self._is_last_axis_boradcast:

            strict_op.append("broadcast_for_tensor")

        for i in self._origin_op:
            if i["op"] in strict_op:
                return True
        return False

    # pylint: disable=no-self-use, consider-using-enumerate
    def arg_sort(self, sort_list):
        """
        arg_sort
        """
        res_dict = {}
        index = []
        for i in range(len(sort_list)):
            res_dict[sort_list[i]] = i

        for i in sorted(res_dict.keys()):
            index.append(res_dict[i])

        return index

    # pylint: disable=no-self-use
    def reorder_list(self, list_need_reorder, index):
        """
        reorder_list
        """
        res_list = []
        for i in index:
            res_list.append(list_need_reorder[i])
        return res_list

    def _optimal_reduce_sum_4d_5d_schedule(self):
        # cache read and write
        read_buffer = self.local_cache_read()
        self.local_cache_write()

        # compute inline
        self.local_compute_inline()

        # tiling
        shape = self._shape_to_list(self._origin_op[0]["src_buffer"][0].shape)
        res_xo, res_xi = self._schedule[self._res_tensor].split(self._res_tensor.op.axis[0], \
                                                                factor=shape[0] // \
                                                                self.device_core_num)
        self.xouter.append(res_xo)
        self.xinner.append(res_xi)

        # compute_at
        self._reduce_index = self._reduce_index[0]
        self._need_compute_at_after = True
        self._compute_at_after_reduce_buffer = self._res_tensor
        self._compute_at_after_reduce_axis = self.xouter[0]
        self._compute_at_before_reduce_buffer = self._res_tensor
        self._compute_at_before_reduce_axis = self.xouter[0]
        self.local_compute_at(read_buffer)
        self._compute_at_after_reduce_axis = self.xinner[0]
        self._compute_at_before_reduce_axis = self.xinner[0]
        self.local_compute_at(read_buffer)

        # bind
        self._multi_core_bind_axis = self.xouter[0]
        block = tvm.thread_axis("blockIdx.x")
        self._schedule[self._res_tensor].bind(self._multi_core_bind_axis, block)
        cce_emitinsn_params.cceEmitParamsIns.insert_param("thread_block", block)
        cce_emitinsn_params.cceEmitParamsIns.insert_param("ub_max_enable_buff_size", \
                                                          self._ub_max_buff)

        # emit_insn include double_buffer
        if self.dsl_type == DSL_REDUCE_TYPE_MAP["single_reduce_sum_float32"]:
            self._schedule[self._res_tensor].emit_insn(self.xinner[0], \
                                                       "reduce_2_3_axis_reduce_sum_optimal")
        elif self.dsl_type == DSL_REDUCE_TYPE_MAP["cast_single_reduce_sum_4d"]:
            self._schedule[self._res_tensor].emit_insn( \
                self.xinner[0], "reduce_2_3_axis_reduce_sum_cast_4D_optimal")
        elif self.dsl_type == DSL_REDUCE_TYPE_MAP["cast_single_reduce_mean_4d"]:
            self._schedule[self._res_tensor].emit_insn( \
                self.xinner[0], "reduce_2_3_axis_reduce_mean_cast_4D_optimal")

    # pylint: disable=too-many-boolean-expressions
    def _check_optimal_condition(self):
        type_len_map = {"float16": 1,
                        "float32": 2}
        check_result = True

        # input type = fp32 && only a sum
        if len(self._origin_op) == 1:
            tmp_op = self._origin_op[0]
            if self._origin_op[0]["op"].find("reduce_sum") != -1 and \
                    self._origin_tensor[0].dtype == "float32":
                self.dsl_type = DSL_REDUCE_TYPE_MAP["single_reduce_sum_float32"]
            else:
                check_result = False
        # input type = fp16 && cast.fp32 -> sum -> cast.fp16
        elif len(self._origin_op) == 3:
            tmp_op = self._origin_op[1]
            if self._origin_op[0]["op"].find("elewise_single_cast") != -1 and \
                    self._origin_op[1]["op"].find("reduce_sum") != -1 and \
                    self._origin_op[2]["op"].find("elewise_single_cast") != -1 and \
                    self._origin_tensor[0].dtype == "float16":
                self.dsl_type = DSL_REDUCE_TYPE_MAP["cast_single_reduce_sum_4d"]
            else:
                check_result = False
        # reduce mean
        elif len(self._origin_op) == 2:
            tmp_op = self._origin_op[0]
            if self._origin_op[0]["op"].find("reduce_sum") != -1 and \
                    self._origin_op[1]["op"].find("elewise_single_VS_mul") != -1 and \
                    self._origin_tensor[0].dtype == "float16":
                check_result = False
                if len(self._origin_op[1]["args"]) == 1:
                    arg = self._origin_op[1]["args"][0]
                    if isinstance(arg, tvm.expr.FloatImm):
                        input_b = arg.value
                        cce_emitinsn_params.cceEmitParamsIns.insert_param("const_mul", input_b)
                        self.dsl_type = DSL_REDUCE_TYPE_MAP["cast_single_reduce_mean_4d"]
                        check_result = True
            else:
                check_result = False
        else:
            check_result = False

        if not check_result:
            return False

        dtype = self._origin_tensor[0].dtype
        nburst_limit = cce_params.VECTOR_COPY_NBURST_LIMIT
        block_width_fp16 = cce_params.VECTOR_SINGLE_BLOCK_WIDTH_FP16
        block_width = block_width_fp16 // type_len_map[dtype]
        # only enable optimal when size enough large
        min_size = 4 * block_width
        shape = self._shape_to_list(tmp_op["src_buffer"][0].shape)
        buffer_size = 0
        if len(shape) == 5:
            buffer_size = (shape[0] // self.device_core_num) * shape[1] * shape[3] * shape[4]
        elif len(shape) == 4:
            buffer_size = (shape[0] // self.device_core_num) * shape[1]
        else:
            return False
        # NCHW or NC1HWC0 and N % (block_dim) = 0: can enable all core
        if shape[0] % self.device_core_num != 0:
            return False

        verify_shape = shape[1] > 1 and shape[2] > 1 and shape[3] > 1
        verify_reduce_axis_num = tmp_op["reduce_axis_num"] == [2, 3]
        # 5HD: copy to ub limitation UB_MAX_ENABLE_BUFF, copy command limitation 4096
        # vector compute 256B align
        verify_5d = len(shape) == 5 and \
                    self.dsl_type == DSL_REDUCE_TYPE_MAP["single_reduce_sum_float32"] and \
                    dtype == "float32" and shape[4] == block_width_fp16
        verify_buff_limit_5d = (buffer_size * 3) * type_len_map[dtype] * 2 < self._ub_max_buff
        verify_vector_align_5d = (buffer_size // shape[3]) % ELEMENTS_VECTOR_OP_FP16 == 0
        verify_nburst_limit_5d = (shape[0] // self.device_core_num) * shape[1] < nburst_limit
        check_result = verify_5d and verify_buff_limit_5d and \
                       verify_vector_align_5d and verify_nburst_limit_5d and \
                       verify_shape and verify_reduce_axis_num
        # check 5d ok, return ture
        if check_result:
            return True

        # 4d: 32B align need: buffer_size % 16(8 for fp32) == 0
        verify_4d = len(shape) == 4 and \
                    (self.dsl_type == DSL_REDUCE_TYPE_MAP["cast_single_reduce_sum_4d"] or \
                     self.dsl_type == DSL_REDUCE_TYPE_MAP["cast_single_reduce_mean_4d"]) and \
                    dtype == "float16"
        verify_vector_align_4d = buffer_size % block_width == 0
        verify_buff_size_4d = buffer_size >= min_size
        verify_shape_4d = True
        m_stride = 2
        while ((shape[2] * shape[3] * m_stride) % block_width) != 0:
            m_stride *= 2
        if (buffer_size // m_stride) \
                < (shape[2] * shape[3] * type_len_map[dtype] // ELEMENTS_VECTOR_OP_FP16):
            verify_shape_4d = False
        check_result = verify_4d and verify_vector_align_4d and \
                       verify_buff_size_4d and verify_shape_4d and \
                       verify_shape and verify_reduce_axis_num

        return check_result
