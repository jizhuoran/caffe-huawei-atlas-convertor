#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright 2018 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

concat schedule:
Optimizing schedule for concat operation.
"""
import math
from functools import reduce as functools_reduce

from te import platform as cceconf
from te import tvm


def _get_align_factor(dtype):
    # base on the diff data type, get the align_factor
    align_factor = 16
    dtype_bytes = 1
    if dtype in ("int8", "uint8"):
        align_factor = 32
        dtype_bytes = 1
    elif dtype in ("float16", "int16", "uint16"):
        align_factor = 16
        dtype_bytes = 2
    elif dtype in ("float32", "int32", "uint32"):
        align_factor = 8
        dtype_bytes = 4
    elif dtype in ("int64", "uint64"):
        align_factor = 4
        dtype_bytes = 8

    return align_factor, dtype_bytes


# pylint: disable=locally-disabled,unused-argument,too-many-locals,too-many-branches,too-many-statements,too-many-instance-attributes
class CceConcatOp():
    """
    class of cce index op

    Parameters
    ----------
    scope : cal buffer like local.UB

    need_tensorize : if need to doing tensorize when using calculate

    need_pragma : if need to doing paagma when using calculate

    Returns
    -------
    CceOp_instance : instance of CceOp

    """

    def __init__(self, scope, need_tensorize=True, need_pragma=True):
        # record data ddr levels, only equals to -1 or 0 in now stage
        self._need_tensorize = need_tensorize
        self._need_pragma = need_pragma
        self._op_list = []
        self._read_buffer = []
        self._write_buffer = []
        self._scope = scope
        self._segment_op_pos = 0

        self._align_factor = 0
        self._dtype_bytes = 0
        self._shapes = [[]]
        self._schedule = None
        self._concat_axis = None

        self._spec_node_list = []
        if self._scope.lower().find('.ub') != -1:
            self._total_size = cceconf.get_soc_spec("UB_SIZE")
        else:
            raise RuntimeError("only support UB buffer now")

    def schedule_for_two_dims(self, res, spec_node_list, sch_list, min_axis):
        """
        the schedule for two dims situation
        """
        self._schedule = sch_list[0]
        input_tensors = res.op.input_tensors
        factors = []
        input_shapes = []
        input_ub = []
        equal_flag = True

        if input_tensors[0].shape[0].value != input_tensors[1].shape[0].value:
            equal_flag = False

        cores = []
        core_num = 1

        dim_num = input_tensors[0].shape[0].value
        if dim_num == 1000:
            bind_cores = 20
        else:
            bind_cores = 32

        if equal_flag:
            for cur_tensor in input_tensors:
                cur_shape = cur_tensor.shape
                input_shapes.append(cur_tensor.shape)
                cores.append(bind_cores)
                cur_factor = cur_shape[0] // bind_cores
                factors.append(cur_factor)
                input_ub.append(
                    self._schedule.cache_read(cur_tensor, self._scope, [res]))
        else:
            min_axis = 720
            for cur_tensor in input_tensors:
                cur_shape = cur_tensor.shape
                input_shapes.append(cur_tensor.shape)
                cur_factor = core_num * (cur_shape[0] // min_axis)
                cores.append(cur_factor)
                cur_factor = cur_shape[0] // cur_factor
                factors.append(cur_factor)
                input_ub.append(
                    self._schedule.cache_read(cur_tensor, self._scope, [res]))

        res_factor = 0
        for c_factor in cores:
            res_factor = res_factor + c_factor

        res_factor = res.shape[0] // res_factor
        axis_one, axis_two = self._schedule[res].split(res.op.axis[0],
                                                       factor=res_factor)

        i = 0
        for cur_tensor in input_ub:
            cur_axis = self._schedule[cur_tensor].split(cur_tensor.op.axis[0],
                                                        factor=factors[i])
            self._schedule[cur_tensor].compute_at(self._schedule[res], axis_one)
            self._schedule[cur_tensor].emit_insn(cur_axis[1], cceconf.dma_copy)
            i = i + 1

        self._schedule[res].emit_insn(axis_two, cceconf.dma_copy)
        self._schedule[res].bind(axis_one, tvm.thread_axis('blockIdx.x'))
        sch_list[0] = self._schedule

        return True

    def check_optimization_rule(self, res):
        """
        check rules for two dims
        """
        self._concat_axis = self._concat_axis
        input_tensors = res.op.input_tensors
        output_shape = res.shape

        concat_axis = [184320, 46080, 11520, 2880, 720]
        concat_axis_two = [184320, 11520]
        concat_axis_one = [184320, 46080]
        check_one = False
        check_two = 0
        if len(input_tensors) == 2:
            tensor_1 = input_tensors[0].shape
            tensor_2 = input_tensors[1].shape

            if len(output_shape) == 2:
                if tensor_1[0].value not in concat_axis_two:
                    check_one = False
                elif tensor_2[0].value != tensor_1[0].value:
                    check_one = False
                elif tensor_1[1].value != 4:
                    check_one = False
                elif tensor_2[1].value != 4:
                    check_one = False
                elif output_shape[1].value != 4:
                    check_one = False
                else:
                    check_one = True
            elif len(output_shape) == 1:
                if tensor_1[0].value not in concat_axis_one:
                    check_one = False
                elif tensor_2[0].value != tensor_1[0].value:
                    check_one = False
                else:
                    check_one = True
            else:
                check_one = False
        elif len(input_tensors) == 5:
            num = 0
            if len(output_shape) != 2:
                check_one = False
            elif output_shape[1].value != 4:
                check_one = False
            else:
                flag = 1
                for cur_compute in input_tensors:
                    cur_shape = cur_compute.shape
                    if cur_shape[0].value != concat_axis[num]:
                        check_one = False
                        flag = 0
                        break
                    if cur_shape[1].value != 4:
                        check_one = False
                        flag = 0
                        break
                    num = num + 1
                if flag == 1:
                    check_one = True
        else:
            check_one = False
        if len(input_tensors) == 80:
            sp_flag = True
            for sp_tensor in input_tensors:
                if sp_tensor.shape[0].value != 1000:
                    sp_flag = False
                    break
            if sp_flag and len(input_tensors[0].shape) == 1:
                check_one = True

        return check_one, check_two

    def schedule(self, res, spec_node_list, sch_list):
        """
        the schedule processes of concat

        Parameters
        ----------
        res: placeholder
            the placeholder of result
        spec_node_list: list
            reverse for spec node list
        sch_list: list
            schedule objects list

        Returns
        -------
        Ture: bool
            the result of schedule
        """
        result, min_axis = self.check_optimization_rule(res)
        if result:
            self.schedule_for_two_dims(res, spec_node_list, sch_list, min_axis)
            return True

        self._schedule = sch_list[0]
        self._spec_node_list = spec_node_list
        self._align_factor, self._dtype_bytes = _get_align_factor(res.dtype)
        self._get_input_tensors(res)

        write_cache_list = [self._schedule.cache_write(res, self._scope)]

        read_cache_list = []
        for in_tensor in self._read_buffer:
            read_cache_list.append(
                self._schedule.cache_read(in_tensor, self._scope,
                                          write_cache_list))

        is_db_buffer = True
        for i in self._shapes[1:len(self._shapes) - 1]:
            if i[self._concat_axis] != self._shapes[0][self._concat_axis]:
                is_db_buffer = False

        split_axis, ub_split_axis, split_factor, bind_core_flag \
            = self.concat_op_tilling(is_db_buffer)

        comput_at_axis = self._schedule[res].op.axis[self._concat_axis]
        res_emit_axis = self._schedule[res].op.axis[self._concat_axis]
        bind_axis = self._schedule[res].op.axis[0]
        fuse_flag = True
        if self._concat_axis < 2 or split_axis < 2:
            fuse_flag = False

        if split_factor != 0:
            if (ub_split_axis + 1 < split_axis and is_db_buffer is True and
                    bind_core_flag is True):
                reorder_index = ub_split_axis + 1
                if reorder_index == 1:
                    fuse_flag = False

                self._schedule[res].reorder(
                    self._schedule[res].op.axis[split_axis],
                    self._schedule[res].op.axis[reorder_index])
                for cache_tensor in read_cache_list:
                    self._schedule[cache_tensor].reorder(
                        self._schedule[cache_tensor].op.axis[split_axis],
                        self._schedule[cache_tensor].op.axis[reorder_index])

            split_outer, split_inner = self._schedule[res].split(
                self._schedule[res].op.axis[split_axis], split_factor)
            comput_at_axis = split_outer
            if split_axis == 0:
                out_loop = self._shapes[-1][0] // split_factor
                bind_axis = split_outer
                if out_loop > 64:
                    split_factor_new = math.ceil(out_loop / 64)
                    split_outer, split_inner_new = self._schedule[res].split(
                        split_outer, split_factor_new)
                    bind_axis = split_outer
                    comput_at_axis = split_inner_new

            if split_axis >= self._concat_axis:
                res_emit_axis = split_inner

            for cache_tensor in read_cache_list:
                self._schedule[cache_tensor].compute_at(self._schedule[res],
                                                        comput_at_axis)
        else:
            self._schedule[res].reorder(self._schedule[res].op.axis[split_axis],
                                        self._schedule[res].op.axis[0])
            for cache_tensor in read_cache_list:
                self._schedule[cache_tensor].reorder(
                    self._schedule[cache_tensor].op.axis[split_axis],
                    self._schedule[cache_tensor].op.axis[0])

        for cache_tensor in write_cache_list:
            self._schedule[cache_tensor].storage_align(comput_at_axis,
                                                       self._align_factor, 0)

        if is_db_buffer:
            for cache_tensor in read_cache_list:
                self._schedule[cache_tensor].double_buffer()

        for cache_tensor in read_cache_list:
            if split_axis == 0 and split_factor == 0:
                self._schedule[cache_tensor].emit_insn(
                    cache_tensor.op.axis[self._concat_axis], cceconf.dma_copy)
            else:
                _, cahe_inner = self._schedule[cache_tensor].split(
                    self._schedule[cache_tensor].op.axis[split_axis],
                    split_factor)
                self._schedule[cache_tensor].storage_align(
                    cahe_inner, self._align_factor, 0)
                self._schedule[cache_tensor].emit_insn(cahe_inner,
                                                       cceconf.dma_copy)

        for cache_tensor in write_cache_list:
            self._schedule[cache_tensor].compute_inline()
        self._schedule[res].emit_insn(res_emit_axis, cceconf.dma_copy)
        sch_list[0] = self._schedule

        if self._shapes[-1][0] > 32 and split_axis != 0:
            factor = math.ceil(self._shapes[-1][0] / 32)
            bind_outer, _ = self._schedule[res].split(
                self._schedule[res].op.axis[0], factor)
            bind_axis = bind_outer
        elif split_axis > 1 and self._shapes[-1][0] < 32 and fuse_flag:
            bind_axis = self._schedule[res].fuse(self._schedule[res].op.axis[0],
                                                 self._schedule[res].op.axis[1])
            if self._shapes[-1][0] * self._shapes[-1][1] > 32:
                bind_axis, _ = self._schedule[res].split(bind_axis, 32)

        if bind_core_flag:
            self._schedule[res].bind(bind_axis, tvm.thread_axis('blockIdx.x'))

        return True

    def _get_input_tensors(self, res):
        write_buffer = [res]
        read_buffer = []
        shapes = []
        for in_tensor in list(res.op.input_tensors):
            read_buffer.append(in_tensor)
            shape = [
                int(in_tensor.shape[i].value)
                for i in range(len(in_tensor.shape))
            ]
            shapes.append(shape)
        self._read_buffer = list(read_buffer)
        self._write_buffer = list(write_buffer)

        shape = [int(res.shape[i].value) for i in range(len(res.shape))]
        shapes.append(shape)
        self._shapes = list(shapes)

        # get concat axis
        self._concat_axis = 0
        for i in range(len(self._shapes[-1])):
            if self._shapes[-1][i] != self._shapes[0][i]:
                self._concat_axis = i
                break

    def concat_op_tilling(self, is_db_buffer):
        """
        calculate compute at axis and split factor

        input
        -----
            is_db_buffer: if true enable double buffer
        output
        ------
            split_axis: int
                the axis index used for split
            ub_split_axis: int
                the axis index used for reorder
            split_factor: int
                the factor used for split
            bind_core_flag: bool
                the flag for bind multi aicores
        """

        def _get_maxinum_common_divisor(left, right):
            # get maximum common divisor
            left, right = (left, right) if left >= right else (right, left)
            while right:
                left, right = right, left % right
            return left

        if is_db_buffer:
            storage_factor = 2
        else:
            storage_factor = 1

        ub_size = cceconf.get_soc_spec("UB_SIZE")
        ub_size_align = ((ub_size // self._align_factor) * self._align_factor //
                         self._dtype_bytes // storage_factor)

        res_shape = self._shapes[-1]
        split_axis = len(res_shape)
        split_factor = 0
        num_inputs = len(self._shapes) - 1

        # calculate compute_at axis according to UB size
        for k in range(0, len(res_shape)):
            axis_size = functools_reduce(lambda i, j: i * j, res_shape[k:])

            if axis_size <= ub_size_align:
                split_axis = k - 1
                split_factor = ub_size_align // axis_size
                break

        if split_axis < 0:
            ub_split_axis = 0
        else:
            ub_split_axis = split_axis

        input_dim_nums = len(res_shape)
        if split_axis == input_dim_nums and axis_size > ub_size_align:
            split_axis = len(res_shape) - 1
            split_factor = ub_size_align // 2 // num_inputs

        bind_core_flag = False
        input_axis = [
            self._shapes[i][self._concat_axis] for i in range(num_inputs)
        ]

        # if not need to split
        if split_axis == input_dim_nums or split_axis < 0:
            # the concat axis must be split for emit_insn
            split_factor = input_axis[0]
            if num_inputs == 2:
                split_factor = input_axis[0]
            else:
                for i in input_axis[1:]:
                    split_factor = _get_maxinum_common_divisor(i, split_factor)
            split_axis = self._concat_axis
        else:
            if split_axis > self._concat_axis:
                if (axis_size / num_inputs) % self._align_factor == 0 \
                        and input_dim_nums > 0:
                    bind_core_flag = True
            elif split_axis == self._concat_axis:
                split_factor = 1
                if split_axis == len(res_shape) - 1:
                    concat_size = self._shapes[0][-1]
                else:
                    concat_size = functools_reduce(lambda i, j: i * j,
                                                   res_shape[split_axis + 1:])
                if concat_size % self._align_factor == 0:
                    bind_core_flag = True
            else:
                split_axis = self._concat_axis
                split_factor = input_axis[0]

                if num_inputs > 2:
                    for i in input_axis[1:]:
                        split_factor = _get_maxinum_common_divisor(
                            i, split_factor)

                concat_size = functools_reduce(lambda i, j: i * j,
                                               res_shape[self._concat_axis:])

                if split_factor % self._align_factor == 0:
                    bind_core_flag = True

            if split_axis == 0:
                if concat_size % self._align_factor == 0:
                    bind_core_flag = True
                elif split_axis == self._concat_axis:
                    bind_core_flag = True
                else:
                    bind_core_flag = False

        return split_axis, ub_split_axis, split_factor, bind_core_flag
