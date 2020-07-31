#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

concat_tik
"""
import math
import numpy as np
from te import tik
from te import platform as tbe_platform


# pylint: disable=too-many-instance-attributes,unused-argument
# pylint: disable=too-many-statements,too-many-lines,too-many-locals
# pylint: disable=too-many-branches,too-many-return-statements
class ConcatSchedule:
    """
        Function: use to store concat base parameters
        Modify : 2020-3-25
    """

    def __init__(self,
                 input_data,
                 output_data,
                 concat_axis,
                 kernel_name="concat"):
        """
        Init concat base parameters

        Parameters
        ----------
        input_data: list, tuple
            data of input
        output_data: dict
            data of output
        concat_axis: int
            axis for concat
        kernel_name: str
            the name of the operator

        Returns
        -------
        None
        """
        self.tik_instance = tik.Tik()
        self.aicore_num = \
            tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)

        self.dtype = input_data[0].get("dtype").lower()
        self.output_shape = output_data.get("shape")

        self.concat_axis = concat_axis % len(self.output_shape)

        self.input_shapes, self.concat_axis, self.output_shape = \
            self.reshape_simple(input_data, concat_axis)

        self.input_tensors, self.output_tensor = self.init_gm_tensor(
            self.input_shapes, self.output_shape, self.dtype)

        dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(self.dtype) // 8
        self.ele_each_block = 32 // dtype_bytes_size

        self.ub_half_size = \
            (tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
             // dtype_bytes_size // 2 - self.ele_each_block)

        self.max_dims = 0
        self.max_input_dim_size = 0
        self.kernel_name = kernel_name
        self.is_remainder_half = False
        self.is_all_align = False
        self.is_use_scalar = False
        self.is_vector_branches = False
        self.is_special_shape = False
        # for vector mask scalar list, the max len is 16
        self.pre_mask_scalar_list = []
        self.post_mask_scalar_list = []

    def reshape_simple(self, shape_list, concat_axis):
        """
        Init concat base parameters

        Parameters
        ----------
        shape_list: list
            input shapes
        concat_axis: int
            axis for concat

        Returns
        -------
        input_shapes: list
            input shapes
        concat_axis: int
            axis for concat
        """
        input_shapes = []
        if concat_axis == 0:
            for _, input_dict in enumerate(shape_list):
                shape_input = input_dict.get("shape")
                shape_input = (int(np.prod(shape_input)),)
                input_shapes.append(shape_input)
            self.output_shape = (int(np.prod(self.output_shape)),)
            concat_axis = 0
        else:
            for _, input_dict in enumerate(shape_list):
                shape_input = input_dict.get("shape")
                out_dim = int(np.prod(shape_input[0:concat_axis]))
                if out_dim == 1:
                    shape_input = (int(np.prod(shape_input)),)
                else:
                    inner_dim = int(np.prod(shape_input[concat_axis:]))
                    shape_input = (out_dim, inner_dim)
                input_shapes.append(shape_input)

            if len(input_shapes[0]) == 2:
                out_dim = int(np.prod(self.output_shape[0:concat_axis]))
                inner_dim = int(np.prod(self.output_shape[concat_axis:]))
                self.output_shape = (out_dim, inner_dim)
                concat_axis = 1
            else:
                self.output_shape = (int(np.prod(self.output_shape)),)
                concat_axis = 0

        # calcu the output shape again
        out_shape = list(input_shapes[0]).copy()
        out_shape[concat_axis] = 0
        for _, input_shape in enumerate(input_shapes):
            out_shape[concat_axis] = \
                out_shape[concat_axis] + input_shape[concat_axis]

        return input_shapes, concat_axis, out_shape

    # pylint: disable=too-many-branches
    def check_tik_supported(self):
        """
        check if tik schedule support this shape

        Returns
        -------
        if_supported: bool
            if tik schedule support this shape
        """
        # get flag whether the concat size mod one block
        is_all_align = check_dim_size_align(self.input_shapes,
                                            self.concat_axis,
                                            self.ele_each_block)
        self.is_all_align = is_all_align

        # get flag whether the concat size mod half block
        self.is_remainder_half = \
            check_dim_size_align(self.input_shapes,
                                 self.concat_axis,
                                 self.ele_each_block // 2)
        if is_all_align:
            self.is_remainder_half = False

        # vector_branch check begin
        max_dims, _, _ = self.get_max_dims_remainder_half()
        use_vector_branch, dtype_support = check_use_vector_branch(
            self.input_shapes, self.dtype, self.output_shape,
            self.concat_axis, self.ele_each_block)

        if self.is_remainder_half and dtype_support:
            use_vector_branch = True

        if max_dims == 0:
            use_vector_branch = False
        # vector_branch check end

        # special shape check begin
        if len(self.input_shapes) == 2 and len(self.input_shapes[1]) == 2:
            if [self.input_shapes[0][1], self.input_shapes[1][1]] == [3120, 1]:
                self.is_special_shape = True
                return True
        # special shape check end

        # scalar_branch check begin
        use_scalar = check_use_scalar(
            self.input_shapes, self.concat_axis, self.ele_each_block,
            self.output_shape, self.get_ub_size_scalar())
        # scalar_branch check end

        # inputs mod half block can go use_vector_branch
        if self.is_remainder_half and use_vector_branch:
            self.is_vector_branches = True
            return True
        # inputs mod one block can go use_vector_branch
        if is_all_align and use_vector_branch:
            self.is_vector_branches = True
            return True

        # when use_scalar is true and concat size more than one block
        # can go use_vector_branch
        if use_scalar and use_vector_branch \
                and self.output_shape[self.concat_axis] > self.ele_each_block:
            self.is_vector_branches = True
            return True

        if use_scalar and (not is_all_align):
            self.is_use_scalar = True
            return True

        # check for data_move
        if_supported = True
        the_max_size = 0
        for _, input_shape in enumerate(self.input_shapes):
            concat_axis_len = input_shape[self.concat_axis]
            the_max_size = max(the_max_size, concat_axis_len)
            if concat_axis_len < self.ele_each_block:
                if_supported = False

        if len(self.output_shape) == 2 and is_all_align \
                and the_max_size // self.ele_each_block \
                // self.aicore_num < 100 and self.output_shape[0] > 128:
            if_supported = False

        if len(self.output_shape) == 1 and self.output_shape[0] < 4096:
            return False

        return if_supported

    def init_gm_tensor(self, input_shapes, output_shape, dtype):
        """
        init gm tensor

        Parameters
        ----------
        input_shapes: list
            shape of input tensors
        output_shape: list
            shape of output tensor
        dtype: str
            data type

        Returns
        -------
        input_tensors: tik tensor
            input gm tensor
        output_tensor: tik tensor
            output gm tensor
        """
        input_tensors = []
        for index, tensor_shape in enumerate(input_shapes):
            tensor_name = "gm_input_" + str(index)
            gm_tensor = self.tik_instance.Tensor(
                dtype, tensor_shape, name=tensor_name, scope=tik.scope_gm)
            input_tensors.append(gm_tensor)

        output_tensor = self.tik_instance.Tensor(
            dtype, output_shape, name="gm_output", scope=tik.scope_gm)

        return input_tensors, output_tensor

    def gen_vector_mask_scalar(self):
        """
        gen vector mask scalar for vcetor vadd

        ex:
        for fp32, gen 16 maske scalar:
            mask:
              0000000000000000000000000000000000000000000000000000000000000000
            scalar value: 0
            mask:
              1000000010000000100000001000000010000000100000001000000010000000
            scalar value: 9259542123273814144
            ......
            mask:
              1111111011111110111111101111111011111110111111101111111011111110
            scalar value: 18374403900871474942

            mask:
              0000000000000000000000000000000000000000000000000000000000000000
            scalar value: 0
            mask:
              0000000100000001000000010000000100000001000000010000000100000001
            scalar value: 72340172838076673
            ......
            mask:
              0111111101111111011111110111111101111111011111110111111101111111
            scalar value: 9187201950435737471
        """
        # calcu all mask
        for i in range(self.ele_each_block):
            mask0_scalar = self.tik_instance.Scalar(dtype="uint64")
            mask0_scalar.set_as(
                gen_vector_mask(i,
                                self.ele_each_block,
                                mask_mode="PRE"))
            self.pre_mask_scalar_list.append(mask0_scalar)
        for i in range(self.ele_each_block):
            mask0_scalar = self.tik_instance.Scalar(dtype="uint64")
            mask0_scalar.set_as(
                gen_vector_mask(i,
                                self.ele_each_block,
                                mask_mode="POST"))
            self.post_mask_scalar_list.append(mask0_scalar)

    def get_loop_para(self, input_shape, concat_axis):
        """
        Get concat loop parameters

        Parameters
        ----------
        input_shape: list
            input shapes
        concat_axis: int
            axis for concat

        Returns
        -------
        loop_num: int
            number of loop
        ele_num_each_loop: int
            element number for each loop
        ele_last: int
            element number of last loop
        """
        total_ele_num = input_shape[concat_axis]
        ele_last = total_ele_num % self.aicore_num

        if ele_last == 0:
            ele_num_each_loop = total_ele_num // self.aicore_num
            use_core_num = self.aicore_num
        else:
            ele_num_each_loop = get_ceil_int(total_ele_num, self.aicore_num)
            ele_last = total_ele_num % ele_num_each_loop
            use_core_num = (total_ele_num // ele_num_each_loop) + 1

        return use_core_num, ele_num_each_loop, ele_last

    # pylint: disable=too-many-locals,too-many-statements
    def concat_compute_for_each_tensor(self, tensor_list, input_tensor_info,
                                       output_tensor_info, ele_num):
        """
        concat each tensor

        Parameters
        ----------
        tensor_list: list
            ub tensor for data move in
        input_tensor_info: list
            input gm tensor, offset
        output_tensor_info: list
            input gm tensor, offset
        ele_num: int
            element number

        Returns
        -------
        None
        """
        input_tensor, move_in_index = input_tensor_info
        output_tensor, move_out_index = output_tensor_info
        loop_burst_len = 0
        if ele_num < self.ub_half_size:
            loop_num = 0
            last_ele = ele_num
            ele_each_loop = 0
        else:
            if ele_num % self.ub_half_size < self.ele_each_block:
                ub_size = self.ub_half_size - self.ele_each_block
            else:
                ub_size = self.ub_half_size
            last_ele = ele_num % ub_size
            if last_ele == 0:
                loop_num = ele_num // ub_size
                ele_each_loop = ub_size
                loop_burst_len = ub_size // self.ele_each_block
            else:
                loop_num, ele_each_loop = cal_loop(ele_num, ub_size)
                loop_burst_len = ele_each_loop // self.ele_each_block
                if ele_each_loop % self.ele_each_block != 0:
                    loop_burst_len = loop_burst_len + 1
                last_ele = ele_num % ele_each_loop

        ping_pang_flag = 0
        loop_burst_len = int(loop_burst_len)
        if loop_num > 0:
            with self.tik_instance.for_range(
                    0, loop_num // 2) as inner_loop:
                ub_tensor = tensor_list[0]
                offset = inner_loop * 2 * ele_each_loop
                self.tik_instance.data_move(ub_tensor,
                                            input_tensor[move_in_index
                                                         + offset],
                                            0, 1, loop_burst_len, 0, 0)
                self.tik_instance.data_move(
                    output_tensor[move_out_index + offset], ub_tensor,
                    0, 1, loop_burst_len, 0, 0)
                ub_tensor = tensor_list[1]
                offset = (inner_loop * 2 + 1) * ele_each_loop
                self.tik_instance.data_move(ub_tensor,
                                            input_tensor[move_in_index
                                                         + offset],
                                            0, 1, loop_burst_len, 0, 0)
                self.tik_instance.data_move(
                    output_tensor[move_out_index + offset], ub_tensor,
                    0, 1, loop_burst_len, 0, 0)
            if loop_num % 2 == 1:
                offset = (loop_num - 1) * ele_each_loop
                ub_tensor = tensor_list[ping_pang_flag]
                self.tik_instance.data_move(ub_tensor,
                                            input_tensor[move_in_index
                                                         + offset],
                                            0, 1, loop_burst_len, 0, 0)
                self.tik_instance.data_move(
                    output_tensor[move_out_index + offset], ub_tensor,
                    0, 1, loop_burst_len, 0, 0)
                ping_pang_flag = (ping_pang_flag + 1) % 2

        if last_ele > 0:
            offset = loop_num * ele_each_loop
            loop_burst_len = last_ele // self.ele_each_block
            ub_tensor = tensor_list[ping_pang_flag]
            if loop_burst_len > 0:
                self.tik_instance.data_move(ub_tensor,
                                            input_tensor[move_in_index
                                                         + offset],
                                            0, 1, loop_burst_len, 0, 0)
                self.tik_instance.data_move(
                    output_tensor[move_out_index + offset], ub_tensor,
                    0, 1, loop_burst_len, 0, 0)
                ping_pang_flag = (ping_pang_flag + 1) % 2

            if last_ele % self.ele_each_block != 0:
                ub_tensor = tensor_list[ping_pang_flag]
                offset = ele_num - self.ele_each_block
                self.tik_instance.data_move(ub_tensor,
                                            input_tensor[move_in_index
                                                         + offset],
                                            0, 1, 1, 0, 0)
                self.tik_instance.data_move(
                    output_tensor[move_out_index + offset], ub_tensor,
                    0, 1, 1, 0, 0)

    def concat_compute_each_core(self, core_index):
        """
        concat input tensor on each core

        Parameters
        ----------
        core_index: int
            aicore index

        Returns
        -------
        None
        """
        if len(self.input_shapes[0]) > 1:
            out_loop = self.input_shapes[0][0]
        else:
            out_loop = 1

        # init ub for double buff
        ub_tensor = self.tik_instance.Tensor(
            self.dtype, (self.ub_half_size,),
            name="ub_tensor",
            scope=tik.scope_ubuf)
        ub_tensor_1 = self.tik_instance.Tensor(
            self.dtype, (self.ub_half_size,),
            name="ub_tensor_1",
            scope=tik.scope_ubuf)

        ub_list = [ub_tensor, ub_tensor_1]

        # define fuc for one ele_num
        def _run_one_core(out_loop_idx, _out_offset,
                          ele_num_each_core, ele_num_process,
                          input_idx):
            input_idx_shape = self.input_shapes[input_idx]
            move_in_idx = \
                (ele_num_each_core * core_index
                 + out_loop_idx * input_idx_shape[self.concat_axis])
            move_out_idx = \
                (_out_offset + ele_num_each_core * core_index
                 + out_loop_idx * self.output_shape[self.concat_axis])
            self.concat_compute_for_each_tensor(
                ub_list,
                [self.input_tensors[input_idx], move_in_idx],
                [self.output_tensor, move_out_idx], ele_num_process)

        one_core_flag = False
        tensor_index = 0
        ele_num_each_loop = 0
        # do concat the input to output one by one
        out_offset = 0
        for tensor_index, input_shape in enumerate(self.input_shapes):
            # get ele number of one core for one input
            use_core_num, ele_num_each_loop, ele_last = \
                self.get_loop_para(input_shape, self.concat_axis)
            # when ele_num in core is less than ele_each_block
            # all core process the same data
            one_core_flag = False
            if ele_num_each_loop < self.ele_each_block or \
                    0 < ele_last < self.ele_each_block or \
                    use_core_num != self.aicore_num:
                one_core_flag = True
                ele_last = 0
                ele_num_each_loop = input_shape[self.concat_axis]

            def _run_one_input_double_buff(ele_num):
                with self.tik_instance.for_range(
                        0, out_loop // 2) as loop_index:
                    one_core_idx = loop_index*2
                    if one_core_flag:
                        _run_one_core(one_core_idx, out_offset,
                                      0, ele_num,
                                      tensor_index)
                    else:
                        _run_one_core(one_core_idx, out_offset,
                                      ele_num_each_loop, ele_num,
                                      tensor_index)
                    one_core_idx = loop_index*2 + 1
                    ub_list.reverse()
                    if one_core_flag:
                        _run_one_core(one_core_idx, out_offset,
                                      0, ele_num,
                                      tensor_index)
                    else:
                        _run_one_core(one_core_idx, out_offset,
                                      ele_num_each_loop, ele_num,
                                      tensor_index)
                    ub_list.reverse()
                if out_loop % 2 == 1:
                    one_core_idx = out_loop - 1
                    if one_core_flag:
                        _run_one_core(one_core_idx, out_offset,
                                      0, ele_num,
                                      tensor_index)
                    else:
                        _run_one_core(one_core_idx, out_offset,
                                      ele_num_each_loop, ele_num,
                                      tensor_index)
                    ub_list.reverse()

            if ele_last == 0:
                _run_one_input_double_buff(ele_num_each_loop)
            else:
                with self.tik_instance.if_scope(
                        core_index < self.aicore_num - 1):
                    _run_one_input_double_buff(ele_num_each_loop)
                with self.tik_instance.else_scope():
                    _run_one_input_double_buff(ele_last)

            out_offset += input_shape[self.concat_axis]

    def get_loop_para_scalar(self, input_shapes):
        """
        Get concat loop parameters

        Parameters
        ----------
        input_shapes: list
            input shapes

        Returns
        -------
        loop_num: int
            number of loop
        last_loop_num: int
            number of last loop
        core_used : int
            user core num
        """
        # get min loop because one core process one block at least
        if self.output_shape[self.concat_axis] < self.ele_each_block:
            min_loop = get_ceil_int(self.ele_each_block,
                                    self.output_shape[self.concat_axis])
        else:
            min_loop = 1
        out_loop = input_shapes[0][0]
        loop_num = out_loop // self.aicore_num
        if loop_num < min_loop:
            loop_num = min_loop
        last_loop_num = out_loop % loop_num
        # calcu the core need
        core_used = get_ceil_int(out_loop, loop_num)

        return loop_num, last_loop_num, core_used

    def concat_last_dim_for_scalar(self):
        """copy all data from src to des
        """
        loop_num, last_loop_num, core_used = \
            self.get_loop_para_scalar(self.input_shapes)

        # when first dim is small, will not active all aicore
        self.aicore_num = core_used

        if last_loop_num == 0:
            with self.tik_instance.for_range(
                    0, self.aicore_num,
                    block_num=self.aicore_num) as index:
                in_offset = index * loop_num
                out_offset = \
                    (index * loop_num * self.output_shape[self.concat_axis])
                compute_loop = loop_num
                self.concat_compute_each_core_scalar(
                    in_offset, out_offset, compute_loop)
        else:
            with self.tik_instance.for_range(
                    0, self.aicore_num,
                    block_num=self.aicore_num) as index:
                with self.tik_instance.if_scope(
                        index < self.aicore_num - 1):
                    in_offset = index * loop_num
                    out_offset = \
                        (index * loop_num
                         * self.output_shape[self.concat_axis])
                    compute_loop = loop_num
                    self.concat_compute_each_core_scalar(
                        in_offset, out_offset, compute_loop)
                with self.tik_instance.else_scope():
                    in_offset = (self.aicore_num - 1) * loop_num
                    out_offset = ((self.aicore_num - 1) * loop_num *
                                  self.output_shape[self.concat_axis])
                    compute_loop = last_loop_num
                    self.concat_compute_each_core_scalar(
                        in_offset, out_offset, compute_loop)

    # pylint: disable=too-many-arguments
    def concat_each_tensor_loop_scalar(self, ub_input, input_tensor,
                                       output_tensor, compute_loop,
                                       last_axis_len):
        """
        Get concat loop parameters

        Parameters
        ----------
        ub_input: tik tensor
            ub tensor of input
        input_tensor: tik tensor
            gm tensor of input
        output_tensor: tik tensor
            gm tensor of output
        compute_loop: Expr
            loop number for compute
        last_axis_len: Expr
            the length of last axis

        Returns
        -------
        None
        """
        burst_len = math.ceil(
            float(compute_loop * last_axis_len) / float(self.ele_each_block))

        self.tik_instance.data_move(ub_input, input_tensor, 0,
                                    1, burst_len, 0, 0)

        with self.tik_instance.for_range(0, compute_loop) as index_0:
            with self.tik_instance.for_range(0, last_axis_len) as index_1:
                src_offset = index_0 * last_axis_len + index_1
                dst_offset = index_0 * self.output_shape[1] + index_1
                output_tensor[dst_offset] = ub_input[src_offset]

    def concat_each_tensor_scalar(self, input_offset, output_tensor,
                                  output_offset,
                                  compute_loop):
        """Get concat loop parameters
        """
        output_size = self.get_ub_size_scalar()
        ub_output = self.tik_instance.Tensor(
            self.dtype, (output_size,), name="ub_output",
            scope=tik.scope_ubuf)

        tensor_offset = 0
        ub_input = self.tik_instance.Tensor(
            self.dtype, (output_size,), name="ub_input", scope=tik.scope_ubuf)
        for tensor_index, _ in enumerate(self.input_shapes):
            last_axis_len = self.input_shapes[tensor_index][1]
            read_offset = input_offset * last_axis_len
            write_offset = tensor_offset
            self.concat_each_tensor_loop_scalar(
                ub_input, self.input_tensors[tensor_index][read_offset],
                ub_output[write_offset], compute_loop, last_axis_len)

            tensor_offset += self.input_shapes[tensor_index][1]

        burst_len = \
            self.output_shape[1] * compute_loop // self.ele_each_block
        last = self.output_shape[1] * compute_loop % self.ele_each_block

        # when burst_len is zero, it must be the last core
        # so neednot process the tail data
        if burst_len == 0:
            burst_len = 1
            last = 0

        ub_last = None
        if last != 0:
            ub_last = self.tik_instance.Tensor(
                self.dtype, (self.ele_each_block,),
                name="ub_last",
                scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, self.ele_each_block) as index:
                src_index = compute_loop * self.output_shape[
                    1] - self.ele_each_block + index
                ub_last[index] = ub_output[src_index]

        self.tik_instance.data_move(output_tensor[output_offset],
                                    ub_output, 0, 1, burst_len,
                                    0, 0)

        if last != 0:
            dst_offset = compute_loop * self.output_shape[
                1] - self.ele_each_block
            self.tik_instance.data_move(
                output_tensor[output_offset + dst_offset],
                ub_last, 0, 1, 1, 0, 0)

    def get_ub_size_scalar(self):
        """
        split ub

        Returns
        -------
        output_ub_size: int
        """
        output_ub_size = self.ub_half_size // 2

        return output_ub_size

    def concat_compute_each_core_scalar(self, in_offset, out_offset,
                                        compute_loop):
        """
        concat input tensor on each core

        Parameters
        ----------
        in_offset: int
            input tensor read offset
        out_offset: int
            output tensor write offset
        compute_loop: int
            compute loop

        Returns
        -------
        None
        """
        output_size = self.get_ub_size_scalar()

        outer_loop, inner_loop, loop_last = get_tensor_loop_para(
            compute_loop, self.output_shape[1], output_size)

        if outer_loop != 0:
            with self.tik_instance.for_range(0, outer_loop) as index:
                input_offset = in_offset + index * inner_loop
                output_offset = \
                    (out_offset + index * inner_loop * self.output_shape[1])
                self.concat_each_tensor_scalar(
                    input_offset, self.output_tensor,
                    output_offset, inner_loop)
        if loop_last != 0:
            input_offset = in_offset + outer_loop * inner_loop
            output_offset = \
                (out_offset + outer_loop * inner_loop * self.output_shape[1])
            self.concat_each_tensor_scalar(input_offset,
                                           self.output_tensor,
                                           output_offset,
                                           loop_last)

    def get_max_dims_remainder_half(self):
        """get_max_dims_remainder_half
        """
        if len(self.output_shape) == 1 \
                or self.output_shape[0] <= self.ele_each_block:
            return 0, 0, 0

        loop_num_list, _, _ = \
            get_offset_and_mask(self.concat_axis, self.input_shapes,
                                self.output_shape, self.ele_each_block)
        loop_num = max(loop_num_list)
        thread_num = 2
        # get max input size
        max_input_dim_size = 0
        for _, shape in enumerate(self.input_shapes):
            max_input_dim_size = \
                max(max_input_dim_size, shape[self.concat_axis])

        # get ub size for one dim, 1 out + 2 max size of input
        ub_need_one_dim = \
            max_input_dim_size*2 + self.output_shape[self.concat_axis] \
            + 2*self.ele_each_block

        if self.is_all_align:
            max_dims = self.ub_half_size // ub_need_one_dim
        else:
            max_dims = \
                (self.ub_half_size - max_input_dim_size) // ub_need_one_dim
        max_dims = (max_dims // (loop_num*8))*loop_num*8
        if max_dims <= 0:
            max_dims = \
                (self.ub_half_size*2 - max_input_dim_size) // ub_need_one_dim
            max_dims = (max_dims // (loop_num*8))*loop_num*8
            thread_num = 1

        max_dims = max(max_dims, 0)

        self.max_input_dim_size = max_input_dim_size

        return max_dims, thread_num, max_input_dim_size

    def concat_last_dim_vector_branch(self):
        """copy all data from src to des
        """
        # when input dtype is int32, can not use vector command to vadds
        if self.ele_each_block == 8:
            self.dtype = "float32"
        elif self.ele_each_block == 16:
            self.dtype = "float16"

        # gen vector mask scalar reg
        self.gen_vector_mask_scalar()

        # core 0 process more self.ele_each_block dims if need
        if self.is_all_align:
            core_zero_tail = 0
        else:
            core_zero_tail = self.ele_each_block
        data_size_first_dim = self.input_shapes[0][0] - core_zero_tail

        # get the max size in input shape
        self.max_dims, thread_num, self.max_input_dim_size = \
            self.get_max_dims_remainder_half()

        inner_loop = self.ele_each_block
        core_len = get_ceil_int(data_size_first_dim, inner_loop)
        core_len = get_ceil_int(core_len, self.aicore_num)
        if core_len == 0:
            core_len = 1

        dims_per_core = core_len * inner_loop
        core_used = data_size_first_dim // dims_per_core
        if data_size_first_dim % dims_per_core != 0:
            core_used = core_used + 1
        tail_dims_core = \
            data_size_first_dim - (core_used - 1)*dims_per_core

        if core_used == 1:
            dims_per_core = tail_dims_core

        concat_fuc = self.proc_vector_scedule
        # for core loop
        with self.tik_instance.for_range(
                0, core_used, block_num=core_used) as _core_index:
            # core 0
            with self.tik_instance.if_scope(
                    _core_index == 0):
                core_dims_offset = 0
                concat_fuc(dims_per_core + core_zero_tail, core_dims_offset,
                           thread_num, core_zero_tail)
            # core equal
            with self.tik_instance.else_scope():
                core_dims_offset = _core_index * dims_per_core + core_zero_tail
                if tail_dims_core != dims_per_core:
                    with self.tik_instance.if_scope(
                            _core_index < (core_used - 1)):
                        concat_fuc(dims_per_core,
                                   core_dims_offset, thread_num)
                    with self.tik_instance.else_scope():
                        concat_fuc(tail_dims_core,
                                   core_dims_offset, thread_num)
                else:
                    concat_fuc(dims_per_core, core_dims_offset, thread_num)

    def proc_vector_scedule(self, dims_len, dims_offset,
                            thread_num, core_zero=0):
        """proc_vector_scedule
        """
        if core_zero != 0:
            dims_len = dims_len - core_zero
            dims_core_zero_offset = core_zero
            # copy self.ele_each_block dims
            copy_tensor = self.tik_instance.Tensor(
                self.dtype,
                (self.max_input_dim_size + self.ele_each_block,),
                name="copy_tensor", scope=tik.scope_ubuf)
            with self.tik_instance.for_range(
                    0, core_zero) as _dims_loop:
                _offset = 0
                for input_index, input_shape in enumerate(self.input_shapes):
                    if input_index != 0:
                        _offset = \
                            self.input_shapes[
                                input_index - 1][self.concat_axis] \
                            + _offset
                    copy_len = get_ceil_int(input_shape[self.concat_axis],
                                            self.ele_each_block)
                    input_offset = \
                        input_shape[self.concat_axis]*_dims_loop
                    output_offset = \
                        self.output_shape[self.concat_axis]*_dims_loop \
                        + _offset
                    self.tik_instance.data_move(
                        copy_tensor,
                        self.input_tensors[input_index][input_offset],
                        0, 1, copy_len, 0, 0)
                    self.tik_instance.data_move(
                        self.output_tensor[output_offset], copy_tensor,
                        0, 1, copy_len, 0, 0)
        else:
            dims_core_zero_offset = 0
        dims_loop = dims_len // self.max_dims
        dims_len = dims_len - self.max_dims*dims_loop
        if dims_loop < 2:
            thread_num = 1

        def _run_one_segment(_dims_len, _segment_index):
            """_run_one_segment
            """
            # get gm dim offset
            dim_offset = \
                dims_offset + _segment_index*self.max_dims \
                + dims_core_zero_offset
            self.function_for_vector_concat(_dims_len, dim_offset, core_zero)

        with self.tik_instance.for_range(
                0, dims_loop, thread_num=thread_num) as _dims_loop:
            _run_one_segment(self.max_dims, _dims_loop)
        if dims_len != 0:
            _run_one_segment(dims_len, dims_loop)

    # pylint: disable=too-many-statements,too-many-branches
    def function_for_vector_concat(self, dims_len, dims_offset, core_zero=0):
        """function_for_half_block
        """
        ub_tensor = self.tik_instance.Tensor(
            self.dtype,
            (self.max_dims*self.max_input_dim_size + self.ele_each_block,),
            name="ub_tensor", scope=tik.scope_ubuf)
        ub_tensor_1 = self.tik_instance.Tensor(
            self.dtype,
            (self.max_dims*self.max_input_dim_size + self.ele_each_block,),
            name="ub_tensor_1", scope=tik.scope_ubuf)
        out_tensor = self.tik_instance.Tensor(
            self.dtype, (self.max_dims*self.output_shape[1],),
            name="out_tensor", scope=tik.scope_ubuf)

        loop_num_list, start_offset_all, mask_value_all = \
            get_offset_and_mask(self.concat_axis, self.input_shapes,
                                self.output_shape, self.ele_each_block)

        def run_one_input(input_idx,):
            list_mask_value = mask_value_all[input_idx]
            loop_num = loop_num_list[input_idx]

            input_len = self.input_shapes[input_idx][1] * dims_len
            input_offset = dims_offset * self.input_shapes[input_idx][1]

            # calcu vector par
            repeat = get_ceil_int(dims_len, loop_num*8)
            dst_m0 = self.output_shape[1]*loop_num // self.ele_each_block
            src_m0 = \
                self.input_shapes[input_idx][1]*loop_num // self.ele_each_block
            dst_m1 = dst_m0*8
            src_m1 = src_m0*8
            repeat_one_time = False
            if dst_m1 > 255 or src_m1 > 255:
                repeat_one_time = True

            # when loop_num greater than the first dim, will change loop_num
            if loop_num > self.output_shape[0] - core_zero:
                loop_num = self.output_shape[0] - core_zero

            def vadds_proc(des_addr, src_ub, src_addr, vector_mask=None):
                if vector_mask is None:
                    vector_mask = self.ele_each_block*8

                if not repeat_one_time:
                    self.tik_instance.vadds(vector_mask, out_tensor[des_addr],
                                            src_ub[src_addr], 0,
                                            repeat, dst_m0,
                                            src_m0, dst_m1, src_m1)
                else:
                    for repeat_idx in range(repeat):
                        _out_offset = \
                            des_addr + repeat_idx*dst_m1*self.ele_each_block
                        self.tik_instance.vadds(
                            vector_mask,
                            out_tensor[_out_offset],
                            src_ub[src_addr
                                   + repeat_idx*src_m1*self.ele_each_block],
                            0, 1, dst_m0, src_m0, 8, 8)

            for i in range(loop_num):
                if (i + input_idx) % 2 == 0:
                    ub_use = ub_tensor
                else:
                    ub_use = ub_tensor_1

                copy_offset = \
                    (self.ele_each_block - list_mask_value[i][0]) \
                    % self.ele_each_block \
                    - self.input_shapes[input_idx][1]*i
                if self.input_shapes[input_idx][1] % self.ele_each_block == 0 \
                        and loop_num > 2:
                    burst_len = \
                        self.input_shapes[input_idx][1] // self.ele_each_block
                    if list_mask_value[i][0] != 0:
                        burst_len = burst_len + 1
                    nbust = dims_len // loop_num
                    if i < dims_len % loop_num:
                        nbust = nbust + 1

                    # when burst_len or nbust less 0,
                    # so no data to move break
                    if burst_len <= 0 or nbust <= 0:
                        break

                    des_stride = src_m0 - burst_len
                    src_sttide = src_m0 - burst_len
                    self.tik_instance.data_move(
                        ub_use,
                        self.input_tensors[input_idx][
                            input_offset - copy_offset],
                        0, nbust, burst_len, src_sttide, des_stride)
                else:
                    burst_len = \
                        get_ceil_int(input_len + self.ele_each_block
                                     - self.input_shapes[input_idx][1]*i,
                                     self.ele_each_block)
                    if burst_len <= 0:
                        break
                    self.tik_instance.data_move(
                        ub_use,
                        self.input_tensors[input_idx][
                            input_offset - copy_offset],
                        0, 1, burst_len, 0, 0)

                offset = 0
                outub_offset = start_offset_all[input_idx][i]
                if list_mask_value[i][0] != 0:
                    out_offset = \
                        (outub_offset // self.ele_each_block) \
                        * self.ele_each_block
                    _mask_scalar = \
                        self.pre_mask_scalar_list[list_mask_value[i][0]]
                    vadds_proc(out_offset, ub_use, offset,
                               [_mask_scalar, _mask_scalar])
                    outub_offset = outub_offset + list_mask_value[i][0]
                    offset = offset + self.ele_each_block
                if repeat_one_time or list_mask_value[i][1] > 2:
                    # use data move to process
                    if list_mask_value[i][1] != 0:
                        out_offset = \
                            (outub_offset // self.ele_each_block) \
                            * self.ele_each_block
                        nbust_len = list_mask_value[i][1]
                        nbust = repeat*8
                        des_stride = dst_m0 - nbust_len
                        src_sttide = src_m0 - nbust_len
                        self.tik_instance.data_move(
                            out_tensor[out_offset],
                            ub_use[offset], 0,
                            nbust, nbust_len, src_sttide, des_stride)
                        outub_offset = \
                            outub_offset \
                            + list_mask_value[i][1]*self.ele_each_block
                        offset = \
                            offset + self.ele_each_block*list_mask_value[i][1]
                else:
                    # use vadds to process
                    for _ in range(list_mask_value[i][1]):
                        out_offset = \
                            (outub_offset // self.ele_each_block) \
                            * self.ele_each_block
                        vadds_proc(out_offset, ub_use, offset)
                        outub_offset = outub_offset + self.ele_each_block
                        offset = offset + self.ele_each_block

                if list_mask_value[i][2] != 0:
                    out_offset = \
                        (outub_offset // self.ele_each_block) \
                        * self.ele_each_block
                    _mask_scalar = \
                        self.post_mask_scalar_list[list_mask_value[i][2]]
                    vadds_proc(out_offset, ub_use, offset,
                               [_mask_scalar, _mask_scalar])

        # concat input one by one
        input_num = len(self.input_shapes)
        tensor_offset = 0
        for _, input_index in enumerate(range(input_num // 2)):
            _index = input_index*2 + 0
            last_axis_len = self.input_shapes[_index][1]
            read_offset = dims_offset * last_axis_len
            write_offset = tensor_offset
            if last_axis_len < self.ele_each_block \
                    and not self.is_remainder_half:
                self.concat_each_tensor_loop_scalar(
                    ub_tensor, self.input_tensors[_index][read_offset],
                    out_tensor[write_offset], dims_len, last_axis_len)
            else:
                run_one_input(_index)
            tensor_offset += last_axis_len

            _index = input_index*2 + 1
            last_axis_len = self.input_shapes[_index][1]
            read_offset = dims_offset * last_axis_len
            write_offset = tensor_offset
            if last_axis_len < self.ele_each_block \
                    and not self.is_remainder_half:
                self.concat_each_tensor_loop_scalar(
                    ub_tensor_1, self.input_tensors[_index][read_offset],
                    out_tensor[write_offset], dims_len, last_axis_len)
            else:
                run_one_input(_index)
            tensor_offset += last_axis_len

        if input_num % 2 != 0:
            _index = input_num - 1
            last_axis_len = self.input_shapes[_index][1]
            read_offset = dims_offset * last_axis_len
            write_offset = tensor_offset
            if last_axis_len < self.ele_each_block \
                    and not self.is_remainder_half:
                self.concat_each_tensor_loop_scalar(
                    ub_tensor, self.input_tensors[_index][read_offset],
                    out_tensor[write_offset], dims_len, last_axis_len)
            else:
                run_one_input(_index)

        out_len = dims_len * self.output_shape[1]
        out_burst_len = get_ceil_int(out_len, self.ele_each_block)
        out_gm_offset = dims_offset * self.output_shape[1]
        self.tik_instance.data_move(
            self.output_tensor[out_gm_offset], out_tensor, 0,
            1, out_burst_len, 0, 0)

    def data_move_cut_by_fisrt_dim(self):
        """data_move_cut_by_fisrt_dim
        """
        data_size_first_dim = self.input_shapes[0][0]
        inner_loop = self.ele_each_block
        core_len = get_ceil_int(data_size_first_dim, inner_loop)
        core_len = get_ceil_int(core_len, self.aicore_num)
        if core_len == 0:
            core_len = 1

        dims_per_core = core_len * inner_loop
        core_used = data_size_first_dim // dims_per_core
        if data_size_first_dim % dims_per_core != 0:
            core_used = core_used + 1
        tail_dims_core = \
            data_size_first_dim - (core_used - 1)*dims_per_core
        concat_fuc = self.proc_data_scedule
        # for core loop
        with self.tik_instance.for_range(
                0, core_used, block_num=core_used) as _core_index:
            core_dims_offset = _core_index * dims_per_core
            if tail_dims_core != dims_per_core:
                # for copy segment loop
                with self.tik_instance.if_scope(
                        _core_index < (core_used - 1)):
                    concat_fuc(_core_index, core_dims_offset, dims_per_core)

                with self.tik_instance.else_scope():
                    concat_fuc(_core_index, core_dims_offset, tail_dims_core)
            else:
                concat_fuc(_core_index, core_dims_offset, dims_per_core)

    def proc_data_scedule(self, _core_index, core_dims_offset, core_process):
        """proc_data_scedule
        """
        output_gm_offset = [0, 3120]

        def _copy_one_dim(src_gm, des_gm, _dims_idx,
                          _input_idx, dim_size, _ub_tensor):
            src_offset = _dims_idx*dim_size
            des_offset = \
                _dims_idx*self.output_shape[1] + output_gm_offset[_input_idx]
            burst_len = get_ceil_int(dim_size,
                                     self.ele_each_block)
            if dim_size == 1:
                src_offset = dim_size - self.ele_each_block + src_offset
                des_offset = dim_size - self.ele_each_block + des_offset
            self.tik_instance.data_move(
                _ub_tensor,
                src_gm[src_offset],
                0, 1, burst_len, 1, 1)
            self.tik_instance.data_move(
                des_gm[des_offset],
                _ub_tensor,
                0, 1, burst_len, 1, 1)

        input_num = len(self.input_shapes)

        ub_tensor = self.tik_instance.Tensor(
            self.dtype,
            (self.max_input_dim_size + self.ele_each_block,),
            name="ub_tensor", scope=tik.scope_ubuf)
        ub_tensor_1 = self.tik_instance.Tensor(
            self.dtype,
            (self.max_input_dim_size + self.ele_each_block,),
            name="ub_tensor_1", scope=tik.scope_ubuf)
        for _, _index in enumerate(range(input_num)):
            input_index = input_num - _index - 1
            with self.tik_instance.for_range(
                    0, core_process // 2) as _dims_loop:
                _copy_one_dim(self.input_tensors[input_index],
                              self.output_tensor,
                              _dims_loop*2 + core_dims_offset,
                              input_index,
                              self.input_shapes[input_index][1],
                              ub_tensor)
                _copy_one_dim(self.input_tensors[input_index],
                              self.output_tensor,
                              _dims_loop*2 + 1 + core_dims_offset,
                              input_index,
                              self.input_shapes[input_index][1],
                              ub_tensor_1)
            if (core_process % 2) != 0:
                _copy_one_dim(self.input_tensors[input_index],
                              self.output_tensor,
                              core_process - 1 + core_dims_offset,
                              input_index,
                              self.input_shapes[input_index][1],
                              ub_tensor)

    def concat_compute(self):
        """
        build concat op

        Returns
        -------
        None
        """
        if self.is_special_shape:
            self.data_move_cut_by_fisrt_dim()
        elif self.is_vector_branches:
            # vadds vector branch
            self.concat_last_dim_vector_branch()
        elif self.is_use_scalar:
            # scalar move branch
            self.concat_last_dim_for_scalar()
        else:
            # tensor move branch
            with self.tik_instance.for_range(
                    0, self.aicore_num, block_num=self.aicore_num) as index:
                self.concat_compute_each_core(index)

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=self.input_tensors,
            outputs=(self.output_tensor,),
            enable_l2=False)

        return self.tik_instance


def get_tensor_loop_para(loop_num, loop_len, ub_size):
    """
    compute loop parameters

    Parameters
    ----------
    loop_num: int
        loop numbers
    loop_len: int
        the length of each loop
    ub_size: int
        the ub block size

    Returns
    -------
    outer_loop: int
        outer loop number
    inner_loop: int
        inner loop number
    loop_last: int
        last loop number
    """
    if loop_num * loop_len > ub_size:
        inner_loop = ub_size // loop_len
        outer_loop = loop_num // inner_loop
        loop_last = loop_num % inner_loop
    else:
        outer_loop = 0
        inner_loop = 0
        loop_last = loop_num

    return outer_loop, inner_loop, loop_last


def check_use_scalar(shapes_list, axis, ele_each_block,
                     out_shape, scalar_ubsize):
    """
    check if use scalar method

    Parameters
    ----------
    shapes_list: list
        shapes of input tensors
    axis: int
        axis of concat
    ele_each_block: int
        ele number in one block
    out_shape: list
        shape of input
    scalar_ubsize: int
        ele number for scalar

    Returns
    -------
    use_scalar: bool
        if use scalar method
    """
    out_size = out_shape[axis]
    # scalar must be concat by last dim
    if axis != 1:
        return False
    # scalar output size of last must be less than scalar_ubsize
    if scalar_ubsize < out_size:
        return False

    use_scalar = False
    for _, input_shape in enumerate(shapes_list):
        if input_shape[axis] < ele_each_block * 32:
            use_scalar = True
        # when scalar input size greater then 4096, not use scalar
        if input_shape[axis] > 4096:
            use_scalar = False
            break

    return use_scalar


def check_use_vector_branch(shapes_list, dtype, output_shape,
                            axis, ele_each_block):
    """check_use_vector_branch
    """
    # check concat with last dim
    if axis != (len(output_shape) - 1) or len(output_shape) == 1:
        return False, False

    # check support dtype
    dtype_support = True
    support_dtype = ("float32", "int32", "float16")
    if dtype not in support_dtype:
        dtype_support = False
        return False, dtype_support

    input_max_size = 0
    for _, input_shape in enumerate(shapes_list):
        input_max_size = max(input_max_size, input_shape[axis])

    if input_max_size < ele_each_block:
        return False, dtype_support

    return True, dtype_support


def cal_loop(ele_num, ub_size):
    """
    calcute loop

    Parameters
    ----------
    ele_num: int
        total number
    ub_size: int
        ele number in one block

    Returns
    -------
    loop: int
        loop number
    ele_each_loop: int
        ele number in one loop
    """
    loop = ele_num // ub_size
    tail = ele_num % ub_size
    ele_each_loop = ub_size
    if tail <= (ub_size * 0.8):
        loop = loop * 2
        ele_each_loop = ele_num // loop
        if ele_num % loop != 0:
            loop = ele_num // ub_size
            ele_each_loop = ub_size

    return loop, ele_each_loop


def get_offset_and_mask(dim, shape_list, output_shape,
                        align_len):
    """
    get offset and mask
    for exp:
       input:
         align_len = 8 (fp32)
         input_1 = [n, 16]
         input_2 = [n, 12]
         output = [n, 28]
       output:
         min_align = 2
         start_offset_all = [[0, 28], [16, 44]]
         mask_value_all = [[[0, 2, 0], [4, 1, 4]], [[0, 1, 4], [4, 1, 0]]]

    Parameters
    ----------
    dim: int
        concat axis
    shape_list: list
        input shape list for concat
    output_shape: list
        output shape for concat
    align_len: int
        ele number in one block

    Returns
    -------
    loop_list : list
         min align loop num per input
    start_offset_all : list
        the offset in the align loop num for inputs
    mask_value_all: list
        the mask in the align loop num for inputs
        [first_mask, repeat, last_mask]
        repeat = align_len // align_len
    """
    offset = 0
    start_offset_all = []
    mask_value_all = []
    loop_list = []
    for _, input_index in enumerate(range(len(shape_list))):
        # calcu repeat loop with input and output
        mask_value = []
        start_offset = []
        concat_size = shape_list[input_index][dim]
        loop_idx = 0
        input_offset_list = []

        for idx, _ in enumerate(range(align_len)):
            # calcu input and output align loop for each input
            input_offset = (concat_size*idx) % align_len
            output_offset = (output_shape[dim]*idx) % align_len
            input_offset_list.append(input_offset)
            if len(input_offset_list) == 1:
                pass
            elif input_offset == output_offset \
                    and input_offset == input_offset_list[0]:
                loop_idx = idx
                break
            else:
                loop_idx = align_len

            mask_1 = \
                align_len \
                - ((offset + output_shape[dim]*idx) % align_len)
            mask_1 = mask_1 % align_len
            mask_1 = mask_1 % align_len
            mask_2 = (concat_size - mask_1) // align_len
            mask_3 = (concat_size - mask_1) % align_len

            mask_value.append([mask_1, mask_2, mask_3])
            start_offset.append(offset + output_shape[dim]*idx)

        loop_list.append(loop_idx)
        start_offset_all.append(start_offset)
        mask_value_all.append(mask_value)
        offset = offset + concat_size

    return loop_list, start_offset_all, mask_value_all


def gen_vector_mask(tail_num, ele_num, mask_mode="POST"):
    """
    gen_vector_mask
    for ex:
       input:
         tail_num = 4
         ele_num = 8
         mask_mode = POST
       output:
         mask = 11110000*8   mask_mode = POST
         mask = 00001111*8   mask_mode = PRE

    Parameters
    ----------
    tail_num: int
        tail mask num
    ele_num: int
        ele num in one block
    mask_mode: str
        POST or PRE

    Returns
    -------
    mask_dec: int
        mask value for vcetor
    """
    zero_mask_str = "0" * (ele_num - tail_num)
    one_mask_ste = "1" * tail_num
    if mask_mode == "POST":
        mask_str = zero_mask_str + one_mask_ste
    else:
        mask_str = one_mask_ste + zero_mask_str

    all_mask_str = mask_str * (64 // ele_num)
    mask_dec = int(all_mask_str, 2)

    return mask_dec


def get_ceil_int(int1, int2):
    """get cel for input1 and input2
    """
    if int1 == 0:
        return 1
    _result = int1 // int2
    if int1 % int2 == 0:
        return _result

    return _result + 1


def check_dim_size_align(shape_list, axis, align_size):
    """
    check if dim_size_align

    Parameters
    ----------
    shape_list: list
        shapes of input tensors
    axis: int
        axis of concat

    align_mode: str
        remainder_half or remainder_blcok

    Returns
    -------
    is_align: bool
        if align
    """
    is_align = True
    for _, input_shape in enumerate(shape_list):
        if input_shape[axis] % align_size != 0:
            is_align = False
            break

    return is_align

