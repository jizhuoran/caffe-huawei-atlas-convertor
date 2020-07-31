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

concat_last_dim
"""
from functools import reduce as functools_reduce
from te import tik
from te import platform as tbe_platform

# vnchwconv can deal 16*16
TRANSPOSE_SIZE = 256
# one block can save the size of fp16
ONE_BLOCK_FP16_SIZE = 16


def get_ceil_int(int1, int2):
    """get cel for input1 and input2
    """
    if int1 == 0:
        return 1
    _result = int1 // int2
    if int1 % int2 == 0:
        return _result

    return _result + 1


# pylint: disable=too-many-instance-attributes,too-many-arguments
# pylint: disable=too-many-locals,too-many-statements,too-many-branches
class ConcatWithVnchw:
    """Function: use to finish ConcatLastDim main functions
    """

    def __init__(self, input_data, output_data, kernel_name="concat_last_dim"):
        """init concat base parameters
        """
        self.input_shapes = []
        self.data_dtype = input_data[0].get("dtype").lower()
        self.gm_in = []
        self.last_dim = input_data[0].get("shape")[-1]
        self.input_num = len(input_data)
        for index, input_dict in enumerate(input_data):
            shape_input = input_dict.get("shape")
            self.input_shapes.append(shape_input)

        self.output_shape = output_data.get("shape")

        self.kernel_name = kernel_name

        if self.data_dtype == "float32":
            self.last_dim = self.last_dim * 2

        self.tik_instance = tik.Tik()
        self.core_num = tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.CORE_NUM)

        self.src_size = int(self.get_tensor_size_in_fp16(self.input_shapes[0]))
        self.dst_size = int(self.get_tensor_size_in_fp16(self.output_shape))

        self.use_last_dim = False
        self.use_last_dim_odd = False

        for index, _ in enumerate(self.input_shapes):
            self.gm_in.append(
                self.tik_instance.Tensor(
                    "float16", (self.src_size,),
                    scope=tik.scope_gm,
                    name="data_gm_in_{}".format(index)))

        self.gm_out = self.tik_instance.Tensor(
            "float16", (self.dst_size,), scope=tik.scope_gm, name="data_gm_out")

    def whether_last_dim_same(self):
        """check whether all the last dim of inputs are the same
        """
        last_dim_the_same = True
        input_last_dim = self.input_shapes[0][-1]
        for i, _ in enumerate(self.input_shapes):
            if input_last_dim != self.input_shapes[i][-1]:
                last_dim_the_same = False
                break
        return last_dim_the_same

    def check_vnchw_supported(self):
        """
        check if vnchw schedule support this shape

        Returns
        -------
        if_supported: bool
            if vnchw schedule support this shape
        """
        last_dim_the_same = self.whether_last_dim_same()
        if not last_dim_the_same \
                or len(self.output_shape) == 1 \
                or self.input_num == 1:
            return False

        input_last_dim = self.input_shapes[0][-1]
        output_last_dim = self.output_shape[-1]

        if output_last_dim != self.input_num * input_last_dim:
            return False

        sup_shape = [1, 2, 4, 8]
        sup_count = [2, 4, 8, 16]
        factor = 1
        if self.data_dtype == "float32":
            factor = 2

        if self.data_dtype in ["float32", "float16"] \
                and input_last_dim in sup_shape \
                and self.input_num in sup_count \
                and output_last_dim * factor <= 16 and self.src_size >= 256:
            self.use_last_dim = True

        if self.data_dtype == "float16" \
                and input_last_dim == 1 \
                and self.input_num == 3 \
                and self.src_size >= TRANSPOSE_SIZE * 8:
            self.use_last_dim_odd = True

        return self.use_last_dim or self.use_last_dim_odd

    def get_tensor_size_in_fp16(self, data_shape):
        """get_tensor_size_in_fp16
        """
        data_size = functools_reduce(lambda x, y: x * y, data_shape)
        fp16_size = data_size
        if self.data_dtype == "float32":
            fp16_size = fp16_size * 2
        return fp16_size

    def concat_last_dim_one_core(self, src_core_offset, des_core_offset,
                                 core_pro, mov_tail, max_mov_num):
        """concat_last_dim_one_core
        """
        # per core scedule
        if mov_tail != 0:
            core_pro = core_pro - 1

        core_pro = max(core_pro, 0)
        core_loop = core_pro // max_mov_num
        core_tail = core_pro % max_mov_num

        input_ub_0 = \
            self.tik_instance.Tensor("float16", (256*max_mov_num,),
                                     tik.scope_ubuf, "input_ub_0")
        vnchw_ub_0 = \
            self.tik_instance.Tensor("float16",
                                     (256*self.input_num*max_mov_num,),
                                     tik.scope_ubuf, "vnchw_ub_0")
        out_ub_0 = \
            self.tik_instance.Tensor("float16",
                                     (256*self.input_num*max_mov_num,),
                                     tik.scope_ubuf, "out_ub_0")
        input_ub_1 = self.tik_instance.Tensor("float16", (256*max_mov_num,),
                                              tik.scope_ubuf, "input_ub_1")
        vnchw_ub_1 = \
            self.tik_instance.Tensor("float16",
                                     (256*self.input_num*max_mov_num,),
                                     tik.scope_ubuf, "vnchw_ub_1")
        out_ub_1 = self.tik_instance.Tensor("float16",
                                            (256*self.input_num*max_mov_num,),
                                            tik.scope_ubuf, "out_ub_1")

        tiling_ub_list_0 = [input_ub_0, vnchw_ub_0, out_ub_0]
        tiling_ub_list_1 = [input_ub_1, vnchw_ub_1, out_ub_1]

        def _run_copy_input_and_vnchw(input_idx, ub_list, gm_input_offset,
                                      run_mov_num, copy_tail):
            copy_len = run_mov_num * TRANSPOSE_SIZE - copy_tail
            nburst = get_ceil_int(copy_len,
                                  ONE_BLOCK_FP16_SIZE)
            ub_copy, ub_vnchw, _, gm_input = ub_list
            # copy gm to ub
            self.tik_instance.data_move(ub_copy,
                                        gm_input[gm_input_offset],
                                        0, 1, nburst, 0, 0)
            # vnchwconv to ub_vnchw
            _src_addrs = [
                ub_copy[ONE_BLOCK_FP16_SIZE * x]
                for x in range(ONE_BLOCK_FP16_SIZE)
            ]
            dst_offset_ub = self.last_dim * input_idx
            dst_loop = ONE_BLOCK_FP16_SIZE // self.last_dim
            inner_offset_ub = self.last_dim * self.input_num
            _dst_addrs = []
            for dloop in range(dst_loop):
                for in_loop in range(self.last_dim):
                    _dst_addrs.append(
                        ub_vnchw[
                            (dst_offset_ub + dloop * inner_offset_ub + in_loop)
                            * ONE_BLOCK_FP16_SIZE])
            _dst_rep_stride = 16 * self.input_num
            _src_rep_stride = 16
            if run_mov_num == 1:
                _dst_rep_stride = 0
                _src_rep_stride = 0
            self.tik_instance.vnchwconv(False, False, _dst_addrs,
                                        _src_addrs, run_mov_num,
                                        _dst_rep_stride, _src_rep_stride)

        def _run_vnchw_all_to_out(ub_list, gm_output_offset,
                                  run_mov_num, copy_tail):
            _, ub_vnchw, ub_out, gm_output = ub_list
            _dst_rep_stride = (256 // 16) * self.input_num
            _src_rep_stride = (256 // 16) * self.input_num
            if run_mov_num == 1:
                _dst_rep_stride = 0
                _src_rep_stride = 0
            for i_idx in range(self.input_num):
                _src_addrs = [
                    ub_vnchw[i_idx * TRANSPOSE_SIZE + ONE_BLOCK_FP16_SIZE * x]
                    for x in range(ONE_BLOCK_FP16_SIZE)
                ]
                _dst_addrs = [
                    ub_out[(i_idx + x * self.input_num) *
                           ONE_BLOCK_FP16_SIZE]
                    for x in range(ONE_BLOCK_FP16_SIZE)
                ]
                self.tik_instance.vnchwconv(False, False, _dst_addrs,
                                            _src_addrs, run_mov_num,
                                            _dst_rep_stride, _src_rep_stride)
            copy_len = \
                (run_mov_num * TRANSPOSE_SIZE - copy_tail)*self.input_num
            nburst = get_ceil_int(copy_len,
                                  ONE_BLOCK_FP16_SIZE)
            # copy ub to gm
            self.tik_instance.data_move(gm_output[gm_output_offset], ub_out, 0,
                                        1, nburst, 0, 0)

        def _run_one_loop(tiling_ub_list, _loop_offset,
                          run_mov_num, copy_tail=0):
            src_offset = \
                src_core_offset \
                + _loop_offset
            dst_offset = \
                des_core_offset \
                + _loop_offset * self.input_num
            # copy input one by one and vnchwconv input to vnchw_ub
            for i_idx in range(self.input_num):
                _run_copy_input_and_vnchw(
                    i_idx,
                    tiling_ub_list + [self.gm_in[i_idx]],
                    src_offset, run_mov_num, copy_tail
                )

            # vnchwconv vnchw_ub to res_ub and copy un to gm out
            _run_vnchw_all_to_out(
                tiling_ub_list + [self.gm_out],
                dst_offset, run_mov_num, copy_tail
            )

        with self.tik_instance.for_range(
                0, core_loop // 2) as loop_idx:
            _idx = loop_idx*2
            _run_one_loop(tiling_ub_list_0,
                          _idx * TRANSPOSE_SIZE * max_mov_num,
                          max_mov_num)
            _idx = loop_idx*2 + 1
            _run_one_loop(tiling_ub_list_1,
                          _idx * TRANSPOSE_SIZE * max_mov_num,
                          max_mov_num)

        if core_loop % 2 == 1:
            _idx = core_loop - 1
            _run_one_loop(tiling_ub_list_0,
                          _idx * TRANSPOSE_SIZE * max_mov_num,
                          max_mov_num)

        if core_tail != 0:
            _idx = core_loop
            _run_one_loop(tiling_ub_list_1,
                          _idx * TRANSPOSE_SIZE * max_mov_num,
                          core_tail)

        if mov_tail != 0:
            _offset = core_loop * TRANSPOSE_SIZE * max_mov_num \
                      + core_tail * TRANSPOSE_SIZE
            _run_one_loop(tiling_ub_list_0, _offset, 1, mov_tail)

    def concat_last_dim_compute(self):
        """concat_last_dim_compute
        """
        ub_half_size = \
            int(tbe_platform.CceProductParams().getParams("Unified_Buffer")
                // 2 // 2 - 16)
        max_mov_num = \
            ub_half_size // TRANSPOSE_SIZE // (self.input_num*2 + 1)

        # core scedule
        mov_num = (self.src_size + TRANSPOSE_SIZE - 1) // TRANSPOSE_SIZE
        mov_tail = mov_num*TRANSPOSE_SIZE - self.src_size

        move_num_per_core = get_ceil_int(mov_num, self.core_num)

        core_used = mov_num // move_num_per_core
        if mov_num % move_num_per_core != 0:
            core_used = core_used + 1
        move_num_core_tail = \
            mov_num - (core_used - 1)*move_num_per_core

        # define concat fuction
        concat_fuc = self.concat_last_dim_one_core

        with self.tik_instance.for_range(
                0, core_used, block_num=core_used) as core_idx:
            src_offset_core = \
                core_idx * move_num_per_core * TRANSPOSE_SIZE
            dst_offset_core = src_offset_core * self.input_num
            if mov_tail == 0 and move_num_core_tail == move_num_per_core:
                concat_fuc(
                    src_offset_core, dst_offset_core,
                    move_num_per_core, 0, max_mov_num)
            else:
                with self.tik_instance.if_scope(
                        core_idx < (core_used - 1)):
                    concat_fuc(
                        src_offset_core, dst_offset_core,
                        move_num_per_core, 0, max_mov_num)

                with self.tik_instance.else_scope():
                    concat_fuc(
                        src_offset_core, dst_offset_core,
                        move_num_core_tail, mov_tail, max_mov_num)

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=self.gm_in,
            outputs=[self.gm_out],
            enable_l2=False)

        return self.tik_instance


    def do_concat_vnchw(self):
        """do_concat_vnchw for last dims
        """
        self.concat_last_dim_compute()
