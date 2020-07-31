#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

cumsum
"""

from te import tik
from te import platform as tbe_platform

from impl.constant_util import DATA_TYPE_FP16
from impl.constant_util import BLOCK_SIZE
from impl.constant_util import DATA_TYPE_INT8
from impl.constant_util import VECTOR_BYTE_SIZE
from impl.constant_util import STRIDE_ONE
from impl.constant_util import REPEAT_STRIDE_EIGHT
from impl.constant_util import DEFAULT_BURST_LEN
from impl.constant_util import STRIDE_ZERO
from impl.constant_util import DATA_TYPE_UINT8
from impl.common_util import get_data_size

# A maximum of 25k can be calculated at a time.
MAX_COMPUTE_SIZE = 25 * 1024

# const value 1
VALUE_ONE = 1

# const value 0
VALUE_ZERO = 0

# const value 2
VALUE_TWO = 2

# const value -1
NEG_ONE = -1

# repeat stride 4 for vconv
STRIDE_FOUR = 4

# type of cumsum op
SUM_TYPE = "sum"

# type of cumprod op
PROD_TYPE = "prod"

# type of cumlogsumexp
LOGSUMEXP_TYPE = "logsumexp"

# handle position of tail
TAIL = "tail"

# handle postiion of head
HEAD = "head"


# pylint: disable=useless-object-inheritance, import-error
class CumBase(object):
    """
        Function: use to store cumsum base parameters
        Modify : 2019-10-08
    """

    def __init__(self, shape, axis, dtype):
        """
        init the base param of cumsum

        Parameters
        ----------
        shape: the shape of tensor
        axis: cumulative axis
        dtype: the data type of tensor

        Returns
        -------
        None

        """
        self.tik_instance = tik.Tik()
        self.each_loop = shape[axis]
        self.dsize = get_data_size(dtype)
        self.each, self.each_tail = self.get_each(shape, axis)
        self.reserved = self.get_reserved()
        self.dtype = dtype

    def get_each(self, shape, axis):
        """
        Calculate the length of each separate accumulation.

        Parameters
        ----------
        shape: the shape of tensor
        axis: cumulative axis

        Returns
        -------
        each: the length of each separate accumulation

        """
        self.each = VALUE_ONE
        for k in range(axis + VALUE_ONE, len(shape)):
            self.each = self.each * shape[k]
        each_tail = self.each % (BLOCK_SIZE // self.dsize)

        return self.each, each_tail

    def get_reserved(self):
        """
        Prevent tensor overflow and calculate the additional length.

        Returns
        -------
        reserved: the additional length

        """
        if self.each * self.dsize % BLOCK_SIZE != VALUE_ZERO:
            reserved = BLOCK_SIZE // (self.each_loop * self.each) + VALUE_ONE
        else:
            reserved = VALUE_ZERO

        return reserved


class CumTensor(CumBase):
    """
        Function: use to store cumsum tensor
        Modify : 2019-10-08
    """

    def __init__(self, shape, axis, dtype):
        """
        init cumsum tensor

        Parameters
        ----------
        shape: the shape of tensor
        axis: cumulative axis
        dtype: the data type of tensor

        Returns
        -------
        None

        """
        super(CumTensor, self).__init__(shape, axis, dtype)
        self.mov_len = int(self.get_mov_len())
        self.mov_loop = int(self.each // self.mov_len)
        self.mov_tail = int(
            (self.each - self.mov_len * self.mov_loop) % self.mov_len)
        self.rdtype = DATA_TYPE_FP16 if \
            dtype in (DATA_TYPE_UINT8, DATA_TYPE_INT8) else dtype
        self.rdsize = VALUE_TWO if \
            dtype in (DATA_TYPE_UINT8, DATA_TYPE_INT8) else self.dsize
        self.mask = VECTOR_BYTE_SIZE // self.rdsize

    def check_dtype_in_u8s8(self):
        """
        check data type whether in int8 or uint8

        Returns
        -------
        None

        """
        return self.dtype in (DATA_TYPE_UINT8, DATA_TYPE_INT8)

    def get_total_loop(self, shape, axis):
        """
        get total loop of shape

        Parameters
        ----------
        shape: the shape of tensor
        axis: the cumulative axis

        Returns
        -------
        total_loop: the total loop

        """
        self.dsize = self.dsize
        total_loop = VALUE_ONE
        for j in range(VALUE_ZERO, axis):
            total_loop = total_loop * shape[j]

        return total_loop

    def get_mov_len(self):
        """
        Calculate the size of one move.

        Returns
        -------
        mov_len: the size of one move

        """
        max_size = MAX_COMPUTE_SIZE
        rdsize = VALUE_TWO if self.check_dtype_in_u8s8() else self.dsize
        if max_size >= (self.each * rdsize):
            mov_len = self.each
        else:
            mov_len = max_size // rdsize

        return mov_len

    def get_temp_ubtensor(self):
        """
        get temp tensor for multi core

        Returns
        -------
        last_32b: ub temp tensor

        """

        return self.tik_instance.Tensor(self.dtype, (BLOCK_SIZE,),
                                        tik.scope_ubuf, "last_32B")

    def get_outer_loop(self, shape, axis):
        """
        Calculate the number of times the outer loop.

        Parameters
        ----------
        shape: the shape of tensor
        axis: cumulative axis

        Returns
        -------
        outer_loop: the number of times the outer loop

        """
        total_loop = self.get_total_loop(shape, axis)

        if self.each * self.dsize < BLOCK_SIZE or (
                self.mov_tail > VALUE_ZERO and self.mov_tail
                * self.dsize < BLOCK_SIZE):
            block_num = VALUE_ONE
        block_num = tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.CORE_NUM)
        if block_num > total_loop:
            outer_loop = VALUE_ONE
            block_num = total_loop
            outer_tail = VALUE_ZERO
        else:
            outer_loop = total_loop // block_num
            outer_tail = total_loop - block_num * outer_loop

        return block_num, total_loop, outer_loop, outer_tail


class CumTilingParam(CumTensor):
    """
        Function: Used to calculate the tiling parameter.
        Modify: 2019-10-08
    """

    def __init__(self, shape, axis, dtype):
        """
        init the tiling input param

        Parameters
        ----------
        shape: the shape of tensor
        axis: cumulative axis
        dtype: the data type of tensor

        Returns
        -------
        None
        """
        super(CumTilingParam, self).__init__(shape, axis, dtype)
        self.block_num, self.total_loop, self.outer_loop, self.outer_tail = \
            self.get_outer_loop(shape, axis)
        self.exclusive = False
        self.reverse = False

    def get_repeat(self, length):
        """
        Calculate the times of repeat

        Returns
        -------
        repeat: the times of repeat

        """
        # head 256B align, tail 32B align
        if length * self.rdsize % VECTOR_BYTE_SIZE == VALUE_ZERO:
            repeat = length * self.rdsize // VECTOR_BYTE_SIZE
        else:
            repeat = length * self.rdsize // VECTOR_BYTE_SIZE + VALUE_ONE

        return repeat

    def get_offset(self, e_cycle):
        """
        Calculate the offset.

        Parameters
        ----------
        e_cycle: the loop time of each

        Returns
        -------
        in_offset: the offset of move in
        out_offset: the offset of move out

        """
        if not self.exclusive and self.reverse:
            in_offset = self.each_loop - VALUE_ONE - VALUE_TWO * e_cycle
            out_offset = self.each_loop - VALUE_ONE - VALUE_TWO * e_cycle
        elif self.exclusive and self.reverse:
            in_offset = self.each_loop - VALUE_TWO * e_cycle
            out_offset = self.each_loop - VALUE_ONE - VALUE_TWO * e_cycle
        elif self.exclusive and not self.reverse:
            in_offset = NEG_ONE
            out_offset = VALUE_ZERO
        else:
            in_offset = VALUE_ZERO
            out_offset = VALUE_ZERO

        return in_offset, out_offset

    def get_burlen_by_mlen(self, length):
        """
        Calculate the tail length of a move in instruct

        Returns
        -------
        burstlen: the tail length of a move in instruct

        """
        if (length * self.dsize) % BLOCK_SIZE == VALUE_ZERO:
            burstlen = length * self.dsize // BLOCK_SIZE
        else:
            burstlen = length * self.dsize // BLOCK_SIZE + VALUE_ONE

        return burstlen

    def set_ext_params(self, exclusive, reverse):
        """
        set expansion param

        Parameters
        ----------
        exclusive: if `True`, perform exclusive cumsum
        reverse: indicates whether to reverse calculation

        Returns
        -------
        None

        """
        self.exclusive = exclusive
        self.reverse = reverse


class CumComputer(CumTilingParam):
    """
        Function: use to compute the cumsum
        Modify: 2019-10-08
    """
    def __init__(self, input_x, axis, kernel_name, ctype):
        """
        init the input param

        Parameters
        ----------
        input_x: shape and dtype
        axis: cumulative axis
        kernel_name: kernel name
        ctype: computer type , "sum" or "prod"

        """
        super(CumComputer, self).__init__(input_x.get("shape"), axis,
                                          input_x.get("dtype"))
        self.ctype = ctype
        self.kernel_name = kernel_name
        self.need_special, self.spe_position = self.get_multi_special_position()
        # gm tensor
        self.input_x_gm = self.tik_instance \
            .Tensor(self.dtype, (self.total_loop + self.reserved,
                                 self.each_loop, self.each),
                    name="input_x_gm",
                    scope=tik.scope_gm)
        self.output_out_gm = self.tik_instance. \
            Tensor(self.dtype, (self.total_loop + self.reserved,
                                self.each_loop, self.each),
                   name="output_out_gm",
                   scope=tik.scope_gm)

    def get_multi_special_position(self):
        """
        Calculate the position where the multi-core special processing is
        required.

        Returns
        -------
        need_special: where the multi-core special processing is required
        position: the position

        """
        need_special = False
        position = HEAD
        if self.block_num > VALUE_ONE and self.each_tail != VALUE_ZERO:
            need_special = True
            position = HEAD if self.mov_tail == VALUE_ZERO else TAIL

        return need_special, position

    def post_multicore(self, burlen, idx, real_out, tail_idx):
        """
        Multi-core postprocessing

        Parameters
        ----------
        burlen: the param of dma instruction
        idx: index
        real_out: ub out
        tail_idx: tail index

        Returns
        -------
        last_32B: last 32B store data not aligned with 32B

        """
        last_32b = self.get_temp_ubtensor()
        self.tik_instance.data_move(last_32b, self.output_out_gm[
            idx[0], idx[1], idx[2] + tail_idx],
                                    VALUE_ZERO, VALUE_ONE, VALUE_ONE,
                                    STRIDE_ZERO, STRIDE_ZERO)
        tmp_scalar = self.tik_instance.Scalar(self.dtype)
        for i in range(self.each_tail):
            tmp_scalar.set_as(real_out[burlen * BLOCK_SIZE // self.dsize + i])
            last_32b[BLOCK_SIZE // self.dsize - self.each_tail + i] \
                .set_as(tmp_scalar)

        return last_32b

    def pre_multicore(self, burlen, position):
        """
        Multi-core preprocessing

        Parameters
        ----------
        burlen: the param of dma instruction
        position: the position of tensor

        Returns
        -------
        burlen: Processed burlen
        tail_idx: Indicates the offset index

        """
        tail_idx = VALUE_ZERO
        if self.need_special and position == self.spe_position:
            burlen = burlen - VALUE_ONE
            tail_idx = burlen * BLOCK_SIZE // self.dsize + \
                       self.each_tail - BLOCK_SIZE // self.dsize

        return burlen, tail_idx

    def get_tik_instance(self):
        """
        get the instance of tik

        Returns
        -------
        tik_instance: the instance of tik

        """
        self.cum_computer()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.input_x_gm,),
                                   outputs=(self.output_out_gm,),
                                   enable_l2=False)

        return self.tik_instance

    def t_vdup_to_gm(self, last_ret, last_ori, idx, position):
        """
        It is used to process the special first axis and is compatible with
        u8s8.

        Parameters
        ----------
        last_ret: Stores temporary results.
        last_ori: Used to store the original type.
        idx: index
        position: the position of each , "head" or "tail"

        Returns
        -------
        None

        """

        burlen = self.get_burlen_by_mlen(self.mov_len) \
            if position == HEAD else self.get_burlen_by_mlen(self.mov_tail)
        repeat = self.get_repeat(self.mov_len) \
            if position == HEAD else self.get_repeat(self.mov_tail)

        # Check whether multi-core special processing is required.
        burlen, tail_idx = self.pre_multicore(burlen, position)
        # SUM_TYE: 0, PROD_TYPE: 1, LOGSUMEXP: min
        if self.ctype == SUM_TYPE:
            value = VALUE_ZERO
        elif self.ctype == PROD_TYPE:
            value = VALUE_ONE
        elif self.ctype == LOGSUMEXP_TYPE:
            if self.dtype == "float16":
                value = -2 ** 15 * 1.9991
            elif self.dtype == "float32":
                value = -2 ** 127 * 1.9999999

        self.tik_instance.vector_dup(self.mask, last_ret, value, repeat,
                                     STRIDE_ONE,
                                     REPEAT_STRIDE_EIGHT)
        ub_out = last_ret
        if self.check_dtype_in_u8s8():
            self.tik_instance.vconv(self.mask, "", last_ori, last_ret,
                                    repeat, STRIDE_ONE,
                                    STRIDE_ONE, STRIDE_FOUR,
                                    REPEAT_STRIDE_EIGHT)
            ub_out = last_ori
        if burlen != 0:
            self.tik_instance.data_move(self.output_out_gm[idx], ub_out,
                                        VALUE_ZERO,
                                        DEFAULT_BURST_LEN,
                                        burlen, STRIDE_ZERO, STRIDE_ZERO)
        if self.need_special and position == self.spe_position:
            last_32b = self.post_multicore(burlen, idx, ub_out, tail_idx)

            self.tik_instance.data_move(
                self.output_out_gm[idx[0], idx[1], idx[2] + tail_idx],
                last_32b, VALUE_ZERO, DEFAULT_BURST_LEN,
                VALUE_ONE, STRIDE_ZERO, STRIDE_ZERO)
        if self.ctype == LOGSUMEXP_TYPE:
            self.tik_instance.vector_dup(self.mask, last_ret, VALUE_ZERO,
                                         repeat,
                                         STRIDE_ONE,
                                         REPEAT_STRIDE_EIGHT)

    def t_dma_in(self, ub_in, ori, idx, burlen):
        """
        The dma command is compatible with the u8s8 type.

        Parameters
        ----------
        ub_in: ub tensor
        ori: ub tensor with original type
        idx: index
        burlen: dma param

        Returns
        -------
        None

        """
        real_in = ori if self.check_dtype_in_u8s8() else ub_in
        repeat = self.get_repeat(self.mov_len)

        self.tik_instance.data_move(real_in, self.input_x_gm[idx], VALUE_ZERO,
                                    DEFAULT_BURST_LEN,
                                    burlen, STRIDE_ZERO, STRIDE_ZERO)

        if self.check_dtype_in_u8s8():
            self.tik_instance.vconv(self.mask, "", ub_in, ori, repeat,
                                    VALUE_ONE, VALUE_ONE, REPEAT_STRIDE_EIGHT,
                                    STRIDE_FOUR)

    def t_dma_direct_out(self, last_ret, last_ori, idx, position):
        """
        The dma command is compatible with the u8s8 type.

        Parameters
        ----------
        last_ret: the tensor store last result
        last_ori: the tensor store last result with original type
        idx: index
        position: the position of each , "head" or "tail"

        Returns
        -------
        None

        """
        burlen = self.get_burlen_by_mlen(self.mov_len) \
            if position == HEAD else self.get_burlen_by_mlen(self.mov_tail)

        # Check whether multi-core special processing is required.
        burlen, tail_idx = self.pre_multicore(burlen, position)

        real_out = last_ori if self.check_dtype_in_u8s8() else last_ret
        if burlen != 0:
            self.tik_instance.data_move(self.output_out_gm[idx], real_out,
                                        VALUE_ZERO,
                                        DEFAULT_BURST_LEN, burlen, STRIDE_ZERO,
                                        STRIDE_ZERO)

        if self.need_special and position == self.spe_position:
            last_32b = self.post_multicore(burlen, idx, real_out, tail_idx)

            self.tik_instance.data_move(
                self.output_out_gm[idx[0], idx[1], idx[2] + tail_idx],
                last_32b, VALUE_ZERO, DEFAULT_BURST_LEN,
                VALUE_ONE, STRIDE_ZERO, STRIDE_ZERO)

    def t_dma_trans_out(self, last_ret, last_ori, idx, position):
        """
        The dma command is compatible with the u8s8 type.

        Parameters
        ----------
        last_ret: the tensor store last result
        last_ori: the tensor store last result with original type
        idx: index
        position: the position of each , "head" or "tail"

        Returns
        -------
        None

        """
        before = self.tik_instance.Tensor(self.dtype, (BLOCK_SIZE,),
                                          name="before",
                                          scope=tik.scope_ubuf)

        burlen = self.get_burlen_by_mlen(self.mov_len) \
            if position == HEAD else self.get_burlen_by_mlen(self.mov_tail)
        repeat = self.get_repeat(self.mov_len) \
            if position == HEAD else self.get_repeat(self.mov_tail)

        # Check whether multi-core special processing is required.
        burlen, tail_idx = self.pre_multicore(burlen, position)

        real_out = last_ret
        if self.check_dtype_in_u8s8():
            self.tik_instance.vconv(self.mask, "", last_ori, last_ret,
                                    repeat, STRIDE_ONE,
                                    STRIDE_ONE, STRIDE_FOUR,
                                    REPEAT_STRIDE_EIGHT)
            real_out = last_ori

        if self.mov_loop == VALUE_ONE  and self.reverse and \
                (self.mov_len * self.dsize) % BLOCK_SIZE != VALUE_ZERO:
            self.tik_instance.data_move(
                before, self.output_out_gm[idx[0], idx[1] + VALUE_ONE, idx[2]],
                VALUE_ZERO, DEFAULT_BURST_LEN,
                VALUE_ONE, STRIDE_ZERO, STRIDE_ZERO)
            tail_idx_fix = self.get_burlen_by_mlen(self.mov_len) * BLOCK_SIZE \
                           // self.dsize + self.each_tail - BLOCK_SIZE // \
                           self.dsize

            with self.tik_instance.for_range(
                    0, BLOCK_SIZE // self.dsize - self.each_tail) as t_idx:
                temp_scalar = self.tik_instance.Scalar(dtype=self.dtype)
                temp_scalar.set_as(before[t_idx])
                real_out[tail_idx_fix + t_idx].set_as(temp_scalar)
        if burlen != 0:
            self.tik_instance.data_move(self.output_out_gm[idx],
                                        real_out, VALUE_ZERO,
                                        DEFAULT_BURST_LEN,
                                        burlen, STRIDE_ZERO, STRIDE_ZERO)
        if self.need_special and position == self.spe_position:
            last_32b = self.post_multicore(burlen, idx, real_out, tail_idx)

            self.tik_instance.data_move(
                self.output_out_gm[idx[0], idx[1], idx[2] + tail_idx],
                last_32b, VALUE_ZERO, DEFAULT_BURST_LEN,
                VALUE_ONE, STRIDE_ZERO, STRIDE_ZERO)

    def cum_computer(self):
        """
        Calculation process of the operator

        Returns
        -------
        None

        """
        with self.tik_instance.for_range(0, self.block_num,
                                         block_num=self.block_num) as block_i:
            self.handle_out_loop(block_i, self.outer_loop)

        # handle kernel tail
        if self.outer_tail != VALUE_ZERO:
            self.handle_out_loop(self.block_num, self.outer_tail)

    def handle_out_loop(self, block_i, loop_num):
        """
        Multi-core processing data of the entire block

        Parameters
        ----------
        block_i: block index
        loop_num: loop number

        Returns
        -------
        None

        """
        with self.tik_instance.for_range(VALUE_ZERO, loop_num) as o_cycle:
            o_idx = o_cycle + block_i * self.outer_loop

            # handle mov tail first because overlap
            if self.mov_tail != VALUE_ZERO:
                self.handle_mov_tail(o_idx)

            self.handle_mov_loop(o_idx)

    def handle_mov_loop(self, o_cycle):
        """
        Calculation Process

        Parameters
        ----------
        o_cycle: outer cycle

        Returns
        -------
        None

        """

        thread_num = VALUE_TWO if self.mov_loop > VALUE_ONE else VALUE_ONE
        with self.tik_instance.for_range(VALUE_ZERO, self.mov_loop,
                                         thread_num=thread_num) as m_cycle:
            # ub tensor
            input_x_ub = self.tik_instance. \
                Tensor(self.rdtype,
                       (MAX_COMPUTE_SIZE // self.rdsize + VECTOR_BYTE_SIZE,),
                       name="input_x_ub",
                       scope=tik.scope_ubuf)
            last_ret = self.tik_instance. \
                Tensor(self.rdtype,
                       (MAX_COMPUTE_SIZE // self.rdsize + VECTOR_BYTE_SIZE,),
                       name="last_ret",
                       scope=tik.scope_ubuf)
            last_ori = self.tik_instance. \
                Tensor(self.dtype,
                       (MAX_COMPUTE_SIZE // self.dsize + VECTOR_BYTE_SIZE,),
                       name="last_ori",
                       scope=tik.scope_ubuf)

            burstlen = self.get_burlen_by_mlen(self.mov_len)
            repeat = self.get_repeat(self.mov_len)
            if self.exclusive and not self.reverse:
                idx = [o_cycle, VALUE_ZERO, m_cycle * self.mov_len]
                self.t_vdup_to_gm(last_ret, last_ori, idx, HEAD)

            elif not self.exclusive and not self.reverse:
                idx = [o_cycle, VALUE_ZERO, m_cycle * self.mov_len]
                self.t_dma_in(last_ret, last_ori, idx, burstlen)
                self.t_dma_direct_out(last_ret, last_ori, idx, HEAD)

            elif not self.exclusive and self.reverse:
                idx = [o_cycle, self.each_loop - VALUE_ONE,
                       m_cycle * self.mov_len]
                self.t_dma_in(last_ret, last_ori, idx, burstlen)
                self.t_dma_direct_out(last_ret, last_ori, idx, HEAD)

            elif self.exclusive and self.reverse:
                idx = [o_cycle, self.each_loop - VALUE_ONE,
                       m_cycle * self.mov_len]
                self.t_vdup_to_gm(last_ret, last_ori, idx, HEAD)

            tmp_ori = self.tik_instance. \
                Tensor(self.dtype,
                       (BLOCK_SIZE,),
                       name="tmp_ori",
                       scope=tik.scope_ubuf)

            if self.each_loop == VALUE_ONE and self.exclusive:
                self.tik_instance.data_move(tmp_ori, self.input_x_gm,
                                            VALUE_ZERO, VALUE_ONE,
                                            VALUE_ONE, STRIDE_ZERO, STRIDE_ZERO)

            with self.tik_instance.for_range(VALUE_ONE,
                                             self.each_loop) as e_cycle:
                in_offset, out_offset = self.get_offset(e_cycle)

                idx = [o_cycle, e_cycle + in_offset,
                       m_cycle * self.mov_len]
                self.t_dma_in(input_x_ub, last_ori, idx, burstlen)
                if self.ctype == SUM_TYPE:
                    self.tik_instance.vadd(self.mask, last_ret, input_x_ub,
                                           last_ret,
                                           repeat, STRIDE_ONE, STRIDE_ONE,
                                           STRIDE_ONE,
                                           REPEAT_STRIDE_EIGHT,
                                           REPEAT_STRIDE_EIGHT,
                                           REPEAT_STRIDE_EIGHT)
                    idx = [o_cycle, e_cycle + out_offset,
                           m_cycle * self.mov_len]
                    self.t_dma_trans_out(last_ret, last_ori, idx, HEAD)

                elif self.ctype == PROD_TYPE:
                    self.tik_instance.vmul(self.mask, last_ret, input_x_ub,
                                           last_ret,
                                           repeat, STRIDE_ONE, STRIDE_ONE,
                                           STRIDE_ONE,
                                           REPEAT_STRIDE_EIGHT,
                                           REPEAT_STRIDE_EIGHT,
                                           REPEAT_STRIDE_EIGHT)
                    idx = [o_cycle, e_cycle + out_offset,
                           m_cycle * self.mov_len]
                    self.t_dma_trans_out(last_ret, last_ori, idx, HEAD)

                elif self.ctype == LOGSUMEXP_TYPE:
                    self.tik_instance.vexp(self.mask, input_x_ub, input_x_ub,
                                           repeat, STRIDE_ONE, STRIDE_ONE,
                                           REPEAT_STRIDE_EIGHT,
                                           REPEAT_STRIDE_EIGHT)
                    if not self.exclusive:
                        with self.tik_instance.if_scope(e_cycle == VALUE_ONE):
                            self.tik_instance.vexp(self.mask, last_ret,
                                                   last_ret, repeat,
                                                   STRIDE_ONE, STRIDE_ONE,
                                                   REPEAT_STRIDE_EIGHT,
                                                   REPEAT_STRIDE_EIGHT)
                    self.tik_instance.vadd(self.mask, last_ret, input_x_ub,
                                           last_ret, repeat,
                                           STRIDE_ONE, STRIDE_ONE, STRIDE_ONE,
                                           REPEAT_STRIDE_EIGHT,
                                           REPEAT_STRIDE_EIGHT,
                                           REPEAT_STRIDE_EIGHT)
                    # because last_ori only used in uint8 or int8
                    # and logsumexp need not to support the above dtype
                    # so use it to store the vln value
                    self.tik_instance.vln(self.mask, last_ori, last_ret,
                                          repeat,
                                          STRIDE_ONE, STRIDE_ONE,
                                          REPEAT_STRIDE_EIGHT,
                                          REPEAT_STRIDE_EIGHT)

                    idx = [o_cycle, e_cycle + out_offset,
                           m_cycle * self.mov_len]
                    self.t_dma_trans_out(last_ori, last_ori, idx, HEAD)

    def handle_mov_tail(self, o_cycle):
        """
        Calculation Process

        Parameters
        ----------
        o_cycle: outer cycle

        Returns
        -------
        None

        """
        # ub tensor
        input_x_ub = self.tik_instance. \
            Tensor(self.rdtype,
                   (MAX_COMPUTE_SIZE // self.rdsize + VECTOR_BYTE_SIZE,),
                   name="input_x_ub",
                   scope=tik.scope_ubuf)
        last_ret = self.tik_instance. \
            Tensor(self.rdtype,
                   (MAX_COMPUTE_SIZE // self.rdsize + VECTOR_BYTE_SIZE,),
                   name="last_ret",
                   scope=tik.scope_ubuf)
        last_ori = self.tik_instance. \
            Tensor(self.dtype,
                   (MAX_COMPUTE_SIZE // self.dsize + VECTOR_BYTE_SIZE,),
                   name="last_ori",
                   scope=tik.scope_ubuf)
        burstlen = self.get_burlen_by_mlen(self.mov_tail)
        repeat = self.get_repeat(self.mov_tail)
        if self.exclusive and not self.reverse:
            idx = [o_cycle, VALUE_ZERO, self.mov_loop * self.mov_len]
            self.t_vdup_to_gm(last_ret, last_ori, idx, TAIL)

        elif not self.exclusive and not self.reverse:
            idx = [o_cycle, VALUE_ZERO, self.mov_loop * self.mov_len]
            self.t_dma_in(last_ret, last_ori, idx, burstlen)
            self.t_dma_direct_out(last_ret, last_ori, idx, TAIL)

        elif not self.exclusive and self.reverse:
            idx = [o_cycle, self.each_loop - VALUE_ONE,
                   self.mov_loop * self.mov_len]
            self.t_dma_in(last_ret, last_ori, idx, burstlen)
            self.t_dma_direct_out(last_ret, last_ori, idx, TAIL)

        elif self.exclusive and self.reverse:
            idx = [o_cycle, self.each_loop - VALUE_ONE,
                   self.mov_loop * self.mov_len]
            self.t_vdup_to_gm(last_ret, last_ori, idx, TAIL)

        tmp_ori = self.tik_instance. \
            Tensor(self.dtype,
                   (BLOCK_SIZE,),
                   name="tmp_ori",
                   scope=tik.scope_ubuf)

        if self.each_loop == VALUE_ONE and self.exclusive:
            self.tik_instance.data_move(tmp_ori, self.input_x_gm,
                                        VALUE_ZERO, VALUE_ONE,
                                        VALUE_ONE, STRIDE_ZERO, STRIDE_ZERO)

        with self.tik_instance.for_range(VALUE_ONE, self.each_loop) as e_cycle:
            in_offset, out_offset = self.get_offset(e_cycle)

            idx = [o_cycle, e_cycle + in_offset, self.mov_loop * self.mov_len]
            self.t_dma_in(input_x_ub, last_ori, idx, burstlen)

            if self.ctype == SUM_TYPE:
                self.tik_instance.vadd(self.mask, last_ret, input_x_ub, last_ret,
                                       repeat, STRIDE_ONE, STRIDE_ONE,
                                       STRIDE_ONE,
                                       REPEAT_STRIDE_EIGHT, REPEAT_STRIDE_EIGHT,
                                       REPEAT_STRIDE_EIGHT)
                idx = [o_cycle, e_cycle + out_offset,
                       self.mov_loop * self.mov_len]
                self.t_dma_trans_out(last_ret, last_ori, idx, TAIL)

            elif self.ctype == PROD_TYPE:
                self.tik_instance.vmul(self.mask, last_ret, input_x_ub, last_ret,
                                       repeat, STRIDE_ONE, STRIDE_ONE,
                                       STRIDE_ONE,
                                       REPEAT_STRIDE_EIGHT, REPEAT_STRIDE_EIGHT,
                                       REPEAT_STRIDE_EIGHT)
                idx = [o_cycle, e_cycle + out_offset,
                       self.mov_loop * self.mov_len]
                self.t_dma_trans_out(last_ret, last_ori, idx, TAIL)
            elif self.ctype == LOGSUMEXP_TYPE:
                self.tik_instance.vexp(self.mask, input_x_ub, input_x_ub,
                                       repeat, STRIDE_ONE, STRIDE_ONE,
                                       REPEAT_STRIDE_EIGHT,
                                       REPEAT_STRIDE_EIGHT)

                if not self.exclusive:
                    with self.tik_instance.if_scope(e_cycle == VALUE_ONE):
                        self.tik_instance.vexp(self.mask, last_ret, last_ret,
                                               repeat,
                                               STRIDE_ONE, STRIDE_ONE,
                                               REPEAT_STRIDE_EIGHT,
                                               REPEAT_STRIDE_EIGHT)
                self.tik_instance.vadd(self.mask, last_ret, input_x_ub,
                                       last_ret, repeat,
                                       STRIDE_ONE, STRIDE_ONE, STRIDE_ONE,
                                       REPEAT_STRIDE_EIGHT,
                                       REPEAT_STRIDE_EIGHT,
                                       REPEAT_STRIDE_EIGHT)
                self.tik_instance.vln(self.mask, last_ori, last_ret,
                                      repeat,
                                      STRIDE_ONE, STRIDE_ONE,
                                      REPEAT_STRIDE_EIGHT,
                                      REPEAT_STRIDE_EIGHT)
                idx = [o_cycle, e_cycle + out_offset,
                       self.mov_loop * self.mov_len]
                self.t_dma_trans_out(last_ret, last_ori, idx, TAIL)


def get_computer_by_ctype(input_x, axis, kernel_name, ctype):
    """
    Obtain the computer template.

    Parameters
    ----------
    input_x: dict, shape and dtype
    axis: the cumulative axis
    kernel_name: kernel name
    ctype: computer type

    Returns
    -------
    the instance of computer template

    """

    return CumComputer(input_x, axis, kernel_name, ctype)
