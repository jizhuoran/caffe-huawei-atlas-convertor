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

swap_ci
"""

from te import tik
from topi.cce import util


# data type of fp16
FP16 = "float16"
# data type of fp32
FP32 = "float32"
# data type of int32
INT32 = "int32"
# one block size takes up 32b
BLOCK_SIZE = 32
# instruction's default sid is 0
SID = 0
# C0 is 16
C0 = 16
# the max stride of data move instruction
MAX_GAP_SIZE = 65536
# 16K size
UB_16K_SIZE = 16*1024
# length of fp16 and fp32 data type
TYPE_LEN_DICT = {FP16: 2, FP32: 4}
# number of element of fp16 and fp32 data type in one block
BLOCK_ELEM_NUM = {FP16: 16, FP32: 8}
# number of element of fp16 and fp32 data type in one vector
VEC_ELEM_NUM = {FP16: 128, FP32: 64}

# one VA register is 128 bits, and is evenly divided into 8 parts
TWO_VA_EIGHT_PART = 8*2
# digit 4
DIGIT_4 = 4
# digit 5
DIGIT_5 = 5
# digit 128
DIGIT_128 = 128


def ceil_value(value, factor):
    """
    if not divide exactly then plus 1

    Parameters
    ----------
    value:  input number
    factor: factor

    Returns
    -------
    ceil value
    """
    return (value + factor - 1) // factor


def align_value(value, factor):
    """
    Alignment based on factor.

    Parameters
    ----------
    value: input number
    factor: alignment base

    Returns
    -------
    res:
    """
    return (value + factor - 1) // factor*factor


# pylint: disable=too-many-instance-attributes, too-few-public-methods
class SwapParams():
    """
    Function: class that set swap_ci Parameters
    """
    def __init__(self):
        """
        constructor of SwapParams

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.loop_num_out = 0
        self.src_one_batch_offset = 0
        self.dst_one_batch_offset = 0

        # size of 16 HW after padding
        self.input_c0_size = 0
        self.input_c1 = 0
        self.channel_gap = 0
        self.cube_channel_distance = 0
        self.dest_offset = 0

        # stage1 parameters
        self.loop_num = 0
        self.inner_loop = 0
        self.load_size = 0
        self.load_size_l = 0
        self.hw_is_aligned = False
        self.s1_bursts = 0
        self.s1_burst_len = 0
        self.s1_burst_len_l = 0
        self.s1_src_stride = 0
        self.s1_src_stride_l = 0
        self.s1_dest_stride = 0
        self.s1_tail = 0

        # stage2 parameters
        self.s2_repeat = 0
        self.s2_src_rep_stride = 0
        self.s2_dest_rep_stride = 0

        self.s2_repeat_l = 0
        self.s2_src_rep_stride_l = 0
        self.s2_dest_rep_stride_l = 0

        self.s2_va_src_idx = [0]*TWO_VA_EIGHT_PART
        self.s2_va_src_idx_l = [0]*TWO_VA_EIGHT_PART
        self.s2_va_dst_idx = [0]*TWO_VA_EIGHT_PART

        # stage3 parameters
        self.s3_src_stride = 0
        self.s3_dest_stride = 0
        self.s3_bursts = 0
        self.s3_burst_len = 0
        self.s3_burst_len_l = 0


# pylint: disable=invalid-name, too-many-arguments
# pylint: disable=too-many-locals, too-many-instance-attributes
class SwapClass():
    """
    Function: class that execute swap_ci
    """
    def input_param_check(self, profile):
        """
        check if the inputs are valid

        Parameters
        ----------
        profile: Dprofile, ai_core profile explanation

        Returns
        -------
        None
        """
        product_name = profile.get_product_name()
        if product_name in ("mini", "cloud", "hisi-es"):
            util.check_dtype_rule(self.dtype, (FP16,))
            util.check_dtype_rule(self.y_dtype, (FP16,))
        else:
            util.check_dtype_rule(self.dtype, (FP16, FP32))
            util.check_dtype_rule(self.y_dtype, (FP16, FP32))

        if self.dtype != self.y_dtype:
            raise RuntimeError("dtype in x and y must be equal")

        util.check_shape_rule(self.x_shape)
        util.check_tensor_shape_size(self.x_shape)
        util.check_shape_rule(self.y_shape)
        util.check_tensor_shape_size(self.y_shape)
        util.check_kernel_name(self.kernel_name)
        # x must be 4D, NCHW
        if len(self.x_shape) != DIGIT_4:
            raise RuntimeError("input params check error,"
                               " x shape must be 4D: NCHW")
        if len(self.y_shape) != DIGIT_5:
            raise RuntimeError("input params check error, y shape must be 5HD")

        if self.group_size >= DIGIT_128:
            raise RuntimeError("input params check error,"
                               " group_size must be less than 128")

        calc_c = self.output_dim*self.group_size*self.group_size
        if self.x_shape[1] != calc_c and \
                self.x_shape[1] != align_value(calc_c, C0):
            raise RuntimeError("input_param_check, input fm channel number"
                               " does not match layer parameters,", calc_c)
        if self.x_shape[0] != self.y_shape[0] or \
                self.x_shape[2] != self.y_shape[2] or \
                self.x_shape[3] != self.y_shape[3] or self.y_shape[1] != \
                ceil_value(self.output_dim, C0)*self.group_size*self.group_size:
            raise RuntimeError("input params check error,"
                               " x shape and y shape is not match")

    def __init__(self, x_dict, y_dict, param_tup, params_obj):
        """
        constructor of SwapClass

        Parameters
        ----------
        x_dict: dict describes input fm, nchw
        y_dict: output size and data type, 5HD
        param_tup: contain output_dim, group_size, and kernel_name
            output_dim: number of output channels for psroipooling
            group_size: number of groups encoding position sensitive score maps
            kernel_name: kernel name of swap_ci op
        params_obj: SwapParams Class object

        Returns
        -------
        None
        """
        self.x_shape = x_dict["shape"]
        self.dtype = x_dict["dtype"].lower()
        self.y_shape = y_dict["shape"]
        self.y_dtype = y_dict["dtype"].lower()
        self.output_dim = param_tup[0]
        self.group_size = param_tup[1]
        self.kernel_name = param_tup[2]
        self.params = params_obj

        profile = tik.Dprofile()
        self.input_param_check(profile)

        self.dsize = TYPE_LEN_DICT[self.dtype]
        self.fm_batch = self.x_shape[0]
        self.fm_c = self.x_shape[1]
        self.fm_h = self.x_shape[2]
        self.fm_w = self.x_shape[3]
        self.hw = self.fm_h*self.fm_w
        self.x_shape_size = self.fm_batch*self.fm_c*self.hw
        self.y_shape_size = self.y_shape[0]*self.y_shape[1]*self.y_shape[2]*\
                            self.y_shape[3]*self.y_shape[4]

        self.k2 = self.group_size*self.group_size
        self.vec_elem_num = VEC_ELEM_NUM[self.dtype]
        self.mask = self.vec_elem_num
        self.ub_size = profile.get_unified_buffer_size()
        # divide the available UB space into four parts
        one_buf = (self.ub_size - UB_16K_SIZE) // 4
        self.ub_one_buf = one_buf if (one_buf % BLOCK_SIZE == 0) \
            else (one_buf // BLOCK_SIZE*BLOCK_SIZE)
        self.ub_one_buf_elem = self.ub_one_buf // self.dsize
        self.tik_instance = tik.Tik(profile, disable_debug=True)
        self.aicore_num = profile.get_aicore_num()

        self.x = None
        self.y = None

        self.init_parameters()

    def init_parameters(self):
        """
        init parameters

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.params.loop_num_out = self.k2
        self.params.src_one_batch_offset = self.fm_c*self.hw
        self.params.dst_one_batch_offset = self.k2*align_value(self.output_dim,
                                                               C0)*self.hw

        tail = self.output_dim - self.output_dim // C0*C0
        self.params.s1_tail = tail if tail > 0 else C0
        # size of 16 HW after padding
        self.params.input_c0_size = align_value(self.hw, \
                BLOCK_ELEM_NUM[self.dtype])*C0*self.dsize
        self.params.input_c1 = ceil_value(self.output_dim, C0)
        self.params.channel_gap = (self.k2 - 1)*self.hw
        # the distance between the first 16 groups and the second 16 groups in
        # the original output_dim groups (each group has k^2 HW)
        self.params.cube_channel_distance = C0*(self.k2*self.hw)
        # Offset between two groups of k^2 groups of results
        self.params.dest_offset = self.params.input_c1*C0*self.hw

    def stage1_params_no_slice(self):
        """
        set parameters of moving data from out to ub (stage1),
        HW can move in at one time.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        params = self.params
        params.loop_num = params.input_c1
        params.inner_loop = 1
        params.load_size = params.input_c0_size
        params.load_size_l = params.input_c0_size

        if self.hw*self.dsize % BLOCK_SIZE == 0 and \
                params.channel_gap*self.dsize // BLOCK_SIZE < MAX_GAP_SIZE:
            params.hw_is_aligned = True
            params.s1_bursts = C0
            params.s1_bursts_l = params.s1_tail
            params.s1_burst_len = self.hw*self.dsize // BLOCK_SIZE
            params.s1_burst_len_l = params.s1_burst_len
            params.s1_src_stride = params.channel_gap*self.dsize // BLOCK_SIZE
            params.s1_src_stride_l = params.s1_src_stride
            params.s1_dest_stride = 0
        else:
            # loop C0 times, one burst per loop
            params.hw_is_aligned = False
            params.s1_bursts = 1
            params.s1_bursts_l = 1
            params.s1_burst_len = align_value(self.hw, \
                    BLOCK_ELEM_NUM[self.dtype])*self.dsize // BLOCK_SIZE
            params.s1_burst_len_l = params.s1_burst_len
            params.s1_src_stride = 0
            params.s1_src_stride_l = 0
            params.s1_dest_stride = 0

    def stage1_params_need_slice(self):
        """
        set parameters of moving data from out to ub (stage1),
        HW can not move in at one time, need slice.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        params = self.params
        params.inner_loop = ceil_value(params.input_c0_size, self.ub_one_buf)
        params.loop_num = params.input_c1*params.inner_loop
        params.load_size = self.ub_one_buf
        params.load_size_l = params.load_size \
            if (params.input_c0_size % params.load_size == 0) \
            else (params.input_c0_size % params.load_size)

        params.s1_burst_len = (params.load_size_l // C0) // BLOCK_SIZE
        # the last src_stride is bigger when load_size_l!=0
        params.s1_src_stride = params.channel_gap*self.dsize // BLOCK_SIZE + \
                (self.hw*self.dsize // BLOCK_SIZE - params.s1_burst_len)
        if self.hw*self.dsize % BLOCK_SIZE == 0 and \
                params.s1_src_stride < MAX_GAP_SIZE:
            params.hw_is_aligned = True
            params.s1_bursts = C0
            params.s1_bursts_l = params.s1_tail
            # HW needs to split, part of HW is moved each time
            params.s1_burst_len = (params.load_size // C0) // BLOCK_SIZE
            params.s1_src_stride = params.channel_gap*self.dsize // BLOCK_SIZE \
                    + (self.hw*self.dsize // BLOCK_SIZE - params.s1_burst_len)
            params.s1_burst_len_l = (params.load_size_l // C0) // BLOCK_SIZE
            params.s1_src_stride_l = params.channel_gap*self.dsize//BLOCK_SIZE \
                    + (self.hw*self.dsize // BLOCK_SIZE - params.s1_burst_len_l)
            params.s1_dest_stride = 0
        else:
            params.hw_is_aligned = False
            params.s1_bursts = 1
            params.s1_bursts_l = 1
            params.s1_burst_len = (params.load_size // C0) // BLOCK_SIZE
            params.s1_src_stride = 0
            params.s1_src_stride_l = 0
            params.s1_burst_len_l = (params.load_size_l // C0) // BLOCK_SIZE
            params.s1_dest_stride = 0

    def set_params_stage1(self):
        """
        set parameters of moving data from out to ub (stage1).

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.params.input_c0_size <= self.ub_one_buf:
            self.stage1_params_no_slice()
        else:
            self.stage1_params_need_slice()

    def set_params_stage2(self):
        """
        set parameters of vnchwconv (stage2),

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        params = self.params
        # parameters of xt register for vnchwconv
        # size of one HW or part of one HW after padding
        s2_one_vec_size = params.load_size // C0
        params.s2_repeat = s2_one_vec_size // BLOCK_SIZE
        params.s2_repeat = 1 if params.s2_repeat == 0 else params.s2_repeat
        params.s2_src_rep_stride = 0 if params.s2_repeat == 1 else 1
        params.s2_dest_rep_stride = 0 if params.s2_repeat == 1 else 16

        s2_one_vec_size_l = params.load_size_l // C0
        params.s2_repeat_l = s2_one_vec_size_l // BLOCK_SIZE
        params.s2_repeat_l = 1 if params.s2_repeat_l == 0 \
                else params.s2_repeat_l
        params.s2_src_rep_stride_l = 0 if params.s2_repeat_l == 1 else 1
        params.s2_dest_rep_stride_l = 0 if params.s2_repeat_l == 1 else 16

        for i in range(TWO_VA_EIGHT_PART):
            params.s2_va_src_idx[i] = i*(s2_one_vec_size // self.dsize)
            params.s2_va_src_idx_l[i] = i*(s2_one_vec_size_l // self.dsize)
            params.s2_va_dst_idx[i] = i*(BLOCK_SIZE // self.dsize)

    def set_params_stage3(self):
        """
        set parameters of moving data from ub to out (stage3).

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        params = self.params
        if params.input_c0_size <= self.ub_one_buf:
            params.s3_src_stride = 0
            params.s3_dest_stride = 0
            params.s3_bursts = 1
            params.s3_burst_len = self.hw*C0*self.dsize // BLOCK_SIZE
        else:
            params.s3_src_stride = 0
            params.s3_dest_stride = 0
            params.s3_bursts = 1
            params.s3_burst_len = params.load_size // BLOCK_SIZE
            load_size_l_real = self.hw*self.dsize*C0 - \
                               params.load_size*(params.inner_loop - 1)
            params.s3_burst_len_l = load_size_l_real // BLOCK_SIZE

    def set_swap_ci_params(self):
        """
        set parameters

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.set_params_stage1()
        self.set_params_stage2()
        self.set_params_stage3()

    def process_stage1(self, batch_id, loop_index, input_buf_ub):
        """
        move data from out to ub (stage1 process)

        Parameters
        ----------
        batch_id: the batch id of input fm
        loop_index: the loop index
        input_buf_ub: the ub that store fm of data move

        Returns
        -------
        None
        """
        params = self.params
        loop_i = self.tik_instance.Scalar(INT32, name="loop_i")
        loop_i.set_as(loop_index % params.loop_num)
        is_last = self.tik_instance.Scalar(INT32, name="is_last")
        is_last.set_as((loop_i + 1) % params.inner_loop)

        dest_stride = params.s1_dest_stride
        loop_sub_g = self.tik_instance.Scalar(INT32, name="loop_sub_g")
        loop_sub_g.set_as(loop_i / params.inner_loop)
        bursts = self.tik_instance.Scalar(INT32, name="bursts")
        move_times = self.tik_instance.Scalar(INT32, name="move_times")
        with self.tik_instance.if_scope(loop_sub_g == (params.input_c1 - 1)):
            bursts.set_as(params.s1_bursts_l)
            move_times.set_as(params.s1_tail)
        with self.tik_instance.else_scope():
            bursts.set_as(params.s1_bursts)
            move_times.set_as(C0)

        burst_len = self.tik_instance.Scalar(INT32, name="burst_len")
        src_stride = self.tik_instance.Scalar(INT32, name="src_stride")
        with self.tik_instance.if_scope(is_last != 0):
            burst_len.set_as(params.s1_burst_len)
            src_stride.set_as(params.s1_src_stride)
        with self.tik_instance.else_scope():
            burst_len.set_as(params.s1_burst_len_l)
            src_stride.set_as(params.s1_src_stride_l)

        addr_index = self.tik_instance.Scalar(INT32, name="addr_index")
        # HW not need slice
        if params.inner_loop == 1:
            if params.hw_is_aligned:
                # loop_index/loop_num is loop group
                addr_index.set_as(batch_id*params.src_one_batch_offset +
                                  loop_i*params.cube_channel_distance +
                                  (loop_index / params.loop_num)*self.hw)
                self.tik_instance.data_move(input_buf_ub, self.x[addr_index],
                                            SID, bursts, burst_len,
                                            src_stride, dest_stride)
            else:
                ub_addr_stride = (params.input_c0_size // C0) // self.dsize
                with self.tik_instance.for_range(0, move_times) as mov_i:
                    addr_index.set_as(batch_id*params.src_one_batch_offset +
                                      loop_i*params.cube_channel_distance +
                                      (loop_index / params.loop_num)*self.hw +
                                      self.k2*mov_i*self.hw)
                    self.tik_instance.data_move(
                        input_buf_ub[ub_addr_stride*mov_i], self.x[addr_index],
                        SID, bursts, burst_len, src_stride, dest_stride)
        else:
            inner_loop_i = self.tik_instance.Scalar(INT32, name="inner_loop_i")
            inner_loop_i.set_as(loop_i % params.inner_loop)

            # loop_i/inner_loop is loop sub group
            if params.hw_is_aligned:
                addr_index.set_as(batch_id*params.src_one_batch_offset + \
                    (loop_i / params.inner_loop)*params.cube_channel_distance +\
                    (loop_index / params.loop_num)*self.hw + \
                    (params.s1_burst_len*BLOCK_SIZE//self.dsize)*inner_loop_i)
                self.tik_instance.data_move(input_buf_ub, self.x[addr_index],
                                            SID, bursts, burst_len,
                                            src_stride, dest_stride)
            else:
                ub_addr_stride = self.tik_instance.Scalar(INT32,
                                                          name="ub_addr_stride")
                with self.tik_instance.if_scope(is_last != 0):
                    ub_addr_stride.set_as((params.load_size//C0) // self.dsize)
                with self.tik_instance.else_scope():
                    ub_addr_stride.set_as(params.load_size_l//C0 // self.dsize)

                with self.tik_instance.for_range(0, move_times) as mov_i:
                    addr_index.set_as(batch_id*params.src_one_batch_offset + \
                            (loop_i / params.inner_loop)* \
                                    params.cube_channel_distance + \
                            (loop_index / params.loop_num)*self.hw + \
                            self.k2*mov_i*self.hw + \
                            params.s1_burst_len*BLOCK_SIZE // self.dsize* \
                                    inner_loop_i)
                    self.tik_instance.data_move(
                        input_buf_ub[ub_addr_stride*mov_i], self.x[addr_index],
                        SID, bursts, burst_len, src_stride, dest_stride)

    def process_stage2(self, loop_index, src_ub, dst_ub):
        """
        transposes with the vnchwconv instruction (stage2 process)

        Parameters
        ----------
        loop_index: the loop index
        src_ub: the ub that store input fm of data move
        dst_ub: the ub that store output fm

        Returns
        -------
        None
        """
        params = self.params
        loop_i = self.tik_instance.Scalar(INT32, name="loop_i")
        loop_i.set_as(loop_index % params.loop_num)
        is_last = self.tik_instance.Scalar(INT32, name="is_last")
        is_last.set_as((loop_i + 1) % params.inner_loop)

        dst_list = [dst_ub[params.s2_va_dst_idx[i]] for i in
                    range(TWO_VA_EIGHT_PART)]

        with self.tik_instance.if_scope(is_last != 0):
            src_list = [src_ub[params.s2_va_src_idx[i]] for i in
                        range(TWO_VA_EIGHT_PART)]
            self.tik_instance.vnchwconv(False, False, dst_list, src_list, \
                    params.s2_repeat, params.s2_dest_rep_stride, \
                    params.s2_src_rep_stride)
        with self.tik_instance.else_scope():
            src_list_l = [src_ub[params.s2_va_src_idx_l[i]] for i in
                          range(TWO_VA_EIGHT_PART)]
            self.tik_instance.vnchwconv(False, False, dst_list, src_list_l,
                                        params.s2_repeat_l,
                                        params.s2_dest_rep_stride_l,
                                        params.s2_src_rep_stride_l)

    def process_stage3(self, batch_id, loop_index, output_buf_ub):
        """
        move result data from ub to out (stage3 process)

        Parameters
        ----------
        batch_id: the batch id of input fm
        loop_index: the loop index
        output_buf_ub: the ub that store output fm

        Returns
        -------
        None
        """
        params = self.params
        addr_index = self.tik_instance.Scalar(INT32, name="addr_index")
        if params.inner_loop == 1:
            addr_index.set_as(batch_id*params.dst_one_batch_offset + \
                              loop_index*self.hw*C0)
            self.tik_instance.data_move(self.y[addr_index], output_buf_ub, \
                    SID, params.s3_bursts, params.s3_burst_len, \
                    params.s3_src_stride, params.s3_dest_stride)
        else:
            loop_i = self.tik_instance.Scalar(INT32, name="loop_i")
            loop_i.set_as(loop_index % params.loop_num)
            is_last = self.tik_instance.Scalar(INT32, name="is_last")
            is_last.set_as((loop_i + 1) % params.inner_loop)

            inner_loop_i = self.tik_instance.Scalar(INT32, name="inner_loop_i")
            inner_loop_i.set_as(loop_i % params.inner_loop)

            addr_index.set_as(batch_id*params.dst_one_batch_offset + \
                    (loop_index / params.loop_num)*params.dest_offset + \
                    (loop_i / params.inner_loop)*self.hw*C0 + \
                    (params.s3_burst_len*BLOCK_SIZE // self.dsize)*inner_loop_i)

            with self.tik_instance.if_scope(is_last != 0):
                self.tik_instance.data_move(self.y[addr_index], output_buf_ub,
                                            SID, params.s3_bursts,
                                            params.s3_burst_len,
                                            params.s3_src_stride,
                                            params.s3_dest_stride)
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(self.y[addr_index], output_buf_ub,
                                            SID, params.s3_bursts,
                                            params.s3_burst_len_l,
                                            params.s3_src_stride,
                                            params.s3_dest_stride)

    def swap_ci_process(self, batch_id):
        """
        swap channel of input fm

        Parameters
        ----------
        batch_id: the batch id of input fm

        Returns
        -------
        None
        """
        loop_num_all = self.params.loop_num_out*self.params.loop_num
        # 2 means ping pong
        thread_num = 2 if loop_num_all > 1 else 1

        with self.tik_instance.for_range(0, loop_num_all,
                                         thread_num=thread_num) as loop_index:
            with self.tik_instance.new_stmt_scope():
                buf_shape = (self.ub_one_buf_elem,)
                input_buf_ub = self.tik_instance.Tensor(self.dtype, buf_shape,
                                                        name="input_buf_ub",
                                                        scope=tik.scope_ubuf)
                output_buf_ub = self.tik_instance.Tensor(self.dtype, buf_shape,
                                                         name="output_buf_ub",
                                                         scope=tik.scope_ubuf)

                self.process_stage1(batch_id, loop_index, input_buf_ub)
                self.process_stage2(loop_index, input_buf_ub, output_buf_ub)
                self.process_stage3(batch_id, loop_index, output_buf_ub)

    def swap_ci_compute(self):
        """
        compute of swap_ci.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.set_swap_ci_params()

        if self.fm_batch > self.aicore_num:
            block_num = self.aicore_num
            outer_loop = self.fm_batch // self.aicore_num
            outer_tail = self.fm_batch % self.aicore_num
        else:
            block_num = self.fm_batch
            outer_loop = 0
            outer_tail = self.fm_batch

        with self.tik_instance.for_range(0, block_num,
                                         block_num=block_num) as block_i:
            # process one batch in one aicore
            if outer_loop > 0:
                with self.tik_instance.for_range(0, outer_loop) as loop_i:
                    self.swap_ci_process(loop_i * block_num + block_i)

            if outer_tail > 0:
                with self.tik_instance.if_scope(block_i < outer_tail):
                    self.swap_ci_process(outer_loop * block_num + block_i)

    def swap_ci_main(self):
        """
        Main process of swap_ci.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.x = self.tik_instance.Tensor(self.dtype, self.x_shape, \
                name="x", scope=tik.scope_gm).reshape((self.x_shape_size,))
        self.y = self.tik_instance.Tensor(self.dtype, shape=self.y_shape, \
                name="y", scope=tik.scope_gm).reshape((self.y_shape_size,))

        self.swap_ci_compute()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.x,),
                                   outputs=(self.y,))


@util.check_input_type(dict, dict, int, int, str)
def swap_ci(x_dict, y_dict, output_dim, group_size, kernel_name="swap_ci"):
    """
    swap_ci interface.

    Parameters
    ----------
    x_dict: feature map size and data type, NCHW
    y_dict: output size and data type, 5HD
    output_dim: number of output channels for psroipooling
    group_size: number of groups encoding position sensitive score maps
    kernel_name: kernel name of swap_ci op

    Returns
    -------
    tik_instance
    """
    swap_params = SwapParams()
    param_tup = (output_dim, group_size, kernel_name)
    swap_class = SwapClass(x_dict, y_dict, param_tup, swap_params)
    swap_class.swap_ci_main()

    return swap_class.tik_instance
