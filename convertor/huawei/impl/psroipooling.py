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

PSROIPooling
"""

from te import tik
from topi.cce import util
from te import platform as tbe_platform


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
# C0 is 16 in davinci
C0 = 16
# instruction mask 64
MASK64 = 64
# one burst
BURST_1 = 1
# default repeat time
REPEAT_1 = 1
# repeat 2 time
REPEAT_2 = 2
# repeat 4 time
REPEAT_4 = 4
# stride zero
STRIDE_ZERO = 0
# stride one
STRIDE_ONE = 1
# default repeat stride length
REP_STRIDE_EIGHT = 8
# default deqscale in vconv instruction
DEQSCALE = 1.0
# the max stride of data move instruction
MAX_GAP_SIZE = 65536
# length of fp16 and fp32 data type
TYPE_LEN_DICT = {FP16: 2, FP32: 4}
# number of element of fp16 and fp32 data type in one block
BLOCK_ELEM_NUM = {FP16: 16, FP32: 8}
# number of element of fp16 and fp32 data type in one vector
VEC_ELEM_NUM = {FP16: 128, FP32: 64}
# repeat times of fp16 and fp32 data type in vconv instruction
REP_TIMES = {FP16: 2, FP32: 1}
# repeat stride of fp16 and fp32 data type in vconv instruction
REP_STRIDE = {FP16: 4, FP32: 8}

# digit 256
DIGIT_256 = 256
# digit 255
DIGIT_255 = 255
# digit 128
DIGIT_128 = 128
# digit 64
DIGIT_64 = 64
# digit 4
DIGIT_4 = 4
# digit 5
DIGIT_5 = 5
# digit 2
DIGIT_2 = 2
# digit 8
DIGIT_8 = 8
# 0.1
POINT_1 = 0.1
# 0.5
POINT_5 = 0.5
# 1.0
ONE_POINT = 1.0
# neg two
NEG_TWO = -2
# neg one
NEG_ONE = -1


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


# pylint: disable=invalid-name, too-many-locals, too-many-arguments
# pylint: disable=too-many-instance-attributes, too-many-lines
class PsroiClass():
    """
    Function: class that execute psroipooling
    """
    def input_param_check(self, product_name):
        """
        check if the inputs are valid

        Parameters
        ----------
        profile: Dprofile, ai_core profile explanation

        Returns
        -------
        None
        """
        if product_name in ("Ascend310", "Ascend910", "Hi3796CV300ES"):
            util.check_dtype_rule(self.dtype, (FP16,))
            util.check_dtype_rule(self.roi_dtype, (FP16,))
        else:
            util.check_dtype_rule(self.dtype, (FP16, FP32))
            util.check_dtype_rule(self.roi_dtype, (FP16, FP32))

        if self.dtype != self.roi_dtype or self.dtype != self.y_dtype:
            raise RuntimeError("dtype in x, rois and y must be equal")

        util.check_shape_rule(self.x_shape)
        util.check_tensor_shape_size(self.x_shape)
        util.check_shape_rule(self.roi_shape)
        util.check_tensor_shape_size(self.roi_shape)
        util.check_shape_rule(self.y_shape)
        util.check_tensor_shape_size(self.y_shape)
        util.check_kernel_name(self.kernel_name)
        # x and y must be 5HD
        if len(self.x_shape) != DIGIT_5:
            raise RuntimeError("input params check error, x shape must be 5HD")
        if len(self.y_shape) != DIGIT_5:
            raise RuntimeError("input params check error, y shape must be 5HD")
        if self.roi_shape[0] != self.x_shape[0]:
            raise RuntimeError("input params check error, rois batch must be"
                               " equal x batch")
        if self.roi_shape[1] != DIGIT_5:
            raise RuntimeError("input params check error, rois shape[1] must"
                               " be equal 5")
        if self.roi_shape[0]*self.roi_shape[2] != self.y_shape[0]:
            raise RuntimeError("input params check error,"
                               " rois all num must be equal y shape[0]")
        if self.group_size >= DIGIT_128:
            raise RuntimeError("input params check error,"
                               " group_size must be less than 128")
        if self.x_shape[1]//self.y_shape[1] != self.y_shape[2]*self.y_shape[3]:
            raise RuntimeError("input params check error, x shape[1] is"
                               " invalid")
        if self.group_size != self.y_shape[2] or \
                self.group_size != self.y_shape[3]:
            raise RuntimeError("input params check error, y shape[2]"
                               " and shape[3] must be equal group_size")
        if ceil_value(self.output_dim, C0) != self.y_shape[1]:
            raise RuntimeError("input params check error,"
                               " output_dim is invalid")

    def __init__(self, x_dict, rois_dict, y_dict, params, kernel_name):
        """
        constructor of PsroiClass

        Parameters
        ----------
        x_dict: dict describes input fm, NC1HWC0
        rois_dict: dict describes input rois
        params: a tuple, contain output_dim, group_size, spatial_scale
        kernel_name: name of kernel

        Returns
        -------
        None
        """
        self.x_shape = x_dict["shape"]
        self.dtype = x_dict["dtype"].lower()
        self.roi_dtype = rois_dict["dtype"].lower()
        self.roi_shape = rois_dict["shape"]
        self.y_dtype = y_dict["dtype"].lower()
        self.y_shape = y_dict["shape"]
        self.output_dim = params[0]
        self.group_size = params[1]
        self.spatial_scale = params[2]
        self.kernel_name = kernel_name

        profile = tik.Dprofile()

        product_name = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
        self.input_param_check(product_name)

        self.dsize = TYPE_LEN_DICT[self.dtype]
        self.fm_batch = self.x_shape[0]
        self.fm_c1 = self.x_shape[1]
        self.fm_h = self.x_shape[2]
        self.fm_w = self.x_shape[3]
        self.fm_c0 = self.x_shape[4]
        self.fm_c = self.fm_c1*self.fm_c0
        self.hw = self.fm_h*self.fm_w
        self.x_data_size = (self.fm_batch*self.fm_c*self.hw)*self.dsize
        # roi num of one batch, roi_shape is (batch, 5, rois_num)
        self.roi_num_b = self.roi_shape[2]

        self.k2 = self.group_size*self.group_size
        self.vec_elem_num = VEC_ELEM_NUM[self.dtype]
        self.ub_size = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
        # divide the available UB space into four parts
        self.ub_one_buf = self.ub_size // 4
        self.ub_one_buf_elem = self.ub_one_buf // self.dsize
        self.l1_size = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.L1_SIZE)
        self.tik_instance = tik.Tik(profile, disable_debug=True)
        self.aicore_num = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)

        self.roi_num_step = self.vec_elem_num
        self.mask = self.roi_num_step

        # set parameters
        self.inner_c1 = self.fm_c1 // self.k2
        self.inner_c = self.inner_c1*C0
        self.inner_c_size = self.inner_c*self.dsize
        self.inner_c_offset_size = self.inner_c_size*self.hw
        self.c0_offset_size = C0*self.hw*self.dsize

        self.bin_load_stride = (self.hw*C0 - C0)*self.dsize // BLOCK_SIZE
        self.bin_load_out_stride = (self.k2*C0 - C0)*self.dsize // BLOCK_SIZE

        self.x = None
        self.rois = None
        self.y = None
        # L1 cache is used, when input x can be put down in L1
        self.cache_in_l1 = False
        self.cache_l1 = None
        self.const_0_127_ub = None

    def load_rois_to_ub(self, rois_ub, rois_offset, roi_step):
        """
        load rois data to ub from gm.

        Parameters
        ----------
        rois_ub: a tensor, which store rois data
        rois_offset: the roi offset of current loop in block_id aicore
        roi_step: the number of rois per loop in process, 64(fp32) or 128(fp16)

        Returns
        -------
        None
        """
        burst_len = roi_step * self.dsize // BLOCK_SIZE
        for i in range(DIGIT_5):
            self.tik_instance.data_move(rois_ub[i, 0], \
                                        self.rois[rois_offset + i*self.roi_num_b], SID, \
                                        BURST_1, burst_len, STRIDE_ZERO, STRIDE_ZERO)

    def spatial_scale_rois(self, rois_ub, rois_floor_ub, rois_spatial_ub,
                           roi_step):
        """
        compute the width and height of rois and bin.

        Parameters
        ----------
        rois_ub: input rois data in ub, (5, roi_step).
            batch_id,batch_id,batch_id...
            x1,x1,x1...
            y1,y1,y1...
            x2,x2,x2...
            y2,y2,y2...
        rois_floor_ub: store rois data of convert to s32
        rois_spatial_ub: store the width and height of rois and bin in ub
        roi_step: the number of rois per loop in process, 64(fp32) or 128(fp16)

        Returns
        -------
        None
        """
        point_one_ub = self.tik_instance.Tensor(self.dtype, (roi_step,),
                                                name="point_one_ub",
                                                scope=tik.scope_ubuf)
        self.tik_instance.vadds(self.mask, rois_ub[1, 0], rois_ub[1, 0],
                                POINT_5, REPEAT_4, STRIDE_ONE, STRIDE_ONE,
                                REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
        # rois_floor_ub[0]: batch id; rois_floor_ub[1-4]: roi coordinates
        # vconv.floor: f162s32r or f322s32r
        self.tik_instance.vconv(MASK64, 'floor', rois_floor_ub, rois_ub,
                                REP_TIMES[self.dtype]*DIGIT_5,
                                STRIDE_ONE, STRIDE_ONE,
                                REP_STRIDE_EIGHT, REP_STRIDE[self.dtype])
        # s322f16: vconv.deq, or s322f32: vconv
        if self.dtype == FP16:
            self.tik_instance.vconv(MASK64, '', rois_spatial_ub,
                                    rois_floor_ub[1, 0],
                                    REP_TIMES[self.dtype]*DIGIT_4,
                                    STRIDE_ONE, STRIDE_ONE,
                                    REP_STRIDE[self.dtype], REP_STRIDE_EIGHT,
                                    deqscale=DEQSCALE)
        else:
            self.tik_instance.vconv(MASK64, '', rois_spatial_ub,
                                    rois_floor_ub[1, 0],
                                    REP_TIMES[self.dtype]*DIGIT_4,
                                    STRIDE_ONE, STRIDE_ONE,
                                    REP_STRIDE[self.dtype], REP_STRIDE_EIGHT)
        self.tik_instance.vadds(self.mask, rois_spatial_ub[2, 0],
                                rois_spatial_ub[2, 0], ONE_POINT, REPEAT_2,
                                STRIDE_ONE, STRIDE_ONE,
                                REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
        # multiply spatial
        self.tik_instance.vmuls(self.mask, rois_spatial_ub, rois_spatial_ub,
                                self.spatial_scale, REPEAT_4, STRIDE_ONE,
                                STRIDE_ONE, REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)

        # roi width and height: roi_end_w-roi_start_w, roi_end_h-roi_start_h
        self.tik_instance.vsub(self.mask, rois_spatial_ub[4, 0],
                               rois_spatial_ub[2, 0], rois_spatial_ub, REPEAT_2,
                               STRIDE_ONE, STRIDE_ONE, STRIDE_ONE,
                               REP_STRIDE_EIGHT, REP_STRIDE_EIGHT,
                               REP_STRIDE_EIGHT)
        self.tik_instance.vector_dup(self.mask, point_one_ub, POINT_1, REPEAT_1,
                                     STRIDE_ONE, REP_STRIDE_EIGHT)
        self.tik_instance.vmax(self.mask, rois_spatial_ub[4, 0],
                               rois_spatial_ub[4, 0], point_one_ub, REPEAT_2,
                               STRIDE_ONE, STRIDE_ONE, STRIDE_ONE,
                               REP_STRIDE_EIGHT, REP_STRIDE_EIGHT, STRIDE_ZERO)

        pooled_k_recip = self.tik_instance.Scalar(self.dtype, \
                                                  name="pooled_k_recip", init_value=1.0 / self.group_size)
        # bin width and height
        self.tik_instance.vmuls(self.mask, rois_spatial_ub[6, 0],
                                rois_spatial_ub[4, 0], pooled_k_recip,
                                REPEAT_2, STRIDE_ONE, STRIDE_ONE,
                                REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)

    def newton_div(self, dst, divisor, dividend, repeat):
        """
        use newton_div to improve performance

        Parameters
        ----------
        dst: vdiv's dest tensor
        divisor: vdiv's src0 tensor
        dividend: vdiv's src1 tensor
        repeat: vdiv's needs repeat times

        Returns
        -------
        None
        """
        if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ("Ascend910", "Ascend620", "Ascend610"):
            self.tik_instance.vdiv(self.mask, dst, divisor, dividend, repeat,
                                   STRIDE_ONE, STRIDE_ONE, STRIDE_ONE,
                                   REP_STRIDE_EIGHT, REP_STRIDE_EIGHT,
                                   STRIDE_ZERO)
        else:
            with self.tik_instance.new_stmt_scope():
                t_tensor = self.tik_instance.Tensor(self.dtype, \
                                                    dividend.shape, name="t_tensor", scope=tik.scope_ubuf)
                self.tik_instance.vrec(self.mask, t_tensor, dividend, REPEAT_1,
                                       STRIDE_ONE, STRIDE_ONE,
                                       REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
                # Newton start
                self.tik_instance.vmul(self.mask, dividend, dividend, t_tensor,
                                       REPEAT_1, STRIDE_ONE, STRIDE_ONE,
                                       STRIDE_ONE, REP_STRIDE_EIGHT,
                                       REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
                self.tik_instance.vadds(self.mask, dividend, dividend, NEG_TWO,
                                        REPEAT_1, STRIDE_ONE, STRIDE_ONE,
                                        REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
                self.tik_instance.vmul(self.mask, dividend, dividend, t_tensor,
                                       REPEAT_1, STRIDE_ONE, STRIDE_ONE,
                                       STRIDE_ONE, REP_STRIDE_EIGHT,
                                       REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
                self.tik_instance.vmuls(self.mask, dividend, dividend, NEG_ONE,
                                        REPEAT_1, STRIDE_ONE, STRIDE_ONE,
                                        REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
                # Newton end

                # divisor * (1/dividend)
                self.tik_instance.vmul(self.mask, dst, divisor, dividend,
                                       repeat, STRIDE_ONE, STRIDE_ONE,
                                       STRIDE_ONE, REP_STRIDE_EIGHT,
                                       REP_STRIDE_EIGHT, STRIDE_ZERO)

    def process_one_bin_1(self, params):
        """
        process one bin of roi: inner_c1 > 1, bin_all_dsize <= self.ub_one_buf,
                                and bin_load_stride < MAX_GAP_SIZE
        Parameters
        ----------
        params: param is a dict, contains multiple keys

        Returns
        -------
        None
        """
        load_count_ceil = ceil_value(self.inner_c, self.vec_elem_num)
        load_count_align = load_count_ceil*self.vec_elem_num

        ub_bin_input_buf = self.tik_instance.Tensor(self.dtype, \
                                                    (self.ub_one_buf_elem,), name="ub_bin_input_buf", \
                                                    scope=tik.scope_ubuf)
        ub_bin_output_buf = self.tik_instance.Tensor(self.dtype, \
                                                     (load_count_align,), name="ub_bin_output_buf", \
                                                     scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(self.mask, ub_bin_output_buf, 0,
                                     load_count_ceil, STRIDE_ONE,
                                     REP_STRIDE_EIGHT)

        x_src = self.x
        if self.cache_in_l1:
            x_src = self.cache_l1

        # move feature map from gm to ub
        burst_len = C0*self.dsize // BLOCK_SIZE
        inner_dst_addr_stread = self.inner_c

        with self.tik_instance.for_range(0, params["h_width"]) as i:
            with self.tik_instance.for_range(0, params["w_width"]) as j:
                self.tik_instance \
                    .data_move(ub_bin_input_buf[(i*params["w_width"] + j)
                                                *inner_dst_addr_stread],
                               x_src[params["scalar_roi_batch_id"],
                                     params["bin_i_offset"]*self.inner_c1,
                                     params["h_start"]+i,
                                     params["w_start"]+j, 0],
                               SID, self.inner_c1, burst_len,
                               self.bin_load_stride, STRIDE_ZERO)

        # avg pooling
        add_loop = ceil_value(self.inner_c, self.vec_elem_num)
        for i in range(0, add_loop):
            src0_rep_stride = self.inner_c_size // BLOCK_SIZE

            with self.tik_instance.if_scope(params["bin_area"] < DIGIT_256):
                self.tik_instance.vadd(self.mask, \
                                       ub_bin_output_buf[self.vec_elem_num*i], \
                                       ub_bin_input_buf[self.vec_elem_num*i], \
                                       ub_bin_output_buf[self.vec_elem_num*i], \
                                       params["bin_area"], STRIDE_ONE, STRIDE_ONE, \
                                       STRIDE_ONE, STRIDE_ZERO, src0_rep_stride, STRIDE_ZERO)
            with self.tik_instance.else_scope():
                tail = params["bin_area"] % DIGIT_255
                times = params["bin_area"] // DIGIT_255
                with self.tik_instance.for_range(0, times) as times_i:
                    self.tik_instance.vadd(self.mask, \
                                           ub_bin_output_buf[self.vec_elem_num*i], \
                                           ub_bin_input_buf[self.vec_elem_num*i +
                                                            times_i*DIGIT_255*self.inner_c], \
                                           ub_bin_output_buf[self.vec_elem_num*i], \
                                           DIGIT_255, STRIDE_ONE, STRIDE_ONE, STRIDE_ONE, \
                                           STRIDE_ZERO, src0_rep_stride, STRIDE_ZERO)
                with self.tik_instance.if_scope(tail != 0):
                    self.tik_instance.vadd(self.mask, \
                                           ub_bin_output_buf[self.vec_elem_num*(add_loop-1)], \
                                           ub_bin_input_buf[self.vec_elem_num*(add_loop-1) +
                                                            times*DIGIT_255*self.inner_c], \
                                           ub_bin_output_buf[self.vec_elem_num*(add_loop-1)], \
                                           tail, STRIDE_ONE, STRIDE_ONE, STRIDE_ONE, \
                                           STRIDE_ZERO, src0_rep_stride, STRIDE_ZERO)

        bin_area_fp_ub = self.tik_instance.Tensor(self.dtype, \
                                                  (self.vec_elem_num,), name="bin_area_fp_ub", \
                                                  scope=tik.scope_ubuf)
        bin_area_int32 = self.tik_instance.Tensor(INT32, \
                                                  (DIGIT_64*REP_TIMES[self.dtype],), name="bin_area_int32", \
                                                  scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(DIGIT_64, bin_area_int32,
                                     params["bin_area"], REP_TIMES[self.dtype],
                                     STRIDE_ONE, REP_STRIDE_EIGHT)
        # s322f16:vconv.deq, or s322f32:vconv
        if self.dtype == FP16:
            self.tik_instance.vconv(MASK64, '', bin_area_fp_ub, bin_area_int32,
                                    REP_TIMES[self.dtype], STRIDE_ONE,
                                    STRIDE_ONE, REP_STRIDE[self.dtype],
                                    REP_STRIDE_EIGHT, deqscale=DEQSCALE)
        else:
            self.tik_instance.vconv(MASK64, '', bin_area_fp_ub, bin_area_int32,
                                    REP_TIMES[self.dtype], STRIDE_ONE,
                                    STRIDE_ONE, REP_STRIDE[self.dtype],
                                    REP_STRIDE_EIGHT)
        self.newton_div(ub_bin_output_buf, ub_bin_output_buf, bin_area_fp_ub,
                        load_count_ceil)

        # move result to gm
        self.tik_instance.data_move(self.y[params["cur_roi_output_offset"], 0,
                                           params["ph"], params["pw"], 0],
                                    ub_bin_output_buf, SID, self.inner_c1,
                                    burst_len, STRIDE_ZERO,
                                    self.bin_load_out_stride)

    def process_one_bin_2(self, params):
        """
        process one bin of roi: inner_c1 == 1, or
                                (bin_all_dsize > self.ub_one_buf, or
                                 bin_load_stride > MAX_GAP_SIZE)
        Parameters
        ----------
        params: param is a dict, contains multiple keys

        Returns
        -------
        None
        """
        bursts_s = self.tik_instance.Scalar(INT32, name="bursts_s")
        bursts_s.set_as(params["h_width"])
        burst_len_s = self.tik_instance.Scalar(INT32, name="burst_len_s")
        burst_len_s.set_as(params["w_width"]*C0*self.dsize // BLOCK_SIZE)
        src_stride_s = self.tik_instance.Scalar(INT32, name="src_stride_s")
        src_stride_s.set_as(
            (self.fm_w - params["w_width"])*C0*self.dsize // BLOCK_SIZE)

        x_src = self.x
        if self.cache_in_l1:
            x_src = self.cache_l1

        with self.tik_instance.for_range(0, self.inner_c1) as loop_i:
            load_count_ceil = ceil_value(C0, self.vec_elem_num)
            load_count_align = load_count_ceil*self.vec_elem_num
            ub_bin_input_buf = self.tik_instance.Tensor(self.dtype, \
                                                        (self.ub_one_buf_elem,), name="ub_bin_input_buf", \
                                                        scope=tik.scope_ubuf)
            ub_bin_output_buf = self.tik_instance.Tensor(self.dtype, \
                                                         (load_count_align,), name="ub_bin_output_buf", \
                                                         scope=tik.scope_ubuf)
            self.tik_instance.vector_dup(self.mask, ub_bin_output_buf, 0,
                                         load_count_ceil, STRIDE_ONE,
                                         REP_STRIDE_EIGHT)
            # move feature map from gm to ub
            self.tik_instance.data_move(ub_bin_input_buf, \
                                        x_src[params["scalar_roi_batch_id"],
                                              params["bin_i_offset"]*self.inner_c1+loop_i,
                                              params["h_start"], params["w_start"], 0], \
                                        SID, bursts_s, burst_len_s, src_stride_s, STRIDE_ZERO)

            # avg pooling
            src0_rep_stride = self.inner_c_size // BLOCK_SIZE

            with self.tik_instance.if_scope(params["bin_area"] < DIGIT_256):
                self.tik_instance.vadd(self.mask, ub_bin_output_buf, \
                                       ub_bin_input_buf, ub_bin_output_buf, \
                                       params["bin_area"], STRIDE_ONE, STRIDE_ONE, \
                                       STRIDE_ONE, STRIDE_ZERO, src0_rep_stride, STRIDE_ZERO)
            with self.tik_instance.else_scope():
                tail = params["bin_area"] % DIGIT_255
                times = params["bin_area"] // DIGIT_255
                with self.tik_instance.for_range(0, times) as times_i:
                    self.tik_instance.vadd(self.mask, ub_bin_output_buf, \
                                           ub_bin_input_buf[C0*DIGIT_255*times_i], \
                                           ub_bin_output_buf, DIGIT_255, \
                                           STRIDE_ONE, STRIDE_ONE, STRIDE_ONE, \
                                           STRIDE_ZERO, src0_rep_stride, STRIDE_ZERO)
                with self.tik_instance.if_scope(tail != 0):
                    self.tik_instance.vadd(self.mask, ub_bin_output_buf, \
                                           ub_bin_input_buf[C0*DIGIT_255*times], \
                                           ub_bin_output_buf, tail, \
                                           STRIDE_ONE, STRIDE_ONE, STRIDE_ONE, \
                                           STRIDE_ZERO, src0_rep_stride, STRIDE_ZERO)

            bin_area_fp_ub = self.tik_instance.Tensor(self.dtype, \
                                                      (self.vec_elem_num,), name="bin_area_fp_ub", \
                                                      scope=tik.scope_ubuf)
            bin_area_int32 = self.tik_instance.Tensor(INT32, \
                                                      (DIGIT_64*REP_TIMES[self.dtype],), name="bin_area_int32", \
                                                      scope=tik.scope_ubuf)
            self.tik_instance.vector_dup(MASK64, bin_area_int32, \
                                         params["bin_area"], REP_TIMES[self.dtype], STRIDE_ONE, \
                                         REP_STRIDE_EIGHT)
            # s322f16:vconv.deq, or s322f32:vconv
            if self.dtype == FP16:
                self.tik_instance.vconv(MASK64, '', bin_area_fp_ub,
                                        bin_area_int32, REP_TIMES[self.dtype],
                                        STRIDE_ONE, STRIDE_ONE,
                                        REP_STRIDE[self.dtype],
                                        REP_STRIDE_EIGHT, deqscale=DEQSCALE)
            else:
                self.tik_instance.vconv(MASK64, '', bin_area_fp_ub,
                                        bin_area_int32, REP_TIMES[self.dtype],
                                        STRIDE_ONE, STRIDE_ONE,
                                        REP_STRIDE[self.dtype],
                                        REP_STRIDE_EIGHT)

            self.newton_div(ub_bin_output_buf, ub_bin_output_buf,
                            bin_area_fp_ub, load_count_ceil)

            # move result to gm
            burst_len = C0*self.dsize // BLOCK_SIZE
            self.tik_instance.data_move(self.y[params["cur_roi_output_offset"],
                                               loop_i*C0, params["ph"],
                                               params["pw"], 0],
                                        ub_bin_output_buf, SID, BURST_1,
                                        burst_len, STRIDE_ZERO, STRIDE_ZERO)

    def process_one_bin(self, params):
        """
        process one bin of roi.

        Parameters
        ----------
        params: param is a dict, contains multiple keys

        Returns
        -------
        None
        """
        # bin area is 0
        with self.tik_instance.if_scope(params["bin_area"] == 0):
            load_count = self.inner_c
            v_dup_0_repeat = ceil_value(load_count, self.vec_elem_num)
            out_shape = (v_dup_0_repeat*self.vec_elem_num,)
            ub_bin_output_buf = self.tik_instance.Tensor(self.dtype, \
                                                         out_shape, name="ub_bin_output_buf", scope=tik.scope_ubuf)
            self.tik_instance.vector_dup(self.mask, ub_bin_output_buf, 0,
                                         v_dup_0_repeat, STRIDE_ONE,
                                         REP_STRIDE_EIGHT)
            # move result to gm
            burst_len = C0*self.dsize // BLOCK_SIZE
            dst_stride = self.bin_load_out_stride
            self.tik_instance.data_move(self.y[params["cur_roi_output_offset"], \
                                               0, params["ph"], params["pw"], 0], ub_bin_output_buf, SID, \
                                        self.inner_c1, burst_len, STRIDE_ZERO, dst_stride)

        with self.tik_instance.else_scope():
            if self.inner_c1 == 1:
                self.process_one_bin_2(params)
            else:
                with self.tik_instance.if_scope(
                        tik.all(params["bin_all_dsize"] <= self.ub_one_buf,
                                self.bin_load_stride < MAX_GAP_SIZE)):
                    self.process_one_bin_1(params)

                with self.tik_instance.else_scope():
                    # need loop self.inner_c1 time
                    self.process_one_bin_2(params)

    def process_one_roi(self, params):
        """
        process one roi.

        Parameters
        ----------
        params: param is a dict, contains multiple keys

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, self.group_size) as ph:
            params["ph"] = ph
            # h coordinates of bin
            h_start = self.tik_instance.Scalar(INT32, name="h_start")
            h_start.set_as(params["bin_start_h_floor"][ph])
            h_end = self.tik_instance.Scalar(INT32, name="h_end")
            h_end.set_as(params["bin_end_h_ceil"][ph + 1])
            params["h_start"] = h_start
            params["h_end"] = h_end

            # ping pong
            thread_num = 2
            if self.group_size == 1:
                thread_num = 1
            with self.tik_instance.for_range(0, self.group_size,
                                             thread_num=thread_num) as pw:
                params["pw"] = pw
                # w coordinates of bin
                w_start = self.tik_instance.Scalar(INT32, name="w_start")
                w_start.set_as(params["bin_start_w_floor"][pw])
                w_end = self.tik_instance.Scalar(INT32, name="w_end")
                w_end.set_as(params["bin_end_w_ceil"][pw + 1])
                params["w_start"] = w_start
                params["w_end"] = w_end

                bin_i_offset = self.tik_instance.Scalar(INT32,
                                                        name="bin_i_offset")
                # bin_i offset of in roi, 0~(group_size^2-1)
                bin_i_offset.set_as(ph * self.group_size + pw)
                params["bin_i_offset"] = bin_i_offset

                w_width = self.tik_instance.Scalar(INT32, name="w_width")
                h_width = self.tik_instance.Scalar(INT32, name="h_width")
                bin_area = self.tik_instance.Scalar(INT32, name="bin_area")
                w_width.set_as(w_end - w_start)
                with self.tik_instance.if_scope(w_end <= w_start):
                    w_width.set_as(0)
                h_width.set_as(h_end - h_start)
                with self.tik_instance.if_scope(h_end <= h_start):
                    h_width.set_as(0)
                bin_area.set_as(w_width*h_width)
                params["w_width"] = w_width
                params["h_width"] = h_width
                params["bin_area"] = bin_area

                bin_all_dsize = self.tik_instance.Scalar(INT32,
                                                         name="bin_all_dsize")
                bin_all_dsize.set_as(bin_area * self.inner_c_size)
                bin_c0_dsize = self.tik_instance.Scalar(INT32,
                                                        name="bin_c0_dsize")
                bin_c0_dsize.set_as(bin_area * C0 * self.dsize)
                params["bin_all_dsize"] = bin_all_dsize
                params["bin_c0_dsize"] = bin_c0_dsize

                self.process_one_bin(params)

    def process_step1_roi(self, rois_floor_ub, rois_spatial_ub,
                          rois_num_offset, step_i_offset, step_i_num):
        """
        process roi_num roi cyclically, and process roi_step roi each time.

        Parameters
        ----------
        rois_floor_ub: rois data, s32
        rois_spatial_ub: a tensor, the width and height of rois and bin
        step_i_offset: the roi offset of this loop in block_id aicore
        rois_num_offset: a Scalar, the offset in block_id aicore
        step_i_num: the number of rois of one loop in process

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, step_i_num) as roi_i:
            params = {}
            scalar_roi_batch_id = self.tik_instance.Scalar(INT32, \
                                                           name="scalar_roi_batch_id")
            scalar_roi_batch_id.set_as(rois_floor_ub[0, roi_i])
            params["scalar_roi_batch_id"] = scalar_roi_batch_id

            cur_roi_output_offset = self.tik_instance.Scalar(INT32, \
                                                             name="cur_roi_output_offset")
            cur_roi_output_offset.set_as(rois_num_offset + step_i_offset +
                                         roi_i)
            params["cur_roi_output_offset"] = cur_roi_output_offset

            scalar_roi_start_w = self.tik_instance.Scalar(self.dtype, \
                                                          name="scalar_roi_start_w")
            scalar_roi_start_w.set_as(rois_spatial_ub[0, roi_i])
            scalar_roi_start_h = self.tik_instance.Scalar(self.dtype, \
                                                          name="scalar_roi_start_h")
            scalar_roi_start_h.set_as(rois_spatial_ub[1, roi_i])

            scalar_bin_width = self.tik_instance.Scalar(self.dtype, \
                                                        name="scalar_bin_width")
            scalar_bin_width.set_as(rois_spatial_ub[6, roi_i])
            scalar_bin_height = self.tik_instance.Scalar(self.dtype, \
                                                         name="scalar_bin_height")
            scalar_bin_height.set_as(rois_spatial_ub[7, roi_i])

            bin_start_w_ub = self.tik_instance.Tensor(self.dtype, (DIGIT_128,), \
                                                      name="bin_start_w_ub", scope=tik.scope_ubuf)
            bin_start_w_floor = self.tik_instance.Tensor(INT32, (DIGIT_128,), \
                                                         name="bin_start_w_floor", scope=tik.scope_ubuf)
            bin_end_w_ceil = self.tik_instance.Tensor(INT32, (DIGIT_128,), \
                                                      name="bin_end_w_ceil", scope=tik.scope_ubuf)
            # scalar_roi_start_w + scalar_bin_width*(0...127)
            self.tik_instance.vmuls(self.mask, bin_start_w_ub,
                                    self.const_0_127_ub, scalar_bin_width,
                                    DIGIT_128 // self.vec_elem_num,
                                    STRIDE_ONE, STRIDE_ONE,
                                    REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
            self.tik_instance.vadds(self.mask, bin_start_w_ub, bin_start_w_ub,
                                    scalar_roi_start_w,
                                    DIGIT_128 // self.vec_elem_num,
                                    STRIDE_ONE, STRIDE_ONE,
                                    REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
            # vconv.floor: f162s32f or f322s32f
            self.tik_instance.vconv(MASK64, 'floor', bin_start_w_floor,
                                    bin_start_w_ub, DIGIT_128 // DIGIT_64,
                                    STRIDE_ONE, STRIDE_ONE, REP_STRIDE_EIGHT,
                                    REP_STRIDE[self.dtype])
            self.tik_instance.vconv(MASK64, 'ceil', bin_end_w_ceil,
                                    bin_start_w_ub, DIGIT_128 // DIGIT_64,
                                    STRIDE_ONE, STRIDE_ONE, REP_STRIDE_EIGHT,
                                    REP_STRIDE[self.dtype])

            bin_start_h_ub = self.tik_instance.Tensor(self.dtype, (DIGIT_128,), \
                                                      name="bin_start_h_ub", scope=tik.scope_ubuf)
            bin_start_h_floor = self.tik_instance.Tensor(INT32, (DIGIT_128,), \
                                                         name="bin_start_h_floor", scope=tik.scope_ubuf)
            bin_end_h_ceil = self.tik_instance.Tensor(INT32, (DIGIT_128,), \
                                                      name="bin_end_h_ceil", scope=tik.scope_ubuf)
            # scalar_roi_start_h + scalar_bin_height*(0...127)
            self.tik_instance.vmuls(self.mask, bin_start_h_ub,
                                    self.const_0_127_ub, scalar_bin_height,
                                    DIGIT_128 // self.vec_elem_num,
                                    STRIDE_ONE, STRIDE_ONE,
                                    REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
            self.tik_instance.vadds(self.mask, bin_start_h_ub, bin_start_h_ub,
                                    scalar_roi_start_h,
                                    DIGIT_128 // self.vec_elem_num,
                                    STRIDE_ONE, STRIDE_ONE,
                                    REP_STRIDE_EIGHT, REP_STRIDE_EIGHT)
            # vconv.floor: f162s32f or f322s32f
            self.tik_instance.vconv(MASK64, 'floor', bin_start_h_floor,
                                    bin_start_h_ub, DIGIT_128 // DIGIT_64,
                                    STRIDE_ONE, STRIDE_ONE, REP_STRIDE_EIGHT,
                                    REP_STRIDE[self.dtype])
            self.tik_instance.vconv(MASK64, 'ceil', bin_end_h_ceil,
                                    bin_start_h_ub, DIGIT_128 // DIGIT_64,
                                    STRIDE_ONE, STRIDE_ONE, REP_STRIDE_EIGHT,
                                    REP_STRIDE[self.dtype])

            # vmax(,0)
            dup_tmp_ub = self.tik_instance.Tensor(INT32, (DIGIT_64,), \
                                                  name="dup_tmp_ub", scope=tik.scope_ubuf)
            self.tik_instance.vector_dup(MASK64, dup_tmp_ub, 0, REPEAT_1,
                                         STRIDE_ONE, REP_STRIDE_EIGHT)
            self.tik_instance.vmax(MASK64, bin_start_w_floor, bin_start_w_floor,
                                   dup_tmp_ub, DIGIT_128 // DIGIT_64,
                                   STRIDE_ONE, STRIDE_ONE, STRIDE_ONE,
                                   REP_STRIDE_EIGHT, REP_STRIDE_EIGHT,
                                   STRIDE_ZERO)
            self.tik_instance.vmax(MASK64, bin_end_w_ceil, bin_end_w_ceil,
                                   dup_tmp_ub, DIGIT_128 // DIGIT_64,
                                   STRIDE_ONE, STRIDE_ONE, STRIDE_ONE,
                                   REP_STRIDE_EIGHT, REP_STRIDE_EIGHT,
                                   STRIDE_ZERO)
            self.tik_instance.vmax(MASK64, bin_start_h_floor, bin_start_h_floor,
                                   dup_tmp_ub, DIGIT_128 // DIGIT_64,
                                   STRIDE_ONE, STRIDE_ONE, STRIDE_ONE,
                                   REP_STRIDE_EIGHT, REP_STRIDE_EIGHT,
                                   STRIDE_ZERO)
            self.tik_instance.vmax(MASK64, bin_end_h_ceil, bin_end_h_ceil,
                                   dup_tmp_ub, DIGIT_128 // DIGIT_64,
                                   STRIDE_ONE, STRIDE_ONE, STRIDE_ONE,
                                   REP_STRIDE_EIGHT, REP_STRIDE_EIGHT,
                                   STRIDE_ZERO)

            # vmin(,width/height)
            self.tik_instance.vector_dup(MASK64, dup_tmp_ub, self.fm_w,
                                         REPEAT_1, STRIDE_ONE, REP_STRIDE_EIGHT)
            self.tik_instance.vmin(MASK64, bin_start_w_floor, bin_start_w_floor,
                                   dup_tmp_ub, DIGIT_128 // DIGIT_64,
                                   STRIDE_ONE, STRIDE_ONE, STRIDE_ONE,
                                   REP_STRIDE_EIGHT, REP_STRIDE_EIGHT,
                                   STRIDE_ZERO)
            self.tik_instance.vmin(MASK64, bin_end_w_ceil, bin_end_w_ceil,
                                   dup_tmp_ub, DIGIT_128 // DIGIT_64,
                                   STRIDE_ONE, STRIDE_ONE, STRIDE_ONE,
                                   REP_STRIDE_EIGHT, REP_STRIDE_EIGHT,
                                   STRIDE_ZERO)
            self.tik_instance.vector_dup(MASK64, dup_tmp_ub, self.fm_h,
                                         REPEAT_1, STRIDE_ONE, REP_STRIDE_EIGHT)
            self.tik_instance.vmin(MASK64, bin_start_h_floor, bin_start_h_floor,
                                   dup_tmp_ub, DIGIT_128 // DIGIT_64,
                                   STRIDE_ONE, STRIDE_ONE, STRIDE_ONE,
                                   REP_STRIDE_EIGHT, REP_STRIDE_EIGHT,
                                   STRIDE_ZERO)
            self.tik_instance.vmin(MASK64, bin_end_h_ceil, bin_end_h_ceil,
                                   dup_tmp_ub, DIGIT_128 // DIGIT_64,
                                   STRIDE_ONE, STRIDE_ONE, STRIDE_ONE,
                                   REP_STRIDE_EIGHT, REP_STRIDE_EIGHT,
                                   STRIDE_ZERO)
            params["bin_start_h_floor"] = bin_start_h_floor
            params["bin_end_h_ceil"] = bin_end_h_ceil
            params["bin_start_w_floor"] = bin_start_w_floor
            params["bin_end_w_ceil"] = bin_end_w_ceil

            self.process_one_roi(params)

    def process_rois(self, roi_step, rois_num_offset, roi_loop, roi_step_l):
        """
        process roi_num roi cyclically, and process roi_step roi each time.

        Parameters
        ----------
        roi_step: the number of rois per loop in process, 64(fp32) or 128(fp16)
        rois_num_offset: the offset in block_id aicore
        roi_loop: loops of processing roi_num rois
        roi_step_l: the number of rois in last loop

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, roi_loop) as inner_i:
            rois_ub = self.tik_instance.Tensor(self.dtype, (DIGIT_5, roi_step),
                                               name="rois_ub",
                                               scope=tik.scope_ubuf)
            rois_offset = self.tik_instance.Scalar(INT32, name="rois_offset")
            rois_offset.set_as(rois_num_offset + roi_step*inner_i)
            # move rois data to ub from gm
            self.load_rois_to_ub(rois_ub, rois_offset, roi_step)

            # calculate spatial scale rois
            rois_floor_ub = self.tik_instance.Tensor(INT32, \
                                                     (DIGIT_5, roi_step), \
                                                     name="rois_floor_ub", scope=tik.scope_ubuf)
            rois_spatial_ub = self.tik_instance.Tensor(self.dtype, \
                                                       (DIGIT_8, roi_step), \
                                                       name="rois_spatial_ub", scope=tik.scope_ubuf)
            self.spatial_scale_rois(rois_ub, rois_floor_ub, rois_spatial_ub,
                                    roi_step)

            with self.tik_instance.if_scope(inner_i == (roi_loop - 1)):
                self.process_step1_roi(rois_floor_ub, rois_spatial_ub, \
                                       rois_num_offset, roi_step*inner_i, roi_step_l)

            with self.tik_instance.else_scope():
                self.process_step1_roi(rois_floor_ub, rois_spatial_ub, \
                                       rois_num_offset, roi_step*inner_i, roi_step)

    def process_rois_multi_batch(self, roi_step, rois_num_offset, roi_loop,
                                 roi_step_l, batch_id):
        """
        process roi_num roi cyclically, and process roi_step roi each time.

        Parameters
        ----------
        roi_step: the number of rois per loop in process, 64(fp32) or 128(fp16)
        rois_num_offset: the offset in block_id aicore
        roi_loop: loops of processing roi_num rois
        roi_step_l: the number of rois in last loop

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, roi_loop) as inner_i:
            rois_ub = self.tik_instance.Tensor(self.dtype, (DIGIT_5, roi_step),
                                               name="rois_ub",
                                               scope=tik.scope_ubuf)
            # rois addr offset
            rois_offset = self.tik_instance.Scalar(INT32, name="rois_offset")
            rois_offset.set_as(rois_num_offset + \
                               batch_id*(self.roi_num_b*self.roi_shape[1]) + \
                               roi_step*inner_i)
            # move rois data to ub from gm
            self.load_rois_to_ub(rois_ub, rois_offset, roi_step)

            # calculate spatial scale rois
            rois_floor_ub = self.tik_instance.Tensor(INT32, (DIGIT_5, roi_step),
                                                     name="rois_floor_ub",
                                                     scope=tik.scope_ubuf)
            rois_spatial_ub = self.tik_instance.Tensor(self.dtype,
                                                       (DIGIT_8, roi_step),
                                                       name="rois_spatial_ub",
                                                       scope=tik.scope_ubuf)
            self.spatial_scale_rois(rois_ub, rois_floor_ub, rois_spatial_ub,
                                    roi_step)

            with self.tik_instance.if_scope(inner_i == (roi_loop - 1)):
                self.process_step1_roi(rois_floor_ub, rois_spatial_ub, \
                                       rois_num_offset + self.roi_num_b*batch_id, \
                                       roi_step*inner_i, roi_step_l)

            with self.tik_instance.else_scope():
                self.process_step1_roi(rois_floor_ub, rois_spatial_ub, \
                                       rois_num_offset + self.roi_num_b*batch_id, \
                                       roi_step*inner_i, roi_step)

    def cache_fm_l1(self):
        """
        cache fm in L1, if fm size  is smaller than L1.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.x_data_size <= self.l1_size:
            self.cache_in_l1 = True
            self.cache_l1 = self.tik_instance.Tensor(self.dtype, \
                                                     self.x_shape, name="cache_l1", scope=tik.scope_cbuf)
            # cache fm in L1
            burst_len = self.x_data_size // BLOCK_SIZE
            self.tik_instance.data_move(self.cache_l1, self.x, SID, BURST_1,
                                        burst_len, STRIDE_ZERO, STRIDE_ZERO)

    def init_const_0_127_ub(self):
        """
        init const_0_127_ub, which store 0.0 - 127.0.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.const_0_127_ub = self.tik_instance.Tensor(self.dtype, \
                                                       (DIGIT_128,), name="const_0_127_ub", scope=tik.scope_ubuf)
        const_0_127_int32 = self.tik_instance.Tensor(INT32, (DIGIT_128,), \
                                                     name="const_0_127_int32", scope=tik.scope_ubuf)
        with self.tik_instance.for_range(0, DIGIT_128) as i:
            const_0_127_int32[i].set_as(i)
        # s322f16:vconv.deq, or s322f32:vconv
        if self.dtype == FP16:
            self.tik_instance.vconv(MASK64, '', self.const_0_127_ub,
                                    const_0_127_int32, REPEAT_2, STRIDE_ONE,
                                    STRIDE_ONE, REP_STRIDE[self.dtype],
                                    REP_STRIDE_EIGHT, deqscale=DEQSCALE)
        else:
            self.tik_instance.vconv(MASK64, '', self.const_0_127_ub,
                                    const_0_127_int32, REPEAT_2, STRIDE_ONE,
                                    STRIDE_ONE, REP_STRIDE[self.dtype],
                                    REP_STRIDE_EIGHT)

    def psroi_pooling_compute(self):
        """
        compute of psroipooling.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        outer_loop = self.roi_num_b // self.aicore_num
        outer_tail = self.roi_num_b % self.aicore_num
        roi_step = self.roi_num_step

        # outer_loop is 0
        num1, num2 = 1, 1
        block_num = outer_tail
        roi_loop1, roi_step1_l = 1, 1
        roi_loop2, roi_step2_l = 1, 1
        if outer_loop > 0:
            block_num = self.aicore_num
            if outer_tail > 0:
                num1 = outer_loop + 1
                num2 = outer_loop
            else:
                num1 = outer_loop
                num2 = outer_loop

            roi_loop1 = ceil_value(num1, roi_step)
            roi_step1_l = roi_step if (num1 % roi_step == 0) \
                else (num1 % roi_step)
            roi_loop2 = ceil_value(num2, roi_step)
            roi_step2_l = roi_step if (num2 % roi_step == 0) \
                else (num2 % roi_step)

        with self.tik_instance.for_range(0, block_num,
                                         block_num=block_num) as block_id:
            # process of one aicore
            self.cache_fm_l1()
            self.init_const_0_127_ub()

            rois_num_offset = self.tik_instance.Scalar(INT32,
                                                       name="rois_num_offset")

            if self.fm_batch == 1:
                # process roi nums: num1
                with self.tik_instance.if_scope(block_id < outer_tail):
                    # rois_num_offset is the offset in block_id aicore
                    rois_num_offset.set_as(block_id*num1)
                    self.process_rois(roi_step, rois_num_offset, roi_loop1,
                                      roi_step1_l)
                # process roi nums: num2
                with self.tik_instance.else_scope():
                    if outer_loop > 0:
                        rois_num_offset.set_as(outer_tail*num1 +
                                               (block_id - outer_tail)*num2)
                        self.process_rois(roi_step, rois_num_offset, roi_loop2,
                                          roi_step2_l)

            else:
                # process roi nums: num1*fm_batch
                with self.tik_instance.if_scope(block_id < outer_tail):
                    # rois_num_offset is the offset in block_id aicore
                    with self.tik_instance.for_range(0, self.fm_batch) \
                            as batch_id:
                        rois_num_offset.set_as(block_id*num1)
                        self.process_rois_multi_batch(roi_step, \
                                                      rois_num_offset, roi_loop1, \
                                                      roi_step1_l, batch_id)
                # process roi nums: num2*fm_batch
                with self.tik_instance.else_scope():
                    if outer_loop > 0:
                        with self.tik_instance.for_range(0, self.fm_batch) \
                                as batch_id:
                            rois_num_offset.set_as(outer_tail*num1 + \
                                                   (block_id - outer_tail)*num2)
                            self.process_rois_multi_batch(roi_step, \
                                                          rois_num_offset, roi_loop2, \
                                                          roi_step2_l, batch_id)

    def psroi_pooling_main(self):
        """
        Main process of psroipooling.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.x = self.tik_instance.Tensor(self.dtype, self.x_shape,
                                          name="x", scope=tik.scope_gm)
        rois_shape = (self.roi_shape[0]*self.roi_shape[2]*self.roi_shape[1] +
                      self.vec_elem_num,)
        self.rois = self.tik_instance.Tensor(self.dtype, rois_shape,
                                             name="rois", scope=tik.scope_gm)
        self.y = self.tik_instance.Tensor(self.dtype, shape=self.y_shape,
                                          name="y", scope=tik.scope_gm)

        self.psroi_pooling_compute()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.x, self.rois),
                                   outputs=(self.y,))


# pylint: disable=too-many-arguments
@util.check_input_type(dict, dict, dict, int, int, float, str)
def psroipooling(x_dict, rois_dict, y_dict, output_dim, group_size,
                 spatial_scale, kernel_name="psroipooling"):
    """
    psroipooling interface.

    Parameters
    ----------
    x_dict: feature map size and data type, 5HD
    rois_dict: rois_dict size and data type, (batch, 5, rois_num), rois all
                nums is batch*rois_num
    y_dict: output size and data type, 5HD
    output_dim: number of output channels
    group_size: number of groups encoding position sensitive score maps
    spatial_scale: spatial scale
    kernel_name: kernel name of psroipooling op

    Returns
    -------
    tik_instance
    """
    psroi_instance = PsroiClass(x_dict, rois_dict, y_dict,
                                (output_dim, group_size, spatial_scale),
                                kernel_name)
    psroi_instance.psroi_pooling_main()

    return psroi_instance.tik_instance
