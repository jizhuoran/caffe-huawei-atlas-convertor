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

clip_boxes_d
"""

from te import tik
from topi.cce import util

SHAPE_SIZE_LIMIT = 65500
CONFIG_ONE = 1
CONFIG_TWO = 2
CONFIG_FOUR = 4
CONFIG_EIGHT = 8
CONFIG_SIXTEEN = 16
CONFIG_DATA_ALIGN = 32
CONFIG_DATA_TRANS = 64
CONFIG_MASK = 128
MATRIX = 256
CONFIG_UB_LIMITED = 4096
IF_USE_V200 = ("aic", "vec")

class InitConst:
    """
    define some const numbers
    these const numbers are for vector operators
    """

    def __init__(self):
        # const number for vector operators
        self.dstorsrc_blk_stride1 = CONFIG_ONE
        self.dstorsrc_rep_stride1 = CONFIG_EIGHT
        self.dstorsrc_blk_stride2 = CONFIG_TWO
        self.dstorsrc_rep_stride2 = CONFIG_SIXTEEN
        self.mask = CONFIG_MASK

    def set_dstorsrc_blk_stride1(self, dstorsrc_blk_stride):
        """
        set dstorsrc_blk_stride1
        return: None
        """
        self.dstorsrc_blk_stride1 = dstorsrc_blk_stride

    def set_dstorsrc_rep_stride1(self, dstorsrc_rep_stride):
        """
        set detorsrc_rep_stride1
        return: None
        """
        self.dstorsrc_rep_stride1 = dstorsrc_rep_stride


class ConstList(InitConst):
    """
    define the const numbers
    these const numbers are related to the size of memory
    """

    def __init__(self):
        super(ConstList, self).__init__()
        # const number for float16, each block contains 32B
        self.num_one_blk = CONFIG_SIXTEEN
        # const number for vector op, 8 blk each
        self.num_one_vecop = CONFIG_EIGHT
        # const number for trans op
        self.num_one_trans = CONFIG_DATA_TRANS
        # const number for ND, D=4
        self.num_d = CONFIG_FOUR

    def set_num_one_blk(self, num_one_blk):
        """
        set  num_one_blk
        return: None
        """
        self.num_one_blk = num_one_blk

    def set_num_one_trans(self, num_one_trans):
        """
        set num_one_trans
        return: None
        """
        self.num_one_trans = num_one_trans


def ceil_div(num_a, num_bulk):
    """
    calculate number of bulk needed
    num_a: the num of  input boxes
    num_bulk : the num of elements each bulk
    return  the num of bulk at least needed
    """

    return (num_a + num_bulk - CONFIG_ONE) // num_bulk


class TilingFunc:
    """
    planning the method for data tiling
      tot_of_blk: total num of block
      num_of_blk: num of block for each loop
      num_of_trans: num of transpose is needed, 16*16 for each
      loop_time: loop time
      thread_num: whether pingpang buffer is needed
    """

    def __init__(self, shape):
        #  num_of_boxes <= 4096  not  double buffer, no multi_core
        num_of_boxes = shape[0]
        if num_of_boxes <= CONFIG_UB_LIMITED:
            self.thread_num = CONFIG_ONE
            self.loop_time = CONFIG_ONE
            # first should be times of 4, thus the data is times of 16, and can be moved to UB
            #  num of  block    for data move
            self.tot_of_blk = ceil_div(num_of_boxes, CONFIG_FOUR)
            self.num_of_blk = self.tot_of_blk
            # num of 16*block   for data transpose
            self.num_of_trans = ceil_div(self.num_of_blk, CONFIG_SIXTEEN)
        else:
            # the suggested num_of_block moved to UB once
            num_half_buf = CONFIG_UB_LIMITED//CONFIG_TWO
            #  Use pingpang Buffer
            self.thread_num = CONFIG_TWO
            #  tot num of blocks
            self.tot_of_blk = ceil_div(num_of_boxes, CONFIG_FOUR)
            #  the num of kernels  needed
            loop_time = ceil_div(self.tot_of_blk*CONFIG_FOUR,
                                 num_half_buf)
            #  num of boxes  each time
            num_half_buf = ceil_div(self.tot_of_blk*CONFIG_FOUR,
                                    loop_time)
            #  each time the memory size should be times of 256=32*4*2
            self.num_of_blk = ceil_div(num_half_buf, CONFIG_DATA_ALIGN)*CONFIG_EIGHT
            self.loop_time = ceil_div(self.tot_of_blk,
                                      self.num_of_blk)

            # num of 16*block   for data transpose    (each loop)
            self.num_of_trans = ceil_div(self.num_of_blk,
                                         CONFIG_SIXTEEN)

    def set_thread_num(self, thread_num):
        """
        set thread_num
        return: None
        """
        self.thread_num = thread_num

    def set_num_of_blk(self, num_of_blk):
        """
        set num_of_blk
        return: None
        """
        self.num_of_blk = num_of_blk


class InitMiddleTensor:
    """
    init the middle tensors
    these tensors are located in UB
    """

    def __init__(self, tik_instance, const_num, num_of_trans):
        # the size of image, construct a vector for computing
        if not tik.Dprofile().get_product_name() in IF_USE_V200:
            self.width_ub = tik_instance.Tensor("float16",
                                                (const_num.num_one_vecop*const_num.num_one_blk,
                                                 CONFIG_ONE),
                                                name="width_ub",
                                                scope=tik.scope_ubuf)
            self.height_ub = tik_instance.Tensor("float16",
                                                 (const_num.num_one_vecop*const_num.num_one_blk,
                                                  CONFIG_ONE),
                                                 name="height_ub",
                                                 scope=tik.scope_ubuf)
        self.anchors_ub = tik_instance.Tensor("float16",
                                              (num_of_trans*const_num.num_one_trans,
                                               const_num.num_d),
                                              name="anchors_ub",
                                              scope=tik.scope_ubuf)
        self.boxes_ub = tik_instance.Tensor("float16",
                                            (const_num.num_d,
                                             num_of_trans*const_num.num_one_trans),
                                            name="boxes_ub",
                                            scope=tik.scope_ubuf)
        self.res_temp1_ub = tik_instance.Tensor("float16",
                                                (const_num.num_d,
                                                 num_of_trans*const_num.num_one_trans),
                                                name="res_temp1_ub",
                                                scope=tik.scope_ubuf)
        self.res_temp2_ub = tik_instance.Tensor("float16",
                                                (const_num.num_d,
                                                 num_of_trans*const_num.num_one_trans),
                                                name="res_temp2_ub",
                                                scope=tik.scope_ubuf)
        self.res_ub = tik_instance.Tensor("float16",
                                          (num_of_trans*const_num.num_one_trans,
                                           const_num.num_d),
                                          name="res_ub",
                                          scope=tik.scope_ubuf)

    def set_imgh_vec(self, tik_instance, img_h, const_num):
        """
        set the image height vector
        return: None
        """
        if not tik.Dprofile().get_product_name() in IF_USE_V200:
            tik_instance.vector_dup(const_num.mask,
                                    self.height_ub[0],
                                    float(img_h),
                                    const_num.dstorsrc_blk_stride1,
                                    const_num.dstorsrc_blk_stride1,
                                    const_num.dstorsrc_rep_stride1)

    def set_imgw_vec(self, tik_instance, img_w, const_num):
        """
        set the image width vector
        return: None
        """
        if not tik.Dprofile().get_product_name() in IF_USE_V200:
            tik_instance.vector_dup(const_num.mask,
                                    self.width_ub[0],
                                    float(img_w),
                                    const_num.dstorsrc_blk_stride1,
                                    const_num.dstorsrc_blk_stride1,
                                    const_num.dstorsrc_rep_stride1)


def processing_one_loop(tik_instance, data_gm, tiling_para, img_size, offset):
    """
    Using Pingpang, this func is one loop processing
    param tik_instance: tik container
    param data_gm: in and out data tensors in DDR
    param tiling_para: tiling
    param img_size: (img_h. img_w)
    param offset: loop id
    return: None
    """

    const_num = ConstList()
    anchors = data_gm[0]
    res_anchors = data_gm[1]
    img_h, img_w = img_size

    data_tensor = InitMiddleTensor(tik_instance, const_num,
                                   tiling_para.num_of_trans)

    # move data from DDR to UB
    tik_instance.data_move(data_tensor.anchors_ub[0],
                           anchors[tiling_para.num_of_blk*const_num.num_one_blk*offset],
                           const_num.dstorsrc_blk_stride1,
                           const_num.dstorsrc_blk_stride1,
                           tiling_para.num_of_blk,
                           0, 0)

    #  do the transpose for the input 16*16
    dst_list = [data_tensor.boxes_ub[const_num.num_one_blk * i]
                for i in range(const_num.num_one_blk)]
    src_list = [data_tensor.anchors_ub[const_num.num_one_blk * i]
                for i in range(const_num.num_one_blk)]
    # with tik_instance.if_scope(tiling_para.num_of_trans == CONFIG_ONE):
    if tiling_para.num_of_trans == CONFIG_ONE:
        tik_instance.vnchwconv(True, False, dst_list, src_list,
                               tiling_para.num_of_trans, 0, 0)
    else:
        tik_instance.vnchwconv(True, False, dst_list, src_list,
                               tiling_para.num_of_trans,
                               const_num.dstorsrc_rep_stride2,
                               const_num.dstorsrc_rep_stride2)

    # do relu, comparing with 0
    tik_instance.vrelu(const_num.mask,
                       data_tensor.res_temp1_ub[0],
                       data_tensor.boxes_ub[0],
                       tiling_para.num_of_trans*CONFIG_TWO,
                       const_num.dstorsrc_blk_stride1,
                       const_num.dstorsrc_blk_stride1,
                       const_num.dstorsrc_rep_stride1,
                       const_num.dstorsrc_rep_stride1)

    # do the comparing
    if tik.Dprofile().get_product_name() in IF_USE_V200:
        tik_instance.vmins(const_num.mask,
                           data_tensor.res_temp2_ub[0],
                           data_tensor.res_temp1_ub[0],
                           img_w,
                           tiling_para.num_of_trans,
                           const_num.dstorsrc_blk_stride2,
                           const_num.dstorsrc_blk_stride2,
                           const_num.dstorsrc_rep_stride2,
                           const_num.dstorsrc_rep_stride2)
        tik_instance.vmins(const_num.mask,
                           data_tensor.res_temp2_ub[CONFIG_SIXTEEN],
                           data_tensor.res_temp1_ub[CONFIG_SIXTEEN],
                           img_h,
                           tiling_para.num_of_trans,
                           const_num.dstorsrc_blk_stride2,
                           const_num.dstorsrc_blk_stride2,
                           const_num.dstorsrc_rep_stride2,
                           const_num.dstorsrc_rep_stride2)
    else:
        # init the vector img_h, img_w
        data_tensor.set_imgh_vec(tik_instance, img_h, const_num)
        data_tensor.set_imgw_vec(tik_instance, img_w, const_num)
        tik_instance.vmin(const_num.mask,
                          data_tensor.res_temp2_ub[0],
                          data_tensor.res_temp1_ub[0],
                          data_tensor.width_ub[0],
                          tiling_para.num_of_trans,
                          const_num.dstorsrc_blk_stride2,
                          const_num.dstorsrc_blk_stride2,
                          const_num.dstorsrc_blk_stride1,
                          const_num.dstorsrc_rep_stride2,
                          const_num.dstorsrc_rep_stride2,
                          0)
        tik_instance.vmin(const_num.mask,
                          data_tensor.res_temp2_ub[CONFIG_SIXTEEN],
                          data_tensor.res_temp1_ub[CONFIG_SIXTEEN],
                          data_tensor.height_ub[0],
                          tiling_para.num_of_trans,
                          const_num.dstorsrc_blk_stride2,
                          const_num.dstorsrc_blk_stride2,
                          const_num.dstorsrc_blk_stride1,
                          const_num.dstorsrc_rep_stride2,
                          const_num.dstorsrc_rep_stride2,
                          0)

    # Data  transpose
    dst_list = [data_tensor.res_ub[const_num.num_one_blk * i]
                for i in range(const_num.num_one_blk)]
    src_list = [data_tensor.res_temp2_ub[const_num.num_one_blk * i]
                for i in range(const_num.num_one_blk)]
    if tiling_para.num_of_trans == CONFIG_ONE:
        tik_instance.vnchwconv(True, False, dst_list, src_list,
                               tiling_para.num_of_trans, 0, 0)
    else:
        tik_instance.vnchwconv(True, False, dst_list, src_list,
                               tiling_para.num_of_trans,
                               const_num.dstorsrc_rep_stride2,
                               const_num.dstorsrc_rep_stride2)

    #  move data from UB  to DDR
    tik_instance.data_move(res_anchors[tiling_para.num_of_blk*const_num.num_one_blk*offset],
                           data_tensor.res_ub[0],
                           const_num.dstorsrc_blk_stride1,
                           const_num.dstorsrc_blk_stride1,
                           tiling_para.num_of_blk,
                           0, 0)


def processing_tail(tik_instance, data_gm, tiling_para, img_size):
    """

    :param tik_instance:
    :param data_gm:
    :param tiling_para:
    :param img_size:
    :return:
    """

    const_num = ConstList()
    anchors = data_gm[0]
    res_anchors = data_gm[1]
    img_h, img_w = img_size

    data_tensor = InitMiddleTensor(tik_instance, const_num,
                                   tiling_para.num_of_trans)

    # move data from DDR to UB
    tik_instance.data_move(data_tensor.anchors_ub[0],
                           anchors[tiling_para.num_of_blk * const_num.num_one_blk *
                                   (tiling_para.loop_time - CONFIG_ONE)],
                           const_num.dstorsrc_blk_stride1,
                           const_num.dstorsrc_blk_stride1,
                           tiling_para.tot_of_blk -
                           tiling_para.num_of_blk * (tiling_para.loop_time-CONFIG_ONE),
                           0, 0)

    #  do the transpose for the input 16*16
    dst_list = [data_tensor.boxes_ub[const_num.num_one_blk * i]
                for i in range(const_num.num_one_blk)]
    src_list = [data_tensor.anchors_ub[const_num.num_one_blk * i]
                for i in range(const_num.num_one_blk)]
    # with tik_instance.if_scope(tiling_para.num_of_trans == CONFIG_ONE):
    if tiling_para.num_of_trans == CONFIG_ONE:
        tik_instance.vnchwconv(True, False, dst_list, src_list,
                               tiling_para.num_of_trans, 0, 0)
    else:
        tik_instance.vnchwconv(True, False, dst_list, src_list,
                               tiling_para.num_of_trans,
                               const_num.dstorsrc_rep_stride2,
                               const_num.dstorsrc_rep_stride2)

    # do relu, comparing with 0
    tik_instance.vrelu(const_num.mask,
                       data_tensor.res_temp1_ub[0],
                       data_tensor.boxes_ub[0],
                       tiling_para.num_of_trans*CONFIG_TWO,
                       const_num.dstorsrc_blk_stride1,
                       const_num.dstorsrc_blk_stride1,
                       const_num.dstorsrc_rep_stride1,
                       const_num.dstorsrc_rep_stride1)

    # do the comparing
    if tik.Dprofile().get_product_name() in IF_USE_V200:
        tik_instance.vmins(const_num.mask,
                           data_tensor.res_temp2_ub[0],
                           data_tensor.res_temp1_ub[0],
                           img_w,
                           tiling_para.num_of_trans,
                           const_num.dstorsrc_blk_stride2,
                           const_num.dstorsrc_blk_stride2,
                           const_num.dstorsrc_rep_stride2,
                           const_num.dstorsrc_rep_stride2)
        tik_instance.vmins(const_num.mask,
                           data_tensor.res_temp2_ub[CONFIG_SIXTEEN],
                           data_tensor.res_temp1_ub[CONFIG_SIXTEEN],
                           img_h,
                           tiling_para.num_of_trans,
                           const_num.dstorsrc_blk_stride2,
                           const_num.dstorsrc_blk_stride2,
                           const_num.dstorsrc_rep_stride2,
                           const_num.dstorsrc_rep_stride2)
    else:
        # init the vector height and width
        data_tensor.set_imgh_vec(tik_instance, img_h, const_num)
        data_tensor.set_imgw_vec(tik_instance, img_w, const_num)

        tik_instance.vmin(const_num.mask,
                          data_tensor.res_temp2_ub[0],
                          data_tensor.res_temp1_ub[0],
                          data_tensor.width_ub[0],
                          tiling_para.num_of_trans,
                          const_num.dstorsrc_blk_stride2,
                          const_num.dstorsrc_blk_stride2,
                          const_num.dstorsrc_blk_stride1,
                          const_num.dstorsrc_rep_stride2,
                          const_num.dstorsrc_rep_stride2,
                          0)
        tik_instance.vmin(const_num.mask,
                          data_tensor.res_temp2_ub[CONFIG_SIXTEEN],
                          data_tensor.res_temp1_ub[CONFIG_SIXTEEN],
                          data_tensor.height_ub[0],
                          tiling_para.num_of_trans,
                          const_num.dstorsrc_blk_stride2,
                          const_num.dstorsrc_blk_stride2,
                          const_num.dstorsrc_blk_stride1,
                          const_num.dstorsrc_rep_stride2,
                          const_num.dstorsrc_rep_stride2,
                          0)

    # Data  transpose
    dst_list = [data_tensor.res_ub[const_num.num_one_blk * i]
                for i in range(const_num.num_one_blk)]
    src_list = [data_tensor.res_temp2_ub[const_num.num_one_blk * i]
                for i in range(const_num.num_one_blk)]
    if tiling_para.num_of_trans == CONFIG_ONE:
        tik_instance.vnchwconv(True, False, dst_list, src_list,
                               tiling_para.num_of_trans, 0, 0)
    else:
        tik_instance.vnchwconv(True, False, dst_list, src_list,
                               tiling_para.num_of_trans,
                               const_num.dstorsrc_rep_stride2,
                               const_num.dstorsrc_rep_stride2)

    #  move data from UB  to DDR
    tik_instance.data_move(res_anchors[tiling_para.num_of_blk * const_num.num_one_blk *
                                       (tiling_para.loop_time-CONFIG_ONE)],
                           data_tensor.res_ub[0],
                           const_num.dstorsrc_blk_stride1,
                           const_num.dstorsrc_blk_stride1,
                           tiling_para.tot_of_blk -
                           tiling_para.num_of_blk * (tiling_para.loop_time-CONFIG_ONE),
                           0, 0)


def clip_boxes_d_compute(boxes_input, img_w, img_h, kernel_name="clip_boxes"):
    """
    the compute process of clip_boxes
    input:
     boxes_input:a dict, include shape, and dtype
     img_w: width of the image
     img_h: height of the image
     kernel_name: the kernel name
    return:
     the tik container
    """

    const_num = ConstList()
    tiling_para = TilingFunc(boxes_input.get("shape"))

    #  start the TIK container
    tik_instance = tik.Tik(tik.Dprofile(), True)

    anchors = tik_instance.Tensor("float16",
                                  (tiling_para.tot_of_blk*const_num.num_d,
                                   const_num.num_d),
                                  name="anchors",
                                  scope=tik.scope_gm)
    res_anchors = tik_instance.Tensor("float16",
                                      (tiling_para.tot_of_blk*const_num.num_d,
                                       const_num.num_d),
                                      name="res_anchors",
                                      scope=tik.scope_gm)

    with tik_instance.for_range(0, tiling_para.loop_time - CONFIG_ONE,
                                thread_num=tiling_para.thread_num) as loop_i:
        processing_one_loop(tik_instance,
                            (anchors, res_anchors),
                            tiling_para,
                            (img_h, img_w),
                            loop_i)

    # the tail processing
    processing_tail(tik_instance,
                    (anchors, res_anchors),
                    tiling_para,
                    (img_h, img_w))

    tik_instance.BuildCCE(kernel_name=kernel_name, inputs=[anchors], outputs=[res_anchors])
    return tik_instance


def check_clip_boxes_input_dict(boxes_input, boxes_output):
    """
    check the input parameters -- tensor
    input:
      boxes_input: an dict, include shape, and dtype of input
      boxes_output: an dict, include shape, and dtype of output
    return: None
    """

    input_shape = boxes_input.get("shape")
    input_dtype = boxes_input.get("dtype")
    output_shape = boxes_output.get("shape")
    output_dtype = boxes_output.get("dtype")

    if input_shape is None or input_dtype is None:
        raise RuntimeError("The boxes_input/boxes_output must include 'shape' and 'dtype'!")

    # the shape and type of the output  should be the same as the input
    if input_shape != output_shape:
        raise RuntimeError("The shape of output should be the same as the input!")

    # Check the size of the input shape
    util.check_shape_rule(input_shape)
    if len(input_shape) != CONFIG_TWO:
        raise RuntimeError("The input shape should be two dimension only!")
    n_x, n_y = input_shape
    if n_x <= 0 or n_x > SHAPE_SIZE_LIMIT:
        raise RuntimeError("N dimension of inputs should be in [1, %d]" % SHAPE_SIZE_LIMIT)
    if n_y != CONFIG_FOUR:
        raise RuntimeError("The last dimension of xxx tensor must be 4!")

    if input_dtype.lower() != "float16":
        raise RuntimeError("The dtype of input must be float16!")
    if input_dtype != output_dtype:
        raise RuntimeError("The dtype of output should be the same as the input!")


def check_clip_boxes_input_attr(img_w, img_h):
    """
    check the input parameters  -- attr
    input:
      img_w: width of the image
      img_h: height of the image
    return: None
    """

    if not isinstance(img_h, int):
        raise RuntimeError("img_h should be Int!")

    if not isinstance(img_w, int):
        raise RuntimeError("img_w should be Int!")
    # the size of the image  should  be lager than zero
    if img_h <= 0 or img_w <= 0:
        raise RuntimeError("img_h/img_w should be larger than zero!")


@util.check_input_type(dict, dict, (tuple, list), str)
def clip_boxes_d(boxes_input, boxes_output, img_size, kernel_name="clip_boxes"):
    """
    the External interface function
    input:
      boxes_input: an dict, include shape, and dtype of input
      boxes_output: an dict, include shape, and dtype of output
      img_w: width of the image
      img_h: height of the image
      kernel_name: the kernel name
    return:
      the tik container
    """

    if len(img_size) != CONFIG_TWO:
        raise RuntimeError("img_size should be [img_h, img_w]!")

    img_h, img_w = img_size
    check_clip_boxes_input_dict(boxes_input, boxes_output)
    check_clip_boxes_input_attr(img_w, img_h)

    if len(kernel_name) > util.MAX_KERNEL_NAEM_LEN:
        raise RuntimeError("kernel_name len must be less than 200!")
    util.check_kernel_name(kernel_name)

    tik_instance = clip_boxes_d_compute(boxes_input, img_w, img_h, kernel_name=kernel_name)
    return tik_instance
