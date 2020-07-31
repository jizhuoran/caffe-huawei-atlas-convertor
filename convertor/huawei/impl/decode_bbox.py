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

decode_bbox
"""

from functools import reduce
from te import tik
from topi.cce import util


# General limitation of the reduce size for input shape: 2**31
SHAPE_SIZE_LIMIT = 2147483648
RESERVED_UB = 20480
CONFIG_ONE_HALF = 0.5
CONFIG_ONE = 1
CONFIG_TWO = 2
CONFIG_THREE = 3
CONFIG_FOUR = 4
CONFIG_FIVE = 5
CONFIG_EIGHT = 8
CONFIG_TWELVE = 12
CONFIG_SIXTEEN = 16
CONFIG_DATA_SIZE = 32
CONFIG_FLOAT_SIZE = 64
CONFIG_BLOCK_SIZE = 128
MATRIX = 256


def ceil_div(divisor, dividend):
    """
    ceil_div
    Returns
    -------
    None
    """
    res = (divisor + dividend - CONFIG_ONE) // dividend
    return res


class InitTikParam:
    """
    init the  middle tensor
    Returns
    -------
    None
    """
    def __init__(self):
        self.product_name = tik.Dprofile().get_product_name()
        self.total_ub = tik.Dprofile().get_unified_buffer_size()
        self.available_ub_size = (self.total_ub - RESERVED_UB) // CONFIG_TWO
        self.aicore_num = tik.Dprofile().get_aicore_num()
        if self.aicore_num >= CONFIG_EIGHT:
            self.aicore_num = CONFIG_EIGHT
        print("product_name:", self.product_name)
        print("total_ub:", self.total_ub)
        print("available_ub_size:", self.available_ub_size)
        print("aicore_num", self.aicore_num)

    def set_product_name(self):
        """
        set product name
        Parameters
        ----------

        Returns
        -------
        product name
        """
        self.product_name = tik.Dprofile().get_product_name()

    def set_ub_buf(self):
        """
        set ub buffer size
        Parameters
        ----------

        Returns
        -------
        total available ub buffer size
        """
        self.total_ub = tik.Dprofile().get_unified_buffer_size()


class InitShape(InitTikParam):
    """
    calculating the shape
    Returns
    -------
    None
    """

    def __init__(self, shape):
        super(InitShape, self).__init__()
        # The minimum Number ono core one time
        # one loop shape is 16*16*16 one_buf_count 16*16*4
        self.one_loop_count = (self.available_ub_size //
                               (CONFIG_SIXTEEN * CONFIG_SIXTEEN * CONFIG_TWELVE)) * CONFIG_SIXTEEN
        self.one_loop_shape = self.one_loop_count * CONFIG_SIXTEEN
        self.one_buf_count = self.one_loop_count * CONFIG_FOUR
        self.one_loop_number = CONFIG_TWO
        self.one_c0_shape = self.one_loop_shape * self.one_loop_number
        # shape of Input : (N,C1,H,W,C0)
        self.input_shape = shape
        # shape of Output : ND = (N*C1*H*W,C0//CONFIG_FOUR)
        self.proposal_shape = (reduce(lambda x, y: x * y, shape[:])) // CONFIG_FOUR

    def set_input_shape(self, input_shape):
        """
        set_input_shape
        :return:
        """
        self.input_shape = input_shape

    def set_output_shape(self, output_shape):
        """
        set_input_shape
        :return:
        """
        self.input_shape = output_shape


class InitEightCoreShape(InitShape):
    """
    calculating the shape
    Returns
    -------
    None
    """

    def __init__(self, shape):
        super(InitEightCoreShape, self).__init__(shape)
        self.core_mul_loop = self.proposal_shape // self.one_buf_count
        if self.product_name == "cloud":
            self.aicore_num = CONFIG_EIGHT
        if self.product_name == "aic":
            self.aicore_num = CONFIG_EIGHT
        if self.core_mul_loop >= self.aicore_num:
            self.core_number = self.aicore_num
            self.one_core_loop_time = self.core_mul_loop // self.core_number
            if self.core_mul_loop >= (self.aicore_num * CONFIG_TWO):
                self.thread_number = CONFIG_TWO
            else:
                self.thread_number = CONFIG_ONE
            # one core shape
            self.one_core_shape = self.one_core_loop_time * self.one_loop_shape
        else:
            self.core_number = CONFIG_ONE
            self.thread_number = CONFIG_ONE
            self.one_core_shape = 0
            self.one_core_loop_time = 0

    def set_input_shape(self, input_shape):
        """
        set_input_shape
        :return:
        """
        self.input_shape = input_shape

    def set_output_shape(self, output_shape):
        """
        set_input_shape
        :return:
        """
        self.input_shape = output_shape


class TilingFunc(InitEightCoreShape):
    """
    calculating the shape
    Returns
    -------
    None
    """

    def __init__(self, shape):
        super(TilingFunc, self).__init__(shape)
        # eight core shape
        self.eight_core_shape = self.one_core_shape * self.core_number
        # 8 core shape left
        self.left_shape = self.proposal_shape * CONFIG_FOUR - self.eight_core_shape
        self.left_one_core_loop = (self.left_shape // CONFIG_FOUR) // self.one_buf_count
        self.tail_shape = self.proposal_shape * CONFIG_FOUR - self.eight_core_shape - \
                          (self.left_one_core_loop * self.one_loop_shape)
        self.left_one_core_loop_time = self.left_one_core_loop
        if self.left_one_core_loop == CONFIG_ONE:
            self.left_thread_number = CONFIG_ONE
        else:
            self.left_thread_number = CONFIG_TWO

    def set_input_shape(self, input_shape):
        """
        set_input_shape
        :return:
        """
        self.input_shape = input_shape

    def set_output_shape(self, output_shape):
        """
        set_input_shape
        :return:
        """
        self.input_shape = output_shape


class InitMiddleTensor:
    """
    init the input output tensor
    Returns
    -------
    None
    """

    def __init__(self, tik_instance, shape):
        self.anchors = tik_instance.Tensor("float16", shape.input_shape, name="anchors",
                                           scope=tik.scope_gm)
        self.box_predictions = tik_instance.Tensor("float16", shape.input_shape,
                                                   name="box_predictions", scope=tik.scope_gm)
        self.result_gm = tik_instance.Tensor("float16", (shape.proposal_shape, CONFIG_FOUR),
                                             name="result_gm",
                                             scope=tik.scope_gm)
        self.a_buffer_ub = tik_instance.Tensor("float16", (shape.one_loop_count, CONFIG_SIXTEEN),
                                               name="a_buffer_ub", scope=tik.scope_ubuf)
        self.b_buffer_ub = tik_instance.Tensor("float16", (shape.one_loop_count, CONFIG_SIXTEEN),
                                               name="b_buffer_ub", scope=tik.scope_ubuf)
        self.c_buffer_ub = tik_instance.Tensor("float16", (shape.one_loop_count, CONFIG_SIXTEEN),
                                               name="c_buffer_ub", scope=tik.scope_ubuf)
        self.d_buffer_ub = tik_instance.Tensor("float16", (shape.one_loop_count, CONFIG_SIXTEEN),
                                               name="d_buffer_ub", scope=tik.scope_ubuf)

    def set_a_buffer_ub(self, a_buffer_ub):
        """
        set_a_buffer_ub
        :return:
        """
        self.a_buffer_ub = a_buffer_ub

    def set_b_buffer_ub(self, b_buffer_ub):
        """
        set_b_buffer_ub
        :return:
        """
        self.b_buffer_ub = b_buffer_ub


class InitTensor(InitMiddleTensor):
    """
    init the  middle tensor
    Returns
    -------
    None
    """

    def __init__(self, tik_instance, shape):
        super(InitTensor, self).__init__(tik_instance, shape)
        self.e_buffer_ub = tik_instance.Tensor("float16", (CONFIG_FOUR, shape.one_buf_count),
                                               name="e_buffer_ub", scope=tik.scope_ubuf)
        self.f_buffer_ub = tik_instance.Tensor("float16", (CONFIG_FOUR, shape.one_buf_count),
                                               name="f_buffer_ub", scope=tik.scope_ubuf)
        self.g_buffer_ub = tik_instance.Tensor("float16", (shape.one_buf_count, CONFIG_SIXTEEN),
                                               name="g_buffer_ub", scope=tik.scope_ubuf)
        self.clip_vector = tik_instance.Tensor("float16", (CONFIG_ONE, shape.one_buf_count),
                                               name="clip_vector", scope=tik.scope_ubuf)
        self.half_wb_hb_buffer_ub = tik_instance.Tensor("float16",
                                                        (shape.one_buf_count * CONFIG_TWO,),
                                                        name="half_wb_hb_buffer_ub",
                                                        scope=tik.scope_ubuf)
        self.repeat_scalar = tik_instance.Scalar("int32", name="repeat_scalar")

    def set_e_buffer_ub(self, e_buffer_ub):
        """
        set_e_buffer_ub
        :return:
        """
        self.e_buffer_ub = e_buffer_ub

    def set_f_buffer_ub(self, f_buffer_ub):
        """
        set_f_buffer_ub
        :return:
        """
        self.f_buffer_ub = f_buffer_ub


def calculate_process(tik_instance, data_tensor, current_data_address, repeat_data, shape):
    """
       Parameters
       ----------
       data_tensor          : the input output and middle tensor, only support fp16
       current_data_address : current_data_address for each core or each loop
       repeat_data          : repeat times for each loop of each core
       shape                : the init shape for calculate
       Returns
       -------
       None
    """
    # move anchors to a_buffer_ub in ub from out
    tik_instance.data_move(data_tensor.a_buffer_ub, data_tensor.anchors[current_data_address], 0,
                           CONFIG_ONE, repeat_data, 0, 0)
    tik_instance.data_move(data_tensor.b_buffer_ub,
                           data_tensor.box_predictions[current_data_address], 0, CONFIG_ONE,
                           repeat_data, 0, 0)
    # transpose x1,y1,x2,y2... x1,y1,x2,y2 ... to c_buffer_ub from a_buffer_ub
    # transpose tx,ty,tw,th;tx,ty,tw,th; ... to d_buffer_ub from b_buffer_ub
    with tik_instance.for_range(0, shape.one_loop_count // CONFIG_SIXTEEN) as i:
        tik_instance.vtranspose(data_tensor.c_buffer_ub[MATRIX * i],
                                data_tensor.a_buffer_ub[MATRIX * i])
    with tik_instance.for_range(0, shape.one_loop_count // CONFIG_SIXTEEN) as i:
        tik_instance.vtranspose(data_tensor.d_buffer_ub[MATRIX * i],
                                data_tensor.b_buffer_ub[MATRIX * i])
    # the formula is wa = x2 - x1
    tik_instance.vsub(CONFIG_BLOCK_SIZE, data_tensor.e_buffer_ub,
                      data_tensor.c_buffer_ub[CONFIG_TWO, 0],
                      data_tensor.c_buffer_ub, data_tensor.repeat_scalar, CONFIG_ONE,
                      CONFIG_FOUR, CONFIG_FOUR, CONFIG_EIGHT, CONFIG_FOUR * CONFIG_EIGHT,
                      CONFIG_FOUR * CONFIG_EIGHT)
    # the formula is ha = y2 - y1
    tik_instance.vsub(CONFIG_BLOCK_SIZE, data_tensor.e_buffer_ub[CONFIG_ONE, 0],
                      data_tensor.c_buffer_ub[CONFIG_THREE, 0],
                      data_tensor.c_buffer_ub[CONFIG_ONE, 0], data_tensor.repeat_scalar, CONFIG_ONE,
                      CONFIG_FOUR, CONFIG_FOUR, CONFIG_EIGHT, CONFIG_FOUR * CONFIG_EIGHT,
                      CONFIG_FOUR * CONFIG_EIGHT)
    # the formula is xa = 0.5*(x1 + x2)
    tik_instance.vadd(CONFIG_BLOCK_SIZE, data_tensor.e_buffer_ub[CONFIG_TWO, 0],
                      data_tensor.c_buffer_ub[CONFIG_TWO, 0],
                      data_tensor.c_buffer_ub, data_tensor.repeat_scalar, CONFIG_ONE,
                      CONFIG_FOUR, CONFIG_FOUR, CONFIG_EIGHT, CONFIG_FOUR * CONFIG_EIGHT,
                      CONFIG_FOUR * CONFIG_EIGHT)
    # the formula is ya = 0.5*(y1 + y2)
    tik_instance.vadd(CONFIG_BLOCK_SIZE, data_tensor.e_buffer_ub[CONFIG_THREE, 0],
                      data_tensor.c_buffer_ub[CONFIG_ONE, 0],
                      data_tensor.c_buffer_ub[CONFIG_THREE, 0], data_tensor.repeat_scalar,
                      CONFIG_ONE, CONFIG_FOUR, CONFIG_FOUR, CONFIG_EIGHT,
                      CONFIG_FOUR * CONFIG_EIGHT,
                      CONFIG_FOUR * CONFIG_EIGHT)
    tik_instance.vmuls(CONFIG_BLOCK_SIZE, data_tensor.e_buffer_ub[CONFIG_TWO * shape.one_buf_count],
                       data_tensor.e_buffer_ub[CONFIG_TWO * shape.one_buf_count], CONFIG_ONE_HALF,
                       data_tensor.repeat_scalar * CONFIG_TWO, CONFIG_ONE, CONFIG_ONE, CONFIG_EIGHT,
                       CONFIG_EIGHT)
    # the formula is wb = exp(min(tw, clip_vector)) * wa
    tik_instance.vmin(CONFIG_BLOCK_SIZE, data_tensor.f_buffer_ub,
                      data_tensor.d_buffer_ub[CONFIG_TWO, 0],
                      data_tensor.clip_vector, data_tensor.repeat_scalar, CONFIG_ONE,
                      CONFIG_FOUR, CONFIG_ONE, CONFIG_EIGHT, CONFIG_FOUR * CONFIG_EIGHT,
                      CONFIG_EIGHT)
    tik_instance.vexp(CONFIG_BLOCK_SIZE, data_tensor.f_buffer_ub, data_tensor.f_buffer_ub,
                      data_tensor.repeat_scalar,
                      CONFIG_ONE, CONFIG_ONE, CONFIG_EIGHT, CONFIG_EIGHT)
    # the formula is hb = exp(min(th, clip_vector)) * ha
    tik_instance.vmin(CONFIG_BLOCK_SIZE, data_tensor.f_buffer_ub[CONFIG_ONE, 0],
                      data_tensor.d_buffer_ub[CONFIG_THREE, 0],
                      data_tensor.clip_vector, data_tensor.repeat_scalar, CONFIG_ONE,
                      CONFIG_FOUR, CONFIG_ONE, CONFIG_EIGHT, CONFIG_FOUR * CONFIG_EIGHT,
                      CONFIG_EIGHT)
    tik_instance.vexp(CONFIG_BLOCK_SIZE, data_tensor.f_buffer_ub[CONFIG_ONE, 0],
                      data_tensor.f_buffer_ub[CONFIG_ONE, 0], data_tensor.repeat_scalar,
                      CONFIG_ONE, CONFIG_ONE, CONFIG_EIGHT, CONFIG_EIGHT)
    tik_instance.vmul(CONFIG_BLOCK_SIZE, data_tensor.f_buffer_ub,
                      data_tensor.f_buffer_ub,
                      data_tensor.e_buffer_ub, data_tensor.repeat_scalar * CONFIG_TWO, CONFIG_ONE,
                      CONFIG_ONE, CONFIG_ONE, CONFIG_EIGHT, CONFIG_EIGHT, CONFIG_EIGHT)
    # the formula is xb = tx * wa + xa
    tik_instance.vmul(CONFIG_BLOCK_SIZE, data_tensor.f_buffer_ub[CONFIG_TWO, 0],
                      data_tensor.d_buffer_ub,
                      data_tensor.e_buffer_ub, data_tensor.repeat_scalar, CONFIG_ONE,
                      CONFIG_FOUR, CONFIG_ONE, CONFIG_EIGHT, CONFIG_FOUR * CONFIG_EIGHT,
                      CONFIG_EIGHT)
    # the formula is yb = ty * ha + ya
    tik_instance.vmul(CONFIG_BLOCK_SIZE, data_tensor.f_buffer_ub[CONFIG_THREE, 0],
                      data_tensor.d_buffer_ub[CONFIG_ONE, 0],
                      data_tensor.e_buffer_ub[CONFIG_ONE, 0], data_tensor.repeat_scalar, CONFIG_ONE,
                      CONFIG_FOUR, CONFIG_ONE, CONFIG_EIGHT, CONFIG_FOUR * CONFIG_EIGHT,
                      CONFIG_EIGHT)
    tik_instance.vadd(CONFIG_BLOCK_SIZE, data_tensor.f_buffer_ub[CONFIG_TWO * shape.one_buf_count],
                      data_tensor.f_buffer_ub[CONFIG_TWO * shape.one_buf_count],
                      data_tensor.e_buffer_ub[CONFIG_TWO * shape.one_buf_count],
                      data_tensor.repeat_scalar * CONFIG_TWO, CONFIG_ONE,
                      CONFIG_ONE, CONFIG_ONE, CONFIG_EIGHT, CONFIG_EIGHT, CONFIG_EIGHT)
    # the formula is 0.5 * wb  && 0.5 * hb
    tik_instance.vmuls(CONFIG_BLOCK_SIZE, data_tensor.half_wb_hb_buffer_ub,
                       data_tensor.f_buffer_ub, CONFIG_ONE_HALF,
                       data_tensor.repeat_scalar * CONFIG_TWO, CONFIG_ONE, CONFIG_ONE, CONFIG_EIGHT,
                       CONFIG_EIGHT)
    # the formula is x1 = xb - 0.5 * wb;
    tik_instance.vsub(CONFIG_BLOCK_SIZE, data_tensor.g_buffer_ub,
                      data_tensor.f_buffer_ub[CONFIG_TWO, 0],
                      data_tensor.half_wb_hb_buffer_ub, data_tensor.repeat_scalar, CONFIG_FOUR,
                      CONFIG_ONE, CONFIG_ONE, CONFIG_FOUR * CONFIG_EIGHT,
                      CONFIG_EIGHT, CONFIG_EIGHT)
    # the formula is y1 = yb - 0.5 * hb;
    tik_instance.vsub(CONFIG_BLOCK_SIZE, data_tensor.g_buffer_ub[CONFIG_ONE, 0],
                      data_tensor.f_buffer_ub[CONFIG_THREE, 0],
                      data_tensor.half_wb_hb_buffer_ub[shape.one_buf_count],
                      data_tensor.repeat_scalar, CONFIG_FOUR,
                      CONFIG_ONE, CONFIG_ONE, CONFIG_FOUR * CONFIG_EIGHT,
                      CONFIG_EIGHT, CONFIG_EIGHT)
    # the formula is x2 = xb + 0.5 * wb;
    tik_instance.vadd(CONFIG_BLOCK_SIZE, data_tensor.g_buffer_ub[CONFIG_TWO, 0],
                      data_tensor.f_buffer_ub[CONFIG_TWO, 0],
                      data_tensor.half_wb_hb_buffer_ub, data_tensor.repeat_scalar, CONFIG_FOUR,
                      CONFIG_ONE, CONFIG_ONE, CONFIG_FOUR * CONFIG_EIGHT,
                      CONFIG_EIGHT, CONFIG_EIGHT)
    # the formula is y2 = yb + 0.5 * hb;
    tik_instance.vadd(CONFIG_BLOCK_SIZE, data_tensor.g_buffer_ub[CONFIG_THREE, 0],
                      data_tensor.f_buffer_ub[CONFIG_THREE, 0],
                      data_tensor.half_wb_hb_buffer_ub[shape.one_buf_count],
                      data_tensor.repeat_scalar, CONFIG_FOUR,
                      CONFIG_ONE, CONFIG_ONE, CONFIG_FOUR * CONFIG_EIGHT,
                      CONFIG_EIGHT, CONFIG_EIGHT)
    with tik_instance.for_range(0, shape.one_loop_count // CONFIG_SIXTEEN) as i:
        tik_instance.vtranspose(data_tensor.a_buffer_ub[MATRIX * i],
                                data_tensor.g_buffer_ub[MATRIX * i])
    tik_instance.data_move(data_tensor.result_gm[current_data_address], data_tensor.a_buffer_ub,
                           0, CONFIG_ONE, repeat_data, 0, 0)


def decode_bbox_compute(shape, decode_clip, kernel_name):
    """
      Parameters
      ----------
      shape        : the init shape for calculate
      decode_clip  : parameters for clip bbox
      kernel_name  : kernel_name
      Returns
      -------
      None
   """
    tik_instance = tik.Tik(tik.Dprofile(), True)
    data_tensor = InitTensor(tik_instance, shape)
    data_tensor.repeat_scalar.set_as(
        (shape.one_buf_count + (CONFIG_BLOCK_SIZE - CONFIG_ONE)) // CONFIG_BLOCK_SIZE)
    tik_instance.vector_dup(CONFIG_BLOCK_SIZE, data_tensor.clip_vector, decode_clip,
                            data_tensor.repeat_scalar, CONFIG_ONE, CONFIG_EIGHT, 0)
    repeat_data = shape.one_loop_shape * CONFIG_TWO // CONFIG_DATA_SIZE
    # calculate the whole 8 core data (8 * loop_time * loop_data) ,maybe the data = 0
    if shape.core_mul_loop >= shape.aicore_num:
        with tik_instance.for_range(0, shape.core_number, block_num=shape.core_number) as block_num:
            with tik_instance.for_range(0, shape.one_core_loop_time,
                                        thread_num=shape.thread_number) as loop_i:
                current_data_address = block_num * shape.one_core_shape + \
                                       (loop_i // shape.one_loop_number) * shape.one_c0_shape + \
                                       (loop_i % shape.one_loop_number) * shape.one_loop_shape
                calculate_process(tik_instance, data_tensor, current_data_address,
                                  repeat_data, shape)
    # calculate the left data (data_all - (8 * loop_time * loop_data))
    if shape.left_one_core_loop != 0:
        with tik_instance.for_range(0, shape.left_one_core_loop_time,
                                    thread_num=shape.left_thread_number) as loop_j:
            current_data_address = shape.eight_core_shape + \
                                   (loop_j // shape.one_loop_number) * shape.one_c0_shape + \
                                   (loop_j % shape.one_loop_number) * shape.one_loop_shape
            calculate_process(tik_instance, data_tensor, current_data_address,
                              repeat_data, shape)
    if shape.tail_shape != 0:
        current_data_address = shape.eight_core_shape + \
                               shape.left_one_core_loop_time * shape.one_loop_shape
        repeat_data_tail = ((((shape.proposal_shape * CONFIG_FOUR) - current_data_address)
                             * CONFIG_TWO) + (CONFIG_DATA_SIZE - CONFIG_ONE)) \
                           // CONFIG_DATA_SIZE
        calculate_process(tik_instance, data_tensor, current_data_address,
                          repeat_data_tail, shape)

    tik_instance.BuildCCE(kernel_name=kernel_name,
                          inputs=(data_tensor.box_predictions, data_tensor.anchors),
                          outputs=data_tensor.result_gm, enable_l2=False)


def check_format_shape(format_box_predictions, format_anchors, format_decoded_boxes):
    """
      Parameters
      ----------
      format_box_predictions : format_box_predictions
      format_anchors  : format_anchors
      format_decoded_boxes  : format_decoded_boxes
      Returns
      -------
      None
   """
    if format_box_predictions not in ("NCHW", "ND", "NHWC"):
        raise RuntimeError(
            "format of box_predictions must be type of NHWC or ND or NCHW")
    if format_anchors not in ("NCHW", "ND", "NHWC"):
        raise RuntimeError(
            "format of anchors must be type of NHWC or ND or NCHW")
    if format_decoded_boxes not in ("NCHW", "ND", "NHWC"):
        raise RuntimeError(
            "format of decoded_boxes must be type of NHWC or ND or NCHW")
    if format_box_predictions != format_anchors:
        raise RuntimeError(
            "format of 2 input must be same")


@util.check_input_type(dict, dict, dict, float, str)
def decode_bbox(box_predictions, anchors, decoded_boxes, decode_clip, kernel_name="decode_bbox"):
    """
    calculating data

    Parameters
    ----------
    box_predictions : shape and dtype of input
    anchors : shape and dtype of input
    decoded_boxes : shape and dtype of output, should be same shape and type as input
    decode_clip : decode_clip
    kernel_name : kernel name, default value is "decode_bbox"
    Returns
    -------
    None
    """

    # check param & data
    shape_box_predictions = box_predictions.get("shape")
    shape_anchors = anchors.get("shape")
    shape_decoded_boxes = decoded_boxes.get("shape")
    util.check_kernel_name(kernel_name)
    format_box_predictions = box_predictions.get("format")
    format_anchors = anchors.get("format")
    format_decoded_boxes = decoded_boxes.get("format")
    check_format_shape(format_box_predictions, format_anchors,
                       format_decoded_boxes)
    util.check_shape_rule(shape_box_predictions,
                          CONFIG_TWO, CONFIG_FOUR, None)
    util.check_shape_rule(shape_anchors,
                          CONFIG_TWO, CONFIG_FOUR, None)
    util.check_shape_rule(shape_decoded_boxes,
                          CONFIG_TWO, CONFIG_TWO, None)
    util.check_shape_size(shape_box_predictions, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape_anchors, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape_decoded_boxes, SHAPE_SIZE_LIMIT)
    input_data_type = box_predictions.get("dtype").lower()
    util.check_dtype_rule(input_data_type, ("float16",))
    input_data_type = anchors.get("dtype").lower()
    util.check_dtype_rule(input_data_type, ("float16",))
    output_data_type = decoded_boxes.get("dtype").lower()
    util.check_dtype_rule(output_data_type, ("float16",))
    if (reduce(lambda x, y: x * y, shape_box_predictions[:])) \
            != (reduce(lambda x, y: x * y, shape_decoded_boxes[:])) or \
            (reduce(lambda x, y: x * y, shape_anchors[:])) \
            != (reduce(lambda x, y: x * y, shape_decoded_boxes[:])):
        raise RuntimeError("the input shape(shape_box_predictions or anchors)"
                           "is not equal to shape_decoded_boxes")
    if not isinstance(decode_clip, (float, int)):
        raise RuntimeError("input param type of decode_clip should be Float")
    if decode_clip < 0 or decode_clip > 10:
        raise RuntimeError("input param decode_clip can't be negtive and shoud be [0,10]! ")
    if shape_decoded_boxes[-1] != CONFIG_FOUR:
        raise RuntimeError("output decoded_boxes shape D should equal to 4)")
    # init the tiling shape
    shape = TilingFunc(shape_box_predictions)
    # calculate the deocede_bbox
    decode_bbox_compute(shape, decode_clip, kernel_name)
