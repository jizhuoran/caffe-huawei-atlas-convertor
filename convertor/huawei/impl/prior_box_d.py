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

prior_box
"""

# pylint: disable=locally-disabled,ungrouped-imports,import-error,unused-import,wrong-import-order
from te import platform as tbe_platform
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te import tvm
import topi
from te.platform.cce_build import build_config
import numpy
import math
from functools import reduce as functools_reduce

# size of 5HD format
DIM_5HD = 5
# size of c0 for fp16 fp32
C0 = 16
# size of c0 for uint8 int8
INT8_C0_SIZE = 32
# index of height in 5HD format
INDEX_H = 2
# index of width in 5HD format
INDEX_W = 3


# pylint: disable=locally-disabled,too-many-arguments
def check_parameters(min_size, max_size, img_h, img_w,
                     step_h, step_w, variance):
    if len(min_size) <= 0:
        raise RuntimeError("must provide min_size.")
    min_size_list = list(range(len(min_size)))
    for i in min_size_list:
        if min_size[i] <= 0:
            raise RuntimeError("min_size must be positive. \
                actual min_size value is %d" % (min_size[i]))

    if len(max_size) > 0:
        if len(max_size) != len(min_size):
            raise RuntimeError("max_size_size must be equal to min_size_size, \
                while max_size_size is %d, min_size_size is %d." \
                % (len(max_size), len(min_size)))
        max_size_list = list(range(len(max_size)))
        for i in max_size_list:
            if max_size[i] <= min_size[i]:
                raise RuntimeError("max_size must be greater than min_size, \
                    while actual max_size is %f, actual min_size is %f." \
                    % (max_size[i], min_size[i]))

    if img_h != 0 or img_w != 0:
        if img_h < 0:
            raise RuntimeError("img_h should be larger than 0, \
                while actual img_h is %d" % (img_h))
        if img_w < 0:
            raise RuntimeError("img_w should be larger than 0, \
                while actual img_w is %d" % (img_w))
    else:
        img_h = 0
        img_w = 0

    if step_h != 0 or step_w != 0:
        if step_h < 0:
            raise RuntimeError("step_h should be larger than 0, \
                while actual step_h is %f" % (step_h))
        if step_w < 0:
            raise RuntimeError("step_w should be larger than 0, \
                while actual step_w is %f" % (step_w))
    else:
        step_h = 0
        step_w = 0

    if len(variance) > 1:
        if len(variance) != 4:
            raise RuntimeError("Must and only provide 4 variance, \
                while actual number is %f" % (len(variance)))
    variance_list = list(range(len(variance)))
    for i in variance_list:
        if variance[i] <= 0:
            raise RuntimeError("variance value must be larger than 0, \
                while actual variance value is %f" % (variance[i]))

    return img_h, img_w, step_h, step_w


# pylint: disable=locally-disabled,too-many-arguments,too-many-locals
def prior_box_check(feature, img, data_h, data_w, min_size, max_size,
                    img_h, img_w, step_h, step_w, variance, kernel_name):
    shape_feature = feature.get("shape")
    shape_img = img.get("shape")
    shape_h_data = data_h.get("shape")
    shape_w_data = data_w.get("shape")

    util.check_shape_rule(shape_feature)
    util.check_shape_rule(shape_img)
    util.check_shape_rule(shape_h_data)
    util.check_shape_rule(shape_w_data)
    util.check_tensor_shape_size(shape_feature)
    util.check_tensor_shape_size(shape_img)
    util.check_tensor_shape_size(shape_h_data)
    util.check_tensor_shape_size(shape_w_data)

    dtype = feature.get("dtype")
    feature_dtype = dtype.lower()
    feature_format = feature.get("format")

    product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
    if product in ("Hi3796CV300ES"):
        check_list = ["float16"]
    else:
        check_list = ["float16", "float32"]

    if feature_dtype not in check_list:
        raise RuntimeError("prior_box only support %s."
                           % (",".join(check_list)))

    if feature_format not in ("NCHW", "NC1HWC0"):
        raise RuntimeError(
            "format is invalid, format only support NCHW and 5HD")

    if feature_format == "NC1HWC0":
        if len(shape_feature) != DIM_5HD:
            raise RuntimeError(
                "The dim of tensor must be %d"
                ", actual dim is %d" % (DIM_5HD, len(shape_feature)))

        shape_c0 = C0
        if shape_feature[DIM_5HD - 1] != shape_c0:
            raise RuntimeError(
                "The value of C0 must be %d,"
                " actual input is (%d)"
                % (shape_c0, shape_feature[DIM_5HD - 1]))


    util.check_kernel_name(kernel_name)
    img_h, img_w, step_h, step_w = \
    check_parameters(min_size, max_size, img_h, img_w,
                     step_h, step_w, variance)
    return img_h, img_w, step_h, step_w


def buffer_mapping(schedule, op_list):
    """
    buffer data
    Parameters
    ---------
    schedule : op schedule
    oplist: list of op
    Returns
    -------
    None
    """
    for i_op in op_list:
        if i_op.op.name == "result":
            continue
        schedule[i_op].set_scope(tbe_platform.scope_ubuf)


def ins_emit(schedule, op_list, axis_list, ins_list):
    """
    when int8 or uint8 spilt  axis to cal
    Parameters
    ---------

    schedule:schedule
    op_list:ins list
    axis_list:axis list
    ins_list:ins list

    Returns
    -------
    NOne
"""
    length = len(op_list)
    for i in range(0, length):
        schedule[op_list[i]].emit_insn(axis_list[i], ins_list[i])


# pylint: disable=locally-disabled,too-many-arguments,too-many-locals,too-many-statements,invalid-name
def prior_box_compute(feature, img, data_h, data_w, box_height, box_width, y, \
        rec_img_h, rec_img_w, step_h, step_w, clip, offset, scale, variance):
    """
    calculating data

    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input_x
    output_y : dict
        dict of output_y, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "prior_box"

    Returns
    -------
    output tensor
    """

    """
    TODO:
    Please refer to the TE DSL Manual, And code here with TE DSL.
    """
    tensor_dic = {}
    tensor_list = []
    op_list = []
    ins_list = []
    shape_data_h = data_h.get("shape")
    shape_data_w = data_w.get("shape")
    data_dtype = data_h.get("dtype")
    shape_box = box_height.get("shape")
    box_dtype = box_height.get("dtype")
    shape_image = img.get("shape")
    shape_image_h = shape_image[2]
    feature_input = tvm.placeholder(shape_data_h, name="feature_input", \
        dtype=data_dtype)
    img_input = tvm.placeholder((shape_image_h,), name="img_input", \
        dtype=data_dtype)
    data_h_input = tvm.placeholder(shape_data_h, name="data_h_input", \
        dtype=data_dtype)
    data_w_input = tvm.placeholder(shape_data_w, name="data_w_input", \
        dtype=data_dtype)
    box_height_input = tvm.placeholder(shape_box, name="box_h_input", \
        dtype=box_dtype)
    box_width_input = tvm.placeholder(shape_box, name="box_w_input", \
        dtype=box_dtype)
    tensor_list.append(feature_input)
    tensor_list.append(img_input)
    tensor_list.append(data_h_input)
    tensor_list.append(data_w_input)
    tensor_list.append(box_height_input)
    tensor_list.append(box_width_input)

    feature_ub = tvm.compute(shape_data_h, lambda *i: feature_input(*i), \
                             name="feature_ub")
    img_ub = tvm.compute((shape_image_h,), lambda *i: img_input(*i), \
                         name="img_ub")
    tensor_dic["feature_ub"] = feature_ub
    op_list += [feature_ub]
    ins_list += ["dma_copy"]
    tensor_dic["img_ub"] = img_ub
    op_list += [img_ub]
    ins_list += ["dma_copy"]

    move_value = tvm.const(0.0, data_dtype)
    feature_move = tvm.compute(shape_data_h, \
                               lambda *i: feature_ub(*i) * move_value, \
                               name="feature_move")
    img_move = tvm.compute((shape_image_h,), \
        lambda *i: img_ub(*i) * move_value, name="img_move")
    tensor_dic["feature_move"] = feature_move
    op_list += [feature_move]
    ins_list += ["vector_muls"]
    tensor_dic["img_move"] = img_move
    op_list += [img_move]
    ins_list += ["vector_muls"]

    data_h_ub_temp = tvm.compute(shape_data_h, lambda *i: data_h_input(*i), \
                                 name="data_h_ub_temp")
    data_w_ub = tvm.compute(shape_data_w, lambda *i: data_w_input(*i), \
                            name="data_w_ub")
    box_height_ub = tvm.compute(shape_box, lambda *i: box_height_input(*i), \
                                name="box_height_ub")
    box_width_ub = tvm.compute(shape_box, lambda *i: box_width_input(*i), \
                               name="box_width_ub")
    tensor_dic["data_h_ub_temp"] = data_h_ub_temp
    op_list += [data_h_ub_temp]
    ins_list += ["dma_copy"]
    tensor_dic["data_w_ub"] = data_w_ub
    op_list += [data_w_ub]
    ins_list += ["dma_copy"]
    tensor_dic["box_height_ub"] = box_height_ub
    op_list += [box_height_ub]
    ins_list += ["dma_copy"]
    tensor_dic["box_width_ub"] = box_width_ub
    op_list += [box_width_ub]
    ins_list += ["dma_copy"]

    data_h_ub_temp1 = tvm.compute(shape_data_h, \
        lambda *i: data_h_ub_temp(*i) + feature_move(*i), \
        name="data_h_ub_temp1")
    data_h_ub = tvm.compute(shape_data_h, \
        lambda *i: data_h_ub_temp1(*i) + img_move[0], name="data_h_ub")
    tensor_dic["data_h_ub_temp1"] = data_h_ub_temp1
    op_list += [data_h_ub_temp1]
    ins_list += ["vector_add"]
    tensor_dic["data_h_ub"] = data_h_ub
    op_list += [data_h_ub]
    ins_list += ["vector_adds"]

    offset_value = tvm.const(offset, data_dtype)
    step_w_value = tvm.const(step_w, data_dtype)
    step_h_value = tvm.const(step_h, data_dtype)
    rec_img_w_value = tvm.const(rec_img_w, data_dtype)
    rec_img_h_value = tvm.const(rec_img_h, data_dtype)
    scale_value = tvm.const(scale, data_dtype)
    scale_oppo = 0 - scale
    scale_value_oppo = tvm.const(scale_oppo, data_dtype)

    # define 1 or 4 variance_value
    if len(variance) == 1:
        variance_value = tvm.const(variance[0], data_dtype)
    else:
        variance_value0 = tvm.const(variance[0], data_dtype)
        variance_value1 = tvm.const(variance[1], data_dtype)
        variance_value2 = tvm.const(variance[2], data_dtype)
        variance_value3 = tvm.const(variance[3], data_dtype)

    w_offset = tvm.compute(shape_data_w, \
        lambda *i: data_w_ub(*i) + offset_value, name="w_offset")
    h_offset = tvm.compute(shape_data_h, \
        lambda *i: data_h_ub(*i) + offset_value, name="h_offset")
    center_x = tvm.compute(shape_data_w, \
        lambda *i: w_offset(*i) * step_w_value, name="center_x")
    center_y = tvm.compute(shape_data_h, \
        lambda *i: h_offset(*i) * step_h_value, name="center_y")
    tensor_dic["w_offset"] = w_offset
    op_list += [w_offset]
    ins_list += ["vector_adds"]
    tensor_dic["h_offset"] = h_offset
    op_list += [h_offset]
    ins_list += ["vector_adds"]
    tensor_dic["center_x"] = center_x
    op_list += [center_x]
    ins_list += ["vector_muls"]
    tensor_dic["center_y"] = center_y
    op_list += [center_y]
    ins_list += ["vector_muls"]

    box_width_scale = tvm.compute(shape_box, \
        lambda *i: box_width_ub(*i) * scale_value, name="box_width_scale")
    box_height_scale = tvm.compute(shape_box, \
        lambda *i: box_height_ub(*i) * scale_value, name="box_height_scale")
    box_width_scale_oppo = tvm.compute(shape_box, \
        lambda *i: box_width_ub(*i) * scale_value_oppo, \
        name="box_width_scale_oppo")
    box_height_scale_oppo = tvm.compute(shape_box, \
        lambda *i: box_height_ub(*i) * scale_value_oppo, \
        name="box_height_scale_oppo")
    tensor_dic["box_width_scale"] = box_width_scale
    op_list += [box_width_scale]
    ins_list += ["vector_muls"]
    tensor_dic["box_height_scale"] = box_height_scale
    op_list += [box_height_scale]
    ins_list += ["vector_muls"]
    tensor_dic["box_width_scale_oppo"] = box_width_scale_oppo
    op_list += [box_width_scale_oppo]
    ins_list += ["vector_muls"]
    tensor_dic["box_height_scale_oppo"] = box_height_scale_oppo
    op_list += [box_height_scale_oppo]
    ins_list += ["vector_muls"]


    num_box = shape_box[0]
    h_length = shape_data_h[0]
    w_length = shape_data_w[0]

    center_x_minus_calc = tvm.compute((w_length, num_box), \
        lambda w, c: center_x[w, 0, 0, 0] + box_width_scale_oppo[c, 0, 0, 0], \
        name="center_x_minus_calc")
    center_y_minus_calc = tvm.compute((h_length, num_box), \
        lambda h, c: center_y[h, 0, 0, 0] + box_height_scale_oppo[c, 0, 0, 0],\
        name="center_y_minus_calc")
    center_x_add_calc = tvm.compute((w_length, num_box), \
        lambda w, c: center_x[w, 0, 0, 0] + box_width_scale[c, 0, 0, 0], \
        name="center_x_add_calc")
    center_y_add_calc = tvm.compute((h_length, num_box), \
        lambda h, c: center_y[h, 0, 0, 0] + box_height_scale[c, 0, 0, 0], \
        name="center_y_add_calc")
    tensor_dic["center_x_minus_calc"] = center_x_minus_calc
    op_list += [center_x_minus_calc]
    ins_list += ["vector_add"]
    tensor_dic["center_y_minus_calc"] = center_y_minus_calc
    op_list += [center_y_minus_calc]
    ins_list += ["vector_add"]
    tensor_dic["center_x_add_calc"] = center_x_add_calc
    op_list += [center_x_add_calc]
    ins_list += ["vector_add"]
    tensor_dic["center_y_add_calc"] = center_y_add_calc
    op_list += [center_y_add_calc]
    ins_list += ["vector_add"]

    top_data_xmin = tvm.compute((w_length, num_box), \
        lambda *i: center_x_minus_calc(*i) * rec_img_w_value, \
        name="top_data_xmin")
    top_data_ymin = tvm.compute((h_length, num_box), \
        lambda *i: center_y_minus_calc(*i) * rec_img_h_value, \
        name="top_data_ymin")
    top_data_xmax = tvm.compute((w_length, num_box), \
        lambda *i: center_x_add_calc(*i) * rec_img_w_value, \
        name="top_data_xmax")
    top_data_ymax = tvm.compute((h_length, num_box), \
        lambda *i: center_y_add_calc(*i) * rec_img_h_value, \
        name="top_data_ymax")
    tensor_dic["top_data_xmin"] = top_data_xmin
    op_list += [top_data_xmin]
    ins_list += ["vector_muls"]
    tensor_dic["top_data_ymin"] = top_data_ymin
    op_list += [top_data_ymin]
    ins_list += ["vector_muls"]
    tensor_dic["top_data_xmax"] = top_data_xmax
    op_list += [top_data_xmax]
    ins_list += ["vector_muls"]
    tensor_dic["top_data_ymax"] = top_data_ymax
    op_list += [top_data_ymax]
    ins_list += ["vector_muls"]

    top_data_res1 = tvm.compute((h_length, w_length, num_box), \
        lambda a, b, c: top_data_xmin[b, c] + move_value, \
        name="top_data_res1")
    top_data_res2 = tvm.compute((h_length, w_length, num_box), \
        lambda a, b, c: top_data_ymin[a, c] + move_value, \
        name="top_data_res2")
    top_data_res3 = tvm.compute((h_length, w_length, num_box), \
        lambda a, b, c: top_data_xmax[b, c] + move_value, \
        name="top_data_res3")
    top_data_res4 = tvm.compute((h_length, w_length, num_box), \
        lambda a, b, c: top_data_ymax[a, c] + move_value, \
        name="top_data_res4")
    tensor_dic["top_data_res1"] = top_data_res1
    op_list += [top_data_res1]
    ins_list += ["vector_add"]
    tensor_dic["top_data_res2"] = top_data_res2
    op_list += [top_data_res2]
    ins_list += ["vector_add"]
    tensor_dic["top_data_res3"] = top_data_res3
    op_list += [top_data_res3]
    ins_list += ["vector_add"]
    tensor_dic["top_data_res4"] = top_data_res4
    op_list += [top_data_res4]
    ins_list += ["vector_add"]
    top_data = tvm.compute((h_length, w_length, num_box, 4), \
    lambda a, b, c, idx: tvm.select(idx < 1, top_data_res1[a, b, c], \
                           tvm.select(idx < 2, top_data_res2[a, b, c], \
                              tvm.select(idx < 3, top_data_res3[a, b, c], \
                                top_data_res4[a, b, c], \
                                ))), name="top_data")


    tensor_dic["top_data"] = top_data
    op_list += [top_data]
    ins_list += ["data_mov"]

    top_data_true = top_data
    if clip:
        top_data_temp = tvm.compute((h_length, w_length, num_box, 4), \
            lambda *i: tvm.max(top_data(*i), 0), name="top_data_temp")
        top_data_true = tvm.compute((h_length, w_length, num_box, 4), \
            lambda *i: tvm.min(top_data_temp(*i), 1), name="top_data_true")
        tensor_dic["top_data_temp"] = top_data_temp
        op_list += [top_data_temp]
        ins_list += ["vector_maxs"]
        tensor_dic["top_data_true"] = top_data_true
        op_list += [top_data_true]
        ins_list += ["vector_mins"]

    if len(variance) == 1:
        variance_data = tvm.compute((h_length, w_length, num_box, 4), \
         lambda a, b, c, idx: tvm.select(idx < 1, variance_value, \
                                tvm.select(idx < 2, variance_value, \
                                 tvm.select(idx < 3, variance_value, \
                                    variance_value, \
                                    ))), name="variance_data")
        tensor_dic["variance_data"] = variance_data
        op_list += [variance_data]
        ins_list += ["data_mov"]
    else:
        variance_data = tvm.compute((h_length, w_length, num_box, 4), \
         lambda a, b, c, idx: tvm.select(idx < 1, variance_value0, \
                                tvm.select(idx < 2, variance_value1, \
                                 tvm.select(idx < 3, variance_value2, \
                                    variance_value3, \
                                    ))), name="variance_data")
        tensor_dic["variance_data"] = variance_data
        op_list += [variance_data]
        ins_list += ["data_mov"]

    y = tvm.compute((1, 2, h_length, w_length, num_box, 4), \
     lambda i, idx, j, k, l, m: tvm.select(idx == 1, \
                                           variance_data[j, k, l, m],
                                           top_data_true[j, k, l, m], \
                                           ), name='result')
    tensor_dic["y"] = y
    op_list += [y]
    ins_list += ["dma_copy"]
    tensor_list.append(y)
    return op_list, ins_list, tensor_dic, y, tensor_list


def get_compute_axis(schedule, tensor_dic):
    compute_at_axis = {}
    compute_at_axis["feature_ub_axis"] = \
        schedule[tensor_dic.get("y")].op.axis[3]
    compute_at_axis["img_ub_axis"] = \
        schedule[tensor_dic.get("y")].op.axis[3]
    compute_at_axis["feature_move_axis"] = \
        schedule[tensor_dic.get("y")].op.axis[3]
    compute_at_axis["img_move_axis"] = \
        schedule[tensor_dic.get("y")].op.axis[3]
    compute_at_axis["data_h_ub_temp_axis"] = \
        schedule[tensor_dic.get("y")].op.axis[3]
    compute_at_axis["data_w_ub_axis"] = \
        schedule[tensor_dic.get("y")].op.axis[3]
    compute_at_axis["box_height_ub_axis"] = \
        schedule[tensor_dic.get("y")].op.axis[3]
    compute_at_axis["box_width_ub_axis"] = \
        schedule[tensor_dic.get("y")].op.axis[3]
    compute_at_axis["data_h_ub_temp1_axis"] = \
        schedule[tensor_dic.get("y")].op.axis[3]
    compute_at_axis["data_h_ub_axis"] = \
        schedule[tensor_dic.get("y")].op.axis[3]
    compute_at_axis["w_offset_axis"] = \
        schedule[tensor_dic.get("y")].op.axis[3]
    compute_at_axis["h_offset_axis"] = \
        schedule[tensor_dic.get("y")].op.axis[3]
    compute_at_axis["center_x_axis"] = \
        schedule[tensor_dic.get("y")].op.axis[3]
    compute_at_axis["center_y_axis"] = \
        schedule[tensor_dic.get("y")].op.axis[3]
    compute_at_axis["box_width_scale_axis"] = \
        schedule[tensor_dic.get("y")].op.axis[3]
    compute_at_axis["box_height_scale_axis"] = \
        schedule[tensor_dic.get("y")].op.axis[3]
    compute_at_axis["box_width_scale_oppo_axis"] = \
        schedule[tensor_dic.get("y")].op.axis[3]
    compute_at_axis["box_height_scale_oppo_axis"] = \
        schedule[tensor_dic.get("y")].op.axis[3]

    compute_at_axis["center_x_minus_calc_axis"] = \
    schedule[tensor_dic.get("top_data_xmin")].op.axis[0]
    compute_at_axis["center_y_minus_calc_axis"] = \
    schedule[tensor_dic.get("top_data_ymin")].op.axis[0]
    compute_at_axis["center_x_add_calc_axis"] = \
    schedule[tensor_dic.get("top_data_xmax")].op.axis[0]
    compute_at_axis["center_y_add_calc_axis"] = \
    schedule[tensor_dic.get("top_data_ymax")].op.axis[0]

    compute_at_axis["top_data_xmin_axis"] = \
    schedule[tensor_dic.get("top_data_res1")].op.axis[1]
    compute_at_axis["top_data_ymin_axis"] = \
    schedule[tensor_dic.get("top_data_res2")].op.axis[0]
    compute_at_axis["top_data_xmax_axis"] = \
    schedule[tensor_dic.get("top_data_res3")].op.axis[1]
    compute_at_axis["top_data_ymax_axis"] = \
    schedule[tensor_dic.get("top_data_res4")].op.axis[0]

    compute_at_axis["top_data_res1_axis"] = \
    schedule[tensor_dic.get("top_data")].op.axis[1]
    compute_at_axis["top_data_res2_axis"] = \
    schedule[tensor_dic.get("top_data")].op.axis[1]
    compute_at_axis["top_data_res3_axis"] = \
    schedule[tensor_dic.get("top_data")].op.axis[1]
    compute_at_axis["top_data_res4_axis"] = \
    schedule[tensor_dic.get("top_data")].op.axis[1]

    compute_at_axis["top_data_axis"] = \
    schedule[tensor_dic.get("y")].op.axis[3]
    compute_at_axis["variance_data_axis"] = \
    schedule[tensor_dic.get("y")].op.axis[3]

    return compute_at_axis





def prior_compute(schedule, ops, axis):

    if not ops:
        raise RuntimeError("operation list is empty")
    length = len(ops)
    if length < 2:
        # no op need integrating
        return
    integration_op = schedule[ops[-1]]
    for i in range(0, length-1):
        schedule[ops[i]].compute_at(integration_op, axis)



def get_ins_emit_axis(ops, last_axis):
    if not ops:
        raise RuntimeError("operation list is empty")
    axis_list = []
    length = len(ops)
    for i in range(0, length-1):
            axis_list += [ops[i].op.axis[0]]
    axis_list += [last_axis]
    return axis_list



def double_buf(schedule, ops):
    if not ops:
        raise RuntimeError("operation list is empty")
    length = len(ops)
    if length < 2:
        # no op need double buffer
        return
    for i in range(0, 6):
        schedule[ops[i]].double_buffer()


def multicore_factor_calculate(shape, element):
    """
    the compute produce, calculate multicore information
    """
    if not shape:
        raise RuntimeError("input shape is empty")

    device_core_num = \
        tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)

    split_axis = -1
    split_size = 0
    shape_0 = int(shape[0])
    shape_1 = int(shape[1])
    shape_2 = int(shape[2])
    shape_3 = int(shape[3])
    shape_4 = int(shape[4])
    shape_5 = int(shape[5])
    out_size = shape_0 * shape_1 * shape_2 * shape_3 * shape_4 * shape_5

    if shape_0 >= device_core_num:
        npart_0 = device_core_num
        npart_1 = 1
        npart_2 = 1
        npart_3 = 1
        npart_4 = 1
        split_axis = 0
        split_size = math.ceil(shape_0 // device_core_num)
    elif device_core_num // shape_0 <= shape_1:
        npart_0 = shape_0
        npart_1 = device_core_num // shape_0
        npart_2 = 1
        npart_3 = 1
        npart_4 = 1
        split_axis = 1
        split_size = math.ceil(shape_1 / (device_core_num // shape_0))
    elif device_core_num // shape_0 // shape_1 <= shape_2:
        npart_0 = shape_0
        npart_1 = shape_1
        npart_2 = (device_core_num // shape_0 // shape_1)
        npart_3 = 1
        npart_4 = 1
        split_axis = 2
        split_size = math.ceil(shape_2 / (device_core_num // shape_0 \
                               // shape_1))
    elif device_core_num // shape_0 // shape_1 // shape_2 <= shape_3:
        npart_0 = shape_0
        npart_1 = shape_1
        npart_2 = shape_2
        npart_3 = (device_core_num // shape_0 // shape_1 // shape_2)
        npart_4 = 1
        split_axis = 3
        split_size = math.ceil(shape_3 / (device_core_num // shape_0 \
                               // shape_1 // shape_2))
    elif device_core_num // shape_0 // shape_1 // shape_2 // shape_3 \
         <= shape_4:
        npart_0 = shape_0
        npart_1 = shape_1
        npart_2 = shape_2
        npart_3 = shape_3
        npart_4 = (device_core_num // shape_0 // shape_1 // shape_2 \
                   // shape_3)
        split_axis = 4
        split_size = math.ceil(shape_4 / (device_core_num // shape_0 \
                               // shape_1 // shape_2 // shape_3))
    else:
        npart_0 = shape_0
        npart_1 = shape_1
        npart_2 = shape_2
        npart_3 = shape_3
        npart_4 = shape_4
        split_axis = 5
        split_size = 1

    total_npart = npart_0 * npart_1 * npart_2 * npart_3 * npart_4
    fuse_num = -1
    if out_size % total_npart != element:
        if shape_1 * shape_5 % element == 0:
            fuse_num = 3
            npart_0 = 1
            npart_1 = 1
            npart_2 = shape_2
            npart_3 = shape_3
            npart_4 = shape_4
            split_axis = 2
            split_size = 1
        elif shape_1 * shape_5 * shape_4 % element == 0:
            fuse_num = 2
            npart_0 = 1
            npart_1 = 1
            npart_2 = shape_2
            npart_3 = shape_3
            npart_4 = 1
            split_axis = 2
            split_size = 1
        elif shape_1 * shape_5 * shape_4 * shape_3 % element == 0:
            fuse_num = 1
            npart_0 = 1
            npart_1 = 1
            npart_2 = shape_2
            npart_3 = 1
            npart_4 = 1
            split_axis = 2
            split_size = 1
            if shape_2 == 1:
                fuse_num = 0
                split_axis = 0
                split_size = 1
        else:
            fuse_num = 0
            npart_0 = shape_0
            npart_1 = 1
            npart_2 = 1
            npart_3 = 1
            npart_4 = 1
            split_axis = 0
            split_size = 1
    return npart_0, npart_1, npart_2, npart_3, npart_4, split_axis, split_size, fuse_num


def tiling_factor_calculate(shape, split_axis_0, split_size, dtype, UB_SIZE_LIMIT, fuse_num):
    """
    do tiling calculate
    """
    if not shape:
        raise RuntimeError("input shape is empty")

    feature_dtype = dtype.lower()
    shape_0 = int(shape[0])
    shape_1 = int(shape[1])
    shape_2 = int(shape[2])
    shape_3 = int(shape[3])
    shape_4 = int(shape[4])
    shape_5 = int(shape[5])

    if feature_dtype == 'float32':
        temp = UB_SIZE_LIMIT // (shape_5 * 4)
    else:
        temp = UB_SIZE_LIMIT // (shape_5 * 2)

    split_flag = False
    split_axis = 0
    split_factor = 0
    if split_axis_0 == 0:
        if fuse_num == -1:
            if temp >= split_size * shape_1 * shape_2 * shape_3 * shape_4:
                # no split
                split_flag = False
            elif temp < split_size * shape_1 * shape_2 * shape_3 * shape_4 and \
                 temp >= shape_1 * shape_2 * shape_3 * shape_4:
                # split on n.inner
                split_flag = True
                split_axis = 0
                split_factor = int(temp // (shape_1 * shape_2 * shape_3 * shape_4))
            elif temp < shape_1 * shape_2 * shape_3 * shape_4 and temp >= \
                 shape_2 * shape_3 * shape_4:
                # split on h
                split_flag = True
                split_axis = 1
                split_factor = int(temp // (shape_2 * shape_3 * shape_4))
            elif temp < shape_2 * shape_3 * shape_4 and temp >= shape_3 * shape_4:
                split_flag = True
                split_axis = 2
                split_factor = int(temp // (shape_3 * shape_4))
            elif temp < shape_3 * shape_4 and temp >= shape_4:
                split_flag = True
                split_axis = 3
                split_factor = int(temp // shape_4)
            elif temp < shape_4:
                # split on w
                split_flag = True
                split_axis = 4
                split_factor = int(temp)
        else:
            if temp >= split_size * shape_1 * shape_2 * shape_3 * shape_4:
                # no split
                split_flag = False
            elif temp < shape_1 * shape_2 * shape_3 * shape_4 and temp >= shape_2 * shape_3 * shape_4:
                # split on h
                split_flag = True
                split_axis = 1
                split_factor = int(temp // (shape_2 * shape_3 * shape_4))
            elif temp < shape_2 * shape_3 * shape_4 and temp >= shape_3 * shape_4:
                # split on h
                split_flag = True
                split_axis = 2
                split_factor = int(temp // (shape_3 * shape_4))
            elif temp < shape_3 * shape_4 and temp >= shape_4:
                # split on h
                split_flag = True
                split_axis = 3
                split_factor = int(temp // shape_4)
            elif temp < shape_4:
                # split on w
                split_flag = True
                split_axis = 4
                split_factor = int(temp)
    if split_axis_0 == 1:
        if temp >= split_size * shape_2 * shape_3 * shape_4:
            # no split
            split_flag = False
        elif temp < split_size * shape_2 * shape_3 * shape_4 and temp >= \
             shape_2 * shape_3 * shape_4:
            # split on h
            split_flag = True
            split_axis = 1
            split_factor = int(temp // (shape_2 * shape_3 * shape_4))
        elif temp < shape_2 * shape_3 * shape_4 and temp >= shape_3 * shape_4:
            # split on h
            split_flag = True
            split_axis = 2
            split_factor = int(temp // (shape_3 * shape_4))
        elif temp < shape_3 * shape_4 and temp >= shape_4:
            # split on h
            split_flag = True
            split_axis = 3
            split_factor = int(temp // shape_4)
        elif temp < shape_4:
            # split on w
            split_flag = True
            split_axis = 4
            split_factor = int(temp)
    if split_axis_0 == 2:
        if fuse_num == -1:
            if temp >= split_size * shape_3 * shape_4:
                # no split
                split_flag = False
            elif temp < split_size * shape_3 * shape_4 and temp >= shape_3 * shape_4:
                # split on h
                split_flag = True
                split_axis = 2
                split_factor = int(temp // (shape_3 * shape_4))
            elif temp < shape_3 * shape_4 and temp >= shape_4:
                # split on h
                split_flag = True
                split_axis = 3
                split_factor = int(temp // shape_4)
            elif temp < shape_4:
                # split on w
                split_flag = True
                split_axis = 4
                split_factor = int(temp)
        elif fuse_num == 3:
            if temp >= split_size * shape_0 * shape_1:
                # no split
                split_flag = False
            elif temp < shape_1:
                # split on w
                split_flag = True
                split_axis = 1
                split_factor = int(temp)
        elif fuse_num == 2:
            if temp >= split_size * shape_0 * shape_1 * shape_4:
                # no split
                split_flag = False
            elif temp < shape_1 * shape_4 and temp >= shape_4:
                # split on h
                split_flag = True
                split_axis = 1
                split_factor = int(temp // shape_4)
            elif temp < shape_4:
                # split on w
                split_flag = True
                split_axis = 4
                split_factor = int(temp)
        elif fuse_num == 1:
            if temp >= split_size * shape_0 * shape_1 * shape_4 * shape_3:
                # no split
                split_flag = False
            elif temp < shape_1 * shape_4 * shape_3 and temp >= shape_4 * shape_3:
                # split on h
                split_flag = True
                split_axis = 1
                split_factor = int(temp // (shape_4 * shape_3))
            elif temp < shape_3 * shape_4 and temp >= shape_4:
                # split on h
                split_flag = True
                split_axis = 3
                split_factor = int(temp // shape_4)
            elif temp < shape_4:
                # split on w
                split_flag = True
                split_axis = 4
                split_factor = int(temp)
    if split_axis_0 == 3:
        if temp >= split_size * shape_4:
            # no split
            split_flag = False
        elif temp < split_size * shape_4 and temp >= shape_4:
            # split on h
            split_flag = True
            split_axis = 3
            split_factor = int(temp // shape_4)
        elif temp < shape_4:
            # split on w
            split_flag = True
            split_axis = 4
            split_factor = int(temp)
    if split_axis_0 == 4:
        if temp >= split_size:
            # no split
            split_flag = False
        else:
            # split on w
            split_flag = True
            split_axis = 4
            split_factor = int(temp)
    if split_axis_0 == 5:
        # no split
        split_flag = False

    return split_flag, split_axis, split_factor

def align(schedule, ops, tensor_dic, clip, factor=16, offset=0):
    """
    determine if aligning needs to be enabled
    """
    if not ops:
        raise RuntimeError("operation list is empty")
    length = len(ops)
    if length <= 3:
        # no op need aligning
        return

    for i in range(0, length):
        shape_len = len(ops[i].shape)
        if shape_len > 1 and tensor_dic.get("top_data")!=ops[i] and \
           tensor_dic.get("variance_data")!=ops[i] and \
           tensor_dic.get("top_data_temp")!=ops[i] and \
           tensor_dic.get("top_data_true")!=ops[i]:
            schedule[ops[i]].storage_align(ops[i].op.axis[shape_len - 1],
                                           factor, offset)


@util.check_input_type(dict, dict, dict, dict, dict, dict, dict, (list, tuple),
                       (list, tuple), int, int, float, float, bool, bool,
                       float, (list, tuple), str)
# pylint: disable=locally-disabled,too-many-arguments,too-many-locals,invalid-name
def prior_box_d(feature, img, data_h, data_w, box_height, box_width, y,
                min_size, max_size, img_h=0, img_w=0, step_h=0.0, step_w=0.0,
                flip=True, clip=False, offset=0.5, variance=[0.1],
                kernel_name="prior_box"):

    """
    calculating data

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "prior_box"

    Returns
    -------
    None
    """

    """
    TODO:
    Please refer to the TE DSL Manual, And code here with TE DSL.
    """
    img_h, img_w, step_h, step_w = prior_box_check(feature, img, data_h, \
     data_w, min_size, max_size, img_h, img_w, step_h, step_w, variance, kernel_name)
    shape_img = img.get("shape")
    shape_feature = feature.get("shape")
    dtype = feature.get("dtype")

    if img_h == 0 or img_w == 0:
        img_h = shape_img[INDEX_H]
        img_w = shape_img[INDEX_W]

    rec_img_h = 1.0 / img_h
    rec_img_w = 1.0 / img_w

    if step_h == 0 or step_w == 0:
        step_h = 1.0 * shape_img[INDEX_H] / shape_feature[INDEX_H]
        step_w = 1.0 * shape_img[INDEX_W] / shape_feature[INDEX_W]
    scale = 0.5

    op_list, ins_list, tensor_dic, y, tensor_list = prior_box_compute(feature, img, data_h, \
     data_w, box_height, box_width, y, rec_img_h, rec_img_w, step_h, step_w, \
    clip, offset, scale, variance)

    UB_SIZE_LIMIT = \
        tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
    UB_SIZE_LIMIT = UB_SIZE_LIMIT / 21

    schedule = tvm.create_schedule(y.op)
    dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(dtype) // 8
    # 32 means one block size(32 Bytes), divide by 32 to get the numbers of data that
    # can be stored in one block.
    element = 32 // dtype_bytes_size
    align(schedule, op_list, tensor_dic, clip, element, 0)


    # muti core
    npart_0, npart_1, npart_2, npart_3, npart_4, split_axis_0, split_size, fuse_num = \
        multicore_factor_calculate(tensor_dic.get("y").shape, element)
    xr1o, xr1i = schedule[y].split(y.op.axis[0], nparts=npart_0)
    xr2o, xr2i = schedule[y].split(y.op.axis[1], nparts=npart_1)
    xho, xhi = schedule[y].split(y.op.axis[2], nparts=npart_2)
    xwo, xwi = schedule[y].split(y.op.axis[3], nparts=npart_3)
    xno, xni = schedule[y].split(y.op.axis[4], nparts=npart_4)

    schedule[y].reorder(xr1o, xr2o, xho, xwo, xno, xr1i, xr2i, xhi, xwi, xni, \
                    y.op.axis[5])
    block_axis = schedule[y].fuse(xr1o, xr2o, xho, xwo, xno)
    schedule[y].bind(block_axis, tvm.thread_axis("blockIdx.x"))

    # tiling strategy
    split_flag, split_axis, split_factor = \
        tiling_factor_calculate(tensor_dic.get("y").shape, split_axis_0, \
            split_size, dtype, UB_SIZE_LIMIT, fuse_num)

    if split_flag:
        if split_axis == 0:
            xo, xi = schedule[y].split(xr1i, factor=split_factor)
        elif split_axis == 1:
            xo, xi = schedule[y].split(xr2i, factor=split_factor)
        elif split_axis == 2:
            xo, xi = schedule[y].split(xhi, factor=split_factor)
        elif split_axis == 3:
            xo, xi = schedule[y].split(xwi, factor=split_factor)
        elif split_axis == 4:
            xo, xi = schedule[y].split(xni, factor=split_factor)


        prior_compute(schedule, op_list, xo)

        buffer_mapping(schedule, op_list)

        double_buf(schedule, op_list)

        axis_list = get_ins_emit_axis(op_list, xi)
        ins_emit(schedule, op_list, axis_list, ins_list)
    else:
        # schedule optimize
        prior_compute(schedule, op_list, block_axis)

        buffer_mapping(schedule, op_list)

        double_buf(schedule, op_list)

        # instructions replace
        if split_axis_0 == 0:
            axis_list = get_ins_emit_axis(op_list, xr1i)
        elif split_axis_0 == 1:
            axis_list = get_ins_emit_axis(op_list, xr2i)
        elif split_axis_0 == 2:
            axis_list = get_ins_emit_axis(op_list, xhi)
        elif split_axis_0 == 3:
            axis_list = get_ins_emit_axis(op_list, xwi)
        elif split_axis_0 == 4 or split_axis_0 == 5:
            axis_list = get_ins_emit_axis(op_list, xni)

        ins_emit(schedule, op_list, axis_list, ins_list)

    with build_config:
        tvm.build(schedule, tensor_list, "cce", name=kernel_name)
