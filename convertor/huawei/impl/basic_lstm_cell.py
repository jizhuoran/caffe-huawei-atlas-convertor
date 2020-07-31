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

basiclstm_cell
"""
# pylint: disable=locally-disabled,ungrouped-imports,import-error
from te import tvm
from te.platform import cce_conf
import te.platform.cce_params as cce
from topi.cce import util
from te.platform.cce_build import build_config
import topi
import te.lang.cce

# Platform specific unit size
C0 = 16

MIN_FP32 = 2**(-126)

NONETYPE = type(None)


def get_nz_dim(shape, unit=16):
    """
    get nz dim
    Parameters
    ----------
    shape : tensor shape
    Returns
    -------
    """
    # Check if already in Nz format
    if len(shape) == 4 and shape[-1] == shape[-2] and shape[-1] == 16:
        return shape
    dim0 = shape[-2]
    dim1 = shape[-1]
    dimremain = tuple(shape[0:-2])
    return dimremain + (dim1 // unit, dim0 // unit, unit, unit)


def get_dtype_size(dtype):
    """
    get_dtype_size
    Parameters
    ----------
    dtype : tensor dtype
    Returns
    -------
    """
    size = 2
    if dtype == "float32":
        size = 4
    return size


def newton_iteration(shape, tensor_x_rec, tensor_x, symbol):
    """
    the function of newton_iteration
    Parameters
    ----------
    shape : tensor shape
    tensor_x_rec : tensor
    tensor_x : tensor
    symbol : tensor symbol
    Returns
    -------
    """
    dtype_c = tensor_x_rec.dtype
    const_num_neg_two = tvm.const(-2, dtype=dtype_c)
    const_num_neg_one = tvm.const(-1, dtype=dtype_c)

    tensor_newton_mul0 = tvm.compute(shape,
                                     lambda *i: tensor_x_rec(*i) * tensor_x(*i),
                                     name="tensor_newton_mul0" + symbol)
    tensor_newton_add = tvm.compute(shape,
                                    lambda *i: tensor_newton_mul0(*i) + const_num_neg_two,
                                    name="tensor_newton_add" + symbol)
    tensor_newton_mul1 = tvm.compute(shape,
                                     lambda *i: tensor_newton_add(*i) * tensor_x_rec(*i),
                                     name="tensor_newton_mul1" + symbol)
    tensor_newton_mul2 = tvm.compute(shape,
                                     lambda *i: tensor_newton_mul1(*i) * const_num_neg_one,
                                     name="tensor_newton_mul2" + symbol)
    return tensor_newton_mul0, tensor_newton_add, tensor_newton_mul1, tensor_newton_mul2


# pylint: disable=locally-disabled,too-many-locals
def sigmoid(shape, tensor_allgate_ub, tensor_one, symbol, is_cloud):
    """
    the function of sigmoid
    Parameters
    ----------
    shape : tensor shape
    tensor_allgate_ub : tensor
    symbol : tensor symbol
    Returns
    -------
    """
    dtype_c = tensor_allgate_ub.dtype
    const_num_neg_one = tvm.const(-1, dtype=dtype_c)
    const_num_one = tvm.const(1, dtype=dtype_c)
    tensor_ub_neg_allgate = tvm.compute(shape,
                                        lambda a, b, c, d: tensor_allgate_ub[a, b, c, d] *
                                        const_num_neg_one,
                                        name="tensor_gate_neg_" + symbol)
    tensor_ub_allgate_exp_fp16 = tensor_ub_neg_allgate
    if not is_cloud:
        tensor_ub_allgate_exp_fp16 = tvm.compute(shape,
                                                 lambda *i: topi.cast(tensor_ub_neg_allgate(*i),
                                                                      "float16"),
                                                 name="tensor_gate_exp_fp16_" + symbol)
    tensor_ub_allgate_exp = tvm.compute(shape, lambda *i: tvm.exp(tensor_ub_allgate_exp_fp16(*i)),
                                        name="tensor_gate_exp_" + symbol)
    tensor_ub_allgate_exp_fp32 = tensor_ub_allgate_exp
    if not is_cloud:
        tensor_ub_allgate_exp_fp32 = tvm.compute(shape,
                                                 lambda *i: topi.cast(tensor_ub_allgate_exp(*i),
                                                                      dtype_c),
                                                 name="tensor_gate_exp_fp32_" + symbol)
    tensor_ub_allgate_add = tvm.compute(shape,
                                        lambda *i: tensor_ub_allgate_exp_fp32(*i) + const_num_one,
                                        name="tensor_gate_add_" + symbol)
    if is_cloud:
        tensor_newton_mul0 = None
        tensor_newton_add = None
        tensor_newton_mul1 = None
        tensor_ub_allgate_sigmoid = tvm.compute(shape,
                                                lambda *i: tensor_one(*i) /
                                                tensor_ub_allgate_add(*i),
                                                name="tensor_gate_sigmoid_" + symbol)
        tensor_newton_mul2 = tensor_ub_allgate_sigmoid
    else:
        tensor_ub_allgate_sigmoid = tvm.compute(shape,
                                                lambda *i: const_num_one /
                                                tensor_ub_allgate_add(*i),
                                                name="tensor_gate_sigmoid_" + symbol)
        tensor_newton_mul0, tensor_newton_add, \
        tensor_newton_mul1, tensor_newton_mul2 = \
            newton_iteration(shape, tensor_ub_allgate_sigmoid, tensor_ub_allgate_add, symbol)

    return tensor_ub_neg_allgate, tensor_ub_allgate_exp_fp16, \
           tensor_ub_allgate_exp, tensor_ub_allgate_exp_fp32, \
           tensor_ub_allgate_add, tensor_ub_allgate_sigmoid, \
           tensor_newton_mul0, tensor_newton_add, \
           tensor_newton_mul1, tensor_newton_mul2


# pylint: disable=locally-disabled,too-many-locals,too-many-statements
def tanh(shape, tensor, symbol, is_cloud):
    """
    the function of tanh
    Parameters
    ----------
    shape : tensor shape
    tensor : tensor
    symbol : tensor symbol
    Returns
    -------
    """
    dtype_c = tensor.dtype
    const_num_neg_one = tvm.const(-1, dtype="float32")
    const_num_one = tvm.const(1, dtype="float32")
    const_num_two = tvm.const(2, dtype=dtype_c)
    const_num_two_fp32 = tvm.const(-2, dtype="float32")
    res = {}
    operation = {}
    if is_cloud:
        tensor_ub_input = tensor
        if dtype_c == "float16":
            tensor_ub_input = tvm.compute(shape,
                                          lambda *i: topi.cast(tensor(*i), "float32"),
                                          name="tensor_ub_input_" + symbol)
        tensor_ub_abs = tvm.compute(shape,
                                    lambda *i: tvm.abs(tensor_ub_input(*i)),
                                    name="tensor_ub_abs_" + symbol)
        tensor_ub_abs_add = tvm.compute(shape,
                                        lambda *i: tensor_ub_abs(*i) + MIN_FP32,
                                        name="tensor_ub_abs_add_" + symbol)
        tensor_ub_tmpdiv = tvm.compute(shape,
                                       lambda *i: tensor_ub_input(*i) / tensor_ub_abs_add(*i),
                                       name="tensor_ub_tmpdiv_" + symbol)
        tensor_ub_mul = tvm.compute(shape,
                                    lambda *i: tensor_ub_tmpdiv(*i) * const_num_neg_one,
                                    name="tensor_ub_mul_" + symbol)
        tensor_ub_abs_mul = tvm.compute(shape,
                                        lambda *i: tensor_ub_abs(*i) * const_num_two_fp32,
                                        name="tensor_ub_abs_mul_" + symbol)
        tensor_ub_exp = tvm.compute(shape,
                                    lambda *i: tvm.exp(tensor_ub_abs_mul(*i)),
                                    name="tensor_ub_exp_" + symbol)
        tensor_ub_sub = tvm.compute(shape,
                                    lambda *i: tensor_ub_exp(*i) + const_num_neg_one,
                                    name="tensor_ub_add_" + symbol)
        tensor_ub_add = tvm.compute(shape,
                                    lambda *i: tensor_ub_exp(*i) + const_num_one,
                                    name="tensor_ub_add_" + symbol)
        tensor_ub_div = tvm.compute(shape,
                                    lambda *i: tensor_ub_sub(*i) / tensor_ub_add(*i),
                                    name="tensor_ub_div_" + symbol)
        tensor_ub_tanh = tvm.compute(shape,
                                     lambda *i: tensor_ub_mul(*i) * tensor_ub_div(*i),
                                     name="tensor_ub_tanh_" + symbol)
        res["tensor_ub_input"] = tensor_ub_input
        operation["tensor_ub_input"] = "vector_conv"
        res["tensor_ub_abs"] = tensor_ub_abs
        operation["tensor_ub_abs"] = "vector_abs"
        res["tensor_ub_abs_add"] = tensor_ub_abs_add
        operation["tensor_ub_abs_add"] = "vector_add"
        res["tensor_ub_tmpdiv"] = tensor_ub_tmpdiv
        operation["tensor_ub_tmpdiv"] = "vector_div"
        res["tensor_ub_mul"] = tensor_ub_mul
        operation["tensor_ub_mul"] = "vector_mul"
        res["tensor_ub_abs_mul"] = tensor_ub_abs_mul
        operation["tensor_ub_abs_mul"] = "vector_mul"
        res["tensor_ub_exp"] = tensor_ub_exp
        operation["tensor_ub_exp"] = "vector_exp"
        res["tensor_ub_sub"] = tensor_ub_sub
        operation["tensor_ub_sub"] = "vector_add"
        res["tensor_ub_add"] = tensor_ub_add
        operation["tensor_ub_add"] = "vector_add"
        res["tensor_ub_div"] = tensor_ub_div
        operation["tensor_ub_div"] = "vector_div"
        res["tensor_ub_tanh"] = tensor_ub_tanh
        operation["tensor_ub_tanh"] = "vector_mul"
    else:
        tensor_ub_two = tvm.compute(shape, lambda *i: tensor(*i) * const_num_two,
                                    name="tensor_ub_two_" + symbol)
        tensor_ub_exp_fp16 = tensor_ub_two
        if dtype_c == "float32":
            tensor_ub_exp_fp16 = tvm.compute(shape,
                                             lambda *i: topi.cast(tensor_ub_two(*i), "float16"),
                                             name="tensor_ub_exp_fp16_" + symbol)
        tensor_ub_exp = tvm.compute(shape,
                                    lambda *i: tvm.exp(tensor_ub_exp_fp16(*i)),
                                    name="tensor_ub_exp_" + symbol)
        tensor_ub_exp_fp32 = tvm.compute(shape, lambda *i: topi.cast(tensor_ub_exp(*i), "float32"),
                                         name="tensor_ub_exp_fp32_" + symbol)
        tensor_ub_add = tvm.compute(shape, lambda *i: tensor_ub_exp_fp32(*i) + const_num_one,
                                    name="tensor_ub_add_" + symbol)
        tensor_ub_sub = tvm.compute(shape, lambda *i: tensor_ub_exp_fp32(*i) + const_num_neg_one,
                                    name="tensor_ub_sub_" + symbol)
        tensor_ub_rec = tvm.compute(shape,
                                    lambda *i: const_num_one / tensor_ub_add(*i),
                                    name="tensor_ub_rec_" + symbol)
        tensor_newton_mul0, tensor_newton_add, \
        tensor_newton_mul1, tensor_newton_mul2 = \
            newton_iteration(shape, tensor_ub_rec, tensor_ub_add, symbol)
        tensor_ub_tanh = tvm.compute(shape, lambda *i: tensor_ub_sub(*i) * tensor_newton_mul2(*i),
                                     name="tensor_ub_tanh_" + symbol)
        res["tensor_ub_input"] = tensor_ub_two
        operation["tensor_ub_input"] = "vector_mul"
        res["tensor_ub_exp_fp16"] = tensor_ub_exp_fp16
        operation["tensor_ub_exp_fp16"] = "vector_conv"
        res["tensor_ub_exp"] = tensor_ub_exp
        operation["tensor_ub_exp"] = "vector_exp"
        res["tensor_ub_exp_fp32"] = tensor_ub_exp_fp32
        operation["tensor_ub_exp_fp32"] = "vector_conv"
        res["tensor_ub_add"] = tensor_ub_add
        operation["tensor_ub_add"] = "vector_add"
        res["tensor_ub_sub"] = tensor_ub_sub
        operation["tensor_ub_sub"] = "vector_add"
        res["tensor_ub_rec"] = tensor_ub_rec
        operation["tensor_ub_rec"] = "vector_rec"
        res["tensor_ub_tanh"] = tensor_ub_tanh
        operation["tensor_ub_tanh"] = "vector_mul"
        res["tensor_newton_mul0"] = tensor_newton_mul0
        operation["tensor_newton_mul0"] = "vector_mul"
        res["tensor_newton_add"] = tensor_newton_add
        operation["tensor_newton_add"] = "vector_add"
        res["tensor_newton_mul1"] = tensor_newton_mul1
        operation["tensor_newton_mul1"] = "vector_mul"
        res["tensor_newton_mul2"] = tensor_newton_mul2
        operation["tensor_newton_mul2"] = "vector_mul"
    return res, operation


# pylint: disable=locally-disabled,too-many-boolean-expressions,invalid-name,too-many-arguments
def basiclstm_cell_check(x, h, c, w, b, ct, ht, it, ft, jt, ot, tanhct, mask, keep_prob):
    """
    the main function of check basic_lstm_cell
    Parameters
    ----------
    x : matmul left x
    h : matmul left h
    c : lstm cell state last time
    w : matmul right w
    b : matmul bias
    ct : lstm cell state this time
    ht : lstm cell output
    it : input gate
    jt : new gate
    ft : forget gate
    ot : output gate
    tanhct : tanh(ot)
    Returns
    -------
    """
    # Check x, h, w, ht dtype
    if x["dtype"] != "float16" or h["dtype"] != "float16" or \
            w["dtype"] != "float16" or ht["dtype"] != "float16":
        raise RuntimeError("x, h, w, ht supports x with dtype float16 only!")
    # Check c, b, ct, it, ft, jt, ot, tanhct dtype
    if c["dtype"] != b["dtype"] or b["dtype"] != ct["dtype"] or \
            ct["dtype"] != it["dtype"] or it["dtype"] != ft["dtype"] or \
            ft["dtype"] != jt["dtype"] or jt["dtype"] != ot["dtype"] or \
            ot["dtype"] != tanhct["dtype"]:
        raise RuntimeError("c, b, ct, it, ft, jt, ot,"
                           " tanhct dtype not match!")
    if c["dtype"] not in ["float16", "float32"]:
        raise RuntimeError("c, b, ct, it, ft, jt, ot,"
                           " tanhct dtype supports float16 and float32 only!")
    # check mask dtype
    if keep_prob != 1.0 and mask["dtype"] not in ["uint8"]:
        raise RuntimeError("mask dtype supports uint8 only!")


def getVselResult(tensor_mask, tensor_x):
    """
    the main function of x_dropout
    Parameters
    ----------
    tensor_mask: tensor_mask
    tensor_x : tensor_x
    Returns
    -------
    """
    dtype_x = tensor_x.dtype
    op_name = "emit_insn_elewise_multiple_sel"
    mode = 'bit'
    shape_condition = te.lang.cce.util.shape_to_list(tensor_mask.shape)
    shape = shape_condition
    shape[-1] = shape[-1] * 8

    def get_indice(indice):
        """
        get indice
        """
        res_index = []
        for i, value in enumerate(indice):
            if i == len(indice) - 1:
                res_index.append(value // 8)
            else:
                res_index.append(value)
        return res_index

    zero_tensor = tvm.const(0, dtype_x)

    def _compute(*indice):
        res_index = get_indice(indice)
        return tvm.select(tensor_mask(*res_index).astype('bool'), tensor_x(*indice), zero_tensor)

    op_name = op_name + '|' + mode
    with tvm.tag_scope(op_name):
        tensor_x_ub_sel = tvm.compute(shape, _compute, name="tensor_x_ub_sel")
    return tensor_x_ub_sel


# Currently not support fusion with dropout
# pylint: disable=locally-disabled,too-many-statements,unused-argument,too-many-arguments,too-many-locals,unnecessary-lambda,invalid-name,too-many-branches,consider-iterating-dictionary
@util.check_input_type(dict, dict, dict, dict, dict, (dict, NONETYPE), dict, dict, dict,
                       dict, dict, dict, dict, float, float, bool,
                       (str, None), str)
def basic_lstm_cell(x, h, c, w, b, mask, ct, ht, it, jt, ft, ot, tanhct,
                    keep_prob=1.0, forget_bias=1.0, state_is_tuple=True,
                    activation="tanh", kernel_name="BasicLSTMCell"):
    """
    the main function of the basic_lstm_cell
    Parameters
    ----------
    x : matmul left x
    h : matmul left h
    c : lstm cell state last time
    w : matmul right w
    b : matmul bias
    ct : lstm cell state this time
    ht : lstm cell output
    it : input gate
    jt : new gate
    ft : forget gate
    ot : output gate
    tanhct : tanh(ot)
    keep_prob : dropout Percentage
    forgat_bias : bias of forget_gate, default: 1.0
    state_is_tuple : x and h is tuple, default: true
    activation : activation function, default: tanh
    kernel_name : kernal_name, default: BasicLSTMCell
    Returns
    -------
    """


    # Perform parameter check
    x_shape = x.get("shape")
    h_shape = h.get("shape")
    c_shape = c.get("shape")
    w_shape = w.get("shape")
    b_shape = b.get("shape")
    util.check_tensor_shape_size(x_shape)
    util.check_tensor_shape_size(h_shape)
    util.check_tensor_shape_size(c_shape)
    util.check_tensor_shape_size(w_shape)
    util.check_tensor_shape_size(b_shape)

    basiclstm_cell_check(x, h, c, w, b, ct, ht, it, ft, jt, ot, tanhct, mask, keep_prob)

    is_cloud = cce_conf.api_check_support("te.lang.cce.vexp", "float32")

    shape_x = get_nz_dim(x.get("shape"))
    shape_h = get_nz_dim(h.get("shape"))
    dtype_x = x.get("dtype").lower()
    dtype_c = c.get("dtype").lower()

    input_dim, batch_dim = shape_x[0:2]
    hidden_dim = shape_h[0]

    shape_bias = b["shape"]
    shape_bias = (shape_bias[0]/16, 16)
    matmul_left_shape = (batch_dim, input_dim + hidden_dim, C0, C0)
    matmul_right_shape = w["shape"]
    matmul_result_shape = h["shape"]
    gate_shape = h["shape"]
    matmul_right_shape_unit = list(matmul_right_shape)
    matmul_right_shape_unit[1] = matmul_right_shape_unit[1] // 4

    # Inputs in gm
    tensor_x = tvm.placeholder(shape_x, name='tensor_x', dtype=dtype_x)
    tensor_h = tvm.placeholder(shape_h, name='tensor_h', dtype=dtype_x)
    tensor_c = tvm.placeholder(gate_shape, name='tensor_c', dtype=dtype_c)
    tensor_w = tvm.placeholder(matmul_right_shape, name='tensor_w', dtype=dtype_x)
    tensor_b = tvm.placeholder(shape_bias, name='tensor_b', dtype=dtype_c)

    if keep_prob == 1.0:
        # Concatenate tensor_x and tensor_h
        tensor_xh_l1 = tvm.compute(matmul_left_shape,
                                   lambda *indice: tvm.select(indice[1] < input_dim,
                                                              tensor_x[indice[1],
                                                                       indice[0],
                                                                       indice[2],
                                                                       indice[3]],
                                                              tensor_h[indice[1] - input_dim,
                                                                       indice[0],
                                                                       indice[2],
                                                                       indice[3]]),
                                   name="tensor_xh_l1", tag="concat")
    else:
        axis0, axis1, axis2, axis3 = shape_x
        shape_mask = (axis0, axis1, axis2, axis3 // 8)
        dtype_mask = mask["dtype"]
        tensor_x_ub = tvm.compute(shape_x, lambda *i: tensor_x(*i), name="tensor_x_ub")
        tensor_mask = tvm.placeholder(shape_mask, name="tensor_mask", dtype=dtype_mask)
        tensor_mask_ub = tvm.compute(shape_mask, lambda *i: tensor_mask(*i), name="tensor_mask_ub")
        scalar_keep_prob = tvm.const(keep_prob, dtype_x)
        tensor_x_ub_prob = tvm.compute(shape_x,
                                       lambda *i: tensor_x_ub(*i) / scalar_keep_prob,
                                       name="tensor_x_ub_prob")
        tensor_x_ub_sel = getVselResult(tensor_mask_ub, tensor_x_ub_prob)
        # Concatenate tensor_x and tensor_h
        tensor_xh_l1 = tvm.compute(matmul_left_shape,
                                   lambda *indice: tvm.select(indice[1] < input_dim,
                                                              tensor_x_ub_sel[indice[1],
                                                                              indice[0],
                                                                              indice[2],
                                                                              indice[3]],
                                                              tensor_h[indice[1] - input_dim,
                                                                       indice[0],
                                                                       indice[2],
                                                                       indice[3]]),
                                   name="tensor_xh_l1", tag="concat")

    # Tensor w and xh from GM to L1, L0A
    tensor_xh_loa = tvm.compute(matmul_left_shape,
                                lambda *i: tensor_xh_l1(*i),
                                name='tensor_xh_l0a')

    #  load 4 W to l1
    def _index_it(*index):
        return index[0], index[1], index[2], index[3]

    def _index_jt(*index):
        return index[0], index[1] + hidden_dim, index[2], index[3]

    def _index_ft(*index):
        return index[0], index[1] + hidden_dim * 2, index[2], index[3]

    def _index_ot(*index):
        return index[0], index[1] + hidden_dim * 3, index[2], index[3]

    tensor_w_l1_it = tvm.compute(matmul_right_shape, lambda *i: tensor_w(*_index_it(*i)),
                                 name='tensor_w_l1_it')
    tensor_w_l1_jt = tvm.compute(matmul_right_shape, lambda *i: tensor_w(*_index_jt(*i)),
                                 name='tensor_w_l1_jt')
    tensor_w_l1_ft = tvm.compute(matmul_right_shape, lambda *i: tensor_w(*_index_ft(*i)),
                                 name='tensor_w_l1_ft')
    tensor_w_l1_ot = tvm.compute(matmul_right_shape, lambda *i: tensor_w(*_index_ot(*i)),
                                 name='tensor_w_l1_ot')

    tensor_w_l0b_it = tvm.compute(matmul_right_shape_unit, lambda *i: tensor_w_l1_it(*i),
                                  name='tensor_w_l0b_it')
    tensor_w_l0b_jt = tvm.compute(matmul_right_shape_unit, lambda *i: tensor_w_l1_jt(*i),
                                  name='tensor_w_l0b_jt')
    tensor_w_l0b_ft = tvm.compute(matmul_right_shape_unit, lambda *i: tensor_w_l1_ft(*i),
                                  name='tensor_w_l0b_ft')
    tensor_w_l0b_ot = tvm.compute(matmul_right_shape_unit, lambda *i: tensor_w_l1_ot(*i),
                                  name='tensor_w_l0b_ot')

    # Copy b from gm to ub
    tensor_b_ub = tvm.compute(shape_bias, lambda *i: tensor_b(*i), name='tensor_b_ub')
    # Determine if cast is needed
    tensor_b_ub_true = tensor_b_ub
    if dtype_c == "float16":
        tensor_b_ub_true = tvm.compute(shape_bias, lambda *i: topi.cast(tensor_b_ub(*i), "float32"),
                                       name="tensor_b_ub_true")

    # Copy bias to L0C
    tensor_b_loc_it = tvm.compute(matmul_result_shape,
                                  lambda i0, i1, i2, i3: tensor_b_ub_true[i0, i3],
                                  name='tensor_b_loc_it')
    tensor_b_loc_jt = tvm.compute(matmul_result_shape,
                                  lambda i0, i1, i2, i3: tensor_b_ub_true[i0 + hidden_dim, i3],
                                  name='tensor_b_loc_jt')
    tensor_b_loc_ft = tvm.compute(matmul_result_shape,
                                  lambda i0, i1, i2, i3: tensor_b_ub_true[i0 + hidden_dim * 2, i3],
                                  name='tensor_b_loc_ft')
    tensor_b_loc_ot = tvm.compute(matmul_result_shape,
                                  lambda i0, i1, i2, i3: tensor_b_ub_true[i0 + hidden_dim * 3, i3],
                                  name='tensor_b_loc_ot')

    # Do matmul(xh * w_it)
    reduce_kb_it = tvm.reduce_axis((0, input_dim + hidden_dim), name='reduce_kb_it')
    reduce_kp_it = tvm.reduce_axis((0, C0), name='reduce_kp_it')
    tensor_matmul_l0c_it = tvm.compute(
        matmul_result_shape, lambda nb, mb, mp, np: tvm.sum(
            (tensor_xh_loa[mb, reduce_kb_it, mp, reduce_kp_it] *
             tensor_w_l0b_it[reduce_kb_it, nb, np, reduce_kp_it]).astype(
                 "float32"), axis=[reduce_kb_it, reduce_kp_it]),
        name='tensor_matmul_l0c_it', attrs={'input_order': 'positive'})
    # Matmul accumulation it + b_it
    tensor_matmul_result_l0c_it = tvm.compute(matmul_result_shape,
                                              lambda *i: tensor_b_loc_it(*i) +
                                              tensor_matmul_l0c_it(*i),
                                              name="tensor_matmul_result_l0c_it")

    # Do matmul(xh * w_jt)
    reduce_kb_jt = tvm.reduce_axis((0, input_dim + hidden_dim), name='reduce_kb_jt')
    reduce_kp_jt = tvm.reduce_axis((0, C0), name='reduce_kp_jt')
    tensor_matmul_l0c_jt = tvm.compute(
        matmul_result_shape, lambda nb, mb, mp, np: tvm.sum(
            (tensor_xh_loa[mb, reduce_kb_jt, mp, reduce_kp_jt] *
             tensor_w_l0b_jt[reduce_kb_jt, nb, np, reduce_kp_jt]).astype(
                 "float32"), axis=[reduce_kb_jt, reduce_kp_jt]),
        name='tensor_matmul_l0c_jt', attrs={'input_order': 'positive'})
    # Matmul accumulation jt + b_jt
    tensor_matmul_result_l0c_jt = tvm.compute(matmul_result_shape,
                                              lambda *i: tensor_b_loc_jt(*i) +
                                              tensor_matmul_l0c_jt(*i),
                                              name="tensor_matmul_result_l0c_jt")

    # Do matmul(xh * w_ft)
    reduce_kb_ft = tvm.reduce_axis((0, input_dim + hidden_dim), name='reduce_kb_ft')
    reduce_kp_ft = tvm.reduce_axis((0, C0), name='reduce_kp_ft')
    tensor_matmul_l0c_ft = tvm.compute(
        matmul_result_shape, lambda nb, mb, mp, np: tvm.sum(
            (tensor_xh_loa[mb, reduce_kb_ft, mp, reduce_kp_ft] *
             tensor_w_l0b_ft[reduce_kb_ft, nb, np, reduce_kp_ft]).astype(
                 "float32"), axis=[reduce_kb_ft, reduce_kp_ft]),
        name='tensor_matmul_l0c_ft', attrs={'input_order': 'positive'})
    # Matmul accumulation ft + b_ft
    tensor_matmul_result_l0c_ft = tvm.compute(matmul_result_shape,
                                              lambda *i: tensor_b_loc_ft(*i) +
                                              tensor_matmul_l0c_ft(*i),
                                              name="tensor_matmul_result_l0c_ft")

    # Do matmul(xh * w_ot)
    reduce_kb_ot = tvm.reduce_axis((0, input_dim + hidden_dim), name='reduce_kb_ot')
    reduce_kp_ot = tvm.reduce_axis((0, C0), name='reduce_kp_ot')
    tensor_matmul_l0c_ot = tvm.compute(
        matmul_result_shape, lambda nb, mb, mp, np: tvm.sum(
            (tensor_xh_loa[mb, reduce_kb_ot, mp, reduce_kp_ot] *
             tensor_w_l0b_ot[reduce_kb_ot, nb, np, reduce_kp_ot]).astype(
                 "float32"), axis=[reduce_kb_ot, reduce_kp_ot]),
        name='tensor_matmul_l0c_ot', attrs={'input_order': 'positive'})
    # Matmul accumulation ot + b_ot
    tensor_matmul_result_l0c_ot = tvm.compute(matmul_result_shape,
                                              lambda *i: tensor_b_loc_ot(*i) +
                                              tensor_matmul_l0c_ot(*i),
                                              name="tensor_matmul_result_l0c_ot")

    # Move it, jt, ft, ot to UB
    it_ub_temp = tvm.compute(gate_shape,
                             lambda *i: tensor_matmul_result_l0c_it(*i), name='it_ub_temp')
    jt_ub_temp = tvm.compute(gate_shape,
                             lambda *i: tensor_matmul_result_l0c_jt(*i), name='jt_ub_temp')
    ft_ub_temp = tvm.compute(gate_shape,
                             lambda *i: tensor_matmul_result_l0c_ft(*i), name='ft_ub_temp')
    ot_ub_temp = tvm.compute(gate_shape,
                             lambda *i: tensor_matmul_result_l0c_ot(*i), name='ot_ub_temp')

    temp_type = ft_ub_temp.dtype
    const_num_one = tvm.const(forget_bias, dtype=temp_type)
    ft_ub_temp_fbias = tvm.compute(gate_shape,
                                   lambda *i: ft_ub_temp(*i) + const_num_one,
                                   name="ft_ub_temp_fbias")

    tensor_one = tvm.compute(gate_shape, lambda *i: const_num_one, name='tensor_one')
    # Do sigmoid(It,Ft,Ot) calculation
    tensor_gate_neg_it, tensor_gate_exp_fp16_it, \
    tensor_gate_exp_it, tensor_gate_exp_fp32_it, \
    tensor_gate_add_it, tensor_gate_sigmoid_it, \
    tensor_newton_mul0_it, tensor_newton_add_it, \
    tensor_newton_mul1_it, tensor_newton_mul2_it = \
        sigmoid(gate_shape, it_ub_temp, tensor_one, "it", is_cloud)

    tensor_gate_neg_ft, tensor_gate_exp_fp16_ft, \
    tensor_gate_exp_ft, tensor_gate_exp_fp32_ft, \
    tensor_gate_add_ft, tensor_gate_sigmoid_ft, \
    tensor_newton_mul0_ft, tensor_newton_add_ft, \
    tensor_newton_mul1_ft, tensor_newton_mul2_ft = \
        sigmoid(gate_shape, ft_ub_temp_fbias, tensor_one, "ft", is_cloud)

    tensor_gate_neg_ot, tensor_gate_exp_fp16_ot, \
    tensor_gate_exp_ot, tensor_gate_exp_fp32_ot, \
    tensor_gate_add_ot, tensor_gate_sigmoid_ot, \
    tensor_newton_mul0_ot, tensor_newton_add_ot, \
    tensor_newton_mul1_ot, tensor_newton_mul2_ot = \
        sigmoid(gate_shape, ot_ub_temp, tensor_one, "ot", is_cloud)

    # calc tanh(jt)
    tanh_jt_list, tanh_jt_op_list = tanh(gate_shape, jt_ub_temp, "jt", is_cloud)

    # move i j f o To GM
    it_ub = tensor_newton_mul2_it
    jt_ub = tanh_jt_list["tensor_ub_tanh"]
    ft_ub = tensor_newton_mul2_ft
    ot_ub = tensor_newton_mul2_ot
    if dtype_c == "float16":
        it_ub = tvm.compute(gate_shape,
                            lambda *i: topi.cast(it_ub(*i), dtype_c),
                            name="it_ub_fp16")
        jt_ub = tvm.compute(gate_shape,
                            lambda *i: topi.cast(jt_ub(*i), dtype_c),
                            name="jt_ub_fp16")
        ft_ub = tvm.compute(gate_shape,
                            lambda *i: topi.cast(ft_ub(*i), dtype_c),
                            name="ft_ub_fp16")
        ot_ub = tvm.compute(gate_shape,
                            lambda *i: topi.cast(ot_ub(*i), dtype_c),
                            name="ot_ub_fp16")
    # Move it, jt, ft, ot to GM
    it = tvm.compute(gate_shape, lambda *i: it_ub(*i), name='it')
    jt = tvm.compute(gate_shape, lambda *i: jt_ub(*i), name='jt')
    ft = tvm.compute(gate_shape, lambda *i: ft_ub(*i), name='ft')
    ot = tvm.compute(gate_shape, lambda *i: ot_ub(*i), name='ot')

    # Move it, jt, ft, ot back(Fake)
    it_ub_fake = tvm.compute(gate_shape, lambda *i: it(*i), name='it_ub_fake')
    jt_ub_fake = tvm.compute(gate_shape, lambda *i: jt(*i), name='jt_ub_fake')
    ft_ub_fake = tvm.compute(gate_shape, lambda *i: ft(*i), name='ft_ub_fake')
    ot_ub_fake = tvm.compute(gate_shape, lambda *i: ot(*i), name='ot_ub_fake')

    # Move c to ub
    tensor_c_ub = tvm.compute(gate_shape, lambda *i: tensor_c(*i), name='tensor_c_ub')
    tensor_cf_ub = tvm.compute(gate_shape,
                               lambda *i: tensor_c_ub(*i) * ft_ub_fake(*i),
                               name="tensor_cf_ub")
    tensor_ji_ub = tvm.compute(gate_shape,
                               lambda *i: jt_ub_fake(*i) * it_ub_fake(*i),
                               name="tensor_ji_ub")
    tensor_ct_ub = tvm.compute(gate_shape,
                               lambda *i: tensor_cf_ub(*i) + tensor_ji_ub(*i),
                               name="tensor_ct_ub")
    # Move ct to gm
    ct = tvm.compute(gate_shape, lambda *i: tensor_ct_ub(*i), name="ct")
    # Move ct back(Fake)
    tensor_ct_ub_fake = tvm.compute(gate_shape, lambda *i: ct(*i), name="ct_ub_fake")
    # calc tanh(ct)
    tanh_ct_list, tanh_ct_op_list = tanh(gate_shape, tensor_ct_ub_fake, "ct", is_cloud)

    tensor_ub_tanh_ct_true = tanh_ct_list["tensor_ub_tanh"]
    if dtype_c == "float16":
        tensor_ub_tanh_ct_true = tvm.compute(gate_shape,
                                             lambda *i:
                                             topi.cast(tanh_ct_list["tensor_ub_tanh"](*i),
                                                       "float16"),
                                             name="tensor_ub_tanh_ct_true")
    # Move tanhct to gm
    tanhct = tvm.compute(gate_shape, lambda *i: tensor_ub_tanh_ct_true(*i), name="tanhct")
    # Move tanhct back(Fake)
    tensor_ub_tanh_ct_fake = tvm.compute(gate_shape, lambda *i: tanhct(*i), name="tanhct_ub_fake")

    tensor_ht_ub = tvm.compute(gate_shape,
                               lambda *i: tensor_ub_tanh_ct_fake(*i) * ot_ub_fake(*i),
                               name="tensor_ht_ub")
    tensor_ht_ub_last = tensor_ht_ub
    if dtype_c == "float32":
        tensor_ht_ub_last = tvm.compute(gate_shape,
                                        lambda *i: topi.cast(tensor_ht_ub(*i), "float16"),
                                        name="tensor_ht_ub_last")
    # Move ht to gm
    ht = tvm.compute(gate_shape, lambda *i: tensor_ht_ub_last(*i), name="ht")

    ###################################################
    # Create initial schedule
    ###################################################
    schedule_list = [ht.op]
    s = tvm.create_schedule(schedule_list)
    ###################################################
    # Data flow control
    ###################################################
    # Move xh, w to L1
    s[tensor_xh_l1].set_scope(cce.scope_cbuf)
    s[tensor_w_l1_it].set_scope(cce.scope_cbuf)
    s[tensor_w_l1_jt].set_scope(cce.scope_cbuf)
    s[tensor_w_l1_ft].set_scope(cce.scope_cbuf)
    s[tensor_w_l1_ot].set_scope(cce.scope_cbuf)
    # Move xh from L1 to L0A
    s[tensor_xh_loa].set_scope(cce.scope_ca)
    # Move w from L1 to L0B
    s[tensor_w_l0b_it].set_scope(cce.scope_cb)

    s[tensor_w_l0b_jt].set_scope(cce.scope_cb)
    s[tensor_w_l0b_ft].set_scope(cce.scope_cb)
    s[tensor_w_l0b_ot].set_scope(cce.scope_cb)
    # Move b to UB
    s[tensor_b_ub].set_scope(cce.scope_ubuf)
    s[tensor_b_ub_true].set_scope(cce.scope_ubuf)
    # Move b from UB to L0C
    s[tensor_b_loc_it].set_scope(cce.scope_cc)
    s[tensor_b_loc_jt].set_scope(cce.scope_cc)
    s[tensor_b_loc_ft].set_scope(cce.scope_cc)
    s[tensor_b_loc_ot].set_scope(cce.scope_cc)
    # Set Matmul operation in L0A+L0B=L0C
    s[tensor_matmul_l0c_it].set_scope(cce.scope_cc)
    s[tensor_matmul_l0c_jt].set_scope(cce.scope_cc)
    s[tensor_matmul_l0c_ft].set_scope(cce.scope_cc)
    s[tensor_matmul_l0c_ot].set_scope(cce.scope_cc)
    s[tensor_matmul_result_l0c_it].set_scope(cce.scope_cc)
    s[tensor_matmul_result_l0c_jt].set_scope(cce.scope_cc)
    s[tensor_matmul_result_l0c_ft].set_scope(cce.scope_cc)
    s[tensor_matmul_result_l0c_ot].set_scope(cce.scope_cc)
    # Move Matmul and accumulation result to UB
    s[it_ub_temp].set_scope(cce.scope_ubuf)
    s[jt_ub_temp].set_scope(cce.scope_ubuf)
    s[ft_ub_temp].set_scope(cce.scope_ubuf)
    s[ot_ub_temp].set_scope(cce.scope_ubuf)
    s[it_ub].set_scope(cce.scope_ubuf)
    s[jt_ub].set_scope(cce.scope_ubuf)
    s[ft_ub].set_scope(cce.scope_ubuf)
    s[ot_ub].set_scope(cce.scope_ubuf)
    # Move result to GM
    it_gm = s.cache_write(it, cce.scope_ubuf)
    ot_gm = s.cache_write(ot, cce.scope_ubuf)
    ft_gm = s.cache_write(ft, cce.scope_ubuf)
    jt_gm = s.cache_write(jt, cce.scope_ubuf)
    # Move result back (Fake)
    it_ub_fake_ub = s.cache_read(it_ub_fake, cce.scope_ubuf, [tensor_ji_ub])
    s[it_ub_fake].compute_inline()
    jt_ub_fake_ub = s.cache_read(jt_ub_fake, cce.scope_ubuf, [tensor_ji_ub])
    s[jt_ub_fake].compute_inline()
    ft_ub_fake_ub = s.cache_read(ft_ub_fake, cce.scope_ubuf, [tensor_cf_ub])
    s[ft_ub_fake].compute_inline()
    ot_ub_fake_ub = s.cache_read(ot_ub_fake, cce.scope_ubuf, [tensor_ht_ub])
    s[ot_ub_fake].compute_inline()
    # Set sigmoid(it, ft, ot) operation on UB
    s[ft_ub_temp_fbias].set_scope(cce.scope_ubuf)
    all_sigmoid_operations = (tensor_gate_neg_it, tensor_gate_exp_fp16_it,
                              tensor_gate_exp_it, tensor_gate_exp_fp32_it,
                              tensor_gate_add_it, tensor_gate_sigmoid_it,
                              tensor_gate_neg_ft, tensor_gate_exp_fp16_ft,
                              tensor_gate_exp_ft, tensor_gate_exp_fp32_ft,
                              tensor_gate_add_ft, tensor_gate_sigmoid_ft,
                              tensor_gate_neg_ot, tensor_gate_exp_fp16_ot,
                              tensor_gate_exp_ot, tensor_gate_exp_fp32_ot,
                              tensor_gate_add_ot, tensor_gate_sigmoid_ot)
    for operation in all_sigmoid_operations:
        s[operation].set_scope(cce.scope_ubuf)

    # Set tanh(jt) operation on UB
    for t in tanh_ct_list.keys():
        s[tanh_ct_list[t]].set_scope(cce.scope_ubuf)
    for t in tanh_jt_list.keys():
        s[tanh_jt_list[t]].set_scope(cce.scope_ubuf)
    # Move c to UB
    s[tensor_c_ub].set_scope(cce.scope_ubuf)
    # Set c * sigmoid(ft) operation on UB
    s[tensor_cf_ub].set_scope(cce.scope_ubuf)
    # Set tanh(jt) * sigmoid(it) operation on UB
    s[tensor_ji_ub].set_scope(cce.scope_ubuf)
    # Set ct operation on UB
    s[tensor_ct_ub].set_scope(cce.scope_ubuf)
    # Move ct to GM
    ct_gm = s.cache_write(ct, cce.scope_ubuf)
    # Move result back (Fake)
    if is_cloud:
        if tensor_ct_ub_fake.dtype == "float16":
            tensor_ct_ub_fake_ub = s.cache_read(tensor_ct_ub_fake, cce.scope_ubuf,
                                                tanh_ct_list["tensor_ub_input"])
        else:
            tensor_ct_ub_fake_ub = s.cache_read(tensor_ct_ub_fake, cce.scope_ubuf,
                                                [tanh_ct_list["tensor_ub_abs"],
                                                 tanh_ct_list["tensor_ub_tmpdiv"]])
    else:
        tensor_ct_ub_fake_ub = s.cache_read(tensor_ct_ub_fake,
                                            cce.scope_ubuf, tanh_ct_list["tensor_ub_input"])

    # Set tanh(ct) operation on UB
    s[tensor_ub_tanh_ct_true].set_scope(cce.scope_ubuf)
    # Move tanhct to GM
    tanhct_gm = s.cache_write(tanhct, cce.scope_ubuf)
    # Move result back (Fake)
    tensor_ub_tanh_ct_fake_ub = s.cache_read(tensor_ub_tanh_ct_fake, cce.scope_ubuf, [tensor_ht_ub])
    s[tensor_ub_tanh_ct_fake].compute_inline()
    # Set tanh(ct) * sigmoid(ot) on UB
    s[tensor_ht_ub].set_scope(cce.scope_ubuf)
    s[tensor_ht_ub_last].set_scope(cce.scope_ubuf)

    # Split core bind axis
    dtype_c_size = get_dtype_size(dtype_c)
    dtype_x_size = get_dtype_size(dtype_x)
    block_num = cce_conf.get_soc_spec(cce_conf.CORE_NUM)
    l0_size = cce_conf.get_soc_spec(cce_conf.L0A_SIZE)
    ub_limit = 60 * 1024
    # block_tilling
    block_batch_npart = batch_dim
    if batch_dim > block_num:
        block_batch_npart = block_num
    block_hidden_npart = block_num // block_batch_npart
    block_batch_factor = batch_dim // block_batch_npart
    block_hidden_factor = max(1, hidden_dim // block_hidden_npart)
    # ub tilling
    k_axis = hidden_dim + input_dim
    one_mn_size = k_axis * C0 * C0 * dtype_x_size
    batch_factor = min(int(l0_size / one_mn_size), block_batch_factor)
    hidden_factor = min(int(l0_size / one_mn_size), block_hidden_factor)

    ub_used_size = batch_factor * hidden_factor * C0 * C0 * dtype_c_size
    ub_used_one_hidden_size = batch_factor * C0 * C0 * dtype_c_size
    if ub_used_one_hidden_size > ub_limit:
        hidden_factor = 1
        batch_factor = batch_factor - 1
        ub_used_one_hidden_size = batch_factor * C0 * C0 * dtype_c_size
        while ub_used_one_hidden_size > ub_limit:
            batch_factor = batch_factor - 1
            ub_used_one_hidden_size = batch_factor * C0 * C0 * dtype_c_size
    elif ub_used_size > ub_limit:
        hidden_factor = hidden_factor - 1
        ub_used_size = batch_factor * hidden_factor * C0 * C0 * dtype_c_size
        while ub_used_size > ub_limit:
            hidden_factor = hidden_factor - 1
            ub_used_size = batch_factor * hidden_factor * C0 * C0 * dtype_c_size

    axis_1_o, axis_1_i = s[ht].split(ht.op.axis[1], nparts=block_batch_npart)
    axis_1_i_0, axis_1_i_i = s[ht].split(axis_1_i, factor=batch_factor)
    axis_0_o, axis_0_i = s[ht].split(ht.op.axis[0], factor=block_hidden_factor)
    axis_0_i_o, axis_0_i_i = s[ht].split(axis_0_i, factor=hidden_factor)
    s[ht].reorder(axis_1_o, axis_0_o, axis_1_i_0, axis_0_i_o, axis_1_i_i, axis_0_i_i)
    fused_axis_block = s[ht].fuse(axis_1_o, axis_0_o)
    fused_xh_at = s[ht].fuse(axis_1_i_0, axis_0_i_o)
    fused_axis_at = s[ht].fuse(axis_1_i_i, axis_0_i_i)
    s[ht].bind(fused_axis_block, tvm.thread_axis("blockIdx.x"))
    s[tensor_xh_l1].compute_at(s[ht], fused_xh_at)

    ##################################################
    # Compute stage scope
    ###################################################
    compute_at_axis = fused_axis_at
    # Merge matmul computation related operations
    s[tensor_w_l0b_it].compute_at(s[ht], compute_at_axis)
    s[tensor_w_l0b_jt].compute_at(s[ht], compute_at_axis)
    s[tensor_w_l0b_ft].compute_at(s[ht], compute_at_axis)
    s[tensor_w_l0b_ot].compute_at(s[ht], compute_at_axis)
    s[tensor_b_ub].compute_at(s[ht], compute_at_axis)
    s[tensor_b_loc_it].compute_at(s[ht], compute_at_axis)
    s[tensor_b_loc_jt].compute_at(s[ht], compute_at_axis)
    s[tensor_b_loc_ft].compute_at(s[ht], compute_at_axis)
    s[tensor_b_loc_ot].compute_at(s[ht], compute_at_axis)
    s[tensor_matmul_l0c_it].compute_at(s[ht], compute_at_axis)
    s[tensor_matmul_l0c_jt].compute_at(s[ht], compute_at_axis)
    s[tensor_matmul_l0c_ft].compute_at(s[ht], compute_at_axis)
    s[tensor_matmul_l0c_ot].compute_at(s[ht], compute_at_axis)
    # Merge matmul computation related operations with multiple consumers
    s[tensor_xh_loa].compute_at(s[ht], compute_at_axis)
    s[tensor_w_l1_it].compute_at(s[ht], compute_at_axis)
    s[tensor_w_l1_jt].compute_at(s[ht], compute_at_axis)
    s[tensor_w_l1_ft].compute_at(s[ht], compute_at_axis)
    s[tensor_w_l1_ot].compute_at(s[ht], compute_at_axis)
    s[tensor_b_ub_true].compute_at(s[ht], compute_at_axis)
    # Merge matmul result moving related operations
    s[tensor_matmul_result_l0c_it].compute_at(s[ht], compute_at_axis)
    s[tensor_matmul_result_l0c_jt].compute_at(s[ht], compute_at_axis)
    s[tensor_matmul_result_l0c_ft].compute_at(s[ht], compute_at_axis)
    s[tensor_matmul_result_l0c_ot].compute_at(s[ht], compute_at_axis)
    if it_ub is not it_ub_temp:
        s[it_ub_temp].compute_at(s[ht], compute_at_axis)
        s[jt_ub_temp].compute_at(s[ht], compute_at_axis)
        s[ft_ub_temp].compute_at(s[ht], compute_at_axis)
        s[ot_ub_temp].compute_at(s[ht], compute_at_axis)
    s[it_ub].compute_at(s[ht], compute_at_axis)
    s[jt_ub].compute_at(s[ht], compute_at_axis)
    s[ft_ub].compute_at(s[ht], compute_at_axis)
    s[ot_ub].compute_at(s[ht], compute_at_axis)
    s[it_gm].compute_at(s[ht], compute_at_axis)
    s[jt_gm].compute_at(s[ht], compute_at_axis)
    s[ft_gm].compute_at(s[ht], compute_at_axis)
    s[ot_gm].compute_at(s[ht], compute_at_axis)
    s[it].compute_at(s[ht], compute_at_axis)
    s[jt].compute_at(s[ht], compute_at_axis)
    s[ft].compute_at(s[ht], compute_at_axis)
    s[ot].compute_at(s[ht], compute_at_axis)
    # Merge matmul moved result with sigmoid operations
    s[it_ub_fake_ub].compute_at(s[ht], compute_at_axis)
    s[ft_ub_fake_ub].compute_at(s[ht], compute_at_axis)
    s[ft_ub_temp_fbias].compute_at(s[ht], compute_at_axis)
    s[ot_ub_fake_ub].compute_at(s[ht], compute_at_axis)

    # Merge sigmoid(it, ft, ot)operations
    def sigmoid_compute_at(sigma_group, sch):
        last_operation = None
        for sig_operation in sigma_group:
            if last_operation is None:
                last_operation = sig_operation
                continue
            sch[last_operation].compute_at(s[ht], compute_at_axis)
            last_operation = sig_operation
    sigmoid_group = [tensor_gate_neg_it, tensor_gate_exp_fp16_it, tensor_gate_exp_it,
                     tensor_gate_exp_fp32_it, tensor_gate_add_it, tensor_gate_sigmoid_it]
    sigmoid_compute_at(sigmoid_group, s)
    sigmoid_group = [tensor_gate_neg_ft, tensor_gate_exp_fp16_ft, tensor_gate_exp_ft,
                     tensor_gate_exp_fp32_ft, tensor_gate_add_ft, tensor_gate_sigmoid_ft]
    sigmoid_compute_at(sigmoid_group, s)
    sigmoid_group = [tensor_gate_neg_ot, tensor_gate_exp_fp16_ot, tensor_gate_exp_ot,
                     tensor_gate_exp_fp32_ot, tensor_gate_add_ot, tensor_gate_sigmoid_ot]
    sigmoid_compute_at(sigmoid_group, s)
    # Merge tanh(jt) operations
    s[jt_ub_fake_ub].compute_at(s[ht], compute_at_axis)
    for t in tanh_ct_list.keys():
        s[tanh_ct_list[t]].compute_at(s[ht], compute_at_axis)
    for t in tanh_jt_list.keys():
        s[tanh_jt_list[t]].compute_at(s[ht], compute_at_axis)
    # Merge c * sigmoid(ft) operations
    s[tensor_c_ub].compute_at(s[ht], compute_at_axis)
    s[tensor_gate_sigmoid_ft].compute_at(s[ht], compute_at_axis)
    s[tensor_gate_sigmoid_it].compute_at(s[ht], compute_at_axis)
    # Merge ct = c * sigmoid(ft) + tanh(jt) * sigmoid(it) operations
    s[tensor_cf_ub].compute_at(s[ht], compute_at_axis)
    s[tensor_ji_ub].compute_at(s[ht], compute_at_axis)
    # Merge ct moving related operations
    s[tensor_ct_ub].compute_at(s[ht], compute_at_axis)
    s[ct_gm].compute_at(s[ht], compute_at_axis)
    s[ct].compute_at(s[ht], compute_at_axis)
    # Merge tanh(ct) operations
    s[tensor_ct_ub_fake_ub].compute_at(s[ht], compute_at_axis)
    s[tensor_ub_tanh_ct_true].compute_at(s[ht], compute_at_axis)
    s[tanhct_gm].compute_at(s[ht], compute_at_axis)
    s[tanhct].compute_at(s[ht], compute_at_axis)
    # Merge ht = tanh(ct) * sigmoid(ot) operations
    s[tensor_ub_tanh_ct_fake_ub].compute_at(s[ht], compute_at_axis)
    s[tensor_gate_sigmoid_ot].compute_at(s[ht], compute_at_axis)
    # Merge ht moving related operations
    s[tensor_ht_ub].compute_at(s[ht], compute_at_axis)
    s[tensor_ht_ub_last].compute_at(s[ht], compute_at_axis)

    ###################################################
    # Emit Insn
    ###################################################
    def emit_on_self(tensor, axisnum=0, op='dma_copy'):
        s[tensor].emit_insn(s[tensor].op.axis[axisnum], op)
    # Emit insn for matmul computation input dma_copy operations
    initial_dma_copy_ops = (tensor_w_l1_it,
                            tensor_w_l1_jt, tensor_w_l1_ft,
                            tensor_w_l1_ot, tensor_xh_loa,
                            tensor_w_l0b_it, tensor_w_l0b_jt,
                            tensor_w_l0b_ft, tensor_w_l0b_ot,
                            tensor_b_ub, tensor_b_loc_it,
                            tensor_b_loc_jt, tensor_b_loc_ft,
                            tensor_b_loc_ot)
    for t in initial_dma_copy_ops:
        emit_on_self(t)
    # Emit insn for matmul operations
    mad_pattern = cce.GEMM_MODE
    mad_dict = {"mad_pattern": mad_pattern,
                "init_bias": 1}
    s[tensor_b_loc_it].pragma(s[tensor_b_loc_it].op.axis[0], 'reuse_output', 0)
    s[tensor_matmul_l0c_it].pragma(s[tensor_matmul_l0c_it].op.axis[0], 'replace_output', 0)
    s[tensor_matmul_result_l0c_it].pragma(s[tensor_matmul_result_l0c_it].op.axis[0],
                                          'replace_output', 0)
    s[tensor_matmul_l0c_it].emit_insn(s[tensor_matmul_l0c_it].op.axis[0], 'mad', mad_dict)
    s[tensor_matmul_result_l0c_it].emit_insn(s[tensor_matmul_result_l0c_it].op.axis[0],
                                             'phony_insn')

    s[tensor_b_loc_jt].pragma(s[tensor_b_loc_jt].op.axis[0], 'reuse_output', 1)
    s[tensor_matmul_l0c_jt].pragma(s[tensor_matmul_l0c_jt].op.axis[0], 'replace_output', 1)
    s[tensor_matmul_result_l0c_jt].pragma(s[tensor_matmul_result_l0c_jt].op.axis[0],
                                          'replace_output', 1)
    s[tensor_matmul_l0c_jt].emit_insn(s[tensor_matmul_l0c_jt].op.axis[0], 'mad', mad_dict)
    s[tensor_matmul_result_l0c_jt].emit_insn(s[tensor_matmul_result_l0c_jt].op.axis[0],
                                             'phony_insn')

    s[tensor_b_loc_ft].pragma(s[tensor_b_loc_ft].op.axis[0], 'reuse_output', 2)
    s[tensor_matmul_l0c_ft].pragma(s[tensor_matmul_l0c_ft].op.axis[0], 'replace_output', 2)
    s[tensor_matmul_result_l0c_ft].pragma(s[tensor_matmul_result_l0c_ft].op.axis[0],
                                          'replace_output', 2)
    s[tensor_matmul_l0c_ft].emit_insn(s[tensor_matmul_l0c_ft].op.axis[0], 'mad', mad_dict)
    s[tensor_matmul_result_l0c_ft].emit_insn(s[tensor_matmul_result_l0c_ft].op.axis[0],
                                             'phony_insn')

    s[tensor_b_loc_ot].pragma(s[tensor_b_loc_ot].op.axis[0], 'reuse_output', 3)
    s[tensor_matmul_l0c_ot].pragma(s[tensor_matmul_l0c_ot].op.axis[0], 'replace_output', 3)
    s[tensor_matmul_result_l0c_ot].pragma(s[tensor_matmul_result_l0c_ot].op.axis[0],
                                          'replace_output', 3)
    s[tensor_matmul_l0c_ot].emit_insn(s[tensor_matmul_l0c_ot].op.axis[0], 'mad', mad_dict)
    s[tensor_matmul_result_l0c_ot].emit_insn(s[tensor_matmul_result_l0c_ot].op.axis[0],
                                             'phony_insn')
    # Matmul dma copy operations
    matmul_dma_copy_ops = (it_ub_temp, jt_ub_temp, ft_ub_temp, ot_ub_temp)
    for t in matmul_dma_copy_ops:
        emit_on_self(t)
    # Matmul bias dma copy operations
    if dtype_c == "float16":
        matmul_dma_copy_ops = (tensor_b_ub_true, it_ub, jt_ub, ft_ub, ot_ub)
        for t in matmul_dma_copy_ops:
            emit_on_self(t, op='vector_conv')
    # Matmul result dma copy operations
    s[it_ub].reused_by(it_ub_fake_ub)
    s[jt_ub].reused_by(jt_ub_fake_ub)
    s[ft_ub].reused_by(ft_ub_fake_ub)
    s[ot_ub].reused_by(ot_ub_fake_ub)
    matmul_res_dma_copy_ops = (it, jt, ft, ot)
    matmul_res_phony_insn = (it_ub_fake_ub, jt_ub_fake_ub, ft_ub_fake_ub, ot_ub_fake_ub)
    for t in matmul_res_dma_copy_ops:
        emit_on_self(t, op='dma_copy')
    for t in matmul_res_phony_insn:
        emit_on_self(t, op='phony_insn')

    if not is_cloud:
        all_newton_operations_mul = (tensor_newton_mul0_it, tensor_newton_mul1_it,
                                     tensor_newton_mul2_it, tensor_newton_mul0_ft,
                                     tensor_newton_mul1_ft, tensor_newton_mul2_ft,
                                     tensor_newton_mul0_ot, tensor_newton_mul1_ot,
                                     tensor_newton_mul2_ot)

        all_newton_operations_add = (tensor_newton_add_it, tensor_newton_add_ft,
                                     tensor_newton_add_ot)

        for operation in all_newton_operations_mul:
            s[operation].set_scope(cce.scope_ubuf)
            emit_on_self(operation, op='vector_mul')
            s[operation].compute_at(s[ht], compute_at_axis)

        for operation in all_newton_operations_add:
            s[operation].set_scope(cce.scope_ubuf)
            emit_on_self(operation, op='vector_add')
            s[operation].compute_at(s[ht], compute_at_axis)
        emit_on_self(tensor_gate_sigmoid_it, op='vector_rec')
        emit_on_self(tensor_gate_sigmoid_ft, op='vector_rec')
        emit_on_self(tensor_gate_sigmoid_ot, op='vector_rec')
        emit_on_self(tensor_gate_exp_fp16_it, op='vector_conv')
        emit_on_self(tensor_gate_exp_fp32_it, op='vector_conv')
        emit_on_self(tensor_gate_exp_fp16_ft, op='vector_conv')
        emit_on_self(tensor_gate_exp_fp32_ft, op='vector_conv')
        emit_on_self(tensor_gate_exp_fp16_ot, op='vector_conv')
        emit_on_self(tensor_gate_exp_fp32_ot, op='vector_conv')
    else:
        s[tensor_one].set_scope(cce.scope_ubuf)
        s[tensor_one].compute_at(s[ht], compute_at_axis)
        emit_on_self(tensor_one, op='vector_dup')
        emit_on_self(tensor_gate_sigmoid_it, op='vector_div')
        emit_on_self(tensor_gate_sigmoid_ft, op='vector_div')
        emit_on_self(tensor_gate_sigmoid_ot, op='vector_div')
    if dtype_c == "float16":
        emit_on_self(tensor_ub_tanh_ct_true, op='vector_conv')

    emit_on_self(tensor_gate_neg_it, op='vector_mul')
    emit_on_self(tensor_gate_exp_it, op='vector_exp')
    emit_on_self(tensor_gate_add_it, op='vector_add')
    emit_on_self(tensor_gate_neg_ft, op='vector_mul')
    emit_on_self(tensor_gate_exp_ft, op='vector_exp')
    emit_on_self(tensor_gate_add_ft, op='vector_add')
    emit_on_self(ft_ub_temp_fbias, op='vector_add')
    emit_on_self(tensor_gate_neg_ot, op='vector_mul')
    emit_on_self(tensor_gate_exp_ot, op='vector_exp')
    emit_on_self(tensor_gate_add_ot, op='vector_add')
    emit_on_self(tensor_c_ub, op='dma_copy')
    emit_on_self(tensor_cf_ub, op='vector_mul')
    emit_on_self(tensor_ji_ub, op='vector_mul')
    emit_on_self(tensor_ct_ub, op='vector_add')
    emit_on_self(ct, op='dma_copy')
    s[tensor_ct_ub].reused_by(tensor_ct_ub_fake_ub)
    emit_on_self(tensor_ct_ub_fake_ub, op='phony_insn')
    emit_on_self(tanhct, op='dma_copy')
    s[tensor_ub_tanh_ct_true].reused_by(tensor_ub_tanh_ct_fake_ub)
    emit_on_self(tensor_ub_tanh_ct_fake_ub, op='phony_insn')
    emit_on_self(tensor_ht_ub, op='vector_mul')

    for t in tanh_ct_list.keys():
        if is_cloud and t == "tensor_ub_input":
            if tanh_ct_list[t].dtype == "float16":
                emit_on_self(tanh_ct_list[t], op=tanh_ct_op_list[t])
            else:
                continue
        elif not is_cloud and t == "tensor_ub_exp_fp16":
            if tensor_ct_ub_fake.dtype == "float32":
                emit_on_self(tanh_ct_list[t], op=tanh_ct_op_list[t])
            else:
                continue
        else:
            emit_on_self(tanh_ct_list[t], op=tanh_ct_op_list[t])
    for t in tanh_jt_list.keys():
        if not is_cloud:
            emit_on_self(tanh_jt_list[t], op=tanh_jt_op_list[t])
        elif t == "tensor_ub_input":
            if jt_ub_temp.dtype == "float16":
                emit_on_self(tanh_jt_list[t], op=tanh_jt_op_list[t])
            else:
                continue
        else:
            emit_on_self(tanh_jt_list[t], op=tanh_jt_op_list[t])


    if dtype_c == "float32":
        emit_on_self(tensor_ht_ub_last, op='vector_conv')

    s[ht].emit_insn(s[ht].op.axis[2], 'dma_copy')
    build_list = [tensor_x, tensor_h, tensor_c, tensor_w, tensor_b, ct, ht, it, jt, ft, ot, tanhct]

    # deal dropout
    if keep_prob != 1.0:
        s[tensor_x_ub].set_scope(cce.scope_ubuf)
        s[tensor_mask_ub].set_scope(cce.scope_ubuf)
        s[tensor_x_ub_prob].set_scope(cce.scope_ubuf)
        s[tensor_x_ub_sel].set_scope(cce.scope_ubuf)
        s[tensor_x_ub].compute_at(s[tensor_xh_l1], tensor_xh_l1.op.axis[0])
        s[tensor_mask_ub].compute_at(s[tensor_xh_l1], tensor_xh_l1.op.axis[0])
        s[tensor_x_ub_prob].compute_at(s[tensor_xh_l1], tensor_xh_l1.op.axis[0])
        s[tensor_x_ub_sel].compute_at(s[tensor_xh_l1], tensor_xh_l1.op.axis[0])
        emit_on_self(tensor_x_ub, op='dma_copy')
        emit_on_self(tensor_mask_ub, op='dma_copy')
        emit_on_self(tensor_x_ub_prob, op='vector_div')
        emit_on_self(tensor_x_ub_sel, op='elewise_multiple_sel')
        s[tensor_xh_l1].emit_insn(tensor_xh_l1.op.axis[1], 'dma_copy')
        build_list = [tensor_x, tensor_h, tensor_c, tensor_w, tensor_b,
                      tensor_mask, ct, ht, it, jt, ft, ot, tanhct]
    else:
        s[tensor_xh_l1].emit_insn(tensor_xh_l1.op.axis[0], 'dma_copy')
    s[tensor_ct_ub_fake].compute_inline()

    with build_config:
        tvm.build(s, build_list, "cce", name=kernel_name)
