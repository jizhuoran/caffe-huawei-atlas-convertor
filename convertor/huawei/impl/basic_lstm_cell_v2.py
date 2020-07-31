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

basic_lstm_cell_v2
"""

# pylint: disable=locally-disabled,import-error,unused-import,ungrouped-imports
import te.lang.cce
from te import tvm, platform as cceconf
import te.platform.cce_params as cce
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.platform.cce_build import build_config
import topi

NONETYPE = type(None)
C0 = 16


# pylint: disable=locally-disabled,unnecessary-lambda,too-many-locals,no-else-return,too-many-lines,too-many-arguments,cell-var-from-loop
def handle_static(w_xc_x_static, gate_shape, build_list, product_info,
                  tensor_list, scope_list, operation_list):
    """
    handle the static input, it, ot, ft, gt.
    """
    static_dtype = w_xc_x_static["dtype"]
    static_shape = w_xc_x_static["shape"]
    output_dim = gate_shape[0]
    tensor_static = tvm.placeholder(static_shape, name='tensor_x_staic', dtype=static_dtype)
    build_list["w_xc_x_static"] = tensor_static
    symbol = ["it", "ft", "ot", "gt"]
    def _index(str_in):
        if str_in == "it":
            return 0
        elif str_in == "ft":
            return 1
        elif str_in == "ot":
            return 2
        return 3
    def _tensor(str_in):
        if str_in == "it":
            return tensor_list["it_ub"]
        elif str_in == "ft":
            return tensor_list["ft_ub"]
        elif str_in == "ot":
            return tensor_list["ot_ub"]
        return tensor_list["gt_ub"]
    for index_name in symbol:
        tensor_static_ub_tmp = tvm.compute(gate_shape,
                                           lambda a, b, c, d:
                                           tensor_static[a + output_dim
                                                         * _index(index_name),
                                                         b, c, d],
                                           name="tensor_static_ub_" + index_name)
        tensor_list["tensor_static_ub_" + index_name] = tensor_static_ub_tmp
        scope_list["tensor_static_ub_" + index_name] = cce.scope_ubuf
        operation_list["tensor_static_ub_" + index_name] = "dma_copy"
        tensor_static_ub_tmp_true = tensor_static_ub_tmp
        if not product_info["hisi_es"] and static_dtype == "float16":
            tensor_static_ub_tmp_true = tvm.compute(gate_shape,
                                                    lambda *i:
                                                    topi.cast(tensor_static_ub_tmp(*i),
                                                              "float32"),
                                                    name="tensor_static_ub_true_"
                                                    + index_name)
            tensor_list["tensor_static_ub_true_" + index_name] = tensor_static_ub_tmp_true
            scope_list["tensor_static_ub_true_" + index_name] = cce.scope_ubuf
            operation_list["tensor_static_ub_true_" + index_name] = "vector_conv"
        tmp_ub_true = tvm.compute(gate_shape,
                                  lambda *i:
                                  tensor_static_ub_tmp_true(*i) +
                                  _tensor(index_name)(*i),
                                  name=index_name + "_ub_true")
        tensor_list[index_name + "_ub_true"] = tmp_ub_true
        scope_list[index_name + "_ub_true"] = cce.scope_ubuf
        operation_list[index_name + "_ub_true"] = "vector_add"


# pylint: disable=locally-disabled,too-many-statements,unnecessary-lambda,too-many-locals,invalid-name,too-many-arguments
@fusion_manager.register("basic_lstm_cell_v2")
def basic_lstm_cell_v2_compute(x, cont, w_xc_x_static, w_xh, bias, h_t, c_t,
                               expose_hidden, product_info):
    """
    the compute for LSTM op, return the compute object
    """
    build_list = {}
    tensor_list = {}
    scope_list = {}
    operation_list = {}
    gate_shape = h_t["shape"]
    dtype_c = c_t["dtype"]
    dtype_h = h_t["dtype"]

    get_matmul_tensor(x, cont, h_t, w_xh, bias, expose_hidden, build_list,
                      product_info["hisi_es"], tensor_list, scope_list, operation_list)


    it_ub = tvm.compute(gate_shape,
                        lambda *i: tensor_list["tensor_matmul_result_l0c_it"](*i),
                        name="it_ub")
    ft_ub = tvm.compute(gate_shape,
                        lambda *i: tensor_list["tensor_matmul_result_l0c_ft"](*i),
                        name="ft_ub")
    ot_ub = tvm.compute(gate_shape,
                        lambda *i: tensor_list["tensor_matmul_result_l0c_ot"](*i),
                        name="ot_ub")
    gt_ub = tvm.compute(gate_shape,
                        lambda *i: tensor_list["tensor_matmul_result_l0c_gt"](*i),
                        name="gt_ub")
    tensor_list["it_ub"] = it_ub
    scope_list["it_ub"] = cce.scope_ubuf
    operation_list["it_ub"] = "dma_copy"
    tensor_list["ft_ub"] = ft_ub
    scope_list["ft_ub"] = cce.scope_ubuf
    operation_list["ft_ub"] = "dma_copy"
    tensor_list["ot_ub"] = ot_ub
    scope_list["ot_ub"] = cce.scope_ubuf
    operation_list["ot_ub"] = "dma_copy"
    tensor_list["gt_ub"] = gt_ub
    scope_list["gt_ub"] = cce.scope_ubuf
    operation_list["gt_ub"] = "dma_copy"

    it_ub_true = it_ub
    ft_ub_true = ft_ub
    ot_ub_true = ot_ub
    gt_ub_true = gt_ub
    if w_xc_x_static is not None:
        handle_static(w_xc_x_static, gate_shape, build_list,
                      product_info, tensor_list, scope_list, operation_list)
        it_ub_true = tensor_list["it_ub_true"]
        ft_ub_true = tensor_list["ft_ub_true"]
        ot_ub_true = tensor_list["ot_ub_true"]
        gt_ub_true = tensor_list["gt_ub_true"]

    shape_cont = cont["shape"]
    # [N,] -> [N//16, 16]
    ub_cont = [shape_cont[0] // 16, 16]
    tensor_cont_ub = tensor_list["tensor_cont_ub"]
    tensor_cont_f_ub_tmp = tensor_cont_ub
    if tensor_cont_ub.dtype != ft_ub_true.dtype:
        tensor_cont_f_ub_tmp = tvm.compute(ub_cont,
                                           lambda *i:
                                           topi.cast(tensor_cont_ub(*i),
                                                     dtype=ft_ub_true.dtype),
                                           name="tensor_cont_f_ub_tmp")
        tensor_list["tensor_cont_f_ub_tmp"] = tensor_cont_f_ub_tmp
        scope_list["tensor_cont_f_ub_tmp"] = cce.scope_ubuf
        operation_list["tensor_cont_f_ub_tmp"] = "vector_conv"


    # Do sigmoid(It,Ft,ot) calculation
    tensor_one = tvm.compute(gate_shape,
                             lambda *i:
                             tvm.const(1, dtype=gt_ub_true.dtype),
                             name='tensor_one')
    if product_info["cloud"]:
        tensor_list["tensor_one"] = tensor_one
        scope_list["tensor_one"] = cce.scope_ubuf
        operation_list["tensor_one"] = "vector_dup"

    sigmoid(gate_shape, it_ub_true, tensor_one,
            product_info, "it", tensor_list,
            scope_list, operation_list)
    sigmoid(gate_shape, ft_ub_true,
            tensor_one, product_info, "ft",
            tensor_list, scope_list, operation_list)
    ft_cont_ub_true = tvm.compute(gate_shape,
                                  lambda i0, i1, i2, i3:
                                  tensor_list["tensor_gate_sigmoid_ft"][i0, i1, i2, i3]
                                  * tensor_cont_f_ub_tmp[i1, i2],
                                  name="ft_cont_ub_true")

    tensor_list["ft_cont_ub_true"] = ft_cont_ub_true
    scope_list["ft_cont_ub_true"] = cce.scope_ubuf
    operation_list["ft_cont_ub_true"] = "vector_muls"
    sigmoid(gate_shape, ot_ub_true,
            tensor_one, product_info, "ot",
            tensor_list, scope_list, operation_list)
    # Do tanh(gt) calculation
    tanh(gate_shape, gt_ub_true, product_info,
         "gt", tensor_list, scope_list, operation_list)

    # calc ft*c + it*gt
    if expose_hidden:
        tensor_c = tvm.placeholder(gate_shape,
                                   name='tensor_c', dtype=dtype_c)
        build_list["c"] = tensor_c
        tensor_c_ub = tvm.compute(gate_shape,
                                  lambda *i: tensor_c(*i),
                                  name='tensor_c_ub')
        tensor_list["tensor_c_ub"] = tensor_c_ub
        scope_list["tensor_c_ub"] = cce.scope_ubuf
        operation_list["tensor_c_ub"] = "dma_copy"
    else:
        tensor_c_ub = tvm.compute(gate_shape,
                                  lambda *i: tvm.const(0, dtype=dtype_c),
                                  name='tensor_c_ub')
        tensor_list["tensor_c_ub"] = tensor_c_ub
        scope_list["tensor_c_ub"] = cce.scope_ubuf
        operation_list["tensor_c_ub"] = "vector_dup"
    tensor_c_ub_true = tensor_c_ub
    if not product_info["hisi_es"] and dtype_c == "float16":
        tensor_c_ub_true = tvm.compute(gate_shape,
                                       lambda *i:
                                       topi.cast(tensor_c_ub(*i), "float32"),
                                       name="tensor_c_ub_true")
        tensor_list["tensor_c_ub_true"] = tensor_c_ub_true
        scope_list["tensor_c_ub_true"] = cce.scope_ubuf
        operation_list["tensor_c_ub_true"] = "vector_conv"
    tensor_cf_ub = tvm.compute(gate_shape,
                               lambda *i: tensor_c_ub_true(*i) *
                               tensor_list["ft_cont_ub_true"](*i),
                               name="tensor_cf_ub")
    tensor_list["tensor_cf_ub"] = tensor_cf_ub
    scope_list["tensor_cf_ub"] = cce.scope_ubuf
    operation_list["tensor_cf_ub"] = "vector_mul"
    tensor_gi_ub = tvm.compute(gate_shape,
                               lambda *i: tensor_list["tensor_ub_tanh_gt"](*i) *
                               tensor_list["tensor_gate_sigmoid_it"](*i),
                               name="tensor_gi_ub")
    tensor_list["tensor_gi_ub"] = tensor_gi_ub
    scope_list["tensor_gi_ub"] = cce.scope_ubuf
    operation_list["tensor_gi_ub"] = "vector_mul"
    tensor_ct_ub = tvm.compute(gate_shape,
                               lambda *i: tensor_cf_ub(*i) + tensor_gi_ub(*i),
                               name="tensor_ct_ub")
    tensor_list["tensor_ct_ub"] = tensor_ct_ub
    scope_list["tensor_ct_ub"] = cce.scope_ubuf
    operation_list["tensor_ct_ub"] = "vector_add"

    tensor_ct_ub_true = tensor_ct_ub
    if dtype_c == "float16" and tensor_ct_ub.dtype != dtype_c:
        tensor_ct_ub_true = tvm.compute(gate_shape,
                                        lambda *i:
                                        topi.cast(tensor_ct_ub(*i), "float16"),
                                        name="tensor_ct_ub_true")
        tensor_list["tensor_ct_ub_true"] = tensor_ct_ub_true
        scope_list["tensor_ct_ub_true"] = cce.scope_ubuf
        operation_list["tensor_ct_ub_true"] = "vector_conv"
    elif dtype_c == "float32" and tensor_ct_ub.dtype != dtype_c:
        tensor_ct_ub_true = tvm.compute(gate_shape,
                                        lambda *i:
                                        topi.cast(tensor_ct_ub(*i), "float32"),
                                        name="tensor_ct_ub_true")
        tensor_list["tensor_ct_ub_true"] = tensor_ct_ub_true
        scope_list["tensor_ct_ub_true"] = cce.scope_ubuf
        operation_list["tensor_ct_ub_true"] = "vector_conv"
    # Move ct to gm
    ct = tvm.compute(gate_shape, lambda *i: tensor_ct_ub_true(*i), name="ct")
    build_list["ct"] = ct
    tensor_list["ct"] = ct
    operation_list["ct"] = "dma_copy"
    # Move ct back(Fake)
    tensor_ct_ub_fake = tvm.compute(gate_shape, lambda *i: ct(*i), name="ct_ub_fake")
    tensor_list["tensor_ct_ub_fake"] = tensor_ct_ub_fake
    tensor_ct_ub_fake_true = tensor_ct_ub_fake
    if not product_info["hisi_es"] and tensor_ct_ub_fake.dtype == "float16":
        tensor_ct_ub_fake_true = tvm.compute(gate_shape,
                                             lambda *i:
                                             topi.cast(tensor_ct_ub_fake(*i), "float32"),
                                             name="tensor_ct_ub_fake_true")
        tensor_list["tensor_ct_ub_fake_true"] = tensor_ct_ub_fake_true
        scope_list["tensor_ct_ub_fake_true"] = cce.scope_ubuf
        operation_list["tensor_ct_ub_fake_true"] = "vector_conv"
    # calc tanh(ct)
    tanh(gate_shape, tensor_ct_ub_fake_true, product_info,
         "ct", tensor_list, scope_list, operation_list)

    tensor_ht_ub = tvm.compute(gate_shape,
                               lambda *i:
                               tensor_list["tensor_gate_sigmoid_ot"](*i) *
                               tensor_list["tensor_ub_tanh_ct"](*i),
                               name="tensor_ht_ub")
    tensor_list["tensor_ht_ub"] = tensor_ht_ub
    scope_list["tensor_ht_ub"] = cce.scope_ubuf
    operation_list["tensor_ht_ub"] = "vector_mul"

    tensor_ht_ub_true = tensor_ht_ub
    if not product_info["hisi_es"] and dtype_h == "float16":
        tensor_ht_ub_true = tvm.compute(gate_shape,
                                        lambda *i:
                                        topi.cast(tensor_ht_ub(*i), "float16"),
                                        name="tensor_ht_ub_true")
        tensor_list["tensor_ht_ub_true"] = tensor_ht_ub_true
        scope_list["tensor_ht_ub_true"] = cce.scope_ubuf
        operation_list["tensor_ht_ub_true"] = "vector_conv"
    # Move ht to gm
    ht = tvm.compute(gate_shape, lambda *i: tensor_ht_ub_true(*i), name="ht")
    tensor_list["ht"] = ht
    build_list["ht"] = ht
    return tensor_list, scope_list, operation_list, build_list


# pylint: disable=locally-disabled,too-many-statements,cell-var-from-loop,unnecessary-lambda,too-many-locals,invalid-name,too-many-arguments
def get_matmul_tensor(x, cont, h_t, w_xh, bias,
                      expose_hidden, build_list, is_hisi_es,
                      tensor_list, scope_list, operation_list):
    """
    compute for matmul, return matmul compute object
    """
    dtype_h = h_t["dtype"]
    shape_h = h_t["shape"]
    shape_x = x["shape"]
    dtype_x = x["dtype"]
    shape_w = w_xh["shape"]
    dtype_w = w_xh["dtype"]
    shape_bias = bias["shape"]
    shape_bias = (shape_bias[0]/16, 16)
    dtype_bias = bias["dtype"]
    shape_cont = cont["shape"]
    ub_cont = [shape_cont[0] // 16, 16]
    dtype_cont = cont["dtype"]
    tensor_x = tvm.placeholder(shape_x, name='tensor_x', dtype=dtype_x)
    tensor_w = tvm.placeholder(shape_w, name='tensor_w', dtype=dtype_w)

    tensor_b = tvm.placeholder(shape_bias, name='tensor_b', dtype=dtype_bias)
    tensor_cont = tvm.placeholder(ub_cont, name='tensor_cont', dtype=dtype_cont)
    build_list["x"] = tensor_x
    build_list["w_xh"] = tensor_w
    build_list["bias"] = tensor_b
    build_list["cont"] = tensor_cont

    input_dim, batch_dim = shape_x[0:2]
    output_dim = shape_h[0]
    left_shape = (batch_dim, input_dim + output_dim, C0, C0)
    right_shape = list(shape_w)
    right_shape[1] = right_shape[1] // 4
    matmul_shape = shape_h

    symbol = ["it", "ft", "ot", "gt"]

    def _index_w(str_name, *index):
        if str_name == "it":
            return index[0], index[1], index[2], index[3]
        elif str_name == "ft":
            return index[0], index[1] + output_dim, index[2], index[3]
        elif str_name == "ot":
            return index[0], index[1] + output_dim * 2, index[2], index[3]
        return index[0], index[1] + output_dim * 3, index[2], index[3]

    def _index_bias(str_name):
        if str_name == "it":
            return 0
        elif str_name == "ft":
            return 1
        elif str_name == "ot":
            return 2
        return 3

    matmul_type = "float32"
    if is_hisi_es:
        matmul_type = "float16"
    if expose_hidden:
        tensor_h = tvm.placeholder(shape_h, name='tensor_h', dtype=dtype_h)
        build_list["h"] = tensor_h
        tensor_h_ub = tvm.compute(shape_h,
                                  lambda *i: tensor_h(*i), name='tensor_h_ub')
        tensor_list["tensor_h_ub"] = tensor_h_ub
        scope_list["tensor_h_ub"] = cce.scope_ubuf
        operation_list["tensor_h_ub"] = "dma_copy"
    else:
        tensor_h_ub = tvm.compute(shape_h,
                                  lambda *i: tvm.const(0, dtype=dtype_h),
                                  name='tensor_h_ub')
        tensor_list["tensor_h_ub"] = tensor_h_ub
        scope_list["tensor_h_ub"] = cce.scope_ubuf
        operation_list["tensor_h_ub"] = "vector_dup"

    tensor_h_ub_true = tensor_h_ub
    if dtype_h == "float32":
        tensor_h_ub_true = tvm.compute(shape_h,
                                       lambda *i:
                                       topi.cast(tensor_h_ub, dtype="float16"),
                                       name="tensor_h_ub_true")
        tensor_list["tensor_h_ub_true"] = tensor_h_ub_true
        scope_list["tensor_h_ub_true"] = cce.scope_ubuf
        operation_list["tensor_h_ub_true"] = "vector_conv"

    tensor_cont_ub = tvm.compute(ub_cont,
                                 lambda *i: tensor_cont(*i),
                                 name="tensor_cont_ub")

    tensor_list["tensor_cont_ub"] = tensor_cont_ub
    scope_list["tensor_cont_ub"] = cce.scope_ubuf
    operation_list["tensor_cont_ub"] = "dma_copy"
    tensor_cont_ub_tmp = tensor_cont_ub
    if dtype_cont != "float16":
        tensor_cont_ub_tmp = tvm.compute(ub_cont,
                                         lambda *i:
                                         topi.cast(tensor_cont_ub(*i), dtype="float16"),
                                         name="tensor_cont_ub_tmp")
        tensor_list["tensor_cont_ub_tmp"] = tensor_cont_ub_tmp
        scope_list["tensor_cont_ub_tmp"] = cce.scope_ubuf
        operation_list["tensor_cont_ub_tmp"] = "vector_conv"

    tensor_h_cont_ub = tvm.compute(shape_h,
                                   lambda i0, i1, i2, i3:
                                   tensor_h_ub_true[i0, i1, i2, i3] *
                                   tensor_cont_ub_tmp[i1, i2],
                                   name="tensor_h_cont_ub")
    tensor_list["tensor_h_cont_ub"] = tensor_h_cont_ub
    scope_list["tensor_h_cont_ub"] = cce.scope_ubuf
    operation_list["tensor_h_cont_ub"] = "vector_muls"

    for t in symbol:
        tensor_xh_l1_tmp = tvm.compute(left_shape,
                                       lambda *indice:
                                       tvm.select(indice[1] < input_dim,
                                                  tensor_x[indice[1],
                                                           indice[0],
                                                           indice[2],
                                                           indice[3]],
                                                  tensor_h_cont_ub[indice[1] - input_dim,
                                                                   indice[0],
                                                                   indice[2],
                                                                   indice[3]]),
                                       name="tensor_xh_l1_" + t, tag="concat")
        tensor_list["tensor_xh_l1_" + t] = tensor_xh_l1_tmp
        scope_list["tensor_xh_l1_" + t] = cce.scope_cbuf

        if t == "ot":
            operation_list["tensor_xh_l1_" + t] = "dma_copy"
        else:
            operation_list["tensor_xh_l1_" + t] = "phony_insn"

        tensor_xh_loa_tmp = tvm.compute(left_shape,
                                        lambda *i: tensor_xh_l1_tmp(*i),
                                        name='tensor_xh_l0a_' + t)
        tensor_list["tensor_xh_l0a_" + t] = tensor_xh_loa_tmp
        scope_list["tensor_xh_l0a_" + t] = cce.scope_ca

        if t == "ot":
            operation_list["tensor_xh_l0a_" + t] = "dma_copy"
        else:
            operation_list["tensor_xh_l0a_" + t] = "phony_insn"


        tensor_w_l1_tmp = tvm.compute(right_shape,
                                      lambda *i: tensor_w(*_index_w(t, *i)),
                                      name='tensor_w_l1_' + t)
        tensor_list["tensor_w_l1_" + t] = tensor_w_l1_tmp
        scope_list["tensor_w_l1_" + t] = cce.scope_cbuf
        operation_list["tensor_w_l1_" + t] = "dma_copy"

        tensor_w_l0b_tmp = tvm.compute(right_shape,
                                       lambda *i: tensor_w_l1_tmp(*i),
                                       name='tensor_w_l0b_' + t)
        tensor_list["tensor_w_l0b_" + t] = tensor_w_l0b_tmp
        scope_list["tensor_w_l0b_" + t] = cce.scope_cb
        operation_list["tensor_w_l0b_" + t] = "dma_copy"

        tensor_b_ub_tmp = tvm.compute(shape_bias,
                                      lambda i0, i1:
                                      tensor_b[_index_bias(t) *
                                               output_dim + i0, i1],
                                      name='tensor_b_ub_' + t)
        tensor_list["tensor_b_ub_" + t] = tensor_b_ub_tmp
        scope_list["tensor_b_ub_" + t] = cce.scope_ubuf
        operation_list["tensor_b_ub_" + t] = "dma_copy"

        tensor_b_ub_true_tmp = tensor_b_ub_tmp
        if not is_hisi_es and dtype_bias == "float16":
            tensor_b_ub_true_tmp = tvm.compute(shape_bias,
                                               lambda *i:
                                               topi.cast(tensor_b_ub_tmp(*i), "float32"),
                                               name="tensor_b_ub_true_" + t)
            tensor_list["tensor_b_ub_true_" + t] = tensor_b_ub_true_tmp
            scope_list["tensor_b_ub_true_" + t] = cce.scope_ubuf
            operation_list["tensor_b_ub_true_" + t] = "vector_conv"
        tensor_b_loc_tmp = tvm.compute(matmul_shape,
                                       lambda i0, i1, i2, i3:
                                       tensor_b_ub_true_tmp[i0, i3],
                                       name='tensor_b_loc_' + t)
        tensor_list["tensor_b_loc_" + t] = tensor_b_loc_tmp
        scope_list["tensor_b_loc_" + t] = cce.scope_cc
        operation_list["tensor_b_loc_" + t] = "dma_copy"

        reduce_kb = tvm.reduce_axis((0, input_dim + output_dim), name='reduce_kb_' + t)
        reduce_kp = tvm.reduce_axis((0, C0), name='reduce_kp_' + t)
        tensor_matmul_l0c_tmp = tvm.compute(
            matmul_shape, lambda nb, mb, mp, np: tvm.sum(
                (tensor_xh_loa_tmp[mb, reduce_kb, mp, reduce_kp] *
                 tensor_w_l0b_tmp[reduce_kb, nb, np, reduce_kp]).astype(
                     matmul_type), axis=[reduce_kb, reduce_kp]),
            name='tensor_matmul_l0c_' + t, attrs={'input_order': 'positive'})
        tensor_list["tensor_matmul_l0c_" + t] = tensor_matmul_l0c_tmp
        scope_list["tensor_matmul_l0c_" + t] = cce.scope_cc
        # Matmul accumulation it + b_it
        tensor_matmul_result_l0c_tmp = tvm.compute(matmul_shape,
                                                   lambda *i: tensor_b_loc_tmp(*i) +
                                                   tensor_matmul_l0c_tmp(*i),
                                                   name="tensor_matmul_result_l0c_" + t)
        tensor_list["tensor_matmul_result_l0c_" + t] = tensor_matmul_result_l0c_tmp
        scope_list["tensor_matmul_result_l0c_" + t] = cce.scope_cc
        operation_list["tensor_matmul_result_l0c_" + t] = "phony_insn"


# pylint: disable=locally-disabled,too-many-statements,cell-var-from-loop,unnecessary-lambda,too-many-locals,invalid-name,too-many-arguments
def newton_iteration(shape, tensor_x_rec,
                     tensor_x, symbol, tensor_list, scope_list, operation_list):
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
                                     name="tensor_newton_mul0_" + symbol)
    tensor_list["tensor_newton_mul0_" + symbol] = tensor_newton_mul0
    scope_list["tensor_newton_mul0_" + symbol] = cce.scope_ubuf
    operation_list["tensor_newton_mul0_" + symbol] = "vector_mul"
    tensor_newton_add = tvm.compute(shape,
                                    lambda *i: tensor_newton_mul0(*i) + const_num_neg_two,
                                    name="tensor_newton_add_" + symbol)
    tensor_list["tensor_newton_add_" + symbol] = tensor_newton_add
    scope_list["tensor_newton_add_" + symbol] = cce.scope_ubuf
    operation_list["tensor_newton_add_" + symbol] = "vector_add"
    tensor_newton_mul1 = tvm.compute(shape,
                                     lambda *i: tensor_newton_add(*i) * tensor_x_rec(*i),
                                     name="tensor_newton_mul1_" + symbol)
    tensor_list["tensor_newton_mul1_" + symbol] = tensor_newton_mul1
    scope_list["tensor_newton_mul1_" + symbol] = cce.scope_ubuf
    operation_list["tensor_newton_mul1_" + symbol] = "vector_mul"
    tensor_newton_mul2 = tvm.compute(shape,
                                     lambda *i: tensor_newton_mul1(*i) * const_num_neg_one,
                                     name="tensor_newton_mul2_" + symbol)
    return tensor_newton_mul2


# pylint: disable=locally-disabled,too-many-statements,cell-var-from-loop,unnecessary-lambda,too-many-locals,invalid-name,too-many-arguments
def sigmoid(shape, tensor_allgate_ub, tensor_one,
            product_info, symbol, tensor_list, scope_list, operation_list):
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
    tensor_list["tensor_gate_neg_" + symbol] = tensor_ub_neg_allgate
    scope_list["tensor_gate_neg_" + symbol] = cce.scope_ubuf
    operation_list["tensor_gate_neg_" + symbol] = "vector_mul"
    tensor_ub_allgate_exp_fp16 = tensor_ub_neg_allgate
    if product_info["mini"]:
        tensor_ub_allgate_exp_fp16 = tvm.compute(shape,
                                                 lambda *i: topi.cast(tensor_ub_neg_allgate(*i),
                                                                      "float16"),
                                                 name="tensor_gate_exp_fp16_" + symbol)
        tensor_list["tensor_gate_exp_fp16_" + symbol] = tensor_ub_allgate_exp_fp16
        scope_list["tensor_gate_exp_fp16_" + symbol] = cce.scope_ubuf
        operation_list["tensor_gate_exp_fp16_" + symbol] = "vector_conv"
    tensor_ub_allgate_exp = tvm.compute(shape,
                                        lambda *i: tvm.exp(tensor_ub_allgate_exp_fp16(*i)),
                                        name="tensor_gate_exp_" + symbol)
    tensor_list["tensor_gate_exp_" + symbol] = tensor_ub_allgate_exp
    scope_list["tensor_gate_exp_" + symbol] = cce.scope_ubuf
    operation_list["tensor_gate_exp_" + symbol] = "vector_exp"
    tensor_ub_allgate_exp_fp32 = tensor_ub_allgate_exp
    if not product_info["hisi_es"]:
        tensor_ub_allgate_exp_fp32 = tvm.compute(shape,
                                                 lambda *i:
                                                 topi.cast(tensor_ub_allgate_exp(*i),
                                                           dtype_c),
                                                 name="tensor_gate_exp_fp32_" + symbol)
        tensor_list["tensor_gate_exp_fp32_" + symbol] = tensor_ub_allgate_exp_fp32
        scope_list["tensor_gate_exp_fp32_" + symbol] = cce.scope_ubuf
        operation_list["tensor_gate_exp_fp32_" + symbol] = "vector_conv"
    tensor_ub_allgate_add = tvm.compute(shape,
                                        lambda *i:
                                        tensor_ub_allgate_exp_fp32(*i) + const_num_one,
                                        name="tensor_gate_add_" + symbol)
    tensor_list["tensor_gate_add_" + symbol] = tensor_ub_allgate_add
    scope_list["tensor_gate_add_" + symbol] = cce.scope_ubuf
    operation_list["tensor_gate_add_" + symbol] = "vector_add"

    if product_info["cloud"]:
        tensor_ub_allgate_sigmoid = tvm.compute(shape,
                                                lambda *i: tensor_one(*i) /
                                                tensor_ub_allgate_add(*i),
                                                name="tensor_gate_sigmoid_" + symbol)
        tensor_newton_mul2 = tensor_ub_allgate_sigmoid
        tensor_list["tensor_gate_sigmoid_" + symbol] = tensor_newton_mul2
        scope_list["tensor_gate_sigmoid_" + symbol] = cce.scope_ubuf
        operation_list["tensor_gate_sigmoid_" + symbol] = "vector_div"
    else:
        tensor_ub_allgate_sigmoid = tvm.compute(shape,
                                                lambda *i:
                                                const_num_one / tensor_ub_allgate_add(*i),
                                                name="tensor_gate_sigmoid_" + symbol)
        tensor_list["tensor_gate_sigmoid_tmp_" + symbol] = tensor_ub_allgate_sigmoid
        scope_list["tensor_gate_sigmoid_tmp_" + symbol] = cce.scope_ubuf
        operation_list["tensor_gate_sigmoid_tmp_" + symbol] = "vector_rec"
        tensor_newton_mul2 = newton_iteration(shape,
                                              tensor_ub_allgate_sigmoid,
                                              tensor_ub_allgate_add,
                                              symbol, tensor_list,
                                              scope_list, operation_list)
        tensor_list["tensor_gate_sigmoid_" + symbol] = tensor_newton_mul2
        scope_list["tensor_gate_sigmoid_" + symbol] = cce.scope_ubuf
        operation_list["tensor_gate_sigmoid_" + symbol] = "vector_mul"


# pylint: disable=locally-disabled,too-many-statements,cell-var-from-loop,unnecessary-lambda,too-many-locals,invalid-name,too-many-arguments
def tanh(shape, tensor, product_info, symbol,
         tensor_list, scope_list, operation_list):
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
    const_num_one = tvm.const(1, dtype=dtype_c)
    const_num_two = tvm.const(-2, dtype=dtype_c)
    const_fp32_min = tvm.const(2 ** (-126), dtype=dtype_c)

    tensor_ub_two_abs = tvm.compute(shape, lambda *i: tvm.abs(tensor(*i)), name="tensor_ub_two_abs_"+symbol)
    tensor_list["tensor_ub_two_abs_" + symbol] = tensor_ub_two_abs
    scope_list["tensor_ub_two_abs_" + symbol] = cce.scope_ubuf
    operation_list["tensor_ub_two_abs_" + symbol] = "vector_abs"

    tensor_ub_two = tvm.compute(shape, lambda *i: tensor_ub_two_abs(*i) * const_num_two,
                                name="tensor_ub_two_" + symbol)
    tensor_list["tensor_ub_two_" + symbol] = tensor_ub_two
    scope_list["tensor_ub_two_" + symbol] = cce.scope_ubuf
    operation_list["tensor_ub_two_" + symbol] = "vector_mul"

    tensor_ub_exp_fp16 = tensor_ub_two
    if product_info["mini"] and dtype_c == "float32":
        tensor_ub_exp_fp16 = tvm.compute(shape,
                                         lambda *i:
                                         topi.cast(tensor_ub_two(*i), "float16"),
                                         name="tensor_ub_exp_fp16_" + symbol)
        tensor_list["tensor_ub_exp_fp16_" + symbol] = tensor_ub_exp_fp16
        scope_list["tensor_ub_exp_fp16_" + symbol] = cce.scope_ubuf
        operation_list["tensor_ub_exp_fp16_" + symbol] = "vector_conv"

    tensor_ub_exp = tvm.compute(shape,
                                lambda *i: tvm.exp(tensor_ub_exp_fp16(*i)),
                                name="tensor_ub_exp_" + symbol)
    tensor_list["tensor_ub_exp_" + symbol] = tensor_ub_exp
    scope_list["tensor_ub_exp_" + symbol] = cce.scope_ubuf
    operation_list["tensor_ub_exp_" + symbol] = "vector_exp"
    tensor_ub_exp_fp32 = tensor_ub_exp
    if dtype_c == "float32":
        tensor_ub_exp_fp32 = tvm.compute(shape,
                                         lambda *i:
                                         topi.cast(tensor_ub_exp(*i), "float32"),
                                         name="tensor_ub_exp_fp32_" + symbol)
        tensor_list["tensor_ub_exp_fp32_" + symbol] = tensor_ub_exp_fp32
        scope_list["tensor_ub_exp_fp32_" + symbol] = cce.scope_ubuf
        operation_list["tensor_ub_exp_fp32_" + symbol] = "vector_conv"

    tensor_mul_temp =  tvm.compute(
        shape, lambda *i: tensor_ub_exp_fp32(*i) * tensor(*i), name="tensor_mul_temp_"+symbol)
    tensor_list["tensor_mul_temp_" + symbol] = tensor_mul_temp
    scope_list["tensor_mul_temp_" + symbol] = cce.scope_ubuf
    operation_list["tensor_mul_temp_" + symbol] = "vector_mul"

    tensor_sub_temp = tvm.compute(
        shape, lambda *i: tensor(*i) - tensor_mul_temp(*i), name="tensor_sub_temp_"+symbol)
    tensor_list["tensor_sub_temp_" + symbol] = tensor_sub_temp
    scope_list["tensor_sub_temp_" + symbol] = cce.scope_ubuf
    operation_list["tensor_sub_temp_" + symbol] = "vector_sub"

    tenosr_add_min_temp = tvm.compute(
        shape, lambda *i: tensor_ub_two_abs(*i) + const_fp32_min, name="tenosr_add_min_temp_"+symbol)
    tensor_list["tenosr_add_min_temp_" + symbol] = tenosr_add_min_temp
    scope_list["tenosr_add_min_temp_" + symbol] = cce.scope_ubuf
    operation_list["tenosr_add_min_temp_" + symbol] = "vector_add"

    tenosr_add_1_temp = tvm.compute(
        shape, lambda *i: tensor_ub_exp_fp32(*i) + const_num_one, name="tenosr_add_1_temp_" + symbol)
    tensor_list["tenosr_add_1_temp_" + symbol] = tenosr_add_1_temp
    scope_list["tenosr_add_1_temp_" + symbol] = cce.scope_ubuf
    operation_list["tenosr_add_1_temp_" + symbol] = "vector_add"

    tenosr_down_temp = tvm.compute(
        shape, lambda *i: tenosr_add_1_temp(*i)*tenosr_add_min_temp(*i), name="tenosr_down_temp_" + symbol)
    tensor_list["tenosr_down_temp_" + symbol] = tenosr_down_temp
    scope_list["tenosr_down_temp_" + symbol] = cce.scope_ubuf
    operation_list["tenosr_down_temp_" + symbol] = "vector_mul"

    tensor_ub_rec = tvm.compute(shape,
                                lambda *i: const_num_one / tenosr_down_temp(*i),
                                name="tensor_ub_rec_" + symbol)
    tensor_list["tensor_ub_rec_" + symbol] = tensor_ub_rec
    scope_list["tensor_ub_rec_" + symbol] = cce.scope_ubuf
    operation_list["tensor_ub_rec_" + symbol] = "vector_rec"

    tensor_newton_mul2 = newton_iteration(shape, tensor_ub_rec, tenosr_down_temp,
                                          symbol, tensor_list,
                                          scope_list, operation_list)
    tensor_list["tensor_ub_tanh_newton_" + symbol] = tensor_newton_mul2
    scope_list["tensor_ub_tanh_newton_" + symbol] = cce.scope_ubuf
    operation_list["tensor_ub_tanh_newton_" + symbol] = "vector_mul"

    tensor_ub_tanh = tvm.compute(shape,
                                 lambda *i:
                                 tensor_sub_temp(*i) * tensor_newton_mul2(*i),
                                 name="tensor_ub_tanh_" + symbol)
    tensor_list["tensor_ub_tanh_" + symbol] = tensor_ub_tanh
    scope_list["tensor_ub_tanh_" + symbol] = cce.scope_ubuf
    operation_list["tensor_ub_tanh_" + symbol] = "vector_mul"

# pylint: disable=locally-disabled,too-many-statements,cell-var-from-loop,unnecessary-lambda,too-many-locals,invalid-name,too-many-arguments,consider-using-in
def basiclstm_cell_v2_check(x, w_xc_x_static, h, c, w_xh, h_t, c_t,
                            expose_hidden, num_output, product_info):
    """
    check params for the LSTM input and output and attr.
    """
    dtype_all = {"float16", "float32"} if not product_info["hisi_es"] else {"float16"}
    dtype_str = "float16 or float32" if not product_info["hisi_es"] else "float16"
    shape_x = x["shape"]
    shape_h = h_t["shape"]
    n_dim = shape_x[1]
    intput_dim = shape_x[0]
    output_dim = shape_h[0]
    if num_output != 0 and num_output != output_dim * C0:
        raise RuntimeError("num_output[%s] is not equal"
                           " output_dim[%s]!"%(str(num_output), str(num_output * C0)))
    if x["dtype"] != "float16" or w_xh["dtype"] != "float16":
        raise RuntimeError("x, w supports dtype float16 only!")
    if w_xh["shape"][0] != output_dim + intput_dim or w_xh["shape"][1] != 4 * output_dim:
        raise RuntimeError("w_xh shape is wrong, please check!")
    if expose_hidden:
        if h is None or c is None:
            raise RuntimeError("h, c can not be None when expose_hidden is True!")
        if h["dtype"] not in dtype_all or c["dtype"] not in dtype_all:
            raise RuntimeError("h, c supports dtype(%s) only!"%(dtype_str))
        if h["shape"][0] != output_dim or h["shape"][1] != n_dim or \
                c["shape"][0] != output_dim or c["shape"][1] != n_dim:
            raise RuntimeError("h or c shape is wrong, please check!")
    if h_t["dtype"] not in dtype_all or c_t["dtype"] not in dtype_all:
        raise RuntimeError("h_t, c_t supports dtype(%s) only!"%(dtype_str))
    if c_t["shape"][0] != output_dim or c_t["shape"][1] != n_dim or shape_h[1] != n_dim:
        raise RuntimeError("h_t or c_t shape is wrong, please check!")
    if w_xc_x_static is not None:
        if w_xc_x_static["dtype"] not in dtype_all:
            raise RuntimeError("w_xc_x_static supports dtype(%s) only!"%(dtype_str))
        if w_xc_x_static["shape"][1] != n_dim or w_xc_x_static["shape"][0] != 4 * output_dim:
            raise RuntimeError("w_xc_x_static shape is wrong, please check!")


# pylint: disable=locally-disabled,too-many-statements,too-many-branches,unnecessary-lambda,too-many-locals,invalid-name,too-many-arguments
def basic_lstm_cell_v2_schedule(tensor_list, scope_list,
                                operation_list, build_list,
                                product_info, tilling_info, kernel_name):
    """
    do the schedule for the LSTM compute.
    """
    ht = tensor_list["ht"]
    schedule_list = [ht.op]
    s = tvm.create_schedule(schedule_list)

    for key in tensor_list.keys():
        if key in scope_list.keys():
            s[tensor_list[key]].set_scope(scope_list[key])
        if key in operation_list.keys()  and\
                        key != "tensor_h_cont_ub" and key != "ft_cont_ub_true":
            s[tensor_list[key]].emit_insn(s[tensor_list[key]].op.axis[0], operation_list[key])

        if key == "tensor_h_cont_ub" or key == "ft_cont_ub_true":
            s[tensor_list[key]].reorder(tensor_list[key].op.axis[1], tensor_list[key].op.axis[2], tensor_list[key].op.axis[0], tensor_list[key].op.axis[3])
            s[tensor_list[key]].emit_insn(s[tensor_list[key]].op.axis[0], operation_list[key])

    s[tensor_list["tensor_xh_l1_ot"]].reused_by(tensor_list["tensor_xh_l1_it"],
                                                tensor_list["tensor_xh_l1_ft"],
                                                tensor_list["tensor_xh_l1_gt"])
    s[tensor_list["tensor_xh_l0a_ot"]].reused_by(tensor_list["tensor_xh_l0a_it"],
                                                 tensor_list["tensor_xh_l0a_ft"],
                                                 tensor_list["tensor_xh_l0a_gt"])
    # handle matmul info

    mad_pattern = cce.GEMM_MODE
    # split matmul
    symbol = ["it", "ft", "ot", "gt"]
    l1_factor = tilling_info["l1_factor"]
    for t in symbol:
        s[tensor_list["tensor_b_loc_" + t]].reused_by(
            tensor_list["tensor_matmul_l0c_" + t],
            tensor_list["tensor_matmul_result_l0c_" + t])
        tmp = tensor_list["tensor_matmul_l0c_" + t]

        block_n_o, block_n_i = s[tmp].split(tmp.op.axis[1],
                                            factor=tilling_info["block_n_factor"])
        block_out_o, block_out_i = s[tmp].split(tmp.op.axis[0],
                                                factor=tilling_info["block_out_factor"])
        l1_n_outer, l1_n_inner = s[tmp].split(block_n_i,
                                              factor=tilling_info["n_factor"]) #safe
        l1_out_outer, l1_out_inner = s[tmp].split(block_out_i,
                                                  factor=tilling_info["out_factor"])
        l1_k_outer, l1_k_inner = s[tmp].split(tmp.op.reduce_axis[0],
                                              factor=tilling_info["k_factor"])
        l0_n_outer, l0_n_inner = s[tmp].split(l1_n_inner,
                                              factor=tilling_info["n_factor"])
        l0_out_outer, l0_out_inner = s[tmp].split(l1_out_inner,
                                                  factor=tilling_info["out_factor"])

        l0_k_outer, l0_k_inner = s[tmp].split(l1_k_inner,
                                              factor=tilling_info["k_factor"])
        s[tmp].reorder(block_n_o, block_out_o, l1_n_outer, l1_out_outer,
                       l1_k_outer, l0_n_outer, l0_out_outer, l0_k_outer,
                       l0_n_inner, l0_out_inner, tmp.op.axis[2],
                       tmp.op.axis[3], l0_k_inner, tmp.op.reduce_axis[1])
        s[tensor_list["tensor_xh_l0a_" + t]].compute_at(s[tmp], l0_k_outer)
        s[tensor_list["tensor_w_l0b_" + t]].compute_at(s[tmp], l0_k_outer)
        if l1_factor != 1:
            s[tensor_list["tensor_xh_l1_" + t]].split(s[tensor_list["tensor_xh_l1_" + t]].op.axis[1], factor=l1_factor)
        s[tensor_list["tensor_xh_l1_" + t]].compute_at(s[tmp], l1_k_outer)
        s[tensor_list["tensor_w_l1_" + t]].compute_at(s[tmp], l1_k_outer)
        mad_dict = {"mad_pattern": mad_pattern,
                    "k_outer": [l1_k_outer, l0_k_outer],
                    "init_bias": 1}
        s[tmp].emit_insn(l0_n_inner, 'mad', mad_dict)

    # split ht
    ht_0 = ht.shape[0].value
    ht_1 = ht.shape[1].value
    axis_1_o, axis_1_i = s[ht].split(ht.op.axis[1], factor=tilling_info["block_n_factor"])

    axis_1_i_0, axis_1_i_i = s[ht].split(axis_1_i, factor=tilling_info["n_factor"])
    axis_0_o, axis_0_i = s[ht].split(ht.op.axis[0],
                                     factor=tilling_info["block_out_factor"])
    axis_0_o_o, axis_0_o_i = s[ht].split(axis_0_o, factor=1)
    axis_0_i_o, axis_0_i_i = s[ht].split(axis_0_i, factor=tilling_info["out_factor"])

    s[ht].reorder(axis_1_o, axis_0_o_o, axis_0_o_i,
                  axis_1_i_0, axis_0_i_o, axis_1_i_i, axis_0_i_i)

    compute_at_axis = axis_0_o_i
    for t in symbol:
        s[tensor_list["tensor_xh_l1_"+t]].double_buffer()
        s[tensor_list["tensor_w_l1_"+t]].double_buffer()
        s[tensor_list["tensor_b_ub_"+t]].double_buffer()
        if "w_xc_x_static" in build_list:
            s[tensor_list["tensor_static_ub_"+t]].double_buffer()

    s[tensor_list["tensor_cont_ub"]].double_buffer()

    if "tensor_h_ub" in tensor_list:
        s[tensor_list["tensor_h_ub"]].double_buffer()
        s[tensor_list["tensor_c_ub"]].double_buffer()

    s[tensor_list["it_ub"]].double_buffer()
    s[tensor_list["ft_ub"]].double_buffer()
    s[tensor_list["ot_ub"]].double_buffer()
    s[tensor_list["gt_ub"]].double_buffer()

    if (ht_1 // tilling_info["block_n_factor"]) > 1\
            and (ht_0 // tilling_info["block_n_factor"]) % 2 == 0:
        core_outer = s[ht].split(axis_1_o, nparts=2)
        s[ht].bind(core_outer[0], tvm.thread_axis("blockIdx.x"))
    elif (ht_0 // tilling_info["block_out_factor"]) > 1 \
            and (ht_0 // tilling_info["block_out_factor"]) % 2 == 0:
        core_outer = s[ht].split(axis_0_o_o, nparts=2)
        s[ht].bind(core_outer[0], tvm.thread_axis("blockIdx.x"))

    special_symbol = {"tensor_xh_l0a_it", "tensor_xh_l0a_ft",
                      "tensor_xh_l0a_ot", "tensor_xh_l0a_gt",
                      "tensor_w_l0b_it", "tensor_w_l0b_ft",
                      "tensor_w_l0b_ot", "tensor_w_l0b_gt",
                      "tensor_xh_l1_it", "tensor_xh_l1_ft",
                      "tensor_xh_l1_ot", "tensor_xh_l1_gt",
                      "tensor_w_l1_it", "tensor_w_l1_ft",
                      "tensor_w_l1_ot", "tensor_w_l1_gt", "ht"}

    if tensor_list["tensor_cont_ub"].dtype != "float16":
        s[tensor_list["tensor_cont_ub"]].\
            storage_align(tensor_list["tensor_cont_ub"].op.axis[1], 8, 0)
        s[tensor_list["tensor_cont_ub_tmp"]].\
            storage_align(tensor_list["tensor_cont_ub_tmp"].op.axis[1], 8, 0)
    else:
        s[tensor_list["tensor_cont_ub"]].\
            storage_align(tensor_list["tensor_cont_ub"].op.axis[1], 16, 0)
    if "ft_ub_true" in tensor_list:
        ft_key = "ft_ub_true"
    else:
        ft_key = "ft_ub"
    if tensor_list["tensor_cont_ub"].dtype != tensor_list[ft_key].dtype:
        if tensor_list["tensor_cont_ub"].dtype != "float16":
            s[tensor_list["tensor_cont_f_ub_tmp"]].\
                storage_align(tensor_list["tensor_cont_f_ub_tmp"].op.axis[1], 16, 0)
        else:
            s[tensor_list["tensor_cont_f_ub_tmp"]].\
                storage_align(tensor_list["tensor_cont_f_ub_tmp"].op.axis[1], 8, 0)


    for key in tensor_list.keys():
        if key not in special_symbol:
            s[tensor_list[key]].compute_at(s[ht], compute_at_axis)
    ct_gm = s.cache_write(tensor_list["ct"], cce.scope_ubuf)
    s[ct_gm].compute_at(s[ht], compute_at_axis)
    if not product_info["hisi_es"] and\
                    tensor_list["tensor_ct_ub_fake"].dtype == "float16":
        tensor_ct_ub_fake_ub = s.\
            cache_read(tensor_list["tensor_ct_ub_fake"], cce.scope_ubuf,
                       [tensor_list["tensor_ct_ub_fake_true"]])
    else:
        tensor_ct_ub_fake_ub = s.\
            cache_read(tensor_list["tensor_ct_ub_fake"], cce.scope_ubuf,
                       [tensor_list["tensor_ub_two_abs_ct"],
                        tensor_list["tensor_mul_temp_ct"],
                        tensor_list["tensor_sub_temp_ct"]])

    s[tensor_ct_ub_fake_ub].compute_at(s[ht], compute_at_axis)
    s[tensor_list["tensor_ct_ub"]].reused_by(tensor_ct_ub_fake_ub)
    s[tensor_ct_ub_fake_ub].\
        emit_insn(s[tensor_ct_ub_fake_ub].op.axis[0], 'phony_insn')
    s[ht].emit_insn(s[ht].op.axis[2], 'dma_copy')
    s[tensor_list["tensor_ct_ub_fake"]].compute_inline()


    build_symbol = ["x", "cont",
                    "w_xc_x_static", "h", "c", "w_xh", "bias", "ht", "ct"]
    new_build_list = []
    for t in build_symbol:
        if t in build_list.keys():
            new_build_list += [build_list[t]]
    with build_config:
        tvm.build(s, new_build_list, "cce", name=kernel_name)


def get_tilling(x, h_t, product_info):
    block_num = cceconf.CceProductParams().getParams("Device_core_num")
    l0_size = cceconf.CceProductParams().getParams("L0B_Buffer")
    ub_size = cceconf.CceProductParams().getParams("Unified_Buffer")
    ub_limit = ub_size // 4

    x_shape = x["shape"]
    ht_shape = h_t["shape"]
    input_dim = x_shape[0]
    n_dim = x_shape[1]
    out_dim = ht_shape[0]

    tilling_info = {}
    # block tilling
    n_core = True
    if n_dim % block_num == 0:
        n_core = False
        block_n_npart = block_num
    else:
        block_n_npart = 1

    if out_dim % block_num == 0 and n_core:
        block_out_npart = block_num
    else:
        block_out_npart = 1

    block_n_factor = n_dim // block_n_npart
    block_out_factor = out_dim // block_out_npart

    dtype_mad_size = 2

    if product_info["hisi_es"]:
        dtype_size = 2
    else:
        dtype_size = 4

    def _decrement_out_factor(temp_out_factor, block_out):
        div = (block_out // temp_out_factor) + 1
        while block_out % div != 0 and div < block_out:
            div = div + 1
        res = block_out // div

        return res

    def _get_ub_used_size(n_factor, out_factor):
        res = (1*n_factor * out_factor * C0 * C0 + n_factor*C0 + 3*out_factor*C0) * dtype_size
        return res

    while (_get_ub_used_size(block_n_factor, block_out_factor) > ub_limit) and block_n_factor > 1:
        block_n_factor = _decrement_out_factor(block_n_factor, n_dim)

    while (_get_ub_used_size(block_n_factor, block_out_factor) > ub_limit) and block_out_factor > 1:
        block_out_factor = _decrement_out_factor(block_out_factor, out_dim)

    #ub tilling
    k_factor = out_dim + input_dim
    one_mn_size = k_factor * C0 * C0 * dtype_mad_size
    if one_mn_size > l0_size:
        k_factor = 64
        one_mn_size = k_factor * C0 * C0 * dtype_mad_size
    k_factor = 64
    one_mn_size = k_factor * C0 * C0 * dtype_mad_size
    n_factor = min(int(l0_size / one_mn_size), block_n_factor)
    out_factor = min(int(l0_size / one_mn_size), block_out_factor)

    if block_n_factor % n_factor != 0:
        n_factor = _decrement_out_factor(n_factor, block_n_factor)

    if block_out_factor % out_factor != 0:
        out_factor = _decrement_out_factor(out_factor, block_out_factor)

    def gcd(var1, var2):
        var1, var2 = (var1, var2) if var1 >= var2 else (var1, var2)
        while var2:
            var1, var2 = var2, var1 % var2
        return var1
    if input_dim == out_dim and input_dim <= k_factor:
        l1_factor = input_dim
    elif (input_dim > out_dim and input_dim % out_dim ==0) or (input_dim < out_dim and out_dim % input_dim == 0):
        l1_factor = min(input_dim, out_dim)
    else :
        l1_factor = gcd(input_dim, out_dim)
    tilling_info["block_n_factor"] = block_n_factor
    tilling_info["block_out_factor"] = block_out_factor
    tilling_info["k_factor"] = k_factor
    tilling_info["n_factor"] = n_factor
    tilling_info["out_factor"] = out_factor
    tilling_info["l1_factor"] = l1_factor

    return tilling_info


# pylint: disable=locally-disabled,unused-argument,too-many-branches,unnecessary-lambda,too-many-locals,invalid-name,too-many-arguments
@util.check_input_type(dict, dict, (dict, NONETYPE), (dict, NONETYPE),
                       (dict, NONETYPE), dict, dict,
                       (dict, NONETYPE), dict, dict, int, bool, float, bool, int,
                       (int, list, tuple), str)
def basic_lstm_cell_v2(x=None, cont=None, w_xc_x_static=None,
                       h=None, c=None, w_xh=None, bias=None,
                       w_xh_deqscale=None,
                       h_t=None, c_t=None, num_output=0,
                       expose_hidden=False, xh_scale=0.0,
                       sqrt_mode=False, xh_offset=0, w_xh_offset=0,
                       kernel_name="BasicLSTMCellV2"):
    """
    Parameters
    ----------
    x : dict
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        float32,the format can be [FRACTAL_NZ]
    cont : dict
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        float32, the format can be [ND]
    w_xc_x_static : dict
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        float32, the format can be [FRACTAL_NZ]
    h : dict
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        the format can be [FRACTAL_NZ]
    c : dict
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        float32, the format can be [FRACTAL_NZ]
    w_xh : dict
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        the format can be [FRACTAL_NZ]
    bias : dict
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        float32, the format can be [ND]
    w_xh_deqscale : dict
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        float32, the format can be [ND]
    h_t:
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        the format can be [FRACTAL_NZ]
    c_t:
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        the format can be [FRACTAL_NZ]
    num_output : int
        The number of N.
    expose_hidden : bool
        The input_h and input_c is None.
    xh_scale : float
        The xh_scale value.
    sqrt_mode : bool
        The value of sqrt_mode.
    xh_offset: int
        The value of xh_offset.
    w_xh_offset : list
        The value of w_xh_offset.
    kernel_name : str
        cce kernel name, default value == "BasicLSTMCellV2"

    Returns
    -------
    None
    """
    is_hisi_es = False
    is_mini = True
    is_cloud = False
    num_output = ((num_output + 15) // 16)*16
    product_info = {}
    product_info["hisi_es"] = is_hisi_es
    product_info["mini"] = is_mini
    product_info["cloud"] = is_cloud
    x['shape'] = [x['shape'][1], x['shape'][2], x['shape'][3], x['shape'][4]]
    basiclstm_cell_v2_check(x, w_xc_x_static, h, c, w_xh, h_t, c_t,
                            expose_hidden, num_output, product_info)

    tensor_list, scope_list,\
    operation_list, build_list = basic_lstm_cell_v2_compute(x, cont,
                                                            w_xc_x_static,
                                                            w_xh,
                                                            bias, h_t, c_t,
                                                            expose_hidden,
                                                            product_info)

    tilling_info = get_tilling(x, h_t, product_info)

    basic_lstm_cell_v2_schedule(tensor_list, scope_list,
                                operation_list, build_list, product_info, tilling_info, kernel_name)
