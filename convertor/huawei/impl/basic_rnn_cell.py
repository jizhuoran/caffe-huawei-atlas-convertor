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

basic_rnn_cell
"""
# pylint: disable=too-many-lines
import math

from te import tvm
from te.platform import cce_conf
from te.platform.cce_build import build_config
import te.platform.cce_params as cce
import topi
from topi.cce import util

MIN_FP32 = 2**(-126)
NONETYPE = type(None)


# pylint: disable=too-many-instance-attributes
class BasicRNNCell:
    """
        Function: use to store BasicRNNCell base parameters
        Modify : 2020-4-15
    """

    # pylint: disable=too-many-arguments,too-many-locals
    def __init__(self,
                 x,
                 cont,
                 w_xh_x_static,
                 h_0,
                 w_xh,
                 bias_h,
                 w_hh,
                 w_ho,
                 bias_o,
                 o_t,
                 h_t,
                 expose_hidden=False,
                 num_output=0,
                 kernel_name="basicrnn_cell"):
        """
        Init BasicRNNCell base parameters

        Parameters
        ----------
        x: dict
            data of input
        cont: dict
            data of cont
        w_xh_x_static: dict
            data of w_xh_x_static
        h_0: dict
            data of h_0
        w_xh: dict
            data of w_xh
        w_hh: dict
            data of w_hh
        w_ho: dict
            data of w_ho
        bias_h: dict
            data of bias_h
        bias_o: dict
            data of bias_o
        o_t: dict
            data of o_t
        h_t: dict
            data of o_t
        expose_hidden: bool
            if expose hidden state
        num_output: int
            number of output
        kernel_name: str
            the name of the operator

        Returns
        -------
        None
        """
        self.kernel_name = kernel_name
        self.tensor_list1 = {}
        self.tensor_list2 = {}
        self.emit_cmd = {}
        self.scope_list = {}
        self.tanh_ht_tensor = None
        self.tanh_ot_tensor = None
        self.expose_hidden = expose_hidden
        self.num_output = num_output

        self.has_static = True
        if w_xh_x_static is None:
            self.has_static = False

        dtypes = {
            "x": x.get("dtype").lower(),
            "w_xh": w_xh.get("dtype").lower(),
            "w_ho": w_ho.get("dtype").lower(),
            "bias_h": bias_h.get("dtype").lower(),
            "bias_o": bias_o.get("dtype").lower(),
            "o_t": o_t.get("dtype").lower(),
            "h_t": h_t.get("dtype").lower()
        }

        shapes = {
            "x": x.get("shape"),
            "w_xh": w_xh.get("shape"),
            "w_ho": w_ho.get("shape"),
            "bias_h": (math.ceil(float(bias_h.get("shape")[0]) / 16), 16),
            "bias_o": (math.ceil(float(bias_o.get("shape")[0]) / 16), 16),
            "o_t": o_t.get("shape"),
            "h_t": h_t.get("shape")
        }

        datas = {
            "x":
                tvm.placeholder(shapes["x"], name="x", dtype=dtypes["x"]),
            "w_xh":
                tvm.placeholder(
                    shapes["w_xh"], name="w_xh", dtype=dtypes["w_xh"]),
            "w_ho":
                tvm.placeholder(
                    shapes["w_ho"], name="w_ho", dtype=dtypes["w_ho"]),
            "bias_h":
                tvm.placeholder(
                    shapes["bias_h"], name="bias_h", dtype=dtypes["bias_h"]),
            "bias_o":
                tvm.placeholder(
                    shapes["bias_o"], name="bias_o", dtype=dtypes["bias_o"])
        }

        dims = {
            "batch_dim": shapes["x"][1],
            "input_dim": shapes["x"][0],
            "hidden_dim": shapes["w_xh"][1]
        }

        if self.has_static:
            dtypes["w_xh_x_static"] = w_xh_x_static.get("dtype").lower()
            shapes["w_xh_x_static"] = w_xh_x_static.get("shape")
            datas["w_xh_x_static"] = tvm.placeholder(
                shapes["w_xh_x_static"],
                name="w_xh_x_static",
                dtype=dtypes["w_xh_x_static"])

        if self.expose_hidden:
            dtypes["h_0"] = h_0.get("dtype").lower()
            dtypes["cont"] = cont.get("dtype").lower()
            dtypes["w_hh"] = w_hh.get("dtype").lower()

            shapes["cont"] = (math.ceil(float(cont.get("shape")[0]) / 16), 16)
            shapes["h_0"] = h_0.get("shape")
            shapes["w_hh"] = w_hh.get("shape")

            datas["cont"] = tvm.placeholder(
                shapes["cont"], name="cont", dtype=dtypes["cont"])
            datas["h_0"] = tvm.placeholder(
                shapes["h_0"], name="h_0", dtype=dtypes["h_0"])
            datas["w_hh"] = tvm.placeholder(
                shapes["w_hh"], name="w_hh", dtype=dtypes["w_hh"])

        self.check_input_parameters(dtypes, shapes, dims)

        self.shapes = shapes
        self.dtypes = dtypes
        self.datas = datas
        self.dims = dims

    def check_input_parameters(self, dtypes, shapes, dims):
        """
        Check the input parameters

        Parameters
        ----------
        dtypes: dict
            dtypes of inputs
        shapes: dict
            shapes of inputs
        dims: dict
            dims length

        Returns
        -------
        None
        """
        # check dtypes
        util.check_dtype_rule(dtypes["x"].lower(), ("float16",))
        util.check_dtype_rule(dtypes["w_xh"].lower(), ("float16",))
        util.check_dtype_rule(dtypes["w_ho"].lower(), ("float16",))
        util.check_dtype_rule(dtypes["bias_h"].lower(),
                              ("float16", "float32", "int16", "int32"))
        util.check_dtype_rule(dtypes["bias_o"].lower(),
                              ("float16", "float32", "int16", "int32"))
        util.check_dtype_rule(dtypes["o_t"].lower(), ("float16", "float32"))
        util.check_dtype_rule(dtypes["h_t"].lower(), ("float16", "float32"))

        # check shapes
        for key in shapes:
            if key in ("bias_h", "bias_o", "cont"):
                util.check_shape_rule(shapes[key], min_dim=2, max_dim=2)
                util.check_tensor_shape_size(shapes[key])
            else:
                util.check_shape_rule(shapes[key], min_dim=4, max_dim=4)
                util.check_tensor_shape_size(shapes[key])

        batch_dim = dims["batch_dim"]
        input_dim = dims["input_dim"]
        hidden_dim = dims["hidden_dim"]
        check_shapes(shapes["x"], (input_dim, batch_dim, 16, 16))

        check_shapes(shapes["w_xh"], (input_dim, hidden_dim, 16, 16))
        check_shapes(shapes["w_ho"], (hidden_dim, hidden_dim, 16, 16))
        check_shapes(shapes["bias_h"], (hidden_dim, 16))
        check_shapes(shapes["bias_o"], (hidden_dim, 16))
        check_shapes(shapes["o_t"], (hidden_dim, batch_dim, 16, 16))
        check_shapes(shapes["h_t"], (hidden_dim, batch_dim, 16, 16))

        if self.expose_hidden:
            util.check_dtype_rule(
                dtypes["h_0"].lower(), ("float16", "float32"))
            util.check_dtype_rule(dtypes["cont"].lower(), ("float16",))
            util.check_dtype_rule(dtypes["w_hh"].lower(), ("float16",))
            check_shapes(shapes["cont"], (batch_dim, 16))
            check_shapes(shapes["h_0"], (hidden_dim, batch_dim, 16, 16))
            check_shapes(shapes["w_hh"], (hidden_dim, hidden_dim, 16, 16))

        if self.has_static:
            util.check_dtype_rule(
                dtypes["w_xh_x_static"].lower(), ("float16", "float32"))
            check_shapes(
                shapes["w_xh_x_static"], (hidden_dim, batch_dim, 16, 16))

    def get_ht_tensors(self):
        """
        get ht compute tensor list
        Parameters
        ----------
        None

        Returns
        -------
        ht_tensors: dict
            tensor list of ht compute
        """
        ht_tensors = self.tanh_ht_tensor.copy()
        ht_tensors["l0c_wht_xt"] = self.tensor_list1["l0c_wht_xt"]
        ht_tensors["l0c_wht_xt_bias_h"] = self.tensor_list1[
            "l0c_wht_xt_bias_h"]
        ht_tensors["ub_wht_xt_bias_h"] = self.tensor_list1["ub_wht_xt_bias_h"]
        ht_tensors["ub_bias_h"] = self.tensor_list1["ub_bias_h"]
        ht_tensors["l0c_bias_h"] = self.tensor_list1["l0c_bias_h"]

        if self.expose_hidden:
            ht_tensors["l0c_whh_ht"] = self.tensor_list1["l0c_whh_ht"]
            ht_tensors["ub_whh_ht"] = self.tensor_list1["ub_whh_ht"]
            ht_tensors["ub_cont"] = self.tensor_list1["ub_cont"]
            ht_tensors["ub_cont_fp32"] = self.tensor_list1["ub_cont_fp32"]
            ht_tensors["ub_whh_ht_cont"] = self.tensor_list1["ub_whh_ht_cont"]
            ht_tensors["ub_ht_tmp1"] = self.tensor_list1["ub_ht_tmp1"]

        if self.has_static:
            ht_tensors["ub_w_xh_x_static"] = self.tensor_list1[
                "ub_w_xh_x_static"]
            if self.dtypes["w_xh_x_static"] == "float16":
                ht_tensors["ub_w_xh_x_static_fp32"] = self.tensor_list1[
                    "ub_w_xh_x_static_fp32"]
            ht_tensors["ub_ht_tmp2"] = self.tensor_list1["ub_ht_tmp2"]

        if self.dtypes["h_t"] == "float16":
            ht_tensors["ub_ht_fp16"] = self.tensor_list1["ub_ht_fp16"]

        return ht_tensors

    def get_ot_tensors(self):
        """
        get ot compute tensor list
        Parameters
        ----------
        None

        Returns
        -------
        ot_tensors: dict
            tensor list of ot compute
        """
        ot_tensors = self.tanh_ot_tensor.copy()
        ot_tensors["ub_bias_o"] = self.tensor_list2["ub_bias_o"]
        ot_tensors["l0c_bias_o"] = self.tensor_list2["l0c_bias_o"]
        ot_tensors["l0c_who_ht"] = self.tensor_list2["l0c_who_ht"]
        ot_tensors["l0c_who_ht_bias_o"] = self.tensor_list2[
            "l0c_who_ht_bias_o"]
        ot_tensors["ub_who_ht_bias_o"] = self.tensor_list2["ub_who_ht_bias_o"]
        if self.dtypes["o_t"] == "float16":
            ot_tensors["ub_ot_fp16"] = self.tensor_list2["ub_ot_fp16"]

        return ot_tensors

    # pylint: disable=too-many-statements
    def basic_rnn_cell_schedule(self, schedule_list):
        """
        Compute at operate for ot

        Parameters
        ----------
        schedule_list: list
            the output tensors need to schedule

        Returns
        -------
        sch: tvm schedule
            schedule operator
        """
        sch = tvm.create_schedule(schedule_list)

        batch_dim = int(self.datas["x"].shape[1])
        input_dim = int(self.datas["x"].shape[0])
        hidden_dim = int(self.datas["w_ho"].shape[0])

        emit_cmd_list = self.emit_cmd

        tensors = self.tensor_list1.copy()
        tensors.update(self.tensor_list2)
        scope_list = self.scope_list
        for key in scope_list:
            sch[tensors[key]].set_scope(scope_list[key])

        for key in emit_cmd_list:
            tensor = tensors[key]
            op_name = emit_cmd_list[key]

            if key == "ub_whh_ht_cont":
                sch[tensor].reorder(
                    tensor.op.axis[2], tensor.op.axis[1],
                    tensor.op.axis[0], tensor.op.axis[3])
                sch[tensor].emit_insn(sch[tensor].op.axis[1], op_name)
            else:
                sch[tensor].emit_insn(sch[tensor].op.axis[0], op_name)

        tilling_info1 = get_tilling(batch_dim, input_dim, hidden_dim)
        mad_tensors_1 = {
            "l0c": tensors["l0c_wht_xt"],
            "l1_left": tensors["l1_x"],
            "l1_right": tensors["l1_w_xh"],
            "l0a": tensors["l0a_x"],
            "l0b": tensors["l0b_w_xh"],
        }
        matmul_schedule(sch, mad_tensors_1, tilling_info1, True)

        # matmul schedule for l0c_wht_xt
        sch[tensors["l0c_bias_h"]].reused_by(
            tensors["l0c_wht_xt"], tensors["l0c_wht_xt_bias_h"])

        tilling_info2 = get_tilling(batch_dim, hidden_dim, hidden_dim)
        if self.expose_hidden:
            mad_tensors_2 = {
                "l0c": tensors["l0c_whh_ht"],
                "l1_left": tensors["l1_h_0"],
                "l1_right": tensors["l1_w_hh"],
                "l0a": tensors["l0a_h_0"],
                "l0b": tensors["l0b_w_hh"],
            }
            compute_at_axis = matmul_schedule(
                sch, mad_tensors_2, tilling_info2, False)

            if self.dtypes["h_0"] == "float32":
                sch[tensors["ub_h_0"]].compute_at(
                    sch[mad_tensors_2["l0c"]], compute_at_axis)
                sch[tensors["h_0_fp16"]].compute_at(
                    sch[mad_tensors_2["l0c"]], compute_at_axis)

        # split ht
        gm_ht = tensors["gm_ht"]
        m_o, m_i = sch[gm_ht].split(
            gm_ht.op.axis[1], factor=tilling_info2["m_l0"])
        m_o_o, m_o_i = sch[gm_ht].split(
            m_o, factor=tilling_info2["m_l1"])
        n_o, n_i = sch[gm_ht].split(
            gm_ht.op.axis[0], factor=tilling_info2["n_l0"])
        n_o_o, n_o_i = sch[gm_ht].split(
            n_o, factor=tilling_info2["n_l1"])

        sch[gm_ht].reorder(
            m_o_o, m_o_i, n_o_o, n_o_i, n_i, m_i, gm_ht.op.axis[2],
            gm_ht.op.axis[3])

        ht_tensors = self.get_ht_tensors()

        for key in ht_tensors:
            sch[ht_tensors[key]].compute_at(sch[gm_ht], n_o_i)

        sch[gm_ht].emit_insn(sch[gm_ht].op.axis[2], "dma_copy")

        tilling_info2 = get_tilling(batch_dim, hidden_dim, hidden_dim)
        mad_tensors_3 = {
            "l0c": tensors["l0c_who_ht"],
            "l1_left": tensors["l1_ht"],
            "l1_right": tensors["l1_w_ho"],
            "l0a": tensors["l0a_ht"],
            "l0b": tensors["l0b_w_ho"],
        }
        compute_at_axis = matmul_schedule(
            sch, mad_tensors_3, tilling_info2, True)

        if self.dtypes["h_t"] == "float32":
            sch[tensors["ub_ht_new"]].compute_at(
                sch[mad_tensors_3["l0c"]], compute_at_axis)
            sch[tensors["ub_ht_fp16"]].compute_at(
                sch[mad_tensors_3["l0c"]], compute_at_axis)
        sch[gm_ht].compute_at(sch[mad_tensors_3["l0c"]], compute_at_axis)

        sch[tensors["l0c_bias_o"]].reused_by(
            tensors["l0c_who_ht"], tensors["l0c_who_ht_bias_o"])

        tilling_info2 = get_tilling(batch_dim, hidden_dim, hidden_dim)
        # split ot
        gm_ot = tensors["gm_ot"]
        m_o, m_i = sch[gm_ot].split(
            gm_ot.op.axis[1], factor=tilling_info2["m_l0"])
        m_o_o, m_o_i = sch[gm_ot].split(
            m_o, factor=tilling_info2["m_l1"])
        n_o, n_i = sch[gm_ot].split(
            gm_ot.op.axis[0], factor=tilling_info2["n_l0"])
        n_o_o, n_o_i = sch[gm_ot].split(
            n_o, factor=tilling_info2["n_l1"])

        sch[gm_ot].reorder(
            m_o_o, m_o_i, n_o_o, n_o_i, n_i, m_i, gm_ot.op.axis[2],
            gm_ot.op.axis[3])

        ot_tensors = self.get_ot_tensors()

        for key in ot_tensors:
            sch[ot_tensors[key]].compute_at(sch[gm_ot], n_o_i)

        sch[gm_ot].emit_insn(sch[gm_ot].op.axis[2], "dma_copy")

        res_empty = tensors["res_empty"]
        m_o, m_i = sch[res_empty].split(
            res_empty.op.axis[1], factor=tilling_info2["m_l0"])
        m_o_o, m_o_i = sch[res_empty].split(
            m_o, factor=tilling_info2["m_l1"])
        n_o, n_i = sch[res_empty].split(
            res_empty.op.axis[0], factor=tilling_info2["n_l0"])
        n_o_o, n_o_i = sch[res_empty].split(
            n_o, factor=tilling_info2["n_l1"])

        sch[res_empty].reorder(
            m_o_o, m_o_i, n_o_o, n_o_i, n_i, m_i,
            res_empty.op.axis[2], res_empty.op.axis[3])

        sch[gm_ht].compute_at(sch[res_empty], m_o_i)
        sch[gm_ot].compute_at(sch[res_empty], m_o_i)
        sch[res_empty].emit_insn(sch[res_empty].op.axis[2], "phony_insn")

        bind, _ = sch[res_empty].split(
            m_o_o, factor=tilling_info2["block"])
        sch[res_empty].bind(bind, tvm.thread_axis("blockIdx.x"))

        return sch

    # pylint: disable=unnecessary-lambda
    def compute_h_0_whh(self, wht_xt_bias_h):
        """
        calculating h_0_whh

        Parameters
        ----------
        wht_xt_bias_h : TVM tensor

        Returns
        -------
        output tensor
        """
        matmul_res_shape = (self.dims["hidden_dim"], self.dims["batch_dim"],
                            16, 16)
        # Tensor h_0 from GM to L1, L0A
        h_0_fp16 = self.datas["h_0"]
        if self.dtypes["h_0"] == "float32":
            ub_h_0 = tvm.compute(
                (self.dims["hidden_dim"], self.dims["batch_dim"], 16, 16),
                lambda *i: self.datas["h_0"](*i),
                name='ub_h_0')
            self.tensor_list1["ub_h_0"] = ub_h_0
            self.emit_cmd["ub_h_0"] = "dma_copy"
            self.scope_list["ub_h_0"] = cce.scope_ubuf

            h_0_fp16 = tvm.compute(
                ub_h_0.shape,
                lambda *i: topi.cast(ub_h_0(*i), "float16"),
                name="h_0_fp16")
            self.tensor_list1["h_0_fp16"] = h_0_fp16
            self.emit_cmd["h_0_fp16"] = "vector_conv"
            self.scope_list["h_0_fp16"] = cce.scope_ubuf

        l1_h_0 = tvm.compute(
            (self.dims["batch_dim"], self.dims["hidden_dim"], 16, 16),
            lambda i0, i1, i2, i3: h_0_fp16[i1, i0, i2, i3],
            name='l1_h_0')
        self.tensor_list1["l1_h_0"] = l1_h_0
        self.emit_cmd["l1_h_0"] = "dma_copy"
        self.scope_list["l1_h_0"] = cce.scope_cbuf
        l0a_h_0 = tvm.compute(
            l1_h_0.shape, lambda *i: l1_h_0(*i), name='l0a_w_hh')
        self.tensor_list1["l0a_h_0"] = l0a_h_0
        self.emit_cmd["l0a_h_0"] = "dma_copy"
        self.scope_list["l0a_h_0"] = cce.scope_ca

        # Tensor w_hh from GM to L1, L0B
        l1_w_hh = tvm.compute(
            self.datas["w_hh"].shape,
            lambda *i: self.datas["w_hh"](*i),
            name='l1_w_hh')
        self.tensor_list1["l1_w_hh"] = l1_w_hh
        self.emit_cmd["l1_w_hh"] = "dma_copy"
        self.scope_list["l1_w_hh"] = cce.scope_cbuf
        l0b_w_hh = tvm.compute(
            l1_w_hh.shape, lambda *i: l1_w_hh(*i), name='l0b_h_0')
        self.tensor_list1["l0b_w_hh"] = l0b_w_hh
        self.emit_cmd["l0b_w_hh"] = "dma_copy"
        self.scope_list["l0b_w_hh"] = cce.scope_cb

        reduce_kb = tvm.reduce_axis((0, self.dims["hidden_dim"]),
                                    name='reduce_kb')
        reduce_kp = tvm.reduce_axis((0, 16), name='reduce_kp')
        l0c_whh_ht = tvm.compute(
            matmul_res_shape,
            lambda nb, mb, mp, np: tvm.sum(
                (l0a_h_0[mb, reduce_kb, mp, reduce_kp] * l0b_w_hh[
                    reduce_kb, nb, np, reduce_kp]).astype("float32"),
                axis=[reduce_kb, reduce_kp]),
            name='l0c_whh_ht',
            attrs={'input_order': 'positive'})
        self.tensor_list1["l0c_whh_ht"] = l0c_whh_ht
        self.scope_list["l0c_whh_ht"] = cce.scope_cc

        # Move whh_ht to UB
        ub_whh_ht = tvm.compute(
            matmul_res_shape, lambda *i: l0c_whh_ht(*i), name='ub_whh_ht')
        self.tensor_list1["ub_whh_ht"] = ub_whh_ht
        self.emit_cmd["ub_whh_ht"] = "dma_copy"
        self.scope_list["ub_whh_ht"] = cce.scope_ubuf

        # Move cont to UB
        ub_cont = tvm.compute(
            self.datas["cont"].shape,
            lambda *i: self.datas["cont"](*i),
            name='ub_cont')
        self.tensor_list1["ub_cont"] = ub_cont
        self.emit_cmd["ub_cont"] = "dma_copy"
        self.scope_list["ub_cont"] = cce.scope_ubuf

        if ub_cont.dtype == "float16":
            ub_cont_fp32 = tvm.compute(
                ub_cont.shape,
                lambda *i: topi.cast(ub_cont(*i), "float32"),
                name="ub_cont_fp32")
            self.tensor_list1["ub_cont_fp32"] = ub_cont_fp32
            self.emit_cmd["ub_cont_fp32"] = "vector_conv"
            self.scope_list["ub_cont_fp32"] = cce.scope_ubuf
        else:
            ub_cont_fp32 = ub_cont
        ub_whh_ht_cont = tvm.compute(
            matmul_res_shape,
            lambda i0, i1, i2, i3: ub_whh_ht[i0, i1, i2, i3] * ub_cont_fp32[
                i1, i2],
            name='ub_whh_ht_cont')
        self.tensor_list1["ub_whh_ht_cont"] = ub_whh_ht_cont
        self.emit_cmd["ub_whh_ht_cont"] = "vector_mul"
        self.scope_list["ub_whh_ht_cont"] = cce.scope_ubuf

        # Matmul accumulation wht_xt_bias_h + whh_ht_cont
        ub_ht_tmp1 = tvm.compute(
            matmul_res_shape,
            lambda *i: wht_xt_bias_h(*i) + ub_whh_ht_cont(*i),
            name="ub_ht_tmp1")
        self.tensor_list1["ub_ht_tmp1"] = ub_ht_tmp1
        self.emit_cmd["ub_ht_tmp1"] = "vector_add"
        self.scope_list["ub_ht_tmp1"] = cce.scope_ubuf

        return ub_ht_tmp1

    # pylint: disable=unnecessary-lambda,too-many-branches
    def basic_rnn_cell_compute(self):
        """
        calculating data

        Parameters
        ----------
        input_x : TVM tensor
            the placeholder of input_x
        output_y : dict
            dict of output_y, include keys(shape and dtype)
        kernel_name : str
            kernel name, default value is "basicrnn_cell"

        Returns
        -------
        output tensor
        """
        matmul_res_shape = (self.dims["hidden_dim"], self.dims["batch_dim"], 16,
                            16)
        # Tensor x from GM to L1, L0A
        l1_x = tvm.compute(
            (self.dims["batch_dim"], self.dims["input_dim"], 16, 16),
            lambda i0, i1, i2, i3: self.datas["x"][i1, i0, i2, i3],
            name='l1_x')
        self.tensor_list1["l1_x"] = l1_x
        self.emit_cmd["l1_x"] = "dma_copy"
        self.scope_list["l1_x"] = cce.scope_cbuf

        l0a_x = tvm.compute(l1_x.shape, lambda *i: l1_x(*i), name='l0a_x')
        self.tensor_list1["l0a_x"] = l0a_x
        self.emit_cmd["l0a_x"] = "dma_copy"
        self.scope_list["l0a_x"] = cce.scope_ca

        # Tensor w_xh from GM to L1, L0B
        l1_w_xh = tvm.compute(
            self.datas["w_xh"].shape,
            lambda *i: self.datas["w_xh"](*i),
            name='l1_w_xh')
        self.tensor_list1["l1_w_xh"] = l1_w_xh
        self.emit_cmd["l1_w_xh"] = "dma_copy"
        self.scope_list["l1_w_xh"] = cce.scope_cbuf

        l0b_w_xh = tvm.compute(
            l1_w_xh.shape, lambda *i: l1_w_xh(*i), name='l0b_w_xh')
        self.tensor_list1["l0b_w_xh"] = l0b_w_xh
        self.emit_cmd["l0b_w_xh"] = "dma_copy"
        self.scope_list["l0b_w_xh"] = cce.scope_cb

        # Copy bias from GM to UB
        ub_bias_h = tvm.compute(
            self.datas["bias_h"].shape,
            lambda *i: self.datas["bias_h"](*i),
            name='ub_bias_h')
        self.tensor_list1["ub_bias_h"] = ub_bias_h
        self.emit_cmd["ub_bias_h"] = "dma_copy"
        self.scope_list["ub_bias_h"] = cce.scope_ubuf
        if ub_bias_h.dtype == "float16":
            l0c_bias_h = tvm.compute(
                matmul_res_shape,
                lambda i0, i1, i2, i3: ub_bias_h[i0, i3].astype("float32"),
                name='l0c_bias_h')
        else:
            l0c_bias_h = tvm.compute(
                matmul_res_shape,
                lambda i0, i1, i2, i3: ub_bias_h[i0, i3],
                name='l0c_bias_h')
        self.tensor_list1["l0c_bias_h"] = l0c_bias_h
        self.emit_cmd["l0c_bias_h"] = "dma_copy"
        self.scope_list["l0c_bias_h"] = cce.scope_cc

        reduce_kb = tvm.reduce_axis((0, self.dims["input_dim"]),
                                    name='reduce_kb')
        reduce_kp = tvm.reduce_axis((0, 16), name='reduce_kp')

        l0c_wht_xt = tvm.compute(
            matmul_res_shape,
            lambda nb, mb, mp, np: tvm.sum(
                (l0a_x[mb, reduce_kb, mp, reduce_kp] * l0b_w_xh[
                    reduce_kb, nb, np, reduce_kp]).astype("float32"),
                axis=[reduce_kb, reduce_kp]),
            name='l0c_wht_xt',
            attrs={'input_order': 'positive'})
        self.tensor_list1["l0c_wht_xt"] = l0c_wht_xt
        self.scope_list["l0c_wht_xt"] = cce.scope_cc

        # Matmul accumulation wht_xt + bias_h
        l0c_wht_xt_bias_h = tvm.compute(
            matmul_res_shape,
            lambda *i: l0c_bias_h(*i) + l0c_wht_xt(*i),
            name="l0c_wht_xt_bias_h")
        self.tensor_list1["l0c_wht_xt_bias_h"] = l0c_wht_xt_bias_h
        self.emit_cmd["l0c_wht_xt_bias_h"] = "phony_insn"
        self.scope_list["l0c_wht_xt_bias_h"] = cce.scope_cc

        # Move ht to UB
        ub_wht_xt_bias_h = tvm.compute(
            matmul_res_shape,
            lambda *i: l0c_wht_xt_bias_h(*i),
            name='ub_wht_xt_bias_h')
        self.tensor_list1["ub_wht_xt_bias_h"] = ub_wht_xt_bias_h
        self.emit_cmd["ub_wht_xt_bias_h"] = "dma_copy"
        self.scope_list["ub_wht_xt_bias_h"] = cce.scope_ubuf

        if self.expose_hidden:
            ub_ht_tmp1 = self.compute_h_0_whh(ub_wht_xt_bias_h)
        else:
            ub_ht_tmp1 = ub_wht_xt_bias_h

        if self.has_static:
            # Copy bias from GM to UB
            ub_w_xh_x_static = tvm.compute(
                matmul_res_shape,
                lambda *i: self.datas["w_xh_x_static"](*i),
                name='ub_w_xh_x_static')
            self.tensor_list1["ub_w_xh_x_static"] = ub_w_xh_x_static
            self.emit_cmd["ub_w_xh_x_static"] = "dma_copy"
            self.scope_list["ub_w_xh_x_static"] = cce.scope_ubuf

            if ub_w_xh_x_static.dtype == "float16":
                ub_w_xh_x_static_fp32 = tvm.compute(
                    ub_w_xh_x_static.shape,
                    lambda *i: topi.cast(ub_w_xh_x_static(*i), "float32"),
                    name="ub_w_xh_x_static_fp32")
                self.tensor_list1[
                    "ub_w_xh_x_static_fp32"] = ub_w_xh_x_static_fp32
                self.emit_cmd["ub_w_xh_x_static_fp32"] = "vector_conv"
                self.scope_list["ub_w_xh_x_static_fp32"] = cce.scope_ubuf
            else:
                ub_w_xh_x_static_fp32 = ub_w_xh_x_static
            ub_ht_tmp2 = tvm.compute(
                matmul_res_shape,
                lambda *i: ub_ht_tmp1(*i) + ub_w_xh_x_static_fp32(*i),
                name="ub_ht_tmp2")
            self.tensor_list1["ub_ht_tmp2"] = ub_ht_tmp2
            self.emit_cmd["ub_ht_tmp2"] = "vector_add"
            self.scope_list["ub_ht_tmp2"] = cce.scope_ubuf
        else:
            ub_ht_tmp2 = ub_ht_tmp1

        tanh_ht_tensor, ht_tanh_op, ht_tanh_scope = tanh_compute(
            ub_ht_tmp2.shape, ub_ht_tmp2, "ht")

        if self.dtypes["h_t"] == "float16":
            ub_ht_fp16 = tvm.compute(
                matmul_res_shape,
                lambda *i: topi.cast(
                    tanh_ht_tensor["ub_tanh_ht"](*i), "float16"),
                name='ub_ht_fp16')
            tanh_ht_tensor["ub_ht_fp16"] = ub_ht_fp16
            ht_tanh_op["ub_ht_fp16"] = "vector_conv"
            ht_tanh_scope["ub_ht_fp16"] = cce.scope_ubuf
            ub_ht = ub_ht_fp16
        else:
            ub_ht = self.tensor_list1["ub_tanh_ht"]

        self.tanh_ht_tensor = tanh_ht_tensor
        self.scope_list.update(ht_tanh_scope)
        self.tensor_list1.update(tanh_ht_tensor)
        self.emit_cmd.update(ht_tanh_op)

        gm_ht = tvm.compute(
            matmul_res_shape, lambda *i: ub_ht(*i), name='gm_ht')
        self.tensor_list1["gm_ht"] = gm_ht
        self.scope_list["gm_ht"] = cce.scope_gm

        # Tensor ht from GM to L1, L0A
        if gm_ht.dtype == "float32":
            ub_ht_new = tvm.compute(
                matmul_res_shape, lambda *i: gm_ht(*i), name='ub_ht_new')
            self.tensor_list2["ub_ht_new"] = ub_ht_new
            self.emit_cmd["ub_ht_new"] = "dma_copy"
            self.scope_list["ub_ht_new"] = cce.scope_ubuf
            ub_ht_fp16 = tvm.compute(
                ub_ht_new.shape,
                lambda *i: topi.cast(ub_ht_new(*i), "float16"),
                name="ub_ht_fp16")
            self.tensor_list2["ub_ht_fp16"] = ub_ht_fp16
            self.emit_cmd["ub_ht_fp16"] = "vector_conv"
            self.scope_list["ub_ht_fp16"] = cce.scope_ubuf
        else:
            ub_ht_fp16 = gm_ht

        l1_ht = tvm.compute(
            (self.dims["batch_dim"], self.dims["hidden_dim"], 16, 16),
            lambda i0, i1, i2, i3: ub_ht_fp16[i1, i0, i2, i3],
            name='l1_ht')
        self.tensor_list2["l1_ht"] = l1_ht
        self.emit_cmd["l1_ht"] = "dma_copy"
        self.scope_list["l1_ht"] = cce.scope_cbuf

        l0a_ht = tvm.compute(l1_ht.shape, lambda *i: l1_ht(*i), name='l0a_ht')
        self.tensor_list2["l0a_ht"] = l0a_ht
        self.emit_cmd["l0a_ht"] = "dma_copy"
        self.scope_list["l0a_ht"] = cce.scope_ca

        # Tensor w_ho from ub to L1, L0B
        l1_w_ho = tvm.compute(
            self.datas["w_ho"].shape,
            lambda *i: self.datas["w_ho"](*i),
            name='l1_w_ho')
        self.tensor_list2["l1_w_ho"] = l1_w_ho
        self.emit_cmd["l1_w_ho"] = "dma_copy"
        self.scope_list["l1_w_ho"] = cce.scope_cbuf
        l0b_w_ho = tvm.compute(
            l1_w_ho.shape, lambda *i: l1_w_ho(*i), name='l0b_w_ho')
        self.tensor_list2["l0b_w_ho"] = l0b_w_ho
        self.emit_cmd["l0b_w_ho"] = "dma_copy"
        self.scope_list["l0b_w_ho"] = cce.scope_cb

        # Copy bias from GM to UB
        ub_bias_o = tvm.compute(
            self.datas["bias_o"].shape,
            lambda *i: self.datas["bias_o"](*i),
            name='ub_bias_o')
        self.tensor_list2["ub_bias_o"] = ub_bias_o
        self.emit_cmd["ub_bias_o"] = "dma_copy"
        self.scope_list["ub_bias_o"] = cce.scope_ubuf
        if ub_bias_o.dtype == "float16":
            l0c_bias_o = tvm.compute(
                matmul_res_shape,
                lambda i0, i1, i2, i3: ub_bias_o[i0, i3].astype("float32"),
                name='l0c_bias_o')
        else:
            l0c_bias_o = tvm.compute(
                matmul_res_shape,
                lambda i0, i1, i2, i3: ub_bias_o[i0, i3],
                name='l0c_bias_o')
        self.tensor_list2["l0c_bias_o"] = l0c_bias_o
        self.emit_cmd["l0c_bias_o"] = "dma_copy"
        self.scope_list["l0c_bias_o"] = cce.scope_cc

        reduce_kb = tvm.reduce_axis((0, self.dims["hidden_dim"]),
                                    name='reduce_kb')
        reduce_kp = tvm.reduce_axis((0, 16), name='reduce_kp')

        l0c_who_ht = tvm.compute(
            matmul_res_shape,
            lambda nb, mb, mp, np: tvm.sum(
                (l0a_ht[mb, reduce_kb, mp, reduce_kp] * l0b_w_ho[
                    reduce_kb, nb, np, reduce_kp]).astype("float32"),
                axis=[reduce_kb, reduce_kp]),
            name='l0c_who_ht',
            attrs={'input_order': 'positive'})
        self.tensor_list2["l0c_who_ht"] = l0c_who_ht
        self.scope_list["l0c_who_ht"] = cce.scope_cc

        # Matmul accumulation whh_ht + bias_o
        l0c_who_ht_bias_o = tvm.compute(
            matmul_res_shape,
            lambda *i: l0c_bias_o(*i) + l0c_who_ht(*i),
            name="l0c_who_ht_bias_o")
        self.tensor_list2["l0c_who_ht_bias_o"] = l0c_who_ht_bias_o
        self.emit_cmd["l0c_who_ht_bias_o"] = "phony_insn"
        self.scope_list["l0c_who_ht_bias_o"] = cce.scope_cc

        # Move ub_whh_ht_bias_o to UB
        ub_who_ht_bias_o = tvm.compute(
            matmul_res_shape,
            lambda *i: l0c_who_ht_bias_o(*i),
            name='ub_who_ht_bias_o')
        self.tensor_list2["ub_who_ht_bias_o"] = ub_who_ht_bias_o
        self.emit_cmd["ub_who_ht_bias_o"] = "dma_copy"
        self.scope_list["ub_who_ht_bias_o"] = cce.scope_ubuf

        tanh_ot_tensor, tanh_ot_operator, tanh_ot_scope = tanh_compute(
            ub_who_ht_bias_o.shape, ub_who_ht_bias_o, "ot")

        if self.dtypes["o_t"] == "float16":
            ub_ot_fp16 = tvm.compute(
                matmul_res_shape,
                lambda *i: topi.cast(
                    tanh_ot_tensor["ub_tanh_ot"](*i), "float16"),
                name='ub_ot_fp16')
            tanh_ot_tensor["ub_ot_fp16"] = ub_ot_fp16
            tanh_ot_operator["ub_ot_fp16"] = "vector_conv"
            tanh_ot_scope["ub_ot_fp16"] = cce.scope_ubuf
            ub_ot = ub_ot_fp16
        else:
            ub_ot = tanh_ot_tensor["ub_tanh_ot"]

        self.tanh_ot_tensor = tanh_ot_tensor
        self.scope_list.update(tanh_ot_scope)
        self.tensor_list2.update(tanh_ot_tensor)
        self.emit_cmd.update(tanh_ot_operator)

        gm_ot = tvm.compute(
            matmul_res_shape, lambda *i: ub_ot(*i), name='gm_ot')
        self.tensor_list2["gm_ot"] = gm_ot
        self.emit_cmd["gm_ot"] = "dma_copy"
        self.scope_list["gm_ot"] = cce.scope_gm

        res_empty = tvm.compute(
            matmul_res_shape,
            lambda *i: gm_ot(*i) * gm_ht(*i),
            name='res_empty')
        self.tensor_list2["res_empty"] = res_empty
        self.emit_cmd["res_empty"] = "phony_insn"
        self.scope_list["res_empty"] = cce.scope_ubuf

        schedule_list = [res_empty.op]
        sch = self.basic_rnn_cell_schedule(schedule_list)
        if self.has_static:
            build_list = (self.datas["x"], self.datas["cont"],
                          self.datas["w_xh_x_static"],
                          self.datas["h_0"], self.datas["w_xh"],
                          self.datas["bias_h"], self.datas["w_hh"],
                          self.datas["w_ho"], self.datas["bias_o"],
                          gm_ot, gm_ht)
        else:
            if self.expose_hidden:
                build_list = (self.datas["x"], self.datas["cont"],
                              self.datas["h_0"], self.datas["w_xh"],
                              self.datas["bias_h"], self.datas["w_hh"],
                              self.datas["w_ho"], self.datas["bias_o"],
                              gm_ot, gm_ht)
            else:
                build_list = (self.datas["x"], self.datas["w_xh"],
                              self.datas["bias_h"], self.datas["w_ho"],
                              self.datas["bias_o"], gm_ot, gm_ht)

        with build_config:
            tvm.build(sch, build_list, "cce", name=self.kernel_name)

    def set_scedule_scope(self, sch):
        """
        Compute at operate for ot

        Parameters
        ----------
        sch: tvm schedule
            schedule operator

        Returns
        -------
        None
        """
        tensor_list = self.tensor_list1 + self.tensor_list2
        scope_list = self.scope_list
        for key in scope_list:
            sch[tensor_list[key]].set_scope(scope_list[key])


# pylint: disable=too-many-locals
def newton_iteration(shape, tensor_x_rec, tensor_x, symbol, iter_num):
    """
    the function of newton_iteration
    Parameters
    ----------
    shape: tensor shape
    tensor_x_rec: tensor
    tensor_x: tensor
    symbol: tensor symbol

    Returns
    -------
    tensor_list: dict
    scope_list: dict
    emit_list: dict
    """
    dtype_c = tensor_x_rec.dtype
    num_two = tvm.const(2, dtype=dtype_c)
    neg_one = tvm.const(-1, dtype=dtype_c)
    tmp = tensor_x_rec

    tensor_list = {}
    scope_list = {}
    emit_list = {}
    tmp_mul = None
    tmp_neg = None
    tmp_add = None
    for index in range(0, iter_num):
        key = "tmp_mul_" + symbol + str(index)
        tmp_mul = tvm.compute(
            shape, lambda *i: tensor_x(*i) * tmp(*i), name=key)
        tensor_list[key] = tmp_mul
        scope_list[key] = cce.scope_ubuf
        emit_list[key] = "vector_mul"

        key = "tmp_neg_" + symbol + str(index)
        tmp_neg = tvm.compute(
            shape, lambda *i: tmp_mul(*i) * neg_one, name=key)
        tensor_list[key] = tmp_neg
        scope_list[key] = cce.scope_ubuf
        emit_list[key] = "vector_muls"

        key = "tmp_add_" + symbol + str(index)
        tmp_add = tvm.compute(
            shape, lambda *i: tmp_neg(*i) + num_two, name=key)
        tensor_list[key] = tmp_add
        scope_list[key] = cce.scope_ubuf
        emit_list[key] = "vector_adds"

        key = "tmp_" + symbol + str(index)
        tmp = tvm.compute(shape, lambda *i: tmp_add(*i) * tmp(*i), name=key)
        tensor_list[key] = tmp
        scope_list[key] = cce.scope_ubuf
        emit_list[key] = "vector_mul"

    return tensor_list, scope_list, emit_list


# pylint: disable=too-many-locals,too-many-branches
def get_tilling(m_dim, k_dim, n_dim):
    """
    get tilling parameters
    Parameters
    ----------
    k_dim: int
        k axis length

    Returns
    -------
    tilling_info: dict
        tilling parameters
    """
    block_num = cce_conf.get_soc_spec(cce_conf.CORE_NUM)
    l1_size = cce_conf.get_soc_spec(cce_conf.L1_SIZE)
    l1_limit = l1_size // 4 // 2
    ub_size = cce_conf.get_soc_spec(cce_conf.UB_SIZE)
    ub_limit = ub_size // 2

    tilling_info = {}
    m_l0_factor = 1
    block_factor = 1
    if m_dim > block_num:
        block_factor = m_dim // block_num

    n_l0_factor = n_dim
    k_l0_factor = k_dim
    c_0 = 16
    dtype_mad_size = 4
    fracz_size = c_0 * c_0 * dtype_mad_size
    one_mn_size = k_l0_factor * fracz_size
    if k_l0_factor > 32:
        k_l0_factor = 32
        one_mn_size = k_l0_factor * fracz_size
    ub_used = m_l0_factor * n_l0_factor * one_mn_size
    while ub_used > ub_limit:
        if m_l0_factor > 1:
            m_l0_factor -= 1
        else:
            n_l0_factor -= 1
        ub_used = m_l0_factor * n_l0_factor * one_mn_size

    if m_l0_factor > 1:
        while m_dim % m_l0_factor != 0:
            m_l0_factor -= 1

    if n_l0_factor > 1:
        while n_dim % n_l0_factor != 0:
            n_l0_factor -= 1

    if l1_limit > one_mn_size * m_l0_factor:
        m_l1_factor = 1
    else:
        m_l1_factor = l1_limit // (one_mn_size * m_l0_factor)

    if l1_limit > one_mn_size * n_l0_factor:
        n_l1_factor = 1
    else:
        n_l1_factor = l1_limit // (one_mn_size * n_l0_factor)

    tilling_info["block"] = block_factor
    tilling_info["m_l1"] = m_l1_factor
    tilling_info["n_l1"] = n_l1_factor
    tilling_info["k_l1"] = 1
    tilling_info["m_l0"] = 1
    tilling_info["n_l0"] = n_l0_factor
    tilling_info["k_l0"] = k_l0_factor

    return tilling_info


# pylint: disable=too-many-statements
def tanh_compute(shape, input_x, symbol):
    """
    the function of tanh
    Parameters
    ----------
    shape : tensor shape
    input_x : tensor
    symbol : tensor symbol
    Returns
    -------
    """
    res = {}
    operation = {}
    scope = {}

    dtype_x = input_x.dtype
    const_one = tvm.const(1, dtype=dtype_x)
    const_neg_two = tvm.const(-2, dtype=dtype_x)
    const_fp32_min = tvm.const(2 ** (-126), dtype=dtype_x)

    key = "input_abs_" + symbol
    input_abs = tvm.compute(
        shape, lambda *i: tvm.abs(input_x(*i)), name=key)
    res[key] = input_abs
    operation[key] = "vector_abs"
    scope[key] = cce.scope_ubuf

    key = "power_val_" + symbol
    power_val = tvm.compute(
        shape, lambda *i: input_abs(*i) * const_neg_two, name=key)
    res[key] = power_val
    operation[key] = "vector_muls"
    scope[key] = cce.scope_ubuf

    if dtype_x == "float32":
        key = "exp_val_fp16_" + symbol
        exp_val_fp16 = tvm.compute(
            shape, lambda *i: topi.cast(power_val(*i), "float16"), name=key)
        res[key] = exp_val_fp16
        operation[key] = "vector_conv"
        scope[key] = cce.scope_ubuf

        key = "exp_val_" + symbol
        exp_val = tvm.compute(
            shape, lambda *i: tvm.exp(exp_val_fp16(*i)), name=key)
        res[key] = exp_val
        operation[key] = "vector_exp"
        scope[key] = cce.scope_ubuf

        key = "exp_val_fp32_" + symbol
        exp_val_fp32 = tvm.compute(
            shape, lambda *i: topi.cast(exp_val(*i), "float32"), name=key)
        res[key] = exp_val_fp32
        operation[key] = "vector_conv"
        scope[key] = cce.scope_ubuf

        exp_val_true = exp_val_fp32
    else:
        key = "exp_val_" + symbol
        exp_val = tvm.compute(
            shape, lambda *i: tvm.exp(power_val(*i)), name=key)
        res[key] = exp_val
        operation[key] = "vector_exp"
        scope[key] = cce.scope_ubuf
        exp_val_true = exp_val

    key = "up_val_tmp_" + symbol
    up_val_tmp = tvm.compute(
        shape, lambda *i: exp_val_true(*i) * input_x(*i), name=key)
    res[key] = up_val_tmp
    operation[key] = "vector_mul"
    scope[key] = cce.scope_ubuf

    key = "up_val_" + symbol
    up_val = tvm.compute(
        shape, lambda *i: input_x(*i) - up_val_tmp(*i), name=key)
    res[key] = up_val
    operation[key] = "vector_sub"
    scope[key] = cce.scope_ubuf

    key = "input_tmp_" + symbol
    input_tmp = tvm.compute(
        shape, lambda *i: input_abs(*i) + const_fp32_min, name=key)
    res[key] = input_tmp
    operation[key] = "vector_adds"
    scope[key] = cce.scope_ubuf

    key = "down_val_tmp_" + symbol
    down_val_tmp = tvm.compute(
        shape, lambda *i: exp_val_true(*i) + const_one, name=key)
    res[key] = down_val_tmp
    operation[key] = "vector_adds"
    scope[key] = cce.scope_ubuf

    key = "down_val_" + symbol
    down_val = tvm.compute(
        shape, lambda *i: down_val_tmp(*i) * input_tmp(*i), name=key)
    res[key] = down_val
    operation[key] = "vector_mul"
    scope[key] = cce.scope_ubuf

    ub_rec = tvm.compute(
        shape, lambda *i: const_one / down_val(*i), name="ub_rec_" + symbol)
    res["ub_rec_" + symbol] = ub_rec
    operation["ub_rec_" + symbol] = "vector_rec"
    scope["ub_rec_" + symbol] = cce.scope_ubuf

    iter_num = 1
    tensor_list, scope_list, emit_list = newton_iteration(
        shape, ub_rec, down_val, symbol, iter_num)
    res.update(tensor_list)
    operation.update(emit_list)
    scope.update(scope_list)

    newton_res = tensor_list["tmp_" + symbol + str(iter_num - 1)]

    ub_tanh = tvm.compute(
        shape,
        lambda *i: up_val(*i) * newton_res(*i),
        name="ub_tanh_" + symbol)
    res["ub_tanh_" + symbol] = ub_tanh
    operation["ub_tanh_" + symbol] = "vector_mul"
    scope["ub_tanh_" + symbol] = cce.scope_ubuf

    return res, operation, scope


def matmul_schedule(sch, tensors, tilling_info, init_bias):
    """
    matmul schedule

    Parameters
    ----------
    tensors : dict
        tensors need to schedule
    tilling_info: dict
        tilling parameters
    init_bias: bool
        if init matmul bias

    Returns
    -------
    None
    """
    mad_pattern = cce.GEMM_MODE
    # matmul schedule for l0c_whh_ht
    l0c = tensors["l0c"]
    l1_left = tensors["l1_left"]
    l1_right = tensors["l1_right"]
    l0a = tensors["l0a"]
    l0b = tensors["l0b"]

    m_o, m_i = sch[l0c].split(
        l0c.op.axis[1], factor=tilling_info["m_l0"])
    m_o_o, m_o_i = sch[l0c].split(
        m_o, factor=tilling_info["m_l1"])
    n_o, n_i = sch[l0c].split(
        l0c.op.axis[0], factor=tilling_info["n_l0"])
    n_o_o, n_o_i = sch[l0c].split(
        n_o, factor=tilling_info["n_l1"])
    l1_k_outer, l1_k_inner = sch[l0c].split(
        l0c.op.reduce_axis[0], factor=tilling_info["k_l0"])
    l0_k_outer, l0_k_inner = sch[l0c].split(
        l1_k_inner, factor=tilling_info["k_l0"])

    sch[l0c].reorder(
        m_o_o, m_o_i, n_o_o, n_o_i, l1_k_outer, l0_k_outer,
        n_i, m_i, l0c.op.axis[2], l0c.op.axis[3], l0_k_inner,
        l0c.op.reduce_axis[1])

    sch[l0a].compute_at(sch[l0c], l0_k_outer)
    sch[l0b].compute_at(sch[l0c], l0_k_outer)
    sch[l1_left].compute_at(sch[l0c], l1_k_outer)
    sch[l1_right].compute_at(sch[l0c], l1_k_outer)

    sch[l0a].double_buffer()
    sch[l0b].double_buffer()
    sch[l1_left].double_buffer()
    sch[l1_right].double_buffer()

    if init_bias:
        mad_dict = {"mad_pattern": mad_pattern, "init_bias": 1}
    else:
        mad_dict = {
            "mad_pattern": mad_pattern,
            "k_outer": [l1_k_outer, l0_k_outer]
        }
    sch[l0c].emit_insn(n_i, 'mad', mad_dict)

    compute_at_axis = l1_k_outer

    return compute_at_axis

def check_shapes(input_shape, check_shape):
    """
    check input tensor shapes

    Parameters
    ----------
    input_shape: tuple
        input shape
    check_shape: tuple
        the shape expected

    Returns
    -------
    None
    """
    if len(input_shape) != len(check_shape):
        raise RuntimeError("the dim number of input shape is error")

    for index, dim in enumerate(check_shape):
        if input_shape[index] != dim:
            raise RuntimeError("the dim(%d) is error" % (index))


@util.check_input_type(dict, (dict, NONETYPE), (dict, NONETYPE),
                       (dict, NONETYPE), dict, dict, (dict, NONETYPE),
                       dict, dict, dict, dict, bool, int, str)
# pylint: disable=too-many-arguments, invalid-name
def basic_rnn_cell(x,
                   cont,
                   w_xh_x_static,
                   h_0,
                   w_xh,
                   bias_h,
                   w_hh,
                   w_ho,
                   bias_o,
                   o_t,
                   h_t,
                   expose_hidden=False,
                   num_output=0,
                   kernel_name="basicrnn_cell"):
    """
    calculating data

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "basicrnn_cell"

    Returns
    -------
    None
    """
    rnn_cell = BasicRNNCell(x, cont, w_xh_x_static, h_0, w_xh, bias_h,
                            w_hh, w_ho, bias_o, o_t, h_t, expose_hidden,
                            num_output, kernel_name)

    rnn_cell.basic_rnn_cell_compute()
