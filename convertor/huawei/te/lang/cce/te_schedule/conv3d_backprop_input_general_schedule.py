"""
conv3d backprop input general schedule.
"""

# !/usr/bin/python
# -*- coding: UTF-8 -*-
from te.platform import scope_ubuf
from te.platform import scope_ca
from te.platform import scope_cb
from te.platform import scope_cc
from te.platform import scope_cbuf
from te.domain.tiling.tiling_query import tiling_query
from te.platform import CUBE_MKN
from te.platform import build_config
from te import tvm

NUM_3 = 3


def print_ir_conv(process, sch):
    """
    print ir for input sch

    Parameter:
    --------------------------------------------------------------
    :param process: tag
    :param sch: schedule
    :return: IR process
    ---------------------------------------------------------------
    """
    with build_config:
        start = process + " IR start"
        end = process + " IR end\n"
        print(start)
        sch = sch.normalize()
        bounds = tvm.schedule.InferBound(sch)
        stmt = tvm.schedule.ScheduleOps(sch, bounds, True)
        print(stmt)
        print(end)


def general_schedule(tensor, sch_list):  # pylint:disable=R0914, R0915
    """
    auto_schedule for cce AI-CORE.
    For now, only one convolution operation is supported.

    Parameters
    ----------
    sch_list: use sch_list[0] to return conv schedule

    Returns
    -------
    True for sucess, False for no schedule
    """
    # =======================#
    def _cub_process():
        cub_tiling_mc_factor_m0 = cub_tiling_mc_factor * cub_tiling_m0
        cddr_n_outer, cddr_n_for_cub = \
            sch[c_ddr].split(n_after_multicore, factor=cub_tiling_nc_factor)
        cddr_m_outer, cddr_m_for_cub =\
            sch[c_ddr].split(m_after_multicore, factor=cub_tiling_mc_factor_m0)
        sch[c_ddr].reorder(cddr_n_outer, cddr_m_outer,
                           cddr_n_for_cub, cddr_m_for_cub)
        return cddr_n_outer, cddr_m_outer, cddr_n_for_cub

    def _l0c_procees():
        cddr_n_outer_outer, cddr_n_outer_inner = \
            sch[c_ddr].split(cddr_n_outer,
                             factor=cl0_tiling_nc // cub_tiling_nc_factor)
        cddr_m_outer_outer, cddr_m_outer_inner = \
            sch[c_ddr].split(cddr_m_outer,
                             factor=cl0_tiling_mc // cub_tiling_mc_factor)
        sch[c_ddr].reorder(cddr_n_outer_outer, cddr_m_outer_outer,
                           cddr_n_outer_inner, cddr_m_outer_inner)
        al1_at_ddr_m_outer, al1_at_ddr_m_inner = \
            sch[c_ddr].split(cddr_m_outer_outer, factor=al1_tilling_m)
        bl1_at_ddr_n_outer, bl1_at_ddr_n_inner = \
            sch[c_ddr].split(cddr_n_outer_outer, factor=bl1_tilling_n)
        batch_outer, batch_inner = \
            sch[c_ddr].split(batch_after_multicore, factor=1)
        c_ddr_deep_outer, c_ddr_deep_inner = \
            sch[c_ddr].split(d_after_multicore, factor=cddr_deep_factor)
        sch[c_ddr].reorder(
            c_ddr_deep_outer, al1_at_ddr_m_outer, batch_inner,
            bl1_at_ddr_n_outer, bl1_at_ddr_n_inner, al1_at_ddr_m_inner,
            c_ddr_deep_inner)
        col_at_ddr_aixs = al1_at_ddr_m_inner
        return (
            batch_outer, al1_at_ddr_m_outer, batch_inner, bl1_at_ddr_n_outer,
            col_at_ddr_aixs, cddr_m_outer_inner, cddr_m_outer_outer,
            c_ddr_deep_outer, c_ddr_deep_inner)

    def _l0a_and_l0b_process():
        c_col_n_outer, c_col_n_inner = \
            sch[c_col].split(c_col.op.axis[2], factor=bl0_tiling_nb)
        c_col_m_outer, c_col_m_inner = \
            sch[c_col].split(c_col.op.axis[3], factor=al0_m_factor)
        c_col_deep_outer, c_col_deep_inner = sch[c_col].split(c_col.op.axis[1],
                                                              factor=1)
        if kd_reduce_flag:  # pylint:disable=R1705
            reduce_axis_kd, reduce_axis_k1, reduce_axis_k0 =\
                c_col.op.reduce_axis
            reduce_axis_kd_outer, reduce_axis_kd_inner = \
                         sch[c_col].split(reduce_axis_kd, factor=kd_factor)
            c_col_k_outer, c_col_k_inner =\
                sch[c_col].split(reduce_axis_k1,
                                 factor=al0_tiling_ka)
            sch[c_col].reorder(c_col_k_outer, c_col_m_outer,
                               reduce_axis_kd_outer, c_col_n_outer,
                               c_col_deep_outer, reduce_axis_kd_inner,
                               c_col_deep_inner, c_col_n_inner, c_col_m_inner,
                               c_col.op.axis[4], c_col_k_inner, reduce_axis_k0)
            return (
                c_col_deep_outer, c_col_deep_inner, c_col_k_outer,
                c_col_m_outer, c_col_n_outer, reduce_axis_kd,
                reduce_axis_kd_outer, reduce_axis_kd_inner)
        else:
            reduce_axis_k1, reduce_axis_k0 = c_col.op.reduce_axis
            c_col_k_outer, c_col_k_inner =\
                sch[c_col].split(reduce_axis_k1,
                                 factor=al0_tiling_ka)
            sch[c_col].reorder(c_col_k_outer, c_col_m_outer, c_col_n_outer,
                               c_col_deep_outer, c_col_deep_inner,
                               c_col_n_inner, c_col_m_inner, c_col.op.axis[4],
                               c_col_k_inner, reduce_axis_k0)
            return (
                c_col_deep_outer, c_col_deep_inner,
                c_col_k_outer, c_col_m_outer,
                c_col_n_outer)

    def _al1_and_bl1_process():
        if kd_reduce_flag:
            reduce_axis_kd_outer_outer, _ =\
                sch[c_col].split(reduce_axis_kd_outer, kd_tiling_l1_factor)
        if k_al1_factor > k_bl1_factor:
            factor_outer, factor_inner =\
                k_al1_factor//k_bl1_factor, k_bl1_factor
            c_col_k_outer_outer, c_col_k_outer_inner = \
                sch[c_col].split(c_col_k_outer, factor=factor_inner)
            c_col_k_outer_outer_outer, c_col_k_outer_outer_inner = \
                sch[c_col].split(c_col_k_outer_outer, factor=factor_outer)
            bl1_at_l0c_axis, al1_at_l0c_axis =\
                c_col_k_outer_outer_inner, c_col_k_outer_outer_outer
            if kd_reduce_flag:
                sch[c_col].reorder(
                    c_col_k_outer_outer_outer,
                    reduce_axis_kd_outer_outer,
                    c_col_k_outer_outer_inner,
                    c_col_k_outer_inner, c_col_m_outer)
            else:
                sch[c_col].reorder(
                    c_col_k_outer_outer_outer, c_col_k_outer_outer_inner,
                    c_col_k_outer_inner, c_col_m_outer)

        else:
            factor_outer, factor_inner =\
                k_bl1_factor//k_al1_factor, k_al1_factor
            c_col_k_outer_outer, c_col_k_outer_inner = \
                sch[c_col].split(c_col_k_outer, factor=factor_inner)
            c_col_k_outer_outer_outer, c_col_k_outer_outer_inner = \
                sch[c_col].split(c_col_k_outer_outer, factor=factor_outer)
            bl1_at_l0c_axis, al1_at_l0c_axis =\
                c_col_k_outer_outer_outer, c_col_k_outer_outer_inner
            if kd_reduce_flag:
                sch[c_col].reorder(
                    reduce_axis_kd_outer_outer,
                    c_col_k_outer_outer_outer,
                    c_col_k_outer_outer_inner,
                    c_col_k_outer_inner,
                    c_col_m_outer)
            else:
                sch[c_col].reorder(
                    c_col_k_outer_outer_outer, c_col_k_outer_outer_inner,
                    c_col_k_outer_inner, c_col_m_outer)
        reduce_axis_serial = \
            [c_col_k_outer_outer_outer, c_col_k_outer_outer_inner,
             c_col_k_outer_inner]
        return reduce_axis_serial, bl1_at_l0c_axis, al1_at_l0c_axis

    def _aub_process():
        aub_tiling_k, aub_tiling_m, _, _ = tiling.get("AUB_shape")
        aub_tiling_k_factor, aub_tiling_m_factor = \
            aub_tiling_k // (kernel_h * kernel_w * 16), aub_tiling_m
        _, _, _, _, aub_w, _ = list(i.value for i in a_filling.shape)
        a_l1_k_outer, a_l1_k_inner =\
            sch[a_l1].split(sch[a_l1].op.axis[2], factor=aub_tiling_k_factor)
        a_l1_h_outer, a_l1_h_inner =\
            sch[a_l1].split(sch[a_l1].op.axis[3], factor=aub_tiling_m_factor)
        a_l1_w_outer, a_l1_w_inner = sch[a_l1].split(sch[a_l1].op.axis[4],
                                                     factor=aub_w)
        sch[a_l1].reorder(a_l1_k_outer, a_l1_h_outer, a_l1_w_outer,
                          sch[a_l1].op.axis[0], sch[a_l1].op.axis[1],
                          a_l1_k_inner, a_l1_h_inner, a_l1_w_inner)
        return a_l1_h_outer

    def _multi_core():  # pylint:disable=R0914
        block_dim = tiling['block_dim']
        batch_dim, n_dim, m_dim, d_dim = block_dim
        blocks = batch_dim * n_dim * m_dim * d_dim
        if blocks != 1:
            multicore_batch, batch_outer_inner = \
                sch[c_ddr].split(batch_outer, nparts=batch_dim)
            multicore_d, c_ddr_deep_outer_inner = \
                sch[c_ddr].split(c_ddr_deep_outer, nparts=d_dim)
            # split n axis
            multicore_n, bl1_at_ddr_n_outer_inner = \
                sch[c_ddr].split(bl1_at_ddr_n_outer, nparts=n_dim)
            # split m axis
            multicore_m, al1_at_ddr_m_outer_inner = \
                sch[c_ddr].split(al1_at_ddr_m_outer, nparts=m_dim)
            # reorder
            sch[c_ddr].reorder(
                multicore_batch, multicore_d,
                multicore_n, multicore_m,
                batch_outer_inner,
                c_ddr_deep_outer_inner,
                bl1_at_ddr_n_outer_inner,
                al1_at_ddr_m_outer_inner)
            out_fused = sch[c_ddr].fuse(multicore_batch,
                                        multicore_d,
                                        multicore_n,
                                        multicore_m)
            out_fused_out, _ = sch[c_ddr].split(out_fused, nparts=blocks)
            bind_out, _ = sch[c_ddr].split(out_fused_out, 1)
            blockidx = tvm.thread_axis("blockIdx.x")
            sch[c_ddr].bind(bind_out, blockidx)
        else:
            batch_outer_inner = batch_outer
            c_ddr_deep_outer_inner = c_ddr_deep_outer
            bl1_at_ddr_n_outer_inner = bl1_at_ddr_n_outer
            al1_at_ddr_m_outer_inner = al1_at_ddr_m_outer
        return \
            batch_outer_inner, c_ddr_deep_outer_inner,\
            bl1_at_ddr_n_outer_inner, al1_at_ddr_m_outer_inner

    def _tiling_check():
        _tiling_check_none()
        _tiling_check_value()
        _tiling_check_factor()
        _tiling_check_pbuffer()
        if stride_h == 1 and stride_w == 1:
            if tiling.get("AUB_shape") is not None:
                raise RuntimeError("stride = 1 but AUB_shape is not None.")

        if tiling.get("BL0_matrix") == [] and tiling.get("BL1_shape") != []:
            raise RuntimeError("BL0 full load but BL1 not!")

    def _tiling_check_value():
        if tiling.get("BL0_matrix"):
            if al0_tiling_ka != bl0_tiling_kb:
                raise RuntimeError("ka != kb.")
            if bl0_tiling_nb != cl0_tiling_nc:
                raise RuntimeError("nb != nc.")

        if al0_tiling_ma != cl0_tiling_mc:
            raise RuntimeError("ma != mc.")

    def _tiling_check_none():
        if (tiling.get("AL1_shape") is None) or \
           (tiling.get("BL1_shape") is None) or \
           (tiling.get("CUB_matrix") is None):
            raise RuntimeError("AL1_shape/BL1_shape/CUB_matrix "
                               "can't be None.")
        if (tiling.get("AL0_matrix") is None) or \
           (tiling.get("BL0_matrix") is None) or \
           (tiling.get("CL0_matrix") is None):
            raise RuntimeError("AL0_matrix/BL0_matrix/CL0_matrix "
                               "can't be None.")

    def _tiling_check_factor():
        if (kernel_w * kernel_h * dy_cout1) % al0_tiling_ka != 0:
            raise RuntimeError("Co1*Hk*Wk % ka != 0")

        if al1_tilling_k % al0_tiling_ka != 0:
            raise RuntimeError("k_AL1 % ka != 0.")

        if tiling.get("BL1_shape") != [] and tiling.get("BL0_matrix") != []:
            if bl1_tilling_k % bl0_tiling_kb != 0:
                raise RuntimeError("k_BL1 % kb != 0.")

        if cl0_tiling_nc % cub_tiling_nc_factor != 0:
            raise RuntimeError("nc % nc_factor != 0.")

        if tiling.get("BL1_shape"):
            if al1_tilling_k > bl1_tilling_k \
                    and al1_tilling_k % bl1_tilling_k != 0:
                raise RuntimeError("k_AL1 > k_BL1 but k_AL1 % k_BL1 != 0.")
            if bl1_tilling_k > al1_tilling_k \
                    and bl1_tilling_k % al1_tilling_k != 0:
                raise RuntimeError("k_BL1 > k_AL1 but k_BL1 % k_AL1 != 0.")

    def _tiling_check_pbuffer():
        if stride_h > 1 or stride_w > 1:
            if aub_pbuffer not in (1, 2):
                raise RuntimeError("value of AUB_pbuffer can only be 1 or 2")

        if al1_pbuffer not in (1, 2):
            raise RuntimeError("value of AL1_pbuffer can only be 1 or 2")

        if bl1_pbuffer not in (1, 2):
            raise RuntimeError("value of BL1_pbuffer can only be 1 or 2")

        if al0_pbuffer not in (1, 2):
            raise RuntimeError("value of AL0_pbuffer can only be 1 or 2")

        if bl0_pbuffer not in (1, 2):
            raise RuntimeError("value of BL0_pbuffer can only be 1 or 2")

        if l0c_pbuffer not in (1, 2):
            raise RuntimeError("value of L0C_pbuffer can only be 1 or 2")

        if cub_pbuffer not in (1, 2):
            raise RuntimeError("value of CUB_pbuffer can only be 1 or 2")

    def _fetch_tensor_info():  # pylint:disable=R0914, R0915
        tensor_map = {}
        tensor_attr = {}
        stride_d = c_ddr.op.attrs["stride_d"].value
        kernel_d, _, _ = list(i.value for i in c_ddr.op.attrs["kernels"])
        c_fill_zero = c_ddr.op.input_tensors[0]
        c_ub = c_fill_zero.op.input_tensors[0]
        tensor_map['c_fill_zero'] = c_fill_zero
        sch[c_fill_zero].set_scope(scope_ubuf)
        c_col = c_ub.op.input_tensors[0]
        a_col = c_col.op.input_tensors[0]  # im2col_fractal in L0A
        b_col = c_col.op.input_tensors[1]  # weight_transform in L0B
        b_ddr = b_col.op.input_tensors[0]  # weight in ddr
        a_col_before = a_col.op.input_tensors[0]  # im2col_row_major in L1

        tensor_map['c_ub'] = c_ub
        tensor_map['c_col'] = c_col
        tensor_map['a_col'] = a_col
        tensor_map['b_col'] = b_col
        tensor_map['b_ddr'] = b_ddr
        tensor_map['a_col_before'] = a_col_before

        # stride > 1
        if a_col_before.op.input_tensors[0].op.tag == "dy_l1" or \
                a_col_before.op.input_tensors[0].op.tag == "dy_l1_cut":

            a_l1 = a_col_before.op.input_tensors[0]
            a_filling = a_l1.op.input_tensors[0]

            stride_h, stride_w = list(i.value for i in
                                      a_filling.op.attrs["stride_expand"])
            a_ddr = a_filling.op.input_tensors[0]  # dEdY in ddr
            a_zero = a_filling.op.input_tensors[1]  # dEdY_zero in ub
            tensor_map['a_l1'] = a_l1
            tensor_map['a_filling'] = a_filling
            tensor_map['a_ddr'] = a_ddr
            tensor_map['a_zero'] = a_zero
        else:
            a_ddr = a_col_before.op.input_tensors[0]  # dEdY in ddr
            stride_h = 1
            stride_w = 1
            tensor_map['a_ddr'] = a_ddr
        tensor_attr['stride_w'] = stride_w
        tensor_attr['stride_h'] = stride_h
        # dataflow management
        b_l1 = sch.cache_read(b_ddr, scope_cbuf, [b_col])
        tensor_map['b_l1'] = b_l1
        sch[b_col].set_scope(scope_cb)
        if stride_h == 1 and stride_w == 1:
            a_l1 = sch.cache_read(a_ddr, scope_cbuf, [a_col_before])
            tensor_map['a_l1'] = a_l1
        else:
            a_ub = sch.cache_read(a_ddr, scope_ubuf, [a_filling])
            tensor_map['a_ub'] = a_ub
            # generate a_zero in ub
            sch[a_zero].set_scope(scope_ubuf)
            sch[a_filling].set_scope(scope_ubuf)
            # dma : a_filling ub------>L1
            sch[a_l1].set_scope(scope_cbuf)

        sch[a_col_before].set_scope(scope_cbuf)
        sch[a_col].set_scope(scope_ca)

        sch[c_col].set_scope(scope_cc)
        sch[c_ub].set_scope(scope_ubuf)
        padding = list(i.value for i in a_col_before.op.attrs["padding"])
        output_shape = list(i.value for i in c_ddr.op.attrs["output_shape"])
        tensor_attr['padding'] = padding
        tensor_attr['output_shape'] = output_shape
        tensor_attr['stride_d'] = stride_d
        tensor_attr['kernel_d'] = kernel_d
        return tensor_map, tensor_attr

    def _tiling_l0_process():
        if al0_tiling_ma == a_col_ma and \
                al0_tiling_ka == a_col_ka and a_col_batch == 1:
            tiling["AL0_matrix"] = []
        if tiling.get("BL0_matrix"):
            bl0_tiling_kb, bl0_tiling_nb, _, _, _, bl0_tiling_kd = \
            tiling.get("BL0_matrix")
        else:
            bl0_tiling_kd, bl0_tiling_kb, bl0_tiling_nb, _, _, = \
            list(i.value for i in b_col.shape)
            bl0_tiling_nb = bl0_tiling_nb//n_dim
        return bl0_tiling_kb, bl0_tiling_nb, bl0_tiling_kd

    def _tiling_l1_process():
        if tiling.get("AL1_shape"):
            al1_tilling_k, al1_tilling_m, _, _ = tiling.get("AL1_shape")
            if al1_tilling_k == kernel_h*kernel_w*al1_co1*al1_co0 and \
               al1_tilling_m == c_l0c_hw \
               // (CUBE_MKN[c_col.dtype]["mac"][0]*cl0_tiling_mc):
                tiling["AL1_shape"] = []
        else:
            # batch = 1 other axes full load
            al1_tilling_k = kernel_h * kernel_w * al1_co1 * al1_co0
            al1_tilling_m = \
                c_l0c_hw//(CUBE_MKN[c_col.dtype]["mac"][0] *
                           cl0_tiling_mc) // m_dim
        if tiling.get("BL1_shape"):
            bl1_tilling_k, bl1_tilling_n, _, bl1_tilling_kdparts =\
                tiling.get("BL1_shape")
        else:
            bl1_tilling_k = kernel_h * kernel_w * bl1_co0 * bl1_co1
            bl1_tilling_n = bl1_k1 // (kernel_h *
                                       kernel_w * cl0_tiling_nc) // n_dim
            bl1_tilling_kdparts = 1
        return al1_tilling_k, al1_tilling_m, bl1_tilling_k,\
               bl1_tilling_n, bl1_tilling_kdparts

    def _reorder_management():
        reorder_flag = False
        if k_al1_factor != 1 and k_bl1_factor == 1:
            reorder_flag = True
        if tiling['AL1_shape'] != [] and tiling['BL1_shape'] != [] \
                and k_al1_factor == 1 and k_bl1_factor == 1:
            if tiling['AL1_shape'][1] > tiling['BL1_shape'][1]:
                reorder_flag = True
        if tiling['AL1_shape'] != [] and tiling['BL1_shape'] != [] \
                and k_al1_factor != 1 and k_bl1_factor != 1:
            if tiling['AL1_shape'][1] > tiling['BL1_shape'][1]:
                reorder_flag = True
        if reorder_flag:
            sch[c_ddr].reorder(bl1_at_ddr_n_outer,
                               al1_at_ddr_m_outer, batch_inner)

    def _do_compute_at():
        if not tiling['AL1_shape']:
            sch[a_l1].compute_at(sch[c_ddr], c_ddr_deep_outer)
            sch[a_col_before].compute_at(sch[c_ddr], c_ddr_deep_outer)
        elif al1_tilling_k == kernel_h * kernel_w * al1_co1 * al1_co0:
            sch[a_l1].compute_at(sch[c_ddr], al1_at_ddr_m_outer)
            sch[a_col_before].compute_at(sch[c_ddr], al1_at_ddr_m_outer)
        else:
            sch[a_l1].compute_at(sch[c_col], al1_at_l0c_axis)
            sch[a_col_before].compute_at(sch[c_col], al1_at_l0c_axis)

        # ####bl1_compute_at
        if not tiling['BL1_shape']:
            sch[b_l1].compute_at(sch[c_ddr], batch_outer)
        elif bl1_tilling_k == kernel_h * kernel_w * bl1_co0 * bl1_co1:
            sch[b_l1].compute_at(sch[c_ddr], bl1_at_ddr_n_outer)
        else:
            sch[b_l1].compute_at(sch[c_col], bl1_at_l0c_axis)
        sch[c_ub].compute_at(sch[c_ddr], cddr_m_outer_inner)
        sch[c_fill_zero].compute_at(sch[c_ddr], cddr_m_outer_inner)
        sch[c_col].compute_at(sch[c_ddr], col_at_ddr_aixs)
        sch[a_col].compute_at(sch[c_col], c_col_m_outer)
        if not tiling['BL0_matrix'] and not tiling['BL1_shape']:
            sch[b_col].compute_at(sch[c_ddr], c_ddr_deep_outer)
        else:
            sch[b_col].compute_at(sch[c_col], c_col_n_outer)
        if stride_h > 1 or stride_w > 1:
            sch[a_filling].compute_at(sch[a_l1], a_l1_h_outer)
            sch[a_zero].compute_at(sch[a_l1], a_l1_h_outer)
            sch[a_ub].compute_at(sch[a_l1], a_l1_h_outer)

    def _double_buffer():
        if stride_h > 1 or stride_w > 1:
            if aub_pbuffer == 2:
                sch[a_ub].double_buffer()
                sch[a_filling].double_buffer()
                sch[a_zero].double_buffer()

        if al1_pbuffer == 2:
            sch[a_l1].double_buffer()

        if bl1_pbuffer == 2:
            sch[b_l1].double_buffer()

        if al0_pbuffer == 2:
            sch[a_col].double_buffer()

        if bl0_pbuffer == 2:
            sch[b_col].double_buffer()

        if l0c_pbuffer == 2:
            sch[c_col].double_buffer()

        if cub_pbuffer == 2:
            sch[c_ub].double_buffer()
            sch[c_fill_zero].double_buffer()

    def _emit_insn_process():
        sch[b_l1].emit_insn(sch[b_l1].op.axis[0], "dma_copy")

        sch[b_col].emit_insn(sch[b_col].op.axis[2], "dma_copy")
        if stride_h == 1 and stride_w == 1:
            sch[a_l1].emit_insn(sch[a_l1].op.axis[0], "dma_copy")
        else:
            sch[a_ub].emit_insn(sch[a_ub].op.axis[0], "dma_copy")
            afill_n, afill_d, afill_c, afill_h, afill_w, _ = \
                sch[a_filling].op.axis

            afill_w_out, afill_w_inner = sch[a_filling].split(
                afill_w, factor=stride_w)
            sch[a_filling].reorder(
                afill_w_inner,
                afill_n,
                afill_d,
                afill_c,
                afill_h,
                afill_w_out)
            sch[a_filling].unroll(afill_w_inner)
            sch[a_filling].reused_by(a_zero)
            sch[a_zero].emit_insn(sch[a_zero].op.axis[0], "vector_dup")
            sch[a_filling].emit_insn(afill_n, "vector_muls")
            sch[a_l1].emit_insn(sch[a_l1].op.axis[0], "dma_copy")

        setfmatrix_dict = {"conv_kernel_h": kernel_h,
                           "conv_kernel_w": kernel_w,
                           "conv_padding_top": padu,
                           "conv_padding_bottom": padd,
                           "conv_padding_left": padl,
                           "conv_padding_right": padr,
                           "conv_stride_h": 1,
                           "conv_stride_w": 1,
                           "conv_fm_c": a_l1.shape[2] * a_l1.shape[5],
                           "conv_fm_h": a_l1.shape[3],
                           "conv_fm_w": a_l1.shape[4]}

        sch[a_col_before].emit_insn(a_col_before.op.axis[1],
                                    'set_fmatrix', setfmatrix_dict)
        _, a_col_deep_inner = sch[a_col].split(sch[a_col].op.axis[1], factor=1)
        sch[a_col].emit_insn(a_col_deep_inner, 'im2col')
        deep_index = c_ddr_deep_outer * cddr_deep_factor + c_col_deep_outer

        if kd_reduce_flag:
            axis_kd = reduce_axis_kd_outer * kd_factor + reduce_axis_kd_inner
            mad_dict = {"mad_pattern": 2,
                        "k_outer": [reduce_axis_serial[0],
                                    reduce_axis_serial[1],
                                    reduce_axis_serial[2],
                                    reduce_axis_kd],
                        'k_cond':
                            tvm.all(
                                deep_index + pad_head -
                                tvm.min(a_ddr.shape[1] - 1,
                                        (deep_index + pad_head)//stride_d)
                                * stride_d - axis_kd == 0,
                                reduce_axis_serial[0].var == 0,
                                reduce_axis_serial[1].var == 0,
                                reduce_axis_serial[2].var == 0)}

            mad_dict_stride1 = {"mad_pattern": 2,
                                "k_outer": [reduce_axis_serial[0],
                                            reduce_axis_serial[1],
                                            reduce_axis_serial[2],
                                            reduce_axis_kd],
                                'k_cond':
                                    tvm.all(
                                        axis_kd +
                                        tvm.min(0, dy_depth - 1
                                                - pad_head - deep_index) == 0,
                                        reduce_axis_serial[0].var == 0,
                                        reduce_axis_serial[1].var == 0,
                                        reduce_axis_serial[2].var == 0)}
        else:
            mad_dict_originovrlap = {"mad_pattern": 2,
                                     "k_outer":  [reduce_axis_serial[0],
                                                  reduce_axis_serial[1],
                                                  reduce_axis_serial[2]]}

        if (stride_d == kernel_d) and \
                (a_col_d*stride_d == c_l0c_d + pad_head + pad_tail) and \
                (not kd_reduce_flag):
            sch[c_col].emit_insn(
                c_col_deep_inner, "mad", mad_dict_originovrlap)
        elif stride_d == 1:
            sch[c_col].emit_insn(
                c_col_deep_inner, "mad", mad_dict_stride1)
        else:
            sch[c_col].emit_insn(
                c_col_deep_inner, "mad", mad_dict)

        sch[c_fill_zero].reused_by(c_ub)
        sch[c_fill_zero].emit_insn(sch[c_fill_zero].op.axis[0], "vector_dup")
        sch[c_ub].emit_insn(sch[c_ub].op.axis[0], "dma_copy")
        sch[c_ddr].emit_insn(cddr_n_for_cub, "dma_copy")

    c_ddr = tensor
    sch = sch_list[0]
    tensor_map, tensor_attr = _fetch_tensor_info()
    c_ub = tensor_map.get("c_ub")
    c_col = tensor_map.get("c_col")
    a_col = tensor_map.get("a_col")
    b_col = tensor_map.get("b_col")
    b_ddr = tensor_map.get("b_ddr")
    a_col_before = tensor_map.get("a_col_before")
    a_l1 = tensor_map.get("a_l1")
    a_filling = tensor_map.get("a_filling")
    a_zero = tensor_map.get("a_zero")
    a_ddr = tensor_map.get("a_ddr")
    b_l1 = tensor_map.get("b_l1")
    a_ub = tensor_map.get("a_ub")
    output_shape = tensor_attr.get("output_shape")
    padding = tensor_attr.get("padding")
    stride_h = tensor_attr.get("stride_h")
    stride_w = tensor_attr.get("stride_w")
    stride_d = tensor_attr.get("stride_d")
    kernel_d = tensor_attr.get("kernel_d")
    c_fill_zero = tensor_map.get("c_fill_zero")

    # =========================tiling_query======================#
    padu, padd, padl, padr = padding
    pad_head, pad_tail = list(i.value for i in c_ddr.op.attrs["depth_pad"])
    tensor_attr['pad_head'] = pad_head
    tensor_attr['pad_tail'] = pad_tail
    _, _, _, _, kernel_h, kernel_w, _ = \
        list(i.value for i in a_col_before.shape)
    img_shape = list(i.value for i in a_ddr.shape)
    _, dy_depth, dy_cout1, _, _, _ = img_shape
    b_ddr_kd, b_ddr_k1, b_ddr_n1, b_ddr_n0, b_ddr_k0 \
        = list(i.value for i in b_ddr.shape)
    # Cout, Cin1, Hk, Wk, Cin0
    filter_shape = (b_ddr_n1*b_ddr_n0,
                    b_ddr_kd, b_ddr_k1//(kernel_h*kernel_w),
                    kernel_h, kernel_w, b_ddr_k0)
    cddr_batch, cddr_depth, cdder_c1, cdder_h, cdder_w, cdder_c0 = output_shape
    tiling_output = (cddr_batch,
                     cddr_depth, cdder_h, cdder_w, cdder_c1*cdder_c0)
    kd_reduce_flag = bool(len(c_col.op.reduce_axis) == NUM_3)

    # tiling_auery
    tiling = tiling_query(a_shape=img_shape,
                          b_shape=filter_shape,
                          c_shape=tiling_output,
                          a_dtype='float16',
                          b_dtype='float16',
                          c_dtype='float16',
                          mad_dtype='float32',
                          padl=padl,
                          padr=padr,
                          padu=padu,
                          padd=padd,
                          padf=pad_head,
                          padb=pad_tail,
                          strideh=1, stridew=1,
                          strideh_expand=stride_h,
                          stridew_expand=stride_w,
                          strided=stride_d,
                          bias_flag=False,
                          op_tag="conv3d_backprop_input")

    if stride_w == 1 and stride_h == 1:
        tiling['AUB_shape'] = None
    _, n_dim, m_dim, _ = tiling.get("block_dim")
    aub_pbuffer = tiling.get("manual_pingpong_buffer").get("AUB_pbuffer")
    al1_pbuffer = tiling.get("manual_pingpong_buffer").get("AL1_pbuffer")
    bl1_pbuffer = tiling.get("manual_pingpong_buffer").get("BL1_pbuffer")
    al0_pbuffer = tiling.get("manual_pingpong_buffer").get("AL0_pbuffer")
    bl0_pbuffer = tiling.get("manual_pingpong_buffer").get("BL0_pbuffer")
    l0c_pbuffer = tiling.get("manual_pingpong_buffer").get("CL0_pbuffer")
    cub_pbuffer = tiling.get("manual_pingpong_buffer").get("CUB_pbuffer")
    _, _, al1_co1, _, _, al1_co0 = list(i.value for i in a_l1.shape)
    _, c_l0c_d, _, c_l0c_hw, _ = list(i.value for i in c_col.shape)
    _, bl1_k1, bl1_co1, bl1_co0, _ = list(i.value for i in b_l1.shape)
    a_col_shape = list(i.value for i in a_col.shape)
    a_col_batch, a_col_d, a_col_ma, a_col_ka, _, _ = a_col_shape
    cub_tiling_nc_factor, cub_tiling_mc_factor, cub_tiling_m0, _,\
        _, _ = tiling.get("CUB_matrix")
    cl0_tiling_nc, cl0_tiling_mc, _, _, _, _ \
        = tiling.get("CL0_matrix")
    al0_tiling_ma, al0_tiling_ka, al0_tiling_m0, _, _, al0_tiling_dfactor \
        = tiling.get("AL0_matrix")
    bl0_tiling_kb, bl0_tiling_nb, bl0_tiling_kd = _tiling_l0_process()
    al1_tilling_k, al1_tilling_m, bl1_tilling_k,\
    bl1_tilling_n, bl1_tilling_kdparts = _tiling_l1_process()

    # tiling_check
    _tiling_check()

    # axis management
    batch_after_multicore, d_after_multicore, n_after_multicore,\
    m_after_multicore = c_ddr.op.axis[0], c_ddr.op.axis[1],\
                        c_ddr.op.axis[2], c_ddr.op.axis[3]
    # cub
    cddr_n_outer, cddr_m_outer, cddr_n_for_cub = _cub_process()
    # l0c
    cddr_deep_factor = al0_tiling_dfactor
    kd_factor = bl0_tiling_kd
    kd_tiling_l1_factor = bl1_tilling_kdparts
    batch_outer, al1_at_ddr_m_outer, batch_inner, bl1_at_ddr_n_outer,\
    col_at_ddr_aixs, cddr_m_outer_inner, _, c_ddr_deep_outer,\
    _ = _l0c_procees()

    # l0a_l0b
    al0_m_factor = al0_tiling_ma * al0_tiling_m0
    if kd_reduce_flag is False:
        c_col_deep_outer, c_col_deep_inner, c_col_k_outer, c_col_m_outer, \
        c_col_n_outer = _l0a_and_l0b_process()
    else:
        c_col_deep_outer, c_col_deep_inner, c_col_k_outer, c_col_m_outer, \
        c_col_n_outer, reduce_axis_kd, reduce_axis_kd_outer,\
        reduce_axis_kd_inner = _l0a_and_l0b_process()

    # l1a_l1b
    k_al1_factor = \
        al1_tilling_k // al0_tiling_ka // CUBE_MKN[c_col.dtype]["mac"][0]
    k_bl1_factor = \
        bl1_tilling_k // bl0_tiling_kb // CUBE_MKN[c_col.dtype]["mac"][0]
    reduce_axis_serial, bl1_at_l0c_axis, al1_at_l0c_axis =\
        _al1_and_bl1_process()

    if stride_h > 1 or stride_w > 1:
        a_l1_h_outer = _aub_process()

    _reorder_management()

    # buffer_align
    if kd_reduce_flag:
        sch[c_col].buffer_align(
            (1, 1),
            (1, 1),
            (1, 1),
            (1, CUBE_MKN[c_col.dtype]["mac"][0]),
            (1, CUBE_MKN[c_col.dtype]["mac"][2]),
            (1, 1),
            (1, 1),
            (1, CUBE_MKN[c_col.dtype]["mac"][1]))
    else:
        sch[c_col].buffer_align(
            (1, 1),
            (1, 1),
            (1, 1),
            (1, CUBE_MKN[c_col.dtype]["mac"][0]),
            (1, CUBE_MKN[c_col.dtype]["mac"][2]),
            (1, 1),
            (1, CUBE_MKN[c_col.dtype]["mac"][1]))
    _, _, _, _, dx_w, _ = output_shape
    sch[a_col_before].buffer_align(
        (1, 1),
        (1, 1),
        (dx_w, dx_w),
        (1, 1),
        (1, 1),
        (1, 1),
        (1, CUBE_MKN[a_col_before.dtype]["mac"][1]))
    sch[c_ub].buffer_align(
        (1, 1),
        (1, 1),
        (1, 1),
        (1, CUBE_MKN[c_ub.dtype]["mac"][0]),
        (1, CUBE_MKN[c_ub.dtype]["mac"][2]))
    sch[c_fill_zero].buffer_align(
        (1, 1),
        (1, 1),
        (1, 1),
        (1, CUBE_MKN[c_ub.dtype]["mac"][0]),
        (1, CUBE_MKN[c_ub.dtype]["mac"][2]))
    batch_outer, c_ddr_deep_outer, bl1_at_ddr_n_outer,\
    al1_at_ddr_m_outer = _multi_core()

    _do_compute_at()

    _double_buffer()

    # emitinsn
    _emit_insn_process()
    return sch
