# -*- coding: UTF-8 -*-
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

conv3d backprop filter schudule.

"""
from __future__ import absolute_import
from __future__ import print_function
from te import tvm
from te.platform import scope_ubuf
from te.platform import scope_ca
from te.platform import scope_cb
from te.platform import scope_cc
from te.platform import scope_cbuf
from te.platform import get_soc_spec
from te.domain.tiling.tiling_query import tiling_query

L1_SIZE = get_soc_spec("L1_SIZE")  # L1 size

# disable double buffer, set True
DEBUG_DOUBLE_BUFFER_OFF = False

CUBE_DIM = 16
FLOAT16_SIZE = 2
CUBE_MUL_SHAPE = 256
OPEN_DOUBLE_BUFFER = 2
DEFAULT_TILING_CASE = 32


def ceil_div(dividend, divisor):
    """
    do division and round up to an integer

    """
    if divisor == 0:
        raise RuntimeError(" division by zero")
    return (dividend + divisor - 1) // divisor


def align(x_1, x_2):
    """
    do align

    """
    if x_2 == 0:
        raise RuntimeError("Division by zero")
    return (x_1 + x_2 - 1) // x_2 * x_2


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
    start = process + " IR start"
    end = process + " IR end\n"
    print(start)
    bounds = tvm.schedule.InferBound(sch)
    stmt = tvm.schedule.ScheduleOps(sch, bounds, True)
    print(stmt)
    print(end)

    return False


class CceConv3dBackpropFilterOp:  # pylint: disable=too-few-public-methods
    """
    CceConv3dBackpropFilterOp: schedule definition of conv3d_backprop_filter

    Functions
    ----------
    __init__ : initialization

    schedule : schedule definition of conv3d_backprop_filter

    """
    def __init__(self, scope, need_tensorize=True, need_pragma=True):
        """
        initialization

        Parameters:
        ----------
        scope : scope definition

        need_tensorize : whether needs tensorize

        need_pragma : whether needs pragma

        Returns
        -------
        None
        """
        self.scope = scope
        self.need_tensorize = need_tensorize
        self.need_pragma = need_pragma
        self.spec_node_list = []

    def schedule(
            self,  # pylint: disable=R0914,R0915
            res,
            spec_node_list,
            sch_list):
        """
        schedule definition of conv3d_backprop_filter

        Parameters:
        ----------
        res :

        spec_node_list :

        sch_list:

        Returns
        -------
        None
        """
        self.spec_node_list = spec_node_list

        def _tiling_shape_check():
            """
            do tiling shape paramters general check

            """

            al1_shape = tiling.get("AL1_shape")
            bl1_shape = tiling.get("BL1_shape")
            al0_matrix = tiling.get("AL0_matrix")
            bl0_matrix = tiling.get("BL0_matrix")
            cl0_matrix = tiling.get("CL0_matrix")
            if al1_shape:
                if al1_shape[0] % al0_matrix[1] != 0:
                    raise RuntimeError("k of AL1_shape should be integral "
                                       "multiple of AL0_matrix")
                if al1_shape[1] < 1:
                    raise RuntimeError("m of AL1_shape should be integral "
                                       "multiple of AL0_matrix")

            if bl1_shape:
                if (bl1_shape[0] // CUBE_DIM) % bl0_matrix[0] != 0:
                    raise RuntimeError("k of BL1_shape should be integral "
                                       "multiple of BL0_matrix")
                if bl1_shape[1] < 1:
                    raise RuntimeError("n of BL1_shape should be integral "
                                       "multiple of BL0_matrix")

            if al0_matrix:
                if al0_matrix[0] != cl0_matrix[1]:
                    raise RuntimeError("mc of AL0_matrix and CL0_matrix "
                                       "should be same")

            if bl0_matrix:
                if bl0_matrix[1] != cl0_matrix[0]:
                    raise RuntimeError("nc of BL0_matrix and CL0_matrix "
                                       "should be same")

            if al0_matrix and bl0_matrix:
                if al0_matrix[1] != bl0_matrix[0]:
                    raise RuntimeError("k of AL0_matrix and BL0_matrix "
                                       "should be same")

        def _tiling_buffer_check():
            """
            Do buffer paramters general check

            """
            block_cout = tiling.get("block_dim")[2]

            al1_pbuff = tiling.get("manual_pingpong_buffer").get("AL1_pbuffer")
            bl1_pbuff = tiling.get("manual_pingpong_buffer").get("BL1_pbuffer")
            al0_pbuff = tiling.get("manual_pingpong_buffer").get("AL0_pbuffer")
            bl0_pbuff = tiling.get("manual_pingpong_buffer").get("BL0_pbuffer")
            l0c_pbuff = tiling.get("manual_pingpong_buffer").get("CL0_pbuffer")
            cub_pbuff = tiling.get("manual_pingpong_buffer").get("CUB_pbuffer")
            cl0_matrix = tiling.get("CL0_matrix")
            cub_matrix = tiling.get("CUB_matrix")
            if cl0_matrix[0] % cub_matrix[0] != 0 \
               or cl0_matrix[1] != cub_matrix[1]:
                raise RuntimeError("invalid CUB_matrix value")
            # blockIdx must be positive int
            if block_cout < 1:
                raise RuntimeError("blockIdx must be positive int")

            # only support no dbuffer/ dbuffer
            if al1_pbuff not in (1, 2):
                raise RuntimeError("value of AL1_pbuffer can only be 1 or 2")

            if bl1_pbuff not in (1, 2):
                raise RuntimeError("value of BL1_pbuffer can only be 1 or 2")

            if al0_pbuff not in (1, 2):
                raise RuntimeError("value of AL0_pbuffer can only be 1 or 2")

            if bl0_pbuff not in (1, 2):
                raise RuntimeError("value of BL0_pbuffer can only be 1 or 2")

            if l0c_pbuff not in (1, 2):
                raise RuntimeError("value of L0C_pbuffer can only be 1 or 2")

            if cub_pbuff not in (1, 2):
                raise RuntimeError("value of CUB_pbuffer can only be 1 or 2")

        def _l1_limit_check():
            """
            do L1 size limit check

            """
            al1_min_byte = CUBE_DIM * CUBE_DIM * FLOAT16_SIZE
            if width_grads >= CUBE_DIM:
                if width_grads % CUBE_DIM == 0:
                    bl1_min_byte = kernel_height * width_fmap * CUBE_DIM *\
                                   FLOAT16_SIZE
                else:
                    bl1_min_byte = (kernel_height+stride_height) \
                                   * width_fmap * CUBE_DIM * FLOAT16_SIZE
            else:
                bl1_align_factor = ceil_div(CUBE_DIM, width_grads)
                if CUBE_DIM % width_grads == 0:
                    bl1_min_byte = (kernel_height+(bl1_align_factor-1)
                                    * stride_height) * width_fmap * CUBE_DIM *\
                                   FLOAT16_SIZE
                else:
                    bl1_min_byte = (kernel_height +
                                    bl1_align_factor * stride_height) \
                                    * width_fmap * CUBE_DIM * FLOAT16_SIZE

            if (al1_min_byte + bl1_min_byte) > L1_SIZE:
                raise RuntimeError("invalid input size due to large kernel"
                                   " size and stride")

        def _atomic_add(sch, res_cc, res_ub, res_ddr):
            """
            achieve atomic add according to refactor dw_cc

            """

            # redefine dw_ddr, dw_ub, dw_cc to achieve atomic write
            ub_reduce = res_ub
            ddr_reduce = res_ddr

            batch, real_k = sch[res_cc].op.reduce_axis
            batch_core, batch_in = sch[res_cc].split(batch,
                                                     nparts=block_dim_batch)

            real_k, k_in = sch[res_cc].split(real_k, CUBE_DIM)
            k_1_multicore, real_k = sch[res_cc].split(real_k,
                                                      nparts=block_dim_hw)

            sch[res_cc].reorder(k_1_multicore, batch_core, batch_in, real_k,
                                k_in)
            fused_atomic_write = sch[res_cc].fuse(k_1_multicore, batch_core)

            # after rfactor op, dw_cc becomes dw_ddr, original dw_ub and dw_ddr
            # will be dropped
            res_ddr = res_cc
            res_cc = sch.rfactor(res_ddr, fused_atomic_write)
            sch[res_cc].set_scope(scope_cc)
            res_ub = sch.cache_read(res_cc, scope_ubuf, [res_ddr])
            return res_cc, res_ub, res_ddr, ub_reduce, ddr_reduce

        def _full_k_check():
            """
            set flag whether axis K is fully loaded in L0A and L0B
            return:
            -------
            full_k_l0a: 1 or 0,
                        1 means K is fully loaded in L0A
            full_k_l0b: 1 or 0,
                        1 means K is fully loaded in L0B
            """

            # if k is fully load in BL1 and
            # there is multi load in N1 and N1 in BL1
            # isn't aligned to kernel_height*kernel_width, then align to it
            if tiling.get("BL1_shape") and tiling.get("BL1_shape")[1] > 1 and \
                    tiling.get("BL1_shape")[1] * tiling.get("BL0_matrix")[1] \
                    % (kernel_height * kernel_width) != 0:
                tiling["BL1_shape"][1] = align(tiling.get("BL1_shape")[1] *
                                               tiling.get("BL0_matrix")[1],
                                               kernel_height * kernel_width) \
                                         // tiling.get("BL0_matrix")[1]

            # whether axis K is fully loaded in L0A and L0B
            # excluding axis batch
            if not tiling["AL0_matrix"]:
                full_k_l0a = 1
            else:
                full_k_l0a = tiling["AL0_matrix"][1] \
                             // ceil_div(hw_pad_1, block_dim_hw)

            if not tiling["BL0_matrix"]:
                full_k_l0b = 1
            else:
                full_k_l0b = tiling["BL0_matrix"][0] \
                             // ceil_div(hw_pad_1, block_dim_hw)

            return full_k_l0a, full_k_l0b

        def _compute_tiling_parts():
            """
            compute the parts or the factors of tensors

            """

            if not tiling["AL0_matrix"]:  # if grads no tiling in L0A
                tiling["AL1_shape"] = []  # then no tiling in L1

            # dw_cc is (fmap_channel_1*kernel_height*kernel_width,
            #          grads_channel_1, C0_grads, C0_fmap)
            dw_tiling_factor = [
                tiling["CL0_matrix"][0], tiling["CL0_matrix"][1]
            ]
            # nparts N, nparts M
            # dw_tiling_nparts only describe the nparts from single core to L0
            dw_tiling_nparts = \
                [ceil_div(fkk // block_dim_cin, dw_tiling_factor[0]),
                 ceil_div(ceil_div(c1_grads, dw_tiling_factor[1]),
                          block_dim_cout)]

            # tiling parameters of dw_ub
            dw_ub_tiling_factor = [
                tiling["CUB_matrix"][0], tiling["CUB_matrix"][1]
            ]
            dw_ub_tiling_nparts = [
                ceil_div(dw_tiling_factor[0], dw_ub_tiling_factor[0]),
                ceil_div(dw_tiling_factor[1], dw_ub_tiling_factor[1])
            ]

            # only support loading one batch to L1 at a time for now
            # cout:out->single core(sc)->L1
            if tiling["AL1_shape"]:  # if grads needs tiling in L1
                if len(tiling["AL1_shape"]) == 1:  # but no C_1 tiling info
                    tiling["AL1_shape"] = \
                        tiling["AL1_shape"] + [1]
                # nparts K1 in L1, nparts M1 in L1
                grads_l1_tiling_nparts = [
                    hw_pad_1 // block_dim_hw //
                    (tiling["AL1_shape"][0] // CUBE_DIM),
                    dw_tiling_nparts[1] // tiling["AL1_shape"][1]
                ]
            else:
                grads_l1_tiling_nparts = [1, 1]

            if tiling["BL1_shape"]:  # if fmap needs tiling in L1
                if len(tiling["BL1_shape"]) == 1:  # but no fkk tiling info
                    tiling["BL1_shape"] = \
                        tiling["BL1_shape"] + [1]  # tiling fkk=1
                # DDR to L1 [nparts K1, nparts N1]
                fmap_l1_tiling_nparts = [
                    hw_pad_1 // block_dim_hw //
                    (tiling["BL1_shape"][0] // CUBE_DIM),
                    dw_tiling_nparts[0] // tiling["BL1_shape"][1]
                ]
            else:
                fmap_l1_tiling_nparts = [1, 1]

            # during L1 to L0 [nparts N1, nparts M1]
            l1_2_l0_tiling_nparts = \
                [dw_tiling_nparts[0] // fmap_l1_tiling_nparts[1],
                 dw_tiling_nparts[1] // grads_l1_tiling_nparts[1]]
            # ka and kb may be different,
            # the min value corresponds to one MMAD,
            # the larger one is []
            if tiling["AL0_matrix"]:  # dw_k equals to ka if L0A needs tiling
                dw_k = tiling["AL0_matrix"][1]
            elif tiling["BL0_matrix"]:
                dw_k = tiling["BL0_matrix"][0]
            else:  # both fully loaded
                dw_k = hw_pad_1 // block_dim_hw

            tiling_patrs_dict = dict()
            tiling_patrs_dict["dw_tiling_factor"] = dw_tiling_factor
            tiling_patrs_dict["dw_tiling_nparts"] = dw_tiling_nparts
            tiling_patrs_dict["dw_ub_tiling_factor"] = dw_ub_tiling_factor
            tiling_patrs_dict["dw_ub_tiling_nparts"] = dw_ub_tiling_nparts
            tiling_patrs_dict["grads_l1_tiling_nparts"] = \
                grads_l1_tiling_nparts
            tiling_patrs_dict["fmap_l1_tiling_nparts"] = fmap_l1_tiling_nparts
            tiling_patrs_dict["l1_2_l0_tiling_nparts"] = l1_2_l0_tiling_nparts
            tiling_patrs_dict["dw_k"] = dw_k
            return tiling_patrs_dict

        def _l0_attach():
            """
            achieve Al0 and Bl0 compute at loc or ddr

            """

            if tiling["AL0_matrix"]:
                if (batch_num_sc == 1) and (full_k_in_l0a == 1):
                    # L0A data is more than that L0C needed，attach to dw_ddr
                    sch[grads_fractal].compute_at(sch[dw_ddr], c_grads_mad_at)
                else:
                    sch[grads_fractal].compute_at(sch[dw_cc], hw_mad_1_mad_at)
            else:  # else: fully load, attach to thread_axis
                sch[grads_fractal].compute_at(sch[dw_ddr], fused_multi_core)

            if tiling["BL0_matrix"]:
                if (batch_num_sc == 1) and (full_k_in_l0b == 1):
                    sch[fmap_fractal].compute_at(sch[dw_ddr], c_fmap_mad_at)
                else:
                    sch[fmap_fractal].compute_at(sch[dw_cc], hw_mad_1_mad_at)
            else:  # else: fully load, attach to thread_axis
                sch[fmap_fractal].compute_at(sch[dw_ddr], fused_multi_core)

        def _al1_attach():
            """
            achieve Al1 compute at l0c or ddr

            """
            if tiling["AL1_shape"]:
                # if axis K needs split, then attach to dw_cc
                if grads_l1_tiling_nparts[0] != 1 or batch_num_sc != 1:
                    sch[grads_matrix].compute_at(sch[dw_cc], al1_at_axis)
                else:  # if axis K fully load in L1, attach to dw_ddr
                    sch[grads_matrix].compute_at(sch[dw_ddr], c_grads_l1_at)
            else:  # else: fully load, attach to thread_axis
                sch[grads_matrix].compute_at(sch[dw_ddr], fused_multi_core)

        def _bl1_attach():
            """
            achieve Bl1 compute at l0c or ddr

            """
            if tiling["BL1_shape"]:
                # if axis K needs split, then attach to dw_cc
                if fmap_l1_tiling_nparts[0] != 1 or batch_num_sc != 1:
                    sch[fmap_matrix].compute_at(sch[dw_cc], bl1_at_axis)
                    if not flag_all_one_case:
                        sch[fmap_l1].compute_at(sch[dw_cc], bl1_at_axis)
                else:  # if axis K fully load in L1, attach to dw_ddr
                    sch[fmap_matrix].compute_at(sch[dw_ddr], c_fmap_l1_at)
                    if not flag_all_one_case:
                        sch[fmap_l1].compute_at(sch[dw_ddr], c_fmap_l1_at)

            else:  # else: fully load, attach to thread_axis
                sch[fmap_matrix].compute_at(sch[dw_ddr], fused_multi_core)
                if not flag_all_one_case:
                    sch[fmap_l1].compute_at(sch[dw_ddr], fused_multi_core)

        def _double_buffer():
            """
            achieve double_buffer

            """
            if not DEBUG_DOUBLE_BUFFER_OFF:
                if tiling.get("manual_pingpong_buffer").get("AL1_pbuffer") \
                        == OPEN_DOUBLE_BUFFER:
                    sch[grads_matrix].double_buffer()

                if tiling.get("manual_pingpong_buffer").get("BL1_pbuffer") \
                        == OPEN_DOUBLE_BUFFER:
                    if not flag_all_one_case:
                        sch[fmap_l1].double_buffer()
                    else:
                        sch[fmap_matrix].double_buffer()

                if tiling.get("manual_pingpong_buffer").get("AL0_pbuffer") \
                        == OPEN_DOUBLE_BUFFER:
                    sch[grads_fractal].double_buffer()

                if tiling.get("manual_pingpong_buffer").get("BL0_pbuffer") \
                        == OPEN_DOUBLE_BUFFER:
                    sch[fmap_fractal].double_buffer()

                if tiling.get("manual_pingpong_buffer").get("CL0_pbuffer") \
                        == OPEN_DOUBLE_BUFFER:
                    sch[dw_cc].double_buffer()

                if tiling.get("manual_pingpong_buffer").get("CUB_pbuffer") \
                        == OPEN_DOUBLE_BUFFER:
                    sch[dw_ub].double_buffer()

        def _emit_insn():
            """
            achieve emit_insn

            """
            setfmatrix_dict = dict()
            setfmatrix_dict["conv_kernel_h"] = kernel_height
            setfmatrix_dict["conv_kernel_w"] = kernel_width
            setfmatrix_dict["conv_padding_top"] = pad_up
            setfmatrix_dict["conv_padding_bottom"] = pad_down
            setfmatrix_dict["conv_padding_left"] = pad_left
            setfmatrix_dict["conv_padding_right"] = pad_right
            setfmatrix_dict["conv_stride_h"] = stride_height
            setfmatrix_dict["conv_stride_w"] = stride_width
            setfmatrix_dict["conv_fm_c"] = featuremap_channel
            setfmatrix_dict["conv_fm_h"] = featuremap_height
            setfmatrix_dict["conv_fm_w"] = featuremap_width
            setfmatrix_dict["conv_dilation_h"] = dilation_height
            setfmatrix_dict["conv_dilation_w"] = dilation_width

            mad_dict = {
                "mad_pattern":
                2,
                "k_outer": [
                    batch_insn_o, hw_mad_1_l1_out_at, hw_mad_1_l1_in_at,
                    hw_mad_1_mad_at
                ]
            }

            sch[grads_matrix].emit_insn(grads_matrix.op.axis[0], 'dma_copy')
            # move grads from L1 to L0A
            sch[grads_fractal].emit_insn(grads_fractal.op.axis[0], 'dma_copy')

            # move fmap from ddr to L1
            if not flag_all_one_case:
                sch[fmap_l1].emit_insn(fmap_l1.op.axis[0], 'dma_copy')
                sch[fmap_matrix].emit_insn(fmap_matrix.op.axis[0],
                                           'set_fmatrix', setfmatrix_dict)
                sch[fmap_fractal].emit_insn(fmap_fractal.op.axis[0], 'im2col')
            else:
                sch[fmap_matrix].emit_insn(fmap_matrix.op.axis[0], 'dma_copy')
                sch[fmap_fractal].emit_insn(fmap_fractal.op.axis[0],
                                            'dma_copy')

            # move dw from L0C to UB
            sch[dw_ub].emit_insn(dw_ub.op.axis[0], 'dma_copy')
            sch[dw_cc].emit_insn(batch_insn, 'mad', mad_dict)

            # move dw form UB to ddr
            sch[dw_ddr].emit_insn(c_fmap_2_ub_insn, 'dma_copy')

            sch[dw_ddr_reduce].emit_insn(dw_ddr_reduce.op.axis[0],
                                         'phony_insn')
            sch[dw_ub_reduce].emit_insn(dw_ub_reduce.op.axis[0], 'phony_insn')

            sch_list.append(dw_ddr)

        # ####################### get computing graph #######################
        dw_ddr = res  # pylint: disable=too-many-statements
        dw_ub = dw_ddr.op.input_tensors[0]
        dw_cc = dw_ub.op.input_tensors[0]
        grads_fractal = dw_cc.op.input_tensors[0]
        fmap_fractal = dw_cc.op.input_tensors[1]
        grads_matrix = grads_fractal.op.input_tensors[0]
        fmap_matrix = fmap_fractal.op.input_tensors[0]
        grads = grads_matrix.op.input_tensors[0]
        load2d_flag = fmap_matrix.op.attrs['load2d_flag'].value
        if load2d_flag:
            fmap = fmap_matrix.op.input_tensors[0]
        else:
            fmap_l1 = fmap_matrix.op.input_tensors[0]
            fmap = fmap_l1.op.input_tensors[0]

        # ########################extract parameters##########################
        default_tiling = {
            'AUB_shape': None,
            'BUB_shape': None,
            'AL1_shape': [CUBE_DIM, 1, 1],
            'BL1_shape': [CUBE_DIM, 1, 1],
            'AL0_matrix': [1, 1, CUBE_DIM, CUBE_DIM, 1],
            'BL0_matrix': [1, 1, CUBE_DIM, CUBE_DIM, 1],
            'CL0_matrix': [1, 1, CUBE_DIM, CUBE_DIM, 1],
            'CUB_matrix': [1, 1, CUBE_DIM, CUBE_DIM, 1],
            'block_dim': [1, 1, 1],
            'cout_bef_batch_flag': 0,
            'A_overhead_opt_flag': 0,
            'B_overhead_opt_flag': 0,
            'manual_pingpong_buffer': {
                'AUB_pbuffer': 1,
                'BUB_pbuffer': 1,
                'AL1_pbuffer': 1,
                'BL1_pbuffer': 1,
                'AL0_pbuffer': 1,
                'BL0_pbuffer': 1,
                'CL0_pbuffer': 1,
                'CUB_pbuffer': 1,
                'UBG_pbuffer': 1
            }
        }
        batch_grads, depth_grads, c1_grads, height_grads, width_grads, c0_grads \
            = list(x.value for x in grads.shape)
        grads_shape = [
            batch_grads, depth_grads, c1_grads, height_grads, width_grads,
            c0_grads
        ]

        batch_fmap, depth_fmap, c1_fmap, height_fmap, width_fmap, c0_fmap \
            = list(x.value for x in fmap.shape)
        fkk, _, _ = list(x.value for x in dw_cc.shape)
        _, hw_pad_1, _, _, _ = list(x.value for x in fmap_fractal.shape)

        fmap_shape = [
            batch_fmap, depth_fmap, c1_fmap, height_fmap, width_fmap, c0_fmap
        ]

        # load_3d parameters
        stride_depth = fmap_matrix.op.attrs['stride'][0].value
        stride_height = fmap_matrix.op.attrs['stride'][1].value
        stride_width = fmap_matrix.op.attrs['stride'][2].value
        pad_front = fmap_matrix.op.attrs['pad'][0].value
        pad_back = fmap_matrix.op.attrs['pad'][1].value
        pad_up = fmap_matrix.op.attrs['pad'][2].value
        pad_down = fmap_matrix.op.attrs['pad'][3].value
        pad_left = fmap_matrix.op.attrs['pad'][4].value
        pad_right = fmap_matrix.op.attrs['pad'][5].value
        kernel_depth = fmap_matrix.op.attrs['kernel_size'][1].value
        kernel_height = fmap_matrix.op.attrs['kernel_size'][3].value
        kernel_width = fmap_matrix.op.attrs['kernel_size'][4].value
        dilation_height = fmap_matrix.op.attrs['dilation'][2].value
        dilation_width = fmap_matrix.op.attrs['dilation'][3].value
        featuremap_channel = kernel_depth * c1_fmap * c0_fmap
        featuremap_height = height_fmap
        featuremap_width = width_fmap

        weight_shape = [
            c1_grads * c0_grads, kernel_depth, kernel_height,
            kernel_width, c1_fmap * c0_fmap
        ]

        sch = sch_list[0]

        def _flag_all_one():
            # special supporting for a unique case, there are 2 conditions:
            # (1) height & weight of x/output_backprop/filter are all 1
            # (2) strides is [1,1]
            flag_all_one_case = False
            height_all_one = False
            width_all_one = False
            if stride_height == 1 and height_grads == 1 and height_fmap == 1 \
                and kernel_height == 1:
                height_all_one = True
            if stride_width == 1 and width_grads == 1 and width_fmap == 1 \
                and kernel_width == 1:
                width_all_one = True
            if height_all_one and width_all_one:
                flag_all_one_case = True

            return flag_all_one_case

        flag_all_one_case = _flag_all_one()

        tiling = tiling_query(grads_shape,
                              fmap_shape,
                              weight_shape,
                              a_dtype=grads.dtype,
                              b_dtype=fmap.dtype,
                              c_dtype=dw_cc.dtype,
                              mad_dtype=dw_cc.dtype,
                              padl=pad_left,
                              padr=pad_right,
                              padu=pad_up,
                              padd=pad_down,
                              strideh=stride_height,
                              stridew=stride_width,
                              strideh_expand=1,
                              stridew_expand=1,
                              dilationh=dilation_height,
                              dilationw=dilation_width,
                              group=1,
                              fused_double_operand_num=0,
                              bias_flag=0,
                              op_tag='conv3d_backprop_filter',
                              padf=pad_front,
                              padb=pad_back,
                              strided=stride_depth)

        _tiling_shape_check()
        _tiling_buffer_check()
        # if no valid tiling found, the flag is as follows
        if tiling["AL0_matrix"][2] == DEFAULT_TILING_CASE:
            tiling = default_tiling
        _l1_limit_check()

        batch_num = batch_grads * depth_grads
        if tiling.get("AUB_shape"):
            block_dim_hw = tiling.get("AUB_shape")[0]
        else:
            block_dim_hw = 1
        block_dim_batch = tiling.get("block_dim")[0]
        block_dim_cout = tiling.get("block_dim")[2]
        block_dim_cin = tiling.get("block_dim")[1]

        sch[grads_matrix].set_scope(scope_cbuf)
        sch[grads_matrix].storage_align(sch[grads_matrix].op.axis[1],
                                        CUBE_MUL_SHAPE, 0)

        sch[grads_fractal].set_scope(scope_ca)
        sch[grads_fractal].buffer_align((1, 1), (1, 1), (1, 1), (1, CUBE_DIM),
                                        (1, CUBE_DIM))

        # fmap_shape_original_matrix is (batch_size*grads_depth,
        #                               grads_height*grads_width,
        #                               kernel_depth*fmap_channel_1,
        #                               kernel_height,
        #                               kernel_width,
        #                               C0_fmap)
        if not flag_all_one_case:
            sch[fmap_l1].set_scope(scope_cbuf)
            sch[fmap_matrix].buffer_align(
                (1, 1), (width_grads, width_grads), (1, 1),
                (kernel_height, kernel_height), (kernel_width, kernel_width),
                (1, CUBE_DIM))
        else:
            sch[fmap_matrix].storage_align(sch[fmap_matrix].op.axis[1],
                                           CUBE_MUL_SHAPE, 0)

        sch[fmap_matrix].set_scope(scope_cbuf)

        sch[fmap_fractal].set_scope(scope_cb)
        sch[fmap_fractal].buffer_align((1, 1), (1, 1), (1, 1), (1, CUBE_DIM),
                                       (1, CUBE_DIM))

        dw_cc, dw_ub, dw_ddr, dw_ub_reduce, dw_ddr_reduce = \
            _atomic_add(sch, dw_cc, dw_ub, dw_ddr)

        # #######################tiling parameters analyze####################
        batch_num_sc = batch_num // block_dim_batch

        full_k_in_l0a, full_k_in_l0b = _full_k_check()

        tiling_patrs_dict = _compute_tiling_parts()
        dw_tiling_factor = tiling_patrs_dict["dw_tiling_factor"]
        dw_tiling_nparts = tiling_patrs_dict["dw_tiling_nparts"]
        dw_ub_tiling_factor = tiling_patrs_dict["dw_ub_tiling_factor"]
        grads_l1_tiling_nparts = \
            tiling_patrs_dict["grads_l1_tiling_nparts"]
        fmap_l1_tiling_nparts = tiling_patrs_dict["fmap_l1_tiling_nparts"]
        l1_2_l0_tiling_nparts = tiling_patrs_dict["l1_2_l0_tiling_nparts"]
        dw_k = tiling_patrs_dict["dw_k"]

        # #############################split axis N##########################
        c_fmap_multicore, c_fmap_mad_at \
            = sch[dw_ddr].split(sch[dw_ddr].op.axis[0], nparts=block_dim_cin)

        c_fmap_mad_at, c_fmap_mad_insn \
            = sch[dw_ddr].split(c_fmap_mad_at, nparts=dw_tiling_nparts[0])

        c_fmap_l1_ori, c_fmap_mad_at \
            = sch[dw_ddr].split(c_fmap_mad_at, nparts=fmap_l1_tiling_nparts[1])

        def _ddr_n_split():
            # for N axis, if Hk and Wk needs split, do explict split
            if not flag_all_one_case:
                if tiling.get("BL1_shape"):
                    nc_cc = tiling.get("CL0_matrix")[0] * \
                            tiling.get("BL1_shape")[1]
                else:
                    nc_cc = kernel_depth * c1_fmap * kernel_width * kernel_height // block_dim_cin

                factor_kw = ceil_div(kernel_width, nc_cc)
                factor_kh = ceil_div(kernel_width * kernel_height,
                                     nc_cc) // factor_kw

                c_fmap_l1_out, c_fmap_l1_at \
                    = sch[dw_ddr].split(c_fmap_l1_ori, factor_kw)

                c_fmap_l1_c1, c_fmap_l1_kh \
                    = sch[dw_ddr].split(c_fmap_l1_out, factor_kh)
            else:
                c_fmap_l1_out, c_fmap_l1_at \
                    = sch[dw_ddr].split(c_fmap_l1_ori, 1)

                c_fmap_l1_c1, c_fmap_l1_kh \
                    = sch[dw_ddr].split(c_fmap_l1_out, 1)
            return c_fmap_l1_c1, c_fmap_l1_kh, c_fmap_l1_at

        c_fmap_l1_c1, c_fmap_l1_kh, c_fmap_l1_at = _ddr_n_split()

        # split axis M
        c_grads_mad_at, c_grads_mad_insn \
            = sch[dw_ddr].split(sch[dw_ddr].op.axis[1],
                                dw_tiling_factor[1]*CUBE_DIM)

        c_grads_multicore, c_grads_mad_at \
            = sch[dw_ddr].split(c_grads_mad_at, nparts=block_dim_cout)

        c_grads_l1_at, c_grads_mad_at = \
            sch[dw_ddr].split(c_grads_mad_at, nparts=grads_l1_tiling_nparts[1])

        # reorder according to requirments of mmad EmitInsn
        sch[dw_ddr].reorder(sch[dw_ddr].op.reduce_axis[0], c_grads_multicore,
                            c_fmap_multicore, c_fmap_l1_c1, c_fmap_l1_kh,
                            c_fmap_l1_at, c_grads_l1_at, c_fmap_mad_at,
                            c_grads_mad_at, c_fmap_mad_insn, c_grads_mad_insn)

        def _ub_and_cc_attach():
            # optimization by move small loops to outer
            reorder_flag = False
            # during L1 to L0, if M loop is smaller, then move to outer
            if l1_2_l0_tiling_nparts[0] > l1_2_l0_tiling_nparts[1]:
                sch[dw_ddr].reorder(c_grads_mad_at, c_fmap_mad_at)
                reorder_flag = True
            # during sc to L1, if M loop is smaller, then move to outer
            if fmap_l1_tiling_nparts[1] > grads_l1_tiling_nparts[1]:
                sch[dw_ddr].reorder(c_grads_l1_at, c_fmap_l1_c1, c_fmap_l1_kh,
                                    c_fmap_l1_at)

            # dw_ub attach
            # dw_ub split
            c_fmap_2_ub_at, c_fmap_2_ub_insn \
                = sch[dw_ddr].split(c_fmap_mad_insn, dw_ub_tiling_factor[0])
            # dw_ub attach
            sch[dw_ub].compute_at(sch[dw_ddr], c_fmap_2_ub_at)

            # dw attach
            if reorder_flag:
                sch[dw_cc].compute_at(sch[dw_ddr], c_fmap_mad_at)
            else:
                sch[dw_cc].compute_at(sch[dw_ddr], c_grads_mad_at)
            return c_fmap_2_ub_insn

        c_fmap_2_ub_insn = _ub_and_cc_attach()

        # dw_cc split
        # get the 3 reduce axis of dw_cc
        batch_axis_sc, k_1_axis_sc, k_0 = sch[dw_cc].op.reduce_axis

        # dw_k is the part for one MMAD
        hw_mad_1_mad_at, hw_mad_1_mad_insn \
            = sch[dw_cc].split(k_1_axis_sc, dw_k)

        # mad_pattern :2 , the 1st axis should be 1, so do a fake split
        batch_insn_o, batch_insn = sch[dw_cc].split(batch_axis_sc, 1)

        # K of AL1 and BL1 can be different, there are 2 split methods
        # on which one is larger
        if grads_l1_tiling_nparts[0] > fmap_l1_tiling_nparts[0]:
            hw_mad_1_l1_at, hw_mad_1_mad_at = sch[dw_cc].split(
                hw_mad_1_mad_at, nparts=grads_l1_tiling_nparts[0])
            hw_mad_1_l1_out_at, hw_mad_1_l1_in_at = sch[dw_cc].split(
                hw_mad_1_l1_at, nparts=fmap_l1_tiling_nparts[0])
            al1_at_axis = hw_mad_1_l1_in_at
            bl1_at_axis = hw_mad_1_l1_out_at
        else:
            hw_mad_1_l1_at, hw_mad_1_mad_at = sch[dw_cc].split(
                hw_mad_1_mad_at, nparts=fmap_l1_tiling_nparts[0])
            hw_mad_1_l1_out_at, hw_mad_1_l1_in_at = sch[dw_cc].split(
                hw_mad_1_l1_at, nparts=grads_l1_tiling_nparts[0])
            al1_at_axis = hw_mad_1_l1_out_at
            bl1_at_axis = hw_mad_1_l1_in_at

        # split dw_cc.op.axis[0](N1), factor is one MMAD
        fkk_mad_at, fkk_mad_insn \
            = sch[dw_cc].split(sch[dw_cc].op.axis[1], dw_tiling_factor[0])

        # split dw_cc.op.axis[1](M1*M0), factor is one MMAD
        lc_mad_at, lc_mad_insn \
            = sch[dw_cc].split(sch[dw_cc].op.axis[2],
                               dw_tiling_factor[1] * CUBE_DIM)

        sch[dw_cc].reorder(fkk_mad_at, lc_mad_at, sch[dw_cc].op.axis[0],
                           batch_insn_o, hw_mad_1_l1_out_at, hw_mad_1_l1_in_at,
                           hw_mad_1_mad_at, batch_insn, fkk_mad_insn,
                           lc_mad_insn, sch[dw_cc].op.axis[3],
                           hw_mad_1_mad_insn, k_0)

        # #############################multi core#############################
        def _bind_core():
            fused_multi_core = \
                sch[dw_ddr].fuse(sch[dw_ddr].op.reduce_axis[0],
                                 c_grads_multicore, c_fmap_multicore)
            fused_multi_core, pragma_at = \
                sch[dw_ddr].split(fused_multi_core, 1)
            block = tvm.thread_axis("blockIdx.x")
            sch[dw_ddr].bind(fused_multi_core, block)
            blocks =\
                block_dim_batch * block_dim_cin * block_dim_cout * block_dim_hw
            if blocks == block_dim_batch:
                sch[dw_ddr].pragma(pragma_at, 'json_info_batchBindOnly')
            return fused_multi_core

        fused_multi_core = _bind_core()
        _l0_attach()
        _al1_attach()
        _bl1_attach()
        _double_buffer()
        _emit_insn()

        return True
