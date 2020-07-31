#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright 2018 Huawei Technologies Co., Ltd
"""
import math

from te import platform as cce
from te import tvm
from te.platform import intrinsic_check_support
from te.platform import get_soc_spec
from te.platform import cce_emitinsn_params as cce_params
from te.platform.cce_conf import CceProductParams
from te.lang.cce import ConvParam # pylint: disable=C0412
from te.platform.cce_policy import get_L1_info
from .max_pool2d_3_3_2_2_schedule import schedule as schedule_max_pool_3_3_2_2
from .max_pool2d_3_3_1_1_schedule import schedule as schedule_max_pool_3_3_1_1

# define the quantize tensor name
CAST_F16_NAME = "cast_f16_ub"
INPUT_NAME = "input_ub"
VMULS_REFORM_NAME = "reform_by_vmuls"
SQRT_NAME = "scale_sqrt_ub"
OFFSET_NAME = "offset_ub"
CAST_I8_NAME = "cast_i8_ub"
VADDS_REFORM_NAME = "reform_by_vadds"

SIZE_OF_FP16 = 2
BLOCK_SIZE = 16
INT8_ALIGN = 32
SIZE_OF_FP32 = 4
MAX_VALUE_OF_16BIT = 65535
ASCEND_QUANT_TAG = "quant"
POOLING2D_TAG_PREFIX = "pooling2d_"
ASCEND_ANTI_QUANT_TAG = "anti_quant"


def _get_l1fusion_device_core_num(is_l1fusion):
    """
    get the device core num
    :param is_l1fusion: is l1 fusion
    :return: device core num
    """
    if is_l1fusion:
        device_core_num = 1
    else:
        device_core_num = get_soc_spec("CORE_NUM")
    return device_core_num


# Example: pylint: disable=R0914,R0915
def pooling2d_tiling(pooling_params, fusion_params=None):
    """
    :params:
    :pooling_params: input params for tiling
    :return: valid tiling params
    """
    pooling_mode = pooling_params["pooling_mode"]
    padding_mode = pooling_params["padding_mode"]

    in_size_h = pooling_params["in_size_h"]
    in_size_w = pooling_params["in_size_w"]

    c1_value = pooling_params["c1_value"]
    c_block_size = pooling_params["c_block_size"]

    window_h = pooling_params["window_h"]
    window_w = pooling_params["window_w"]

    # in SAME mode, pad is a tvm.expr
    pad_top = pooling_params["pad_top"]
    pad_bottom = pooling_params["pad_bottom"]

    stride_h = pooling_params["stride_h"]
    out_size_h = pooling_params["out_size_h"]
    out_size_w = pooling_params["out_size_w"]

    batch_size = pooling_params["batch_size"]

    fused_ascend_quant = pooling_params["fused_ascend_quant"]
    fused_anti_quant = pooling_params["fused_anti_quant"]

    # get available ub size
    ub_size = get_soc_spec("UB_SIZE")
    l1_size = get_soc_spec("L1_SIZE")

    tiling = {}
    gm_to_l1_steph_list = []
    gm_to_l1_steph_c1_factor_list = []
    l1_cut_to_ub_factor_list = []
    l1_cut_to_ub_c1_factor_list = []
    fmap_img2col_choose_list = []
    step_h_choose_list = []

    if fusion_params:
        in_l1_flag = fusion_params.get("in_l1_flag", False)
        l1_fusion_type = fusion_params.get("l1_fusion_type", -1)
    else:
        in_l1_flag = False
        l1_fusion_type = -1
    is_l1fusion = l1_fusion_type in (0, 1)

    can_enable_double_buffer = False
    device_core_num = _get_l1fusion_device_core_num(is_l1fusion)
    enabler_c1_bind_core = \
        batch_size < device_core_num and batch_size < c1_value

    def _clean_residue():
        tiling.clear()
        del gm_to_l1_steph_list[:]
        del gm_to_l1_steph_c1_factor_list[:]
        del l1_cut_to_ub_factor_list[:]
        del l1_cut_to_ub_c1_factor_list[:]
        del fmap_img2col_choose_list[:]
        del step_h_choose_list[:]

    # pylint: too-many-locals, too-many-branches, too-many-statements
    def _try_tiling(ub_tiling_size, l1_tiling_size):
        _clean_residue()

        def check_anti_quant(h_factor, c1_factor):
            if fused_anti_quant:
                n_count = c1_factor*h_factor*in_size_w*BLOCK_SIZE
                return n_count*SIZE_OF_FP16*2 <= ub_tiling_size
            return True

        def check_c1_factor(c1_factor):
            if fused_anti_quant and c1_factor % 2 != 0:
                return False
            if fused_ascend_quant and c1_value != 1 and c1_factor % 2 != 0:
                return False
            return True

        # pylint: too-many-nested-blocks
        def _find_tiling(need_cut_c1=False):
            """
            :gm -> l1 find the max cut
            :make sure Oh*Ow is aligned to c_block_size, so after img2col,
            :it can be split to Z fractal in H dim
            :start from step_h = 1 means the first cut
            :param need_cut_c1: cutC1 or not
            :return: valid tiling params
            """
            del gm_to_l1_steph_list[:]

            for step_h in range(1, out_size_h + 1):
                if step_h * out_size_w % c_block_size == 0:
                    gm_to_l1_cut_h = (step_h - 1) * stride_h + window_h

                    if need_cut_c1:
                        for ci_factor in range(c1_value, 0, -1):
                            if not check_c1_factor(ci_factor):
                                continue

                            # make sure nRepeat param of img2col_l1_ub in [0,255]
                            if ci_factor * window_h * window_w > 255:
                                continue

                            fmap_l1_size = gm_to_l1_cut_h * in_size_w * \
                                           ci_factor * c_block_size * \
                                           SIZE_OF_FP16

                            if fmap_l1_size <= l1_tiling_size:
                                gm_to_l1_steph_list.append(step_h)
                                gm_to_l1_steph_c1_factor_list.append(ci_factor)
                                break
                    else:
                        fmap_l1_size = gm_to_l1_cut_h * in_size_w * c1_value * \
                                       c_block_size * SIZE_OF_FP16

                        if fmap_l1_size <= l1_tiling_size:
                            gm_to_l1_steph_list.append(step_h)
                        else:
                            break

            if need_cut_c1 and not gm_to_l1_steph_list:
                # can not load in ub one time, try to cut_C1
                # and set need_cut_c1 flag
                if out_size_h < c_block_size:
                    bind_core_gap = max(c1_value, device_core_num)
                    for ci_factor in range(c1_value, 0, -1):
                        if not check_c1_factor(ci_factor):
                            continue

                        # make sure nRepeat param of img2col_l1_ub in [0,255]
                        if ci_factor * window_h * window_w > 255:
                            continue

                        fmap_img2col_w = \
                            window_h*window_w*ci_factor*c_block_size
                        fmap_img2col_h = \
                            (out_size_h*out_size_w + c_block_size - 1) // \
                            c_block_size*c_block_size
                        res_w = ci_factor*c_block_size
                        res_h = (out_size_h*out_size_w + c_block_size - 1) // \
                                c_block_size * c_block_size
                        avg_factor_dump_w = c_block_size
                        avg_factor_dump_h = \
                            (out_size_h*out_size_w + c_block_size - 1) // \
                            c_block_size*c_block_size

                        fmap_img2col_size = \
                            fmap_img2col_w*fmap_img2col_h*SIZE_OF_FP16
                        # pooling_out_ub & pooling_ub_5hd
                        res_size = res_h*res_w*SIZE_OF_FP16*2
                        avg_factor_dump_size = \
                            (avg_factor_dump_w*avg_factor_dump_h + 127) // \
                            128*128*SIZE_OF_FP16

                        if pooling_mode == "AVG" and padding_mode == "SAME":
                            # tmp_buffer is exist, the same size as
                            # avg_factor_dump_size, so mul 2 here
                            l1_cut_to_ub_unit_size = \
                                fmap_img2col_size + res_size + \
                                avg_factor_dump_size*2
                        else:
                            l1_cut_to_ub_unit_size = \
                                fmap_img2col_size + res_size

                        if fused_ascend_quant:
                            # for c0 rearrange
                            l1_cut_to_ub_unit_size += res_size

                        size_h_gm_to_l1 = in_size_h
                        if out_size_h == 1:
                            size_h_gm_to_l1 = window_h - pad_top
                        fmap_l1_size = size_h_gm_to_l1 * in_size_w * \
                                       ci_factor * c_block_size * SIZE_OF_FP16

                        is_l1_enough = \
                            fmap_l1_size <= l1_tiling_size and \
                            l1_cut_to_ub_unit_size <= \
                            float(ub_tiling_size) and \
                            check_anti_quant(size_h_gm_to_l1, ci_factor)
                        if is_l1_enough:
                            if enabler_c1_bind_core:
                                cur_bind_core_gap = \
                                    abs((c1_value + ci_factor - 1)//ci_factor -
                                        device_core_num)
                                if cur_bind_core_gap >= bind_core_gap:
                                    continue
                                bind_core_gap = cur_bind_core_gap

                            tiling["gm_to_l1_cut_h"] = in_size_h
                            l1_cut_to_ub_factor = (out_size_h * out_size_w +
                                                   c_block_size - 1) // \
                                                  c_block_size
                            tiling["step_h"] = out_size_h
                            tiling["axis_orig"] = l1_cut_to_ub_factor
                            tiling["l1_cut_to_ub_factor"] = l1_cut_to_ub_factor

                            tiling["cutc1_factor"] = ci_factor
                            tiling["cut_flag"] = "NO_CUT"
                            tiling["NO_CutH_FLAG"] = True

                            is_bind_core = \
                                not enabler_c1_bind_core or \
                                enabler_c1_bind_core and bind_core_gap == 0
                            if is_bind_core:
                                break

                    if "cutc1_factor" not in tiling:
                        tiling["is_find_tiling"] = False

                    return

            def _find_tiling_cuth():
                """
                :cutH tiling
                :return: valid cutH tiling params
                """
                max_l1_cut_to_ub_factor = 1
                for step_h in gm_to_l1_steph_list:
                    axis_outer_split_factor = step_h * out_size_w // \
                                              c_block_size
                    h_factor = (step_h - 1) * stride_h + window_h
                    axis_orig = axis_outer_split_factor

                    while axis_outer_split_factor > 0:
                        if axis_orig % axis_outer_split_factor != 0:
                            axis_outer_split_factor = \
                                axis_outer_split_factor - 1
                            continue

                        fmap_img2col_w = window_h * window_w * c1_value * \
                                         c_block_size
                        fmap_img2col_h = axis_outer_split_factor * c_block_size
                        align_c1 = (c1_value + 1)//2*2
                        res_w = \
                            (align_c1 if fused_ascend_quant else c1_value) * \
                            c_block_size
                        res_h = step_h*out_size_w
                        avg_factor_dump_w = c_block_size
                        avg_factor_dump_h = step_h * out_size_w

                        fmap_img2col_size = \
                            fmap_img2col_w*fmap_img2col_h*SIZE_OF_FP16
                        # pooling_out_ub & pooling_ub_5hd
                        res_size = res_h*res_w*SIZE_OF_FP16*2
                        avg_factor_dump_size = \
                            ((avg_factor_dump_w*avg_factor_dump_h + \
                            127)//128*128)*SIZE_OF_FP16

                        if pooling_mode == "AVG" and padding_mode == "SAME":
                            # tmp_buffer is exist, the same size as
                            # avg_factor_dump_size, so mul 2 here
                            l1_cut_to_ub_unit_size = \
                                fmap_img2col_size + res_size + \
                                avg_factor_dump_size*2
                        else:
                            # tmp_buffer is exist, the same size as res_size,
                            # so mul 2 here
                            l1_cut_to_ub_unit_size = \
                                fmap_img2col_size + res_size*2

                        if fused_ascend_quant:
                            # for c0 rearrange
                            l1_cut_to_ub_unit_size += res_size

                        if l1_cut_to_ub_unit_size >= float(ub_tiling_size) or \
                                not check_anti_quant(h_factor, c1_value):
                            axis_outer_split_factor -= 1
                            continue

                        if axis_outer_split_factor >= max_l1_cut_to_ub_factor:
                            max_l1_cut_to_ub_factor = axis_outer_split_factor
                            l1_cut_to_ub_factor = max_l1_cut_to_ub_factor
                            tiling["l1_cut_to_ub_factor"] = l1_cut_to_ub_factor
                            tiling["step_h"] = step_h
                            gm_to_l1_cut_h = (step_h - 1) * stride_h + window_h
                            tiling["gm_to_l1_cut_h"] = gm_to_l1_cut_h
                            tiling["fmap_img2col_size"] = fmap_img2col_size
                            tiling["axis_orig"] = axis_orig
                            l1_cut_to_ub_factor_list.append(l1_cut_to_ub_factor)
                            fmap_img2col_choose_list.append(fmap_img2col_size)
                            step_h_choose_list.append(step_h)

                        axis_outer_split_factor = axis_outer_split_factor - 1

            def _find_tiling_cuth_cutc1():
                """
                :cutH and cutC1 together.
                :return: valid tiling params.
                """
                max_l1_cut_to_ub_factor = 1
                bind_core_gap = max(c1_value, device_core_num)
                for i, _ in enumerate(gm_to_l1_steph_list):
                    step_h = gm_to_l1_steph_list[i]
                    h_factor = (step_h - 1) * stride_h + window_h
                    cutc1_factor_max = gm_to_l1_steph_c1_factor_list[i]
                    axis_outer_split_factor = \
                        step_h * out_size_w // c_block_size
                    axis_orig = axis_outer_split_factor

                    while axis_outer_split_factor > 0:
                        if axis_orig % axis_outer_split_factor != 0:
                            axis_outer_split_factor = \
                                axis_outer_split_factor - 1
                            continue

                        for cutc1_factor in range(cutc1_factor_max, 0, -1):
                            if not check_c1_factor(cutc1_factor):
                                continue

                            # make sure nRepeat param of img2col_l1_ub in [0,255]
                            if cutc1_factor * window_h * window_w > 255:
                                continue

                            fmap_img2col_w = window_h * window_w * \
                                             cutc1_factor * c_block_size
                            fmap_img2col_h = \
                                axis_outer_split_factor * c_block_size
                            res_w = cutc1_factor * c_block_size
                            res_h = step_h * out_size_w
                            avg_factor_dump_w = c_block_size
                            avg_factor_dump_h = \
                                ((step_h*out_size_w + 128 - 1)//128)*128

                            fmap_img2col_size = \
                                fmap_img2col_w*fmap_img2col_h*SIZE_OF_FP16
                            # pooling_out_ub & pooling_ub_5hd
                            res_size = res_h*res_w*SIZE_OF_FP16*2
                            avg_factor_dump_size = \
                                avg_factor_dump_w * avg_factor_dump_h * \
                                SIZE_OF_FP16

                            is_avg_same = pooling_mode == "AVG" and \
                                          padding_mode == "SAME"
                            if is_avg_same:
                                l1_cut_to_ub_unit_size = \
                                    fmap_img2col_size + res_size + \
                                    avg_factor_dump_size*2
                            else:
                                l1_cut_to_ub_unit_size = \
                                    fmap_img2col_size + res_size*2

                            if fused_ascend_quant:
                                # for c0 rearrage in quantion case
                                l1_cut_to_ub_unit_size += res_size

                            invalid_size = l1_cut_to_ub_unit_size >= \
                                           float(ub_tiling_size) or \
                                           not check_anti_quant(h_factor,
                                                                cutc1_factor)
                            if invalid_size:
                                continue

                            if enabler_c1_bind_core:
                                cur_bind_core_gap = \
                                    abs((c1_value + cutc1_factor - 1) // \
                                        cutc1_factor - device_core_num)
                                if cur_bind_core_gap > bind_core_gap:
                                    continue
                                if cur_bind_core_gap == bind_core_gap and \
                                    axis_outer_split_factor < \
                                    max_l1_cut_to_ub_factor:
                                    continue
                                bind_core_gap = cur_bind_core_gap
                            else:
                                if axis_outer_split_factor < \
                                    max_l1_cut_to_ub_factor:
                                    continue

                            max_l1_cut_to_ub_factor = axis_outer_split_factor
                            l1_cut_to_ub_factor = max_l1_cut_to_ub_factor
                            tiling["l1_cut_to_ub_factor"] = l1_cut_to_ub_factor
                            tiling["step_h"] = step_h
                            gm_to_l1_cut_h = (step_h - 1)*stride_h + window_h
                            tiling["gm_to_l1_cut_h"] = gm_to_l1_cut_h
                            tiling["fmap_img2col_size"] = fmap_img2col_size
                            tiling["axis_orig"] = axis_orig
                            tiling["cutc1_factor"] = cutc1_factor
                            l1_cut_to_ub_factor_list.append(
                                l1_cut_to_ub_factor)
                            fmap_img2col_choose_list.append(fmap_img2col_size)
                            step_h_choose_list.append(step_h)
                            l1_cut_to_ub_c1_factor_list.append(cutc1_factor)

                        axis_outer_split_factor = axis_outer_split_factor - 1

            if need_cut_c1:
                _find_tiling_cuth_cutc1()
            else:
                _find_tiling_cuth()

            if need_cut_c1 and not fmap_img2col_choose_list:
                tiling["is_find_tiling"] = False

        need_cut_c1 = False

        if enabler_c1_bind_core or c1_value * window_h * window_w > 255:
            need_cut_c1 = True

        if not need_cut_c1:
            _find_tiling()

            if not gm_to_l1_steph_list:
                if out_size_h < c_block_size:
                    fmap_img2col_w = window_h*window_w*c1_value*c_block_size
                    fmap_img2col_h = \
                        (out_size_h*out_size_w + c_block_size - 1) // \
                        c_block_size * c_block_size
                    align_c1 = (c1_value + 1)//2*2
                    res_w = \
                        (align_c1 if fused_ascend_quant else c1_value) * \
                        c_block_size
                    res_h = \
                        (out_size_h*out_size_w + c_block_size - 1) // \
                        c_block_size*c_block_size
                    avg_factor_dump_w = c_block_size
                    avg_factor_dump_h = \
                        (out_size_h * out_size_w + c_block_size - 1) // \
                        c_block_size*c_block_size

                    fmap_img2col_size = \
                        fmap_img2col_w*fmap_img2col_h*SIZE_OF_FP16
                    # pooling_out_ub & pooling_ub_5hd
                    res_size = res_h*res_w*SIZE_OF_FP16*2
                    avg_factor_dump_size = \
                        (avg_factor_dump_w * avg_factor_dump_h + 127) // 128 \
                        * 128 * SIZE_OF_FP16

                    if pooling_mode == "AVG" and padding_mode == "SAME":
                        # tmp_buffer is exist, the same size as
                        # avg_factor_dump_size, so mul 2 here
                        l1_cut_to_ub_unit_size = \
                            fmap_img2col_size + res_size + \
                            avg_factor_dump_size*2
                    else:
                        l1_cut_to_ub_unit_size = fmap_img2col_size + res_size

                    if fused_ascend_quant:
                        # for c0 rearrange
                        l1_cut_to_ub_unit_size += res_size

                    size_h_gm_to_l1 = in_size_h
                    if out_size_h == 1:
                        size_h_gm_to_l1 = window_h - pad_top
                    fmap_l1_size = \
                        size_h_gm_to_l1*in_size_w * \
                        c1_value*c_block_size*SIZE_OF_FP16
                    # Oh*Ow is not 16 multiple and
                    # l1_cut_to_ub_unit_size less equal ub_tiling_size
                    # means no need cut, can be load in ub one time
                    if fmap_l1_size < l1_tiling_size and \
                            l1_cut_to_ub_unit_size <= \
                            float(ub_tiling_size) and \
                            check_anti_quant(size_h_gm_to_l1, c1_value):
                        tiling["gm_to_l1_cut_h"] = in_size_h
                        l1_cut_to_ub_factor = (out_size_h * out_size_w +
                                               c_block_size - 1) // c_block_size
                        tiling["step_h"] = out_size_h
                        tiling["axis_orig"] = l1_cut_to_ub_factor
                        tiling["l1_cut_to_ub_factor"] = l1_cut_to_ub_factor
                        tiling["cut_flag"] = "NO_CUT"
                        tiling["NO_CutH_FLAG"] = True
                    # can not load in ub one time, try to cut_C1
                    # and set need_cut_c1 flag
                    else:
                        need_cut_c1 = True
                else:
                    need_cut_c1 = True
            else:
                if not l1_cut_to_ub_factor_list:
                    need_cut_c1 = True

        if need_cut_c1:
            _find_tiling(need_cut_c1)

        # Example: pylint: disable=R0912
        def _find_tiling_not_cutl1_2_ub(is_cut_each_c1=False):
            """
            :cutCi and cutH together, but no need cut l1 to ub
            :img2col all the data in l1 to ub
            :return: valid pooling params
            """
            if fused_ascend_quant and is_cut_each_c1:
                return
            bind_core_gap = max(c1_value, device_core_num)
            for ci_factor in range(c1_value, 0, -1):
                is_continue = is_cut_each_c1 and ci_factor != 1
                if is_continue:
                    continue
                # make sure nRepeat param of img2col_l1_ub in [0,255]
                ci_factor_valid = check_c1_factor(ci_factor) and \
                                  ci_factor*window_h*window_w <= 255

                if not ci_factor_valid:
                    continue

                is_avg_same = pooling_mode == "AVG" and padding_mode == "SAME"
                if is_avg_same:
                    img2col_w = window_h*window_w*ci_factor*c_block_size + \
                                ci_factor*c_block_size*2 + c_block_size*2
                else:
                    img2col_w = window_h*window_w*ci_factor*c_block_size +\
                                ci_factor*c_block_size*3

                if fused_ascend_quant:
                    # for c0 rearrange
                    img2col_w += ci_factor*c_block_size*2

                img2col_h_max = float(ub_tiling_size)/(SIZE_OF_FP16*img2col_w)
                img2col_h = int(
                    math.floor(img2col_h_max / c_block_size)) * c_block_size

                is_continue = (img2col_h_max < out_size_w) or \
                              (img2col_h < out_size_w)
                if is_continue:
                    continue

                while img2col_h - c_block_size > out_size_w:
                    img2col_h = img2col_h - c_block_size

                out_cuth = img2col_h // out_size_w

                if img2col_h % out_size_w != 0:
                    out_cuth += 1

                # this case, the gm_to_l1 will cross three line
                # in case out_size_w % (img2col_h % out_size_w) != 0:
                out_cuth += 1

                in_cuth = (out_cuth - 1) * stride_h + window_h

                gm_to_l1_size = ci_factor * in_cuth * in_size_w * \
                                c_block_size * SIZE_OF_FP16

                is_continue = (gm_to_l1_size > l1_tiling_size) or \
                              (not check_anti_quant(in_cuth, ci_factor))
                if is_continue:
                    continue

                if enabler_c1_bind_core:
                    cur_bind_core_gap = \
                        abs((c1_value + ci_factor - 1)//ci_factor - \
                            device_core_num)
                    if cur_bind_core_gap >= bind_core_gap:
                        continue
                    bind_core_gap = cur_bind_core_gap

                tiling["gm_to_l1_cut_h"] = in_cuth
                tiling["step_h"] = 0
                tiling["axis_orig"] = img2col_h//c_block_size
                tiling["l1_cut_to_ub_factor"] = img2col_h//c_block_size
                tiling["cutc1_factor"] = ci_factor
                tiling["is_find_tiling"] = True

                is_break = not enabler_c1_bind_core or \
                    (enabler_c1_bind_core and bind_core_gap == 0)
                if is_break:
                    break

        def _find_tiling_cut_c1_then_cut_howo():  # pylint: disable=R0914,R0915
            cut_l1_to_ub_factor = 1
            ho_wo_16_outer_factor = (out_size_h * out_size_w +
                                     c_block_size - 1) // c_block_size
            ho_wo_16_outer_factor_ori = ho_wo_16_outer_factor

            is_fuction_find_tilling = False
            while ho_wo_16_outer_factor > 0:
                if ho_wo_16_outer_factor_ori % ho_wo_16_outer_factor != 0:
                    ho_wo_16_outer_factor = ho_wo_16_outer_factor - 1
                    continue

                for cut_c1_factor in range(c1_value, 0, -1):
                    # when c1_value is 1, ci_factor can equals 1
                    is_continue = (fused_ascend_quant and \
                                            c1_value != 1 and \
                                            cut_c1_factor % 2 != 0) or \
                                            cut_c1_factor * window_h * \
                                            window_w > 255
                    if is_continue:
                        continue

                    img2col_fmap_columns = window_h * window_w * \
                                     cut_c1_factor * c_block_size
                    img2col_fmap_rows = ho_wo_16_outer_factor * c_block_size
                    res_weight = cut_c1_factor * c_block_size
                    res_height = out_size_h * out_size_w
                    avg_factor_dump_weight = c_block_size
                    avg_factor_dump_height = ((out_size_h * out_size_w +
                                               128 - 1) // 128) * 128
                    fmap_img2col_size = img2col_fmap_columns * \
                                        img2col_fmap_rows * \
                                        SIZE_OF_FP16
                    # pooling_out_ub & pooling_ub_5hd
                    pooling_res_size = res_height * res_weight * \
                                       SIZE_OF_FP16 * 2
                    avg_factor_dump_size = (avg_factor_dump_weight *
                                            avg_factor_dump_height) * \
                                           SIZE_OF_FP16

                    is_avg_same = pooling_mode == "AVG" and \
                                  padding_mode == "SAME"
                    if is_avg_same:
                        l1_to_ub_unit_size = fmap_img2col_size + \
                                                 pooling_res_size + \
                                                 avg_factor_dump_size * 2
                    else:
                        l1_to_ub_unit_size = fmap_img2col_size + \
                                                 pooling_res_size * 2

                    if l1_to_ub_unit_size >= float(ub_tiling_size):
                        continue

                    if enabler_c1_bind_core:
                        c1_bind_core_gap = abs((c1_value + cut_c1_factor - 1)
                                               // cut_c1_factor -
                                               device_core_num)
                        is_continue = (c1_bind_core_gap > bind_core_gap) or \
                                      (c1_bind_core_gap == bind_core_gap and
                                       ho_wo_16_outer_factor <
                                       cut_l1_to_ub_factor)
                        if is_continue:
                            continue
                        bind_core_gap = c1_bind_core_gap
                    else:
                        if ho_wo_16_outer_factor < cut_l1_to_ub_factor:
                            continue

                    cut_l1_to_ub_factor = ho_wo_16_outer_factor
                    l1_cut_to_ub_factor = cut_l1_to_ub_factor
                    tiling["l1_cut_to_ub_factor"] = l1_cut_to_ub_factor
                    tiling["step_h"] = out_size_h
                    gm_to_l1_cut_h = (out_size_h - 1) * stride_h + window_h
                    tiling["gm_to_l1_cut_h"] = gm_to_l1_cut_h
                    tiling["fmap_img2col_size"] = fmap_img2col_size
                    tiling["axis_orig"] = ho_wo_16_outer_factor_ori
                    tiling["cutc1_factor"] = cut_c1_factor
                    tiling["is_find_tiling"] = True
                    l1_cut_to_ub_factor_list.append(l1_cut_to_ub_factor)
                    fmap_img2col_choose_list.append(fmap_img2col_size)
                    step_h_choose_list.append(out_size_h)
                    l1_cut_to_ub_c1_factor_list.append(cut_c1_factor)
                    is_fuction_find_tilling = True

                ho_wo_16_outer_factor = ho_wo_16_outer_factor - 1
            return is_fuction_find_tilling

        is_cut_l1_to_ub = True
        if "is_find_tiling" in tiling:
            if is_l1fusion and in_l1_flag:
                # this kind of tilling will no cut H, will firstly cut H0W0,
                # then cut ci to find the right tilling
                result_cut_c1_then_cut_howo = \
                    _find_tiling_cut_c1_then_cut_howo()
                if not result_cut_c1_then_cut_howo:
                    is_cut_l1_to_ub = False
                    if not (fused_ascend_quant and c1_value != 1):
                        _find_tiling_not_cutl1_2_ub(True)
            else:
                is_cut_l1_to_ub = False
                _find_tiling_not_cutl1_2_ub()

            if "is_find_tiling" in tiling:
                is_find_tiling = tiling["is_find_tiling"]

                if not is_find_tiling:
                    tiling["is_cut_l1_to_ub"] = is_cut_l1_to_ub
                    tiling["need_cut_c1"] = need_cut_c1
                    return False

        tiling["is_cut_l1_to_ub"] = is_cut_l1_to_ub
        tiling["need_cut_c1"] = need_cut_c1
        return True

    res_try_tiling = _try_tiling(ub_size//2, l1_size//2)
    if res_try_tiling:
        can_enable_double_buffer = True
    else:
        res_try_tiling = _try_tiling(ub_size, l1_size)
        if not res_try_tiling:
            raise RuntimeError("cutH and C1, can not find" +
                               " valid tiling params, cutW support needed")

    is_cut_l1_to_ub = tiling["is_cut_l1_to_ub"]
    need_cut_c1 = tiling["need_cut_c1"]

    # post precession after get tiling params
    step_h = tiling["step_h"]
    res_cut_factor = tiling["axis_orig"]
    cutc1_factor = tiling.get("cutc1_factor", c1_value)
    cuth_tile = tiling["gm_to_l1_cut_h"]
    l1_cut_to_ub_factor = tiling["l1_cut_to_ub_factor"]

    def _get_cuth_info(cuth_tile):
        cuth_stride = 0
        cut_flag = None

        if "NO_CutH_FLAG" in tiling:
            if tiling["NO_CutH_FLAG"] is True:
                cut_flag = tiling["cut_flag"]
                cuth_stride = cuth_tile - (window_h - stride_h)
                cuth_loop = 1
        else:
            if is_cut_l1_to_ub:
                cuth_stride = cuth_tile - (window_h - stride_h)
                cuth_loop = \
                    ((in_size_h + pad_top + pad_bottom - cuth_tile) /
                     float(cuth_stride)) + 1.0
            else:
                cuth_loop = (1.0 * out_size_h * out_size_w) / \
                            (res_cut_factor * c_block_size)

        if is_cut_l1_to_ub:
            if cuth_tile >= in_size_h + pad_top + pad_bottom:
                cuth_tile = in_size_h + pad_top + pad_bottom
                cuth_loop = 1

        return cut_flag, cuth_stride, cuth_loop, cuth_tile

    cut_flag, cuth_stride, cuth_loop, cuth_tile = _get_cuth_info(cuth_tile)

    cut_flag = "CUT"
    if 0 < cuth_loop <= 1:
        cut_flag = "NO_CUT"

    cuth_loop = int(math.ceil(cuth_loop))

    cce_params.cceEmitParamsIns.insert_param("cut_flag", cut_flag)
    cce_params.cceEmitParamsIns.insert_param("step_h", step_h)
    cce_params.cceEmitParamsIns.insert_param("cuth_tile", cuth_tile)
    cce_params.cceEmitParamsIns.insert_param("cuth_stride", cuth_stride)
    cce_params.cceEmitParamsIns.insert_param("cuth_loop", cuth_loop)
    cce_params.cceEmitParamsIns.insert_param("l1_cut_to_ub_factor",
                                             l1_cut_to_ub_factor)
    cce_params.cceEmitParamsIns.insert_param("need_cut_c1", need_cut_c1)
    cce_params.cceEmitParamsIns.insert_param("cutc1_factor", cutc1_factor)
    cce_params.cceEmitParamsIns.insert_param("is_cut_l1_to_ub",
                                             is_cut_l1_to_ub)

    result_list = (l1_cut_to_ub_factor, res_cut_factor,
                   is_cut_l1_to_ub, cutc1_factor,
                   can_enable_double_buffer, cuth_loop)
    return result_list


# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def pooling2d_global_tiling(pooling_params, fusion_params=None):
    """
    :params:
    :pooling_params: input params for tiling
    :return: valid tiling params
    """
    # get gap or gmp related params
    batch_size = pooling_params["batch_size"]
    c1_value = pooling_params["c1_value"]
    in_size_h = pooling_params["in_size_h"]
    in_size_w = pooling_params["in_size_w"]
    c_block_size = pooling_params["c_block_size"]
    pooling_mode = pooling_params["pooling_mode"]

    if fusion_params:
        l1_fusion_type = fusion_params.get("l1_fusion_type", -1)
    else:
        l1_fusion_type = -1
    is_l1fusion = l1_fusion_type in (0, 1)

    # get available ub size
    ub_size = get_soc_spec("UB_SIZE")
    device_core_num = _get_l1fusion_device_core_num(is_l1fusion)
    fused_ascend_quant = pooling_params["fused_ascend_quant"]

    enabler_c1_bind_core = \
        batch_size < device_core_num and batch_size < c1_value

    # get tiling params, cutCi_Hi_Wi
    tiling_params = {}

    # Get vadd vmul vconv ability
    vconv_ability = intrinsic_check_support("Intrinsic_vconv", "f162f32")
    vadd_ability = intrinsic_check_support("Intrinsic_vadd", "float32")
    vmul_ability = intrinsic_check_support("Intrinsic_vmul", "float32")
    fp32_ability = vconv_ability and \
                   vadd_ability and \
                   vmul_ability and \
                   (not get_soc_spec("SOC_VERSION") in ("Ascend310",))

    # pylint: disable=too-many-nested-blocks
    def _try_tiling(ub_size):
        tiling_params.clear()
        bind_core_gap = max(c1_value, device_core_num)
        for ci_factor in range(c1_value, 0, -1):
            if c1_value % ci_factor != 0:
                continue

            is_match = fused_ascend_quant and \
                       c1_value != 1 and ci_factor % 2 != 0
            if is_match:
                continue

            if pooling_mode == "GAP" and fp32_ability:
                # fp16 will cast to fp32 compute
                tensor_in_ub_size = ci_factor * c_block_size *\
                                    in_size_h * in_size_w *\
                                    (SIZE_OF_FP32 + SIZE_OF_FP16)
                result_in_ub_size = (ci_factor*c_block_size + 128 - 1) // \
                                    128 * 128 * SIZE_OF_FP32
            else:
                tensor_in_ub_size = ci_factor * c_block_size *\
                                    in_size_h * in_size_w * SIZE_OF_FP16
                result_in_ub_size = (ci_factor * c_block_size + 128 - 1) // \
                                    128 * 128 * SIZE_OF_FP16

            used_ub_size = tensor_in_ub_size + result_in_ub_size

            if fused_ascend_quant:
                used_ub_size += result_in_ub_size

            # srcStride of copy_gm_to_ubuf is uint16
            if used_ub_size > ub_size or in_size_h * in_size_w > 4096:
                continue

            if enabler_c1_bind_core:
                cur_bind_core_gap = \
                    abs((c1_value + ci_factor - 1) // ci_factor - \
                        device_core_num)
                if cur_bind_core_gap >= bind_core_gap:
                    continue
                bind_core_gap = cur_bind_core_gap

            cut_ci_factor = ci_factor
            cut_hi_factor = in_size_h
            cut_wi_factor = in_size_w
            tiling_params["cut_ci_factor"] = cut_ci_factor
            tiling_params["cut_hi_factor"] = cut_hi_factor
            tiling_params["cut_wi_factor"] = cut_wi_factor
            tiling_params["find_tiling"] = True

            if not enabler_c1_bind_core or \
                (enabler_c1_bind_core and bind_core_gap == 0):
                break

        if fused_ascend_quant and ci_factor == 1:
            ci_factor = 2

        if "find_tiling" not in tiling_params:
            # after cutCi when cut_ci_factor = 1, still can not load in ub,
            # need cutH and Ci together
            # let cutCi as big as it can be and let H as a loop axis,
            # it will be best for result data copy from ub to gm
            for hi_factor in range(in_size_h, 0, -1):
                if in_size_h % hi_factor != 0:
                    continue

                is_match = fused_ascend_quant and \
                           c1_value != 1 and ci_factor % 2 != 0
                if is_match:
                    continue

                # occupied ub size
                if pooling_mode == "GAP" and fp32_ability:
                    # fp16 will cast to fp32 compute
                    tensor_in_ub_size = \
                        ci_factor * hi_factor * in_size_w * c_block_size *\
                        (SIZE_OF_FP32 + SIZE_OF_FP16)
                    result_in_ub_size = \
                        ci_factor * ((c_block_size + 128 - 1) // 128) *\
                        128 * SIZE_OF_FP32
                else:
                    tensor_in_ub_size = \
                        ci_factor * hi_factor * in_size_w *\
                        c_block_size * SIZE_OF_FP16
                    result_in_ub_size = \
                        ci_factor * ((c_block_size + 128 - 1) // 128) *\
                         128 * SIZE_OF_FP16

                used_ub_size = tensor_in_ub_size + result_in_ub_size

                if fused_ascend_quant:
                    used_ub_size += result_in_ub_size

                # lenBurst of copy_gm_to_ubuf is uint16
                if used_ub_size < ub_size and hi_factor * in_size_w <= 4096:
                    cut_ci_factor = ci_factor
                    cut_hi_factor = hi_factor
                    cut_wi_factor = in_size_w
                    tiling_params["cut_ci_factor"] = cut_ci_factor
                    tiling_params["cut_hi_factor"] = cut_hi_factor
                    tiling_params["cut_wi_factor"] = cut_wi_factor
                    tiling_params["find_tiling"] = True
                    break
                continue

            if "find_tiling" not in tiling_params:
                for wi_factor in range(in_size_w, 0, -1):
                    # occupied ub size
                    if pooling_mode == "GAP" and fp32_ability:
                        # fp16 will cast to fp32 compute
                        tensor_in_ub_size = ci_factor*wi_factor*c_block_size\
                                            * (SIZE_OF_FP32 + SIZE_OF_FP16)
                        result_in_ub_size = \
                            ci_factor*((c_block_size + 128 - 1) \
                            // 128)*128*SIZE_OF_FP32
                    else:
                        tensor_in_ub_size = ci_factor * wi_factor\
                                            * c_block_size * SIZE_OF_FP16
                        result_in_ub_size = \
                            ci_factor * \
                            ((c_block_size + 128 - 1) // 128) * \
                            128*SIZE_OF_FP16

                    used_ub_size = tensor_in_ub_size + result_in_ub_size

                    if fused_ascend_quant:
                        used_ub_size += result_in_ub_size

                    # lenBurst of copy_gm_to_ubuf is uint16
                    if used_ub_size < ub_size and wi_factor <= 4096:
                        cut_ci_factor = 1
                        cut_hi_factor = 1
                        cut_wi_factor = wi_factor
                        tiling_params["cut_ci_factor"] = cut_ci_factor
                        tiling_params["cut_hi_factor"] = cut_hi_factor
                        tiling_params["cut_wi_factor"] = cut_wi_factor
                        tiling_params["find_tiling"] = True
                        break

    _try_tiling(ub_size//2)

    enable_double_buffer = True

    # disable double buffer and try tiling use
    # whole ub if can not find valid tiling
    if "find_tiling" not in tiling_params:
        enable_double_buffer = False
        _try_tiling(ub_size)

    if "find_tiling" in tiling_params:
        cut_ci_factor = tiling_params["cut_ci_factor"]
        cut_hi_factor = tiling_params["cut_hi_factor"]
        cut_wi_factor = tiling_params["cut_wi_factor"]
    else:
        raise RuntimeError("Can't find valid tiling.")

    cce_params.cceEmitParamsIns.insert_param("cut_ci_factor", cut_ci_factor)
    cce_params.cceEmitParamsIns.insert_param("cut_hi_factor", cut_hi_factor)
    cce_params.cceEmitParamsIns.insert_param("cut_wi_factor", cut_wi_factor)

    if cut_hi_factor == in_size_h and cut_wi_factor == in_size_w:
        cut_flag = "cutCi"
    elif cut_hi_factor < in_size_h and \
            cut_wi_factor == in_size_w:
        cut_flag = "cutCiHi"
    elif cut_hi_factor == 1 and \
            cut_wi_factor < in_size_w:
        cut_flag = "cutCiHiWi"

    cce_params.cceEmitParamsIns.insert_param("cut_flag", cut_flag)

    loop_cut_ci = (c1_value + cut_ci_factor - 1) // cut_ci_factor
    loop_cut_hi = (in_size_h + cut_hi_factor - 1) // cut_hi_factor
    loop_cut_wi = (in_size_w + cut_wi_factor - 1) // cut_wi_factor

    tiling_params["loop_cut_ci"] = loop_cut_ci
    tiling_params["loop_cut_hi"] = loop_cut_hi
    tiling_params["loop_cut_wi"] = loop_cut_wi
    tiling_params["enable_double_buffer"] = enable_double_buffer

    cce_params.cceEmitParamsIns.insert_param("loop_cut_ci", loop_cut_ci)
    cce_params.cceEmitParamsIns.insert_param("loop_cut_hi", loop_cut_hi)
    cce_params.cceEmitParamsIns.insert_param("loop_cut_wi", loop_cut_wi)

    return tiling_params


def find_bind_core_factor(cherry_num, device_core_num):
    """
    find the best pair for binding core
    :param cherry_num: batch size or others(for future)
    :param device_core_num: device core num
    :return: best factor
    """
    if cherry_num <= device_core_num:
        return 1

    expectation = cherry_num//device_core_num
    if cherry_num % device_core_num == 0:
        return expectation

    factor1, group1 = 1, cherry_num
    for j in range(expectation, 1, -1):
        if cherry_num % j == 0:
            factor1, group1 = j, cherry_num//j
            break

    factor2, _ = cherry_num, 1
    for j in range(expectation + 1, cherry_num, 1):
        if cherry_num % j == 0:
            factor2, _ = j, cherry_num//j
            break

    if factor1*((group1 + device_core_num - 1)//device_core_num) < factor2:
        return factor1

    return factor2


def build_ascend_tensor(res):
    """
    get the quantize compute tensors
    :param res: the placeholder of result
    :return: the quantize compute tensors
    """
    quant_tensor_map = {}
    stack = [res]
    visited_list = []
    while stack:
        cur_tensor = stack.pop()
        visited_list.append(cur_tensor)
        for in_tensor in cur_tensor.op.input_tensors:
            if in_tensor not in visited_list and \
                    not in_tensor.op.tag.startswith(POOLING2D_TAG_PREFIX):
                stack.append(in_tensor)
                quant_tensor_map[in_tensor.name] = in_tensor

    return quant_tensor_map


def _build_anti_tensor(res):
    if res is None:
        return {}

    tensor_map = {}
    stack = [res]
    while stack:
        cur_tensor = stack.pop()
        for tensor in cur_tensor.op.input_tensors:
            if isinstance(tensor.op, tvm.tensor.PlaceholderOp):
                continue
            stack.append(tensor)
            tensor_map[tensor.name] = tensor
    return tensor_map


def _set_anti_buffer_scope(sch, anti_tensor):
    for _, tensor in anti_tensor.items():
        sch[tensor].set_scope(cce.scope_ubuf)


def _anti_compute_at(sch, res, anti_tensor):
    for tensor in anti_tensor.values():
        sch[tensor].compute_at(sch[res], res.op.axis[0])


def _anti_emit_insn(sch, anti_tensor):
    for tensor in anti_tensor.values():
        if tensor.op.name == "input_ub":
            sch[tensor].emit_insn(tensor.op.axis[0], 'dma_copy')
        else:
            sch[tensor].emit_insn(tensor.op.axis[0], 'vector_auto')


def set_ascend_buffer_scope(sch, quant_tensor_map):
    """
    set the scope for quantize tensors
    :param sch: the schedule
    :param quant_tensor_map: the quantize compute tensors
    :return:
    """
    for _, value in quant_tensor_map.items():
        sch[value].set_scope(cce.scope_ubuf)


def set_ascend_compute_at(sch, res, quant_tensor_map, axis):
    """
    set the compute axis for tensors
    :param sch:
    :param res:
    :param quant_tensor_map:
    :param axis:
    :return:
    """
    for _, tensor in quant_tensor_map.items():
        sch[tensor].compute_at(sch[res], axis)
        if tensor.op.name in [VADDS_REFORM_NAME, VMULS_REFORM_NAME]:
            # c0 from 16 to 32
            sch[tensor].split(tensor.op.axis[3], factor=16)


def set_round_emit_insn(round_mode):
    """
    Obtains the conv instruction by the round mode attr

    Parameters
    ----------
    round_mode: the attr of round mode

    Returns
    -------
    instruction
    """
    if get_soc_spec("SOC_VERSION") == "Ascend310":
        # mini
        emit_insn_str = "vector_conv"
    else:
        if round_mode == "Round":
            emit_insn_str = "vector_conv_round"
        elif round_mode == "Ceil":
            emit_insn_str = "vector_conv_ceil"
        elif round_mode == "Floor":
            emit_insn_str = "vector_conv_floor"
        elif round_mode == "Trunc":
            emit_insn_str = "vector_conv_trunc"
        else:
            emit_insn_str = "vector_conv"
    return emit_insn_str


def set_quant_emit_insn(sch, quant_tensor_map, c1_value, attr_dic):
    """
    set quantion emit insn
    """
    round_emit_insn = set_round_emit_insn(attr_dic.get("round_mode"))
    in_dma = "dma_copy" if c1_value % 2 == 0 else "dma_padding"
    if CAST_F16_NAME in quant_tensor_map:
        sch[quant_tensor_map.get(CAST_F16_NAME)].emit_insn(
            sch[quant_tensor_map.get(CAST_F16_NAME)].op.axis[0], 'vector_conv')
    if OFFSET_NAME in quant_tensor_map:
        sch[quant_tensor_map.get(OFFSET_NAME)].emit_insn(
            sch[quant_tensor_map.get(OFFSET_NAME)].op.axis[0], 'vector_adds')
    if SQRT_NAME in quant_tensor_map:
        sch[quant_tensor_map.get(SQRT_NAME)].emit_insn(
            sch[quant_tensor_map.get(SQRT_NAME)].op.axis[0], 'vector_muls')
    if VMULS_REFORM_NAME in quant_tensor_map:
        sch[quant_tensor_map.get(VMULS_REFORM_NAME)].emit_insn(
            sch[quant_tensor_map.get(VMULS_REFORM_NAME)].op.axis[0],
            'vector_muls')
    if VADDS_REFORM_NAME in quant_tensor_map:
        sch[quant_tensor_map.get(VADDS_REFORM_NAME)].emit_insn(
            sch[quant_tensor_map.get(VADDS_REFORM_NAME)].op.axis[0],
            'vector_adds')
    sch[quant_tensor_map.get(CAST_I8_NAME)].emit_insn(
        sch[quant_tensor_map.get(CAST_I8_NAME)].op.axis[0], round_emit_insn)
    if INPUT_NAME in quant_tensor_map:
        sch[quant_tensor_map.get(INPUT_NAME)].emit_insn(
            sch[quant_tensor_map.get(INPUT_NAME)].op.axis[0], in_dma)


def _find_anti_res(tensor):
    if tensor.op.tag == ASCEND_ANTI_QUANT_TAG:
        return tensor
    for node in tensor.op.input_tensors:
        target = _find_anti_res(node)
        if target is not None:
            return target
    return None


def _get_from_attr(attrs, name):
    if name in attrs:
        return attrs[name]
    return None


def _check_blockdims(batch_size, batch_factor, device_core_num):
    if batch_size >= device_core_num and \
            batch_size / batch_factor > MAX_VALUE_OF_16BIT:
        raise RuntimeError("Invalid cut batch factor, "
                           "batch factor should be "
                           "less than 65535.")


def _copy_item(source_dict, target_dict):
    for key, value in source_dict.items():
        if hasattr(value, "value"):
            target_dict[key] = value.value
        else:
            target_dict[key] = value


# pylint: disable=too-many-locals, too-many-branches, too-many-statements, too-many-nested-blocks
def pooling2d_schedule(res, sch_list):
    """
    :params:
    :res: res compute
    :sch_list: schedule list
    :return: True
    """
    sch = sch_list[0]
    pooling_params = {}

    def _preprocess_fusion():
        fused_select_write = res.op.name.find("write_select") >= 0
        pooling_params["fused_select_write"] = fused_select_write

        res_quant = None
        ascend_tensor = None
        ascend_attr = None

        if fused_select_write:
            before_res = res.op.input_tensors[0]
            fused_ascend_quant = before_res.op.tag == ASCEND_QUANT_TAG
            pooling_params["fused_ascend_quant"] = fused_ascend_quant
            anti_res = _find_anti_res(res)
            fused_anti_quant = anti_res is not None
            pooling_params["fused_anti_quant"] = fused_anti_quant
        else:
            fused_ascend_quant = res.op.tag == ASCEND_QUANT_TAG
            pooling_params["fused_ascend_quant"] = fused_ascend_quant
            anti_res = _find_anti_res(res)
            fused_anti_quant = anti_res is not None
            pooling_params["fused_anti_quant"] = fused_anti_quant

        if fused_select_write and fused_ascend_quant:
            res_select_write = res
            res_quant = res_select_write.op.input_tensors[0]

            ascend_tensor = build_ascend_tensor(res)
            pooling2d_res = ascend_tensor["input_ub"].op.input_tensors[0]
            ascend_attr = {
                "scale": res_quant.op.attrs['scale'],
                "sqrt_mode": res_quant.op.attrs['sqrt_mode'],
                "offset": res_quant.op.attrs['offset'],
                "round_mode": res_quant.op.attrs['round_mode'],
            }
        elif fused_select_write:
            res_select_write = res
            pooling2d_res = res_select_write.op.input_tensors[0]
        elif fused_ascend_quant:
            ascend_tensor = build_ascend_tensor(res)
            pooling2d_res = ascend_tensor["input_ub"].op.input_tensors[0]
            ascend_attr = {
                "scale": res.op.attrs['scale'],
                "sqrt_mode": res.op.attrs['sqrt_mode'],
                "offset": res.op.attrs['offset'],
                "round_mode": res.op.attrs['round_mode'],
            }
            res_quant = res
        else:
            pooling2d_res = res

        res_list = (fused_select_write, fused_ascend_quant, fused_anti_quant,
                    anti_res, res_quant, ascend_tensor, pooling2d_res,
                    ascend_attr)
        return res_list

    fused_select_write, fused_ascend_quant, fused_anti_quant, anti_res, \
    res_quant, ascend_tensor, pooling2d_res, ascend_attr = \
        _preprocess_fusion()

    if _get_from_attr(pooling2d_res.op.attrs, "template") == "max_3_3_2_2":
        return schedule_max_pool_3_3_2_2(res, sch_list)
    elif _get_from_attr(pooling2d_res.op.attrs, "template") == "max_3_3_1_1":
        return schedule_max_pool_3_3_1_1(res, sch_list)

    anti_tensor = _build_anti_tensor(anti_res)

    def _get_l1_fusion_params(pooling2d_res):
        fusion_params_map = pooling2d_res.op.attrs['fusion_params']
        fusion_params = {}
        if fusion_params_map:
            for key, value in fusion_params_map.items():
                if hasattr(value, "value"):
                    fusion_params[key] = value.value
                else:
                    fusion_params[key] = value
        return fusion_params

    fusion_params = _get_l1_fusion_params(pooling2d_res)
    in_l1_flag = fusion_params.get("in_l1_flag")
    out_l1_flag = fusion_params.get("out_l1_flag")
    in_select_read_flag = fusion_params.get("in_select_read_flag")
    cce_params.cceEmitParamsIns.insert_params(fusion_params)

    l1_fusion_type = fusion_params.get("l1_fusion_type")
    is_l1fusion = l1_fusion_type in (0, 1)
    is_l2fusion = get_L1_info("L2_fusion_enabled")

    setfmatrix_map = pooling2d_res.op.attrs['setfmatrix_dict']
    setfmatrix_dict = {}
    _copy_item(setfmatrix_map, setfmatrix_dict)

    pooling_params_map = pooling2d_res.op.attrs['pooling_params']
    _copy_item(pooling_params_map, pooling_params)

    cce_params.cceEmitParamsIns.insert_params(pooling_params)
    device_core_num = _get_l1fusion_device_core_num(is_l1fusion)

    # Get vadd vmul vconv ability
    vconv_ability = intrinsic_check_support("Intrinsic_vconv", "f162f32")
    vadd_ability = intrinsic_check_support("Intrinsic_vadd", "float32")
    vmul_ability = intrinsic_check_support("Intrinsic_vmul", "float32")
    fp32_ability = vconv_ability and \
                   vadd_ability and \
                   vmul_ability and \
                   (not get_soc_spec("SOC_VERSION") in ("Ascend310",))
    # get pooling mode, out size
    pooling_mode = pooling_params["pooling_mode"]
    padding_mode = pooling_params["padding_mode"]
    out_size_h = pooling_params["out_size_h"]
    out_size_w = pooling_params["out_size_w"]
    batch_size = pooling_params["batch_size"]
    c1_value = pooling_params["c1_value"]

    # avg or max pooling
    if pooling_mode in ["AVG", "MAX"]:
        # get all tensors from compute
        pooling_ub_5hd = pooling2d_res.op.input_tensors[0]
        if pooling_mode == "AVG":
            pooling_out_ub_mul_factor = pooling_ub_5hd.op.input_tensors[0]
            pooling_out_ub = pooling_out_ub_mul_factor.op.input_tensors[0]
        elif pooling_mode == "MAX":
            pooling_out_ub = pooling_ub_5hd.op.input_tensors[0]

        fmap_fractal_tmp1 = pooling_out_ub.op.input_tensors[0]
        fmap_fractal = fmap_fractal_tmp1.op.input_tensors[0]
        fmap_img2col = fmap_fractal.op.input_tensors[0]
        fmap_l1 = fmap_img2col.op.input_tensors[0]
        tensor_in = fmap_l1.op.input_tensors[0]

        fmap_h = setfmatrix_dict["conv_fm_h"]
        fmap_w = setfmatrix_dict["conv_fm_w"]
        pad_top = setfmatrix_dict["conv_padding_top"]
        pad_bottom = setfmatrix_dict["conv_padding_bottom"]
        pad_left = setfmatrix_dict["conv_padding_left"]
        pad_right = setfmatrix_dict["conv_padding_right"]
        stride_h = setfmatrix_dict["conv_stride_h"]
        stride_w = setfmatrix_dict["conv_stride_w"]
        filter_h = setfmatrix_dict["conv_kernel_h"]
        filter_w = setfmatrix_dict["conv_kernel_w"]

        pooling_cut_factor, res_cut_factor, is_cut_l1_to_ub, cutc1_factor, \
        can_enable_double_buffer, cuth_loop = \
            pooling2d_tiling(pooling_params, fusion_params)

        is_ddr_l1_cut_h_flag = (cuth_loop > 1)
        is_need_skip_read_on_l1 = (in_l1_flag and
                                   is_ddr_l1_cut_h_flag and
                                   cutc1_factor != 1)

        def _set_scope():
            if is_l1fusion:
                ConvParam.l1_fusion_workspace_tensor_list = []
                if in_l1_flag:
                    sch[fmap_l1].set_scope(cce.scope_cbuf_fusion)
                    sch[tensor_in].set_scope(cce.scope_cbuf_fusion)
                    if l1_fusion_type == 1:
                        # regard ddrin_l1out as l1_width_fusion_tensor_in +
                        # l1in_l1out, fmap_l1 will emit phony insn
                        ConvParam.l1_fusion_workspace_tensor_list.\
                            append(tensor_in)
                else:
                    sch[fmap_l1].set_scope(cce.scope_cbuf_fusion)
                    ConvParam.l1_fusion_workspace_tensor_list.append(fmap_l1)
            else:
                sch[fmap_l1].set_scope(cce.scope_cbuf)
            sch[fmap_img2col].set_scope(cce.scope_cbuf)
            sch[fmap_fractal].set_scope(cce.scope_ubuf)
            sch[pooling_out_ub].set_scope(cce.scope_ubuf)
            sch[pooling_ub_5hd].set_scope(cce.scope_ubuf)
            if pooling_mode == "AVG":
                sch[pooling_out_ub_mul_factor].set_scope(cce.scope_ubuf)

            # fuse select_wirte op and ascend_quant op
            if fused_select_write:
                sch[pooling2d_res].set_scope(cce.scope_ubuf)

                if fused_ascend_quant:
                    sch[res_quant].set_scope(cce.scope_ubuf)
                    set_ascend_buffer_scope(sch, ascend_tensor)
                if fused_anti_quant:
                    sch[anti_res].set_scope(cce.scope_cbuf)
                    _set_anti_buffer_scope(sch, anti_tensor)
            else:
                if fused_ascend_quant:
                    sch[pooling2d_res].set_scope(cce.scope_ubuf)
                    set_ascend_buffer_scope(sch, ascend_tensor)
                if fused_anti_quant:
                    sch[anti_res].set_scope(cce.scope_cbuf)
                    _set_anti_buffer_scope(sch, anti_tensor)

            if is_l1fusion and out_l1_flag:
                sch[res].set_scope(cce.scope_cbuf_fusion)

        _set_scope()

        def _enable_double_buffer():
            if can_enable_double_buffer:
                if not is_l1fusion:
                    # l1 fusion can't support double buffer ddr->l1 stage
                    sch[fmap_l1].double_buffer()
                    sch[fmap_l1].preload()
                sch[fmap_fractal].double_buffer()
                sch[fmap_fractal].preload()

        _enable_double_buffer()

        def _process_compute_inline():
            sch[fmap_fractal_tmp1].compute_inline()
            if (is_l1fusion or is_l2fusion) and in_select_read_flag:
                sch[tensor_in].compute_inline()

            is_reset_pooling2d_res = False
            if fused_select_write:
                if fused_ascend_quant:
                    sch[res_quant].compute_inline()
                else:
                    sch[pooling2d_res].compute_inline()
                    is_reset_pooling2d_res = True
            return is_reset_pooling2d_res

        is_reset_pooling2d_res = _process_compute_inline()
        if is_reset_pooling2d_res:
            pooling2d_res = res

        need_correct_c1_factor = fused_ascend_quant and c1_value == 1
        if need_correct_c1_factor:
            cutc1_factor = 2

        pooling2d_res_c1_outer, pooling2d_res_c1_inner = \
            sch[pooling2d_res].split(
                pooling2d_res.op.axis[1],
                factor=cutc1_factor)
        # fractal
        pooling2d_res_2o, pooling2d_res_2i = \
            sch[pooling2d_res].split(pooling2d_res.op.axis[2],
                                     factor=16)
        pooling2d_res_outer, pooling2d_res_inner = \
            sch[pooling2d_res].split(pooling2d_res_2o,
                                     factor=res_cut_factor)
        if fused_ascend_quant:
            res_c1_factor = (cutc1_factor + 1)//2
            res_c1_outer, res_c1_inner = \
                sch[res].split(res.op.axis[1], factor=res_c1_factor)
            # fractal
            res_2o, res_2i = sch[res].split(res.op.axis[2], factor=16)
            res_outer, res_inner = \
                sch[res].split(res_2o, factor=res_cut_factor)
            res_c1_value = (c1_value + 1)//2
            res_c1_outer_value = \
                (res_c1_value + res_c1_factor - 1)//res_c1_factor
        else:
            res_c1_outer = pooling2d_res_c1_outer
            res_c1_inner = pooling2d_res_c1_inner
            # fractal
            res_2o, res_2i = pooling2d_res_2o, pooling2d_res_2i
            res_outer, res_inner = pooling2d_res_outer, pooling2d_res_inner
            res_c1_outer_value = (c1_value + cutc1_factor - 1)//cutc1_factor

        # block tiling
        block_tag = None
        block_axis = None
        bind_none = is_l1fusion or \
                    (batch_size == 1 and res_c1_outer_value == 1)
        bind_batch = batch_size >= device_core_num or \
                     batch_size >= res_c1_outer_value
        if bind_none:
            # no bind core
            batch_outer, batch_inner = sch[res].split(res.op.axis[0], factor=1)
            sch[res].reorder(batch_outer, batch_inner,
                             res_outer, res_c1_outer, res_inner,
                             res_c1_inner, res_2i, res.op.axis[3])
        elif bind_batch:
            # batch bind core
            batch_factor = find_bind_core_factor(batch_size, device_core_num)
            _check_blockdims(batch_size, batch_factor, device_core_num)
            batch_outer, batch_inner = \
                sch[res].split(res.op.axis[0], factor=batch_factor) \
                if batch_size >= device_core_num \
                else sch[res].split(res.op.axis[0], factor=1)
            sch[res].reorder(batch_outer, batch_inner,
                             res_outer, res_c1_outer, res_inner,
                             res_c1_inner, res_2i, res.op.axis[3])
            block_tag = "batch"
            block_axis = batch_outer
        else:
            # C1_outer bind core
            batch_outer, batch_inner = sch[res].split(res.op.axis[0], factor=1)
            sch[res].reorder(res_c1_outer, batch_outer, batch_inner,
                             res_outer, res_inner,
                             res_c1_inner, res_2i, res.op.axis[3])
            block_tag = "c1"
            block_axis = res_c1_outer
        if block_tag is not None:
            thread_block = tvm.thread_axis("blockIdx.x")
            sch[res].bind(block_axis, thread_block)

        # handle pooling2d_res
        if fused_ascend_quant:
            sch[pooling_ub_5hd].compute_inline()
            res_at = res_outer if block_tag == "c1" else res_c1_outer
            sch[pooling2d_res].compute_at(sch[res], res_at)
            set_ascend_compute_at(sch, res, ascend_tensor, res_at)
            set_quant_emit_insn(sch, ascend_tensor, c1_value, ascend_attr)
        else:
            # for pooling_ub_5hd emit_insn
            sch[pooling_ub_5hd].split(pooling_ub_5hd.op.axis[2], factor=16)

        if fused_anti_quant:
            sch[fmap_l1].compute_inline()
            _anti_compute_at(sch, anti_res, anti_tensor)
            reform_by_vmuls = anti_tensor["reform_by_vmuls"]
            sch[reform_by_vmuls].split(reform_by_vmuls.op.axis[1], factor=2)
            res_at = res_outer if block_tag == "c1" else res_c1_outer
            sch[anti_res].compute_at(sch[res], res_at)
            _anti_emit_insn(sch, anti_tensor)
            sch[anti_res].emit_insn(anti_res.op.axis[1], 'dma_copy')

        def _input_ub_buffer_align():
            """
            :align input_ub
            :return:
            """
            if "input_ub" in anti_tensor.keys():
                sch[anti_tensor["input_ub"]].buffer_align((1, 1),
                                                          (1, 1),
                                                          (1, 1),
                                                          (1, 1),
                                                          (1, INT8_ALIGN)
                                                          )

        # pylint: disable=too-many-branches, too-many-statements
        def schedule_cuth_cut_l1_to_ub():
            """
            :schedule of cut l1 to ub
            :return: valid schedule
            """
            pooling2d_res_at = \
                pooling2d_res_outer if block_tag == "c1" \
                else pooling2d_res_c1_outer

            pooling_outer, pooling_inner = \
                sch[pooling_out_ub].split(pooling_out_ub.op.axis[1],
                                          factor=pooling_cut_factor)

            sch[fmap_fractal].compute_at(sch[pooling_out_ub], pooling_outer)

            if pooling_mode == "AVG":
                sch[pooling_out_ub_mul_factor].compute_at(sch[pooling2d_res],
                                                          pooling2d_res_at)

            if not fused_ascend_quant:
                sch[pooling_ub_5hd].compute_at(
                    sch[pooling2d_res], pooling2d_res_at)
            sch[pooling_out_ub].compute_at(
                sch[pooling2d_res], pooling2d_res_at)
            sch[fmap_img2col].compute_at(sch[pooling2d_res], pooling2d_res_at)

            if not fused_anti_quant:
                if not is_need_skip_read_on_l1:
                    sch[fmap_l1].compute_at(sch[pooling2d_res],
                                            pooling2d_res_at)

                def _emit_insn_ddr_l1():
                    if is_l1fusion:
                        if l1_fusion_type == 1:
                            sch[tensor_in].emit_insn(tensor_in.op.axis[0],
                                                     'dma_copy')
                            sch[fmap_l1].emit_insn(fmap_l1.op.axis[0],
                                                   'phony_insn')
                        elif in_l1_flag:
                            sch[fmap_l1].emit_insn(fmap_l1.op.axis[0],
                                                   'phony_insn')
                        else:
                            sch[fmap_l1].emit_insn(fmap_l1.op.axis[0],
                                                   'dma_copy')
                            if is_ddr_l1_cut_h_flag:
                                sch[fmap_l1].pragma(fmap_l1.op.axis[0],
                                                    'jump_data', 1)
                    else:
                        sch[fmap_l1].emit_insn(fmap_l1.op.axis[0], 'dma_copy')

                _emit_insn_ddr_l1()

            fmap_img2col_setps_cuth = res_cut_factor * BLOCK_SIZE // out_size_w

            if is_need_skip_read_on_l1:
                start_pos_h = fmap_img2col_setps_cuth * stride_h
                setfmatrix_dict["conv_fm_offset_h"] = start_pos_h * res_outer

            cce_params.cceEmitParamsIns.insert_param(
                "fmap_img2col_setps_cuth", fmap_img2col_setps_cuth)
            cce_params.cceEmitParamsIns.insert_param("res_cut_factor",
                                                     res_cut_factor)

            def _process_buffer_tile():
                if_fmap_l1_buffer_tile = (is_l1fusion and in_l1_flag and
                                          not is_need_skip_read_on_l1)
                if if_fmap_l1_buffer_tile:
                    in_size_h = pooling_params["in_size_h"]
                    in_size_w = pooling_params["in_size_w"]
                    real_h = (out_size_h - 1) * stride_h + filter_h
                    real_w = (out_size_w - 1) * stride_w + filter_w
                    exist_invalid_data = (real_h < in_size_h) or \
                                         (real_w < in_size_w)
                    if exist_invalid_data:
                        out_cuth = res_cut_factor * BLOCK_SIZE // out_size_w
                        cuth_step = out_cuth * stride_h
                        if res_cut_factor * BLOCK_SIZE % out_size_w != 0:
                            out_cuth = out_cuth + 1
                        if out_size_h == 1:
                            out_cuth = 1
                        input_cuth = (out_cuth - 1) * stride_h + filter_h
                        tiling_fmap_h = tvm.min(fmap_h + pad_top + pad_bottom,
                                                input_cuth)

                        if block_tag == "batch":
                            i0_var = batch_outer.var * batch_factor + \
                                     batch_inner.var
                        else:
                            i0_var = batch_outer.var
                        sch[fmap_l1].buffer_tile(
                            (i0_var, 1),
                            (pooling2d_res_c1_outer * cutc1_factor,
                             tvm.min(cutc1_factor, c1_value -
                                     pooling2d_res_c1_outer * cutc1_factor)),
                            (pooling2d_res_outer.var * cuth_step - pad_top,
                             tiling_fmap_h),
                            (-1 * pad_left, fmap_w + pad_left + pad_right),
                            (0, BLOCK_SIZE)
                        )

            _process_buffer_tile()

            sch[fmap_img2col].buffer_align((1, 1),
                                           (1, 1),
                                           (1, 1),
                                           (1, 1),
                                           (1, 1),
                                           (1, 16))

            sch[fmap_fractal].buffer_align((1, 1),
                                           (1, 1),
                                           (1, 1),
                                           (1, BLOCK_SIZE),
                                           (1, BLOCK_SIZE))

            sch[pooling_out_ub].buffer_align((1, 1),
                                             (1, 1),
                                             (1, 1),
                                             (1, BLOCK_SIZE),
                                             (1, BLOCK_SIZE),
                                             (1, 1))
            if pooling_mode == "AVG":
                sch[pooling_out_ub_mul_factor].buffer_align((1, 1),
                                                            (1, 1),
                                                            (1, 1),
                                                            (1, BLOCK_SIZE),
                                                            (1, BLOCK_SIZE))
                _input_ub_buffer_align()

            def _process_emit_insn():
                sch[fmap_img2col].emit_insn(fmap_img2col.op.axis[0],
                                            'set_fmatrix', setfmatrix_dict)
                sch[fmap_fractal].emit_insn(fmap_fractal.op.axis[0], 'im2col')
                sch[pooling_out_ub].emit_insn(pooling_inner,
                                              'pooling2d_process')

                if fused_ascend_quant:
                    sch[pooling2d_res].emit_insn(pooling2d_res_inner,
                                                 'dma_copy')
                    if fused_select_write:
                        hwc0 = int(sch[res].op.attrs["HWC0"])
                        sch[res].buffer_stride(res.op.axis[1], hwc0, 0)
                        sch[res].emit_insn(res_inner, 'dma_copy')
                    else:
                        sch[res].emit_insn(res_inner, 'dma_copy')
                else:
                    sch[pooling_ub_5hd].emit_insn(
                        pooling_ub_5hd.op.axis[0], 'vector_adds')
                    if fused_select_write:
                        hwc0 = int(sch[pooling2d_res].op.attrs["HWC0"])
                        sch[pooling2d_res].buffer_stride(
                            pooling2d_res.op.axis[1],
                            hwc0, 0)
                        sch[pooling2d_res].emit_insn(res_inner, 'dma_copy')
                    else:
                        sch[pooling2d_res].emit_insn(pooling2d_res_inner,
                                                     'dma_copy')

                if pooling_mode == "AVG":
                    if padding_mode == "SAME":
                        avg_para_dict = {
                            'out_var': res_outer if fused_ascend_quant \
                                       else pooling2d_res_outer
                        }
                        sch[pooling_out_ub_mul_factor].emit_insn(
                            pooling_out_ub_mul_factor.op.axis[0],
                            'pooling2d_avg_mul_factor',
                            avg_para_dict)
                    else:
                        sch[pooling_out_ub_mul_factor].emit_insn(
                            pooling_out_ub_mul_factor.op.axis[0],
                            "elewise_single_VS_mul")

            _process_emit_insn()

        # pylint: disable=too-many-branches, too-many-statements
        def schedule_cuth():
            """
            :schedule for cuth
            :return: valid schedule
            """
            sch[fmap_img2col].buffer_align((1, 1),
                                           (out_size_w, out_size_w),
                                           (1, 1),
                                           (1, 1),
                                           (1, 1),
                                           (1, 16))

            sch[fmap_fractal].buffer_align((1, 1),
                                           (1, 1),
                                           (1, 1),
                                           (1, BLOCK_SIZE),
                                           (1, BLOCK_SIZE))

            sch[pooling_out_ub].buffer_align((1, 1),
                                             (1, 1),
                                             (1, 1),
                                             (1, BLOCK_SIZE),
                                             (1, BLOCK_SIZE),
                                             (1, 1))
            if pooling_mode == "AVG":
                sch[pooling_out_ub_mul_factor].buffer_align((1, 1),
                                                            (1, 1),
                                                            (1, 1),
                                                            (1, BLOCK_SIZE),
                                                            (1, BLOCK_SIZE))
                _input_ub_buffer_align()

            c1_or_res_outer = \
                pooling2d_res_outer if block_tag == "c1" \
                else pooling2d_res_c1_outer

            sch[fmap_img2col].compute_at(sch[pooling2d_res], c1_or_res_outer)

            if not fused_anti_quant:
                if not is_need_skip_read_on_l1:
                    sch[fmap_l1].compute_at(sch[pooling2d_res],
                                            c1_or_res_outer)
                if is_l1fusion:
                    if l1_fusion_type == 1:
                        sch[tensor_in].emit_insn(tensor_in.op.axis[0],
                                                 'dma_copy')
                        sch[fmap_l1].emit_insn(fmap_l1.op.axis[0],
                                               'phony_insn')
                    elif in_l1_flag:
                        sch[fmap_l1].emit_insn(fmap_l1.op.axis[0],
                                               'phony_insn')
                    else:
                        sch[fmap_l1].emit_insn(fmap_l1.op.axis[0], 'dma_copy')
                        if is_ddr_l1_cut_h_flag:
                            sch[fmap_l1].pragma(fmap_l1.op.axis[0],
                                                'jump_data', 1)
                else:
                    sch[fmap_l1].emit_insn(fmap_l1.op.axis[0], 'dma_copy')

            sch[fmap_fractal].compute_at(sch[pooling2d_res], c1_or_res_outer)

            fmap_img2col_setps_cuth = res_cut_factor * BLOCK_SIZE // out_size_w

            if is_need_skip_read_on_l1:
                start_pos_h = fmap_img2col_setps_cuth * stride_h - pad_top
                setfmatrix_dict["conv_fm_offset_h"] = start_pos_h * res_outer

            cce_params.cceEmitParamsIns.insert_param("fmap_img2col_setps_cuth",
                                                     fmap_img2col_setps_cuth)
            cce_params.cceEmitParamsIns.insert_param("res_cut_factor",
                                                     res_cut_factor)

            if pooling_mode == "AVG":
                sch[pooling_out_ub_mul_factor].compute_at(sch[pooling2d_res],
                                                          c1_or_res_outer)

            if not fused_ascend_quant:
                sch[pooling_ub_5hd].compute_at(sch[pooling2d_res],
                                               c1_or_res_outer)
            sch[pooling_out_ub].compute_at(sch[pooling2d_res], c1_or_res_outer)

            def _process_emit_insn():
                sch[fmap_img2col].emit_insn(fmap_img2col.op.axis[0],
                                            'set_fmatrix',
                                            setfmatrix_dict)
                sch[fmap_fractal].emit_insn(fmap_fractal.op.axis[0], 'im2col')
                sch[pooling_out_ub].emit_insn(pooling_out_ub.op.axis[0],
                                              'pooling2d_process')

                if fused_ascend_quant:
                    sch[pooling2d_res].emit_insn(pooling2d_res_inner,
                                                 'dma_copy')
                    if fused_select_write:
                        hwc0 = int(sch[res].op.attrs["HWC0"])
                        sch[res].buffer_stride(res.op.axis[1], hwc0, 0)
                        sch[res].emit_insn(res_inner, 'dma_copy')
                    else:
                        sch[res].emit_insn(res_inner, 'dma_copy')
                else:
                    sch[pooling_ub_5hd].emit_insn(pooling_ub_5hd.op.axis[0],
                                                  'vector_adds')
                    if fused_select_write:
                        hwc0 = int(sch[pooling2d_res].op.attrs["HWC0"])
                        sch[pooling2d_res].buffer_stride(
                            pooling2d_res.op.axis[1], hwc0, 0)
                        sch[pooling2d_res].emit_insn(res_inner, 'dma_copy')
                    else:
                        sch[pooling2d_res].emit_insn(pooling2d_res_inner,
                                                     'dma_copy')

                if pooling_mode == "AVG":
                    if padding_mode == "SAME":
                        avg_para_dict = {
                            'out_var': res_outer if fused_ascend_quant \
                                                 else pooling2d_res_outer
                        }
                        sch[pooling_out_ub_mul_factor].emit_insn(
                            pooling_out_ub_mul_factor.op.axis[0],
                            'pooling2d_avg_mul_factor',
                            avg_para_dict)
                    else:
                        sch[pooling_out_ub_mul_factor].emit_insn(
                            pooling_out_ub_mul_factor.op.axis[0],
                            "elewise_single_VS_mul")

            _process_emit_insn()

        if is_cut_l1_to_ub:
            schedule_cuth_cut_l1_to_ub()
        else:
            schedule_cuth()

    # global avg pooling or global max pooling
    elif pooling_mode in ["GAP", "GMP"]:
        # get tiling params
        tiling_params = pooling2d_global_tiling(pooling_params, fusion_params)
        if fused_ascend_quant:
            sch_list = [sch]
            return pooling_global_quant_schedule(
                sch_list, pooling_mode, pooling2d_res, res,
                tiling_params, pooling_params, fp32_ability,
                ascend_tensor, fusion_params)

        # get all tensors from compute
        if pooling_mode == "GAP":
            if fp32_ability:
                pooling_out_ub_mul_factor_f16 = \
                    pooling2d_res.op.input_tensors[0]
                pooling_out_ub_mul_factor = \
                    pooling_out_ub_mul_factor_f16.op.input_tensors[0]
                pooling_out_ub = pooling_out_ub_mul_factor.op.input_tensors[0]
                tensor_in_ub_f32 = pooling_out_ub.op.input_tensors[0]
            else:
                pooling_out_ub_mul_factor = pooling2d_res.op.input_tensors[0]
                pooling_out_ub = pooling_out_ub_mul_factor.op.input_tensors[0]
        elif pooling_mode == "GMP":
            pooling_out_ub = pooling2d_res.op.input_tensors[0]

        is_gloabl_avg_fp32 = (pooling_mode == "GAP" and fp32_ability)
        if is_gloabl_avg_fp32:
            tensor_in_ub = tensor_in_ub_f32.op.input_tensors[0]
            # set scope for each tensor
            sch[tensor_in_ub].set_scope(cce.scope_ubuf)
            sch[tensor_in_ub_f32].set_scope(cce.scope_ubuf)
            sch[pooling_out_ub].set_scope(cce.scope_ubuf)
        else:
            tensor_in_ub = pooling_out_ub.op.input_tensors[0]
            # set scope for each tensor
            sch[tensor_in_ub].set_scope(cce.scope_ubuf)
            sch[pooling_out_ub].set_scope(cce.scope_ubuf)

        if in_l1_flag:
            tensor_in = tensor_in_ub.op.input_tensors[0]
            if l1_fusion_type == 1:
                sch[tensor_in].set_scope(cce.scope_cbuf)
            else:
                sch[tensor_in].set_scope(cce.scope_cbuf_fusion)

        if pooling_mode == "GAP":
            if fp32_ability:
                sch[pooling_out_ub_mul_factor_f16].set_scope(cce.scope_ubuf)
            sch[pooling_out_ub_mul_factor].set_scope(cce.scope_ubuf)

        if out_l1_flag:
            sch[res].set_scope(cce.scope_cbuf_fusion)

        cut_ci_factor = tiling_params["cut_ci_factor"]
        # shedule part
        ci_outer, ci_inner = sch[pooling2d_res].split(pooling2d_res.op.axis[1],
                                                      factor=cut_ci_factor)

        pooling_out_ub_ci_outer, pooling_out_ub_ci_inner = \
            sch[pooling_out_ub].split(pooling_out_ub.op.axis[1],
                                      factor=cut_ci_factor)

        # use multi block
        block_tag = None
        res_c1_outer_value = (c1_value + cut_ci_factor - 1)//cut_ci_factor
        if is_l1fusion or (batch_size == 1 and res_c1_outer_value == 1):
            # no bind core
            batch_outer, batch_inner = \
                sch[pooling2d_res].split(pooling2d_res.op.axis[0], factor=1)
            sch[pooling2d_res].reorder(batch_outer, batch_inner,
                                       ci_outer, ci_inner,
                                       pooling2d_res.op.axis[2],
                                       pooling2d_res.op.axis[3])
        elif batch_size >= device_core_num or batch_size >= res_c1_outer_value:
            # batch bind core
            if batch_size >= device_core_num:
                batch_factor = \
                    find_bind_core_factor(batch_size, device_core_num)
                _check_blockdims(batch_size, batch_factor, device_core_num)
                batch_outer, batch_inner = \
                    sch[pooling2d_res].split(pooling2d_res.op.axis[0],
                                             factor=batch_factor)
            else:
                batch_outer, batch_inner = \
                    sch[pooling2d_res].split(pooling2d_res.op.axis[0],
                                             factor=1)
            sch[pooling2d_res].reorder(batch_outer, batch_inner,
                                       ci_outer, ci_inner,
                                       pooling2d_res.op.axis[2],
                                       pooling2d_res.op.axis[3])
            block_tag = "batch"
            block_axis = batch_outer
        else:
            # C1_outer bind core
            batch_outer, batch_inner = \
                sch[pooling2d_res].split(pooling2d_res.op.axis[0], factor=1)
            sch[pooling2d_res].reorder(ci_outer, batch_outer,
                                       batch_inner, ci_inner,
                                       pooling2d_res.op.axis[3])
            block_tag = "c1"
            block_axis = ci_outer

        if block_tag is not None:
            thread_block = tvm.thread_axis("blockIdx.x")
            sch[pooling2d_res].bind(block_axis, thread_block)
            cce_params.cceEmitParamsIns.insert_param("thread_block",
                                                     thread_block)
            cce_params.cceEmitParamsIns.insert_param("block_tag", block_tag)

        cut_hi_factor = tiling_params["cut_hi_factor"]
        reduce_hi_outer, reduce_hi_inner = sch[pooling_out_ub].split(
            pooling_out_ub.op.reduce_axis[0], factor=cut_hi_factor)

        cut_wi_factor = tiling_params["cut_wi_factor"]
        reduce_wi_outer, reduce_wi_inner = sch[pooling_out_ub].split(
            pooling_out_ub.op.reduce_axis[1], factor=cut_wi_factor)

        # cutCi Hi and Wi reorder
        # Wi is devided to main part and rest part, so reduce_wi_outer should be
        # outside of reduce_hi_outer, the same like
        # reduce_hi_outer is outside of pooling_out_ub_ci_outer
        block_tag_flag = block_tag is not None and block_tag == "c1"
        if block_tag_flag:
            sch[pooling_out_ub].reorder(pooling_out_ub_ci_outer,
                                        pooling_out_ub.op.axis[0],
                                        reduce_wi_outer,
                                        reduce_hi_outer,
                                        pooling_out_ub_ci_inner,
                                        pooling_out_ub.op.axis[2],
                                        pooling_out_ub.op.axis[3],
                                        reduce_hi_inner,
                                        reduce_wi_inner,
                                        pooling_out_ub.op.axis[4])
        else:
            sch[pooling_out_ub].reorder(pooling_out_ub.op.axis[0],
                                        reduce_wi_outer,
                                        reduce_hi_outer,
                                        pooling_out_ub_ci_outer,
                                        pooling_out_ub_ci_inner,
                                        pooling_out_ub.op.axis[2],
                                        pooling_out_ub.op.axis[3],
                                        reduce_hi_inner,
                                        reduce_wi_inner,
                                        pooling_out_ub.op.axis[4])

        sch[tensor_in_ub].compute_at(sch[pooling_out_ub], reduce_hi_outer)
        if pooling_mode == "GAP" and fp32_ability:
            sch[tensor_in_ub_f32].compute_at(
                sch[pooling_out_ub], reduce_hi_outer)

        if block_tag is not None and block_tag == "c1":
            compute_at_axis = batch_outer
        else:
            compute_at_axis = ci_outer

        sch[pooling_out_ub].compute_at(sch[pooling2d_res], compute_at_axis)

        if tiling_params["enable_double_buffer"]:
            sch[tensor_in_ub].double_buffer()
            sch[tensor_in_ub].preload()

        if pooling_mode == "GAP":
            if fp32_ability:
                _, avg_mul_ci_inner_f16 = \
                    sch[pooling_out_ub_mul_factor_f16].split(
                        pooling_out_ub_mul_factor_f16.op.axis[1],
                        factor=cut_ci_factor)
                sch[pooling_out_ub_mul_factor_f16].compute_at(
                    sch[pooling2d_res], compute_at_axis)

            _, avg_mul_ci_inner = sch[pooling_out_ub_mul_factor].split(
                pooling_out_ub_mul_factor.op.axis[1], factor=cut_ci_factor)
            sch[pooling_out_ub_mul_factor].compute_at(
                sch[pooling2d_res], compute_at_axis)

        if l1_fusion_type == 1:
            sch[tensor_in].emit_insn(tensor_in.op.axis[0], 'dma_copy')

        sch[tensor_in_ub].emit_insn(tensor_in_ub.op.axis[0], 'dma_copy')

        if pooling_mode == "GAP" and fp32_ability:
            sch[tensor_in_ub_f32].emit_insn(
                tensor_in_ub_f32.op.axis[0], 'vector_conv')
        if block_tag is not None and block_tag == "c1":
            sch[pooling_out_ub].emit_insn(
                pooling_out_ub_ci_inner, 'pooling2d_global_process')
        else:
            sch[pooling_out_ub].emit_insn(
                pooling_out_ub_ci_outer, 'pooling2d_global_process')

        if pooling_mode == "GAP":
            if fp32_ability:
                sch[pooling_out_ub_mul_factor_f16].emit_insn(
                    avg_mul_ci_inner_f16, "vector_conv")
            sch[pooling_out_ub_mul_factor].emit_insn(
                avg_mul_ci_inner, "elewise_single_VS_mul")

        sch[pooling2d_res].emit_insn(ci_inner, 'dma_copy')

    return True


def pooling_global_quant_schedule(
        sch_list, pooling_mode, pooling2d_res, res,
        tiling_params, pooling_params, fp32_ability,
        quant_tensor_map, fusion_params={}):
    """
    do schedule for global quant pooling
    """
    # pylint: too-many-arguments
    l1_fusion_type = fusion_params.get("l1_fusion_type", -1)
    in_l1_flag = fusion_params.get("in_l1_flag", False)
    out_l1_flag = fusion_params.get("out_l1_flag", False)
    is_l1fusion = l1_fusion_type in (0, 1)

    sch = sch_list[0]

    batch_size = pooling_params["batch_size"]
    c1_value = pooling_params["c1_value"]

    sch[pooling2d_res].set_scope(cce.scope_ubuf)
    set_ascend_buffer_scope(sch, quant_tensor_map)
    input_ub = quant_tensor_map['input_ub']
    del quant_tensor_map['input_ub']

    cut_ci_factor = tiling_params["cut_ci_factor"]

    cut_ci_factor_res = cut_ci_factor // 2 if cut_ci_factor > 1 else 1

    set_ascend_buffer_scope(sch, quant_tensor_map)

    device_core_num = _get_l1fusion_device_core_num(is_l1fusion)

    def _set_scop_get_rel_tensor():
        pooling_out_ub_mul_factor_f16 = None
        pooling_out_ub_mul_factor = None
        tensor_in_ub_f32 = None
        tensor_in = None
        if pooling_mode == "GAP":
            if fp32_ability:
                pooling_out_ub_mul_factor_f16 = pooling2d_res.op.input_tensors[
                    0]
                pooling_out_ub_mul_factor = \
                    pooling_out_ub_mul_factor_f16.op.input_tensors[0]
                pooling_out_ub = pooling_out_ub_mul_factor.op.input_tensors[0]
                tensor_in_ub_f32 = pooling_out_ub.op.input_tensors[0]
                tensor_in_ub = tensor_in_ub_f32.op.input_tensors[0]
                # set scope for each tensor
                sch[tensor_in_ub_f32].set_scope(cce.scope_ubuf)
                sch[pooling_out_ub_mul_factor_f16].set_scope(cce.scope_ubuf)
            else:
                pooling_out_ub_mul_factor = pooling2d_res.op.input_tensors[0]
                pooling_out_ub = pooling_out_ub_mul_factor.op.input_tensors[0]
                tensor_in_ub = pooling_out_ub.op.input_tensors[0]

            sch[pooling_out_ub_mul_factor].set_scope(cce.scope_ubuf)

            sch[tensor_in_ub].set_scope(cce.scope_ubuf)
            sch[pooling_out_ub].set_scope(cce.scope_ubuf)
        else:
            pooling_out_ub = pooling2d_res.op.input_tensors[0]
            tensor_in_ub = pooling_out_ub.op.input_tensors[0]
            # set scope for each tensor
            sch[tensor_in_ub].set_scope(cce.scope_ubuf)
            sch[pooling_out_ub].set_scope(cce.scope_ubuf)

        if is_l1fusion and in_l1_flag:
            tensor_in = tensor_in_ub.op.input_tensors[0]
            if l1_fusion_type == 1:
                sch[tensor_in].set_scope(cce.scope_cbuf)
            else:
                sch[tensor_in].set_scope(cce.scope_cbuf_fusion)
        if is_l1fusion and out_l1_flag:
            sch[res].set_scope(cce.scope_cbuf_fusion)

        res_list = (pooling_out_ub_mul_factor_f16, pooling_out_ub_mul_factor,
                    pooling_out_ub, tensor_in_ub_f32, tensor_in_ub, tensor_in)
        return res_list

    pooling_out_ub_mul_factor_f16, pooling_out_ub_mul_factor, pooling_out_ub, \
    tensor_in_ub_f32, tensor_in_ub, tensor_in = _set_scop_get_rel_tensor()

    sch[pooling2d_res].compute_inline()
    sch[input_ub].compute_inline()

    # shedule part
    ci_outer, ci_inner = sch[res].split(res.op.axis[1],
                                        factor=cut_ci_factor_res)

    pooling_out_ub_ci_outer, pooling_out_ub_ci_inner = \
        sch[pooling_out_ub].split(pooling_out_ub.op.axis[1],
                                  factor=cut_ci_factor)

    # use multi block
    block_tag = None
    res_c1_outer_value = (c1_value + cut_ci_factor - 1)//cut_ci_factor

    is_no_bind = is_l1fusion or (batch_size == 1 and res_c1_outer_value == 1)
    is_bind_batch = batch_size >= device_core_num or\
                    batch_size >= res_c1_outer_value
    if is_no_bind:
        # no bind core
        batch_outer, batch_inner = sch[res].split(res.op.axis[0], factor=1)
        sch[res].reorder(batch_outer, batch_inner, ci_outer, ci_inner,
                         res.op.axis[2], res.op.axis[3])
        compute_at_axis = ci_outer
    else:
        if is_bind_batch:
            # batch bind core
            if batch_size >= device_core_num:
                batch_factor = \
                    find_bind_core_factor(batch_size, device_core_num)
                batch_outer, batch_inner = sch[res].split(res.op.axis[0],
                                                          factor=batch_factor)
            else:
                batch_outer, batch_inner = sch[res].split(res.op.axis[0],
                                                          factor=1)
            sch[res].reorder(batch_outer, batch_inner, ci_outer, ci_inner,
                             res.op.axis[3])
            block_tag = "batch"
            block_axis = batch_outer
            compute_at_axis = ci_outer
        else:
            # C1_outer bind core
            batch_outer, batch_inner = sch[res].split(res.op.axis[0], factor=1)
            sch[res].reorder(ci_outer, batch_outer, batch_inner, ci_inner,
                             res.op.axis[2], res.op.axis[3])
            block_tag = "c1"
            block_axis = ci_outer
            compute_at_axis = batch_outer

        thread_block = tvm.thread_axis("blockIdx.x")
        sch[res].bind(block_axis, thread_block)
        cce_params.cceEmitParamsIns.insert_param("thread_block", thread_block)
        cce_params.cceEmitParamsIns.insert_param("block_tag", block_tag)

    cut_hi_factor = tiling_params["cut_hi_factor"]
    reduce_hi_outer, reduce_hi_inner = sch[pooling_out_ub].split(
        pooling_out_ub.op.reduce_axis[0], factor=cut_hi_factor)

    cut_wi_factor = tiling_params["cut_wi_factor"]
    reduce_wi_outer, reduce_wi_inner = sch[pooling_out_ub].split(
        pooling_out_ub.op.reduce_axis[1], factor=cut_wi_factor)

    # cutCi Hi and Wi reorder
    # Wi is devided to main part and rest part, so reduce_wi_outer should be
    # outside of reduce_hi_outer, the same like
    # reduce_hi_outer is outside of pooling_out_ub_ci_outer
    is_bind_c1_axis = block_tag is not None and block_tag == "c1"
    if is_bind_c1_axis:
        sch[pooling_out_ub].reorder(
            pooling_out_ub_ci_outer,
            pooling_out_ub.op.axis[0],
            reduce_wi_outer,
            reduce_hi_outer,
            pooling_out_ub_ci_inner,
            pooling_out_ub.op.axis[2],
            pooling_out_ub.op.axis[3],
            reduce_hi_inner,
            reduce_wi_inner,
            pooling_out_ub.op.axis[4])
    else:
        sch[pooling_out_ub].reorder(
            pooling_out_ub.op.axis[0],
            reduce_wi_outer,
            reduce_hi_outer,
            pooling_out_ub_ci_outer,
            pooling_out_ub_ci_inner,
            pooling_out_ub.op.axis[2],
            pooling_out_ub.op.axis[3],
            reduce_hi_inner,
            reduce_wi_inner,
            pooling_out_ub.op.axis[4])

    def _do_compute_at_and_emitinsn():
        """
        do compute_at and emit_insn
        """
        if tiling_params["enable_double_buffer"]:
            sch[tensor_in_ub].double_buffer()
            sch[tensor_in_ub].preload()

        sch[tensor_in_ub].compute_at(sch[pooling_out_ub], reduce_hi_outer)
        is_fp32_mode = pooling_mode == "GAP" and fp32_ability
        if is_fp32_mode:
            sch[tensor_in_ub_f32].compute_at(
                sch[pooling_out_ub], reduce_hi_outer)

        sch[pooling_out_ub].compute_at(sch[res], compute_at_axis)

        if pooling_mode == "GAP":
            if fp32_ability:
                _, avg_mul_ci_inner_f16 = \
                    sch[pooling_out_ub_mul_factor_f16].split(
                        pooling_out_ub_mul_factor_f16.op.axis[1],
                        factor=cut_ci_factor)
                sch[pooling_out_ub_mul_factor_f16].compute_at(
                    sch[res], compute_at_axis)
                sch[pooling_out_ub_mul_factor_f16].emit_insn(
                    avg_mul_ci_inner_f16, "vector_conv")

            avg_mul_ci_outer, _ = sch[pooling_out_ub_mul_factor].split(
                pooling_out_ub_mul_factor.op.axis[1], factor=cut_ci_factor)
            sch[pooling_out_ub_mul_factor].compute_at(
                sch[res], compute_at_axis)
            sch[pooling_out_ub_mul_factor].emit_insn(
                avg_mul_ci_outer, "elewise_single_VS_mul")

        if l1_fusion_type == 1:
            sch[tensor_in].emit_insn(tensor_in.op.axis[0], 'dma_copy')

        sch[tensor_in_ub].emit_insn(tensor_in_ub.op.axis[0], 'dma_copy')

        if is_fp32_mode:
            sch[tensor_in_ub_f32].emit_insn(tensor_in_ub_f32.op.axis[0],
                                            'vector_conv')

        if is_bind_c1_axis:
            sch[pooling_out_ub].emit_insn(pooling_out_ub_ci_inner,
                                          'pooling2d_global_process')
        else:
            sch[pooling_out_ub].emit_insn(pooling_out_ub_ci_outer,
                                          'pooling2d_global_process')

        sch[res].emit_insn(ci_inner, 'dma_copy')

    _do_compute_at_and_emitinsn()

    set_ascend_compute_at(sch, res, quant_tensor_map, compute_at_axis)

    attr_dic = {
        "scale": res.op.attrs['scale'],
        "sqrt_mode": res.op.attrs['sqrt_mode'],
        "offset": res.op.attrs['offset'],
        "round_mode": res.op.attrs['round_mode'],
    }
    set_quant_emit_insn(sch, quant_tensor_map, c1_value, attr_dic)

    return True
