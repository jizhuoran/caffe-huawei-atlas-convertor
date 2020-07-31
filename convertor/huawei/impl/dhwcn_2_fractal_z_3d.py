"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

dhwcn_2_fractal_z_3d
"""

from te import tik
from te import platform as tbe_platform
from topi.cce import util


# UB size in byte
UB_SIZE = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
# AICORE count
CORE_NUM = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
# C0 length
C0_LEN = 16
# repeat up limit
REPEAT_LIMIT = 255
# mask value
MASK_128 = 128


def _ceil_div(value_x, value_y):
    """
    do ceil division
    """
    return (value_x + value_y - 1) // value_y


def _get_vnchwconv_ub_size(col_size, dtype):
    """
    get the ubuf size for vnchwconv branch
    """

    if dtype.lower() == "float16":
        byte_cnt = 2
    elif dtype.lower() == "float32":
        byte_cnt = 4

    # 16 lines, the unit is byte
    need_ub_size = _ceil_div(col_size, C0_LEN) * C0_LEN * C0_LEN * byte_cnt
    # the UB will be split into two parts, the unit is byte
    ub_half_248k_size = 248 * 1024 // 2
    ub_upper_limit = UB_SIZE // 2
    if ub_upper_limit > ub_half_248k_size:
        ub_upper_limit = ub_half_248k_size

    if need_ub_size >= ub_upper_limit:
        ub_size = ub_upper_limit // byte_cnt // C0_LEN * C0_LEN
    else:
        ub_size = need_ub_size // byte_cnt

    return ub_size


def _clean_ubuf(tik_inst, src, src_offset, dup_len):
    """
    clean ubuf to zero
    """

    if src.dtype.lower() == "float16":
        dtype_factor = 2
    elif src.dtype.lower() == "float32":
        dtype_factor = 1
    batch_size = 64

    if dup_len > 0:
        repeat = dup_len // (batch_size * dtype_factor)
        left_elem = dup_len % (batch_size * dtype_factor)
        repeat_loop = repeat // REPEAT_LIMIT
        repeat_left = repeat % REPEAT_LIMIT
        dup_value = float(0)

        if repeat_loop > 0:
            with tik_inst.for_range(0, repeat_loop) as rpt_idx:
                tik_inst.vector_dup(MASK_128,
                                    src[src_offset + rpt_idx *
                                        REPEAT_LIMIT *
                                        batch_size * dtype_factor],
                                    dup_value, REPEAT_LIMIT, 1, 8)

        if repeat_left > 0:
            tik_inst.vector_dup(MASK_128,
                                src[src_offset + repeat_loop *
                                    REPEAT_LIMIT *
                                    batch_size * dtype_factor],
                                dup_value, repeat_left, 1, 8)

        if left_elem > 0:
            tik_inst.vector_dup(left_elem,
                                src[src_offset + repeat *
                                    batch_size * dtype_factor],
                                dup_value, 1, 1, 8)


# pylint: disable=too-many-locals
def _multi_core_on_c(tik_inst, data_in, data_out, shape_in):
    """
    process of multiple core on axis c
    """

    axis_d, axis_h, axis_w, axis_c, axis_n = shape_in
    multi_c_loop_cnt = (axis_c // C0_LEN) // CORE_NUM
    multi_c_loop_left = (axis_c // C0_LEN) % CORE_NUM
    axis_c_left = axis_c % C0_LEN
    vnchw_ub_size = _get_vnchwconv_ub_size(axis_n, data_in.dtype)
    # vnchwconv process 16 lines each time
    vnchw_col_size = vnchw_ub_size // C0_LEN

    in_ub = tik_inst.Tensor(data_in.dtype, (vnchw_ub_size,),
                            name="in_ub", scope=tik.scope_ubuf)
    out_ub = tik_inst.Tensor(data_in.dtype, (vnchw_ub_size,),
                             name="out_ub", scope=tik.scope_ubuf)

    with tik_inst.for_range(0, CORE_NUM, block_num=CORE_NUM) as block_idx:
        with tik_inst.for_range(0, axis_d) as d_idx:
            with tik_inst.for_range(0, axis_h * axis_w) as hw_idx:

                def _inner_process(c_lp_idx, c_idx, c_cnt):
                    """
                    real transfer process for multiple core on axis c
                    """

                    ceil_axis_n = _ceil_div(axis_n, C0_LEN) * C0_LEN
                    n_16_size = ceil_axis_n * C0_LEN
                    idx_list = [0, 1, 2, 3, 4, 5, 6, 7,
                                8, 9, 10, 11, 12, 13, 14, 15]

                    def _inner_dhwcn_2_3d(n_lp_id, col_size):
                        """
                        do transfer from dhwcn to 3d
                        """

                        with tik_inst.for_range(0, c_cnt) as c_index:
                            tik_inst.data_move(
                                in_ub[c_index * vnchw_col_size],
                                data_in[c_index * axis_n +
                                        n_lp_id * vnchw_col_size +
                                        (block_idx +
                                         c_lp_idx * CORE_NUM + c_idx) *
                                        C0_LEN * axis_n +
                                        (hw_idx + d_idx * axis_h * axis_w) *
                                        axis_c * axis_n],
                                0, 1,
                                _ceil_div(col_size, C0_LEN), 0, 0)

                        src_addr_list = [in_ub[vnchw_col_size * i] for i in
                                         idx_list]
                        dst_addr_list = [out_ub[C0_LEN * i] for i in idx_list]
                        repeat_cnt = _ceil_div(col_size, C0_LEN)
                        src_stride = 0 if repeat_cnt == 1 else 1
                        dst_stride = 0 if repeat_cnt == 1 else 16

                        tik_inst.vnchwconv(False, False, dst_addr_list,
                                           src_addr_list,
                                           repeat_cnt, dst_stride, src_stride)

                        # set n left to zero
                        if col_size % C0_LEN:
                            _clean_ubuf(tik_inst, out_ub,
                                        col_size * C0_LEN,
                                        (C0_LEN - col_size % C0_LEN) * C0_LEN)

                        ni_c0_size = C0_LEN * C0_LEN
                        no_ni_c0_size = _ceil_div(axis_n, C0_LEN) * ni_c0_size
                        tik_inst.data_move(
                            data_out[
                                n_lp_id * _ceil_div(vnchw_col_size, C0_LEN) *
                                ni_c0_size +
                                (hw_idx + (c_lp_idx * CORE_NUM + c_idx +
                                           block_idx + d_idx *
                                           _ceil_div(axis_c, C0_LEN)) *
                                 axis_h * axis_w) * no_ni_c0_size],
                            out_ub,
                            0, 1, repeat_cnt * C0_LEN, 0, 0)

                    if n_16_size > vnchw_ub_size:
                        axis_n_loop = axis_n // vnchw_col_size
                        axis_n_left = axis_n % vnchw_col_size

                        if axis_n_loop > 0:
                            with tik_inst.for_range(0,
                                                    axis_n_loop) as n_lp_idx:
                                _inner_dhwcn_2_3d(n_lp_idx, vnchw_col_size)

                        if axis_n_left > 0:
                            _inner_dhwcn_2_3d(axis_n_loop, axis_n_left)

                    else:
                        _inner_dhwcn_2_3d(0, axis_n)

                if multi_c_loop_cnt > 0:
                    with tik_inst.for_range(0, multi_c_loop_cnt) as c_loop_idx:
                        _inner_process(c_loop_idx, 0, C0_LEN)

                if multi_c_loop_left > 0:
                    with tik_inst.if_scope(block_idx < multi_c_loop_left):
                        _inner_process(multi_c_loop_cnt, 0, C0_LEN)

                if axis_c_left > 0:
                    _clean_ubuf(tik_inst,
                                in_ub,
                                axis_c_left * vnchw_col_size,
                                (C0_LEN - axis_c_left) * vnchw_col_size)

                    with tik_inst.if_scope(block_idx < 1):
                        _inner_process(0, axis_c // C0_LEN, axis_c_left)


def _multi_core_on_dhw(tik_inst, data_in, data_out, shape_in):
    """
    process of multiple core on axis d, h, w
    """

    axis_d, axis_h, axis_w, axis_c, _ = shape_in
    # move 16 * 256 elements each time
    out_ub_size = C0_LEN * C0_LEN * C0_LEN
    out_ub = tik_inst.Tensor(data_in.dtype, (out_ub_size,),
                             name="out_ub", scope=tik.scope_ubuf)
    # in order to keep ub size 16 align
    in_ub_size = (UB_SIZE // 2 - out_ub_size) // C0_LEN * C0_LEN
    in_ub = tik_inst.Tensor(data_in.dtype, (in_ub_size,),
                            name="in_ub", scope=tik.scope_ubuf)
    c_list = [i for i in range(axis_c)]
    reg_list = [tik_inst.Scalar(data_in.dtype) for i in c_list]

    dhw_size = axis_d * axis_h * axis_w
    core_data_size = dhw_size // CORE_NUM * axis_c
    core_data_left = dhw_size % CORE_NUM * axis_c
    ni_c0_size = C0_LEN * C0_LEN

    # set out_ub to zero
    _clean_ubuf(tik_inst, out_ub, 0, out_ub_size)

    def _inner_process_dhw(block_index, slice_size, in_offset, out_offset):
        """
        real transfer process for multiple core on axis d, h, w
        """

        # to keep the axis_c align
        ub_align_c_size = in_ub_size // axis_c * axis_c

        def _inner_dhwcn_2_3d_dhw(lp_idx, col_size):
            """
            do transfer from dhwcn to 3d
            """

            tik_inst.data_move(in_ub,
                               data_in[lp_idx * ub_align_c_size +
                                       block_index * slice_size +
                                       in_offset],
                               0, 1, _ceil_div(col_size, C0_LEN), 0, 0)

            c_count = col_size // axis_c
            mv_loop = c_count // C0_LEN
            mv_left = c_count % C0_LEN

            def _move_elements(lp_index, mv_len):
                """
                move elements for output
                """

                if axis_c == C0_LEN:
                    tik_inst.data_move(out_ub,
                                       in_ub[lp_index * C0_LEN * C0_LEN],
                                       0, mv_len, 1, 0, 15)
                else:
                    with tik_inst.for_range(0, mv_len) as len_idx:
                        # mv_len * axis_c
                        for idx in c_list:
                            reg_list[idx].set_as(
                                in_ub[idx + len_idx * axis_c +
                                      lp_index * axis_c * C0_LEN])

                        for idx in c_list:
                            out_ub[len_idx * ni_c0_size + idx].set_as(
                                reg_list[idx])

                tik_inst.data_move(
                    data_out[(lp_index * C0_LEN +
                              lp_idx * (ub_align_c_size // axis_c) +
                              block_index * (slice_size // axis_c)) *
                             ni_c0_size + out_offset],
                    out_ub,
                    0, 1, mv_len * C0_LEN, 0, 0)

            if mv_loop > 0:
                with tik_inst.for_range(0, mv_loop) as mv_lp_idx:
                    _move_elements(mv_lp_idx, C0_LEN)

            if mv_left > 0:
                _move_elements(mv_loop, mv_left)

        if slice_size > in_ub_size:

            slice_loop = slice_size // ub_align_c_size
            slice_left = slice_size % ub_align_c_size

            if slice_loop > 0:
                with tik_inst.for_range(0, slice_loop) as slice_lp_idx:
                    _inner_dhwcn_2_3d_dhw(slice_lp_idx, ub_align_c_size)

            if slice_left > 0:
                _inner_dhwcn_2_3d_dhw(slice_loop, slice_left)

        else:
            _inner_dhwcn_2_3d_dhw(0, slice_size)

    with tik_inst.for_range(0, CORE_NUM, block_num=CORE_NUM) as block_idx:
        if core_data_size > 0:
            _inner_process_dhw(block_idx, core_data_size, 0, 0)

        if core_data_left > 0:
            with tik_inst.if_scope(block_idx < (dhw_size % CORE_NUM)):
                _inner_process_dhw(
                    block_idx, axis_c,
                    core_data_size * CORE_NUM,
                    dhw_size // CORE_NUM * ni_c0_size * CORE_NUM)


# pylint: disable=too-many-statements, too-many-branches
def _multi_core_on_hw(tik_inst, data_in, data_out, shape_in):
    """
    process of multiple core on axis h, w
    """

    axis_d, axis_h, axis_w, axis_c, axis_n = shape_in

    if axis_n == 1:
        # move 16 * 256 elements each time
        out_ub_size = C0_LEN * C0_LEN * C0_LEN
        out_ub = tik_inst.Tensor(data_in.dtype, (out_ub_size,),
                                 name="out_ub", scope=tik.scope_ubuf)

        # in order to keep ub size 16 align
        in_ub_size = (UB_SIZE // 2 - out_ub_size) // C0_LEN * C0_LEN
        in_ub = tik_inst.Tensor(data_in.dtype, (in_ub_size,),
                                name="in_ub", scope=tik.scope_ubuf)

        # set out_ub to zero
        _clean_ubuf(tik_inst, out_ub, 0, out_ub_size)

        if axis_c % C0_LEN:
            c_list = [i for i in range(axis_c % C0_LEN)]
            reg_list = [tik_inst.Scalar(data_in.dtype) for i in c_list]

    else:
        vnchw_ub_size = _get_vnchwconv_ub_size(axis_n, data_in.dtype)
        # vnchwconv process 16 lines each time
        vnchw_col_size = vnchw_ub_size // C0_LEN
        in_ub = tik_inst.Tensor(data_in.dtype, (vnchw_ub_size,),
                                name="in_ub", scope=tik.scope_ubuf)
        out_ub = tik_inst.Tensor(data_in.dtype, (vnchw_ub_size,),
                                 name="out_ub", scope=tik.scope_ubuf)

    hw_size = axis_h * axis_w
    multi_hw_loop_cnt = hw_size // CORE_NUM
    multi_hw_left_cnt = hw_size % CORE_NUM
    c_loop_cnt = axis_c // C0_LEN
    c_left_cnt = axis_c % C0_LEN
    ni_c0_size = C0_LEN * C0_LEN
    c1hw_size = _ceil_div(axis_c, C0_LEN) * hw_size

    with tik_inst.for_range(0, CORE_NUM, block_num=CORE_NUM) as block_idx:
        with tik_inst.for_range(0, axis_d) as d_idx:

            def _inner_process_hw(hw_lp_index):
                """
                real transfer process for multiple core on axis h, w
                """

                def _inner_dhwcn_2_3d_hw_vnchw():
                    """
                    do transfer from dhwcn to 3d by vnchwconv
                    """

                    ceil_axis_n = _ceil_div(axis_n, C0_LEN) * C0_LEN
                    n_16_size = ceil_axis_n * C0_LEN
                    idx_list = [0, 1, 2, 3, 4, 5, 6, 7,
                                8, 9, 10, 11, 12, 13, 14, 15]

                    def _c_loop_process(c_lp_index, c_cnt):
                        """
                        c loop process
                        """

                        def _vnchwconv_transfer_hw(n_lp_index, col_size):
                            """
                            vnchwconv transfer
                            """

                            with tik_inst.for_range(0, c_cnt) as c_line_idx:
                                tik_inst.data_move(
                                    in_ub[c_line_idx * vnchw_col_size],
                                    data_in[c_line_idx * axis_n +
                                            n_lp_index * vnchw_col_size +
                                            c_lp_index * C0_LEN * axis_n +
                                            (block_idx +
                                             hw_lp_index * CORE_NUM +
                                             d_idx * hw_size) *
                                            axis_c * axis_n],
                                    0, 1,
                                    _ceil_div(col_size, C0_LEN), 0, 0)

                            src_addr_list = [in_ub[vnchw_col_size * i]
                                             for i in idx_list]
                            dst_addr_list = [out_ub[C0_LEN * i]
                                             for i in idx_list]
                            repeat_cnt = _ceil_div(col_size, C0_LEN)
                            src_stride = 0 if repeat_cnt == 1 else 1
                            dst_stride = 0 if repeat_cnt == 1 else 16

                            tik_inst.vnchwconv(False, False,
                                               dst_addr_list,
                                               src_addr_list,
                                               repeat_cnt,
                                               dst_stride, src_stride)
                            # set n left to zero
                            if col_size % C0_LEN:
                                _clean_ubuf(
                                    tik_inst, out_ub,
                                    col_size * C0_LEN,
                                    (C0_LEN - col_size % C0_LEN) * C0_LEN)

                            no_ni_c0_size = \
                                _ceil_div(axis_n, C0_LEN) * ni_c0_size
                            tik_inst.data_move(
                                data_out[n_lp_index *
                                         _ceil_div(vnchw_col_size, C0_LEN) *
                                         ni_c0_size +
                                         (c_lp_index * hw_size + block_idx +
                                          hw_lp_index * CORE_NUM +
                                          d_idx * c1hw_size) * no_ni_c0_size],
                                out_ub,
                                0, 1, repeat_cnt * C0_LEN, 0, 0)

                        if n_16_size > vnchw_ub_size:
                            n_loop = axis_n // vnchw_col_size
                            n_left = axis_n % vnchw_col_size

                            if n_loop > 0:
                                with tik_inst.for_range(0, n_loop) as n_lp_idx:
                                    _vnchwconv_transfer_hw(n_lp_idx,
                                                           vnchw_col_size)

                            if n_left > 0:
                                _vnchwconv_transfer_hw(n_loop, n_left)

                        else:
                            _vnchwconv_transfer_hw(0, axis_n)

                    if c_loop_cnt:
                        with tik_inst.for_range(0, c_loop_cnt) as c_lp_idx:
                            _c_loop_process(c_lp_idx, C0_LEN)

                    if c_left_cnt:
                        # clean the un-used lines of 16 lines to zero
                        _clean_ubuf(tik_inst, in_ub,
                                    c_left_cnt * vnchw_col_size,
                                    (C0_LEN - c_left_cnt) * vnchw_col_size)
                        _c_loop_process(c_loop_cnt, c_left_cnt)
                _inner_dhwcn_2_3d_hw_vnchw()

            def _inner_process_hwc(hw_len, in_offset, out_offset):
                """
                real transfer process for multiple core on axis h, w with n==1
                and c <= in_ub_size
                """

                def _dhwcn_2_3d_hw_hwc(hwc_index, col_size, c0_line):
                    """
                    dhwcn transfer to 3d by c with c <= in_ub_size
                    """

                    with tik_inst.for_range(0, c0_line) as c0_idx:
                        tik_inst.data_move(
                            in_ub[c0_idx * hwc_col_size],
                            data_in[(hwc_index * C0_LEN + block_idx * hw_len +
                                     d_idx * hw_size) * axis_c + in_offset +
                                    c0_idx * col_size],
                            0, 1, _ceil_div(col_size, C0_LEN), 0, 0)

                    src_addr_list = [in_ub[hwc_col_size * i] for i in idx_list]
                    dst_addr_list = [in_ub[hwc_ub_size + C0_LEN * i]
                                     for i in idx_list]
                    repeat_cnt1 = _ceil_div(col_size, C0_LEN)
                    src_stride = 0 if repeat_cnt1 == 1 else 1
                    dst_stride = 0 if repeat_cnt1 == 1 else 16
                    # first vnchwconv
                    tik_inst.vnchwconv(False, False,
                                       dst_addr_list,
                                       src_addr_list,
                                       repeat_cnt1,
                                       dst_stride, src_stride)
                    # move ub_2_ub to padding c
                    tik_inst.data_move(
                        in_ub[hwc_ub_size * 2],
                        in_ub[hwc_ub_size],
                        0, col_size // axis_c, axis_c,
                        0, _ceil_div(axis_c, C0_LEN) * C0_LEN - axis_c)

                    repeat_cnt2 = 1
                    src_stride2 = 0
                    dst_stride2 = 0
                    c_cnt = col_size // axis_c
                    c0_cnt = _ceil_div(axis_c, C0_LEN)

                    with tik_inst.for_range(0, c0_cnt) as c0_index:
                        with tik_inst.for_range(0, c_cnt) as c_index:
                            src_tmp_list = [
                                in_ub[hwc_ub_size * 2 +
                                      (c0_index + c_index * c0_cnt) *
                                      ni_c0_size + C0_LEN * i]
                                for i in idx_list]
                            dst_tmp_list = [out_ub[ni_c0_size * i]
                                            for i in idx_list]

                            # mv 16 c0 to make 16 C0_LEN * C0_LEN
                            tik_inst.vnchwconv(False, False,
                                               dst_tmp_list,
                                               src_tmp_list,
                                               repeat_cnt2,
                                               dst_stride2, src_stride2)

                            tik_inst.data_move(
                                data_out[(c_index + c0_index * hw_size +
                                          hwc_index * C0_LEN +
                                          block_idx * hw_len +
                                          d_idx * c0_cnt * hw_size) *
                                         ni_c0_size + out_offset],
                                out_ub,
                                0, c0_line, C0_LEN,
                                0, (c_cnt - 1) * C0_LEN)

                c_count_in_col = \
                    hwc_col_size // (_ceil_div(axis_c, C0_LEN) * C0_LEN)
                actual_c_count_in_col = hw_len // C0_LEN
                c_count_left = hw_len % C0_LEN
                idx_list = [0, 1, 2, 3, 4, 5, 6, 7,
                            8, 9, 10, 11, 12, 13, 14, 15]

                if actual_c_count_in_col:
                    hwc_ub_loop = actual_c_count_in_col // c_count_in_col
                    hwc_ub_left = actual_c_count_in_col % c_count_in_col

                    if hwc_ub_loop:
                        with tik_inst.for_range(0, hwc_ub_loop) as hwc_ub_idx:
                            _dhwcn_2_3d_hw_hwc(
                                hwc_ub_idx * c_count_in_col,
                                c_count_in_col * axis_c, C0_LEN)

                    if hwc_ub_left:
                        _dhwcn_2_3d_hw_hwc(
                            hwc_ub_loop * c_count_in_col,
                            hwc_ub_left * axis_c, C0_LEN)

                if c_count_left:
                    _dhwcn_2_3d_hw_hwc(
                        actual_c_count_in_col,
                        axis_c, c_count_left)

            def _inner_process_c(hw_lp_index):
                """
                real transfer process for multiple core on axis h, w with n==1
                and c > in_ub_size
                """

                def _dhwcn_2_3d_hw_c(c_ub_index, col_size):
                    """
                    dhwcn transfer to 3d by c with c > in_ub_size
                    """

                    tik_inst.data_move(
                        in_ub,
                        data_in[c_ub_index * in_ub_size +
                                (block_idx + hw_lp_index * CORE_NUM +
                                 d_idx * hw_size) * axis_c],
                        0, 1, _ceil_div(col_size, C0_LEN), 0, 0)

                    c0_cnt = col_size // C0_LEN
                    col_left = col_size % C0_LEN
                    if c0_cnt:
                        c0_loop = c0_cnt // C0_LEN
                        c0_loop_left = c0_cnt % C0_LEN
                        with tik_inst.for_range(0, c0_loop) as c0_lp_idx:
                            tik_inst.data_move(
                                out_ub,
                                in_ub[c0_lp_idx * ni_c0_size],
                                0, C0_LEN, 1, 0, 15)

                            tik_inst.data_move(
                                data_out[(block_idx + hw_lp_index * CORE_NUM +
                                          (c0_lp_idx * C0_LEN + c_ub_index *
                                           in_ub_size // C0_LEN +
                                           d_idx * _ceil_div(axis_c, C0_LEN)) *
                                          hw_size) * ni_c0_size],
                                out_ub,
                                0, C0_LEN, C0_LEN, 0, (hw_size - 1) * C0_LEN)

                        if c0_loop_left:
                            tik_inst.data_move(
                                out_ub,
                                in_ub[c0_loop * ni_c0_size],
                                0, c0_loop_left, 1, 0, 15)

                            tik_inst.data_move(
                                data_out[(block_idx + hw_lp_index * CORE_NUM +
                                          (c0_loop * C0_LEN + c_ub_index *
                                           in_ub_size // C0_LEN +
                                           d_idx * _ceil_div(axis_c, C0_LEN)) *
                                          hw_size) * ni_c0_size],
                                out_ub,
                                0, c0_loop_left, C0_LEN,
                                0, (hw_size - 1) * C0_LEN)

                    if col_left:
                        # the left cnt is less than one block,
                        # so clean one block
                        _clean_ubuf(tik_inst, out_ub, 0, C0_LEN)
                        for idx in c_list:
                            reg_list[idx].set_as(in_ub[c0_cnt * C0_LEN + idx])
                        for idx in c_list:
                            out_ub[idx].set_as(reg_list[idx])

                        tik_inst.data_move(
                            data_out[(block_idx + hw_lp_index * CORE_NUM +
                                      (c0_cnt + c_ub_index *
                                       in_ub_size // C0_LEN +
                                       d_idx * _ceil_div(axis_c, C0_LEN)) *
                                      hw_size) * ni_c0_size],
                            out_ub,
                            0, 1, C0_LEN, 0, (hw_size - 1) * C0_LEN)

                c_ub_loop = axis_c // in_ub_size
                c_ub_left = axis_c % in_ub_size

                with tik_inst.for_range(0, c_ub_loop) as c_ub_idx:
                    _dhwcn_2_3d_hw_c(c_ub_idx, in_ub_size)

                if c_ub_left:
                    _dhwcn_2_3d_hw_c(c_ub_loop, c_ub_left)

            if axis_n > 1:
                if multi_hw_loop_cnt:
                    with tik_inst.for_range(0, multi_hw_loop_cnt) as hw_lp_idx:
                        _inner_process_hw(hw_lp_idx)

                if multi_hw_left_cnt:
                    with tik_inst.if_scope(block_idx < multi_hw_left_cnt):
                        _inner_process_hw(multi_hw_loop_cnt)
            else:
                col_threshold = in_ub_size // 3 // C0_LEN // C0_LEN * C0_LEN
                if axis_c > col_threshold or multi_hw_loop_cnt == 1:
                    if multi_hw_loop_cnt:
                        with tik_inst.for_range(
                                0, multi_hw_loop_cnt) as hw_lp_idx:
                            _inner_process_c(hw_lp_idx)

                    if multi_hw_left_cnt:
                        with tik_inst.if_scope(block_idx < multi_hw_left_cnt):
                            _inner_process_c(multi_hw_loop_cnt)

                else:
                    # to split the in_ub_size into 3 parts and align with 16
                    hwc_col_size = col_threshold
                    hwc_ub_size = hwc_col_size * C0_LEN

                    # clean dst address of ub_to_ub one time
                    with tik_inst.if_scope(d_idx == 0):
                        with tik_inst.if_scope(axis_c % C0_LEN):
                            _clean_ubuf(tik_inst, in_ub,
                                        hwc_ub_size * 2, hwc_ub_size)

                    # process multi_hw_loop_cnt * axis_c each core
                    if multi_hw_loop_cnt:
                        _inner_process_hwc(multi_hw_loop_cnt, 0, 0)

                    # the left process axis_c each core
                    if multi_hw_left_cnt:
                        with tik_inst.if_scope(block_idx < multi_hw_left_cnt):
                            _inner_process_hwc(
                                1,
                                multi_hw_loop_cnt * CORE_NUM * axis_c,
                                multi_hw_loop_cnt * CORE_NUM * ni_c0_size)


def _multi_core_on_d(tik_inst, data_in, data_out, shape_in):
    """
    process of multiple core on axis d
    """

    axis_d, axis_h, axis_w, axis_c, axis_n = shape_in

    multi_d_loop_cnt = axis_d // CORE_NUM
    multi_d_left_cnt = axis_d % CORE_NUM
    hw_size = axis_h * axis_w
    idx_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    vnchw_ub_size = _get_vnchwconv_ub_size(axis_n, data_in.dtype)
    # vnchwconv process 16 lines each time
    vnchw_col_size = vnchw_ub_size // C0_LEN
    in_ub = tik_inst.Tensor(data_in.dtype, (vnchw_ub_size,),
                            name="in_ub", scope=tik.scope_ubuf)
    out_ub = tik_inst.Tensor(data_in.dtype, (vnchw_ub_size,),
                             name="out_ub", scope=tik.scope_ubuf)

    with tik_inst.for_range(0, CORE_NUM, block_num=CORE_NUM) as block_idx:

        def _inner_process_d(d_lp_index):
            """
            real transfer process for multiple core on axis h, w
            """

            c_loop_cnt = axis_c // C0_LEN
            c_left = axis_c % C0_LEN

            with tik_inst.for_range(0, hw_size) as hw_idx:

                def _dhwcn_2_3d_d(c_lp_index, c_cnt):
                    """
                    do transfer from dhwcn to 3d by vnchwconv
                    """

                    def _vnchwconv_process_d(n_lp_index, col_size):
                        """
                        vnchwconv transfer process
                        """

                        with tik_inst.for_range(0, c_cnt) as c_index:
                            tik_inst.data_move(
                                in_ub[c_index * vnchw_col_size],
                                data_in[n_lp_index * vnchw_col_size +
                                        c_index * axis_n +
                                        c_lp_index * C0_LEN * axis_n +
                                        hw_idx * axis_c * axis_n +
                                        (block_idx + d_lp_index * CORE_NUM) *
                                        hw_size * axis_c * axis_n],
                                0, 1, _ceil_div(col_size, C0_LEN), 0, 0)

                        src_addr_list = [in_ub[vnchw_col_size * i]
                                         for i in idx_list]
                        dst_addr_list = [out_ub[C0_LEN * i] for i in idx_list]
                        repeat_cnt = _ceil_div(col_size, C0_LEN)
                        src_stride = 0 if repeat_cnt == 1 else 1
                        dst_stride = 0 if repeat_cnt == 1 else 16

                        tik_inst.vnchwconv(False, False,
                                           dst_addr_list,
                                           src_addr_list,
                                           repeat_cnt,
                                           dst_stride, src_stride)
                        # set n left to zero
                        if col_size % C0_LEN:
                            _clean_ubuf(
                                tik_inst, out_ub,
                                col_size * C0_LEN,
                                (C0_LEN - col_size % C0_LEN) * C0_LEN)

                        ni_c0_size = C0_LEN * C0_LEN
                        no_ni_c0_size = _ceil_div(axis_n, C0_LEN) * ni_c0_size
                        c1_cnt = _ceil_div(axis_c, C0_LEN)
                        c1hw_size = c1_cnt * hw_size
                        tik_inst.data_move(
                            data_out[n_lp_index *
                                     _ceil_div(vnchw_col_size, C0_LEN) *
                                     ni_c0_size +
                                     (c_lp_index * hw_size + hw_idx +
                                      (d_lp_index * CORE_NUM + block_idx) *
                                      c1hw_size) * no_ni_c0_size],
                            out_ub,
                            0, 1, repeat_cnt * C0_LEN, 0, 0)

                    if axis_n > vnchw_col_size:
                        n_loop = axis_n // vnchw_col_size
                        n_left = axis_n % vnchw_col_size

                        if n_loop:
                            with tik_inst.for_range(0, n_loop) as n_lp_idx:
                                _vnchwconv_process_d(n_lp_idx,
                                                     vnchw_col_size)

                        if n_left:
                            _vnchwconv_process_d(n_loop, n_left)

                    else:
                        _vnchwconv_process_d(0, axis_n)

                if c_loop_cnt:
                    with tik_inst.for_range(0, c_loop_cnt) as c_lp_idx:
                        _dhwcn_2_3d_d(c_lp_idx, C0_LEN)

                if c_left:
                    # set the lines will not be used to zero
                    _clean_ubuf(tik_inst, in_ub, c_left * vnchw_col_size,
                                (C0_LEN - c_left) * vnchw_col_size)
                    _dhwcn_2_3d_d(c_loop_cnt, c_left)

        if multi_d_loop_cnt:
            with tik_inst.for_range(0, multi_d_loop_cnt) as d_lp_idx:
                _inner_process_d(d_lp_idx)

        if multi_d_left_cnt:
            with tik_inst.if_scope(block_idx < multi_d_left_cnt):
                _inner_process_d(multi_d_loop_cnt)


def dhwcn_2_fractal_z_3d_compute(tik_inst, data_in, data_out, shape_in):
    """
    do dhwcn to fractal_z_3d transfer
    """

    axis_d, _, _, axis_c, axis_n = shape_in
    if axis_c <= C0_LEN and axis_n == 1:
        _multi_core_on_dhw(tik_inst, data_in, data_out, shape_in)
    elif axis_c // C0_LEN // CORE_NUM > 0 and axis_n > 1:
        _multi_core_on_c(tik_inst, data_in, data_out, shape_in)
    elif axis_d // CORE_NUM > 0 and axis_n > 1:
        _multi_core_on_d(tik_inst, data_in, data_out, shape_in)
    else:
        _multi_core_on_hw(tik_inst, data_in, data_out, shape_in)


@util.check_input_type(dict, dict, str, str, str)
def dhwcn_2_fractal_z_3d(src, dst, src_format, dst_format,
                         kernel_name="dhwcn_2_fractal_z_3d"):
    """
    used to transfer dhwcn to fractal_z_3d

    Parameters
    ----------
    src : dict
        shape and dtype of input
    dst : dict
        shape and dtype of output, should be same shape and type as input
    src_format : str
        input format, the value should be "DHWCN"
    dst_format : str
         output format, the value should be "FRACTAL_Z_3D"
    kernel_name : str
        kernel name, default value is "dhwcn_2_fractal_z_3d"

    Returns
    -------
    None
    """
    shape_in = src.get("shape")
    dtype = src.get("dtype")
    input_dtype = dtype.lower()

    util.check_shape_rule(shape_in)
    util.check_tensor_shape_size(shape_in)
    util.check_kernel_name(kernel_name)

    if input_dtype != "float16":
        raise RuntimeError("Now the transfer only support data type float16!")

    if src_format.upper() != "DHWCN" or dst_format.upper() != "FRACTAL_Z_3D":
        raise RuntimeError("The src_format must be DHWCN and"
                           " dst_format must be FRACTAL_Z_3D!")
    input_shape_len = 5
    if len(shape_in) != input_shape_len:
        raise RuntimeError("The length of input shape should be 5!")

    # initial Tik
    tik_inst = tik.Tik()
    # define input and output tensors
    data_in = tik_inst.Tensor(input_dtype, shape_in,
                              tik.scope_gm, "data_in")
    shape_out = (shape_in[0], _ceil_div(shape_in[3], C0_LEN),
                 shape_in[1] * shape_in[2],
                 _ceil_div(shape_in[4], C0_LEN),
                 C0_LEN, C0_LEN)
    data_out = tik_inst.Tensor(input_dtype, shape_out,
                               tik.scope_gm, "data_out")

    # do transfer
    dhwcn_2_fractal_z_3d_compute(tik_inst, data_in, data_out, shape_in)

    # build cce
    tik_inst.BuildCCE(kernel_name=kernel_name,
                      inputs=[data_in], outputs=[data_out])
