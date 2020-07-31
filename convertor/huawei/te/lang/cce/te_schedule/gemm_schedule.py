#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

gemm schedule
"""

from __future__ import absolute_import  # pylint: disable=too-many-lines

import functools
from te.platform.cce_build import build_config
from te.domain.tiling.tiling_query import tiling_query
import te.platform.cce_params as cce
from te.lang.cce.boost_schedule_kit import ScheduleAgent
from te.lang.cce.boost_schedule_kit import Compare
from te.platform import CUBE_MKN
import tvm  # pylint: disable=import-error

class Params():
    """
    all params and print function
    """
    DEBUG_PARAM = True
    DEBUG_IR = True
    CUB_FUSED_NUM = {
        'fp16fp16': 4, 'fp16fp32': 1, 'int8int32': 2, 'int8fp32': 1
    }
    AUB_FUSED_NUM = {
        'fp16fp16': 0, 'fp16fp32': 0, 'int8int32': 0, 'int8fp32': 40
    }
    BUB_FUSED_NUM = {
        'fp16fp16': 0, 'fp16fp32': 0, 'int8int32': 0, 'int8fp32': 40
    }
    INPUT_SIZE = {
        'fp16fp16': 2, 'fp16fp32': 2, 'int8int32': 1, 'int8fp32': 1
    }
    L1_L0_SIZE = {
        'fp16fp16': 2, 'fp16fp32': 2, 'int8int32': 1, 'int8fp32': 2
    }
    OUTPUT_SIZE = {
        'fp16fp16': 2, 'fp16fp32': 4, 'int8int32': 4, 'int8fp32': 4
    }
    MAD_TYPE = {
        'fp16fp16': 'float32', 'fp16fp32': 'float32',
        'int8int32': 'int32', 'int8fp32': 'float32'
    }
    ops_mode = 'fp16fp16'
    ops_format_mode = 'ND'
    is_a_matrix_align = True
    is_b_matrix_align = True
    block_in = cce.BLOCK_IN
    block_reduce = cce.BLOCK_REDUCE
    CONST_AL1_SHAPE_DIM = 4
    CONST_BL1_SHAPE_DIM = 4
    UB_SPACE_SIZE = 248 * 1024
    L1_SPACE_SIZE = 1024 * 1024
    L0_SPACE_SIZE = 64 * 1024
    L0C_SPACE_SIZE = 256 * 1024
    FRACT_FLAG = [False, False, False, False]  # [a, b, c, bias]
    TENSOR_MAP = {}
    TILING = {}
    DIM_MAP = {}
    DATA_SIZE = {
        'int8': 1,
        'int32': 4,
        'float16': 2,
        'float32': 4
    }

    def __init__(self):
        self.ops_mode = 'fp16fp16'

    def print_debug(self, *info):
        """
        print log if debug
        :param info:
        :return:
        """
        if self.DEBUG_PARAM:
            print(info)

    def print_ir_matmul(self, process, sch):
        """
        print ir for input sch
        :param process: tag
        :param sch: schedule
        :return: IR process
        """
        if self.DEBUG_IR:
            with build_config:
                start = process + " IR start"
                end = process + " IR end\n"
                print(start)
                bounds = tvm.schedule.InferBound(sch)
                stmt = tvm.schedule.ScheduleOps(sch, bounds, True)
                print(stmt)
                print(end)


def get_ops_mode():
    """
    Get ops mode from input and output
    :return:
    """

    a_type = Params.TENSOR_MAP["a_placehold"].dtype
    a_format = Params.TENSOR_MAP["a_placehold"].shape
    c_type = Params.TENSOR_MAP["c_gm"].dtype
    if len(a_format) == 2:
        Params.ops_format_mode = 'ND'
    else:
        Params.ops_format_mode = 'Nz'
    if a_type == 'float16' and c_type == 'float16':
        Params.ops_mode = 'fp16fp16'
        Params.block_reduce = cce.BLOCK_REDUCE
    elif a_type == 'float16' and c_type == 'float32':
        Params.ops_mode = 'fp16fp32'
        Params.block_reduce = cce.BLOCK_REDUCE
    elif a_type == 'int8' and c_type == 'int32':
        Params.ops_mode = 'int8int32'
        Params.block_reduce = cce.BLOCK_REDUCE_INT8
    elif a_type == 'int8' and c_type == 'float32':
        Params.ops_mode = 'int8fp32'
        Params.block_reduce = cce.BLOCK_REDUCE
    else:
        raise RuntimeError('Unsupported data type, {} in, {} out'.format(
            a_type, c_type))


def int_ceil_div(divisor_a, divisor_b):
    """
    round up function
    :param divisor_a: int.
    :param divisor_b: int.
    :return: int
    """
    if divisor_b == 0:
        raise RuntimeError("division by zero")
    return (divisor_a + divisor_b - 1) // divisor_b


def get_all_tensors(res):
    """
    get all tensor
    :param res: tensor
    :return: list
    """

    all_tensor = dict()
    all_tensor['res'] = res

    def get(tensor):
        """
        find all tensor
        :param tensor: c_gm
        :return: all tensor
        """

        tensor_list = tensor.op.input_tensors
        for one_tensor in tensor_list:
            # check which tensor has not been checked
            if one_tensor.op.name not in all_tensor:
                all_tensor[one_tensor.op.name] = one_tensor
                get(one_tensor)

    get(res)
    return all_tensor


def _small_m_k_n(a_m, a_k, b_n):
    small_am_count = 0
    small_ak_count = 0
    small_bn_count = 0
    if 1 <= a_m < 16:
        small_am_count = 1
    if 1 <= a_k < 16:
        small_ak_count = 1
    if 1 <= b_n < 16:
        small_bn_count = 1
    small_m_k_n_count = small_am_count * small_ak_count * small_bn_count
    return small_m_k_n_count


def _check_align(a_m, a_k, b_n, stridew, divide_factor):
    if 1 <= b_n <= divide_factor:
        stridew = 1
    elif (a_m % divide_factor != 0) or (
            a_k % divide_factor != 0) or (b_n % divide_factor != 0):
        stridew = 0
    elif (a_m % divide_factor == 0) and \
            (a_k % divide_factor == 0) and (b_n % divide_factor == 0):
        stridew = 1
    return stridew


def _check_align_int8(b_n, stridew, divide_factor):
    if 1 <= b_n <= divide_factor or b_n % divide_factor == 0:
        stridew = 1
    elif b_n % divide_factor < 8:
        stridew = 0
    return stridew


def get_tiling_result_nd():
    '''
    :param None:
    :return: TILING result and data_byte
    '''
    a_type = Params.TENSOR_MAP["a_placehold"].dtype
    b_type = Params.TENSOR_MAP["b_placehold"].dtype
    c_type = Params.TENSOR_MAP["c_gm"].dtype

    a_m = Params.DIM_MAP['a_shape'][0]
    a_k = Params.DIM_MAP['a_shape'][1]
    b_n = Params.DIM_MAP['b_shape'][1]

    if a_type == 'float16':
        a_shape = [1, (Params.DIM_MAP['a_shape'][1] + 16 - 1) // 16,
                   (Params.DIM_MAP['a_shape'][0] + 16 - 1) // 16, 16, 16]
        b_shape = [Params.DIM_MAP['b_shape'][0],
                   (Params.DIM_MAP['b_shape'][1] + 16 - 1) // 16,
                   1, 1, 16]
    else:
        a_shape = [1, (Params.DIM_MAP['a_shape'][1] + 32-1) // 32,\
        (((Params.DIM_MAP['a_shape'][0] + 32 - 1) // 32)
         * 32 // 16), 16, 32]
        b_shape = [((Params.DIM_MAP['b_shape'][0] + 32 - 1) // 32) * 32,
                   (Params.DIM_MAP['b_shape'][1] + 32 - 1)
                   // 32, 1, 1, 32]
    pad_l = 10
    pad_r = 10
    if c_type == 'float16':
        fused_num = 5
    else:
        fused_num = 1

    mad_type = Params.MAD_TYPE.get(Params.ops_mode)

    stridew = 1
    if Params.ops_mode == 'int8int32':
        divide_factor = 32
        stridew = _check_align_int8(b_n, stridew, divide_factor)
    else:
        divide_factor = 16
        stridew = _check_align(a_m, a_k, b_n, stridew, divide_factor)
    tiling = tiling_query(a_shape, b_shape, c_shape=None,
                          a_dtype=a_type, b_dtype=b_type, c_dtype=c_type,
                          mad_dtype=mad_type,
                          padl=pad_l, padr=pad_r,
                          padu=0, padd=0, strideh=0, stridew=stridew,
                          strideh_expand=1, stridew_expand=1,
                          dilationh=1, dilationw=1, group=1,
                          fused_double_operand_num=fused_num,
                          bias_flag=0, op_tag='matmul')

    Params().print_debug('-----------auto tiling result-----------------')
    Params().print_debug(tiling)
    Params().print_debug('----------------------------------------------')
    return tiling


def set_data_layout(res, sch):  # pylint: disable=too-many-statements
    """
    get DIM_MAP which contains all ops

    Parameter:
    ----------------------------------------------------------
    :param res: op
    :param sch: schedule
    :return: None
    ----------------------------------------------------------
    """

    all_tensor = get_all_tensors(res)

    def _set_align_mode():
        a_matrix = all_tensor.get('tensor_a')
        b_matrix = all_tensor.get('tensor_b')
        for item in a_matrix.shape:
            if item.value % 32 != 0:
                Params.is_a_matrix_align = False
                break
        for item in b_matrix.shape:
            if item.value % 32 != 0:
                Params.is_b_matrix_align = False
                break


    def _init_common_tensor():
        Params.TENSOR_MAP["c_gm"] = all_tensor.get('res')
        Params.TENSOR_MAP["a_placehold"] = all_tensor.get('tensor_a')
        Params.TENSOR_MAP["b_placehold"] = all_tensor.get('tensor_b')
        Params.TENSOR_MAP["bias"] = all_tensor.get('tensor_bias')
        Params.TENSOR_MAP["alpha"] = all_tensor.get('tensor_alpha')
        Params.TENSOR_MAP["beta"] = all_tensor.get('tensor_beta')

        # tensor in aicore
        if not (Params.TENSOR_MAP["c_gm"].dtype == "float32" and \
                len(Params.TENSOR_MAP["c_gm"].shape) == 2):
            Params.TENSOR_MAP["c_ub"] = all_tensor.get('tensor_c_ub')
            sch[Params.TENSOR_MAP["c_ub"]].set_scope(cce.scope_ubuf)
        Params.TENSOR_MAP["a_l0a"] = all_tensor.get('tensor_a_l0a')
        sch[Params.TENSOR_MAP["a_l0a"]].set_scope(cce.scope_ca)
        Params.TENSOR_MAP["a_l1"] = all_tensor.get('tensor_a_l1')
        sch[Params.TENSOR_MAP["a_l1"]].set_scope(cce.scope_cbuf)
        Params.TENSOR_MAP["b_l0b"] = all_tensor.get('tensor_b_l0b')
        sch[Params.TENSOR_MAP["b_l0b"]].set_scope(cce.scope_cb)
        Params.TENSOR_MAP["b_l1"] = all_tensor.get('tensor_b_l1')
        sch[Params.TENSOR_MAP["b_l1"]].set_scope(cce.scope_cbuf)
        Params.TENSOR_MAP["c_l0c"] = all_tensor.get('tensor_c')
        sch[Params.TENSOR_MAP["c_l0c"]].set_scope(cce.scope_cc)
        Params.TENSOR_MAP["bias_ub"] = all_tensor.get('tensor_bias_ub')
        sch[Params.TENSOR_MAP["bias_ub"]].set_scope(cce.scope_ubuf)
        Params.TENSOR_MAP["beta_bias_ub"] = \
            all_tensor.get('tensor_beta_bias_ub')
        sch[Params.TENSOR_MAP["beta_bias_ub"]].set_scope(cce.scope_ubuf)
        Params.TENSOR_MAP["beta_ub"] = all_tensor.get('tensor_beta_ub')
        sch[Params.TENSOR_MAP["beta_ub"]].set_scope(cce.scope_ubuf)
        Params.TENSOR_MAP["alpha_ub"] = all_tensor.get('tensor_alpha_ub')
        sch[Params.TENSOR_MAP["alpha_ub"]].set_scope(cce.scope_ubuf)
        Params.TENSOR_MAP["alpha_c_ub"] = all_tensor.get('tensor_alpha_c_ub')
        sch[Params.TENSOR_MAP["alpha_c_ub"]].set_scope(cce.scope_ubuf)
        Params.TENSOR_MAP["c_before_mul_ub"] = \
            all_tensor.get('tensor_c_before_mul_ub')
        sch[Params.TENSOR_MAP["c_before_mul_ub"]].set_scope(cce.scope_ubuf)
        Params.TENSOR_MAP["c_ub_temp"] = all_tensor.get('tensor_c_ub_temp')
        sch[Params.TENSOR_MAP["c_ub_temp"]].set_scope(cce.scope_ubuf)

    def _init_fp16_fp16_tensor():
        if Params.ops_mode == 'fp16fp16':
            Params.TENSOR_MAP["float32_bias_ub"] = \
                all_tensor.get('tensor_float32_bias_ub')
            sch[Params.TENSOR_MAP["float32_bias_ub"]].set_scope(cce.scope_ubuf)
            Params.TENSOR_MAP["beta_temp_ub"] = \
                all_tensor.get('tensor_beta_temp_ub')
            sch[Params.TENSOR_MAP["beta_temp_ub"]].set_scope(cce.scope_ubuf)
            Params.TENSOR_MAP["alpha_temp_ub"] = \
                all_tensor.get('tensor_alpha_temp_ub')
            sch[Params.TENSOR_MAP["alpha_temp_ub"]].set_scope(cce.scope_ubuf)

        if Params.ops_format_mode == 'ND':
            Params.TENSOR_MAP["a_normalize_ub"] = \
                all_tensor.get('tensor_a_normalize_ub')
            sch[Params.TENSOR_MAP["a_normalize_ub"]].set_scope(cce.scope_ubuf)
            Params.TENSOR_MAP["a_fract_k_ub"] = \
            all_tensor.get('a_fract_k')
            sch[Params.TENSOR_MAP["a_fract_k_ub"]].set_scope(cce.scope_ubuf)
            Params.TENSOR_MAP["b_normalize_ub"] = \
            all_tensor.get('tensor_b_normalize_ub')
            sch[Params.TENSOR_MAP["b_normalize_ub"]].set_scope(cce.scope_ubuf)
            Params.TENSOR_MAP["b_fract_ub"] = \
                all_tensor.get('b_fract')
            sch[Params.TENSOR_MAP["b_fract_ub"]].set_scope(cce.scope_ubuf)
            if (Params.ops_mode == 'fp16fp16' or
                    Params.ops_mode == 'fp16fp32'):
                Params.TENSOR_MAP["b_zero"] = \
                    all_tensor.get('tensor_b_zero')
                sch[Params.TENSOR_MAP["b_zero"]].set_scope(cce.scope_ubuf)
            if Params.ops_mode == 'int8int32':
                if Params.is_b_matrix_align is False:
                    Params.TENSOR_MAP["b_zero"] = \
                        all_tensor.get('tensor_b_zero')
                    sch[Params.TENSOR_MAP["b_zero"]].set_scope(cce.scope_ubuf)
                if Params.is_a_matrix_align is False:
                    Params.TENSOR_MAP["a_zero"] = all_tensor.get(
                        'tensor_a_zero')
                    sch[Params.TENSOR_MAP["a_zero"]].set_scope(cce.scope_ubuf)
                Params.TENSOR_MAP["b_transpose"] = \
                    all_tensor.get('b_transpose')
                sch[Params.TENSOR_MAP["b_transpose"]].set_scope(cce.scope_ubuf)

    def _init_int8_fp32_tensor():
        if Params.ops_mode == 'int8fp32':
            Params.TENSOR_MAP["a_ub"] = all_tensor.get('tensor_a_ub')
            sch[Params.TENSOR_MAP["a_ub"]].set_scope(cce.scope_ubuf)
            Params.TENSOR_MAP["float16_a_ub"] = \
                all_tensor.get('tensor_float16_a_ub')
            sch[Params.TENSOR_MAP["float16_a_ub"]].set_scope(cce.scope_ubuf)
            Params.TENSOR_MAP["zz_a_ub"] = all_tensor.get('tensor_zz_a_ub')
            sch[Params.TENSOR_MAP["zz_a_ub"]].set_scope(cce.scope_ubuf)

            Params.TENSOR_MAP["b_ub"] = all_tensor.get('tensor_b_ub')
            sch[Params.TENSOR_MAP["b_ub"]].set_scope(cce.scope_ubuf)
            Params.TENSOR_MAP["float16_b_ub"] = \
                all_tensor.get('tensor_float16_b_ub')
            sch[Params.TENSOR_MAP["float16_b_ub"]].set_scope(cce.scope_ubuf)
            Params.TENSOR_MAP["zn_b_ub"] = all_tensor.get('tensor_zn_b_ub')
            sch[Params.TENSOR_MAP["zn_b_ub"]].set_scope(cce.scope_ubuf)

    def _init_fract_tensor():
        if 'tensor_bias_ub_fract' in all_tensor:
            Params.TENSOR_MAP["bias_ub_fract"] = \
                all_tensor.get('tensor_bias_ub_fract')
            sch[Params.TENSOR_MAP["bias_ub_fract"]].set_scope(cce.scope_ubuf)

    def _init_map():
        # fill in dimmap
        Params.DIM_MAP['out_shape'] = [int(i) for i in
                                       res.shape]
        Params.DIM_MAP['a_shape'] = [int(i) for i in
                                     Params.TENSOR_MAP["a_placehold"].shape]
        Params.DIM_MAP['A_matrix_dim'] = [int(i) for i in
                                          Params.TENSOR_MAP["a_l0a"].shape]
        Params.DIM_MAP['B_matrix_dim'] = [int(i) for i in
                                          Params.TENSOR_MAP["b_l0b"].shape]
        Params.DIM_MAP['b_shape'] = [int(i) for i in
                                     Params.TENSOR_MAP["b_placehold"].shape]

    _set_align_mode()
    _init_common_tensor()
    get_ops_mode()
    _init_fp16_fp16_tensor()
    _init_int8_fp32_tensor()
    _init_fract_tensor()
    _init_map()


def get_tiling():  # pylint: disable=too-many-statements
    """
    get tilling parameter from tilling guery and check all parameter

    Parameter:
    ---------------------------------------------------------
    :return: None
    ---------------------------------------------------------
    """

    # check tilling parts
    # get data amount in AL1 BL1
    def _get_data_amount_l1(l1_shape, isdouble, data_size):
        """
        using tilling parameter calculate data amount in l1

        Parameters:
        ---------------------------------------------------
        :param l1_shape:  'AL1_shape' or 'BL1_shape'
        :param isdouble:  True or False
        :return:  data amount in l1_shape
        ---------------------------------------------------
        """
        data_amount_l1 = 0
        if Params.TILING.get(l1_shape) is None:
            raise RuntimeError('{} can not be None'.format(l1_shape))
        if Params.TILING.get(l1_shape) == []:
            if l1_shape == 'AL1_shape':
                data_amount_l1 = \
                    functools.reduce(lambda x, y: x * y,
                                     Params.DIM_MAP['A_matrix_dim'][1:]) \
                    // Params.TILING["block_dim"][2]
            if l1_shape == 'BL1_shape':
                data_amount_l1 = \
                    functools.reduce(lambda x, y: x * y,
                                     Params.DIM_MAP['B_matrix_dim']) \
                    // Params.TILING["block_dim"][1]
        else:
            l1_k = Params.TILING.get(l1_shape)[0]
            l1_mn = Params.TILING.get(l1_shape)[1]
            if l1_k == 0 or l1_mn == 0:
                raise RuntimeError('l1_k or l1_mn can not be zero')
            if l1_k % Params.block_reduce != 0:
                raise RuntimeError('l1_k can not be divided by BLOCK_REDUCE')
            if l1_shape == 'AL1_shape':
                data_amount_l1 = l1_k * l1_mn * \
                                 Params.TILING.get('CL0_matrix')[1] * \
                                 cce.BLOCK_IN * data_size

            else:
                data_amount_l1 = l1_k * l1_mn * \
                                 Params.TILING.get('CL0_matrix')[0] * \
                                 cce.BLOCK_OUT * data_size
            if isdouble == 2:
                data_amount_l1 = data_amount_l1 * 2
        return data_amount_l1

    # check tilling l0
    def _check_tilling_l0(l0_shape, l0_space, isdouble, data_size):
        """
         check tilling parameter in L0 buffer

         Parameter:
         --------------------------------------------------
        :param l0_shape: 'AL0_matrix' or 'BL0_matrix'
        :param l0_space: LO buffer size
        :param isdouble: True or False
        :return: None
        ---------------------------------------------------
        """
        row = Params.TILING.get(l0_shape)[0]
        col = Params.TILING.get(l0_shape)[1]
        if row == 0 or col == 0:
            raise RuntimeError('k, m, n in L0A/B can not be zero')
        data_amount_l0 = row * col * Params.TILING.get(l0_shape)[2] * \
                         Params.TILING.get(l0_shape)[3] * data_size * isdouble
        if Params.ops_mode == 'int8fp32':
            data_amount_l0 = data_amount_l0 // 2
        Params().print_debug('{} data mount(KB)'.format(l0_shape),
                             data_amount_l0 / 1024)
        if data_amount_l0 > l0_space:
            raise RuntimeError('tilling size exceed L0A/B Buffer')

    # check tilling l0c
    def _check_tilling_l0c(l0c_shape, l0c_space, isdouble):
        """
        check tilling parameter in L0c

        Parameter:
        -----------------------------------------------------
        :param l0c_shape:'CL0_matrix'
        :param l0c_space: LOC buffer size
        :param isdouble: True or False
        :return: None
        -----------------------------------------------------
        """
        cl0_n = Params.TILING.get(l0c_shape)[0]
        cl0_m = Params.TILING.get(l0c_shape)[1]
        if Params.TILING.get('BL0_matrix') != []:
            if cl0_m == 0 or cl0_n == 0:
                raise RuntimeError('cl0_m, cl0_n can not be zero')

            data_amount_cl0 = cl0_m * cl0_n \
                              * Params.TILING.get(l0c_shape)[2] \
                              * Params.TILING.get(l0c_shape)[3] \
                              * 4 * isdouble
            Params().print_debug('data_amount_l0c(KB)',
                                 data_amount_cl0 / 1024)
            if data_amount_cl0 > l0c_space:
                raise RuntimeError('tilling size exceed L0C Buffer')

    def _check_tiling_ub():
        """
        check tilling parameter in ub

        Parameter:
        ------------------------------------------------------
        :return: None
        -------------------------------------------------------
        """

        # check tilling cub
        def _check_tilling_cub():
            """
            check tilling parameter in cub

            Parameter:
            ------------------------------------------------------
            :return: None
            -------------------------------------------------------
            """
            nc_factor = Params.TILING.get('CUB_matrix')[0]
            mc_factor = Params.TILING.get('CUB_matrix')[1]
            if Params.TILING.get('CL0_matrix')[0] % nc_factor != 0:
                raise RuntimeError('nc_factor is not factor of nc')

            manual_pingpong_buffer = \
                Params.TILING.get('manual_pingpong_buffer')
            is_double = manual_pingpong_buffer.get('CUB_pbuffer')
            data_amount_cub = nc_factor \
                              * mc_factor \
                              * Params.TILING.get('CUB_matrix')[2] \
                              * Params.TILING.get('CUB_matrix')[3] \
                              * Params.OUTPUT_SIZE.get(Params.ops_mode) \
                              * is_double \
                              * (Params.CUB_FUSED_NUM.get(Params.ops_mode) + 1)
            return data_amount_cub

        def _check_aub_bub_tiling(shape_head):
            """
            check tilling parameter in ub

            Parameter: shape_head: 'A' or 'B'
            ------------------------------------------------------
            :return: None
            -------------------------------------------------------
            """
            # check ub between l1
            ub_name = shape_head + 'UB_shape'
            l1_name = shape_head + 'L1_shape'
            l0_name = shape_head + 'L0_matrix'
            if Params.TILING.get(ub_name) is None:
                return 0
            data_size = 2
            if Params.ops_mode in ('int8fp32', 'int8int32'):
                data_size = 1
            if Params.TILING.get(ub_name)[0] \
                    % Params.TILING.get(l1_name)[0] != 0:
                raise RuntimeError('k_{head}UB should be multiple of '
                                   'the k_{head}L1'.format(head=shape_head))
            if shape_head == 'A':
                m_n_pos = 0
                m_n_tags = 'm'
            else:
                m_n_pos = 1
                m_n_tags = 'n'
            if Params.TILING.get(ub_name)[1] % \
                    (Params.TILING.get(l1_name)[1] *
                     Params.TILING.get(l0_name)[m_n_pos]) != 0:
                raise RuntimeError('{m_n_tags}_{shape_head}UB '
                                   'should be multiple of the '
                                   '{m_n_tags}_{shape_head}L1' \
                                   .format(m_n_tags=m_n_tags,
                                           shape_head=shape_head))
            manual_pingpong_buffer = \
                Params.TILING.get('manual_pingpong_buffer')
            is_double = manual_pingpong_buffer.get(shape_head + 'UB_pbuffer')
            data_amount_ub = Params.TILING.get(ub_name)[0] \
                             * Params.TILING.get(ub_name)[1] \
                             * 16 * is_double * data_size
            return data_amount_ub

        data_amount_aub = _check_aub_bub_tiling('A')*(
            Params.AUB_FUSED_NUM.get(Params.ops_mode) // 10 + 1)
        data_amount_bub = _check_aub_bub_tiling('B')*(
            Params.BUB_FUSED_NUM.get(Params.ops_mode) // 10 + 1)
        data_amount_cub = _check_tilling_cub()
        Params().print_debug('data_amount_aub(KB)', data_amount_aub / 1024)
        Params().print_debug('data_amount_bub(KB)', data_amount_bub / 1024)
        Params().print_debug('data_amount_cub(KB)', data_amount_cub / 1024)
        if Params.ops_mode == 'fp16fp16':
            alpha_beta_size = 2 * 2
        else:
            alpha_beta_size = 4 * 2

        total_size_ub = data_amount_aub + \
                        data_amount_bub + \
                        data_amount_cub + \
                        alpha_beta_size
        Params().print_debug('total_data_amount_ub(KB)', total_size_ub / 1024)
        if total_size_ub > Params.UB_SPACE_SIZE:
            raise RuntimeError('tilling size exceed UB Buffer')

    def _get_tiling_l0a_l0b(cl0_matrix, l0_matrix, instr):
        '''
        :param cl0_matrix:
        :param l0_matrix:
        :param instr:
        :return:
        '''
        if Params.ops_mode in ('fp16fp16', 'fp16fp32'):
            dtype = 'float16'
        else:
            dtype = 'int8'

        k_dim = Params.DIM_MAP.get("A_matrix_dim")[-3]
        if instr == 'A':
            if l0_matrix != []:
                full_ab = [cl0_matrix[1],
                           l0_matrix[0], CUBE_MKN[dtype]['mac'][0],
                           CUBE_MKN[dtype]['mac'][1], 1]
            else:
                full_ab = [cl0_matrix[1],
                           k_dim, CUBE_MKN[dtype]['mac'][0],
                           CUBE_MKN[dtype]['mac'][1], 1]
        elif instr == 'B':
            if l0_matrix != []:
                full_ab = [l0_matrix[1],
                           cl0_matrix[0], CUBE_MKN[dtype]['mac'][2],
                           CUBE_MKN[dtype]['mac'][1], 1]
            else:
                full_ab = [k_dim,
                           cl0_matrix[0], CUBE_MKN[dtype]['mac'][2],
                           CUBE_MKN[dtype]['mac'][1], 1]
        else:
            raise RuntimeError("instr should be A or B")

        return full_ab

    def get_tiling_result():
        '''
        :param None:
        :return: TILING result and data_byte
        '''
        a_type = Params.TENSOR_MAP["a_placehold"].dtype
        b_type = Params.TENSOR_MAP["b_placehold"].dtype
        c_type = Params.TENSOR_MAP["c_gm"].dtype

        if a_type == 'float16':
            a_shape = [1, Params.DIM_MAP['a_shape'][0],
                       Params.DIM_MAP['a_shape'][1], 16, 16]
            b_shape = [Params.DIM_MAP['b_shape'][1] * 16,
                       Params.DIM_MAP['b_shape'][0], 1, 1, 16]
        else:
            a_shape = [1, Params.DIM_MAP['a_shape'][0],
                       Params.DIM_MAP['a_shape'][1], 16, 32]
            b_shape = [Params.DIM_MAP['b_shape'][0] * 32,
                       Params.DIM_MAP['b_shape'][1], 1, 1, 16]

        pad_l = Params.AUB_FUSED_NUM.get(Params.ops_mode)
        pad_r = Params.BUB_FUSED_NUM.get(Params.ops_mode)
        fused_num = Params.CUB_FUSED_NUM.get(Params.ops_mode)
        mad_type = Params.MAD_TYPE.get(Params.ops_mode)
        tiling = tiling_query(a_shape, b_shape, c_shape=None,
                              a_dtype=a_type, b_dtype=b_type, c_dtype=c_type,
                              mad_dtype=mad_type,
                              padl=pad_l, padr=pad_r,
                              padu=0, padd=0, strideh=1, stridew=1,
                              strideh_expand=1, stridew_expand=1,
                              dilationh=1, dilationw=1, group=1,
                              fused_double_operand_num=fused_num,
                              bias_flag=0, op_tag='matmul')

        Params().print_debug('-----------auto tiling result-----------------')
        Params().print_debug(tiling)
        Params().print_debug('----------------------------------------------')
        return tiling

    def _check_tiling_al1():
        manual_pingpong_buffer = Params.TILING.get('manual_pingpong_buffer')
        data_amount_al1 = 0
        if Params.TILING.get("AL1_shape") is None:
            raise RuntimeError('AL1_shape can not be None')
        if Params.TILING.get("AL1_shape") != [] \
                and len(Params.TILING.get("AL1_shape")) \
                != Params.CONST_AL1_SHAPE_DIM:
            raise RuntimeError('AL1_shape should be Four')
        data_amount_al1 = \
            _get_data_amount_l1('AL1_shape',
                                manual_pingpong_buffer.get('AL1_pbuffer'),
                                Params.INPUT_SIZE.get(Params.ops_mode))
        return data_amount_al1

    def _check_tiling_bl1():
        manual_pingpong_buffer = Params.TILING.get('manual_pingpong_buffer')
        data_amount_bl1 = 0
        if Params.TILING.get('BL1_shape') is None:
            raise RuntimeError('BL1 can not be None')
        if Params.TILING.get("BL1_shape") != [] \
                and len(Params.TILING.get("BL1_shape")) \
                != Params.CONST_BL1_SHAPE_DIM:
            raise RuntimeError('BL1_shape should be Four')
        data_amount_bl1 = \
            _get_data_amount_l1('BL1_shape',
                                manual_pingpong_buffer.get('BL1_pbuffer'),
                                Params.INPUT_SIZE.get(Params.ops_mode))
        return data_amount_bl1

    def _check_mul_al1_bl1():
        if Params.TILING.get('BL1_shape') and Params.TILING.get('AL1_shape'):
            k_al1 = Params.TILING.get("AL1_shape")[0]
            k_bl1 = Params.TILING.get("BL1_shape")[0]
            if k_al1 % k_bl1 != 0 and k_bl1 % k_al1 != 0:
                raise RuntimeError('kal1 should be divisible by kbl1 or kbl1'
                                   'should be divisible by kal1 ')
            if k_al1 % (Params.TILING.get('AL0_matrix')[1]
                        * Params.block_reduce) != 0:
                raise RuntimeError('ka should be divisible by kal1')
            if Params.TILING.get('BL0_matrix') and k_bl1 % \
                    (Params.TILING.get('BL0_matrix')[0]
                     * Params.block_reduce) != 0:
                raise RuntimeError('kb should be divisible by kbl1')

    def _check_tiling_l0a_l0b(data_amount_l1b):
        data_size = Params.L1_L0_SIZE.get(Params.ops_mode)
        manual_pingpong_buffer = Params.TILING.get('manual_pingpong_buffer')
        if Params.TILING.get('AL0_matrix') == []:
            Params.TILING['AL0_matrix'] = \
                _get_tiling_l0a_l0b(Params.TILING.get('CL0_matrix'),
                                    Params.TILING.get('BL0_matrix'), 'A')

        if Params.TILING.get('BL0_matrix') == []:
            Params.TILING['BL0_matrix'] = \
                _get_tiling_l0a_l0b(Params.TILING.get('CL0_matrix'),
                                    Params.TILING.get('AL0_matrix'), 'B')

        # check tilling in AL0 BL0
        if Params.TILING.get('AL0_matrix') is None \
                or Params.TILING.get('AL0_matrix') == []:
            raise RuntimeError('tiling[AL0_matrix] can not be None or []')
        _check_tilling_l0('AL0_matrix',
                          Params.L0_SPACE_SIZE,
                          manual_pingpong_buffer.get('AL0_pbuffer'),
                          data_size)
        if Params.TILING.get('BL0_matrix') is None:
            raise RuntimeError('tiling[BL0_matrix] can not be None')
        if Params.TILING.get('BL0_matrix') == []:
            data_amount_l0b = data_amount_l1b
            if data_amount_l0b > Params.L0_SPACE_SIZE:
                raise RuntimeError('tiling size exceed L0B Buffer')
        else:
            _check_tilling_l0('BL0_matrix',
                              Params.L0_SPACE_SIZE,
                              manual_pingpong_buffer.get('BL0_pbuffer'),
                              data_size)
            if Params.TILING.get('AL0_matrix')[1] != \
                    Params.TILING.get('BL0_matrix')[0]:
                raise RuntimeError('axis k in tilling AL0 is '
                                   'not equal to axis k in tilling BL0')

    Params.TILING = get_tiling_result()

    data_amount_al1 = _check_tiling_al1()
    data_amount_bl1 = _check_tiling_bl1()
    Params().print_debug('data_amount_al1:', data_amount_al1 / 1024)
    Params().print_debug('data_amount_bl1:', data_amount_bl1 / 1024)
    if data_amount_al1 + data_amount_bl1 > Params.L1_SPACE_SIZE:
        raise RuntimeError('tiling size exceed L1 Buffer')
    _check_mul_al1_bl1()
    _check_tiling_l0a_l0b(data_amount_bl1)
    manual_pingpong_buffer = Params.TILING.get('manual_pingpong_buffer')
    # check tilling in CL0
    _check_tilling_l0c('CL0_matrix',
                       Params.L0C_SPACE_SIZE,
                       manual_pingpong_buffer.get('CL0_pbuffer'))

    # check tilling in UB
    _check_tiling_ub()


def get_aicore_tiling_factor():
    """
    using tilling parameter calculate factor

    :return: tilling factor from ub to ddr
             tilling factor from l0c to ub
             tilling factor from ddr to AL1
             tilling factor from ddr to Bl1
    """
    l0c_tiling_factor = [Params.TILING["CL0_matrix"][0],
                         Params.TILING["CL0_matrix"][1]]

    # From LOC to GM [NumOfparts for N axis, NumOfparts for M axis ]
    l0c_parts = [int_ceil_div(Params.DIM_MAP.get("out_shape")[0]
                              // Params.TILING['block_dim'][1],
                              l0c_tiling_factor[0]),
                 int_ceil_div(Params.DIM_MAP.get("out_shape")[1]
                              // Params.TILING['block_dim'][2],
                              l0c_tiling_factor[1])]

    l0c_ub_tiling_factor = Params.TILING["CUB_matrix"]
    l0c_ub_parts = [int_ceil_div(l0c_tiling_factor[0],
                                 l0c_ub_tiling_factor[0]),
                    int_ceil_div(l0c_tiling_factor[1],
                                 l0c_ub_tiling_factor[1])]

    if Params.TILING["AL1_shape"]:  # AL1_shape = [n/16, m/16, 16, 16]
        al1_parts = [int_ceil_div(Params.DIM_MAP["A_matrix_dim"][1],
                                  int_ceil_div(Params.TILING["AL1_shape"][0],
                                               Params.block_reduce)),
                     int_ceil_div(l0c_parts[1],
                                  Params.TILING["AL1_shape"][1])]

    else:
        al1_parts = [1, 1]

    if Params.TILING["BL1_shape"]:
        bl1_parts = [int_ceil_div(Params.DIM_MAP["B_matrix_dim"][0],
                                  int_ceil_div(Params.TILING["BL1_shape"][0],
                                               Params.block_reduce)),
                     int_ceil_div(l0c_parts[0],
                                  Params.TILING["BL1_shape"][1])]
    else:
        bl1_parts = [1, 1]

    if Params.TILING["AUB_shape"]:
        aub_parts = [
            int_ceil_div(Params.DIM_MAP["A_matrix_dim"][1],
                         int_ceil_div(Params.TILING["AUB_shape"][0],
                                      Params.block_reduce)),
            int_ceil_div(l0c_parts[1],
                         int_ceil_div(Params.TILING["AUB_shape"][1],
                                      Params.TILING["CL0_matrix"][1]))
        ]

    else:
        aub_parts = [1, 1]

    if Params.TILING["BUB_shape"]:
        bub_parts = [
            int_ceil_div(Params.DIM_MAP["B_matrix_dim"][0],
                         int_ceil_div(Params.TILING["BUB_shape"][0],
                                      Params.block_reduce)),
            int_ceil_div(l0c_parts[0],
                         int_ceil_div(Params.TILING["BUB_shape"][1],
                                      Params.TILING["CL0_matrix"][0]))
        ]
    else:
        bub_parts = [1, 1]

    return l0c_tiling_factor, l0c_ub_parts, al1_parts, bl1_parts, aub_parts, bub_parts


def get_mmad_factor():
    """
    get tilling factor in mmad

    :return:tilling factor for al0
            tilling factor for bl0
            tilling factor for reduce axis
    """
    al0_factor = [Params.TILING.get("AL0_matrix")[0],
                  Params.TILING.get("AL0_matrix")[1]]
    bl0_factor = [Params.TILING.get("BL0_matrix")[0],
                  Params.TILING.get("BL0_matrix")[1]]
    reduce_factor = Params.TILING.get("BL0_matrix")[0]
    return al0_factor, bl0_factor, reduce_factor


def _bind_multi_core(sch, c_gm, bl1_at_c_axis, cl1_out_inner, # pylint: disable=too-many-arguments
                     al1_at_c_axis, c_slice_axis, batch):
    """
    bind multi core

    Parameter:
    sch: schedule
    c_gm: op
    bl1_at_c_axis: axis
    cl1_out_inner: axis
    al1_at_c_axis: axis
    c_slice_axis: axis
    batch: axis
    """
    if "block_dim" in Params.TILING:
        block_dim = Params.TILING["block_dim"]
    else:
        block_dim = [1, 1, 1, 1]
    blockidx = []
    # split batch axis
    batch_out_in = sch[c_gm].split(batch, nparts=block_dim[0])
    c_out_in = sch[c_gm].split(cl1_out_inner, nparts=block_dim[1])
    h_out_in = sch[c_gm].split(c_slice_axis, nparts=block_dim[2])

    # reorder
    sch[c_gm].reorder(batch_out_in[0], c_out_in[0], h_out_in[0],
                      batch_out_in[1], bl1_at_c_axis, c_out_in[1],
                      al1_at_c_axis, h_out_in[1])

    def _do_bind():
        blocks = block_dim[0] * block_dim[1] * block_dim[2]
        if blocks != 1:
            out_fused = sch[c_gm].fuse(batch_out_in[0],
                                       c_out_in[0],
                                       h_out_in[0])
            out_fused_out, _ = sch[c_gm].split(out_fused, nparts=blocks)
            bind_out, _ = sch[c_gm].split(out_fused_out, 1)
            blockidx = tvm.thread_axis("blockIdx.x")
            sch[c_gm].bind(bind_out, blockidx)
        else:
            blockidx = [batch_out_in[0], c_out_in[0], h_out_in[0]]

    _do_bind()
    return batch_out_in[1], h_out_in[1], c_out_in[1], blockidx


def get_l1_at_c_gm_axis(sch, c_gm, aub_parts, # pylint: disable=too-many-arguments
                        bub_parts, al1_parts, bl1_parts, c_in, c_slice_axis):
    """
    get l1 at c gm axis

    Parameter:
    ------------------------------------------------------------------
    :param sch: schedule
    :param c_gm: op
    :param aub_parts: tilling factor for aub
    :param bub_parts: tilling factor for bub
    :param al1_parts: tilling factor for al1
    :param bl1_parts: tilling factor for bl1
    :param c_in: axis
    :param c_slice_axis: axis
    -------------------------------------------------------------------
    """
    al1_part = [al1_parts[0] // aub_parts[0],
                al1_parts[1] // aub_parts[1]]
    bl1_part = [bl1_parts[0] // bub_parts[0],
                bl1_parts[1] // bub_parts[1]]

    bl1_at_c_axis, _ = sch[c_gm].split(c_in, nparts=bl1_part[1])
    al1_at_c_axis, c_slice_axis = sch[c_gm].split(c_slice_axis,
                                                  nparts=al1_part[1])

    return bl1_at_c_axis, al1_at_c_axis, c_slice_axis


def get_l1_mn_axis_at_l0c(sch, c_l0c, al0_m_outer, # pylint: disable=too-many-arguments
                          bl0_n_outer, al1_parts, bl1_parts, aub_parts,
                          bub_parts):
    """
    get l1 mn axis at l0c

    Parameter:
    ------------------------------------------------------------------
    :param sch: schedule
    :param c_gm: op
    :param al0_m_outer: axis
    :param bl0_n_outer: axis
    :param al1_parts: tilling factor for al1
    :param bl1_parts: tilling factor for bl1
    :param aub_parts: tilling factor for aub
    :param bub_parts: tilling factor for bub
    -------------------------------------------------------------------
    """
    aub_part = [al1_parts[0] // aub_parts[0],
                al1_parts[1] // aub_parts[1]]
    bub_part = [bl1_parts[0] // bub_parts[0],
                bl1_parts[1] // bub_parts[1]]
    al0_m_outer_outer, al0_m_outer_inner = sch[c_l0c].split(al0_m_outer,
                                                            nparts=aub_part[1])
    bl0_n_outer_outer, bl0_n_outer_inner = sch[c_l0c].split(bl0_n_outer,
                                                            nparts=bub_part[1])

    return bl0_n_outer_outer, al0_m_outer_outer, bl0_n_outer_inner, al0_m_outer_inner


def get_ub_k_axis_at_l0c(sch, c_l0c, al1_parts, # pylint: disable=too-many-arguments
                         bl1_parts, aub_parts, bub_parts, k_outer_outer):
    """
    get ub k axis at l0c

    Parameter:
    ------------------------------------------------------------------
    :param sch: schedule
    :param c_l0c: op
    :param al1_parts: tilling factor for al1
    :param bl1_parts: tilling factor for bl1
    :param aub_parts: tilling factor for aub
    :param bub_parts: tilling factor for bub
    :param k_outer_outer: axis
    -------------------------------------------------------------------
    """

    def _get_order(k_dict):
        tmp_order = sorted(k_dict.items(), key=lambda d: d[1], reverse=True)
        axis_order = [i[0] for i in tmp_order]
        k_val_order = [i[1] for i in tmp_order]
        return k_val_order, axis_order

    def _change_order(axis_order, name):
        ub_tag = name + 'ub'
        l1_tag = name + 'l1'
        if axis_order.index(ub_tag) < axis_order.index(l1_tag) \
                and k_dict.get(ub_tag) == k_dict.get(l1_tag):
            index_ub = axis_order.index(ub_tag)
            index_l1 = axis_order.index(l1_tag)
            axis_order[index_ub] = l1_tag
            axis_order[index_l1] = ub_tag

    def _shape_check():
        outer_factor = max(aub_parts[0], bub_parts[0])
        inner_factor = min(aub_parts[0], bub_parts[0])
        if outer_factor % inner_factor != 0:
            raise RuntimeError("illegal value of AUB_shape & BUB_shape")

    _shape_check()

    # get order
    k_dict = {
        'aub': aub_parts[0],
        'bub': bub_parts[0],
        'al1': al1_parts[0],
        'bl1': bl1_parts[0]
    }
    k_val_order, axis_order = _get_order(k_dict)
    _change_order(axis_order, 'a')
    _change_order(axis_order, 'b')

    reduce_axis_serial = list()

    def _do_split():
        outer = k_outer_outer
        for axis_parts in k_val_order:
            outer, inner = sch[c_l0c].split(outer, nparts=axis_parts)
            reduce_axis_serial.append(inner)
        reduce_axis_serial.append(outer)
        Params().print_debug('axis_order', axis_order)

    _do_split()
    return reduce_axis_serial, axis_order


def get_l0c_and_l1_axis(sch, c_gm, # pylint: disable=too-many-locals, too-many-arguments
                        l0c_factor, al1_parts, bl1_parts, batch=None):
    """
    get l0c and l1 axis

    Parameter:
    ------------------------------------------------------------------
    :param sch: schedule
    :param c_gm: op
    :param l0c_factor: tilling factor for l0c
    :param al1_parts: tilling factor for al1
    :param bl1_parts: tilling factor for bl1
    :param batch_in_axis: tilling factor for batch
    -------------------------------------------------------------------
    """

    def _get_reorder_flag():
        reorder_flag = False
        if Params.TILING["AL1_shape"] and al1_parts[0] != 1 \
                and Params.TILING["BL1_shape"] and bl1_parts[0] != 1:
            if bl1_parts[1] >= al1_parts[1]:
                reorder_flag = True
        if Params.TILING["AL1_shape"] and al1_parts[0] == 1 \
                and Params.TILING["BL1_shape"] and bl1_parts[0] == 1:
            if bl1_parts[1] >= al1_parts[1]:
                reorder_flag = True
        if Params.TILING["BL1_shape"] and bl1_parts[0] != 1 \
                and Params.TILING["AL1_shape"] and al1_parts[0] == 1:
            reorder_flag = True
        return reorder_flag

    # split c_gm according to factor of loc and out_shape
    l0c_n_outer, l0c_n_inner = sch[c_gm].split(c_gm.op.axis[0], l0c_factor[0])
    l0c_m_outer, l0c_m_inner = sch[c_gm].split(c_gm.op.axis[1], l0c_factor[1])
    sch[c_gm].reorder(l0c_n_outer, l0c_m_outer, l0c_n_inner, l0c_m_inner)
    # split c_gm according to factor of a_l1 and b_l1
    l1_m_outer_outer, l1_m_outer_inner = sch[c_gm].split(l0c_m_outer,
                                                         nparts=al1_parts[1])
    l1_n_outer_outer, cl1_out_inner = sch[c_gm].split(l0c_n_outer,
                                                      nparts=bl1_parts[1])

    # zN type not batch axis, so make one
    if not batch:
        batch, l1_n_outer_outer = sch[c_gm].split(l1_n_outer_outer, nparts=1)
    bl1_at_c_axis = l1_n_outer_outer
    al1_at_c_axis = l1_m_outer_outer
    c_slice_axis = l1_m_outer_inner

    batch_in, c_slice_axis, noii_axis, blockidx = _bind_multi_core(sch, c_gm,
                                                                   bl1_at_c_axis,
                                                                   cl1_out_inner,
                                                                   al1_at_c_axis,
                                                                   c_slice_axis,
                                                                   batch)
    # reorder al1 and bl1 axis according to double buffer
    batch_in_out_axis, batch_in_inner_axis = sch[c_gm].split(batch_in, factor=1)
    reorder_flag = _get_reorder_flag()
    if Params.ops_mode == 'int8fp32':
        reorder_flag = False
    if reorder_flag:
        sch[c_gm].reorder(l1_m_outer_outer,
                          batch_in_inner_axis,
                          l1_n_outer_outer)
    else:
        sch[c_gm].reorder(l1_n_outer_outer,
                          l1_m_outer_outer,
                          batch_in_inner_axis)
    al1_at_c_axis = l1_m_outer_outer

    return batch_in_out_axis, bl1_at_c_axis, al1_at_c_axis, c_slice_axis, l0c_n_inner, \
           l0c_m_inner, l1_m_outer_outer, noii_axis, blockidx


def get_l0a_and_l0b_axis(sch, c_l0c,  # pylint: disable=too-many-arguments
                         new_c_col_axis, al0_axis_factor,
                         bl0_axis_factor, reduce_axis_factor):
    """
    get l0a and l0b axis
    Parameter:
    ---------------------------------------------------------------
    :param sch: schedule
    :param c_l0c: op
    :param new_c_col_axis:
    :param al0_axis_factor:
    :param bl0_axis_factor:
    :param reduce_axis_factor:
    :return:
    ---------------------------------------------------------------
    """
    # split and get axis of reduce, al0_at_axis, bl0_at_axis
    reduce_out, reduce_inner = sch[c_l0c].op.reduce_axis
    al0_m_outer, al0_m_inner = sch[c_l0c].split(
        new_c_col_axis[1], al0_axis_factor[0])
    bl0_n_outer, bl0_n_inner = sch[c_l0c].split(
        new_c_col_axis[0], bl0_axis_factor[1])
    # for reduce axis, al0 and b_l0b should be the same
    k_outer_outer, k_outer_inner = sch[c_l0c].split(reduce_out,
                                                    reduce_axis_factor)

    sch[c_l0c].reorder(k_outer_outer, bl0_n_outer, al0_m_outer,
                       bl0_n_inner, al0_m_inner, new_c_col_axis[2],
                       new_c_col_axis[3], k_outer_inner, reduce_inner)

    return al0_m_outer, bl0_n_outer, k_outer_outer, bl0_n_inner


def get_al1_and_bl1_axis(sch, c_l0c, al1_parts, bl1_parts, k_outer_outer):
    """
    get al1 and bli axis
    Parameter:
    ---------------------------------------------------------------
    :param sch: schedule
    :param c_l0c: op
    :param al1_parts:
    :param bl1_parts:
    :param k_outer_outer:
    :return:
    ---------------------------------------------------------------
    """
    #  ============ a_l1 and b_l1 slice can be different with CUB & CL0 =====
    outer_factor = max(al1_parts[0], bl1_parts[0])
    inner_factor = min(al1_parts[0], bl1_parts[0])
    if outer_factor % inner_factor != 0:
        raise RuntimeError("illegal value of AL1_shape & BL1_shape")
    if al1_parts[0] > bl1_parts[0]:
        k_outer_outer_outer, k_outer_outer_inner = \
            sch[c_l0c].split(k_outer_outer, nparts=al1_parts[0])
        k_outer_outer_outer_outer, k_outer_outer_outer_inner = \
            sch[c_l0c].split(k_outer_outer_outer, nparts=(bl1_parts[0]))
        al1_at_l0c_axis = k_outer_outer_outer_inner
        bl1_at_l0c_axis = k_outer_outer_outer_outer
    else:
        k_outer_outer_outer, k_outer_outer_inner = \
            sch[c_l0c].split(k_outer_outer, nparts=bl1_parts[0])
        k_outer_outer_outer_outer, k_outer_outer_outer_inner = \
            sch[c_l0c].split(k_outer_outer_outer, nparts=(al1_parts[0]))
        al1_at_l0c_axis = k_outer_outer_outer_outer
        bl1_at_l0c_axis = k_outer_outer_outer_inner
    reduce_axis_serial = [k_outer_outer_outer_outer, k_outer_outer_outer_inner,
                          k_outer_outer_inner]
    return al1_at_l0c_axis, bl1_at_l0c_axis, reduce_axis_serial


def gemm_schedule(res, sch_list):# pylint: disable=r0914, r0915, r0912
    """
    schedule enter
    param:
    res: tensor
    sch_list: list of schedule
    """
    sch = sch_list[0]
    set_data_layout(res, sch)
    Params().print_ir_matmul('orgin', sch)
    if Params.ops_format_mode != 'ND':
        get_tiling()

    # get tensor
    a_l1, b_l1, a_l0a, \
    b_l0b, c_l0c, \
    c_gm, c_before_mul_ub = \
        Params.TENSOR_MAP["a_l1"], Params.TENSOR_MAP["b_l1"], \
        Params.TENSOR_MAP["a_l0a"], Params.TENSOR_MAP["b_l0b"], \
        Params.TENSOR_MAP["c_l0c"], \
        Params.TENSOR_MAP["c_gm"], Params.TENSOR_MAP["c_before_mul_ub"]
    if not (Params.ops_mode == 'fp16fp32' and Params.ops_format_mode == 'ND'):
        c_ub = Params.TENSOR_MAP["c_ub"]
    alpha_ub, alpha_c_ub, \
    beta_ub, bias_ub, \
    c_ub_temp, beta_bias_ub = \
        Params.TENSOR_MAP["alpha_ub"], \
        Params.TENSOR_MAP["alpha_c_ub"], \
        Params.TENSOR_MAP["beta_ub"], Params.TENSOR_MAP["bias_ub"], \
        Params.TENSOR_MAP["c_ub_temp"], Params.TENSOR_MAP["beta_bias_ub"]

    if Params.ops_mode == 'fp16fp16':
        alpha_temp_ub = Params.TENSOR_MAP["alpha_temp_ub"]
        beta_temp_ub = Params.TENSOR_MAP["beta_temp_ub"]
        float32_bias_ub = Params.TENSOR_MAP["float32_bias_ub"]

    def _get_a_zero_tensor():
        a_zero = None
        if  Params.is_a_matrix_align is False:
            a_zero = Params.TENSOR_MAP["a_zero"]
        return a_zero

    def _get_b_zero_tensor():
        b_zero = None
        if  Params.is_b_matrix_align is False:
            b_zero = Params.TENSOR_MAP["b_zero"]
        return b_zero

    if Params.ops_format_mode == 'ND':
        a_normalize_ub = Params.TENSOR_MAP["a_normalize_ub"]
        a_fract_k_ub = Params.TENSOR_MAP["a_fract_k_ub"]
        b_normalize_ub = Params.TENSOR_MAP["b_normalize_ub"]
        b_fract_ub = Params.TENSOR_MAP["b_fract_ub"]
        if Params.ops_mode == 'int8int32':
            a_zero = _get_a_zero_tensor()
            b_zero = _get_b_zero_tensor()
            b_matrix_transpose = Params.TENSOR_MAP[
                'b_transpose']
        else:
            b_zero = Params.TENSOR_MAP["b_zero"]

    if Params.ops_mode == 'int8fp32':
        a_ub, float16_a_ub, zz_a_ub, b_ub, float16_b_ub, zn_b_ub = \
            Params.TENSOR_MAP["a_ub"], Params.TENSOR_MAP["float16_a_ub"], \
            Params.TENSOR_MAP["zz_a_ub"], Params.TENSOR_MAP["b_ub"], \
            Params.TENSOR_MAP["float16_b_ub"], Params.TENSOR_MAP["zn_b_ub"]

    if Params.ops_mode == 'int8int32' and Params.ops_format_mode != 'ND':
        bias_ub_fract = Params.TENSOR_MAP['bias_ub_fract']

    # -------------------------------------boost_schedule_kit----------#
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

    def _tiling_l0_process():
        if tiling.get("BL0_matrix") != []:
            bl0_tiling_kb, bl0_tiling_nb, \
            bl0_tiling_n0, bl0_tiling_k0, _, _ = tiling.get("BL0_matrix")
        else:
            bl0_tiling_kb, bl0_tiling_nb, \
            bl0_tiling_n0, bl0_tiling_k0, = list(i.value for i in b_l0b.shape)
        return bl0_tiling_kb, bl0_tiling_nb, bl0_tiling_n0, bl0_tiling_k0

    def _tiling_l1_process():
        if tiling.get("AL1_shape") != []:
            al1_tiling_k, al1_tiling_m, _, _ = tiling.get("AL1_shape")
        else:
            al1_ka, al1_m, al1_k0 = list(i.value for i in a_l1.shape)
            align_m = CUBE_MKN[a_l0a.dtype]["mac"][0]
            al1_m_align = (al1_m + align_m - 1) // align_m * align_m
            al1_tiling_m = al1_m_align // (
                CUBE_MKN[c_l0c.dtype]["mac"][0]*cl0_tiling_mc)
            al1_tiling_k = al1_ka * al1_k0

        if tiling.get("BL1_shape") != []:
            bl1_tiling_k, bl1_tiling_n, _, _ = tiling.get("BL1_shape")
        else:
            bl1_k1, bl1_nb, bl1_k0, _ = list(i.value for i in b_l1.shape)
            bl1_tiling_k = bl1_k1 * bl1_k0
            bl1_tiling_n = bl1_nb // cl0_tiling_nc
        return al1_tiling_k, al1_tiling_m, bl1_tiling_k, bl1_tiling_n

    def _tiling_check_equal():
        if tiling.get("BL0_matrix") != []:
            if al0_tiling_ka != bl0_tiling_kb:
                raise RuntimeError("ka != kb.")
            if bl0_tiling_nb != cl0_tiling_nc:
                raise RuntimeError("nb != nc.")

        if al0_tiling_ma != cl0_tiling_mc:
            raise RuntimeError("ma != mc.")

    def _tiling_check_factor():
        if al1_tiling_k % al0_tiling_ka != 0:
            raise RuntimeError("k_AL1 % ka != 0.")

        if tiling.get("BL1_shape") != [] and tiling.get("BL0_matrix") != []:
            if bl1_tiling_k % bl0_tiling_kb != 0:
                raise RuntimeError("k_BL1 % kb != 0.")

        if cl0_tiling_nc % cub_tiling_nc_factor != 0:
            raise RuntimeError("nc % nc_factor != 0.")

        if tiling.get("BL1_shape") != []:
            if al1_tiling_k > bl1_tiling_k \
                    and al1_tiling_k % bl1_tiling_k != 0:
                raise RuntimeError("k_AL1 > k_BL1 but k_AL1 % k_BL1 != 0.")
            if bl1_tiling_k > al1_tiling_k \
                    and bl1_tiling_k % al1_tiling_k != 0:
                raise RuntimeError("k_BL1 > k_AL1 but k_BL1 % k_AL1 != 0.")

    def _cub_process():
        affine_cub = cub_tiling_mc_factor * cub_tiling_m0, cub_tiling_nc_factor * cub_tiling_n0

        c_before_mul_ub_shape = list(i.value for i in c_before_mul_ub.shape)
        status = Compare.compare([cub_tiling_nc_factor, \
                                  cub_tiling_mc_factor, \
                                  cub_tiling_m0, \
                                  cub_tiling_n0],
                                 c_before_mul_ub_shape)

        if status == Compare.EQUAL:
            pass
        elif status == Compare.LESS_EQ:
            sch_agent.attach_at(c_before_mul_ub, c_gm, affine_shape=affine_cub)
            sch_agent.same_attach(alpha_c_ub, c_before_mul_ub)
            sch_agent.same_attach(bias_ub, c_before_mul_ub)
            sch_agent.same_attach(beta_bias_ub, c_before_mul_ub)
            sch_agent.same_attach(c_ub_temp, c_before_mul_ub)
            if Params.ops_mode == 'fp16fp16':
                sch_agent.same_attach(float32_bias_ub, c_before_mul_ub)
                sch_agent.same_attach(c_ub, c_before_mul_ub)
            elif Params.ops_mode == 'int8int32':
                sch_agent.same_attach(c_ub, c_before_mul_ub)
        else:
            raise RuntimeError("c_ub attach error.")

        return affine_cub

    def _cl0_process(affine_cub):
        affine_l0c = cl0_tiling_mc * cl0_tiling_m0, \
                     cl0_tiling_nc * cl0_tiling_n0

        c_l0c_shape = list(i.value for i in c_l0c.shape)
        status_ori = Compare.compare([cl0_tiling_nc, \
                                      cl0_tiling_mc, \
                                      cl0_tiling_m0, \
                                      cl0_tiling_n0],
                                     c_l0c_shape)
        status = Compare.compare(affine_l0c, affine_cub)
        if status_ori == Compare.EQUAL:
            pass
        elif status == Compare.EQUAL:
            sch_agent.same_attach(c_l0c, c_before_mul_ub)
        elif status == Compare.GREATE_EQ:

            sch_agent.attach_at(c_l0c, c_gm, affine_shape=affine_l0c)
        else:
            raise RuntimeError("tensor_c_l0c attach error.")

        sch[c_l0c].buffer_align(
            (1, 1), \
            (1, 1), \
            (1, CUBE_MKN[c_l0c.dtype]["mac"][0]), \
            (1, CUBE_MKN[c_l0c.dtype]["mac"][2]), \
            (1, 1), \
            (1, CUBE_MKN[c_l0c.dtype]["mac"][1]))

    def _l0a_process():
        l0a2l0c_affine_shape = None, \
                               al0_tiling_ma, \
                               al0_tiling_m0, \
                               cl0_tiling_n0, \
                               al0_tiling_ka, \
                               al0_tiling_k0
        tiling_ori_l0a = al0_tiling_ma, \
                         al0_tiling_ka, \
                         al0_tiling_m0, \
                         al0_tiling_k0
        a_l0a_shape = list(i.value for i in a_l0a.shape)
        status_ori = Compare.compare(tiling_ori_l0a, a_l0a_shape)
        status = Compare.compare([al0_tiling_ma, \
                                  al0_tiling_m0, \
                                  al0_tiling_ka, \
                                  al0_tiling_k0], \
                                 [cl0_tiling_mc, \
                                  cl0_tiling_m0, \
                                  c_col_k1, \
                                  c_col_k0])

        if status_ori == Compare.EQUAL:
            pass
        elif status == Compare.EQUAL:
            sch_agent.same_attach(a_l0a, c_l0c)
        elif status == Compare.LESS_EQ:
            sch_agent.attach_at(a_l0a, c_l0c, affine_shape=l0a2l0c_affine_shape)
        elif status == Compare.GREATE_EQ:
            l0a2out_affine_shape = [al0_tiling_ma * al0_tiling_m0, None]
            sch_agent.attach_at(a_l0a, c_gm, affine_shape=l0a2out_affine_shape)
        else:
            raise RuntimeError("l0a attach error.")

        sch[a_l0a].buffer_align(
            (1, 1), \
            (1, 1), \
            (1, CUBE_MKN[a_l0a.dtype]["mac"][0]), \
            (1, CUBE_MKN[a_l0a.dtype]["mac"][0]))

    def _l0b_process():
        l0b2l0c_affine_shape = bl0_tiling_nb, \
                               None, \
                               cl0_tiling_m0, \
                               bl0_tiling_n0, \
                               bl0_tiling_kb, \
                               bl0_tiling_k0
        tiling_ori_l0b = bl0_tiling_kb, \
                         bl0_tiling_nb, \
                         bl0_tiling_n0, \
                         bl0_tiling_k0
        b_l0b_shape = list(i.value for i in b_l0b.shape)
        status_ori = Compare.compare(tiling_ori_l0b, b_l0b_shape)
        status = Compare.compare([bl0_tiling_nb, \
                                  bl0_tiling_n0, \
                                  bl0_tiling_kb, \
                                  bl0_tiling_k0], \
                                 [cl0_tiling_nc, \
                                  cl0_tiling_n0, \
                                  c_col_k1, \
                                  c_col_k0])
        if status_ori == Compare.EQUAL:
            pass
        elif status == Compare.EQUAL:
            sch_agent.same_attach(b_l0b, c_l0c)
        elif status == Compare.LESS_EQ:
            sch_agent.attach_at(b_l0b, c_l0c, affine_shape=l0b2l0c_affine_shape)
        elif status == Compare.GREATE_EQ:
            l0b2out_affine_shape = [None, bl0_tiling_nb * bl0_tiling_n0]
            sch_agent.attach_at(b_l0b, c_gm, affine_shape=l0b2out_affine_shape)
        else:
            raise RuntimeError("l0b attach error.")

    def _al1_process():
        if al1_tiling_m == 0 or al1_tiling_k == 0:
            return
        l1_ma = al1_tiling_m * al0_tiling_ma
        l1_ka = (al1_tiling_k + al0_tiling_k0 - 1) // al0_tiling_k0
        if Params.ops_mode == 'int8int32':
            tiling_ori_al1 = l1_ka, l1_ma, al0_tiling_m0, al0_tiling_k0
        else:
            tiling_ori_al1 = l1_ka, l1_ma * al0_tiling_m0, al0_tiling_k0
        al1_shape = list(i.value for i in a_l1.shape)
        l1a2l0c_affine_shape = None, \
                               l1_ma, \
                               al0_tiling_m0, \
                               cl0_tiling_n0, \
                               l1_ka, \
                               al0_tiling_k0
        status_ori = Compare.compare(tiling_ori_al1, al1_shape)
        status = Compare.compare([l1_ma, \
                                  al0_tiling_m0, \
                                  l1_ka, \
                                  al0_tiling_k0], \
                                 [cl0_tiling_mc, \
                                  cl0_tiling_m0, \
                                  c_col_k1, \
                                  c_col_k0])
        if status_ori == Compare.EQUAL:
            # al1 full load but tiling.get("AL1_shape") is not []
            pass
        elif status == Compare.EQUAL:
            sch_agent.same_attach(a_l1, c_l0c)
        elif status == Compare.LESS_EQ:
            sch_agent.attach_at(a_l1, c_l0c, affine_shape=l1a2l0c_affine_shape)
        elif status == Compare.GREATE_EQ:
            l1a2out_affine_shape = [l1_ma * al0_tiling_m0, None]
            sch_agent.attach_at(a_l1, c_gm, affine_shape=l1a2out_affine_shape)
        else:
            raise RuntimeError("a_l1 atach error.")

    def _bl1_process():
        if tiling.get("BL1_shape") != []:
            l1_nb = bl1_tiling_n * bl0_tiling_nb
            l1_kb = (bl1_tiling_k + bl0_tiling_k0 - 1) // bl0_tiling_k0
            l1b2l0c_affine_shape = l1_nb, \
                                   None, \
                                   cl0_tiling_m0, \
                                   bl0_tiling_n0, \
                                   l1_kb, \
                                   bl0_tiling_k0
            tiling_ori_bl1 = l1_kb, l1_nb, bl0_tiling_n0, bl0_tiling_k0
            bl1_shape = list(i.value for i in b_l1.shape)
            status_ori = Compare.compare(tiling_ori_bl1, bl1_shape)
            status = Compare.compare([l1_nb, \
                                      bl0_tiling_n0, \
                                      l1_kb, \
                                      bl0_tiling_k0], \
                                     [cl0_tiling_nc, \
                                      cl0_tiling_n0, \
                                      c_col_k1, \
                                      c_col_k0])
            if status_ori == Compare.EQUAL:
                # bl1 full load but tiling.get("BL1_shape") is not []
                pass
            elif status == Compare.EQUAL:
                sch_agent.same_attach(b_l1, c_l0c)
            elif status == Compare.LESS_EQ:
                sch_agent.attach_at(b_l1, c_l0c, affine_shape=l1b2l0c_affine_shape)
            elif status == Compare.GREATE_EQ:
                l1b2out_affine_shape = [None, l1_nb * bl0_tiling_n0]
                sch_agent.attach_at(b_l1, c_gm, affine_shape=l1b2out_affine_shape)
            else:
                raise RuntimeError("b_l1 attach error.")

    def _aub_process():
        l1_ma = al1_tiling_m * al0_tiling_ma
        l1_ka = (al1_tiling_k + al0_tiling_k0 - 1) // al0_tiling_k0

        aub_tiling_k0 = CUBE_MKN[a_fract_k_ub.dtype]["mac"][1]
        aub_tiling_m0 = 16

        a_ub_ori_shape = list(i.value for i in a_fract_k_ub.shape)

        if Params.ops_mode == 'int8int32':
            tiling_ori_aub = (aub_tiling_k+aub_tiling_k0-1)//aub_tiling_k0, \
                             aub_tiling_m, aub_tiling_m0, al0_tiling_k0
            aub_l1_shape = [(aub_tiling_k+aub_tiling_k0-1)//aub_tiling_k0, \
                         aub_tiling_m, aub_tiling_m0, \
                         aub_tiling_k0]
        else:
            tiling_ori_aub = (aub_tiling_k+aub_tiling_k0-1)//aub_tiling_k0, \
                                 aub_tiling_m * aub_tiling_m0, al0_tiling_k0
            aub_l1_shape = [(aub_tiling_k +
                             aub_tiling_k0 - 1)//aub_tiling_k0, \
                             aub_tiling_m * aub_tiling_m0, \
                             aub_tiling_k0]

        status_ori = Compare.compare(tiling_ori_aub, a_ub_ori_shape)
        status_l1 = Compare.compare(
            [aub_tiling_m, aub_tiling_m0,
             (aub_tiling_k + aub_tiling_k0 - 1) // aub_tiling_k0,
             aub_tiling_k0], [l1_ma, al0_tiling_m0, l1_ka, al0_tiling_k0])
        status_l0c = Compare.compare(
            [aub_tiling_m, aub_tiling_m0,
             (aub_tiling_k + aub_tiling_k0 - 1) // aub_tiling_k0,
             aub_tiling_k0], [cl0_tiling_mc, cl0_tiling_m0, c_col_k1,
                              c_col_k0])

        if status_ori == Compare.EQUAL:
            pass
        elif status_l1 == Compare.EQUAL:
            sch_agent.same_attach(a_fract_k_ub, a_l1)
        elif status_l1 == Compare.LESS_EQ:
            sch_agent.attach_at(a_fract_k_ub, a_l1, aub_l1_shape)
        else:
            if status_l0c == Compare.EQUAL:
                sch_agent.same_attach(a_fract_k_ub, c_l0c)
            elif status_l0c == Compare.LESS_EQ:
                aub_l0c_affine_shape = (
                    None, aub_tiling_m, al0_tiling_m0, cl0_tiling_n0,
                    (aub_tiling_k + aub_tiling_k0 - 1) // aub_tiling_k0,
                    aub_tiling_k0
                )
                sch_agent.attach_at(a_fract_k_ub, c_l0c,
                                    affine_shape=aub_l0c_affine_shape)
            else:
                aub_out_affine_shape = [aub_tiling_m * aub_tiling_m0, None]
                sch_agent.attach_at(a_fract_k_ub, c_gm,
                                    affine_shape=aub_out_affine_shape)
        sch_agent.same_attach(a_normalize_ub, a_fract_k_ub)


    def _bub_process():
        l1_nb = bl1_tiling_n * bl0_tiling_nb
        l1_kb = (bl1_tiling_k + bl0_tiling_k0 - 1) // bl0_tiling_k0

        bub_tiling_k0 = CUBE_MKN[b_fract_ub.dtype]["mac"][1]
        bub_tiling_n0 = 16

        b_ub_ori_shape = list(i.value for i in b_fract_ub.shape)

        tiling_ori_bub = (bub_tiling_k + bub_tiling_k0 - 1) // bub_tiling_k0, \
                         bub_tiling_n, bub_tiling_k0, bub_tiling_n0

        status_ori = Compare.compare(tiling_ori_bub, b_ub_ori_shape)
        status_l1 = Compare.compare(
            [bub_tiling_n, bub_tiling_n0,
             (bub_tiling_k + bub_tiling_k0 - 1) // bub_tiling_k0,
             bub_tiling_k0], [l1_nb, bl0_tiling_n0, l1_kb, bl0_tiling_k0])
        status_l0c = Compare.compare(
            [bub_tiling_n, bub_tiling_n0,
             (bub_tiling_k + bub_tiling_k0 - 1) // bub_tiling_k0,
             bub_tiling_k0],
            [cl0_tiling_nc, cl0_tiling_n0, c_col_k1, c_col_k0])
        if status_ori == Compare.EQUAL:
            pass
        elif status_l1 == Compare.EQUAL:
            sch_agent.same_attach(b_fract_ub, b_l1)
        elif status_l1 == Compare.LESS_EQ:
            bub_shape = [(bub_tiling_k + bub_tiling_k0 - 1) // bub_tiling_k0, \
                         bub_tiling_n, \
                         bub_tiling_k0, \
                         bub_tiling_n0]
            sch_agent.attach_at(b_fract_ub, b_l1, bub_shape)
        else:
            if status_l0c == Compare.EQUAL:
                sch_agent.same_attach(b_fract_ub, c_l0c)
            elif status_l0c == Compare.LESS_EQ:
                bub_l0c_affine_shape = (
                    bub_tiling_n, None, cl0_tiling_m0, bl0_tiling_n0,
                    (bub_tiling_k + bub_tiling_k0 - 1) // bub_tiling_k0,
                    bub_tiling_k0)
                sch_agent.attach_at(b_fract_ub, c_l0c,
                                    affine_shape=bub_l0c_affine_shape)
            else:
                bub_out_affine_shape = [None, bub_tiling_n * bub_tiling_n0]
                sch_agent.attach_at(b_fract_ub, c_gm,
                                    affine_shape=bub_out_affine_shape)
        sch_agent.same_attach(b_normalize_ub, b_fract_ub)

    def _do_padding_ub_process():
        if Params.ops_mode == 'int8int32':
            if Params.is_b_matrix_align is False:
                sch_agent.same_attach(b_zero, b_fract_ub)
            if Params.is_a_matrix_align is False:
                sch_agent.same_attach(a_zero, a_fract_k_ub)
            sch_agent.same_attach(b_matrix_transpose, b_fract_ub)
        else:
            sch_agent.same_attach(b_zero, b_fract_ub)


    def _do_l1_process():
        if al1_tiling_k < bl1_tiling_k:
            _al1_process()
            _bl1_process()
        else:
            _bl1_process()
            _al1_process()

    def _do_ub_process():
        if aub_tiling_k < bub_tiling_k:
            _aub_process()
            _bub_process()
        else:
            _bub_process()
            _aub_process()

    def _double_buffer():
        if tiling.get("manual_pingpong_buffer").get("AL1_pbuffer") == 2:
            sch[a_l1].double_buffer()

        if tiling.get("manual_pingpong_buffer").get("BL1_pbuffer") == 2:
            sch[b_l1].double_buffer()

        if tiling.get("manual_pingpong_buffer").get("AL0_pbuffer") == 2:
            sch[a_l0a].double_buffer()

        if tiling.get("manual_pingpong_buffer").get("BL0_pbuffer") == 2:
            sch[b_l0b].double_buffer()

        if tiling.get("manual_pingpong_buffer").get("CL0_pbuffer") == 2:
            sch[c_l0c].double_buffer()

        if tiling.get("manual_pingpong_buffer").get("CUB_pbuffer") == 2:
            if Params.ops_mode == 'int8int32':
                sch[c_ub].double_buffer()
                sch[c_before_mul_ub].double_buffer()
                sch[alpha_c_ub].double_buffer()
                sch[bias_ub].double_buffer()
                sch[beta_bias_ub].double_buffer()
                sch[c_ub_temp].double_buffer()

    def _buffer_align():
        sch[bias_ub].buffer_align((1, 16), (1, 16))
        sch[beta_bias_ub].buffer_align((1, 16), (1, 16))
        sch[c_ub_temp].buffer_align((1, 16), (1, 16))
        if Params.ops_mode == 'fp16fp16':
            sch[float32_bias_ub].buffer_align((1, 16), (1, 16))
        if Params.ops_mode == 'int8int32':
            sch[c_ub].buffer_align((1, 16), (1, 16))


    def _emit_insn_common():# pylint:disable=too-many-locals
        scopes_intrins = sch_agent[c_l0c].intrin_scopes(6)
        scope_insn = scopes_intrins[0]
        scopes_intrins_cub = sch_agent[c_before_mul_ub].intrin_scopes(4)
        scope_cub = scopes_intrins_cub[0]
        scopes_intrins_c = sch_agent[c_gm].intrin_scopes(2)
        scope_c = scopes_intrins_c[0]
        sch_agent[b_normalize_ub].emit_insn(
            sch_agent[b_normalize_ub].op.axis[0], 'dma_copy')
        sch_agent[b_l1].emit_insn(sch_agent[b_l1].op.axis[0], 'dma_copy')
        sch_agent[b_l0b].emit_insn(sch_agent[b_l0b].op.axis[0], 'dma_copy')
        sch_agent[a_normalize_ub].emit_insn(
            sch_agent[a_normalize_ub].op.axis[0], 'dma_copy')
        sch_agent[a_l1].emit_insn(sch_agent[a_l1].op.axis[0], 'dma_copy')
        sch_agent[a_l0a].emit_insn(sch_agent[a_l0a].op.axis[0], 'dma_copy')
        inner_k_axis = sch_agent[c_l0c].get_relate_scope(
            c_l0c.op.reduce_axis[0], scope_insn)
        if inner_k_axis:
            mad_dict = {"mad_pattern": 0,
                        "k_outer": sch_agent[c_l0c].get_relate_scope(
                            c_l0c.op.reduce_axis[0], scope_insn)}
        else:
            inner_nb, inner_mb, inner_mp, inner_np, inner_kb, inner_kp = scopes_intrins
            inner_ko, inner_ki = sch_agent[c_l0c].split(inner_kb, nparts=1)
            sch_agent[c_l0c].reorder(inner_ko, inner_nb, inner_mb, inner_mp,
                                     inner_np, inner_ki, inner_kp)
            mad_dict = {"mad_pattern": 0, "k_outer": [inner_ko]}
        sch_agent[c_l0c].emit_insn(scope_insn, 'mad', mad_dict)
        sch_agent[c_before_mul_ub].emit_insn(scope_cub, 'dma_copy')
        sch_agent[alpha_c_ub].emit_insn(
            sch_agent[alpha_c_ub].op.axis[0], 'vector_muls')
        sch_agent[bias_ub].emit_insn(
            sch_agent[bias_ub].op.axis[0], 'dma_copy')
        sch_agent[beta_bias_ub].emit_insn(
            sch_agent[beta_bias_ub].op.axis[0], 'vector_muls')
        outer1, inner1 = sch_agent[c_ub_temp].split(c_ub_temp.op.axis[0], 16)
        outer2, inner2 = sch_agent[c_ub_temp].split(c_ub_temp.op.axis[1], 16)
        sch_agent[c_ub_temp].reorder(outer1, outer2, inner1, inner2)
        sch_agent[c_ub_temp].emit_insn(inner1, 'vector_add')
        sch_agent[c_gm].emit_insn(scope_c, 'dma_copy', {'no_overlap': 1})


    def _emit_insn_int8int32():
        sch_agent[b_fract_ub].emit_insn(
            sch_agent[b_fract_ub].op.axis[0], 'dma_copy')
        sch_agent[a_fract_k_ub].emit_insn(
            sch_agent[a_fract_k_ub].op.axis[0], 'dma_copy')
        if Params.is_a_matrix_align is False:
            sch_agent[a_zero].emit_insn(
                sch_agent[a_zero].op.axis[0], 'vector_dup')
        if Params.is_b_matrix_align is False:
            sch_agent[b_zero].emit_insn(
                sch_agent[b_zero].op.axis[0], 'vector_dup')
        k_outer, k_inner = sch_agent[b_matrix_transpose].split(
            b_matrix_transpose.op.axis[1], factor=32)
        sch_agent[b_matrix_transpose].reorder(
            k_outer, b_matrix_transpose.op.axis[0], k_inner)
        sch_agent[b_matrix_transpose].emit_insn(
            sch_agent[b_matrix_transpose].op.axis[0], 'vnchwconv')
        sch_agent[c_ub].emit_insn(
            sch_agent[c_ub].op.axis[0], 'dma_copy')
        sch_agent[alpha_ub].emit_insn(
            sch_agent[alpha_ub].op.axis[0], 'dma_copy')
        sch_agent[beta_ub].emit_insn(
            sch_agent[beta_ub].op.axis[0], 'dma_copy')


    def _emit_insn_fp16fp16():
        sch_agent[b_zero].emit_insn(
            sch_agent[b_zero].op.axis[0], 'vector_dup')
        sch_agent[b_fract_ub].emit_insn(
            sch_agent[b_fract_ub].op.axis[0], 'vector_auto')
        sch_agent[a_fract_k_ub].emit_insn(
            sch_agent[a_fract_k_ub].op.axis[0], 'vector_auto')
        if Params.ops_mode == 'fp16fp16':
            sch_agent[alpha_temp_ub].emit_insn(
                sch_agent[alpha_temp_ub].op.axis[0], 'dma_copy')
            sch_agent[beta_temp_ub].emit_insn(
                sch_agent[beta_temp_ub].op.axis[0], 'dma_copy')
            sch_agent[alpha_ub].emit_insn(
                sch_agent[alpha_ub].op.axis[0], 'vector_conv')
            sch_agent[beta_ub].emit_insn(
                sch_agent[beta_ub].op.axis[0], 'vector_conv')
            sch_agent[float32_bias_ub].emit_insn(
                sch_agent[float32_bias_ub].op.axis[0], 'vector_conv')
            sch_agent[c_ub].emit_insn(
                sch_agent[c_ub].op.axis[0], 'vector_conv')
        else:
            sch_agent[alpha_ub].emit_insn(
                sch_agent[alpha_ub].op.axis[0], 'dma_copy')
            sch_agent[beta_ub].emit_insn(
                sch_agent[beta_ub].op.axis[0], 'dma_copy')

    def _buffer_reuse_fp16fp16():
        sch_agent[c_before_mul_ub].reused_by(alpha_c_ub)
        if Params.ops_mode == 'fp16fp16':
            sch_agent[float32_bias_ub].reused_by(beta_bias_ub)
            sch_agent[bias_ub].reused_by(c_ub)
        else:
            sch_agent[bias_ub].reused_by(beta_bias_ub)


    def _buffer_reuse_int8int32():
        sch_agent[c_before_mul_ub].reused_by(alpha_c_ub)
        if Params.is_a_matrix_align is False:
            sch_agent[a_normalize_ub].reused_by(a_zero)
        if Params.is_b_matrix_align is False:
            sch_agent[b_normalize_ub].reused_by(b_zero)
        sch[c_ub_temp].reused_by(c_ub)
        sch[beta_bias_ub].reused_by(c_ub_temp)
        sch[bias_ub].reused_by(beta_bias_ub)


    # -------------------------------------boost_schedule_kit end------#

    if Params.ops_format_mode == 'ND':
        tiling = get_tiling_result_nd()
        # tiling cub
        cub_tiling_nc_factor, cub_tiling_mc_factor, \
        cub_tiling_m0, cub_tiling_n0, _, _ = tiling.get("CUB_matrix")

        # tiling l0c
        cl0_tiling_nc, cl0_tiling_mc, \
        cl0_tiling_m0, cl0_tiling_n0, _, _ = tiling.get("CL0_matrix")

        # tiling l0a
        al0_tiling_ma, al0_tiling_ka, \
        al0_tiling_m0, al0_tiling_k0, _, _ = tiling.get("AL0_matrix")

        # tiling l0b
        bl0_tiling_kb, bl0_tiling_nb, \
        bl0_tiling_n0, bl0_tiling_k0 = _tiling_l0_process()

        # tiling al1, bl1
        al1_tiling_k, al1_tiling_m, \
        bl1_tiling_k, bl1_tiling_n = _tiling_l1_process()

        # tiling aub, bub
        if tiling.get("AUB_shape") is not None:
            aub_tiling_k, aub_tiling_m, _, _ = tiling.get("AUB_shape")
        if tiling.get("BUB_shape") is not None:
            bub_tiling_k, bub_tiling_n, _, _ = tiling.get("BUB_shape")

        _tiling_check_equal()
        _tiling_check_factor()

        c_col_k1, c_col_k0 = list(
            ax.dom.extent.value for ax in c_l0c.op.reduce_axis)

        Params().print_ir_matmul('orgin', sch)

        sch_agent = ScheduleAgent(sch)
        affine_cub = _cub_process()
        _cl0_process(affine_cub)

        _l0a_process()
        _l0b_process()
        _do_l1_process()
        _do_ub_process()
        _do_padding_ub_process()

        _buffer_align()
        _double_buffer()

        if Params.ops_mode == 'int8int32':
            multi_core_min_slice = 32
        else:
            multi_core_min_slice = 16

        if c_gm.shape[1].value >= multi_core_min_slice:
            ax_m, ax_n = sch_agent[c_gm].get_active_scopes()
            _, n_dim, m_dim, _ = tiling.get("block_dim")
            ax_core = sch_agent[c_gm].bind_core([ax_m, ax_n], [m_dim, n_dim])
            sch_agent.root_stage_at(c_gm, ax_core)

        if Params.ops_mode == 'int8int32':
            _buffer_reuse_int8int32()
            _emit_insn_int8int32()
        else:
            _buffer_reuse_fp16fp16()
            _emit_insn_fp16fp16()
        _emit_insn_common()


        sch_agent.apply()
        Params().print_ir_matmul('orgin_end', sch)
    else:
        # --------------get factor and parts from tiling----------------
        l0c_factor, l0c_ub_parts, al1_parts, bl1_parts, aub_parts, bub_parts = \
            get_aicore_tiling_factor()
        al0_axis_factor, bl0_axis_factor, reduce_axis_factor = get_mmad_factor()
        # -----------split and get axis of l0c, al1, bl1 or may aub , bub----------------
        if Params.ops_mode == 'int8fp32':
            batch_in_out_axis, bub_at_c_axis, aub_at_c_axis, \
            c_slice_axis, l0c_n_inner, l0c_m_inner, _, c_in, _ = \
                get_l0c_and_l1_axis(sch, c_gm, l0c_factor, aub_parts, bub_parts)
            bl1_at_c_axis, al1_at_c_axis, c_slice_axis = \
                get_l1_at_c_gm_axis(sch, c_gm, aub_parts, bub_parts,
                                    al1_parts, bl1_parts, c_in, c_slice_axis)
        else:
            batch_in_out_axis, bl1_at_c_axis, al1_at_c_axis, \
            c_slice_axis, l0c_n_inner, l0c_m_inner, _, _, _ = \
                get_l0c_and_l1_axis(sch, c_gm, l0c_factor, al1_parts, bl1_parts)

        # -----------attach tensor of CUB----------------
        def _get_cub_at_axis(sch, c_gm, l0c_n_inner, l0c_m_inner, l0c_ub_parts):
            """
            get_cub_at_axis
            param: sch:schedule
            param: c_gm:tensor
            param: l0c_n_inner:axis
            param: l0c_m_inner:axis
            param: l0c_ub_parts: spilt parts
            return c_gm_at_axis, l0c_n_inner_inner
            """
            l0c_n_inner_outer, l0c_n_inner_inner = sch[c_gm].split(l0c_n_inner,
                                                                   nparts=
                                                                   l0c_ub_parts[
                                                                       0])
            l0c_m_inner_outer, l0c_m_inner_inner = sch[c_gm].split(l0c_m_inner,
                                                                   nparts=
                                                                   l0c_ub_parts[
                                                                       1])
            sch[c_gm].reorder(l0c_n_inner_outer, l0c_m_inner_outer,
                              l0c_n_inner_inner, l0c_m_inner_inner)
            c_gm_at_axis = l0c_m_inner_outer
            return c_gm_at_axis, l0c_n_inner_inner

        c_gm_at_axis, l0c_n_inner_inner = _get_cub_at_axis(sch, c_gm,
                                                           l0c_n_inner,
                                                           l0c_m_inner,
                                                           l0c_ub_parts)

        def _attach_ub(sch, c_gm, at_axis):
            """
            tensor cub
            """
            tensor_cub_list = [
                c_ub,
                c_before_mul_ub,
                alpha_ub,
                alpha_c_ub,
                beta_ub,
                bias_ub,
                c_ub_temp,
                beta_bias_ub
            ]
            if Params.ops_mode == 'fp16fp16':
                tensor_cub_list += [
                    alpha_temp_ub,
                    beta_temp_ub,
                    float32_bias_ub
                ]
            elif Params.ops_mode == 'int8int32':
                tensor_cub_list += [
                    bias_ub_fract
                ]
            for tensor in tensor_cub_list:
                sch[tensor].compute_at(sch[c_gm], at_axis)

        _attach_ub(sch, c_gm, c_gm_at_axis)

        # -----------attach tensor of l0c----------------

        sch[c_l0c].compute_at(sch[c_gm], c_slice_axis)

        new_c_col_axis = [sch[c_l0c].op.axis[0], sch[c_l0c].op.axis[1],
                          sch[c_l0c].op.axis[2], sch[c_l0c].op.axis[3]]
        al0_m_outer, bl0_n_outer, k_outer_outer, bl0_n_inner = \
            get_l0a_and_l0b_axis(sch, c_l0c, new_c_col_axis, al0_axis_factor,
                                 bl0_axis_factor, reduce_axis_factor)
        l0a_at_axis = al0_m_outer
        l0b_at_axis = bl0_n_outer
        # ---split and get axis of al1_at_l0c_axis, bl1_at_l0c_axis---
        if Params.ops_mode == "int8fp32":
            bl0_n_outer_outer, al0_m_outer_outer, \
            bl0_n_outer_inner, al0_m_outer_inner = \
                get_l1_mn_axis_at_l0c(sch, c_l0c, al0_m_outer, bl0_n_outer,
                                      al1_parts, bl1_parts, aub_parts,
                                      bub_parts)

            reduce_axis_serial, axis_order = \
                get_ub_k_axis_at_l0c(sch, c_l0c, al1_parts, bl1_parts,
                                     aub_parts, bub_parts, k_outer_outer)
            aub_at_l0c_axis = reduce_axis_serial[axis_order.index('aub') + 1]
            bub_at_l0c_axis = reduce_axis_serial[axis_order.index('bub') + 1]
            al1_at_l0c_axis = reduce_axis_serial[axis_order.index('al1') + 1]
            bl1_at_l0c_axis = reduce_axis_serial[axis_order.index('bl1') + 1]

            def _reorder_mkn_axis():
                if (axis_order.index('aub') in [2, 3]) \
                        and (axis_order.index('bub') in [2, 3]):
                    if axis_order.index('al1') > axis_order.index('bl1'):
                        Params().print_debug('order 1-1')
                        sch[c_l0c].reorder(reduce_axis_serial[4],
                                           reduce_axis_serial[3],
                                           al0_m_outer_outer,
                                           reduce_axis_serial[2],
                                           bl0_n_outer_outer,
                                           reduce_axis_serial[1],
                                           reduce_axis_serial[0],
                                           bl0_n_outer_inner,
                                           al0_m_outer_inner)
                    else:
                        Params().print_debug('order 1-2')
                        sch[c_l0c].reorder(reduce_axis_serial[4],
                                           reduce_axis_serial[3],
                                           bl0_n_outer_outer,
                                           reduce_axis_serial[2],
                                           al0_m_outer_outer,
                                           reduce_axis_serial[1],
                                           reduce_axis_serial[0],
                                           bl0_n_outer_inner,
                                           al0_m_outer_inner)
                elif axis_order.index('aub') == 3:
                    Params().print_debug('order 2')
                    sch[c_l0c].reorder(reduce_axis_serial[4],
                                       al0_m_outer_outer,
                                       reduce_axis_serial[3],
                                       reduce_axis_serial[2],
                                       bl0_n_outer_outer,
                                       reduce_axis_serial[1],
                                       reduce_axis_serial[0],
                                       bl0_n_outer_inner,
                                       al0_m_outer_inner)
                elif axis_order.index('bub') == 3:
                    Params().print_debug('order 3')
                    sch[c_l0c].reorder(reduce_axis_serial[4],
                                       bl0_n_outer_outer,
                                       reduce_axis_serial[3],
                                       reduce_axis_serial[2],
                                       al0_m_outer_outer,
                                       reduce_axis_serial[1],
                                       reduce_axis_serial[0],
                                       bl0_n_outer_inner,
                                       al0_m_outer_inner)

            _reorder_mkn_axis()
            l0a_at_axis = al0_m_outer_inner
            l0b_at_axis = bl0_n_outer_inner

        else:
            al1_at_l0c_axis, bl1_at_l0c_axis, reduce_axis_serial = \
                get_al1_and_bl1_axis(sch, c_l0c, al1_parts, bl1_parts,
                                     k_outer_outer)

        # -----------attach tensor of a_l0a----------------
        sch[a_l0a].compute_at(sch[c_l0c], l0a_at_axis)
        sch[b_l0b].compute_at(sch[c_l0c], l0b_at_axis)

        Params().print_ir_matmul('before attach aub bub', sch)

        def _attach_aub_bub(al1_at_tensor, bl1_at_tensor):
            if Params.ops_mode == 'int8fp32':
                sch[zz_a_ub].split(zz_a_ub.op.axis[1], factor=2)
                zn_b_ub_k_out, _ = sch[zn_b_ub].split(zn_b_ub.op.axis[0],
                                                      factor=2)
                tensor_list = [a_ub, float16_a_ub, zz_a_ub]
                if Params.TILING["AUB_shape"]:
                    if aub_parts[0] != 1 or al1_at_tensor == 'c_l0c':
                        Params().print_debug("a_ub at c_l0c")
                        at_tensor = c_l0c
                        at_axis = aub_at_l0c_axis
                    else:
                        Params().print_debug("a_ub at c_gm")
                        at_tensor = c_gm
                        at_axis = aub_at_c_axis
                else:
                    Params().print_debug("a_ub at c_gm2")
                    at_tensor = c_gm
                    at_axis = batch_in_out_axis

                for tensor in tensor_list:
                    sch[tensor].compute_at(sch[at_tensor], at_axis)

                tensor_list = [b_ub, float16_b_ub, zn_b_ub]
                if Params.TILING["BUB_shape"]:
                    if bub_parts[0] != 1 or bl1_at_tensor == 'c_l0c':
                        Params().print_debug("b_ub at c_l0c")
                        at_tensor = c_l0c
                        at_axis = bub_at_l0c_axis
                    else:
                        Params().print_debug("b_ub at c_gm")
                        at_tensor = c_gm
                        at_axis = bub_at_c_axis
                else:
                    Params().print_debug("b_ub at c_gm 2")
                    at_tensor = c_gm
                    at_axis = batch_in_out_axis
                for tensor in tensor_list:
                    sch[tensor].compute_at(sch[at_tensor], at_axis)
                return zn_b_ub_k_out
            return None

        def _attach_al1_bl1():
            if Params.TILING["AL1_shape"]:
                if al1_parts[0] != 1:
                    Params().print_debug("a_l1 at c_l0c")
                    sch[a_l1].compute_at(sch[c_l0c], al1_at_l0c_axis)
                    al1_at_tensor = 'c_l0c'
                else:
                    Params().print_debug("a_l1 at c_gm")
                    sch[a_l1].compute_at(sch[c_gm], al1_at_c_axis)
                    al1_at_tensor = 'c_gm'
            else:
                Params().print_debug("a_l1 at c_gm 2")
                sch[a_l1].compute_at(sch[c_gm], batch_in_out_axis)
                al1_at_tensor = 'c_gm'

            if Params.TILING["BL1_shape"]:
                if bl1_parts[0] != 1:
                    Params().print_debug("b_l1 at c_l0c")
                    sch[b_l1].compute_at(sch[c_l0c], bl1_at_l0c_axis)
                    bl1_at_tensor = 'c_l0c'
                else:
                    Params().print_debug("b_l1 at c_gm")
                    sch[b_l1].compute_at(sch[c_gm], bl1_at_c_axis)
                    bl1_at_tensor = 'c_gm'
            else:
                Params().print_debug("b_l1 at c_gm2")
                sch[b_l1].compute_at(sch[c_gm], batch_in_out_axis)
                bl1_at_tensor = 'c_gm'

            return al1_at_tensor, bl1_at_tensor

        def _do_reused_by():
            # reused_by
            if Params.ops_mode == 'fp16fp16':
                sch[c_before_mul_ub].reused_by(alpha_c_ub, c_ub_temp, c_ub)

                sch[float32_bias_ub].reused_by(beta_bias_ub)

            elif Params.ops_mode == 'fp16fp32':
                sch[c_before_mul_ub].reused_by(alpha_c_ub, c_ub_temp, c_ub)
                sch[bias_ub].reused_by(beta_bias_ub)
            elif Params.ops_mode == 'int8int32':
                sch[c_before_mul_ub].reused_by(alpha_c_ub, c_ub_temp, c_ub)
                sch[bias_ub].reused_by(beta_bias_ub)
            elif Params.ops_mode == 'int8fp32':
                sch[c_before_mul_ub].reused_by(alpha_c_ub, c_ub_temp, c_ub)
                sch[bias_ub].reused_by(beta_bias_ub)

        def _do_double_buffer():
            # double buffer
            # a_l1 b_l1
            temp_tensor_list = list()
            if Params.TILING.get("manual_pingpong_buffer")["AL1_pbuffer"] == 2 \
                    and (Params.TILING["AL1_shape"] != []):
                temp_tensor_list += [a_l1]
            if Params.TILING.get("manual_pingpong_buffer")["BL1_pbuffer"] == 2 \
                    and (Params.TILING["BL1_shape"] != []):
                temp_tensor_list += [b_l1]

            # L0A L0B
            if Params.TILING.get("manual_pingpong_buffer")["AL0_pbuffer"] == 2:
                temp_tensor_list += [a_l0a]
            if Params.TILING.get("manual_pingpong_buffer")["BL0_pbuffer"] == 2:
                temp_tensor_list += [b_l0b]

            # c_l0c C_UB
            if Params.TILING.get("manual_pingpong_buffer")["CL0_pbuffer"] == 2:
                temp_tensor_list += [c_l0c]
            if Params.TILING.get("manual_pingpong_buffer")["CUB_pbuffer"] == 2:
                temp_tensor_list += [alpha_ub, alpha_c_ub, beta_ub,
                                     bias_ub, c_ub_temp, beta_bias_ub,
                                     c_ub, c_before_mul_ub]

                if Params.ops_mode == 'fp16fp16':
                    temp_tensor_list += [alpha_temp_ub,
                                         beta_temp_ub,
                                         float32_bias_ub]
                elif Params.ops_mode == 'int8int32':
                    temp_tensor_list += [bias_ub_fract]

            if Params.TILING.get("manual_pingpong_buffer")["AUB_pbuffer"] == 2:
                temp_tensor_list += [float16_a_ub, a_ub, zz_a_ub]
            if Params.TILING.get("manual_pingpong_buffer")["BUB_pbuffer"] == 2:
                temp_tensor_list += [float16_b_ub, b_ub, zn_b_ub]

            for temp_tensor in temp_tensor_list:
                sch[temp_tensor].double_buffer()

        def _do_intrin_mapping(zn_b_ub_k_out):
            # intrin mapping
            temp_tensor_list = [a_l0a, b_l0b, c_ub, c_before_mul_ub,
                                alpha_ub, beta_ub, bias_ub]
            if Params.TILING["AL1_shape"] is not None:
                temp_tensor_list.append(a_l1)
            if Params.TILING["BL1_shape"] is not None:
                temp_tensor_list.append(b_l1)
            if Params.ops_mode == 'fp16fp16':
                temp_tensor_list += [alpha_temp_ub, beta_temp_ub,
                                     float32_bias_ub]
            if Params.ops_mode == 'int8int32':
                sch[bias_ub_fract].emit_insn(bias_ub_fract.op.axis[0],
                                             'vector_adds')
            if Params.ops_mode == 'int8fp32':
                sch[zz_a_ub].emit_insn(zz_a_ub.op.axis[0], 'vector_auto')
                sch[zn_b_ub].emit_insn(zn_b_ub_k_out, "vector_auto")
                temp_tensor_list += [float16_a_ub, float16_b_ub, a_ub, b_ub]
            for temp_tensor in temp_tensor_list:
                sch[temp_tensor].emit_insn(temp_tensor.op.axis[0], 'dma_copy')

            sch[c_gm].emit_insn(l0c_n_inner_inner, 'dma_copy')
            sch[alpha_c_ub].emit_insn(alpha_c_ub.op.axis[0], 'vector_muls')

            if alpha_c_ub is not None:
                sch[c_ub_temp].emit_insn(c_ub_temp.op.axis[0], 'vector_add')
            else:
                sch[c_ub_temp].emit_insn(c_ub_temp.op.axis[0], 'vector_muls')

            sch[beta_bias_ub].emit_insn(beta_bias_ub.op.axis[0], 'vector_muls')

            if Params.ops_mode == "int8fp32":
                mad_dict = {"mad_pattern": cce.GEMM_MODE,
                            "k_outer": [reduce_axis_serial[0],
                                        reduce_axis_serial[1],
                                        reduce_axis_serial[2],
                                        reduce_axis_serial[3],
                                        reduce_axis_serial[4]]}
            else:
                mad_dict = {"mad_pattern": cce.GEMM_MODE,
                            "k_outer": [reduce_axis_serial[0],
                                        reduce_axis_serial[1],
                                        reduce_axis_serial[2]]}
            sch[c_l0c].emit_insn(bl0_n_inner, 'mad', mad_dict)

        Params().print_ir_matmul('finish', sch)

        al1_at_tensor, bl1_at_tensor = _attach_al1_bl1()
        zn_b_ub_k_out = _attach_aub_bub(al1_at_tensor, bl1_at_tensor)
        _do_reused_by()
        _do_double_buffer()
        _do_intrin_mapping(zn_b_ub_k_out)
    # clear global cache
    Params.TILING.clear()
    Params.DIM_MAP.clear()
    Params.TENSOR_MAP.clear()
    Params.is_b_matrix_align = True
    Params.is_a_matrix_align = True
    return True
