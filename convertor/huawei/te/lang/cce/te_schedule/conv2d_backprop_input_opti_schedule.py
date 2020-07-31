"""
Copyright 2018 Huawei Technologies Co., Ltd

conv2d_backprop_input schedule
"""
import functools  # pylint: disable=C0302
import operator
import re

import topi
from te import tvm
from te import platform as cce
from te.lang.cce import util
from te.platform import CUBE_MKN
from te.platform import intrinsic_check_support
from te.platform import get_soc_spec
from te.domain.tiling.tiling_query import tiling_query


# Don't modify,used in log_util
DX_SUPPORT_TAG_LOG_PREFIX = '#Conv2DBackpropInput only support#'
# default False
DEBUG_MODE = False
CONST_L1_SHAPE_DIM = 4
UB_SPACE_SIZE = get_soc_spec("UB_SIZE")
L1_SPACE_SIZE = get_soc_spec("L1_SIZE")
L0A_SPACE_SIZE = get_soc_spec("L0A_SIZE")
L0B_SPACE_SIZE = get_soc_spec("L0B_SIZE")
L0C_SPACE_SIZE = get_soc_spec("L0C_SIZE")
DTYPE_BYTE_MAP = {"float16": 2,
                  "float32": 4,
                  "int8": 1,
                  "int32": 4}
CUB_BUFFER_LIMIT = 4096
EPSINON = 1e-6
TENSOR_MAP = {}
TILING = {}
DIM_MAP = {}

FUSION_DX_DRELU = "dx+drelu"
FUSION_DX_ADD = "dx+vadd"
FUSION_DX_ADD_DRELU = "dx+vadd+drelu"
FUSION_DX_DEQUANT = "dx+dequant"
FUSION_DX_DEQUANT_QUANT = "dx+dequant+quant"
FUSION_NONE = ""
FUSION_TYPE_2_OPERAND_NUM = {FUSION_NONE: 0,
                             FUSION_DX_DRELU: 0.0625,
                             FUSION_DX_ADD: 1,
                             FUSION_DX_ADD_DRELU: 1.0625,
                             FUSION_DX_DEQUANT: 0,
                             FUSION_DX_DEQUANT_QUANT: 0
                            }

# broadcast should be 16
BRC_STANDARD_BLOCK_SIZE = 16

class DeconvParam:
    """
    class of DeconvTilingParam
    """

    def __init__(self):
        pass

    para_map = {"DATA_AMOUNT_CUB": 0,
                "FUSION_TYPE": FUSION_NONE}

    @staticmethod
    def update_para_map(key, value):
        """
        updata para map with key and value
        """
        DeconvParam.para_map[key] = value

    @staticmethod
    def get_para_map(key):
        """
        get value by key
        """
        return DeconvParam.para_map[key]


def print_debug(*params):
    '''
    print log if debug
    :param params: infos
    :return: None
    '''
    if DEBUG_MODE:
        print(params)


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
    if DEBUG_MODE or process == "debug":
        start = process + " IR start"
        end = process + " IR end\n"
        sch = sch.normalize()
        print(start)
        bounds = tvm.schedule.InferBound(sch)
        stmt = tvm.schedule.ScheduleOps(sch, bounds, True)
        print(stmt)
        print(end)


def _calc_channel_wise_para(tensor):
    def coeff(dtype, base):
        pattern = re.compile(r'[a-z]*(\d+)')
        base_res = pattern.match(base)
        dtype_res = pattern.match(dtype)
        if not base_res:
            raise AttributeError('base(%s) of coeff not match pattern' % base)
        if not dtype_res:
            raise AttributeError('x(%s) of coeff not match pattern' % dtype)
        return int_ceil_div(int(dtype_res.group(1)), int(base_res.group(1)))

    ub_gm_dtype = tensor.dtype
    l0c_ub_dtype = TENSOR_MAP["c_ub"].dtype
    cub_channel_coefficient = coeff(l0c_ub_dtype, ub_gm_dtype) - 1
    coefficient = 1
    if TENSOR_MAP.get("dilate_ub") is not None:
        dilate_hw = TENSOR_MAP.get("dilate_ub").op.attrs["dilate"]
        coefficient = int(dilate_hw[0] * dilate_hw[1])
    if TENSOR_MAP.get("input_ub") is not None:
        cub_channel_coefficient += 2 / coefficient
    if TENSOR_MAP.get("quant_reform") is not None:
        cub_channel_coefficient += 2 / coefficient
    if TENSOR_MAP.get("dilate_ub") is None and TENSOR_MAP.get("quant_vmuls_ub") is not None \
            or TENSOR_MAP.get("quant_vadds_ub") is not None:
        cub_channel_coefficient += 2

    fused_channel_wise = [0, 0, cub_channel_coefficient]

    return fused_channel_wise


def _get_data_amount_l1(l1_shape, isdouble):
    """
    using tilling parameter calculate data amount in l1

    Parameters:
    ---------------------------------------------------
    :param l1_shape:  'AL1_shape' or 'BL1_shape'
    :param isdouble:  True or False
    :return:  data amount in l1_shape
    ---------------------------------------------------
    """
    if TILING.get(l1_shape) is None:
        raise RuntimeError('{} can not be None'.format(l1_shape))
    if TILING.get(l1_shape) != [] \
            and len(TILING.get(l1_shape)) != CONST_L1_SHAPE_DIM:
        raise RuntimeError('{} should be {}'
                           .format(l1_shape, CONST_L1_SHAPE_DIM))

    if TILING.get(l1_shape) == []:
        if l1_shape == 'AL1_shape':
            data_amount_l1 = functools.reduce(lambda x, y: x * y,
                                              DIM_MAP['A_matrix_dim'][1:]) \
                             // TILING["block_dim"][2]
        if l1_shape == 'BL1_shape':
            data_amount_l1 = functools.reduce(lambda x, y: x * y,
                                              DIM_MAP['B_matrix_dim']) \
                             // TILING["block_dim"][1]
    else:
        block_m, block_k, block_n = CUBE_MKN[TENSOR_MAP.get("b_l1").dtype]["mac"]
        l1_k = TILING.get(l1_shape)[0]
        l1_mn = TILING.get(l1_shape)[1]
        if l1_k == 0 or l1_mn == 0:
            raise RuntimeError('l1_k or l1_mn can not be zero')
        if l1_k % block_k != 0:
            raise RuntimeError('l1_k can not be divided by {}'.format(block_k))
        if l1_shape == 'AL1_shape':
            data_amount_l1 = l1_k * l1_mn * TILING.get('CL0_matrix')[1] \
                             * block_m \
                             * DTYPE_BYTE_MAP[TENSOR_MAP.get("a_l1").dtype]
        else:
            data_amount_l1 = l1_k * l1_mn * TILING.get('CL0_matrix')[0] \
                             * block_n \
                             * DTYPE_BYTE_MAP[TENSOR_MAP.get("b_l1").dtype]
        if isdouble == 2:
            data_amount_l1 = data_amount_l1 * 2
    print_debug('{} data_amount_l1:{}'
                .format(l1_shape, data_amount_l1 / 1024))
    return data_amount_l1


def _check_tilling_l0(l0_shape, l0_space, isdouble):
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
    row = TILING.get(l0_shape)[0]
    col = TILING.get(l0_shape)[1]
    if row == 0 or col == 0:
        raise RuntimeError('k, m, n in L0A/B can not be zero')
    data_amount_l0 = row * col * TILING.get(l0_shape)[2] \
                     * TILING.get(l0_shape)[3] \
                     * DTYPE_BYTE_MAP[TENSOR_MAP.get("b_l0b").dtype] * isdouble
    print_debug('data_amount_l0A/B[KB]:', data_amount_l0 / 1024)
    if data_amount_l0 > l0_space:
        raise RuntimeError('tilling size exceed L0A/B Buffer')


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
    cl0_m, cl0_n = TILING.get(l0c_shape)[1], TILING.get(l0c_shape)[0]
    if TILING.get('BL0_matrix') != []:
        bl0_n = TILING.get('BL0_matrix')[1]
        if cl0_m == 0 or cl0_n == 0:
            raise RuntimeError('cl0_m, cl0_n can not be zero')
        if cl0_n != bl0_n:
            raise RuntimeError('axis n in tilling BL0 '
                               'is not equal to axis n in tilling CL0')
    data_amount_cl0 = cl0_m * cl0_n * TILING.get(l0c_shape)[2] \
                      * TILING.get(l0c_shape)[3] \
                      * DTYPE_BYTE_MAP[TENSOR_MAP.get("c_l0c").dtype] * isdouble
    print_debug('data_amount_l0C[KB]:', data_amount_cl0 / 1024)
    if data_amount_cl0 > l0c_space:
        raise RuntimeError('tilling size exceed L0C Buffer')


def _check_tilling_cub(cub_matrix, strideh, stridew, cub_space, isdouble):
    """
    check tilling parameter in cub

    Parameter:
    ------------------------------------------------------
    :param cub_matrix: 'CUB_matrix'
    :param strideh: stride_h
    :param stridew: stride_w
    :param cub_space: UB buffer size
    :param isdouble: True or False
    :return: None
    -------------------------------------------------------
    """
    def _get_dilate_cub_size():
        block_m, _, _ = CUBE_MKN[TENSOR_MAP.get("c_ub").dtype]["mac"]
        if cl0_m_extent < DIM_MAP['img_shape'][3]:
            raise RuntimeError('mc of CL0_matrix '
                               'smaller than weight of Image')
        if DIM_MAP['img_shape'][3] > block_m:
            check_ifmc_falg = \
                bool((cl0_m_extent // DIM_MAP['img_shape'][3]) *
                     DIM_MAP['img_shape'][3] * strideh * stridew
                     <= CUB_BUFFER_LIMIT)
            if cl0_m_extent % DIM_MAP['img_shape'][3] == 0 \
                    and check_ifmc_falg \
                    and DIM_MAP['img_shape'][2] % \
                    (cl0_m_extent // DIM_MAP['img_shape'][3]) == 0:
                n_is_hfactor = cl0_m_extent // DIM_MAP['img_shape'][3]
            else:
                n_is_hfactor = (cl0_m_extent - block_m) \
                               // DIM_MAP['img_shape'][3]
        else:
            check_ifmc_falg_s = False
            if cl0_m_extent % DIM_MAP['img_shape'][3] == 0:
                n_is_hfactor = cl0_m_extent // DIM_MAP['img_shape'][3]
                while DIM_MAP['img_shape'][2] % n_is_hfactor != 0:
                    n_is_hfactor = n_is_hfactor - 1
                check_ifmc_falg_s = bool(n_is_hfactor
                                         * DIM_MAP['img_shape'][3]
                                         * DIM_MAP['dilate_dim'][0]
                                         * DIM_MAP['dilate_dim'][1]
                                         > CUB_BUFFER_LIMIT)
            if cl0_m_extent % DIM_MAP['img_shape'][3] != 0 \
                    or check_ifmc_falg_s:
                n_is_hfactor = ((cl0_m_extent - block_m)
                                // DIM_MAP['img_shape'][3])
                while DIM_MAP['img_shape'][2] % n_is_hfactor != 0:
                    n_is_hfactor = n_is_hfactor - 1
        real_m = n_is_hfactor * DIM_MAP['img_shape'][3]
        dilate_cub_size = (1 + strideh * stridew) * nc_factor * real_m * \
                          TILING.get(cub_matrix)[3] \
                          * DTYPE_BYTE_MAP[TENSOR_MAP.get("c_ub").dtype] * isdouble
        return dilate_cub_size

    nc_factor, mc_factor = TILING.get(cub_matrix)[0], \
                           TILING.get(cub_matrix)[1]
    if mc_factor != TILING.get('CL0_matrix')[1]:
        raise RuntimeError('mc_factor is not equal to mc')
    if TILING.get('CL0_matrix')[0] % nc_factor != 0:
        raise RuntimeError('nc_factor is not factor of nc')
    cl0_m_extent = TILING["CL0_matrix"][1] * TILING["CL0_matrix"][2]
    if strideh > 1 or stridew > 1:
        data_amount_cub = _get_dilate_cub_size()
    else:
        data_amount_cub = nc_factor * mc_factor \
                          * TILING.get(cub_matrix)[2] \
                          * TILING.get(cub_matrix)[3] \
                          * DTYPE_BYTE_MAP[TENSOR_MAP.get("c_ub").dtype] * isdouble
    DeconvParam.update_para_map("DATA_AMOUNT_CUB", data_amount_cub)
    print_debug('DATA_AMOUNT_CUB[KB]:',
                DeconvParam.get_para_map("DATA_AMOUNT_CUB") / 1024)

    if DeconvParam.get_para_map("DATA_AMOUNT_CUB") > cub_space:
        raise RuntimeError('tilling size exceed CUB Buffer')


def _get_tiling_l0a_l0b(cl0_matrix, l0_matrix, instr):
    ''' get l0a and l0b matrix '''
    k_dim = DIM_MAP.get("A_matrix_dim")[-3]
    if instr == 'A':
        block_m, block_k, block_n = CUBE_MKN[TENSOR_MAP.get("a_l0a").dtype]["mac"]
        # l0_matrix is bl0_matrix:[kb, nb, n0, k0]
        if l0_matrix != []:
            full_ab = [cl0_matrix[1], l0_matrix[0],
                       block_m, block_k, 1]
        else:
            full_ab = [cl0_matrix[1], k_dim,
                       block_m, block_k, 1]
    elif instr == 'B':
        block_m, block_k, block_n = CUBE_MKN[TENSOR_MAP.get("b_l0b").dtype]["mac"]
        # l0_matrix is al0_matrix:[ma, ka, m0, k0]
        if l0_matrix != []:
            full_ab = [l0_matrix[1], cl0_matrix[0],
                       block_n, block_k, 1]
        else:
            full_ab = [k_dim, cl0_matrix[0], block_n, block_k, 1]
    else:
        raise RuntimeError("instr should be A or B")

    return full_ab


def _check_tilinng_k_l1():
    _, block_k, _ = CUBE_MKN[TENSOR_MAP.get("b_l1").dtype]["mac"]
    k_al1 = TILING.get("AL1_shape")[0]
    k_bl1 = TILING.get("BL1_shape")[0]
    if k_al1 % k_bl1 != 0 and k_bl1 % k_al1 != 0:
        raise RuntimeError('kal1 should be divisible by kbl1 or kbl1'
                           'should be divisible by kal1 ')
    if k_al1 % (TILING.get('AL0_matrix')[1] * block_k) != 0:
        raise RuntimeError('ka should be divisible by kal1')
    if TILING.get('BL0_matrix') \
            and k_bl1 % (TILING.get('BL0_matrix')[0] * block_k) != 0:
        raise RuntimeError('kb should be divisible by kbl1')


def _check_tiling_bl0_matrix(manual_pingpong_buffer, data_amount_l1b):
    if TILING.get('BL0_matrix') is None:
        raise RuntimeError('tiling[BL0_matrix] can not be None')
    if TILING.get('BL0_matrix') == []:
        data_amount_l0b = data_amount_l1b
        if data_amount_l0b > L0B_SPACE_SIZE:
            raise RuntimeError('tiling size exceed L0B Buffer')
    else:
        _check_tilling_l0('BL0_matrix', L0B_SPACE_SIZE,
                          manual_pingpong_buffer.get('BL0_pbuffer'))
        if TILING.get('AL0_matrix')[1] != TILING.get('BL0_matrix')[0]:
            raise RuntimeError('axis k in tilling AL0 is not '
                               'equal to axis k in tilling BL0')


def _check_ub_size_limit():
    block_m, _, block_n = CUBE_MKN[TENSOR_MAP.get("b_l1").dtype]["mac"]
    shape_deq_scale = TENSOR_MAP.get("deq_scale").shape
    size_deq_scale = functools.reduce(lambda x, y: x * y, shape_deq_scale)

    # min tiling size is 1*1*16*16*2(float16) or 1*1*16*32(int8)
    min_nc, min_mc = 2, 2
    for factor in range(1, DIM_MAP["img_shape"][2]):
        if factor * DIM_MAP["img_shape"][3] % block_m == 0:
            min_mc = 1
            break

    size_cub_min = min_nc * min_mc * block_m * block_n * 2

    # has deq_scale, dequant1, quant_reform, quant_vmuls_ub at least
    size_cub_min_total = size_deq_scale * 2 + size_cub_min * 3

    if TENSOR_MAP.get("dilate_ub") is not None:
        shape_dilate = TENSOR_MAP.get("dilate_ub").op.attrs["dilate"]
        size_cub_dilate = size_cub_min * shape_dilate[0] * shape_dilate[1]
        size_cub_min_total += size_cub_dilate

    if TENSOR_MAP.get("dequant_relu") is not None:
        size_cub_min_total += size_cub_min
    if TENSOR_MAP.get("dequant2_scalar") is not None:
        size_cub_min_total += size_cub_min
    if TENSOR_MAP.get("input_ub") is not None:
        size_cub_min_total += size_cub_min

    if operator.ge(int(size_cub_min_total), UB_SPACE_SIZE):
        TILING.clear()
        DIM_MAP.clear()
        TENSOR_MAP.clear()
        raise RuntimeError("Exceed Unified_Buffer max limit")


def get_tiling(tensor, fusion_type):
    """
    get tilling parameter from tilling guery and check all parameter
    """

    _, block_k, block_n = CUBE_MKN[TENSOR_MAP.get("b_l1").dtype]["mac"]
    # if filter dtype is int8, than channel block_size is 32
    if tensor.dtype == "int32" or fusion_type in (
            FUSION_DX_DEQUANT, FUSION_DX_DEQUANT_QUANT):
        co_dim, ci_dim, _, _ = DIM_MAP['filter_shape']
    else:
        ci_dim, co_dim, _, _ = DIM_MAP['filter_shape']
    filter_shape = [co_dim * block_k, ci_dim, 1, 1, block_n]

    if TENSOR_MAP.get("dilate_ub") is None:
        strideh, stridew = 1, 1
    else:
        strideh, stridew = DIM_MAP['dilate_dim']
    # times of the dx ub space
    fused_channel_wise = _calc_channel_wise_para(tensor)

    if fusion_type == FUSION_DX_DEQUANT_QUANT:
        filter_shape[1] = (filter_shape[1] + 1) // 2 * 2
        _check_ub_size_limit()

    bias_flag = _get_bias_flag()

    global TILING
    if not TILING:  # pylint:disable=W0603
        TILING = tiling_query(DIM_MAP['img_shape'],
                              filter_shape, c_shape=None,
                              a_dtype=TENSOR_MAP["img_placehold"].dtype,
                              b_dtype=TENSOR_MAP["filter_placehold"].dtype,
                              c_dtype=tensor.dtype,
                              mad_dtype=TENSOR_MAP["c_l0c"].dtype,
                              padl=0, padr=0, padu=0, padd=0,
                              strideh=1, stridew=1,
                              strideh_expand=strideh, stridew_expand=stridew,
                              dilationh=1, dilationw=1, group=1,
                              fused_double_operand_num
                              =FUSION_TYPE_2_OPERAND_NUM.get(fusion_type),
                              fused_channel_wise=fused_channel_wise,
                              bias_flag=bias_flag,
                              op_tag='conv2d_backprop_input')

    print_debug('opti dx shape:', 'filter:', filter_shape,
                'dy:', DIM_MAP['img_shape'], 'dx:', DIM_MAP['out_img_shape'])
    print_debug('tiling:', TILING)

    if TILING.get('AL0_matrix') == []:
        TILING['AL0_matrix'] = _get_tiling_l0a_l0b(TILING.get('CL0_matrix'),
                                                   TILING.get('BL0_matrix'),
                                                   'A')

    if TILING.get('BL0_matrix') == []:
        TILING['BL0_matrix'] = _get_tiling_l0a_l0b(TILING.get('CL0_matrix'),
                                                   TILING.get('AL0_matrix'),
                                                   'B')

    manual_pingpong_buffer = TILING.get('manual_pingpong_buffer')
    data_amount_l1a = _get_data_amount_l1('AL1_shape',
                                          manual_pingpong_buffer.get(
                                              'AL1_pbuffer'))
    data_amount_l1b = _get_data_amount_l1('BL1_shape',
                                          manual_pingpong_buffer.get(
                                              'BL1_pbuffer'))

    if (data_amount_l1a + data_amount_l1b) > L1_SPACE_SIZE:
        raise RuntimeError('tiling size exceed L1 Buffer')
    if TILING.get('BL1_shape') and TILING.get('AL1_shape'):
        _check_tilinng_k_l1()

    # check tilling in AL0 BL0
    if TILING.get('AL0_matrix') is None or TILING.get('AL0_matrix') == []:
        raise RuntimeError('tiling[AL0_matrix] can not be None or []')
    _check_tilling_l0('AL0_matrix', L0A_SPACE_SIZE,
                      manual_pingpong_buffer.get('AL0_pbuffer'))

    _check_tiling_bl0_matrix(manual_pingpong_buffer, data_amount_l1b)

    # check tilling in CL0
    _check_tilling_l0c('CL0_matrix', L0C_SPACE_SIZE,
                       manual_pingpong_buffer.get('CL0_pbuffer'))

    # check tilling in CUB  attention:light when stride get  #########
    _check_tilling_cub('CUB_matrix', strideh, stridew,
                       UB_SPACE_SIZE,
                       manual_pingpong_buffer.get('CUB_pbuffer'))


def _get_bias_flag():
    if TENSOR_MAP.get("bias_add_vector") is not None or TENSOR_MAP.get(
            "c_add_bias") is not None:
        bias_flag = 1
    else:
        bias_flag = 0
    return bias_flag


def int_ceil_div(divisor_a, divisor_b):
    """
    round up function

    Paramater:
    :param divisor_a: int.
    :param divisor_b: int.
    :return: int
    """
    if divisor_b == 0:
        raise RuntimeError("division by zero")
    return (divisor_a + divisor_b - 1) // divisor_b


def _get_src_tensor(tensor, index):
    """
    get input tensor according to the specified index
    :param tensor: Tensor for getting input tensor
    :param index: specified index
    :return: specified input tensor
    """
    if tensor is not None and tensor.op.input_tensors:
        return tensor.op.input_tensors[index]
    return None


def _get_target_tensor(tensor, tag):
    """
    Get the specified tensor
    :param tensor: start tensor
    :param tag: tag of specified tensor
    :return: specified tensor
    """
    while tensor is not None and not tensor.op.name.endswith(tag):
        tensor = _get_src_tensor(tensor, 0)
    return tensor


def _check_dx_opti(tensor):
    l0c = _get_target_tensor(tensor, "C")
    l0b = _get_src_tensor(l0c, 1)
    kernel_h, kernel_w = l0b.op.attrs["kernel_hw"]
    if kernel_h.value != 1 or kernel_w.value != 1:
        return False
    return True


def _check_vector(tensor, dequant_para):
    """
    check whether it is vector dequant fusion
    :param tensor: start tensor
    :return: dequant para or None
    """
    relu_flag = tensor.op.attrs["relu_flag"]
    dx_out = _get_src_tensor(tensor, 0)
    deq_scale = _get_src_tensor(tensor, 1)
    if dx_out.op.tag != "conv2d_backprop_input_opti":
        return False

    if _check_dx_opti(dx_out):
        dequant_para["relu_flag"] = relu_flag
        dequant_para["deq_scale"] = deq_scale
        dequant_para["dx_out"] = dx_out
        dequant_para["dequant_type"] = "vector"
        return dequant_para
    return False


def _check_vector_sqrt(tensor, dequant_para):
    """
    check whether it is vector dequant fusion with sqrt mode
    :param tensor: start tensor
    :return: dequant para or None
    """
    dequant_remove_pad = _get_src_tensor(tensor, 0)
    if dequant_remove_pad.op.tag != "dequant_remove_pad":
        return False
    dequant1_vector = _get_src_tensor(dequant_remove_pad, 0)
    if dequant1_vector.op.tag != "dequant1_vector":
        return False
    relu_flag = dequant1_vector.op.attrs["relu_flag"]
    dx_out = _get_src_tensor(dequant1_vector, 0)
    deq_scale = _get_src_tensor(dequant1_vector, 1)
    if dx_out.op.tag != "conv2d_backprop_input_opti":
        return False

    if _check_dx_opti(dx_out):
        dequant_para["relu_flag"] = relu_flag
        dequant_para["deq_scale"] = deq_scale
        dequant_para["dx_out"] = dx_out
        dequant_para["dequant_type"] = "vector_sqrt"
        return dequant_para
    return False


def _check_scalar(tensor, dequant_para):
    """
    check whether it is scalar dequant fusion
    :param tensor: start tensor
    :return: dequant para or None
    """
    dx_out = _get_src_tensor(tensor, 0)
    deq_scale = _get_src_tensor(tensor, 1)
    if dx_out.op.tag != "conv2d_backprop_input_opti":
        return False

    if _check_dx_opti(dx_out):
        dequant_para["deq_scale"] = deq_scale
        dequant_para["dx_out"] = dx_out
        dequant_para["dequant_type"] = "scalar"
        return dequant_para
    return False


def _check_scalar_relu(tensor, dequant_para):
    """
    check whether it is scalar dequant fusion with relu
    :param tensor: start tensor
    :return: dequant para or None
    """
    dequant_remove_pad = _get_src_tensor(tensor, 0)
    if dequant_remove_pad.op.tag != "dequant_remove_pad":
        return False
    dequant_scalar = _get_src_tensor(dequant_remove_pad, 0)
    if dequant_scalar.op.tag != "dequant1_scale":
        return False
    dx_out = _get_src_tensor(dequant_scalar, 0)
    deq_scale = _get_src_tensor(dequant_scalar, 1)
    if dx_out.op.tag != "conv2d_backprop_input_opti":
        return False

    if _check_dx_opti(dx_out):
        dequant_para["deq_scale"] = deq_scale
        dequant_para["dx_out"] = dx_out
        dequant_para["dequant_type"] = "scalar_relu"
        return dequant_para
    return False


def _check_scalar_sqrt(tensor, dequant_para):
    """
    check whether it is scalar dequant fusion with sqrt mode
    :param tensor: start tensor
    :return: dequant para or None
    """
    dequant1_scalar = _get_src_tensor(tensor, 0)
    if dequant1_scalar.op.tag != "dequant1_scale":
        return False
    dx_out = _get_src_tensor(dequant1_scalar, 0)
    deq_scale = _get_src_tensor(dequant1_scalar, 1)
    if dx_out.op.tag != "conv2d_backprop_input_opti":
        return False

    if _check_dx_opti(dx_out):
        dequant_para["deq_scale"] = deq_scale
        dequant_para["dx_out"] = dx_out
        dequant_para["dequant_type"] = "scalar_sqrt"
        return dequant_para
    return False


def _check_scalar_sqrt_relu(tensor, dequant_para):
    """
    check whether it is scalar dequant fusion with sqrt mode and relu
    :param tensor: start tensor
    :return: dequant para or None
    """
    dequant_remove_pad = _get_src_tensor(tensor, 0)
    if dequant_remove_pad.op.tag != "dequant_remove_pad":
        return False
    dequant1_scalar = _get_src_tensor(dequant_remove_pad, 0)
    if dequant1_scalar.op.tag != "dequant1_scale":
        return False
    dx_out = _get_src_tensor(dequant1_scalar, 0)
    deq_scale = _get_src_tensor(dequant1_scalar, 1)
    if dx_out.op.tag != "conv2d_backprop_input_opti":
        return False

    if _check_dx_opti(dx_out):
        dequant_para["deq_scale"] = deq_scale
        dequant_para["dx_out"] = dx_out
        dequant_para["dequant_type"] = "scalar_sqrt_relu"
        return dequant_para
    return False


def _identify_dequant_type(tensor):
    """
    Identify types of dequant fusion
    """

    dequant_para = {}
    if tensor is None:
        return None

    if tensor.op.tag == "dequant2_vector":
        dequant_para = _check_vector_sqrt(tensor, dequant_para)
    elif tensor.op.tag == "dequant_relu":
        dequant_para = _check_scalar_relu(tensor, dequant_para)
    elif tensor.op.tag == "dequant_remove_pad":
        src_tensor = _get_src_tensor(tensor, 0)
        if src_tensor.op.tag == "dequant1_vector":
            dequant_para = _check_vector(src_tensor, dequant_para)
        elif src_tensor.op.tag == "dequant1_scale":
            dequant_para = _check_scalar(src_tensor, dequant_para)
        else:
            return None
    elif tensor.op.tag == "dequant2_scale":
        src_tensor = _get_src_tensor(tensor, 0)
        if src_tensor.op.tag == "dequant_remove_pad":
            dequant_para = _check_scalar_sqrt(src_tensor, dequant_para)
        elif src_tensor.op.tag == "dequant_relu":
            dequant_para = _check_scalar_sqrt_relu(src_tensor, dequant_para)
    if not dequant_para:
        return None
    return dequant_para


def _identify_dequant_quant_type(tensor):
    """
    Identify types of quant fusion
    """
    def _check_dequant(tensor):
        input_ub = _get_src_tensor(tensor, 0)
        if input_ub.op.name != "input_ub":
            return None
        dequant_res = _get_src_tensor(input_ub, 0)
        return dequant_res

    def _identify_quant_type(src_tensor):
        if src_tensor.op.name == "reform_by_vadds":
            dequant_res = _check_dequant(src_tensor)
            quant_para["quant_type"] = "offset"
        elif src_tensor.op.name == "offset_ub":
            src_tensor = _get_src_tensor(src_tensor, 0)
            if src_tensor.op.name == "scale_sqrt_ub":
                reform_by_vmuls = _get_src_tensor(src_tensor, 0)
                if reform_by_vmuls.op.name != "reform_by_vmuls":
                    return None
                dequant_res = _check_dequant(reform_by_vmuls)
                quant_para["quant_type"] = "scale_sqrt_offset"
            elif src_tensor.op.name == "reform_by_vmuls":
                dequant_res = _check_dequant(src_tensor)
                quant_para["quant_type"] = "scale_offset"
            else:
                return None
        elif src_tensor.op.name == "scale_sqrt_ub":
            reform_by_vmuls = _get_src_tensor(src_tensor, 0)
            if reform_by_vmuls.op.name != "reform_by_vmuls":
                return None
            dequant_res = _check_dequant(reform_by_vmuls)
            quant_para["quant_type"] = "scale_sqrt"
        elif src_tensor.op.name == "reform_by_vmuls":
            dequant_res = _check_dequant(src_tensor)
            quant_para["quant_type"] = "scale"
        else:
            raise RuntimeError("unsupported quant fusion type")
        quant_para["dequant_res"] = dequant_res
        return dequant_res

    quant_para = {}
    if tensor.op.tag == "quant":
        quant_para["scale"] = tensor.op.attrs["scale"].value
        quant_para["sqrt_mode"] = tensor.op.attrs["sqrt_mode"].value
        quant_para["offset"] = tensor.op.attrs["offset"].value
        quant_para["round_mode"] = tensor.op.attrs["round_mode"].value

        cast_i8_ub = _get_src_tensor(tensor, 0)
        if cast_i8_ub.op.name != "cast_i8_ub":
            return None
        src_tensor = _get_src_tensor(cast_i8_ub, 0)
        dequant_res = _identify_quant_type(src_tensor)
        dequant_para = _identify_dequant_type(dequant_res)
    else:
        dequant_para = _identify_dequant_type(tensor)

    if dequant_para and dequant_para.get("dequant_type") is None:
        quant_para = {}

    return dequant_para, quant_para


def _dilate_tensor(raw_tensor, attr_list):
    """
    dilate in ub
    :param raw_tensor: tensor for dilate
    :param out_shape_h: int, new h
    :param out_shape_w: int, new w
    :param dilate_h: stride_h
    :param dilate_w: stride_w
    :param img_w: dedy_w
    :return: dilated tensor
    """
    h_dim, w_dim = attr_list["out_hw"]
    dilate_h, dilate_w = attr_list["dilate"]
    img_w = attr_list["img_w"]
    dilate_shape = [raw_tensor.shape[0], raw_tensor.shape[1],
                    h_dim*w_dim, raw_tensor.shape[3]]

    dx_zero = tvm.compute(
        dilate_shape,
        lambda *indice: tvm.convert(0).astype(raw_tensor.dtype),
        name=raw_tensor.name + "_dx_zero",
        tag="init_zero")

    dilate_tensor = tvm.compute(
        dilate_shape,
        lambda n, c1, hw, c0:
        tvm.select(tvm.all((hw // w_dim) % dilate_h == 0,
                           (hw % w_dim) % dilate_w == 0),
                   raw_tensor[n,
                              c1,
                              ((hw // w_dim) // dilate_h)*img_w
                              + (hw % w_dim // dilate_w),
                              c0],
                   dx_zero[n, c1, hw, c0]),
        name=raw_tensor.name + "_dilation",
        tag="conv2d_backprop_input_opti",
        attrs={'dilate': [dilate_h, dilate_w],
               'out_hw': [h_dim, w_dim]})

    TENSOR_MAP["dilate_tensor"] = dilate_tensor
    return dilate_tensor


def _dequant_vector(input_tensor, real_shape, deq_scale, sqrt_mode=0, relu_flag=0):
    """
    dequant with vector parameter
    :param x: tensor, deconv res in l0c
    :param deq_scale: tensor, dequant parameter
    :param sqrt_mode: int, do sqrt when sqrt_mode=1
    :param relu_flag: int, do relu when relu_flag=1
    :return: dequant res
    """
    if int(relu_flag) == 1:
        dequant1_vector = tvm.compute(
            input_tensor.shape,
            lambda i, j, k, l:
            tvm.relu(input_tensor(i, j, k, l)
                     .astype("float16")*deq_scale(0, j, 0, 0, l)),
            name='dequant1', tag="dequant1_vector")
    else:
        dequant1_vector = tvm.compute(
            input_tensor.shape,
            lambda i, j, k, l:
            input_tensor(i, j, k, l).astype("float16") *
            deq_scale(0, j, 0, 0, l),
            name='dequant1', tag="dequant1_vector")

    def _ub_elem_func(*index):
        return dequant1_vector(*index)
    dequant_remove_pad = tvm.compute(
        real_shape,
        _ub_elem_func,
        name='dequant_remove_pad',
        tag="dequant_remove_pad")

    TENSOR_MAP["dequant1_vector"] = dequant1_vector
    TENSOR_MAP["c_ub"] = dequant1_vector
    TENSOR_MAP["dequant_remove_pad"] = dequant_remove_pad

    if int(sqrt_mode) == 1:
        dequant2_vector = tvm.compute(
            dequant_remove_pad.shape,
            lambda i, j, k, l:
            (dequant_remove_pad(i, j, k, l) *
             deq_scale(0, j, 0, 0, l)),
            name='dequant2', tag='dequant2_vector')
        TENSOR_MAP["dequant2_vector"] = dequant2_vector
        return dequant2_vector

    return dequant_remove_pad


def _dequant_scalar(input_tensor, deq_scale, sqrt_mode=0, relu_flag=0):
    """
    dequant with scalar parameter
    :param x: tensor, deconv res in l0c
    :param deq_scale: tensor, dequant parameter
    :param sqrt_mode: int, do sqrt when sqrt_mode=1
    :param relu_flag: int, do relu when relu_flag=1
    :return: dequant res
    """
    dequant1_scalar = tvm.compute(
        input_tensor.shape,
        lambda i, j, k, l:
        (input_tensor(i, j, k, l).astype("float16")*deq_scale(0, 0, 0, 0, 0)),
        name='dequant1', tag="dequant1_scalar")
    TENSOR_MAP["dequant1_scalar"] = dequant1_scalar
    TENSOR_MAP["c_ub"] = dequant1_scalar

    if relu_flag == 1:
        dequant1_scalar = tvm.compute(
            dequant1_scalar.shape,
            lambda *indices: tvm.relu(dequant1_scalar(*indices)),
            name="dequant_relu", tag="dequant_relu")
        TENSOR_MAP["dequant_relu"] = dequant1_scalar

    if sqrt_mode == 1:
        dequant2_scalar = tvm.compute(
            dequant1_scalar.shape,
            lambda i, j, k, l:
            (dequant1_scalar(i, j, k, l)*deq_scale(0, 0, 0, 0, 0)),
            name='dequant2', tag='dequant2_scalar')
        TENSOR_MAP["dequant2_scalar"] = dequant2_scalar
        return dequant2_scalar

    return dequant1_scalar


def _reget_dx_dequant_map(dequant_para, tensor):
    """
    Regenerate the calculation graph
    :param tensor: last tensor in the graph
    :return: last tensor in new graph
    """

    dequant_type = dequant_para.get("dequant_type")
    deq_scale = dequant_para.get("deq_scale")
    c_ub = _get_src_tensor(dequant_para.get("dx_out"), 0)
    dx_l0c = _get_target_tensor(tensor, "C")

    dilate_flag = c_ub.op.name.endswith("_dilation")
    real_shape = util.shape_to_list(dx_l0c.shape)
    real_shape[2] = _get_target_tensor(dx_l0c, "_col").shape[2]

    if dequant_type == "vector":
        dequant_ub = _dequant_vector(dx_l0c, real_shape, deq_scale,
                                     sqrt_mode=0,
                                     relu_flag=dequant_para.get("relu_flag"))
    elif dequant_type == "vector_sqrt":
        dequant_ub = _dequant_vector(dx_l0c, real_shape, deq_scale,
                                     sqrt_mode=1,
                                     relu_flag=dequant_para.get("relu_flag"))
    elif dequant_type == "scalar":
        dequant_ub = _dequant_scalar(dx_l0c, deq_scale,
                                     sqrt_mode=0, relu_flag=0)
    elif dequant_type == "scalar_relu":
        dequant_ub = _dequant_scalar(dx_l0c, deq_scale,
                                     sqrt_mode=0, relu_flag=1)
    elif dequant_type == "scalar_sqrt":
        dequant_ub = _dequant_scalar(dx_l0c, deq_scale,
                                     sqrt_mode=1, relu_flag=0)
    elif dequant_type == "scalar_sqrt_relu":
        dequant_ub = _dequant_scalar(dx_l0c, deq_scale,
                                     sqrt_mode=1, relu_flag=1)
    else:
        raise RuntimeError("unsupported dequant fusion type")

    TENSOR_MAP["deq_scale"] = deq_scale
    TENSOR_MAP["dequant_ub"] = dequant_ub
    TENSOR_MAP["dequant_type"] = dequant_type
    return dequant_ub, dilate_flag, c_ub


def _quant_reform(input_tensor, input_shape, output_shape):
    """
    5 dim input tensor C0 change
    Parameters
    ----------
    input_tensor : input tensor

    input_shape : the shape of input tensor

    output_shape :the shape of output tensor

    scale_val : the val of scale

    Returns
    -------
    res tensor
    """

    def reform_compute_generate(tensor, in_shape, out_shape):
        in_shape = list(in_shape)
        out_shape = list(out_shape)
        n_dim = len(in_shape)

        def lambda_func(*indice):
            new_indice = [indice[0],
                          (indice[1] * out_shape[n_dim - 1] +
                           indice[n_dim - 1])
                          // in_shape[n_dim - 1]] \
                         + list(indice[2:n_dim - 1]) + \
                         [(indice[1] * out_shape[n_dim - 1] +
                           indice[n_dim - 1])
                          % in_shape[n_dim - 1]]
            return tensor(*new_indice)

        return lambda_func

    quant_reform = tvm.compute(output_shape,
                               reform_compute_generate(
                                   input_tensor, input_shape,
                                   output_shape),
                               name='quant_reform')
    TENSOR_MAP["quant_reform"] = quant_reform
    return quant_reform


def _compute_scale(input_tensor, out_shape, scale_val, sqrt_mode):
    """
    5 dim input tensor C0 change
    Parameters
    ----------
    input_tensor : input tensor

    input_shape : the shape of input tensor

    output_shape :the shape of output tensor

    scale_val : the val of scale

    Returns
    -------
    res tensor
    """
    quant_vmuls_ub = tvm.compute(
        out_shape,
        lambda *indices: input_tensor[indices] * scale_val,
        name="quant_vmuls_ub")
    TENSOR_MAP["quant_vmuls_ub"] = quant_vmuls_ub

    if sqrt_mode == 1:
        quant_vmuls_ub = tvm.compute(
            out_shape,
            lambda *indice: quant_vmuls_ub(*indice) * scale_val,
            name="scale_sqrt_ub")
        TENSOR_MAP["scale_sqrt_ub"] = quant_vmuls_ub

    return quant_vmuls_ub


def _compute_offset(input_tensor, out_shape, offset):
    """
    the compute of scale
    """
    offset_value = tvm.const(offset, "float16")

    quant_vadds_ub = tvm.compute(
        out_shape,
        lambda *indice: input_tensor(*indice) + offset_value,
        name="quant_vadds_ub")
    TENSOR_MAP["quant_vadds_ub"] = quant_vadds_ub

    return quant_vadds_ub


def _compute_quant(input_tensor, out_shape, attr_list):
    """
    the compute of quant
    """
    scale, offset, sqrt_mode = attr_list

    quant_res = input_tensor
    if abs(scale - 1.0) >= EPSINON:
        scale_value = tvm.const(scale, "float16")
        quant_res = _compute_scale(
            quant_res, out_shape, scale_value, sqrt_mode)
    if abs(offset - 0.0) >= EPSINON:
        quant_res = _compute_offset(
            quant_res, out_shape, offset)

    cast_i8_ub = tvm.compute(
        out_shape,
        lambda *indice: topi.cast(quant_res(*indice), "int8"),
        name='cast_i8_ub')
    TENSOR_MAP["cast_i8_ub"] = cast_i8_ub

    return cast_i8_ub


def reget_dx_dequant_quant_map(tensor):
    """
    Regenerate the calculation graph
    :param tensor: last tensor in the graph
    :return: last tensor in new graph
    """
    def _cub2ddr(c_ub, output_shape):
        """compute tensor c_ddr"""
        def _ub_elem_func(*index):
            return c_ub(*index)
        c_ddr = tvm.compute(
            output_shape,
            _ub_elem_func,
            tag="conv2d_backprop_input_opti",
            name=c_ub.name + "_img")
        return c_ddr

    dequant_para, quant_para = _identify_dequant_quant_type(tensor)

    # not dequant or quant fusion
    if dequant_para is None or dequant_para.get("dequant_type") is None:
        return tensor
    # dx + dequant
    if dequant_para.get("dequant_type") is not None \
            and quant_para.get("quant_type") is None:
        dequant_ub, dilate_flag, c_ub = _reget_dx_dequant_map(dequant_para, tensor)

        if dilate_flag is True:
            dequant_ub = _dilate_tensor(dequant_ub, c_ub.op.attrs)
            TENSOR_MAP["dilate_ub"] = dequant_ub

        output_shape = list(dequant_ub.shape)
        output_shape[2] = dequant_para.get("dx_out").op.attrs["hw_dim"]
        res = _cub2ddr(dequant_ub, output_shape)
    # dx + dequant + quant
    else:
        TENSOR_MAP["quant_type"] = quant_para.get("quant_type")
        dequant_res = quant_para.get("dequant_res")
        dequant_ub, dilate_flag, c_ub = _reget_dx_dequant_map(dequant_para, dequant_res)

        dequant_shape = util.shape_to_list(dequant_ub.shape)
        dequant_shape[2] = dequant_para.get("dx_out").op.attrs["hw_dim"]
        # make cout 32 aligned
        if dequant_shape[1] % 2 != 0:
            read_shape = dequant_shape[:]
            read_shape[1] = read_shape[1] + 1
            dequant_ub = tvm.compute(
                read_shape,
                lambda *indice: tvm.select(indice[1] <= dequant_shape[1] - 1,
                                           dequant_ub(*indice),
                                           tvm.const(0, dtype="float16")),
                name='input_ub',
                attrs={"c_out": dequant_shape[1]})
            TENSOR_MAP["input_ub"] = dequant_ub

        quant_shape = [dequant_shape[0], (dequant_shape[1] + 1) // 2,
                       dequant_shape[2], dequant_shape[3]*2]
        quant_reform = _quant_reform(dequant_ub, dequant_shape, quant_shape)

        if dilate_flag is True:
            quant_reform = _dilate_tensor(quant_reform, c_ub.op.attrs)
            TENSOR_MAP["dilate_ub"] = quant_reform

        attr_list = (quant_para["scale"],
                     quant_para["offset"],
                     quant_para["sqrt_mode"])

        quant_shape[2] = dequant_para.get("dx_out").op.attrs["hw_dim"]
        quant_ub = _compute_quant(quant_reform, quant_shape, attr_list)
        res = _cub2ddr(quant_ub, quant_shape)

        TENSOR_MAP["round_mode"] = quant_para.get("round_mode")
    return res


def reget_dx_tensor_list(tensor_list, real_out_tensors):
    """
    get new tensor_list
    """
    dequant_para, _ = _identify_dequant_quant_type(tensor_list[-1])
    # not dequant or quant fusion
    if dequant_para is None or dequant_para.get("dequant_type") is None:
        pass
    else:
        tensor_list = tensor_list[:-1]
        tensor_list.append(real_out_tensors[0])

    return tensor_list


def _set_data_layout(res, sch):  # pylint: disable=R0914,R0915
    """
    get DIM_MAP which contains all ops

    Parameter:
    ----------------------------------------------------------
    :param res: op
    :param sch: schedule
    :return: None
    ----------------------------------------------------------
    """
    def _ub_set_scope(ub_tensor_map):
        """
        set tensor from tensor_map in ub buffer
        """
        deq_read_cache = []
        for tensor_name in ub_tensor_map:
            if TENSOR_MAP.get(tensor_name) is not None:
                tensor = TENSOR_MAP[tensor_name]
                sch[tensor].set_scope(cce.scope_ubuf)
                if tensor_name in("dequant_relu", "dequant_remove_pad"):
                    continue
                deq_read_cache.append(tensor)
        return deq_read_cache

    def _get_tensor_dx_gm(tensor_add_res):
        '''
        get dx_gm tensor by add_res tensor
        :param sch:
        :param tensor_add_res: add_res tensor
        :return: dx_gm tensor
        '''
        global TENSOR_MAP  # pylint: disable=W0603
        tensor_add_left = tensor_add_res.op.input_tensors[0]
        tensor_add_right = tensor_add_res.op.input_tensors[1]
        if tensor_add_left.op.tag == "conv2d_backprop_input_opti":
            tensor_dx_gm = tensor_add_left
            tensor_add_input_gm = tensor_add_right
        else:
            tensor_dx_gm = tensor_add_right
            tensor_add_input_gm = tensor_add_left

        tensor_add_input_ub = sch.cache_read(tensor_add_input_gm,
                                             cce.scope_ubuf,
                                             [tensor_add_res])
        TENSOR_MAP["add_input_ub"] = tensor_add_input_ub
        return tensor_dx_gm

    def _check_dx_fusion_type(res, fusion_tensor_map):
        """
        check fusion type and set buffer
        """

        if res.op.tag == "emit_insn_elewise_multiple_sel|bool":
            drelu_gm = res
            # dx+add+drelu
            if "elewise_binary_add" in drelu_gm.op.input_tensors[1].op.tag:
                DeconvParam.update_para_map("FUSION_TYPE", FUSION_DX_ADD_DRELU)
                tensor_add_res = drelu_gm.op.input_tensors[1]
                sch[tensor_add_res].set_scope(cce.scope_ubuf)
                TENSOR_MAP["add_res_ub"] = tensor_add_res
                tensor_dx_gm = _get_tensor_dx_gm(tensor_add_res)
            # dx+drelu
            else:
                DeconvParam.update_para_map("FUSION_TYPE", FUSION_DX_DRELU)
                tensor_dx_gm = drelu_gm.op.input_tensors[1]

            tensor_bitmask_gm = drelu_gm.op.input_tensors[0]
            sch[tensor_dx_gm].set_scope(cce.scope_ubuf)
            tensor_bitmask = sch.cache_read(
                tensor_bitmask_gm, cce.scope_ubuf, [drelu_gm])
            tensor_drelu = sch.cache_write(drelu_gm, cce.scope_ubuf)

            fusion_tensor_map["bitmask_ub"] = tensor_bitmask
            fusion_tensor_map["drelu_ub"] = tensor_drelu
            fusion_tensor_map["fusion_dx_gm"] = tensor_dx_gm  # inter_gm
        # dx+add
        elif res.op.tag == "elewise_binary_add_disable":
            add_res_gm = res
            DeconvParam.update_para_map("FUSION_TYPE", FUSION_DX_ADD)
            tensor_dx_gm = _get_tensor_dx_gm(add_res_gm)
            sch[tensor_dx_gm].set_scope(cce.scope_ubuf)
            add_res_ub = sch.cache_write(add_res_gm, cce.scope_ubuf)
            fusion_tensor_map["add_res_ub"] = add_res_ub
            fusion_tensor_map["fusion_dx_gm"] = tensor_dx_gm  # inter_gm
        # dx+dequant+(quant)
        elif fusion_tensor_map.get("dequant_type") is not None:
            DeconvParam.update_para_map("FUSION_TYPE", FUSION_DX_DEQUANT)
            deq_scale = fusion_tensor_map["deq_scale"]
            ub_tensor_map = ("dequant1_vector", "dequant2_vector",
                             "dequant1_scalar", "dequant2_scalar",
                             "dequant_relu", "dequant_remove_pad")
            deq_read_cache = _ub_set_scope(ub_tensor_map)
            deq_scale_ub = sch.cache_read(
                deq_scale, cce.scope_ubuf, deq_read_cache)
            fusion_tensor_map["deq_scale_ub"] = deq_scale_ub
            tensor_dx_gm = res

            if fusion_tensor_map.get("quant_type") is not None:
                DeconvParam.update_para_map("FUSION_TYPE",
                                            FUSION_DX_DEQUANT_QUANT)
                ub_tensor_map = ("cast_i8_ub", "quant_reform",
                                 "quant_vmuls_ub", "quant_vadds_ub",
                                 "scale_sqrt_ub", "input_ub")
                _ub_set_scope(ub_tensor_map)
        # dx
        elif res.op.tag == "conv2d_backprop_input_opti":
            DeconvParam.update_para_map("FUSION_TYPE", FUSION_NONE)
            tensor_dx_gm = res
        else:
            raise RuntimeError(DX_SUPPORT_TAG_LOG_PREFIX
                               + ' unsupported data flow')
        return tensor_dx_gm, fusion_tensor_map

    def _get_ub_tensor(fusion_type):
        if fusion_type == FUSION_DX_DEQUANT_QUANT:
            tensor_dilate_ub = TENSOR_MAP.get("dilate_ub")
        else:
            if tensor_dx_gm.op.input_tensors[0].op.name == "bias_add_vector":
                bias_add_vector = tensor_dx_gm.op.input_tensors[0]
                tensor_dilate_ub = bias_add_vector.op.input_tensors[0]
                tensor_bias = bias_add_vector.op.input_tensors[1]
                sch[bias_add_vector].set_scope(cce.scope_ubuf)
                bias_ub = sch.cache_read(
                    tensor_bias, cce.scope_ubuf, [bias_add_vector]
                )
                TENSOR_MAP["bias_add_vector"] = bias_add_vector
                TENSOR_MAP["bias_ub"] = bias_ub
            else:
                tensor_dilate_ub = tensor_dx_gm.op.input_tensors[0]

        if tensor_dilate_ub is not None \
                and tensor_dilate_ub.op.tag == "conv2d_backprop_input_opti":
            TENSOR_MAP["dilate_ub"] = tensor_dilate_ub
            sch[tensor_dilate_ub].set_scope(cce.scope_ubuf)
            tensor_cub = tensor_dilate_ub.op.input_tensors[0]
            tensor_fillling_zero = tensor_dilate_ub.op.input_tensors[1]
            TENSOR_MAP["tensor_fillling_zero"] = tensor_fillling_zero
            sch[tensor_fillling_zero].set_scope(cce.scope_ubuf)
        else:
            if tensor_dx_gm.op.input_tensors[0].op.name == "bias_add_vector":
                tensor_cub = tensor_dilate_ub
            else:
                tensor_cub = tensor_dx_gm.op.input_tensors[0]

        if fusion_type in (FUSION_DX_DEQUANT, FUSION_DX_DEQUANT_QUANT):
            tensor_cub = TENSOR_MAP["c_ub"]

        return tensor_cub

    global TENSOR_MAP  # pylint: disable=W0603
    global DIM_MAP  # pylint: disable=W0603

    print_debug("dx fusion tag:", res.op.tag)
    tensor_dx_gm, TENSOR_MAP = _check_dx_fusion_type(res, TENSOR_MAP)
    fusion_type = DeconvParam.get_para_map("FUSION_TYPE")

    # get tensor of l0c, l1,l0a,l0b
    tensor_cub = _get_ub_tensor(fusion_type)
    if tensor_cub.op.input_tensors[0].name == "c_add_bias":
        c_add_bias = tensor_cub.op.input_tensors[0]
        bias_l0c = c_add_bias.op.input_tensors[0]
        tensor_mmad = c_add_bias.op.input_tensors[1]
        bias_ub_brc = bias_l0c.op.input_tensors[0]
        tensor_bias = bias_ub_brc.op.input_tensors[0]
        bias_ub = sch.cache_read(tensor_bias, cce.scope_ubuf, [bias_ub_brc])
        TENSOR_MAP["c_add_bias"] = c_add_bias
        TENSOR_MAP["bias_l0c"] = bias_l0c
        TENSOR_MAP["bias_ub_brc"] = bias_ub_brc
        TENSOR_MAP["bias_ub"] = bias_ub
        TENSOR_MAP["tensor_bias"] = tensor_bias
    else:
        tensor_mmad = tensor_cub.op.input_tensors[0]
    a_l0a = tensor_mmad.op.input_tensors[0]
    dedy_col = a_l0a.op.input_tensors[0]
    dedy = dedy_col.op.input_tensors[0]
    weight_l0 = tensor_mmad.op.input_tensors[1]
    weight_l1 = weight_l0.op.input_tensors[0]

    # set scope
    sch[tensor_cub].set_scope(cce.scope_ubuf)
    TENSOR_MAP["c_ub"] = tensor_cub
    sch[dedy_col].set_scope(cce.scope_cbuf)
    storage_align_size = 256
    if dedy_col.dtype == "int8":
        storage_align_size = 512
    sch[dedy_col].storage_align(
        sch[dedy_col].op.axis[1], storage_align_size, 0)
    TENSOR_MAP["a_l1"] = dedy_col
    sch[a_l0a].set_scope(cce.scope_ca)
    l0a_m0, l0a_k0, _ = CUBE_MKN[a_l0a.dtype]['mac']
    sch[a_l0a].buffer_align(
        (1, 1), (1, 1), (1, 1), (1, l0a_m0), (1, l0a_k0))
    TENSOR_MAP["a_l0a"] = a_l0a
    if TENSOR_MAP.get("c_add_bias") is not None:
        sch[c_add_bias].set_scope(cce.scope_cc)
        sch[bias_l0c].set_scope(cce.scope_cc)
        sch[bias_ub_brc].set_scope(cce.scope_ubuf)
    sch[weight_l1].set_scope(cce.scope_cbuf)
    TENSOR_MAP["b_l1"] = weight_l1
    sch[weight_l0].set_scope(cce.scope_cb)
    TENSOR_MAP["b_l0b"] = weight_l0

    sch[tensor_mmad].set_scope(cce.scope_cc)
    TENSOR_MAP["c_l0c"] = tensor_mmad
    TENSOR_MAP["c_gm"] = res
    TENSOR_MAP["img_placehold"] = dedy
    TENSOR_MAP["filter_placehold"] = weight_l1.op.input_tensors[0]

    # fill in dimmap
    DIM_MAP['out_img_shape'] = [int(i) for i in res.shape]
    DIM_MAP['img_shape'] = [int(i) for i in TENSOR_MAP["img_placehold"].shape]
    DIM_MAP['A_matrix_dim'] = [int(i) for i in dedy_col.shape]
    DIM_MAP['B_matrix_dim'] = [int(i) for i in weight_l0.shape]
    DIM_MAP['filter_shape'] = [int(i) for i in
                               weight_l1.op.input_tensors[0].shape]
    if TENSOR_MAP.get("dilate_ub") is not None:
        DIM_MAP['dilate_dim'] \
            = [int(i) for i in TENSOR_MAP["dilate_ub"].op.attrs["dilate"]]
        DIM_MAP['out_hwdim'] \
            = [int(i) for i in TENSOR_MAP["dilate_ub"].op.attrs["out_hw"]]


def _get_aicore_tiling_factor():
    """
    using tilling parameter calculate factor

    :return: tilling factor from ub to ddr
         tilling factor from l0c to ub
         tilling factor from ddr to AL1
         tilling factor from ddr to Bl1
    """

    def _get_undilate_loc_m(l0c_tiling_factor):

        if l0c_tiling_factor[1] < DIM_MAP.get("img_shape")[-2]:
            raise RuntimeError('mc of CL0_matrix small than weight of Image')
        if DIM_MAP['img_shape'][3] > block_m:
            check_ifmc_falg = bool((mc_from_tiling // DIM_MAP['img_shape'][3])
                                   * DIM_MAP['img_shape'][3]
                                   * DIM_MAP['dilate_dim'][0]
                                   * DIM_MAP['dilate_dim'][1]
                                   <= CUB_BUFFER_LIMIT)
            if mc_from_tiling % DIM_MAP['img_shape'][3] == 0 \
                    and check_ifmc_falg \
                    and DIM_MAP['img_shape'][2] \
                    % (mc_from_tiling // DIM_MAP['img_shape'][3]) == 0:
                n_is_hfactor = ((mc_from_tiling) // DIM_MAP['img_shape'][3])
            else:
                n_is_hfactor = ((mc_from_tiling - block_m) //
                                DIM_MAP['img_shape'][3])
        else:
            check_ifmc_falg_s = False
            if mc_from_tiling % DIM_MAP['img_shape'][3] == 0:
                n_is_hfactor = mc_from_tiling // DIM_MAP['img_shape'][3]
                while DIM_MAP['img_shape'][2] % n_is_hfactor != 0:
                    n_is_hfactor = n_is_hfactor - 1
                check_ifmc_falg_s = bool(
                    n_is_hfactor * DIM_MAP['img_shape'][3] *
                    DIM_MAP['dilate_dim'][0] * DIM_MAP['dilate_dim'][1]
                    > CUB_BUFFER_LIMIT)
            if mc_from_tiling % DIM_MAP['img_shape'][3] != 0 \
                    or check_ifmc_falg_s:
                n_is_hfactor = ((mc_from_tiling - block_m) //
                                DIM_MAP['img_shape'][3])
                while DIM_MAP['img_shape'][2] % n_is_hfactor != 0:
                    n_is_hfactor = n_is_hfactor - 1

        l0c_tiling_factor[1] = DIM_MAP.get("out_hwdim")[1] * n_is_hfactor * \
                               DIM_MAP['dilate_dim'][0]
        if l0c_tiling_factor[1] == 0:
            raise RuntimeError('nw can not be zero')
        undilate_l0c_m = n_is_hfactor * DIM_MAP['img_shape'][3]
        return undilate_l0c_m

    block_m = CUBE_MKN[TENSOR_MAP.get("c_l0c").dtype]["mac"][0]
    block_k = CUBE_MKN[TENSOR_MAP.get("b_l1").dtype]["mac"][1]
    # get factor from l0c, ub to ddr
    l0c_tiling_factor = [TILING["CL0_matrix"][0], # sizeOfpart for N axis
                         TILING["CL0_matrix"][1]
                         * TILING["CL0_matrix"][2]] # M
    mc_from_tiling = TILING["CL0_matrix"][1] * TILING["CL0_matrix"][2]

    undilate_l0c_m = (mc_from_tiling // DIM_MAP['img_shape'][3]) * \
                     DIM_MAP['img_shape'][3]

    need_buffer_tile = False
    if DIM_MAP.get('dilate_dim') is not None:
        undilate_l0c_m = _get_undilate_loc_m(l0c_tiling_factor)
        if undilate_l0c_m % block_m != 0:
            need_buffer_tile = True
    if TENSOR_MAP.get("drelu_ub") is not None:
        if l0c_tiling_factor[1] % block_m != 0 or \
                (DIM_MAP['out_img_shape'][2] - DIM_MAP['out_img_shape'][2] //
                 l0c_tiling_factor[1]
                 * l0c_tiling_factor[1]) % block_m != 0:
            TILING["CUB_matrix"][0] = 1

    # From LOC to GM [NumOfparts for N axis, NumOfparts for M axis ]
    l0c_parts = [
        int_ceil_div(DIM_MAP.get("out_img_shape")[1] // TILING['block_dim'][1],
                     l0c_tiling_factor[0]),
        int_ceil_div(DIM_MAP.get("out_img_shape")[2] // TILING['block_dim'][2],
                     l0c_tiling_factor[1])]

    # l0c_ub_tiling_factor is num of parts from l0c to ub
    l0c_ub_tiling_factor = TILING["CUB_matrix"]  # [n_part, m_part(not used)]
    # C_UB_factor is size of each part from l0c to ub, second item is 1
    # From L0C to UB,[NumOfparts for N axis, NumOfparts for M axis]
    l0c_ub_parts = [int_ceil_div(l0c_tiling_factor[0],
                                 l0c_ub_tiling_factor[0]),
                    int_ceil_div(
                        l0c_tiling_factor[1],
                        l0c_ub_tiling_factor[1] * l0c_ub_tiling_factor[2])]
    print_debug('l0c_ub_tiling_factor:', l0c_ub_tiling_factor,
                'l0c_ub_parts:', l0c_ub_parts)

    if TILING["AL1_shape"]:  # AL1_shape = [C1,H*W,16,16],batch=1
        # parts of k-axis from DDR to L1---need div by H*W
        al1_parts = [int_ceil_div(DIM_MAP["A_matrix_dim"][1],
                                  int_ceil_div(TILING["AL1_shape"][0],
                                               block_k)),
                     int_ceil_div(l0c_parts[1], TILING["AL1_shape"][1])]
    else:
        al1_parts = [1, 1]

    if TILING["BL1_shape"]:
        if (l0c_parts[0] % TILING["BL1_shape"][1]) != 0:
            raise RuntimeError(
                "second value of BL1_shape should be factor of n block num")
        bl1_parts = [int_ceil_div(DIM_MAP["B_matrix_dim"][0],
                                  int_ceil_div(TILING["BL1_shape"][0],
                                               block_k)),
                     int_ceil_div(l0c_parts[0], TILING["BL1_shape"][1])]
    else:
        bl1_parts = [1, 1]

    return l0c_tiling_factor, l0c_ub_parts, al1_parts, bl1_parts, \
           undilate_l0c_m, need_buffer_tile


def _get_mmad_factor():
    """
    get tilling factor in mmad

    :return:tilling factor for al0
            tilling factor for bl0
            tilling factor for reduce axis
    """
    al0_factor = [TILING.get("AL0_matrix")[0], TILING.get("AL0_matrix")[1]]
    bl0_factor = [TILING.get("BL0_matrix")[0], TILING.get("BL0_matrix")[1]]
    reduce_factor = TILING.get("BL0_matrix")[0]
    return al0_factor, bl0_factor, reduce_factor


def _bind_multi_core(  # pylint: disable=R0913,R0914
        sch, c_gm, l1_n_outer_outer, l1_n_out_inner,
        l1_m_outer_outer, l1_m_outer_inner):
    if "block_dim" in TILING:
        block_dim = TILING["block_dim"]
    else:
        block_dim = [1, 1, 1, 1]
    blockidx = []
    # split batch axis
    batch_out, batch_in = sch[c_gm].split(c_gm.op.axis[0], nparts=block_dim[0])
    l1_n_out_inner_out, l1_n_out_inner_in = \
        sch[c_gm].split(l1_n_out_inner, nparts=block_dim[1])
    l1_m_outer_inner_out, l1_m_outer_inner_in = \
        sch[c_gm].split(l1_m_outer_inner, nparts=block_dim[2])

    # reorder
    sch[c_gm].reorder(batch_out, l1_n_out_inner_out, l1_m_outer_inner_out,
                      batch_in,
                      l1_n_outer_outer, l1_n_out_inner_in, l1_m_outer_outer,
                      l1_m_outer_inner_in)

    blocks = block_dim[0] * block_dim[1] * block_dim[2]
    if blocks != 1:
        out_fused = sch[c_gm].fuse(batch_out, l1_n_out_inner_out,
                                   l1_m_outer_inner_out)
        out_fused_out, _ = sch[c_gm].split(out_fused, nparts=blocks)
        bind_out, bind_in = sch[c_gm].split(out_fused_out, 1)
        blockidx = tvm.thread_axis("blockIdx.x")
        sch[c_gm].bind(bind_out, blockidx)
        if blocks == block_dim[0]:
            sch[c_gm].pragma(bind_in, 'json_info_batchBindOnly')
    else:
        blockidx = [batch_out, l1_n_out_inner_out, l1_m_outer_inner_out]
    return batch_in, l1_m_outer_inner_in, l1_n_out_inner_in, blockidx


def _get_l0c_and_l1_axis(  # pylint: disable=R0914
        sch, c_gm, l0c_factor, al1_parts, bl1_parts):
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

    def _get_reorder_flag(al1_parts, bl1_parts):
        reorder_flag = False
        if TILING["AL1_shape"] and al1_parts[0] != 1 \
                and TILING["BL1_shape"] and bl1_parts[0] != 1:
            if bl1_parts[1] >= al1_parts[1]:
                reorder_flag = True
        if TILING["AL1_shape"] and al1_parts[0] == 1 \
                and TILING["BL1_shape"] and bl1_parts[0] == 1:
            if bl1_parts[1] >= al1_parts[1]:
                reorder_flag = True
        if TILING["BL1_shape"] and bl1_parts[0] != 1 \
                and TILING["AL1_shape"] and al1_parts[0] == 1:
            reorder_flag = True
        print_debug('reorder_flag:', reorder_flag)
        return reorder_flag

    # split c_gm according to factor of loc and out_shape
    l0c_n_outer, l0c_n_inner = sch[c_gm].split(c_gm.op.axis[1], l0c_factor[0])
    l0c_m_outer, l0c_m_inner = sch[c_gm].split(c_gm.op.axis[2], l0c_factor[1])
    sch[c_gm].reorder(l0c_n_outer, l0c_m_outer, l0c_n_inner, l0c_m_inner)

    # split c_gm according to factor of a_l1 and b_l1
    l1_m_outer_outer, l1_m_outer_inner = sch[c_gm].split(l0c_m_outer,
                                                         nparts=al1_parts[1])
    l1_n_outer_outer, l1_n_out_inner = sch[c_gm].split(l0c_n_outer,
                                                       nparts=bl1_parts[1])

    batch_in, l1_m_outer_inner_in, l1_n_out_inner_in, blockidx = \
        _bind_multi_core(sch, c_gm, l1_n_outer_outer, l1_n_out_inner,
                         l1_m_outer_outer, l1_m_outer_inner)
    # reorder al1 and bl1 axis according to double buffer
    batch_in_out_axis, batch_in_inner_axis \
        = sch[c_gm].split(batch_in, factor=1)

    reorder_flag = _get_reorder_flag(al1_parts, bl1_parts)
    print_ir_conv("before reorder", sch)
    if reorder_flag:
        sch[c_gm].reorder(l1_m_outer_outer, batch_in_inner_axis,
                          l1_n_outer_outer)
    else:
        sch[c_gm].reorder(l1_n_outer_outer, l1_m_outer_outer,
                          batch_in_inner_axis)
    print_ir_conv("after reorder", sch)
    return batch_in_out_axis, l1_n_outer_outer, batch_in_inner_axis, \
           l1_m_outer_inner_in, l0c_n_inner, l0c_m_inner, l1_m_outer_outer, \
           l1_n_out_inner_in, blockidx


def _get_l0a_and_l0b_axis(  # pylint: disable=R0913,R0914
        sch, c_l0c, new_c_col_axis, al0_axis_factor,
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
    al0_m_out, al0_m_inner = sch[c_l0c].split(
        new_c_col_axis[2],
        al0_axis_factor[0] * CUBE_MKN[c_l0c.dtype]["mac"][0])
    bl0_n_outer, bl0_n_inner = sch[c_l0c].split(
        new_c_col_axis[1], bl0_axis_factor[1])
    # for reduce axis, al0 and b_l0b should be the same
    k_outer_outer, k_outer_inner = sch[c_l0c].split(reduce_out,
                                                    reduce_axis_factor)
    _, batch_l0c_inner = sch[c_l0c].split(c_l0c.op.axis[0], 1)

    sch[c_l0c].reorder(k_outer_outer, bl0_n_outer,
                       al0_m_out, batch_l0c_inner,
                       bl0_n_inner, al0_m_inner, new_c_col_axis[3],
                       k_outer_inner, reduce_inner)

    return al0_m_out, bl0_n_outer, k_outer_outer, batch_l0c_inner

def _get_al1_and_bl1_axis(sch, c_l0c, al1_parts, bl1_parts, k_outer_outer):
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
    reduce_axis_serial = [k_outer_outer_outer_outer,
                          k_outer_outer_outer_inner,
                          k_outer_outer_inner]
    return al1_at_l0c_axis, bl1_at_l0c_axis, reduce_axis_serial


def _dilate_schedule(sch, dilate_ub, out_w, dilate_w, dilate_h):
    '''
    :param sch:
    :param dilate_ub:
    :param out_h:
    :param out_w:
    :param dilate_h:
    :param dilate_w:
    :return:
    '''
    h_axis, w_axis = sch[dilate_ub].split(dilate_ub.op.axis[2], out_w)
    wo_axis, wi_axis = sch[dilate_ub].split(w_axis, dilate_w)
    ho_axis, hi_axis = sch[dilate_ub].split(h_axis, dilate_h)
    sch[dilate_ub].unroll(hi_axis)
    sch[dilate_ub].reorder(wi_axis,
                           dilate_ub.op.axis[0],
                           dilate_ub.op.axis[1],
                           ho_axis, wo_axis)
    sch[dilate_ub].unroll(wi_axis)
    return wo_axis


def opti_schedule(tensor, sch_list):  # pylint: disable=R0915,R0914
    """
    the schedule of conv2d_backprop_opti_input

    Parameter:
    -------------------------------------------------------------------
    :param tensor: a tensor
    :param sch_list: a schedule list
    :return: schedule
    -------------------------------------------------------------------
    """

    def _do_buffer_tile(fusion_type):
        def _get_cub_buffertile_m_min():
            # cub buffertile hw axis
            block_m = CUBE_MKN[TENSOR_MAP.get("c_ub").dtype]["mac"][0]
            mm_coefficient_factor = undilate_l0c_m
            moo_coefficient_unzero = int_ceil_div(
                int_ceil_div(DIM_MAP['out_img_shape'][2], l0c_factor[1]),
                al1_parts[1])
            moo_coefficient = 0 if al1_parts[1] == 1 \
                else moo_coefficient_unzero
            moio_coefficient = 0 if TILING['block_dim'][2] == 1 \
                else int_ceil_div(moo_coefficient_unzero, TILING['block_dim'][2])
            moii_coefficient = 0 if int_ceil_div(moo_coefficient_unzero,
                                                 TILING['block_dim'][2]) == 1 else 1
            cub_buffertile_m_min = (moo_coefficient * tile_axis.var +
                                    moio_coefficient * moio_axis +
                                    moii_coefficient * c_slice_axis) * \
                                   mm_coefficient_factor // block_m \
                                   * block_m
            return cub_buffertile_m_min

        def _get_cub_buffertile_n_min():
            # cub buffertile cout axis
            no_coefficient = l0c_factor[0]
            noo_coefficient_unzero = int_ceil_div(
                int_ceil_div(DIM_MAP['out_img_shape'][1],
                             l0c_factor[0]), bl1_parts[1])
            noo_coefficient = 0 if bl1_parts[1] == 1 \
                else noo_coefficient_unzero
            noio_coefficient = 0 if TILING['block_dim'][1] == 1 \
                else int_ceil_div(noo_coefficient_unzero, TILING['block_dim'][1])
            noii_coefficient = 0 if int_ceil_div(noo_coefficient_unzero,
                                                 TILING['block_dim'][1]) == 1 else 1
            nio_coefficient = 0 if l0c_ub_parts[0] == 1 \
                else int_ceil_div(l0c_factor[0], l0c_ub_parts[0])
            cub_buffertile_n_min = (bl1_at_c_axis.var * noo_coefficient +
                                    noio_coefficient * noio_axis +
                                    noii_coefficient * noii_axis.var) \
                                    * no_coefficient \
                                    + nio_coefficient * l0c_n_inner_outer.var
            return cub_buffertile_n_min

        l0c_factor_tile = TILING['CL0_matrix'][1] * TILING['CL0_matrix'][2]

        # multi core and one core
        if isinstance(blockidx, list):
            batcho_axis = blockidx[0]
            noio_axis = blockidx[1]
            moio_axis = blockidx[2]
        else:
            batcho_axis = blockidx.var // (TILING['block_dim'][1] * TILING['block_dim'][2])
            noio_axis = blockidx.var // TILING['block_dim'][2] % TILING['block_dim'][1]
            moio_axis = blockidx.var % TILING['block_dim'][2]
        # cub buffertile batch axis
        batch_factor = int_ceil_div(DIM_MAP['img_shape'][0],
                                    TILING["block_dim"][0])
        batcho_coefficient = 0 if TILING['block_dim'][0] == 1 else batch_factor
        batchio_coefficient = 0 if batch_factor == 1 else 1
        batch_dim = [batcho_axis * batcho_coefficient +
                     batch_in_out_axis.var * batchio_coefficient, 1]
        cub_buffertile_m_min = _get_cub_buffertile_m_min()
        cub_buffertile_m_extend = l0c_factor_tile

        cub_buffertile_n_min = _get_cub_buffertile_n_min()
        cub_buffertile_n_extend = TILING['CUB_matrix'][0]

        if fusion_type in [FUSION_DX_DEQUANT_QUANT, FUSION_DX_DEQUANT]:
            cub_buffertile_n_min, cub_buffertile_n_extend = None, None
        sch[c_ub].buffer_tile((batch_dim[0], batch_dim[1]),
                              (cub_buffertile_n_min, cub_buffertile_n_extend),
                              (cub_buffertile_m_min, cub_buffertile_m_extend),
                              (0, 16))

    def _attach_ub(fusion_type):
        if fusion_type in [FUSION_DX_ADD_DRELU, FUSION_DX_DRELU]:
            if fusion_type == FUSION_DX_ADD_DRELU:
                sch[add_res_ub].compute_at(sch[c_gm], l0c_m_inner_outer)
                sch[add_input_ub].compute_at(sch[c_gm], add_input_at)
            sch[drelu_ub].compute_at(sch[c_gm], l0c_m_inner_outer)
            sch[bitmask_ub].compute_at(sch[c_gm], l0c_m_inner_outer)
            sch[fusion_dx_gm].compute_at(sch[c_gm], l0c_m_inner_outer)
        elif fusion_type == FUSION_DX_ADD:
            sch[add_res_ub].compute_at(sch[c_gm], l0c_m_inner_outer)
            sch[add_input_ub].compute_at(sch[c_gm], add_input_at)
            sch[fusion_dx_gm].compute_at(sch[c_gm], l0c_m_inner_outer)
        elif fusion_type in (FUSION_DX_DEQUANT, FUSION_DX_DEQUANT_QUANT):
            for tensor_name in TENSOR_MAP:
                if tensor_name in ("dequant1_vector", "dequant2_vector",
                                   "dequant1_scalar", "dequant2_scalar",
                                   "dequant_relu", "deq_scale_ub",
                                   "cast_i8_ub", "quant_vadds_ub",
                                   "quant_vmuls_ub", "scale_sqrt_ub",
                                   "input_ub", "quant_reform",
                                   "dequant_remove_pad"):
                    sch[TENSOR_MAP[tensor_name]].compute_at(
                        sch[c_gm], l0c_m_inner_outer)

            if fusion_type == FUSION_DX_DEQUANT_QUANT:
                sch[TENSOR_MAP["dequant_ub"]].compute_at(
                    sch[c_gm], l0c_m_inner_outer)

        if bias_add_vector_ub is not None:
            sch[bias_ub].compute_at(sch[c_gm], batch_in_out_axis)
            sch[bias_add_vector_ub].compute_at(sch[c_gm], l0c_m_inner_outer)

        if bias_ub_brc is not None:
            sch[bias_ub].compute_at(sch[c_gm], batch_in_out_axis)

        if dilate_ub is not None:
            filling_zero_ub = TENSOR_MAP["tensor_fillling_zero"]
            sch[dilate_ub].compute_at(sch[c_gm], l0c_m_inner_outer)
            sch[filling_zero_ub].compute_at(sch[c_gm], l0c_m_inner_outer)

        sch[c_ub].compute_at(sch[c_gm], l0c_m_inner_outer)

    def _attach_al1_bl1():
        # attach tensor of al1 and bl1 to c_l0c
        if TILING["AL1_shape"]:
            print_debug('al1_parts[0]:', al1_parts[0])
            if al1_parts[0] != 1:
                sch[a_l1].compute_at(sch[c_l0c], al1_at_l0c_axis)
            else:
                sch[a_l1].compute_at(sch[c_gm], al1_at_c_axis)
        else:  # TILING["AL1_shape"]=[]
            sch[a_l1].compute_at(sch[c_gm], batch_in_out_axis)

        if TILING["BL1_shape"]:
            print_debug('bl1_parts[0]:', bl1_parts[0])
            if bl1_parts[0] != 1:
                sch[b_l1].compute_at(sch[c_l0c], bl1_at_l0c_axis)
            else:  # bl1_parts[0] == 1
                sch[b_l1].compute_at(sch[c_gm], bl1_at_c_axis)
        else:  # TILING["BL1_shape"]=[]
            sch[b_l1].compute_at(sch[c_gm], batch_in_out_axis)

    def _do_double_buffer(fusion_type):
        # a_l1 b_l1
        if TILING.get("manual_pingpong_buffer")["AL1_pbuffer"] == 2 \
                and (TILING["AL1_shape"] != []):
            sch[a_l1].double_buffer()
        if TILING.get("manual_pingpong_buffer")["BL1_pbuffer"] == 2 \
                and (TILING["BL1_shape"] != []):
            sch[b_l1].double_buffer()

        # L0A L0B
        if TILING.get("manual_pingpong_buffer")["AL0_pbuffer"] == 2:
            sch[a_l0a].double_buffer()
        if TILING.get("manual_pingpong_buffer")["BL0_pbuffer"] == 2:
            sch[b_l0b].double_buffer()

        # c_l0c
        _double_buffer_l0c()

        # C_UB
        _double_buffer_cub(fusion_type)

    def _double_buffer_l0c():
        if TILING.get("manual_pingpong_buffer")["CL0_pbuffer"] == 2:
            sch[c_l0c].double_buffer()
            if bias_l0c is not None:
                sch[bias_l0c].double_buffer()
                sch[c_add_bias].double_buffer()
                sch[bias_l0c].preload()
                sch[bias_ub_brc].double_buffer()
                sch[bias_ub_brc].preload()

    def _double_buffer_cub(fusion_type):
        if TILING.get("manual_pingpong_buffer")["CUB_pbuffer"] == 2:
            sch[c_ub].double_buffer()
            if bias_add_vector_ub is not None:
                sch[bias_add_vector_ub].double_buffer()
            if dilate_ub is not None:
                filling_zero_ub = TENSOR_MAP["tensor_fillling_zero"]
                sch[dilate_ub].double_buffer()
                sch[filling_zero_ub].double_buffer()
            if fusion_type in [FUSION_DX_ADD_DRELU, FUSION_DX_DRELU]:
                sch[fusion_dx_gm].double_buffer()
                sch[drelu_ub].double_buffer()
                if fusion_type == FUSION_DX_ADD_DRELU:
                    sch[add_res_ub].double_buffer()
                    if dilate_ub is not None:
                        sch[add_input_ub].double_buffer()
                        sch[add_input_ub].preload()
            elif fusion_type == FUSION_DX_ADD:
                sch[fusion_dx_gm].double_buffer()
                sch[add_res_ub].double_buffer()
                if dilate_ub is not None:
                    sch[add_input_ub].double_buffer()
                    sch[add_input_ub].preload()
            elif fusion_type in (FUSION_DX_DEQUANT, FUSION_DX_DEQUANT_QUANT):
                db_tensor_list = ("dequant2_scalar", "dequant2_vector",
                                  "dequant_relu", "input_ub", "quant_reform",
                                  "quant_vadds_ub", "quant_vmuls_ub",
                                  "scale_sqrt_ub", "cast_i8_ub",
                                  "quant_reform_dilation", "dequant_remove_pad")
                db_tensor_list = [TENSOR_MAP[tensor] for tensor in TENSOR_MAP
                                  if tensor in db_tensor_list]

                for tensor in db_tensor_list:
                    sch[tensor].double_buffer()

    def _do_reused_by(fusion_type):
        def _tensor_reused_by_list(reused_tensor, reused_tensor_list):
            reused_tensor_list = [TENSOR_MAP[tensor] for tensor in TENSOR_MAP
                                  if tensor in reused_tensor_list]
            sch[reused_tensor].reused_by(*reused_tensor_list)
        def _dequant_quant_reused_by():
            reused_list = ("dequant2_vector")
            _tensor_reused_by_list(c_ub, reused_list)
            if fusion_type == FUSION_DX_DEQUANT_QUANT:
                reuse_list = ["scale_sqrt_ub", "quant_vmuls_ub", "cast_i8_ub", "quant_vadds_ub"]
                reused_tensor = None
                if TENSOR_MAP.get("dilate_ub") is not None:
                    reused_tensor = TENSOR_MAP.get("dilate_ub")
                else:
                    reused_tensor_list = ("quant_vmuls_ub", "quant_vadds_ub", "scale_sqrt_ub")
                    for tensor in reused_tensor_list:
                        reused_tensor = TENSOR_MAP.get(tensor)
                        reuse_list.remove(tensor)
                        if reused_tensor is not None:
                            break
                if reused_tensor is not None:
                    _tensor_reused_by_list(reused_tensor, reuse_list)

        if dilate_ub is not None:
            dx_output_ub = dilate_ub
        else:
            dx_output_ub = c_ub
        if fusion_type == FUSION_DX_ADD_DRELU:
            if dilate_ub is not None:
                filling_zero_ub = TENSOR_MAP["tensor_fillling_zero"]
                sch[filling_zero_ub].reused_by(add_input_ub)
                sch[dx_output_ub].reused_by(fusion_dx_gm, add_res_ub)
            else:
                sch[dx_output_ub].reused_by(fusion_dx_gm, drelu_ub, add_res_ub)
        elif fusion_type == FUSION_DX_DRELU:
            sch[dx_output_ub].reused_by(fusion_dx_gm, drelu_ub)
        elif fusion_type == FUSION_DX_ADD:
            if dilate_ub is not None:
                filling_zero_ub = TENSOR_MAP["tensor_fillling_zero"]
                sch[filling_zero_ub].reused_by(add_input_ub)
                sch[dx_output_ub].reused_by(fusion_dx_gm)
            else:
                sch[dx_output_ub].reused_by(fusion_dx_gm, add_res_ub)

        if fusion_type in (FUSION_DX_DEQUANT, FUSION_DX_DEQUANT_QUANT):
            _dequant_quant_reused_by()

    def _fusion_intrin_mapping(fusion_type):
        def _add_res_ub_insn():
            if dilate_ub is None:
                sch[add_res_ub].emit_insn(add_res_ub.op.axis[0], "vector_add")
            else:
                sch[add_res_ub].emit_insn(add_res_ub.op.axis[0], "phony_insn")
        if fusion_type in [FUSION_DX_ADD_DRELU, FUSION_DX_DRELU]:
            sch[bitmask_ub].emit_insn(bitmask_ub.op.axis[0], 'dma_copy')
            sch[drelu_ub].emit_insn(
                drelu_ub.op.axis[0], "vector_selects_bool")
            sch[fusion_dx_gm].emit_insn(fusion_dx_gm.op.axis[0], 'phony_insn')
            if fusion_type == FUSION_DX_ADD_DRELU:
                sch[add_input_ub].emit_insn(
                    add_input_ub.op.axis[0], 'dma_copy')
                _add_res_ub_insn()
        elif fusion_type == FUSION_DX_ADD:
            sch[add_input_ub].emit_insn(add_input_ub.op.axis[0], 'dma_copy')
            _add_res_ub_insn()
            sch[fusion_dx_gm].emit_insn(fusion_dx_gm.op.axis[0], 'phony_insn')
        elif fusion_type in (FUSION_DX_DEQUANT, FUSION_DX_DEQUANT_QUANT):
            tensor_emit_map = {"deq_scale_ub": "dma_copy",
                               "dequant2_vector": "vector_auto",
                               "dequant2_scalar": "vector_auto",
                               "dequant_relu": "vector_relu",
                               "scale_sqrt_ub": "vector_auto",
                               "input_ub": "dma_padding",
                               "quant_vadds_ub": "vector_adds",
                               "quant_vmuls_ub": "vector_muls",
                               "dequant_remove_pad": "dma_copy"
                              }

            for tensor_name in TENSOR_MAP:
                if tensor_emit_map.get(tensor_name) is not None:
                    sch[TENSOR_MAP[tensor_name]].emit_insn(
                        TENSOR_MAP[tensor_name].op.axis[0],
                        tensor_emit_map[tensor_name])
                elif tensor_name in ("dequant1_vector", "dequant1_scalar"):
                    dequant1_ub = TENSOR_MAP[tensor_name]
                    emit = tensor_name.split('_')[1]
                    axis_index = 2 if emit == "vector" else 0
                    sch[dequant1_ub].pragma(
                        dequant1_ub.op.axis[axis_index], 'deq_scale', emit)
                elif tensor_name == "quant_reform":
                    reform_ub = TENSOR_MAP[tensor_name]
                    ndim = len(sch[reform_ub].op.axis)
                    coo, _ = sch[reform_ub].split(
                        sch[reform_ub].op.axis[ndim - 1],
                        CUBE_MKN["float16"]["mac"][1])
                    axis_list = sch[reform_ub].op.axis[0:ndim - 1]
                    sch[reform_ub].reorder(coo, *axis_list)
                    sch[reform_ub].emit_insn(
                        sch[reform_ub].op.axis[2], "vector_auto")
                elif tensor_name == "cast_i8_ub":
                    cast_i8_ub = TENSOR_MAP[tensor_name]
                    round_mode_emit_insn \
                        = 'vector_conv_%s' \
                          % TENSOR_MAP['round_mode'].lower()
                    if not intrinsic_check_support(
                            "Intrinsic_vconv", "f162s8a"):
                        round_mode_emit_insn = 'vector_conv'
                    sch[cast_i8_ub].emit_insn(
                        cast_i8_ub.op.axis[0], round_mode_emit_insn)

    def _intrin_mapping(fusion_type):
        if TILING["AL1_shape"] is not None:
            sch[a_l1].emit_insn(a_l1.op.axis[0], 'dma_copy')
        sch[b_l1].emit_insn(b_l1.op.axis[0], 'dma_copy')
        sch[b_l0b].emit_insn(b_l0b.op.axis[0], 'dma_copy')
        sch[a_l0a].emit_insn(a_l0a.op.axis[0], 'dma_copy')
        if fusion_type not in (FUSION_DX_DEQUANT, FUSION_DX_DEQUANT_QUANT):
            sch[c_ub].emit_insn(c_ub.op.axis[0], 'dma_copy')
        sch[c_gm].emit_insn(l0c_n_inner_inner, 'dma_copy')

        if dilate_ub is not None:
            filling_zero_ub = TENSOR_MAP["tensor_fillling_zero"]

            if bias_add_vector_ub is not None:
                sch[dilate_ub].reused_by(filling_zero_ub, bias_add_vector_ub)
            else:
                sch[dilate_ub].reused_by(filling_zero_ub)
            if fusion_type in (FUSION_DX_ADD_DRELU, FUSION_DX_ADD):
                sch[filling_zero_ub].emit_insn(
                    sch[filling_zero_ub].op.axis[0], "phony_insn")
            else:
                sch[filling_zero_ub].emit_insn(
                    sch[filling_zero_ub].op.axis[0], "vector_dup")
            vadd_at_axis = _dilate_schedule(sch, dilate_ub,
                                            DIM_MAP.get("out_hwdim")[1],
                                            DIM_MAP.get("dilate_dim")[1],
                                            DIM_MAP.get("dilate_dim")[0])
            sch[dilate_ub].emit_insn(vadd_at_axis, "vector_add")
        elif bias_add_vector_ub is not None:
            sch[c_ub].reused_by(bias_add_vector_ub)

        if bias_add_vector_ub is not None:
            sch[bias_ub].emit_insn(sch[bias_ub].op.axis[0], "dma_copy")
            sch[bias_add_vector_ub].emit_insn(
                sch[bias_add_vector_ub].op.axis[0], 'vector_auto',
            )

        if fusion_type != FUSION_NONE:
            _fusion_intrin_mapping(fusion_type)
        mad_dict = {"mad_pattern": 2,
                    "k_outer": [reduce_axis_serial[0],
                                reduce_axis_serial[1],
                                reduce_axis_serial[2]]}

        if bias_ub_brc is not None:
            sch[bias_l0c].reused_by(c_add_bias, c_l0c)
            sch[c_add_bias].emit_insn(c_add_bias.op.axis[0], 'phony_insn')
            cc_outer, _ = sch[bias_l0c].split(
                bias_l0c.op.axis[2], BRC_STANDARD_BLOCK_SIZE)
            sch[bias_l0c].emit_insn(cc_outer, 'dma_copy')
            sch[bias_ub].emit_insn(bias_ub.op.axis[0], 'dma_copy')
            sch[bias_ub_brc].emit_insn(bias_ub_brc.op.axis[0], 'vector_auto')
            mad_dict["init_bias"] = 1

        sch[c_l0c].emit_insn(batch_l0c_inner, 'mad', mad_dict)
        print_ir_conv("intrin mapping", sch)

    TILING.clear()
    dx_res = tensor
    sch = sch_list[0]
    print_ir_conv("schedule", sch)

    # set scope for all tensor
    _set_data_layout(dx_res, sch)
    fusion_type = DeconvParam.get_para_map("FUSION_TYPE")

    print_ir_conv("set scope", sch)

    # get tensor
    a_l1, b_l1, a_l0a, b_l0b, c_ub, dilate_ub, c_l0c, c_gm = \
        TENSOR_MAP.get("a_l1"), TENSOR_MAP.get("b_l1"), \
        TENSOR_MAP.get("a_l0a"), TENSOR_MAP.get("b_l0b"), \
        TENSOR_MAP.get("c_ub"), TENSOR_MAP.get("dilate_ub"), \
        TENSOR_MAP.get("c_l0c"), TENSOR_MAP.get("c_gm")
    drelu_ub, bitmask_ub, add_res_ub, add_input_ub, fusion_dx_gm \
        = TENSOR_MAP.get("drelu_ub"), TENSOR_MAP.get("bitmask_ub"), \
          TENSOR_MAP.get("add_res_ub"), TENSOR_MAP.get("add_input_ub"), \
          TENSOR_MAP.get("fusion_dx_gm")
    bias_add_vector_ub, bias_ub \
        = TENSOR_MAP.get("bias_add_vector"), \
          TENSOR_MAP.get("bias_ub")
    bias_ub_brc, bias_l0c, c_add_bias \
        = TENSOR_MAP.get("bias_ub_brc"), TENSOR_MAP.get("bias_l0c"), \
          TENSOR_MAP.get("c_add_bias")

    get_tiling(tensor, fusion_type)

    # get factor and parts from tiling
    l0c_factor, l0c_ub_parts, al1_parts, bl1_parts, \
    undilate_l0c_m, need_buffer_tile = _get_aicore_tiling_factor()
    al0_axis_factor, bl0_axis_factor, reduce_axis_factor = _get_mmad_factor()
    print_ir_conv("bind to core", sch)

    # split and get axis of l0c, al1, bl1
    batch_in_out_axis, l1_n_outer_outer, batch_in_inner_axis, \
    l1_m_outer_inner_in, l0c_n_inner, l0c_m_inner, tile_axis, \
    noii_axis, blockidx = \
        _get_l0c_and_l1_axis(sch, c_gm, l0c_factor, al1_parts, bl1_parts)
    al1_at_c_axis = batch_in_inner_axis
    bl1_at_c_axis = l1_n_outer_outer
    c_slice_axis = l1_m_outer_inner_in
    print_ir_conv("split with al1 and bl1 factor", sch)

    # attach tensor of CUB
    l0c_n_inner_outer, l0c_n_inner_inner = \
        sch[c_gm].split(l0c_n_inner, nparts=l0c_ub_parts[0])
    l0c_m_inner_outer, l0c_m_inner_inner = \
        sch[c_gm].split(l0c_m_inner, nparts=1)
    add_input_at, l0c_m_inner_outer = \
        sch[c_gm].split(l0c_m_inner_outer, nparts=1)
    sch[c_gm].reorder(l0c_n_inner_outer, add_input_at,
                      l0c_m_inner_outer, l0c_n_inner_inner,
                      l0c_m_inner_inner)
    print_ir_conv("reorder loc", sch)

    _attach_ub(fusion_type)
    print_ir_conv("attach CUB", sch)

    # attach tensor of l0c
    new_c_col_axis = [sch[c_l0c].op.axis[0], sch[c_l0c].op.axis[1],
                      sch[c_l0c].op.axis[2], sch[c_l0c].op.axis[3]]
    sch[c_l0c].compute_at(sch[c_gm], c_slice_axis)
    if bias_l0c is not None:
        sch[bias_l0c].compute_at(sch[c_gm], c_slice_axis)
        sch[c_add_bias].compute_at(sch[c_gm], c_slice_axis)
        sch[bias_ub_brc].compute_at(sch[c_gm], c_slice_axis)

    print_ir_conv("attach l0c", sch)

    # split and get axis of reduce, al0_at_axis, bl0_at_axis
    al0_m_out, bl0_n_outer, k_outer_outer, batch_l0c_inner = \
        _get_l0a_and_l0b_axis(sch, c_l0c, new_c_col_axis, al0_axis_factor,
                              bl0_axis_factor, reduce_axis_factor)
    print_ir_conv("split with al0/bl0/reduce factor", sch)

    # attach tensor of a_l0a
    sch[a_l0a].compute_at(sch[c_l0c], al0_m_out)
    sch[b_l0b].compute_at(sch[c_l0c], bl0_n_outer)
    print_ir_conv("attach l0a/l0b", sch)

    # split and get axis of al1_at_l0c_axis, bl1_at_l0c_axis
    al1_at_l0c_axis, bl1_at_l0c_axis, reduce_axis_serial = \
        _get_al1_and_bl1_axis(sch, c_l0c, al1_parts, bl1_parts, k_outer_outer)

    _attach_al1_bl1()
    print_ir_conv("attach al1/bl1", sch)
    if need_buffer_tile:
        _do_buffer_tile(fusion_type)
        print_ir_conv("after_tile", sch)
    else:
        sch[c_ub].buffer_align((1, 1), (1, 1),
                               (1, CUBE_MKN["float16"]["mac"][0]),
                               (1, CUBE_MKN["float16"]["mac"][0]))
    # double buffer
    _do_double_buffer(fusion_type)
    print_ir_conv("enable double buffer", sch)

    _do_reused_by(fusion_type)
    print_ir_conv("reused_by", sch)

    # preload
    if DeconvParam.get_para_map("DATA_AMOUNT_CUB") * (
            1 + 2*FUSION_TYPE_2_OPERAND_NUM.get(fusion_type)) \
            <= UB_SPACE_SIZE:
        print_debug('dx opti ub preload enable.')
        if fusion_type == FUSION_DX_DRELU:
            sch[bitmask_ub].double_buffer()
            sch[bitmask_ub].preload()
        elif fusion_type == FUSION_DX_ADD_DRELU:
            sch[bitmask_ub].double_buffer()
            sch[bitmask_ub].preload()
            if dilate_ub is None:
                sch[add_input_ub].double_buffer()
                sch[add_input_ub].preload()
        elif fusion_type == FUSION_DX_ADD:
            if dilate_ub is None:
                sch[add_input_ub].double_buffer()
                sch[add_input_ub].preload()
        print_ir_conv("preload", sch)

    # intrin mapping
    _intrin_mapping(fusion_type)

    # clear global cache
    TILING.clear()
    DIM_MAP.clear()
    TENSOR_MAP.clear()
    return sch
