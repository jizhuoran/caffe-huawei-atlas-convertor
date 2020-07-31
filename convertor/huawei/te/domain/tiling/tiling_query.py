"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
Query the tiling of conv
"""
import json
import os
import math
import numpy as np
from te import tvm
from te.platform import CUBE_MKN
from te.platform.cce_conf import get_soc_spec
from te.domain.tiling.tiling_helper import TILING_INSTANCE
from te import platform as cce

AUTO_TILING_TYPE = 0
CCE_TILING_TYPE = 1
REPOSITORY_TILING_TYPE = 2
PRIORITY_TILING_TYPE = 3
CUSTOM_TILING_TYPE = 4
TUNING_TILING_TYPE = 5

MINI_FLAG = 1
CLOUD_FLAG = 2
LHISI_V300ES_FLAG = 3
MINI_V200_FLAG = 4
MINI_V200MDC_FLAG = 5

MAX_UINT64 = 2**64 - 1
MAX_UINT32 = 4294967295
MAX_UINT16 = 65535
MAX_UINT8 = 255
MAX_UINT4 = 15
MAX_BOOL = 1

SHAPE_LENGTH4 = 4
SHAPE_LENGTH5 = 5
SHAPE_LENGTH6 = 6

C04 = 4

# Encode the type of tensor data
DTYPE_DICT = {
        'uint8': 0,
        'int8': 1,
        'float16': 2,
        'int32': 3,
        'float32': 4
}

# Encode the type of op
OP_TYPE_DICT = {'conv2d': 0, 'conv2d_backprop_input': 1, \
                'conv2d_backprop_filter': 2, \
                'depthwise_conv2d_forward': 3, 'depthwise_bp_input': 4,
                'depthwise_bp_filter': 5, 'depthwise_conv2d_native_v200': 6,
                'matmul': 7, 'convolution': 0, 'convolution_3d': 8,
                'conv3d_backprop_input': 9, 'conv3d_backprop_filter': 10}

# define the support tiling type
SUPPORT_TILING_TYPE = {
    "atc_repository_tiling": REPOSITORY_TILING_TYPE,
    "atc_tuning_tiling": TUNING_TILING_TYPE,
    "auto_tiling": AUTO_TILING_TYPE,
    "cce_tiling": CCE_TILING_TYPE,
    "repository_tiling": REPOSITORY_TILING_TYPE,
    "custom_tiling": CUSTOM_TILING_TYPE,
    "priority_tiling": PRIORITY_TILING_TYPE,
    "tuning_tiling": TUNING_TILING_TYPE
}

def get_platform_info():
    """Get the information of hardware
        Parameters
        ----------
        Returns
        -------
        platform_type : num
        The platform type.
    """
    # version
    version_dict = {
        "Ascend310": 1,
        "Ascend910": 2,
        "Hi3796CV300ES": 3,
        "Ascend610": 4,
        "Ascend620": 5,
    }
    return version_dict[get_soc_spec("SOC_VERSION")]


def is_conv3d_op_tag(op_tag):
    """Check if Conv3D type"""
    return op_tag in OP_TYPE_DICT and \
           (OP_TYPE_DICT[op_tag] >= OP_TYPE_DICT['convolution_3d']) and \
           (OP_TYPE_DICT[op_tag] <= OP_TYPE_DICT['conv3d_backprop_filter'])


def tiling_query(a_shape, b_shape, c_shape=None, a_dtype='float16', \
                 b_dtype='float16', c_dtype='float16', mad_dtype='float16', \
                 padl=0, padr=0, padu=0, padd=0, strideh=1, stridew=1, \
                 strideh_expand=1, stridew_expand=1, dilationh=1, \
                 dilationw=1, group=1, fused_double_operand_num=0, \
                 bias_flag=False, op_tag='conv2d', \
                 fused_coefficient=None, \
                 fused_channel_wise=None, \
                 padf=0, padb=0, strided=0):
    """Query the tiling of convolution/convolution 3d from module

    Parameters
    ----------
    A_shape: 5D/6D shape information of feature map
        Input argument.

    B_shape: 5D/6D shape information of filter
        Input argument

    C_shape: 5D/6D shape information of output
        Input argument

    A_dtype, B_dtype, C_dtype, mad_dtype: type of data
        0 represent uint8; 1 represent int8; 2 represent float16,
        3 represent float32

    padl,  padr, padu, padd, padf, padb: the pad of six direction
        padl, padr = padW
        padu, padd = padH
        padf, padb = padD

    strideH, strideW, strideD: the stride of three direction
        Input argument

    dilationH, dilationW: the dilation param of
        H and W direction of dilation convolution
        Input argument

    strideH_expand, strideW_expand: the param of expansion on
        H and W direction of A_shape
        Input argument

    group: the param of group convolution
        Input argument

    fused_double_operand_num: the param of num of
        fused double operand vector operator
        Input argument

    bias_flag: whether there is bias
        bool value

    op_tag: the tag of operator
        string: see OP_TYPE_DICT

    fused_coefficient: the params of num of
        fused double operand vector operator
        list value

    fused_channel_wise: the params of num of LOA,LOB,LOC in chanel direction
        list value

    Returns
    -------
    tiling : dict
        The result.
    """
    # preprocess the param
    fused_coefficient = ([0, 0, 0]
                         if fused_coefficient is None else fused_coefficient)
    fused_channel_wise = ([0, 0, 0] if fused_channel_wise is None else
                          fused_channel_wise)
    _, _, fused_loc_coefficient = list(fused_coefficient)
    _, _, fused_loc_channel_wise = list(fused_channel_wise)
    fused_double_operand_num = fused_double_operand_num + \
                    fused_loc_channel_wise + fused_loc_coefficient

    platform = get_platform_info()
    params_in = {"A_shape": a_shape, "B_shape": b_shape, "C_shape": c_shape,
                 "A_dtype": a_dtype, "B_dtype": b_dtype, "C_dtype": c_dtype,
                 "mad_dtype": mad_dtype,
                 "padl": padl, "padr": padr, "padu": padu, "padd": padd,
                 "padf": padf, "padb": padb,
                 "strideH": strideh, "strideW": stridew, "strideD": strided,
                 "strideH_expand": strideh_expand,
                 "strideW_expand": stridew_expand,
                 "dilationH": dilationh, "dilationW": dilationw,
                 "group": group,
                 "fused_double_operand_num": fused_double_operand_num,
                 "bias_flag": bias_flag, "op_type": op_tag,
                 "platform": platform}

    check_params_in(params_in)
    shape_encode_array = encode_array(params_in)
    tiling_type = TILING_INSTANCE.get_tiling_type()
    # End to end scene, the tiling type is used to query
    # the tiling from repository
    tiling = obtain_tiling(shape_encode_array, tiling_type, params_in)

    return tiling


def check_params_in(params_in):
    """
    check the input params

    Parameters
    ----------
    params_in: the input params

    Returns
    -------
    """
    def check_param_type(param, type_list):
        """check whether the type of param is correct

        Parameters
        ----------
        param: instance
            the instance to check
        type_list: type
            type of data
        """
        check_list = tuple(type_list)
        if not isinstance(param, check_list):
            raise TypeError("the type of param is error, \
                only support %s, but the type of param is %s" % \
                (str(type_list), type(param)))

    def check_param_length(param, length_list):
        """check whether the length of param is correct

        Parameters
        ----------
        param: instance
            the instance to check
        length_list: list
            length of data
        """
        if not len(param) in length_list:
            raise ValueError("the length of param is error, \
                only support %s, but the length of param is %s" % \
                (str(length_list), len(param)))

    def check_support_range(param, support_dict):
        """check whether the range of param is correct

        Parameters
        ----------
        param: instance
            the instance to check
        support_dict: dict
            support context of data
        """
        if not param in support_dict.keys():
            raise ValueError("the input param is error, \
                only support %s,  but the param is %s" % \
                (str(support_dict.keys()), param))

    # check the type of param
    check_param_type(params_in, [dict])
    check_param_type(params_in.get('A_shape'), [list, tuple])
    check_param_type(params_in.get('B_shape'), [list, tuple])
    # check the type and value and C_shape
    if isinstance(params_in.get('C_shape'), (list, tuple)):
        if not (len(params_in.get('C_shape')) in \
            [SHAPE_LENGTH4, SHAPE_LENGTH5, SHAPE_LENGTH6]):
            raise ValueError("the length of param is error, \
                only support 4 or 5")
    elif params_in.get('C_shape') is not None:
        raise TypeError("the type of param is error, \
                only support list tuple or None, but the type of param \
                is %s" % type(params_in.get('C_shape')))
    # check the type of param
    check_param_type(params_in.get('A_dtype'), [str])
    check_param_type(params_in.get('B_dtype'), [str])
    check_param_type(params_in.get('C_dtype'), [str])
    check_param_type(params_in.get('mad_dtype'), [str])
    check_param_type(params_in.get('padl'), [int])
    check_param_type(params_in.get('padr'), [int])
    check_param_type(params_in.get('padu'), [int])
    check_param_type(params_in.get('padd'), [int])
    check_param_type(params_in.get('strideH'), [int])
    check_param_type(params_in.get('strideW'), [int])
    check_param_type(params_in.get('strideH_expand'), [int])
    check_param_type(params_in.get('strideW_expand'), [int])
    check_param_type(params_in.get('dilationH'), [int])
    check_param_type(params_in.get('dilationW'), [int])
    check_param_type(params_in.get('group'), [int])
    check_param_type(params_in.get('fused_double_operand_num'), [int, float])
    check_param_type(params_in.get('bias_flag'), [bool, int])
    check_param_type(params_in.get('op_type'), [str])
    check_param_type(params_in.get('platform'), [int])

    # check the length of param
    check_param_length(params_in.get('A_shape'),
                       [SHAPE_LENGTH5, SHAPE_LENGTH6])
    check_param_length(params_in.get('B_shape'),
                       [SHAPE_LENGTH5, SHAPE_LENGTH6])
    check_support_range(params_in.get('A_dtype'), DTYPE_DICT)
    check_support_range(params_in.get('B_dtype'), DTYPE_DICT)
    check_support_range(params_in.get('C_dtype'), DTYPE_DICT)
    check_support_range(params_in.get('mad_dtype'), DTYPE_DICT)
    check_support_range(params_in.get('op_type'), OP_TYPE_DICT)


def obtain_tiling(shape_encode_array, tiling_type, params_in):
    """Get tilling

    Parameters
    ----------
    shape_encode_array: encoded shape info
    tiling_type: tiling type

    Returns
    -------
    tiling
    """
    if tiling_type not in SUPPORT_TILING_TYPE:
        raise ValueError(
            'tiling_type: {} is not supported.'.format(tiling_type))
    tiling_type_num = SUPPORT_TILING_TYPE[tiling_type]
    if tiling_type_num != TUNING_TILING_TYPE:
        tiling_result = tvm.get_global_func("_tiling_query")
        ret = tiling_result(shape_encode_array, tiling_type_num)
        res = list(ret.asnumpy())
        tiling = decode(res)
        TILING_INSTANCE.set_params(params_in)
    else:
        tiling = TILING_INSTANCE.get_tiling(params_in)

    return tiling


def encode_array(params_in):
    """encode the information of shape to the list of uint32 digit

    Parameters
    ----------
    params_in: dict of params
        include all information of shape

    Returns
    -------
    shape_encode_array : tvm.nd.array
        The encoded params, array
    """
    _ca0 = params_in["A_shape"][-1]
    if _ca0 != C04:
        config = CUBE_MKN[params_in["B_dtype"]]
        _ca0 = config['mac'][1]

    params_dict = {}
    if is_conv3d_op_tag(params_in['op_type']):
        params_dict['batchA'] = params_in["A_shape"][0]
        params_dict['da'] = params_in["A_shape"][1]
        params_dict['ca1'] = ((params_in["A_shape"][2]
                               * params_in["A_shape"][5]) + _ca0 - 1) // _ca0
        params_dict['ca0'] = _ca0
        params_dict['ha'] = params_in["A_shape"][3]
        params_dict['wa'] = params_in["A_shape"][4]
        params_dict['batchB'] = params_in['B_shape'][0]
        params_dict['db'] = params_in["B_shape"][1]
        params_dict['cb'] = params_in['B_shape'][2] * params_in['B_shape'][5]
        params_dict['hb'] = params_in['B_shape'][3]
        params_dict['wb'] = params_in['B_shape'][4]
        params_dict['padf'] = params_in['padf']
        params_dict['padb'] = params_in['padb']
        params_dict['strideD'] = params_in['strideD']
    else:
        params_dict['batchA'] = params_in["A_shape"][0]
        params_dict['da'] = 0
        params_dict['ca1'] = ((params_in["A_shape"][1]
                               * params_in["A_shape"][4]) + _ca0 - 1) // _ca0
        params_dict['ca0'] = _ca0
        params_dict['ha'] = params_in["A_shape"][2]
        params_dict['wa'] = params_in["A_shape"][3]
        params_dict['batchB'] = params_in['B_shape'][0]
        params_dict['db'] = 0
        params_dict['cb'] = params_in['B_shape'][1] * params_in['B_shape'][4]
        params_dict['hb'] = params_in['B_shape'][2]
        params_dict['wb'] = params_in['B_shape'][3]
        params_dict['padf'] = 0
        params_dict['padb'] = 0
        params_dict['strideD'] = 0

    params_dict['dc'] = 0
    params_dict['hc'] = 0
    params_dict['wc'] = 0
    if params_in['op_type'] == 'conv2d_backprop_filter'\
        or params_in['op_type'] == 'depthwise_bp_filter':
        params_dict['hc'] = params_in['C_shape'][2]
        params_dict['wc'] = params_in['C_shape'][3]
    elif params_in['op_type'] == 'conv2d_backprop_input' or\
            params_in['op_type'] == 'matmul':
        if isinstance(params_in['C_shape'], list):
            params_dict['hc'] = params_in['C_shape'][2]
            params_dict['wc'] = params_in['C_shape'][3]
        else:
            # set default value of hc and wc
            pass
    elif params_in['op_type'] == 'conv3d_backprop_input' or \
            params_in['op_type'] == 'conv3d_backprop_filter':
        params_dict['dc'] = params_in['C_shape'][1]
        params_dict['hc'] = params_in['C_shape'][2]
        params_dict['wc'] = params_in['C_shape'][3]

    params_dict['A_type_encode'] = DTYPE_DICT[params_in['A_dtype']]
    params_dict['B_dtype_encode'] = DTYPE_DICT[params_in['B_dtype']]
    params_dict['C_dtype_encode'] = DTYPE_DICT[params_in['C_dtype']]
    params_dict['mad_dtype_encode'] = DTYPE_DICT[params_in['mad_dtype']]
    params_dict['padl'] = params_in['padl']
    params_dict['padr'] = params_in['padr']
    params_dict['padu'] = params_in['padu']
    params_dict['padd'] = params_in['padd']
    params_dict['strideH'] = params_in['strideH']
    params_dict['strideW'] = params_in['strideW']
    params_dict['strideH_expand'] = params_in['strideH_expand']
    params_dict['strideW_expand'] = params_in['strideW_expand']
    params_dict['dilationH'] = params_in['dilationH']
    params_dict['dilationW'] = params_in['dilationW']
    params_dict['group'] = params_in['group']
    # The param encode fixed point data,
    # the data is accurate to two decimal places
    params_dict['fused_double_operand_num'] = math.ceil( \
                                params_in['fused_double_operand_num'] * 100)
    params_dict['bias_flag'] = params_in['bias_flag']
    params_dict['op_tag_encode'] = OP_TYPE_DICT[params_in['op_type']]
    params_dict['platform'] = params_in['platform']

    params_encode = encode(params_dict)
    shape_encode = np.array([params_encode[0], params_encode[1],
                             params_encode[2], params_encode[3],
                             params_encode[4], params_encode[5],
                             params_encode[6], params_encode[7],
                             params_encode[8], params_encode[9],
                             params_encode[10], params_encode[11],
                             params_encode[12], params_encode[13],
                             params_encode[14], params_encode[15]],
                            dtype="uint32")

    return tvm.nd.array(shape_encode)

def encode(params):
    """encode the information of shape to the list of uint32 digit

    Parameters
    ----------
    params: dict of params
        include all information of shape

    Returns
    -------
    params_encode : list of encoded params
        The encoded params, include 8 uint32 numbers
    """
    # according to the define of TBETilingArgs structure,
    # the combination of multiple members occupy 32 bits storage space
    # The storage order of 32-byte aligned members is positive,
    # and the order of structure members stored in 32-byte is opposite
    params_encode = []
    # batchA occupies 32 bits
    params_encode.append(params['batchA'])
    # batchB occupies 32 bits
    params_encode.append(params['batchB'])
    # cb occupies 32 bits
    params_encode.append(params['cb'])
    # ca1 occupies 32 bits
    params_encode.append(params['ca1'])
    # ha occupies 16 bits, wa occupies 16bit,
    # the combination occupies 32 bits
    params_encode.append((params['wa'] << 16) + params['ha'])
    # hb occupies 16 bits, wb occupies 16 bits,
    # the combination occupies 32 bits
    params_encode.append((params['wb'] << 16) + params['hb'])
    # hc occupies 16 bits, wc occupies 16 bits,
    # the combination occupies 32 bits
    params_encode.append((params['wc'] << 16) + params['hc'])
    # fusedDoubleOperandNum occupies 16 bits, ca0 occupies 8 bits,
    # aType occupies 8 bits, the combination occupies 32 bits
    params_encode.append((params['A_type_encode'] << 24) + \
        (params['ca0'] << 16) + params['fused_double_operand_num'])
    # bType occupies 8 bits, cType occupies 8 bits,
    # madType: 8 bits, padl: 8 bits, the combination occupies 32 bits
    params_encode.append((params['padl'] << 24) + \
        (params['mad_dtype_encode'] << 16) + \
        (params['C_dtype_encode'] << 8) + params['B_dtype_encode'])
    # padr occupies 8 bits, padu occupies 8 bits,
    # padd: 8 bits, strideH: 8 bits, the combination occupies 32 bits
    params_encode.append((params['strideH'] << 24) + \
        (params['padd'] << 16) + (params['padu'] << 8) + params['padr'])
    # strideW occupies 8 bits, strideHExpand occupies 8 bits,
    # strideWExpand: 8 bits, dilationH: 8 bits,
    # the combination occupies 32 bits
    params_encode.append((params['dilationH'] << 24) + \
        (params['strideW_expand'] << 16) + \
        (params['strideH_expand'] << 8) + params['strideW'])
    # dilationW occupies 8 bits, group occupies 8 bits,
    # opTag occupies 8 bits, biasFlag occupies 1 bits,
    # platForm occupies 7 bits, the combination occupies 32 bits
    params_encode.append((params['platform'] << 25) + \
        (params['bias_flag'] << 24) + (params['op_tag_encode'] << 16) + \
        (params['group'] << 8) + params['dilationW'])
    # da occupies 32 bits
    params_encode.append(params['da'])
    # db occupies 32 bits
    params_encode.append(params['db'])
    # padf occupies 8 bits, padb occupies 8 bits,
    # strideD: 8 bits, the combination occupies 24 bits
    params_encode.append((params['strideD'] << 16) + \
                         (params['padb'] << 8) + params['padf'])
    # dc occupies 32 bits
    params_encode.append(params['dc'])
    return params_encode

def decode(tiling_encode):
    """decode the information of tiling from the list of uint32 digit

    Parameters
    ----------
    tiling_encode: list of encoded tiling
        encoded tiling, includes 11 uint32 digits

    Returns
    -------
    tiling : dict of tiling
        The decoded tiling
    """

    # according to the define of TBETilingResult structure,
    # the one member of tiling_encode occupies 32 bits
    # and includes multiple members of TBETilingResult structure
    # tiling_encode[0] includes 32-bit digits, AL1_shape_0 occupies 32-bit
    al1_shape_0 = (tiling_encode[0] & MAX_UINT32)
    # tiling_encode[1] includes 32-bit digits, BL1_shape_0 occupies 32-bit
    bl1_shape_0 = (tiling_encode[1] & MAX_UINT32)
    # tiling_encode[2] includes 32-bit digits,
    # AL1_shape_1 occupies low 16-bit , AL1_shape_2 occupies high 16-bit
    al1_shape_1 = ((tiling_encode[2] & MAX_UINT16))
    al1_shape_2 = ((tiling_encode[2] >> 16) & MAX_UINT16)
    # tiling_encode[3] includes AL1_shape_3 and BL1_shape_1,
    # AL1_shape_3 occupies low 16-bit, BL1_shape_1 occupies high 16-bit
    al1_shape_3 = (tiling_encode[3] & MAX_UINT16)
    bl1_shape_1 = ((tiling_encode[3] >> 16) & MAX_UINT16)
    # tiling_encode[4] includes BL1_shape_2 and BL1_shape_3,
    # BL1_shape_2 occupies low 16-bit, BL1_shape_3 occupies high 16-bit
    bl1_shape_2 = (tiling_encode[4] & MAX_UINT16)
    bl1_shape_3 = ((tiling_encode[4] >> 16) & MAX_UINT16)
    # tiling_encode[5] includes AL0_matrix_0 and AL0_matrix_1,
    # AL0_matrix_0 occupies low 16-bit, AL0_matrix_1 occupies high 16-bit
    al0_matrix_0 = (tiling_encode[5] & MAX_UINT16)
    al0_matrix_1 = ((tiling_encode[5] >> 16) & MAX_UINT16)
    # tiling_encode[6] includes AL0_matrix_2, AL0_matrix_3 and AL0_matrix_4,
    # AL0_matrix_2 occupies low 8-bit, AL0_matrix_3 occupies middle 8-bit,
    # AL0_matrix_4 occupies high 16-bit
    al0_matrix_2 = (tiling_encode[6] & MAX_UINT8)
    al0_matrix_3 = ((tiling_encode[6] >> 8) & MAX_UINT8)
    al0_matrix_4 = ((tiling_encode[6] >> 16) & MAX_UINT16)
    # tiling_encode[7] includes AL0_matrix_5 and BL0_matrix_0,
    # AL0_matrix_5 occupies low 16-bit, BL0_matrix_0 occupies high 16-bit
    al0_matrix_5 = (tiling_encode[7] & MAX_UINT16)
    bl0_matrix_0 = ((tiling_encode[7] >> 16) & MAX_UINT16)
    # tiling_encode[8] includes BL0_matrix_1, BL0_matrix_2 and BL0_matrix_3,
    # BL0_matrix_1 occupies low 16-bit, # BL0_matrix_2 occupies middle 8-bit,
    # BL0_matrix_3 occupies high 8-bit
    bl0_matrix_1 = (tiling_encode[8] & MAX_UINT16)
    bl0_matrix_2 = ((tiling_encode[8] >> 16) & MAX_UINT8)
    bl0_matrix_3 = ((tiling_encode[8] >> 24) & MAX_UINT8)
    # tiling_encode[9] includes BL0_matrix_4 and BL0_matrix_5,
    # BL0_matrix_4 occupies low 16-bit, BL0_matrix_5 occupies high 16-bit
    bl0_matrix_4 = (tiling_encode[9] & MAX_UINT16)
    bl0_matrix_5 = ((tiling_encode[9] >> 16) & MAX_UINT16)
    # tiling_encode[10] includes CL0_matrix_0 and CL0_matrix_1,
    # CL0_matrix_0 occupies low 16-bit, CL0_matrix_1 occupies high 16-bit
    cl0_matrix_0 = (tiling_encode[10] & MAX_UINT16)
    cl0_matrix_1 = ((tiling_encode[10] >> 16) & MAX_UINT16)
    # tiling_encode[11] includes CL0_matrix_2, CL0_matrix_3 and CL0_matrix_4,
    # CL0_matrix_2 occupies low 8-bit, # CL0_matrix_3 occupies middle 8-bit,
    # CL0_matrix_4 occupies high 16-bit
    cl0_matrix_2 = (tiling_encode[11] & MAX_UINT8)
    cl0_matrix_3 = ((tiling_encode[11] >> 8) & MAX_UINT8)
    cl0_matrix_4 = ((tiling_encode[11] >> 16) & MAX_UINT16)
    # tiling_encode[12] includes CL0_matrix_5 and CUB_matrix_0,
    # CL0_matrix_5 occupies low 16-bit, CUB_matrix_0 occupies high 16-bit
    cl0_matrix_5 = (tiling_encode[12] & MAX_UINT16)
    cub_matrix_0 = ((tiling_encode[12] >> 16) & MAX_UINT16)
    # tiling_encode[13] includes CUB_matrix_1, CUB_matrix_2 and CUB_matrix_3,
    # CUB_matrix_1 occupies low 16-bit,
    # CUB_matrix_2 occupies middle 8-bit, CUB_matrix_3 occupies high 8-bit
    cub_matrix_1 = (tiling_encode[13] & MAX_UINT16)
    cub_matrix_2 = ((tiling_encode[13] >> 16) & MAX_UINT8)
    cub_matrix_3 = ((tiling_encode[13] >> 24) & MAX_UINT8)
    # tiling_encode[14] includes CUB_matrix_4 and CUB_matrix_5,
    # CUB_matrix_4 occupies low 16-bit, CUB_matrix_5 occupies high 16-bit
    cub_matrix_4 = (tiling_encode[14] & MAX_UINT16)
    cub_matrix_5 = ((tiling_encode[14] >> 16) & MAX_UINT16)
    # tiling_encode[15] includes AUB_shape_0, AUB_shape_0 occupies 32-bit
    aub_shape_0 = (tiling_encode[15] & MAX_UINT32)
    # tiling_encode[16] includes BUB_shape_0, BUB_shape_0 occupies 32-bit
    bub_shape_0 = (tiling_encode[16] & MAX_UINT32)
    # tiling_encode[17] includes AUB_shape_1 and AUB_shape_2,
    # AUB_shape_1 occupies low 16-bit, AUB_shape_2 occupies high 16-bit
    aub_shape_1 = (tiling_encode[17] & MAX_UINT16)
    aub_shape_2 = ((tiling_encode[17] >> 16) & MAX_UINT16)
    # tiling_encode[18] includes AUB_shape_3 and BUB_shape_1,
    # AUB_shape_3 occupies low 16-bit, BUB_shape_1 occupies high 16-bit
    aub_shape_3 = (tiling_encode[18] & MAX_UINT16)
    bub_shape_1 = ((tiling_encode[18] >> 16) & MAX_UINT16)
    # tiling_encode[19] includes BUB_shape_2 and BUB_shape_3,
    # BUB_shape_2 occupies low 16-bit, BUB_shape_3 occupies high 16-bit
    bub_shape_2 = (tiling_encode[19] & MAX_UINT16)
    bub_shape_3 = ((tiling_encode[19] >> 16) & MAX_UINT16)
    # tiling_encode[20] includes batch_dim and n_dim,
    # batch_dim occupies low 16-bit, n_dim occupies high 16-bit
    batch_dim = (tiling_encode[20] & MAX_UINT16)
    n_dim = ((tiling_encode[20] >> 16) & MAX_UINT16)
    # tiling_encode[21] includes m_dim and group_dim,
    # m_dim occupies low 16-bit, group_dim occupies high 16-bit
    m_dim = (tiling_encode[21] & MAX_UINT16)
    group_dim = ((tiling_encode[21] >> 16) & MAX_UINT16)
    # tiling_encode[22] includes AUB_pbuffer, BUB_pbuffer,
    # AL1_pbuffer, BL1_pbuffer, AL0_pbuffer, BL0_pbuffer,
    # CL0_pbuffer and CUB_pbuffer,
    # AUB_pbuffer occupies low 16-bit, BUB_pbuffer occupies middle 4-bit,
    # AL1_pbuffer occupies next 4-bit, BL1_pbuffer occupies next 4-bit,
    # AL0_pbuffer: 4 bits, BL0_pbuffer: 4 bits,
    # CL0_pbuffer: 4 bits, CUB_pbuffer: 4 bits
    aub_pbuffer = (tiling_encode[22] & MAX_UINT4)
    bub_pbuffer = ((tiling_encode[22] >> 4) & MAX_UINT4)
    al1_pbuffer = ((tiling_encode[22] >> 8) & MAX_UINT4)
    bl1_pbuffer = ((tiling_encode[22] >> 12) & MAX_UINT4)
    al0_pbuffer = ((tiling_encode[22] >> 16) & MAX_UINT4)
    bl0_pbuffer = ((tiling_encode[22] >> 20) & MAX_UINT4)
    cl0_pbuffer = ((tiling_encode[22] >> 24) & MAX_UINT4)
    cub_pbuffer = ((tiling_encode[22] >> 28) & MAX_UINT4)
    # tiling_encode[23] includes UBG_pbuffer, n_bef_batch_flag,
    # n_bef_group_flag, batch_bef_group_flag,
    # A_overhead_opt_flag and B_overhead_opt_flag,
    # UBG_pbuffer occupies low 4-bit, n_bef_batch_flag occupies next 1-bit,
    # n_bef_group_flag: 1 bits, batch_bef_group_flag: 1 bits,
    # A_overhead_opt_flag: 1 bits, B_overhead_opt_flag occupies 1 bit,
    ubg_pbuffer = (tiling_encode[23] & MAX_UINT4)
    n_bef_batch_flag = ((tiling_encode[23] >> 4) & MAX_BOOL)
    n_bef_group_flag = ((tiling_encode[23] >> 5) & MAX_BOOL)
    batch_bef_group_flag = ((tiling_encode[23] >> 6) & MAX_BOOL)
    a_overhead_opt_flag = ((tiling_encode[23] >> 7) & MAX_BOOL)
    b_overhead_opt_flag = ((tiling_encode[23] >> 8) & MAX_BOOL)

    # BUB_shape_2 support special value None
    if bub_shape_2 == 0:
        bub_shape_2 = None

    # BL1_shape_2 support special value None
    if bl1_shape_2 == 0:
        bl1_shape_2 = None

    # BL0_matrix_4 support special value None
    if bl0_matrix_4 == 0:
        bl0_matrix_4 = None

    # default set  channel_wise_flag

    aub_channel_wise_flag = None
    bub_channel_wise_flag = None
    cub_channel_wise_flag = True

    # Fill the dictionary of Tiling
    tiling = {"AUB_shape": [aub_shape_0, aub_shape_1, aub_shape_2, \
                            aub_shape_3], \
            "BUB_shape": [bub_shape_0, bub_shape_1, bub_shape_2, \
                            bub_shape_3], \
            "AL1_shape": [al1_shape_0, al1_shape_1, al1_shape_2, \
                            al1_shape_3], \
            "BL1_shape": [bl1_shape_0, bl1_shape_1, bl1_shape_2, \
                            bl1_shape_3], \
            "AL0_matrix": [al0_matrix_0, al0_matrix_1, al0_matrix_2, \
                            al0_matrix_3, al0_matrix_4, al0_matrix_5], \
            "BL0_matrix": [bl0_matrix_0, bl0_matrix_1, bl0_matrix_2, \
                            bl0_matrix_3, bl0_matrix_4, bl0_matrix_5], \
            "CL0_matrix": [cl0_matrix_0, cl0_matrix_1, cl0_matrix_2, \
                            cl0_matrix_3, cl0_matrix_4, cl0_matrix_5], \
            "CUB_matrix": [cub_matrix_0, cub_matrix_1, cub_matrix_2, \
                            cub_matrix_3, cub_matrix_4, cub_matrix_5], \
            "block_dim": [batch_dim, n_dim, m_dim, group_dim], \
            "n_bef_batch_flag": n_bef_batch_flag, \
            "n_bef_group_flag": n_bef_group_flag, \
            "batch_bef_group_flag": batch_bef_group_flag, \
            "A_overhead_opt_flag": a_overhead_opt_flag, \
            "B_overhead_opt_flag": b_overhead_opt_flag, \
            "AUB_channel_wise_flag": aub_channel_wise_flag, \
            "BUB_channel_wise_flag": bub_channel_wise_flag, \
            "CUB_channel_wise_flag": cub_channel_wise_flag, \
            "manual_pingpong_buffer": {"AUB_pbuffer": aub_pbuffer, \
                                        "BUB_pbuffer": bub_pbuffer, \
                                        "AL1_pbuffer": al1_pbuffer, \
                                        "BL1_pbuffer": bl1_pbuffer, \
                                        "AL0_pbuffer": al0_pbuffer, \
                                        "BL0_pbuffer": bl0_pbuffer, \
                                        "CL0_pbuffer": cl0_pbuffer, \
                                        "CUB_pbuffer": cub_pbuffer, \
                                        "UBG_pbuffer": ubg_pbuffer}}

    # AUB_shape support special value None
    if aub_shape_0 == 0:
        aub_shape_0 = None
        tiling["AUB_shape"] = aub_shape_0

    # BUB_shape support special value None
    if bub_shape_0 == 0:
        bub_shape_0 = None
        tiling["BUB_shape"] = bub_shape_0

    # AL1_shape support special value [] and None
    if al1_shape_0 == MAX_UINT32:
        al1_shape_0 = []
        tiling["AL1_shape"] = al1_shape_0
    elif al1_shape_0 == 0:
        al1_shape_0 = None
        tiling["AL1_shape"] = al1_shape_0

    # BL1_shape support special value [] and None
    if bl1_shape_0 == 0:
        bl1_shape_0 = None
        tiling["BL1_shape"] = bl1_shape_0
    elif bl1_shape_0 == MAX_UINT32:
        bl1_shape_0 = []
        tiling["BL1_shape"] = bl1_shape_0

    # BL0_matrix support special value []
    if bl0_matrix_0 == MAX_UINT16:
        tiling['BL0_matrix'] = []

    return tiling
