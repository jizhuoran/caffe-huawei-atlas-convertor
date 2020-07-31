#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

TBE operator param encoder
"""
import math
import numpy as np
from copy import deepcopy
from te import tvm
from te.platform import CUBE_MKN
from te.domain.tiling.op_param_encode.operator_params_encoder \
    import BaseClassParamsEncoder


C04 = 4
# define the type of memory
DDR_MEMORY = 0
L1_MEMORY = 1
L2_MEMORY = 2
EMPTY_MEMORY = 3

# define the type of L1 fusion
DEFAULT_VALUE = -1
L1_DEPTH_FUSION = 0
L1_BREADTH_FUSION = 1
L1_NO_FUSION = 2
L2_FUSION = 3
L2_NO_FUSION = 2

# define the const value
CONST_VALUE0 = 0
CONST_VALUE1 = 1

# length of shape
SHAPE_LENGTH2 = 2
SHAPE_LENGTH3 = 3
SHAPE_LENGTH4 = 4
SHAPE_LENGTH5 = 5


class Conv2dParamsEncoder(BaseClassParamsEncoder):
    """
    Child class for conv2d Params Encoder
    """

    def __init__(self):
        '''
        init the super class
        '''
        super(Conv2dParamsEncoder, self).__init__()

    def encode_array(self, input_args):
        """
        encode the input params to tvm.nd.array

        Parameters
        ----------
        input_args: the input params

        Returns
        -------
        tvm.nd.array: the NDArray
        """
        params_in = deepcopy(input_args)
        # first: check the params from the interface
        self.check_info_dict(params_in)
        # second: preprocess the params from the interface
        self.preprocess_info_dict(params_in)

        # third: encode the params to tvm.nd.array
        return self.encode(params_in)

    def check_info_dict(self, params_in):
        """
        check the input params

        Parameters
        ----------
        params_in: the input params

        Returns
        -------
        """
        # preprocess the param
        fused_coefficient = params_in.get('fused_coefficient', None)
        params_in['fused_coefficient'] = ([0, 0, 0] if \
            fused_coefficient is None else fused_coefficient)
        fused_channel_wise = params_in.get('fused_channel_wise', None)
        params_in['fused_channel_wise'] = ([0, 0, 0] if \
            fused_channel_wise is None else fused_channel_wise)

        # check the type of param
        self.check_param_type(params_in, [dict])
        self.check_param_type(params_in.get('a_shape'), [list])
        self.check_param_type(params_in.get('b_shape'), [list])
        self.check_param_type(params_in.get('c_shape'), [list])
        # check the type of param
        self.check_param_type(params_in.get('a_dtype'), [str])
        self.check_param_type(params_in.get('b_dtype'), [str])
        self.check_param_type(params_in.get('c_dtype'), [str])
        self.check_param_type(params_in.get('mad_dtype'), [str])
        self.check_param_type(params_in.get('pad'), [list, tuple])
        self.check_param_type(params_in.get('stride'), [list, tuple])
        self.check_param_type(params_in.get( \
            'dilation'), [list, tuple])
        self.check_param_type(params_in.get('fused_coefficient'), [list])
        self.check_param_type(params_in.get('fused_channel_wise'), [list])
        self.check_param_type(params_in.get('group', CONST_VALUE1), [int])
        self.check_param_type(params_in.get('bias_flag', False), [bool, int])
        self.check_param_type(params_in.get('op_type', 'conv2d'), [str])
        self.check_param_type(params_in.get( \
            'in_fm_memory_type'), [list])
        self.check_param_type(params_in.get( \
            'out_fm_memory_type'), [list])
        self.check_param_type(params_in.get('l1_fusion_type'), [int])
        self.check_param_type(params_in.get( \
            'fusion_type', CONST_VALUE0), [int])
        self.check_param_type( \
            params_in.get('kernel_name', "conv2d_kernel"), [str])
        self.check_param_type(params_in.get( \
            'reserved_ub', CONST_VALUE0), [int])

        # check the length of param
        self.check_param_length(params_in.get('a_shape'), [SHAPE_LENGTH5])
        self.check_param_length(params_in.get('b_shape'), [SHAPE_LENGTH5])
        self.check_param_length(params_in.get('c_shape'), [SHAPE_LENGTH5])
        self.check_param_length(params_in.get('pad'), [SHAPE_LENGTH4])
        self.check_param_length(params_in.get('stride'), [SHAPE_LENGTH2])
        self.check_param_length(params_in.get( \
            'dilation'), [SHAPE_LENGTH2])
        self.check_param_length(params_in.get('fused_coefficient'), \
            [SHAPE_LENGTH3])
        self.check_param_length(params_in.get('fused_channel_wise'), \
            [SHAPE_LENGTH3])

        # check the support range of param
        self.check_support_range(params_in.get('a_dtype'), self.dtype_dict)
        self.check_support_range(params_in.get('b_dtype'), self.dtype_dict)
        self.check_support_range(params_in.get('c_dtype'), self.dtype_dict)
        self.check_support_range(params_in.get('mad_dtype'), \
            self.dtype_dict)
        self.check_support_range(params_in.get('op_type', 'conv2d'), \
            self.op_type_dict)

    def preprocess_info_dict(self, params_in):
        """encode the information of shape to the list of uint32 digit

        Parameters
        ----------
        params_in: dict of params
            include all information of shape

        Returns
        -------
        """
        def encode_memory_type(type_flag, memory_type_list):
            """
            encode the input params to encode_value

            Parameters
            ----------
            memory_type_list: the input memory list

            Returns
            -------
            encode_value: encode value
            """
            # get the length of memory_type_list
            value_length = len(memory_type_list)
            # define the encode table
            encode_table = {EMPTY_MEMORY: 0, \
                            L1_MEMORY: 1,
                            L2_MEMORY: 2,
                            DDR_MEMORY: 3}
            # encode the encode_value
            encode_value = 0
            encode_index = 0
            while value_length:
                # if the type_flag is output, using ternary code
                if type_flag == "out":
                    encode_value += \
                        encode_table[memory_type_list[encode_index]]* \
                        (4**encode_index)
                # if if the type_flag is input, using binary code
                elif type_flag == "in":
                    encode_value += memory_type_list[encode_index]* \
                        (3**encode_index)
                else:
                    raise ValueError("input type_list not support")
                value_length -= 1
                encode_index += 1

            return encode_value

        # get the missing information on interface
        l1_fusion_type = params_in.get('l1_fusion_type', DEFAULT_VALUE)
        # source buffer of input and destination buffer of output
        in_fm_memory_type = params_in.get('in_fm_memory_type', \
            [DDR_MEMORY])
        out_fm_memory_type = params_in.get('out_fm_memory_type', \
            [DDR_MEMORY])
        # process the fusion type
        # if fuison type is depth fusion, then source and destination is DDR,
        # the l1_fusion_type is L1_no_fusion
        # 0 represent L1_depth_fusion; 1 represent L1_breadth_fusion,
        # 2 represent L1_no_fusion; 3 represent L2_fusion
        if l1_fusion_type == DEFAULT_VALUE:
            l1_fusion_type = L1_NO_FUSION
        # 2 represent L2_no_fusion; 3 represent L2_fusion
        if (L2_MEMORY in in_fm_memory_type) or \
            (L2_MEMORY in out_fm_memory_type):
            l2_fusion_type = L2_FUSION
        else:
            l2_fusion_type = L2_NO_FUSION

        # encode the memory type
        in_fm_memory_type_encode = encode_memory_type("in", \
            in_fm_memory_type)
        out_fm_memory_type_encode = encode_memory_type("out", \
            out_fm_memory_type)
        # set the default value of these params
        op_type = params_in.get('op_type', 'conv2d')
        fusion_type = params_in.get('fusion_type', CONST_VALUE0)
        kernel_name = params_in.get('kernel_name', "conv2d_kernel")
        a_dtype = params_in.get('a_dtype', 'float16')
        b_dtype = params_in.get('b_dtype', 'float16')
        c_dtype = params_in.get('c_dtype', 'float16')
        mad_dtype = params_in.get('mad_dtype', 'float16')
        bias_flag = params_in.get('bias_flag', False)
        bias_flag = (1 if bias_flag else 0)

        # the channel align to unit of cube
        _ca0 = params_in["b_shape"][4]
        if _ca0 != C04:
            config = CUBE_MKN[params_in["b_dtype"]]
            _ca0 = config['mac'][1]

        a_shape = params_in.get('a_shape')
        a_shape[1] = (a_shape[1]*a_shape[4] + _ca0 - 1) // _ca0
        a_shape[4] = _ca0
        b_shape = params_in.get('b_shape')
        b_shape[1] = (b_shape[1]*b_shape[4] + _ca0 - 1) // _ca0
        b_shape[4] = _ca0
        c_shape = params_in.get('c_shape')
        c_shape = ([0, 0, 0, 0, 0] if c_shape is None else c_shape)

        # processing fixed-point number
        fused_coefficient = params_in.get('fused_coefficient')
        fused_coefficient = [math.ceil(100*elt) for elt in fused_coefficient]
        fused_channel_wise = params_in.get('fused_channel_wise')
        fused_channel_wise = [math.ceil(100*elt) \
                              for elt in fused_channel_wise]
        reserved_ub = params_in.get('reserved_ub', 0)

        # processed params
        params_in['a_shape'] = a_shape
        params_in['b_shape'] = b_shape
        params_in['c_shape'] = c_shape
        params_in['a_dtype'] = self.dtype_dict.get(a_dtype)
        params_in['b_dtype'] = self.dtype_dict.get(b_dtype)
        params_in['c_dtype'] = self.dtype_dict.get(c_dtype)
        params_in['mad_dtype'] = self.dtype_dict.get(mad_dtype)
        # l1_fusion_type have three states:  0 represent L1_depth_fusion;
        # 1 represent L1_breadth_fusion, 2 represent L1_no_fusion
        params_in['l1_fusion_type'] = l1_fusion_type
        # 2 represent L2_no_fusion; 3 represent L2_fusion
        params_in['l2_fusion_type'] = l2_fusion_type
        # source buffer of input and destination buffer of output
        params_in['in_fm_memory_type'] = in_fm_memory_type_encode
        params_in['out_fm_memory_type'] = out_fm_memory_type_encode
        # set the default value of these params
        params_in['pad'] = params_in.get('pad', [0, 0, 0, 0])
        params_in['stride'] = params_in.get('stride', \
            [CONST_VALUE1, CONST_VALUE1])
        params_in['dilation'] = params_in.get('dilation', \
            [CONST_VALUE1, CONST_VALUE1])
        params_in['group'] = params_in.get('group', CONST_VALUE1)
        params_in['bias_flag'] = bias_flag
        params_in['op_type'] = self.op_type_dict.get(op_type)
        params_in['fusion_type'] = fusion_type
        params_in['kernel_name'] = kernel_name
        # the fused_channel_wise and fused_coefficient are fixed-point number
        # account to two decimal places
        params_in['fused_coefficient'] = fused_coefficient
        params_in['fused_channel_wise'] = fused_channel_wise
        params_in['reserved_ub'] = reserved_ub

    def encode(self, params):
        """encode the information of shape to the list of uint32 digit

        Parameters
        ----------
        params: dict of params
            include all information of shape

        Returns
        -------
        params_encode : list of encoded params
            The encoded params, include uint32 numbers
        """
        # encode the dict to list
        params_encode = list()
        # encode the op_type
        params_encode.append(params.get('op_type'))
        # encode the l1_fusion_type
        params_encode.append(params.get('l1_fusion_type'))
        # encode the l2_fusion_type
        params_encode.append(params.get('l2_fusion_type'))
        # encode the shape(five dimensions)
        self.encode_list(params.get('a_shape'), params_encode)
        self.encode_list(params.get('b_shape'), params_encode)
        self.encode_list(params.get('c_shape'), params_encode)
        # encode the data type
        params_encode.append(params.get('a_dtype'))
        params_encode.append(params.get('b_dtype'))
        params_encode.append(params.get('c_dtype'))
        params_encode.append(params.get('mad_dtype'))
        # encode the pad
        self.encode_list(params.get('pad'), params_encode)
        # encode the stride
        self.encode_list(params.get('stride'), params_encode)
        # encode the dilation
        self.encode_list(params.get('dilation'), params_encode)
        # encode the group and bias_flag
        params_encode.append(params.get('group'))
        params_encode.append(params.get('bias_flag'))
        # encode the in_fm_memory_type and out_fm_memory_type
        params_encode.append(params.get('in_fm_memory_type'))
        params_encode.append(params.get('out_fm_memory_type'))
        # encode the fused_coefficient(three dimensions) and
        # fused_channel_wise is three dimensions
        self.encode_list(params.get('fused_coefficient'), params_encode)
        self.encode_list(params.get('fused_channel_wise'), params_encode)
        # encode the fusion_type
        params_encode.append(params.get('fusion_type'))
        params_encode.append(params.get('reserved_ub'))

        shape_encode = np.array(params_encode, dtype="uint32")

        return tvm.nd.array(shape_encode)
