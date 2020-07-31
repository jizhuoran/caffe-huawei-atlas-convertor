#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

TBE operator param encoder
"""
from te.domain.tiling.op_param_encode.conv2d_params_encoder \
    import Conv2dParamsEncoder

# define the support encoder of operator
SUPPORT_ENCODER_MAP = {
    "conv2d": Conv2dParamsEncoder
}


class ParamsEncoder():
    """
    factory class for conv2d Params Encoder
    """

    def __init__(self, op_type):
        """
        init the specific object
        """
        self.encoder = mapping_op_type(op_type)()

    def encode_array(self, info_dict):
        """
        encode the info_dict

        Parameters
        ----------
        info_dict: the input params

        Returns
        -------
        tvm.nd.array: the NDArray
        """
        self.encoder.check_info_dict(info_dict)
        return self.encoder.encode_array(info_dict)


def mapping_op_type(op_type):
    """
    map the op_type to object of specific class

    Parameters
    ----------
    op_type: the input type of operator

    Returns
    -------
    class_name: the specific class
    """
    if op_type in SUPPORT_ENCODER_MAP.keys():
        return SUPPORT_ENCODER_MAP[op_type]
    else:
        raise TypeError("only support the operator: %s, \
            but the input is %s" % (SUPPORT_ENCODER_MAP.keys(), op_type))
