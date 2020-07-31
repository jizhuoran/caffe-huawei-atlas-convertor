#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0. You may not use this
file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

common function
"""

import re
from functools import reduce as functools_reduce
from te.platform.fusion_manager import fusion_manager

SHAPE_SIZE_LIMIT = 2 ** 31 - 1
SHAPE_SIZE_ZERO = 0
RANK_ZERO = 0
RANK_LIMIT = 8
DIM_LIMIT = 2 ** 31 - 1
ZERO_DIM = 0
# the max len of kernel_name
MAX_KERNEL_NAEM_LEN = 200

REQUIRED_INPUT = "required_input"
OPTION_INPUT = "option_input"
DYNAMIC_INPUT = "dynamic_input"

REQUIRED_OUTPUT = "required_output"
OPTION_OUTPUT = "option_output"
DYNAMIC_OUTPUT = "dynamic_output"

# in proto attr can be a Tensor/BYTES/LIST_TYPE Type, but not te fusion don't support this type
REQUIRED_ATTR_INT = "REQUIRED_ATTR_INT"
REQUIRED_ATTR_FLOAT = "REQUIRED_ATTR_FLOAT"
REQUIRED_ATTR_STR = "REQUIRED_ATTR_STR"
REQUIRED_ATTR_BOOL = "REQUIRED_ATTR_BOOL"
REQUIRED_ATTR_TYPE = "REQUIRED_ATTR_TYPE"
REQUIRED_ATTR_LIST_INT = "REQUIRED_ATTR_LIST_INT"
REQUIRED_ATTR_LIST_FLOAT = "REQUIRED_ATTR_LIST_FLOAT"
REQUIRED_ATTR_LIST_BOOL = "REQUIRED_ATTR_LIST_BOOL"
REQUIRED_ATTR_LIST_LIST_INT = "REQUIRED_ATTR_LIST_LIST_INT"

OPTION_ATTR_INT = "OPTION_ATTR_INT"
OPTION_ATTR_FLOAT = "OPTION_ATTR_FLOAT"
OPTION_ATTR_STR = "OPTION_ATTR_STR"
OPTION_ATTR_BOOL = "OPTION_ATTR_BOOL"
OPTION_ATTR_TYPE = "OPTION_ATTR_TYPE"
OPTION_ATTR_LIST_INT = "OPTION_ATTR_LIST_INT"
OPTION_ATTR_LIST_FLOAT = "OPTION_ATTR_LIST_FLOAT"
OPTION_ATTR_LIST_BOOL = "OPTION_ATTR_LIST_BOOL"
OPTION_ATTR_LIST_LIST_INT = "OPTION_ATTR_LIST_LIST_INT"

KERNEL_NAME = "kernel_name"

OP_ERROR_CODE_000 = 'E80000'
OP_ERROR_CODE_001 = 'E80001'
OP_ERROR_CODE_002 = 'E80002'
OP_ERROR_CODE_003 = 'E80003'
OP_ERROR_CODE_004 = 'E80004'
OP_ERROR_CODE_005 = 'E80005'
OP_ERROR_CODE_006 = 'E80006'
OP_ERROR_CODE_007 = 'E80007'
OP_ERROR_CODE_008 = 'E80008'
OP_ERROR_CODE_009 = 'E80009'
OP_ERROR_CODE_010 = 'E80010'
OP_ERROR_CODE_011 = 'E80011'
OP_ERROR_CODE_012 = 'E80012'
OP_ERROR_CODE_013 = 'E80013'
OP_ERROR_CODE_014 = 'E80014'
OP_ERROR_CODE_015 = 'E80015'
OP_ERROR_CODE_016 = 'E80016'
OP_ERROR_CODE_017 = 'E80017'


class OpParamInfoKey:  # pylint: disable=too-few-public-methods
    """
    Define op params
    """
    def __init__(self):
        pass

    SHAPE = "shape"
    FORMAT = "format"
    ORI_SHAPE = "ori_shape"
    ORI_FORMAT = "ori_format"
    D_TYPE = "dtype"


class TensorFormat:  # pylint: disable=too-few-public-methods
    """
    Define op params
    """
    def __init__(self):
        pass

    ND = "ND"
    NCHW = "NCHW"
    NHWC = "NHWC"
    NDHWC = "NDHWC"
    NCDHW = "NCDHW"

    NC1HWC0 = "NC1HWC0"
    NC1HWC0_C04 = "NC1HWC0_C04"
    NDC1HWC0 = "NDC1HWC0"
    FRACTAL_NZ = "FRACTAL_NZ"

    HWCN = "HWCN"
    DHWCN = "DHWCN"
    FRACTAL_Z = "FRACTAL_Z"
    FRACTAL_Z_C04 = "FRACTAL_Z_C04"
    C1HWNCoC0 = "C1HWNCoC0"
    FRACTAL_Z_3D = "FRACTAL_Z_3D"


ALL_FORMAT_LIST = [TensorFormat.__dict__[d_key]
                   for d_key in TensorFormat.__dict__ if "__" not in d_key]
ALL_DTYPE_LIST = ("int8", "uint8", "int16", "uint16", "int32", "uint32",
                  "int64", "uint64", "float16", "float32", "float64", "bool")
OP_NAME = ""
PARAM_NAME = ""
def check_op_params(*type_args,  # pylint: disable=too-many-locals,too-many-statements
                    **type_kwargs):  # pylint: disable=unused-argument,
    """
    check op params
    """
    input_params = [REQUIRED_INPUT, OPTION_INPUT, DYNAMIC_INPUT]
    output_params = [REQUIRED_OUTPUT, OPTION_OUTPUT, DYNAMIC_OUTPUT]
    required_attr_params = [REQUIRED_ATTR_STR, REQUIRED_ATTR_FLOAT,
                            REQUIRED_ATTR_INT, REQUIRED_ATTR_BOOL,
                            REQUIRED_ATTR_TYPE, REQUIRED_ATTR_LIST_INT,
                            REQUIRED_ATTR_LIST_BOOL, REQUIRED_ATTR_LIST_FLOAT,
                            REQUIRED_ATTR_LIST_LIST_INT]
    list_type_attr = [REQUIRED_ATTR_LIST_BOOL, REQUIRED_ATTR_LIST_INT,
                      REQUIRED_ATTR_LIST_FLOAT, REQUIRED_ATTR_LIST_LIST_INT,
                      OPTION_ATTR_LIST_BOOL, OPTION_ATTR_LIST_INT,
                      OPTION_ATTR_LIST_FLOAT, OPTION_ATTR_LIST_LIST_INT]

    def _check_input_output_key(op_param, param_name, op_name=OP_NAME):
        # check all necessary information(shape, format, ori_shape, ori_format, dtype)
        if not isinstance(op_param, dict):
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_003
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['param_type'] = 'dict'
            error_info['actual_type'] = op_param.__class__.__name__
            raise RuntimeError(error_info, "In op[%s], the parameter[%s]'s type should be [%s],  "
                                           "but actually is [%s]." % (error_info['op_name'],\
                                            error_info['param_name'], error_info['param_type']\
                                            , error_info['actual_type']))
        if OpParamInfoKey.SHAPE not in op_param.keys():
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_004
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['key'] = OpParamInfoKey.SHAPE
            raise RuntimeError(error_info,
                               "In op[%s], the input[%s] does not contain the item[%s]."
                               % (error_info['op_name'], error_info['param_name']\
                                  , error_info['key']))
        if OpParamInfoKey.FORMAT not in op_param.keys():
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_004
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['key'] = OpParamInfoKey.FORMAT
            raise RuntimeError(error_info,
                               "In op[%s], the input[%s] does not contain the item[%s]."
                               % (error_info['op_name'], error_info['param_name']\
                                  , error_info['key']))
        if OpParamInfoKey.ORI_SHAPE not in op_param.keys():
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_004
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['key'] = OpParamInfoKey.ORI_SHAPE
            raise RuntimeError(error_info,
                               "In op[%s], the input[%s] does not contain the item[%s]."
                               % (error_info['op_name'], error_info['param_name']\
                                  , error_info['key']))
        if OpParamInfoKey.ORI_FORMAT not in op_param.keys():
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_004
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['key'] = OpParamInfoKey.ORI_FORMAT
            raise RuntimeError(error_info,
                               "In op[%s], the input[%s] does not contain the item[%s]."
                               % (error_info['op_name'], error_info['param_name']\
                                  , error_info['key']))
        if OpParamInfoKey.D_TYPE not in op_param.keys():
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_004
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['key'] = OpParamInfoKey.D_TYPE
            raise RuntimeError(error_info,
                               "In op[%s], the input[%s] does not contain the item[%s]."
                               % (error_info['op_name'], error_info['param_name']\
                                  , error_info['key']))
    def _check_input_output_dict(op_param, param_name, op_name=OP_NAME):
        _check_input_output_key(op_param, param_name, op_name)

        check_shape(op_param[OpParamInfoKey.SHAPE])
        check_shape(op_param[OpParamInfoKey.ORI_SHAPE])

        if op_param[OpParamInfoKey.FORMAT] not in TensorFormat.__dict__.keys():
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_015
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['excepted_format_list'] = ",".join(ALL_FORMAT_LIST)
            error_info['format'] = op_param[OpParamInfoKey.FORMAT]

            raise RuntimeError(error_info, "In op[%s], the format of input[%s] "
                                           "should be one of [%s], but actually is [%s]."
                               %(error_info['op_name'], error_info['param_name']\
                                 , error_info['excepted_format_list'], error_info['format']))

        if op_param[OpParamInfoKey.ORI_FORMAT] not in TensorFormat.__dict__.keys():
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_014
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['excepted_format_list'] = ",".join(ALL_FORMAT_LIST)
            error_info['format'] = op_param[OpParamInfoKey.ORI_FORMAT]
            raise RuntimeError(error_info,
                               "In op[%s], the ori format of input[%s] should be one of [%s]"
                               ", but actually is [%s]."\
                               %(error_info['op_name'], error_info['param_name']\
                                 , ",".join(ALL_FORMAT_LIST), OpParamInfoKey.ORI_FORMAT))

        if op_param[OpParamInfoKey.D_TYPE] not in ALL_DTYPE_LIST:
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_008
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['excepted_dtype_list'] = ",".join(ALL_DTYPE_LIST)
            error_info['dtype'] = op_param[OpParamInfoKey.D_TYPE]
            raise RuntimeError(error_info,
                               "In op[%s], the parameter[%s]'s dtype should be "
                               "one of [%s], but actually is [%s]." %
                               (error_info['op_name'], error_info['param_name']
                                , error_info['excepted_dtype_list'], error_info['dtype']))

    def _check_input(op_param, param_name, param_type, op_name=OP_NAME):
        if param_type == REQUIRED_INPUT:
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_001
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            if op_param is None:
                raise RuntimeError(error_info, "In op[%s], the mandatory "
                                               "parameter[%s] is missed."\
                                   % (error_info['op_name'], error_info['param_name']))
            _check_input_output_dict(op_param, param_name, op_name)
        elif param_type == OPTION_INPUT:
            if op_param is not None:
                _check_input_output_dict(op_param, param_name, op_name)
        else:
            if not isinstance(op_param, (list, tuple)):
                error_info = {}
                error_info['errCode'] = OP_ERROR_CODE_003
                error_info['op_name'] = op_name
                error_info['param_name'] = param_name
                error_info['param_type'] = "list truple"
                error_info['actual_type'] = op_param.__class__.__name__
                raise RuntimeError(error_info, "In op[%s], the parameter[%s]'s type should be [%s]"
                                               ",  but actually is [%s]."\
                                   % (op_name, param_name, error_info['param_type']\
                                      , error_info['actual_type']))
            if not op_param:
                error_info = {}
                error_info['errCode'] = OP_ERROR_CODE_001
                error_info['op_name'] = op_name
                error_info['param_name'] = param_name
                raise RuntimeError(error_info, "In op[%s], the mandatory parameter[%s]"
                                               " is missed." % (op_name, param_name))
            for one_input in op_param:
                _check_input_output_dict(one_input, param_name, op_name)

    def _check_output(op_param, param_name, param_type, op_name=OP_NAME):
        if param_type == REQUIRED_OUTPUT:
            if op_param is None:
                error_info = {}
                error_info['errCode'] = OP_ERROR_CODE_001
                error_info['op_name'] = op_name
                error_info['param_name'] = param_name
                raise RuntimeError(error_info, "In op[%s], the mandatory parameter[%s]"
                                               " is missed." % (op_name, param_name))

            _check_input_output_dict(op_param, param_name, op_name)
        elif param_type == OPTION_OUTPUT:
            if op_param is not None:
                _check_input_output_dict(op_param, param_name, op_name)
        else:
            if not isinstance(op_param, (list, tuple)):
                error_info = {}
                error_info['errCode'] = OP_ERROR_CODE_003
                error_info['op_name'] = op_name
                error_info['param_name'] = param_name
                error_info['param_type'] = "list tuple"
                error_info['actual_type'] = op_param.__class__.__name__
                raise RuntimeError(error_info, "In op[%s], the parameter[%s]'s "
                                               "type should be [%s],  but actually is [%s]."\
                                   % (op_name, param_name, error_info['param_type']\
                                      , error_info['actual_type']))
            if not op_param:
                error_info = {}
                error_info['errCode'] = OP_ERROR_CODE_001
                error_info['op_name'] = op_name
                error_info['param_name'] = param_name
                raise RuntimeError(error_info, "In op[%s], the mandatory"
                                               " parameter[%s] is missed."\
                                   % (op_name, param_name))
            for one_input in op_param:
                _check_input_output_dict(one_input, param_name, op_name)

    def _check_attr_type(op_param, param_name, py_type, py_type_name, op_name=OP_NAME):
        if not isinstance(op_param, py_type):
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_003
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['param_type'] = str(py_type)
            error_info['actual_type'] = op_param.__class__.__name__
            raise RuntimeError(error_info,
                               "In op[%s], the parameter[%s]'s type should be [%s],"
                               " but actually is [%s]."
                               % (error_info['op_name'], error_info['param_name']\
                                  , error_info['param_type'], error_info['actual_type']))

    def _check_list_attr_element(op_param, param_name, py_type, py_type_name, op_name=OP_NAME):
        if not isinstance(op_param, py_type):
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_003
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['param_type'] = str(py_type)
            error_info['actual_type'] = op_param.__class__.__name__
            raise RuntimeError(error_info,
                               "In op[%s], the parameter[%s]'s type should be [%s],"
                               " but actually is [%s]."
                               % (error_info['op_name'], error_info['param_name']\
                                  , error_info['param_type'], error_info['actual_type']))

    def _check_list_attr(op_param, param_name, param_type, op_name=OP_NAME):
        if not isinstance(op_param, (list, tuple)):
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_003
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            error_info['param_type'] = "list tuple"
            error_info['actual_type'] = op_param.__class__.__name__
            raise RuntimeError(error_info,
                               "In op[%s], the parameter[%s]'s type should be [%s],"
                               "  but actually is [%s]."\
                               % (error_info['op_name'], error_info['param_name']\
                                  , error_info['param_type'], error_info['actual_type']))

        if param_type in [REQUIRED_ATTR_LIST_BOOL, OPTION_ATTR_LIST_BOOL]:
            for one_attr in op_param:
                _check_list_attr_element(one_attr, param_name, bool, "bool", op_name)

        if param_type in [REQUIRED_ATTR_LIST_INT, OPTION_ATTR_LIST_INT]:
            for one_attr in op_param:
                _check_list_attr_element(one_attr, param_name, int, "int", op_name)

        if param_type in [REQUIRED_ATTR_LIST_FLOAT, OPTION_ATTR_LIST_FLOAT]:
            for one_attr in op_param:
                _check_list_attr_element(one_attr, param_name, float, "float", op_name)

        if param_type in [REQUIRED_ATTR_LIST_LIST_INT, OPTION_ATTR_LIST_LIST_INT]:
            for one_attr in op_param:
                if not isinstance(one_attr, (list, tuple)):
                    error_info = {}
                    error_info['errCode'] = OP_ERROR_CODE_003
                    error_info['op_name'] = op_name
                    error_info['param_name'] = param_name
                    error_info['param_type'] = "list tuple"
                    error_info['actual_type'] = op_param.__class__.__name__
                    raise RuntimeError(error_info,
                                       "In op[%s], the parameter[%s]'s type should be [%s],"
                                       " but actually is [%s]."\
                                       % (error_info['op_name'], error_info['param_name']\
                                          , error_info['param_type'], error_info['actual_type']))

                for ele in one_attr:
                    _check_list_attr_element(ele, param_name, int, "int", op_name)

    def _check_attr(op_param, param_name, param_type, op_name=OP_NAME):
        if op_param is None and param_type in required_attr_params:

            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_001
            error_info['op_name'] = op_name
            error_info['param_name'] = param_name
            raise RuntimeError(error_info,
                               "In op[%s], the mandatory parameter[%s] is missed."\
                               % (op_name, param_name))
        if not op_param:
            return

        if param_type in [REQUIRED_ATTR_INT, OPTION_ATTR_INT]:
            _check_attr_type(op_param, param_name, int, "int", op_name)

        if param_type in [REQUIRED_ATTR_FLOAT, OPTION_ATTR_FLOAT]:
            _check_attr_type(op_param, param_name, float, "float", op_name)

        if param_type in [REQUIRED_ATTR_STR, OPTION_ATTR_STR]:
            _check_attr_type(op_param, param_name, str, "string", op_name)

        if param_type in [REQUIRED_ATTR_BOOL, OPTION_ATTR_BOOL]:
            _check_attr_type(op_param, param_name, bool, "bool", op_name)

        if param_type in [REQUIRED_ATTR_TYPE, OPTION_ATTR_TYPE]:
            if op_param not in ALL_DTYPE_LIST:

                error_info = {}
                error_info['errCode'] = OP_ERROR_CODE_003
                error_info['op_name'] = op_name
                error_info['param_name'] = param_name
                error_info['param_type'] = " ".join(ALL_DTYPE_LIST)
                error_info['actual_type'] = op_param.__class__.__name__
                raise RuntimeError(error_info,
                                   "In op[%s], the parameter[%s]'s dtype should"
                                   " be one of [%s], but actually is [%s]."\
                                   % (error_info['op_name'], error_info['param_name']\
                                , error_info['param_type'], error_info['actual_type']))

        if param_type in list_type_attr:
            _check_list_attr(op_param, param_name, param_type, op_name)

    def _check_kernel_name(kernel_name):
        """
        check kernel_name
        """
        if not isinstance(kernel_name, str):
            raise RuntimeError("kernel name must be string type, actual is not.")

        if len(kernel_name) > MAX_KERNEL_NAEM_LEN:
            raise RuntimeError("kernel_name len must be less than %d, "
                               "but got %d" % (MAX_KERNEL_NAEM_LEN, len(kernel_name)))

        pattern = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
        if not pattern.match(kernel_name):
            raise RuntimeError(
                "kernel_name can only contain letters, numbers and underscores, and begin with "
                "underscores or letters")

    def _check_one_op_param(op_param, param_name, param_type, op_name=OP_NAME):

        if param_type in input_params:
            _check_input(op_param, param_name, param_type, op_name)
        elif param_type in output_params:
            _check_output(op_param, param_name, param_type, op_name)
        elif param_type == KERNEL_NAME:
            if op_param is None:
                return
            _check_kernel_name(op_param)
        else: # else is attr_params:
            _check_attr(op_param, param_name, param_type, op_name)

    def _out_wrapper(func):
        formal_parameter = func.__code__.co_varnames
        formal_parameter_list = list(zip(formal_parameter, type_args))

        def _in_wrapper(*args, **kwargs):
            for i, one_args in enumerate(args):
                op_name = func.__name__
                _check_one_op_param(one_args, formal_parameter_list[i][0],
                                    formal_parameter_list[i][1], op_name)
            for arg_key in kwargs:
                for name_type in formal_parameter_list:
                    if arg_key == name_type[0]:
                        _check_one_op_param(kwargs[arg_key], arg_key, name_type[1], op_name)
                        break

            return func(*args, **kwargs)

        return _in_wrapper

    return _out_wrapper


def check_shape(shape, min_dim=0, max_dim=DIM_LIMIT,  # pylint: disable=too-many-arguments
                min_rank=0, max_rank=RANK_LIMIT,  # pylint: disable=too-many-arguments
                min_size=0, max_size=SHAPE_SIZE_LIMIT):  # pylint: disable=too-many-arguments
    """
    check shape size
    """
    if not isinstance(shape, (tuple, list)):

        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_003
        error_info['op_name'] = OP_NAME
        error_info['param_name'] = PARAM_NAME
        error_info['param_type'] = "list tuple"
        error_info['actual_type'] = shape.__class__.__name__
        raise RuntimeError(error_info,
                           "In op, the parameter's type should be [%s], "
                           "but actually is [%s]." %(error_info['param_type']\
                            , error_info['actual_type']))

    for dim in shape:
        if not isinstance(dim, int):
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_003
            error_info['op_name'] = OP_NAME
            error_info['param_name'] = 'dim'
            error_info['param_type'] = 'int'
            error_info['actual_type'] = dim.__class__.__name__
            raise RuntimeError(error_info,
                               "In op, the parameter's type should be [%s],  "
                               "but actually is [%s]." %(error_info['param_type']\
                                , error_info['actual_type']))

    if len(shape) < min_rank or len(shape) > max_rank:
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_012
        error_info['op_name'] = OP_NAME
        error_info['param_name'] = PARAM_NAME
        error_info['min_value'] = min_rank
        error_info['max_value'] = max_rank
        error_info['real_value'] = len(shape)
        raise RuntimeError(error_info,
                           "In op, the num of dimensions of input should be in"
                           " the range of [%s, %s], but actually is [%s]."\
                           % (min_rank, max_rank, len(shape)))

    for _, dim in enumerate(shape):
        if dim < min_dim or dim > max_dim:
            raise RuntimeError(
                "The axis value must be more than %d and less than %d, "
                "actual input is %d" % (min_dim, max_dim, dim))
    if shape:
        shape_size = functools_reduce(lambda x, y: x * y, shape[:])
    else:
        shape_size = 1
    if shape_size < min_size or shape_size > max_size:

        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_011
        error_info['op_name'] = OP_NAME
        error_info['param_name'] = PARAM_NAME
        error_info['max_value'] = max_size
        error_info['real_value'] = shape_size
        raise RuntimeError(error_info,
                           "In op, the shape size(product of all dimensions) of "
                           "input should less than [%s], but actually is [%s]."\
                           % (max_size, shape_size))


def check_dtype(dtype, check_list=ALL_DTYPE_LIST):
    """
    The common check rule for tensor dtype
    """
    if dtype is None:
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_007
        raise RuntimeError(error_info, "In op, the input's dtype could not be none.")
    if dtype.lower() not in check_list:
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_008
        error_info['excepted_dtype_list'] = ",".join(check_list)
        error_info['dtype'] = dtype.lower()
        raise RuntimeError(error_info, "In op, the parameter's dtype should be one of [%s]"
                                       ", but actually is [%s]."\
                           % (error_info['excepted_dtype_list'], error_info['dtype']))


def check_format(data_format, check_list=ALL_FORMAT_LIST):
    """
    The common check rule for tensor dtype
    """

    if data_format is None:
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_017
        raise RuntimeError(error_info, "In op, the input's format could not be none")

    if data_format not in check_list:
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_015
        error_info['excepted_format_list'] = ",".join(check_list)
        error_info['format'] = data_format.lower()
        raise RuntimeError(error_info, "In op, the format of input should "
                                       "be one of [%s], but actually is [%s]."\
                           %(error_info['excepted_format_list'], error_info['format']))


def squeeze_shape(shape):
    """
    squeeze shape
    """
    squeezed_shape = [i for i in shape if i > 1]
    if not squeezed_shape:
        squeezed_shape = [1]

    return squeezed_shape

def wrap_axes_to_positive(axes, rank):
    """
    wrap axis to positive
    """
    if isinstance(axes, (tuple, list)):
        local_axes = axes
    else:
        local_axes = [axes]
    res_axes = []
    for axis in local_axes:
        if rank <= axis or axis < -rank:
            raise RuntimeError("Axis must between [-%d, %d)." % (rank, rank))
        if axis < 0:
            laxis = axis + rank
        else:
            laxis = axis
        res_axes.append(laxis)

    return res_axes


def refine_shape_axes(shape, axes):
    """
    refine shape and axes for reduce ops, fused reduced axes, and fused not reduced axes
    result is a tuple of (shape, axes)
    for example:
        input: shape is (2,3,4,5,6), axes is (1, -3)
        output: (2, 12, 30), (1,)

    Parameters
    ----------
    shape : shape which need refine

    axes : axes which need refine

    Returns
    -------
    shape : list
        refined shape

    axes : list
        refined axes

    """
    if len(shape) == 1:
        return shape, axes
    wrapped_axes = wrap_axes_to_positive(axes, len(shape))
    wrapped_axes = sorted(wrapped_axes)
    refined_axes = []
    reduce_flag = -1
    refined_shape = []
    for idx, dim in enumerate(shape):
        if dim == 1:
            # dim is one, not need reduce skip
            continue
        tmp_flag = 1 if idx in wrapped_axes else 0
        if reduce_flag == 1 and tmp_flag == 1:
            # continues reduce
            refined_shape[-1] *= dim
        elif reduce_flag == 0 and tmp_flag == 0:
            # continues no reduce
            refined_shape[-1] *= dim
        else:
            refined_shape.append(dim)
            if tmp_flag == 1:
                refined_axes.append(idx)
            reduce_flag = tmp_flag

    if not refined_shape:
        refined_shape.append(1)

    return refined_shape, refined_axes


def broadcast_shapes(shape1, shape2, op_name=OP_NAME, param_name_input1='', param_name_input2=''):
    """
    two input shapes produce three output shape
    """
    shape1 = list(shape1)
    shape2 = list(shape2)
    flag = 0
    if len(shape1) < len(shape2):
        shape1, shape2 = shape2, shape1
        flag = 1

    output_shape_len = len(shape1)
    dec = output_shape_len - len(shape2)
    for i in range(dec):
        shape2 = [1] + shape2

    out_shape = []
    for i in range(output_shape_len):
        if (shape1[i] != shape2[i]) and (shape1[i] != 1) and (shape2[i] != 1):
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_013
            error_info['op_name'] = op_name
            error_info['input1_name'] = param_name_input1
            error_info['input2_name'] = param_name_input2
            error_info['input1_shape'] = ",".join(str(i) for i in shape1)
            error_info['input2_shape'] = ",".join(str(i) for i in shape2)
            raise RuntimeError(error_info, "In op[%s], the inputs[%s][%s] could "
                                           "not be broadcast together with shapes[%s][%s]."
                               % (op_name, param_name_input1, param_name_input2,\
                                  error_info['input1_shape'], error_info['input2_shape']))
        out_shape.append(shape1[i] if shape1[i] > shape2[i] else shape2[i])

    if flag == 1:
        shape1, shape2 = shape2, shape1

    return shape1, shape2, out_shape


def refine_shapes_for_broadcast(shape1, shape2):
    """
    Fusing the axes for the input shapes
    """
    def _delete_one(shape1, shape2):
        # delete 1 when both 1
        shape1_new = []
        shape2_new = []
        for i, (shape1_i, shape2_i) in enumerate(zip(shape1, shape2)):
            if (shape1_i != shape2_i) or \
                    (shape1_i == shape2_i and shape1_i != 1):
                shape1_new.append(shape1[i])
                shape2_new.append(shape2[i])
        if shape1_new == [] and shape2_new == []:
            shape1_new = [1]
            shape2_new = [1]
        return shape1_new, shape2_new

    if fusion_manager.get_build_cfg() == "disable":
        return shape1, shape2

    shape1 = list(shape1)
    shape2 = list(shape2)
    flag = 0
    if len(shape1) < len(shape2):
        shape1, shape2 = shape2, shape1
        flag = 1

    output_shape_len = len(shape1)
    dec = output_shape_len - len(shape2)
    for i in range(dec):
        shape2 = [1] + shape2

    shape1, shape2 = _delete_one(shape1, shape2)

    fused_shape1 = []
    fused_shape2 = []
    fused_shape1.append(shape1[0])
    fused_shape2.append(shape2[0])
    j = 0

    for i, (shape1_i, shape2_i) in enumerate(zip(shape1, shape2)):
        if i == 0:
            pass
        elif shape1_i == shape2_i and shape1[i - 1] == shape2[i - 1]:
            fused_shape1[j] *= shape1[i]
            fused_shape2[j] *= shape2[i]
        elif shape1_i != shape2_i and shape1[i - 1] != shape2[i - 1] \
                and (shape1_i == shape1[i - 1] or shape2_i == shape2[i - 1]):
            fused_shape1[j] *= shape1[i]
            fused_shape2[j] *= shape2[i]
        else:
            j += 1
            if i != 0:
                fused_shape1.append(shape1[i])
                fused_shape2.append(shape2[i])

    if flag == 1:
        fused_shape1, fused_shape2 = fused_shape2, fused_shape1

    return fused_shape1, fused_shape2
