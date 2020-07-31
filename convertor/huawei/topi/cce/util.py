#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this
file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

common function
"""
# pylint: disable=import-error, unidiomatic-typecheck,no-else-return,consider-using-enumerate
import re
from functools import reduce
from te import tvm
from te.platform import cce_conf
from te import platform as cce

# the min dim of shaoe
DEFAULT_MIN_SHAPE_DIM = 1
# the max dim of shape
DEFAULT_MAX_SHAPE_DIM = 8
# the max num of each axis of shape
DEFAULT_MAX_SHAPE_NUM = 200000000
# the max num of all reduce axis of shape
MAX_REDUCE_SHAPE_NUM = 200000000
# the max len of kernel_name
MAX_KERNEL_NAEM_LEN = 200
# cloud product
VERSION_CLOUD = "cloud"
# mini product
VERSION_MINI = "mini"
# smallhisi product
VERSION_SHISI = "smallhisi"
VERSION_MINI_1951 = "mini_1951"
VERSION_MINI_1951M = "mini_1951m"
# the max size of SHAPE, value = 2^31
SHAPE_SIZE_LIMIT = 2147483648


def check_input_type_dict(input_dict, input_key, input_name):
    """
    check input parameter type for new type: dict
    rule1: key of input_dict should be in the input_key
    rule2: type of input_dict[shape] should be in (list, tuple), if have shape
    rule3: type of input_dict[dtype] should be in (str), if have dtype

    Parameters
    ----------
    input_dict: dict
        input_dict
    input_key: list or tuple
        all input key list, the key of input must in input_key
    input_name: str
        input param name, only used for error print

    Returns
    -------
    None
    """

    def _check_input_type(input_key, input_type):
        if not isinstance(input_dict[input_key], input_type):
            raise RuntimeError(
                "Input parameter error, please check input parameter!")

    for key in input_dict.keys():
        if key not in input_key:
            raise RuntimeError(
                "Input parameter value must have property, please check!")

        # check shape's type of input_dict, if have shape
        if key == "shape":
            _check_input_type(key, (list, tuple))

        # check dtype's type of input_dict, if have dtype
        if key == "dtype":
            _check_input_type(key, (str,))


# pylint: disable=unused-argument
def check_input_type(*type_args, **type_kwargs):
    """
    check input parameter type
    """

    def out_wrapper(func):
        """
        out_wrapper

        :param func: func
        :return: None
        """
        formal_parameter = func.__code__.co_varnames
        formal_parameter_list = list(zip(formal_parameter, type_args))

        def in_wrapper(*args, **kwargs):
            """
            in_wrapper
            :param args: args
            :param kwargs: kwargs
            :return: None
            """
            for i in range(len(args)):
                # add for new input dict, if dict, will check shape and dtype
                if isinstance(args[i], dict):
                    check_input_type_dict(args[i], args[i].keys(),
                                          formal_parameter_list[i][0])
                if not isinstance(args[i], formal_parameter_list[i][1]):
                    raise RuntimeError(
                        "Input parameter type error, please check!")
            for i in kwargs:
                for j in formal_parameter_list:
                    if i in j:
                        if not isinstance(kwargs[i], j[1]):
                            raise RuntimeError(
                                "Input parameter type error, please check!")
                        break
            return func(*args, **kwargs)

        return in_wrapper

    return out_wrapper


def check_dtype_rule(dtype, check_list):
    """
    The common check rule for tensor dtype
    """
    if dtype is None:
        raise RuntimeError("dtype is None")

    if dtype.lower() not in check_list:
        raise RuntimeError("The data type is not supported."
                           "Please check the data type!")


def check_shape_rule(shape, min_dim=None, max_dim=None, max_shape_num=None):
    """
    The common check rule for tensor shape
    """
    if min_dim is None:
        min_dim = DEFAULT_MIN_SHAPE_DIM

    if max_dim is None:
        max_dim = DEFAULT_MAX_SHAPE_DIM

    if max_shape_num is None:
        max_shape_num = DEFAULT_MAX_SHAPE_NUM

    if not isinstance(shape, (tuple, list)):
        raise RuntimeError(
            "Wrong type of input shape,"
            "the shape must be type of tuple or list!")

    if len(shape) < min_dim or len(shape) > max_dim:
        raise RuntimeError(
            "The ndim of input must more than %d and less than %d"
            ", actual input is %d" % (min_dim, max_dim, len(shape)))

    for i in range(len(shape)):
        if type(shape[i]) != int or shape[i] <= 0 or shape[i] > max_shape_num:
            raise RuntimeError(
                "The type of axis must be positive int and value more than"
                "0 and less than max shape!!")


def check_reduce_shape_rule(shape):
    """
    check the shape of reduce axis must be less than MAX_REDUCE_SHAPE_NUM
    :param shape: inout shape
    """
    # the shape of reduce axis must be less than MAX_REDUCE_SHAPE_NUM
    # reduce_shape_num = 1

    from functools import reduce
    product = reduce(lambda x, y: x * y, shape[:])  # product of all dimension

    if product > MAX_REDUCE_SHAPE_NUM:
        raise RuntimeError(
            "The all reduce axis num of shape must be less than %d, actual input is %d" % (
                MAX_REDUCE_SHAPE_NUM, product))


def check_reduce_need_refine(shape, reduce_axis):
    """
    # if the reduce axis correspond to shape[axis] is 1,
    we can not refine the shape,or the reduce axis will be wrong
    shape : shape of data

    reduce_axis : list, tuple or int  axis want to reduce

    :return: True or False
    """

    # if the reduce axis correspond to shape[axis] is 1,
    # we can not refine the shape,or the reduce axis will be wrong
    if hasattr(reduce_axis, 'index'):
        for i in reduce_axis:
            if shape[i] == 1:
                return False
    else:
        if shape[reduce_axis] == 1:
            return False

    return True


# pylint: disable=too-many-branches
def shape_refine(shape, reduce_axis=None):
    """
    refine shape to drop 1 in shape according to reduce axis,
    if input is just shape, result is shape, and if inputs are shape and axis,
    result is a tuple of (shape, axis)

    Parameters
    ----------
    shape : shape of data

    reduce_axis : list, tuple or int
        axis want to reduce

    keepdims: if keepdims = True, we should not refine the shape

    Returns
    -------
    shape : list
        refined shape

    reduce_axis : list
        if input parameters send reduce axis, this will be the output.
        if all the reduce axis is illegal like the length of reduce axis is 1,
        a empty list([]) will be returned.

    """

    def __refine_shape_no_reduce(shape_local):
        refined_shape = [i for i in shape_local if i > 1]
        if not refined_shape:
            refined_shape = [1]
        return refined_shape

    res_shape = []
    res_reduce_axis = []
    if reduce_axis is not None:
        # if the reduce axis correspond to shape[axis] is 1,
        # we can not refine the shape,or the reduce axis will be wrong
        if not check_reduce_need_refine(shape, reduce_axis):

            if hasattr(reduce_axis, 'index'):
                return shape, reduce_axis
            else:
                return shape, [reduce_axis]

        if isinstance(reduce_axis, (tuple, list)):
            res_reduce_axis = reduce_axis[:]
        else:
            res_reduce_axis = [reduce_axis]
        res_reduce_axis = sorted(refine_axis(reduce_axis, shape))
        if not res_reduce_axis:
            return (__refine_shape_no_reduce(shape), [])
        res_shape = shape[:]
        refined_shape = []
        count = 0
        for i in res_shape:
            if i > 1:
                refined_shape.append(i)
                count += 1
            else:
                for j in range(len(res_reduce_axis)):
                    if res_reduce_axis[j] > count:
                        res_reduce_axis[j] -= 1

        return refined_shape, res_reduce_axis

    else:
        return __refine_shape_no_reduce(shape)


def refine_axis(axis, shape):
    """
    refine axis

    Parameters
    ----------
    axis :
        axis want to reduce

    shape : shape of data

    Returns
    -------
    res_reduce_axis : list
        refined axis
    """
    if isinstance(axis, (tuple, list)):
        local_axis = axis
    else:
        local_axis = [axis]
    res_axis = []
    shape_len = len(shape)
    for i in local_axis:
        if i < 0:
            laxis = shape_len + i
        else:
            laxis = i
        if (laxis >= shape_len) or (laxis < 0):
            raise RuntimeError("wrong axis.")
        res_axis.append(laxis)
    res_reduce_axis = []
    for i in res_axis:
        if shape[i] > 1:
            res_reduce_axis.append(i)
    return res_reduce_axis


def get_divisors(num):
    """
    compute the divisors of num

    Parameters
    ----------
    num: Inpute number

    Returns
    -------
    divisors: List
    """
    divisors = []
    tmp_var = num
    while 0 < tmp_var <= num:
        if num % tmp_var == 0:
            divisors.append(tmp_var)
        tmp_var -= 1
    divisors.reverse()
    return divisors


def _axis_value_type_check(shape_len, value):
    """
    Check the value of the axis
    """
    if type(value) != int:
        raise RuntimeError("type of axis value should be int")
    if value >= shape_len or value < -shape_len:
        raise RuntimeError("input axis is out of range, axis value can be from %d to %d" % (
            -shape_len, shape_len - 1))
    if value < 0:
        value = shape_len + value
    return value


def axis_check(shape_len, axis):
    """
    Check the value of axis and return the sorted axis
    """
    if not hasattr(axis, 'index'):
        axis = _axis_value_type_check(shape_len, axis)
        return axis
    else:
        for i in range(len(axis)):
            axis[i] = _axis_value_type_check(shape_len, axis[i])

    axis = list(set(axis))
    axis.sort()
    return axis


def simplify_axis_shape(shape, axis):
    """
    simplify the shape and aixs
    """
    axis1 = []
    shape1 = []
    merge_num = 0
    length = shape[0]

    for i in range(len(axis)):
        if i == 0:
            length = shape[axis[0]]
            axis1.append(axis[0])
        else:
            if axis[i] - axis[i - 1] == 1:
                length = length*shape[axis[i]]
                merge_num = merge_num + 1
            else:
                shape1.append(length)
                for j in range(axis[i - 1], axis[i] - 1):
                    shape1.append(shape[j + 1])
                axis1.append(axis[i] - merge_num)
                length = shape[axis[i]]
    shape1.append(length)
    if axis1 == []:
        axis1 = [0]
    else:
        shape1 = list(shape[:axis[0]]) + shape1 + list(shape[axis[-1] + 1:])

    shape_final = []
    axis_final = []
    axis_fuse_sum = 0
    pre_axis = -1
    for axes in axis1:
        shape_noreduce = shape1[pre_axis + 1 : axes]
        if len(shape_noreduce) > 1:
            shape_final.append(reduce(lambda x, y:x*y, shape_noreduce))
        else:
            shape_final += shape_noreduce

        if len(shape_noreduce) > 0:
           axis_fuse_sum += len(shape_noreduce) - 1
        axis_final.append(axes - axis_fuse_sum)
        shape_final.append(shape1[axes])
        pre_axis = axes

    shape_noreduce = shape1[pre_axis + 1:]
    if len(shape_noreduce) > 1:
        shape_final.append(reduce(lambda x, y:x*y, shape_noreduce))
    else:
        shape_final += shape_noreduce

    return shape_final, axis_final


def produce_shapes(shape1, shape2):
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
            raise RuntimeError("input shapes not match!")
        out_shape.append(shape1[i] if shape1[i] > shape2[i] else shape2[i])

    if flag == 1:
        shape1, shape2 = shape2, shape1

    return shape1, shape2, out_shape


def create_param_ptr(value_list, p_type, ptr_name):
    """ alloc a ptr to store value_list"""

    def _select_from_list(_i, _list):
        ret = _list[0]
        for j in range(0, len(_list)):
            ret = tvm.select(_i == j, _list[j], ret)
        return ret

    tvm_var_list = [tvm.const(value, dtype=p_type) for value in value_list]
    p_ptr = tvm.compute((len(tvm_var_list),), lambda i: _select_from_list(i, tvm_var_list),
                        name=ptr_name)
    return p_ptr


def get_device_api_dtype(dtype):
    """
    get_device_api_dtype

    Parameters
    ----------
    dtype: type

    Returns
    -------
    number of type
    """
    device_api_dtype_maps = {"float32": 0, "float16": 1, "int32": 2, "double": 3}
    return device_api_dtype_maps[dtype]


def is_scalar(shape):
    """
    verify that tensor is scalar
    ----------
    shape: shape of data

    Returns
    -------
    True or False
    """
    if isinstance(shape, (list, tuple)):
        if len(shape) == 1 and shape[0] == 1:
            return True
    return False


def scalar2tensor_one(shape):
    """
    if the input_shape is [],convert the input_shape to [1]
    ----------
    shape: shape of input tensor

    Returns
    -------
    list:[1]
    """
    if isinstance(shape, (list, tuple)):
        if not shape:
            return [1]
    return shape


def check_shape_size(shape, limit):
    """
    check shape size for operator
    ----------
    shape: shape of data

    limit: limit of the product of all dimension

    Returns
    -------
    None
    """
    from functools import reduce
    product = reduce(lambda x, y: x * y, shape[:])  # product of all dimension

    if product > limit:
        raise RuntimeError("the shape is too large to calculate")

    return product


def compare_tensor_dict_key(dict1, dict2, dict_key):
    """
    compare the key value between dict1 and dict2,
    the value is not equal, will raise error

    Parameters
    ----------
    dict1: dict
        input dict1
    dict2: dict
        input dict2
    dict_key: str
        the key that will be compare

    Returns
    -------
    None
    """
    if not isinstance(dict1, dict):
        raise RuntimeError("the input dict1 is not dict")
    if not isinstance(dict2, dict):
        raise RuntimeError("the input dict2 is not dict")

    if dict_key not in dict1.keys():
        raise RuntimeError("There is no value for this input type,"
                           "please check the input!")
    if dict_key not in dict2.keys():
        raise RuntimeError("There is no value for this input type,"
                           "please check the input")

    value1 = dict1.get(dict_key)
    value2 = dict2.get(dict_key)

    if isinstance(value1, (list, tuple)):
        value1 = list(value1)
    if isinstance(value2, (list, tuple)):
        value2 = list(value2)

    if not isinstance(value1, type(value2)):
        raise RuntimeError("The two input types are inconsistent!."
                           "The input types must be the same")
    if isinstance(value1, (str,)):
        value_cmp_1 = value1.lower()
        value_cmp_2 = value2.lower()
        if value1.lower() != value2.lower():
            raise RuntimeError("Input one and input two are not equal!")
    elif isinstance(value1, (list, tuple,)):
        if value1 != value2:
            raise RuntimeError("Input one and input two are not equal!")


def axis_transfrom_5d(axis, data_format):
    """
    4d format axis to 5d mapping
    """
    if data_format == "NCHW":
        if axis < 0:
            axis = axis - 1
    elif data_format == "NHWC":
        if axis == -4:
            axis = -5
        elif axis == -1:
            axis = -4
        elif axis == 1:
            axis = 2
        elif axis == 2:
            axis = 3
        elif axis == 3:
            axis = 1
    return axis


def check_tensor_shape_size(shape):
    """
    check shape size for operator
    In order not to affect the old operators,create a similar function

    Parameters
    ----------
    shape: list or tuple
        shape of data

    Returns
    -------
    None
    """
    shape_size = check_shape_size(shape, SHAPE_SIZE_LIMIT)
    return shape_size


def check_kernel_name(kernel_name):
    """
    check kernel_name
    ----------
    kernel_name: str or None

    Returns
    -------
    None
    """
    if kernel_name is None:
        return

    if not isinstance(kernel_name, str):
        try:
            kernel_name = str(kernel_name)
        except ValueError:
            raise ValueError(
                "kernel name input error! It must be string or None")

    if len(kernel_name) > MAX_KERNEL_NAEM_LEN:
        raise ValueError(
            "kernel_name len must be less than %d, "
            "but got %d" % (MAX_KERNEL_NAEM_LEN, len(kernel_name)))

    pattern = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
    if not pattern.match(kernel_name):
        raise RuntimeError("kernel_name can only contain letters, numbers and underscores, "
                           "and begin with underscores or letters")


def get_product_version():
    """
    get product version
    ----------

    Returns
    -------
    cloud: cloud product
    mini: mini product
    """
    soc_version = cce_conf.get_soc_spec("SOC_VERSION")
    if soc_version in ["Ascend310"]:
        version = VERSION_MINI
    elif soc_version in ["Ascend910"]:
        version = VERSION_CLOUD
    elif soc_version in ["Hi3796CV300ES", "Hi3796CV300CS"]:
        version = VERSION_SHISI
    elif soc_version in ["Ascend610"]:
        version = VERSION_MINI_1951
    elif soc_version in ["Ascend620"]:
        version = VERSION_MINI_1951M
    else:
        raise RuntimeError("Current product version is wrong!Please check!!")

    return version


def is_v200_version():
    """
    check if 1951/1951m version
    ----------

    Returns
    -------
    True: 1951 version
    False: Other version
    """
    version = get_product_version()
    if version in (VERSION_MINI_1951, VERSION_MINI_1951M):
        return True
    return False


def is_lhisi_version():
    """
    check if 3796ES version
    -------

    Returns
    -------
    True: 3796ES version
    False: Other version
    """
    version = get_product_version()
    if version == VERSION_SHISI:
        return True
    return False


def is_cloud_version():
    """
    check if cloud-1980 version
    ----------

    Returns
    -------
    True: cloud version
    False: Other version
    """
    version = get_product_version()
    if version == VERSION_CLOUD:
        return True
    return False


def is_mini_version():
    """
    check if mini version
    -------

    Returns
    -------
    True: mini version
    False: Other version
    """
    version = get_product_version()
    if version == VERSION_MINI:
        return True
    return False

def is_mini_or_lhisi_version():
    """
    check if mini or lhisi version
    -------

    Returns
    -------
    True: mini version or lhisi
    False: Other version
    """
    version = get_product_version()
    if version in (VERSION_MINI, VERSION_SHISI):
        return True
    return False


def check_and_init_5hdc_reduce_support(input_tensor, axis, kernel_name):
    """5HD Special param for 5hd schedule"""
    if "format" in input_tensor and input_tensor["format"] == "NC1HWC0" and \
            1 in axis and 4 in axis and input_tensor["dtype"] == "float16":
        if "ori_shape" in input_tensor and "ori_format" in input_tensor:
            cce.cce_emitinsn_params.cceEmitParamsIns. \
                insert_param_with_penetration("5HDOriShape_" + kernel_name,
                                              input_tensor["ori_shape"])
            cce.cce_emitinsn_params.cceEmitParamsIns. \
                insert_param_with_penetration("5HDOriFormat_" + kernel_name,
                                              input_tensor["ori_format"])
            return True
        else:
            raise RuntimeError("Original shape needed for 5HD reduce")
    return False
