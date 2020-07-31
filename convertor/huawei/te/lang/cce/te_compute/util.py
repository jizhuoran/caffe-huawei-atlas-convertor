#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright 2018 Huawei Technologies Co., Ltd
"""
# pylint: disable=import-error
from decorator import decorator
from te import platform as cceconf
from te import tvm
from te.platform import intrinsic_check_support
from te.platform import get_soc_spec

# Save op's output dtype, when first call the template api,we will save the dtype.
# Before auto scheduling,get the dtype and convert the res tensor to this dtype,
# and set the dtype to None.

DTYPE_MAP = {
    "float32": "f32",
    "float16": "f16",
    "int8": "s8",
    "uint8": "u8",
    "int32": "s32",
}

DSL_CHECK_SUPPORT_MAP = {
    "broadcast": {
        "AllSoc": ("float16", "float32", "int32", "int16", "uint16", \
                   "int8", "uint8"),
        "Ascend310": ("float16", "float32", "int32", "int16", "uint16", \
                      "int8", "uint8"),
        "Ascend910": ("float16", "float32", "int32", "int16", "uint16", \
                      "int8", "uint8"),
        "Ascend610": ("float16", "float32", "int32", "int16", "uint16", \
                      "int8", "uint8"),
        "Ascend620": ("float16", "float32", "int32", "int16", "uint16", \
                      "int8", "uint8"),
        "Hi3796CV300ES": ("float16", "float32", "int32", "int16", "uint16", \
                          "int8", "uint8"),
    },

    # segment
    "unsorted_segment_sum": {
        "AllSoc": ("float16", "float32", "int32"),
        "Ascend310": ("float16", "float32", "int32"),
        "Ascend910": ("float16", "float32", "int32"),
        "Ascend610": ("float16", "float32", "int32"),
        "Ascend620": ("float16", "float32", "int32"),
        "Hi3796CV300ES": ("float16", "float32", "int32"),
    },
    "unsorted_segment_mean": {
        "AllSoc": ("float16", "float32", "int32"),
        "Ascend310": ("float16", "float32", "int32"),
        "Ascend910": ("float16", "float32", "int32"),
        "Ascend610": ("float16", "float32", "int32"),
        "Ascend620": ("float16", "float32", "int32"),
        "Hi3796CV300ES": ("float16", "float32", "int32"),
    },
    "unsorted_segment_prod": {
        "AllSoc": ("float16", "float32", "int32"),
        "Ascend310": ("float16", "float32", "int32"),
        "Ascend910": ("float16", "float32", "int32"),
        "Ascend610": ("float16", "float32", "int32"),
        "Ascend620": ("float16", "float32", "int32"),
        "Hi3796CV300ES": ("float16", "float32", "int32"),
    },
    "unsorted_segment_min": {
        "AllSoc": ("float16", "float32", "int32"),
        "Ascend310": ("float16", "float32", "int32"),
        "Ascend910": ("float16", "float32", "int32"),
        "Ascend610": ("float16", "float32", "int32"),
        "Ascend620": ("float16", "float32", "int32"),
        "Hi3796CV300ES": ("float16", "float32", "int32"),
    },
    "unsorted_segment_max": {
        "AllSoc": ("float16", "float32", "int32"),
        "Ascend310": ("float16", "float32", "int32"),
        "Ascend910": ("float16", "float32", "int32"),
        "Ascend610": ("float16", "float32", "int32"),
        "Ascend620": ("float16", "float32", "int32"),
        "Hi3796CV300ES": ("float16", "float32", "int32"),
    },

    # inplce
    "inplace_add": {
        "AllSoc": ("float16", "float32", "int32"),
        "Ascend310": ("float16", "float32", "int32"),
        "Ascend910": ("float16", "float32", "int32"),
        "Ascend610": ("float16", "float32", "int32"),
        "Ascend620": ("float16", "float32", "int32"),
        "Hi3796CV300ES": ("float16", "float32", "int32"),
    },
    "inplace_sub": {
        "AllSoc": ("float16", "float32", "int32"),
        "Ascend310": ("float16", "float32", "int32"),
        "Ascend910": ("float16", "float32", "int32"),
        "Ascend610": ("float16", "float32", "int32"),
        "Ascend620": ("float16", "float32", "int32"),
        "Hi3796CV300ES": ("float16", "float32", "int32"),
    },
    "inplace_update": {
        "AllSoc": ("float16", "float32", "int32"),
        "Ascend310": ("float16", "float32", "int32"),
        "Ascend910": ("float16", "float32", "int32"),
        "Ascend610": ("float16", "float32", "int32"),
        "Ascend620": ("float16", "float32", "int32"),
        "Hi3796CV300ES": ("float16", "float32", "int32"),
    },

    # ceil/floor/round/trunc
    "ceil": {
        "AllSoc": ("float16",),
        "Ascend310": ("float16",),
        "Ascend910": ("float16", "float32"),
        "Ascend610": ("float16", "float32"),
        "Ascend620": ("float16", "float32"),
        "Hi3796CV300ES": ("float16",),
    },
    "floor": {
        "AllSoc": ("float16",),
        "Ascend310": ("float16",),
        "Ascend910": ("float16", "float32"),
        "Ascend610": ("float16", "float32"),
        "Ascend620": ("float16", "float32"),
        "Hi3796CV300ES": ("float16",),
    },
    "round": {
        "AllSoc": ("float16",),
        "Ascend310": ("float16",),
        "Ascend910": ("float16", "float32"),
        "Ascend610": ("float16", "float32"),
        "Ascend620": ("float16", "float32"),
        "Hi3796CV300ES": ("float16",),
    },
    "trunc": {
        "AllSoc": ("float16",),
        "Ascend310": (),
        "Ascend910": ("float16", "float32"),
        "Ascend610": ("float16", "float32"),
        "Ascend620": ("float16", "float32"),
        "Hi3796CV300ES": ("float16",),
    },
    "round_d": {
        "AllSoc": ("float16",),
        "Ascend310": (),
        "Ascend910": ("float16", "float32"),
        "Ascend610": ("float16", "float32"),
        "Ascend620": ("float16", "float32"),
        "Hi3796CV300ES": ("float16",),
    },

    # reduce
    "reduce_sum": {
        "AllSoc": ("float16",),
        "Ascend310": ("float16", "float32"),
        "Ascend910": ("float16", "float32"),
        "Ascend610": ("float16", "float32"), # int32: nlst support, last not
        "Ascend620": ("float16", "float32"),
        "Hi3796CV300ES": ("float16",),
    },
    "reduce_max": {
        "AllSoc": ("float16",),
        "Ascend310": ("float16", "float32"), # fp32:last need priority_flag
        "Ascend910": ("float16", "float32"),
        "Ascend610": ("float16", "float32"),
        "Ascend620": ("float16", "float32"), # int32: nlst support, last not
        "Hi3796CV300ES": ("float16",),
    },
    "reduce_min": {
        "AllSoc": ("float16",),
        "Ascend310": ("float16", "float32"), # int32: nlst support, last not
        "Ascend910": ("float16", "float32"), # fp32:last need priority_flag
        "Ascend610": ("float16", "float32"),
        "Ascend620": ("float16", "float32"),
        "Hi3796CV300ES": ("float16",),
    },
    "reduce_prod": {
        "AllSoc": ("float16",),
        "Ascend310": ("float16", "float32"), # int32: nlst/last support
        "Ascend910": ("float16", "float32"),
        "Ascend610": ("float16", "float32"),
        "Ascend620": ("float16", "float32"),
        "Hi3796CV300ES": ("float16",),
    },

    # elewise
    "vadd": {
        "AllSoc": ("float16", "int32"),
        "Ascend310": ("float16", "float32", "int32"),
        "Ascend910": ("float16", "float32", "int32"),
        "Ascend610": ("float16", "float32", "int32"),
        "Ascend620": ("float16", "float32", "int32"),
        "Hi3796CV300ES": ("float16", "int32"),
    },
    "vsub": {
        "AllSoc": ("float16", "int32"),
        "Ascend310": ("float16", "float32", "int32"),
        "Ascend910": ("float16", "float32", "int32"),
        "Ascend610": ("float16", "float32", "int32"),
        "Ascend620": ("float16", "float32", "int32"),
        "Hi3796CV300ES": ("float16", "int32"),
    },
    "vmul": {
        "AllSoc": ("float16", "int32"),
        "Ascend310": ("float16", "float32", "int32"),
        "Ascend910": ("float16", "float32", "int32"),
        "Ascend610": ("float16", "float32", "int32"),
        "Ascend620": ("float16", "float32", "int32"),
        "Hi3796CV300ES": ("float16", "int32"),
    },
    "vdiv": {
        "AllSoc": ("float16", "int32"),
        "Ascend310": ("float16", "float32", "int32"),
        "Ascend910": ("float16", "float32", "int32"),
        "Ascend610": ("float16", "float32", "int32"),
        "Ascend620": ("float16", "float32", "int32"),
        "Hi3796CV300ES": ("float16", "int32"),
    },
    "vmod": {
        "AllSoc": ("float16",),
        "Ascend310": ("float16",),
        "Ascend910": ("float16", "float32"),
        "Ascend610": ("float16", "float32"),
        "Ascend620": ("float16", "float32"),
        "Hi3796CV300ES": ("float16",),
    },
    "vmin": {
        "AllSoc": ("float16", "int32"),
        "Ascend310": ("float16", "float32", "int32"),
        "Ascend910": ("float16", "float32", "int32"),
        "Ascend610": ("float16", "float32", "int32"),
        "Ascend620": ("float16", "float32", "int32"),
        "Hi3796CV300ES": ("float16", "int32"),
    },
    "vmax": {
        "AllSoc": ("float16", "int32"),
        "Ascend310": ("float16", "float32", "int32"),
        "Ascend910": ("float16", "float32", "int32"),
        "Ascend610": ("float16", "float32", "int32"),
        "Ascend620": ("float16", "float32", "int32"),
        "Hi3796CV300ES": ("float16", "int32"),
    },
    "vadds": {
        "AllSoc": ("float16", "int32"),
        "Ascend310": ("float16", "float32", "int32"),
        "Ascend910": ("float16", "float32", "int32"),
        "Ascend610": ("float16", "float32", "int32"),
        "Ascend620": ("float16", "float32", "int32"),
        "Hi3796CV300ES": ("float16", "int32"),
    },
    "vmins": {
        "AllSoc": ("float16", "int32"),
        "Ascend310": ("float16", "float32", "int32"),
        "Ascend910": ("float16", "float32", "int32"),
        "Ascend610": ("float16", "float32", "int32"),
        "Ascend620": ("float16", "float32", "int32"),
        "Hi3796CV300ES": ("float16", "int32"),
    },
    "vmaxs": {
        "AllSoc": ("float16", "int32"),
        "Ascend310": ("float16", "float32", "int32"),
        "Ascend910": ("float16", "float32", "int32"),
        "Ascend610": ("float16", "float32", "int32"),
        "Ascend620": ("float16", "float32", "int32"),
        "Hi3796CV300ES": ("float16", "int32"),
    },
    "vmuls": {
        "AllSoc": ("float16", "int32"),
        "Ascend310": ("float16", "float32", "int32"),
        "Ascend910": ("float16", "float32", "int32"),
        "Ascend610": ("float16", "float32", "int32"),
        "Ascend620": ("float16", "float32", "int32"),
        "Hi3796CV300ES": ("float16", "int32"),
    },
    "vnot": {
        "AllSoc": ("int16", "uint16"),
        "Ascend310": ("int16", "uint16"),
        "Ascend910": ("int16", "uint16"),
        "Ascend610": ("int16", "uint16"),
        "Ascend620": ("int16", "uint16"),
        "Hi3796CV300ES": ("int16", "uint16"),
    },
    "vor": {
        "AllSoc": ("int16", "uint16"),
        "Ascend310": ("int16", "uint16"),
        "Ascend910": ("int16", "uint16"),
        "Ascend610": ("int16", "uint16"),
        "Ascend620": ("int16", "uint16"),
        "Hi3796CV300ES": ("int16", "uint16"),
    },
    "vand": {
        "AllSoc": ("int16", "uint16"),
        "Ascend310": ("int16", "uint16"),
        "Ascend910": ("int16", "uint16"),
        "Ascend610": ("int16", "uint16"),
        "Ascend620": ("int16", "uint16"),
        "Hi3796CV300ES": ("int16", "uint16"),
    },
    "vcmp": {
        "AllSoc": ("float16",),
        "Ascend310": ("float16",),
        "Ascend910": ("float16", "float32"),
        "Ascend610": ("float16", "float32"),
        "Ascend620": ("float16", "float32"),
        "Hi3796CV300ES": ("float16",),
    },
    "vlogic": {
        "AllSoc": ("bool",),
        "Ascend310": ("bool",),
        "Ascend910": ("bool",),
        "Ascend610": ("bool",),
        "Ascend620": ("bool",),
        "Hi3796CV300ES": ("bool",),
    },
    "vsel": {
        "AllSoc": ("float16",),
        "Ascend310": ("float16",),
        "Ascend910": ("float16", "float32"),
        "Ascend610": ("float16", "float32"),
        "Ascend620": ("float16", "float32"),
        "Hi3796CV300ES": ("float16",),
    },
    "vcmpsel": {
        "AllSoc": ("float16",),
        "Ascend310": ("float16",),
        "Ascend910": ("float16", "float32"),
        "Ascend610": ("float16", "float32"),
        "Ascend620": ("float16", "float32"),
        "Hi3796CV300ES": ("float16",),
    },
    "vlog": {
        "AllSoc": ("float16",),
        "Ascend310": ("float16",),
        "Ascend910": ("float16", "float32"),
        "Ascend610": ("float16", "float32"),
        "Ascend620": ("float16", "float32"),
        "Hi3796CV300ES": ("float16",),
    },
    "vexp": {
        "AllSoc": ("float16",),
        "Ascend310": ("float16",),
        "Ascend910": ("float16", "float32"),
        "Ascend610": ("float16", "float32"),
        "Ascend620": ("float16", "float32"),
        "Hi3796CV300ES": ("float16",),
    },
    "vabs": {
        "AllSoc": ("float16",),
        "Ascend310": ("float16", "float32"),
        "Ascend910": ("float16", "float32"),
        "Ascend610": ("float16", "float32"),
        "Ascend620": ("float16", "float32"),
        "Hi3796CV300ES": ("float16",),
    },
    "vrec": {
        "AllSoc": ("float16",),
        "Ascend310": ("float16", "float32"),
        "Ascend910": ("float16", "float32"),
        "Ascend610": ("float16", "float32"),
        "Ascend620": ("float16", "float32"),
        "Hi3796CV300ES": ("float16",),
    },
    "vrelu": {
        "AllSoc": ("float16",),
        "Ascend310": ("float16",),
        "Ascend910": ("float16",),
        "Ascend610": ("float16", "float32"),
        "Ascend620": ("float16", "float32"),
        "Hi3796CV300ES": ("float16",),
    },
    "vsqrt": {
        "AllSoc": ("float16",),
        "Ascend310": ("float16", "float32"),
        "Ascend910": ("float16", "float32"),
        "Ascend610": ("float16", "float32"),
        "Ascend620": ("float16", "float32"),
        "Hi3796CV300ES": ("float16",),
    },
    "vrsqrt": {
        "AllSoc": ("float16",),
        "Ascend310": ("float16", "float32"),
        "Ascend910": ("float16", "float32"),
        "Ascend610": ("float16", "float32"),
        "Ascend620": ("float16", "float32"),
        "Hi3796CV300ES": ("float16",),
    },
    "vaxpy": {
        "AllSoc": ("float16",),
        "Ascend310": ("float16", "float32"),
        "Ascend910": ("float16", "float32"),
        "Ascend610": ("float16", "float32"),
        "Ascend620": ("float16", "float32"),
        "Hi3796CV300ES": ("float16",),
    },
    "vmla": {
        "AllSoc": ("float16",),
        "Ascend310": ("float16", "float32"),
        "Ascend910": ("float16", "float32"),
        "Ascend610": ("float16", "float32"),
        "Ascend620": ("float16", "float32"),
        "Hi3796CV300ES": ("float16",),
    },
    "vmadd": {
        "AllSoc": ("float16",),
        "Ascend310": ("float16", "float32"),
        "Ascend910": ("float16", "float32"),
        "Ascend610": ("float16", "float32"),
        "Ascend620": ("float16", "float32"),
        "Hi3796CV300ES": ("float16",),
    },
    "vmaddrelu": {
        "AllSoc": ("float16",),
        "Ascend310": ("float16", "float32"),
        "Ascend910": ("float16", "float32"),
        "Ascend610": ("float16", "float32"),
        "Ascend620": ("float16", "float32"),
        "Hi3796CV300ES": ("float16",),
    },

    # common
    "round_to": {
        "AllSoc": ("float16",),
        "Ascend310": ("float16", "float32", "int32"),
        "Ascend910": ("float16", "float32", "int32"),
        "Ascend610": ("float16", "float32"), # int32: schedule not support
        "Ascend620": ("float16", "float32"), # int32: schedule not support
        "Hi3796CV300ES": ("float16",), # int32: schedule not support
    },
    "cast_to": {
        "AllSoc": ("f162f32", "f162s8", "f162u8", "f162s32", \
                   "s82f16", "s82u8", "u82f16", "u82s8", \
                   "s322f16", "s322s8", "s322u8", "s322f32"),
        "Ascend310": ("f322f16", "f322s8", "f322u8", "f322s32", \
                      "f162f32", "f162s8", "f162u8", "f162s32", \
                      "s82f16", "s82u8", "u82f16", "u82s8", \
                      "s322f16", "s322s8", "s322u8", "s322f32"),
        "Ascend910": ("f322f16", "f322s8", "f322u8", "f322s32", \
                      "f162f32", "f162s8", "f162u8", "f162s32", \
                      "s82f16", "s82u8", "u82f16", "u82s8", \
                      "s322f16", "s322s8", "s322u8", "s322f32"),
        "Ascend610": ("f322f16", "f322s8", "f322u8", "f322s32", \
                      "f162f32", "f162s8", "f162u8", "f162s32", \
                      "s82f16", "s82u8", "u82f16", "u82s8", \
                      "s322f16", "s322s8", "s322u8", "s322f32"),
        "Ascend620": ("f322f16", "f322s8", "f322u8", "f322s32", \
                      "f162f32", "f162s8", "f162u8", "f162s32", \
                      "s82f16", "s82u8", "u82f16", "u82s8", \
                      "s322f16", "s322s8", "s322u8", "s322f32"),
        "Hi3796CV300ES": ("f162f32", "f162s8", "f162u8", "f162s32", \
                          "s82f16", "s82u8", "u82f16", "u82s8", \
                          "s322f16", "s322s8", "s322u8", "s322f32"),
    },

    "conv": {
        "AllSoc": ("float16",),
    },
    "compute_four2five": {
        "AllSoc": ("float16",),
    },
    "compute_five2four": {
        "AllSoc": ("float16",),
    },
    "matmul": {
        "AllSoc": ("float16", "f162f16", "f162f32"),
    },
    "pooling2d": {
        "AllSoc": ("float16",),
    },
    "concat": {
        "AllSoc": ("int8", "int16", "int32", "int64", \
                   "uint8", "uint16", "uint32", "uint64", \
                   "float16", "float32"),
    },
}


def dsl_support_dtype(dsl_name):
    """
    dsl_support_dtype
    """
    if not isinstance(dsl_name, str):
        return []

    if dsl_name in ("reduce_sum", "sum"):
        dsl_name = "reduce_sum"

    all_support_dtype = DSL_CHECK_SUPPORT_MAP.get(dsl_name)
    if all_support_dtype is None:
        return []

    soc_ver = get_soc_spec("SOC_VERSION")
    soc_support_dtype = all_support_dtype.get(soc_ver)
    if soc_support_dtype is None:
        soc_support_dtype = all_support_dtype.get("AllSoc")
        if soc_support_dtype is None:
            return []

    return list(soc_support_dtype)



def dsl_check_support(dsl_api, dtype=None):
    """
    dsl_check_support
    """
    if not dsl_api.startswith("te.lang.cce."):
        return False
    if (dtype is not None) and (not isinstance(dtype, str)):
        return False

    dsl_name = dsl_api.split("te.lang.cce.")[-1]
    if dsl_name in ("reduce_sum", "sum"):
        dsl_name = "reduce_sum"

    all_support_dtype = DSL_CHECK_SUPPORT_MAP.get(dsl_name)
    if all_support_dtype is None:
        return False

    soc_ver = get_soc_spec("SOC_VERSION")
    soc_support_dtype = all_support_dtype.get(soc_ver)
    if soc_support_dtype is None:
        soc_support_dtype = all_support_dtype.get("AllSoc")
        if soc_support_dtype is None:
            return False

    if (dtype not in (None, "")) and (dtype not in soc_support_dtype):
        return False

    return True


def check_input_type(*type_args, **type_kwargs):
    """
    check input parameter type
    """
    # for pylint, reserved argument, otherwise "Unused argument"
    type_kwargs = type_kwargs

    def out_wrapper(func):
        """
        out_wrapper
        """
        formal_parameter = func.__code__.co_varnames
        formal_parameter_list = list(zip(formal_parameter, type_args))

        def in_wrapper(*args, **kwargs):
            """
            in_wrapper
            """
            # pylint: disable=consider-using-enumerate
            for i in range(len(args)):
                if not isinstance(args[i], formal_parameter_list[i][1]):
                    raise RuntimeError(
                        "the input parameter %s must be %s, \
                            while type of input is %s"
                        % (formal_parameter_list[i][0],
                           formal_parameter_list[i][1], type(args[i])))
            for i in kwargs:
                for j in formal_parameter_list:
                    if i in j:
                        if not isinstance(kwargs[i], j[1]):
                            raise RuntimeError(
                                "the input parameter %s must be %s, \
                                    while type of input is %s"
                                % (i, j[1], type(kwargs[i])))
                        break
            return func(*args, **kwargs)

        return in_wrapper

    return out_wrapper


# pylint: disable=too-many-branches
@decorator
def dtype_check_decorator(func, *args, **kwargs):
    """
    dtype_check_decorator
    """
    func_name = func.__name__
    if func_name == "broadcast":
        if isinstance(args[0], int):
            judge_dtype = "int32"
        elif isinstance(args[0], float):
            judge_dtype = "float16"
        else:
            judge_dtype = args[0].dtype
    elif func_name == "concat":
        if not isinstance(args[0], list):
            raise RuntimeError("The first input type must be list")
        if not isinstance(args[0][0], tvm.tensor.Tensor):
            raise RuntimeError(
                "The first input type must be list of tvm.tensor")
        judge_dtype = args[0][0].dtype
    else:
        if not isinstance(args[0], tvm.tensor.Tensor):
            raise RuntimeError("The first input type must be tvm.tensor")
        judge_dtype = args[0].dtype

    if not dsl_check_support("te.lang.cce."+func_name, judge_dtype):
        raise RuntimeError("te.lang.cce.%s is not supported %s!"
                           % (func_name, judge_dtype))

    return func(*args, **kwargs)


def str_to_tuple(str_local):
    """
    str_to_tuple
    """
    if str_local:
        return str_local.split(",")
    return []


def is_cast_support(src_type, dst_type):
    """
    is_cast_support
    """
    if src_type not in DTYPE_MAP:
        raise RuntimeError("%s is unsupported dtype!" % src_type)

    if dst_type not in DTYPE_MAP:
        raise RuntimeError("%s is unsupported dtype!" % dst_type)

    if src_type == dst_type:
        return True

    cast_type = DTYPE_MAP[src_type] + "2" + DTYPE_MAP[dst_type]

    if cast_type == "s322f16":
        cast_type = "deq"

    return intrinsic_check_support("Intrinsic_vconv", cast_type)


def judge_var(num):
    """
    judge var if a tvm.var, tvm.const or python data type
    """
    var_dict = {
        "python_const": [int, float],
        "tvm_const": [tvm.expr.IntImm, tvm.expr.UIntImm, tvm.expr.FloatImm],
        "tvm_var": [tvm.expr.Var]
    }
    num_type = type(num)
    for i in var_dict:
        if num_type in var_dict[i]:
            return i
    raise RuntimeError("Input var Error")


def shape_to_list(shape):
    """
    translate tvm.shape to list type in python
    """
    tmp = []
    for i in shape:
        if isinstance(i, tvm.expr.Var):
            tmp.append(i)
        else:
            tmp.append(i.value)
    return tmp


def int_ceil_div(num_a, num_b):
    """
    upper division
    """
    if num_b == 0:
        raise RuntimeError(" division by zero")
    return (num_a + num_b - 1) // num_b


def get_and_res(flag_a, flag_b):
    """
    logical AND
    """
    return flag_a and flag_b


def get_or_res(flag_a, flag_b):
    """
    logical OR
    """
    return flag_a or flag_b


def refine_axis(axis, shape):
    """
    refine_axis
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
    return sorted(res_axis)


def _check(bool_res, append_str):
    if not bool_res:
        raise RuntimeError(append_str)


def auto_cast_tensor(tensor, intr, supported_types=None, is_auto_cast=True):
    """
    auto_cast_tensor
    """
    from .cast_compute import _cast
    if isinstance(tensor, tvm.tensor.Tensor):
        dtype = tensor.dtype
        intr_is_support_dtype = False
        intr_is_support_fp32 = False
        if supported_types is None:
            intrinsic = "Intrinsic_" + intr
            intr_is_support_dtype = intrinsic_check_support(intrinsic, dtype)
            intr_is_support_fp32 = intrinsic_check_support(intrinsic,
                                                           "float32")
        else:
            intr_is_support_dtype = (dtype in supported_types)
            intr_is_support_fp32 = ("float32" in supported_types)

        if not intr_is_support_dtype:
            if intr_is_support_fp32 and is_cast_support(dtype, "float32"):
                tensor = _cast(tensor, "float32", is_auto_cast)
            else:
                tensor = _cast(tensor, "float16", is_auto_cast)

    return tensor


def get_tvm_scalar(scalar, dtype):
    """
    get_tvm_scalar
    """
    scalar_type = judge_var(scalar)
    if scalar_type == "tvm_const" and scalar.dtype != dtype:
        scalar = tvm.const(scalar.value, dtype=dtype)

    if scalar_type == "python_const":
        scalar = tvm.const(scalar, dtype=dtype)

    return scalar


def check_input_tensor_shape(tensor_shape):
    """
    check_tensor_shape
    """
    shape = tensor_shape
    if isinstance(tensor_shape, tvm.tensor.Tensor):
        shape = shape_to_list(tensor_shape.shape)

    for val in shape:
        if isinstance(val, int) is False or val <= 0:
            raise RuntimeError(
                "The input shape value must be a positive integer")


def _axis_value_type_check(shape_len, value):
    """
    Check the value of the axis
    """
    if not isinstance(value, int):
        raise RuntimeError("type of axis value should be int")
    if value >= shape_len or value < -shape_len:
        raise RuntimeError(
            "input axis is out of range, axis value can be from %d to %d" %
            (-shape_len, shape_len - 1))
    if value < 0:
        value = shape_len + value
    return value


def reduce_axis_check(shape_len, axis):
    """
    Check the value of axis and return the sorted axis
    """
    axis = list(axis)
    if not hasattr(axis, 'index'):
        axis = _axis_value_type_check(shape_len, axis)
        return axis
    # pylint: disable=consider-using-enumerate
    for i in range(len(axis)):
        axis[i] = _axis_value_type_check(shape_len, axis[i])

    axis = list(set(axis))
    axis.sort()
    return axis
