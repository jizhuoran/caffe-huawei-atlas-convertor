"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

ascend_dequant
"""
import operator
import te.lang.cce

from te import tvm
from topi import generic
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import check_shape
from topi.cce import util


# pylint: disable=locally-disabled, too-many-arguments, unused-argument,
# pylint: disable=invalid-name, too-many-locals,unnecessary-lambda
def _check_params(x, deq_scaler, kernel_name):
    """
    check the parameters including shape, dtype, kernel_name, attr.
    """
    x_shape = x.get("shape")
    deq_shape = deq_scaler.get("shape")

    x_format = x.get("format")
    deq_format = deq_scaler.get("format")

    x_dtype = x.get("dtype").lower()
    deq_dtype = deq_scaler.get("dtype").lower()
    x_format_list = ["NC1HWC0", "FRACTAL_NZ"]
    if x_format not in x_format_list:
        raise RuntimeError(
            "ascend dequant x only support [NC1HWC0,FRACTAL_NZ]")
    if deq_format != "NC1HWC0":
        raise RuntimeError(
            "ascend dequant deq only support NC1HWC0")
    if x_format == "NC1HWC0":
        if len(x_shape) != 5:
            raise RuntimeError(
                "ascend dequant x only support the length of shape is 4 or 5")
    if x_format == "FRACTAL_NZ":
        if len(x_shape) != 4:
            raise RuntimeError(
                "ascend dequant x only support the length of shape is 4 or 5")
    if len(deq_shape) != 5:
        raise RuntimeError(
            "ascend dequant deq only support the length of shape is 5 ")

    if deq_shape[0] != 1 or deq_shape[2] != 1 or deq_shape[3] != 1:
        raise RuntimeError(
            "ascend dequant deq shape must be 1 in n,h,w")

    if x_dtype != "int32":
        raise RuntimeError(
            "ascend dequant x only support dtype is int32 ")

    deq_dtype_check = "float16"
    if util.is_v200_version():
        deq_dtype_check = "uint64"

    if deq_dtype != deq_dtype_check:
        raise RuntimeError(
            "ascend dequant x only support dtype is float16 or uint64 ")

    check_shape(x_shape)
    check_shape(deq_shape)

    util.check_kernel_name(kernel_name)


def _matmul_compute(x, x_shape, deq_scale, sqrt_mode, relu_flag,
                    shape_matmul_origin, tensor_format):
    """
    dequant for matmul
    """
    if util.is_v200_version():
        res_f16 = tvm.compute(x_shape, lambda i, j, k, l:
                              tvm.deq_cast(x(i, j, k, l),
                                           deq_scale(0, 0, 0, 0, 0),
                                           dtype="float16"),
                              name='dequant', tag="dequant_scale")
    else:
        res_f16 = tvm.compute(x_shape,
                              lambda i, j, k, l: (x(i, j, k, l).astype(
                                  "float16") * deq_scale(0, 0, 0, 0, 0)),
                              name='dequant', tag="dequant", )
        if sqrt_mode:
            res_f16 = tvm.compute(
                x_shape,
                lambda i, j, k, l: (res_f16(i, j, k, l).astype("float16") *
                                    deq_scale(0, 0, 0, 0, 0)),
                name='dequant_sqrt', tag="dequant_sqrt")

        if relu_flag:
            res_f16 = tvm.compute(x_shape,
                                  lambda *indices: tvm.relu(res_f16[indices]),
                                  name="dequant_relu", tag="dequant_relu")
    if tensor_format == "NC1HWC0":
        # convert fractal_z to ND
        res_out = tvm.compute(shape_matmul_origin, lambda i, j: res_f16[
            j // 16, i // 16, i % 16, j % 16], name='dequant_ND',
                              tag='dequant_ND', attrs={'format': 'NC1HWC0'})
    else:
        # nz format
        res_out = tvm.compute(x_shape, lambda *i: res_f16[i],
                              name='dequant_NZ', tag='dequant_NZ',
                              attrs={'format': 'FRACTAL_NZ'})
    return res_out


def _vector_dequant_v100(x, x_shape, align_shape, deq_scale, relu_flag,
                         sqrt_mode):
    """
    dequant for vector in v100

    """
    if relu_flag:
        res_f16 = tvm.compute(
            align_shape,
            lambda i, j, k, l: tvm.relu(x(i, j, k, l).astype("float16") *
                                        deq_scale(0, j, 0, 0, l)),
            name='dequant1', tag="dequant1_vector", attrs={"relu_flag": 1})

    else:
        res_f16 = tvm.compute(
            align_shape,
            lambda i, j, k, l: x(i, j, k, l).astype(
                "float16") * deq_scale(0, j, 0, 0, l),
            name='dequant1', tag="dequant1_vector", attrs={"relu_flag": 0})

    res = tvm.compute(x_shape, lambda *indice: res_f16(*indice),
                      name='dequant_remove_pad',
                      tag="dequant_remove_pad")

    if sqrt_mode:
        res = tvm.compute(
            x_shape, lambda i, j, k, l: (res(i, j, k, l) *
                                         deq_scale(0, j, 0, 0, l)),
            name='dequant2', tag='dequant2_vector')

    return res


def _scalar_dequant_v100(x, x_shape, align_shape, deq_scale, relu_flag,
                         sqrt_mode):
    """
    dequant for scale in v100

    """
    res_f16 = tvm.compute(
        align_shape,
        lambda i, j, k, l: (x(i, j, k, l).astype("float16") *
                            deq_scale(0, 0, 0, 0, 0)),
        name='dequant1', tag="dequant1_scale")

    res = tvm.compute(x_shape, lambda *indice: res_f16(*indice),
                      name='dequant_remove_pad',
                      tag="dequant_remove_pad")

    if relu_flag:
        res = tvm.compute(x_shape, lambda *indices: tvm.relu(
            res(*indices)),
                          name="dequant_relu",
                          tag="dequant_relu")
    if sqrt_mode:
        res = tvm.compute(
            x_shape,
            lambda i, j, k, l: (res(i, j, k, l) *
                                deq_scale(0, 0, 0, 0, 0)),
            name='dequant2', tag='dequant2_scale', )

    return res


def _vector_dequant_v200(x, x_shape, align_shape, deq_scale, relu_flag):
    """
    dequant for vector in v200

    """

    res_f16 = tvm.compute(align_shape, lambda i, j, k, l:
                          tvm.vdeq_cast(x(i, j, k, l),
                                        deq_scale(0, j, 0, 0, l),
                                        dtype="float16",
                                        do_relu=relu_flag),
                          name='dequant', tag="dequant_vector")

    res = tvm.compute(x_shape, lambda *indice: res_f16(*indice),
                      name='dequant_remove_pad',
                      tag="dequant_remove_pad")

    return res


def _vector_depthwise_fused(x, x_shape, align_shape, deq_scale, relu_flag,
                            sqrt_mode):
    """
    dequant for vector in v100

    """

    if relu_flag == True:
        res_f16 = tvm.compute(
            align_shape,
            lambda i, j, a, k, l: tvm.relu(
                x(i, j // 2, j % 2, k, l).astype("float16") *
                deq_scale(0, j, 0, 0, l)),
            name='dequant1', tag="dequant1_vector", attrs={"relu_flag": 1})

    elif relu_flag == False:
        res_f16 = tvm.compute(
            align_shape,
            lambda i, j, a, k, l: x(i, j // 2, j % 2, k, l).astype(
                "float16") * deq_scale(0, j, a, 0, l),
            name='dequant1', tag="dequant1_vector", attrs={"relu_flag": 0})

    align_shape[3] = x_shape[3].value

    if sqrt_mode == False:
        res = tvm.compute(align_shape, lambda *indice: res_f16(*indice),
                          name='dequant_remove_pad',
                          tag="dequant_remove_pad", attrs={"sqrt_flag": 0})
    elif sqrt_mode == True:
        res_sqrt = tvm.compute(
            align_shape, lambda i, j, a, k, l: (res_f16(i, j, a, k, l) *
                                                deq_scale(0, j, a, 0, l)),
            name='dequant2', tag='dequant2_vector')

        res = tvm.compute(align_shape, lambda *indice: res_sqrt(*indice),
                          name='dequant2_remove_pad',
                          tag="dequant2_remove_pad", attrs={"sqrt_flag": 1})
    return res


def _scalar_dequant_v200(x, x_shape, align_shape, deq_scale):
    """
    dequant for scale in v200

    """
    res_f16 = tvm.compute(align_shape, lambda i, j, k, l:
                          tvm.deq_cast(x(i, j, k, l),
                                       deq_scale(0, 0, 0, 0, 0),
                                       dtype="float16"),
                          name='dequant', tag="dequant_scale")

    res = tvm.compute(x_shape, lambda *indice: res_f16(*indice),
                      name='dequant_remove_pad',
                      tag="dequant_remove_pad")

    return res


@fusion_manager.register("ascend_dequant")
def ascend_dequant_compute(x, deq_scale, y, sqrt_mode=False, relu_flag=False,
                           kernel_name='ascend_dequant'):
    """
    int32 -> fp16

    Parameters:
     ----------
    x : the placeholder of input

    deq_scale: the placeholder of dequant num

    offset: the placeholder of offset num

    y : the dict of output.

    sqrt_mode : the sqrt mode when true the result to do sqrt

    relu_flag : the relu mode when true the result to do relu

    kernel_name : cce kernel name, default value is "ascend_dequant"

    Returns:

    res : the result of ascend_dequant
    -------
    None
    """

    def shape_to_list(shape):
        """
        trans shape to list shape
        """
        tmp = []
        for i in shape:
            tmp.append(i.value)
        return tmp

    x_shape = x.shape
    deq_shape = deq_scale.shape
    x_shape_list = shape_to_list(x_shape)
    deq_shape_list = shape_to_list(deq_shape)

    tensor_flag = False

    align_shape = x_shape_list
    if x.op.tag != "depthwise_conv2d":
        align_shape[2] = (align_shape[2] + 15) // 16 * 16

    if x.op.tag == "matmul" or x.op.tag == "matmul_gemv":
        tensor_format = "NC1HWC0"
        if x.op.attrs:
            if 'format' in x.op.attrs:
                # UB convergence scenario, input shape C1,N1,N0,C0
                tensor_format = x.op.attrs['format']
        shape_matmul_origin = x.op.attrs['shape']
        res = _matmul_compute(x, x_shape, deq_scale, sqrt_mode,
                              relu_flag, shape_matmul_origin, tensor_format)
        return res
    if x.op.tag == "depthwise_conv2d":
        align_shape[4] = 16
        align_shape[3] = (x_shape_list[3] + 15) // 16 * 16
        align_shape[2] = 1
        align_shape[1] = (deq_shape_list[1] * deq_shape_list[4]) // 16
        align_shape[0] = x_shape_list[0]
        res = _vector_depthwise_fused(x, x_shape, align_shape, deq_scale,
                                      relu_flag, sqrt_mode)
        return res

    if operator.eq((deq_shape_list[1] * deq_shape_list[4]),
                   (x_shape_list[1] * x_shape_list[3])):
        tensor_flag = True

    if tensor_flag:
        if util.is_v200_version():
            res = _vector_dequant_v200(x, x_shape, align_shape, deq_scale,
                                       relu_flag)
        else:
            res = _vector_dequant_v100(x, x_shape, align_shape, deq_scale,
                                       relu_flag, sqrt_mode)
    else:
        if util.is_v200_version():
            res = _scalar_dequant_v200(x, x_shape, align_shape, deq_scale)
        else:
            res = _scalar_dequant_v100(x, x_shape, align_shape, deq_scale,
                                       relu_flag, sqrt_mode)

    return res


def _dequant_v200_v2(x_l0c, deq_ub, align_shape, x_shape, relu_flag,
                     tensor_flag):
    """
    dequant for vector in v200

    """
    if tensor_flag:
        res_f16 = tvm.compute(align_shape, lambda i, j, k, l:
                              tvm.vdeq_cast(x_l0c(i, j, k, l),
                                            deq_ub(0, j, 0, l),
                                            dtype="float16",
                                            do_relu=relu_flag),
                              name='dequant_to_fp16', tag="dequant_vector")

    else:
        res_f16 = tvm.compute(align_shape, lambda i, j, k, l:
                              tvm.deq_cast(x_l0c(i, j, k, l),
                                           deq_ub(0, 0, 0, 0),
                                           dtype="float16"),
                              name='dequant_to_fp16', tag="dequant_scale")
    is_scalar = 1
    if tensor_flag:
        is_scalar = 0
    res = tvm.compute(x_shape, lambda *indice: res_f16(*indice),
                      name='res', tag="dequant_res",
                      attrs={'is_scalar': is_scalar})

    return res


def _vector_dequant_v100_v2(x_l0c, deq_ub, align_shape, x_shape, relu_flag,
                            sqrt_mode):
    """
    dequant for vector in v100

    """
    if relu_flag:
        res = tvm.compute(
            align_shape,
            lambda i, j, k, l: tvm.relu(x_l0c(i, j, k, l).astype("float16") *
                                        deq_ub(0, j, 0, l)),
            name='dequant_to_fp16')

    else:
        res = tvm.compute(
            align_shape,
            lambda i, j, k, l: x_l0c(i, j, k, l).astype(
                "float16") * deq_ub(0, j, 0, l),
            name='dequant_to_fp16')

    if sqrt_mode:
        res = tvm.compute(
            x_shape, lambda i, j, k, l: (res(i, j, k, l) * deq_ub(0, j, 0, l)),
            name='dequant_sqrt')

    res = tvm.compute(x_shape, lambda *indice: res(*indice),
                      name="res", tag='dequant_res',
                      attrs={'sqrt_mode': sqrt_mode,
                             'relu_mode': relu_flag,
                             'is_scalar': 0})

    return res


def _scalar_dequant_v100_v2(x_l0c, deq_ub, align_shape, x_shape, relu_flag,
                            sqrt_mode):
    """
    dequant for scale in v100

    """
    res = tvm.compute(
        align_shape,
        lambda i, j, k, l: (x_l0c(i, j, k, l).astype("float16") *
                            deq_ub(0, 0, 0, 0)),
        name='dequant_to_fp16')

    if sqrt_mode:
        res = tvm.compute(
            x_shape,
            lambda i, j, k, l: (res(i, j, k, l) * deq_ub(0, 0, 0, 0)),
            name='dequant_sqrt')

    if relu_flag:
        res = tvm.compute(x_shape, lambda *indices: tvm.relu(
            res(*indices)), name="dequant_relu")

    res = tvm.compute(x_shape, lambda *indice: res(*indice),
                      name="res", tag='dequant_res',
                      attrs={
                          'sqrt_mode': sqrt_mode,
                          'relu_mode': relu_flag,
                          'is_scalar': 1
                      })
    return res


def ascend_dequant_compute_v2(x, deq_scale, y, sqrt_mode=False,
                              relu_flag=False, kernel_name='ascend_dequant'):
    """
    int32 -> fp16

    Parameters:
     ----------
    x : the placeholder of input

    deq_scale: the placeholder of dequant num

    offset: the placeholder of offset num

    y : the dict of output.

    sqrt_mode : the sqrt mode when true the result to do sqrt

    relu_flag : the relu mode when true the result to do relu

    kernel_name : cce kernel name, default value is "ascend_dequant"

    Returns:

    res : the result of ascend_dequant
    -------
    None
    """

    x_shape_list = te.lang.cce.util.shape_to_list(x.shape)
    deq_shape_list = te.lang.cce.util.shape_to_list(deq_scale.shape)
    tensor_flag = False

    if operator.eq((deq_shape_list[1] * deq_shape_list[3]),
                   (x_shape_list[1] * x_shape_list[3])):
        tensor_flag = True

    align_shape = te.lang.cce.util.shape_to_list(x.shape)
    align_shape[2] = (align_shape[2] + 15) // 16 * 16

    x_ub = tvm.compute(x.shape, lambda *i: x(*i),
                       name='x_ub', tag="dequant_x_ub")
    deq_ub = tvm.compute(deq_scale.shape, lambda *i: deq_scale(*i),
                         name='deq_ub', tag="dequant_deq_ub")
    x_l0c = tvm.compute(align_shape, lambda *i: x_ub(*i),
                        name='x_l0c', tag="dequant_x_l0c")

    if tensor_flag:
        if util.is_v200_version():
            res = _dequant_v200_v2(x_l0c, deq_ub, align_shape, x.shape,
                                   relu_flag, tensor_flag)
        else:
            res = _vector_dequant_v100_v2(x_l0c, deq_ub, align_shape, x.shape,
                                          relu_flag, sqrt_mode)
    else:
        if util.is_v200_version():
            res = _dequant_v200_v2(x_l0c, deq_ub, align_shape, x.shape,
                                   relu_flag,
                                   tensor_flag)
        else:
            res = _scalar_dequant_v100_v2(x_l0c, deq_ub, align_shape, x.shape,
                                          relu_flag, sqrt_mode)
    return res


@util.check_input_type((dict), (dict), (dict), bool, bool, str)
def ascend_dequant(x, deq_scale, y, sqrt_mode=False, relu_mode=False,
                   kernel_name='ascend_dequant'):
    """
    int32 -> fp16

    Parameters:
    ----------
    x : the dict of input

    deq_scale: the dict of dequant num

    offset: the dict of offset num

    y : the dict of output.

    sqrt_mode : the sqrt mode when true the result to do sqrt

    relu_flag : the relu mode when true the result to do relu

    kernel_name : cce kernel name, default value is "ascend_dequant"

    Returns:
    -------
    None
    """

    _check_params(x, deq_scale, kernel_name)

    shape_x = x.get("shape")
    shape_deq = deq_scale.get("shape")

    dtype_x = x.get("dtype")
    dtype_deq = deq_scale.get("dtype")
    x_format = x.get("format")

    if dtype_deq == "uint64" and sqrt_mode:
        raise RuntimeError(
            "ascend dequant when deq_scale dtype is uint64,"
            "sqrt_mode only support False ")

    if x_format == "NC1HWC0":
        # n, C1, H*W, C0
        shape_x = [shape_x[0], shape_x[1], shape_x[2] * shape_x[3], shape_x[4]]
        shape_deq = [shape_deq[0], shape_deq[1], shape_deq[2] * shape_deq[3],
                     shape_deq[4]]
    else:
        # C1,N1,N0,C0 change to 1,C1,N1*N0,C0 equivalence N,C1,H*W,C0
        shape_x = [1, shape_x[0], shape_x[1] * shape_x[2], shape_x[3]]
        shape_deq = [1, shape_deq[0],
                     shape_deq[1] * shape_deq[2], shape_deq[3]]

    input_x = tvm.placeholder(shape_x, dtype_x, "x")
    input_deq = tvm.placeholder(shape_deq, dtype_deq, "deq_scale")

    with tvm.target.cce():
        res = ascend_dequant_compute_v2(input_x, input_deq, y, sqrt_mode,
                                        relu_mode, kernel_name)
        sch = generic.auto_schedule(res)
        config = {"name": kernel_name,
                  "tensor_list": [input_x, input_deq, res]}
        te.lang.cce.cce_build_code(sch, config)
