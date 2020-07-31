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

gemm_compute
"""

from __future__ import absolute_import  # pylint: disable=too-many-lines

import te.platform.cce_params as cce
from te.platform import intrinsic_check_support

import topi  # pylint: disable=import-error, ungrouped-imports
from te import tvm

from . import util
from .util import check_input_tensor_shape


def shape_check(tensor_a,  # pylint: disable=C0301, R0912, R0913, R0914, R0915
                tensor_b, tensor_bias, tensor_alpha, tensor_beta,
                trans_a, trans_b, format_a, format_b, dst_dtype):
    """
    Check the given input if legal

    Parameters:
    shape_a: list or tuple
            Shape of the first tensor a with rank > 1
    shape_b:  list or tuple
            Shape of the second tensor b with the same type with a,
            and shape_a, shape_b must be 2 dims
    shape_bias: list or tuple
            Shape of bias, only support the input data format with ND
    trans_a: bool
            If True, shape_a is transposed before multiplication
    trans_b: bool
            If True, shape_b is transposed before multiplication
    is_fractal: bool
            If True, the input data format of a and b must be fractal format

    Returns None
    """

    in_a_dtype = tensor_a.dtype
    in_b_dtype = tensor_b.dtype

    check_input_tensor_shape(tensor_a)
    check_input_tensor_shape(tensor_b)

    shape_a = [i.value for i in tensor_a.shape]
    shape_b = [i.value for i in tensor_b.shape]
    shape_bias = ()

    shape_len_a = len(shape_a)
    shape_len_b = len(shape_b)

    is_fractal_a = format_a != "ND"
    is_fractal_b = format_b != "ND"

    if tensor_bias is not None:
        shape_bias = [i.value for i in tensor_bias.shape]

    if (in_a_dtype in ("uint8", "int8")) and in_b_dtype == "int8":
        k_block_size = cce.BLOCK_REDUCE_INT8
    else:
        k_block_size = cce.BLOCK_REDUCE

    if dst_dtype == "int32":
        for index, value in enumerate(shape_bias):
            if index == 0:
                block = cce.BLOCK_IN
            else:
                block = cce.BLOCK_OUT
            shape_bias[index] = ((value + block - 1) // block) * block

    def _check_dtype():
        # check type of tensor_alpha and tensor_beta
        if tensor_alpha.dtype != tensor_beta.dtype:
            raise RuntimeError("dtype of alpha and beta are not same!")

        if dst_dtype != tensor_alpha.dtype:
            raise RuntimeError("dtype of alpha/beta and dst are not same!")

        # ND and ND only support 'float16'
        if not is_fractal_a and not is_fractal_b:
            if in_a_dtype == "int8" and dst_dtype != "int32":
                raise RuntimeError(
                    "only support 'float16' input datatype for 'ND' and 'ND' "
                    "format.")
        # ND and fractal support 'float16' and 'b8'
        else:
            if not (in_a_dtype == "float16" and in_b_dtype == "float16") and \
                    not (in_a_dtype in ("uint8", "int8") and (
                            in_b_dtype == "int8")):
                raise RuntimeError(
                    "only support float16 & float16 and uint8/int8 & int8 intput "
                    "data type.")

        if dst_dtype not in ("float16", "float32", "int32"):
            raise RuntimeError(
                "dst dtype only support float16 or float32 or int32!")

    def _check_fractal():
        if format_a not in ("ND", "fractal"):
            raise RuntimeError("format_a must be ND or fractal!")

        if format_b not in ("ND", "fractal"):
            raise RuntimeError("format_b must be ND or fractal!")

        # fractal and ND not support
        if is_fractal_a and not is_fractal_b:
            raise RuntimeError("Not support A is fractal and B is ND!")

        if (in_a_dtype in ("uint8", "int8")) and (in_b_dtype == "int8"):
            if not is_fractal_a and is_fractal_b:
                if trans_a:
                    raise RuntimeError(
                        "Not support A transpose for u8/s8 input and 'ND' & 'fractal'.")

        if (is_fractal_a == is_fractal_b) and (shape_len_a != shape_len_b):
            raise RuntimeError("A and B shape length should be equal.")

    _check_dtype()
    _check_fractal()

    def _check_shape():
        if is_fractal_a:
            if shape_len_a not in (4, 5):
                raise RuntimeError(
                    "for fractal input data, only support tensor's dim is 4 or 5!")
        else:
            if shape_len_a not in (2, 3):
                raise RuntimeError(
                    "for nd input data, only support tensor's dim is 2 or 3!")

        if is_fractal_b:
            if shape_len_b not in (4, 5):
                raise RuntimeError(
                    "for fractal input data, only support tensor's dim is 4 or 5!")
        else:
            if shape_len_b not in (2, 3):
                raise RuntimeError(
                    "for nd input data, only support tensor's dim is 2 or 3!")

        if shape_len_a in (3, 5):
            if tensor_a.shape[0].value != tensor_b.shape[0].value:
                raise RuntimeError("the shape's batch size not equal!")

    _check_shape()

    def _check_a_m_k_n():
        is_vector_a = False
        if not is_fractal_a:
            # shape_len_a is 2 or 3
            if trans_a:
                m_shape = shape_a[shape_len_a - 1]
                km_shape = shape_a[shape_len_a - 2]
                # non 16 multi result in buffer not align while transport
                if m_shape != cce.BLOCK_VECTOR and m_shape % cce.BLOCK_IN != 0:
                    raise RuntimeError(
                        "for ND input, shape_m must be %d or %d multi "
                        "for A transport" % (cce.BLOCK_VECTOR, cce.BLOCK_IN))
            else:
                m_shape = shape_a[shape_len_a - 2]
                km_shape = shape_a[shape_len_a - 1]
            real_shape_m = m_shape
        else:
            if trans_a:
                m_shape = shape_a[shape_len_a - 3]
                km_shape = shape_a[shape_len_a - 4]
                a_block_reduce = shape_a[shape_len_a - 1]
                a_block_in = shape_a[shape_len_a - 2]
            else:
                m_shape = shape_a[shape_len_a - 4]
                km_shape = shape_a[shape_len_a - 3]
                a_block_reduce = shape_a[shape_len_a - 1]
                a_block_in = shape_a[shape_len_a - 2]
            real_shape_m = m_shape * a_block_in

            if a_block_reduce != k_block_size:
                raise RuntimeError(
                    "for fractal input,tensor_a's shape last 2 dim must be %d or %d"
                    % (cce.BLOCK_IN, cce.BLOCK_VECTOR))

            if a_block_in not in (cce.BLOCK_VECTOR, cce.BLOCK_IN):
                raise RuntimeError(
                    "for fractal input,tensor_a's shape last 2 dim must be %d or %d"
                    % (cce.BLOCK_IN, cce.BLOCK_VECTOR))
            if a_block_in == cce.BLOCK_VECTOR:
                is_vector_a = True
                if m_shape != cce.BLOCK_VECTOR:
                    raise RuntimeError(
                        "for fractal input,tensor_a's shape last 2 dim "
                        "must be %d or %d" % (cce.BLOCK_IN, cce.BLOCK_VECTOR))
                if km_shape % (cce.BLOCK_IN) != 0:
                    raise RuntimeError(
                        "for fractal gevm input,K should be multiple of %d"
                        % (cce.BLOCK_IN * k_block_size))
        return km_shape, real_shape_m, is_vector_a

    km_shape, real_shape_m, is_vector_a = _check_a_m_k_n()

    def _check_b_m_k_n(is_vector_a):  # pylint: disable=too-many-branches
        is_gemv = False
        b_block_reduce = 1
        b_block_out = 1

        def _get_nd_m_k_n():
            # shape_len_b is 2 or 3
            if trans_b:
                kn_shape = shape_b[shape_len_b - 1]
                n_shape = shape_b[shape_len_b - 2]
            else:
                kn_shape = shape_b[shape_len_b - 2]
                n_shape = shape_b[shape_len_b - 1]

            return kn_shape, n_shape

        if not is_fractal_b:
            kn_shape, n_shape = _get_nd_m_k_n()
        else:
            if trans_b:
                kn_shape = shape_b[shape_len_b - 3]
                n_shape = shape_b[shape_len_b - 4]
                b_block_reduce = shape_b[shape_len_b - 2]
                b_block_out = shape_b[shape_len_b - 1]
            else:
                kn_shape = shape_b[shape_len_b - 4]
                n_shape = shape_b[shape_len_b - 3]
                b_block_reduce = shape_b[shape_len_b - 1]
                b_block_out = shape_b[shape_len_b - 2]

            if b_block_reduce != k_block_size:
                raise RuntimeError(
                    "for fractal input,tensor_b's shape last 2 dim must be %d"
                    % (k_block_size))

            if b_block_out not in (cce.BLOCK_VECTOR, cce.BLOCK_IN):
                raise RuntimeError(
                    "for fractal input,tensor_b's shape last 2 dim must be %d or %d"
                    % (cce.BLOCK_IN, cce.BLOCK_VECTOR))
            if b_block_out == cce.BLOCK_VECTOR:
                is_gemv = True
                if is_vector_a:
                    raise RuntimeError("input shape M and N can't both be 1")
                if n_shape != 1:
                    raise RuntimeError(
                        "for fractal input,tensor_b's shape last 2 dim "
                        "must be %d or %d" % (cce.BLOCK_IN, cce.BLOCK_VECTOR))
                if kn_shape % (cce.BLOCK_IN) != 0:
                    raise RuntimeError(
                        "for fractal gemv input,K should be multiple of %d"
                        % (cce.BLOCK_IN * k_block_size))
                # gemv u8/s8 is transed to gevm(s8/u8), s8/u8 is not support for mad intri
                if in_a_dtype == "uint8" and in_b_dtype == "int8":
                    raise RuntimeError(
                        "b8 gemv only support int8 & int8, current type is %s "
                        "and %s." % (in_a_dtype, in_b_dtype))

        return is_gemv, b_block_out, kn_shape, n_shape

    is_gemv, b_block_out, kn_shape, n_shape = _check_b_m_k_n(is_vector_a)

    def _check_a_between_b():
        if is_fractal_a == is_fractal_b:
            if km_shape != kn_shape:
                raise RuntimeError("reduce axis not same")

    _check_a_between_b()

    def renew_is_gemv(is_gemv):
        if not is_fractal_a and not is_fractal_b:
            is_gemv = n_shape == 1
        return is_gemv

    is_gemv = renew_is_gemv(is_gemv)

    if in_b_dtype == "int8":
        if is_gemv:
            if trans_a:
                # Load2D intri has error from L1 to L0B transport for b8
                raise RuntimeError(
                    "Not support A transpose for gemv b8 input.")
        else:
            if trans_b:
                # Load2D intri has error from L1 to L0B transport for b8
                raise RuntimeError(
                    "Not support B transpose for gevm or gemm b8 input.")

    def _check_bias():
        # 2d case
        if shape_bias:
            if len(shape_bias) == 2:
                if shape_bias not in \
                        ([real_shape_m, n_shape * b_block_out],
                         [1, n_shape * b_block_out]):
                    raise RuntimeError("bias shape must be [m,n] or [1,n]")
            elif len(shape_bias) == 4:
                if shape_bias[:2] != [n_shape, real_shape_m // cce.BLOCK_IN]:
                    raise RuntimeError("bias shape error")
            else:
                raise RuntimeError("bias shape must be [b,m,n] or [1,m,n] "
                                   "or [1,1,n] or [m,n] or [1,n] or [n]")


@util.check_input_type(tvm.tensor.Tensor, tvm.tensor.Tensor, tvm.tensor.Tensor,
                       tvm.tensor.Tensor, bool, bool, str,
                       str, str, (type(None), tvm.tensor.Tensor))
def gemm(tensor_a,  # pylint: disable=R1702, R0912, R0913, R0914, R0915
         tensor_b, tensor_alpha, tensor_beta, trans_a=False,
         trans_b=False, format_a="ND", format_b="ND", dst_dtype="float16",
         tensor_bias=None, quantize_params=None):
    """
    algorithm: mmad
    calculating  matrix multiplication, C=alpha_num*A*B+beta_num*C

    Parameters:
    tensor_a : the first tensor a

    tensor_b : second tensor b with the same type and shape with a

              If tensor_a/tensor_b is int8/uint8,then L0A must be 16*32,L0B
              must be 32*16.
              If A is transpose , then AShape classification matrix must be
              32*16 in gm/L1,then it is 16*32 in L0A.
              If B is transpose , then BShape classification matrix must be
              16*32 in gm/L1,then it is 32*16 in L0B.

    trans_a : if True, a needs to be transposed

    trans_b : if True, b needs to be transposed

    is_fractal: If type is bool, a and b's format both be fractal or ND,
                default is ND;
                If type is list, len must be 2, [0] is is_fractal_a,
                [1] is is_fractal_b

    alpha_num: scalar used for multiplication

    beta_num: scalar used for multiplication

    dst_dtype: output data type,support "float16" "float32", default is "float16"

    tensor_bias :the bias with used to init L0C for tensor c

    quantize_params: quantization parameters,
            not None means enable quantization, it is dictionary structure

        quantize_alg: quantize mode,
            support 'NON_OFFSET' 'HALF_OFFSET_A' 'HALF_OFFSET_B' 'ALL_OFFSET'

        scale_mode_a: tensor_a inbound quantization mode,
                support 'SCALAR' and 'VECTOR'
        scale_mode_b: tensor_b inbound quantization mode,
                support 'SCALAR' and 'VECTOR'
        scale_mode_out: out tensor quantization mode,
                support 'SCALAR' and 'VECTOR'

        sqrt_mode_a: tensor_a inbound sqrt mode, support 'NON_SQRT' and 'SQRT'
        sqrt_mode_b: tensor_b inbound sqrt mode, support 'NON_SQRT' and 'SQRT'
        sqrt_mode_out: out tensor sqrt mode, support 'NON_SQRT' and 'SQRT'

        scale_q_a: scale placeholder for tensor_a inbound quantization
        offset_q_a: offset placeholder for tensor_a inbound quantization
        scale_q_b: scale placeholder for tensor_b inbound quantization
        offset_q_b: offset placeholder for tensor_b inbound quantization

        scale_drq: scale placeholder for requantization or dequantization
        offset_drq: scale placeholder for requantization or dequantization

    Returns None
    """

    nz_a = False
    if format_a == "FRACTAL_NZ":
        nz_a = True
        format_a = "fractal"

    nz_b = False
    if format_b == "FRACTAL_NZ":
        nz_b = True
        format_b = "fractal"

    def _compute_alpha_beta():
        if tensor_alpha.dtype == "float16":
            tensor_alpha_temp_ub = tvm.compute(
                tensor_alpha.shape,
                lambda *indices: tensor_alpha(  # pylint: disable=W0108
                    *indices),
                name='tensor_alpha_temp_ub',
            )

            tensor_beta_temp_ub = tvm.compute(
                tensor_beta.shape,
                lambda *indices: tensor_beta(  # pylint: disable=W0108
                    *indices),
                name='tensor_beta_temp_ub',
            )

            tensor_alpha_ub = tvm.compute(
                tensor_alpha.shape,
                lambda *indices: topi.cast(tensor_alpha_temp_ub(
                    *indices), dtype="float32"),
                name='tensor_alpha_ub',
            )
            tensor_beta_ub = tvm.compute(
                tensor_beta.shape,
                lambda *indices: topi.cast(tensor_beta_temp_ub(
                    *indices), dtype="float32"),
                name='tensor_beta_ub',
            )
        else:
            tensor_alpha_ub = tvm.compute(
                tensor_alpha.shape,
                lambda *indices: tensor_alpha(  # pylint: disable=W0108
                    *indices),
                name='tensor_alpha_ub',
            )
            tensor_beta_ub = tvm.compute(
                tensor_beta.shape,
                lambda *indices: tensor_beta(  # pylint: disable=W0108
                    *indices),
                name='tensor_beta_ub',
            )
        return tensor_alpha_ub, tensor_beta_ub

    tensor_alpha_ub, tensor_beta_ub = _compute_alpha_beta()

    shape_check(tensor_a, tensor_b, tensor_bias, tensor_alpha, tensor_beta,
                trans_a, trans_b, format_a, format_b, dst_dtype)

    in_a_dtype = tensor_a.dtype
    in_b_dtype = tensor_b.dtype

    def _get_output_type():
        l0c_support_fp32 = intrinsic_check_support("Intrinsic_mmad", "f162f32")

        def _out_dtype():
            if in_a_dtype == "float16" and in_b_dtype == "float16":
                if dst_dtype not in ("float16", "float32"):
                    raise RuntimeError(
                        "dst_dtype must be 'float16' or 'float32'.")
                out_dtype = "float32"
                if not l0c_support_fp32:
                    out_dtype = "float16"
            elif (in_a_dtype == "int8" and in_b_dtype == "int8") or \
                    (in_a_dtype == "uint8" and in_b_dtype == "int8"):
                out_dtype = "int32"
            else:
                raise RuntimeError("data type of tensor not supported")

            if (out_dtype == dst_dtype) and (quantize_params is not None):
                raise RuntimeError(
                    "quantize parameter 'quantize_params' is unexpected.")

            if dst_dtype not in (out_dtype, "float16") and not (
                    dst_dtype == "float32" and out_dtype == "int32"
            ):
                raise RuntimeError(
                    "dst_dtype[%s] should be 'float16' for a_type[%s] and b_type[%s]."
                    % (dst_dtype, in_a_dtype, in_b_dtype))
            return out_dtype

        out_dtype = _out_dtype()

        if (out_dtype not in (dst_dtype, "float32")) and (
                quantize_params is None
        ) and not (dst_dtype == "float32" and out_dtype == "int32"):
            raise RuntimeError("Lack of quantize parameter 'quantize_params'.")
        if (quantize_params is not None) and (
                not isinstance(quantize_params, dict)):
            raise RuntimeError(
                "'quantize_params' should be dict type.")

        if in_a_dtype == "int8" and dst_dtype == "float32":
            out_dtype = "float32"
        return l0c_support_fp32, out_dtype

    l0c_support_fp32, out_dtype = _get_output_type()

    tensor_a_length = len(tensor_a.shape)
    tensor_b_length = len(tensor_b.shape)

    is_fractal_a = format_a != "ND"
    is_fractal_b = format_b != "ND"

    def _get_bias_shape():
        if tensor_bias.dtype != dst_dtype:
            raise RuntimeError(
                "Tensor bias type error, should be '%s'" % dst_dtype)

        bias_shape = list(tensor_bias.shape)
        if len(bias_shape) == 2:
            origin_bias_shape = bias_shape.copy()
            for index, value in enumerate(bias_shape):
                if index == 0:
                    block = cce.BLOCK_IN
                else:
                    block = cce.BLOCK_OUT
                bias_shape[index] = ((value + block - 1) // block) * block
        else:
            origin_bias_shape = None
        return bias_shape, origin_bias_shape

    if tensor_bias is not None:
        bias_shape, origin_bias_shape = _get_bias_shape()

    def _get_block():
        if in_a_dtype == "float16":
            block_reduce = cce.BLOCK_REDUCE
        else:
            block_reduce = cce.BLOCK_REDUCE_INT8

        block_in = cce.BLOCK_IN
        block_out = cce.BLOCK_OUT
        return block_reduce, block_in, block_out

    block_reduce, block_in, block_out = _get_block()
    gm_a_shape_normalize = []

    def _get_a_martix_shape(gm_a_shape_normalize):
        if trans_a:
            if is_fractal_a:
                m_shape = tensor_a.shape[tensor_a_length - 3].value
                m_shape_ori = m_shape
                km_shape = tensor_a.shape[tensor_a_length - 4].value
                km_shape_ori = km_shape
                gm_a_shape_normalize = tensor_a.shape
            else:
                m_shape = (
                    tensor_a.shape[tensor_a_length - 1].value +
                    block_in - 1) // block_in
                m_shape_ori = tensor_a.shape[tensor_a_length - 1].value
                km_shape = tensor_a.shape[tensor_a_length -
                                          2].value // block_reduce
                km_shape_ori = tensor_a.shape[tensor_a_length - 2].value
                gm_a_shape_normalize.append(km_shape * block_reduce)
                gm_a_shape_normalize.append(m_shape * block_in)
        else:
            if is_fractal_a:
                m_shape = tensor_a.shape[tensor_a_length - 4].value
                m_shape_ori = m_shape
                km_shape = tensor_a.shape[tensor_a_length - 3].value
                km_shape_ori = km_shape
                gm_a_shape_normalize = tensor_a.shape
            else:
                if in_a_dtype == 'int8':
                    m_shape = (((tensor_a.shape[
                        tensor_a_length - 2].value + 32 - 1) // 32) * 32) // 16
                else:
                    m_shape = (tensor_a.shape[
                        tensor_a_length - 2].value + block_in - 1) // block_in
                m_shape_ori = tensor_a.shape[tensor_a_length - 2].value
                km_shape = (tensor_a.shape[
                    tensor_a_length - 1].value + block_reduce - 1) \
                           // block_reduce
                km_shape_ori = tensor_a.shape[tensor_a_length - 1].value
                gm_a_shape_normalize.append(m_shape * block_in)
                gm_a_shape_normalize.append(km_shape * block_reduce)

        return m_shape, m_shape_ori, km_shape, km_shape_ori, gm_a_shape_normalize

    m_shape, m_shape_ori, km_shape, km_shape_ori, \
        gm_a_shape_normalize = _get_a_martix_shape(gm_a_shape_normalize)

    gm_b_shape_normalize = []

    def _get_b_martix_shape(gm_b_shape_normalize):
        if trans_b:
            if is_fractal_b:
                kn_shape = tensor_b.shape[tensor_b_length - 3].value
                kn_shape_ori = kn_shape
                n_shape = tensor_b.shape[tensor_b_length - 4].value
                n_shape_ori = n_shape
                gm_b_shape_normalize = tensor_b.shape
            else:
                kn_shape = tensor_b.shape[tensor_b_length -
                                          1].value // block_reduce
                kn_shape_ori = tensor_b.shape[tensor_b_length - 1].value
                n_shape = tensor_b.shape[tensor_b_length -
                                         2].value // block_out
                n_shape_ori = tensor_b.shape[tensor_b_length - 2].value
                gm_b_shape_normalize.append(kn_shape * block_reduce)
                gm_b_shape_normalize.append(n_shape * block_out)
        else:
            if is_fractal_b:
                kn_shape = tensor_b.shape[tensor_b_length - 4].value
                kn_shape_ori = kn_shape
                n_shape = tensor_b.shape[tensor_b_length - 3].value
                n_shape_ori = n_shape
                gm_b_shape_normalize = tensor_b.shape
            else:
                kn_shape = (tensor_b.shape[
                    tensor_b_length - 2].value + block_reduce - 1) // block_reduce
                kn_shape_ori = tensor_b.shape[tensor_b_length - 2].value
                if in_b_dtype == 'int8':
                    n_shape = (((tensor_b.shape[
                        tensor_b_length - 1].value + 32 - 1) // 32) * 32) // 16
                else:
                    n_shape = (tensor_b.shape[
                        tensor_b_length - 1].value + block_out - 1) \
                              // block_out
                n_shape_ori = tensor_b.shape[tensor_b_length - 1].value
                gm_b_shape_normalize.append(kn_shape * block_reduce)
                gm_b_shape_normalize.append(n_shape * block_out)

        return kn_shape, n_shape, n_shape_ori, kn_shape_ori, gm_b_shape_normalize

    kn_shape, n_shape, n_shape_ori, kn_shape_ori, gm_b_shape_normalize \
        = _get_b_martix_shape(gm_b_shape_normalize)

    def _check_k():
        # check shape
        if km_shape != kn_shape:
            raise RuntimeError("the k shape is wrong in mmad")

    _check_k()

    def _check_shape():
        if is_fractal_a:
            if trans_a:
                if not (tensor_a.shape[tensor_a_length - 1].value
                        == block_reduce and tensor_a.shape[
                            tensor_a_length - 2].value == block_in):
                    raise RuntimeError("AShape classification matrix is wrong")
            else:
                if not (tensor_a.shape[tensor_a_length - 2].value == block_in and
                        tensor_a.shape[
                            tensor_a_length - 1].value == block_reduce):
                    raise RuntimeError("AShape classification matrix is wrong")
        if is_fractal_b:
            if trans_b:
                if not (tensor_b.shape[
                        tensor_b_length - 2].value == block_reduce and
                        tensor_b.shape[tensor_b_length - 1].value == block_out):
                    raise RuntimeError("BShape classification matrix is wrong")
            else:
                if not (tensor_b.shape[tensor_b_length - 2].value == block_out and
                        tensor_b.shape[tensor_b_length - 1].value == block_reduce):
                    raise RuntimeError("BShape classification matrix is wrong")

    _check_shape()

    def _get_reduce():
        # kBurstAxis and kPointAxis
        if in_a_dtype == "int8" and dst_dtype == "float32":
            reduce_kp = tvm.reduce_axis((0, 16), name='kp')
            reduce_kb = tvm.reduce_axis((0, km_shape * 2), name='kb')
        else:
            reduce_kp = tvm.reduce_axis((0, block_reduce), name='kp')
            reduce_kb = tvm.reduce_axis((0, km_shape), name='kb')
        return reduce_kp, reduce_kb

    reduce_kp, reduce_kb = _get_reduce()

    def _get_optmt_flag():
        optmt_a = 0
        optmt_b = 0
        optmt_c = 0
        if in_a_dtype in {"float16", "int8"}:
            optmt_a = 1
        if in_b_dtype in {"float16", "int8"}:
            optmt_b = 1
        if dst_dtype in ("float16", "float32", "int32"):
            optmt_c = 1
        return optmt_a, optmt_b, optmt_c

    optmt_a, optmt_b, optmt_c = _get_optmt_flag()

    out_shape = (int(n_shape), int(m_shape), int(block_in), int(block_out))
    out_shape_ori = [int(m_shape_ori), int(n_shape_ori)]

    def check_shape_align(shape, factor):
        is_align = True
        for item in shape:
            if item.value % factor != 0:
                is_align = False
                break
        return is_align

    def _compute_bias():
        tensor_bias_ub_fract, tensor_beta_bias_ub, tensor_bias_ub = None, None, None
        if len(bias_shape) == 2:
            if not is_fractal_a:
                bias_m_shape_ori = tensor_bias.shape[0]
                bias_n_shape_ori = tensor_bias.shape[1]
                ub_bias_shape_normalize = [
                    m_shape * block_in, n_shape * block_out]
                tensor_bias_ub = tvm.compute(
                    ub_bias_shape_normalize,
                    lambda i, j: tvm.select(
                        i < bias_m_shape_ori,
                        tvm.select(
                            j < bias_n_shape_ori,
                            tensor_bias[i, j],
                            tvm.convert(0).astype(tensor_bias.dtype)),
                        tvm.convert(0).astype(tensor_bias.dtype)),
                    name='tensor_bias_ub'
                )
            else:
                tensor_bias_ub = tvm.compute(
                    bias_shape,
                    lambda i, j: tvm.select(
                        j < origin_bias_shape[-1],
                        tvm.select(
                            i < origin_bias_shape[-2],
                            tensor_bias[i, j],
                            tvm.convert(0).astype(dst_dtype)),
                        tvm.convert(0).astype(dst_dtype)),
                    name='tensor_bias_ub'
                )
                tensor_bias_ub_fract = tvm.compute(
                    out_shape, lambda i, j, k, l: tensor_bias_ub[
                        j * block_in + k, i * block_out + l] + 0,
                    name='tensor_bias_ub_fract')
        elif len(bias_shape) == 4:
            tensor_bias_ub = tvm.compute(
                out_shape, lambda *indices: tensor_bias(# pylint: disable=W0108
                    *indices),
                name='tensor_bias_ub')

        if tensor_bias_ub_fract is not None:
            if tensor_beta_ub.dtype == 'float32' and \
                    tensor_bias_ub_fract.dtype == 'float16':
                tensor_float32_bias_ub = tvm.compute(
                    tensor_bias_ub_fract.shape,
                    lambda *indices: topi.cast(
                        tensor_bias_ub_fract(*indices), dtype='float32'
                    ),
                    name='tensor_float32_bias_ub',
                )
                tensor_beta_bias_ub = tvm.compute(
                    tensor_bias_ub_fract.shape,
                    lambda *indices: tensor_beta_ub[0] *
                    tensor_float32_bias_ub(*indices),
                    name='tensor_beta_bias_ub',
                )
            else:
                tensor_beta_bias_ub = tvm.compute(
                    tensor_bias_ub_fract.shape,
                    lambda *indices: tensor_beta_ub[0] *
                    tensor_bias_ub_fract(*indices),
                    name='tensor_beta_bias_ub',
                )
        else:
            if tensor_beta_ub.dtype == 'float32' and tensor_bias_ub.dtype == 'float16':
                tensor_float32_bias_ub = tvm.compute(
                    tensor_bias_ub.shape,
                    lambda *indices: topi.cast(tensor_bias_ub(
                        *indices), dtype='float32'),
                    name='tensor_float32_bias_ub',
                )
                tensor_beta_bias_ub = tvm.compute(
                    tensor_bias_ub.shape,
                    lambda *indices: tensor_beta_ub[0] *
                    tensor_float32_bias_ub(*indices),
                    name='tensor_beta_bias_ub',
                )
            else:
                tensor_beta_bias_ub = tvm.compute(
                    tensor_bias_ub.shape,
                    lambda *indices: tensor_beta_ub[0] *
                    tensor_bias_ub(*indices),
                    name='tensor_beta_bias_ub',
                )
        return tensor_beta_bias_ub

    tensor_beta_bias_ub = _compute_bias()

    def _part_not_trans():
        if is_fractal_a:
            if nz_a:
                tensor_a_l1 = tvm.compute(
                    gm_a_shape_normalize,
                    lambda *indices: tensor_a(  # pylint: disable=W0108
                        *indices),
                    name='tensor_a_l1')
                tensor_a_l0a = tvm.compute(
                    (m_shape, km_shape, block_in, block_reduce),
                    lambda i, j, k, l: tensor_a_l1[i, j, l, k],
                    name='tensor_a_l0a')
            else:
                tensor_a_l1 = tvm.compute(
                    gm_a_shape_normalize,
                    lambda *indices: tensor_a(  # pylint: disable=W0108
                        *indices),
                    name='tensor_a_l1')
                tensor_a_l0a = tvm.compute(
                    gm_a_shape_normalize,
                    lambda *indices: tensor_a_l1(  # pylint: disable=W0108
                        *indices),
                    name='tensor_a_l0a')
        else:
            tensor_a_l1_shape = (
                m_shape, km_shape, block_in, block_reduce)
            if in_a_dtype == 'int8':
                is_a_align = check_shape_align(tensor_a.shape, 32)
                if is_a_align is False:
                    tensor_a_zero = tvm.compute(gm_a_shape_normalize,\
                     lambda *indice: tvm.const(0).astype(in_a_dtype),\
                     name="tensor_a_zero", tag="init_zero")
                    tensor_a_normalize_ub = tvm.compute(gm_a_shape_normalize,\
                     lambda i, j: tvm.select(i < m_shape_ori,\
                      tvm.select(j < km_shape_ori, tensor_a[i, j],\
                      tensor_a_zero[i, j]), tensor_a_zero[i, j]),\
                      name='tensor_a_normalize_ub')
                else:
                    tensor_a_normalize_ub = tvm.compute(
                        gm_a_shape_normalize,
                        lambda i, j: tensor_a[i, j],
                        name='tensor_a_normalize_ub')
                tensor_a_fract_k_shape = (
                    m_shape, km_shape, block_in, block_reduce)
                tensor_a_fract_k = tvm.compute(
                    tensor_a_fract_k_shape,
                    lambda i, j, k, l: tensor_a_normalize_ub[
                        i * block_in + k, j * block_reduce + l],
                    name='a_fract_k')
                tensor_a_l1 = tvm.compute(
                    tensor_a_fract_k_shape,
                    lambda *indices:# pylint: disable=W0108
                    tensor_a_fract_k(
                        *indices),
                    name='tensor_a_l1')
                tensor_a_l0a = tvm.compute(tensor_a_l1_shape,\
                     lambda *indices: tensor_a_l1(# pylint: disable=W0108
                         *indices), name='tensor_a_l0a')
            else:
                tensor_a_normalize_ub = tvm.compute(
                    gm_a_shape_normalize,
                    lambda i, j: tvm.select(  # pylint: disable=W0108
                        i < m_shape_ori,
                        tvm.select(
                            j < km_shape_ori,
                            tensor_a[i, j],
                            tvm.convert(0).astype("float16")),
                        tvm.convert(0).astype("float16")),
                    name='tensor_a_normalize_ub')
                tensor_a_fract_k_shape = (
                    km_shape, m_shape * block_in, block_reduce)
                tensor_a_fract_k = tvm.compute(
                    tensor_a_fract_k_shape,
                    lambda i, j, k: tensor_a_normalize_ub[
                        j, i * block_reduce + k],
                    name='a_fract_k')
                tensor_a_l1 = tvm.compute(
                    tensor_a_fract_k_shape,
                    lambda i, j, k: tensor_a_fract_k[i, j, k],
                    name='tensor_a_l1')
                tensor_a_l0a = tvm.compute(
                    tensor_a_l1_shape,
                    lambda i, j, k, l: tensor_a_l1[j, i * block_in + k, l],
                    name='tensor_a_l0a'
                )
        return tensor_a_l0a

    def _compute_a_matrix():  # pylint: disable=too-many-branches
        if not trans_a:
            tensor_a_l0a = _part_not_trans()
        else:
            def _part_trans():
                if is_fractal_a:
                    if nz_a:
                        if in_a_dtype == "int8" and dst_dtype == "float32":
                            tensor_a_ub = tvm.compute(
                                gm_a_shape_normalize,
                                lambda *indices:# pylint: disable=W0108
                                tensor_a(*indices),
                                name="tensor_a_ub",
                            )
                            tensor_float16_a_ub = tvm.compute(
                                gm_a_shape_normalize,
                                lambda *indices: topi.cast(
                                    tensor_a_ub(*indices), "float16"),
                                name="tensor_float16_a_ub",
                            )
                            new_a_shape = [
                                gm_a_shape_normalize[1],
                                gm_a_shape_normalize[0] * 2,
                                gm_a_shape_normalize[2],
                                gm_a_shape_normalize[3] // 2,
                            ]
                            tensor_zz_a_ub = tvm.compute(
                                new_a_shape,
                                lambda i, j, k, l: tensor_float16_a_ub[
                                    j // 2, i, k, (j * 16 + l) % 32],
                                name="tensor_zz_a_ub",
                            )
                            tensor_a_l1 = tvm.compute(\
                            new_a_shape,\
                            lambda *indices:# pylint: disable=W0108
                            tensor_zz_a_ub(*indices),
                            name='tensor_a_l1')
                            tensor_a_l0a = tvm.compute(\
                            new_a_shape,
                            lambda *indices:# pylint: disable=W0108
                            tensor_a_l1(
                                *indices), name='tensor_a_l0a')
                        else:
                            tensor_a_l1 = tvm.compute(\
                            gm_a_shape_normalize,
                            lambda *indices: tensor_a(# pylint: disable=W0108
                                *indices),
                            name='tensor_a_l1')
                            tensor_a_l0a = tvm.compute(
                                (m_shape, km_shape, block_in, block_reduce),
                                lambda i, j, k, l: tensor_a_l1[j, i, k, l],
                                name='tensor_a_l0a')
                    else:
                        tensor_a_l1 = tvm.compute(
                            gm_a_shape_normalize,
                            lambda *indices: tensor_a(  # pylint: disable=W0108
                                *indices),
                            name='tensor_a_l1')
                        tensor_a_l0a = tvm.compute(
                            (m_shape, km_shape, block_in, block_reduce),
                            lambda i, j, k, l: tensor_a_l1[j, i, l, k],
                            name='tensor_a_l0a')
                else:
                    tensor_a_ub = tvm.compute(
                        gm_a_shape_normalize,
                        lambda *indices: tensor_a(  # pylint: disable=W0108
                            *indices),
                        name='tensor_a_ub')
                    if optmt_a == 1:
                        tensor_a_l1_shape = (
                            m_shape, km_shape, block_reduce, block_in)
                        tensor_a_ub_fract = tvm.compute(
                            tensor_a_l1_shape, lambda i, j, k, l:
                            tensor_a_ub[j * block_reduce +
                                        k, i * block_in + l],
                            name='tensor_a_ub_fract')
                        tensor_a_l1 = tvm.compute(
                            tensor_a_l1_shape,
                            lambda *indices:# pylint: disable=W0108
                            tensor_a_ub_fract(
                                *indices),
                            name='tensor_a_l1')
                        tensor_a_l0a_shape = (
                            m_shape, km_shape, block_in, block_reduce)
                        tensor_a_l0a = tvm.compute(
                            tensor_a_l0a_shape,
                            lambda i, j, k, l: tensor_a_l1[i, j, l, k],
                            name='tensor_a_l0a')
                    else:
                        tensor_a_l1_shape = (
                            km_shape, m_shape, block_reduce, block_in)
                        tensor_a_l1 = tvm.compute(
                            tensor_a_l1_shape, lambda i, j, k, l:
                            tensor_a_ub[i * block_reduce +
                                        k, j * block_in + l],
                            name='tensor_a_l1')
                        tensor_a_l0a_shape = (
                            m_shape, km_shape, block_in, block_reduce)
                        tensor_a_l0a = tvm.compute(
                            tensor_a_l0a_shape,
                            lambda i, j, k, l: tensor_a_l1[j, i, l, k],
                            name='tensor_a_l0a')
                return tensor_a_l0a

            tensor_a_l0a = _part_trans()
        return tensor_a_l0a

    tensor_a_l0a = _compute_a_matrix()

    def _part_not_trans():
        if is_fractal_b:
            if nz_b:
                tensor_b_l1 = tvm.compute(
                    tensor_b.shape,
                    lambda *indices: tensor_b(  # pylint: disable=W0108
                        *indices),
                    name='tensor_b_l1')
                tensor_b_l0b = tvm.compute(
                    tensor_b.shape,
                    lambda *indices: tensor_b_l1(  # pylint: disable=W0108
                        *indices),
                    name='tensor_b_l0b')
            else:
                if in_b_dtype == "int8" and dst_dtype == "float32":
                    tensor_b_ub = tvm.compute(
                        tensor_b.shape,
                        lambda *indices: tensor_b(  # pylint: disable=W0108
                            *indices),
                        name="tensor_b_ub",
                    )
                    tensor_float16_b_ub = tvm.compute(
                        tensor_b.shape,
                        lambda *indices: topi.cast(tensor_b_ub(*indices),
                                                   "float16"),
                        name="tensor_float16_b_ub",
                    )
                    new_b_shape = [
                        tensor_b.shape[0] * 2,
                        tensor_b.shape[1],
                        tensor_b.shape[2],
                        tensor_b.shape[3] // 2,
                    ]
                    tensor_zn_b_ub = tvm.compute(
                        new_b_shape,
                        lambda i, j, k, l: tensor_float16_b_ub[
                            i // 2, j, k, (i * 16 + l) % 32],
                        name="tensor_zn_b_ub",
                    )
                    tensor_b_l1 = tvm.compute(
                        new_b_shape,
                        lambda *indices: tensor_zn_b_ub(# pylint: disable=W0108
                            *indices),
                        name='tensor_b_l1')
                    tensor_b_l0b = tvm.compute(
                        new_b_shape,
                        lambda *indices: tensor_b_l1(# pylint: disable=W0108
                            *indices),
                        name='tensor_b_l0b')
                else:
                    tensor_b_l1 = tvm.compute(
                        tensor_b.shape,
                        lambda *indices: tensor_b(# pylint: disable=W0108
                            *indices),
                        name='tensor_b_l1')
                    tensor_b_l0b = tvm.compute(
                        tensor_b.shape,
                        lambda *indices: tensor_b_l1(# pylint: disable=W0108
                            *indices),
                        name='tensor_b_l0b')
        else:
            tensor_b_l1_shape = (
                kn_shape, n_shape, block_reduce, block_out)
            if in_b_dtype == 'int8':
                is_b_align = check_shape_align(tensor_b.shape, 32)
                tensor_b_l1_shape = (
                    kn_shape, n_shape, block_out, block_reduce)
                tensor_b_ub_shape = (
                    kn_shape * block_reduce, n_shape * block_out)
                if is_b_align is False:
                    tensor_b_zero = tvm.compute(
                        tensor_b_ub_shape,
                        lambda *indice: tvm.const(0).astype(in_b_dtype),
                        name="tensor_b_zero",
                        tag="init_zero")
                    tensor_b_normalize_ub = tvm.compute(
                        tensor_b_ub_shape,
                        lambda i, j: tvm.select(i < kn_shape_ori,\
                         tvm.select(j < n_shape_ori, tensor_b[i, j],\
                          tensor_b_zero[i, j]),\
                          tensor_b_zero[i, j]), name='tensor_b_normalize_ub')
                else:
                    tensor_b_normalize_ub = tvm.compute(
                        tensor_b_ub_shape, lambda i, j:
                        tensor_b[i, j], name='tensor_b_normalize_ub')
                tensor_b_transpose_shape = (
                    n_shape * block_out, kn_shape * block_reduce)
                tensor_b_transpose = tvm.compute(tensor_b_transpose_shape,\
                     lambda i, j: tensor_b_normalize_ub[j, i],\
                      name='b_transpose')
                tensor_b_fract = tvm.compute(
                    (kn_shape, n_shape, block_out, block_reduce),
                    lambda i, j, k, l: tensor_b_transpose[
                        j * block_in + k, i * block_reduce + l],
                    name='b_fract')
                tensor_b_l1 = tvm.compute(
                    tensor_b_l1_shape,
                    lambda *indices:# pylint: disable=W0108
                    tensor_b_fract(*indices), name='tensor_b_l1')
                tensor_b_l0b = tvm.compute(
                    (kn_shape, n_shape, block_out, block_reduce),
                    lambda i, j, k, l: tensor_b_l1[i, j, k, l],
                    name='tensor_b_l0b')
            else:
                tensor_b_normalize_ub = tvm.compute(
                    gm_b_shape_normalize,
                    lambda i, j: tvm.select(# pylint: disable=W0108
                        i < kn_shape_ori,
                        tvm.select(
                            j < n_shape_ori,
                            tensor_b[i, j],
                            tvm.convert(0).astype(in_b_dtype)),
                        tvm.convert(0).astype(in_b_dtype)),
                    name='tensor_b_normalize_ub')
                tensor_b_zero = tvm.compute(
                    tensor_b_l1_shape,
                    lambda *indice: tvm.convert(0).astype(in_b_dtype),
                    name="tensor_b_zero",
                    tag="init_zero")
                tensor_b_fract = tvm.compute(
                    tensor_b_l1_shape,
                    lambda i, j, k, l:
                    tvm.select(
                        tvm.all(
                            i * block_reduce + k < kn_shape *
                            block_reduce,
                            i * block_reduce + k >= 0),
                        tensor_b_normalize_ub[i * block_reduce + k,
                                              j * block_out + l],
                        tensor_b_zero[i, j, k, l]),
                    name='b_fract')
                tensor_b_l1 = tvm.compute(
                    tensor_b_l1_shape,
                    lambda *indices: tensor_b_fract(# pylint: disable=W0108
                        *indices),
                    name='tensor_b_l1')
                tensor_b_l0b = tvm.compute(
                    (kn_shape, n_shape, block_out, block_reduce),
                    lambda i, j, k, l: tensor_b_l1[i, j, l, k],
                    name='tensor_b_l0b')
        return tensor_b_l0b

    def _compute_b_matrix():  # pylint: disable=too-many-branches
        if not trans_b:
            tensor_b_l0b = _part_not_trans()
        else:
            def _part_trans():
                if is_fractal_b:
                    if nz_b:
                        tensor_b_l1 = tvm.compute(
                            tensor_b.shape,
                            lambda *indices: tensor_b(# pylint: disable=W0108
                                *indices),
                            name='tensor_b_l1')
                        tensor_b_l0b = tvm.compute(
                            (kn_shape, n_shape, block_out, block_reduce),
                            lambda i, j, k, l: tensor_b_l1[j, i, l, k],
                            name='tensor_b_l0b')
                    else:
                        tensor_b_l1 = tvm.compute(
                            tensor_b.shape,
                            lambda *indices: tensor_b(# pylint: disable=W0108
                                *indices),
                            name='tensor_b_l1')
                        tensor_b_l0b = tvm.compute(
                            (kn_shape, n_shape, block_out, block_reduce),
                            lambda i, j, k, l: tensor_b_l1[j, i, l, k],
                            name='tensor_b_l0b')
                else:
                    tensor_b_ub = tvm.compute(
                        tensor_b.shape,
                        lambda *indices: tensor_b(# pylint: disable=W0108
                            *indices),
                        name='tensor_b_ub')

                    if optmt_b == 1:
                        tensor_b_l1_shape = (
                            kn_shape, n_shape, block_out, block_reduce)
                        tensor_b_ub_fract = tvm.compute(
                            tensor_b_l1_shape, lambda i, j, k, l:
                            tensor_b_ub[j * block_out +
                                        k, i * block_reduce + l],
                            name='tensor_b_ub_fract')
                        tensor_b_l1 = tvm.compute(
                            tensor_b_l1_shape,
                            lambda *indices: # pylint: disable=W0108
                            tensor_b_ub_fract(
                                *indices),
                            name='tensor_b_l1')
                        tensor_b_l0b_shape = (
                            kn_shape, n_shape, block_out, block_reduce)
                        tensor_b_l0b = tvm.compute(
                            tensor_b_l0b_shape,
                            lambda i, j, k, l: tensor_b_l1[i, j, k, l],
                            name='tensor_b_l0b')
                    else:
                        tensor_b_l1_shape = (
                            n_shape, kn_shape, block_out, block_reduce)
                        tensor_b_l1 = tvm.compute(
                            tensor_b_l1_shape, lambda i, j, k, l:
                            tensor_b_ub[i * block_out +
                                        k, j * block_reduce + l],
                            name='tensor_b_l1')
                        tensor_b_l0b_shape = (
                            kn_shape, n_shape, block_out, block_reduce)
                        tensor_b_l0b = tvm.compute(
                            tensor_b_l0b_shape,
                            lambda i, j, k, l: tensor_b_l1[j, i, k, l],
                            name='tensor_b_l0b')
                return tensor_b_l0b

            tensor_b_l0b = _part_trans()

        return tensor_b_l0b

    tensor_b_l0b = _compute_b_matrix()

    def _compute_c_martix():
        if block_in != cce.BLOCK_VECTOR:  # gemm
            # define mad compute
            tensor_c = tvm.compute(
                out_shape, lambda nb, mb, mp, np: tvm.sum(
                    (tensor_a_l0a[mb, reduce_kb, mp, reduce_kp] *
                     tensor_b_l0b[reduce_kb, nb, np, reduce_kp]).astype(out_dtype),
                    axis=[reduce_kb, reduce_kp]),
                name='tensor_c', attrs={'input_order': 'positive'})
            tensor_c_ub = get_tensor_c_ub(
                tensor_c, out_shape, tensor_bias,
                tensor_alpha_ub, l0c_support_fp32,
                tensor_beta_bias_ub, dst_dtype,
                is_fractal_a
            )

            if is_fractal_a and is_fractal_b:
                tensor_c_gm = tvm.compute(
                    out_shape,
                    lambda *indices: tensor_c_ub(# pylint: disable=W0108
                        *indices),
                    name='tensor_c_gm', tag='gemm')
            else:
                # ND out shape is dim 2, shape m is original value
                if optmt_c == 1:
                    tensor_c_gm = tvm.compute(
                        out_shape_ori,
                        lambda i, j: tvm.select(
                            i < m_shape_ori,
                            tvm.select(j < n_shape_ori, tensor_c_ub[i, j])),
                        name='tensor_c_gm',
                        tag='gemm')
                else:
                    tensor_c_gm = tvm.compute(
                        out_shape_ori, lambda i, j: tensor_c_ub[
                            j // 16, i // 16, i % 16, j % 16],
                        name='tensor_c_gm', tag='gemm')
        return tensor_c_gm

    tensor_c_gm = _compute_c_martix()
    return tensor_c_gm


def get_tensor_c_ub(  # pylint: disable=too-many-arguments
        tensor_c, out_shape, tensor_bias, tensor_alpha_ub,
        l0c_support_fp32, tensor_beta_bias_ub, dst_dtype,
        is_fractal_a
):
    """calculate tensor_c_ub"""
    tensor_c_before_mul_ub = tvm.compute(
        out_shape,
        lambda *indices: tensor_c(*indices),# pylint: disable=W0108
        name='tensor_c_before_mul_ub',
    )
    temp = tensor_c_before_mul_ub
    if temp.dtype == "int32" and dst_dtype == "float32":
        tensor_c_float16_before_mul_ub = tvm.compute(
            out_shape,
            lambda *indices: topi.cast(tensor_c_before_mul_ub(
                *indices), dtype="float16"),
            name="tensor_c_float16_before_mul_ub",
        )
        tensor_c_float32_before_mul_ub = tvm.compute(
            out_shape,
            lambda *indices: topi.cast(tensor_c_float16_before_mul_ub(
                *indices), dtype="float32"),
            name="tensor_c_float32_before_mul_ub",
        )
        temp = tensor_c_float32_before_mul_ub

    if tensor_bias is not None:
        tensor_alpha_c_ub = tvm.compute(
            out_shape,
            lambda *indices: temp(*indices) * tensor_alpha_ub[0],
            name='tensor_alpha_c_ub',
        )
        if not is_fractal_a:
            tensor_c_ub_temp = tvm.compute(
                tensor_beta_bias_ub.shape,
                lambda i, j: tensor_beta_bias_ub[i, j] + tensor_alpha_c_ub[
                    j // 16, i // 16, i % 16, j % 16],
                name='tensor_c_ub_temp',
            )
        else:
            tensor_c_ub_temp = tvm.compute(
                out_shape,
                lambda *indices: tensor_alpha_c_ub(*indices) +
                tensor_beta_bias_ub(*indices),
                name='tensor_c_ub_temp',
            )
    else:
        tensor_c_ub_temp = tvm.compute(
            out_shape,
            lambda *indices: temp(*indices) * tensor_alpha_ub[0],
            name='tensor_c_ub_temp',
        )
    if dst_dtype == 'float16' and l0c_support_fp32:
        tensor_c_ub = tvm.compute(
            tensor_c_ub_temp.shape,
            lambda *indices: topi.cast(
                tensor_c_ub_temp(*indices),
                dtype='float16',
            ),
            name='tensor_c_ub',
        )
    elif dst_dtype == 'float32' and l0c_support_fp32 \
            and not is_fractal_a:
        tensor_c_ub = tensor_c_ub_temp
    else:
        tensor_c_ub = tvm.compute(
            tensor_c_ub_temp.shape,
            lambda *indices: tensor_c_ub_temp( # pylint: disable=W0108
                *indices),
            name='tensor_c_ub',
        )
    return tensor_c_ub
