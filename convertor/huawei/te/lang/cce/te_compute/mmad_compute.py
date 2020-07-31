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

mmad_compute
"""
# pylint: disable=too-many-lines
from __future__ import absolute_import

from functools import reduce as functools_reduce

import te.platform.cce_params as cce
import te.platform.cce_conf as conf

# pylint: disable=import-error, ungrouped-imports
import topi
from te import tvm
from te.lang.cce.te_compute.util import shape_to_list

from . import util
from .util import check_input_tensor_shape


def elecnt_of_shape(shape):
    """
    calculate reduce shape
    """
    return functools_reduce(lambda x, y: x*y, shape)


# shape limit for matmul
# int32's max value
SHAPE_SIZE_LIMIT = 2**31 - 1

def shape_check(tensor_a, tensor_b, # pylint: disable=C0301, R0912, R0913, R0914, R0915
                tensor_bias, trans_a, trans_b, format_a, format_b, dst_dtype):
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

    if format_a not in ("ND", "fractal"):
        raise RuntimeError("format_a must be ND or fractal!")

    if format_b not in ("ND", "fractal"):
        raise RuntimeError("format_b must be ND or fractal!")

    is_fractal_a = format_a != "ND"
    is_fractal_b = format_b != "ND"

    # fractal and ND not support
    if is_fractal_a and not is_fractal_b:
        raise RuntimeError("Not support A is fractal and B is ND!")
    # ND and ND only support 'float16'
    if not is_fractal_a and not is_fractal_b:
        if in_a_dtype != "float16" or in_b_dtype != "float16":
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

    if (in_a_dtype in ("uint8", "int8")) and (in_b_dtype == "int8"):
        if not is_fractal_a and is_fractal_b:
            if trans_a:
                raise RuntimeError(
                    "Not support A transpose for u8/s8 input and 'ND' & 'fractal'.")

    def __check_shape_len():
        if (is_fractal_a == is_fractal_b) and \
                (shape_len_b not in [shape_len_a - 1, shape_len_a]):
            return False
        return True

    if not __check_shape_len():
        raise RuntimeError("A and B shape length should be equal.")

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

    def __check_batch_size():
        if shape_len_a == shape_len_b and \
                tensor_a.shape[0].value != tensor_b.shape[0].value:
            return False
        return True

    batch = None
    if shape_len_a in (3, 5):
        if not __check_batch_size():
            raise RuntimeError("the shape's batch size not equal!")
        batch = tensor_a.shape[0].value

    if tensor_bias is not None:
        shape_bias = [i.value for i in tensor_bias.shape]

    k_block_size = cce.BLOCK_REDUCE
    if (in_a_dtype in ("uint8", "int8")) and in_b_dtype == "int8":
        k_block_size = cce.BLOCK_REDUCE_INT8

    dst_dtype_check_list = ["float16", "float32", "int32"]
    if dst_dtype not in dst_dtype_check_list:
        raise RuntimeError(
            "dst dtype only support float16 or float32 or int32!")

    is_gemv = False
    is_vector_a = False
    a_block_in = 1
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
            a_block_reduce = shape_a[shape_len_a - 2]
            a_block_in = shape_a[shape_len_a - 1]
        else:
            m_shape = shape_a[shape_len_a - 4]
            km_shape = shape_a[shape_len_a - 3]
            a_block_reduce = shape_a[shape_len_a - 1]
            a_block_in = shape_a[shape_len_a - 2]
        real_shape_m = m_shape*a_block_in

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
                    % (cce.BLOCK_IN*k_block_size))
    b_block_reduce = 1
    b_block_out = 1
    if not is_fractal_b:
        # shape_len_b is 2 or 3
        if trans_b:
            kn_shape = shape_b[shape_len_b - 1]
            n_shape = shape_b[shape_len_b - 2]
        else:
            kn_shape = shape_b[shape_len_b - 2]
            n_shape = shape_b[shape_len_b - 1]

        if n_shape != 1 and n_shape % cce.BLOCK_IN != 0:
            raise RuntimeError(
                "input shape N should be multiple of %d or 1" % (cce.BLOCK_IN))

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
                    % (cce.BLOCK_IN*k_block_size))
            # gemv u8/s8 is transed to gevm(s8/u8), s8/u8 is not support for mad intri
            if in_a_dtype == "uint8" and in_b_dtype == "int8":
                raise RuntimeError(
                    "b8 gemv only support int8 & int8, current type is %s "
                    "and %s." % (in_a_dtype, in_b_dtype))
    if is_fractal_a == is_fractal_b:
        if km_shape != kn_shape:
            raise RuntimeError("reduce axis not same")
    elif not is_fractal_a and is_fractal_b:
        if km_shape != (kn_shape*b_block_reduce):
            raise RuntimeError(
                "Km shape should be equal whit kn*block")

    if not is_fractal_a and not is_fractal_b:
        if m_shape == 1 and n_shape == 1:
            raise RuntimeError("input shape M and N can't both be 1")
        if n_shape == 1:
            if kn_shape % (cce.BLOCK_IN*cce.BLOCK_IN) != 0:
                raise RuntimeError(
                    "input shape K should be multiple of %d"
                    % (cce.BLOCK_IN*cce.BLOCK_IN))
        elif km_shape % k_block_size != 0:
            raise RuntimeError(
                "input shape K should be multiple of %d" % (cce.BLOCK_IN))

        is_gemv = n_shape == 1

    if in_b_dtype == "int8":
        if is_gemv:
            if trans_a:
                # Load2D intri has error from L1 to L0B transport for b8
                raise RuntimeError("Not support A transpose for gemv b8 input.")
        else:
            if trans_b:
                # Load2D intri has error from L1 to L0B transport for b8
                raise RuntimeError(
                    "Not support B transpose for gevm or gemm b8 input.")
    # 2d case
    if shape_bias:
        if is_gemv:
            if len(shape_bias) == 1:
                raise RuntimeError(
                    "bias shape for gemv must be [m,1] or [1,m,1] or ", \
                    "[b,m,1], curr is ", shape_bias)
            if len(shape_bias) == 2:
                if shape_bias != [real_shape_m, 1]:
                    raise RuntimeError(
                        "bias shape for gemv must be [m,1] or [1,m,1] or ", \
                        "[b,m,1] for gemv, curr is ", shape_bias)
            elif len(shape_bias) == 3:
                if batch is None:
                    raise RuntimeError(
                        "tensor A and tensor B lack of batch "
                        "while bias has batch")
                if shape_bias not in ([1, real_shape_m, 1], \
                        [batch, real_shape_m, 1]):
                    raise RuntimeError(
                        "bias shape for gemv must be [m,1] or [1,m,1] or ", \
                        "[b,m,1] for gemv, curr is ", shape_bias)
            else:
                raise RuntimeError(
                    "bias shape must be [m,1] or [1,m,1] or ", \
                    "[b,m,1] for gemv, curr is ", shape_bias)
        else:
            if len(shape_bias) == 1:
                if shape_bias[0] != n_shape*b_block_out:
                    raise RuntimeError(
                        "broadcast bias shape must be equal to shape n")
            elif len(shape_bias) == 2:
                if shape_bias not in ([1, n_shape*b_block_out], ):
                    raise RuntimeError("bias shape must be [1,n], curr is ", \
                                       shape_bias)
            elif len(shape_bias) == 3:
                if batch is None:
                    raise RuntimeError(
                        "tensor A and tensor B lack of batch "
                        "while bias has batch")
                if shape_bias not in ([1, 1, n_shape*b_block_out], \
                        [batch, 1, n_shape*b_block_out]):
                    raise RuntimeError(
                        "bias shape must be [n,] or [1,n] or [1,1,n] or " \
                        "[b,1,n] for gevm and gemm, current is ", shape_bias)
            else:
                raise RuntimeError(
                    "bias shape must be [n,] or [1,n] or [1,1,n] or " \
                    "[b,1,n] for gevm and gemm, current is ", shape_bias)


def check_quantize_params(quantize_params=None): # pylint: disable=R0912
    """
    Parameters:
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
    """
    # check input quantization parameter info
    if quantize_params is None:
        return
    # quantization parameter default value
    quantize_mode = "NON_OFFSET"
    scale_out = "SCALAR"

    # check quantize_alg value
    if "quantize_alg" not in quantize_params:
        raise RuntimeError(
            "Lack of 'quantize_alg', need to supply it.")
    quantize_mode = quantize_params["quantize_alg"]
    if quantize_mode not in ("NON_OFFSET", "HALF_OFFSET_A"):
        raise RuntimeError(
            "'quantize_alg' is %s, it should be "
            "'NON_OFFSET' or 'HALF_OFFSET_A'." % (quantize_mode))
    # check inbound scale mode paras
    if "scale_mode_a" in quantize_params:
        raise RuntimeError(
            "Inbound scale mode a function is not supported.")

    if "scale_mode_b" in quantize_params:
        raise RuntimeError(
            "Inbound scale mode b function is not supported.")

    if "scale_q_a" in quantize_params:
        raise RuntimeError(
            "Inbound scale quant a function is not supported.")

    if "offset_q_a" in quantize_params:
        raise RuntimeError(
            "Inbound offset quant a function is not supported.")

    if "scale_q_b" in quantize_params:
        raise RuntimeError(
            "Inbound scale quant b function is not supported.")

    if "offset_q_b" in quantize_params:
        raise RuntimeError(
            "Inbound offset quant b function is not supported.")

    # check outbound scale mode paras
    if "scale_mode_out" not in quantize_params:
        raise RuntimeError(
            "Lack of 'scale_mode_out', need to supply it.")

    scale_out = quantize_params["scale_mode_out"]
    if scale_out not in ("SCALAR", "VECTOR"):
        raise RuntimeError(
            "'scale_mode_out' is %s, should be 'SCALAR' or 'VECTOR'."
            % (scale_out))

    # check inbound scale mode paras
    if "sqrt_mode_a" in quantize_params or "sqrt_mode_b" in quantize_params:
        raise RuntimeError(
            "Inbound sqrt mode function is not supported.")
    # check outbound sqrt mode paras
    if "sqrt_mode_out" not in quantize_params:
        raise RuntimeError(
            "Lack of 'sqrt_mode_out', need to supply it.")


def get_quantize_params(quantize_params=None, out_type="float16"):
    """
    algorithm: check matmul quantize parameters

    Parameters:
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

    Returns:
        scale_drq: DISABLE: dequant, ENABLE: requant
        scale_drq_tensor: scale drq tensor
        sqrt_out: NON_SQRT: none sqrt quantize, SQRT: sqrt quantize
    """
    # quantization parameter default value
    sqrt_out = "NON_SQRT"
    scale_drq_tensor = None
    scale_drq = "DISABLE"

    # check input quantization parameter info
    if quantize_params is not None:
        sqrt_out = quantize_params["sqrt_mode_out"]
        if sqrt_out not in ("NON_SQRT", "SQRT"):
            raise RuntimeError(
                "'scale_mode_out' is %s, should be 'NON_SQRT' or 'SQRT'."
                % (sqrt_out))
        # check out dyte and tensor drq
        if out_type == "float16":
            if "scale_drq" not in quantize_params:
                raise RuntimeError(
                    "'scale_drq' is None, should not be None "
                    "while out_dtype is 'float16'.")
            scale_drq = "ENABLE"
            scale_drq_tensor = quantize_params["scale_drq"]
            if scale_drq_tensor is None:
                raise RuntimeError(
                    "scale_drq_tensor is None, need to supply it.")
            if "offset_drq" in quantize_params:
                raise RuntimeError(
                    "'offset_drq' is unnecessary, please delete it.")
        else:
            raise RuntimeError(
                "'dst_dtype' is '%s', should be 'float16'" % (out_type))

    return scale_drq, scale_drq_tensor, sqrt_out

@util.check_input_type(tvm.tensor.Tensor, tvm.tensor.Tensor, bool, bool, str,
                       str, float, float, str, (type(None), tvm.tensor.Tensor))
def matmul(tensor_a, # pylint: disable=W0108, R1702, R0912, R0913, R0914, R0915
           tensor_b, trans_a=False, trans_b=False, format_a="ND", format_b="ND",
           alpha_num=1.0, beta_num=0.0, dst_dtype="float16", tensor_bias=None,
           quantize_params=None):
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

    shape_check(tensor_a, tensor_b, tensor_bias, trans_a, trans_b, format_a,
                format_b, dst_dtype)

    in_a_dtype = tensor_a.dtype
    in_b_dtype = tensor_b.dtype

    l0c_support_fp32 = 1
    # used for inner_product and ascend_dequant UB fusion
    is_fusion_mode = False
    support_type = conf.getValue("Intrinsic_mmad")
    if "f162f32" not in support_type:
        l0c_support_fp32 = 0
    if in_a_dtype == "float16" and in_b_dtype == "float16":
        if dst_dtype not in ("float16", "float32"):
            raise RuntimeError("dst_dtype must be 'float16' or 'float32'.")
        out_dtype = "float32"
        if l0c_support_fp32 == 0:
            out_dtype = "float16"
    elif (in_a_dtype == "int8" and in_b_dtype == "int8") or \
            (in_a_dtype == "uint8" and in_b_dtype == "int8"):
        out_dtype = "int32"
    else:
        raise RuntimeError("data type of tensor not supported")

    if (out_dtype == dst_dtype) and (quantize_params is not None):
        raise RuntimeError(
            "quantize parameter 'quantize_params' is unexpected.")

    if dst_dtype not in (out_dtype, "float16"):
        raise RuntimeError(
            "dst_dtype[%s] should be 'float16' for a_type[%s] and b_type[%s]."
            % (dst_dtype, in_a_dtype, in_b_dtype))

    if (in_a_dtype == "int8") and (dst_dtype == "float16") and (quantize_params is None):
        is_fusion_mode = True
    if (out_dtype not in (dst_dtype, "float32")) and (quantize_params is None) \
            and not is_fusion_mode:
        raise RuntimeError("Lack of quantize parameter 'quantize_params'.")
    if (quantize_params is not None) and (
            not isinstance(quantize_params, dict)):
        raise RuntimeError(
            "'quantize_params' should be dict type.")

    tensor_a_length = len(tensor_a.shape)
    tensor_b_length = len(tensor_b.shape)

    is_fractal_a = format_a != "ND"
    is_fractal_b = format_b != "ND"

    if tensor_bias is not None:
        if quantize_params is None and not is_fusion_mode:
            if tensor_bias.dtype != dst_dtype:
                raise RuntimeError(
                    "Tensor bias type error, should be '%s'" % dst_dtype)
        else:
            if tensor_bias.dtype != out_dtype:
                raise RuntimeError("Tensor bias type is '%s', should be '%s'"
                                   % (tensor_bias.dtype, out_dtype))

        bias_shape = []
        if elecnt_of_shape(tensor_bias.shape).value == 1:
            bias_shape = [1]
        else:
            for i in tensor_bias.shape:
                if bias_shape:
                    bias_shape.append(i.value)
                elif i.value != 0 and i.value != 1:
                    # first element vlaue should be > 1
                    bias_shape.append(i.value)

    if in_a_dtype == "float16":
        block_reduce = cce.BLOCK_REDUCE
    else:
        block_reduce = cce.BLOCK_REDUCE_INT8

    block_in = cce.BLOCK_IN
    block_out = cce.BLOCK_OUT

    gm_a_shape_normalize = []
    if trans_a:
        if is_fractal_a:
            m_shape = tensor_a.shape[tensor_a_length - 3].value
            m_shape_ori = m_shape
            km_shape = tensor_a.shape[tensor_a_length - 4].value
            vm_shape = tensor_a.shape[tensor_a_length - 1].value
            gm_a_shape_normalize = tensor_a.shape
        else:
            m_shape = (tensor_a.shape[
                tensor_a_length - 1].value + block_in - 1) // block_in
            m_shape_ori = tensor_a.shape[tensor_a_length - 1].value
            km_shape = tensor_a.shape[tensor_a_length - 2].value // block_reduce
            vm_shape = 16
            if tensor_a.shape[tensor_a_length - 1].value == 1:
                m_shape = 1
                vm_shape = 1
            if tensor_a_length in (3, 5):
                gm_a_shape_normalize.append(tensor_a.shape[0])
            gm_a_shape_normalize.append(km_shape*block_reduce)
            gm_a_shape_normalize.append(m_shape*vm_shape)

    else:
        if is_fractal_a:
            m_shape = tensor_a.shape[tensor_a_length - 4].value
            m_shape_ori = m_shape
            km_shape = tensor_a.shape[tensor_a_length - 3].value
            vm_shape = tensor_a.shape[tensor_a_length - 2].value
            gm_a_shape_normalize = tensor_a.shape
        else:
            m_shape = (tensor_a.shape[
                tensor_a_length - 2].value + block_in - 1) // block_in
            m_shape_ori = tensor_a.shape[tensor_a_length - 2].value
            km_shape = tensor_a.shape[tensor_a_length - 1].value // block_reduce
            vm_shape = 16
            if tensor_a.shape[tensor_a_length - 2].value == 1:
                m_shape = 1
                vm_shape = 1

            if tensor_a_length in (3, 5):
                gm_a_shape_normalize.append(tensor_a.shape[0])
            gm_a_shape_normalize.append(m_shape*vm_shape)
            gm_a_shape_normalize.append(km_shape*block_reduce)

    if trans_b:
        if is_fractal_b:
            kn_shape = tensor_b.shape[tensor_b_length - 3].value
            n_shape = tensor_b.shape[tensor_b_length - 4].value
            vn_shape = tensor_b.shape[tensor_b_length - 1].value
        else:
            kn_shape = tensor_b.shape[tensor_b_length - 1].value // block_reduce
            n_shape = tensor_b.shape[tensor_b_length - 2].value // block_out
            vn_shape = 16
            if tensor_b.shape[tensor_b_length - 2].value == 1:
                n_shape = 1
                vn_shape = 1
    else:
        if is_fractal_b:
            kn_shape = tensor_b.shape[tensor_b_length - 4].value
            n_shape = tensor_b.shape[tensor_b_length - 3].value
            vn_shape = tensor_b.shape[tensor_b_length - 2].value
        else:
            kn_shape = tensor_b.shape[tensor_b_length - 2].value // block_reduce
            n_shape = tensor_b.shape[tensor_b_length - 1].value // block_out
            vn_shape = 16
            if tensor_b.shape[tensor_b_length - 1].value == 1:
                n_shape = 1
                vn_shape = 1

    def gevm_block_in_value(m_shape, vm_shape, km_shape, block_in_ori):
        """
        calculate k!= block_in*block_reduce gevm block_in
        """
        block_in = block_in_ori
        if m_shape == 1 and vm_shape == 1 and km_shape % block_in == 0:
            block_in = cce.BLOCK_VECTOR
        return block_in

    block_in = gevm_block_in_value(m_shape, vm_shape, km_shape, block_in)

    if n_shape == 1 and vn_shape == 1:
        block_out = cce.BLOCK_VECTOR

    # check shape
    if km_shape != kn_shape:
        raise RuntimeError("the k shape is wrong in mmad")
    if alpha_num != 1.0 or beta_num != 0.0:
        raise RuntimeError("we not support this situation now!")

    if is_fractal_a:
        if trans_a:
            if not (tensor_a.shape[tensor_a_length - 2].value == block_reduce \
                    and tensor_a.shape[tensor_a_length - 1].value == block_in):
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

    # define reduce axis
    # kBurstAxis
    reduce_kb = tvm.reduce_axis((0, km_shape), name='kb')
    # kPointAxis
    reduce_kp = tvm.reduce_axis((0, block_reduce), name='kp')

    check_quantize_params(quantize_params)
    scale_drq, scale_drq_tensor, sqrt_out = \
        get_quantize_params(quantize_params, dst_dtype)

    # vadds function only support fp16 and fp32
    optmt_a = 0
    optmt_b = 0
    optmt_c = 0
    if in_a_dtype == "float16":
        optmt_a = 1
    if in_b_dtype == "float16":
        optmt_b = 1
    if dst_dtype == "float16":
        optmt_c = 1

    # not gemv
    if block_out != cce.BLOCK_VECTOR: # pylint: disable=too-many-nested-blocks
        if tensor_a_length in (2, 4):
            out_shape = (
                int(n_shape), int(m_shape), int(block_in), int(block_out))
            out_shape_ori = [int(m_shape_ori), int(n_shape*block_out)]
            if tensor_bias is not None:
                # bias only be [n,] and [1,n] for gevm and gemm
                if len(tensor_bias.shape) == 1:
                    tensor_bias_ub = tvm.compute(
                        bias_shape, lambda i:
                        tensor_bias[i], name='tensor_bias_ub')
                else:
                    tensor_bias_ub = tvm.compute(
                        bias_shape, lambda i: tensor_bias[0, i],
                        name='tensor_bias_ub')
            if not trans_a:
                if is_fractal_a:
                    if nz_a:
                        tensor_a_l1 = tvm.compute(
                            gm_a_shape_normalize,
                            lambda *indices: tensor_a[indices], name='tensor_a_l1')
                        tensor_a_l0a = tvm.compute(
                            (m_shape, km_shape, block_in, block_reduce),
                            lambda i, j, k, l: tensor_a_l1[i, j, l, k],
                            name='tensor_a_l0a')
                    else:
                        tensor_a_l1 = tvm.compute(
                            gm_a_shape_normalize,
                            lambda *indices: tensor_a[indices], name='tensor_a_l1')
                        tensor_a_l0a = tvm.compute(
                            gm_a_shape_normalize,
                            lambda *indices: tensor_a_l1[indices],
                            name='tensor_a_l0a')
                else:
                    tensor_a_ub = tvm.compute(
                        tensor_a.shape,
                        lambda *indices: tensor_a[indices], name='tensor_a_ub')

                    tensor_a_l1_shape = (
                        m_shape, km_shape, block_in, block_reduce)
                    if optmt_a == 1:
                        tensor_a_ub_fract = tvm.compute(
                            tensor_a_l1_shape, lambda i, j, k, l:
                            tensor_a_ub[i*block_in + k, j*block_reduce + l],
                            name='tensor_a_ub_fract')

                        tensor_a_l1 = tvm.compute(
                            tensor_a_l1_shape,
                            lambda *indices: tensor_a_ub_fract[indices],
                            name='tensor_a_l1')
                    else:
                        tensor_a_l1 = tvm.compute(
                            tensor_a_l1_shape, lambda i, j, k, l:
                            tensor_a_ub[i*block_in + k, j*block_reduce + l],
                            name='tensor_a_l1')

                    tensor_a_l0a = tvm.compute(
                        tensor_a_l1.shape,
                        lambda *indices: tensor_a_l1[indices],
                        name='tensor_a_l0a')
            else:
                if is_fractal_a:
                    if nz_a:
                        tensor_a_l1 = tvm.compute(
                            gm_a_shape_normalize,
                            lambda *indices: tensor_a[indices], name='tensor_a_l1')
                        tensor_a_l0a = tvm.compute(
                            (m_shape, km_shape, block_in, block_reduce),
                            lambda i, j, k, l: tensor_a_l1[j, i, k, l],
                            name='tensor_a_l0a')
                    else:
                        tensor_a_l1 = tvm.compute(
                            gm_a_shape_normalize,
                            lambda *indices: tensor_a[indices], name='tensor_a_l1')
                        tensor_a_l0a = tvm.compute(
                            (m_shape, km_shape, block_in, block_reduce),
                            lambda i, j, k, l: tensor_a_l1[j, i, l, k],
                            name='tensor_a_l0a')

                else:
                    tensor_a_ub = tvm.compute(
                        tensor_a.shape,
                        lambda *indices: tensor_a[indices], name='tensor_a_ub')
                    if optmt_a == 1:
                        tensor_a_l1_shape = (
                            m_shape, km_shape, block_reduce, block_in)
                        tensor_a_ub_fract = tvm.compute(
                            tensor_a_l1_shape, lambda i, j, k, l:
                            tensor_a_ub[j*block_reduce + k, i*block_in + l],
                            name='tensor_a_ub_fract')
                        tensor_a_l1 = tvm.compute(
                            tensor_a_l1_shape,
                            lambda *indices: tensor_a_ub_fract[indices],
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
                            tensor_a_ub[i*block_reduce + k, j*block_in + l],
                            name='tensor_a_l1')
                        tensor_a_l0a_shape = (
                            m_shape, km_shape, block_in, block_reduce)
                        tensor_a_l0a = tvm.compute(
                            tensor_a_l0a_shape,
                            lambda i, j, k, l: tensor_a_l1[j, i, l, k],
                            name='tensor_a_l0a')

            if not trans_b:
                if is_fractal_b:
                    if nz_b:
                        tensor_b_l1 = tvm.compute(
                            tensor_b.shape, lambda *indices: tensor_b[indices],
                            name='tensor_b_l1')
                        tensor_b_l0b = tvm.compute(
                            tensor_b.shape, lambda *indices: tensor_b_l1[indices],
                            name='tensor_b_l0b')
                    else:
                        tensor_b_l1 = tvm.compute(
                            tensor_b.shape, lambda *indices: tensor_b[indices],
                            name='tensor_b_l1')
                        tensor_b_l0b = tvm.compute(
                            tensor_b.shape, lambda *indices: tensor_b_l1[indices],
                            name='tensor_b_l0b')
                else:
                    tensor_b_ub = tvm.compute(
                        tensor_b.shape, lambda *indices: tensor_b[indices],
                        name='tensor_b_ub')

                    tensor_b_l1_shape = (
                        kn_shape, n_shape, block_reduce, block_out)
                    if optmt_b == 1:
                        tensor_b_ub_fract = tvm.compute(
                            tensor_b_l1_shape, lambda i, j, k, l:
                            tensor_b_ub[i*block_reduce + k, j*block_out + l],
                            name='tensor_b_ub_fract')

                        tensor_b_l1 = tvm.compute(
                            tensor_b_l1_shape,
                            lambda *indices: tensor_b_ub_fract[indices],
                            name='tensor_b_l1')
                    else:
                        tensor_b_l1 = tvm.compute(
                            tensor_b_l1_shape, lambda i, j, k, l:
                            tensor_b_ub[i*block_reduce + k, j*block_out + l],
                            name='tensor_b_l1')

                    tensor_b_l0b = tvm.compute(
                        (kn_shape, n_shape, block_out, block_reduce),
                        lambda i, j, k, l: tensor_b_l1[i, j, l, k],
                        name='tensor_b_l0b')
            else:
                if is_fractal_b:
                    if nz_b:
                        tensor_b_l1 = tvm.compute(
                            tensor_b.shape, lambda *indices: tensor_b[indices],
                            name='tensor_b_l1')
                        tensor_b_l0b = tvm.compute(
                            (kn_shape, n_shape, block_out, block_reduce),
                            lambda i, j, k, l: tensor_b_l1[j, i, l, k],
                            name='tensor_b_l0b')
                    else:
                        tensor_b_l1 = tvm.compute(
                            tensor_b.shape, lambda *indices: tensor_b[indices],
                            name='tensor_b_l1')
                        tensor_b_l0b = tvm.compute(
                            (kn_shape, n_shape, block_out, block_reduce),
                            lambda i, j, k, l: tensor_b_l1[j, i, l, k],
                            name='tensor_b_l0b')
                else:
                    tensor_b_ub = tvm.compute(
                        tensor_b.shape, lambda *indices: tensor_b[indices],
                        name='tensor_b_ub')

                    if optmt_b == 1:
                        tensor_b_l1_shape = (
                            kn_shape, n_shape, block_out, block_reduce)
                        tensor_b_ub_fract = tvm.compute(
                            tensor_b_l1_shape, lambda i, j, k, l:
                            tensor_b_ub[j*block_out + k, i*block_reduce + l],
                            name='tensor_b_ub_fract')
                        tensor_b_l1 = tvm.compute(
                            tensor_b_l1_shape,
                            lambda *indices: tensor_b_ub_fract[indices],
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
                            tensor_b_ub[i*block_out + k, j*block_reduce + l],
                            name='tensor_b_l1')
                        tensor_b_l0b_shape = (
                            kn_shape, n_shape, block_out, block_reduce)
                        tensor_b_l0b = tvm.compute(
                            tensor_b_l0b_shape,
                            lambda i, j, k, l: tensor_b_l1[j, i, k, l],
                            name='tensor_b_l0b')

            if block_in != cce.BLOCK_VECTOR:  # gemm
                # define mad compute
                tensor_c = tvm.compute(
                    out_shape, lambda nb, mb, mp, np: tvm.sum(
                        (tensor_a_l0a[mb, reduce_kb, mp, reduce_kp] *
                         tensor_b_l0b[reduce_kb, nb, np, reduce_kp]).astype(
                             out_dtype), axis=[reduce_kb, reduce_kp]),
                    name='tensor_c', attrs={'input_order': 'positive'})
                if tensor_bias is None:
                    if scale_drq_tensor is not None:
                        tensor_c_ub = tvm.compute(
                            out_shape, lambda *indices: (tensor_c[indices]) *
                            scale_drq_tensor[0], name='tensor_c_ub',
                            attrs={'scale_drq': scale_drq,
                                   'sqrt_out': sqrt_out,
                                   'nz_b': nz_b})
                    elif is_fusion_mode:
                        tensor_c_ub = tvm.compute(
                            out_shape, lambda *indices: tensor_c[indices],
                            name='tensor_c_ub', attrs={'scale_drq': scale_drq,
                                                       'sqrt_out': sqrt_out,
                                                       'shape': out_shape_ori,
                                                       'nz_b': nz_b},
                            tag='matmul')
                    elif dst_dtype == 'float16' and l0c_support_fp32 == 1:
                        tensor_c_ub = tvm.compute(
                            out_shape,
                            lambda *indices: topi.cast(tensor_c[indices],
                                                       dtype='float16'),
                            name='tensor_c_ub', attrs={'scale_drq': scale_drq,
                                                       'sqrt_out': sqrt_out,
                                                       'nz_b': nz_b})
                    else:
                        tensor_c_ub = tvm.compute(
                            out_shape, lambda *indices: tensor_c[indices],
                            name='tensor_c_ub', attrs={'scale_drq': scale_drq,
                                                       'sqrt_out': sqrt_out,
                                                       'nz_b': nz_b})

                else:
                    if tensor_bias.dtype == 'float16' and l0c_support_fp32 == 1:
                        # tensor_bias_ub must be [n,]
                        tensor_bias_l0c = tvm.compute(
                            out_shape, lambda i, j, k, l: topi.cast(
                                tensor_bias_ub[i*block_out + l],
                                dtype='float32'), name='tensor_bias_l0c')
                    else:
                        # tensor_bias_ub must be [n,]
                        tensor_bias_l0c = tvm.compute(
                            out_shape, lambda i, j, k, l: tensor_bias_ub[
                                i*block_out + l], name='tensor_bias_l0c')

                    tensor_c_add_bias = tvm.compute(out_shape, lambda *indices:
                                                    tensor_bias_l0c[indices] +
                                                    tensor_c[indices],
                                                    name='tensor_c_add_bias')

                    if scale_drq_tensor is not None:
                        tensor_c_ub = tvm.compute(
                            out_shape, lambda *indices: (
                                tensor_c_add_bias[indices] *
                                scale_drq_tensor[0]),
                            name='tensor_c_ub', attrs={'scale_drq': scale_drq,
                                                       'sqrt_out': sqrt_out,
                                                       'nz_b': nz_b})
                    elif is_fusion_mode:
                        tensor_c_ub = tvm.compute(
                            out_shape,
                            lambda *indices: tensor_c_add_bias[indices],
                            name='tensor_c_ub', attrs={'scale_drq': scale_drq,
                                                       'sqrt_out': sqrt_out,
                                                       'shape': out_shape_ori,
                                                       'nz_b': nz_b},
                            tag='matmul')
                    elif dst_dtype == 'float16' and l0c_support_fp32 == 1:
                        tensor_c_ub = tvm.compute(
                            out_shape, lambda *indices: topi.cast(
                                tensor_c_add_bias[indices], dtype='float16'),
                            name='tensor_c_ub', attrs={'scale_drq': scale_drq,
                                                       'sqrt_out': sqrt_out,
                                                       'nz_b': nz_b})
                    else:
                        tensor_c_ub = tvm.compute(
                            out_shape,
                            lambda *indices: tensor_c_add_bias[indices],
                            name='tensor_c_ub', attrs={'scale_drq': scale_drq,
                                                       'sqrt_out': sqrt_out,
                                                       'nz_b': nz_b})

                if is_fusion_mode:
                    return tensor_c_ub

                if is_fractal_a and is_fractal_b:
                    tensor_c_gm = tvm.compute(
                        out_shape, lambda *indices: tensor_c_ub[indices],
                        name='tensor_c_gm', tag='matmul')
                else:
                    # ND out shape is dim 2, shape m is original value
                    if optmt_c == 1:
                        tensor_c_ub_fract = tvm.compute(
                            out_shape_ori, lambda i, j: tensor_c_ub[
                                j // 16, i // 16, i % 16, j % 16],
                            name='tensor_c_ub_fract')
                        tensor_c_gm = tvm.compute(
                            out_shape_ori,
                            lambda *indices: tensor_c_ub_fract[indices],
                            name='tensor_c_gm', tag='matmul')
                    else:
                        tensor_c_gm = tvm.compute(
                            out_shape_ori, lambda i, j: tensor_c_ub[
                                j // 16, i // 16, i % 16, j % 16],
                            name='tensor_c_gm', tag='matmul')
            else:  # gevm
                orig_shape = list(out_shape)
                orig_shape[-2] = block_in

                # define mad
                tensor_c = tvm.compute(
                    (n_shape, m_shape, block_out, block_out),
                    lambda nb, mb, mp, np: tvm.sum(
                        (tensor_a_l0a[mb, reduce_kb, mp, reduce_kp] *
                         tensor_b_l0b[reduce_kb, nb, np, reduce_kp]).astype(
                             out_dtype), axis=[reduce_kb, reduce_kp]),
                    name='tensor_c', attrs={'input_order': 'positive'})
                if tensor_bias is not None:
                    # tensor_bias_ub only be [n,]
                    if tensor_bias.dtype == 'float16' and l0c_support_fp32 == 1:
                        tensor_bias_l0c = tvm.compute(
                            tensor_c.shape, lambda i, j, k, l: topi.cast(
                                tensor_bias_ub[i*block_out + l],
                                dtype='float32'), name='tensor_bias_l0c')
                    else:
                        tensor_bias_l0c = tvm.compute(
                            tensor_c.shape, lambda i, j, k, l: tensor_bias_ub[
                                i*block_out + l], name='tensor_bias_l0c')

                    tensor_c_add_bias = tvm.compute(
                        tensor_c.shape,
                        lambda *indices: tensor_bias_l0c[indices] + tensor_c[indices],
                        name='tensor_c_add_bias')
                    if scale_drq_tensor is not None:
                        tensor_c_ub = tvm.compute(
                            orig_shape,
                            lambda *indices: tensor_c_add_bias[indices] *
                            scale_drq_tensor[0], name='tensor_c_ub',
                            attrs={'scale_drq': scale_drq,
                                   'sqrt_out': sqrt_out,
                                   'nz_b': nz_b})
                    elif is_fusion_mode:
                        tensor_c_ub = tvm.compute(
                            orig_shape,
                            lambda *indices: tensor_c_add_bias[indices],
                            name='tensor_c_ub', attrs={'scale_drq': scale_drq,
                                                       'sqrt_out': sqrt_out,
                                                       'shape': out_shape_ori,
                                                       'nz_b': nz_b},
                            tag='matmul_gemv')
                    elif dst_dtype == 'float16' and l0c_support_fp32 == 1:
                        tensor_c_ub = tvm.compute(
                            out_shape, lambda *indices: topi.cast(
                                tensor_c_add_bias[indices], dtype='float16'),
                            name='tensor_c_ub', attrs={'scale_drq': scale_drq,
                                                       'sqrt_out': sqrt_out,
                                                       'nz_b': nz_b})
                    else:
                        tensor_c_ub = tvm.compute(
                            orig_shape,
                            lambda *indices: tensor_c_add_bias[indices],
                            name='tensor_c_ub', attrs={'scale_drq': scale_drq,
                                                       'sqrt_out': sqrt_out,
                                                       'nz_b': nz_b})

                else:
                    if scale_drq_tensor is not None:
                        tensor_c_ub = tvm.compute(
                            orig_shape, lambda *indices: tensor_c[indices] *
                            scale_drq_tensor[0], name='tensor_c_ub',
                            attrs={'scale_drq': scale_drq,
                                   'sqrt_out': sqrt_out,
                                   'nz_b': nz_b})
                    elif is_fusion_mode:
                        tensor_c_ub = tvm.compute(
                            orig_shape, lambda *indices: tensor_c[indices],
                            name='tensor_c_ub', attrs={'scale_drq': scale_drq,
                                                       'sqrt_out': sqrt_out,
                                                       'shape': out_shape_ori,
                                                       'nz_b': nz_b},
                            tag='matmul_gemv')
                    elif dst_dtype == 'float16' and l0c_support_fp32 == 1 and not is_fusion_mode:
                        tensor_c_ub = tvm.compute(
                            out_shape,
                            lambda *indices: topi.cast(tensor_c[indices],
                                                       dtype='float16'),
                            name='tensor_c_ub', attrs={'scale_drq': scale_drq,
                                                       'sqrt_out': sqrt_out,
                                                       'nz_b': nz_b})
                    else:
                        tensor_c_ub = tvm.compute(
                            orig_shape, lambda *indices: tensor_c[indices],
                            name='tensor_c_ub', attrs={'scale_drq': scale_drq,
                                                       'sqrt_out': sqrt_out,
                                                       'nz_b': nz_b})

                if is_fusion_mode:
                    return tensor_c_ub

                if is_fractal_a and is_fractal_b:
                    tensor_c_gm = tvm.compute(
                        orig_shape, lambda *indices: tensor_c_ub[indices],
                        name='tensor_c_gm', tag='matmul_gemv')
                else:
                    # ND out shape is dim 2, shape m is original value
                    if optmt_c == 1:
                        tensor_c_ub_fract = tvm.compute(
                            out_shape_ori, lambda i, j: tensor_c_ub[
                                j // 16, i // 16, i % 16, j % 16],
                            name='tensor_c_ub_fract')
                        tensor_c_gm = tvm.compute(
                            out_shape_ori,
                            lambda *indices: tensor_c_ub_fract[indices],
                            name='tensor_c_gm', tag='matmul_gemv')
                    else:
                        tensor_c_gm = tvm.compute(
                            out_shape_ori, lambda i, j: tensor_c_ub[
                                j // 16, i // 16, i % 16, j % 16],
                            name='tensor_c_gm', tag='matmul_gemv')

        else:
            # have batch size
            batch_shape = tensor_a.shape[0].value

            out_shape = (batch_shape, n_shape, m_shape, block_in, block_out)
            out_shape_ori = [int(batch_shape), int(m_shape_ori),
                             int(n_shape*block_out)]

            if tensor_bias is not None:
                # tensor_bias shape only be [n,], [1,n] and [1,1,n],
                # bias_shape only be [n,] for gevm and gemm
                if len(bias_shape) == 1:
                    if len(tensor_bias.shape) == 1:
                        tensor_bias_ub = tvm.compute(
                            bias_shape, lambda i: tensor_bias[i],
                            name='tensor_bias_ub')
                    elif len(tensor_bias.shape) == 2:
                        tensor_bias_ub = tvm.compute(
                            bias_shape, lambda i: tensor_bias[0, i],
                            name='tensor_bias_ub')
                    else:
                        tensor_bias_ub = tvm.compute(
                            bias_shape, lambda i: tensor_bias[0, 0, i],
                            name='tensor_bias_ub')
                elif len(bias_shape) == 3:
                    # bias_shape only be (batch, 1, n)
                    tensor_bias_ub = tvm.compute(
                        bias_shape, lambda *indices: tensor_bias[indices],
                        name='tensor_bias_ub')
            if not trans_a:
                if is_fractal_a:
                    if nz_a:
                        tensor_a_l1 = tvm.compute(
                            gm_a_shape_normalize,
                            lambda *indices: tensor_a[indices], name='tensor_a_l1')
                        tensor_a_l0a = tvm.compute(
                            (batch_shape, m_shape, km_shape, block_in, block_reduce),
                            lambda batch, i, j, k, l: tensor_a_l1[batch, i, j, l, k],
                            name='tensor_a_l0a')
                    else:
                        tensor_a_l1 = tvm.compute(
                            gm_a_shape_normalize,
                            lambda *indices: tensor_a[indices], name='tensor_a_l1')
                        tensor_a_l0a = tvm.compute(
                            gm_a_shape_normalize,
                            lambda *indices: tensor_a_l1[indices],
                            name='tensor_a_l0a')
                else:
                    tensor_a_ub = tvm.compute(
                        tensor_a.shape, lambda *i: tensor_a[i],
                        name='tensor_a_ub')

                    tensor_a_l1_shape = (
                        batch_shape, m_shape, km_shape, block_in, block_reduce)
                    if optmt_a == 1:
                        tensor_a_ub_fract = tvm.compute(
                            tensor_a_l1_shape, lambda batch, i, j, k, l:
                            tensor_a_ub[batch, i*block_in + k,
                                        j*block_reduce + l],
                            name='tensor_a_ub_fract')
                        tensor_a_l1 = tvm.compute(
                            tensor_a_l1_shape,
                            lambda *indices: tensor_a_ub_fract[indices],
                            name='tensor_a_l1')
                        tensor_a_l0a = tvm.compute(
                            tensor_a_l1_shape,
                            lambda *indices: tensor_a_l1[indices],
                            name='tensor_a_l0a')
                    else:
                        tensor_a_l1 = tvm.compute(
                            tensor_a_l1_shape, lambda batch, i, j, k, l:
                            tensor_a_ub[batch, i*block_in + k,
                                        j*block_reduce + l], name='tensor_a_l1')

                    tensor_a_l0a = tvm.compute(
                        tensor_a_l1.shape, lambda *indices:
                        tensor_a_l1[indices], name='tensor_a_l0a')
            else:
                if is_fractal_a:
                    if nz_a:
                        tensor_a_l1 = tvm.compute(
                            gm_a_shape_normalize,
                            lambda *indices: tensor_a[indices], name='tensor_a_l1')
                        tensor_a_l0a = tvm.compute(
                            (batch_shape, m_shape, km_shape, block_in, block_reduce),
                            lambda batch, i, j, k, l: tensor_a_l1[batch, j, i, k, l],
                            name='tensor_a_l0a')
                    else:
                        tensor_a_l1 = tvm.compute(
                            gm_a_shape_normalize,
                            lambda *indices: tensor_a[indices], name='tensor_a_l1')
                        tensor_a_l0a = tvm.compute(
                            (batch_shape, m_shape, km_shape, block_in,
                             block_reduce),
                            lambda batch, i, j, k, l: tensor_a_l1[
                                batch, j, i, l, k], name='tensor_a_l0a')
                else:
                    tensor_a_ub = tvm.compute(
                        tensor_a.shape, lambda *i: tensor_a[i],
                        name='tensor_a_ub')

                    if optmt_a == 1:
                        tensor_a_l1_shape = (
                            batch_shape, m_shape, km_shape, block_reduce, block_in)
                        tensor_a_ub_fract = tvm.compute(
                            tensor_a_l1_shape, lambda batch, i, j, k, l:
                            tensor_a_ub[batch, j*block_reduce + k,
                                        i*block_in + l],
                            name='tensor_a_ub_fract')
                        tensor_a_l1 = tvm.compute(
                            tensor_a_l1_shape,
                            lambda *indices: tensor_a_ub_fract[indices],
                            name='tensor_a_l1')
                        tensor_a_l0a_shape = (
                            batch_shape, m_shape, km_shape, block_in, block_reduce)
                        if block_in != cce.BLOCK_VECTOR:
                            lambda_func = lambda batch, i, j, k, l: tensor_a_l1[
                                batch, i, j, l, k]
                        else:
                            lambda_func = lambda batch, i, j, k, l: tensor_a_l1[
                                batch, i, j, k, l]
                        tensor_a_l0a = tvm.compute(
                            tensor_a_l0a_shape, lambda_func, name='tensor_a_l0a')
                    else:
                        tensor_a_l1_shape = (
                            batch_shape, km_shape, m_shape, block_reduce, block_in)
                        tensor_a_l1 = tvm.compute(
                            tensor_a_l1_shape, lambda batch, i, j, k, l:
                            tensor_a_ub[batch, i*block_reduce + k,
                                        j*block_in + l],
                            name='tensor_a_l1')
                        tensor_a_l0a_shape = (
                            batch_shape, m_shape, km_shape, block_in, block_reduce)
                        if block_in != cce.BLOCK_VECTOR:
                            lambda_func = lambda batch, i, j, k, l: tensor_a_l1[
                                batch, j, i, l, k]
                        else:
                            lambda_func = lambda batch, i, j, k, l: tensor_a_l1[
                                batch, j, i, k, l]
                        tensor_a_l0a = tvm.compute(
                            tensor_a_l0a_shape, lambda_func, name='tensor_a_l0a')

            if not trans_b:
                if is_fractal_b:
                    if nz_b:
                        tensor_b_l1 = tvm.compute(
                            tensor_b.shape, lambda *indices: tensor_b[indices],
                            name='tensor_b_l1')
                        tensor_b_l0b = tvm.compute(
                            tensor_b.shape, lambda *indices: tensor_b_l1[indices],
                            name='tensor_b_l0b')
                    else:
                        tensor_b_l1 = tvm.compute(
                            tensor_b.shape, lambda *indices: tensor_b[indices],
                            name='tensor_b_l1')
                        tensor_b_l0b = tvm.compute(
                            tensor_b.shape, lambda *indices: tensor_b_l1[indices],
                            name='tensor_b_l0b')
                else:
                    tensor_b_ub = tvm.compute(
                        tensor_b.shape, lambda *indices: tensor_b[indices],
                        name='tensor_b_ub')

                    def __get_tensor_l0b_for_not_trans_and_not_fractal():
                        if tensor_b_length in (2, 4):
                            tensor_b_l1_shape = (
                                kn_shape, n_shape, block_reduce, block_out)
                            if optmt_b == 1:
                                tensor_b_ub_fract = tvm.compute(
                                    tensor_b_l1_shape, lambda i, j, k, l:
                                    tensor_b_ub[i*block_reduce + k,
                                                j*block_out + l],
                                    name='tensor_b_ub_fract')

                                tensor_b_l1 = tvm.compute(
                                    tensor_b_l1_shape,
                                    lambda *indices: tensor_b_ub_fract[indices],
                                    name='tensor_b_l1')
                            else:
                                tensor_b_l1 = tvm.compute(
                                    tensor_b_l1_shape,
                                    lambda i, j, k, l:
                                    tensor_b_ub[i*block_reduce + k,
                                                j*block_out + l],
                                    name='tensor_b_l1')
                            tensor_b_l0b = tvm.compute(
                                (kn_shape, n_shape, block_out,
                                 block_reduce), lambda i, j, k, l: tensor_b_l1[
                                     i, j, l, k], name='tensor_b_l0b')
                        else:
                            tensor_b_l1_shape = (
                                batch_shape, kn_shape, n_shape, block_reduce, block_out)
                            if optmt_b == 1:
                                tensor_b_ub_fract = tvm.compute(
                                    tensor_b_l1_shape, lambda batch, i, j, k, l:
                                    tensor_b_ub[batch, i*block_reduce + k,
                                                j*block_out + l],
                                    name='tensor_b_ub_fract')

                                tensor_b_l1 = tvm.compute(
                                    tensor_b_l1_shape,
                                    lambda *indices: tensor_b_ub_fract[indices],
                                    name='tensor_b_l1')
                            else:
                                tensor_b_l1 = tvm.compute(
                                    tensor_b_l1_shape,
                                    lambda batch, i, j, k, l:
                                    tensor_b_ub[batch, i*block_reduce + k,
                                                j*block_out + l],
                                    name='tensor_b_l1')
                            tensor_b_l0b = tvm.compute(
                                (batch_shape, kn_shape, n_shape, block_out,
                                 block_reduce), lambda batch, i, j, k, l: tensor_b_l1[
                                     batch, i, j, l, k], name='tensor_b_l0b')
                        return tensor_b_l0b

                    tensor_b_l0b = __get_tensor_l0b_for_not_trans_and_not_fractal()
            else:
                if is_fractal_b:
                    tensor_b_l1 = tvm.compute(
                        tensor_b.shape, lambda *indices: tensor_b[indices],
                        name='tensor_b_l1')

                    def __get_tensor_l0b_for_trans_and_fractal():
                        if tensor_b_length in (2, 4):
                            tensor_b_l0b = tvm.compute(
                                (kn_shape, n_shape, block_out,
                                 block_reduce), lambda i, j, k, l: tensor_b_l1[
                                     j, i, l, k], name='tensor_b_l0b')
                        else:
                            tensor_b_l0b = tvm.compute(
                                (batch_shape, kn_shape, n_shape, block_out,
                                 block_reduce),
                                lambda batch, i, j, k, l: tensor_b_l1[
                                    batch, j, i, l, k], name='tensor_b_l0b')
                        return tensor_b_l0b

                    tensor_b_l0b = __get_tensor_l0b_for_trans_and_fractal()
                else:
                    tensor_b_ub = tvm.compute(
                        tensor_b.shape, lambda *indices: tensor_b[indices],
                        name='tensor_b_ub')

                    def __get_tensor_l0b_for_trans_and_not_fractal():
                        if tensor_b_length in (2, 4):
                            if optmt_b == 1:
                                tensor_b_l1_shape = (
                                    kn_shape, n_shape, block_out, block_reduce)
                                tensor_b_ub_fract = tvm.compute(
                                    tensor_b_l1_shape, lambda i, j, k, l:
                                    tensor_b_ub[j*block_out + k,
                                                i*block_reduce + l],
                                    name='tensor_b_ub_fract')
                                tensor_b_l1 = tvm.compute(
                                    tensor_b_l1_shape,
                                    lambda *indices: tensor_b_ub_fract[indices],
                                    name='tensor_b_l1')
                                tensor_b_l0b_shape = (
                                    kn_shape, n_shape, block_out, block_reduce)
                                tensor_b_l0b = tvm.compute(

                                    tensor_b_l0b_shape,
                                    lambda i, j, k, l: tensor_b_l1[
                                        i, j, k, l], name='tensor_b_l0b')
                            else:
                                tensor_b_l1_shape = (
                                    n_shape, kn_shape, block_out, block_reduce)
                                tensor_b_l1 = tvm.compute(
                                    tensor_b_l1_shape, lambda i, j, k, l:
                                    tensor_b_ub[i*block_out + k,
                                                j*block_reduce + l],
                                    name='tensor_b_l1')
                                tensor_b_l0b_shape = (
                                    kn_shape, n_shape, block_out, block_reduce)
                                tensor_b_l0b = tvm.compute(
                                    tensor_b_l0b_shape,
                                    lambda i, j, k, l: tensor_b_l1[
                                        j, i, k, l], name='tensor_b_l0b')
                        else:
                            if optmt_b == 1:
                                tensor_b_l1_shape = (
                                    batch_shape, kn_shape, n_shape, block_out, block_reduce)
                                tensor_b_ub_fract = tvm.compute(
                                    tensor_b_l1_shape, lambda batch, i, j, k, l:
                                    tensor_b_ub[batch, j * block_out + k,
                                                i * block_reduce + l],
                                    name='tensor_b_ub_fract')
                                tensor_b_l1 = tvm.compute(
                                    tensor_b_l1_shape,
                                    lambda *indices: tensor_b_ub_fract[indices],
                                    name='tensor_b_l1')
                                tensor_b_l0b_shape = (
                                    batch_shape, kn_shape, n_shape, block_out, block_reduce)
                                tensor_b_l0b = tvm.compute(
                                    tensor_b_l0b_shape,
                                    lambda batch, i, j, k, l: tensor_b_l1[
                                        batch, i, j, k, l], name='tensor_b_l0b')
                            else:
                                tensor_b_l1_shape = (
                                    batch_shape, n_shape, kn_shape, block_out, block_reduce)
                                tensor_b_l1 = tvm.compute(
                                    tensor_b_l1_shape, lambda batch, i, j, k, l:
                                    tensor_b_ub[batch, i * block_out + k,
                                                j * block_reduce + l],
                                    name='tensor_b_l1')
                                tensor_b_l0b_shape = (
                                    batch_shape, kn_shape, n_shape, block_out, block_reduce)
                                tensor_b_l0b = tvm.compute(
                                    tensor_b_l0b_shape,
                                    lambda batch, i, j, k, l: tensor_b_l1[
                                        batch, j, i, k, l], name='tensor_b_l0b')
                        return tensor_b_l0b

                    tensor_b_l0b = __get_tensor_l0b_for_trans_and_not_fractal()

            if block_in != cce.BLOCK_VECTOR:
                # define mad compute
                def __get_tensor_c_for_not_block_in_vector():
                    if tensor_b_length in (2, 4):
                        tensor_c = tvm.compute(
                            out_shape, lambda batch, nb, mb, mp, np: tvm.sum((
                                tensor_a_l0a[
                                    batch, mb, reduce_kb, mp, reduce_kp] *
                                tensor_b_l0b[
                                    reduce_kb, nb, np, reduce_kp]).astype(
                                        out_dtype), axis=[reduce_kb, reduce_kp]),
                            name='tensor_c', attrs={'input_order': 'positive'})
                    else:
                        tensor_c = tvm.compute(
                            out_shape, lambda batch, nb, mb, mp, np: tvm.sum((
                                tensor_a_l0a[
                                    batch, mb, reduce_kb, mp, reduce_kp] *
                                tensor_b_l0b[
                                    batch, reduce_kb, nb, np, reduce_kp]).astype(
                                        out_dtype), axis=[reduce_kb, reduce_kp]),
                            name='tensor_c', attrs={'input_order': 'positive'})
                    return tensor_c

                tensor_c = __get_tensor_c_for_not_block_in_vector()

                if tensor_bias is not None:
                    if tensor_bias.dtype == 'float16' and l0c_support_fp32 == 1:
                        # tensor_bias_ub shape only be [n,]
                        if len(tensor_bias_ub.shape) == 1:
                            tensor_bias_l0c = tvm.compute(
                                out_shape, lambda batch, i, j, k, l: topi.cast(
                                    tensor_bias_ub[i*block_out + l],
                                    dtype='float32'), name='tensor_bias_l0c')
                        elif len(tensor_bias_ub.shape) == 3:
                            tensor_bias_l0c = tvm.compute(
                                out_shape, lambda batch, i, j, k, l:
                                topi.cast(tensor_bias_ub[batch,
                                                         0, i*block_out + l],
                                          dtype='float32'),
                                name='tensor_bias_l0c')
                    else:
                        # tensor_bias_ub shape only be [n,]
                        if len(tensor_bias_ub.shape) == 1:
                            tensor_bias_l0c = tvm.compute(
                                out_shape,
                                lambda batch, i, j, k, l: tensor_bias_ub[
                                    i*block_out + l], name='tensor_bias_l0c')
                        elif len(tensor_bias_ub.shape) == 3:
                            tensor_bias_l0c = tvm.compute(
                                out_shape, lambda batch, i, j, k, l:
                                tensor_bias_ub[batch,
                                               0, i*block_out + l],
                                name='tensor_bias_l0c')
                    tensor_c_add_bias = tvm.compute(
                        out_shape,
                        lambda *indices: tensor_bias_l0c[indices] + tensor_c(
                            *indices), name='tensor_c_add_bias')

                    if scale_drq_tensor is not None:
                        tensor_c_ub = tvm.compute(
                            out_shape,
                            lambda *indices: tensor_c_add_bias[indices] *
                            scale_drq_tensor[0], name='tensor_c_ub',
                            attrs={'scale_drq': scale_drq,
                                   'sqrt_out': sqrt_out,
                                   'nz_b': nz_b})
                    elif dst_dtype == 'float16' and l0c_support_fp32 == 1:
                        tensor_c_ub = tvm.compute(
                            out_shape, lambda *indices: topi.cast(
                                tensor_c_add_bias[indices], dtype='float16'),
                            name='tensor_c_ub', attrs={'scale_drq': scale_drq,
                                                       'sqrt_out': sqrt_out,
                                                       'nz_b': nz_b})
                    else:
                        tensor_c_ub = tvm.compute(
                            out_shape,
                            lambda *indices: tensor_c_add_bias[indices],
                            name='tensor_c_ub', attrs={'scale_drq': scale_drq,
                                                       'sqrt_out': sqrt_out,
                                                       'nz_b': nz_b})
                else:
                    if scale_drq_tensor is not None:
                        tensor_c_ub = tvm.compute(
                            out_shape, lambda *indices: tensor_c[indices] *
                            scale_drq_tensor[0], name='tensor_c_ub',
                            attrs={'scale_drq': scale_drq,
                                   'sqrt_out': sqrt_out,
                                   'nz_b': nz_b})
                    elif dst_dtype == 'float16' and l0c_support_fp32 == 1:
                        tensor_c_ub = tvm.compute(
                            out_shape,
                            lambda *indices: topi.cast(tensor_c[indices],
                                                       dtype='float16'),
                            name='tensor_c_ub', attrs={'scale_drq': scale_drq,
                                                       'sqrt_out': sqrt_out,
                                                       'nz_b': nz_b})
                    else:
                        tensor_c_ub = tvm.compute(
                            out_shape, lambda *indices: tensor_c[indices],
                            name='tensor_c_ub', attrs={'scale_drq': scale_drq,
                                                       'sqrt_out': sqrt_out,
                                                       'nz_b': nz_b})

                if is_fractal_a and is_fractal_b:
                    tensor_c_gm = tvm.compute(
                        out_shape, lambda *indices: tensor_c_ub[indices],
                        name='tensor_c_gm', tag="matmul")
                else:
                    # ND out shape is dim 2, shape m is original value
                    if optmt_c == 1:
                        tensor_c_ub_fract = tvm.compute(
                            out_shape_ori, lambda batch, i, j: tensor_c_ub[
                                batch, j // 16, i // 16, i % 16, j % 16],
                            name='tensor_c_ub_fract')
                        tensor_c_gm = tvm.compute(
                            out_shape_ori,
                            lambda *indices: tensor_c_ub_fract[indices],
                            name='tensor_c_gm', tag='matmul')
                    else:
                        tensor_c_gm = tvm.compute(
                            out_shape_ori, lambda batch, i, j: tensor_c_ub[
                                batch, j // 16, i // 16, i % 16, j % 16],
                            name='tensor_c_gm', tag="matmul")
            else:
                # gevm mode, define mad
                def __get_tensor_c_for_block_in_vector():
                    if tensor_b_length in (2, 4):
                        tensor_c = tvm.compute(
                            (batch_shape, n_shape, m_shape, block_out, block_out),
                            lambda batch, nb, mb, mp, np:
                            tvm.sum((tensor_a_l0a[batch, mb, reduce_kb, mp, reduce_kp] *
                                     tensor_b_l0b[reduce_kb, nb, np, reduce_kp])
                                    .astype(out_dtype),
                                    axis=[reduce_kb, reduce_kp]),
                            name='tensor_c', attrs={'input_order': 'positive'})
                    else:
                        tensor_c = tvm.compute(
                            (batch_shape, n_shape, m_shape, block_out, block_out),
                            lambda batch, nb, mb, mp, np:
                            tvm.sum((tensor_a_l0a[batch, mb, reduce_kb, mp, reduce_kp] *
                                     tensor_b_l0b[batch, reduce_kb, nb, np, reduce_kp])
                                    .astype(out_dtype),
                                    axis=[reduce_kb, reduce_kp]),
                            name='tensor_c', attrs={'input_order': 'positive'})
                    return tensor_c

                tensor_c = __get_tensor_c_for_block_in_vector()

                # define reduce
                orig_shape = shape_to_list(tensor_c.shape)
                orig_shape[-2] = block_in

                if tensor_bias is not None:
                    # tensor_bias_ub just is [n,] and [batch, 1, n], no [1, n]
                    if tensor_bias.dtype == 'float16' and l0c_support_fp32 == 1:
                        # tensor_bias_ub shape only be [n,] and [batch, 1, n]
                        if len(tensor_bias_ub.shape) == 1:
                            tensor_bias_l0c = tvm.compute(
                                orig_shape, lambda batch, i, j, k, l:
                                topi.cast(tensor_bias_ub[i*block_out + l],
                                          dtype='float32'),
                                name='tensor_bias_l0c')
                        elif len(tensor_bias_ub.shape) == 3:
                            tensor_bias_l0c = tvm.compute(
                                orig_shape, lambda batch, i, j, k, l:
                                topi.cast(tensor_bias_ub[batch, 0, i *
                                                         block_out + l],
                                          dtype='float32'),
                                name='tensor_bias_l0c')
                    else:
                        # tensor_bias_ub shape only be [n,] and [batch, 1, n]
                        if len(tensor_bias_ub.shape) == 1:
                            tensor_bias_l0c = tvm.compute(
                                orig_shape, lambda batch, i, j, k, l:
                                tensor_bias_ub[i*block_out + l],
                                name='tensor_bias_l0c')
                        elif len(tensor_bias_ub.shape) == 3:
                            tensor_bias_l0c = tvm.compute(
                                orig_shape,
                                lambda batch, i, j, k, l: tensor_bias_ub[
                                    batch, 0, i*block_out + l],
                                name='tensor_bias_l0c')

                    tensor_c_add_bias = tvm.compute(
                        tensor_c.shape, lambda *indices:
                        tensor_bias_l0c[indices] + tensor_c[indices],
                        name='tensor_c_add_bias')
                    if scale_drq_tensor is not None:
                        tensor_c_ub = tvm.compute(
                            orig_shape,
                            lambda *indices: tensor_c_add_bias[indices] *
                            scale_drq_tensor[0], name='tensor_c_ub',
                            attrs={'scale_drq': scale_drq,
                                   'sqrt_out': sqrt_out,
                                   'nz_b': nz_b})
                    elif dst_dtype == 'float16' and l0c_support_fp32 == 1:
                        tensor_c_ub = tvm.compute(
                            orig_shape, lambda *indices: topi.cast(
                                tensor_c_add_bias[indices], dtype='float16'),
                            name='tensor_c_ub', attrs={'scale_drq': scale_drq,
                                                       'sqrt_out': sqrt_out,
                                                       'nz_b': nz_b})
                    else:
                        tensor_c_ub = tvm.compute(
                            orig_shape,
                            lambda *indices: tensor_c_add_bias[indices],
                            name='tensor_c_ub', attrs={'scale_drq': scale_drq,
                                                       'sqrt_out': sqrt_out,
                                                       'nz_b': nz_b})
                else:
                    if scale_drq_tensor is not None:
                        tensor_c_ub = tvm.compute(
                            orig_shape, lambda *indices: tensor_c[indices] *
                            scale_drq_tensor[0], name='tensor_c_ub',
                            attrs={'scale_drq': scale_drq,
                                   'sqrt_out': sqrt_out,
                                   'nz_b': nz_b})
                    elif dst_dtype == 'float16' and l0c_support_fp32 == 1:
                        tensor_c_ub = tvm.compute(
                            orig_shape,
                            lambda *indices: topi.cast(tensor_c[indices],
                                                       dtype='float16'),
                            name='tensor_c_ub', attrs={'scale_drq': scale_drq,
                                                       'sqrt_out': sqrt_out,
                                                       'nz_b': nz_b})
                    else:
                        tensor_c_ub = tvm.compute(
                            orig_shape, lambda *indices: tensor_c[indices],
                            name='tensor_c_ub', attrs={'scale_drq': scale_drq,
                                                       'sqrt_out': sqrt_out,
                                                       'nz_b': nz_b})
                if is_fractal_a and is_fractal_b:
                    tensor_c_gm = tvm.compute(
                        orig_shape, lambda *indices: tensor_c_ub[indices],
                        name='tensor_c_gm', tag='matmul_gemv')
                else:
                    # ND out shape is dim 2, shape m is original value
                    if optmt_c == 1:
                        tensor_c_ub_fract = tvm.compute(
                            out_shape_ori, lambda batch, i, j: tensor_c_ub[
                                batch, j // 16, i // 16, i % 16, j % 16],
                            name='tensor_c_ub_fract')
                        tensor_c_gm = tvm.compute(
                            out_shape_ori,
                            lambda *indices: tensor_c_ub_fract[indices],
                            name='tensor_c_gm', tag='matmul_gemv')
                    else:
                        tensor_c_gm = tvm.compute(
                            out_shape_ori, lambda batch, i, j: tensor_c_ub[
                                batch, j // 16, i // 16, i % 16, j % 16],
                            name='tensor_c_gm', tag="matmul_gemv")
    else:
        # gemv,c=A*B=(B`A`)`,so B`A` is gevm
        if tensor_a_length in (2, 4):
            if trans_a:
                if is_fractal_a:
                    tensor_a_l1 = tvm.compute(
                        gm_a_shape_normalize,
                        lambda *indices: tensor_a[indices], name='tensor_a_l1')
                    tensor_a_l0a = tvm.compute(
                        (km_shape, m_shape, block_in, block_reduce),
                        lambda i, j, l, k: tensor_a_l1[i, j, k, l],
                        name='tensor_a_l0a')
                else:
                    tensor_a_ub = tvm.compute(
                        tensor_a.shape,
                        lambda *indices: tensor_a[indices], name='tensor_a_ub')
                    tensor_a_l1_shape = (
                        km_shape, m_shape, block_reduce, block_in)
                    if optmt_a == 1:
                        tensor_a_ub_fract = tvm.compute(
                            tensor_a_l1_shape, lambda i, j, k, l:
                            tensor_a_ub[i*block_reduce + k, j*block_in + l],
                            name='tensor_a_ub_fract')
                        tensor_a_l1 = tvm.compute(
                            tensor_a_l1_shape,
                            lambda *indices: tensor_a_ub_fract[indices],
                            name='tensor_a_l1')
                    else:
                        tensor_a_l1 = tvm.compute(
                            tensor_a_l1_shape, lambda i, j, k, l: tensor_a_ub[
                                i*block_reduce + k, j*block_in + l],
                            name='tensor_a_l1')

                    tensor_a_l0a = tvm.compute(
                        (km_shape, m_shape, block_in, block_reduce),
                        lambda i, j, k, l: tensor_a_l1[i, j, l, k],
                        name='tensor_a_l0a')
            else:
                if is_fractal_a:
                    tensor_a_l1 = tvm.compute(
                        gm_a_shape_normalize,
                        lambda *indices: tensor_a[indices], name='tensor_a_l1')
                    tensor_a_l0a = tvm.compute(
                        (km_shape, m_shape, block_in, block_reduce),
                        lambda i, j, k, l: tensor_a_l1[j, i, k, l],
                        name='tensor_a_l0a')
                else:
                    tensor_a_ub = tvm.compute(
                        tensor_a.shape,
                        lambda *indices: tensor_a[indices], name='tensor_a_ub')
                    if optmt_a == 1:
                        tensor_a_l1_shape = (
                            km_shape, m_shape, block_in, block_reduce)
                        tensor_a_ub_fract = tvm.compute(
                            tensor_a_l1_shape, lambda i, j, k, l:
                            tensor_a_ub[j*block_in + k, i*block_reduce + l],
                            name='tensor_a_ub_fract')
                        tensor_a_l1 = tvm.compute(
                            tensor_a_l1_shape,
                            lambda *indices: tensor_a_ub_fract[indices],
                            name='tensor_a_l1')
                        tensor_a_l0a = tvm.compute(
                            (km_shape, m_shape, block_in, block_reduce),
                            lambda i, j, k, l: tensor_a_l1[i, j, k, l],
                            name='tensor_a_l0a')
                    else:
                        tensor_a_l1_shape = (
                            m_shape, km_shape, block_in, block_reduce)
                        tensor_a_l1 = tvm.compute(
                            tensor_a_l1_shape, lambda i, j, k, l:
                            tensor_a_ub[i*block_in + k, j*block_reduce + l],
                            name='tensor_a_l1')
                        tensor_a_l0a = tvm.compute(
                            (km_shape, m_shape, block_in, block_reduce),
                            lambda i, j, k, l: tensor_a_l1[j, i, k, l],
                            name='tensor_a_l0a')

            if trans_b:
                if is_fractal_b:
                    tensor_b_l1 = tvm.compute(
                        tensor_b.shape, lambda *indices: tensor_b[indices],
                        name='tensor_b_l1')
                    tensor_b_l0b = tvm.compute(
                        (n_shape, kn_shape, block_out, block_reduce),
                        lambda i, j, k, l: tensor_b_l1[i, j, l, k],
                        name='tensor_b_l0b')
                else:
                    tensor_b_ub = tvm.compute(
                        tensor_b.shape, lambda *i: tensor_b[i],
                        name='tensor_b_ub')
                    if optmt_b == 1:
                        tensor_b_l1_shape = (
                            n_shape, kn_shape, block_out, block_reduce)
                        tensor_b_ub_fract = tvm.compute(
                            tensor_b_l1_shape, lambda i, j, k, l:
                            tensor_b_ub[i*block_out + k, j*block_reduce + l],
                            name='tensor_b_ub_fract')
                        tensor_b_l1 = tvm.compute(
                            tensor_b_l1_shape,
                            lambda *indices: tensor_b_ub_fract[indices],
                            name='tensor_b_l1')
                        tensor_b_l0b = tvm.compute(
                            tensor_b_l1_shape,
                            lambda i, j, k, l: tensor_b_l1[i, j, k, l],
                            name='tensor_b_l0b')
                    else:
                        tensor_b_l1_shape = (
                            n_shape, kn_shape, block_out, block_reduce)
                        tensor_b_l1 = tvm.compute(
                            tensor_b_l1_shape, lambda i, j, k, l:
                            tensor_b_ub[i*block_out + k, j*block_reduce + l],
                            name='tensor_b_l1')
                        # (n_shape, kn_shape, block_reduce, block_out)
                        tensor_b_l0b = tvm.compute(
                            tensor_b_l1_shape,
                            lambda i, j, k, l: tensor_b_l1[i, j, k, l],
                            name='tensor_b_l0b')
            else:
                if is_fractal_b:
                    tensor_b_l1 = tvm.compute(
                        tensor_b.shape, lambda *indices: tensor_b[indices],
                        name='tensor_b_l1')
                    tensor_b_l0b = tvm.compute(
                        (n_shape, kn_shape, block_out, block_reduce),
                        lambda i, j, k, l: tensor_b_l1[j, i, k, l],
                        name='tensor_b_l0b')
                else:
                    tensor_b_ub = tvm.compute(
                        tensor_b.shape, lambda *i: tensor_b[i],
                        name='tensor_b_ub')
                    if optmt_b == 1:
                        tensor_b_l1_shape = (
                            n_shape, kn_shape, block_reduce, block_out)
                        tensor_b_ub_fract = tvm.compute(
                            tensor_b_l1_shape, lambda i, j, k, l:
                            tensor_b_ub[j*block_reduce + k, i*block_out + l],
                            name='tensor_b_ub_fract')
                        tensor_b_l1 = tvm.compute(
                            tensor_b_l1_shape,
                            lambda *indices: tensor_b_ub_fract[indices],
                            name='tensor_b_l1')
                        tensor_b_l0b_shape = (
                            n_shape, kn_shape, block_out, block_reduce)
                        tensor_b_l0b = tvm.compute(
                            tensor_b_l0b_shape,
                            lambda i, j, k, l: tensor_b_l1[i, j, l, k],
                            name='tensor_b_l0b')
                    else:
                        tensor_b_l1_shape = (
                            kn_shape, n_shape, block_reduce, block_out)
                        tensor_b_l1 = tvm.compute(
                            tensor_b_l1_shape, lambda i, j, k, l:
                            tensor_b_ub[i*block_reduce + k, j*block_out + l],
                            name='tensor_b_l1')
                        tensor_b_l0b_shape = (
                            n_shape, kn_shape, block_out, block_reduce)
                        tensor_b_l0b = tvm.compute(
                            tensor_b_l0b_shape,
                            lambda i, j, k, l: tensor_b_l1[j, i, l, k],
                            name='tensor_b_l0b')

            # define reduce
            orig_shape = [int(m_shape), int(n_shape), int(block_in),
                          int(block_in)]
            orig_shape[-2] = block_out
            out_shape_ori = [int(n_shape*block_out), int(m_shape*block_in)]
            # define mad
            tensor_c = tvm.compute(
                (m_shape, n_shape, block_in, block_in),
                lambda nb, mb, mp, np: tvm.sum(
                    (tensor_b_l0b[mb, reduce_kb, mp, reduce_kp]*tensor_a_l0a[
                        reduce_kb, nb, np, reduce_kp]).astype(out_dtype),
                    axis=[reduce_kb, reduce_kp]), name='tensor_c',
                attrs={'input_order': 'negative'})

            if tensor_bias is not None:
                tensor_bias_ub = tvm.compute(
                    tensor_bias.shape, lambda *indices: tensor_bias[indices],
                    name='tensor_bias_ub')
                # bias shape only support [m,1]
                if tensor_bias.dtype == 'float16' and l0c_support_fp32 == 1:
                    tensor_bias_l0c = tvm.compute(
                        orig_shape, lambda i, j, k, l:
                        topi.cast(tensor_bias_ub[i*block_in + l, 0],
                                  dtype='float32'),
                        name='tensor_bias_l0c')
                else:
                    tensor_bias_l0c = tvm.compute(
                        orig_shape, lambda i, j, k, l:
                        tensor_bias_ub[i*block_in + l, 0],
                        name='tensor_bias_l0c')

                tensor_c_add_bias = tvm.compute(
                    tensor_c.shape, lambda *indices:
                    tensor_bias_l0c[indices] + tensor_c[indices],
                    name='tensor_c_add_bias')
                if scale_drq_tensor is not None:
                    tensor_c_ub = tvm.compute(
                        orig_shape,
                        lambda *indices: tensor_c_add_bias[indices]*
                        scale_drq_tensor[0], name='tensor_c_ub',
                        attrs={'scale_drq': scale_drq, 'sqrt_out': sqrt_out, 'nz_b': nz_b})
                elif dst_dtype == 'float16' and l0c_support_fp32 == 1:
                    tensor_c_ub = tvm.compute(
                        orig_shape,
                        lambda *indices: topi.cast(tensor_c_add_bias[indices],
                                                   dtype='float16'),
                        name='tensor_c_ub',
                        attrs={'scale_drq': scale_drq, 'sqrt_out': sqrt_out, 'nz_b': nz_b})
                else:
                    tensor_c_ub = tvm.compute(
                        orig_shape,
                        lambda *indices: tensor_c_add_bias[indices],
                        name='tensor_c_ub',
                        attrs={'scale_drq': scale_drq, 'sqrt_out': sqrt_out, 'nz_b': nz_b})
            else:
                if scale_drq_tensor is not None:
                    tensor_c_ub = tvm.compute(
                        orig_shape,
                        lambda *indices: tensor_c[indices]*scale_drq_tensor[
                            0], name='tensor_c_ub',
                        attrs={'scale_drq': scale_drq, 'sqrt_out': sqrt_out, 'nz_b': nz_b})
                elif dst_dtype == 'float16' and l0c_support_fp32 == 1:
                    tensor_c_ub = tvm.compute(
                        orig_shape,
                        lambda *indices: topi.cast(tensor_c[indices],
                                                   dtype='float16'),
                        name='tensor_c_ub',
                        attrs={'scale_drq': scale_drq, 'sqrt_out': sqrt_out, 'nz_b': nz_b})
                else:
                    tensor_c_ub = tvm.compute(
                        orig_shape, lambda *indices: tensor_c[indices],
                        name='tensor_c_ub',
                        attrs={'scale_drq': scale_drq, 'sqrt_out': sqrt_out, 'nz_b': nz_b})
            if is_fractal_a and is_fractal_b:
                tensor_c_gm = tvm.compute(
                    orig_shape, lambda *indices: tensor_c_ub[indices],
                    name='tensor_c_gm', tag='matmul_gemv')
            else:
                # ND out shape is dim 2, shape m is original value
                if optmt_c == 1:
                    tensor_c_ub_fract = tvm.compute(
                        out_shape_ori, lambda i, j:
                        tensor_c_ub[j // 16, i // 16, i % 16, j % 16],
                        name='tensor_c_ub_fract')
                    tensor_c_gm = tvm.compute(
                        out_shape_ori,
                        lambda *indices: tensor_c_ub_fract[indices],
                        name='tensor_c_gm', tag='matmul_gemv')
                else:
                    tensor_c_gm = tvm.compute(
                        out_shape_ori, lambda i, j:
                        tensor_c_ub[j // 16, i // 16, i % 16, j % 16],
                        name='tensor_c_gm', tag="matmul_gemv")

        else:
            # have batch size
            batch_shape = tensor_a.shape[0].value
            out_shape_ori = [batch_shape, int(n_shape*block_out),
                             int(m_shape_ori)]

            if trans_a:
                if is_fractal_a:
                    tensor_a_l1 = tvm.compute(
                        gm_a_shape_normalize,
                        lambda *indices: tensor_a[indices], name='tensor_a_l1')
                    tensor_a_l0a = tvm.compute(
                        (batch_shape, km_shape, m_shape, block_in,
                         block_reduce), lambda batch, i, j, l, k:
                        tensor_a_l1[batch, i, j, k, l], name='tensor_a_l0a')
                else:
                    tensor_a_ub = tvm.compute(
                        tensor_a.shape,
                        lambda *indices: tensor_a[indices], name='tensor_a_ub')
                    tensor_a_l1_shape = (
                        batch_shape, km_shape, m_shape, block_reduce, block_in)
                    if optmt_a == 1:
                        tensor_a_ub_fract = tvm.compute(
                            tensor_a_l1_shape, lambda batch, i, j, k, l:
                            tensor_a_ub[
                                batch, i*block_reduce + k, j*block_in + l],
                            name='tensor_a_ub_fract')

                        tensor_a_l1 = tvm.compute(
                            tensor_a_l1_shape,
                            lambda *indices: tensor_a_ub_fract[indices],
                            name='tensor_a_l1')
                    else:
                        tensor_a_l1 = tvm.compute(
                            tensor_a_l1_shape, lambda batch, i, j, k, l:
                            tensor_a_ub[
                                batch, i*block_reduce + k, j*block_in + l],
                            name='tensor_a_l1')

                    tensor_a_l0a = tvm.compute(
                        (batch_shape, km_shape, m_shape, block_in,
                         block_reduce),
                        lambda batch, i, j, k, l: tensor_a_l1[
                            batch, i, j, l, k], name='tensor_a_l0a')
            else:
                if is_fractal_a:
                    tensor_a_l1 = tvm.compute(
                        gm_a_shape_normalize,
                        lambda *indices: tensor_a[indices], name='tensor_a_l1')
                    tensor_a_l0a = tvm.compute(
                        (batch_shape, km_shape, m_shape, block_in,
                         block_reduce),
                        lambda batch, i, j, k, l: tensor_a_l1[
                            batch, j, i, k, l], name='tensor_a_l0a')
                else:
                    tensor_a_ub = tvm.compute(
                        tensor_a.shape, lambda *i: tensor_a[i],
                        name='tensor_a_ub')
                    if optmt_a == 1:
                        tensor_a_l1_shape = (
                            batch_shape, km_shape, m_shape, block_in, block_reduce)
                        tensor_a_ub_fract = tvm.compute(
                            tensor_a_l1_shape, lambda batch, i, j, k, l:
                            tensor_a_ub[
                                batch, j*block_in + k, i*block_reduce + l],
                            name='tensor_a_ub_fract')
                        tensor_a_l1 = tvm.compute(
                            tensor_a_l1_shape,
                            lambda *indices: tensor_a_ub_fract[indices],
                            name='tensor_a_l1')
                        tensor_a_l0a_shape = (
                            batch_shape, km_shape, m_shape, block_in, block_reduce)
                        tensor_a_l0a = tvm.compute(
                            tensor_a_l0a_shape,
                            lambda batch, i, j, k, l: tensor_a_l1[
                                batch, i, j, k, l], name='tensor_a_l0a')
                    else:
                        tensor_a_l1_shape = (
                            batch_shape, m_shape, km_shape, block_in, block_reduce)
                        tensor_a_l1 = tvm.compute(
                            tensor_a_l1_shape, lambda batch, i, j, k, l:
                            tensor_a_ub[
                                batch, i*block_in + k, j*block_reduce + l],
                            name='tensor_a_l1')
                        tensor_a_l0a_shape = (
                            batch_shape, km_shape, m_shape, block_in, block_reduce)
                        tensor_a_l0a = tvm.compute(
                            tensor_a_l0a_shape,
                            lambda batch, i, j, k, l: tensor_a_l1[
                                batch, j, i, k, l], name='tensor_a_l0a')
            if trans_b:
                if is_fractal_b:
                    tensor_b_l1 = tvm.compute(
                        tensor_b.shape, lambda *indices: tensor_b[indices],
                        name='tensor_b_l1')

                    def __get_tensor_l0b_for_trans_and_fractal():
                        if tensor_b_length in (2, 4):
                            tensor_b_l0b = tvm.compute(
                                (n_shape, kn_shape, block_out,
                                 block_reduce), lambda i, j, k, l: tensor_b_l1[
                                     i, j, l, k], name='tensor_b_l0b')
                        else:
                            tensor_b_l0b = tvm.compute(
                                (batch_shape, n_shape, kn_shape, block_out,
                                 block_reduce), lambda batch, i, j, k, l: tensor_b_l1[
                                     batch, i, j, l, k], name='tensor_b_l0b')
                        return tensor_b_l0b

                    tensor_b_l0b = __get_tensor_l0b_for_trans_and_fractal()
                else:
                    tensor_b_ub = tvm.compute(
                        tensor_b.shape, lambda *indices: tensor_b[indices],
                        name='tensor_b_ub')

                    def __get_tensor_l0b_for_trans_and_not_fractal():
                        if tensor_b_length in (2, 4):
                            tensor_b_l1_shape = (
                                n_shape, kn_shape, block_out, block_reduce)
                            if optmt_b == 1:
                                tensor_b_ub_fract = tvm.compute(
                                    tensor_b_l1_shape, lambda i, j, k, l:
                                    tensor_b_ub[
                                        i*block_out + k, j*block_reduce + l],
                                    name='tensor_b_ub_fract')
                                tensor_b_l1 = tvm.compute(
                                    tensor_b_l1_shape,
                                    lambda *indices: tensor_b_ub_fract[indices],
                                    name='tensor_b_l1')
                            else:
                                tensor_b_l1 = tvm.compute(
                                    tensor_b_l1_shape, lambda i, j, k, l:
                                    tensor_b_ub[
                                        i*block_out + k, j*block_reduce + l],
                                    name='tensor_b_l1')
                            tensor_b_l0b = tvm.compute(
                                tensor_b_l1_shape,
                                lambda i, j, k, l: tensor_b_l1[
                                    i, j, k, l], name='tensor_b_l0b')
                        else:
                            tensor_b_l1_shape = (
                                batch_shape, n_shape, kn_shape, block_out, block_reduce)
                            if optmt_b == 1:
                                tensor_b_ub_fract = tvm.compute(
                                    tensor_b_l1_shape, lambda batch, i, j, k, l:
                                    tensor_b_ub[
                                        batch, i*block_out + k, j*block_reduce + l],
                                    name='tensor_b_ub_fract')
                                tensor_b_l1 = tvm.compute(
                                    tensor_b_l1_shape,
                                    lambda *indices: tensor_b_ub_fract[indices],
                                    name='tensor_b_l1')
                            else:
                                tensor_b_l1 = tvm.compute(
                                    tensor_b_l1_shape, lambda batch, i, j, k, l:
                                    tensor_b_ub[
                                        batch, i*block_out + k, j*block_reduce + l],
                                    name='tensor_b_l1')
                            tensor_b_l0b = tvm.compute(
                                tensor_b_l1_shape,
                                lambda batch, i, j, k, l: tensor_b_l1[
                                    batch, i, j, k, l], name='tensor_b_l0b')
                        return tensor_b_l0b

                    tensor_b_l0b = __get_tensor_l0b_for_trans_and_not_fractal()
            else:
                if is_fractal_b:
                    tensor_b_l1 = tvm.compute(
                        tensor_b.shape, lambda *indices: tensor_b[indices],
                        name='tensor_b_l1')

                    def __get_tensor_l0b_for_not_trans_and_fractal():
                        if tensor_b_length in (2, 4):
                            tensor_b_l0b = tvm.compute(
                                (n_shape, kn_shape, block_out,
                                 block_reduce), lambda i, j, k, l: tensor_b_l1[
                                     j, i, k, l], name='tensor_b_l0b')
                        else:
                            tensor_b_l0b = tvm.compute(
                                (batch_shape, n_shape, kn_shape, block_out,
                                 block_reduce), lambda batch, i, j, k, l: tensor_b_l1[
                                     batch, j, i, k, l], name='tensor_b_l0b')
                        return tensor_b_l0b

                    tensor_b_l0b = __get_tensor_l0b_for_not_trans_and_fractal()
                else:
                    tensor_b_ub = tvm.compute(
                        tensor_b.shape, lambda *indices: tensor_b[indices],
                        name='tensor_b_ub')

                    def __get_tensor_l0b_for_not_trans_and_not_fractal():
                        if tensor_b_length in (2, 4):
                            if optmt_b == 1:
                                tensor_b_l1_shape = (
                                    n_shape, kn_shape, block_reduce, block_out)
                                tensor_b_ub_fract = tvm.compute(
                                    tensor_b_l1_shape, lambda i, j, k, l:
                                    tensor_b_ub[
                                        j*block_reduce + k, i*block_out + l],
                                    name='tensor_b_ub_fract')
                                tensor_b_l1 = tvm.compute(
                                    tensor_b_l1_shape,
                                    lambda *indices: tensor_b_ub_fract[indices],
                                    name='tensor_b_l1')
                                tensor_b_l0b_shape = (
                                    n_shape, kn_shape, block_out, block_reduce)
                                tensor_b_l0b = tvm.compute(
                                    tensor_b_l0b_shape,
                                    lambda i, j, k, l: tensor_b_l1[
                                        i, j, l, k], name='tensor_b_l0b')
                            else:
                                tensor_b_l1_shape = (
                                    kn_shape, n_shape, block_reduce, block_out)
                                tensor_b_l1 = tvm.compute(
                                    tensor_b_l1_shape, lambda i, j, k, l:
                                    tensor_b_ub[
                                        i*block_reduce + k, j*block_out + l],
                                    name='tensor_b_l1')
                                tensor_b_l0b_shape = (
                                    n_shape, kn_shape, block_out, block_reduce)
                                tensor_b_l0b = tvm.compute(
                                    tensor_b_l0b_shape,
                                    lambda i, j, k, l: tensor_b_l1[
                                        j, i, l, k], name='tensor_b_l0b')
                        else:
                            if optmt_b == 1:
                                tensor_b_l1_shape = (
                                    batch_shape, n_shape, kn_shape, block_reduce, block_out)
                                tensor_b_ub_fract = tvm.compute(
                                    tensor_b_l1_shape, lambda batch, i, j, k, l:
                                    tensor_b_ub[
                                        batch, j*block_reduce + k, i*block_out + l],
                                    name='tensor_b_ub_fract')
                                tensor_b_l1 = tvm.compute(
                                    tensor_b_l1_shape,
                                    lambda *indices: tensor_b_ub_fract[indices],
                                    name='tensor_b_l1')
                                tensor_b_l0b_shape = (
                                    batch_shape, n_shape, kn_shape, block_out, block_reduce)
                                tensor_b_l0b = tvm.compute(
                                    tensor_b_l0b_shape,
                                    lambda batch, i, j, k, l: tensor_b_l1[
                                        batch, i, j, l, k], name='tensor_b_l0b')
                            else:
                                tensor_b_l1_shape = (
                                    batch_shape, kn_shape, n_shape, block_reduce, block_out)
                                tensor_b_l1 = tvm.compute(
                                    tensor_b_l1_shape, lambda batch, i, j, k, l:
                                    tensor_b_ub[
                                        batch, i*block_reduce + k, j*block_out + l],
                                    name='tensor_b_l1')
                                tensor_b_l0b_shape = (
                                    batch_shape, n_shape, kn_shape, block_out, block_reduce)
                                tensor_b_l0b = tvm.compute(
                                    tensor_b_l0b_shape,
                                    lambda batch, i, j, k, l: tensor_b_l1[
                                        batch, j, i, l, k], name='tensor_b_l0b')
                        return tensor_b_l0b

                    tensor_b_l0b = __get_tensor_l0b_for_not_trans_and_not_fractal()

            # define mad
            def __get_tensor_c():
                if tensor_b_length in (2, 4):
                    tensor_c = tvm.compute(
                        (batch_shape, m_shape, n_shape, block_in, block_in),
                        lambda batch, nb, mb, mp, np: tvm.sum(
                            (tensor_b_l0b[mb, reduce_kb, mp, reduce_kp] *
                             tensor_a_l0a[batch, reduce_kb, nb, np, reduce_kp]).astype
                            (out_dtype), axis=[reduce_kb, reduce_kp]),
                        name='tensor_c', attrs={'input_order': 'negative'})
                else:
                    tensor_c = tvm.compute(
                        (batch_shape, m_shape, n_shape, block_in, block_in),
                        lambda batch, nb, mb, mp, np: tvm.sum(
                            (tensor_b_l0b[batch, mb, reduce_kb, mp, reduce_kp] *
                             tensor_a_l0a[batch, reduce_kb, nb, np, reduce_kp]).astype
                            (out_dtype), axis=[reduce_kb, reduce_kp]),
                        name='tensor_c', attrs={'input_order': 'negative'})
                return tensor_c

            tensor_c = __get_tensor_c()
            # define reduce
            orig_shape = shape_to_list(tensor_c.shape)
            orig_shape[-2] = block_out

            if tensor_bias is not None:
                tensor_bias_ub = tvm.compute(
                    tensor_bias.shape, lambda *indices: tensor_bias[indices],
                    name='tensor_bias_ub')

                # bias shape support [m,1] or [1,m,1] or [b,m,1]
                if len(bias_shape) == 2:
                    if len(tensor_bias.shape) == 2:
                        if tensor_bias.dtype == 'float16' and \
                                l0c_support_fp32 == 1:
                            tensor_bias_l0c = tvm.compute(
                                orig_shape,
                                lambda batch, i, j, k, l: topi.cast(
                                    tensor_bias_ub[i*block_in + l, 0],
                                    dtype='float32'), name='tensor_bias_l0c')
                        else:
                            tensor_bias_l0c = tvm.compute(
                                orig_shape,
                                lambda batch, i, j, k, l: tensor_bias_ub[
                                    i*block_in + l, 0],
                                name='tensor_bias_l0c')
                    else:
                        if tensor_bias.dtype == 'float16' and \
                                l0c_support_fp32 == 1:
                            tensor_bias_l0c = tvm.compute(
                                orig_shape,
                                lambda batch, i, j, k, l: topi.cast(
                                    tensor_bias_ub[0, i*block_in + l, 0],
                                    dtype='float32'), name='tensor_bias_l0c')
                        else:
                            tensor_bias_l0c = tvm.compute(
                                orig_shape,
                                lambda batch, i, j, k, l: tensor_bias_ub[
                                    0, i*block_in + l, 0],
                                name='tensor_bias_l0c')
                elif len(bias_shape) == 3:
                    if tensor_bias.dtype == 'float16' and l0c_support_fp32 == 1:
                        tensor_bias_l0c = tvm.compute(
                            orig_shape, lambda batch, i, j, k, l: topi.cast(
                                tensor_bias_ub[batch, i*block_in + l, 0],
                                dtype='float32'), name='tensor_bias_l0c')
                    else:
                        tensor_bias_l0c = tvm.compute(
                            orig_shape,
                            lambda batch, i, j, k, l: tensor_bias_ub[
                                batch, i*block_in + l, 0],
                            name='tensor_bias_l0c')

                tensor_c_add_bias = tvm.compute(
                    tensor_c.shape,
                    lambda *indices: tensor_bias_l0c[indices] + tensor_c(
                        *indices), name='tensor_c_add_bias')
                if scale_drq_tensor is not None:
                    tensor_c_ub = tvm.compute(
                        orig_shape,
                        lambda *indices: tensor_c_add_bias[indices] *
                        scale_drq_tensor[0], name='tensor_c_ub',
                        attrs={'scale_drq': scale_drq, 'sqrt_out': sqrt_out, 'nz_b': nz_b})
                elif dst_dtype == 'float16' and l0c_support_fp32 == 1:
                    tensor_c_ub = tvm.compute(
                        orig_shape,
                        lambda *indices: topi.cast(tensor_c_add_bias[indices],
                                                   dtype='float16'),
                        name='tensor_c_ub',
                        attrs={'scale_drq': scale_drq, 'sqrt_out': sqrt_out, 'nz_b': nz_b})
                else:
                    tensor_c_ub = tvm.compute(
                        orig_shape,
                        lambda *indices: tensor_c_add_bias[indices],
                        name='tensor_c_ub',
                        attrs={'scale_drq': scale_drq, 'sqrt_out': sqrt_out, 'nz_b': nz_b})
            else:
                # gemv/gevm in nd or fractal just copy in continuous
                if scale_drq_tensor is not None:
                    tensor_c_ub = tvm.compute(
                        orig_shape,
                        lambda *indices: tensor_c[indices]*scale_drq_tensor[
                            0], name='tensor_c_ub',
                        attrs={'scale_drq': scale_drq, 'sqrt_out': sqrt_out, 'nz_b': nz_b})
                elif dst_dtype == 'float16' and l0c_support_fp32 == 1:
                    tensor_c_ub = tvm.compute(
                        orig_shape,
                        lambda *indices: topi.cast(tensor_c[indices],
                                                   dtype='float16'),
                        name='tensor_c_ub',
                        attrs={'scale_drq': scale_drq, 'sqrt_out': sqrt_out, 'nz_b': nz_b})
                else:
                    tensor_c_ub = tvm.compute(
                        orig_shape, lambda *indices: tensor_c[indices],
                        name='tensor_c_ub',
                        attrs={'scale_drq': scale_drq, 'sqrt_out': sqrt_out, 'nz_b': nz_b})
            if is_fractal_a and is_fractal_b:
                tensor_c_gm = tvm.compute(
                    orig_shape, lambda *indices: tensor_c_ub[indices],
                    name='tensor_c_gm', tag='matmul_gemv')
            else:
                # ND out shape is dim 2, shape m is original value
                if optmt_c == 1:
                    tensor_c_ub_fract = tvm.compute(
                        out_shape_ori, lambda batch, i, j: tensor_c_ub[
                            batch, j // 16, i // 16, i % 16, j % 16],
                        name='tensor_c_ub_fract')
                    tensor_c_gm = tvm.compute(
                        out_shape_ori,
                        lambda *indices: tensor_c_ub_fract[indices],
                        name='tensor_c_gm', tag='matmul_gemv')
                else:
                    tensor_c_gm = tvm.compute(
                        out_shape_ori, lambda batch, i, j: tensor_c_ub[
                            batch, j // 16, i // 16, i % 16, j % 16],
                        name='tensor_c_gm', tag="matmul_gemv")

    return tensor_c_gm
