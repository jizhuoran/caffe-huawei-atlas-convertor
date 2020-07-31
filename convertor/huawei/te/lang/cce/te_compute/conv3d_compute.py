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

conv3d compute
"""
from __future__ import division
from te import tvm
from te.platform import CUBE_MKN
from te.platform import get_soc_spec
from te.domain.tiling.tiling_query import tiling_query
from te.utils import op_utils
from . import util as te_util
from .cube_util import im2col_fractal_3d, im2col_row_major

OP_TAG = "conv3d_"
TENSOR_MAP = {}
DIM_MAP = {}
NAME_INDEX = [0]
SQRT = {}
# filterD must be in [1,255]
FILTER_DHW_MIN = 1
FILTER_DHW_MAX = 255
# pad must be in [0,255]
PAD_MIN = 0
PAD_MAX = 255
# stride must be in [1,63]
STRIDE_MIN = 1
STRIDE_MAX = 63

# fmap H and W must be in [1, 4096]
FMAP_HW_MIN = 1
FMAP_HW_MAX = 4096


def check_d_dimension(fmap_d, filter_d, pad_d, stride_d):
    if filter_d < FILTER_DHW_MIN or filter_d > FILTER_DHW_MAX:
        raise RuntimeError("kernel D must be in [1,255].")

    if (fmap_d + pad_d[0] + pad_d[1]) < filter_d:
        raise RuntimeError(
            "the depth of feature map after padding"
            "can not be less than shape_filter's")

    if pad_d[0] < PAD_MIN or pad_d[1] < PAD_MIN or \
        pad_d[0] > PAD_MAX or pad_d[1] > PAD_MAX:
        raise RuntimeError("padd must be in [0,255].")

    if pad_d[0] >= filter_d or pad_d[1] >= filter_d:
        raise RuntimeError("padd must be less than shape_filter's")

    if stride_d < STRIDE_MIN or stride_d > STRIDE_MAX:
        raise RuntimeError("strided must be in [1,63].")


def check_h_dimension(fmap_h, filter_h, pad_h, stride_h):
    if fmap_h < FMAP_HW_MIN or fmap_h > FMAP_HW_MAX:
        raise RuntimeError("feature H must be in [1,4096].")

    if filter_h < FILTER_DHW_MIN or filter_h > FILTER_DHW_MAX:
        raise RuntimeError("kernel H must be in [1,255].")

    if pad_h[0] < PAD_MIN or pad_h[1] < PAD_MIN or \
        pad_h[0] > PAD_MAX or  pad_h[1] > PAD_MAX:
        raise RuntimeError("padh must be in [0,255].")

    if filter_h > (fmap_h + pad_h[0] + pad_h[1]):
        # Chip Design demand, Load3D
        raise RuntimeError("feature H(after pad) must >= kernel H")

    if stride_h < STRIDE_MIN or stride_h > STRIDE_MAX:
        raise RuntimeError("strideh must be in [1,63].")

    if pad_h[0] >= filter_h or pad_h[1] >= filter_h:
        raise RuntimeError("kernel H must > Pad H")


def check_w_dimension(fmap_w, filter_w, pad_w, stride_w):
    if fmap_w < FMAP_HW_MIN or fmap_w > FMAP_HW_MAX:
        raise RuntimeError("feature W must be in [1,4096].")

    if filter_w < FILTER_DHW_MIN or filter_w > FILTER_DHW_MAX:
        raise RuntimeError("kernel W must be in [1,255].")

    if pad_w[0] < PAD_MIN or pad_w[1] < PAD_MIN or \
        pad_w[0] > PAD_MAX or pad_w[1] > PAD_MAX:
        raise RuntimeError("padw must be in [0,255].")

    if filter_w > (fmap_w + pad_w[0] + pad_w[1]):
        # Chip Design demand, Load3D
        raise RuntimeError("feature W(after pad) must >= kernel W")

    if stride_w < STRIDE_MIN or stride_w > STRIDE_MAX:
        raise RuntimeError("stridew must be in [1,63].")


def check_conv3d_shape(shape_fm, shape_filter, pads, stride_dhw, fmp_dtype,
                       w_dtype):
    """
    algorithm: check the input params of conv3d

    Parameters
    ----------
    shape_fm: the shape of feature, format is 'NCDHW'.
        a list/tuple of 'int' that has length `== 5`

    shape_filter: the shape of filter, format is 'NCDHW'.
        a list of 'int' that has length `== 5`

    pads: tuple/list of 6 integers
        [pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right]

    stride_dhw: A list of `ints` that has length `== 3`.

    fmp_dtype: the dtype of feature

    w_dtype: the dtype of filter

    Returns
    -------
    None
    """
    if shape_fm[1] != shape_filter[1]:
        raise RuntimeError("input feature map channel should"
                           "equal to filter channel")

    fmap_n, fmap_c, fmap_d, fmap_h, fmap_w = shape_fm
    filter_n, filter_c, filter_d, filter_h, filter_w = shape_filter

    pad_d = [pads[0], pads[1]]
    check_d_dimension(fmap_d, filter_d, pad_d, stride_dhw[0])

    pad_h = [pads[2], pads[3]]
    check_h_dimension(fmap_h, filter_h, pad_h, stride_dhw[1])

    pad_w = [pads[4], pads[5]]
    check_w_dimension(fmap_w, filter_w, pad_w, stride_dhw[2])


    # C dimension should align 16
    block_size_k = CUBE_MKN[fmp_dtype]['mac'][1]
    block_size_m = CUBE_MKN[fmp_dtype]['mac'][0]
    famp_c = ((fmap_c + block_size_k - 1) //
               block_size_k) * block_size_k
    filter_c = fmap_c
    block_size_n = CUBE_MKN[w_dtype]['mac'][2]
    filter_n = ((filter_n + block_size_n - 1) //
                block_size_n) * block_size_n

    # calculated by h_i and w_i
    h_out = (fmap_h + (pad_h[0] + pad_h[1]) - filter_h) // stride_dhw[1] + 1
    w_out = (fmap_w + (pad_w[0] + pad_w[1]) - filter_w) // stride_dhw[2] + 1
    d_out = (fmap_d + (pad_d[0] + pad_d[1]) - filter_d) // stride_dhw[0] + 1

    load2d_pass_flag = (filter_d == 1) and (filter_h == 1) and \
                       (filter_w == 1) and \
                       (list(pads) == [0, 0, 0, 0, 0, 0]) and \
                       (list(stride_dhw) == [1, 1, 1])

    #  Chip Design demand only h_dimesion constraint
    only_fhkh_pass_flag = (1 <= filter_h <= 11) and \
                          (stride_dhw[1] == 1) and \
                          (h_out == 1)

    #  Chip Design demand both h_dimesion and w_dimension constraint
    fhkh_fwkw_pass_flag = (1 <= filter_w <= 11) and (1 <= filter_h <= 11) and \
                          (stride_dhw[1] == 1) and (stride_dhw[2] == 1) and \
                          (h_out == 1) and (w_out == 1)

    if load2d_pass_flag or only_fhkh_pass_flag or fhkh_fwkw_pass_flag:
        pass
    else:
        if w_out < 2:
            # Chip Design demand w_out must >=2
            raise RuntimeError("FM_W + pad_left + pad_right - KW>=strideW")

        if h_out < 2:
            # Chip Design demand h_out must >=2
            raise RuntimeError("FM_H + pad_top + pad_bottom - KH>=strideH")

    # check for not bigger than L1
    l1_buffer_size = get_soc_spec("L1_SIZE")
    m_bit_ratio = {"float16": 2, "int8": 1}
    point_per_w = (fmap_w - filter_w +
                   pad_w[0] + pad_w[1]) // stride_dhw[2] + 1
    w_in = block_size_m // point_per_w + 2
    tmp = ((w_in - 1) * stride_dhw[1] + filter_h) * fmap_w
    max_feature_map_l1 = block_size_k * tmp * m_bit_ratio[w_dtype]

    if max_feature_map_l1 > l1_buffer_size:
        raise RuntimeError(
            "Input feature is too large, "
            "the minimum tiling may exceeds L1_Buffer")


class Conv3DParam:
    """
    class of ConvParam
    """

    def __init__(self):
        pass

    def get_tensor_map(self):
        """
         get the tensor_map in convparam
        """
        return self.TENSOR_MAP

    TENSOR_MAP = {}
    dim_map = {}
    tiling = None
    tiling_query_param = {}


def cube_3d_compute(fmap,
                    weight,
                    mad_dtype,
                    res_dtype,
                    pads,
                    stride_dhw,
                    shape_filter_ncdhw,
                    cyclebuffer_flag,
                    bias=False,
                    tiling=None):
    """
    conv

    Parameters
    ----------
    fmap : tvm.tensor, Feature Map

    weight: tvm.tensor, Filter

    mad_dtype : the compute data type

    res_dtype : the result data type

    pads: the padding shape
        [head, tail, top, bottom, left, right]

    stride_dhw: the stride value
        [stride_d, stride_h, stride_w]

    shape_filter_ncdhw: the filter shape

    bias: the tag for bias or not

    tiling: default none, tiling

    -------
    Returns

    wrapped_tensor
    """
    in_dtype = fmap.dtype
    w_dtype = weight.dtype

    TENSOR_MAP["fmap"] = fmap
    TENSOR_MAP["filter"] = weight

    if isinstance(bias, tvm.tensor.Tensor):
        TENSOR_MAP["bias"] = bias

    fmap_shape = te_util.shape_to_list(fmap.shape)
    batch_size = fmap_shape[0]
    fmap_d = fmap_shape[1]
    fmap_c1 = fmap_shape[2]
    fmap_h = fmap_shape[3]
    fmap_w = fmap_shape[4]
    fmap_c0 = fmap_shape[5]
    filter_cout, _, filter_d, filter_h, filter_w = shape_filter_ncdhw
    pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right = pads
    stride_d, stride_h, stride_w = stride_dhw

    TENSOR_MAP["filter_d"] = filter_d

    height_out = (fmap_h + pad_top + pad_bottom - filter_h) // stride_h + 1
    width_out = (fmap_w + pad_left + pad_right - filter_w) // stride_w + 1
    d_out = (fmap_d + pad_head + pad_tail - filter_d) // stride_d + 1

    config = CUBE_MKN[in_dtype]
    block_size_k = config['mac'][1]
    block_size_m = config['mac'][0]
    opti_h_flag = filter_h == 1 and stride_h > 1
    TENSOR_MAP["opti_h_flag"] = opti_h_flag
    TENSOR_MAP["d_out"] = d_out
    TENSOR_MAP["d_dim"] = tiling["block_dim"][-1]

    fmap_fuse_shape = (batch_size * d_out, filter_d * fmap_c1, fmap_h, fmap_w,
                       fmap_c0)
    fuse_fmap_tensor = get_fuse_fmap_tensor(fmap_fuse_shape,
                                            fmap,
                                            d_out,
                                            filter_d,
                                            stride_d,
                                            stride_h,
                                            pad_head,
                                            tiling,
                                            opti_h_flag,
                                            cyclebuffer_flag,
                                            tag=OP_TAG)
    TENSOR_MAP["fmap_do_tensor"] = fuse_fmap_tensor

    # set_fmatrix
    # new data layout (N,C1,H,W,C0) -> (N,HoWo,C1,Hk,Wk,C0)
    fmap_im2col_row_major_shape = (fmap_fuse_shape[0], height_out * width_out,
                                   fmap_fuse_shape[1], filter_h, filter_w,
                                   fmap_c0)
    pad_hw = pads[2:]
    stride_hw = [stride_h, stride_w]
    fmap_im2col_row_major_res = im2col_row_major(fmap_im2col_row_major_shape,
                                                 fuse_fmap_tensor,
                                                 filter_w,
                                                 pad_hw,
                                                 stride_hw,
                                                 fmap.dtype,
                                                 opti_h_flag,
                                                 tag=OP_TAG)
    TENSOR_MAP["fmap_im2col_row_major_res"] = fmap_im2col_row_major_res

    # im2col
    # small-z-big-Z
    howo_mad = (height_out * width_out + block_size_m -
                1) // block_size_m * block_size_m

    # new data layout (N,HoWo,C1,Hk,Wk,C0) -> (N,loop_m,loop_k,cube_m,cube_k)
    fmap_im2col_fractal_shape = (fmap_fuse_shape[0], howo_mad // block_size_m,
                                 fmap_fuse_shape[1] * filter_h * filter_w,
                                 block_size_m, block_size_k)
    fmap_im2col_fractal_res = im2col_fractal_3d(fmap_im2col_fractal_shape,
                                                fmap_im2col_row_major_res,
                                                fmap_c1,
                                                d_out,
                                                filter_d,
                                                stride_d,
                                                cyclebuffer_flag,
                                                tag=OP_TAG)
    TENSOR_MAP["fmap_im2col_fractal_res"] = fmap_im2col_fractal_res

    config = CUBE_MKN[res_dtype]

    l0a_load2d_flag = get_load2d_flag(stride_dhw, pads, shape_filter_ncdhw)

    TENSOR_MAP["l0a_load2d_flag"] = l0a_load2d_flag

    mad_shape = (fmap_fuse_shape[0],
                 (filter_cout + config['mac'][2] - 1) // (config['mac'][2]),
                 howo_mad, config['mac'][2])

    config = CUBE_MKN[w_dtype]

    if l0a_load2d_flag:
        c_col = mad_by_load2d(mad_shape, fmap, weight, config, mad_dtype, pads,
                              stride_d, d_out, filter_d)
    else:
        c_col = mad(mad_shape, fmap_im2col_fractal_res, weight, config,
                    mad_dtype, pads, stride_d, d_out, fmap_d, filter_d)
    TENSOR_MAP["c_col"] = c_col

    conv_shape = (fmap_fuse_shape[0],
                  (filter_cout + config['mac'][2] - 1) // (config['mac'][2]),
                  height_out * width_out, config['mac'][2])
    DIM_MAP["out_img_shape"] = conv_shape

    c_ub = tvm.compute(mad_shape,
                       lambda n, i, j, k: c_col(n, i, j, k).astype(res_dtype),
                       name='C_UB',
                       tag=OP_TAG + "C_UB",
                       attrs={
                           'true_shape': conv_shape,
                           'sqrt': False,
                           'res_dtype': res_dtype,
                           'kernel_h': filter_h,
                           'kernel_w': filter_w,
                           'padding': pads[2:],
                           'stride': stride_dhw[1:]
                       })

    TENSOR_MAP["c_ub"] = c_ub
    dim_map1 = im2col_dim(te_util.shape_to_list(fuse_fmap_tensor.shape),
                          shape_filter_ncdhw, list(pads), list(stride_dhw),
                          config)
    dim_map_copy = DIM_MAP.copy()
    dim_map_copy.update(dim_map1)

    Conv3DParam.TENSOR_MAP = TENSOR_MAP
    Conv3DParam.dim_map = dim_map_copy
    Conv3DParam.tiling = None

    return c_ub


def get_fuse_fmap_tensor(fmap_fuse_shape, fmap, d_out, kernel_d, stride_d,
                         stride_h, pad_head, tiling, opti_h_flag,
                         cyclebuffer_flag, tag):
    """
    calculate expand tensor
    Parameters

    ----------
    fmap_fuse_shape : the shape of new tensor

    fmap : the input feature

    d_out : the D dimension of out shape

    stride_d : the D dimension of strides

    pad_head : the pad head of pads

    tag : the tensor tag

    Returns
    -------
    new tensor
    """
    _, fmap_d, fmap_c1, _, _, _ = fmap.shape
    # multi core
    d_dim = tiling["block_dim"][-1]
    if cyclebuffer_flag:
        if opti_h_flag:
            fmap_fuse_shape = list(fmap_fuse_shape)
            fmap_fuse_shape[2] = (fmap_fuse_shape[2] - 1) // stride_h + 1
            fuse_fmap_tensor = tvm.compute(
                fmap_fuse_shape,
                lambda n, dc, h, w, c0: tvm.select(
                    tvm.all(
                        n % d_out * stride_d + (dc // fmap_c1 + n % d_out *
                                                (kernel_d - stride_d)) % kernel_d -
                        pad_head >= 0, n % d_out * stride_d + \
                        (dc // fmap_c1 + n % d_out * \
                        (kernel_d - stride_d)) % kernel_d - pad_head < fmap_d,
                        tvm.any(
                            n % d_out * stride_d + \
                            (dc // fmap_c1 + n % d_out * \
                            (kernel_d - stride_d)) % kernel_d > \
                            (n % d_out - 1) * stride_d + kernel_d - 1, n % \
                            (d_out // d_dim) == 0)),
                    fmap(
                        n // d_out, n % d_out * stride_d +\
                        (dc // fmap_c1 + n % d_out * (kernel_d - stride_d)\
                        ) % kernel_d - pad_head, dc % fmap_c1, h*stride_h, w, c0)),
                name='fuse_fmap_tensor',
                tag=tag + 'fuse_fmap_tensor')
        else:
            fuse_fmap_tensor = tvm.compute(
                fmap_fuse_shape,
                lambda n, dc, h, w, c0: tvm.select(
                    tvm.all(
                        n % d_out * stride_d + (dc // fmap_c1 + n % d_out *
                                                (kernel_d - stride_d)) % kernel_d -
                        pad_head >= 0, n % d_out * stride_d + \
                        (dc // fmap_c1 + n % d_out * \
                        (kernel_d - stride_d)) % kernel_d - pad_head < fmap_d,
                        tvm.any(
                            n % d_out * stride_d + \
                            (dc // fmap_c1 + n % d_out * \
                            (kernel_d - stride_d)) % kernel_d > \
                            (n % d_out - 1) * stride_d + kernel_d - 1, n % \
                            (d_out // d_dim) == 0)),
                    fmap(
                        n // d_out, n % d_out * stride_d +\
                        (dc // fmap_c1 + n % d_out * (kernel_d - stride_d)\
                        ) % kernel_d - pad_head, dc % fmap_c1, h, w, c0)),
                name='fuse_fmap_tensor',
                tag=tag + 'fuse_fmap_tensor')
    else:
        if opti_h_flag:
            fmap_fuse_shape = list(fmap_fuse_shape)
            fmap_fuse_shape[2] = (fmap_fuse_shape[2] - 1) // stride_h + 1
            fuse_fmap_tensor = tvm.compute(
                fmap_fuse_shape,
                lambda n, dc, h, w, c0: tvm.select(
                    tvm.all((n % d_out) * stride_d - pad_head + dc // fmap_c1 >= 0,
                            (n % d_out) * \
                            stride_d - pad_head + dc // fmap_c1 < fmap_d),
                    fmap(n // d_out, (n % d_out) * stride_d - pad_head + dc // \
                         fmap_c1, dc % fmap_c1, h*stride_h, w, c0)),
                name='fuse_fmap_tensor',
                tag=tag + 'fuse_fmap_tensor')
        else:
            fuse_fmap_tensor = tvm.compute(
                fmap_fuse_shape,
                lambda n, dc, h, w, c0: tvm.select(
                    tvm.all((n % d_out) * stride_d - pad_head + dc // fmap_c1 >= 0,
                            (n % d_out) * \
                            stride_d - pad_head + dc // fmap_c1 < fmap_d),
                    fmap(n // d_out, (n % d_out) * stride_d - pad_head + dc // \
                         fmap_c1, dc % fmap_c1, h, w, c0)),
                name='fuse_fmap_tensor',
                tag=tag + 'fuse_fmap_tensor')

    return fuse_fmap_tensor


def mad_by_load2d(mad_shape, fmap, weight, config, mad_dtype, pads, stride_d,
                  d_out, filter_d):
    """
    calculate mad
    Parameters

    ----------
    mad_shape : the shape of new tensor

    fmap : the input feature

    weight : the input filter

    config : the MKN config

    mad_dtype : the compute dtype of mad

    Returns
    -------
    new tensor
    """
    fmap_shape = te_util.shape_to_list(fmap.shape)
    batch_size = fmap_shape[0]
    fmap_d = fmap_shape[1]
    fmap_c1 = fmap_shape[2]
    fmap_h = fmap_shape[3]
    fmap_w = fmap_shape[4]
    fmap_c0 = fmap_shape[5]
    shape_al1_load2d = (batch_size * fmap_d, fmap_c1, fmap_h * fmap_w, fmap_c0)
    al1_load2d = tvm.compute(
        shape_al1_load2d,
        lambda n, c1, m, c0: fmap(n // fmap_d, n % fmap_d, c1, m // fmap_w, m %
                                  fmap_w, c0),
        name=OP_TAG + "al1_load2d")
    TENSOR_MAP["al1_load2d"] = al1_load2d

    hw_dim = te_util.int_ceil_div(fmap_h * fmap_w,
                                  CUBE_MKN[fmap.dtype]["mac"][0])
    shape_al0_load2d = (batch_size * fmap_d, hw_dim, fmap_c1,
                        CUBE_MKN[fmap.dtype]["mac"][0], fmap_c0)

    al0_load2d = tvm.compute(
        shape_al0_load2d,
        lambda n, m1, c1, m0, c0: al1_load2d(
            n, c1, m0 + CUBE_MKN[fmap.dtype]["mac"][0] * m1, c0),
        name=OP_TAG + "al0_load2d")
    TENSOR_MAP["al0_load2d"] = al0_load2d

    c_col = mad(mad_shape, al0_load2d, weight, config, mad_dtype, pads,
                stride_d, d_out, fmap_d, filter_d)
    return c_col


def get_load2d_flag(stride, pads, shape_filter_ncdhw):
    """
    calculate use load2d or not
    Parameters

    ----------
    stride : the input strides

    pads : the input pads

    shape_filter_ncdhw : the shape of filter

    Returns
    -------
    True or False
    """
    l0a_load2d_flag = False
    _, _, filter_d, filter_h, filter_w = shape_filter_ncdhw

    if list(pads) == [0, 0, 0, 0, 0, 0] and list(stride) == [1, 1, 1] and \
                    [filter_d, filter_h, filter_w] == [1, 1, 1]:
        l0a_load2d_flag = True
    return l0a_load2d_flag


def get_cyclebuffer_flag(tiling, shape_w, w_dtype, channel_c1, stride_d,
                         l0a_load2d_flag):
    """
    calculate whether to do cyclebuffer

    Parameters
    ----------
    tiling : tiling_new

    shape_w : filter shape

    channel_c1 : fmap c1

    stride_d : d channel stride

    l0a_load2d_flag : whether fmap to load2d

    return
    ----------
    cyclebuffer_flag

    """
    cyclebuffer_flag = False
    filter_d = shape_w[1]
    cyc_size = 0
    if tiling["AL1_shape"]:
        cyc_size = int(tiling["AL1_shape"][0] * tiling["AL1_shape"][-1] // \
                       (shape_w[-3] * shape_w[-2] * CUBE_MKN[w_dtype]['mac'][1]))
    if cyc_size == filter_d * channel_c1:
        cyclebuffer_flag = True
    if l0a_load2d_flag or filter_d <= stride_d:
        cyclebuffer_flag = False

    return cyclebuffer_flag


def im2col_dim(shape_fmap, shape_filter_ncdhw, pads, stride_dhw, config):
    """
    calculate shape
    Parameters

    ----------
    shape_fmap : shape of feature

    shape_filter_ncdhw : shape of filter

    pads : the padding shape

    stride_dhw : the stride value

    config : the MKN infor

    Returns
    -------
    img_shape, fmap_matrix_dim
    """
    mac_dim = config['mac']

    batch, fmap_c1, fmap_h, fmap_w, fmap_c0 = shape_fmap
    filter_cout, _, _, filter_h, filter_w = shape_filter_ncdhw
    _, _, pad_top, pad_bottom, pad_left, pad_right = pads

    out_h = ((fmap_h + pad_top + pad_bottom) - filter_h) // stride_dhw[1] + 1
    out_w = ((fmap_w + pad_left + pad_right) - filter_w) // stride_dhw[2] + 1

    fmap_valid_dim = (batch, out_h * out_w,
                      fmap_c1 * filter_h * filter_w * fmap_c0)

    fmap_matrix_dim = (fmap_valid_dim[0],
                       ((fmap_valid_dim[-2] + mac_dim[0] - 1) // mac_dim[0]),
                       ((fmap_valid_dim[-1] + mac_dim[1] - 1) // mac_dim[1]),
                       mac_dim[0], mac_dim[1])

    filter_valid_dim = (fmap_valid_dim[-1], filter_cout)

    filter_matrix_dim = ((filter_valid_dim[-2] + mac_dim[1] - 1) // mac_dim[1],
                         (filter_valid_dim[-1] + mac_dim[2] - 1) // mac_dim[2],
                         mac_dim[2], mac_dim[1])

    return {
        "img_shape": shape_fmap,
        "fmap_matrix_dim": fmap_matrix_dim,
        "filter_matrix_dim": filter_matrix_dim,
        "shape_filter_ncdhw": shape_filter_ncdhw
    }


def mad(mad_shape, fmap, weight, config, mad_dtype, pads, stride_d, d_out,
        fmap_d, filter_d):
    """
    calculate mad result tensor
    Parameters
    ----------
    mad_shape : shape of mad result

    fmap : feature map

    weight : filter

    config: the config of cube

    mad_dtype: dtype of mad output

    pads: input pad

    stride_d: stride for d channel

    d_out: output d channel

    fmap_d: input fmap d channel

    filter_d: input filter d channel

    Returns
    -------
    mad result tensor
    """
    block_size = config['mac'][1]
    block_size_m = config['mac'][0]
    pad_head = pads[0]
    c1khkw = weight.shape[0] // filter_d

    axis_k1 = tvm.reduce_axis((0, weight.shape[0]), name='k1')
    axis_k0 = tvm.reduce_axis((0, block_size), name='k0')

    if mad_dtype in ["float16", "int32"]:
        mode = 'f162f16'
    else:
        mode = 'f162f32'

    c_col = tvm.compute(
        mad_shape,
        lambda n, index_j1, i, index_j0: tvm.sum(
            (fmap[n, i // block_size_m, axis_k1, i % block_size_m, axis_k0] *
             weight[axis_k1, index_j1, index_j0, axis_k0]).astype(mad_dtype),
                                                 axis=[axis_k1, axis_k0]),
        name='mad1',
        tag=OP_TAG + "c_col",
        attrs={
            'mode': mode,
            'pad_head': pad_head,
            'fmap_d': fmap_d,
            'stride_d': stride_d,
            'd_out': d_out
        })
    return c_col


def bias_add(in_tensor0, in_tensor1):
    """
    calculate conv res + bias in UB
    Parameters
    ----------
    in_tensor0: cnv res tensor

    in_tensor1: bias vector

    Returns
    -------
    in_tensor0+in_tensor1 tensor
    """
    dim_map = {}
    dim_map["out_img_shape"] = te_util.shape_to_list(in_tensor0.shape)
    NAME_INDEX[0] += 1

    with tvm.tag_scope('conv_vector_bias_add'):
        c_add_vector = tvm.compute(
            dim_map["out_img_shape"],
            lambda *indice: in_tensor0(*indice) + in_tensor1(indice[
                1] * CUBE_MKN[in_tensor0.dtype]['mac'][2] + indice[3]),
            name='bias_add_vector' + "_cc_" + str(NAME_INDEX[0]))
    return c_add_vector


def remove_pad(res, res_remove_pad_shape):
    """
    remove pad
    Parameters
    ----------
    res: input tensor

    res_remove_pad_shape: true shape

    Returns
    -------
    res_remove_pad tensor
    """
    NAME_INDEX[0] += 1
    with tvm.tag_scope('conv_vector_remove_pad'):
        res_tensor = tvm.compute(res_remove_pad_shape,
                                 lambda *indice: res(*indice),
                                 name='remove_pad' + "_cc_" +
                                 str(NAME_INDEX[0]))
    return res_tensor


@tvm.target.generic_func
def conv3d(data, weight, para_dict):
    """
    conv

    Parameters
    ----------
    data: feature map

    weight: filter

    para_dict: dict of params

    Returns
    -------
    tensor : res
    """
    in_dtype = data.dtype
    w_dtype = weight.dtype
    bias_tensor = para_dict["bias_tensor"]
    bias_flag = (bias_tensor is not None)

    pads = para_dict["pads"]
    pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right = pads
    pad_d = [pad_head, pad_tail]
    pad_w = [pad_left, pad_right]
    pad_h = [pad_top, pad_bottom]

    stride_dhw = para_dict["stride_dhw"]
    stride_d, stride_h, stride_w = stride_dhw

    shape_filter_ncdhw = para_dict["shape_filter_ncdhw"]
    filter_n, filter_c, filter_d, filter_h, filter_w = shape_filter_ncdhw

    mad_dtype = para_dict["mad_dtype"]
    res_dtype = para_dict["res_dtype"]

    block_size_k = CUBE_MKN[w_dtype]['mac'][1]
    filter_c1 = (filter_c + block_size_k - 1) // block_size_k
    shape_w_ndc1hwc0 = (filter_n, filter_d, filter_c1, filter_h, filter_w,
                        block_size_k)

    fmap_shape_ndc1hwc0 = te_util.shape_to_list(data.shape)

    Conv3DParam.tiling_query_param = {
        "fmap_shape_ndc1hwc0": fmap_shape_ndc1hwc0,
        "shape_w_ndc1hwc0": shape_w_ndc1hwc0,
        "in_dtype": in_dtype,
        "w_dtype": w_dtype,
        "res_dtype": res_dtype,
        "mad_dtype": mad_dtype,
        "padw": pad_w,
        "padh": pad_h,
        "padd": pad_d,
        "strideh": stride_h,
        "stridew": stride_w,
        "strided": stride_d,
        "bias_flag": bias_flag,
        "default_tiling": False
    }

    tiling_new = tiling_query(a_shape=fmap_shape_ndc1hwc0,
                              b_shape=shape_w_ndc1hwc0,
                              a_dtype=in_dtype,
                              b_dtype=w_dtype,
                              c_dtype=res_dtype,
                              mad_dtype=mad_dtype,
                              padl=pad_w[0],
                              padr=pad_w[1],
                              padu=pad_h[0],
                              padd=pad_h[1],
                              padf=pad_d[0],
                              padb=pad_d[1],
                              strideh=stride_h,
                              stridew=stride_w,
                              strided=stride_d,
                              bias_flag=bias_flag,
                              op_tag="convolution_3d")

    TENSOR_MAP["tiling_new"] = tiling_new
    l0a_load2d_flag = get_load2d_flag(stride_dhw, pads, shape_filter_ncdhw)
    cyclebuffer_flag = get_cyclebuffer_flag(tiling_new, shape_w_ndc1hwc0,
                                            w_dtype, fmap_shape_ndc1hwc0[2],
                                            stride_d, l0a_load2d_flag)

    TENSOR_MAP["cyclebuffer_flag"] = cyclebuffer_flag

    conv_res = cube_3d_compute(data,
                               weight,
                               mad_dtype,
                               res_dtype,
                               pads,
                               stride_dhw,
                               shape_filter_ncdhw,
                               cyclebuffer_flag,
                               bias=False,
                               tiling=tiling_new)
    res = conv_res
    if bias_flag:
        res = bias_add(conv_res, bias_tensor)

    # Remove H-aligned data in the output shape
    res_remove_pad_shape = list(res.shape)
    res_remove_pad_shape[2] = conv_res.op.attrs['true_shape'][2].value
    res_remove_pad = remove_pad(res, res_remove_pad_shape)

    return res_remove_pad
