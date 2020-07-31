"""
Copyright 2019 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Compute of depthwise conv2d.
"""
# pylint: disable=import-error
from te import tvm
import te.platform.cce_params as cce_params
from te.lang.cce.te_compute import common
from topi.cce import util
from te import platform as tbe_platform

BLOCK_SIZE = cce_params.BLOCK_REDUCE
BLOCK_INT8_SIZE = cce_params.BLOCK_REDUCE_INT8
FRACTAL_M = 16
FRACTAL_N = 16
NAME_INDEX = [0]


def shape_to_list(shape):
    """
    translate tvm.shape to list type in python
    """
    return [i.value for i in shape]


def bias_add(res, bias):
    """
    calculate depthwise res + bias in UB
    Parameters
    ----------
    res: res_shape = (fmap_n, fmap_c1, 1, output_h*output_w,\
                             cce_params.BLOCK_REDUCE)

    bias: bias vector = (filter_c1*filter_c0)

    Returns
    -------
    res+bias tensor
    """
    dim_map = {}
    dim_map["out_img_shape"] = shape_to_list(res.shape)
    c_add_vector = tvm.compute(dim_map["out_img_shape"], lambda *indice:
    res(*indice) + \
    bias(indice[1] * cce_params.BLOCK_REDUCE \
         + indice[4]), \
                               name='bias_add_vector' + "_cc", \
                               tag='depthwise_conv2d')
    return c_add_vector


# pylint: disable=locally-disabled,too-many-arguments,too-many-locals
def _img2col(input_img, col_shape, filter_h, filter_w, pad, stride, dilations):
    """
    img2col for depthwise conv2d backprop filter

    Parameters
    ----------
    input_img : tvm tensor
        tensor to img2col.

    col_shape : tuple or list
        shape of output.

    filter_h: int
        height of filter.

    filter_w: int
        width of filter.

    pad: tuple or list
        pad data.

    stride: tuple or list or int
        stride of convolution.

    dilations: tuple or list or int
        dilations of convolution.

    Returns
    -------
        tensor after img2col.
    """
    def _img2col_compute(indices):
        """img2col for depthwise conv2d backprop filter"""
        # input_img.shape is (fmap_cg, fmap_c1, fmap_n, fmap_h, fmap_w, fmap_c0)
        _, _, _, fmap_h, fmap_w, _ = input_img.shape
        col_cg, col_n, col_howo, col_c1, col_hw, col_ww, col_c0 = indices
        stride_h, stride_w = stride
        dilation_h, dilation_w = dilations
        pad_top, _, pad_left, pad_right = pad

        kernel_dilate_w = (filter_w - 1) * dilation_w + 1
        output_w = (fmap_w.value + pad_left + pad_right -
                    kernel_dilate_w) // stride_w + 1

        img_cg_index = col_cg
        ori_n_index = col_n
        img_c1_index = col_c1
        ori_h_index = (col_howo // output_w) * stride_h + col_hw * dilation_h
        img_w_index = (col_howo % output_w) * stride_w + col_ww * dilation_w
        img_c0_index = col_c0

        return tvm.select(
            tvm.any(ori_h_index < pad_top,
                    ori_h_index > fmap_h.value + pad_top - 1,
                    img_w_index < pad_left,
                    img_w_index > fmap_w.value + pad_left - 1),
            tvm.const(0.0, input_img.dtype),
            input_img(img_cg_index, ori_n_index, img_c1_index,
                      ori_h_index - pad_top, img_w_index - pad_left,
                      img_c0_index))

    return tvm.compute(col_shape,
                       lambda *indices: _img2col_compute(indices),
                       name='im2col_row_major',
                       tag='im2col_row_major',
                       attrs={
                           'kernel_h': filter_h,
                           'kernel_w': filter_w,
                           'padding': pad,
                           'stride': stride
                       })


# pylint: disable=locally-disabled,invalid-name
def _im2col_fractal(col_shape, img, dout_h, dout_w):
    """
    fractal(in L0B) for depthwise conv2d backprop filter after img2col

    Parameters
    ----------
    col_shape : tuple or list
        shape of output.

    img : tvm tensor
        tensor to img2col.

    dout_h: int
        height of dout.

    dout_w: int
        width of dout.

    Returns
    -------
        tensor in fractal(in L0B).
    """
    def __im2col_fractal_indices(indices, img, dout_h, dout_w):
        """fractal(in L0B) for depthwise conv2d backprop filter after img2col"""
        _, _, _, _, kernel_h, kernel_w, _ = img.shape
        cg, b, i1, j1, j0, i0 = indices

        n_index = b
        hw_index = i1 * BLOCK_SIZE + i0
        howo = dout_h * dout_w

        c1_index = (((j1 * BLOCK_SIZE + j0) // BLOCK_SIZE) //
                    kernel_w.value) // kernel_h.value
        kh_index = ((j1 * BLOCK_SIZE + j0) // BLOCK_SIZE) // kernel_w.value
        kw_index = ((j1 * BLOCK_SIZE + j0) // BLOCK_SIZE) % kernel_w.value
        c0_index = (j1 * BLOCK_SIZE + j0) % BLOCK_SIZE

        return tvm.select(
            tvm.any(hw_index < 0, hw_index > howo - 1),
            tvm.const(0.0, img.dtype),
            img(cg, n_index, hw_index, c1_index, kh_index, kw_index, c0_index))

    return tvm.compute(col_shape,
                       lambda *indices: __im2col_fractal_indices(
                           indices, img, dout_h, dout_w),
                       name='im2col_fractal',
                       tag='im2col_fractal')


# pylint: disable=locally-disabled,too-many-locals
def _backprop_filter_matmul(mad_shape, left_tensor, right_tensor, dout_hw,
                            dout_n, res_type):
    """
    depthwiese conv2d backprop filter matmul.

    Parameters
    ----------
    mad_shape : tuple or list
        shape of output.

    left_tensor : tvm tensor
        tensor to batch matmul, which in L0A.

    right_tensor: int
        tensor to batch matmul, which in L0B.

    dout_hw: int
        dout_h * dout_w

    dout_n: int
        dout_n

    res_type: str
        dtype of output in batch matmul.

    Returns
    -------
        tensor in fractal(in L0C).
    """
    # (fmap_cgroup, fmap_c1*kernel_h*kernel_w, dout_c1 * BLOCK_SIZE,
    #  BLOCK_SIZE)
    k_n = tvm.reduce_axis((0, dout_n), name='k_n')
    k_hw = tvm.reduce_axis((0, dout_hw), name='k_hw')
    k1 = k_hw.var // BLOCK_SIZE
    k0 = k_hw.var % BLOCK_SIZE
    # left_tensor.shape is (fmap_cgroup, dout_n, dout_c1,
    #                       dout_hw_pad // BLOCK_SIZE,
    #                       BLOCK_SIZE, BLOCK_SIZE)
    # right_tensor.shape is (fmap_cgroup, dout_n, dout_hw_pad // BLOCK_SIZE,
    #                        fmap_c1*kernel_h*kernel_w, BLOCK_SIZE,
    #                        BLOCK_SIZE)
    return tvm.compute(
        mad_shape,
        lambda cg, j1, i, j0: tvm.sum(
            (left_tensor[cg, k_n, i // BLOCK_SIZE, k1, i % BLOCK_SIZE, k0] *
             right_tensor[cg, k_n, k1, j1, j0, k0]).astype(res_type),
            axis=[k_n, k_hw]),
        name='mad')


# pylint: disable=locally-disabled,too-many-locals
def depthwise_conv2d_compute(fmap, weight, depthwise_res_dtype, stride, \
                             pad, dilation, para_dict):
    """
    algorithm: depthwise_conv2d_compute

    calculating  depthwise convolution compute

    Parameters
    ----------
    fmap : feature map placehold
        5-D shape of input tensor [N, C1, H, W, C0]

    weight : filter placehold
        5-D shape of filter tensor [C1, H, W, Co, C0]

    depthwise_res_dtype : dtype of depthwise UB result

    stride : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    pad : padding added to each dimension of the input

    dilation : the dilation factor for each dimension of input

    para_dict : bias tensor dict

    Returns
    -------
    depthwise_res : result tensor
       forward depthwise result of out
    """

    fmap_shape = [int(i.value) for i in fmap.shape]
    weight_shape = [int(i.value) for i in weight.shape]
    util.check_shape_rule(fmap_shape)
    util.check_shape_rule(weight_shape)

    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    if len(fmap_shape) == 6:
        fmap_n, fmap_c1, _, fmap_h, fmap_w, _ = fmap_shape
    else:
        fmap_n, fmap_c1, fmap_h, fmap_w, _ = fmap_shape
    filter_c1, kernel_h, kernel_w, filter_co, filter_c0 = weight_shape

    pad_top, pad_bottom, pad_left, pad_right = pad
    full_height = fmap_h + pad_top + pad_bottom
    full_width = fmap_w + pad_left + pad_right
    effective_filter_h = (kernel_h - 1) * dilation_h + 1
    effective_filter_w = (kernel_w - 1) * dilation_w + 1
    output_h = (full_height - effective_filter_h) // stride_h + 1
    output_w = (full_width - effective_filter_w) // stride_w + 1

    offset_x = para_dict.get('offset_x')
    if offset_x is None:
        offset_x = 0
    bias_tensor = para_dict.get('bias_tensor')
    dsl_flag = para_dict.get('dsl_flag')
    # set_fmatrix
    fmap_im2col_row_major_shape = (fmap_n, fmap_c1, output_h * output_w, 1,
                                   kernel_h, kernel_w, BLOCK_SIZE)
    feature_col = common.im2col_6d(fmap, fmap_im2col_row_major_shape, kernel_h,
                                   kernel_w, pad, stride)

    # float32 for cloud, float16 for mini, int8 for quant;
    # How to tell the two get_version
    if not tbe_platform.intrinsic_check_support("Intrinsic_mmad", "f162f32")\
            and fmap.dtype == "float16":
        mad_out_dtype = "float16"
        mad_res_block_size = BLOCK_SIZE
        fractal_n_split = 1
        if depthwise_res_dtype != "float16":
            raise RuntimeError(
                "when fmap dtype float16, depthwise_res_dtype must float16"
                " but dtype %s" % depthwise_res_dtype)
    elif fmap.dtype == "int8" or fmap.dtype == "uint8":
        mad_out_dtype = "int32"
        mad_res_block_size = BLOCK_INT8_SIZE
        fractal_n_split = 2
        if depthwise_res_dtype != "int32":
            raise RuntimeError(
                "when fmap dtype int8/uint8, depthwise_res_dtype must int32"
                " but dtype %s" % depthwise_res_dtype)
    elif tbe_platform.intrinsic_check_support("Intrinsic_mmad", "f162f32") \
            and fmap.dtype == "float16":
        mad_out_dtype = "float32"
        mad_res_block_size = BLOCK_SIZE
        fractal_n_split = 1
        if depthwise_res_dtype != "float16":
            raise RuntimeError(
                "when fmap dtype float16, depthwise_res_dtype must float16"
                " but dtype %s" % depthwise_res_dtype)
    else:
        raise RuntimeError(
            "mad result dtype only surport int32/float16/float32,"
            " but mad dtype %s" % mad_out_dtype)

    fmap_im2col_row_major_shape = (fmap_n, fmap_c1, output_h * output_w, 1,
                                   kernel_h, kernel_w, mad_res_block_size)
    feature_col = common.im2col_6d(fmap, fmap_im2col_row_major_shape, kernel_h,
                                   kernel_w, pad, stride, offset_x)
    # fractal M == 16; fractal N == 16; quant mode,
    # M*K*N == 16*32*16; no-quant mode, M*K*N == 16*16*16
    howo_mad = (output_h * output_w + FRACTAL_M - 1) // FRACTAL_M * FRACTAL_M
    fmap_im2col_fractal_shape = (fmap_n, fmap_c1, howo_mad // FRACTAL_M, \
                                 1 * kernel_h * kernel_w, FRACTAL_M,
                                 mad_res_block_size)
    feature_col_pad = common.im2col_fractal_6d(fmap_im2col_fractal_shape,
                                               feature_col)

    if mad_out_dtype == "int32":
        filter_reshape = tvm.compute(
            (filter_c1, kernel_h * kernel_w, 2, FRACTAL_N, filter_c0),
            lambda cg, hw, co1, co0, c0: weight(
                cg, hw // kernel_w, hw % kernel_w, co1 * FRACTAL_N + co0, c0),
            name='filter_reshape')
    elif mad_out_dtype == "float16" or mad_out_dtype == "float32":
        filter_reshape = tvm.compute(
            (filter_c1, kernel_h * kernel_w, 1, filter_co, filter_c0),
            lambda cg, hw, co1, co0, c0: weight(cg, hw // kernel_w, hw %
                                                kernel_w, co0, c0),
            name='filter_reshape')
    else:
        raise RuntimeError(
            "mad result dtype only surport int32/float16/float32,"
            " but mad dtype %s" % mad_out_dtype)

    mad_shape = (fmap_n, fmap_c1, fractal_n_split, howo_mad, FRACTAL_N)

    mad_res = common.mad(mad_shape, feature_col_pad, filter_reshape,
                         mad_out_dtype)

    # if bias true, increase data flow, else hide bias
    if bias_tensor is not None and bias_tensor != {}:
        bias_flag = True
        # bias model: data flow
        # dsl_flag == 1: bias add in l0c, then cast in ub
        # dsl_flag == 0: cast in ub, then bias add in ub
        if dsl_flag == True and mad_out_dtype == "int32":
            bias_ub_brc_shape = list(mad_shape)
            bias_ub_brc_shape[3] = bias_ub_brc_shape[3] // 16
            # bias UB broadcast ho*wo//16
            bias_ub_brc = tvm.compute(bias_ub_brc_shape, lambda i, j, a, k, l: \
                bias_tensor(j * 2 * 16 + a * 16 + l), \
                                      name='bias_ub_brc')
            # bias l0c broadcast ho*wo
            bias_l0c = tvm.compute(mad_shape, lambda i1, j1, a1, k1, l1: \
                bias_ub_brc(i1, j1, a1, k1 // 16, l1), \
                                   name='bias_l0c')
            # l0c add
            mad_res = tvm.compute(mad_shape,
                                  lambda *index: bias_l0c(*index) + mad_res(
                                      *index), \
                                  name='c_col_bias')
        # l0c---ub cast
        depthwise_cast = tvm.compute(
            mad_res.shape,
            lambda *index: mad_res(*index).astype(depthwise_res_dtype),
            name='depthwise_cast',
            attrs={
                'kernel_h': kernel_h,
                'kernel_w': kernel_w,
                'padding': pad,
                'stride': stride
            })
        if dsl_flag == True and mad_out_dtype == "int32":
            depthwise_res_bias = depthwise_cast
        else:
            depthwise_res_bias = bias_add(depthwise_cast, bias_tensor)
    else:
        bias_flag = False
        depthwise_cast = tvm.compute(
            mad_res.shape,
            lambda *index: mad_res(*index).astype(depthwise_res_dtype),
            name='depthwise_cast',
            attrs={
                'kernel_h': kernel_h,
                'kernel_w': kernel_w,
                'padding': pad,
                'stride': stride
            })
        depthwise_res_bias = depthwise_cast

    res_shape = (fmap_n, fmap_c1, fractal_n_split, output_h * output_w,
                 FRACTAL_N)
    depthwise_res = tvm.compute(
        res_shape,
        lambda *index: depthwise_res_bias(*index).astype(depthwise_res_dtype),
        name='depthwise_res',
        tag='depthwise_conv2d',
        attrs={
            "bias_flag": bias_flag,
            "dsl_flag": dsl_flag
        })
    return depthwise_res


# pylint: disable=locally-disabled,invalid-name
def depthwise_conv2d_backprop_filter_d_compute(fmap, dout, kernel_h, kernel_w,
                                               stride, pad, dilations,
                                               w_dtype):
    """
    compute of depthwise conv2d backprop filter

    Parameters
    ----------
    fmap : tvm tensor
        feature map tensor in tvm.

    dout : tvm tensor
        dout tensor in tvm.

    kernel_h: int
        height of filter.

    kernel_w: int
        width of filter.

    stride: tuple or list or int
        stride of convolution.

    pad: list
        padding added to each dimension of the input.

    w_dtype: str
        the dtype of dfilter.

    Returns
    -------
    depthwise_dfilter_res: tvm tensor
        the tensor of output.
    """
    def _ceil(x):
        """
        Return the least multiple of 16 integer number
        which is greater than or equal to x.
        """
        return ((x + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE

    fmap_dtype = fmap.dtype
    dout_dtype = dout.dtype
    util.check_dtype_rule(fmap_dtype, ('float16', ))
    util.check_dtype_rule(dout_dtype, ('float16', ))
    util.check_dtype_rule(w_dtype, ('float32', ))

    fmap_shape = (int(i.value) for i in fmap.shape)
    dout_shape = (int(i.value) for i in dout.shape)

    # NCgC1HiWiC0 (C1 = 1)
    fmap_n, fmap_cgroup, fmap_c1, fmap_h, fmap_w, fmap_c0 = fmap_shape
    # NCgC1HoWoC0 (C1 = 1)
    dout_n, _, dout_c1, dout_h, dout_w, dout_c0 = dout_shape

    stride_h, stride_w = stride
    dilation_h, dilation_w = dilations

    pad_top, pad_bottom, pad_left, pad_right = pad
    full_height = fmap_h + pad_top + pad_bottom
    full_width = fmap_w + pad_left + pad_right
    effective_filter_h = (kernel_h - 1) * dilation_h + 1
    effective_filter_w = (kernel_w - 1) * dilation_w + 1
    output_h = (full_height - effective_filter_h) // stride_h + 1
    output_w = (full_width - effective_filter_w) // stride_w + 1

    if output_h != dout_h or output_w != dout_w:
        raise RuntimeError("Error fmap or dout shape input!")

    fmap_trans_shape = (fmap_cgroup, fmap_n, fmap_c1, fmap_h, fmap_w, fmap_c0)
    fmap_transpose = tvm.compute(
        fmap_trans_shape,
        lambda cg, n, c1, h, w, c0: fmap(n, cg, c1, h, w, c0),
        name='fmap_transpose')

    # set_fmatrix CgC1NHiWiC0 --> Cg,NHoWo,C1,Hw,Ww,C0
    A_im2col_row_major_shape = (fmap_cgroup, dout_n, dout_h * dout_w, fmap_c1,
                                kernel_h, kernel_w, BLOCK_SIZE)
    feature_col = _img2col(fmap_transpose, A_im2col_row_major_shape, kernel_h,
                           kernel_w, pad, (stride_h, stride_w),
                           (dilation_h, dilation_w))

    dout_hw_pad = _ceil(dout_h * dout_w)

    # Cg,NHoWo,C1,Hw,Ww,C0 (axis K in cube is NHoWo, axis N in cube is C1HwWwC0)
    # --> Cg,NHoWo // 16,C1HwWw,16,16 (nZ in L0B)
    A_im2col_fractal_shape = (fmap_cgroup, dout_n, dout_hw_pad // BLOCK_SIZE,
                              fmap_c1 * kernel_h * kernel_w, BLOCK_SIZE,
                              BLOCK_SIZE)

    # dw_in_L0C (from backward filter view) is transposed dw_in_L0B
    #  (from forward view).
    # dw.T = dout.T <matmul> im2col(fmap)
    # (Co, Ci1.Hk.Wk.Ci0) = (Co, Ho.Wo.N) <matmul> (Ho.Wo.N, Ci1.Hk.Wk.Ci0)
    feature_col_pad = _im2col_fractal(A_im2col_fractal_shape, feature_col,
                                      dout_h, dout_w)

    # NCgC1HoWoC0 --> CgC1NHoWoC0
    dout_trans_shape = (fmap_cgroup, dout_n, dout_c1, dout_h * dout_w, dout_c0)
    dout_transpose = tvm.compute(dout_trans_shape,
                                 lambda cg, n, c1, hw, c0: dout(
                                     n, cg, c1, hw // dout_w, hw % dout_w, c0),
                                 name='dout_transpose')

    def _dout_fractal_compute(index, dout_ph):
        """Transform shape in zZ with block pad."""
        cg, n, c1, hw1, c0, hw0 = index
        hw = hw1 * BLOCK_SIZE + hw0
        return dout_ph(cg, n, c1, hw, c0)

    # CgC1NHoWoC0 (axis M in cube is C1C0, axis K in cube is NHoWo)
    # --> Cg,C1,NHoWo // 16,C0,16 (zZ in L0A)
    dout_fractal_shape = (fmap_cgroup, dout_n, dout_c1,
                          dout_hw_pad // BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
    dout_fractal = tvm.compute(
        dout_fractal_shape,
        lambda *index: _dout_fractal_compute(index, dout_transpose),
        name='dout_fractal')

    # dw is float32 only.
    res_dtype = "float32"
    # axis M in cube is C1C0, axis N in cube is C1HwWwC0
    # Cg,C1HwWw,C1C0,C0 (zN in L0C)
    mad_shape = (fmap_cgroup, fmap_c1 * kernel_h * kernel_w,
                 dout_c1 * BLOCK_SIZE, BLOCK_SIZE)
    mad_res = _backprop_filter_matmul(mad_shape, dout_fractal, feature_col_pad,
                                      dout_h * dout_w, dout_n, res_dtype)

    depthwise_dfilter = tvm.compute(
        mad_shape,
        lambda cg, hw, co, c0: tvm.select(co == c0, mad_res(cg, hw, co, c0),
                                          tvm.const(0.0, mad_res.dtype)),
        name='depthwise_dfilter')

    depthwise_dfilter_res = tvm.compute(
        mad_shape,
        lambda *index: depthwise_dfilter(*index).astype(w_dtype),
        name='depthwise_dfilter_res',
        attrs={
            'kernel_h': kernel_h,
            'kernel_w': kernel_w,
            'padding': pad,
            'stride': (stride_h, stride_w),
            'dilations': (dilation_h, dilation_w)
        })

    return depthwise_dfilter_res


# pylint: disable=locally-disabled,too-many-locals,too-many-statements
def depthwise_conv2d_backprop_input_d_compute(input_shape, weight, dout,
                                              weight_sizes, strides, pads):
    """
    Computes the gradients of depthwise convolution with respect to the input.

    Parameters
    ----------
    input_shape: a list or tuple representing the shape of input,
                6D format [N, C1, 1, H, W, C0]

    weight: a tensor, 5D with shape [C1, Hf*Wf, 1, C0, C0]

    dout: a tensor, 6D format [N, Co1, 1, Ho, Wo, C0]

    weight_sizes: a list or tuple of two ints,
                  the height and width of the weight of the convolution

    strides: a list or tuple of two ints, the stride of the sliding window for
             height and width of the input of the convolution

    pads: padding added to each dimension of the input

    Returns
    -------
    dx_res: compute of the gradients of depthwise convolution
            with respect to the input
    """
    dout_dtype = dout.dtype
    dout_shape = (int(i.value) for i in dout.shape)

    dout_n, dout_cgroup, dout_c1, dout_h, dout_w, dout_c0 = dout_shape
    stride_h, stride_w = strides
    weight_height, weight_width = weight_sizes
    input_h, input_w = input_shape[3], input_shape[4]
    pad_top, _, pad_left, _ = pads

    # get the dialted shape, padding and strides of out_backprop
    dilated_padded_h = input_shape[3] + weight_height - 1
    dilated_padded_w = input_shape[4] + weight_width - 1

    dilated_h = dout_h * stride_h - (stride_h - 1)
    dilated_w = dout_w * stride_w - (stride_w - 1)

    dilated_shape = (input_shape[0], input_shape[1], input_shape[2], dilated_h,
                     dilated_w, input_shape[5])

    dilated_pad_top = weight_height - 1 - pad_top
    dilated_pad_bottom = dilated_padded_h - dilated_pad_top - dilated_h
    dilated_pad_left = weight_width - 1 - pad_left
    dilated_pad_right = dilated_padded_w - dilated_pad_left - dilated_w

    dilated_pad = (dilated_pad_top, dilated_pad_bottom, dilated_pad_left,
                   dilated_pad_right)

    dilated_strides = (1, 1)

    if dilated_pad_top >= weight_height or \
        dilated_pad_bottom >= weight_height or \
        dilated_pad_left >= weight_width or \
        dilated_pad_right >= weight_width:
        raise RuntimeError("kernel W or H must > Pad")
    # compute of out_backprop dilation
    dout_dilated = tvm.compute(
        dilated_shape,
        lambda n, cg, c1, h, w, c0: tvm.select(
            tvm.all(h % strides[0] == 0, w % strides[1] == 0), dout[
                n, cg, c1, h // strides[0], w // strides[1], c0],
            tvm.const(0, dout.dtype)),
        attrs={'strides': strides},
        name='dout_dilated')

    # image to column of dilated out_backprop
    dout_im2col_row_major_shape = (dout_n, dout_cgroup, input_h * input_w,
                                   dout_c1, weight_height, weight_width,
                                   BLOCK_SIZE)
    dout_col = common.im2col_6d(dout_dilated, dout_im2col_row_major_shape,
                                weight_height, weight_width, dilated_pad,
                                dilated_strides)

    hiwi_mad = (input_h * input_w + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE

    dout_im2col_fractal_shape = (dout_n, dout_cgroup, hiwi_mad // BLOCK_SIZE,
                                 dout_c1 * weight_height * weight_width,
                                 BLOCK_SIZE, BLOCK_SIZE)

    dout_col_pad = common.im2col_fractal_6d(dout_im2col_fractal_shape,
                                            dout_col)

    # rotate weight with a degree of 180
    weight_rotated = tvm.compute(
        weight.shape,
        lambda cg, khkw, co1, co0, c0: weight[
            cg, (weight_height - 1 - khkw // weight_width) * weight_width +
            (weight_width - 1 - khkw % weight_width), co1, co0, c0],
        name='weight_rotated')

    # float32 for cloud, float16 for mini. How to tell the two get_version
    if not tbe_platform.intrinsic_check_support("Intrinsic_mmad", "f162f32"):
        mad_out_dtype = "float16"
    else:
        mad_out_dtype = "float32"
    # matrix multiplication of dilated out_backprop and rotated weight
    mad_shape = (dout_n, dout_cgroup, dout_c1, hiwi_mad, dout_c0)
    mad_res = common.mad(mad_shape, dout_col_pad, weight_rotated,
                         mad_out_dtype)

    # cast dX from float32 to float16
    dx_cast = tvm.compute(mad_res.shape,
                          lambda *index: mad_res(*index).astype(dout_dtype),
                          name='dx_cast')

    # remove the padding of dX
    res_shape = (dout_n, dout_cgroup, dout_c1, input_h * input_w, dout_c0)
    dx_res = tvm.compute(res_shape,
                         lambda *index: dx_cast(*index).astype(dout_dtype),
                         name='dx_res',
                         attrs={
                             'weight_height': weight_height,
                             'weight_width': weight_width,
                             'dilated_pad': dilated_pad,
                             'dilated_strides': dilated_strides
                         })
    return dx_res
