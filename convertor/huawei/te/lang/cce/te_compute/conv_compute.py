"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

conv2d DSL interface.
"""
from __future__ import division
import math
from te import tvm
from te.platform import cce_conf
from te.platform import CUBE_MKN
from te.platform.cce_buffer import cur_cce_product_params as cce_product
from topi.cce import util
from topi.cce.util import is_v200_version
from topi.cce.util import is_lhisi_version
from topi.cce.util import is_mini_or_lhisi_version

# fmapH, fmapW must be in [1,4096]
FMAP_HW_MIN = 1
FMAP_HW_MAX = 4096

# filterH, filterW must be in [1,255]
FILTER_HW_MIN = 1
FILTER_HW_MAX = 255

# padH, padW must be in [0,255]
PAD_MIN = 0
PAD_MAX = 255

# stride must be in [1,63]
STRIDE_MIN = 1
STRIDE_MAX = 63

# dilate must be in [1,255]
DILATE_MIN = 1
DILATE_MAX = 255
CONV_SHAPE_DIM = 4


def check_conv_shape(shape_in, shape_w, pad_top, pad_bottom, \
    pad_left, pad_right, strideh, stridew, in_dtype, w_dtype, fusion_para, \
                     optim_dict=None, dilateh=1, dilatew=1):
    """

    Parameters
    ----------
    shape_in : shape of data_in

    shape_w : shape of filter

    padh: the padding shape in H

    padw: the padding shape in weight

    strideh: the stride value in H

    stridew: the stride value in weight

    dilateh: the dilate value in H

    dilatew: the dilate value in weight

    optim_dict: optimize feature dict

    in_dtype : the feature map data type

    w_dtype : the weight data type

    fusion_para: the config for Lx Fusion

    Returns
    -------
    None

    """

    def fusion_para_check(fusion_para, pad_top, pad_bottom, shape_in):
        """
         check Lx fusion para
        """
        def handle_valid_shape():
            if not slice_offset:
                raise RuntimeError("if valid_shape exists, "
                                   "offset can not be []")
            slice_offset_check_flg = (slice_offset[2] < shape_in[2]) and \
            (slice_offset[2] >= 0) \
            and slice_offset[0] == 0 and slice_offset[1] == 0 and \
            slice_offset[3] == 0 and slice_offset[4] == 0

            valid_shape_check_flg = (valid_shape[2] + slice_offset[2] <= \
                shape_in[2]) and (valid_shape[2] >= 0) \
            and valid_shape[0] == shape_in[0] and valid_shape[3] == \
            shape_in[3] and valid_shape[1]*valid_shape[4] == shape_in[1]

            if not slice_offset_check_flg:
                raise RuntimeError("Invalid slice_offset", slice_offset)
            if not valid_shape_check_flg:
                raise RuntimeError("Invalid valid_shape", \
                    valid_shape, shape_in)

            if fusion_para.get("input_memory_type") == 1:
                if slice_offset[2] == 0 and pad_bottom != 0:
                    raise RuntimeError("if it is the first cut shape, "
                                       "the pad_bottom must be 0")
                if slice_offset[2] == (shape_in[2] - valid_shape[2]) and \
                pad_top != 0:
                    raise RuntimeError("if it is the last cut shape, "
                                       "the pad_top must be 0")
                if (slice_offset[2] > 0 and slice_offset[2] < \
                    (shape_in[2] - valid_shape[2])) and \
                    (pad_top != 0 or pad_bottom != 0):
                    raise RuntimeError("if it is the middle cut shape, "
                                       "the pad_top and pad_bottom must be 0")

        if fusion_para is None:
            fusion_para = {"input_memory_type": 0, \
            "output_memory_type": 0, \
            "valid_shape": (), \
            "slice_offset": (), \
            "l1_fusion_type": -1}
        valid_shape = fusion_para.get("valid_shape")
        slice_offset = fusion_para.get("slice_offset")
        l1_fusion_type = fusion_para.get("l1_fusion_type")
        input_memory_type = fusion_para.get("input_memory_type")
        output_memory_type = fusion_para.get("output_memory_type")

        if l1_fusion_type == -1:
            if input_memory_type == 1 or output_memory_type == 1:
                raise RuntimeError("input_memory_type/output_memory_type"
                    " must be 0 when l1_fusion_type is -1", \
                    input_memory_type,
                    output_memory_type)

        if valid_shape:
            handle_valid_shape()

    def _l1_buffer_size_check(max_feature_map_l1, fusion_para):
        """
         check for not bigger than L1 size
        """
        l1_buffer_size = cce_conf.get_soc_spec("L1_SIZE")
        l1_fusion_type = fusion_para.get("l1_fusion_type")
        if (l1_fusion_type == 1) or (l1_fusion_type == 0):
            pass
        elif max_feature_map_l1 > l1_buffer_size:
            raise RuntimeError(
                "Input is too large, the minimum tiling may exceed L1_Buffer")

    if shape_in[2] < FMAP_HW_MIN or shape_in[2] > FMAP_HW_MAX:
        raise RuntimeError("feature H must be in [1,4096].")
    if shape_in[3] < FMAP_HW_MIN or shape_in[3] > FMAP_HW_MAX:
        raise RuntimeError("feature W must be in [1,4096].")

    util.check_shape_rule(shape_in, CONV_SHAPE_DIM, CONV_SHAPE_DIM)
    util.check_shape_rule(shape_w, CONV_SHAPE_DIM, CONV_SHAPE_DIM)

    if shape_in[1] != shape_w[1]:
        raise RuntimeError("input feature map channel should equal "
                           "to filter channel")

    if optim_dict is None:
        optim_dict = {"c0_optim_flg": False}
    block_size_k = CUBE_MKN[in_dtype]['mac'][1]
    shape_in[1] = ((shape_in[1] + block_size_k - 1) //
                   block_size_k)*block_size_k
    # InQuant feature_map_channel_in%16=0, but weight_channel_in%32=0
    shape_w[1] = ((shape_in[1] + block_size_k - 1) //
                  block_size_k)*block_size_k
    if optim_dict["c0_optim_flg"]:
        shape_in[1] = 4
        shape_w[1] = 4
    h_i = shape_in[2]
    w_i = shape_in[3]
    h_k = shape_w[2]
    w_k = shape_w[3]

    # V200 does not support dialtion
    if is_v200_version() and (dilateh != 1 or dilatew != 1):
        raise RuntimeError("V200 does not support dilation")

    # dilateh, dilatew check
    if dilateh < DILATE_MIN or dilateh > DILATE_MAX:
        raise RuntimeError("dilateh must be in [1,255].")
    if dilatew < DILATE_MIN or dilatew > DILATE_MAX:
        raise RuntimeError("dilatew must be in [1,255].")

    hk_dilation = (h_k - 1)*dilateh + 1
    wk_dilation = (w_k - 1)*dilatew + 1

    # calculated by h_i and w_i
    h_out = (h_i + (pad_top + pad_bottom) - hk_dilation) // strideh + 1
    # calculated by h_i and w_i
    w_out = (w_i + (pad_left + pad_right) - wk_dilation) // stridew + 1

    def _check_load3d_constraint():
        load2d_pass_flag = (h_k == 1) and (w_k == 1) and \
                           (pad_top == 0) and (pad_bottom == 0) and \
                           (pad_left == 0) and (pad_right == 0) and \
                           (strideh == 1) and (stridew == 1)
        #  Chip Design demand only h_dimesion constraint
        only_fhkh_flag = (h_i + (pad_top + pad_bottom) - hk_dilation) == 0
        #  Chip Design demand both h_dimesion and w_dimension constraint
        fhkh_fwkw_pass_flag = (1 <= wk_dilation <= 11) and \
                              (hk_dilation == wk_dilation) and \
                              ((h_i + (pad_top + pad_bottom) -
                                hk_dilation) == 0) and \
                              ((w_i + (pad_left + pad_right) -
                                wk_dilation) == 0)

        if h_out < 1 or w_out < 1:
            raise RuntimeError("output shape should greater than 0, "
                               "please check input shape\n")
        if h_out == 1 and w_out == 1:
            if not fhkh_fwkw_pass_flag:
                raise RuntimeError("op [Conv2D] output featuremap h*w == 1*1, "
                                   "the input parameter must follow rule: "
                                   "Filter_h(after dilation) == "
                                   "Filter_w(after dilation) and "
                                   "1 <=Filter_w(after dilation) <= 11")
        elif h_out == 1:
            if not (only_fhkh_flag or load2d_pass_flag):
                raise RuntimeError("op [Conv2D] output featuremap h == 1, "
                                   "the input parameter must follow rule: "
                                   "Fmap_h(after padding) == "
                                   "Filter_h(after dilation)")
        elif w_out == 1:
            if not load2d_pass_flag:
                raise RuntimeError("op [Conv2D] output featuremap w == 1, "
                                   "the input parameter must follow rule: "
                                   "filters_h x filter_w == 1x1 and "
                                   "paddings = [0, 0, 0, 0] and "
                                   "stride_h x stride_w = 1x1")
        else:
            pass

    _check_load3d_constraint()

    w_block_size_n = CUBE_MKN[w_dtype]['mac'][2]
    shape_w[0] = ((shape_w[0] + w_block_size_n - 1) // \
        w_block_size_n)*w_block_size_n

    # filterH, filterW check(before dilation according to chip design demand )
    if shape_w[2] < FILTER_HW_MIN or shape_w[2] > FILTER_HW_MAX:
        raise RuntimeError("kernel H must be in [1,255].")
    if shape_w[3] < FILTER_HW_MIN or shape_w[3] > FILTER_HW_MAX:
        raise RuntimeError("kernel W must be in [1,255].")

    # padh, padw check
    if pad_top < PAD_MIN or pad_bottom < PAD_MIN or \
            pad_top > PAD_MAX or  pad_bottom > PAD_MAX:
        raise RuntimeError("pad_top or pad_bottom must be in [0,255].")
    if pad_left < PAD_MIN or pad_right < PAD_MIN or \
            pad_left > PAD_MAX or pad_right > PAD_MAX:
        raise RuntimeError("pad_left or pad_right must be in [0,255].")

    # strideh, stridew check
    if strideh < STRIDE_MIN or strideh > STRIDE_MAX:
        raise RuntimeError("strideh must be in [1,63].")
    if stridew < STRIDE_MIN or stridew > STRIDE_MAX:
        raise RuntimeError("stridew must be in [1,63].")

    config = CUBE_MKN[w_dtype]
    ci0 = config['mac'][1]
    if ci0 <= 0:
        raise RuntimeError("ci0 must > 0")

    fusion_para_check(fusion_para, pad_top, pad_bottom, shape_in)

    # check for not bigger than L1
    m_bit_ratio = {"float16": 2, "int8": 1}
    point_per_w = math.floor((w_i - wk_dilation + pad_left + \
        pad_right) / stridew) + 1
    w_in = math.floor(config['mac'][0] / point_per_w) + 2
    tmp = ((w_in - 1)*strideh + hk_dilation)*w_i
    max_feature_map_l1 = ci0*tmp*m_bit_ratio[w_dtype]

    _l1_buffer_size_check(max_feature_map_l1, fusion_para)

    return shape_in, shape_w

class ConvParam:
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

    @classmethod
    def set_default(cls):
        cls.TENSOR_MAP.clear()
        cls.dim_map.clear()
        cls.tiling = None
        cls.tiling_query_param.clear()
        cls.convbn1_flag = False
        cls.l1_fusion_workspace_tensor_list = None
        cls.fusion_para = {"input_memory_type": 0, \
            "output_memory_type": 0, "valid_shape": (), "slice_offset": (), \
            "l1_fusion_type": -1}
        cls.conv_deq_req_double_out = False
        cls.swrite_flag = False
        cls.swrite_dequant_flag = False


    TENSOR_MAP = {}
    dim_map = {}
    tiling = None
    tiling_query_param = {}
    convbn1_flag = False
    l1_fusion_workspace_tensor_list = []
    fusion_para = {"input_memory_type": 0, "output_memory_type": 0,
                   "valid_shape": (), "slice_offset": (), "l1_fusion_type": -1}
    conv_deq_req_double_out = False
    swrite_flag = False
    swrite_dequant_flag = False
    compress_index_shape = {}
    fractal_weight_size = {}


def shape_to_list(shape):
    """
    translate tvm.shape to list type in python
    """
    if isinstance(shape, (list, tuple)):
        return shape
    tmp = []
    for i in shape:
        tmp.append(i.value)
    return tmp


def int_ceil_div(num_a, num_b):
    """
    upper division
    """
    if num_b == 0:
        raise RuntimeError(" division by zero")
    return (num_a + num_b - 1) // num_b


def _fmap_c0_check_value(dtype, optim_dict):
    """
    fmap c0 check value
    """
    fmap_c0_check_value = 4 if optim_dict["c0_optim_flg"] and \
    (is_v200_version() or is_lhisi_version()) else CUBE_MKN[dtype]['mac'][1]
    return fmap_c0_check_value

OP_TAG = "convolution_"
TENSOR_MAP = {}
DIM_MAP = {}
NAME_INDEX = [0]
SQRT = {}

@tvm.target.generic_func
def conv_compress(inputs, weight_compress, compress_index, compress_index_shape,
                  para_dict, optim_dict=None, dsl_flag=True):
    weight_compress_shape = weight_compress.shape
    fractal_weight_size = tvm.var("fractal_weight_size", dtype="int32")
    ConvParam.compress_index_shape = compress_index_shape
    ConvParam.fractal_weight_size = fractal_weight_size
    weight = tvm.compute(weight_compress_shape, lambda i, j, k, l: tvm.unzip( \
        compress_index(((l + weight_compress_shape[-1] * (k + \
        weight_compress_shape[-2] * (j + i * weight_compress_shape[-3]))) // \
        fractal_weight_size) * 8), weight_compress(i, j, k, l)), \
        name='weight_unzip')
    res = conv(inputs, weight, para_dict, optim_dict, dsl_flag)
    return res

@tvm.target.generic_func
def conv(data, weight, para_dict, optim_dict=None, dsl_flag=True):
    """
    conv

    Parameters
    ----------
    data: feature map

    weight: filter

    para_dict: dict of params

    dsl_flag: true if not from topi

    Returns
    -------
    tensor : res
    """
    def _v200_l0c2ub(c_col, res_dtype):
        c_ub = tvm.compute(ConvParam.mad_shape,
                           lambda *indices: c_col(*indices).astype(res_dtype),
                           name='C_UB',
                           tag=OP_TAG + "C_UB")
        TENSOR_MAP["c_ub"] = c_ub
        return c_ub

    def _fmap_ddr2l1(fmap, fmap_shape, strideh_opti_flag, valid_shape):
        # remove invalid fmap data in h for optimization
        if strideh_opti_flag:
            if valid_shape:
                fmap_l1_shape = valid_shape  # NC1HWC0
                fmap_l1_shape[2] = (fmap_l1_shape[2] - 1) // stride_h + 1

                offset = ConvParam.fusion_para["slice_offset"]
                n_offset, c1_offset, h_offset, w_offset, c0_offset = offset

                fmap_l1 = tvm.compute(fmap_l1_shape, lambda n_idx, \
                                      ci1_idx, \
                                      hi_idx, \
                                      wi_idx, \
                                      ci0_idx: fmap[n_idx, \
                                      ci1_idx, \
                                      (hi_idx + h_offset)*stride_h, \
                                      wi_idx, \
                                      ci0_idx], \
                                      name="fmap_l1")
                TENSOR_MAP["fmap_l1"] = fmap_l1
                return fmap_l1
            else:
                fmap_l1_shape = fmap_shape  # NC1HWC0
                fmap_l1_shape[2] = (fmap_l1_shape[2] - 1) // stride_h + 1

                fmap_l1 = tvm.compute(fmap_l1_shape, lambda n_idx, \
                                      ci1_idx, \
                                      hi_idx, \
                                      wi_idx, \
                                      ci0_idx: fmap[n_idx, \
                                      ci1_idx, \
                                      hi_idx*stride_h, \
                                      wi_idx, \
                                      ci0_idx], \
                                      name="fmap_l1")
                TENSOR_MAP["fmap_l1"] = fmap_l1
                return fmap_l1
        return None

    def _row_major_c0_value(fmap_shape, optim_dict):
        # row major res c0 value
        row_major_c0_value = 4 if optim_dict["c0_optim_flg"] else fmap_shape[4]
        return row_major_c0_value

    def _v100_cal_im2col_row_major(fmap, fmap_im2col_row_major_shape, \
                                   fmap_l1, optim):
        filter_w, padding, stride, dilate, strideh_opti_flag = optim

        if strideh_opti_flag:
            fmap_im2col_row_major_res = im2col_row_major(
                fmap_im2col_row_major_shape, fmap_l1, \
                filter_w, padding, (1, stride_w), \
                dilate, fmap.dtype)
        else:
            fmap_im2col_row_major_res = im2col_row_major(
                fmap_im2col_row_major_shape, \
                fmap, filter_w, padding, stride, \
                dilate, fmap.dtype)
        TENSOR_MAP["fmap_im2col_row_major_res"] = fmap_im2col_row_major_res
        return fmap_im2col_row_major_res

    def _cube_compute(data, weight, mad_dtype, tiling=None, \
        optim_dict=None, bias=None):

        def _config_sread_flag():
            if fmap.op.tag == "strided_read":
                TENSOR_MAP["fmap"] = fmap.op.input_tensors[0] # A_DDR
                TENSOR_MAP["strided_read_flag"] = True
            else:
                TENSOR_MAP["fmap"] = fmap
                TENSOR_MAP["strided_read_flag"] = False

        def _fusion_fmap_select(fmap):
            """
             check L1 fusion fmap select
            """
            valid_shape = ConvParam.fusion_para.get("valid_shape")

            offset = ConvParam.fusion_para.get("slice_offset")
            input_memory_type = ConvParam.fusion_para.get("input_memory_type")
            if offset and input_memory_type != 1 :
                if TENSOR_MAP["strideh_opti_flag"]:
                    # do it in _fmap_ddr2l1
                    pass
                else:
                    n_offset, c1_offset, h_offset, w_offset, c0_offset = offset
                    data_res = tvm.compute(valid_shape, lambda n, c1, h, w, c0:
                                           fmap(n + n_offset,
                                                c1 + c1_offset,
                                                h + h_offset,
                                                w + w_offset,
                                                c0 + c0_offset),
                                           name="fusion_fmap_select")
                    fmap = data_res
                    TENSOR_MAP['fusion_fmap_select'] = fmap
            return fmap

        def _mad_res(l0a_load2d_flag):
            """
             mad res
            """
            if l0a_load2d_flag:
                shape_al1_load2d = (batch_size,
                                    in_channel_c1,
                                    feature_map_h*feature_map_w,
                                    in_channel_c0)
                al1_load2d = tvm.compute(shape_al1_load2d, \
                    lambda n, c1, m, c0: fmap( \
                        n, c1, m // feature_map_w, m % feature_map_w, c0), \
                    name=OP_TAG + "al1_load2d")
                TENSOR_MAP["al1_load2d"] = al1_load2d

                hw_dim = int_ceil_div(feature_map_h*feature_map_w, \
                    CUBE_MKN[fmap.dtype]["mac"][0])
                shape_al0_load2d = (batch_size, hw_dim, in_channel_c1, \
                    CUBE_MKN[fmap.dtype]["mac"][0], \
                    in_channel_c0)

                al0_load2d = tvm.compute(shape_al0_load2d, \
                    lambda n, m1, c1, m0, c0: al1_load2d( \
                        n, c1, m0 + CUBE_MKN[fmap.dtype]["mac"][0]*m1, c0), \
                    name=OP_TAG + "al0_load2d")
                TENSOR_MAP["al0_load2d"] = al0_load2d

                c_col = mad(mad_shape, al0_load2d, weight, config, mad_dtype)
            else:
                c_col = mad(mad_shape, fmap_im2col_fractal_res, weight, \
                    config, mad_dtype)
            return c_col

        def _get_l0a_load2d_flag():
            """
             get l0a_load2d_flag
            """
            l0a_load2d_flag = False
            if (padding == (0, 0, 0, 0)) and (stride == (1, 1) \
            and w_dtype == "float16") \
                and (filter_h*filter_w == 1) and (not valid_shape):
                l0a_load2d_flag = True
                optim_dict["c0_optim_flg"] = False

            l1_fusion_type = ConvParam.fusion_para.get("l1_fusion_type")
            if  (l1_fusion_type == 0) or (l1_fusion_type == 1):
                l0a_load2d_flag = False
            return l0a_load2d_flag

        fmap = data
        in_dtype = fmap.dtype

        _config_sread_flag()

        TENSOR_MAP["filter"] = weight

        strideh_opti_flag = (filter_h == 1 and stride_h > 1) \
            and not optim_dict["c0_optim_flg"]

        input_memory_type = ConvParam.fusion_para.get("input_memory_type")
        if input_memory_type == 1:
            strideh_opti_flag =False

        TENSOR_MAP["strideh_opti_flag"] = strideh_opti_flag
        TENSOR_MAP["c0_optim_flg"] = optim_dict["c0_optim_flg"] \
            and (not is_v200_version() and not is_lhisi_version())

        fmap = _fusion_fmap_select(fmap)
        valid_shape = ConvParam.fusion_para.get("valid_shape")

        fmap_shape = shape_to_list(fmap.shape)

        if valid_shape:
            fmap_shape = valid_shape

        batch_size = fmap_shape[0]
        in_channel_c1 = fmap_shape[1]
        feature_map_h = fmap_shape[2]
        feature_map_w = fmap_shape[3]
        in_channel_c0 = fmap_shape[4]
        weight_shape = shape_to_list(weight.shape)
        out_channel_c1 = weight_shape[1]
        out_channel_c0 = weight_shape[2]
        out_channel = out_channel_c1*out_channel_c0

        padding_top = pad_h[0]
        padding_bottom = pad_h[1]
        padding_left = pad_w[0]
        padding_right = pad_w[1]
        filter_h_dilate = (filter_h - 1)*dilate_h + 1
        filter_w_dilate = (filter_w - 1)*dilate_w + 1

        height_out = (feature_map_h + padding_top + \
                      padding_bottom - filter_h_dilate) // stride_h + 1
        width_out = (feature_map_w + padding_left + \
                     padding_right - filter_w_dilate) // stride_w + 1

        config = CUBE_MKN[in_dtype]
        block_size_k = config['mac'][1]
        block_size_m = config['mac'][0]
        padding = (padding_top, padding_bottom, padding_left, padding_right)
        stride = (stride_h, stride_w)
        dilate = (dilate_h, dilate_w)

        ConvParam.padding = padding
        # DDR -> L1
        fmap_l1 = _fmap_ddr2l1(fmap, fmap_shape, strideh_opti_flag, \
            valid_shape)

        # set_fmatrix
        # calculate im2col_row_major
        in_channel_c0_row_major_res = _row_major_c0_value(
            fmap_shape, optim_dict)
        fmap_im2col_row_major_shape = (batch_size,
                                       height_out*width_out,
                                       in_channel_c1,
                                       filter_h,
                                       filter_w,
                                       in_channel_c0_row_major_res)
        fmap_im2col_row_major_res = \
        _v100_cal_im2col_row_major(fmap, \
            fmap_im2col_row_major_shape, \
            fmap_l1, \
            [filter_w, padding, stride, \
            dilate, strideh_opti_flag])

        # im2col
        # small-z-big-Z
        input_memory_type = ConvParam.fusion_para.get("input_memory_type")
        if valid_shape and input_memory_type == 1:
            height_out = (valid_shape[2] + padding_top + \
                padding_bottom - filter_h_dilate) // stride_h + 1
            width_out = (valid_shape[3] + padding_left + \
                padding_right - filter_w_dilate) // stride_w + 1
        howo_mad = (height_out*width_out + block_size_m -
                    1) // block_size_m*block_size_m
        k_size = (
            in_channel_c0_row_major_res*in_channel_c1*filter_h*filter_w + \
            block_size_k - 1) // block_size_k
        fmap_im2col_fractal_shape = (batch_size, \
                                     howo_mad // block_size_m, \
                                     k_size, \
                                     block_size_m, \
                                     block_size_k)
        fmap_im2col_fractal_res = im2col_fractal(
            fmap_im2col_fractal_shape, fmap_im2col_row_major_res, \
            config, fmap.dtype)

        if is_v200_version() or is_lhisi_version():
            input_k_block = (in_channel_c1*filter_h*filter_w*in_channel_c0 +
                             block_size_k - 1) // block_size_k * block_size_k
            row_major_reshape_shape = (batch_size, howo_mad, input_k_block)
            row_major_reshape_res = \
            _im2col_row_major_reshape(row_major_reshape_shape, \
                fmap_im2col_row_major_res, \
                fmap.dtype)
            fmap_im2col_fractal_res = \
            _im2col_fractal_v200(fmap_im2col_fractal_shape, \
                row_major_reshape_res, \
                config)
            TENSOR_MAP["row_major_reshape_res"] = row_major_reshape_res
        TENSOR_MAP["fmap_im2col_fractal_res"] = fmap_im2col_fractal_res

        config = CUBE_MKN[res_dtype]
        mad_shape = (batch_size, (out_channel + config['mac'][2] - 1) // (
            config['mac'][2]), howo_mad, config['mac'][2])

        config = CUBE_MKN[w_dtype]
        ConvParam.mad_shape = mad_shape

        l0a_load2d_flag = _get_l0a_load2d_flag()

        TENSOR_MAP["l0a_load2d_flag"] = l0a_load2d_flag

        c_col = _mad_res(l0a_load2d_flag)
        TENSOR_MAP["c_col"] = c_col

        conv_shape = (batch_size, (out_channel + config['mac'][2] - 1) // (
            config['mac'][2]), height_out*width_out, config['mac'][2])
        DIM_MAP["out_img_shape"] = conv_shape
        ConvParam.conv_shape = conv_shape

        filter_shape = [out_channel, filter_h, filter_w, 1]
        dim_map1 = im2col_dim(shape_to_list(fmap.shape), \
                              filter_shape, list(padding), list(stride), \
                              list(dilate), config)
        dim_map_copy = DIM_MAP.copy()
        dim_map_copy.update(dim_map1)

        ConvParam.dim_map = dim_map_copy
        ConvParam.tiling = tiling
        TENSOR_MAP["conv_vector_fused_flag"] = False
        TENSOR_MAP["bias_optimize_flag"] = False

        if isinstance(bias, tvm.tensor.Tensor):
            TENSOR_MAP["bias"] = bias
        bias_tensor_flag = isinstance(bias, tvm.tensor.Tensor)
        bias_l0c = None
        bias_optimize_flag = True

        howo_mad = (height_out*width_out + block_size_m -
                    1) // block_size_m*block_size_m

        mad_shape = (batch_size, (out_channel + config['mac'][2] - 1) // (
            config['mac'][2]), howo_mad, config['mac'][2])
        conv_shape = DIM_MAP["out_img_shape"]
        if bias_tensor_flag:
            if bias_optimize_flag:
                bias_ub_brc_shape = list(mad_shape)
                bias_ub_brc_shape[2] = bias_ub_brc_shape[2] // 16
                bias_ub_brc = tvm.compute(bias_ub_brc_shape, \
                    lambda i, j, k, l: \
                    bias_tensor(j * config['mac'][2] + l), \
                    name=OP_TAG + 'bias_ub_brc')
                bias_l0c = tvm.compute(mad_shape, lambda i1, j1, k1, l1: \
                                       bias_ub_brc(i1, j1, k1 // 16, l1), \
                                       name=OP_TAG + 'bias_l0c')
                TENSOR_MAP["bias_ub_brc"] = bias_ub_brc
                TENSOR_MAP["bias_l0c"] = bias_l0c
            else:
                bias_l0c = \
                tvm.compute(mad_shape, lambda i1, j1, k1, l1: \
                    bias_tensor(j1 * config['mac'][2] + l1), \
                    name=OP_TAG + 'bias_l0c')
                TENSOR_MAP["bias_l0c"] = bias_l0c

        TENSOR_MAP["bias_optimize_flag"] = bias_optimize_flag
        config = CUBE_MKN[w_dtype]

        if bias_tensor_flag:
            c_col = tvm.compute(mad_shape, lambda *index: \
                bias_l0c(*index) + c_col(*index), \
                                name=OP_TAG + 'c_col_bias')
            TENSOR_MAP["c_col_bias"] = c_col

        ConvParam.TENSOR_MAP = TENSOR_MAP
        return c_col

    def conv_and_quant_compute(data, weight, mad_dtype, res_dtype, \
        pad_h, pad_w, stride_h, \
        stride_w, dilate_h, dilate_w, filter_h, filter_w, bias=False, \
        no_vector=False, tiling=None, conv_fused_flag=False, \
        optim_dict=None, kernel_name=None):
        """
        conv

        Parameters
        ----------
        data : tvm.tensor, Feature Map

        weight: tvm.tensor, Filter

        res_dtype : the result data type

        pad_h: the padding shape in height

        pad_w: the padding shape in weight

        stride_h: the stride value in height

        stride_w: the stride value in weight

        dilate_h: the dilate value in H

        dilate_w: the dilate value in Weight

        filter_h: kernel size of height

        filter_w: kernel_size of weight

        bias: the tag for bias or not

        drq_scale: scale tensor of DeQuant or ReQuant

        scalr_vector_flag: the tag for scalar mode or vector mode

        offset_pad: offset_pad tensor for ReQuant in half-offset mode

        no_vector: the tag for conv has vector compute or not

        tiling: default none, tiling

        conv_fused_flag: the tag indicates conv fusion
        -------
        Returns

        wrapped_tensor
        """
        fmap = data
        in_dtype = fmap.dtype

        fmap_shape = shape_to_list(fmap.shape)
        batch_size = fmap_shape[0]

        valid_shape = ConvParam.fusion_para.get("valid_shape")
        if valid_shape:
            feature_map_h = valid_shape[2]
            feature_map_w = valid_shape[3]
        else:
            feature_map_h = fmap_shape[2]
            feature_map_w = fmap_shape[3]

        weight_shape = shape_to_list(weight.shape)
        out_channel_c1 = weight_shape[1]
        out_channel_c0 = weight_shape[2]
        out_channel = out_channel_c1*out_channel_c0

        padding_top = pad_h[0]
        padding_bottom = pad_h[1]
        padding_left = pad_w[0]
        padding_right = pad_w[1]

        filter_h_dilate = (filter_h - 1)*dilate_h + 1
        filter_w_dilate = (filter_w - 1)*dilate_w + 1
        height_out = (feature_map_h + padding_top + \
                      padding_bottom - filter_h_dilate) // stride_h + 1
        width_out = (feature_map_w + padding_left + \
                     padding_right - filter_w_dilate) // stride_w + 1

        config = CUBE_MKN[in_dtype]
        block_size_m = config['mac'][0]
        padding = (padding_top, padding_bottom, padding_left, padding_right)
        stride = (stride_h, stride_w)
        dilate = (dilate_h, dilate_w)

        c_col = _cube_compute(fmap, weight, mad_dtype, \
            tiling, optim_dict, bias)

        howo_mad = (height_out*width_out + block_size_m -
                    1) // block_size_m*block_size_m

        mad_shape = (batch_size, \
            (out_channel + config['mac'][2] - 1) // (config['mac'][2]), \
            howo_mad, \
            config['mac'][2])
        conv_shape = DIM_MAP["out_img_shape"]
        config = CUBE_MKN[w_dtype]
        c_ub = tvm.compute(mad_shape, lambda n, i, j, k: \
            c_col(n, i, j, k).astype(res_dtype), \
            name='C_UB', tag=OP_TAG + "C_UB", \
            attrs={'true_shape': conv_shape, \
            'no_vector': no_vector, \
            'sqrt': False, \
            'res_dtype': res_dtype, \
            'kernel_h': filter_h, \
            'kernel_w': filter_w, \
            'padding': padding, \
            'stride': stride, \
            'dilate': dilate, \
            'width_out': width_out, \
            'kernel_name': kernel_name})

        TENSOR_MAP["c_ub"] = c_ub
        res_c = tvm.compute(conv_shape, \
            lambda *indice: c_ub[indice], name='C', \
            tag=OP_TAG + "C", \
            attrs={'width_out': width_out})
        TENSOR_MAP["C"] = res_c

        conv_dsl_outputs = [TENSOR_MAP["fmap_im2col_row_major_res"], \
                            TENSOR_MAP["fmap_im2col_fractal_res"], \
                            c_col, c_ub, res_c]

        if not no_vector:
            TENSOR_MAP["C"] = None

        filter_shape = [out_channel, filter_h, filter_w, 1]
        dim_map1 = im2col_dim(shape_to_list(fmap.shape), \
            filter_shape, list(padding), \
            list(stride), list(dilate), config)
        dim_map_copy = DIM_MAP.copy()
        dim_map_copy.update(dim_map1)

        if conv_fused_flag:
            TENSOR_MAP["conv_vector_fused_flag"] = True
        else:
            TENSOR_MAP["conv_vector_fused_flag"] = False

        ConvParam.TENSOR_MAP = TENSOR_MAP
        ConvParam.dim_map = dim_map_copy
        ConvParam.tiling = tiling

        if conv_fused_flag:
            return conv_dsl_outputs[-1]
        return conv_dsl_outputs[-2]

    def im2col_dim(img_shape, filter_shape, pad, stride, dilate, config):
        """
        calculate shape
        Parameters

        ----------
        img_shape : shape of feature

        filter_shape : shape of filter

        pad: the padding shape

        stride: the stride value

        dilate: the dilate value

        Returns
        -------
        img_shape, fmap_matrix_dim
        """
        mac_dim = config['mac']

        batch = img_shape[0]
        out_h = ((img_shape[-3] + pad[2] + pad[3]) - \
                ((filter_shape[-3]-1)*dilate[0] + 1)) // stride[0] + 1
        out_w = ((img_shape[-2] + pad[0] + pad[1]) - \
                ((filter_shape[-2]-1)*dilate[1] + 1)) // stride[1] + 1

        fmap_valid_dim = (batch, out_h*out_w, \
            img_shape[-4]*img_shape[-1]*filter_shape[-2]*filter_shape[-3])

        fmap_matrix_dim = (batch, \
            ((fmap_valid_dim[-2] + mac_dim[0] - 1) // mac_dim[0]), \
            ((fmap_valid_dim[-1] + mac_dim[1] - 1) // mac_dim[1]), \
            mac_dim[0], mac_dim[1])

        filter_valid_dim = (img_shape[-4]*filter_shape[-3]*filter_shape[-2] \
                            *img_shape[-1], filter_shape[-4]*filter_shape[-1])

        filter_matrix_dim = ((filter_valid_dim[-2] + mac_dim[1] - 1) \
            // mac_dim[1], \
            (filter_valid_dim[-1] + mac_dim[2] - 1) // mac_dim[2], \
            mac_dim[2], mac_dim[1])

        return {
            "img_shape": img_shape,
            "fmap_matrix_dim": fmap_matrix_dim,
            "filter_matrix_dim": filter_matrix_dim}

    def im2col_row_major(
            fmap_im2col_vm_shape, fmap, kernel_w, padding, stride, \
            dilate, compute_dtype):
        """
        calculate im2col_row_major tensor

        Parameters
        ----------
        fmap_im2col_vm_shape : shape of fmap_im2col_row_major

        fmap : feature map

        kernel_w: the kernel value in  w

        padding: the padding shape

        stride: the stride value

        dilate: the dilate value

        compute_dtype: dtype of compute result

        Returns
        -------
        fmap_im2col_row_major tensor
        """

        def __im2col_row_major_indices(indices, fmap, kernel_w, \
            padding, stride, dilate):
            """
            calculate im2col_row_major tvm lambda function
            Parameters
            ----------
            indices : indices in lambda function

            fmap : feature map

            padding: the padding shape

            stride: the stride value

            dilate: the dilate value

            Returns
            -------
            im2col_row_major tvm lambda function
            """
            _, _, input_h, input_w, _ = fmap.shape
            n_index, howo, c1_index, k_h, k_w, c0_index = indices
            stride_h, stride_w = stride
            dilate_h, dilate_w = dilate
            padding_top, _, padding_left, padding_right = padding
            width_out = (input_w.value + padding_left + padding_right \
                - ((kernel_w - 1)*dilate_w + 1)) // (stride_w) + 1

            h_index = (howo // width_out)*stride_h + k_h*dilate_h
            w_index = (howo % width_out)*stride_w + k_w*dilate_w
            input_memory_type = ConvParam.fusion_para.get("input_memory_type")
            slice_offset = ConvParam.fusion_para.get("slice_offset")
            offset = slice_offset[2] if (slice_offset and \
                input_memory_type == 1) else 0
            return tvm.select( \
                tvm.any(h_index < padding_top, \
                    h_index > input_h.value + padding_top - 1, \
                    w_index < padding_left, \
                    w_index > input_w.value + padding_left - 1), \
                tvm.const(offset_x, compute_dtype), \
                fmap(n_index, c1_index, h_index - padding_top + offset, \
                    w_index - padding_left, c0_index))

        return tvm.compute(fmap_im2col_vm_shape, lambda *indices: \
            __im2col_row_major_indices(indices, fmap, kernel_w, padding, \
                stride, dilate), name='im2col_row_major', \
            tag=OP_TAG + 'im2col_row_major')

    def im2col_fractal(fmap_im2col_shape, fmap, \
        config, compute_dtype):
        """
        calculate im2col_fractal tensor
        Parameters
        ----------
        fmap_im2col_shape : shape of fmap_im2col

        fmap : feature map

        config: the config of cube

        compute_dtype: dtype of compute result

        Returns
        -------
        fmap_im2col_fractal tensor
        """

        def __im2col_fractal_indices(indices, fmap):
            """
            calculate im2col_fractal tvm lambda function
            Parameters
            ----------
            indices : indices in lambda function

            fmap : feature map

            Returns
            -------
            im2col_fractal tvm lambda function
            """
            block_size = config['mac'][1]
            block_size_m = config['mac'][0]
            _, howo, _, kernel_h, kernel_w, _ = fmap.shape
            batch_size, index_i1, index_j1, index_i0, index_j0 = indices
            n_index = batch_size

            hw_index = index_i1*block_size_m + index_i0

            c1_index = (((index_j1*block_size + index_j0) // block_size) //
                        kernel_w.value) // kernel_h.value

            kh_index = (((index_j1*block_size + index_j0) // block_size) //
                        kernel_w.value) % kernel_h.value

            kw_index = ((index_j1*block_size + index_j0) \
                // block_size) % kernel_w.value

            c0_index = (index_j1*block_size + index_j0) % block_size
            if optim_dict["c0_optim_flg"]:
                c1_index = 0
                kh_index = (index_j1*4 + index_j0 // 4) // kernel_w.value
                kw_index = (index_j1*4 + index_j0 // 4) % kernel_w.value
                c0_index = index_j0 % 4
            dtype = compute_dtype

            return tvm.select( \
                tvm.any(hw_index < 0, hw_index > howo.value - 1), \
                tvm.const(0.0, dtype), \
                fmap(n_index, hw_index, \
                    c1_index, kh_index, kw_index, c0_index))

        return tvm.compute(fmap_im2col_shape, lambda *indices:
                           __im2col_fractal_indices(indices, fmap),
                           name='im2col_fractal',
                           tag=OP_TAG + 'im2col_fractal')

    def _im2col_row_major_reshape(fmap_im2col_shape, \
        fmap_row_major, compute_dtype):
        """
        merage im2col_row_major axis of input_C1, filter_h, filter_w, input_C0
        Parameters
        ----------
        fmap_im2col_shape : shape of fmap_im2col

        fmap_row_major : feature map after im2col_row_major

        compute_dtype: dtype of compute result

        Returns
        -------
        row_major_reshape tensor
        """
        _, howo, input_c1, filter_h, filter_w, input_c0 = fmap_row_major.shape
        row_major_reshape = tvm.compute(
            fmap_im2col_shape, lambda i, j, k: tvm.select(
                tvm.all(k < input_c1*filter_h*filter_w*input_c0, j < howo),
                fmap_row_major(i, j, k // (filter_h*filter_w*input_c0),
                               k // (filter_w*input_c0) % filter_h,
                               k // (input_c0) % (filter_w),
                               k % input_c0), tvm.const(0.0, compute_dtype)),
            name="row_major_reshape",
            tag=OP_TAG + 'row_major_reshape')

        return row_major_reshape

    def _im2col_fractal_v200(fmap_im2col_shape, \
        im2col_row_major_reshape, config):
        """
        calculate im2col_fractal tensor
        Parameters
        ----------
        fmap_im2col_shape : shape of fmap_im2col

        im2col_row_major_reshape : feature map of _im2col_row_major_reshape

        config: the config of cube

        compute_dtype: dtype of compute result

        Returns
        -------
        fmap_im2col_fractal tensor
        """
        block_size_m = config['mac'][0]
        block_size_k = config['mac'][1]
        res_im2col_fractal = tvm.compute(
            fmap_im2col_shape, lambda i, j, k, l, m: im2col_row_major_reshape(
                i, j*block_size_m + l, k*block_size_k + m),
            name="_im2col_fractal",
            tag=OP_TAG + '_im2col_fractal')

        return res_im2col_fractal

    def mad(mad_shape, fmap, weight, config, mad_dtype):

        """
        calculate mad result tensor
        Parameters
        ----------
        mad_shape : shape of mad result

        fmap : feature map

        weight : filter

        config: the config of cube

        mad_dtype: dtype of mad output

        Returns
        -------
        mad result tensor
        """
        block_size = config['mac'][1]
        block_size_m = config['mac'][0]
        axis_k1 = tvm.reduce_axis((0, weight.shape[0]), name='k1')
        axis_k0 = tvm.reduce_axis((0, block_size), name='k0')

        if mad_dtype in ["float16", "int32"]:
            mode = 'f162f16'
        else:
            mode = 'f162f32'

        c_col = tvm.compute(
            mad_shape,
            lambda n, index_j1, i, index_j0:
            tvm.sum((
                fmap[n,
                     i // block_size_m,
                     axis_k1,
                     i % block_size_m,
                     axis_k0] * \
                weight[axis_k1,
                       index_j1,
                       index_j0,
                       axis_k0]).astype(mad_dtype),
                    axis=[axis_k1, axis_k0]),
            name='mad1',
            tag=OP_TAG + "c_col",
            attrs={'mode': mode})
        return c_col

    def bias_add(in_tensor0, in_tensor1):
        """
        calculate conv res + bias in UB
        Parameters
        ----------
        in_tensor0: conv res tensor

        in_tensor1: bias vector

        Returns
        -------
        in_tensor0+in_tensor1 tensor
        """
        dim_map = {}
        dim_map["out_img_shape"] = shape_to_list(in_tensor0.shape)
        NAME_INDEX[0] += 1

        with tvm.tag_scope('conv_vector_bias_add'):
            c_add_vector = \
            tvm.compute(dim_map["out_img_shape"], lambda *indice: \
                in_tensor0(*indice) + \
                in_tensor1(indice[1]*CUBE_MKN[in_tensor0.dtype]['mac'][2] \
                    + indice[3]), \
                name='bias_add_vector' + "_cc_" + str(NAME_INDEX[0]), \
                attrs={'width_out': in_tensor0.op.attrs["width_out"]})
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
            res_tensor = tvm.compute(res_remove_pad_shape, \
                lambda *indice: res( \
                    *indice), name='remove_pad' + "_cc_" + str(NAME_INDEX[0]))
        return res_tensor


    def check_optim_dict(optim_dict, para_dict, data, weight):
        if isinstance(optim_dict, dict) and "c0_optim_flg" in \
        optim_dict.keys() and isinstance(optim_dict["c0_optim_flg"], bool):
            pass
        else:
            raise RuntimeError("Invalid optim_dict check")

        kernel_one_one = (para_dict["filter_h"] == 1) and \
            (para_dict["filter_w"] == 1)
        if optim_dict["c0_optim_flg"]:
            c0_value = _fmap_c0_check_value(weight.dtype, optim_dict)
            if weight.dtype != "int8" and data.shape[1].value == 1 \
            and data.shape[4].value == c0_value \
            and weight.shape[3].value == 16 and not kernel_one_one:
                pass
            else:
                raise RuntimeError("Invalid config for c0=4 optimize feature.")


    def check_para_dict(para_dict, dtype, wtype):

        def check_para_dict_more():
            if "mad_dtype" not in para_dict:
                if wtype == "int8":
                    mad_dtype = "int32"
                elif is_lhisi_version():
                    mad_dtype = "float16"
                else:
                    mad_dtype = "float32"
                para_dict["mad_dtype"] = mad_dtype
            if "offset_x" not in para_dict:
                para_dict["offset_x"] = 0
            if "kernel_name" not in para_dict:
                para_dict["kernel_name"] = "conv2d"

        if not isinstance(para_dict, dict):
            RuntimeError("the third Input parameter must be a dict")
        if "pad_h" not in para_dict:
            if "padh" in para_dict:
                para_dict["pad_h"] = para_dict["padh"]
            else:
                raise RuntimeError("para_dict must contain pad_h")
        if "pad_w" not in para_dict:
            if "padw" in para_dict:
                para_dict["pad_w"] = para_dict["padw"]
            else:
                raise RuntimeError("para_dict must contain pad_w")
        if "stride_h" not in para_dict:
            if "strideh" in para_dict:
                para_dict["stride_h"] = para_dict["strideh"]
            else:
                raise RuntimeError("para_dict must contain stride_h")
        if "stride_w" not in para_dict:
            if "stridew" in para_dict:
                para_dict["stride_w"] = para_dict["stridew"]
            else:
                raise RuntimeError("para_dict must contain stride_w")
        if "dilate_h" not in para_dict:
            if "dilateh" in para_dict:
                para_dict["dilate_h"] = para_dict["dilateh"]
            else:
                para_dict["dilate_h"] = 1
        if "dilate_w" not in para_dict:
            if "dilatew" in para_dict:
                para_dict["dilate_w"] = para_dict["dilatew"]
            else:
                para_dict["dilate_w"] = 1
        if "filter_h" not in para_dict:
            if "filterh" in para_dict:
                para_dict["filter_h"] = para_dict["filterh"]
            else:
                raise RuntimeError("para_dict must contain filter_h")
        if "filter_w" not in para_dict:
            if "filterw" in para_dict:
                para_dict["filter_w"] = para_dict["filterw"]
            else:
                raise RuntimeError("para_dict must contain filter_w")

        check_para_dict_more()


    def check_data(data, optim_dict):
        if not isinstance(data, tvm.tensor.Tensor):
            raise RuntimeError(
                "the first Input parameter must be a tvm.tensor.Tensor")
        if len(data.shape) != 5:
            raise RuntimeError(
                "the first Input parameter must be a 5 dim tvm.tensor.Tensor")
        check_dtype_list = ('float16', )
        if is_v200_version() or is_mini_or_lhisi_version():
            check_dtype_list = ('int8', "float16")
        util.check_dtype_rule(data.dtype, check_dtype_list)

        block_size_k = 4 if optim_dict["c0_optim_flg"] and \
            (is_v200_version() or is_lhisi_version()) else \
            CUBE_MKN[data.dtype]['mac'][1]
        if data.shape[4].value != block_size_k:
            raise RuntimeError( \
                "the last dim of first "
                "Input parameter must be  %d" % block_size_k)


    def check_weight(weight):
        if not isinstance(weight, tvm.tensor.Tensor):
            raise RuntimeError(
                "the first Input parameter must be a tvm.tensor.Tensor")
        if len(weight.shape) != 4:
            raise RuntimeError(
                "the first Input parameter must be a 4 dim tvm.tensor.Tensor")

        check_dtype_list = ('float16', )
        if is_v200_version() or is_mini_or_lhisi_version():
            check_dtype_list = ('int8', "float16")

        util.check_dtype_rule(weight.dtype, check_dtype_list)
        block_size_k = CUBE_MKN[weight.dtype]['mac'][1]

        if weight.shape[3].value != block_size_k:
            raise RuntimeError( \
                "the last dim of first Input parameter must be %d" % \
                block_size_k)


    def _save_conv_param():
        ConvParam.pad_h = pad_h
        ConvParam.pad_w = pad_w
        ConvParam.stride_h = para_dict["stride_h"]
        ConvParam.stride_w = para_dict["stride_w"]
        ConvParam.dilate_h = para_dict["dilate_h"]
        ConvParam.dilate_w = para_dict["dilate_w"]
        ConvParam.filter_h = para_dict["filter_h"]
        ConvParam.filter_w = para_dict["filter_w"]
        ConvParam.mad_dtype = para_dict.get("mad_dtype")
        ConvParam.fusion_para = para_dict["fusion_para"]
        ConvParam.kernel_name = para_dict["kernel_name"]

    def _get_fmap_shape_nc1hwc0():
        """
        get fmap_shape_nc1hwc0
        """
        if para_dict.get("fusion_para") is not None:
            ConvParam.fusion_para = para_dict["fusion_para"]
            valid_shape = ConvParam.fusion_para.get("valid_shape")
            if valid_shape:
                fmap_shape_nc1hwc0 = ConvParam.fusion_para.get("valid_shape")
            else:
                fmap_shape_nc1hwc0 = tuple(shape_to_list(data.shape))
        else:
            fmap_shape_nc1hwc0 = tuple(shape_to_list(data.shape))
        return list(fmap_shape_nc1hwc0)

    def _get_dsl_fmap_shape_nc1hwc0():
        """
        get fmap_shape_nc1hwc0 for dsl interface
        """
        valid_shape = ConvParam.fusion_para.get("valid_shape")
        if valid_shape:
            fmap_shape_nc1hwc0 = tuple(shape_to_list(valid_shape))
        else:
            fmap_shape_nc1hwc0 = tuple(shape_to_list(data.shape))
        return fmap_shape_nc1hwc0

    def _fusion_para_get():
        """
        get fusion_para
        """
        if para_dict.get("fusion_para") is None:
            para_dict["fusion_para"] = {"input_memory_type": 0, \
            "output_memory_type": 0, \
            "valid_shape": (), "slice_offset": (), "l1_fusion_type": -1}

    ConvParam.set_default()

    if optim_dict is None:
        optim_dict = {"c0_optim_flg": False}
    pad_h = para_dict["pad_h"]
    pad_w = para_dict["pad_w"]
    if isinstance(pad_h, int):
        pad_h = [pad_h, pad_h]
    pad_top = pad_h[0]
    pad_bottom = pad_h[1]
    if isinstance(pad_w, int):
        pad_w = [pad_w, pad_w]
    pad_left = pad_w[0]
    pad_right = pad_w[1]
    if dsl_flag:
        check_optim_dict(optim_dict, para_dict, data, weight)
        check_data(data, optim_dict)
        check_weight(weight)
        check_para_dict(para_dict, data.dtype, weight.dtype)
        kernel_name = para_dict["kernel_name"]

        in_dtype = data.dtype
        w_dtype = weight.dtype

        stride_h = para_dict["stride_h"]
        stride_w = para_dict["stride_w"]
        dilate_h = para_dict["dilate_h"]
        dilate_w = para_dict["dilate_w"]
        filter_h = para_dict["filter_h"]
        filter_w = para_dict["filter_w"]
        offset_x = para_dict["offset_x"]
        mad_dtype = para_dict.get("mad_dtype")

        data_shape = shape_to_list(data.shape)
        in_channel_c1 = data_shape[1]
        in_channel_c0 = data_shape[4]
        in_channel_c = in_channel_c1*in_channel_c0
        if optim_dict["c0_optim_flg"]:
            in_channel_c = 4
        weight_shape = shape_to_list(weight.shape)
        out_channel_c1 = weight_shape[1]
        out_channel_c0 = weight_shape[2]
        out_channel = out_channel_c1*out_channel_c0
        _fusion_para_get()
        _save_conv_param()

        if not optim_dict["c0_optim_flg"] and \
                data_shape[1] != ((weight_shape[0] / filter_h) / filter_w):
            raise RuntimeError("data_shape[1] must equal to \
                ((weight_shape[0]/filter_h)/filter_w)")

        shape_in = [data_shape[0], in_channel_c, \
                    data_shape[2], data_shape[3]]
        shape_w = [out_channel, in_channel_c, filter_h, filter_w]
        res_dtype = "float16"
        if (is_v200_version() or is_mini_or_lhisi_version()) and \
                in_dtype == "int8" and w_dtype == "int8":
            res_dtype = "int32"
        ConvParam.res_dtype = res_dtype
        check_conv_shape(
            shape_in, shape_w, pad_top, pad_bottom, pad_left, \
            pad_right, stride_h, stride_w, in_dtype, w_dtype, \
            para_dict["fusion_para"], \
            optim_dict, dilateh=dilate_h, dilatew=dilate_w)

        block_size_k = CUBE_MKN[w_dtype]['mac'][1]
        if optim_dict["c0_optim_flg"]:
            block_size_k = 4
            in_channel_c = 4
        shape_w_nc1hwc0 = (
            out_channel, (in_channel_c + block_size_k - 1) //
            block_size_k, filter_h, filter_w, block_size_k)

        bias_tensor = None
        bias_tensor_flag = False

        if "bias_tensor" in para_dict.keys():
            bias_tensor = para_dict["bias_tensor"]
            if isinstance(bias_tensor, tvm.tensor.Tensor):
                bias_tensor_flag = True

        shape_fmap_nc1hwc0 = _get_dsl_fmap_shape_nc1hwc0()
        # for tiling c0 optim
        shape_fmap_nc1hwc0 = list(shape_fmap_nc1hwc0)
        shape_fmap_nc1hwc0[4] = _row_major_c0_value( \
            shape_fmap_nc1hwc0, optim_dict)
        tiling = None

        default_tiling = False
        if ("default_tiling" in ConvParam.tiling_query_param) and \
                ConvParam.tiling_query_param["default_tiling"]:
            default_tiling = ConvParam.tiling_query_param["default_tiling"]

        ConvParam.tiling_query_param = {
            "fmap_shape_nc1hwc0": shape_fmap_nc1hwc0,
            "shape_w_nc1hwc0": shape_w_nc1hwc0,
            "in_dtype": in_dtype,
            "w_dtype": w_dtype,
            "res_dtype": res_dtype,
            "mad_dtype": mad_dtype,
            "padw": pad_w,
            "padh": pad_h,
            "strideh": stride_h,
            "stridew": stride_w,
            "dilateh": dilate_h,
            "dilatew": dilate_w,
            "bias_flag": bias_tensor_flag,
            "default_tiling": default_tiling}

        if (is_v200_version() or is_mini_or_lhisi_version()) \
                and in_dtype == "int8":
            ConvParam.tiling_query_param["bias_flag"] = bias_tensor_flag
            conv_res = _cube_compute(data, weight, mad_dtype, \
                tiling=tiling, optim_dict=optim_dict, bias=bias_tensor)
            res_remove_pad_shape = list(conv_res.shape)
            res_remove_pad_shape[2] = DIM_MAP["out_img_shape"][2]
            res_remove_pad = remove_pad(conv_res, res_remove_pad_shape)
            TENSOR_MAP["l0c_remove_pad"] = res_remove_pad
            return res_remove_pad

        conv_res = conv_and_quant_compute(
            data, weight, mad_dtype, res_dtype, \
            pad_h, pad_w, stride_h, stride_w, dilate_h, dilate_w, \
            filter_h, filter_w, bias=False, no_vector=False, \
            tiling=tiling, conv_fused_flag=True, \
            optim_dict=optim_dict, kernel_name=kernel_name)

        if bias_tensor_flag:
            TENSOR_MAP["fp16_bias"] = bias_tensor
            res = bias_add(conv_res, bias_tensor)
            return res

        return conv_res

    in_dtype = data.dtype
    w_dtype = weight.dtype
    res_dtype = para_dict["res_dtype"]
    ConvParam.res_dtype = res_dtype
    bias_tensor = para_dict["bias_tensor"]
    bias_flag = (bias_tensor is not None)
    offset_x = para_dict["offset_x"]
    kernel_name = para_dict["kernel_name"]

    check_optim_dict(optim_dict, para_dict, data, weight)
    check_para_dict(para_dict, in_dtype, w_dtype)
    kernel_name = para_dict["kernel_name"]

    stride_h = para_dict["stride_h"]
    stride_w = para_dict["stride_w"]
    dilate_h = para_dict["dilate_h"]
    dilate_w = para_dict["dilate_w"]
    filter_h = para_dict["filter_h"]
    filter_w = para_dict["filter_w"]

    data_shape = shape_to_list(data.shape)
    in_channel_c1 = data_shape[1]
    in_channel_c0 = data_shape[4]
    in_channel_c = in_channel_c1*in_channel_c0

    weight_shape = shape_to_list(weight.shape)
    out_channel_c1 = weight_shape[1]
    out_channel_c0 = weight_shape[2]
    out_channel = out_channel_c1*out_channel_c0

    mad_dtype = para_dict.get("mad_dtype")
    _save_conv_param()

    block_size_k = CUBE_MKN[w_dtype]['mac'][1]

    if optim_dict["c0_optim_flg"]:
        block_size_k = 4
        in_channel_c = 4
    shape_w_nc1hwc0 = (out_channel, (in_channel_c + block_size_k - 1) //
                       block_size_k, filter_h, filter_w, block_size_k)
    shape_fmap_nc1hwc0 = _get_fmap_shape_nc1hwc0()
    shape_fmap_nc1hwc0[4] = _row_major_c0_value(data_shape, optim_dict)
    ConvParam.tiling_query_param = {
        "fmap_shape_nc1hwc0": shape_fmap_nc1hwc0,
        "shape_w_nc1hwc0": shape_w_nc1hwc0,
        "in_dtype": in_dtype,
        "w_dtype": w_dtype,
        "res_dtype": res_dtype,
        "mad_dtype": mad_dtype,
        "padw": pad_w,
        "padh": pad_h,
        "strideh": stride_h,
        "stridew": stride_w,
        "dilateh": dilate_h,
        "dilatew": dilate_w,
        "bias_flag": bias_flag,
        "default_tiling": False}

    tiling = None

    if (is_mini_or_lhisi_version() or is_v200_version()) \
    and in_dtype == 'int8':
        conv_res = _cube_compute(data, weight, mad_dtype, tiling=tiling, \
                                 optim_dict=optim_dict, bias=bias_tensor)
        res = _v200_l0c2ub(conv_res, res_dtype)
    else:
        if bias_flag:
            # for float16 conv, bias add in UB, so bias=0(no bias add in cube)
            TENSOR_MAP["fp16_bias"] = bias_tensor
            conv_res = \
            conv_and_quant_compute(data, weight, mad_dtype, res_dtype, pad_h, \
                pad_w, stride_h, stride_w, \
                dilate_h, dilate_w, filter_h, filter_w, \
                bias=False, no_vector=False, tiling=tiling, \
                optim_dict=optim_dict, kernel_name=kernel_name)
            res = bias_add(conv_res, bias_tensor)
        else:
            conv_res = \
            conv_and_quant_compute(data, weight, mad_dtype, res_dtype, pad_h, \
                pad_w, stride_h, stride_w, \
                dilate_h, dilate_w, filter_h, filter_w, \
                bias=False, \
                no_vector=True, tiling=tiling, \
                optim_dict=optim_dict, kernel_name=kernel_name)
            res = conv_res

    res_remove_pad_shape = list(res.shape)
    res_remove_pad_shape[2] = DIM_MAP["out_img_shape"][2]
    res_remove_pad = remove_pad(res, res_remove_pad_shape)

    return res_remove_pad
