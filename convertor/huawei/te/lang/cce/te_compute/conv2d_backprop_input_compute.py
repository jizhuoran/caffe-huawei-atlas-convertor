"""
conv2d backprop input DSL interface.
"""

#!/usr/bin/python
# -*- coding: UTF-8 -*-
from te import tvm
from te.platform import CUBE_MKN
from te.lang.cce.te_compute import util
from .conv2d_backprop_input_general_compute import DeConvPattern
from .conv2d_backprop_input_opti_compute import DeConvKernelSize1Pattern

NoneType = type(None)

# shape dim
DY_SHAPE_DIM = 5
FILTER_SHAPE_DIM = 4
FILTER_DIM = 4
DX_SHAPE_DIM = 4
STRIDES_DIM = 2
PADDING_DIM = 4
DILATION_DIM = 4

# padH, padW must be in [0,255]
PAD_MIN = 0
PAD_MAX = 255

# dilation must be in [1,255]
DILATION_MIN = 1
DILATION_MAX = 255

# filterH, filterW must be in [1,255]
FILTER_HW_MIN = 1
FILTER_HW_MAX = 255
FILTER_HW_MUL_MIN = 1
FILTER_HW_MUL_MAX = 255

DY_FILLING_HW_MIN = 2
DY_FILLING_HW_MAX = 4096

# fmapH, fmapW must be in [2,4096]
DX_HW_MIN = 2
DX_HW_MAX = 4096

# stride must be in [1,64]
STRIDE_MIN = 1
STRIDE_MAX = 63
STRIDE_MUL_MIN = 1
STRIDE_MUL_MAX = 256

# the bytes length of several dtype
BIT_RATIO_DICT = {"int32": 4, "float32": 4, "float16": 2,
                  "uint8": 1, "int8": 1, "uint4": 0.5, "int4": 0.5}
# same as (2**63-1)
DATA_SIZE_MAX = 9223372036854775807


def check_input_params(filters, # pylint: disable=R0913,R0914,R0915
                       out_backprop, filter_sizes, input_sizes,
                       strides, padding, dilations, res_dtype):
    """
    check the input params of conv2d_backprop_input_compute

    Parameters
    ----------
    filters : weight tensor of fractal shape

    out_backprop : 5D dE/dY tensor

    filter_sizes : shape of weight, [N, C, H, W]

    input_sizes : shape of dE/dX, [N, C, H, W]

    strides : list of strides, [strideh, stridew]

    padding : list of padding, [pad_up, pad_down, pad_left, pad_right]

    dilations : [1, 1, 1, 1] by default

    res_dtype : dE/dX data type, "float16" by default

    Returns
    ----------
    None
    """

    def _check_shape_rule(shape_arg, shape_dim, shape_dtype, shape_name):
        if len(shape_arg) != shape_dim:
            raise RuntimeError("{} must "
                               "be length of {}.".format(shape_name,
                                                         str(shape_dim)))
        for i in shape_arg:
            if not isinstance(i, shape_dtype):
                raise RuntimeError("Element in {} must "
                                   "be of {}.".format(shape_name,
                                                      str(shape_dtype)))

    # check dtype
    def _check_dtype(valid_dtype_dict):
        if filters.dtype not in valid_dtype_dict["filter"]:
            raise RuntimeError("filter dtype must "
                               "in {}.".format(valid_dtype_dict["filter"]))

        if out_backprop.dtype not in valid_dtype_dict["dedy"]:
            raise RuntimeError("out_backprop dtype must "
                               "in {}.".format(valid_dtype_dict["dedy"]))

        if filters.dtype != out_backprop.dtype:
            raise RuntimeError("filter's dtype should "
                               "be equal to out_backprop's dtype.")

        if res_dtype not in valid_dtype_dict["dx"][filters.dtype]:
            raise RuntimeError("dx dtype must "
                               "in {}.".format(valid_dtype_dict["dx"]))

    # check shape
    def _check_shape():
        if len(filters.shape) != FILTER_SHAPE_DIM:
            raise RuntimeError("filter must be in "
                               "the format of [k1, n1, n0, k0].")

        if len(out_backprop.shape) != DY_SHAPE_DIM:
            raise RuntimeError("out_backprop must be in the format of "
                               "[No, Co1, Ho, Wo, Co0].")

        _check_shape_rule(filter_sizes, FILTER_DIM, int, "filter_sizes")

        _check_shape_rule(input_sizes, DX_SHAPE_DIM, int, "input_sizes")

        _check_shape_rule(strides, STRIDES_DIM, int, "strides")

        _check_shape_rule(padding, PADDING_DIM, int, "padding")

        _check_shape_rule(dilations, DILATION_DIM, int, "dilations")

    valid_dtype_dict = {}
    valid_dtype_dict["filter"] = ("float16", "int8")
    valid_dtype_dict["dedy"] = ("float16", "int8")
    valid_dtype_dict["dx"] = {"float16": "float16", "int8": "int32"}

    _check_dtype(valid_dtype_dict)
    _check_shape()

    # begin to fetch params
    if filters.dtype == "int8":
        filter_cout1, _, filter_cin0, filter_cout0 \
            = list(i.value for i in filters.shape)
        filter_cout1 = filter_cout1 / filter_sizes[2] / filter_sizes[3]
    else:
        _, filter_cout1, filter_cout0, filter_cin0 \
            = list(i.value for i in filters.shape)

    dy_batch, dy_c1, dy_h, dy_w, dy_c0 = \
        list(i.value for i in out_backprop.shape)
    filter_cout, filter_cin, filter_h, filter_w = filter_sizes
    dx_batch, dx_c, dx_h, dx_w = input_sizes
    stride_h, stride_w = strides
    pad_up, pad_down, pad_left, pad_right = padding
    dilation_n, dilation_c, dilation_h, dilation_w = dilations
    filter_h_dilation = (filter_h - 1) * dilation_h + 1
    filter_w_dilation = (filter_w - 1) * dilation_w + 1

    _, dedy_k0, _ = CUBE_MKN[out_backprop.dtype]['mac']
    _, w_k0, w_n0 = CUBE_MKN[filters.dtype]['mac']

    filter_cout = (filter_cout + w_k0 - 1) // w_k0 * w_k0

    # special cases
    dy_filling_hw_min, dx_hw_min = DY_FILLING_HW_MIN, DX_HW_MIN
    # limitation by chip:
    # if kernel h,w in [1,11] and fmap h/w after padding equals to filter h/w
    # load3d support h,w is 1
    if (1 <= filter_h <= 11) and (1 <= filter_w <= 11) \
        and (dx_h + pad_up + pad_down == filter_h
             or dx_w + pad_left + pad_right == filter_w):
        dy_filling_hw_min = 1
        dx_hw_min = 1

    # dy
    def _check_dy():
        if dy_c0 != dedy_k0:
            raise RuntimeError("dy_c0 must be {}.".format(dedy_k0))

        if dy_h*stride_h < dy_filling_hw_min or \
        dy_h*stride_h > DY_FILLING_HW_MAX:
            raise RuntimeError("dy_h*stride_h must be in [2, 4096].")

        if filter_h == 1 and filter_w == 1:
            if dy_w*stride_w*stride_h < dy_filling_hw_min or \
            dy_w*stride_w*stride_h > DY_FILLING_HW_MAX:
                raise RuntimeError("dy_w*stride_w*stride_h "
                                   "must be in [2, 4096].")
        else:
            if dy_w*stride_w < dy_filling_hw_min or \
            dy_w*stride_w > DY_FILLING_HW_MAX:
                raise RuntimeError("dy_w*stride_w must be in [2, 4096].")

    # w
    # check filter shape and filter_sizes from topi
    def _check_filter():
        if filter_cout0 != w_k0:
            raise RuntimeError("filter_cout0 must be {}.".format(w_k0))

        if filter_cout1*filter_cout0 != filter_cout:
            raise RuntimeError("filter_cout1*filter_cout0 must "
                               "be equal to filter_cout.")

        if filter_cin0 != w_n0:
            raise RuntimeError("filter_cin0 must be {}.".format(w_n0))

        if dy_c1*dy_c0 != filter_cout:
            raise RuntimeError("dy_c must be equal to filter_cout0.")

        if filter_h < FILTER_HW_MIN or filter_h > FILTER_HW_MAX:
            raise RuntimeError("filter_h must be in [1, 255].")

        if filter_w < FILTER_HW_MIN or filter_w > FILTER_HW_MAX:
            raise RuntimeError("filter_w must be in [1, 255].")

        if filter_h*filter_w < FILTER_HW_MUL_MIN or \
                filter_h*filter_w > FILTER_HW_MUL_MAX:
            raise RuntimeError("filter_h*filter_w must be in [1, 255].")

        def _check_max(x_1, x_2, name_1, name_2):
            if x_1 > x_2:
                raise RuntimeError("{} must <= {}.".format(name_1, name_2))
        _check_max(filter_h_dilation, dx_h + pad_up + pad_down,
                   "filter_h after dilation", "dx_h after pad")
        _check_max(filter_w_dilation, dx_w + pad_left + pad_right,
                   "filtre_w after dilation", "dx_w after pad")

    # dx
    def _check_dx():
        if dx_h < dx_hw_min or dx_h > DX_HW_MAX:
            raise RuntimeError("dx_h must be in [1, 4096].")

        if dx_w < dx_hw_min or dx_w > DX_HW_MAX:
            raise RuntimeError("dx_w must be in [1, 4096].")

        if dx_batch != dy_batch:
            raise RuntimeError("dx_batch must be equal to dy_batch.")

        if dx_c != filter_cin:
            raise RuntimeError("dx_cin must be equal to filter_cin.")

        dx_h_after_pad = dx_h + pad_up + pad_down
        if ((dx_h_after_pad - filter_h_dilation) // stride_h + 1) != dy_h:
            raise RuntimeError("dx_h and dy_h dismatch.")

        dx_w_after_pad = dx_w + pad_left + pad_right
        if ((dx_w_after_pad - filter_w_dilation) // stride_w + 1) != dy_w:
            raise RuntimeError("dx_w and dy_w dismatch.")

    # strides
    def _check_strides():
        if stride_h < STRIDE_MIN or stride_h > STRIDE_MAX:
            raise RuntimeError("stride_h must be in [1, 63].")

        if stride_w < STRIDE_MIN or stride_w > STRIDE_MAX:
            raise RuntimeError("stride_w must be in [1, 63].")

        if stride_h*stride_w < STRIDE_MUL_MIN or \
            stride_h*stride_w > STRIDE_MUL_MAX:
            raise RuntimeError("stride_h*stride_w must be in [1, 256].")

    # padding
    def _check_padding():
        def _check_border(attr_name, attr_value):
            if attr_value < PAD_MIN or attr_value > PAD_MAX:
                raise RuntimeError(
                    "{}'s value must be in [{}, {}].".format(attr_name,
                                                             PAD_MIN,
                                                             PAD_MAX))

        def _check_max(x_1, x_2, name_1, name_2):
            if x_1 > x_2:
                raise RuntimeError("{} must <= {}.".format(name_1, name_2))

        _check_border("pad_up", pad_up)
        _check_border("pad_down", pad_down)
        _check_border("pad_left", pad_left)
        _check_border("pad_right", pad_right)

        _check_max(pad_up, filter_h_dilation,
                   "pad_up", "filter_h_dilation")
        _check_max(pad_down, filter_h_dilation,
                   "pad_down", "filter_h_dilation")
        _check_max(pad_left, filter_w_dilation,
                   "pad_left", "filter_w_dilation")
        _check_max(pad_right, filter_w_dilation,
                   "pad_right", "filter_w_dilation")

    # dilation
    def _check_dilation():
        if dilation_n != 1 or dilation_c != 1:
            raise RuntimeError("Dilations in the batch and depth "
                               "dimensions must be 1")
        if dilation_h < DILATION_MIN or dilation_h > DILATION_MAX:
            raise RuntimeError("dilation_h must "
                               "be in [{},{}]".format(DILATION_MIN,
                                                      DILATION_MAX))
        if dilation_w < DILATION_MIN or dilation_w > DILATION_MAX:
            raise RuntimeError("dilation_w must "
                               "be in [{},{}]".format(DILATION_MIN,
                                                      DILATION_MAX))


    # check L1 exceed buffer
    def _check_l1_buffer():
        bl1_size = filter_h_dilation * filter_w_dilation * 512
        al1_w_value = dy_w * stride_w

        if dx_w > 16:
            al1_h_value = filter_h_dilation + 1
        elif 16 % dx_w == 0:
            al1_h_value = filter_h_dilation + 16 // dx_w - 1
        else:
            al1_h_value = filter_h_dilation + 16 // dx_w + 1

        al1_size = al1_h_value * al1_w_value * 32
        if al1_size + bl1_size > 1024*1024:
            raise RuntimeError("Shape exceeds L1 buffer limitations!")

    # 64 bits limitation check
    def _check_chip_limitation():
        def _align(x_1, x_2):
            return (x_1 + x_2 - 1) // x_2 * x_2

        def _check_64bits_limitation(attr_name, attr_value, dtype=None):
            if dtype is None:
                bit_ratio = BIT_RATIO_DICT.get("float16")
            else:
                bit_ratio = BIT_RATIO_DICT.get(dtype)
            if attr_value*bit_ratio > DATA_SIZE_MAX:
                raise RuntimeError("{} exceed 64 bit limitations!"
                                   .format(attr_name))

        _, dedy_k0, _ = CUBE_MKN[out_backprop.dtype]['mac']
        _, w_k0, w_n0 = CUBE_MKN[filters.dtype]['mac']

        fmap_size = dx_batch * _align(dx_c, dedy_k0) * dx_h * dx_w
        dedy_size = dy_batch * dy_c1 * dy_h * dy_w * dy_c0
        filter_size = _align(filter_cout, w_k0) \
                      * _align(filter_cin, w_n0) * filter_h * filter_w
        _check_64bits_limitation("fmap_size",
                                 fmap_size, dtype=res_dtype)
        _check_64bits_limitation("dedy_size",
                                 dedy_size, dtype=out_backprop.dtype)
        _check_64bits_limitation("filter_size",
                                 filter_size, dtype=filters.dtype)

    _check_dy()
    _check_filter()
    _check_dx()
    _check_strides()
    _check_padding()
    _check_dilation()
    _check_l1_buffer()
    _check_chip_limitation()


@util.check_input_type(tvm.tensor.Tensor, tvm.tensor.Tensor,
                       (list, tuple), (list, tuple), (list, tuple),
                       (list, tuple), (list, tuple),
                       str, (tvm.tensor.Tensor, NoneType))
def conv2d_backprop_input_compute(filters,  # pylint: disable=R0913,R0914
                                  out_backprop, filter_sizes,
                                  input_sizes, strides, padding, dilations,
                                  res_dtype="float16", tensor_bias=None):

    """
    DSL interface of conv2d backprop input

    Parameters
    ----------
    filters : weight tensor of fractal shape

    out_backprop : 5D dE/dY tensor

    filter_sizes : shape of weight, [N, C, H, W]

    input_sizes : shape of dE/dX, [N, C, H, W]

    strides : list of strides, [strideh, stridew]

    padding : list of padding, [pad_up, pad_down, pad_left, pad_right]

    dilations : [1, 1, 1, 1] by default

    res_dtype : dE/dX data type, "float16" by default

    Returns
    ----------
    dx_ddr: dE/dX tensor
    """
    check_input_params(filters, out_backprop, filter_sizes,
                       input_sizes, strides, padding, dilations, res_dtype)

    _, _, filter_h, filter_w = filter_sizes
    dx_batch, dx_c, dx_h, dx_w = input_sizes
    ceil = lambda A, B: (A + B - 1) // B
    _, _, dx_n0 = CUBE_MKN[res_dtype]['mac']
    shape_dx = (dx_batch, ceil(dx_c, dx_n0), dx_h, dx_w, dx_n0)

    if filter_h == 1 and filter_w == 1 and padding == [0, 0, 0, 0]:
        pattc = DeConvKernelSize1Pattern(filter_sizes,
                                         strides=strides,
                                         pad=padding,
                                         output_shape=shape_dx)
    else:
        pattc = DeConvPattern(filter_sizes, strides=strides,
                              pad=padding, output_shape=shape_dx,
                              dilations=dilations)

    dy_col = pattc.generate_a(out_backprop)
    w_col = pattc.generate_b(filters)
    dx_ddr = pattc.generate_c(dy_col, w_col, tensor_bias=tensor_bias)

    return dx_ddr
