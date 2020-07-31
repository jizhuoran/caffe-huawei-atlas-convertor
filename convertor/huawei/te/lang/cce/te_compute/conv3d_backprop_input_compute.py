"""
conv3d backprop input DSL interface.
"""

#!/usr/bin/python
# -*- coding: UTF-8 -*-
from te import tvm
from te.platform import CUBE_MKN
from te.lang.cce.te_compute import util
from .conv3d_backprop_input_general_compute import DeConvPattern


# shape dim
DY_SHAPE_DIM = 5
FILTER_SHAPE_DIM = 4
FILTER_DIM = 4
DX_SHAPE_DIM = 4
STRIDES_DIM = 2
PADDING_DIM = 4

# padH, padW must be in [0,255]
PAD_MIN = 0
PAD_MAX = 255

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

DATA_SIZE_MAX = 9223372036854775807

@util.check_input_type(tvm.tensor.Tensor, tvm.tensor.Tensor, (list, tuple),
                       (list, tuple), (list, tuple), (list, tuple),
                       (list, tuple), str)
def conv3d_backprop_input_compute(filters, # pylint: disable=R0913,R0914
                                  out_backprop, filter_sizes,
                                  input_sizes, strides, padding,
                                  dilations, res_dtype="float16"):

    """
    DSL interface of conv3d backprop input

    Parameters
    ----------
    filters : weight tensor of fractal shape

    out_backprop : 5D dE/dY tensor

    filter_sizes : shape of weight, [D, H, W, N, C]

    input_sizes : shape of dE/dX, [N, D, H, W, C]

    strides : list of strides, [stridebatch,
                                strided, strideh, stridew, stridechannel]

    padding : list of padding, [pad_front, pad_tail,
                                pad_up, pad_down, pad_left, pad_right]

    dilations : [1, 1, 1, 1, 1] by default

    res_dtype : dE/dX data type, "float16" by default

    Returns
    ----------
    dx_ddr: dE/dX tensor
    """
    dx_batch, dx_d, dx_h, dx_w, dx_c = input_sizes
    ceil = lambda A, B: (A + B - 1) // B
    _, _, config_n0 = CUBE_MKN[res_dtype]['mac']
    shape_dx = (dx_batch, dx_d, ceil(dx_c, config_n0), dx_h, dx_w, config_n0)
    pattc = DeConvPattern(filter_sizes, strides=strides,
                          pad=padding, output_shape=shape_dx,
                          dilations=dilations)
    dy_col = pattc.generate_a(out_backprop)
    w_col = pattc.generate_b(filters)
    dx_ddr = pattc.generate_c(dy_col, w_col)
    return dx_ddr
