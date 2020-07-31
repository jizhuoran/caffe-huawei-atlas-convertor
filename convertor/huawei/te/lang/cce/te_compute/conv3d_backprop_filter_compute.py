# -*- coding: UTF-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

conv3d backprop filter DSL interface.

"""
from __future__ import absolute_import
from __future__ import print_function
import functools
from te import tvm
from te.lang.cce.te_compute import util
from topi.cce.util import is_lhisi_version

# fractal size, only support 16 for now
BLOCK_SIZE = 16
# maximum of int64 (2**63 - 1)
DATA_SIZE_LIMIT_INT64 = 9223372036864776807
# maximum of int32 (2**31 - 1)
DATA_SIZE_LIMIT_INT32 = 2147483647


def check_shape_rule(shape, dim, formats, name):
    """
    check shape

    """
    if len(shape) != dim:
        raise RuntimeError("shape format of %s must be %s" % (name, formats))
    for dim_x in shape:
        if (not isinstance(dim_x, int)) or dim_x <= 0:
            raise RuntimeError(
                "The type of axis of {} must be positive int".format(name))


def ceil_div(dividend, divisor):
    """
    do division and round up to an integer

    """
    if divisor == 0:
        raise RuntimeError(" division by zero")
    return (dividend + divisor - 1) // divisor


def check_attr_rule(attr, dim, attr_limit, formats, name):
    """
    check attribute

    """
    attr_min = attr_limit[0]
    attr_max = attr_limit[1]
    if len(attr) != dim:
        raise RuntimeError("shape format of %s must be %s" % (name, formats))
    for attr_x in attr:
        if (not isinstance(attr_x, int)) \
                or attr_x < attr_min or attr_x > attr_max:
            raise RuntimeError("{} must be in range of [{},{}]".format(
                name, attr_min, attr_max))


def check_variable_range(variable, minimum, maximum, name):
    """
    check variable range

    """
    if variable < minimum or variable > maximum:
        raise RuntimeError("{} must be in range of [{},{}]." \
                           .format(name, minimum, maximum))


def check_addressing_rule(shape, byte_count, limit):
    """
    check addressing limit

    """
    # product of all dimension
    product = functools.reduce(lambda x, y: x * y, shape[:])
    if product * byte_count > limit:
        raise RuntimeError("the shape is too large to calculate")


class Conv3dBackpropFilter:
    """
    Conv3dBackpropFilter: compute definition of conv3d_backprop_filter

    Functions
    ----------
    __init__ : initialization

    deconv_dw_input_check_1: parameters check part 1

    deconv_dw_input_check_2: parameters check part 2

    deconv_dw_compute : compute process

    grads_2_matrix : compute definition of loading grads to cbuf

    grads_2_fractal : compute definition of load_3d

    fmap_2_col_matrix : compute definition of set_fmatrix

    fmap_2_col_fractal : compute definition of load_3d

    mad : compute definition of mmad

    """
    def __init__(self,
                 input_x,
                 out_backprop,
                 filter_sizes,
                 strides,
                 padding,
                 dilations,
                 res_dtype="float32"):
        """
        initialization

        Parameters:
        ----------
        input_x : the featuremap data, tvm.placeholder, 6hd shape
        [N, D, C1, H, W, C0]

        out_backprop : the grads data, tvm.placeholder, 6hd shape
        [N, DO, GRADS_C1, HO, WO, GRADS_C0]

        filter_sizes : 4-D shape, specifies the filter sizes
        [GRADS_C, DK, C, DH, DW]

        strides : 2-D shape in height and width dimension
        [STRIDE_D, STRIDE_H, STRIDE_W]

        padding : 4-D shape in up/down/left/right dimension
        [PAD_D, PAD_D, PAD_H, PAD_H, PAD_W, PAD_W]

        dilations : 4-D shape in batch/channel/height/width dimension

        res_dtype : the output data type

        Returns
        -------
        None
        """

        self.shapelist = {}

        self.fmap = input_x
        self.grads = out_backprop

        self.weight_shape = list(filter_sizes)

        self.fmap_dtype = input_x.dtype
        self.grads_dtype = out_backprop.dtype
        self.res_dtype = res_dtype

        self.pad = list(padding)
        self.stride = list(strides)
        self.dilation = list(dilations)

        self.optag = "conv3d_backprop_filter"

        # 6hd shape
        # [N, DO, GRADS_C1, HO, WO, GRADS_C0]
        self.shape_grads_6hd = list(x.value for x in self.grads.shape)
        # [N, D, C1, H, W, C0]
        self.shape_x_6hd = list(x.value for x in self.fmap.shape)

        self.shapelist['grads_6hd'] = self.shape_grads_6hd
        self.shapelist['fmap_6hd'] = self.shape_x_6hd

        # flag of special case
        self.flag_all_one_case = False
        self.dw_ddr = []
        self.res_tensor = self.dw_ddr  # return tensor of this file to topi
        # special supporting for a unique case, there are 2 conditions:
        # (1) height & weight of x/output_backprop/filter are all 1
        # (2) strides is [1,1]
        if self.stride[1:] == [1, 1] and self.shape_x_6hd[3:5] == [1, 1] \
            and self.shape_grads_6hd[3:5] == [1, 1] \
            and self.weight_shape[3:5] == [1, 1]:
            self.flag_all_one_case = True

        self.flag_load3d_special_case = False
        # special supporting for another unique case, there are 2 conditions:
        # (1) weight_height, weight_width in range[1,11]
        # (2) fmap_height/width + pad_top/left + pad_bottom/right is
        # weight_height/width
        _, fmap_depth, _, fmap_height, fmap_width, _ = self.shape_x_6hd
        _, kernel_depth, _, kernel_height, kernel_width = self.weight_shape
        pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right = self.pad

        if (1 <= kernel_height <= 11) and (1 <= kernel_width <= 11) \
            and (fmap_height + pad_top + pad_bottom == kernel_height
                 or fmap_width + pad_left + pad_right == kernel_width or \
                     fmap_depth + pad_front+pad_back == kernel_depth):
            self.flag_load3d_special_case = True

    def deconv_dw_input_check_1(self):
        """
        do input parameters check part1

        """
        # check of data type
        if self.fmap_dtype != "float16":
            raise RuntimeError("data type of x only supports float16")
        if self.grads_dtype != "float16":
            raise RuntimeError("data type of out_backprop "
                               "only supports float16")
        if is_lhisi_version() and self.res_dtype != "float16":
            raise RuntimeError("data type of result "
                               "only supports float16 for lhisi")
        if not is_lhisi_version() and self.res_dtype != "float32":
            raise RuntimeError("data type of result only supports float32")

        # check shape
        # each element must be positive int
        check_shape_rule(self.shape_x_6hd, 6, "NDC1HWC0", "x")
        check_shape_rule(self.shape_grads_6hd, 6, "NDC1HWC0", "out_backprop")
        check_shape_rule(self.weight_shape, 5, "NDCHW", "filter_sizes")

        _, grads_depth, _, grads_height, grads_width, _ \
            = self.shape_grads_6hd
        _, fmap_depth, _, fmap_height, fmap_width, _ = self.shape_x_6hd
        _, kernel_depth, _, kernel_height, kernel_width = self.weight_shape

        if ceil_div(self.weight_shape[0], 16) != self.shape_grads_6hd[2]:
            raise RuntimeError("filter number must be same with out_backprop")

        if ceil_div(self.weight_shape[2], 16) != self.shape_x_6hd[2]:
            raise RuntimeError("filter channel must be same with x")

        # individual range check
        if not self.flag_all_one_case:
            if not self.flag_load3d_special_case:
                check_variable_range(grads_depth, 2, 4096,
                                     "depth of out_backprop")
                check_variable_range(grads_height, 2, 4096,
                                     "height of out_backprop")
                check_variable_range(grads_width, 2, 4096,
                                     "width of out_backprop")
            else:
                check_variable_range(grads_depth, 1, 4096,
                                     "depth of out_backprop")
                check_variable_range(grads_height, 1, 4096,
                                     "height of out_backprop")
                check_variable_range(grads_width, 1, 4096,
                                     "width of out_backprop")
        check_variable_range(fmap_depth, 1, 4096, "depth of x")
        check_variable_range(fmap_height, 1, 4096, "height of x")
        check_variable_range(fmap_width, 1, 4096, "width of x")
        check_variable_range(kernel_depth, 1, 256, "depth of filter")
        check_variable_range(kernel_height, 1, 256, "height of filter")
        check_variable_range(kernel_width, 1, 256, "width of filter")

        check_attr_rule(self.stride, 3, [1, 63], "[strideD, strideH, strideW]",
                        "stride")
        check_attr_rule(self.pad, 6, [0, 266],
                        "[front,back,up,down,left,right]", "pad")
        return True

    def deconv_dw_input_check_2(self):
        """
        do input parameters check part2

        """
        stride_depth, stride_height, stride_width = self.stride
        pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right \
            = self.pad
        _, grads_depth, grads_channel_1, grads_height, grads_width, grads_c0 \
            = self.shape_grads_6hd
        _, fmap_depth, fmap_channel_1, fmap_height, fmap_width, fmap_c0 \
            = self.shape_x_6hd
        _, kernel_depth, _, kernel_height, kernel_width = self.weight_shape
        dilationn, dilationd, dilationc, dilationh, dilationw = self.dilation

        def _check_dilation():
            dilation_min = 1
            dilation_max = 266
            if dilationn * dilationc != 1:
                raise RuntimeError("Dilations in the batch and channel "
                                   "dimensions must be 1")
            dilation_dhw = [dilationd, dilationh, dilationw]
            for item in dilation_dhw:
                if item < dilation_min or item > dilation_max:
                    raise RuntimeError("{} must be in [{},{}]".format(
                        item, dilation_min, dilation_max))

        _check_dilation()

        if self.shape_x_6hd[5] != BLOCK_SIZE:
            raise RuntimeError("axis C0 of x must be 16")

        if self.shape_grads_6hd[5] != BLOCK_SIZE:
            raise RuntimeError("axis C0 of out_backprop must be 16")
        # batch_size should be same
        if self.shape_x_6hd[0] != self.shape_grads_6hd[0]:
            raise RuntimeError("batch size of x and out_backprop must be same")

        # coupling range check
        dilation_kernel_depth = (kernel_depth - 1) * dilationd + 1
        computed_grads_depth = (fmap_depth - dilation_kernel_depth + \
                                 pad_front + pad_back)//stride_depth + 1
        if computed_grads_depth != grads_depth:
            raise RuntimeError("mismatch between depth of out_backprop and x")

        dilation_kernel_height = (kernel_height - 1) * dilationh + 1
        computed_grads_height = (fmap_height - dilation_kernel_height + \
                                 pad_top + pad_bottom)//stride_height + 1
        if computed_grads_height != grads_height:
            raise RuntimeError("mismatch between height of out_backprop and x")

        dilation_kernel_width = (kernel_width - 1) * dilationw + 1
        computed_grads_width = (fmap_width - dilation_kernel_width + \
                                 pad_left + pad_right)//stride_width + 1
        if computed_grads_width != grads_width:
            raise RuntimeError("mismatch between width of out_backprop and x")

        fmap_depth_after_pad = fmap_depth + pad_front + pad_back
        fmap_height_after_pad = fmap_height + pad_top + pad_bottom
        fmap_width_after_pad = fmap_width + pad_left + pad_right
        if dilation_kernel_height > fmap_height_after_pad \
            or dilation_kernel_width > fmap_width_after_pad \
            or dilation_kernel_depth > fmap_depth_after_pad:
            raise RuntimeError("depth/height/width of filter "
                               "cannot exceed that of x")

        def _check_pad():
            if pad_front >= dilation_kernel_height \
                or pad_back >= dilation_kernel_height:
                raise RuntimeError("pad in front/back should "
                                   "less than depth of filter")

            if pad_top >= dilation_kernel_height \
                or pad_bottom >= dilation_kernel_height:
                raise RuntimeError("pad in up/down should "
                                   "less than height of filter")

            if pad_left >= dilation_kernel_width \
                or pad_right >= dilation_kernel_width:
                raise RuntimeError("pad in left/right should "
                                   "less than width of filter")

        _check_pad()

        # int64 addressing limit of tvm
        kernel_fractal = (kernel_depth * fmap_channel_1 * kernel_height *
                          kernel_width, grads_channel_1 * grads_c0, fmap_c0)

        check_addressing_rule(self.shape_grads_6hd, 2, DATA_SIZE_LIMIT_INT64)
        check_addressing_rule(self.shape_x_6hd, 2, DATA_SIZE_LIMIT_INT64)
        # because of atomic write, dw_ubuf does not tiled, cannot exceed int32
        # limit (change to int64 limit after tvm v0.6)
        check_addressing_rule(kernel_fractal, 4, DATA_SIZE_LIMIT_INT32)
        return True

    def deconv_dw_access(self):
        """
        complete compute generation, including input check,
        compute definition and result record

        """

        self.deconv_dw_input_check_1()
        self.deconv_dw_input_check_2()
        self.deconv_dw_compute()
        self.res_tensor = self.dw_ddr  # return tensor of this file to topi

    def deconv_dw_compute(self):
        """
        complege compute definition

        """
        def _grads_2_fractal(grads_shape_fractal, grads_2_matrix):
            """
            compute definiton of loading grads_matrix to L0A

            Parameters:
            ----------
            grads_shape_fractal : shape of result tensor in L0A

            grads_2_matrix : input tensor in L1

            Returns
            -------
            None
            """
            def __grads_2_fractal_compute(indices, grads_2_matrix):
                """
                do coordinate calculation

                """

                batch_indices, grads_c1_indices, hw_mad_1_indices, \
                grads_c0_indices, hw_mad_0_indices = indices

                batch_size_index = batch_indices
                grads_channel_1_index = grads_c1_indices
                grads_hw_index = hw_mad_1_indices * BLOCK_SIZE \
                                 + hw_mad_0_indices
                grads_c0_index = grads_c0_indices

                return grads_2_matrix(batch_size_index, grads_channel_1_index,
                                      grads_hw_index, grads_c0_index)

            return tvm.compute(grads_shape_fractal,
                               lambda *indices: \
                                   __grads_2_fractal_compute(indices,
                                                             grads_2_matrix),
                               name='grads_2_fractal',
                               tag='grads_2_fractal')

        def _fmap_2_fractal_load2d(fmap_shape_fractal, fmap_2_matrix):
            """
            compute definiton of loading fmap_matrix to L0B

            Parameters:
            ----------
            fmap_shape_fractal : shape of result tensor in L0B

            fmap_2_matrix : input tensor in L1

            Returns
            -------
            None
            """
            def __fmap_2_fractal_load2d_compute(indices, fmap_2_matrix):
                """
                do coordinate calculation
                """
                batch_indices, hw_mad_1_indices, fmap_c1_indices, \
                fmap_c0_indices, hw_mad_0_indices = indices

                batch_size_index = batch_indices
                fmap_channel_1_index = fmap_c1_indices
                fmap_hw_index = hw_mad_1_indices * BLOCK_SIZE \
                                + hw_mad_0_indices
                fmap_c0_index = fmap_c0_indices

                return fmap_2_matrix(batch_size_index, fmap_channel_1_index,
                                     fmap_hw_index, fmap_c0_index)

            return tvm.compute(
                fmap_shape_fractal,
                lambda *indices: __fmap_2_fractal_load2d_compute(
                    indices, fmap_2_matrix),
                name='famp_2_fractal',
                tag='famp_2_fractal')

        fmap_dtype = self.fmap_dtype

        batch_size, grads_depth, grads_channel_1, grads_height, grads_width, \
        grads_c0 = self.shape_grads_6hd
        _, fmap_depth, fmap_channel_1, fmap_height, fmap_width, fmap_c0 \
            = self.shape_x_6hd
        _, kernel_depth, _, kernel_height, kernel_width = self.weight_shape
        # align to 16
        hw_mad = (grads_height*grads_width + BLOCK_SIZE - 1) \
                 // BLOCK_SIZE*BLOCK_SIZE

        # move grads to L1
        grads_shape_matrix = (batch_size * grads_depth, grads_channel_1,
                              grads_height * grads_width, grads_c0)
        self.shapelist['grads_matrix'] = grads_shape_matrix

        grads_matrix = self.grads_2_matrix(grads_shape_matrix, self.grads)

        # move grads_matrix to L0A and do transpose
        grads_shape_fractal = (batch_size * grads_depth, grads_channel_1,
                               hw_mad // BLOCK_SIZE, grads_c0, BLOCK_SIZE)

        self.shapelist['grads_fractal'] = grads_shape_fractal
        grads_fractal = _grads_2_fractal(grads_shape_fractal, \
                                         grads_matrix)

        stride_depth, _, _ = self.stride
        pad_front, _, _, _, _, _ = self.pad

        def tensor_to_al1():
            fmap_al1_shape = (batch_size * grads_depth,
                              kernel_depth * fmap_channel_1, fmap_height,
                              fmap_width, fmap_c0)
            fmap_l1 = tvm.compute(
                fmap_al1_shape,
                lambda i, j, k, l, m: tvm.select(
                    tvm.all((i % grads_depth) * stride_depth + j //
                            fmap_channel_1 >= pad_front,
                            (i % grads_depth) * stride_depth + j //
                            fmap_channel_1 < pad_front + fmap_depth),
                    self.fmap(
                        i // grads_depth, i % grads_depth * stride_depth + j //
                        fmap_channel_1 - pad_front, j % fmap_channel_1, k, l, m
                    )),
                name='fmap_l1',
                tag='fmap_l1')
            return fmap_l1

        if not self.flag_all_one_case:
            fmap_l1 = tensor_to_al1()

        if not self.flag_all_one_case:
            # shape of fmap_original_matrix, corresponding to set_fmatrix
            fmap_shape_original_matrix = (batch_size * grads_depth,
                                          grads_height * grads_width,
                                          kernel_depth * fmap_channel_1,
                                          kernel_height, kernel_width, fmap_c0)

            self.shapelist['fmap_original_matrix'] = fmap_shape_original_matrix

            fmap_matrix = self.fmap_2_matrix(fmap_shape_original_matrix,
                                             fmap_l1, fmap_dtype)
            # move fmap to L0B
            fmap_shape_fmap_matrix = (batch_size * grads_depth,
                                      hw_mad // BLOCK_SIZE, kernel_depth *
                                      fmap_channel_1 * kernel_height *
                                      kernel_width, fmap_c0, BLOCK_SIZE)
            self.shapelist['fmap_fmap_matrix'] = fmap_shape_fmap_matrix

            fmap_fractal = self.fmap_2_fractal(fmap_shape_fmap_matrix,
                                               fmap_matrix, fmap_dtype)

        # else: all_one_case, using load_2d instead of load_3d
        else:
            # shape of fmap_matrix
            fmap_shape_matrix = (batch_size * grads_depth,
                                 kernel_depth * fmap_channel_1,
                                 fmap_height * fmap_width, fmap_c0)

            self.shapelist['fmap_matrix'] = fmap_shape_matrix

            fmap_matrix = self.fmap_2_matrix_load2d(fmap_shape_matrix,
                                                    self.fmap)
            # move fmap to L0B
            fmap_shape_fractal = (batch_size * grads_depth, hw_mad //
                                  BLOCK_SIZE, kernel_depth * fmap_channel_1 *
                                  kernel_height * kernel_width, fmap_c0,
                                  BLOCK_SIZE)
            self.shapelist['fmap_matrix'] = fmap_shape_fractal

            fmap_fractal = _fmap_2_fractal_load2d(fmap_shape_fractal,
                                                  fmap_matrix)
        # shape of result dw [n1,m,n0]
        dw_shape = (kernel_depth * fmap_channel_1 * kernel_height *
                    kernel_width, grads_channel_1 * grads_c0, fmap_c0)
        self.shapelist['dw'] = dw_shape

        # do mmad
        dw_cc = self.mad(dw_shape, grads_fractal, fmap_fractal)

        def _lambda_func_dw_cc(*params):
            return dw_cc(*params)

        # move dw_cc to UB
        dw_ubuf = tvm.compute(dw_shape,
                              _lambda_func_dw_cc,
                              name='dw_ubuf',
                              tag="dw_ubuf")

        def _lambda_func_dw_ubuf(*params):
            return dw_ubuf(*params)

        # move to ddr
        self.dw_ddr = tvm.compute(dw_shape,
                                  _lambda_func_dw_ubuf,
                                  name='dw_ddr',
                                  tag=self.optag + "dw_ddr")

        return 1

    def grads_2_matrix(self, grads_shape_matrix, grads):
        """
        compute definiton of loading grads to L1

        Parameters:
        ----------
        grads_shape_matrix : shape of result tensor in L1

        grads : input tensor in ddr

        Returns
        -------
        None
        """
        def __grads_2_matrix_compute(indices, grads):
            """
            do coordinate calculation
            """
            grads_depth = self.shapelist['grads_6hd'][1]
            grads_width = self.shapelist['grads_6hd'][4]

            batch_indices, grads_c1_indices, hw_mad_indices, grads_c0_indices \
            = indices

            # calculate index of grads according to indice of grads_matrix
            batch_size_index = batch_indices // grads_depth
            grads_depth_index = batch_indices % grads_depth
            grads_channel_1_index = grads_c1_indices
            grads_height_index = (hw_mad_indices // grads_width)
            grads_width_index = (hw_mad_indices % grads_width)
            grads_c0_index = grads_c0_indices

            return grads(batch_size_index, grads_depth_index,
                         grads_channel_1_index, grads_height_index,
                         grads_width_index, grads_c0_index)

        return tvm.compute(
            grads_shape_matrix,
            lambda *indices: __grads_2_matrix_compute(indices, grads),
            name='grads_2_matrix',
            tag='grads_2_matrix')

    def fmap_2_matrix(self, fmap_shape_original_matrix, fmap, fmap_dtype):
        """
        compute definiton of set_fmatrix

        Parameters:
        ----------
        fmap_shape_original_matrix : shape of result tensor in L1
        in shape (batch_size*grads_depth,
                  grads_height*grads_width,
                  kernel_depth*fmap_channel_1,
                  kernel_height,
                  kernel_width,
                  fmap_c0)

        fmap : input tensor in L1

        fmap_dtype : data type of fmap
        in shape (batch_size, fmap_depth, fmap_channel_1, fmap_height, fmap_width, C0)

        Returns
        -------
        None
        """
        def __fmap_2_matrix_compute(indices,
                                    fmap,
                                    kernel_width,
                                    pad_left=0,
                                    pad_right=0,
                                    pad_top=0,
                                    strideh=1,
                                    stridew=1,
                                    dilationh=1,
                                    dilationw=1):
            """
            do coordinate calculation

            """

            _, _, _, fmap_height, fmap_width, _ = self.shapelist['fmap_6hd']

            batch_indices, \
            hw_fuse_indices, \
            fmap_c1_indices, \
            kernel_height_indices, \
            kernel_width_indices, \
            fmap_c0_indices = indices

            dilation_kernel_width = kernel_width \
                                    + (kernel_width - 1) * (dilationw - 1)
            fmap_width_after_pad = fmap_width + pad_left + pad_right
            width_out = (fmap_width_after_pad - dilation_kernel_width) \
                        // stridew + 1

            n_index = batch_indices
            c1_index = fmap_c1_indices
            h_index = (hw_fuse_indices // width_out) * strideh \
                      + kernel_height_indices * dilationh
            w_index = (hw_fuse_indices % width_out) * stridew \
                      + kernel_width_indices * dilationw
            c0_index = fmap_c0_indices
            # if index belongs to padding and 16 align, select 0
            return tvm.select(tvm.any(h_index < pad_top,
                                      h_index > fmap_height + pad_top - 1,
                                      w_index < pad_left,
                                      w_index > fmap_width + pad_left - 1),
                              tvm.const(0.0, fmap_dtype),
                              fmap(n_index, c1_index, h_index-pad_top, \
                                   w_index-pad_left, c0_index))

        _, _, pad_top, _, pad_left, pad_right = self.pad
        _, strideh, stridew = self.stride
        _, _, _, dilationh, dilationw = self.dilation

        kernel_width = fmap_shape_original_matrix[4]
        return tvm.compute(
            fmap_shape_original_matrix,
            lambda *indices: __fmap_2_matrix_compute(indices,
                                                     fmap,
                                                     kernel_width,
                                                     pad_left=pad_left,
                                                     pad_right=pad_right,
                                                     pad_top=pad_top,
                                                     strideh=strideh,
                                                     stridew=stridew,
                                                     dilationh=dilationh,
                                                     dilationw=dilationw),
            name='fmap_2_col_matrix',
            tag='fmap_2_col_matrix',
            attrs={
                'pad': self.pad,
                'stride': self.stride,
                'dilation': self.dilation,
                'kernel_size': self.weight_shape,
                'load2d_flag': False
            })

    def fmap_2_matrix_load2d(self, fmap_shape_matrix, fmap):
        """
        compute definiton of loading fmap to L1

        Parameters:
        ----------
        fmap_shape_matrix : shape of result tensor in L1

        fmap : input tensor in ddr

        Returns
        -------
        None
        """
        def __fmap_2_matrix_load2d_compute(indices, fmap):
            """
            do coordinate calculation

            """
            fmap_width = self.shapelist['fmap_6hd'][4]
            fmap_channel_1 = self.shapelist['fmap_6hd'][2]
            grads_depth = self.shapelist['grads_6hd'][1]
            stride_depth = self.stride[0]
            batch_indices, fmap_c1_indices, hw_mad_indices, fmap_c0_indices \
            = indices
            batch_size_index = batch_indices // grads_depth
            fmap_depth_index = batch_indices % grads_depth * stride_depth + \
                               fmap_c1_indices // fmap_channel_1
            fmap_channel_1_index = fmap_c1_indices % fmap_channel_1
            fmap_height_index = (hw_mad_indices // fmap_width)
            fmap_width_index = (hw_mad_indices % fmap_width)
            fmap_c0_index = fmap_c0_indices
            return fmap(batch_size_index, fmap_depth_index,
                        fmap_channel_1_index, fmap_height_index,
                        fmap_width_index, fmap_c0_index)

        return tvm.compute(
            fmap_shape_matrix,
            lambda *indices: __fmap_2_matrix_load2d_compute(indices, fmap),
            name='fmap_2_matrix',
            tag='fmap_2_matrix',
            attrs={
                'pad': self.pad,
                'stride': self.stride,
                'dilation': self.dilation,
                'kernel_size': self.weight_shape,
                "load2d_flag": True
            })

    def fmap_2_fractal(self, fmap_shape_fmap_matrix, fmap_2_col_matrix,
                       fmap_dtype):
        """
        compute definiton of loading fmap to L0B

        Parameters:
        ----------
        fmap_shape_fmap_matrix : shape of result tensor in L0B
        in shape (batch_size*grads_depth,
                  hw_mad//block_size_K,
                  kernel_depth*fmap_channel_1*kernel_height*kernel_width,
                  fmap_c0,
                  block_size_K)

        fmap_2_col_matrix : input tensor in L1
        in shape (batch_size*grads_depth,
                  grads_height*grads_width,
                  kernel_depth*fmap_channel_1,
                  kernel_height,
                  kernel_width,
                  fmap_c0)

        fmap_dtype : data type of fmap_2_col_matrix


        Returns
        -------
        None
        """
        def __fmap_2_fractal_compute(indices, fmap_2_col_matrix):
            """
            do coordinate calculation

            """

            _, hw_fuse, _, kernel_height, kernel_width, _ \
                = self.shapelist['fmap_original_matrix']

            n_vm_index, hw_mad_1_indices, fkk_indices, \
            fmap_c0_indices, hw_mad_0_indices \
                = indices

            hw_vm_index = hw_mad_1_indices * BLOCK_SIZE + hw_mad_0_indices
            c1_vm_index = ((
                (fkk_indices * BLOCK_SIZE + fmap_c0_indices) // BLOCK_SIZE) //
                           kernel_width) // kernel_height
            kh_vm_index = ((
                (fkk_indices * BLOCK_SIZE + fmap_c0_indices) // BLOCK_SIZE) //
                           kernel_width) % kernel_height
            kw_vm_index = ((fkk_indices * BLOCK_SIZE + fmap_c0_indices) //
                           BLOCK_SIZE) % kernel_width
            c0_vm_index = \
                (fkk_indices*BLOCK_SIZE + fmap_c0_indices) % BLOCK_SIZE

            # select padding and 16 align
            return tvm.select(
                tvm.any(hw_vm_index < 0, hw_vm_index > hw_fuse - 1),
                tvm.const(0.0, fmap_dtype),
                fmap_2_col_matrix(n_vm_index, hw_vm_index, c1_vm_index,
                                  kh_vm_index, kw_vm_index, c0_vm_index))

        return tvm.compute(fmap_shape_fmap_matrix,
                           lambda *indices: __fmap_2_fractal_compute(
                               indices, fmap_2_col_matrix),
                           name='fmap_2_col_fractal',
                           tag='fmap_2_col_fractal')

    def mad(self, mad_shape, grads, fmap):
        """
        calculate mad result tensor
        Parameters
        ----------
        mad_shape : result shape
        (kernel_depth*fmap_channel_1*kernel_height*kernel_width, grads_channel, fmap_c0)

        grads : tensor in L0A
        grads_shape_fractal = (batch_size*grads_depth,
                               grads_channel_1,
                               hw_mad//block_size_K,
                               grads_c0,
                               block_size_K)

        fmap : tensor in L0B
        fmap_shape_fmap_matrix = (batch_size*grads_depth,
                                  hw_mad//block_size_K,
                                  kernel_depth*fmap_channel_1*kernel_height*kernel_width,
                                  fmap_c0,
                                  block_size_K)

        Returns
        -------
        None
        """

        batch_size, grads_depth, _, grads_height, grads_width, _ \
            = self.shapelist['grads_6hd']
        hw_fuse = grads_height * grads_width

        batch_axis = tvm.reduce_axis((0, batch_size * grads_depth),
                                     name='axis_b')
        k_axis = tvm.reduce_axis((0, hw_fuse), name='axis_k')

        k_1 = k_axis.var // 16
        k_0 = k_axis.var % 16

        mode = "f162f32"
        if self.res_dtype == "float16":
            mode = "f162f16"

        c_col = tvm.compute(
            mad_shape,
            lambda fkk, grads_c, fmap_c0: tvm.sum(
                (grads[batch_axis, grads_c // 16, k_1, grads_c % 16, k_0] *
                 fmap[batch_axis, k_1, fkk, fmap_c0, k_0]).astype(self.
                                                                  res_dtype),
                axis=[batch_axis, k_axis]),
            name='dw',
            tag="dw",
            attrs={'mode': mode})
        return c_col


@util.check_input_type(tvm.tensor.Tensor, tvm.tensor.Tensor, (list, tuple),
                       (list, tuple), (list, tuple), (list, tuple), str)
def conv3d_backprop_filter_compute(input_x,
                                   out_backprop,
                                   filter_sizes,
                                   strides=None,
                                   padding=None,
                                   dilations=None,
                                   res_dtype="float32"):
    """
    the DSL interface of conv3d backprop filter compute

    Parameters:
    ----------
    x : the featuremap data, tvm.placeholder, 6hd shape

    out_backprop : the grads data, tvm.placeholder, 6hd shape

    filter_sizes : 4-D shape, specifies the filter sizes

    strides : 2-D shape, specifies in height and width dimension

    padding : 4-D shape, specifies in up/down/left/right dimension

    dilations : 4-D shape, specifies in batch/channel/height/width dimension

    res_dtype : the output data type

    Returns
    -------
    result tensor of conv3d_backprop_filter compute

    """
    if not strides:
        strides = [1, 1, 1]
    if not padding:
        padding = [0, 0, 0, 0, 0, 0]
    if not dilations:
        dilations = [1, 1, 1, 1, 1]
    deconv_dw_object = Conv3dBackpropFilter(input_x,
                                            out_backprop,
                                            filter_sizes,
                                            strides=strides,
                                            padding=padding,
                                            dilations=dilations,
                                            res_dtype=res_dtype)
    deconv_dw_object.deconv_dw_access()

    return deconv_dw_object.res_tensor
