#!/usr/bin/env python
# -*- coding:utf-8 -*-
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

strided_slice_grad
"""

from impl import pad_d
from topi.cce import util
from te import platform as tbe_platform
from te import tik
from impl.strided_slice_d import _init_parameter
# General limitation of the reduce size for input shape: 2**31
SHAPE_SIZE_LIMIT = 2147483648
BLOCK_SIZE = 32


# pylint: disable=invalid-name, too-many-instance-attributes
# pylint: disable=too-many-arguments, useless-object-inheritance
# pylint: disable=too-many-locals, too-many-statements
# pylint: disable=attribute-defined-outside-init, unused-argument
# pylint: disable=attribute-defined-outside-init, chained-comparison
class StridedSliceGradLastDimCompute(object):
    """
    the compute for stridedslicegrad in last dim situation
    """
    def __init__(self, shape, begin, size, dtype, kernel_name):
        self.dim_product = 1
        self.input_dim_last = 1
        self.output_dim_last = 1
        self.begin_last = 1
        self.dtype = dtype
        self.kernel_name = kernel_name
        self.ele_size = tbe_platform.cce_intrin.get_bit_len(self.dtype) // 8
        # align size for product dim, to make sure out data is 32B align
        self.product_dim_align_size = BLOCK_SIZE // self.ele_size

        # check only last dim to be sliced
        for i, (shape_i, begin_i, size_i) in \
                enumerate(zip(reversed(shape),
                              reversed(begin), reversed(size))):
            if i != 0:
                if shape_i != size_i:
                    self.check_result = False
                    return

                self.dim_product *= shape_i
            else:
                if begin_i < 0:
                    begin_i += shape_i
                self.input_dim_last = shape_i
                self.begin_last = begin_i
                self.output_dim_last = size_i

        # for moving data continuously, only small last dim is allowed
        # last dim data size <= 340B
        if self.input_dim_last * self.ele_size > 340:
            self.check_result = False
            return

        # for dividing cores easily, only big product dim is allowed
        # product dim >= aicore_num * 32 // ele_size
        aicore_num = tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.CORE_NUM)
        if self.dim_product < self.product_dim_align_size * aicore_num:
            self.check_result = False
            return

        self.check_result = True

    def check(self):
        """
        return check_result
        """
        return self.check_result

    def _get_block_tiling(self, product, core, block_idx):
        task_size = self.product_dim_align_size
        if product % task_size == 0:
            tasks = product // task_size
        else:
            tasks = product // task_size + 1

        begin = self.tik_instance.Scalar(dtype="int64")
        size = self.tik_instance.Scalar(dtype="int64")
        if tasks % core == 0:
            begin.set_as(block_idx * (tasks // core) * task_size)
            size.set_as((tasks // core) * task_size)
        else:
            pack1 = tasks // core + 1
            pack2 = tasks // core
            with self.tik_instance.if_scope(block_idx >= tasks % core):
                begin.set_as(pack1 * block_idx * task_size - (block_idx - tasks % core) * task_size)
                size.set_as(pack2 * task_size)
            with self.tik_instance.else_scope():
                begin.set_as(pack1 * block_idx * task_size)
                size.set_as(pack1 * task_size)

        with self.tik_instance.if_scope(block_idx == (core - 1)):
            size.set_as(product - begin)
        return begin, size

    def strided_slice_grad(self):
        """
        schedule for strided_slice_grad
        """
        if not self.check_result:
            raise RuntimeError("conditions of SliceLastDimCompute are not fulfilled")

        tik_instance = tik.Tik()
        self.tik_instance = tik_instance
        aicore_num = tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.CORE_NUM)
        ub_size = tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.UB_SIZE)

        pad_value = tik_instance.Scalar(dtype=self.dtype, init_value=0)
        x = tik_instance.Tensor(self.dtype,
                                (self.dim_product, self.input_dim_last),
                                name="x", scope=tik.scope_gm)
        y = tik_instance.Tensor(self.dtype,
                                (self.dim_product, self.output_dim_last),
                                name="y", scope=tik.scope_gm)

        with tik_instance.for_range(0, aicore_num,
                                    block_num=aicore_num) as block_idx:
            dim_product_begin, dim_product_size = \
                self._get_block_tiling(self.dim_product, aicore_num, block_idx)
            max_dim_product = ub_size // self.ele_size \
                              // (self.input_dim_last + self.output_dim_last) \
                              // self.product_dim_align_size * self.product_dim_align_size
            loops = tik_instance.Scalar(dtype="int64")
            loops.set_as(dim_product_size // max_dim_product)
            with tik_instance.if_scope(dim_product_size % max_dim_product == 0):
                loops.set_as(loops - 1)

            with tik_instance.for_range(0, loops) as i:
                dim_product_begin_in_loop = i * max_dim_product
                dim_product_size_in_loop = max_dim_product

                x_ub = tik_instance.Tensor(self.dtype,
                                           (max_dim_product,
                                            self.input_dim_last), \
                                           name="x_ub", scope=tik.scope_ubuf)
                y_ub = tik_instance.Tensor(self.dtype,
                                           (max_dim_product,
                                            self.output_dim_last), \
                                           name="y_ub", scope=tik.scope_ubuf)

                output_size_in_loop = dim_product_size_in_loop \
                                      * self.output_dim_last * self.ele_size
                burst_length_out = output_size_in_loop // BLOCK_SIZE
                tik_instance.data_move(y_ub,
                                       y[(dim_product_begin +
                                          dim_product_begin_in_loop)
                                         * self.output_dim_last],
                                       0, 1, burst_length_out, 0, 0)

                with tik_instance.for_range(0, dim_product_size_in_loop) as j:
                    idx_x = j * self.input_dim_last
                    idx_y = j * self.output_dim_last
                    for k in range(self.input_dim_last):
                        max_num = self.begin_last + self.output_dim_last
                        if (k >= self.begin_last) and (k < max_num):
                            x_ub[idx_x + k] = y_ub[idx_y + k - self.begin_last]
                        else:
                            x_ub[idx_x + k] = pad_value

                input_size_in_loop = dim_product_size_in_loop \
                                     * self.input_dim_last * self.ele_size
                burst_length = input_size_in_loop // BLOCK_SIZE
                tik_instance.data_move(x[(dim_product_begin +
                                          dim_product_begin_in_loop)
                                         * self.input_dim_last],
                                       x_ub,
                                       0, 1, burst_length, 0, 0)

            # last loop
            i = loops
            dim_product_begin_in_loop = i * max_dim_product
            dim_product_size_in_loop = dim_product_size - dim_product_begin_in_loop

            x_ub = tik_instance.Tensor(self.dtype, (max_dim_product, self.input_dim_last), \
                                       name="x_ub", scope=tik.scope_ubuf)
            y_ub = tik_instance.Tensor(self.dtype, (max_dim_product, self.output_dim_last), \
                                       name="y_ub", scope=tik.scope_ubuf)

            output_size_in_loop = dim_product_size_in_loop * self.output_dim_last * self.ele_size
            burst_length_out = tik_instance.Scalar(dtype="int64")
            burst_length_out.set_as(output_size_in_loop // BLOCK_SIZE)
            with tik_instance.if_scope(output_size_in_loop % BLOCK_SIZE != 0):
                burst_length_out.set_as(burst_length_out + 1)
            tik_instance.data_move(y_ub,
                                   y[(dim_product_begin +
                                      dim_product_begin_in_loop)
                                     * self.output_dim_last],
                                   0, 1, burst_length_out, 0, 0)

            with tik_instance.for_range(0, dim_product_size_in_loop) as j:
                idx_x = j * self.input_dim_last
                idx_y = j * self.output_dim_last
                for k in range(self.input_dim_last):
                    max_num = (self.begin_last + self.output_dim_last)
                    if (k >= self.begin_last) and (k < max_num):
                        x_ub[idx_x + k] = y_ub[idx_y + k - self.begin_last]
                    else:
                        x_ub[idx_x + k] = pad_value

            input_size_in_loop = dim_product_size_in_loop * self.input_dim_last * self.ele_size
            burst_length = tik_instance.Scalar(dtype="int64")
            burst_length.set_as(input_size_in_loop // BLOCK_SIZE)
            with tik_instance.if_scope(input_size_in_loop % BLOCK_SIZE != 0):
                burst_length.set_as(burst_length + 1)
            tik_instance.data_move(x[(dim_product_begin +
                                      dim_product_begin_in_loop)
                                     * self.input_dim_last],
                                   x_ub,
                                   0, 1, burst_length, 0, 0)

        tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=[y], outputs=[x])


def _update_begin_end(input_shape, begin, end, begin_mask, end_mask):
    """ Calculate the value of padding by input parameters.

    Parameters
    ----------
    input_shape: list or tuple.
        shape of input.
    begin: list or tuple.
        represents the index of the first value to select.
    end: list or tuple.
        represents the index of the last value to select.
    begin_mask: int
        a bit mask where a bit i being 1 means to ignore the begin value and instead use the
        largest interval possible.
    end_mask: int
        analogous to `begin_mask`.

    Returns
    -------
    begin_shape: list.
        shape of 'begin' after mask handle
    end_shape: list.
        shape of 'end' after mask handle
    """
    begin_shape = list(begin)
    end_shape = list(end)

    if end_shape[-1] > input_shape[-1]:
        end_shape[-1] = input_shape[-1]

    # If the ith bit of begin_mask is set, begin[i] is ignored,
    # and the fullest possible range in that dimension is used instead.
    # end_mask works analogously, except with the end range.
    for i, _ in enumerate(zip(input_shape, begin_shape, end_shape)):
        # process begin_mask
        if (begin_mask & 2**i) == 2**i:
            begin_shape[i] = 0
        # process end_mask
        if (end_mask & 2**i) == 2**i:
            end_shape[i] = input_shape[i]

    return begin_shape, end_shape


def _get_paddings(shape_x, begin_shape, end_shape):
    """ Calculate the value of padding by input parameters.

    Parameters
    ----------
    shape_x: list or tuple.
        shape of output.
    begin_shape: list or tuple.
        represents the index of the first value to select.
    end_shape: list or tuple.
        represents the index of the last value to select.

    Returns
    -------
    paddings: list.
        indicates how many zeros to add after the contents of `shape_dy` in every dimension
    """
    paddings = []
    for begin_i, shape_x_i, end_i in zip(begin_shape, shape_x, end_shape):
        if begin_i < 0:
            begin_i += shape_x_i
        if end_i < 0:
            end_i += shape_x_i
        paddings.append([begin_i, shape_x_i - end_i])

    return paddings


def _check_shape_parameter(shape_x, shape_dy, begin, end, strides):
    """ Check whether the input shape meets the requirements.

    Parameters
    ----------
    shape_x: list or tuple.
        shape of output.
    shape_dy: list or tuple.
        shape of input.
    begin: list or tuple.
        represents the index of the first value to select.
    end: list or tuple.
        represents the index of the last value to select.
    strides: list or tuple.
        step length to select.

    Returns
    -------
    None.
    """
    # length of 'shape_x, shape_dy, begin, end, strides' must be the same
    if not (len(end) == len(begin) and \
            len(shape_x) == len(shape_dy) and \
            len(shape_x) == len(begin) and \
            len(shape_x) == len(strides)):
        raise RuntimeError("shape length mismatch!")

    # value of begin must less equal to end, and it's range is (0, shape_x_i).
    for i, (shape_x_i, begin_i, end_i) in enumerate(zip(shape_x, begin, end)):
        if begin_i < 0:
            begin_i += shape_x_i
        if end_i < 0:
            end_i += shape_x_i
        if not ((begin_i >= 0) and (end_i <= shape_x_i)
                and (begin_i <= end_i)):
            raise RuntimeError("Bound Over: begin[%d]:%d, end[%d]:%d, shape_x[%d]:%d\n" \
                               % (i, begin[i], i, end[i], i, shape_x_i))

    # value of strides must all be 1.
    for i, strides_i in enumerate(strides):
        if strides_i != 1:
            raise RuntimeError("Value of the strides[%d]:%d must be 1!" % (i, strides_i))


def _check_mask(input_mask):
    """ Check whether the value of the input mask is 0.

    Parameters
    ----------
    input_mask: int.
        value of the input mask.

    Returns
    -------
    None.
    """
    if input_mask != 0:
        raise RuntimeError("ellipsis_mask,new_axis_mask,shrink_axis_mask only support 0 currently")


# pylint: disable=locally-disabled,too-many-arguments,too-many-locals

@util.check_input_type(dict, dict, (list, tuple), (list, tuple), (list, tuple), (list, tuple),
                       int, int, int, int, int, str)
def strided_slice_grad_d(dy, output, shape, begin, end, strides, begin_mask=0,
                         end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0,
                         kernel_name="strided_slice_grad_d"):

    """ Since `StridedSlice` cuts out pieces of its `input` which is size`shape_dy`, its gradient
    will have the same shape (which is passed here as `shape_x`). The gradient will be zero in any
    element that the slice does not select.

    Parameters
    ----------
    dy : dict
        shape and dtype of input
    output_x : dict
        shape and dtype of out
    shape : list or tuple.
        shape of input
    begin: list or tuple.
        represents the index of the first value to select.
    end: list or tuple.
        represents the index of the last value to select.
    strides: list or tuple.
        step length to select.
    begin_mask: int
        a bitmask where a bit i being 1 means to ignore the begin value and instead use the
        largest interval possible.
    end_mask: int
        analogous to `begin_mask`.
    ellipsis_mask: int
        a bitmask where bit `i` being 1 means the `i`th position is actually an ellipsis.
    new_axis_mask: int
        a bitmask where bit `i` being 1 means the `i`th specification creates a
        new shape 1 dimension.
    shrink_axis_mask: int
        a bitmask where bit `i` implies that the `i`th specification should shrink
        the dimensionality.
    kernel_name : str
        cce kernel name, default value is "strided_slice_grad_d"

    Returns
    -------
    None.
    """
    shape_dy = dy.get("shape")
    input_dtype = dy.get("dtype").lower()

    util.check_dtype_rule(input_dtype, ("float16", "float32", "int8", "uint8", "int32"))
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape)
    util.check_shape_rule(shape_dy)
    util.check_shape_size(shape, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape_dy, SHAPE_SIZE_LIMIT)

    _check_mask(new_axis_mask)
    _check_mask(shrink_axis_mask)

    shape = list(shape)
    begin = list(begin)
    end = list(end)
    strides = list(strides)
    begin_shape, end_shape, stride_shape = _init_parameter(shape, begin, end,
                                                           strides, begin_mask, end_mask,
                                                           ellipsis_mask, new_axis_mask,
                                                           shrink_axis_mask)

    _check_shape_parameter(shape, shape_dy, begin_shape, end_shape, stride_shape)

    last_dim_compute = StridedSliceGradLastDimCompute(shape,
                                                      begin_shape,
                                                      shape_dy,
                                                      input_dtype, kernel_name)
    if last_dim_compute.check():
        last_dim_compute.strided_slice_grad()
    else:
        paddings = _get_paddings(shape, begin_shape, end_shape)

        # Call the pad operator due to gradient of 'StridedSlice' is the same as 'pad'
        # when the strides is 1.
        # pad.pad_cce(shape_dy, paddings, dtype, "CONSTANT", pad_value, kernel_name, need_build,
        #            need_print)
        dy_dict = {"shape": shape_dy, "dtype": input_dtype}
        pad_d(dy_dict, {}, paddings, kernel_name)
