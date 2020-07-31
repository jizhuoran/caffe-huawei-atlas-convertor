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

slice_d
"""
from __future__ import absolute_import
from __future__ import print_function

from functools import reduce as functools_reduce

from te import platform as cce
from te.platform import insn_cmd
from te import tvm
from te import tik
from te.platform.cce_build import build_config
from te.platform.fusion_manager import fusion_manager
import te.platform.cce_params as cce_params
from topi.cce import util

BLOCK_SIZE = 32
# available ub size
UB_SIZE_B = cce.cce_conf.get_soc_spec(cce.cce_conf.UB_SIZE)
# available number of cores
AICORE_NUM = cce.cce_conf.get_soc_spec(cce.cce_conf.CORE_NUM)


# pylint: disable=locally-disabled,too-many-lines,too-many-statements
# pylint: disable=locally-disabled,too-many-instance-attributes
# pylint: disable=locally-disabled,too-many-arguments
# pylint: disable=locally-disabled,too-many-locals,singleton-comparison
# pylint: disable=locally-disabled,invalid-name,attribute-defined-outside-init
class SliceLastDimCompute(object):
    """
    class of SliceLastDimCompute for last dim small scene

    """
    def __init__(self, shape, begin, size, dtype, kernel_name):
        self.dim_product = 1
        self.input_dim_last = 1
        self.output_dim_last = 1
        self.begin_last = 1
        self.dtype = dtype
        self.kernel_name = kernel_name
        self.ele_size = cce.cce_intrin.get_bit_len(self.dtype) // 8
        # align size for product dim, to make sure out data is 32B align
        self.product_dim_align_size = BLOCK_SIZE // self.ele_size

        # check only last dim to be sliced
        for i, (shape_i, begin_i, size_i) in enumerate(zip(reversed(shape),
                                                           reversed(begin),
                                                           reversed(size))):
            if i != 0:
                if shape_i != size_i:
                    self.check_result = False
                    return
                else:
                    self.dim_product *= shape_i
            else:
                self.input_dim_last = shape_i
                self.begin_last = begin_i
                self.output_dim_last = size_i

        # for moving data continuously, only small last dim is allowed
        # last dim data size <= 32B
        if self.input_dim_last * self.ele_size > 2 * BLOCK_SIZE:
            self.check_result = False
            return

        if self.output_dim_last * self.ele_size > BLOCK_SIZE:
            self.check_result = False
            return

        # for dividing cores easily, only big product dim is allowed
        # product dim >= aicore_num * 32 // ele_size
        aicore_num = AICORE_NUM
        if self.dim_product < self.product_dim_align_size * aicore_num:
            self.check_result = False
            return

        self.check_result = True

    def check(self):
        """
        function of checking for result

        """
        return self.check_result

    def _get_block_tiling(self, product, core, block_idx):
        """
        function of getting block tiling

        """
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
                begin.set_as(pack1 * block_idx * task_size -
                             (block_idx - tasks % core) * task_size)
                size.set_as(pack2 * task_size)
            with self.tik_instance.else_scope():
                begin.set_as(pack1 * block_idx * task_size)
                size.set_as(pack1 * task_size)

        with self.tik_instance.if_scope(block_idx == (core - 1)):
            size.set_as(product - begin)
        return begin, size

    def slice(self):
        """
        function of slice compute

        """
        if self.check_result == False:
            raise RuntimeError(
                "conditions of SliceLastDimCompute are not fulfilled")

        tik_instance = tik.Tik()
        self.tik_instance = tik_instance
        aicore_num = AICORE_NUM
        ub_size = UB_SIZE_B

        x = tik_instance.Tensor(self.dtype,
                                (self.dim_product, self.input_dim_last),
                                name="x", scope=tik.scope_gm)
        y = tik_instance.Tensor(self.dtype,
                                (self.dim_product, self.output_dim_last),
                                name="y", scope=tik.scope_gm)

        with tik_instance.for_range(0, aicore_num, block_num=aicore_num)\
                as block_idx:
            dim_product_begin, dim_product_size = self._get_block_tiling(
                self.dim_product, aicore_num, block_idx)
            max_dim_product = ub_size // self.ele_size\
                              // (self.input_dim_last + self.output_dim_last)\
                              // self.product_dim_align_size\
                              * self.product_dim_align_size
            loops = tik_instance.Scalar(dtype="int64")
            loops.set_as(dim_product_size // max_dim_product)
            with tik_instance.if_scope(dim_product_size % max_dim_product == 0):
                loops.set_as(loops - 1)

            with tik_instance.for_range(0, loops) as i:
                dim_product_begin_in_loop = i * max_dim_product
                dim_product_size_in_loop = max_dim_product

                x_ub = tik_instance.Tensor(self.dtype,
                                           (max_dim_product,
                                            self.input_dim_last),
                                           name="x_ub", scope=tik.scope_ubuf)
                y_ub = tik_instance.Tensor(self.dtype,
                                           (max_dim_product,
                                            self.output_dim_last),
                                           name="y_ub", scope=tik.scope_ubuf)

                input_size_in_loop = dim_product_size_in_loop\
                                     * self.input_dim_last * self.ele_size
                burst_length = input_size_in_loop // BLOCK_SIZE
                tik_instance.data_move(x_ub,
                                       x[(dim_product_begin
                                          + dim_product_begin_in_loop)
                                         * self.input_dim_last],
                                       0, 1, burst_length, 0, 0)

                with tik_instance.for_range(0, dim_product_size_in_loop) as j:
                    idx_x = j * self.input_dim_last + self.begin_last
                    idx_y = j * self.output_dim_last
                    for k in range(self.output_dim_last):
                        y_ub[idx_y + k] = x_ub[idx_x + k]

                output_size_in_loop = dim_product_size_in_loop\
                                      * self.output_dim_last * self.ele_size
                burst_length_out = output_size_in_loop // BLOCK_SIZE
                tik_instance.data_move(y[(dim_product_begin
                                          + dim_product_begin_in_loop)
                                         * self.output_dim_last],
                                       y_ub,
                                       0, 1, burst_length_out, 0, 0)

            # last loop
            i = loops
            dim_product_begin_in_loop = i * max_dim_product
            dim_product_size_in_loop = dim_product_size\
                                       - dim_product_begin_in_loop

            x_ub = tik_instance.Tensor(self.dtype,
                                       (max_dim_product, self.input_dim_last),
                                       name="x_ub", scope=tik.scope_ubuf)
            y_ub = tik_instance.Tensor(self.dtype,
                                       (max_dim_product, self.output_dim_last),
                                       name="y_ub", scope=tik.scope_ubuf)

            input_size_in_loop = dim_product_size_in_loop\
                                 * self.input_dim_last * self.ele_size
            burst_length = tik_instance.Scalar(dtype="int64")
            burst_length.set_as(input_size_in_loop // BLOCK_SIZE)
            with tik_instance.if_scope(input_size_in_loop % BLOCK_SIZE != 0):
                burst_length.set_as(burst_length + 1)
            tik_instance.data_move(x_ub,
                                   x[(dim_product_begin
                                      + dim_product_begin_in_loop)
                                     * self.input_dim_last],
                                   0, 1, burst_length, 0, 0)

            with tik_instance.for_range(0, dim_product_size_in_loop) as j:
                idx_x = j * self.input_dim_last + self.begin_last
                idx_y = j * self.output_dim_last
                for k in range(self.output_dim_last):
                    y_ub[idx_y + k] = x_ub[idx_x + k]

            output_size_in_loop = dim_product_size_in_loop\
                                  * self.output_dim_last * self.ele_size
            burst_length_out = tik_instance.Scalar(dtype="int64")
            burst_length_out.set_as(output_size_in_loop // BLOCK_SIZE)
            with tik_instance.if_scope(output_size_in_loop % BLOCK_SIZE != 0):
                burst_length_out.set_as(burst_length_out + 1)
            tik_instance.data_move(y[(dim_product_begin
                                      + dim_product_begin_in_loop)
                                     * self.output_dim_last],
                                   y_ub,
                                   0, 1, burst_length_out, 0, 0)

        tik_instance.BuildCCE(kernel_name=self.kernel_name,
                              inputs=[x], outputs=[y])


class SliceDiffLastDimCompute(object):
    """
    class of SliceLastDimCompute for last dim small scene

    """
    def __init__(self, shape, begin, size, dtype, kernel_name):
        self.dim_product = 1
        self.input_dim_last = 1
        self.output_dim_last = 1
        self.begin_last = 1
        self.begin_first = 1
        self.all_dim = len(shape)
        self.dtype = dtype
        self.kernel_name = kernel_name
        self.ele_size = cce.cce_intrin.get_bit_len(self.dtype) // 8
        # align size for product dim, to make sure out data is 32B align
        self.product_dim_align_size = BLOCK_SIZE // self.ele_size

        if self.all_dim < 3:
            self.check_result = False
            return

        # check only last dim to be sliced
        for i, (shape_i, begin_i, size_i) in enumerate(zip(reversed(shape),
                                                           reversed(begin),
                                                           reversed(size))):
            if i == 0:
                self.input_dim_last = shape_i
                self.begin_last = begin_i
                self.output_dim_last = size_i
            elif i == self.all_dim - 1:
                if not (shape_i > 1 and size_i == 1):
                    self.check_result = False
                    return
                else:
                    self.begin_first = begin_i
            else:
                if shape_i != size_i:
                    self.check_result = False
                    return
                else:
                    self.dim_product *= shape_i

        # for moving data continuously, only small last dim is allowed
        # last dim data size <= 32B
        if self.input_dim_last * self.ele_size > 2 * BLOCK_SIZE:
            self.check_result = False
            return

        if self.output_dim_last * self.ele_size > BLOCK_SIZE:
            self.check_result = False
            return

        # for dividing cores easily, only big product dim is allowed
        # product dim >= aicore_num * 32 // ele_size
        aicore_num = AICORE_NUM
        if self.dim_product < self.product_dim_align_size * aicore_num:
            self.check_result = False
            return

        self.check_result = True

    def check(self):
        """
        function of checking for result

        """
        return self.check_result

    def _get_block_tiling(self, product, core, block_idx):
        """
        function of getting block tiling

        """
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
                begin.set_as(pack1 * block_idx * task_size -
                             (block_idx - tasks % core) * task_size)
                size.set_as(pack2 * task_size)
            with self.tik_instance.else_scope():
                begin.set_as(pack1 * block_idx * task_size)
                size.set_as(pack1 * task_size)

        with self.tik_instance.if_scope(block_idx == (core - 1)):
            size.set_as(product - begin)
        return begin, size

    def slice(self):
        """
        function of slice compute

        """
        if self.check_result == False:
            raise RuntimeError(
                "conditions of SliceLastDimCompute are not fulfilled")

        tik_instance = tik.Tik()
        self.tik_instance = tik_instance
        aicore_num = AICORE_NUM
        ub_size = UB_SIZE_B

        x = tik_instance.Tensor(self.dtype,
                                (self.dim_product, self.input_dim_last),
                                name="x", scope=tik.scope_gm)
        y = tik_instance.Tensor(self.dtype,
                                (self.dim_product, self.output_dim_last),
                                name="y", scope=tik.scope_gm)

        with tik_instance.for_range(0, aicore_num, block_num=aicore_num)\
                as block_idx:
            dim_product_begin, dim_product_size = self._get_block_tiling(
                self.dim_product, aicore_num, block_idx)
            max_dim_product = ub_size // self.ele_size\
                              // (self.input_dim_last + self.output_dim_last)\
                              // self.product_dim_align_size\
                              * self.product_dim_align_size
            loops = tik_instance.Scalar(dtype="int64")
            loops.set_as(dim_product_size // max_dim_product)
            input_ele = self.dim_product * self.input_dim_last
            with tik_instance.if_scope(
                dim_product_size % max_dim_product == 0):
                loops.set_as(loops - 1)

            with tik_instance.for_range(0, loops) as i:
                dim_product_begin_in_loop = i * max_dim_product
                dim_product_size_in_loop = max_dim_product

                x_ub = tik_instance.Tensor(self.dtype,
                                           (max_dim_product,
                                            self.input_dim_last),
                                           name="x_ub", scope=tik.scope_ubuf)
                y_ub = tik_instance.Tensor(self.dtype,
                                           (max_dim_product,
                                            self.output_dim_last),
                                           name="y_ub", scope=tik.scope_ubuf)

                input_size_in_loop = dim_product_size_in_loop\
                                     * self.input_dim_last * self.ele_size
                burst_length = input_size_in_loop // BLOCK_SIZE
                tik_instance.data_move(x_ub,
                                       x[(self.begin_first * input_ele)
                                         + (dim_product_begin
                                            + dim_product_begin_in_loop)
                                         * self.input_dim_last],
                                       0, 1, burst_length, 0, 0)

                with tik_instance.for_range(0, dim_product_size_in_loop) as j:
                    idx_x = j * self.input_dim_last + self.begin_last
                    idx_y = j * self.output_dim_last
                    for k in range(self.output_dim_last):
                        y_ub[idx_y + k] = x_ub[idx_x + k]

                output_size_in_loop = dim_product_size_in_loop\
                                      * self.output_dim_last * self.ele_size
                burst_length_out = output_size_in_loop // BLOCK_SIZE
                tik_instance.data_move(y[(dim_product_begin
                                          + dim_product_begin_in_loop)
                                         * self.output_dim_last],
                                       y_ub,
                                       0, 1, burst_length_out, 0, 0)

            # last loop
            i = loops
            dim_product_begin_in_loop = i * max_dim_product
            dim_product_size_in_loop = dim_product_size\
                                       - dim_product_begin_in_loop

            x_ub = tik_instance.Tensor(self.dtype,
                                       (max_dim_product, self.input_dim_last),
                                       name="x_ub", scope=tik.scope_ubuf)
            y_ub = tik_instance.Tensor(self.dtype,
                                       (max_dim_product, self.output_dim_last),
                                       name="y_ub", scope=tik.scope_ubuf)

            input_size_in_loop = dim_product_size_in_loop\
                                 * self.input_dim_last * self.ele_size
            burst_length = tik_instance.Scalar(dtype="int64")
            burst_length.set_as(input_size_in_loop // BLOCK_SIZE)
            with tik_instance.if_scope(input_size_in_loop % BLOCK_SIZE != 0):
                burst_length.set_as(burst_length + 1)
            tik_instance.data_move(x_ub,
                                   x[(self.begin_first * input_ele)
                                     + (dim_product_begin
                                        + dim_product_begin_in_loop)
                                     * self.input_dim_last],
                                   0, 1, burst_length, 0, 0)

            with tik_instance.for_range(0, dim_product_size_in_loop) as j:
                idx_x = j * self.input_dim_last + self.begin_last
                idx_y = j * self.output_dim_last
                for k in range(self.output_dim_last):
                    y_ub[idx_y + k] = x_ub[idx_x + k]

            output_size_in_loop = dim_product_size_in_loop\
                                  * self.output_dim_last * self.ele_size
            burst_length_out = tik_instance.Scalar(dtype="int64")
            burst_length_out.set_as(output_size_in_loop // BLOCK_SIZE)
            with tik_instance.if_scope(output_size_in_loop % BLOCK_SIZE != 0):
                burst_length_out.set_as(burst_length_out + 1)
            tik_instance.data_move(y[(dim_product_begin
                                      + dim_product_begin_in_loop)
                                     * self.output_dim_last],
                                   y_ub,
                                   0, 1, burst_length_out, 0, 0)

        tik_instance.BuildCCE(kernel_name=self.kernel_name,
                              inputs=[x], outputs=[y])


def _check_parameters(shape, dtype, begin, size, kernel_name):
    """
    check the parameters including shape, dtype, begin, size and kernel_name

    """
    util.check_dtype_rule(dtype, ("int8", "int16", "int32", "int64", "uint8",
                                  "uint16", "uint32", "uint64",
                                  "float16", "float32"))
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape)
    util.check_tensor_shape_size(shape)

    if not (len(shape) == len(begin) and len(shape) == len(size)):
        raise RuntimeError(
            "the length of begin and size must be equal to shape!")

    for i, (shape_i, begin_i) in enumerate(zip(shape, begin)):
        if not (isinstance(begin[i], int) and 0 <= begin_i < shape_i):
            raise RuntimeError(
                "value of begin must be int,"
                " greater than or equal to 0, and less than shape!")

    for i, (shape_i, size_i) in enumerate(zip(shape, size)):
        if not (isinstance(size[i], int) and -1 <= size_i <= shape_i
                and size_i != 0):
            raise RuntimeError(
                "value of size must be int, greater than or equal to -1,"
                " less than or equal to shape, and cannot be equal to 0!")

    end = _get_end(shape, begin, size)
    for i, (shape_i, end_i) in enumerate(zip(shape, end)):
        if end[i] <= 0 or end_i > shape_i:
            raise RuntimeError(
                "value of end must be greater than 0,"
                " less than or equal to shape!")


def _get_end(shape, begin, size):
    """
    calculate value of end according to shape, begin and size

    """
    end = []
    for i, (begin_i, size_i) in enumerate(zip(begin, size)):
        if size_i == -1:
            end.append(shape[i])
        else:
            end.append(begin_i + size_i)

    return end


def _update_params(shape, begin, size):
    """
    update the shape, begin and size parameters

    """
    output_shape = []
    for i, item in enumerate(size):
        if item != -1:
            output_shape.append(item)
        else:
            output_shape.append(shape[i] - begin[i])

    input_shape_new = list(shape)
    out_shape_new = list(output_shape)
    begin_new = list(begin)

    flag = -1
    for i, (shape_i, size_i) in enumerate(zip(reversed(input_shape_new),
                                              reversed(out_shape_new))):
        if not (shape_i == 1 and size_i == 1):
            flag = i
            break

    # remove the last number of shape and size that are all 1
    # for more efficient tilling strategy
    while flag > 0:
        input_shape_new.pop()
        out_shape_new.pop()
        begin_new.pop()
        flag -= 1

    return input_shape_new, begin_new, out_shape_new


def _is_special_shape(out_shape):
    """
    whether the shape needs special tilling method

    """
    if not (len(out_shape) > 2 and out_shape[-1] == 1):
        return False

    if out_shape[-1] == 1:
        count = 0
        for item in out_shape:
            if item != 1:
                count += 1
        if count > 1:
            return True

    return False


def _get_last_not_one(shape):
    """
    get the first axis which is not one from the back

    """
    flag = -1
    axis = 0
    for i, item in enumerate(reversed(shape)):
        if item > 1:
            flag = i
            break
    if flag != -1:
        axis = len(shape) - flag - 1

    return axis


def _get_factor(ele_zero, ele_cnt, total_ele, no_remainder):
    """
    get split factor for _tilling_one_axis function

    """
    split_factor = 1
    if no_remainder:
        for i in reversed(list(range(1, ele_zero))):
            if ele_zero % i == 0 and i*ele_cnt <= total_ele:
                split_factor = i
                break
    else:
        for i in reversed(list(range(1, ele_zero))):
            if i*ele_cnt <= total_ele:
                split_factor = i
                break

    return split_factor


def _tilling_axis(shape, dtype, no_remainder):
    """
    calculate the split parameters according to different shapes

    """
    ub_size_bytes = UB_SIZE_B - 1024
    dtype_bytes_size = cce.cce_intrin.get_bit_len(dtype) // 8
    # 32 means one block size(32 Bytes), divide by 32 to get
    # the numbers of data that can be stored in one block.
    flag = 32 // dtype_bytes_size
    element_new = ((shape[-1] + flag - 1) // flag)*flag
    shape_new = []
    for i in shape:
        shape_new.append(i)
    shape_new[-1] = int(element_new)

    total_ele = ub_size_bytes // dtype_bytes_size
    split_axis = 0
    split_factor = 1
    for i, _ in enumerate(shape_new):
        ele_cnt = functools_reduce(lambda x, y: x*y, shape_new[i:])
        if ele_cnt <= total_ele:
            split_axis = i - 1
            split_factor = total_ele // ele_cnt
            if no_remainder and i == 1 and shape_new[0] % split_factor != 0:
                split_factor = _get_factor(shape_new[0], ele_cnt, total_ele,
                                           no_remainder)
            break
        elif i == len(shape) - 1:
            if len(shape) == 1:
                split_axis = 0
                split_factor = _get_factor(shape_new[0], 1, total_ele,
                                           no_remainder)
            else:
                split_axis = i
                split_factor = total_ele
            break

    if split_axis < 0:
        split_axis = 0
        split_factor = shape_new[0]

    if _is_special_shape(shape):
        last_not_one_axis = _get_last_not_one(shape)
        if split_axis < last_not_one_axis:
            split_axis = last_not_one_axis
            split_factor = shape[last_not_one_axis]

    if no_remainder:
        device_core_num = AICORE_NUM
        if len(shape) >= 2 and split_axis == 0\
                and shape[0] >= device_core_num\
                and shape[0] < (2 * device_core_num)\
                and shape[0] < 65535:
            split_factor = 1

    return split_axis, split_factor


def _get_align_axis(out_shape):
    """
    get the axis_info when applying the align

    """
    flag = -1
    if out_shape[-1] != 1:
        axis = len(out_shape) - 2
    else:
        for i, item in enumerate(reversed(out_shape)):
            if item > 1:
                flag = i
                break
        if flag == -1:
            axis = 0
        else:
            axis = len(out_shape) - flag - 1

    return axis


# pylint: disable=locally-disabled,unused-argument,too-many-locals
# pylint: disable=locally-disabled,unnecessary-lambda
@fusion_manager.register("slice_d")
def slice_d_compute(x, y, begin, size, kernel_name="slice_d"):
    """
    calculating: this operation extracts a slice of size size
                 from a tensor input
                 starting at the location specified by begin.

    Parameters
    ----------
    x: tvm.tensor
        tensor of input data
    y: dict
        contains shape and dtype information of output tensor
    begin: list or tuple
        represents the index of the first value to select
    size: list or tuple
        represents the shape of output tensor
    kernel_name: str
        cce kernel name, default value is "slice_d".

    Returns
    -------
    sch: tvm.schedule
        the compute schedule
    res: tvm.tensor
        the output tensor
    """
    def _map_index_norm(*index):
        """
        calculate normal index by strided and begin parameters.

        """
        for i, begin_i in enumerate(begin):
            if i == 0:
                index_org = (index[i] + begin_i,)
            else:
                index_org = index_org + (index[i] + begin_i,)

        return index_org

    data_ub = tvm.compute(size, lambda *i: x(*_map_index_norm(*i)),
                          name='data_ub')
    res = tvm.compute(size, lambda *i: data_ub(*i), name='res')
    sch = tvm.create_schedule(res.op)
    sch[data_ub].set_scope(cce.scope_ubuf)

    float_size = cce.cce_intrin.get_bit_len(x.dtype) // 8
    element_align = cce_params.BLOCK_REDUCE_INT8 // float_size

    if size[-1] < element_align:
        split_axis, split_factor = _tilling_axis(size, x.dtype, False)
        axis_outer, axis_inner = sch[res].split(res.op.axis[split_axis],
                                                factor=split_factor)
    else:
        split_axis, split_factor = _tilling_axis(size, x.dtype, True)
        if split_axis == 0:
            core_num = size[split_axis] // split_factor
        else:
            core_num = size[0]
        if (split_axis == len(size) - 1 and split_factor < element_align)\
                or core_num > 65535:
            split_axis, split_factor = _tilling_axis(size, x.dtype, False)
            axis_outer, axis_inner = sch[res].split(res.op.axis[split_axis],
                                                    factor=split_factor)
        else:
            axis_outer, axis_inner = sch[res].split(res.op.axis[split_axis],
                                                    factor=split_factor)
            if split_axis == 0:
                sch[res].bind(axis_outer, tvm.thread_axis('blockIdx.x'))
            else:
                sch[res].bind(res.op.axis[0], tvm.thread_axis('blockIdx.x'))

    sch[data_ub].compute_at(sch[res], axis_outer)
    sch[data_ub].emit_insn(data_ub.op.axis[split_axis], insn_cmd.DMA_COPY)
    if len(size) >= 2:
        # 32 means one block size(32 Bytes), divide by 32 to get
        # the numbers of data that can be stored in one block.
        element = cce_params.BLOCK_REDUCE_INT8 // float_size
        align_axis = _get_align_axis(size)
        sch[data_ub].storage_align(data_ub.op.axis[align_axis], element, 0)
    sch[res].emit_insn(axis_inner, insn_cmd.DMA_COPY)

    return sch, res


def _ceil_div(value, block):
    """
    integrate the input value by block

    """
    return (value + block - 1) // block


def _ceil_fill(value, block):
    """
    fill the input value by block

    """
    return _ceil_div(value, block)*block


def _new_alloc(tvm_ib, dtype, shape, name, scope):
    """
    decl new buffer for ir builder make function

    """
    buf_var = tvm_ib.allocate(dtype, shape, name=name, scope=scope)
    new_buffer = tvm.decl_buffer(shape, buf_var.dtype, name=name,
                                 scope=scope, data=buf_var)

    return new_buffer


def _func_gm_to_ub(args):
    """
    function of moving data from data to data_ub

    """
    tvm_ib, param, data, data_ub, data_offset, ub_offset, ori_nburst,\
    burst_len, src_stride, dst_stride = args

    with tvm_ib.if_scope(ori_nburst > 0):
        with tvm_ib.if_scope(burst_len > 0):
            with tvm_ib.if_scope(burst_len <= 65535):
                with tvm_ib.if_scope(src_stride >= 0):
                    with tvm_ib.if_scope(dst_stride >= 0):
                        with tvm_ib.if_scope(dst_stride <= 65535):
                            with tvm_ib.if_scope(src_stride <= 65535):
                                with tvm_ib.if_scope(ori_nburst <= 4095):
                                    tvm_ib.emit(
                                        tvm.call_extern(
                                            data_ub.dtype,
                                            "copy_gm_to_ubuf",
                                            data_ub.access_ptr(
                                                "w", offset=ub_offset),
                                            data.access_ptr(
                                                'r', offset=data_offset),
                                            0, ori_nburst,
                                            burst_len,
                                            src_stride, dst_stride))
                                with tvm_ib.else_scope():
                                    n_burst = 4095
                                    c_cycle = ori_nburst // n_burst
                                    c_mod = ori_nburst % n_burst
                                    with tvm_ib.for_range(0, c_cycle,
                                                          name="num_cy")\
                                            as num_cy:
                                        data_cur = data_offset + (
                                            burst_len + src_stride) \
                                                   * param.get("cp_align_len")\
                                                   * n_burst * num_cy
                                        ub_cur = ub_offset + (
                                            burst_len + dst_stride) \
                                                 * param.get("cp_align_len")\
                                                 * n_burst * num_cy
                                        tvm_ib.emit(
                                            tvm.call_extern(
                                                data_ub.dtype,
                                                "copy_gm_to_ubuf",
                                                data_ub.access_ptr(
                                                    "w",
                                                    offset=ub_cur),
                                                data.access_ptr(
                                                    'r',
                                                    offset=data_cur),
                                                0, n_burst,
                                                burst_len,
                                                src_stride,
                                                dst_stride))
                                    with tvm_ib.if_scope(c_mod > 0):
                                        data_cur = data_offset + (
                                            burst_len + src_stride) \
                                                   * param.get("cp_align_len")\
                                                   * n_burst * c_cycle
                                        ub_cur = ub_offset + (
                                            burst_len + dst_stride) \
                                                 * param.get("cp_align_len")\
                                                 * n_burst * c_cycle
                                        tvm_ib.emit(
                                            tvm.call_extern(
                                                data_ub.dtype,
                                                "copy_gm_to_ubuf",
                                                data_ub.access_ptr(
                                                    "w", offset=ub_cur),
                                                data.access_ptr(
                                                    'r', offset=data_cur),
                                                0, c_mod, burst_len,
                                                src_stride,
                                                dst_stride))
                            with tvm_ib.else_scope():
                                with tvm_ib.for_range(0, ori_nburst,
                                                      name="num_nb") as num_nb:
                                    data_cur = data_offset + (
                                        burst_len + src_stride)\
                                               * param.get("cp_align_len")\
                                               * num_nb
                                    ub_cur = ub_offset + (
                                        burst_len + dst_stride)\
                                             * param.get("cp_align_len")\
                                             * num_nb
                                    tvm_ib.emit(
                                        tvm.call_extern(
                                            data_ub.dtype,
                                            "copy_gm_to_ubuf",
                                            data_ub.access_ptr(
                                                "w", offset=ub_cur),
                                            data.access_ptr(
                                                'r', offset=data_cur),
                                            0, 1, burst_len,
                                            0, 0))


def _func_gm_to_ub_align(args):
    """
    function of moving data from data to data_ub

    """
    tvm_ib, data, data_ub, data_offset, ub_offset, ori_nburst,\
    burst_len, src_stride, dst_stride, cp_align_len = args

    with tvm_ib.if_scope(ori_nburst > 0):
        with tvm_ib.if_scope(burst_len > 0):
            with tvm_ib.if_scope(burst_len <= 65535):
                with tvm_ib.if_scope(src_stride >= 0):
                    with tvm_ib.if_scope(dst_stride >= 0):
                        with tvm_ib.if_scope(dst_stride <= 65535):
                            with tvm_ib.if_scope(src_stride <= 65535):
                                with tvm_ib.if_scope(ori_nburst <= 4095):
                                    tvm_ib.emit(
                                        tvm.call_extern(
                                            data_ub.dtype,
                                            "copy_gm_to_ubuf",
                                            data_ub.access_ptr(
                                                "w", offset=ub_offset),
                                            data.access_ptr(
                                                'r', offset=data_offset),
                                            0, ori_nburst,
                                            burst_len,
                                            src_stride, dst_stride))
                                with tvm_ib.else_scope():
                                    n_burst = 4095
                                    c_cycle = ori_nburst // n_burst
                                    c_mod = ori_nburst % n_burst
                                    with tvm_ib.for_range(0, c_cycle,
                                                          name="num_cy")\
                                            as num_cy:
                                        data_cur = data_offset + (
                                            burst_len + src_stride) \
                                                   * cp_align_len\
                                                   * n_burst * num_cy
                                        ub_cur = ub_offset + (
                                            burst_len + dst_stride) \
                                                 * cp_align_len\
                                                 * n_burst * num_cy
                                        tvm_ib.emit(
                                            tvm.call_extern(
                                                data_ub.dtype,
                                                "copy_gm_to_ubuf",
                                                data_ub.access_ptr(
                                                    "w",
                                                    offset=ub_cur),
                                                data.access_ptr(
                                                    'r',
                                                    offset=data_cur),
                                                0, n_burst,
                                                burst_len,
                                                src_stride,
                                                dst_stride))
                                    with tvm_ib.if_scope(c_mod > 0):
                                        data_cur = data_offset + (
                                            burst_len + src_stride) \
                                                   * cp_align_len\
                                                   * n_burst * c_cycle
                                        ub_cur = ub_offset + (
                                            burst_len + dst_stride) \
                                                 * cp_align_len\
                                                 * n_burst * c_cycle
                                        tvm_ib.emit(
                                            tvm.call_extern(
                                                data_ub.dtype,
                                                "copy_gm_to_ubuf",
                                                data_ub.access_ptr(
                                                    "w", offset=ub_cur),
                                                data.access_ptr(
                                                    'r', offset=data_cur),
                                                0, c_mod, burst_len,
                                                src_stride,
                                                dst_stride))
                            with tvm_ib.else_scope():
                                with tvm_ib.for_range(0, ori_nburst,
                                                      name="num_nb") as num_nb:
                                    data_cur = data_offset + (
                                        burst_len + src_stride)\
                                               * cp_align_len\
                                               * num_nb
                                    ub_cur = ub_offset + (
                                        burst_len + dst_stride)\
                                             * cp_align_len\
                                             * num_nb
                                    tvm_ib.emit(
                                        tvm.call_extern(
                                            data_ub.dtype,
                                            "copy_gm_to_ubuf",
                                            data_ub.access_ptr(
                                                "w", offset=ub_cur),
                                            data.access_ptr(
                                                'r', offset=data_cur),
                                            0, 1, burst_len,
                                            0, 0))


def _reg_mov_row(args):
    """
    move row from data_ub to data_res

    """
    tvm_ib, data_ub, data_res, reg, num_d, num_c, dim_ele_in, dim_ele_out,\
    col_begin, row_begin, row_in, row_out = args

    ele_reg = 8
    r_cycle = row_out // ele_reg
    r_mod = row_out - ele_reg*r_cycle
    reg_zero = 0
    reg_one = 1
    reg_two = 2
    reg_three = 3
    reg_four = 4
    reg_five = 5
    reg_six = 6
    reg_seven = 7

    with tvm_ib.for_range(0, r_cycle, name="num_cr") as num_cr:
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_zero]),
            data_ub.access_ptr('r',
                               offset=(num_d * dim_ele_in
                                       + (col_begin + num_c) * row_in
                                       + row_begin
                                       + (num_cr*ele_reg + reg_zero)))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_one]),
            data_ub.access_ptr('r',
                               offset=(num_d * dim_ele_in
                                       + (col_begin + num_c) * row_in
                                       + row_begin
                                       + (num_cr*ele_reg + reg_one)))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_two]),
            data_ub.access_ptr('r',
                               offset=(num_d * dim_ele_in
                                       + (col_begin + num_c) * row_in
                                       + row_begin
                                       + (num_cr*ele_reg + reg_two)))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_three]),
            data_ub.access_ptr('r',
                               offset=(num_d * dim_ele_in
                                       + (col_begin + num_c) * row_in
                                       + row_begin
                                       + (num_cr*ele_reg + reg_three)))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_four]),
            data_ub.access_ptr('r',
                               offset=(num_d * dim_ele_in
                                       + (col_begin + num_c) * row_in
                                       + row_begin
                                       + (num_cr*ele_reg + reg_four)))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_five]),
            data_ub.access_ptr('r',
                               offset=(num_d * dim_ele_in
                                       + (col_begin + num_c) * row_in
                                       + row_begin
                                       + (num_cr*ele_reg + reg_five)))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_six]),
            data_ub.access_ptr('r',
                               offset=(num_d * dim_ele_in
                                       + (col_begin + num_c) * row_in
                                       + row_begin
                                       + (num_cr*ele_reg + reg_six)))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_seven]),
            data_ub.access_ptr('r',
                               offset=(num_d * dim_ele_in
                                       + (col_begin + num_c) * row_in
                                       + row_begin
                                       + (num_cr*ele_reg + reg_seven)))
        ))

        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=num_d * dim_ele_out
                                + num_c * row_out
                                + (num_cr*ele_reg + reg_zero)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_zero])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=num_d * dim_ele_out
                                + num_c * row_out
                                + (num_cr*ele_reg + reg_one)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_one])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=num_d * dim_ele_out
                                + num_c * row_out
                                + (num_cr*ele_reg + reg_two)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_two])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=num_d * dim_ele_out
                                + num_c * row_out
                                + (num_cr*ele_reg + reg_three)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_three])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=num_d * dim_ele_out
                                + num_c * row_out
                                + (num_cr*ele_reg + reg_four)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_four])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=num_d * dim_ele_out
                                + num_c * row_out
                                + (num_cr*ele_reg + reg_five)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_five])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=num_d * dim_ele_out
                                + num_c * row_out
                                + (num_cr*ele_reg + reg_six)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_six])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=num_d * dim_ele_out
                                + num_c * row_out
                                + (num_cr*ele_reg + reg_seven)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_seven])
        ))

    with tvm_ib.if_scope(r_mod > 0):
        with tvm_ib.for_range(0, r_mod, name="num_er") as num_er:
            tvm_ib.emit(tvm.call_extern(
                data_ub.dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[0]),
                data_ub.access_ptr('r',
                                   offset=(num_d * dim_ele_in
                                           + (col_begin + num_c) * row_in
                                           + row_begin
                                           + (r_cycle*ele_reg + num_er)))
            ))
            tvm_ib.emit(tvm.call_extern(
                data_res.dtype, "reg_mov",
                data_res.access_ptr('w', offset=num_d * dim_ele_out
                                    + num_c * row_out
                                    + (r_cycle*ele_reg + num_er)),
                tvm.call_extern(reg.dtype, "reg", reg[0])
            ))


def _reg_mov_align(args):
    """
    move from data_res to data_tail for cp_align_len

    """
    tvm_ib, data_ub, data_res, reg, one_loop_ele_out_align, cp_align_len = args

    ele_reg = 8
    r_cycle = cp_align_len // ele_reg
    r_mod = cp_align_len - ele_reg * r_cycle
    reg_zero = 0
    reg_one = 1
    reg_two = 2
    reg_three = 3
    reg_four = 4
    reg_five = 5
    reg_six = 6
    reg_seven = 7

    with tvm_ib.for_range(0, r_cycle, name="num_cr") as num_cr:
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_zero]),
            data_ub.access_ptr('r',
                               offset=(one_loop_ele_out_align
                                       + (num_cr * ele_reg + reg_zero)))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_one]),
            data_ub.access_ptr('r',
                               offset=(one_loop_ele_out_align
                                       + (num_cr * ele_reg + reg_one)))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_two]),
            data_ub.access_ptr('r',
                               offset=(one_loop_ele_out_align
                                       + (num_cr * ele_reg + reg_two)))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_three]),
            data_ub.access_ptr('r',
                               offset=(one_loop_ele_out_align
                                       + (num_cr * ele_reg + reg_three)))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_four]),
            data_ub.access_ptr('r',
                               offset=(one_loop_ele_out_align
                                       + (num_cr * ele_reg + reg_four)))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_five]),
            data_ub.access_ptr('r',
                               offset=(one_loop_ele_out_align
                                       + (num_cr * ele_reg + reg_five)))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_six]),
            data_ub.access_ptr('r',
                               offset=(one_loop_ele_out_align
                                       + (num_cr * ele_reg + reg_six)))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_seven]),
            data_ub.access_ptr('r',
                               offset=(one_loop_ele_out_align
                                       + (num_cr * ele_reg + reg_seven)))
        ))

        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(num_cr * ele_reg + reg_zero)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_zero])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(num_cr * ele_reg + reg_one)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_one])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(num_cr * ele_reg + reg_two)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_two])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(num_cr * ele_reg + reg_three)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_three])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(num_cr * ele_reg + reg_four)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_four])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(num_cr * ele_reg + reg_five)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_five])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(num_cr * ele_reg + reg_six)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_six])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(num_cr * ele_reg + reg_seven)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_seven])
        ))

    with tvm_ib.if_scope(r_mod > 0):
        with tvm_ib.for_range(0, r_mod, name="num_er") as num_er:
            tvm_ib.emit(tvm.call_extern(
                data_ub.dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[0]),
                data_ub.access_ptr('r',
                                   offset=(one_loop_ele_out_align
                                           + (r_cycle * ele_reg + num_er)))
            ))
            tvm_ib.emit(tvm.call_extern(
                data_res.dtype, "reg_mov",
                data_res.access_ptr('w', offset=(r_cycle * ele_reg + num_er)),
                tvm.call_extern(reg.dtype, "reg", reg[0])
            ))


def _func_mov_scalar_one(args):
    """
    function of moving data in mov scalar one function

    """
    tvm_ib, data, dst, data_ub, data_res, data_tail, reg, device_core_num,\
    block_index, cp_align_len, col_begin, row_begin, ub_ele, num_g = args

    n_index = num_g * device_core_num + block_index
    _, num_dim_one_core, col_in, row_in = data.shape
    _, _, col_out, row_out = dst.shape

    one_core_ele_in = num_dim_one_core*col_in*row_in
    one_core_ele_out = num_dim_one_core * col_out * row_out

    dim_ele_in = col_in*row_in
    dim_ele_out = col_out*row_out
    num_dim_one_loop = ub_ele // dim_ele_in
    ub_loop = num_dim_one_core // num_dim_one_loop
    num_dim_mod = num_dim_one_core % num_dim_one_loop
    ele_one_loop_in = num_dim_one_loop*dim_ele_in
    ele_mod_in = num_dim_mod*dim_ele_in
    ele_one_loop_out = num_dim_one_loop * dim_ele_out
    ele_mod_out = num_dim_mod * dim_ele_out

    with tvm_ib.for_range(0, ub_loop, name="num_u") as num_u:
        data_offset = n_index * one_core_ele_in + num_u*ele_one_loop_in
        burst_len_data = _ceil_div(ele_one_loop_in, cp_align_len)
        tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                    data_ub.access_ptr("w",
                                                       offset=0),
                                    data.access_ptr('r',
                                                    offset=data_offset),
                                    0, 1, burst_len_data, 0, 0))

        with tvm_ib.for_range(0, num_dim_one_loop, name="num_d") as num_d:
            with tvm_ib.for_range(0, col_out, name="num_c") as num_c:
                args = tvm_ib, data_ub, data_res, reg, num_d, num_c,\
                       dim_ele_in, dim_ele_out, col_begin, row_begin,\
                       row_in, row_out
                _reg_mov_row(args)

        with tvm_ib.if_scope(ele_one_loop_out % cp_align_len > 0):
            with tvm_ib.if_scope(ele_one_loop_out >= cp_align_len):
                ele_one_loop_out_align = ele_one_loop_out - cp_align_len
                dst_offset = n_index * one_core_ele_out\
                             + num_u*ele_one_loop_out
                burst_len_dst = _ceil_div(ele_one_loop_out_align, cp_align_len)
                tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                            dst.access_ptr('w',
                                                           offset=dst_offset),
                                            data_res.access_ptr("r", offset=0),
                                            0, 1, burst_len_dst, 0, 0))
                args = tvm_ib, data_res, data_tail, reg,\
                       ele_one_loop_out_align, cp_align_len
                _reg_mov_align(args)

                tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                            dst.access_ptr(
                                                'w',
                                                offset=dst_offset
                                                + ele_one_loop_out_align),
                                            data_tail.access_ptr(
                                                "r", offset=0),
                                            0, 1, 1, 0, 0))
        with tvm_ib.else_scope():
            dst_offset = n_index * one_core_ele_out + num_u*ele_one_loop_out
            burst_len_dst = _ceil_div(ele_one_loop_out, cp_align_len)
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w',
                                                       offset=dst_offset),
                                        data_res.access_ptr("r", offset=0),
                                        0, 1, burst_len_dst, 0, 0))

    with tvm_ib.if_scope(num_dim_mod > 0):
        data_offset = n_index * one_core_ele_in + ub_loop * ele_one_loop_in
        burst_len_data = _ceil_div(ele_mod_in, cp_align_len)
        tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                    data_ub.access_ptr("w",
                                                       offset=0),
                                    data.access_ptr('r',
                                                    offset=data_offset),
                                    0, 1, burst_len_data, 0, 0))

        with tvm_ib.for_range(0, num_dim_mod, name="num_d") as num_d:
            with tvm_ib.for_range(0, col_out, name="num_c") as num_c:
                args = tvm_ib, data_ub, data_res, reg, num_d, num_c,\
                       dim_ele_in, dim_ele_out, col_begin, row_begin,\
                       row_in, row_out
                _reg_mov_row(args)

        with tvm_ib.if_scope(ele_mod_out % cp_align_len > 0):
            with tvm_ib.if_scope(ele_mod_out >= cp_align_len):
                ele_mod_out_align = ele_mod_out - cp_align_len
                dst_offset = n_index * one_core_ele_out\
                             + ub_loop * ele_one_loop_out
                burst_len_dst = _ceil_div(ele_mod_out_align, cp_align_len)
                tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                            dst.access_ptr('w',
                                                           offset=dst_offset),
                                            data_res.access_ptr("r", offset=0),
                                            0, 1, burst_len_dst, 0, 0))
                args = tvm_ib, data_res, data_tail, reg,\
                       ele_mod_out_align, cp_align_len
                _reg_mov_align(args)

                tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                            dst.access_ptr(
                                                'w',
                                                offset=dst_offset
                                                + ele_mod_out_align),
                                            data_tail.access_ptr(
                                                "r", offset=0),
                                            0, 1, 1, 0, 0))
        with tvm_ib.else_scope():
            dst_offset = n_index * one_core_ele_out\
                         + ub_loop * ele_one_loop_out
            burst_len_dst = _ceil_div(ele_mod_out, cp_align_len)
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w',
                                                       offset=dst_offset),
                                        data_res.access_ptr("r", offset=0),
                                        0, 1, burst_len_dst, 0, 0))


def _mov_scalar_one(dst, data, begin):
    """
    function of making ir node builder for mov scalar one scene

    """
    tvm_ib = tvm.ir_builder.create()

    device_core_num = AICORE_NUM
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - 32
    ub_ele = (ub_bytes // 2 // float_size // cp_align_len) * cp_align_len

    data_ub = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                         "data_ub", scope=cce.scope_ubuf)
    data_res = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                          "data_res", scope=cce.scope_ubuf)
    data_tail = _new_alloc(tvm_ib, dst.dtype, cp_align_len,
                           "data_tail", scope=cce.scope_ubuf)
    reg = tvm_ib.allocate(dst.dtype, (8,), name='reg', scope=cce.scope_reg)

    n_i = data.shape[0]
    _, _, col_begin, row_begin = begin

    group_index = n_i // device_core_num
    group_mod = n_i % device_core_num

    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    with tvm_ib.for_range(0, group_index, name="num_g") as num_g:
        args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg,\
               device_core_num, block_index, cp_align_len, col_begin,\
               row_begin, ub_ele, num_g
        _func_mov_scalar_one(args)
    with tvm_ib.if_scope(group_mod > 0):
        with tvm_ib.if_scope(block_index < group_mod):
            args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg,\
                   device_core_num, block_index, cp_align_len, col_begin,\
                   row_begin, ub_ele, group_index
            _func_mov_scalar_one(args)

    return tvm_ib.get()


def _reg_mov_only_col(args):
    """
    move from data_ub to data_res to change the C axis for split dim scene

    """
    tvm_ib, data_ub, data_res, reg, row_len_src, n_begin, col_len = args

    ele_reg = 8
    r_cycle = col_len // ele_reg
    r_mod = col_len - ele_reg*r_cycle
    reg_zero = 0
    reg_one = 1
    reg_two = 2
    reg_three = 3
    reg_four = 4
    reg_five = 5
    reg_six = 6
    reg_seven = 7

    with tvm_ib.for_range(0, r_cycle, name="num_cr") as num_cr:
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_zero]),
            data_ub.access_ptr('r',
                               offset=((num_cr*ele_reg + reg_zero)*row_len_src
                                       + n_begin))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_one]),
            data_ub.access_ptr('r',
                               offset=((num_cr*ele_reg + reg_one)*row_len_src
                                       + n_begin))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_two]),
            data_ub.access_ptr('r',
                               offset=((num_cr*ele_reg + reg_two)*row_len_src
                                       + n_begin))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_three]),
            data_ub.access_ptr('r',
                               offset=((num_cr*ele_reg + reg_three)*row_len_src
                                       + n_begin))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_four]),
            data_ub.access_ptr('r',
                               offset=((num_cr*ele_reg + reg_four)*row_len_src
                                       + n_begin))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_five]),
            data_ub.access_ptr('r',
                               offset=((num_cr*ele_reg + reg_five)*row_len_src
                                       + n_begin))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_six]),
            data_ub.access_ptr('r',
                               offset=((num_cr*ele_reg + reg_six)*row_len_src
                                       + n_begin))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_seven]),
            data_ub.access_ptr('r',
                               offset=((num_cr*ele_reg + reg_seven)*row_len_src
                                       + n_begin))
        ))

        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(num_cr*ele_reg + reg_zero)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_zero])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(num_cr*ele_reg + reg_one)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_one])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(num_cr*ele_reg + reg_two)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_two])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(num_cr*ele_reg + reg_three)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_three])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(num_cr*ele_reg + reg_four)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_four])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(num_cr*ele_reg + reg_five)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_five])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(num_cr*ele_reg + reg_six)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_six])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(num_cr*ele_reg + reg_seven)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_seven])
        ))

    with tvm_ib.if_scope(r_mod > 0):
        with tvm_ib.for_range(0, r_mod, name="num_er") as num_er:
            tvm_ib.emit(tvm.call_extern(
                data_ub.dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[0]),
                data_ub.access_ptr('r',
                                   offset=((r_cycle*ele_reg + num_er)
                                           * row_len_src + n_begin))
            ))
            tvm_ib.emit(tvm.call_extern(
                data_res.dtype, "reg_mov",
                data_res.access_ptr('w', offset=(r_cycle*ele_reg + num_er)),
                tvm.call_extern(reg.dtype, "reg", reg[0])
            ))


def _reg_mov_big_col(args):
    """
    move from data_ub to data_res to change the C axis for split dim scene

    """
    tvm_ib, data_ub, data_res, reg, row_len_src, row_len_dst,\
    n_begin, num_r, col_len = args

    ele_reg = 8
    r_cycle = col_len // ele_reg
    r_mod = col_len - ele_reg * r_cycle
    reg_zero = 0
    reg_one = 1
    reg_two = 2
    reg_three = 3
    reg_four = 4
    reg_five = 5
    reg_six = 6
    reg_seven = 7

    with tvm_ib.for_range(0, r_cycle, name="num_cr") as num_cr:
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_zero]),
            data_ub.access_ptr('r',
                               offset=((num_cr * ele_reg + reg_zero)
                                       * row_len_src
                                       + n_begin + num_r))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_one]),
            data_ub.access_ptr('r',
                               offset=((num_cr * ele_reg + reg_one)
                                       * row_len_src
                                       + n_begin + num_r))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_two]),
            data_ub.access_ptr('r',
                               offset=((num_cr * ele_reg + reg_two)
                                       * row_len_src
                                       + n_begin + num_r))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_three]),
            data_ub.access_ptr('r',
                               offset=((num_cr * ele_reg + reg_three)
                                       * row_len_src
                                       + n_begin + num_r))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_four]),
            data_ub.access_ptr('r',
                               offset=((num_cr * ele_reg + reg_four)
                                       * row_len_src
                                       + n_begin + num_r))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_five]),
            data_ub.access_ptr('r',
                               offset=((num_cr * ele_reg + reg_five)
                                       * row_len_src
                                       + n_begin + num_r))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_six]),
            data_ub.access_ptr('r',
                               offset=((num_cr * ele_reg + reg_six)
                                       * row_len_src
                                       + n_begin + num_r))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_seven]),
            data_ub.access_ptr('r',
                               offset=((num_cr * ele_reg + reg_seven)
                                       * row_len_src
                                       + n_begin + num_r))
        ))

        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=((num_cr * ele_reg + reg_zero)
                                             * row_len_dst + num_r)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_zero])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=((num_cr * ele_reg + reg_one)
                                             * row_len_dst + num_r)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_one])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=((num_cr * ele_reg + reg_two)
                                             * row_len_dst + num_r)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_two])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=((num_cr * ele_reg + reg_three)
                                             * row_len_dst + num_r)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_three])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=((num_cr * ele_reg + reg_four)
                                             * row_len_dst + num_r)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_four])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=((num_cr * ele_reg + reg_five)
                                             * row_len_dst + num_r)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_five])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=((num_cr * ele_reg + reg_six)
                                             * row_len_dst + num_r)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_six])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=((num_cr * ele_reg + reg_seven)
                                             * row_len_dst + num_r)),
            tvm.call_extern(reg.dtype, "reg", reg[reg_seven])
        ))

    with tvm_ib.if_scope(r_mod > 0):
        with tvm_ib.for_range(0, r_mod, name="num_er") as num_er:
            tvm_ib.emit(tvm.call_extern(
                data_ub.dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[0]),
                data_ub.access_ptr('r',
                                   offset=((r_cycle * ele_reg + num_er)
                                           * row_len_src
                                           + n_begin + num_r))
            ))
            tvm_ib.emit(tvm.call_extern(
                data_res.dtype, "reg_mov",
                data_res.access_ptr('w', offset=((r_cycle * ele_reg + num_er)
                                                 * row_len_dst + num_r)),
                tvm.call_extern(reg.dtype, "reg", reg[0])
            ))


def _reg_mov_big_row(args):
    """
    move from data_ub to data_res to change the C axis for split dim scene

    """
    tvm_ib, data_ub, data_res, reg, row_len_src, n_begin,\
    num_c, row_len_dst = args

    ele_reg = 8
    r_cycle = row_len_dst // ele_reg
    r_mod = row_len_dst - ele_reg * r_cycle
    reg_zero = 0
    reg_one = 1
    reg_two = 2
    reg_three = 3
    reg_four = 4
    reg_five = 5
    reg_six = 6
    reg_seven = 7

    with tvm_ib.for_range(0, r_cycle, name="num_cr") as num_cr:
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_zero]),
            data_ub.access_ptr('r',
                               offset=(num_c*row_len_src
                                       + n_begin
                                       + (num_cr * ele_reg + reg_zero)))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_one]),
            data_ub.access_ptr('r',
                               offset=(num_c*row_len_src
                                       + n_begin
                                       + (num_cr * ele_reg + reg_one)))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_two]),
            data_ub.access_ptr('r',
                               offset=(num_c*row_len_src
                                       + n_begin
                                       + (num_cr * ele_reg + reg_two)))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_three]),
            data_ub.access_ptr('r',
                               offset=(num_c*row_len_src
                                       + n_begin
                                       + (num_cr * ele_reg + reg_three)))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_four]),
            data_ub.access_ptr('r',
                               offset=(num_c*row_len_src
                                       + n_begin
                                       + (num_cr * ele_reg + reg_four)))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_five]),
            data_ub.access_ptr('r',
                               offset=(num_c*row_len_src
                                       + n_begin
                                       + (num_cr * ele_reg + reg_five)))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_six]),
            data_ub.access_ptr('r',
                               offset=(num_c*row_len_src
                                       + n_begin
                                       + (num_cr * ele_reg + reg_six)))
        ))
        tvm_ib.emit(tvm.call_extern(
            data_ub.dtype, "reg_mov",
            tvm.call_extern(reg.dtype, "reg", reg[reg_seven]),
            data_ub.access_ptr('r',
                               offset=(num_c*row_len_src
                                       + n_begin
                                       + (num_cr * ele_reg + reg_seven)))
        ))

        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(num_c * row_len_dst
                                             + (num_cr * ele_reg + reg_zero))),
            tvm.call_extern(reg.dtype, "reg", reg[reg_zero])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(num_c * row_len_dst
                                             + (num_cr * ele_reg + reg_one))),
            tvm.call_extern(reg.dtype, "reg", reg[reg_one])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(num_c * row_len_dst
                                             + (num_cr * ele_reg + reg_two))),
            tvm.call_extern(reg.dtype, "reg", reg[reg_two])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(num_c * row_len_dst
                                             + (num_cr * ele_reg
                                                + reg_three))),
            tvm.call_extern(reg.dtype, "reg", reg[reg_three])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(num_c * row_len_dst
                                             + (num_cr * ele_reg
                                                + reg_four))),
            tvm.call_extern(reg.dtype, "reg", reg[reg_four])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(num_c * row_len_dst
                                             + (num_cr * ele_reg
                                                + reg_five))),
            tvm.call_extern(reg.dtype, "reg", reg[reg_five])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(num_c * row_len_dst
                                             + (num_cr * ele_reg
                                                + reg_six))),
            tvm.call_extern(reg.dtype, "reg", reg[reg_six])
        ))
        tvm_ib.emit(tvm.call_extern(
            data_res.dtype, "reg_mov",
            data_res.access_ptr('w', offset=(num_c * row_len_dst
                                             + (num_cr * ele_reg
                                                + reg_seven))),
            tvm.call_extern(reg.dtype, "reg", reg[reg_seven])
        ))

    with tvm_ib.if_scope(r_mod > 0):
        with tvm_ib.for_range(0, r_mod, name="num_er") as num_er:
            tvm_ib.emit(tvm.call_extern(
                data_ub.dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[0]),
                data_ub.access_ptr('r',
                                   offset=(num_c * row_len_src
                                           + n_begin
                                           + (r_cycle * ele_reg + num_er)))
            ))
            tvm_ib.emit(tvm.call_extern(
                data_res.dtype, "reg_mov",
                data_res.access_ptr('w', offset=(num_c * row_len_dst
                                                 + (r_cycle * ele_reg
                                                    + num_er))),
                tvm.call_extern(reg.dtype, "reg", reg[0])
            ))


def _func_one_core_ir(args):
    """
    function of moving data for one core

    """
    tvm_ib, data, dst, data_ub, data_res, data_tail, reg,\
    device_core_num, block_index, src_shape_ele, dst_shape_ele,\
    cp_align_len, n_begin, num_g = args

    _, col_len, row_len_src = data.shape
    row_len_dst = dst.shape[-1]
    n_index = num_g * device_core_num + block_index

    data_offset = n_index * src_shape_ele
    burst_len_data = _ceil_div(src_shape_ele, cp_align_len)
    tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                data_ub.access_ptr("w",
                                                   offset=0),
                                data.access_ptr('r',
                                                offset=data_offset),
                                0, 1, burst_len_data, 0, 0))

    with tvm_ib.if_scope(row_len_dst == 1):
        args = tvm_ib, data_ub, data_res, reg, row_len_src, n_begin, col_len
        _reg_mov_only_col(args)
    with tvm_ib.else_scope():
        with tvm_ib.if_scope(tvm.all(row_len_dst < 8, col_len >= 8)):
            with tvm_ib.for_range(0, row_len_dst, name="num_r") as num_r:
                args = tvm_ib, data_ub, data_res, reg, row_len_src,\
                       row_len_dst, n_begin, num_r, col_len
                _reg_mov_big_col(args)
        with tvm_ib.else_scope():
            with tvm_ib.for_range(0, col_len, name="num_c") as num_c:
                args = tvm_ib, data_ub, data_res, reg, row_len_src,\
                       n_begin, num_c, row_len_dst
                _reg_mov_big_row(args)

    dst_offset = n_index * dst_shape_ele
    with tvm_ib.if_scope(dst_shape_ele % cp_align_len == 0):
        burst_len_dst = dst_shape_ele // cp_align_len
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w', offset=dst_offset),
                                    data_res.access_ptr("r", offset=0),
                                    0, 1, burst_len_dst, 0, 0))
    with tvm_ib.else_scope():
        move_len_align = dst_shape_ele - cp_align_len
        burst_len_dst_a = _ceil_div(move_len_align, cp_align_len)
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w', offset=dst_offset),
                                    data_res.access_ptr("r", offset=0),
                                    0, 1, burst_len_dst_a, 0, 0))
        with tvm_ib.for_range(0, cp_align_len, name="num_a") as num_a:
            tvm_ib.emit(tvm.call_extern(
                data_res.dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[0]),
                data_res.access_ptr('r', offset=(move_len_align + num_a))
            ))
            tvm_ib.emit(tvm.call_extern(
                data_tail.dtype, "reg_mov",
                data_tail.access_ptr('w', offset=num_a),
                tvm.call_extern(reg.dtype, "reg", reg[0])
            ))
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr(
                                        'w',
                                        offset=dst_offset + move_len_align),
                                    data_tail.access_ptr("r", offset=0),
                                    0, 1, 1, 0, 0))


def _one_core_ir(dst, data, n_begin):
    """
    function of making ir node builder for one core scene

    """
    tvm_ib = tvm.ir_builder.create()
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    device_core_num = AICORE_NUM
    ub_bytes = UB_SIZE_B - 64
    ub_ele = (ub_bytes // 2 // float_size // cp_align_len) * cp_align_len

    src_shape_ele = functools_reduce(lambda x, y: x * y, data.shape[1:])
    dst_shape_ele = functools_reduce(lambda x, y: x * y, dst.shape[1:])

    data_ub = _new_alloc(tvm_ib, data.dtype, ub_ele,
                         "data_ub", scope=cce.scope_ubuf)
    data_res = _new_alloc(tvm_ib, data.dtype, ub_ele,
                          "data_res", scope=cce.scope_ubuf)
    data_tail = _new_alloc(tvm_ib, data.dtype, cp_align_len,
                           "data_tail", scope=cce.scope_ubuf)
    reg = tvm_ib.allocate(dst.dtype, (8,), name='reg', scope=cce.scope_reg)

    n_i = data.shape[0]
    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    group_index = n_i // device_core_num
    group_mod = n_i % device_core_num

    with tvm_ib.for_range(0, group_index, name="num_g") as num_g:
        args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg,\
               device_core_num, block_index, src_shape_ele, dst_shape_ele,\
               cp_align_len, n_begin, num_g
        _func_one_core_ir(args)
    with tvm_ib.if_scope(group_mod > 0):
        with tvm_ib.if_scope(block_index < group_mod):
            args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
                   device_core_num, block_index, src_shape_ele, dst_shape_ele, \
                   cp_align_len, n_begin, group_index
            _func_one_core_ir(args)

    return tvm_ib.get()


def _set_mask_slice(before, after):
    """
    calculate MASK in cce

    """
    before = int(before)
    after = int(after)
    mask1 = 0
    mask2 = (2**(64 - after) - 1) - (2**before - 1)
    return mask1, mask2


def _set_mask_slice_fp16(before, after):
    """
    calculate MASK in cce

    """
    before = int(before)
    after = int(after)

    if after >= 64:
        mask1 = 0
        after_new = after - 64
        mask2 = (2**(64 - after_new) - 1) - (2**before - 1)
    elif before >= 64:
        before_new = before - 64
        mask1 = (2**(64 - after) - 1) - (2**before_new - 1)
        mask2 = 0
    else:
        mask1 = (2**(64 - after) - 1)
        mask2 = (2**64 - 1) - (2**before - 1)

    return mask1, mask2


def _func_vadds(args):
    """
    function of moving data with vadds function

    """
    tvm_ib, data_ub, data_res, ub_offset, res_offset,\
    repeat, srcm0, dstm0, srcm1, dstm1, cp_align_len = args
    max_r = 255

    with tvm_ib.if_scope(repeat <= max_r):
        with tvm_ib.if_scope(repeat == 1):
            tvm_ib.emit(tvm.call_extern(data_res.dtype, "vadds",
                                        data_res.access_ptr("w",
                                                            offset=res_offset),
                                        data_ub.access_ptr('r',
                                                           offset=ub_offset),
                                        0, repeat, dstm0, srcm0, 0, 0))
        with tvm_ib.else_scope():
            tvm_ib.emit(tvm.call_extern(data_res.dtype, "vadds",
                                        data_res.access_ptr("w",
                                                            offset=res_offset),
                                        data_ub.access_ptr('r',
                                                           offset=ub_offset),
                                        0, repeat, dstm0, srcm0, dstm1, srcm1))
    with tvm_ib.else_scope():
        zu_repeat = repeat // max_r
        mod_repeat = repeat % max_r
        with tvm_ib.for_range(0, zu_repeat, name="num_zr") as num_zr:
            ub_offset_cur = ub_offset + num_zr*max_r*srcm1*cp_align_len
            res_offset_cur = res_offset + num_zr*max_r*dstm1*cp_align_len
            tvm_ib.emit(
                tvm.call_extern(
                    data_res.dtype, "vadds",
                    data_res.access_ptr("w", offset=res_offset_cur),
                    data_ub.access_ptr('r', offset=ub_offset_cur),
                    0, max_r, dstm0, srcm0, dstm1, srcm1))
        with tvm_ib.if_scope(mod_repeat > 0):
            ub_offset_cur = ub_offset + zu_repeat*max_r*srcm1*cp_align_len
            res_offset_cur = res_offset + zu_repeat*max_r*dstm1*cp_align_len
            with tvm_ib.if_scope(mod_repeat == 1):
                tvm_ib.emit(
                    tvm.call_extern(
                        data_res.dtype, "vadds",
                        data_res.access_ptr("w", offset=res_offset_cur),
                        data_ub.access_ptr('r', offset=ub_offset_cur),
                        0, mod_repeat, dstm0, srcm0, 0, 0))
            with tvm_ib.else_scope():
                tvm_ib.emit(
                    tvm.call_extern(
                        data_res.dtype, "vadds",
                        data_res.access_ptr("w", offset=res_offset_cur),
                        data_ub.access_ptr('r', offset=ub_offset_cur),
                        0, mod_repeat, dstm0, srcm0, dstm1, srcm1))


def _func_sp_vadds_rowzu_mul(args):
    """
    function of moving data for multi row zu

    """
    tvm_ib, data, dst, data_ub, data_res, data_tail, reg, device_core_num,\
    block_index, cp_align_len, ub_ele_half, n_begin, row_zu,\
    before, after, num_g = args

    col_len_shape = data.shape[1]
    row_in = data.shape[2]
    row_out = dst.shape[2]

    dim_ele_in = col_len_shape*row_in
    dim_ele_out = col_len_shape*row_out

    n_index = num_g*device_core_num + block_index
    row_out_align = _ceil_fill(row_out, cp_align_len) + cp_align_len
    row_out_align_block = row_out_align // cp_align_len

    row_out_align_ub = (ub_ele_half // row_out_align // row_zu)*row_zu
    ub_loop = col_len_shape // row_out_align_ub
    ub_mod = col_len_shape % row_out_align_ub
    ub_loop_ele_in = row_out_align_ub*row_in
    ub_loop_ele_out = row_out_align_ub*row_out
    ub_mod_ele_out = ub_mod*row_out

    with tvm_ib.for_range(0, ub_loop, name="num_u") as num_u:
        col_group = row_out_align_ub // row_zu
        for num_rz in range(row_zu):
            gap = (num_rz * row_out) % cp_align_len
            data_offset = n_index * dim_ele_in + num_u * ub_loop_ele_in\
                          + num_rz * row_in - gap + n_begin
            ub_offset = 0
            burst_len_data = _ceil_div(row_out + gap, cp_align_len)
            src_stride = _ceil_div(row_in * row_zu,
                                   cp_align_len) - burst_len_data
            dst_stride = 1 * row_out_align_block - burst_len_data
            args = tvm_ib, data, data_ub[num_rz], data_offset, ub_offset,\
                   col_group, burst_len_data, src_stride,\
                   dst_stride, cp_align_len
            _func_gm_to_ub_align(args)

            with tvm_ib.if_scope(row_out < 64):
                gap = (num_rz * row_out) % cp_align_len
                mask1, mask2 = _set_mask_slice(before[num_rz], after[num_rz])
                tvm_ib.emit(tvm.call_extern(
                    dst.dtype, "set_vector_mask",
                    tvm.const(mask1, dtype="uint64"),
                    tvm.const(mask2, dtype="uint64")))
                ub_offset = 0
                res_offset = num_rz * row_out - gap
                repeat = col_group
                srcm0 = 1
                dstm0 = 1
                srcm1 = row_out_align_block * 1
                dstm1 = row_out * row_zu // cp_align_len
                args = tvm_ib, data_ub[num_rz], data_res,\
                       ub_offset, res_offset,\
                       repeat, srcm0, dstm0, srcm1, dstm1, cp_align_len
                _func_vadds(args)
            with tvm_ib.else_scope():
                gap = (num_rz * row_out) % cp_align_len
                mask1, mask2 = _set_mask_slice(before[num_rz * 2],
                                               after[num_rz * 2])
                tvm_ib.emit(tvm.call_extern(
                    dst.dtype, "set_vector_mask",
                    tvm.const(mask1, dtype="uint64"),
                    tvm.const(mask2, dtype="uint64")))
                ub_offset = 0
                res_offset = num_rz * row_out - gap
                repeat = col_group
                srcm0 = 1
                dstm0 = 1
                srcm1 = row_out_align_block * 1
                dstm1 = row_out * row_zu // cp_align_len
                args = tvm_ib, data_ub[num_rz], data_res,\
                       ub_offset, res_offset,\
                       repeat, srcm0, dstm0, srcm1, dstm1, cp_align_len
                _func_vadds(args)

                mask1, mask2 = _set_mask_slice(before[num_rz * 2 + 1],
                                               after[num_rz * 2 + 1])
                tvm_ib.emit(tvm.call_extern(
                    dst.dtype, "set_vector_mask",
                    tvm.const(mask1, dtype="uint64"),
                    tvm.const(mask2, dtype="uint64")))
                ub_offset = 64
                res_offset = num_rz * row_out - gap + 64
                repeat = col_group
                srcm0 = 1
                dstm0 = 1
                srcm1 = row_out_align_block * 1
                dstm1 = row_out * row_zu // cp_align_len
                args = tvm_ib, data_ub[num_rz], data_res,\
                       ub_offset, res_offset,\
                       repeat, srcm0, dstm0, srcm1, dstm1, cp_align_len
                _func_vadds(args)

        tvm_ib.emit(tvm.call_extern(
            dst.dtype, "set_vector_mask",
            tvm.const(-1, dtype="uint64"),
            tvm.const(-1, dtype="uint64")))

        dst_offset = n_index*dim_ele_out + num_u*ub_loop_ele_out
        with tvm_ib.if_scope(ub_loop_ele_out % cp_align_len > 0):
            ub_loop_ele_out_align = ub_loop_ele_out - cp_align_len
            burst_len_dst = _ceil_div(ub_loop_ele_out_align, cp_align_len)
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w',
                                                       offset=dst_offset),
                                        data_res.access_ptr("r", offset=0),
                                        0, 1, burst_len_dst, 0, 0))
            with tvm_ib.for_range(0, cp_align_len, name="num_a") as num_a:
                tvm_ib.emit(tvm.call_extern(
                    data_res.dtype, "reg_mov",
                    tvm.call_extern(reg.dtype, "reg", reg[0]),
                    data_res.access_ptr('r',
                                        offset=ub_loop_ele_out_align + num_a)
                ))
                tvm_ib.emit(tvm.call_extern(
                    data_tail.dtype, "reg_mov",
                    data_tail.access_ptr('w', offset=num_a),
                    tvm.call_extern(reg.dtype, "reg", reg[0])
                ))

            tvm_ib.emit(
                tvm.call_extern(
                    dst.dtype, "copy_ubuf_to_gm",
                    dst.access_ptr(
                        'w', offset=dst_offset + ub_loop_ele_out_align),
                    data_tail.access_ptr("r", offset=0),
                    0, 1, 1, 0, 0))
        with tvm_ib.else_scope():
            burst_len_dst = _ceil_div(ub_loop_ele_out, cp_align_len)
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w',
                                                       offset=dst_offset),
                                        data_res.access_ptr("r", offset=0),
                                        0, 1, burst_len_dst, 0, 0))

    with tvm_ib.if_scope(ub_mod > 0):
        col_group = ub_mod // row_zu
        col_mod = ub_mod % row_zu

        for num_rz in range(row_zu):
            with tvm_ib.if_scope(num_rz < ub_mod):
                gap = (num_rz * row_out) % cp_align_len
                data_offset = n_index * dim_ele_in + ub_loop * ub_loop_ele_in\
                              + num_rz * row_in - gap + n_begin
                ub_offset = 0
                burst_len_data = _ceil_div(row_out + gap, cp_align_len)
                src_stride = _ceil_div(row_in * row_zu,
                                       cp_align_len) - burst_len_data
                dst_stride = 1 * row_out_align_block - burst_len_data
                n_burst = tvm.select(num_rz < col_mod, col_group + 1,
                                     col_group)
                args = tvm_ib, data, data_ub[num_rz], data_offset, ub_offset, \
                       n_burst, burst_len_data, src_stride, \
                       dst_stride, cp_align_len
                _func_gm_to_ub_align(args)

                with tvm_ib.if_scope(row_out < 64):
                    gap = (num_rz * row_out) % cp_align_len
                    mask1, mask2 = _set_mask_slice(before[num_rz],
                                                   after[num_rz])
                    tvm_ib.emit(tvm.call_extern(
                        dst.dtype, "set_vector_mask",
                        tvm.const(mask1, dtype="uint64"),
                        tvm.const(mask2, dtype="uint64")))
                    ub_offset = 0
                    res_offset = num_rz * row_out - gap
                    repeat = tvm.select(num_rz < col_mod, col_group + 1,
                                        col_group)
                    srcm0 = 1
                    dstm0 = 1
                    srcm1 = row_out_align_block * 1
                    dstm1 = row_out * row_zu // cp_align_len
                    args = tvm_ib, data_ub[
                        num_rz], data_res, ub_offset, res_offset, \
                           repeat, srcm0, dstm0, srcm1, dstm1, cp_align_len
                    _func_vadds(args)
                with tvm_ib.else_scope():
                    gap = (num_rz * row_out) % cp_align_len
                    mask1, mask2 = _set_mask_slice(before[num_rz * 2],
                                                   after[num_rz * 2])
                    tvm_ib.emit(tvm.call_extern(
                        dst.dtype, "set_vector_mask",
                        tvm.const(mask1, dtype="uint64"),
                        tvm.const(mask2, dtype="uint64")))
                    ub_offset = 0
                    res_offset = num_rz * row_out - gap
                    repeat = tvm.select(num_rz < col_mod, col_group + 1,
                                        col_group)
                    srcm0 = 1
                    dstm0 = 1
                    srcm1 = row_out_align_block * 1
                    dstm1 = row_out * row_zu // cp_align_len
                    args = tvm_ib, data_ub[num_rz], data_res,\
                           ub_offset, res_offset,\
                           repeat, srcm0, dstm0, srcm1, dstm1, cp_align_len
                    _func_vadds(args)

                    mask1, mask2 = _set_mask_slice(before[num_rz * 2 + 1],
                                                   after[num_rz * 2 + 1])
                    tvm_ib.emit(tvm.call_extern(
                        dst.dtype, "set_vector_mask",
                        tvm.const(mask1, dtype="uint64"),
                        tvm.const(mask2, dtype="uint64")))
                    ub_offset = 64
                    res_offset = num_rz * row_out - gap + 64
                    repeat = tvm.select(num_rz < col_mod, col_group + 1,
                                        col_group)
                    srcm0 = 1
                    dstm0 = 1
                    srcm1 = row_out_align_block * 1
                    dstm1 = row_out * row_zu // cp_align_len
                    args = tvm_ib, data_ub[num_rz], data_res,\
                           ub_offset, res_offset,\
                           repeat, srcm0, dstm0, srcm1, dstm1, cp_align_len
                    _func_vadds(args)

        tvm_ib.emit(tvm.call_extern(
            dst.dtype, "set_vector_mask",
            tvm.const(-1, dtype="uint64"),
            tvm.const(-1, dtype="uint64")))

        dst_offset = n_index * dim_ele_out + ub_loop * ub_loop_ele_out
        with tvm_ib.if_scope(ub_mod_ele_out % cp_align_len > 0):
            ub_mod_ele_out_align = ub_mod_ele_out - cp_align_len
            burst_len_dst = _ceil_div(ub_mod_ele_out_align, cp_align_len)
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w',
                                                       offset=dst_offset),
                                        data_res.access_ptr("r", offset=0),
                                        0, 1, burst_len_dst, 0, 0))
            with tvm_ib.for_range(0, cp_align_len, name="num_a") as num_a:
                tvm_ib.emit(tvm.call_extern(
                    data_res.dtype, "reg_mov",
                    tvm.call_extern(reg.dtype, "reg", reg[0]),
                    data_res.access_ptr('r',
                                        offset=ub_mod_ele_out_align + num_a)
                ))
                tvm_ib.emit(tvm.call_extern(
                    data_tail.dtype, "reg_mov",
                    data_tail.access_ptr('w',
                                         offset=num_a),
                    tvm.call_extern(reg.dtype, "reg", reg[0])
                ))

            tvm_ib.emit(
                tvm.call_extern(
                    dst.dtype, "copy_ubuf_to_gm",
                    dst.access_ptr(
                        'w', offset=dst_offset + ub_mod_ele_out_align),
                    data_tail.access_ptr("r", offset=0),
                    0, 1, 1, 0, 0))

        with tvm_ib.else_scope():
            burst_len_dst = _ceil_div(ub_mod_ele_out, cp_align_len)
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w',
                                                       offset=dst_offset),
                                        data_res.access_ptr("r", offset=0),
                                        0, 1, burst_len_dst, 0, 0))


def _move_sp_vadds_rowzu_8(dst, data, n_begin, row_zu, before, after):
    """
    function of moving data for row zu 8

    """
    tvm_ib = tvm.ir_builder.create()
    device_core_num = AICORE_NUM
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - 64
    ub_ele = (ub_bytes // float_size // cp_align_len) * cp_align_len
    ub_ele_half = ub_ele // 2
    ub_ele_four = ub_ele // 16

    data_ub_one = _new_alloc(tvm_ib, dst.dtype, ub_ele_four,
                             "data_ub_one", scope=cce.scope_ubuf)
    data_ub_two = _new_alloc(tvm_ib, dst.dtype, ub_ele_four,
                             "data_ub_two", scope=cce.scope_ubuf)
    data_ub_three = _new_alloc(tvm_ib, dst.dtype, ub_ele_four,
                               "data_ub_three", scope=cce.scope_ubuf)
    data_ub_four = _new_alloc(tvm_ib, dst.dtype, ub_ele_four,
                              "data_ub_four", scope=cce.scope_ubuf)
    data_ub_five = _new_alloc(tvm_ib, dst.dtype, ub_ele_four,
                              "data_ub_five", scope=cce.scope_ubuf)
    data_ub_six = _new_alloc(tvm_ib, dst.dtype, ub_ele_four,
                             "data_ub_six", scope=cce.scope_ubuf)
    data_ub_seven = _new_alloc(tvm_ib, dst.dtype, ub_ele_four,
                               "data_ub_seven", scope=cce.scope_ubuf)
    data_ub_eight = _new_alloc(tvm_ib, dst.dtype, ub_ele_four,
                               "data_ub_eight", scope=cce.scope_ubuf)
    data_res = _new_alloc(tvm_ib, dst.dtype, ub_ele_half,
                          "data_res", scope=cce.scope_ubuf)
    data_tail = _new_alloc(tvm_ib, dst.dtype, cp_align_len,
                           "data_tail", scope=cce.scope_ubuf)
    reg = tvm_ib.allocate(dst.dtype, (8,), name='reg', scope=cce.scope_reg)

    data_ub = [data_ub_one, data_ub_two, data_ub_three, data_ub_four,
               data_ub_five, data_ub_six, data_ub_seven, data_ub_eight]

    n_i = data.shape[0]
    group_index = n_i // device_core_num
    group_mod = n_i % device_core_num
    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    with tvm_ib.for_range(0, group_index, name="num_g") as num_g:
        args = tvm_ib, data, dst, data_ub, data_res, data_tail,\
               reg, device_core_num, block_index, cp_align_len, ub_ele_half,\
               n_begin, row_zu, before, after, num_g
        _func_sp_vadds_rowzu_mul(args)
    with tvm_ib.if_scope(group_mod > 0):
        with tvm_ib.if_scope(block_index < group_mod):
            args = tvm_ib, data, dst, data_ub, data_res, data_tail,\
                   reg, device_core_num, block_index, cp_align_len,\
                   ub_ele_half, n_begin, row_zu, before, after, group_index
            _func_sp_vadds_rowzu_mul(args)

    return tvm_ib.get()


def _move_sp_vadds_rowzu_4(dst, data, n_begin, row_zu, before, after):
    """
    function of moving data for row zu 4

    """
    tvm_ib = tvm.ir_builder.create()
    device_core_num = AICORE_NUM
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - 64
    ub_ele = (ub_bytes // float_size // cp_align_len) * cp_align_len
    ub_ele_half = ub_ele // 2
    ub_ele_four = ub_ele // 8

    data_ub_one = _new_alloc(tvm_ib, dst.dtype, ub_ele_four,
                             "data_ub_one", scope=cce.scope_ubuf)
    data_ub_two = _new_alloc(tvm_ib, dst.dtype, ub_ele_four,
                             "data_ub_two", scope=cce.scope_ubuf)
    data_ub_three = _new_alloc(tvm_ib, dst.dtype, ub_ele_four,
                               "data_ub_three", scope=cce.scope_ubuf)
    data_ub_four = _new_alloc(tvm_ib, dst.dtype, ub_ele_four,
                              "data_ub_four", scope=cce.scope_ubuf)
    data_res = _new_alloc(tvm_ib, dst.dtype, ub_ele_half,
                          "data_res", scope=cce.scope_ubuf)
    data_tail = _new_alloc(tvm_ib, dst.dtype, cp_align_len,
                           "data_tail", scope=cce.scope_ubuf)
    reg = tvm_ib.allocate(dst.dtype, (8,), name='reg', scope=cce.scope_reg)

    data_ub = [data_ub_one, data_ub_two, data_ub_three, data_ub_four]

    n_i = data.shape[0]

    group_index = n_i // device_core_num
    group_mod = n_i % device_core_num
    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    with tvm_ib.for_range(0, group_index, name="num_g") as num_g:
        args = tvm_ib, data, dst, data_ub, data_res, data_tail,\
               reg, device_core_num, block_index, cp_align_len, ub_ele_half,\
               n_begin, row_zu, before, after, num_g
        _func_sp_vadds_rowzu_mul(args)
    with tvm_ib.if_scope(group_mod > 0):
        with tvm_ib.if_scope(block_index < group_mod):
            args = tvm_ib, data, dst, data_ub, data_res, data_tail,\
                   reg, device_core_num, block_index, cp_align_len,\
                   ub_ele_half, n_begin, row_zu, before, after, group_index
            _func_sp_vadds_rowzu_mul(args)

    return tvm_ib.get()


def _move_sp_vadds_rowzu_2(dst, data, n_begin, row_zu, before, after):
    """
    function of moving data for row zu 2

    """
    tvm_ib = tvm.ir_builder.create()
    device_core_num = AICORE_NUM
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - 64
    ub_ele = (ub_bytes // float_size // cp_align_len) * cp_align_len
    ub_ele_half = ub_ele // 2
    ub_ele_four = ub_ele // 4

    data_ub_one = _new_alloc(tvm_ib, dst.dtype, ub_ele_four,
                             "data_ub_one", scope=cce.scope_ubuf)
    data_ub_two = _new_alloc(tvm_ib, dst.dtype, ub_ele_four,
                             "data_ub_two", scope=cce.scope_ubuf)
    data_res = _new_alloc(tvm_ib, dst.dtype, ub_ele_half,
                          "data_res", scope=cce.scope_ubuf)
    data_tail = _new_alloc(tvm_ib, dst.dtype, cp_align_len,
                           "data_tail", scope=cce.scope_ubuf)
    reg = tvm_ib.allocate(dst.dtype, (8,), name='reg', scope=cce.scope_reg)

    data_ub = [data_ub_one, data_ub_two]

    n_i = data.shape[0]

    group_index = n_i // device_core_num
    group_mod = n_i % device_core_num
    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    with tvm_ib.for_range(0, group_index, name="num_g") as num_g:
        args = tvm_ib, data, dst, data_ub, data_res, data_tail,\
               reg, device_core_num, block_index, cp_align_len, ub_ele_half,\
               n_begin, row_zu, before, after, num_g
        _func_sp_vadds_rowzu_mul(args)
    with tvm_ib.if_scope(group_mod > 0):
        with tvm_ib.if_scope(block_index < group_mod):
            args = tvm_ib, data, dst, data_ub, data_res, data_tail,\
                   reg, device_core_num, block_index, cp_align_len,\
                   ub_ele_half, n_begin, row_zu, before, after, group_index
            _func_sp_vadds_rowzu_mul(args)

    return tvm_ib.get()


def _func_21_20_fp32(args):
    """
    function of moving data for 21 20 float32 scene

    """
    tvm_ib, data, dst, data_ub, data_res, data_tail, reg, device_core_num,\
    block_index, n_begin, ub_ele_eight, cp_align_len, num_g = args

    _, col_len, row_in = data.shape
    row_out = dst.shape[2]
    row_zu_in = 8
    row_out_block = _ceil_div(row_out, cp_align_len)
    row_out_align = _ceil_fill(row_out, cp_align_len)
    num_col_ub = ub_ele_eight // row_out_align
    num_ub_loop = col_len // 8 // num_col_ub
    num_ub_mod = col_len % (8 * num_col_ub)
    n_index = num_g * device_core_num + block_index

    with tvm_ib.for_range(0, num_ub_loop, name="num_u") as num_u:
        for num_zi in range(row_zu_in):
            data_offset = n_index * col_len * row_in\
                          + num_u * 8 * num_col_ub * row_in\
                          + num_zi * row_in + n_begin - 4 * (num_zi % 2)
            ub_offset = 0
            n_burst = num_col_ub
            burst_len_data = row_out_block
            src_stride = (8 * row_in) // cp_align_len - burst_len_data
            dst_stride = 0
            args = tvm_ib, data, data_ub[num_zi], data_offset, ub_offset,\
                   n_burst, burst_len_data, src_stride,\
                   dst_stride, cp_align_len
            _func_gm_to_ub_align(args)

            mask1, mask2 = _set_mask_slice(0 + 4 * (num_zi % 2),
                                           64 - row_out - 4 * (num_zi % 2))
            tvm_ib.emit(tvm.call_extern(
                dst.dtype, "set_vector_mask",
                tvm.const(mask1, dtype="uint64"),
                tvm.const(mask2, dtype="uint64")))
            ub_offset = 0
            res_offset = num_zi * row_out - 4 * (num_zi % 2)
            repeat = num_col_ub
            srcm0 = 1
            dstm0 = 1
            srcm1 = row_out_block
            dstm1 = row_out * row_zu_in // cp_align_len
            args = tvm_ib, data_ub[num_zi], data_res, \
                   ub_offset, res_offset, \
                   repeat, srcm0, dstm0, srcm1, dstm1, cp_align_len
            _func_vadds(args)

            tvm_ib.emit(tvm.call_extern(
                dst.dtype, "set_vector_mask",
                tvm.const(-1, dtype="uint64"),
                tvm.const(-1, dtype="uint64")))

        dst_offset = n_index * col_len * row_out\
                     + num_u * 8 * num_col_ub * row_out
        burst_len_dst = (8 * num_col_ub * row_out) // cp_align_len
        tvm_ib.emit(
            tvm.call_extern(
                dst.dtype, "copy_ubuf_to_gm",
                dst.access_ptr('w', offset=dst_offset),
                data_res.access_ptr("r", offset=0),
                0, 1, burst_len_dst, 0, 0))

    with tvm_ib.if_scope(num_ub_mod > 0):
        col_group = num_ub_mod // row_zu_in
        col_mod = num_ub_mod % row_zu_in

        for num_zi in range(row_zu_in):
            with tvm_ib.if_scope(num_zi < num_ub_mod):
                data_offset = n_index * col_len * row_in \
                              + num_ub_loop * 8 * num_col_ub * row_in \
                              + num_zi * row_in + n_begin - 4 * (num_zi % 2)
                ub_offset = 0
                n_burst = tvm.select(num_zi < col_mod, col_group + 1,
                                     col_group)
                burst_len_data = row_out_block
                src_stride = (8 * row_in) // cp_align_len - burst_len_data
                dst_stride = 0
                args = tvm_ib, data, data_ub[num_zi], data_offset, ub_offset,\
                       n_burst, burst_len_data, src_stride, \
                       dst_stride, cp_align_len
                _func_gm_to_ub_align(args)

                with tvm_ib.if_scope(num_zi % 2 == 0):
                    mask1, mask2 = _set_mask_slice(0, 64 - row_out)
                    tvm_ib.emit(tvm.call_extern(
                        dst.dtype, "set_vector_mask",
                        tvm.const(mask1, dtype="uint64"),
                        tvm.const(mask2, dtype="uint64")))
                    ub_offset = 0
                    res_offset = num_zi * row_out
                    repeat = tvm.select(num_zi < col_mod, col_group + 1,
                                        col_group)
                    srcm0 = 1
                    dstm0 = 1
                    srcm1 = row_out_block
                    dstm1 = row_out * row_zu_in // cp_align_len
                    args = tvm_ib, data_ub[num_zi], data_res, \
                           ub_offset, res_offset, \
                           repeat, srcm0, dstm0, srcm1, dstm1, cp_align_len
                    _func_vadds(args)
                with tvm_ib.if_scope(num_zi % 2 == 1):
                    mask1, mask2 = _set_mask_slice(4, 64 - row_out - 4)
                    tvm_ib.emit(tvm.call_extern(
                        dst.dtype, "set_vector_mask",
                        tvm.const(mask1, dtype="uint64"),
                        tvm.const(mask2, dtype="uint64")))
                    ub_offset = 0
                    res_offset = num_zi * row_out - 4
                    repeat = tvm.select(num_zi < col_mod, col_group + 1,
                                        col_group)
                    srcm0 = 1
                    dstm0 = 1
                    srcm1 = row_out_block
                    dstm1 = row_out * row_zu_in // cp_align_len
                    args = tvm_ib, data_ub[num_zi], data_res, \
                           ub_offset, res_offset, \
                           repeat, srcm0, dstm0, srcm1, dstm1, cp_align_len
                    _func_vadds(args)

                tvm_ib.emit(tvm.call_extern(
                    dst.dtype, "set_vector_mask",
                    tvm.const(-1, dtype="uint64"),
                    tvm.const(-1, dtype="uint64")))

        dst_offset = n_index * col_len * row_out \
                     + num_ub_loop * 8 * num_col_ub * row_out
        mod_len = num_ub_mod * row_out
        with tvm_ib.if_scope(mod_len % cp_align_len == 0):
            burst_len_dst = mod_len // cp_align_len
            tvm_ib.emit(
                tvm.call_extern(
                    dst.dtype, "copy_ubuf_to_gm",
                    dst.access_ptr('w', offset=dst_offset),
                    data_res.access_ptr("r", offset=0),
                    0, 1, burst_len_dst, 0, 0))
        with tvm_ib.else_scope():
            mod_len_align = mod_len - cp_align_len
            burst_len_dst_a = _ceil_div(mod_len_align, cp_align_len)
            tvm_ib.emit(
                tvm.call_extern(
                    dst.dtype, "copy_ubuf_to_gm",
                    dst.access_ptr('w', offset=dst_offset),
                    data_res.access_ptr("r", offset=0),
                    0, 1, burst_len_dst_a, 0, 0))
            with tvm_ib.for_range(0, cp_align_len, name="num_a") as num_a:
                tvm_ib.emit(tvm.call_extern(
                    data_res.dtype, "reg_mov",
                    tvm.call_extern(reg.dtype, "reg", reg[0]),
                    data_res.access_ptr('r',
                                        offset=mod_len_align + num_a)
                ))
                tvm_ib.emit(tvm.call_extern(
                    data_tail.dtype, "reg_mov",
                    data_tail.access_ptr('w',
                                         offset=num_a),
                    tvm.call_extern(reg.dtype, "reg", reg[0])
                ))
            tvm_ib.emit(
                tvm.call_extern(
                    dst.dtype, "copy_ubuf_to_gm",
                    dst.access_ptr('w', offset=dst_offset + mod_len_align),
                    data_tail.access_ptr("r", offset=0),
                    0, 1, 1, 0, 0))


def _move_sp_vadds_21_20_fp32(dst, data, n_begin):
    """
    function of making ir node builder for 21 20 float32 scene

    """
    tvm_ib = tvm.ir_builder.create()
    device_core_num = AICORE_NUM
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - 64
    ub_ele = (ub_bytes // float_size // cp_align_len) * cp_align_len
    ub_ele_half = ub_ele // 2
    ub_ele_eight = ub_ele // 16

    data_ub_one = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                             "data_ub_one", scope=cce.scope_ubuf)
    data_ub_two = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                             "data_ub_two", scope=cce.scope_ubuf)
    data_ub_three = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                               "data_ub_three", scope=cce.scope_ubuf)
    data_ub_four = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                              "data_ub_four", scope=cce.scope_ubuf)
    data_ub_five = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                              "data_ub_five", scope=cce.scope_ubuf)
    data_ub_six = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                             "data_ub_six", scope=cce.scope_ubuf)
    data_ub_seven = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                               "data_ub_seven", scope=cce.scope_ubuf)
    data_ub_eight = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                               "data_ub_eight", scope=cce.scope_ubuf)

    data_res = _new_alloc(tvm_ib, dst.dtype, ub_ele_half,
                          "data_res", scope=cce.scope_ubuf)
    data_tail = _new_alloc(tvm_ib, dst.dtype, cp_align_len,
                           "data_tail", scope=cce.scope_ubuf)
    reg = tvm_ib.allocate(dst.dtype, (8,), name='reg', scope=cce.scope_reg)

    data_ub = [data_ub_one, data_ub_two, data_ub_three, data_ub_four,
               data_ub_five, data_ub_six, data_ub_seven, data_ub_eight]

    n_i = data.shape[0]
    group_index = n_i // device_core_num
    group_mod = n_i % device_core_num
    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    with tvm_ib.for_range(0, group_index, name="num_g") as num_g:
        args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg,\
               device_core_num, block_index, n_begin, ub_ele_eight,\
               cp_align_len, num_g
        _func_21_20_fp32(args)
    with tvm_ib.if_scope(group_mod > 0):
        with tvm_ib.if_scope(block_index < group_mod):
            args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
                   device_core_num, block_index, n_begin, ub_ele_eight, \
                   cp_align_len, group_index
            _func_21_20_fp32(args)

    return tvm_ib.get()


def _func_91_90_fp32(args):
    """
    function of moving data for 91 90 float32 scene

    """
    tvm_ib, data, dst, data_ub, data_res, data_tail, reg, device_core_num, \
    block_index, n_begin, ub_ele_eight, cp_align_len, num_g = args

    _, col_len, row_in = data.shape
    row_out = dst.shape[2]
    row_zu_in = 8
    row_out_block = _ceil_div(row_out, cp_align_len)
    row_out_align = _ceil_fill(row_out, cp_align_len)
    num_col_ub = ub_ele_eight // row_out_align
    num_ub_loop = col_len // 8 // num_col_ub
    num_ub_mod = col_len % (8 * num_col_ub)
    n_index = num_g * device_core_num + block_index

    with tvm_ib.for_range(0, num_ub_loop, name="num_u") as num_u:
        for num_zi in range(row_zu_in):
            data_offset = n_index * col_len * row_in\
                          + num_u * 8 * num_col_ub * row_in\
                          + num_zi * row_in + n_begin - 2 * (num_zi % 4)
            ub_offset = 0
            n_burst = num_col_ub
            burst_len_data = row_out_block
            src_stride = (8 * row_in) // cp_align_len - burst_len_data
            dst_stride = 0
            args = tvm_ib, data, data_ub[num_zi], data_offset, ub_offset, \
                   n_burst, burst_len_data, src_stride, \
                   dst_stride, cp_align_len
            _func_gm_to_ub_align(args)

            mask1, mask2 = _set_mask_slice(0 + 2 * (num_zi % 4), 0)
            tvm_ib.emit(tvm.call_extern(
                dst.dtype, "set_vector_mask",
                tvm.const(mask1, dtype="uint64"),
                tvm.const(mask2, dtype="uint64")))
            ub_offset = 0
            res_offset = num_zi * row_out - 2 * (num_zi % 4)
            repeat = num_col_ub
            srcm0 = 1
            dstm0 = 1
            srcm1 = row_out_block
            dstm1 = row_out * row_zu_in // cp_align_len
            args = tvm_ib, data_ub[num_zi], data_res, \
                   ub_offset, res_offset, \
                   repeat, srcm0, dstm0, srcm1, dstm1, cp_align_len
            _func_vadds(args)

            mask1, mask2 = _set_mask_slice(0, 128 - 90 - 2 * (num_zi % 4))
            tvm_ib.emit(tvm.call_extern(
                dst.dtype, "set_vector_mask",
                tvm.const(mask1, dtype="uint64"),
                tvm.const(mask2, dtype="uint64")))
            ub_offset = 64
            res_offset = num_zi * row_out - 2 * (num_zi % 4) + 64
            repeat = num_col_ub
            srcm0 = 1
            dstm0 = 1
            srcm1 = row_out_block
            dstm1 = row_out * row_zu_in // cp_align_len
            args = tvm_ib, data_ub[num_zi], data_res, \
                   ub_offset, res_offset, \
                   repeat, srcm0, dstm0, srcm1, dstm1, cp_align_len
            _func_vadds(args)

            tvm_ib.emit(tvm.call_extern(
                dst.dtype, "set_vector_mask",
                tvm.const(-1, dtype="uint64"),
                tvm.const(-1, dtype="uint64")))

        dst_offset = n_index * col_len * row_out \
                     + num_u * 8 * num_col_ub * row_out
        burst_len_dst = (8 * num_col_ub * row_out) // cp_align_len
        tvm_ib.emit(
            tvm.call_extern(
                dst.dtype, "copy_ubuf_to_gm",
                dst.access_ptr('w', offset=dst_offset),
                data_res.access_ptr("r", offset=0),
                0, 1, burst_len_dst, 0, 0))

    with tvm_ib.if_scope(num_ub_mod > 0):
        col_group = num_ub_mod // row_zu_in
        col_mod = num_ub_mod % row_zu_in

        for num_zi in range(row_zu_in):
            with tvm_ib.if_scope(num_zi < num_ub_mod):
                data_offset = n_index * col_len * row_in\
                              + num_ub_loop * 8 * num_col_ub * row_in\
                              + num_zi * row_in + n_begin - 2 * (num_zi % 4)
                ub_offset = 0
                n_burst = tvm.select(num_zi < col_mod, col_group + 1,
                                     col_group)
                burst_len_data = row_out_block
                src_stride = (8 * row_in) // cp_align_len - burst_len_data
                dst_stride = 0
                args = tvm_ib, data, data_ub[num_zi], data_offset, ub_offset, \
                       n_burst, burst_len_data, src_stride, \
                       dst_stride, cp_align_len
                _func_gm_to_ub_align(args)

                mask1, mask2 = _set_mask_slice(0 + 2 * (num_zi % 4), 0)
                tvm_ib.emit(tvm.call_extern(
                    dst.dtype, "set_vector_mask",
                    tvm.const(mask1, dtype="uint64"),
                    tvm.const(mask2, dtype="uint64")))
                ub_offset = 0
                res_offset = num_zi * row_out - 2 * (num_zi % 4)
                repeat = tvm.select(num_zi < col_mod, col_group + 1,
                                    col_group)
                srcm0 = 1
                dstm0 = 1
                srcm1 = row_out_block
                dstm1 = row_out * row_zu_in // cp_align_len
                args = tvm_ib, data_ub[num_zi], data_res, \
                       ub_offset, res_offset, \
                       repeat, srcm0, dstm0, srcm1, dstm1, cp_align_len
                _func_vadds(args)

                mask1, mask2 = _set_mask_slice(0, 128 - 90 - 2 * (num_zi % 4))
                tvm_ib.emit(tvm.call_extern(
                    dst.dtype, "set_vector_mask",
                    tvm.const(mask1, dtype="uint64"),
                    tvm.const(mask2, dtype="uint64")))
                ub_offset = 64
                res_offset = num_zi * row_out - 2 * (num_zi % 4) + 64
                repeat = tvm.select(num_zi < col_mod, col_group + 1,
                                    col_group)
                srcm0 = 1
                dstm0 = 1
                srcm1 = row_out_block
                dstm1 = row_out * row_zu_in // cp_align_len
                args = tvm_ib, data_ub[num_zi], data_res, \
                       ub_offset, res_offset, \
                       repeat, srcm0, dstm0, srcm1, dstm1, cp_align_len
                _func_vadds(args)

                tvm_ib.emit(tvm.call_extern(
                    dst.dtype, "set_vector_mask",
                    tvm.const(-1, dtype="uint64"),
                    tvm.const(-1, dtype="uint64")))

        dst_offset = n_index * col_len * row_out \
                     + num_ub_loop * 8 * num_col_ub * row_out
        mod_len = num_ub_mod * row_out
        with tvm_ib.if_scope(mod_len % cp_align_len == 0):
            burst_len_dst = mod_len // cp_align_len
            tvm_ib.emit(
                tvm.call_extern(
                    dst.dtype, "copy_ubuf_to_gm",
                    dst.access_ptr('w', offset=dst_offset),
                    data_res.access_ptr("r", offset=0),
                    0, 1, burst_len_dst, 0, 0))
        with tvm_ib.else_scope():
            mod_len_align = mod_len - cp_align_len
            burst_len_dst_a = _ceil_div(mod_len_align, cp_align_len)
            tvm_ib.emit(
                tvm.call_extern(
                    dst.dtype, "copy_ubuf_to_gm",
                    dst.access_ptr('w', offset=dst_offset),
                    data_res.access_ptr("r", offset=0),
                    0, 1, burst_len_dst_a, 0, 0))
            with tvm_ib.for_range(0, cp_align_len, name="num_a") as num_a:
                tvm_ib.emit(tvm.call_extern(
                    data_res.dtype, "reg_mov",
                    tvm.call_extern(reg.dtype, "reg", reg[0]),
                    data_res.access_ptr('r',
                                        offset=mod_len_align + num_a)
                ))
                tvm_ib.emit(tvm.call_extern(
                    data_tail.dtype, "reg_mov",
                    data_tail.access_ptr('w',
                                         offset=num_a),
                    tvm.call_extern(reg.dtype, "reg", reg[0])
                ))
            tvm_ib.emit(
                tvm.call_extern(
                    dst.dtype, "copy_ubuf_to_gm",
                    dst.access_ptr('w', offset=dst_offset + mod_len_align),
                    data_tail.access_ptr("r", offset=0),
                    0, 1, 1, 0, 0))


def _move_sp_vadds_91_90_fp32(dst, data, n_begin):
    """
    function of making ir node builder for 91 90 float32 scene

    """
    tvm_ib = tvm.ir_builder.create()
    device_core_num = AICORE_NUM
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - 64
    ub_ele = (ub_bytes // float_size // cp_align_len) * cp_align_len
    ub_ele_half = ub_ele // 2
    ub_ele_eight = ub_ele // 16

    data_ub_one = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                             "data_ub_one", scope=cce.scope_ubuf)
    data_ub_two = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                             "data_ub_two", scope=cce.scope_ubuf)
    data_ub_three = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                               "data_ub_three", scope=cce.scope_ubuf)
    data_ub_four = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                              "data_ub_four", scope=cce.scope_ubuf)
    data_ub_five = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                              "data_ub_five", scope=cce.scope_ubuf)
    data_ub_six = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                             "data_ub_six", scope=cce.scope_ubuf)
    data_ub_seven = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                               "data_ub_seven", scope=cce.scope_ubuf)
    data_ub_eight = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                               "data_ub_eight", scope=cce.scope_ubuf)

    data_res = _new_alloc(tvm_ib, dst.dtype, ub_ele_half,
                          "data_res", scope=cce.scope_ubuf)
    data_tail = _new_alloc(tvm_ib, dst.dtype, cp_align_len,
                           "data_tail", scope=cce.scope_ubuf)
    reg = tvm_ib.allocate(dst.dtype, (8,), name='reg', scope=cce.scope_reg)

    data_ub = [data_ub_one, data_ub_two, data_ub_three, data_ub_four,
               data_ub_five, data_ub_six, data_ub_seven, data_ub_eight]

    n_i = data.shape[0]
    group_index = n_i // device_core_num
    group_mod = n_i % device_core_num
    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    with tvm_ib.for_range(0, group_index, name="num_g") as num_g:
        args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg,\
               device_core_num, block_index, n_begin, ub_ele_eight,\
               cp_align_len, num_g
        _func_91_90_fp32(args)
    with tvm_ib.if_scope(group_mod > 0):
        with tvm_ib.if_scope(block_index < group_mod):
            args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
                   device_core_num, block_index, n_begin, ub_ele_eight, \
                   cp_align_len, group_index
            _func_91_90_fp32(args)

    return tvm_ib.get()


def _func_602_601_fp32(args):
    """
    function of moving data for 602 601 float32 scene

    """
    tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
    device_core_num, block_index, n_begin, ub_ele_eight, \
    cp_align_len, num_g = args

    _, col_len, row_in = data.shape
    row_out = dst.shape[2]
    row_zu_out = 8
    row_out_block = _ceil_div(row_out, cp_align_len)
    row_out_align = _ceil_fill(row_out, cp_align_len)
    row_out_block_t = row_out_block + 1
    row_out_align_t = row_out_align + 8
    num_col_ub = ub_ele_eight // row_out_align_t
    num_ub_loop = col_len // 8 // num_col_ub
    num_ub_mod = col_len % (8 * num_col_ub)
    n_index = num_g * device_core_num + block_index

    with tvm_ib.for_range(0, num_ub_loop, name="num_u") as num_u:
        for num_zi in range(row_zu_out):
            data_offset = n_index * col_len * row_in\
                          + num_u * 8 * num_col_ub * row_in\
                          + num_zi * row_in + n_begin - (num_zi * row_out % 8)
            ub_offset = 0
            n_burst = num_col_ub
            burst_len_data = row_out_block_t
            src_stride = (8 * row_in) // cp_align_len - burst_len_data
            dst_stride = 0
            args = tvm_ib, data, data_ub[num_zi], data_offset, ub_offset, \
                   n_burst, burst_len_data, src_stride, \
                   dst_stride, cp_align_len
            _func_gm_to_ub_align(args)

            with tvm_ib.for_range(0, num_col_ub, name="num_cu") as num_cu:
                mask1, mask2 = _set_mask_slice(0 + (num_zi % 8), 0)
                tvm_ib.emit(tvm.call_extern(
                    dst.dtype, "set_vector_mask",
                    tvm.const(mask1, dtype="uint64"),
                    tvm.const(mask2, dtype="uint64")))
                ub_offset = num_cu * row_out_align_t
                res_offset = num_zi * row_out - (num_zi % 8)\
                             + num_cu * 8 * row_out
                repeat = 1
                srcm0 = 1
                dstm0 = 1
                srcm1 = 0
                dstm1 = 0
                args = tvm_ib, data_ub[num_zi], data_res, \
                       ub_offset, res_offset, \
                       repeat, srcm0, dstm0, srcm1, dstm1, cp_align_len
                _func_vadds(args)

                mask1, mask2 = _set_mask_slice(0, 0)
                tvm_ib.emit(tvm.call_extern(
                    dst.dtype, "set_vector_mask",
                    tvm.const(mask1, dtype="uint64"),
                    tvm.const(mask2, dtype="uint64")))
                ub_offset = num_cu * row_out_align_t + 64
                res_offset = num_zi * row_out - (num_zi % 8)\
                             + num_cu * 8 * row_out + 64
                repeat = 8
                srcm0 = 1
                dstm0 = 1
                srcm1 = 8
                dstm1 = 8
                args = tvm_ib, data_ub[num_zi], data_res, \
                       ub_offset, res_offset, \
                       repeat, srcm0, dstm0, srcm1, dstm1, cp_align_len
                _func_vadds(args)

                mask1, mask2 = _set_mask_slice(0,
                                               (10 * 64 - row_out
                                                - (num_zi % 8)))
                tvm_ib.emit(tvm.call_extern(
                    dst.dtype, "set_vector_mask",
                    tvm.const(mask1, dtype="uint64"),
                    tvm.const(mask2, dtype="uint64")))
                ub_offset = num_cu * row_out_align_t + 64 * 9
                res_offset = num_zi * row_out - (num_zi % 8)\
                             + num_cu * 8 * row_out + 64 * 9
                repeat = 1
                srcm0 = 1
                dstm0 = 1
                srcm1 = 0
                dstm1 = 0
                args = tvm_ib, data_ub[num_zi], data_res, \
                       ub_offset, res_offset, \
                       repeat, srcm0, dstm0, srcm1, dstm1, cp_align_len
                _func_vadds(args)

            tvm_ib.emit(tvm.call_extern(
                dst.dtype, "set_vector_mask",
                tvm.const(-1, dtype="uint64"),
                tvm.const(-1, dtype="uint64")))

        dst_offset = n_index * col_len * row_out \
                     + num_u * 8 * num_col_ub * row_out
        burst_len_dst = (8 * num_col_ub * row_out) // cp_align_len
        tvm_ib.emit(
            tvm.call_extern(
                dst.dtype, "copy_ubuf_to_gm",
                dst.access_ptr('w', offset=dst_offset),
                data_res.access_ptr("r", offset=0),
                0, 1, burst_len_dst, 0, 0))

    with tvm_ib.if_scope(num_ub_mod > 0):
        col_group = num_ub_mod // row_zu_out
        col_mod = num_ub_mod % row_zu_out

        for num_zi in range(row_zu_out):
            with tvm_ib.if_scope(num_zi < num_ub_mod):
                data_offset = n_index * col_len * row_in \
                              + num_ub_loop * 8 * num_col_ub * row_in \
                              + num_zi * row_in + n_begin\
                              - (num_zi * row_out % 8)
                ub_offset = 0
                n_burst = tvm.select(num_zi < col_mod, col_group + 1,
                                     col_group)

                burst_len_data = row_out_block_t
                src_stride = (8 * row_in) // cp_align_len - burst_len_data
                dst_stride = 0
                args = tvm_ib, data, data_ub[num_zi], data_offset, ub_offset,\
                       n_burst, burst_len_data, src_stride,\
                       dst_stride, cp_align_len
                _func_gm_to_ub_align(args)

                mod_zu = tvm.select(num_zi < col_mod, col_group + 1,
                                    col_group)
                with tvm_ib.for_range(0, mod_zu, name="num_cu") as num_cu:
                    mask1, mask2 = _set_mask_slice(0 + (num_zi % 8), 0)
                    tvm_ib.emit(tvm.call_extern(
                        dst.dtype, "set_vector_mask",
                        tvm.const(mask1, dtype="uint64"),
                        tvm.const(mask2, dtype="uint64")))
                    ub_offset = num_cu * row_out_align_t
                    res_offset = num_zi * row_out - (num_zi % 8)\
                                 + num_cu * 8 * row_out
                    repeat = 1
                    srcm0 = 1
                    dstm0 = 1
                    srcm1 = 0
                    dstm1 = 0
                    args = tvm_ib, data_ub[num_zi], data_res, \
                           ub_offset, res_offset, \
                           repeat, srcm0, dstm0, srcm1, dstm1, cp_align_len
                    _func_vadds(args)

                    mask1, mask2 = _set_mask_slice(0, 0)
                    tvm_ib.emit(tvm.call_extern(
                        dst.dtype, "set_vector_mask",
                        tvm.const(mask1, dtype="uint64"),
                        tvm.const(mask2, dtype="uint64")))
                    ub_offset = num_cu * row_out_align_t + 64
                    res_offset = num_zi * row_out - (num_zi % 8)\
                                 + num_cu * 8 * row_out + 64
                    repeat = 8
                    srcm0 = 1
                    dstm0 = 1
                    srcm1 = 8
                    dstm1 = 8
                    args = tvm_ib, data_ub[num_zi], data_res, \
                           ub_offset, res_offset, \
                           repeat, srcm0, dstm0, srcm1, dstm1, cp_align_len
                    _func_vadds(args)

                    mask1, mask2 = _set_mask_slice(0,
                                                   (10 * 64 - row_out
                                                    - (num_zi % 8)))
                    tvm_ib.emit(tvm.call_extern(
                        dst.dtype, "set_vector_mask",
                        tvm.const(mask1, dtype="uint64"),
                        tvm.const(mask2, dtype="uint64")))
                    ub_offset = num_cu * row_out_align_t + 64 * 9
                    res_offset = num_zi * row_out - (num_zi % 8)\
                                 + num_cu * 8 * row_out + 64 * 9
                    repeat = 1
                    srcm0 = 1
                    dstm0 = 1
                    srcm1 = 0
                    dstm1 = 0
                    args = tvm_ib, data_ub[num_zi], data_res, \
                           ub_offset, res_offset, \
                           repeat, srcm0, dstm0, srcm1, dstm1, cp_align_len
                    _func_vadds(args)

                tvm_ib.emit(tvm.call_extern(
                    dst.dtype, "set_vector_mask",
                    tvm.const(-1, dtype="uint64"),
                    tvm.const(-1, dtype="uint64")))

        dst_offset = n_index * col_len * row_out \
                     + num_ub_loop * 8 * num_col_ub * row_out
        mod_len = num_ub_mod * row_out
        with tvm_ib.if_scope(mod_len % cp_align_len == 0):
            burst_len_dst = mod_len // cp_align_len
            tvm_ib.emit(
                tvm.call_extern(
                    dst.dtype, "copy_ubuf_to_gm",
                    dst.access_ptr('w', offset=dst_offset),
                    data_res.access_ptr("r", offset=0),
                    0, 1, burst_len_dst, 0, 0))
        with tvm_ib.else_scope():
            mod_len_align = mod_len - cp_align_len
            burst_len_dst_a = _ceil_div(mod_len_align, cp_align_len)
            tvm_ib.emit(
                tvm.call_extern(
                    dst.dtype, "copy_ubuf_to_gm",
                    dst.access_ptr('w', offset=dst_offset),
                    data_res.access_ptr("r", offset=0),
                    0, 1, burst_len_dst_a, 0, 0))
            with tvm_ib.for_range(0, cp_align_len, name="num_a") as num_a:
                tvm_ib.emit(tvm.call_extern(
                    data_res.dtype, "reg_mov",
                    tvm.call_extern(reg.dtype, "reg", reg[0]),
                    data_res.access_ptr('r',
                                        offset=mod_len_align + num_a)
                ))
                tvm_ib.emit(tvm.call_extern(
                    data_tail.dtype, "reg_mov",
                    data_tail.access_ptr('w',
                                         offset=num_a),
                    tvm.call_extern(reg.dtype, "reg", reg[0])
                ))
            tvm_ib.emit(
                tvm.call_extern(
                    dst.dtype, "copy_ubuf_to_gm",
                    dst.access_ptr('w', offset=dst_offset + mod_len_align),
                    data_tail.access_ptr("r", offset=0),
                    0, 1, 1, 0, 0))


def _move_sp_vadds_602_601_fp32(dst, data, n_begin):
    """
    function of making ir node builder for 602 601 float32 scene

    """
    tvm_ib = tvm.ir_builder.create()
    device_core_num = AICORE_NUM
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - 64
    ub_ele = (ub_bytes // float_size // cp_align_len) * cp_align_len
    ub_ele_half = ub_ele // 2
    ub_ele_eight = ub_ele // 16

    data_ub_one = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                             "data_ub_one", scope=cce.scope_ubuf)
    data_ub_two = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                             "data_ub_two", scope=cce.scope_ubuf)
    data_ub_three = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                               "data_ub_three", scope=cce.scope_ubuf)
    data_ub_four = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                              "data_ub_four", scope=cce.scope_ubuf)
    data_ub_five = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                              "data_ub_five", scope=cce.scope_ubuf)
    data_ub_six = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                             "data_ub_six", scope=cce.scope_ubuf)
    data_ub_seven = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                               "data_ub_seven", scope=cce.scope_ubuf)
    data_ub_eight = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                               "data_ub_eight", scope=cce.scope_ubuf)

    data_res = _new_alloc(tvm_ib, dst.dtype, ub_ele_half,
                          "data_res", scope=cce.scope_ubuf)
    data_tail = _new_alloc(tvm_ib, dst.dtype, cp_align_len,
                           "data_tail", scope=cce.scope_ubuf)
    reg = tvm_ib.allocate(dst.dtype, (8,), name='reg', scope=cce.scope_reg)

    data_ub = [data_ub_one, data_ub_two, data_ub_three, data_ub_four,
               data_ub_five, data_ub_six, data_ub_seven, data_ub_eight]

    n_i = data.shape[0]
    group_index = n_i // device_core_num
    group_mod = n_i % device_core_num
    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    with tvm_ib.for_range(0, group_index, name="num_g") as num_g:
        args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
               device_core_num, block_index, n_begin, ub_ele_eight, \
               cp_align_len, num_g
        _func_602_601_fp32(args)
    with tvm_ib.if_scope(group_mod > 0):
        with tvm_ib.if_scope(block_index < group_mod):
            args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
                   device_core_num, block_index, n_begin, ub_ele_eight, \
                   cp_align_len, group_index
            _func_602_601_fp32(args)

    return tvm_ib.get()


def _func_21_20_fp16(args):
    """
    function of moving data for 21 20 float16 scene

    """
    tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
    device_core_num, block_index, n_begin, ub_ele_eight, \
    cp_align_len, num_g = args

    _, col_len, row_in = data.shape
    row_out = dst.shape[2]

    row_zu_in = 16
    row_out_block = _ceil_div(row_out, cp_align_len)
    row_out_align = _ceil_fill(row_out, cp_align_len)
    num_col_ub = ub_ele_eight // row_out_align
    num_ub_loop = col_len // 16 // num_col_ub
    num_ub_mod = col_len % (16 * num_col_ub)
    n_index = num_g * device_core_num + block_index

    with tvm_ib.for_range(0, num_ub_loop, name="num_u") as num_u:
        for num_zi in range(row_zu_in):
            data_offset = n_index * col_len * row_in\
                          + num_u * 16 * num_col_ub * row_in\
                          + num_zi * row_in + n_begin - 4 * (num_zi % 4)
            ub_offset = 0
            n_burst = num_col_ub
            burst_len_data = row_out_block
            src_stride = (16 * row_in) // cp_align_len - burst_len_data
            dst_stride = 0
            args = tvm_ib, data, data_ub[num_zi], data_offset, ub_offset, \
                   n_burst, burst_len_data, src_stride, \
                   dst_stride, cp_align_len
            _func_gm_to_ub_align(args)

            mask1, mask2 = _set_mask_slice_fp16(0 + 4 * (num_zi % 4),
                                                (128 - row_out
                                                 - 4 * (num_zi % 4)))
            tvm_ib.emit(tvm.call_extern(
                dst.dtype, "set_vector_mask",
                tvm.const(mask1, dtype="uint64"),
                tvm.const(mask2, dtype="uint64")))
            ub_offset = 0
            res_offset = num_zi * row_out - 4 * (num_zi % 4)
            repeat = num_col_ub
            srcm0 = 1
            dstm0 = 1
            srcm1 = row_out_block
            dstm1 = row_out * row_zu_in // cp_align_len
            args = tvm_ib, data_ub[num_zi], data_res, \
                   ub_offset, res_offset, \
                   repeat, srcm0, dstm0, srcm1, dstm1, cp_align_len
            _func_vadds(args)

            tvm_ib.emit(tvm.call_extern(
                dst.dtype, "set_vector_mask",
                tvm.const(-1, dtype="uint64"),
                tvm.const(-1, dtype="uint64")))

        dst_offset = n_index * col_len * row_out\
                     + num_u * 16 * num_col_ub * row_out
        burst_len_dst = (16 * num_col_ub * row_out) // cp_align_len
        tvm_ib.emit(
            tvm.call_extern(
                dst.dtype, "copy_ubuf_to_gm",
                dst.access_ptr('w', offset=dst_offset),
                data_res.access_ptr("r", offset=0),
                0, 1, burst_len_dst, 0, 0))

    with tvm_ib.if_scope(num_ub_mod > 0):
        col_group = num_ub_mod // row_zu_in
        col_mod = num_ub_mod % row_zu_in

        for num_zi in range(row_zu_in):
            with tvm_ib.if_scope(num_zi < num_ub_mod):
                data_offset = n_index * col_len * row_in\
                              + num_ub_loop * 16 * num_col_ub * row_in\
                              + num_zi * row_in + n_begin - 4 * (num_zi % 4)
                ub_offset = 0
                n_burst = tvm.select(num_zi < col_mod, col_group + 1,
                                     col_group)
                burst_len_data = row_out_block
                src_stride = (16 * row_in) // cp_align_len - burst_len_data
                dst_stride = 0
                args = tvm_ib, data, data_ub[num_zi], data_offset, ub_offset,\
                       n_burst, burst_len_data, src_stride, \
                       dst_stride, cp_align_len
                _func_gm_to_ub_align(args)

                mask1, mask2 = _set_mask_slice_fp16(0 + 4 * (num_zi % 4),
                                                    (128 - row_out
                                                     - 4 * (num_zi % 4)))
                tvm_ib.emit(tvm.call_extern(
                    dst.dtype, "set_vector_mask",
                    tvm.const(mask1, dtype="uint64"),
                    tvm.const(mask2, dtype="uint64")))
                ub_offset = 0
                res_offset = num_zi * row_out - 4 * (num_zi % 4)
                repeat = tvm.select(num_zi < col_mod, col_group + 1,
                                    col_group)
                srcm0 = 1
                dstm0 = 1
                srcm1 = row_out_block
                dstm1 = row_out * row_zu_in // cp_align_len
                args = tvm_ib, data_ub[num_zi], data_res, \
                       ub_offset, res_offset, \
                       repeat, srcm0, dstm0, srcm1, dstm1, cp_align_len
                _func_vadds(args)

                tvm_ib.emit(tvm.call_extern(
                    dst.dtype, "set_vector_mask",
                    tvm.const(-1, dtype="uint64"),
                    tvm.const(-1, dtype="uint64")))

        dst_offset = n_index * col_len * row_out \
                     + num_ub_loop * 16 * num_col_ub * row_out
        mod_len = num_ub_mod * row_out
        with tvm_ib.if_scope(mod_len % cp_align_len == 0):
            burst_len_dst = mod_len // cp_align_len
            tvm_ib.emit(
                tvm.call_extern(
                    dst.dtype, "copy_ubuf_to_gm",
                    dst.access_ptr('w', offset=dst_offset),
                    data_res.access_ptr("r", offset=0),
                    0, 1, burst_len_dst, 0, 0))
        with tvm_ib.else_scope():
            mod_len_align = mod_len - cp_align_len
            burst_len_dst_a = _ceil_div(mod_len_align, cp_align_len)
            tvm_ib.emit(
                tvm.call_extern(
                    dst.dtype, "copy_ubuf_to_gm",
                    dst.access_ptr('w', offset=dst_offset),
                    data_res.access_ptr("r", offset=0),
                    0, 1, burst_len_dst_a, 0, 0))
            with tvm_ib.for_range(0, cp_align_len, name="num_a") as num_a:
                tvm_ib.emit(tvm.call_extern(
                    data_res.dtype, "reg_mov",
                    tvm.call_extern(reg.dtype, "reg", reg[0]),
                    data_res.access_ptr('r',
                                        offset=mod_len_align + num_a)
                ))
                tvm_ib.emit(tvm.call_extern(
                    data_tail.dtype, "reg_mov",
                    data_tail.access_ptr('w',
                                         offset=num_a),
                    tvm.call_extern(reg.dtype, "reg", reg[0])
                ))
            tvm_ib.emit(
                tvm.call_extern(
                    dst.dtype, "copy_ubuf_to_gm",
                    dst.access_ptr('w', offset=dst_offset + mod_len_align),
                    data_tail.access_ptr("r", offset=0),
                    0, 1, 1, 0, 0))


def _move_sp_vadds_21_20_fp16(dst, data, n_begin):
    """
    function of making ir node builder for 21 20 float16 scene

    """
    tvm_ib = tvm.ir_builder.create()
    device_core_num = AICORE_NUM
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - 64
    ub_ele = (ub_bytes // float_size // cp_align_len) * cp_align_len
    ub_ele_half = ub_ele // 2
    ub_ele_eight = ub_ele // 32

    data_ub_one = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                             "data_ub_one", scope=cce.scope_ubuf)
    data_ub_two = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                             "data_ub_two", scope=cce.scope_ubuf)
    data_ub_three = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                               "data_ub_three", scope=cce.scope_ubuf)
    data_ub_four = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                              "data_ub_four", scope=cce.scope_ubuf)
    data_ub_five = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                              "data_ub_five", scope=cce.scope_ubuf)
    data_ub_six = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                             "data_ub_six", scope=cce.scope_ubuf)
    data_ub_seven = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                               "data_ub_seven", scope=cce.scope_ubuf)
    data_ub_eight = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                               "data_ub_eight", scope=cce.scope_ubuf)
    data_ub_nine = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                              "data_ub_nine", scope=cce.scope_ubuf)
    data_ub_ten = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                             "data_ub_ten", scope=cce.scope_ubuf)
    data_ub_eleven = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                                "data_ub_eleven", scope=cce.scope_ubuf)
    data_ub_twelve = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                                "data_ub_twelve", scope=cce.scope_ubuf)
    data_ub_thirteen = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                                  "data_ub_thirteen", scope=cce.scope_ubuf)
    data_ub_fourteen = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                                  "data_ub_fourteen", scope=cce.scope_ubuf)
    data_ub_fifteen = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                                 "data_ub_fifteen", scope=cce.scope_ubuf)
    data_ub_sixteen = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                                 "data_ub_sixteen", scope=cce.scope_ubuf)

    data_res = _new_alloc(tvm_ib, dst.dtype, ub_ele_half,
                          "data_res", scope=cce.scope_ubuf)
    data_tail = _new_alloc(tvm_ib, dst.dtype, cp_align_len,
                           "data_tail", scope=cce.scope_ubuf)
    reg = tvm_ib.allocate(dst.dtype, (8,), name='reg', scope=cce.scope_reg)

    data_ub = [data_ub_one, data_ub_two, data_ub_three, data_ub_four,
               data_ub_five, data_ub_six, data_ub_seven, data_ub_eight,
               data_ub_nine, data_ub_ten, data_ub_eleven, data_ub_twelve,
               data_ub_thirteen, data_ub_fourteen,
               data_ub_fifteen, data_ub_sixteen]

    n_i = data.shape[0]
    group_index = n_i // device_core_num
    group_mod = n_i % device_core_num
    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    with tvm_ib.for_range(0, group_index, name="num_g") as num_g:
        args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
               device_core_num, block_index, n_begin, ub_ele_eight, \
               cp_align_len, num_g
        _func_21_20_fp16(args)
    with tvm_ib.if_scope(group_mod > 0):
        with tvm_ib.if_scope(block_index < group_mod):
            args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
                   device_core_num, block_index, n_begin, ub_ele_eight, \
                   cp_align_len, group_index
            _func_21_20_fp16(args)

    return tvm_ib.get()


def _func_91_90_fp16(args):
    """
    function of moving data for 91 90 float16 scene

    """
    tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
    device_core_num, block_index, n_begin, ub_ele_eight, \
    cp_align_len, num_g = args

    _, col_len, row_in = data.shape
    row_out = dst.shape[2]
    row_zu_in = 16
    row_out_block = _ceil_div(row_out, cp_align_len)
    row_out_align = _ceil_fill(row_out, cp_align_len)
    row_out_block_t = row_out_block + 1
    row_out_align_t = row_out_align + 16
    num_col_ub = ub_ele_eight // row_out_align_t
    num_ub_loop = col_len // 16 // num_col_ub
    num_ub_mod = col_len % (16 * num_col_ub)
    n_index = num_g * device_core_num + block_index

    with tvm_ib.for_range(0, num_ub_loop, name="num_u") as num_u:
        for num_zi in range(row_zu_in):
            data_offset = n_index * col_len * row_in\
                          + num_u * 16 * num_col_ub * row_in\
                          + num_zi * row_in + n_begin - (num_zi * row_out % 16)
            ub_offset = 0
            n_burst = num_col_ub
            burst_len_data = row_out_block_t
            src_stride = (16 * row_in) // cp_align_len - burst_len_data
            dst_stride = 0
            args = tvm_ib, data, data_ub[num_zi], data_offset, ub_offset, \
                   n_burst, burst_len_data, src_stride, \
                   dst_stride, cp_align_len
            _func_gm_to_ub_align(args)

            mask1, mask2 = _set_mask_slice_fp16(0 + (num_zi * row_out % 16),
                                                (128 - row_out
                                                 - (num_zi * row_out % 16)))
            tvm_ib.emit(tvm.call_extern(
                dst.dtype, "set_vector_mask",
                tvm.const(mask1, dtype="uint64"),
                tvm.const(mask2, dtype="uint64")))
            ub_offset = 0
            res_offset = num_zi * row_out - (num_zi * row_out % 16)
            repeat = num_col_ub
            srcm0 = 1
            dstm0 = 1
            srcm1 = row_out_block_t
            dstm1 = row_out * row_zu_in // cp_align_len
            args = tvm_ib, data_ub[num_zi], data_res, \
                   ub_offset, res_offset, \
                   repeat, srcm0, dstm0, srcm1, dstm1, cp_align_len
            _func_vadds(args)

            tvm_ib.emit(tvm.call_extern(
                dst.dtype, "set_vector_mask",
                tvm.const(-1, dtype="uint64"),
                tvm.const(-1, dtype="uint64")))

        dst_offset = n_index * col_len * row_out\
                     + num_u * 16 * num_col_ub * row_out
        burst_len_dst = (16 * num_col_ub * row_out) // cp_align_len
        tvm_ib.emit(
            tvm.call_extern(
                dst.dtype, "copy_ubuf_to_gm",
                dst.access_ptr('w', offset=dst_offset),
                data_res.access_ptr("r", offset=0),
                0, 1, burst_len_dst, 0, 0))

    with tvm_ib.if_scope(num_ub_mod > 0):
        col_group = num_ub_mod // row_zu_in
        col_mod = num_ub_mod % row_zu_in

        for num_zi in range(row_zu_in):
            with tvm_ib.if_scope(num_zi < num_ub_mod):
                data_offset = n_index * col_len * row_in\
                          + num_ub_loop * 16 * num_col_ub * row_in\
                          + num_zi * row_in + n_begin - (num_zi * row_out % 16)
                ub_offset = 0
                n_burst = tvm.select(num_zi < col_mod, col_group + 1,
                                     col_group)

                burst_len_data = row_out_block_t
                src_stride = (16 * row_in) // cp_align_len - burst_len_data
                dst_stride = 0
                args = tvm_ib, data, data_ub[num_zi], data_offset, ub_offset,\
                       n_burst, burst_len_data, src_stride,\
                       dst_stride, cp_align_len
                _func_gm_to_ub_align(args)

                mask1, mask2 = _set_mask_slice_fp16(0 + (num_zi * row_out % 16),
                                                    (128 - row_out
                                                     - (num_zi * row_out % 16)))
                tvm_ib.emit(tvm.call_extern(
                    dst.dtype, "set_vector_mask",
                    tvm.const(mask1, dtype="uint64"),
                    tvm.const(mask2, dtype="uint64")))
                ub_offset = 0
                res_offset = num_zi * row_out - (num_zi * row_out % 16)
                repeat = tvm.select(num_zi < col_mod, col_group + 1,
                                    col_group)
                srcm0 = 1
                dstm0 = 1
                srcm1 = row_out_block_t
                dstm1 = row_out * row_zu_in // cp_align_len
                args = tvm_ib, data_ub[num_zi], data_res, \
                       ub_offset, res_offset, \
                       repeat, srcm0, dstm0, srcm1, dstm1, cp_align_len
                _func_vadds(args)

                tvm_ib.emit(tvm.call_extern(
                    dst.dtype, "set_vector_mask",
                    tvm.const(-1, dtype="uint64"),
                    tvm.const(-1, dtype="uint64")))

        dst_offset = n_index * col_len * row_out\
                     + num_ub_loop * 16 * num_col_ub * row_out
        mod_len = num_ub_mod * row_out
        with tvm_ib.if_scope(mod_len % cp_align_len == 0):
            burst_len_dst = mod_len // cp_align_len
            tvm_ib.emit(
                tvm.call_extern(
                    dst.dtype, "copy_ubuf_to_gm",
                    dst.access_ptr('w', offset=dst_offset),
                    data_res.access_ptr("r", offset=0),
                    0, 1, burst_len_dst, 0, 0))
        with tvm_ib.else_scope():
            mod_len_align = mod_len - cp_align_len
            burst_len_dst_a = _ceil_div(mod_len_align, cp_align_len)
            tvm_ib.emit(
                tvm.call_extern(
                    dst.dtype, "copy_ubuf_to_gm",
                    dst.access_ptr('w', offset=dst_offset),
                    data_res.access_ptr("r", offset=0),
                    0, 1, burst_len_dst_a, 0, 0))
            with tvm_ib.for_range(0, cp_align_len, name="num_a") as num_a:
                tvm_ib.emit(tvm.call_extern(
                    data_res.dtype, "reg_mov",
                    tvm.call_extern(reg.dtype, "reg", reg[0]),
                    data_res.access_ptr('r',
                                        offset=mod_len_align + num_a)
                ))
                tvm_ib.emit(tvm.call_extern(
                    data_tail.dtype, "reg_mov",
                    data_tail.access_ptr('w',
                                         offset=num_a),
                    tvm.call_extern(reg.dtype, "reg", reg[0])
                ))
            tvm_ib.emit(
                tvm.call_extern(
                    dst.dtype, "copy_ubuf_to_gm",
                    dst.access_ptr('w', offset=dst_offset + mod_len_align),
                    data_tail.access_ptr("r", offset=0),
                    0, 1, 1, 0, 0))


def _move_sp_vadds_91_90_fp16(dst, data, n_begin):
    """
    function of making ir node builder for 91 90 float16 scene

    """
    tvm_ib = tvm.ir_builder.create()
    device_core_num = AICORE_NUM
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - 64
    ub_ele = (ub_bytes // float_size // cp_align_len) * cp_align_len
    ub_ele_half = ub_ele // 2
    ub_ele_eight = ub_ele // 32

    data_ub_one = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                             "data_ub_one", scope=cce.scope_ubuf)
    data_ub_two = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                             "data_ub_two", scope=cce.scope_ubuf)
    data_ub_three = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                               "data_ub_three", scope=cce.scope_ubuf)
    data_ub_four = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                              "data_ub_four", scope=cce.scope_ubuf)
    data_ub_five = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                              "data_ub_five", scope=cce.scope_ubuf)
    data_ub_six = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                             "data_ub_six", scope=cce.scope_ubuf)
    data_ub_seven = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                               "data_ub_seven", scope=cce.scope_ubuf)
    data_ub_eight = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                               "data_ub_eight", scope=cce.scope_ubuf)
    data_ub_nine = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                              "data_ub_nine", scope=cce.scope_ubuf)
    data_ub_ten = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                             "data_ub_ten", scope=cce.scope_ubuf)
    data_ub_eleven = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                                "data_ub_eleven", scope=cce.scope_ubuf)
    data_ub_twelve = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                                "data_ub_twelve", scope=cce.scope_ubuf)
    data_ub_thirteen = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                                  "data_ub_thirteen", scope=cce.scope_ubuf)
    data_ub_fourteen = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                                  "data_ub_fourteen", scope=cce.scope_ubuf)
    data_ub_fifteen = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                                 "data_ub_fifteen", scope=cce.scope_ubuf)
    data_ub_sixteen = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                                 "data_ub_sixteen", scope=cce.scope_ubuf)

    data_res = _new_alloc(tvm_ib, dst.dtype, ub_ele_half,
                          "data_res", scope=cce.scope_ubuf)
    data_tail = _new_alloc(tvm_ib, dst.dtype, cp_align_len,
                           "data_tail", scope=cce.scope_ubuf)
    reg = tvm_ib.allocate(dst.dtype, (8,), name='reg', scope=cce.scope_reg)

    data_ub = [data_ub_one, data_ub_two, data_ub_three, data_ub_four,
               data_ub_five, data_ub_six, data_ub_seven, data_ub_eight,
               data_ub_nine, data_ub_ten, data_ub_eleven, data_ub_twelve,
               data_ub_thirteen, data_ub_fourteen,
               data_ub_fifteen, data_ub_sixteen]

    n_i = data.shape[0]
    group_index = n_i // device_core_num
    group_mod = n_i % device_core_num
    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    with tvm_ib.for_range(0, group_index, name="num_g") as num_g:
        args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
               device_core_num, block_index, n_begin, ub_ele_eight, \
               cp_align_len, num_g
        _func_91_90_fp16(args)
    with tvm_ib.if_scope(group_mod > 0):
        with tvm_ib.if_scope(block_index < group_mod):
            args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
                   device_core_num, block_index, n_begin, ub_ele_eight, \
                   cp_align_len, group_index
            _func_91_90_fp16(args)

    return tvm_ib.get()


def _func_602_601_fp16(args):
    """
    function of moving data for 602 601 float16 scene

    """
    tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
    device_core_num, block_index, n_begin, ub_ele_eight, \
    cp_align_len, num_g = args

    _, col_len, row_in = data.shape
    row_out = dst.shape[2]
    row_zu_out = 16
    row_out_block = _ceil_div(row_out, cp_align_len)
    row_out_align = _ceil_fill(row_out, cp_align_len)
    row_out_block_t = row_out_block + 1
    row_out_align_t = row_out_align + 16
    num_col_ub = ub_ele_eight // row_out_align_t
    num_ub_loop = col_len // 16 // num_col_ub
    num_ub_mod = col_len % (16 * num_col_ub)
    n_index = num_g * device_core_num + block_index

    with tvm_ib.for_range(0, num_ub_loop, name="num_u") as num_u:
        for num_zi in range(row_zu_out):
            data_offset = n_index * col_len * row_in\
                          + num_u * 16 * num_col_ub * row_in\
                          + num_zi * row_in + n_begin - (num_zi * row_out % 16)
            ub_offset = 0
            n_burst = num_col_ub
            burst_len_data = row_out_block_t
            src_stride = (16 * row_in) // cp_align_len - burst_len_data
            dst_stride = 0
            args = tvm_ib, data, data_ub[num_zi], data_offset, ub_offset, \
                   n_burst, burst_len_data, src_stride, \
                   dst_stride, cp_align_len
            _func_gm_to_ub_align(args)

            with tvm_ib.for_range(0, num_col_ub, name="num_cu") as num_cu:
                mask1, mask2 = _set_mask_slice_fp16((0 + (num_zi * row_out
                                                          % 16)), 0)
                tvm_ib.emit(tvm.call_extern(
                    dst.dtype, "set_vector_mask",
                    tvm.const(mask1, dtype="uint64"),
                    tvm.const(mask2, dtype="uint64")))
                ub_offset = num_cu * row_out_align_t
                res_offset = num_zi * row_out - (num_zi * row_out % 16)\
                             + num_cu * 16 * row_out
                repeat = 1
                srcm0 = 1
                dstm0 = 1
                srcm1 = 0
                dstm1 = 0
                args = tvm_ib, data_ub[num_zi], data_res, \
                       ub_offset, res_offset, \
                       repeat, srcm0, dstm0, srcm1, dstm1, cp_align_len
                _func_vadds(args)

                mask1, mask2 = _set_mask_slice_fp16(0, 0)
                tvm_ib.emit(tvm.call_extern(
                    dst.dtype, "set_vector_mask",
                    tvm.const(mask1, dtype="uint64"),
                    tvm.const(mask2, dtype="uint64")))
                ub_offset = num_cu * row_out_align_t + 128
                res_offset = num_zi * row_out - (num_zi * row_out % 16)\
                             + num_cu * 16 * row_out + 128
                repeat = 3
                srcm0 = 1
                dstm0 = 1
                srcm1 = 8
                dstm1 = 8
                args = tvm_ib, data_ub[num_zi], data_res, \
                       ub_offset, res_offset, \
                       repeat, srcm0, dstm0, srcm1, dstm1, cp_align_len
                _func_vadds(args)

                mask1, mask2 = _set_mask_slice_fp16(0,
                                                    (5 * 128 - row_out
                                                     - (num_zi * row_out % 16)))
                tvm_ib.emit(tvm.call_extern(
                    dst.dtype, "set_vector_mask",
                    tvm.const(mask1, dtype="uint64"),
                    tvm.const(mask2, dtype="uint64")))
                ub_offset = num_cu * row_out_align_t + 128 * 4
                res_offset = num_zi * row_out - (num_zi * row_out % 16)\
                             + num_cu * 16 * row_out + 128 * 4
                repeat = 1
                srcm0 = 1
                dstm0 = 1
                srcm1 = 0
                dstm1 = 0
                args = tvm_ib, data_ub[num_zi], data_res, \
                       ub_offset, res_offset, \
                       repeat, srcm0, dstm0, srcm1, dstm1, cp_align_len
                _func_vadds(args)

            tvm_ib.emit(tvm.call_extern(
                dst.dtype, "set_vector_mask",
                tvm.const(-1, dtype="uint64"),
                tvm.const(-1, dtype="uint64")))

        dst_offset = n_index * col_len * row_out \
                     + num_u * 16 * num_col_ub * row_out
        burst_len_dst = (16 * num_col_ub * row_out) // cp_align_len
        tvm_ib.emit(
            tvm.call_extern(
                dst.dtype, "copy_ubuf_to_gm",
                dst.access_ptr('w', offset=dst_offset),
                data_res.access_ptr("r", offset=0),
                0, 1, burst_len_dst, 0, 0))

    with tvm_ib.if_scope(num_ub_mod > 0):
        col_group = num_ub_mod // row_zu_out
        col_mod = num_ub_mod % row_zu_out

        for num_zi in range(row_zu_out):
            with tvm_ib.if_scope(num_zi < num_ub_mod):
                data_offset = n_index * col_len * row_in\
                          + num_ub_loop * 16 * num_col_ub * row_in\
                          + num_zi * row_in + n_begin - (num_zi * row_out % 16)
                ub_offset = 0
                n_burst = tvm.select(num_zi < col_mod, col_group + 1,
                                     col_group)
                burst_len_data = row_out_block_t
                src_stride = (16 * row_in) // cp_align_len - burst_len_data
                dst_stride = 0
                args = tvm_ib, data, data_ub[num_zi], data_offset, ub_offset,\
                       n_burst, burst_len_data, src_stride,\
                       dst_stride, cp_align_len
                _func_gm_to_ub_align(args)

                mod_zu = tvm.select(num_zi < col_mod, col_group + 1,
                                    col_group)
                with tvm_ib.for_range(0, mod_zu, name="num_cu") as num_cu:
                    mask1, mask2 = _set_mask_slice_fp16(
                        0 + (num_zi * row_out % 16), 0)
                    tvm_ib.emit(tvm.call_extern(
                        dst.dtype, "set_vector_mask",
                        tvm.const(mask1, dtype="uint64"),
                        tvm.const(mask2, dtype="uint64")))
                    ub_offset = num_cu * row_out_align_t
                    res_offset = num_zi * row_out - (num_zi * row_out % 16)\
                                 + num_cu * 16 * row_out
                    repeat = 1
                    srcm0 = 1
                    dstm0 = 1
                    srcm1 = 0
                    dstm1 = 0
                    args = tvm_ib, data_ub[num_zi], data_res, \
                           ub_offset, res_offset, \
                           repeat, srcm0, dstm0, srcm1, dstm1, cp_align_len
                    _func_vadds(args)

                    mask1, mask2 = _set_mask_slice_fp16(0, 0)
                    tvm_ib.emit(tvm.call_extern(
                        dst.dtype, "set_vector_mask",
                        tvm.const(mask1, dtype="uint64"),
                        tvm.const(mask2, dtype="uint64")))
                    ub_offset = num_cu * row_out_align_t + 128
                    res_offset = num_zi * row_out - (num_zi * row_out % 16)\
                                 + num_cu * 16 * row_out + 128
                    repeat = 3
                    srcm0 = 1
                    dstm0 = 1
                    srcm1 = 8
                    dstm1 = 8
                    args = tvm_ib, data_ub[num_zi], data_res, \
                           ub_offset, res_offset, \
                           repeat, srcm0, dstm0, srcm1, dstm1, cp_align_len
                    _func_vadds(args)

                    mask1, mask2 = _set_mask_slice_fp16(0,
                                                        (5 * 128 - row_out
                                                         - (num_zi * row_out
                                                            % 16)))
                    tvm_ib.emit(tvm.call_extern(
                        dst.dtype, "set_vector_mask",
                        tvm.const(mask1, dtype="uint64"),
                        tvm.const(mask2, dtype="uint64")))
                    ub_offset = num_cu * row_out_align_t + 128 * 4
                    res_offset = num_zi * row_out - (num_zi * row_out % 16)\
                                 + num_cu * 16 * row_out + 128 * 4
                    repeat = 1
                    srcm0 = 1
                    dstm0 = 1
                    srcm1 = 0
                    dstm1 = 0
                    args = tvm_ib, data_ub[num_zi], data_res, \
                           ub_offset, res_offset, \
                           repeat, srcm0, dstm0, srcm1, dstm1, cp_align_len
                    _func_vadds(args)

                tvm_ib.emit(tvm.call_extern(
                    dst.dtype, "set_vector_mask",
                    tvm.const(-1, dtype="uint64"),
                    tvm.const(-1, dtype="uint64")))

        dst_offset = n_index * col_len * row_out \
                     + num_ub_loop * 16 * num_col_ub * row_out
        mod_len = num_ub_mod * row_out
        with tvm_ib.if_scope(mod_len % cp_align_len == 0):
            burst_len_dst = mod_len // cp_align_len
            tvm_ib.emit(
                tvm.call_extern(
                    dst.dtype, "copy_ubuf_to_gm",
                    dst.access_ptr('w', offset=dst_offset),
                    data_res.access_ptr("r", offset=0),
                    0, 1, burst_len_dst, 0, 0))
        with tvm_ib.else_scope():
            mod_len_align = mod_len - cp_align_len
            burst_len_dst_a = _ceil_div(mod_len_align, cp_align_len)
            tvm_ib.emit(
                tvm.call_extern(
                    dst.dtype, "copy_ubuf_to_gm",
                    dst.access_ptr('w', offset=dst_offset),
                    data_res.access_ptr("r", offset=0),
                    0, 1, burst_len_dst_a, 0, 0))
            with tvm_ib.for_range(0, cp_align_len, name="num_a") as num_a:
                tvm_ib.emit(tvm.call_extern(
                    data_res.dtype, "reg_mov",
                    tvm.call_extern(reg.dtype, "reg", reg[0]),
                    data_res.access_ptr('r',
                                        offset=mod_len_align + num_a)
                ))
                tvm_ib.emit(tvm.call_extern(
                    data_tail.dtype, "reg_mov",
                    data_tail.access_ptr('w',
                                         offset=num_a),
                    tvm.call_extern(reg.dtype, "reg", reg[0])
                ))
            tvm_ib.emit(
                tvm.call_extern(
                    dst.dtype, "copy_ubuf_to_gm",
                    dst.access_ptr('w', offset=dst_offset + mod_len_align),
                    data_tail.access_ptr("r", offset=0),
                    0, 1, 1, 0, 0))


def _move_sp_vadds_602_601_fp16(dst, data, n_begin):
    """
    function of making ir node builder for 602 601 float16 scene

    """
    tvm_ib = tvm.ir_builder.create()
    device_core_num = AICORE_NUM
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - 64
    ub_ele = (ub_bytes // float_size // cp_align_len) * cp_align_len
    ub_ele_half = ub_ele // 2
    ub_ele_eight = ub_ele // 32

    data_ub_one = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                             "data_ub_one", scope=cce.scope_ubuf)
    data_ub_two = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                             "data_ub_two", scope=cce.scope_ubuf)
    data_ub_three = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                               "data_ub_three", scope=cce.scope_ubuf)
    data_ub_four = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                              "data_ub_four", scope=cce.scope_ubuf)
    data_ub_five = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                              "data_ub_five", scope=cce.scope_ubuf)
    data_ub_six = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                             "data_ub_six", scope=cce.scope_ubuf)
    data_ub_seven = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                               "data_ub_seven", scope=cce.scope_ubuf)
    data_ub_eight = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                               "data_ub_eight", scope=cce.scope_ubuf)
    data_ub_nine = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                              "data_ub_nine", scope=cce.scope_ubuf)
    data_ub_ten = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                             "data_ub_ten", scope=cce.scope_ubuf)
    data_ub_eleven = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                                "data_ub_eleven", scope=cce.scope_ubuf)
    data_ub_twelve = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                                "data_ub_twelve", scope=cce.scope_ubuf)
    data_ub_thirteen = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                                  "data_ub_thirteen", scope=cce.scope_ubuf)
    data_ub_fourteen = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                                  "data_ub_fourteen", scope=cce.scope_ubuf)
    data_ub_fifteen = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                                 "data_ub_fifteen", scope=cce.scope_ubuf)
    data_ub_sixteen = _new_alloc(tvm_ib, dst.dtype, ub_ele_eight,
                                 "data_ub_sixteen", scope=cce.scope_ubuf)

    data_res = _new_alloc(tvm_ib, dst.dtype, ub_ele_half,
                          "data_res", scope=cce.scope_ubuf)
    data_tail = _new_alloc(tvm_ib, dst.dtype, cp_align_len,
                           "data_tail", scope=cce.scope_ubuf)
    reg = tvm_ib.allocate(dst.dtype, (8,), name='reg', scope=cce.scope_reg)

    data_ub = [data_ub_one, data_ub_two, data_ub_three, data_ub_four,
               data_ub_five, data_ub_six, data_ub_seven, data_ub_eight,
               data_ub_nine, data_ub_ten, data_ub_eleven, data_ub_twelve,
               data_ub_thirteen, data_ub_fourteen,
               data_ub_fifteen, data_ub_sixteen]

    n_i = data.shape[0]
    group_index = n_i // device_core_num
    group_mod = n_i % device_core_num
    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    with tvm_ib.for_range(0, group_index, name="num_g") as num_g:
        args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
               device_core_num, block_index, n_begin, ub_ele_eight, \
               cp_align_len, num_g
        _func_602_601_fp16(args)
    with tvm_ib.if_scope(group_mod > 0):
        with tvm_ib.if_scope(block_index < group_mod):
            args = tvm_ib, data, dst, data_ub, data_res, data_tail, reg, \
                   device_core_num, block_index, n_begin, ub_ele_eight, \
                   cp_align_len, group_index
            _func_602_601_fp16(args)

    return tvm_ib.get()


def _move_sp_vadds_32_32_4(dst, data, n_begin):
    """
    function of making ir node builder for 32 32 4 scene

    """
    tvm_ib = tvm.ir_builder.create()
    device_core_num = AICORE_NUM
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size

    col_len = 32
    row_in = 32
    row_out = 4
    data_ub_ele = col_len * row_in
    data_res_ele = col_len * row_out
    data_ub = _new_alloc(tvm_ib, dst.dtype, data_ub_ele,
                         "data_ub", scope=cce.scope_ubuf)
    data_res = _new_alloc(tvm_ib, dst.dtype, data_res_ele,
                          "data_res", scope=cce.scope_ubuf)
    reg = tvm_ib.allocate(dst.dtype, (8,), name='reg', scope=cce.scope_reg)

    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    data_offset = block_index * data_ub_ele
    burst_len_data = data_ub_ele // cp_align_len
    tvm_ib.emit(tvm.call_extern(data_ub.dtype, "copy_gm_to_ubuf",
                                data_ub.access_ptr("w", offset=0),
                                data.access_ptr('r',
                                                offset=data_offset),
                                0, 1, burst_len_data, 0, 0))

    for num_r in range(row_out):
        for num_cz in range(4):
            for num_c in range(8):
                tvm_ib.emit(tvm.call_extern(
                    data_ub.dtype, "reg_mov",
                    tvm.call_extern(reg.dtype, "reg", reg[num_c]),
                    data_ub.access_ptr('r', offset=((num_cz * 8 + num_c) * 32
                                                    + n_begin + num_r))
                ))
                tvm_ib.emit(tvm.call_extern(
                    data_res.dtype, "reg_mov",
                    data_res.access_ptr('w', offset=((num_cz * 8 + num_c) * 4
                                                     + num_r)),
                    tvm.call_extern(reg.dtype, "reg", reg[num_c])
                ))

    dst_offset = block_index * data_res_ele
    burst_len_dst = data_res_ele // cp_align_len
    tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                dst.access_ptr('w', offset=dst_offset),
                                data_res.access_ptr("r", offset=0),
                                0, 1, burst_len_dst, 0, 0))

    return tvm_ib.get()


def _func_same_small_dim(args):
    """
    function of moving data for after dims same scene

    """
    tvm_ib, data, dst, data_ub, data_tail, reg, dim_ele_in, bl_begin,\
    num_dim_one_ub, gap_ele, dim_ele_out, cp_align_len, col_begin, row_len,\
    num_u, num_bl_before_core, n_burst, burst_len_data, num_dim_cur_core = args

    data_offset = (bl_begin + num_bl_before_core) * dim_ele_in \
                  + num_u * num_dim_one_ub * dim_ele_in + col_begin * row_len

    with tvm_ib.if_scope(tvm.all(gap_ele % cp_align_len == 0,
                                 dim_ele_out % cp_align_len == 0)):
        with tvm_ib.if_scope(n_burst == 1):
            src_stride = 0
            dst_stride = 0
            args = tvm_ib, data, data_ub, data_offset, 0, n_burst, \
                   burst_len_data, src_stride, dst_stride, cp_align_len
            _func_gm_to_ub_align(args)
        with tvm_ib.else_scope():
            src_stride = _ceil_div(gap_ele, cp_align_len)
            dst_stride = 0
            args = tvm_ib, data, data_ub, data_offset, 0, n_burst, \
                   burst_len_data, src_stride, dst_stride, cp_align_len
            _func_gm_to_ub_align(args)

    with tvm_ib.else_scope():
        with tvm_ib.if_scope(dim_ele_out % cp_align_len == 0):
            with tvm_ib.for_range(0, n_burst, name="num_nb") as num_nb:
                data_offset_cur = data_offset + num_nb * dim_ele_in
                ub_offset_cur = num_nb * dim_ele_out
                tvm_ib.emit(
                    tvm.call_extern(
                        data_ub.dtype, "copy_gm_to_ubuf",
                        data_ub.access_ptr("w", offset=ub_offset_cur),
                        data.access_ptr('r', offset=data_offset_cur),
                        0, 1, burst_len_data, 0, 0))
        with tvm_ib.else_scope():
            with tvm_ib.for_range(0, n_burst, name="num_nb") as num_nb:
                data_offset_cur = data_offset + num_nb * dim_ele_in
                ub_offset_cur = num_nb * _ceil_fill(dim_ele_out, cp_align_len)
                tvm_ib.emit(
                    tvm.call_extern(
                        data_ub.dtype, "copy_gm_to_ubuf",
                        data_ub.access_ptr("w", offset=ub_offset_cur),
                        data.access_ptr('r', offset=data_offset_cur),
                        0, 1, burst_len_data, 0, 0))

    with tvm_ib.if_scope(dim_ele_out % cp_align_len == 0):
        dst_offset = num_bl_before_core * dim_ele_out\
                     + num_u * num_dim_one_ub * dim_ele_out
        burst_len_dst = (num_dim_cur_core * dim_ele_out) // cp_align_len
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w',
                                                   offset=dst_offset),
                                    data_ub.access_ptr("r", offset=0),
                                    0, 1, burst_len_dst, 0, 0))
    with tvm_ib.else_scope():
        with tvm_ib.for_range(0, num_dim_cur_core, name="num_d") as num_d:
            dst_offset_cur = num_bl_before_core * dim_ele_out\
                             + num_u * num_dim_one_ub * dim_ele_out\
                             + num_d * dim_ele_out
            ub_offset_cur = num_d * _ceil_fill(dim_ele_out, cp_align_len)
            dim_ele_out_before = dim_ele_out - cp_align_len
            burst_len_dst_cur = _ceil_div(dim_ele_out_before, cp_align_len)
            tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w',
                                                       offset=dst_offset_cur),
                                        data_ub.access_ptr(
                                            "r", offset=ub_offset_cur),
                                        0, 1, burst_len_dst_cur, 0, 0))
            with tvm_ib.for_range(0, cp_align_len, name="num_a") as num_a:
                tvm_ib.emit(tvm.call_extern(
                    data_ub.dtype, "reg_mov",
                    tvm.call_extern(reg.dtype, "reg", reg[0]),
                    data_ub.access_ptr(
                        'r', offset=(ub_offset_cur + dim_ele_out_before
                                     + num_a))
                ))
                tvm_ib.emit(tvm.call_extern(
                    data_tail.dtype, "reg_mov",
                    data_tail.access_ptr(
                        'w', offset=num_a),
                    tvm.call_extern(reg.dtype, "reg", reg[0])
                ))
            tvm_ib.emit(
                tvm.call_extern(
                    dst.dtype, "copy_ubuf_to_gm",
                    dst.access_ptr(
                        'w',
                        offset=dst_offset_cur + dim_ele_out_before),
                    data_tail.access_ptr("r", offset=0),
                    0, 1, 1, 0, 0))


def _move_sp_diff_same_small_dim(dst, data, bl_begin, col_begin):
    """
    function of making ir node builder for after dims same and small dim scene

    """
    tvm_ib = tvm.ir_builder.create()
    device_core_num = AICORE_NUM
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - 32
    ub_ele = (ub_bytes // float_size // cp_align_len) * cp_align_len

    data_ub = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                         "data_ub", scope=cce.scope_ubuf)
    data_tail = _new_alloc(tvm_ib, dst.dtype, cp_align_len,
                           "data_tail", scope=cce.scope_ubuf)
    reg = tvm_ib.allocate(dst.dtype, (8,), name='reg', scope=cce.scope_reg)

    _, col_in, row_len = data.shape
    bl_out, col_out, _ = dst.shape
    dim_ele_in = col_in * row_len
    dim_ele_out = col_out * row_len
    dim_ele_out_align = _ceil_fill(dim_ele_out, cp_align_len)
    num_dim_one_ub = ub_ele // dim_ele_out_align

    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    bl_small = bl_out // device_core_num
    bl_mod = bl_out % device_core_num
    bl_num_cur_core = tvm.select(block_index < bl_mod, bl_small + 1, bl_small)

    ub_loop = bl_num_cur_core // num_dim_one_ub
    dim_num_mod = bl_num_cur_core % num_dim_one_ub

    with tvm_ib.for_range(0, ub_loop, name="num_u") as num_u:
        gap_ele = (col_in - col_out) * row_len
        n_burst = num_dim_one_ub
        burst_len_data = _ceil_div(dim_ele_out, cp_align_len)
        with tvm_ib.if_scope(block_index <= bl_mod):
            num_bl_before_core = (bl_small + 1) * block_index
            args = tvm_ib, data, dst, data_ub, data_tail, reg,\
                   dim_ele_in, bl_begin,\
                   num_dim_one_ub, gap_ele, dim_ele_out, cp_align_len,\
                   col_begin, row_len, num_u, num_bl_before_core,\
                   n_burst, burst_len_data, num_dim_one_ub
            _func_same_small_dim(args)
        with tvm_ib.else_scope():
            num_bl_before_core = (bl_small + 1) * bl_mod\
                                 + bl_small * (block_index - bl_mod)
            args = tvm_ib, data, dst, data_ub, data_tail, reg,\
                   dim_ele_in, bl_begin,\
                   num_dim_one_ub, gap_ele, dim_ele_out, cp_align_len,\
                   col_begin, row_len, num_u, num_bl_before_core,\
                   n_burst, burst_len_data, num_dim_one_ub
            _func_same_small_dim(args)

    with tvm_ib.if_scope(dim_num_mod > 0):
        gap_ele = (col_in - col_out) * row_len
        n_burst = dim_num_mod
        burst_len_data = _ceil_div(dim_ele_out, cp_align_len)
        with tvm_ib.if_scope(block_index <= bl_mod):
            num_bl_before_core = (bl_small + 1) * block_index
            args = tvm_ib, data, dst, data_ub, data_tail, reg,\
                   dim_ele_in, bl_begin,\
                   num_dim_one_ub, gap_ele, dim_ele_out, cp_align_len,\
                   col_begin, row_len, ub_loop, num_bl_before_core,\
                   n_burst, burst_len_data, dim_num_mod
            _func_same_small_dim(args)
        with tvm_ib.else_scope():
            num_bl_before_core = (bl_small + 1) * bl_mod\
                                 + bl_small * (block_index - bl_mod)
            args = tvm_ib, data, dst, data_ub, data_tail, reg,\
                   dim_ele_in, bl_begin,\
                   num_dim_one_ub, gap_ele, dim_ele_out, cp_align_len,\
                   col_begin, row_len, ub_loop, num_bl_before_core,\
                   n_burst, burst_len_data, dim_num_mod
            _func_same_small_dim(args)

    return tvm_ib.get()


def _func_same_big_dim(args):
    """
    function of moving data for after dims same and big dim scene

    """
    tvm_ib, data, dst, data_ub, data_tail, reg, dim_ele_in, bl_begin, \
    dim_ele_out, cp_align_len, col_begin, row_len, \
    num_b, num_cu, num_bl_before_core, num_col_ub, num_col_cur_core = args

    move_len = num_col_cur_core * row_len
    data_offset = (bl_begin + num_bl_before_core + num_b) * dim_ele_in \
                  + (col_begin + num_cu * num_col_ub) * row_len
    burst_len_data = _ceil_div(move_len, cp_align_len)
    tvm_ib.emit(
        tvm.call_extern(
            data_ub.dtype, "copy_gm_to_ubuf",
            data_ub.access_ptr("w", offset=0),
            data.access_ptr('r', offset=data_offset),
            0, 1, burst_len_data, 0, 0))

    dst_offset = (num_bl_before_core + num_b) * dim_ele_out\
                 + num_cu * num_col_ub * row_len
    with tvm_ib.if_scope(move_len % cp_align_len == 0):
        burst_len_dst = move_len // cp_align_len
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w',
                                                   offset=dst_offset),
                                    data_ub.access_ptr(
                                        "r", offset=0),
                                    0, 1, burst_len_dst, 0, 0))
    with tvm_ib.else_scope():
        move_len_align = move_len - cp_align_len
        burst_len_dst_a = _ceil_div(move_len_align, cp_align_len)
        tvm_ib.emit(tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w',
                                                   offset=dst_offset),
                                    data_ub.access_ptr(
                                        "r", offset=0),
                                    0, 1, burst_len_dst_a, 0, 0))
        with tvm_ib.for_range(0, cp_align_len, name="num_a") as num_a:
            tvm_ib.emit(tvm.call_extern(
                data_ub.dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[0]),
                data_ub.access_ptr(
                    'r', offset=(move_len_align + num_a))
            ))
            tvm_ib.emit(tvm.call_extern(
                data_tail.dtype, "reg_mov",
                data_tail.access_ptr(
                    'w', offset=num_a),
                tvm.call_extern(reg.dtype, "reg", reg[0])
            ))
        tvm_ib.emit(
            tvm.call_extern(
                dst.dtype, "copy_ubuf_to_gm",
                dst.access_ptr(
                    'w',
                    offset=dst_offset + move_len_align),
                data_tail.access_ptr("r", offset=0),
                0, 1, 1, 0, 0))


def _move_sp_diff_same_big_dim(dst, data, bl_begin, col_begin):
    """
    function of making ir node builder for after dims same and big dim scene

    """
    tvm_ib = tvm.ir_builder.create()
    device_core_num = AICORE_NUM
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - 32
    ub_ele = (ub_bytes // float_size // cp_align_len) * cp_align_len

    data_ub = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                         "data_ub", scope=cce.scope_ubuf)
    data_tail = _new_alloc(tvm_ib, dst.dtype, cp_align_len,
                           "data_tail", scope=cce.scope_ubuf)
    reg = tvm_ib.allocate(dst.dtype, (8,), name='reg', scope=cce.scope_reg)

    _, col_in, row_len = data.shape
    bl_out, col_out, _ = dst.shape
    dim_ele_in = col_in * row_len
    dim_ele_out = col_out * row_len
    row_len_align = _ceil_fill(row_len, cp_align_len)

    num_col_ub = ub_ele // row_len_align
    col_ub_loop = col_out // num_col_ub
    col_ub_mod = col_out % num_col_ub

    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    bl_small = bl_out // device_core_num
    bl_mod = bl_out % device_core_num
    bl_num_cur_core = tvm.select(block_index < bl_mod, bl_small + 1, bl_small)

    with tvm_ib.for_range(0, bl_num_cur_core, name="num_b") as num_b:
        with tvm_ib.for_range(0, col_ub_loop, name="num_cu") as num_cu:
            with tvm_ib.if_scope(block_index <= bl_mod):
                num_bl_before_core = (bl_small + 1) * block_index
                args = tvm_ib, data, dst, data_ub, data_tail, reg,\
                       dim_ele_in, bl_begin, dim_ele_out, cp_align_len,\
                       col_begin, row_len, num_b, num_cu,\
                       num_bl_before_core, num_col_ub, num_col_ub
                _func_same_big_dim(args)
            with tvm_ib.else_scope():
                num_bl_before_core = (bl_small + 1) * bl_mod \
                                     + bl_small * (block_index - bl_mod)
                args = tvm_ib, data, dst, data_ub, data_tail, reg,\
                       dim_ele_in, bl_begin, dim_ele_out, cp_align_len,\
                       col_begin, row_len, num_b, num_cu,\
                       num_bl_before_core, num_col_ub, num_col_ub
                _func_same_big_dim(args)
        with tvm_ib.if_scope(col_ub_mod > 0):
            with tvm_ib.if_scope((col_ub_mod * row_len) >= cp_align_len):
                with tvm_ib.if_scope(block_index <= bl_mod):
                    num_bl_before_core = (bl_small + 1) * block_index
                    args = tvm_ib, data, dst, data_ub, data_tail, reg, \
                           dim_ele_in, bl_begin, dim_ele_out, cp_align_len, \
                           col_begin, row_len, num_b, col_ub_loop, \
                           num_bl_before_core, num_col_ub, col_ub_mod
                    _func_same_big_dim(args)
                with tvm_ib.else_scope():
                    num_bl_before_core = (bl_small + 1) * bl_mod \
                                         + bl_small * (block_index - bl_mod)
                    args = tvm_ib, data, dst, data_ub, data_tail, reg, \
                           dim_ele_in, bl_begin, dim_ele_out, cp_align_len, \
                           col_begin, row_len, num_b, col_ub_loop, \
                           num_bl_before_core, num_col_ub, col_ub_mod
                    _func_same_big_dim(args)
            with tvm_ib.else_scope():
                data_offset = (bl_begin + num_bl_before_core + num_b)\
                              * dim_ele_in \
                              + (col_begin + col_out) * row_len - cp_align_len
                tvm_ib.emit(
                    tvm.call_extern(
                        data_ub.dtype, "copy_gm_to_ubuf",
                        data_ub.access_ptr("w", offset=0),
                        data.access_ptr('r', offset=data_offset),
                        0, 1, 1, 0, 0))
                dst_offset = (num_bl_before_core + num_b) * dim_ele_out \
                             + col_out * row_len - cp_align_len
                tvm_ib.emit(
                    tvm.call_extern(
                        dst.dtype, "copy_ubuf_to_gm",
                        dst.access_ptr('w', offset=dst_offset),
                        data_ub.access_ptr("r", offset=0),
                        0, 1, 1, 0, 0))

    return tvm_ib.get()


def _func_two_diff_same_small_dim(args):
    """
    function of moving data for two dim diff and after dims same
    and small dim scene

    """
    tvm_ib, data, dst, data_ub, data_tail, reg, block_index,\
    col_in, col_out, row_len, num_dim_one_core_in, num_dim_one_core_out,\
    dim_ele_in, dim_ele_out, bl_in, bl_out, bl_begin, col_begin,\
    cp_align_len, num_dim_one_ub, num_bcol, num_du, num_dim_cur = args

    gap_ele = (col_in - col_out) * row_len
    data_offset = (block_index * num_dim_one_core_in
                   + num_bcol * bl_in
                   + bl_begin + num_du * num_dim_one_ub) * dim_ele_in\
                  + col_begin * row_len
    n_burst = num_dim_cur
    with tvm_ib.if_scope(tvm.all(dim_ele_out % cp_align_len == 0,
                                 gap_ele % cp_align_len == 0)):
        burst_len_data = dim_ele_out // cp_align_len
        src_stride = gap_ele // cp_align_len
        with tvm_ib.if_scope(n_burst == 1):
            args = tvm_ib, data, data_ub, data_offset, 0, n_burst, \
                   burst_len_data, 0, 0, cp_align_len
            _func_gm_to_ub_align(args)
        with tvm_ib.else_scope():
            args = tvm_ib, data, data_ub, data_offset, 0, n_burst, \
                   burst_len_data, src_stride, 0, cp_align_len
            _func_gm_to_ub_align(args)
    with tvm_ib.else_scope():
        with tvm_ib.if_scope(dim_ele_out % cp_align_len == 0):
            burst_len_data = dim_ele_out // cp_align_len
            with tvm_ib.for_range(0, n_burst, name="num_nb") as num_nb:
                data_offset_cur = data_offset + num_nb * dim_ele_in
                ub_offset_cur = num_nb * dim_ele_out
                tvm_ib.emit(
                    tvm.call_extern(
                        data_ub.dtype, "copy_gm_to_ubuf",
                        data_ub.access_ptr("w", offset=ub_offset_cur),
                        data.access_ptr('r', offset=data_offset_cur),
                        0, 1, burst_len_data, 0, 0))
        with tvm_ib.else_scope():
            burst_len_data = _ceil_div(dim_ele_out, cp_align_len)
            with tvm_ib.for_range(0, n_burst, name="num_nb") as num_nb:
                data_offset_cur = data_offset + num_nb * dim_ele_in
                ub_offset_cur = num_nb * _ceil_fill(dim_ele_out,
                                                    cp_align_len)
                tvm_ib.emit(
                    tvm.call_extern(
                        data_ub.dtype, "copy_gm_to_ubuf",
                        data_ub.access_ptr("w", offset=ub_offset_cur),
                        data.access_ptr('r', offset=data_offset_cur),
                        0, 1, burst_len_data, 0, 0))

    with tvm_ib.if_scope(dim_ele_out % cp_align_len == 0):
        dst_offset = (block_index * num_dim_one_core_out
                      + num_bcol * bl_out
                      + num_du * num_dim_one_ub) * dim_ele_out
        burst_len_dst = (num_dim_cur * dim_ele_out) // cp_align_len
        tvm_ib.emit(
            tvm.call_extern(
                dst.dtype, "copy_ubuf_to_gm",
                dst.access_ptr('w', offset=dst_offset),
                data_ub.access_ptr("r", offset=0),
                0, 1, burst_len_dst, 0, 0))
    with tvm_ib.else_scope():
        with tvm_ib.for_range(0, num_dim_cur, name="num_dc") as num_dc:
            dst_offset_cur = (block_index * num_dim_one_core_out
                              + num_bcol * bl_out
                              + num_du * num_dim_one_ub + num_dc) * dim_ele_out
            ub_offset_cur = num_dc * _ceil_fill(dim_ele_out, cp_align_len)
            dim_ele_out_before = dim_ele_out - cp_align_len
            burst_len_dst_cur = _ceil_div(dim_ele_out_before, cp_align_len)
            tvm_ib.emit(
                tvm.call_extern(
                    dst.dtype, "copy_ubuf_to_gm",
                    dst.access_ptr('w', offset=dst_offset_cur),
                    data_ub.access_ptr("r", offset=ub_offset_cur),
                    0, 1, burst_len_dst_cur, 0, 0))
            with tvm_ib.for_range(0, cp_align_len, name="num_a") as num_a:
                tvm_ib.emit(tvm.call_extern(
                    data_ub.dtype, "reg_mov",
                    tvm.call_extern(reg.dtype, "reg", reg[0]),
                    data_ub.access_ptr(
                        'r', offset=(ub_offset_cur + dim_ele_out_before
                                     + num_a))
                ))
                tvm_ib.emit(tvm.call_extern(
                    data_tail.dtype, "reg_mov",
                    data_tail.access_ptr(
                        'w', offset=num_a),
                    tvm.call_extern(reg.dtype, "reg", reg[0])
                ))
            tvm_ib.emit(
                tvm.call_extern(
                    dst.dtype, "copy_ubuf_to_gm",
                    dst.access_ptr(
                        'w', offset=dst_offset_cur + dim_ele_out_before),
                    data_tail.access_ptr("r", offset=0),
                    0, 1, 1, 0, 0))


def _move_two_diff_same_small_dim(dst, data, bl_begin, col_begin):
    """
    function of making ir node builder for two dim diff and after dims same
    and small dim scene

    """
    tvm_ib = tvm.ir_builder.create()
    device_core_num = AICORE_NUM
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - 32
    ub_ele = (ub_bytes // float_size // cp_align_len) * cp_align_len

    data_ub = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                         "data_ub", scope=cce.scope_ubuf)
    data_tail = _new_alloc(tvm_ib, dst.dtype, cp_align_len,
                           "data_tail", scope=cce.scope_ubuf)
    reg = tvm_ib.allocate(dst.dtype, (8,), name='reg', scope=cce.scope_reg)

    big_col, bl_in, col_in, row_len = data.shape
    _, bl_out, col_out, _ = dst.shape
    dim_ele_in = col_in * row_len
    dim_ele_out = col_out * row_len
    dim_ele_out_align = _ceil_fill(dim_ele_out, cp_align_len)
    num_dim_one_ub = ub_ele // dim_ele_out_align

    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    big_col_one_core = big_col // device_core_num
    num_dim_one_core_in = big_col_one_core * bl_in
    num_dim_one_core_out = big_col_one_core * bl_out

    dim_ub_loop = bl_out // num_dim_one_ub
    num_dim_mod = bl_out % num_dim_one_ub

    with tvm_ib.for_range(0, big_col_one_core, name="num_bcol") as num_bcol:
        with tvm_ib.for_range(0, dim_ub_loop, name="num_du") as num_du:
            args = tvm_ib, data, dst, data_ub, data_tail, reg, block_index,\
                   col_in, col_out, row_len, num_dim_one_core_in,\
                   num_dim_one_core_out, dim_ele_in, dim_ele_out,\
                   bl_in, bl_out, bl_begin, col_begin, cp_align_len,\
                   num_dim_one_ub, num_bcol, num_du, num_dim_one_ub
            _func_two_diff_same_small_dim(args)
        with tvm_ib.if_scope(num_dim_mod > 0):
            args = tvm_ib, data, dst, data_ub, data_tail, reg, block_index, \
                   col_in, col_out, row_len, num_dim_one_core_in, \
                   num_dim_one_core_out, dim_ele_in, dim_ele_out, \
                   bl_in, bl_out, bl_begin, col_begin, cp_align_len, \
                   num_dim_one_ub, num_bcol, dim_ub_loop, num_dim_mod
            _func_two_diff_same_small_dim(args)

    return tvm_ib.get()


def _func_two_diff_same_big_dim(args):
    """
    function of moving data for two dim diff and after dims same
    and big dim scene

    """
    tvm_ib, data, dst, data_ub, data_tail, reg, block_index,\
    num_dim_one_core_in, num_dim_one_core_out, bl_in, bl_out, row_len,\
    dim_ele_in, dim_ele_out, cp_align_len, bl_begin, col_begin,\
    num_col_one_ub, num_bcol, num_bo, num_cu, num_col_cur_core = args

    move_len = num_col_cur_core * row_len
    data_offset = (block_index * num_dim_one_core_in
                   + num_bcol * bl_in
                   + bl_begin + num_bo) * dim_ele_in \
                  + (col_begin + num_cu * num_col_one_ub) * row_len
    burst_len_data = _ceil_div(move_len, cp_align_len)
    tvm_ib.emit(
        tvm.call_extern(
            data_ub.dtype, "copy_gm_to_ubuf",
            data_ub.access_ptr("w", offset=0),
            data.access_ptr('r', offset=data_offset),
            0, 1, burst_len_data, 0, 0))

    dst_offset = (block_index * num_dim_one_core_out
                  + num_bcol * bl_out + num_bo) * dim_ele_out\
                 + (num_cu * num_col_one_ub) * row_len
    with tvm_ib.if_scope(move_len % cp_align_len == 0):
        burst_len_dst = move_len // cp_align_len
        tvm_ib.emit(
            tvm.call_extern(
                dst.dtype, "copy_ubuf_to_gm",
                dst.access_ptr('w', offset=dst_offset),
                data_ub.access_ptr("r", offset=0),
                0, 1, burst_len_dst, 0, 0))
    with tvm_ib.else_scope():
        move_len_align = move_len - cp_align_len
        burst_len_dst_a = _ceil_div(move_len_align, cp_align_len)
        tvm_ib.emit(
            tvm.call_extern(
                dst.dtype, "copy_ubuf_to_gm",
                dst.access_ptr('w', offset=dst_offset),
                data_ub.access_ptr("r", offset=0),
                0, 1, burst_len_dst_a, 0, 0))
        with tvm_ib.for_range(0, cp_align_len, name="num_a") as num_a:
            tvm_ib.emit(tvm.call_extern(
                data_ub.dtype, "reg_mov",
                tvm.call_extern(reg.dtype, "reg", reg[0]),
                data_ub.access_ptr(
                    'r', offset=(move_len_align + num_a))
            ))
            tvm_ib.emit(tvm.call_extern(
                data_tail.dtype, "reg_mov",
                data_tail.access_ptr(
                    'w', offset=num_a),
                tvm.call_extern(reg.dtype, "reg", reg[0])
            ))
        tvm_ib.emit(
            tvm.call_extern(
                dst.dtype, "copy_ubuf_to_gm",
                dst.access_ptr(
                    'w', offset=dst_offset + move_len_align),
                data_tail.access_ptr("r", offset=0),
                0, 1, 1, 0, 0))


def _move_two_diff_same_big_dim(dst, data, bl_begin, col_begin):
    """
    function of making ir node builder for two dim diff and after dims same
    and big dim scene

    """

    tvm_ib = tvm.ir_builder.create()
    device_core_num = AICORE_NUM
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - 32
    ub_ele = (ub_bytes // float_size // cp_align_len) * cp_align_len

    data_ub = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                         "data_ub", scope=cce.scope_ubuf)
    data_tail = _new_alloc(tvm_ib, dst.dtype, cp_align_len,
                           "data_tail", scope=cce.scope_ubuf)
    reg = tvm_ib.allocate(dst.dtype, (8,), name='reg', scope=cce.scope_reg)

    big_col, bl_in, col_in, row_len = data.shape
    _, bl_out, col_out, _ = dst.shape
    dim_ele_in = col_in * row_len
    dim_ele_out = col_out * row_len
    row_len_align = _ceil_fill(row_len, cp_align_len)

    big_col_one_core = big_col // device_core_num
    num_dim_one_core_in = big_col_one_core * bl_in
    num_dim_one_core_out = big_col_one_core * bl_out

    num_col_one_ub = ub_ele // row_len_align
    col_ub_loop = col_out // num_col_one_ub
    num_col_mod = col_out % num_col_one_ub

    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    with tvm_ib.for_range(0, big_col_one_core, name="num_bcol") as num_bcol:
        with tvm_ib.for_range(0, bl_out, name="num_bo") as num_bo:
            with tvm_ib.for_range(0, col_ub_loop, name="num_cu") as num_cu:
                args = tvm_ib, data, dst, data_ub, data_tail, reg,\
                       block_index, num_dim_one_core_in, num_dim_one_core_out,\
                       bl_in, bl_out, row_len, dim_ele_in, dim_ele_out,\
                       cp_align_len, bl_begin, col_begin, num_col_one_ub,\
                       num_bcol, num_bo, num_cu, num_col_one_ub
                _func_two_diff_same_big_dim(args)
            with tvm_ib.if_scope(num_col_mod > 0):
                with tvm_ib.if_scope((num_col_mod * row_len) >= cp_align_len):
                    args = tvm_ib, data, dst, data_ub, data_tail, reg, \
                           block_index, num_dim_one_core_in,\
                           num_dim_one_core_out, bl_in, bl_out, row_len,\
                           dim_ele_in, dim_ele_out, cp_align_len, bl_begin,\
                           col_begin, num_col_one_ub, num_bcol, num_bo,\
                           col_ub_loop, num_col_mod
                    _func_two_diff_same_big_dim(args)
                with tvm_ib.else_scope():
                    data_offset = (block_index * num_dim_one_core_in
                                   + num_bcol * bl_in
                                   + bl_begin + num_bo) * dim_ele_in\
                                   + (col_begin + col_out) * row_len\
                                   - cp_align_len
                    tvm_ib.emit(
                        tvm.call_extern(
                            data_ub.dtype, "copy_gm_to_ubuf",
                            data_ub.access_ptr("w", offset=0),
                            data.access_ptr('r', offset=data_offset),
                            0, 1, 1, 0, 0))
                    dst_offset = (block_index * num_dim_one_core_out
                                  + num_bcol * bl_out + num_bo) * dim_ele_out\
                                 + col_out * row_len - cp_align_len
                    tvm_ib.emit(
                        tvm.call_extern(
                            dst.dtype, "copy_ubuf_to_gm",
                            dst.access_ptr(
                                'w', offset=dst_offset),
                            data_ub.access_ptr("r", offset=0),
                            0, 1, 1, 0, 0))

    return tvm_ib.get()


def _vconv_one_fp16(args):
    """
    function of vnchwconv for _func_vconv_fp16

    """
    tvm_ib, addr_array, addr_array_buf, one_begin, two_begin,\
    repeat_vconv, src_stride_vconv, dst_stride_vconv, col_row_zu_in = args

    src0_offset = 8*0
    src1_offset = 8*1
    dst0_offset = 8*2
    dst1_offset = 8*3
    src_gap = 32 * col_row_zu_in
    src_eight_gap = src_gap*8
    dst_gap = 32
    dst_eight_gap = dst_gap*8

    with tvm_ib.for_range(0, 8, name="i") as i:
        tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                    tvm.call_extern(addr_array.dtype, "reg",
                                                    addr_array[
                                                        src0_offset + i]),
                                    one_begin + i * src_gap))
        tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                    tvm.call_extern(addr_array.dtype, "reg",
                                                    addr_array[
                                                        src1_offset + i]),
                                    one_begin + src_eight_gap
                                    + i * src_gap))
        tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                    tvm.call_extern(addr_array.dtype, "reg",
                                                    addr_array[
                                                        dst0_offset + i]),
                                    two_begin + i * dst_gap))
        tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                    tvm.call_extern(addr_array.dtype, "reg",
                                                    addr_array[
                                                        dst1_offset + i]),
                                    two_begin + dst_eight_gap
                                    + i * dst_gap))

    tvm_ib.emit(tvm.call_extern("int32",
                                "set_va_reg_sb",
                                "VA0",
                                addr_array_buf.access_ptr("rw",
                                                          offset=src0_offset)))
    tvm_ib.emit(tvm.call_extern("int32",
                                "set_va_reg_sb",
                                "VA1",
                                addr_array_buf.access_ptr("rw",
                                                          offset=src1_offset)))
    tvm_ib.emit(tvm.call_extern("int32",
                                "set_va_reg_sb",
                                "VA2",
                                addr_array_buf.access_ptr("rw",
                                                          offset=dst0_offset)))
    tvm_ib.emit(tvm.call_extern("int32",
                                "set_va_reg_sb",
                                "VA3",
                                addr_array_buf.access_ptr("rw",
                                                          offset=dst1_offset)))

    with tvm_ib.if_scope(repeat_vconv == 1):
        tvm_ib.emit(tvm.call_extern("int32",
                                    "scatter_vnchwconv_b16",
                                    "VA2",
                                    "VA0",
                                    1,
                                    0,
                                    0))
    with tvm_ib.else_scope():
        tvm_ib.emit(tvm.call_extern("int32",
                                    "scatter_vnchwconv_b16",
                                    "VA2",
                                    "VA0",
                                    repeat_vconv,
                                    dst_stride_vconv,
                                    src_stride_vconv))


def _vconv_two_fp16(args):
    """
    function of vnchwconv for _func_vconv_fp16

    """
    tvm_ib, addr_array, addr_array_buf, one_begin, two_begin,\
    repeat_vconv, src_stride_vconv, dst_stride_vconv, col_row_zu_out = args

    src0_offset = 8*0
    src1_offset = 8*1
    dst0_offset = 8*2
    dst1_offset = 8*3
    src_gap = 32
    src_eight_gap = src_gap*8
    dst_gap = 32 * col_row_zu_out
    dst_eight_gap = dst_gap*8

    with tvm_ib.for_range(0, 8, name="i") as i:
        tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                    tvm.call_extern(addr_array.dtype, "reg",
                                                    addr_array[
                                                        src0_offset + i]),
                                    one_begin + i * src_gap))
        tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                    tvm.call_extern(addr_array.dtype, "reg",
                                                    addr_array[
                                                        src1_offset + i]),
                                    one_begin + src_eight_gap
                                    + i * src_gap))
        tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                    tvm.call_extern(addr_array.dtype, "reg",
                                                    addr_array[
                                                        dst0_offset + i]),
                                    two_begin + i * dst_gap))
        tvm_ib.emit(tvm.call_extern("uint64", "reg_mov",
                                    tvm.call_extern(addr_array.dtype, "reg",
                                                    addr_array[
                                                        dst1_offset + i]),
                                    two_begin + dst_eight_gap
                                    + i * dst_gap))

    tvm_ib.emit(tvm.call_extern("int32",
                                "set_va_reg_sb",
                                "VA0",
                                addr_array_buf.access_ptr("rw",
                                                          offset=src0_offset)))
    tvm_ib.emit(tvm.call_extern("int32",
                                "set_va_reg_sb",
                                "VA1",
                                addr_array_buf.access_ptr("rw",
                                                          offset=src1_offset)))
    tvm_ib.emit(tvm.call_extern("int32",
                                "set_va_reg_sb",
                                "VA2",
                                addr_array_buf.access_ptr("rw",
                                                          offset=dst0_offset)))
    tvm_ib.emit(tvm.call_extern("int32",
                                "set_va_reg_sb",
                                "VA3",
                                addr_array_buf.access_ptr("rw",
                                                          offset=dst1_offset)))

    with tvm_ib.if_scope(repeat_vconv == 1):
        tvm_ib.emit(tvm.call_extern("int32",
                                    "scatter_vnchwconv_b16",
                                    "VA2",
                                    "VA0",
                                    1,
                                    0,
                                    0))
    with tvm_ib.else_scope():
        tvm_ib.emit(tvm.call_extern("int32",
                                    "scatter_vnchwconv_b16",
                                    "VA2",
                                    "VA0",
                                    repeat_vconv,
                                    dst_stride_vconv,
                                    src_stride_vconv))


def _small_last_32_512_2(dst, data, n_begin):
    """
    function of making ir node builder for before dims same
    and last dim smaller than one block scene

    """
    tvm_ib = tvm.ir_builder.create()
    device_core_num = AICORE_NUM
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - 64
    ub_ele = (ub_bytes // 2 // float_size // cp_align_len) * cp_align_len

    data_one = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                          "data_one", scope=cce.scope_ubuf)
    data_two = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                          "data_two", scope=cce.scope_ubuf)
    addr_array = tvm_ib.allocate("uint64", (32,), name="addr_array",
                                 scope=cce.scope_reg)
    addr_array_buf = tvm.decl_buffer((32,), "uint64_t", "addr_array_buf",
                                     scope=cce.scope_reg,
                                     data=addr_array)

    c_0 = 16
    n_i, col_len, row_in = data.shape
    row_out = dst.shape[2]
    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)
    n_zu = _ceil_fill(n_i, c_0) // c_0

    with tvm_ib.if_scope(block_index < n_zu):
        _clean_ubuf(tvm_ib, data_one, 0, ub_ele)
        _clean_ubuf(tvm_ib, data_two, 0, ub_ele)
        data_offset = block_index * (tvm.min(c_0, n_i) * col_len * row_in)
        burst_len_data = (tvm.min(c_0, n_i) * col_len * row_in) // cp_align_len
        tvm_ib.emit(tvm.call_extern(data_one.dtype, "copy_gm_to_ubuf",
                                    data_one.access_ptr("w", offset=0),
                                    data.access_ptr('r', offset=data_offset),
                                    0, 1, burst_len_data, 0, 0))

        one_begin = 0
        two_begin = ub_ele * float_size
        repeat_vconv = (col_len * row_in) // cp_align_len
        src_stride_vconv = 1
        dst_stride_vconv = 16
        col_row_zu_in = (col_len * row_in) // cp_align_len
        args = tvm_ib, addr_array, addr_array_buf, one_begin, two_begin, \
               repeat_vconv, src_stride_vconv, dst_stride_vconv, col_row_zu_in
        _vconv_one_fp16(args)

        two_offset = n_begin * c_0
        n_burst = col_len
        burst_len = c_0 // cp_align_len
        src_stride = c_0 // cp_align_len
        dst_stride = 0
        tvm_ib.emit(
            tvm.call_extern(data_one.dtype, "copy_ubuf_to_ubuf",
                            data_one.access_ptr(
                                "w", offset=0),
                            data_two.access_ptr(
                                "r", offset=two_offset),
                            0, n_burst, burst_len,
                            src_stride, dst_stride))

        one_begin = 0
        two_begin = ub_ele * float_size
        repeat_vconv = (col_len * row_out) // cp_align_len
        src_stride_vconv = 16
        dst_stride_vconv = 1
        col_row_zu_out = (col_len * row_out) // cp_align_len
        args = tvm_ib, addr_array, addr_array_buf, one_begin, two_begin, \
               repeat_vconv, src_stride_vconv, dst_stride_vconv, col_row_zu_out
        _vconv_two_fp16(args)

        dst_offset = block_index * (col_len * row_out) * tvm.min(n_i, c_0)
        burst_len_dst = (col_len * row_out) * tvm.min(n_i, c_0) // cp_align_len
        tvm_ib.emit(
            tvm.call_extern(
                dst.dtype, "copy_ubuf_to_gm",
                dst.access_ptr(
                    'w', offset=dst_offset),
                data_two.access_ptr("r", offset=0),
                0, 1, burst_len_dst, 0, 0))

    return tvm_ib.get()


def _small_last_32_1024_2_fp32(dst, data, n_begin):
    """
    function of making ir node builder for before dims same
    and last dim smaller than one block scene

    """
    tvm_ib = tvm.ir_builder.create()
    device_core_num = AICORE_NUM
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - 64
    ub_ele = (ub_bytes // 2 // float_size // cp_align_len) * cp_align_len

    data_one = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                          "data_one", scope=cce.scope_ubuf)
    data_two = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                          "data_two", scope=cce.scope_ubuf)
    addr_array = tvm_ib.allocate("uint64", (32,), name="addr_array",
                                 scope=cce.scope_reg)
    addr_array_buf = tvm.decl_buffer((32,), "uint64_t", "addr_array_buf",
                                     scope=cce.scope_reg,
                                     data=addr_array)

    c_0 = 16
    n_i, col_len, row_in = data.shape
    col_len_half = col_len // 2
    row_out = dst.shape[2]
    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)
    n_zu = _ceil_fill(n_i, c_0) // c_0

    with tvm_ib.if_scope(block_index < (2 * n_zu)):
        _clean_ubuf(tvm_ib, data_one, 0, ub_ele)
        _clean_ubuf(tvm_ib, data_two, 0, ub_ele)

        data_offset = (block_index // 2) * (tvm.min(n_i, c_0)
                                            * col_len * row_in)\
                      + (block_index % 2) * col_len_half * row_in
        burst_len_data = (col_len_half * row_in) // cp_align_len
        n_burst = tvm.min(n_i, c_0)
        src_stride = (col_len_half * row_in) // cp_align_len
        tvm_ib.emit(tvm.call_extern(data_one.dtype, "copy_gm_to_ubuf",
                                    data_one.access_ptr("w", offset=0),
                                    data.access_ptr('r', offset=data_offset),
                                    0, n_burst, burst_len_data, src_stride, 0))

        one_begin = 0
        two_begin = ub_ele * float_size
        repeat_vconv = (col_len_half * row_in) // cp_align_len
        src_stride_vconv = 1
        dst_stride_vconv = 16
        col_row_zu_in = (col_len_half * row_in) // cp_align_len
        args = tvm_ib, addr_array, addr_array_buf, one_begin, two_begin, \
               repeat_vconv, src_stride_vconv, dst_stride_vconv, col_row_zu_in
        _vconv_one_fp16(args)

        two_offset = n_begin * c_0
        n_burst = col_len_half
        burst_len = c_0 // cp_align_len
        src_stride = c_0 // cp_align_len
        dst_stride = 0
        tvm_ib.emit(
            tvm.call_extern(data_one.dtype, "copy_ubuf_to_ubuf",
                            data_one.access_ptr(
                                "w", offset=0),
                            data_two.access_ptr(
                                "r", offset=two_offset),
                            0, n_burst, burst_len,
                            src_stride, dst_stride))

        one_begin = 0
        two_begin = ub_ele * float_size
        repeat_vconv = (col_len_half * row_out) // cp_align_len
        src_stride_vconv = 16
        dst_stride_vconv = 1
        col_row_zu_out = (col_len_half * row_out) // cp_align_len
        args = tvm_ib, addr_array, addr_array_buf, one_begin, two_begin, \
               repeat_vconv, src_stride_vconv, dst_stride_vconv, col_row_zu_out
        _vconv_two_fp16(args)

        dst_offset = (block_index // 2) * (col_len * row_out)\
                     * tvm.min(n_i, c_0)\
                     + (block_index % 2) * (col_len_half * row_out)
        n_burst = tvm.min(n_i, c_0)
        burst_len_dst = (col_len_half * row_out) // cp_align_len
        dst_stride = (col_len_half * row_out) // cp_align_len
        tvm_ib.emit(
            tvm.call_extern(
                dst.dtype, "copy_ubuf_to_gm",
                dst.access_ptr(
                    'w', offset=dst_offset),
                data_two.access_ptr("r", offset=0),
                0, n_burst, burst_len_dst, 0, dst_stride))

    return tvm_ib.get()


def _small_last_32_1917_2(dst, data, n_begin):
    """
    function of making ir node builder for before dims same
    and last dim smaller than one block scene

    """
    tvm_ib = tvm.ir_builder.create()
    device_core_num = AICORE_NUM
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - 64
    ub_ele = (ub_bytes // 2 // float_size // cp_align_len) * cp_align_len

    data_one = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                          "data_one", scope=cce.scope_ubuf)
    data_two = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                          "data_two", scope=cce.scope_ubuf)
    data_tail = _new_alloc(tvm_ib, dst.dtype, cp_align_len,
                           "data_tail", scope=cce.scope_ubuf)
    addr_array = tvm_ib.allocate("uint64", (32,), name="addr_array",
                                 scope=cce.scope_reg)
    addr_array_buf = tvm.decl_buffer((32,), "uint64_t", "addr_array_buf",
                                     scope=cce.scope_reg,
                                     data=addr_array)
    reg = tvm_ib.allocate(dst.dtype, (8,), name='reg', scope=cce.scope_reg)
    reg_addr = tvm_ib.allocate("int32", (8,), name='reg_addr',
                               scope=cce.scope_reg)

    c_0 = 16
    _, col_len, row_in = data.shape
    col_len_qua = _ceil_fill(col_len, 512) // 4
    row_out = dst.shape[2]
    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    with tvm_ib.if_scope(block_index < 8):
        _clean_ubuf(tvm_ib, data_one, 0, ub_ele)
        _clean_ubuf(tvm_ib, data_two, 0, ub_ele)
        zong_col_len = tvm.min((block_index % 4 + 1) * col_len_qua, col_len)
        move_col_len = zong_col_len - (block_index % 4) * col_len_qua
        reg_addr[0] = move_col_len
        mov_align = _ceil_fill(move_col_len * row_in, c_0)
        with tvm_ib.for_range(0, c_0, name="num_c") as num_c:
            data_offset = (block_index // 4) * (c_0 * col_len * row_in) \
                          + (block_index % 4) * col_len_qua * row_in\
                          + num_c * col_len * row_in
            ub_offset = num_c * mov_align
            burst_len_data = _ceil_div(move_col_len * row_in, cp_align_len)
            tvm_ib.emit(tvm.call_extern(data_one.dtype, "copy_gm_to_ubuf",
                                        data_one.access_ptr(
                                            "w", offset=ub_offset),
                                        data.access_ptr(
                                            "r", offset=data_offset),
                                        0, 1, burst_len_data, 0, 0))

        one_begin = 0
        two_begin = ub_ele * float_size
        repeat_vconv = mov_align // cp_align_len
        src_stride_vconv = 1
        dst_stride_vconv = 16
        col_row_zu_in = mov_align // cp_align_len
        args = tvm_ib, addr_array, addr_array_buf, one_begin, two_begin, \
               repeat_vconv, src_stride_vconv, dst_stride_vconv, col_row_zu_in
        _vconv_one_fp16(args)

        two_offset = n_begin * c_0
        n_burst = mov_align // row_in
        burst_len = c_0 // cp_align_len
        src_stride = c_0 // cp_align_len
        dst_stride = 0
        tvm_ib.emit(
            tvm.call_extern(data_one.dtype, "copy_ubuf_to_ubuf",
                            data_one.access_ptr(
                                "w", offset=0),
                            data_two.access_ptr(
                                "r", offset=two_offset),
                            0, n_burst, burst_len,
                            src_stride, dst_stride))

        one_begin = 0
        two_begin = ub_ele * float_size
        repeat_vconv = mov_align // row_in // cp_align_len
        src_stride_vconv = 16
        dst_stride_vconv = 1
        col_row_zu_out = mov_align // row_in // cp_align_len
        args = tvm_ib, addr_array, addr_array_buf, one_begin, two_begin, \
               repeat_vconv, src_stride_vconv, dst_stride_vconv, col_row_zu_out
        _vconv_two_fp16(args)

        with tvm_ib.if_scope(block_index % 4 >= 3):
            with tvm_ib.for_range(0, c_0, name="num_c") as num_c:
                dst_offset = (block_index // 4) * (c_0 * col_len * row_out) \
                             + (block_index % 4) * col_len_qua * row_out \
                             + num_c * col_len * row_out
                two_offset = num_c * (mov_align // row_in)
                move_len_dst = col_len - 3 * col_len_qua
                move_len_dst_align = move_len_dst - cp_align_len
                burst_len_align = _ceil_div(move_len_dst_align, cp_align_len)
                tvm_ib.emit(
                    tvm.call_extern(
                        dst.dtype, "copy_ubuf_to_gm",
                        dst.access_ptr(
                            'w', offset=dst_offset),
                        data_two.access_ptr("r", offset=two_offset),
                        0, 1, burst_len_align, 0, 0))
                with tvm_ib.for_range(0, cp_align_len, name="num_a") as num_a:
                    tvm_ib.emit(tvm.call_extern(
                        data_two.dtype, "reg_mov",
                        tvm.call_extern(reg.dtype, "reg", reg[0]),
                        data_two.access_ptr(
                            'r',
                            offset=two_offset + move_len_dst_align + num_a)
                    ))
                    tvm_ib.emit(tvm.call_extern(
                        data_tail.dtype, "reg_mov",
                        data_tail.access_ptr('w', offset=num_a),
                        tvm.call_extern(reg.dtype, "reg", reg[0])
                    ))
                tvm_ib.emit(
                    tvm.call_extern(
                        dst.dtype, "copy_ubuf_to_gm",
                        dst.access_ptr(
                            'w', offset=dst_offset + move_len_dst_align),
                        data_tail.access_ptr("r", offset=0),
                        0, 1, 1, 0, 0))
        with tvm_ib.else_scope():
            with tvm_ib.for_range(0, c_0, name="num_c") as num_c:
                dst_offset = (block_index // 4) * (c_0 * col_len * row_out)\
                             + (block_index % 4) * col_len_qua * row_out\
                             + num_c * col_len * row_out
                two_offset = num_c * col_len_qua * row_out
                burst_len_dst = col_len_qua * row_out // cp_align_len
                tvm_ib.emit(
                    tvm.call_extern(
                        dst.dtype, "copy_ubuf_to_gm",
                        dst.access_ptr(
                            'w', offset=dst_offset),
                        data_two.access_ptr("r", offset=two_offset),
                        0, 1, burst_len_dst, 0, 0))

    return tvm_ib.get()


def _small_last_65472_2(dst, data, n_begin):
    """
    function of making ir node builder for before dims same
    and last dim smaller than one block scene

    """
    tvm_ib = tvm.ir_builder.create()
    device_core_num = AICORE_NUM
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - 64
    ub_ele = (ub_bytes // 2 // float_size // cp_align_len) * cp_align_len

    data_one = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                          "data_one", scope=cce.scope_ubuf)
    data_two = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                          "data_two", scope=cce.scope_ubuf)
    data_tail = _new_alloc(tvm_ib, dst.dtype, cp_align_len,
                           "data_tail", scope=cce.scope_ubuf)
    addr_array = tvm_ib.allocate("uint64", (32,), name="addr_array",
                                 scope=cce.scope_reg)
    addr_array_buf = tvm.decl_buffer((32,), "uint64_t", "addr_array_buf",
                                     scope=cce.scope_reg,
                                     data=addr_array)
    reg = tvm_ib.allocate(dst.dtype, (8,), name='reg', scope=cce.scope_reg)
    reg_addr = tvm_ib.allocate("int32", (8,), name='reg_addr',
                               scope=cce.scope_reg)

    c_0 = 16
    _, col_len, row_in = data.shape
    col_len_qua = _ceil_fill(col_len, 512) // 8
    row_out = dst.shape[2]
    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    with tvm_ib.if_scope(block_index < 8):
        _clean_ubuf(tvm_ib, data_one, 0, ub_ele)
        _clean_ubuf(tvm_ib, data_two, 0, ub_ele)

        zong_col_len = tvm.min((block_index + 1) * col_len_qua, col_len)
        move_col_len = zong_col_len - block_index * col_len_qua
        reg_addr[0] = move_col_len
        mov_align = _ceil_fill(move_col_len * row_in, c_0)
        with tvm_ib.for_range(0, c_0, name="num_c") as num_c:
            data_offset = block_index * col_len_qua * row_in\
                          + num_c * col_len * row_in
            ub_offset = num_c * mov_align
            burst_len_data = _ceil_div(move_col_len * row_in, cp_align_len)
            tvm_ib.emit(tvm.call_extern(data_one.dtype, "copy_gm_to_ubuf",
                                        data_one.access_ptr(
                                            "w", offset=ub_offset),
                                        data.access_ptr(
                                            "r", offset=data_offset),
                                        0, 1, burst_len_data, 0, 0))

        one_begin = 0
        two_begin = ub_ele * float_size
        repeat_vconv = mov_align // cp_align_len
        src_stride_vconv = 1
        dst_stride_vconv = 16
        col_row_zu_in = mov_align // cp_align_len
        args = tvm_ib, addr_array, addr_array_buf, one_begin, two_begin, \
               repeat_vconv, src_stride_vconv, dst_stride_vconv, col_row_zu_in
        _vconv_one_fp16(args)

        two_offset = n_begin * c_0
        n_burst = mov_align // row_in
        burst_len = c_0 // cp_align_len
        src_stride = c_0 // cp_align_len
        dst_stride = 0
        tvm_ib.emit(
            tvm.call_extern(data_one.dtype, "copy_ubuf_to_ubuf",
                            data_one.access_ptr(
                                "w", offset=0),
                            data_two.access_ptr(
                                "r", offset=two_offset),
                            0, n_burst, burst_len,
                            src_stride, dst_stride))

        one_begin = 0
        two_begin = ub_ele * float_size
        repeat_vconv = mov_align // row_in // cp_align_len
        src_stride_vconv = 16
        dst_stride_vconv = 1
        col_row_zu_out = mov_align // row_in // cp_align_len
        args = tvm_ib, addr_array, addr_array_buf, one_begin, two_begin, \
               repeat_vconv, src_stride_vconv, dst_stride_vconv, col_row_zu_out
        _vconv_two_fp16(args)

        with tvm_ib.if_scope(block_index >= 7):
            with tvm_ib.for_range(0, c_0, name="num_c") as num_c:
                dst_offset = block_index * col_len_qua * row_out\
                             + num_c * col_len * row_out
                two_offset = num_c * (mov_align // row_in)
                move_len_dst = col_len - 7 * col_len_qua
                move_len_dst_align = move_len_dst - cp_align_len
                burst_len_align = _ceil_div(move_len_dst_align, cp_align_len)
                tvm_ib.emit(
                    tvm.call_extern(
                        dst.dtype, "copy_ubuf_to_gm",
                        dst.access_ptr(
                            'w', offset=dst_offset),
                        data_two.access_ptr("r", offset=two_offset),
                        0, 1, burst_len_align, 0, 0))
                with tvm_ib.for_range(0, cp_align_len, name="num_a") as num_a:
                    tvm_ib.emit(tvm.call_extern(
                        data_two.dtype, "reg_mov",
                        tvm.call_extern(reg.dtype, "reg", reg[0]),
                        data_two.access_ptr(
                            'r',
                            offset=two_offset + move_len_dst_align + num_a)
                    ))
                    tvm_ib.emit(tvm.call_extern(
                        data_tail.dtype, "reg_mov",
                        data_tail.access_ptr('w', offset=num_a),
                        tvm.call_extern(reg.dtype, "reg", reg[0])
                    ))
                tvm_ib.emit(
                    tvm.call_extern(
                        dst.dtype, "copy_ubuf_to_gm",
                        dst.access_ptr(
                            'w', offset=dst_offset + move_len_dst_align),
                        data_tail.access_ptr("r", offset=0),
                        0, 1, 1, 0, 0))
        with tvm_ib.else_scope():
            with tvm_ib.for_range(0, c_0, name="num_c") as num_c:
                dst_offset = block_index * col_len_qua * row_out\
                             + num_c * col_len * row_out
                two_offset = num_c * col_len_qua * row_out
                burst_len_dst = col_len_qua * row_out // cp_align_len
                tvm_ib.emit(
                    tvm.call_extern(
                        dst.dtype, "copy_ubuf_to_gm",
                        dst.access_ptr(
                            'w', offset=dst_offset),
                        data_two.access_ptr("r", offset=two_offset),
                        0, 1, burst_len_dst, 0, 0))

    return tvm_ib.get()


def _small_last_65472_4(dst, data, n_begin):
    """
    function of making ir node builder for before dims same
    and last dim smaller than one block scene

    """
    tvm_ib = tvm.ir_builder.create()
    device_core_num = AICORE_NUM
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - 64
    ub_ele = (ub_bytes // 2 // float_size // cp_align_len) * cp_align_len

    data_one = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                          "data_one", scope=cce.scope_ubuf)
    data_two = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                          "data_two", scope=cce.scope_ubuf)
    data_tail = _new_alloc(tvm_ib, dst.dtype, cp_align_len,
                           "data_tail", scope=cce.scope_ubuf)
    addr_array = tvm_ib.allocate("uint64", (32,), name="addr_array",
                                 scope=cce.scope_reg)
    addr_array_buf = tvm.decl_buffer((32,), "uint64_t", "addr_array_buf",
                                     scope=cce.scope_reg,
                                     data=addr_array)
    reg = tvm_ib.allocate(dst.dtype, (8,), name='reg', scope=cce.scope_reg)

    c_0 = 16
    _, col_len, row_in = data.shape
    col_len_qua = _ceil_fill(col_len, 256) // 16
    row_out = dst.shape[2]
    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)

    with tvm_ib.if_scope(block_index < 16):
        _clean_ubuf(tvm_ib, data_one, 0, ub_ele)
        _clean_ubuf(tvm_ib, data_two, 0, ub_ele)
        with tvm_ib.if_scope(block_index < 15):
            move_col_len = col_len_qua
            mov_align = _ceil_fill(move_col_len * row_in, c_0)
            with tvm_ib.for_range(0, c_0, name="num_c") as num_c:
                data_offset = block_index * col_len_qua * row_in \
                              + num_c * col_len * row_in
                ub_offset = num_c * mov_align
                burst_len_data = _ceil_div(move_col_len * row_in, cp_align_len)
                tvm_ib.emit(tvm.call_extern(data_one.dtype, "copy_gm_to_ubuf",
                                            data_one.access_ptr(
                                                "w", offset=ub_offset),
                                            data.access_ptr(
                                                "r", offset=data_offset),
                                            0, 1, burst_len_data, 0, 0))

            one_begin = 0
            two_begin = ub_ele * float_size
            repeat_vconv = mov_align // cp_align_len
            src_stride_vconv = 1
            dst_stride_vconv = 16
            col_row_zu_in = mov_align // cp_align_len
            args = tvm_ib, addr_array, addr_array_buf, one_begin, two_begin,\
                   repeat_vconv, src_stride_vconv, dst_stride_vconv,\
                   col_row_zu_in
            _vconv_one_fp16(args)

            two_offset = n_begin * c_0
            n_burst = mov_align // row_in
            burst_len = c_0 // cp_align_len
            src_stride = (c_0 // cp_align_len) * 3
            dst_stride = 0
            tvm_ib.emit(
                tvm.call_extern(data_one.dtype, "copy_ubuf_to_ubuf",
                                data_one.access_ptr(
                                    "w", offset=0),
                                data_two.access_ptr(
                                    "r", offset=two_offset),
                                0, n_burst, burst_len,
                                src_stride, dst_stride))

            one_begin = 0
            two_begin = ub_ele * float_size
            repeat_vconv = mov_align // row_in // cp_align_len
            src_stride_vconv = 16
            dst_stride_vconv = 1
            col_row_zu_out = mov_align // row_in // cp_align_len
            args = tvm_ib, addr_array, addr_array_buf, one_begin, two_begin,\
                   repeat_vconv, src_stride_vconv, dst_stride_vconv,\
                   col_row_zu_out
            _vconv_two_fp16(args)

            with tvm_ib.for_range(0, c_0, name="num_c") as num_c:
                dst_offset = block_index * col_len_qua * row_out \
                             + num_c * col_len * row_out
                two_offset = num_c * col_len_qua * row_out
                burst_len_dst = col_len_qua * row_out // cp_align_len
                tvm_ib.emit(
                    tvm.call_extern(
                        dst.dtype, "copy_ubuf_to_gm",
                        dst.access_ptr(
                            'w', offset=dst_offset),
                        data_two.access_ptr("r", offset=two_offset),
                        0, 1, burst_len_dst, 0, 0))
        with tvm_ib.else_scope():
            move_col_len = col_len - 15 * col_len_qua
            mov_align = _ceil_fill(move_col_len * row_in, c_0)
            with tvm_ib.for_range(0, c_0, name="num_c") as num_c:
                data_offset = block_index * col_len_qua * row_in \
                              + num_c * col_len * row_in
                ub_offset = num_c * mov_align
                burst_len_data = _ceil_div(move_col_len * row_in, cp_align_len)
                tvm_ib.emit(tvm.call_extern(data_one.dtype, "copy_gm_to_ubuf",
                                            data_one.access_ptr(
                                                "w", offset=ub_offset),
                                            data.access_ptr(
                                                "r", offset=data_offset),
                                            0, 1, burst_len_data, 0, 0))

            one_begin = 0
            two_begin = ub_ele * float_size
            repeat_vconv = mov_align // cp_align_len
            src_stride_vconv = 1
            dst_stride_vconv = 16
            col_row_zu_in = mov_align // cp_align_len
            args = tvm_ib, addr_array, addr_array_buf, one_begin, two_begin,\
                   repeat_vconv, src_stride_vconv, dst_stride_vconv,\
                   col_row_zu_in
            _vconv_one_fp16(args)

            two_offset = n_begin * c_0
            n_burst = mov_align // row_in
            burst_len = c_0 // cp_align_len
            src_stride = (c_0 // cp_align_len) * 3
            dst_stride = 0
            tvm_ib.emit(
                tvm.call_extern(data_one.dtype, "copy_ubuf_to_ubuf",
                                data_one.access_ptr(
                                    "w", offset=0),
                                data_two.access_ptr(
                                    "r", offset=two_offset),
                                0, n_burst, burst_len,
                                src_stride, dst_stride))

            move_out = _ceil_fill(mov_align // row_in, c_0)
            one_begin = 0
            two_begin = ub_ele * float_size
            repeat_vconv = move_out // cp_align_len
            src_stride_vconv = 16
            dst_stride_vconv = 1
            col_row_zu_out = move_out // cp_align_len
            args = tvm_ib, addr_array, addr_array_buf, one_begin, two_begin,\
                   repeat_vconv, src_stride_vconv, dst_stride_vconv,\
                   col_row_zu_out
            _vconv_two_fp16(args)

            with tvm_ib.for_range(0, c_0, name="num_c") as num_c:
                dst_offset = block_index * col_len_qua * row_out \
                             + num_c * col_len * row_out
                two_offset = num_c * move_out
                move_len_dst = col_len - 15 * col_len_qua
                move_len_dst_align = move_len_dst - cp_align_len
                burst_len_align = _ceil_div(move_len_dst_align, cp_align_len)
                tvm_ib.emit(
                    tvm.call_extern(
                        dst.dtype, "copy_ubuf_to_gm",
                        dst.access_ptr(
                            'w', offset=dst_offset),
                        data_two.access_ptr("r", offset=two_offset),
                        0, 1, burst_len_align, 0, 0))
                with tvm_ib.for_range(0, cp_align_len, name="num_a") as num_a:
                    tvm_ib.emit(tvm.call_extern(
                        data_two.dtype, "reg_mov",
                        tvm.call_extern(reg.dtype, "reg", reg[0]),
                        data_two.access_ptr(
                            'r',
                            offset=two_offset + move_len_dst_align + num_a)
                    ))
                    tvm_ib.emit(tvm.call_extern(
                        data_tail.dtype, "reg_mov",
                        data_tail.access_ptr('w', offset=num_a),
                        tvm.call_extern(reg.dtype, "reg", reg[0])
                    ))
                tvm_ib.emit(
                    tvm.call_extern(
                        dst.dtype, "copy_ubuf_to_gm",
                        dst.access_ptr(
                            'w', offset=dst_offset + move_len_dst_align),
                        data_tail.access_ptr("r", offset=0),
                        0, 1, 1, 0, 0))

    return tvm_ib.get()


def _small_last_32_300_3_2(dst, data, n_begin):
    """
    function of making ir node builder for before dims same
    and last dim smaller than one block scene

    """
    tvm_ib = tvm.ir_builder.create()
    device_core_num = AICORE_NUM
    float_size = cce.cce_intrin.get_bit_len(data.dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - 64
    ub_ele = (ub_bytes // 2 // float_size // cp_align_len) * cp_align_len

    data_one = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                          "data_one", scope=cce.scope_ubuf)
    data_two = _new_alloc(tvm_ib, dst.dtype, ub_ele,
                          "data_two", scope=cce.scope_ubuf)
    data_tail = _new_alloc(tvm_ib, dst.dtype, cp_align_len,
                           "data_tail", scope=cce.scope_ubuf)
    addr_array = tvm_ib.allocate("uint64", (32,), name="addr_array",
                                 scope=cce.scope_reg)
    addr_array_buf = tvm.decl_buffer((32,), "uint64_t", "addr_array_buf",
                                     scope=cce.scope_reg,
                                     data=addr_array)
    reg = tvm_ib.allocate(dst.dtype, (8,), name='reg', scope=cce.scope_reg)

    c_0 = 16
    n_i, col_len, row_in = data.shape
    row_out = dst.shape[2]
    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ib.scope_attr(block_index, "thread_extent", device_core_num)
    n_zu = n_i // c_0
    dim_ele_in = c_0 * col_len * row_in
    mov_len_in = col_len * row_in
    mov_len_in_align = _ceil_fill(mov_len_in, cp_align_len)

    with tvm_ib.if_scope(block_index < n_zu):
        _clean_ubuf(tvm_ib, data_one, 0, ub_ele)
        _clean_ubuf(tvm_ib, data_two, 0, ub_ele)
        with tvm_ib.for_range(0, c_0, name="num_c") as num_c:
            data_offset = block_index * dim_ele_in + num_c * mov_len_in
            ub_offset = num_c * mov_len_in_align
            burst_len_data = _ceil_div(mov_len_in, cp_align_len)
            tvm_ib.emit(tvm.call_extern(data_one.dtype, "copy_gm_to_ubuf",
                                        data_one.access_ptr(
                                            "w", offset=ub_offset),
                                        data.access_ptr(
                                            "r", offset=data_offset),
                                        0, 1, burst_len_data, 0, 0))

        one_begin = 0
        two_begin = ub_ele * float_size
        repeat_vconv = mov_len_in_align // cp_align_len
        src_stride_vconv = 1
        dst_stride_vconv = 16
        col_row_zu_in = mov_len_in_align // cp_align_len
        args = tvm_ib, addr_array, addr_array_buf, one_begin, two_begin, \
               repeat_vconv, src_stride_vconv, dst_stride_vconv, col_row_zu_in
        _vconv_one_fp16(args)

        two_offset = n_begin * c_0
        n_burst = col_len
        burst_len = c_0 // cp_align_len * 2
        src_stride = c_0 // cp_align_len
        dst_stride = 0
        tvm_ib.emit(
            tvm.call_extern(data_one.dtype, "copy_ubuf_to_ubuf",
                            data_one.access_ptr(
                                "w", offset=0),
                            data_two.access_ptr(
                                "r", offset=two_offset),
                            0, n_burst, burst_len,
                            src_stride, dst_stride))

        mov_len_out = col_len * row_out
        mov_len_out_align = _ceil_fill(mov_len_out, cp_align_len)
        one_begin = 0
        two_begin = ub_ele * float_size
        repeat_vconv = mov_len_out_align // cp_align_len
        src_stride_vconv = 16
        dst_stride_vconv = 1
        col_row_zu_out = mov_len_out_align // cp_align_len
        args = tvm_ib, addr_array, addr_array_buf, one_begin, two_begin, \
               repeat_vconv, src_stride_vconv, dst_stride_vconv, col_row_zu_out
        _vconv_two_fp16(args)

        dim_ele_out = c_0 * col_len * row_out
        with tvm_ib.if_scope(mov_len_out % cp_align_len == 0):
            dst_offset = block_index * dim_ele_out
            burst_len_dst = dim_ele_out // cp_align_len
            tvm_ib.emit(
                tvm.call_extern(
                    dst.dtype, "copy_ubuf_to_gm",
                    dst.access_ptr(
                        'w', offset=dst_offset),
                    data_two.access_ptr("r", offset=0),
                    0, 1, burst_len_dst, 0, 0))
        with tvm_ib.else_scope():
            with tvm_ib.for_range(0, c_0, name="num_c") as num_c:
                dst_offset = block_index * dim_ele_out + num_c * mov_len_out
                ub_offset = num_c * mov_len_out_align
                mov_len_before = mov_len_out - cp_align_len
                burst_len_dst = _ceil_div(mov_len_before, cp_align_len)
                tvm_ib.emit(
                    tvm.call_extern(
                        dst.dtype, "copy_ubuf_to_gm",
                        dst.access_ptr(
                            'w', offset=dst_offset),
                        data_two.access_ptr("r", offset=ub_offset),
                        0, 1, burst_len_dst, 0, 0))
                with tvm_ib.for_range(0, cp_align_len, name="num_a") as num_a:
                    tvm_ib.emit(tvm.call_extern(
                        data_two.dtype, "reg_mov",
                        tvm.call_extern(reg.dtype, "reg", reg[0]),
                        data_two.access_ptr(
                            'r', offset=(ub_offset + mov_len_before + num_a))
                    ))
                    tvm_ib.emit(tvm.call_extern(
                        data_tail.dtype, "reg_mov",
                        data_tail.access_ptr('w', offset=num_a),
                        tvm.call_extern(reg.dtype, "reg", reg[0])
                    ))

                tvm_ib.emit(
                    tvm.call_extern(
                        dst.dtype, "copy_ubuf_to_gm",
                        dst.access_ptr(
                            'w', offset=dst_offset + mov_len_before),
                        data_tail.access_ptr("r", offset=0),
                        0, 1, 1, 0, 0))

    return tvm_ib.get()


def _clean_ubuf(ib_, src, src_offset, dup_len):
    """
    :param ib_:
        type: tvm.ir_builer
        desc: the instance of ir builer
    :param src:
        type: Tensor
        desc: input tensor
    :param src_offset:
        type: int
        desc: address offset of src
    :param dup_len:
        type: int
        desc: data length need to clean
    :return:
        None
    """
    uint64_all_one = tvm.const(2**64 - 1, dtype="uint64")
    uint64_all_zero = tvm.const(0, dtype="uint64")
    dtype_factor = 32 // cce.cce_intrin.get_bit_len(src.dtype)
    dup_value = tvm.const(0.0, dtype=src.dtype)
    batch_cnt = 64

    if dup_len > 0:
        if src.dtype == "float16":
            ib_.emit(
                tvm.call_extern("uint64", 'set_vector_mask', uint64_all_one,
                                uint64_all_one))
        else:
            ib_.emit(
                tvm.call_extern("uint64", 'set_vector_mask', uint64_all_zero,
                                uint64_all_one))

        repeat = dup_len // (batch_cnt * dtype_factor)
        dup_left = dup_len % (batch_cnt * dtype_factor)
        if repeat >= 255:
            repeat_loop = (repeat + 255 - 1) // 255
            with ib_.for_range(0, repeat_loop) as i:
                with ib_.if_scope(i != repeat_loop - 1):
                    ib_.emit(
                        tvm.call_extern(
                            src.dtype, 'vector_dup',
                            src.access_ptr(
                                "rw",
                                offset=(255 * batch_cnt * dtype_factor) * i +
                                src_offset), dup_value, 255, 1, 1, 8, 8))
                with ib_.else_scope():
                    ib_.emit(
                        tvm.call_extern(
                            src.dtype, "vector_dup",
                            src.access_ptr(
                                "rw",
                                offset=(255 * batch_cnt * dtype_factor) * i +
                                src_offset), dup_value, repeat % 255, 1, 1, 8,
                            8))

        else:
            ib_.emit(
                tvm.call_extern(src.dtype, "vector_dup",
                                src.access_ptr("rw", offset=src_offset),
                                dup_value, repeat, 1, 1, 8, 8))

            if dup_left > 0:
                if dup_left > 64:
                    high_mask = tvm.const(2 ** (dup_left % 64) - 1,
                                          dtype="uint64")
                    ib_.emit(
                        tvm.call_extern("uint64", 'set_vector_mask',
                                        high_mask,
                                        uint64_all_one))
                elif 0 < dup_left <= 64:
                    low_mask = tvm.const(2 ** dup_left - 1, dtype="uint64")
                    ib_.emit(
                        tvm.call_extern("uint64", 'set_vector_mask',
                                        uint64_all_zero,
                                        low_mask))
                ib_.emit(
                    tvm.call_extern(src.dtype, "vector_dup",
                                    src.access_ptr(
                                        "rw",
                                        offset=src_offset +
                                        repeat * batch_cnt * dtype_factor),
                                    dup_value, 1, 1, 1, 8, 8))

        ib_.emit(
            tvm.call_extern("uint64", 'set_vector_mask', uint64_all_one,
                            uint64_all_one))


# pylint: disable=locally-disabled,too-many-return-statements
def _check_scalar_one(shape, size, dtype):
    """
    function of checking whether scalar one scene

    """
    dim = len(shape)
    if dim != 4:
        return False

    for i in range(dim):
        if i == 0 or i == 1:
            if shape[i] != size[i]:
                return False
        if i == 2 or i == 3:
            if shape[i] == size[i]:
                return False

    a_i, b_i, c_i, d_i = shape
    device_core_num = AICORE_NUM
    if a_i < device_core_num:
        if a_i*b_i % device_core_num > 0:
            return False

    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - 32
    ub_ele = (ub_bytes // 2 // float_size // cp_align_len) * cp_align_len

    if c_i*d_i > ub_ele:
        return False

    row_out = size[3]
    if row_out < cp_align_len:
        return False

    if row_out % cp_align_len == 0:
        return False

    return True


def _check_one_core(shape, size, dtype):
    """
    function of checking whether one core branch

    """
    dim = len(shape)
    if dim != 3:
        return False

    if shape[0] != 1:
        return False

    if shape[-1] != 2:
        return False

    if size[-1] != 1:
        return False

    if shape[-1] - size[-1] != 1:
        return False

    for i in range(0, dim - 1):
        if shape[i] != size[i]:
            return False
    if shape[dim - 1] == size[dim - 1]:
        return False

    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_true = UB_SIZE_B - 64
    ub_ele = (ub_true // 2 // float_size // cp_align_len)*cp_align_len

    src_shape_ele = functools_reduce(lambda x, y: x * y, shape[1:])
    dst_shape_ele = functools_reduce(lambda x, y: x * y, size[1:])
    src_shape_align = _ceil_fill(src_shape_ele, cp_align_len)
    dst_shape_align = _ceil_fill(dst_shape_ele, cp_align_len)

    if src_shape_align <= ub_ele and dst_shape_align <= ub_ele:
        return True

    return False


def _get_max_mod(a, b):
    """
    function of getting greatest common divisor

    """
    if b > a:
        temp = a
        a = b
        b = temp

    c = b
    while a % b != 0:
        c = a % b
        a = b
        b = c

    return c


def _get_min_mul(a, b):
    """
    function of getting minimum common multiple

    """
    mod = _get_max_mod(a, b)
    res = (a*b)//mod

    return res


def _get_row_zu(row_out, dtype):
    """
    function of getting row zu

    """
    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size

    row_mod = row_out % cp_align_len
    row_zu = _get_min_mul(cp_align_len, row_mod) // row_mod

    return row_zu


def _get_before_after(row_zu, row_out, dtype):
    """
    function of getting before and after

    """
    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size

    before = []
    after = []
    if row_out < 64:
        for num_rz in range(row_zu):
            gap = (num_rz * row_out) % cp_align_len
            before.append(gap)
            after.append(64 - row_out - gap)
        for num_rz in range(row_zu):
            before.append(0)
            after.append(0)
    else:
        for num_rz in range(row_zu):
            gap = (num_rz * row_out) % cp_align_len
            before.append(gap)
            after.append(0)
            before.append(0)
            after.append(128 - row_out - gap)

    return before, after


# pylint: disable=locally-disabled,too-many-branches
def _last_dim_diff(shape, size, dtype):
    """
    function of checking whether last dim different

    """
    if dtype != "float32":
        return False

    dim = len(shape)
    if dim < 3:
        return False

    for i in range(0, dim - 1):
        if shape[i] != size[i]:
            return False
    if shape[dim - 1] == size[dim - 1]:
        return False

    device_core_num = AICORE_NUM
    if shape[0] < device_core_num:
        return False

    row_in = shape[-1]
    row_out = size[-1]

    if [row_in, row_out] not in [[88, 44], [108, 54], [540, 108],
                                 [135, 27], [264, 44], [270, 54],
                                 [44, 22], [54, 27], [88, 22],
                                 [168, 84], [216, 108], [336, 84],
                                 [22, 11], [27, 13], [44, 11],
                                 [65, 13], [168, 42], [84, 42]]:
        return False

    one_dim = 1
    for i in range(1, dim - 1):
        one_dim = one_dim * size[i]

    if one_dim not in [196, 784, 1764, 3136, 49, 6889]:
        return False

    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - 64
    ub_ele = (ub_bytes // float_size // cp_align_len) * cp_align_len
    ub_ele_half = ub_ele // 2

    row_zu = _get_row_zu(row_out, dtype)
    row_out_align = _ceil_fill(row_out, cp_align_len) + cp_align_len

    if row_out_align > ub_ele_half:
        return False

    if (ub_ele_half // row_out_align) < row_zu:
        return False

    if row_in % cp_align_len == 0:
        row_zu_in = 1
    else:
        row_zu_in = _get_row_zu(row_in, dtype)

    if row_zu_in > row_zu:
        return False

    return True


def _check_sp_diff_same_small_dim(shape, size, dtype):
    """
    function of checking after dims same and small dim branch

    """
    dim = len(shape)
    if dim < 3:
        return False

    for cur_dim in range(2, dim):
        if shape[cur_dim] != size[cur_dim]:
            return False

    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - 32
    ub_ele = (ub_bytes // float_size // cp_align_len) * cp_align_len

    col_out = size[1]
    row_len = 1
    for cur_dim in range(2, dim):
        row_len = row_len * size[cur_dim]

    dim_ele_out = col_out * row_len
    dim_ele_out_align = _ceil_fill(dim_ele_out, cp_align_len)

    if dim_ele_out_align <= ub_ele and dim_ele_out >= cp_align_len:
        return True

    return False


def _check_sp_diff_same_big_dim(shape, size, dtype):
    """
    function of checking after dims same and big dim branch

    """
    dim = len(shape)
    if dim < 3:
        return False

    for cur_dim in range(2, dim):
        if shape[cur_dim] != size[cur_dim]:
            return False

    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - 32
    ub_ele = (ub_bytes // float_size // cp_align_len) * cp_align_len

    col_out = size[1]
    row_len = 1
    for cur_dim in range(2, dim):
        row_len = row_len * size[cur_dim]

    dim_ele_out = col_out * row_len
    dim_ele_out_align = _ceil_fill(dim_ele_out, cp_align_len)
    row_len_align = _ceil_fill(row_len, cp_align_len)

    if row_len_align <= ub_ele < dim_ele_out_align:
        return True

    return False


def _check_two_diff_same_small_dim(shape, size, dtype):
    """
    function of checking two dim diff and after dims same and small dim branch

    """
    dim = len(shape)
    if dim < 4:
        return False

    device_core_num = AICORE_NUM
    if size[0] < device_core_num:
        return False

    if size[0] % device_core_num > 0:
        return False

    if shape[0] != size[0] or shape[1] == size[1] or shape[2] == size[2]:
        return False

    for cur_dim in range(3, dim):
        if shape[cur_dim] != size[cur_dim]:
            return False

    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - 32
    ub_ele = (ub_bytes // float_size // cp_align_len) * cp_align_len

    col_out = size[2]
    row_len = 1
    for cur_dim in range(3, dim):
        row_len = row_len * size[cur_dim]

    dim_ele_out = col_out * row_len
    dim_ele_out_align = _ceil_fill(dim_ele_out, cp_align_len)

    if dim_ele_out_align <= ub_ele and dim_ele_out >= cp_align_len:
        return True

    return False


def _check_two_diff_same_big_dim(shape, size, dtype):
    """
    function of checking two dim diff and after dims same and big dim branch

    """
    dim = len(shape)
    if dim < 4:
        return False

    device_core_num = AICORE_NUM
    if size[0] < device_core_num:
        return False

    if size[0] % device_core_num > 0:
        return False

    if shape[0] != size[0] or shape[1] == size[1] or shape[2] == size[2]:
        return False

    for cur_dim in range(3, dim):
        if shape[cur_dim] != size[cur_dim]:
            return False

    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - 32
    ub_ele = (ub_bytes // float_size // cp_align_len) * cp_align_len

    col_out = size[2]
    row_len = 1
    for cur_dim in range(3, dim):
        row_len = row_len * size[cur_dim]

    dim_ele_out = col_out * row_len
    dim_ele_out_align = _ceil_fill(dim_ele_out, cp_align_len)
    row_len_align = _ceil_fill(row_len, cp_align_len)

    if row_len_align <= ub_ele < dim_ele_out_align:
        return True

    return False


def _check_21_91_602_1_branch(shape, size, dtype):
    """
    function of checking 21/20 91/90 602/601 branch

    """
    dim = len(shape)
    if dim < 3:
        return False

    if dtype != "float16" and dtype != "float32":
        return False

    for i in range(0, dim - 1):
        if shape[i] != size[i]:
            return False

    if shape[dim - 1] == size[dim - 1]:
        return False

    device_core_num = AICORE_NUM
    if shape[0] < device_core_num:
        return False

    row_in = shape[-1]
    row_out = size[-1]

    if [row_in, row_out] not in [[21, 20], [91, 90], [602, 601]]:
        return False

    one_dim = 1
    for i in range(1, dim - 1):
        one_dim = one_dim * size[i]

    if one_dim not in [20, 32, 100, 300, 845, 1917, 3000, 5118, 51150]:
        return False

    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - 64
    ub_ele = (ub_bytes // float_size // cp_align_len) * cp_align_len
    ub_ele_eight = ub_ele // (2 * cp_align_len)

    row_out_align_t = _ceil_fill(row_out, cp_align_len) + cp_align_len
    if ub_ele_eight < row_out_align_t:
        return False

    return True


def _check_32_32_4(shape, size, dtype):
    """
    function of checking 32/32/4 branch

    """
    if dtype != "float16" and dtype != "float32":
        return False

    if list(shape) != [32, 32, 32] or list(size) != [32, 32, 4]:
        return False

    device_core_num = AICORE_NUM
    if device_core_num != 32:
        return False

    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - 64
    ub_ele = (ub_bytes // 2 // float_size // cp_align_len) * cp_align_len

    if ub_ele < 32 * 32:
        return False

    return True


def _check_32_512_2(shape, size, dtype):
    """
    function of checking 32 512/1024 2 branch

    """
    if dtype != "float16" and dtype != "float32":
        return False

    if (shape, size) not in (([32, 512, 2], [32, 512, 1]),
                             ([32, 1024, 2], [32, 1024, 1]),
                             ([32, 512, 1, 2], [32, 512, 1, 1]),
                             ([32, 1024, 1, 2], [32, 1024, 1, 1]),
                             ([10, 512, 2], [10, 512, 1]),
                             ([10, 512, 1, 2], [10, 512, 1, 1]),
                             ([10, 1024, 2], [10, 1024, 1]),
                             ([10, 1024, 1, 2], [10, 1024, 1, 1])):
        return False

    if len(shape) == 3:
        n_i, col_len, row_in = shape
    else:
        n_i, col_len, _, row_in = shape

    device_core_num = AICORE_NUM
    c_0 = 16
    n_zu = _ceil_fill(n_i, c_0) // c_0
    if col_len == 1024 and dtype == "float32":
        if device_core_num < (2 * n_zu):
            return False
    else:
        if device_core_num < n_zu:
            return False

    if col_len == 1024 and dtype == "float32":
        dim_ele = c_0 * col_len * row_in // 2
    else:
        dim_ele = c_0 * col_len * row_in

    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - 64
    ub_ele = (ub_bytes // 2 // float_size // cp_align_len) * cp_align_len

    if dim_ele > ub_ele:
        return False

    return True


def _check_32_1917_2(shape, size, dtype):
    """
    function of checking 32 1917 2 branch

    """
    if dtype != "float16" and dtype != "float32":
        return False

    if shape != [32, 1917, 2] or size != [32, 1917, 1]:
        return False

    device_core_num = AICORE_NUM
    if device_core_num != 32:
        return False

    row_in = shape[-1]

    c_0 = 16
    dim_ele = c_0 * 512 * row_in
    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - 64
    ub_ele = (ub_bytes // 2 // float_size // cp_align_len) * cp_align_len

    if dim_ele > ub_ele:
        return False

    return True


def _check_65472_4_2(shape, size, dtype):
    """
    function of checking 65472 4/2 float16 branch

    """
    if dtype != "float16" and dtype != "float32":
        return False

    if (shape, size) not in (([16, 4092, 2], [16, 4092, 1]),
                             ([16, 4092, 4], [16, 4092, 1]),
                             ([65472, 2], [65472, 1]),
                             ([65472, 4], [65472, 1])):
        return False

    device_core_num = AICORE_NUM
    if device_core_num != 32:
        return False

    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - 64
    ub_ele = (ub_bytes // 2 // float_size // cp_align_len) * cp_align_len

    row_in = shape[-1]
    if row_in == 4:
        col_len = 256
    else:
        col_len = 512

    dim_ele = col_len * row_in * 16
    if dim_ele > ub_ele:
        return False

    return True


def _check_32_300_3_2(shape, size, dtype):
    """
    function of checking 32 300 3/2 branch

    """
    if dtype != "float16" and dtype != "float32":
        return False

    if shape != [32, 300, 3] or size != [32, 300, 2]:
        return False

    device_core_num = AICORE_NUM
    n_i = shape[0]
    c_0 = 16
    n_zu = n_i // c_0
    if device_core_num < n_zu:
        return False

    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size
    ub_bytes = UB_SIZE_B - 64
    ub_ele = (ub_bytes // 2 // float_size // cp_align_len) * cp_align_len
    _, col_len, row_in = shape
    col_len_align = _ceil_fill(col_len, cp_align_len)
    dim_ele = 16 * col_len_align * row_in
    if ub_ele < dim_ele:
        return False

    return True


def _check_not_one_block(shape_new):
    len_s = len(shape_new)
    if len_s >= 2:
        value = functools_reduce(lambda x, y: x * y, shape_new[0:len_s - 1])
        if value > 30000000:
            return False

    return True


@util.check_input_type(dict, dict, (list, tuple), (list, tuple), str)
def slice_d(x, y, begin, size, kernel_name="slice_d"):
    """
    algorithm: slice_d
    calculating: this operation extracts a slice of size size
                 from a tensor input
                 starting at the location specified by begin.

    Parameters
    ----------
    x: dict
        contains shape and dtype information of input tensor
    y: dict
        contains shape and dtype information of output tensor
    begin: list or tuple
        represents the index of the first value to select
    size: list or tuple
        represents the shape of output tensor
    kernel_name: str
        cce kernel name, default value is "slice_d".

    Returns
    -------
    None
    """
    shape = x.get("shape")
    dtype = x.get("dtype").lower()
    _check_parameters(shape, dtype, begin, size, kernel_name)
    shape_new, begin_new, size_new = _update_params(shape, begin, size)

    if len(shape_new) >= 2:
        if shape_new[0] == 1 and size_new[0] == 1:
            shape_new = shape_new[1:]
            begin_new = begin_new[1:]
            size_new = size_new[1:]

    if _check_one_core(shape_new, size_new, dtype):
        dim = len(shape_new)
        zero_dim = shape_new[0]
        one_dim = 1
        for i in range(1, dim - 1):
            one_dim = one_dim * shape_new[i]
        shape_use = [zero_dim, one_dim, shape_new[-1]]
        size_use = [zero_dim, one_dim, size_new[-1]]
        n_begin = begin_new[-1]
        data = tvm.placeholder(shape_use, dtype=dtype, name='data')
        res = tvm.extern(size_use, [data],
                         lambda ins, outs: _one_core_ir(outs[0], ins[0],
                                                        n_begin),
                         name="res", dtype=dtype)

        tensor_list = [data, res]
        sch = tvm.create_schedule(res.op)
        with build_config:
            tvm.build(sch, tensor_list, "cce", name=kernel_name)
    elif _check_32_512_2(shape_new, size_new, dtype):
        if len(shape_new) == 4:
            one_dim = shape_new[0]
            two_dim = shape_new[1]
            three_dim = shape_new[3]
            size_dim = size_new[3]
            shape_new = [one_dim, two_dim, three_dim]
            size_new = [one_dim, two_dim, size_dim]
            n_begin = begin_new[3]
        else:
            n_begin = begin_new[2]

        data = tvm.placeholder(shape_new, dtype=dtype, name='data')

        if (shape_new == [32, 1024, 2] or shape_new == [10, 1024, 2])\
                and dtype == "float32":
            res = tvm.extern(size_new, [data],
                             lambda ins, outs: _small_last_32_1024_2_fp32(
                                 outs[0], ins[0],
                                 n_begin),
                             name="res", dtype=dtype)
        else:
            res = tvm.extern(size_new, [data],
                             lambda ins, outs: _small_last_32_512_2(
                                 outs[0], ins[0],
                                 n_begin),
                             name="res", dtype=dtype)

        tensor_list = [data, res]
        sch = tvm.create_schedule(res.op)
        with build_config:
            tvm.build(sch, tensor_list, "cce", name=kernel_name)
    elif _check_32_1917_2(shape_new, size_new, dtype):
        data = tvm.placeholder(shape_new, dtype=dtype, name='data')
        n_begin = begin_new[2]
        res = tvm.extern(size_new, [data],
                         lambda ins, outs: _small_last_32_1917_2(
                             outs[0], ins[0],
                             n_begin),
                         name="res", dtype=dtype)
        tensor_list = [data, res]
        sch = tvm.create_schedule(res.op)
        with build_config:
            tvm.build(sch, tensor_list, "cce", name=kernel_name)
    elif _check_65472_4_2(shape_new, size_new, dtype):
        if len(shape_new) == 2:
            row_in = shape_new[1]
            shape_new = [16, 4092, row_in]
            size_new = [16, 4092, 1]
        n_begin = begin_new[-1]
        data = tvm.placeholder(shape_new, dtype=dtype, name='data')
        if shape_new[-1] == 2:
            res = tvm.extern(size_new, [data],
                             lambda ins, outs: _small_last_65472_2(
                                 outs[0], ins[0],
                                 n_begin),
                             name="res", dtype=dtype)
        else:
            res = tvm.extern(size_new, [data],
                             lambda ins, outs: _small_last_65472_4(
                                 outs[0], ins[0],
                                 n_begin),
                             name="res", dtype=dtype)
        tensor_list = [data, res]
        sch = tvm.create_schedule(res.op)
        with build_config:
            tvm.build(sch, tensor_list, "cce", name=kernel_name)
    elif _check_32_300_3_2(shape_new, size_new, dtype):
        data = tvm.placeholder(shape_new, dtype=dtype, name='data')
        n_begin = begin_new[-1]
        res = tvm.extern(size_new, [data],
                         lambda ins, outs: _small_last_32_300_3_2(
                             outs[0], ins[0],
                             n_begin),
                         name="res", dtype=dtype)
        tensor_list = [data, res]
        sch = tvm.create_schedule(res.op)
        with build_config:
            tvm.build(sch, tensor_list, "cce", name=kernel_name)
    else:
        last_dim_compute = SliceLastDimCompute(shape_new, begin_new, size_new,
                                               dtype, kernel_name)
        diff_first_last_dim = SliceDiffLastDimCompute(shape_new, begin_new,
                                                      size_new, dtype,
                                                      kernel_name)

        if last_dim_compute.check() and _check_not_one_block(shape_new):
            last_dim_compute.slice()
        elif diff_first_last_dim.check() and _check_not_one_block(shape_new):
            diff_first_last_dim.slice()
        elif _check_scalar_one(shape_new, size_new, dtype):
            a_i = shape_new[0]
            device_core_num = AICORE_NUM
            if a_i < device_core_num:
                b_i = shape_new[1]
                one_dim = device_core_num
                two_dim = a_i * b_i // device_core_num
                shape_new[0] = one_dim
                shape_new[1] = two_dim
                size_new[0] = one_dim
                size_new[1] = two_dim

            data = tvm.placeholder(shape_new, dtype=dtype, name='data')
            res = tvm.extern(size_new, [data],
                             lambda ins, outs: _mov_scalar_one(outs[0], ins[0],
                                                               begin_new),
                             name="res", dtype=dtype)

            tensor_list = [data, res]
            sch = tvm.create_schedule(res.op)
            with build_config:
                tvm.build(sch, tensor_list, "cce", name=kernel_name)
        elif _last_dim_diff(shape_new, size_new, dtype):
            dim = len(shape_new)
            zero_dim = shape_new[0]
            one_dim = 1
            for i in range(1, dim - 1):
                one_dim = one_dim * size_new[i]

            shape_use = [zero_dim, one_dim, shape_new[-1]]
            size_use = [zero_dim, one_dim, size_new[-1]]
            n_begin = begin_new[-1]

            row_zu = _get_row_zu(size_use[2], dtype)
            row_out = size_use[2]
            before, after = _get_before_after(row_zu, row_out, dtype)

            data = tvm.placeholder(shape_use, dtype=dtype, name='data')
            if row_zu == 2:
                res = tvm.extern(size_use, [data],
                                 lambda ins, outs: _move_sp_vadds_rowzu_2(
                                     outs[0], ins[0],
                                     n_begin, row_zu, before, after),
                                 name="res", dtype=dtype)
            elif row_zu == 4:
                res = tvm.extern(size_use, [data],
                                 lambda ins, outs: _move_sp_vadds_rowzu_4(
                                     outs[0], ins[0],
                                     n_begin, row_zu, before, after),
                                 name="res", dtype=dtype)
            elif row_zu == 8:
                res = tvm.extern(size_use, [data],
                                 lambda ins, outs: _move_sp_vadds_rowzu_8(
                                     outs[0], ins[0],
                                     n_begin, row_zu, before, after),
                                 name="res", dtype=dtype)

            tensor_list = [data, res]
            sch = tvm.create_schedule(res.op)
            with build_config:
                tvm.build(sch, tensor_list, "cce", name=kernel_name)
        elif _check_21_91_602_1_branch(shape_new, size_new, dtype):
            dim = len(shape_new)
            zero_dim = shape_new[0]
            one_dim = 1
            for i in range(1, dim - 1):
                one_dim = one_dim * size_new[i]

            shape_use = [zero_dim, one_dim, shape_new[-1]]
            size_use = [zero_dim, one_dim, size_new[-1]]
            n_begin = begin_new[-1]

            data = tvm.placeholder(shape_use, dtype=dtype, name='data')

            if dtype == "float32" and shape_use[-1] == 21\
                    and size_use[-1] == 20:
                res = tvm.extern(size_use, [data],
                                 lambda ins, outs: _move_sp_vadds_21_20_fp32(
                                     outs[0], ins[0],
                                     n_begin),
                                 name="res", dtype=dtype)
            elif dtype == "float32" and shape_use[-1] == 91\
                    and size_use[-1] == 90:
                res = tvm.extern(size_use, [data],
                                 lambda ins, outs: _move_sp_vadds_91_90_fp32(
                                     outs[0], ins[0],
                                     n_begin),
                                 name="res", dtype=dtype)
            elif dtype == "float32" and shape_use[-1] == 602\
                    and size_use[-1] == 601:
                res = tvm.extern(size_use, [data],
                                 lambda ins, outs: _move_sp_vadds_602_601_fp32(
                                     outs[0], ins[0],
                                     n_begin),
                                 name="res", dtype=dtype)
            elif dtype == "float16" and shape_use[-1] == 21\
                    and size_use[-1] == 20:
                res = tvm.extern(size_use, [data],
                                 lambda ins, outs: _move_sp_vadds_21_20_fp16(
                                     outs[0], ins[0],
                                     n_begin),
                                 name="res", dtype=dtype)
            elif dtype == "float16" and shape_use[-1] == 91\
                    and size_use[-1] == 90:
                res = tvm.extern(size_use, [data],
                                 lambda ins, outs: _move_sp_vadds_91_90_fp16(
                                     outs[0], ins[0],
                                     n_begin),
                                 name="res", dtype=dtype)
            elif dtype == "float16" and shape_use[-1] == 602\
                    and size_use[-1] == 601:
                res = tvm.extern(size_use, [data],
                                 lambda ins, outs: _move_sp_vadds_602_601_fp16(
                                     outs[0], ins[0],
                                     n_begin),
                                 name="res", dtype=dtype)

            tensor_list = [data, res]
            sch = tvm.create_schedule(res.op)
            with build_config:
                tvm.build(sch, tensor_list, "cce", name=kernel_name)

        elif _check_sp_diff_same_small_dim(shape_new, size_new, dtype):
            dim = len(shape_new)
            row_len = 1
            for cur_dim in range(2, dim):
                row_len = row_len * size_new[cur_dim]
            shape_use = [shape_new[0], shape_new[1], row_len]
            size_use = [size_new[0], size_new[1], row_len]
            bl_begin = begin_new[0]
            col_begin = begin_new[1]

            data = tvm.placeholder(shape_use, dtype=dtype, name='data')
            res = tvm.extern(size_use, [data],
                             lambda ins, outs: _move_sp_diff_same_small_dim(
                                 outs[0],
                                 ins[0],
                                 bl_begin,
                                 col_begin),
                             name="res", dtype=dtype)
            tensor_list = [data, res]
            sch = tvm.create_schedule(res.op)
            with build_config:
                tvm.build(sch, tensor_list, "cce", name=kernel_name)

        elif _check_sp_diff_same_big_dim(shape_new, size_new, dtype):
            dim = len(shape_new)
            row_len = 1
            for cur_dim in range(2, dim):
                row_len = row_len * size_new[cur_dim]
            shape_use = [shape_new[0], shape_new[1], row_len]
            size_use = [size_new[0], size_new[1], row_len]
            bl_begin = begin_new[0]
            col_begin = begin_new[1]

            data = tvm.placeholder(shape_use, dtype=dtype, name='data')
            res = tvm.extern(size_use, [data],
                             lambda ins, outs: _move_sp_diff_same_big_dim(
                                 outs[0],
                                 ins[0],
                                 bl_begin,
                                 col_begin),
                             name="res", dtype=dtype)
            tensor_list = [data, res]
            sch = tvm.create_schedule(res.op)
            with build_config:
                tvm.build(sch, tensor_list, "cce", name=kernel_name)
        elif _check_two_diff_same_small_dim(shape_new, size_new, dtype):
            dim = len(shape_new)
            row_len = 1
            for cur_dim in range(3, dim):
                row_len = row_len * size_new[cur_dim]
            shape_use = [shape_new[0], shape_new[1], shape_new[2], row_len]
            size_use = [size_new[0], size_new[1], size_new[2], row_len]
            bl_begin = begin_new[1]
            col_begin = begin_new[2]

            data = tvm.placeholder(shape_use, dtype=dtype, name='data')
            res = tvm.extern(size_use, [data],
                             lambda ins, outs: _move_two_diff_same_small_dim(
                                 outs[0],
                                 ins[0],
                                 bl_begin,
                                 col_begin),
                             name="res", dtype=dtype)
            tensor_list = [data, res]
            sch = tvm.create_schedule(res.op)
            with build_config:
                tvm.build(sch, tensor_list, "cce", name=kernel_name)
        elif _check_two_diff_same_big_dim(shape_new, size_new, dtype):
            dim = len(shape_new)
            row_len = 1
            for cur_dim in range(3, dim):
                row_len = row_len * size_new[cur_dim]
            shape_use = [shape_new[0], shape_new[1], shape_new[2], row_len]
            size_use = [size_new[0], size_new[1], size_new[2], row_len]
            bl_begin = begin_new[1]
            col_begin = begin_new[2]

            data = tvm.placeholder(shape_use, dtype=dtype, name='data')
            res = tvm.extern(size_use, [data],
                             lambda ins, outs: _move_two_diff_same_big_dim(
                                 outs[0],
                                 ins[0],
                                 bl_begin,
                                 col_begin),
                             name="res", dtype=dtype)
            tensor_list = [data, res]
            sch = tvm.create_schedule(res.op)
            with build_config:
                tvm.build(sch, tensor_list, "cce", name=kernel_name)
        elif _check_32_32_4(shape_new, size_new, dtype):
            n_begin = begin_new[-1]
            data = tvm.placeholder(shape_new, dtype=dtype, name='data')
            res = tvm.extern(size_new, [data],
                             lambda ins, outs: _move_sp_vadds_32_32_4(outs[0],
                                                                      ins[0],
                                                                      n_begin),
                             name="res", dtype=dtype)

            tensor_list = [data, res]
            sch = tvm.create_schedule(res.op)
            with build_config:
                tvm.build(sch, tensor_list, "cce", name=kernel_name)
        else:
            data = tvm.placeholder(shape_new, dtype=dtype, name='data')
            sch, res = slice_d_compute(data, y, begin_new, size_new,
                                       kernel_name)

            with build_config:
                tvm.build(sch, [data, res], "cce", name=kernel_name)
