#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

elewise speel schedule
"""


from te import tvm

from . import util
from .elewise_schedule import CceOp


# pylint: disable=too-many-branches,too-many-statements,too-many-instance-attributes
# pylint: no-member
class CceSpeelOp(CceOp):
    """
    Base class of cce API

    Parameters
    ----------
    scope : cal buffer like local.UB

    need_tensorize : if need to doing tensorize when using calculate

    need_pragma : if need to doing pragma when using calculate

    need_double_buffer : if need to do double buffer, only support double buffer
     for the buffer in inner of for loop
    Returns
    -------
    CceOp_instance : instance of CceOp
    """
    def core_schedule_reduce(self, res, spec_node_list, sch_list, tensor_map):
        """
        auto_schedule for cce AI-CORE. For now, only N's elewise operation +
        (0 - 1) reduction operations are supported. The last axis must be n*128,
         except the case of reducting last axis, the exception case requires
         the last axis and reduction axis is 16*n

        Parameters
        ----------
        res : list of tensor

        Returns
        -------
        sch: Schedule
            The computation schedule for the op.

        _origin_tensor: list of tvm.Tensor
            return tensors for tvm.build and tvm.lower
        """
        self._res_tensor_list = res
        if hasattr(res, 'index'):
            if len(res) > 1:
                self._is_muti_output = True
                util.get_dst_tensor_map(res, self._dst_tensor_map)
                # get the last out,out1-->....-->out2
                for out in self._res_tensor_list:
                    if out not in self._dst_tensor_map.keys():
                        self._res_tensor = out
            else:
                self._res_tensor = res[0]
        else:
            self._res_tensor = res

        # out1->..->out2,self._res_tensor_list constains the all outs except out2
        self._res_tensor_list.remove(self._res_tensor)

        self._write_buffer = []
        self._read_buffer = []
        self._schedule = sch_list[0]
        self._spec_node_list = spec_node_list
        self.update(self._res_tensor)

        read_buffer = self.local_cache_read()
        _ = self.local_cache_write()
        # cache read again for muti-res
        read_buffer.extend(self.local_cache_read_for_muti_out())

        self._need_compute_at_before = True
        self._need_compute_at_after = False
        self._is_last_reduce = False

        # special operations including data slice, data tiling
        if self._have_reduce:
            self.reduce_schedule(read_buffer)
        else:
            self.elewise_schedule(read_buffer)

        # enable muti core
        if self._need_enable_muticore:
            self.local_enable_muti_core()

        self.local_compute_at(read_buffer)
        ##compute_inline operations
        self.local_compute_inline()

        if not self.check_valid_schedule():
            return False

        # tensorize operations
        if self._need_tensorize:
            self.local_tensorize()

        # pragma operations
        if self._need_pragma:
            self.local_pragma(read_buffer)

        # double buffer
        if self._need_double_buffer:
            self.local_double_buffer(read_buffer)

        sch_list[0] = self._schedule
        return True, None

    def check_valid_schedule(self):
        """
        :return: wether speel schedule can deal
        """
        # if there are more than one independent intermediary tensor, the speel
        # schedule not support
        if len(self._spec_node_list) > 1:
            return False

        if self._have_reduce:
            return self.check_valid_reduction_schedule()
        return self.check_valid_elewise_schedule()

    def check_valid_reduction_schedule(self):
        """
        :return: wether reduction speel schedule can deal
        """
        return True

    def check_valid_elewise_schedule(self):
        """
        :return: wether reduction speel schedule can deal
        """
        return True

    # pylint: disable=too-many-locals
    def elewise_tiling(self, shape):
        """
        simple auto tiling module for elewise op

        Parameters
        ----------
        shape : tensor shape

        Returns
        -------
        int : slice number
        """
        max_ub_count = self.get_max_ub_count()
        rfactor = max_ub_count
        axis = len(shape) - 1

        # find the split axis, shape = (shape[0],.,shape[split_axis], shape[-1])
        # so that shape[split_axis]*shape[split_axis+1]*..*shape[-1]<max_ub_count
        # and  shape[split_axis-1]*shape[split_axis]*...*shape[-1]> max_ub_count
        # pylint: disable=no-member
        if not self.is_strict_lastAxis():
            for tmp_axis in reversed(shape):
                if max_ub_count == 1:
                    outer = shape[axis + 1] // rfactor
                    return outer, axis + 1
                if max_ub_count >= tmp_axis:
                    rfactor = tmp_axis
                    max_ub_count = max_ub_count // tmp_axis
                else:
                    break
                axis -= 1

        # shape1 * shape2 > 1, the outer no need to div block_dim completely
        outer_shape_size = 1
        for i in range(0, axis):
            outer_shape_size = outer_shape_size*shape[i]

        inner_shape_size = 1
        for i in range(axis + 1, len(shape)):
            inner_shape_size = inner_shape_size*shape[i]

        align_type = self._res_tensor.dtype
        # bool is represented by int8
        if align_type == 'bool':
            align_type = 'int8'
        # if the dma_copy_count can not div aligin, the muti core write to gm
        # may conflict, do not enable muti-core
        align_factor, _ = util.get_align_factor(align_type)

        if axis != -1:
            for outer in range(1, shape[axis]):
                # inner < max_ub_count and tail < max_ub_count
                inner = shape[axis] // outer
                tail = shape[axis] - outer*inner
                # if can enbale muti core and not the core bind axis is
                # the non-div-completely axis
                if self._need_enable_muticore and \
                        inner_shape_size % align_factor == 0 and \
                        outer_shape_size == 1:
                    # in this case need non-completely-div muti core
                    if outer % self._block_dim == 0 and inner < max_ub_count \
                            and tail < max_ub_count:
                        return outer, axis
                else:
                    if inner < max_ub_count and tail < max_ub_count:
                        return outer, axis

        # axis = -1 , can load all data to ub
        outer = 1

        if axis == len(shape) - 1:
            return outer, axis

        return outer, axis + 1

    def elewise_schedule(self, read_buffer=None):
        """
        do the elewise pattern schedule
        """
        self._reduce_index = len(self._op)
        self._is_elewise_single_and_broadcast = \
            self._check_is_elewise_single_and_broadcast()
        self._is_last_axis_boradcast, self._last_boradcast_axis = \
            self._check_is_last_axis_broadcast()
        self._need_compute_at_after = False
        shape = self._shape_to_list(self._res_tensor.shape)
        self._last_num, self._split_axis = self.elewise_tiling(shape)
        res_op = self._schedule[self._res_tensor]
        self._res_tensor_axis_len = len(self._res_tensor.op.axis)

        # 1-dim no need to compute at
        # the split axis is the first axis no need to compute at
        if len(shape) > 1 and self._split_axis > 0:
            # the speel api cannot return the split axis, so we should
            # compute at before speel
            self._compute_at_before_reduce_buffer = self._res_tensor
            self._compute_at_before_reduce_axis = res_op.op.axis[
                self._split_axis - 1]

        res_op.speel(res_op.op.axis[self._split_axis], self._last_num)
        # ub to gm axis
        self._res_dma_axis = res_op.op.axis[self._split_axis]
        self._shape_before_reduce = self._shape_to_list(self._res_tensor.shape)
        # gm to ub axis
        self._read_dma_axis = self._split_axis

    def _reduction_tiling(self, shape, reduce_axis):
        """
        simple auto tiling module for reduce op

        Parameters
        ----------
        shape : tensor shape

        Returns
        -------
        int : slice number
        """

        max_ub_count = self.get_max_ub_count()

        # the split axis is always the last axis
        # the res's last axis is res's length -1
        if self._is_keepdims:
            axis = len(shape) - 1
        else:
            axis = len(shape) - len(reduce_axis) - 1

        last_axis = len(shape) - 1
        rfactor = shape[axis]
        if self._is_last_reduce:
            for outer in range(1, shape[last_axis]):
                # if reduce last , in realize pass, we will add pad so that
                # pad's value: outer - shape[axis] % outer
                real_num = shape[last_axis] + outer - shape[last_axis] % outer
                inner = real_num // outer
                if inner < max_ub_count:
                    rfactor = outer
                    break
        else:
            for outer in range(1, shape[last_axis]):
                # num's value: outer*inner + tail
                # inner < max_ub_count and tail < max_ub_count
                inner = shape[last_axis] // outer
                tail = shape[last_axis] - outer*inner
                if inner < max_ub_count and tail < max_ub_count:
                    rfactor = outer
                    break

        return rfactor, axis

    def reduce_schedule(self, read_buffer=None):
        """
        do the reduction pattern schedule with last and nist axis reduce
        """
        self._reduce_index = self._reduce_index[0]
        reduce_op = [self._op[self._reduce_index]]

        reduce_buffer = [reduce_op[0]["cache_buffer"]]
        reduce_sub_buffer = reduce_buffer[0]
        tmp_reduce_axis_num = reduce_op[0]["reduce_axis_num"]
        self._reduce_axis_num = tmp_reduce_axis_num
        self._shape_before_reduce = self._shape_to_list(
            reduce_op[0]['src_buffer'][-1].shape)
        reduce_axis = reduce_op[0]["reduce_axis"]
        self._is_last_reduce = ((len(self._shape_before_reduce) - 1)
                                in tmp_reduce_axis_num)
        reduce_op[0]["self._is_last_reduce"] = self._is_last_reduce

        index = self.arg_sort(tmp_reduce_axis_num)
        tmp_reduce_axis_num = self.reorder_list(tmp_reduce_axis_num, index)
        reduce_axis = self.reorder_list(reduce_axis, index)
        self._schedule[reduce_sub_buffer].reorder(*(reduce_axis))
        self._is_keepdims = self.is_keepdims()
        if self._is_last_reduce:
            real_reduce_axis = reduce_axis[-1]
            src_buffer = reduce_op[0]["src_buffer"][0]
            reduce_sub_buffer = reduce_buffer[0]
            self._res_tensor_axis_len = len(self._res_tensor.op.axis)
            self._last_num, self._split_axis = self._reduction_tiling(
                self._shape_before_reduce, tmp_reduce_axis_num)
            ndim = len(self._schedule[self._res_tensor].op.axis)
            self._need_compute_at_after = True
            if len(src_buffer.shape) == 1:
                self._need_compute_at_after = False

            if len(tmp_reduce_axis_num) == 1:
                self._compute_at_before_reduce_axis = self._schedule[
                    reduce_sub_buffer].op.axis[ndim - 1]
            else:
                self._compute_at_before_reduce_axis = reduce_axis[-2]

            reduce_op[0]["tensorize_axis"] = real_reduce_axis

            self._res_dma_axis = self._schedule[self._res_tensor].op.axis[
                self._res_tensor_axis_len - 1]
            if self._need_compute_at_after:
                self._compute_at_after_reduce_axis = self._res_tensor.op.axis[
                    len(self._res_tensor.op.axis) - 2]
                self._compute_at_after_reduce_buffer = self._res_tensor
            self._compute_at_before_reduce_buffer = reduce_sub_buffer
            self.local_compute_at(read_buffer)

            self._schedule[reduce_sub_buffer].speel(real_reduce_axis,
                                                    self._last_num)

            reduce_op[0]["tensorize_axis"] = real_reduce_axis

        else:
            self._res_tensor_axis_len = len(self._res_tensor.op.axis)
            self._last_num, self._split_axis = self._reduction_tiling(
                self._shape_before_reduce, tmp_reduce_axis_num)
            ndim = len(self._schedule[self._res_tensor].op.axis)
            self._need_compute_at_after = False

            if len(self._res_tensor.shape) > 1:
                self._compute_at_after_reduce_axis = \
                    self._schedule[self._res_tensor].op.axis[
                        len(self._res_tensor.shape) - 2]
                self._compute_at_after_reduce_buffer = self._res_tensor
                self._need_compute_at_after = True

            self._compute_at_before_reduce_buffer = reduce_sub_buffer
            self._compute_at_before_reduce_axis = reduce_axis[-1]
            self.local_compute_at(read_buffer)
            self._schedule[self._res_tensor].speel(
                self._schedule[self._res_tensor].op.axis[self._split_axis],
                self._last_num)
            self._res_dma_axis = self._schedule[self._res_tensor].op.axis[
                len(self._res_tensor.shape) - 1]
            reduce_op[0]["tensorize_axis"] = self._schedule[
                reduce_op[0]["cache_buffer"]].op.axis[self._split_axis]
            ndim = len(self._schedule[
                self._compute_at_before_reduce_buffer].op.axis)

            self._schedule[self._compute_at_before_reduce_buffer].reorder(*(
                reduce_axis + list(
                    self._schedule[
                        self._compute_at_before_reduce_buffer].op.axis)[
                            self._split_axis:ndim]))

        if not self._is_keepdims:
            self._read_dma_axis = self._split_axis + len(tmp_reduce_axis_num)
        else:
            self._read_dma_axis = self._split_axis

    def tensorize_for_op(self, lop):
        """
        emit_insn for single_op
        """
        op_cmd = lop["op"].split("_")
        cache_buffer = lop["cache_buffer"]
        tensorize_axis = lop["tensorize_axis"]
        if op_cmd[0] == "emit":
            elewise_func = lop["op"].split("emit_insn_")[-1]
            self._schedule[cache_buffer].emit_insn(tensorize_axis, elewise_func)
        elif op_cmd[0].lower() == "elewise":
            emit_insn_pragma = self._emit_insn_map.get(lop["op"])
            if emit_insn_pragma and self._is_cast_support(lop):
                if emit_insn_pragma == "vector_multiple":
                    self.emit_multiple(cache_buffer, lop, op_cmd)
                else:
                    self._schedule[cache_buffer].emit_insn(
                        cache_buffer.op.axis[0], emit_insn_pragma)
            else:
                self._schedule[cache_buffer].emit_insn(
                    tensorize_axis, lop["op"])
        elif op_cmd[0].lower() == "broadcast":
            self._schedule[cache_buffer].emit_insn(tensorize_axis, lop["op"])
        # speel schedule for vector tag
        elif op_cmd[0].lower() == "vector":
            self._schedule[cache_buffer].emit_insn(tensorize_axis, lop["op"])
        elif op_cmd[0].lower() == "reduce":
            if self._is_last_reduce:
                self._schedule[cache_buffer].emit_insn(
                    tensorize_axis, "reduce_last_axis_" + lop["op"])
            else:
                self._schedule[cache_buffer].emit_insn(
                    tensorize_axis, "reduce_nlst_axis_" + lop["op"])
        else:
            raise RuntimeError("%s not support" % lop["op"])

        if 'cache_read_for_res' in lop.keys():
            cache_buffer_for_res = lop['cache_read_for_res']
            self._schedule[cache_buffer_for_res].emit_insn(
                lop["tensorize_axis_for_res"], 'phony_insn')

    # pylint: disable=too-many-locals
    def elewise_muti_core(self):
        """
        elewise muti core
        :return:
        """
        shape = self._shape_to_list(self._res_tensor.shape)

        # if all number can load ub once, do not enable muti core
        if self._split_axis == 0 and self._last_num == shape[0]:
            return

        # the shape is split to (shape[0], outer*self._last_num, shape[1])
        dma_copy_count = 1
        if shape[self._split_axis] % self._last_num == 0:
            dma_copy_count = dma_copy_count*shape[self._split_axis] // \
                             self._last_num

        for axis in range(self._split_axis + 1, len(shape)):
            dma_copy_count = dma_copy_count*shape[axis]

        align_type = self._res_tensor.dtype
        # bool is represented by int8
        if align_type == 'bool':
            align_type = 'int8'

        align_factor, _ = util.get_align_factor(align_type)
        # if the dma_copy_count in one core less than align_factor,
        # do not enable muti-core
        if dma_copy_count <= align_factor:
            return

        # the shape is split to (shape[0], outer*self._last_num, shape[1])
        fuse_axis_length = 1
        for axis in range(0, self._split_axis):
            fuse_axis_length = fuse_axis_length*shape[axis]

        # if the outloop is 1, such as (1,1,99991,16), (99991, 16), we should
        # bind the block axis to 99991 and in realize pass do block tiling
        if fuse_axis_length == 1:
            if self._last_num % self._block_dim == 0:
                # it will split the last_num to block_dim*n in pass
                self._schedule[self._res_tensor].pragma(
                    self._res_tensor.op.axis[self._split_axis],
                    "block_dim", self._block_dim)
            else:
                # the block_dim is last_num
                if self._last_num < self.MAX_BLOCK_DIM:
                    self._schedule[self._res_tensor].pragma(
                        self._res_tensor.op.axis[self._split_axis],
                        "block_dim", self._last_num)
            return

        # the shape is split to (shape[0], outer*self._last_num, shape[1])
        need_fuse_axis = []
        for i in range(0, self._split_axis):
            need_fuse_axis.append(
                self._compute_at_before_reduce_buffer.op.axis[i])

        fuse_axis = need_fuse_axis[0]
        for i in range(1, len(need_fuse_axis)):
            fuse_axis = self._schedule[self._compute_at_before_reduce_buffer].\
                fuse(fuse_axis, need_fuse_axis[i])

        # we find minimum factor from [core_number, fuse_axis_length) which can
        # be dived by fuse_axis_length. Then we spilt fuse_axis_length to factor
        # and fuse_axis_length/factor, we will take the factor task to muti_core
        factor = self._block_dim if self._block_dim < fuse_axis_length \
            else fuse_axis_length

        while factor < fuse_axis_length:
            if fuse_axis_length % factor == 0:
                break
            factor += 1

        x_outer, x_inner = self._schedule[self._compute_at_before_reduce_buffer]\
            .split(fuse_axis, nparts=factor)
        self._compute_at_before_reduce_axis = x_inner
        block = tvm.thread_axis("blockIdx.x")
        self._schedule[self._compute_at_before_reduce_buffer].bind(x_outer,
                                                                   block)
