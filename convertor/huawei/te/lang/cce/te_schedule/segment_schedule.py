#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

segment schedule, provide a schedule for segment compute
"""
# pylint: disable=too-many-lines
from functools import reduce as functools_reduce

from te import platform as cceconf
from te import tvm


class CceSegmentOp:
    # pylint: disable=too-many-instance-attributes
    """class of cce index op

    Parameters
    ----------
    scope : cal buffer like local.UB

    need_tensorize : if need to doing tensorize when using calculate

    need_pragma : if need to doing paagma when using calculate

    Returns
    -------
    CceOp_instance : instance of CceOp

    """

    def __init__(self, scope, need_tensorize=True, need_pragma=True, need_double_buffer=False):
        # record data ddr levels, only equals to -1 or 0 in now stage
        self._need_tensorize = need_tensorize
        self._need_pragma = need_pragma
        self._op_list = []
        self._split_axis = None
        self._ub_split_xo = None
        self._ub_split_xi = None
        self._block_split_xo = None
        self._block_split_xi = None
        self._block_split_axis = None
        self._block_split_factor = None
        self._check_ids_tensor = False
        self._need_block_tiling = False
        self._need_font_emit = False
        self._segment_nums = 0
        self._read_buffer = []
        self._write_buffer = []
        self._scope = scope
        self._segment_op_pos = 0
        self._factor_ = 0
        self._schedule = None
        self._need_double_buffer = need_double_buffer
        self._spec_node_list = []
        self._get_op_list_traversed_tensor = set()
        self._goto_speel_ub_use_threshold = 0.5
        self._core_dim = cceconf.get_soc_spec("CORE_NUM")

        if self._scope.lower().find('.ub') != -1:
            self._total_size = cceconf.get_soc_spec("UB_SIZE")
            if self._need_double_buffer:
                self._total_size = self._total_size // 2
        else:
            raise RuntimeError("only support UB buffer now")

    def get_max_ptr(self):
        """
        base on the diff data type, get the max ptr based on usable ub
        """
        dtype = self._write_buffer[0].dtype
        if self._check_ids_tensor:
            dtype_byte = 2
            if dtype == "float16":
                dtype_byte = 2
            elif dtype == "float32":
                dtype_byte = 4
            elif dtype == "int32":
                dtype_byte = 4
        else:
            dtype_byte = 4
            if dtype == "float16":
                dtype_byte = 4
            elif dtype == "float32":
                dtype_byte = 8
            elif dtype == "int32":
                dtype_byte = 8
        return self._total_size // (len(self._write_buffer)*dtype_byte)

    def get_align_factor(self, dtype):
        # pylint: disable=no-self-use
        """
        base on the diff data type, get the align_factor
        """
        align_factor = 16
        if dtype in ("int8", "uint8"):
            align_factor = 32
        elif dtype == "float16":
            align_factor = 16
        else:
            align_factor = 8
        return align_factor

    # pylint: disable=too-many-locals, too-many-branches
    def schedule(self, res, spec_node_list, sch_list):
        """
        int8/uint8 goes to CPU schedule in current version
        """

        self._schedule = sch_list[0]
        self._spec_node_list = spec_node_list

        # =============================== analyse op =========================================
        self._get_op_list(res)
        self._op_list.reverse()
        self._decode_buffer()
        self._locate_segment_op_pos()
        # =============================== data flow =========================================
        write_cache_list = [self._schedule.cache_write(out, self._scope) for out in
                            self._write_buffer]
        for out in self._write_buffer[1:]:
            self._schedule[out].compute_inline()

        read_cache_list = []
        for in_tensor in self._read_buffer:
            reader_list = []
            for op_i in self._op_list:
                if in_tensor in op_i['src_buffer']:
                    for reader_i in op_i['dst_buffer']:
                        idx = self._write_buffer.index(reader_i)
                        reader_list.append(write_cache_list[idx])

            read_cache_list.append(self._schedule.cache_read(in_tensor, self._scope, reader_list))

        # =============================== data split =========================================
        res = self._write_buffer[0]
        res_ub = write_cache_list[0]
        if self._check_ids_tensor:
            self.segment_op_tensor_tiling()
            self._ub_split_xo, self._ub_split_xi, splited = self.split(res)
            self.tensor_reorder_compute_at(res, write_cache_list, read_cache_list)
        else:
            in_shape = [i.value for i in self._op_list[self._segment_op_pos]["src_buffer"][0].shape]

            def refine_dime(in_shape):
                length = len(in_shape)
                for i in range(len(in_shape)):
                    if in_shape[len(in_shape) - 1 - i] == 1:
                        length -= 1
                return length

            length = refine_dime(in_shape)
            if length == 1:
                self._dim_equal_one_schedule(
                    res, write_cache_list, read_cache_list)
                sch_list[0] = self._schedule
                return True
            self._ub_split_xo, self._ub_split_xi, splited = self.split(res)

            if splited is False:
                self._splited_equal_false_schedule(
                    res, write_cache_list, read_cache_list)
                return True

            first_axis = self._schedule[res].op.axis[0]
            reorder_axis_list = self._schedule[res].op.axis[1: self._split_axis] + \
                                [self._ub_split_xo, ] + \
                                [self._schedule[res].op.axis[0], ] + \
                                [self._ub_split_xi, ] + \
                                self._schedule[res].op.axis[self._split_axis + 1:]
            self._schedule[res].reorder(*reorder_axis_list)

            def do_cache_read_write():
                for cache_tensor in \
                        write_cache_list[0:self._segment_op_pos + 1]:
                    self._schedule[cache_tensor].compute_at(
                        self._schedule[res], first_axis)
                for cache_tensor in \
                        write_cache_list[self._segment_op_pos + 1:]:
                    self._schedule[cache_tensor].compute_at(
                        self._schedule[res], self._ub_split_xo)
                for cache_tensor in read_cache_list:
                    self._schedule[cache_tensor].compute_at(
                        self._schedule[res], self._ub_split_xo)
                for cache_tensor in \
                        write_cache_list[self._segment_op_pos + 1:]:
                    self._schedule[cache_tensor].storage_align(
                        self._schedule[cache_tensor].op.axis[0],
                        self.get_align_factor(cache_tensor.dtype), 0)
                for cache_tensor in read_cache_list:
                    self._schedule[cache_tensor].storage_align(
                        self._schedule[cache_tensor].op.axis[0],
                        self.get_align_factor(cache_tensor.dtype), 0)

            do_cache_read_write()

        # =============================== intrinsic =========================================
        if self._check_ids_tensor:
            self._local_emit_insn(res, res_ub, read_cache_list)
        else:
            if self._need_tensorize:
                self.local_tensorize(write_cache_list)
            if self._need_pragma:
                self.local_pragma(read_cache_list, res)
            if self._need_double_buffer:
                self.local_double_buffer(read_cache_list)

        sch_list[0] = self._schedule
        return True

    def tensor_reorder_compute_at(self, res, write_cache_list, read_cache_list):
        """
        when segment_ids is a tensor,
        doing reorder, compute_at and storage_align.
        """
        in_shape = [i.value for i in self._op_list[self._segment_op_pos]["src_buffer"][1].shape]
        if len(in_shape) != 1:
            reorder_axis_list = self._schedule[res].op.axis[1: self._split_axis] + \
                                [self._ub_split_xo, ] + \
                                [self._schedule[res].op.axis[0], ] + \
                                [self._ub_split_xi, ] + \
                                self._schedule[res].op.axis[self._split_axis + 1:]

            self._schedule[res].reorder(*reorder_axis_list)
        res_ub = write_cache_list[0]
        recoder_axis_list_reduce = [self._schedule[res_ub].op.reduce_axis[0], ]\
                                   + self._schedule[res_ub].op.axis[:]
        self._schedule[write_cache_list[0]].reorder(*recoder_axis_list_reduce)

        if self._need_block_tiling and not self._need_font_emit:
            self._segment_block_tiling(res)

        for cache_tensor in write_cache_list[0:self._segment_op_pos + 1]:
            self._schedule[cache_tensor].compute_at(self._schedule[res], self._ub_split_xo)
        for cache_tensor in write_cache_list[self._segment_op_pos + 1:]:
            self._schedule[cache_tensor].compute_at(self._schedule[res], self._ub_split_xo)
        for cache_tensor in read_cache_list:
            self._schedule[cache_tensor].compute_at(self._schedule[res], self._ub_split_xo)
        for cache_tensor in write_cache_list:
            self._schedule[cache_tensor].storage_align(self._schedule[cache_tensor].op.axis[0],
                                                       self.get_align_factor(cache_tensor.dtype), 0)

        for cache_tensor in read_cache_list[self._segment_op_pos + 1:]:
            self._schedule[cache_tensor].storage_align(self._schedule[cache_tensor].op.axis[0],
                                                       self.get_align_factor(cache_tensor.dtype), 0)

    def _splited_equal_false_schedule(self, res, write_cache_list, read_cache_list):
        """
        when splited is false,
        do schedule in this function
        """
        for cache_tensor in write_cache_list[0:self._segment_op_pos + 1]:
            self._schedule[cache_tensor].compute_at(self._schedule[res],
                                                    self._schedule[res].op.axis[0])
        for cache_tensor in write_cache_list[self._segment_op_pos + 1:]:
            self._schedule[cache_tensor].compute_at(self._schedule[res],
                                                    self._schedule[res].op.axis[0])
        for cache_tensor in read_cache_list:
            self._schedule[cache_tensor].compute_at(self._schedule[res],
                                                    self._schedule[res].op.axis[0])
        for cache_tensor in write_cache_list[self._segment_op_pos + 1:]:
            self._schedule[cache_tensor].storage_align(self._schedule[cache_tensor].op.axis[0],
                                                       self.get_align_factor(cache_tensor.dtype), 0)
        for cache_tensor in read_cache_list:
            self._schedule[cache_tensor].storage_align(self._schedule[cache_tensor].op.axis[0],
                                                       self.get_align_factor(cache_tensor.dtype), 0)
        if self._need_tensorize:
            for op_index in range(len(self._op_list)):
                op_name = self._op_list[op_index]['op']
                cache_buffer = write_cache_list[op_index]
                if op_index == self._segment_op_pos:
                    # for using in emit_insn callback in cce_intrin_md
                    cceconf.cce_emitinsn_params.cceEmitParamsIns.insert_params(
                        {"segment_ids": self._op_list[op_index]['segment_ids'],
                         "num_segments": self._op_list[op_index]['num_segments'],
                         "segment_init_value": self._op_list[op_index]['args']})
                    self._schedule[cache_buffer].emit_insn(
                        self._schedule[cache_buffer].op.axis[0],
                        op_name)

                else:
                    if op_index < self._segment_op_pos:
                        self._schedule[cache_buffer].emit_insn(
                            self._schedule[cache_buffer].op.axis[0], op_name)
                    else:
                        self._schedule[cache_buffer].emit_insn(
                            self._schedule[cache_buffer].op.axis[0],
                            op_name)

        if self._need_pragma:
            for cache_tensor in read_cache_list:
                self._schedule[cache_tensor].emit_insn(
                    self._schedule[cache_tensor].op.axis[0],
                    cceconf.dma_copy)
            self._schedule[res].emit_insn(self._schedule[res].op.axis[0],
                                          cceconf.dma_copy, {"no_overlap": 1})

    def _dim_equal_one_schedule(self, res, write_cache_list, read_cache_list):
        """
        when the dim of input_tensor's shape  is 1,
        do schedule in this function
        """
        for cache_tensor in write_cache_list[self._segment_op_pos + 1:]:
            self._schedule[cache_tensor].storage_align(self._schedule[cache_tensor].op.axis[0],
                                                       self.get_align_factor(cache_tensor.dtype), 0)
        for cache_tensor in read_cache_list:
            self._schedule[cache_tensor].storage_align(self._schedule[cache_tensor].op.axis[0],
                                                       self.get_align_factor(cache_tensor.dtype), 0)
        if self._need_tensorize:
            for op_index in range(len(self._op_list)):
                op_name = self._op_list[op_index]['op']
                cache_buffer = write_cache_list[op_index]
                if op_index == self._segment_op_pos:
                    # for using in emit_insn callback in cce_intrin_md
                    cceconf.cce_emitinsn_params.cceEmitParamsIns.insert_params(
                        {"segment_ids": self._op_list[op_index]['segment_ids'],
                         "num_segments": self._op_list[op_index]['num_segments'],
                         "segment_init_value": self._op_list[op_index]['args']})
                    _, inner_axis = self._schedule[cache_buffer].split(
                        self._schedule[cache_buffer].op.axis[0], factor=1)
                    self._schedule[cache_buffer].emit_insn(inner_axis, op_name)

                else:
                    if op_index < self._segment_op_pos:
                        self._schedule[cache_buffer].emit_insn(
                            self._schedule[cache_buffer].op.axis[0], op_name)
                    else:
                        self._schedule[cache_buffer].emit_insn(
                            self._schedule[cache_buffer].op.axis[0],
                            op_name)

        if self._need_pragma:
            for cache_tensor in read_cache_list:
                self._schedule[cache_tensor].emit_insn(
                    self._schedule[cache_tensor].op.axis[0],
                    cceconf.dma_copy)
            self._schedule[res].emit_insn(self._schedule[res].op.axis[0],
                                          cceconf.dma_copy, {"no_overlap": 1})

    def _local_emit_insn(self, res, res_ub, read_cache_list):
        """
        when segment_ids is a tensor,
        doing emit_insn.
        """
        for op_index in range(len(self._op_list)):
            op_name = self._op_list[op_index]['op']
        in_shape = [i.value for i in self._op_list[self._segment_op_pos]["src_buffer"][1].shape]
        for cache_tensor in read_cache_list:
            self._schedule[cache_tensor].emit_insn(
                self._schedule[cache_tensor].op.axis[0],
                cceconf.dma_copy)

        if op_name == "segmentensor_min":
            op_name_emit = "vector_min"
        elif op_name == "segmentensor_prod":
            op_name_emit = "vector_mul"
        else:
            raise RuntimeError("operation %s not support yet" % op_name)

        if len(in_shape) == 1:
            self._schedule[res].emit_insn(self._ub_split_xi, cceconf.dma_copy, {"no_overlap": 1})
            ub_outer_axis, ub_inner_axis = self._schedule[res_ub].split(
                self._schedule[res_ub].op.axis[0], factor=1)
            self._schedule[res_ub].pragma(ub_outer_axis, "sparse_access", ub_outer_axis)
            self._schedule[res_ub].emit_insn(ub_inner_axis, op_name_emit)
        else:
            if self._need_font_emit:
                self._schedule[res].emit_insn(self._ub_split_xi, "mov_backup")
            else:
                self._schedule[res].emit_insn(self._ub_split_xi, cceconf.dma_copy,
                                              {"no_overlap": 1})
            self._schedule[res_ub].pragma(self._schedule[res_ub].op.axis[0], "sparse_access",
                                          self._schedule[res_ub].op.axis[0])
            self._schedule[res_ub].emit_insn(self._schedule[res_ub].op.axis[1], op_name_emit)


    def split(self, res):
        """
        do segment ub tiing, and return split axis
        """
        if self._check_ids_tensor:
            split_xo, split_xi = self._schedule[res].split(
                res.op.axis[self._split_axis], self._factor_)
            return split_xo, split_xi, True

        (self._split_axis, self._factor_, tiling_result) = self.segment_op_tiling()
        if (not tiling_result) and self._factor_ == 0:
            return None, None, False

        if (not tiling_result) and self._factor_ == 1:
            split_xo, split_xi = self._schedule[res].split(
                res.op.axis[self._split_axis], self._factor_)
            return split_xo, split_xi, True

        split_xo, split_xi = self._schedule[res].split(
            res.op.axis[self._split_axis], self._factor_)
        return split_xo, split_xi, True

    def _segment_need_block_tiling(self, shape, datatype):
        """
        segment need block tiling or not
        """
        align_factor = self.get_align_factor(datatype)
        mul_shape = 1
        for i in range(1, len(shape)):
            mul_shape = mul_shape * shape[i]
        if mul_shape < align_factor:
            need_block_tiling = False
        else:
            need_block_tiling = True
        return need_block_tiling

    # pylint: disable=too-many-locals
    def _get_tensor_block_axis_factor(self, shape):
        """
        when segment_ids is a tensor,
        get block axis and block factor.
        """
        bound_size = 1
        block_split_axis = 0
        for i, _ in enumerate(shape):
            bound_size = int(shape[i] * bound_size)
            block_split_axis = i
            if bound_size >= self._core_dim:
                break

        tmp_size = int(bound_size // shape[block_split_axis])

        if bound_size <= self._core_dim:
            outer = shape[block_split_axis]
            block_factor = 1
        else:
            outer = int(self._core_dim // tmp_size)
            block_factor = int((shape[block_split_axis] + outer - 1) // outer)

        if block_split_axis != -1:
            tmp_outer = 1
            for i in range(outer, shape[block_split_axis]+1):
                if shape[block_split_axis] % i == 0:
                    tmp_outer = i
                    break
            outer = tmp_outer
            block_factor = int(shape[block_split_axis] // outer)
        block_split_axis = block_split_axis + 1
        return block_split_axis, block_factor

    # pylint: disable=too-many-branches
    def _segment_block_tiling(self, res):
        """
        when segment_ids is a tensor,
        do block tiling.
        """
        if self._split_axis == 1:
            thread_block = tvm.thread_axis("blockIdx.x")
            self._schedule[res].bind(self._ub_split_xo, thread_block)
            return

        block_xo, _ = self._schedule[res].split(
            res.op.axis[self._block_split_axis], factor=self._block_split_factor)

        if self._block_split_axis == 1:
            bind_axis = block_xo
        else:
            fuse_axis_list = []
            for i in range(1, self._block_split_axis):
                fuse_axis_list.append(res.op.axis[i])
            fuse_axis_list.append(block_xo)
            bind_axis = self._schedule[res].fuse(*fuse_axis_list)

        thread_block = tvm.thread_axis("blockIdx.x")
        self._schedule[res].bind(bind_axis, thread_block)
        return


    def get_max_ub_size(self, in_shape, max_ptr, align_factor):
        """
        when segment_ids is a tensor,
        get the max_ub_size
        """
        datatype = self._op_list[self._segment_op_pos]["src_buffer"][0].dtype
        align_factor_ids = self.get_align_factor(datatype)
        transform_dtype = align_factor // align_factor_ids
        segment_id_size = (in_shape[0] + align_factor_ids - 1) // \
                          align_factor_ids * align_factor_ids
        max_ub_size = int((max_ptr - transform_dtype * segment_id_size) //
                          (in_shape[0] + self._segment_nums))
        max_ub_size = int(max_ub_size // align_factor * align_factor)
        if max_ub_size < align_factor:
            raise RuntimeError("Too large, causing the max_ub_count to be less than 32B")
        return max_ub_size


    def get_tensor_split_axis_factor(self, in_shape, out_shape, max_ptr, align_factor):
        """
        when segment_ids is a tensor,
        get ub_split_axis and ub_split_factor.
        """
        split_axis = len(out_shape) - 1
        bound_size = 1
        factor = 1
        self._need_font_emit = False
        max_ub_size = self.get_max_ub_size(in_shape, max_ptr, align_factor)
        try:
            self.get_max_ub_size(in_shape, max_ptr // 2, align_factor)
        except RuntimeError:
            self._need_double_buffer = False
        else:
            self._need_double_buffer = True
            max_ptr = max_ptr // 2
            max_ub_size = self.get_max_ub_size(in_shape, max_ptr, align_factor)
        for i in reversed(range(1, len(out_shape))):
            bound_size = int(out_shape[i] * bound_size)
            split_axis = i
            if bound_size >= max_ub_size:
                break
        tmp_size = int(bound_size // out_shape[split_axis])

        if bound_size <= max_ub_size:
            self._need_font_emit = False
            factor = out_shape[split_axis]
            return split_axis, factor
        factor = int(max_ub_size // tmp_size)

        if len(out_shape) == 1:
            return split_axis, factor
        mod2count = 0
        while tmp_size % 2 == 0 and tmp_size != 0:
            tmp_size = tmp_size // 2
            mod2count = mod2count + 1

        align_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        mod2need = align_list.index(align_factor) - mod2count
        if mod2need <= 0:
            tail = (out_shape[split_axis] % factor)*tmp_size
            if tail < align_factor:
                self._need_font_emit = True
            else:
                self._need_font_emit = False
            return split_axis, factor

        align_list_factor = align_list[mod2count]
        if factor >= align_list_factor:
            factor = int(factor // align_list_factor*align_list_factor)

        tail = (out_shape[split_axis] % factor)*tmp_size
        if tail < align_factor:
            self._need_font_emit = True
        else:
            self._need_font_emit = False

        return split_axis, factor


    def segment_op_tensor_tiling(self):
        """
        when segment_ids is a tensor,
        do ub_tiling and block tiling.
        """
        max_ptr = self.get_max_ptr()
        out_shape = [i.value for i in self._op_list[self._segment_op_pos]["dst_buffer"][0].shape]
        in_shape = [i.value for i in self._op_list[self._segment_op_pos]["src_buffer"][1].shape]
        datatype = self._op_list[self._segment_op_pos]["dst_buffer"][0].dtype
        align_factor = self.get_align_factor(datatype)
        self._split_axis, self._factor_ = self.get_tensor_split_axis_factor(in_shape, out_shape,
                                                                            max_ptr, align_factor)
        if len(in_shape) == 1:
            self._need_block_tiling = False
            return
        # Multi-core processing cannot be done with data less than 32B
        self._need_block_tiling = self._segment_need_block_tiling(in_shape, datatype)
        if self._split_axis == 1:
            return
        self._block_split_axis, self._block_split_factor = \
            self._get_tensor_block_axis_factor(in_shape[1:self._split_axis])
        return

    def local_tensorize(self, write_cache_list):
        """
        for using in emit_insn callback in cce_intrin_md
        """
        for op_index in range(len(self._op_list)):
            op_name = self._op_list[op_index]['op']
            cache_buffer = write_cache_list[op_index]
            if op_index == self._segment_op_pos:

                # for using in emit_insn callback in cce_intrin_md
                cceconf.cce_emitinsn_params.cceEmitParamsIns.insert_params(
                    {"segment_ids": self._op_list[op_index]['segment_ids'],
                     "num_segments": self._op_list[op_index]['num_segments'],
                     "segment_init_value": self._op_list[op_index]['args']})
                self._schedule[cache_buffer].emit_insn(
                    self._schedule[cache_buffer].op.axis[self._split_axis],
                    op_name)

            else:
                if op_index < self._segment_op_pos:
                    self._schedule[cache_buffer].emit_insn(self._schedule[cache_buffer].op.axis[0],
                                                           op_name)
                else:
                    # op_index is bigger means the op is before segment op
                    # the op before segment op comute at outer split axis, split_xo,
                    # first_axis, split_xi.so the idx of split_xo is split_axis-1
                    # but because of 16 assgin, so the dma copy is emitinsn at split_axis,
                    # here emit_insn at this axis too
                    self._schedule[cache_buffer].emit_insn(
                        self._schedule[cache_buffer].op.axis[self._split_axis],
                        op_name)

    def local_pragma(self, read_cache_list, res):
        """
        for using in emit_insn callback in cce_intrin_md
        """
        for cache_tensor in read_cache_list:
            self._schedule[cache_tensor].emit_insn(
                self._schedule[cache_tensor].op.axis[self._split_axis],
                cceconf.dma_copy)


        # if the move element number once can not div align_factor,
        # we should backup for not covering the old result
        self._schedule[res].emit_insn(self._ub_split_xi, "mov_backup")

    def _locate_segment_op_pos(self):
        """
        get segment op pos
        """
        for idx in range(len(self._op_list)):
            if self._check_ids_tensor:
                if "segmentensor" in self._op_list[idx]["op"]:
                    self._segment_op_pos = idx
                    break
            else:
                if "segment" in self._op_list[idx]["op"]:
                    self._segment_op_pos = idx
                    break

    def _get_op_list(self, tmp):
        """
        get op list para
        """
        str_list = tmp.op.tag.split("|")
        tmp_op = {"op": None,
                  "dst_buffer": [],
                  "src_buffer": [],
                  "args": None,
                  "segment_ids": None,
                  "num_segments": None}
        tmp_op["op"] = str_list[0]
        tmp_op["dst_buffer"].append(tmp)
        for in_tensor in list(tmp.op.input_tensors):
            tmp_op['src_buffer'].append(in_tensor)

        if 'segmentensor' in str_list[0].split('_'):
            self._check_ids_tensor = True
            num_segments = int(str_list[1])
            init_value = float(str_list[2])
            self._segment_nums = num_segments
            tmp_op["num_segments"] = num_segments
            tmp_op["args"] = init_value

        elif 'segment' in str_list[0].split('_'):
            self._check_ids_tensor = False
            segment_ids_str = str_list[1].split(",")
            segment_ids = [int(i) for i in segment_ids_str]
            num_segments = int(str_list[2])
            self._segment_nums = num_segments
            init_value = float(str_list[3])
            tmp_op["segment_ids"] = segment_ids
            tmp_op["num_segments"] = num_segments
            tmp_op["args"] = init_value

        elif tmp_op["op"].find("elewise_single") != -1:
            if hasattr(tmp.op.body[0], 'b'):
                if isinstance((tmp.op.body[0].a), tvm.expr.Call):
                    tmp_op["args"] = [tmp.op.body[0].b]
                else:
                    tmp_op["args"] = [tmp.op.body[0].a]

        elif 'elewise' not in str_list[0].split('_'):
            raise RuntimeError("operation %s not support yet" % str_list[0])

        for in_tensor in list(tmp.op.input_tensors):
            if (not isinstance((in_tensor.op), tvm.tensor.PlaceholderOp)) \
                    and (in_tensor not in self._spec_node_list) \
                    and (in_tensor not in self._get_op_list_traversed_tensor):
                self._get_op_list_traversed_tensor.add(in_tensor)
                self._get_op_list(in_tensor)
        self._op_list.append(tmp_op)

    def _decode_buffer(self):
        """
        decode write buffer and read buffer
        """
        read_buffer = set()
        write_buffer = []
        for op_i in self._op_list:
            read_buffer = read_buffer | set(op_i['src_buffer'])
            write_buffer = write_buffer + op_i['dst_buffer']
        for op_i in self._op_list:
            read_buffer = read_buffer - set(op_i['dst_buffer'])
        self._read_buffer = list(read_buffer)
        self._write_buffer = list(write_buffer)

    def _check_do_in_ai_cpu_better(self, data_shape):
        # pylint: disable=no-self-use
        """
        ai_cpu is better than ai_core or not
        """
        if len(data_shape) == 1:
            return True
        mul_axis = functools_reduce(lambda i, j: i*j, data_shape[1:])
        if mul_axis == 1:
            return True

        return False

    def _calc_and_check_max_ub_use(self, axis, factor, in_shape, out_shape, max_ub_size,
                                   align_factor):
        # pylint: disable=too-many-arguments
        """
        calc max ub use and check max ub use < max ub size
        """
        tensorize_shape = [factor, ]
        for idx in range(axis + 1, len(out_shape)):
            tensorize_shape += [out_shape[idx], ]

        tensorize_size = functools_reduce(lambda i, j: i*j, tensorize_shape)

        if self._check_ids_tensor:
            max_ub_use = (in_shape[0] + self._segment_nums) *\
                         ((tensorize_size + align_factor - 1) // align_factor*align_factor) + \
                         ((in_shape[0] + align_factor - 1) // align_factor*align_factor)
        else:
            max_ub_use = (in_shape[0] + 1)*(
                (tensorize_size + align_factor - 1) // align_factor*align_factor)
            if tensorize_size % align_factor != 0:
                max_ub_use += align_factor
        if max_ub_use <= max_ub_size:
            return max_ub_use, True
        return max_ub_use, False

    def get_greatest_split_axis_factor(self, in_shape, out_shape, max_ptr, align_factor):
        """
        in_shape, the segment op input shape
        out_shape, the segment op output shape
        max ptr, max "max ub use size"
        return split_axis, factor

        in_shape[split_axis] % factor == 0
        """
        # first try split last axis
        split_axis = len(out_shape) - 1
        # first try split factor = 1, then factor+=1,
        # until use the this factor the mov data is bigger then ub size,
        # then the previous factor and split_axis is best
        # when the factor reach the shape[split_axis], split_axis -=1, and factor reset to 1
        # that means we can try split the earlier axis, this can decrease the outer loop
        factor = 1
        while split_axis > 0:
            tmp_factor = 2
            while tmp_factor <= out_shape[split_axis]:
                if out_shape[split_axis] % tmp_factor != 0:
                    tmp_factor += 1
                else:
                    _, valid_ub_use = self._calc_and_check_max_ub_use(split_axis, tmp_factor,
                                                                      in_shape, out_shape, max_ptr,
                                                                      align_factor)
                    if not valid_ub_use:
                        return split_axis, factor
                    factor = tmp_factor
                    tmp_factor += 1
            split_axis -= 1
            factor = 1
        return split_axis, factor

    def segment_op_tiling(self):
        """
        a simple segment max example as follow:
        input data like (4,4,5)
        |------------------------------------------------------------------------------------|
        | 100,101,102,103,104, 110,111,112,113,114, 120,121,122,123,124, 130,131,132,133,134 |
        | 200,201,202,203,204, 210,211,212,213,214, 220,221,222,223,224, 230,231,232,233,234 |
        | 300,301,302,303,304, 310,311,312,313,314, 320,321,322,323,324, 330,331,332,333,334 |
        | 400,401,402,403,404, 410,411,412,413,414, 420,421,422,423,424, 430,431,432,433,434 |
        |------------------------------------------------------------------------------------|
        as the input data above, it's like the data in memory,
        in the memory it's a continuous memory.
        In order to understand the segment operation, show the data as 4 rows,
        4 is the input_shape[0] dim size.
        segment max mean do operation with the data in the same position of each row.
        how to do the operation is refer to the segment_ids:

        output example 1:
        if the segment id [0,0,1,1], the segment id lenght is same with the the input_shape[0]
        the output is : the output shape will be (2,4,5)
        max(100,200) => 200, max(101,201) => 201, max(102,202) => 202 ...
        max(300,400) => 300, max(301,401) => 401, max(302,402) => 402 ...
        then all the output data as follow
        ----------------------------------------------------------------------------------
        200,201,202,203,204, 210,211,212,213,214, 220,221,222,223,224, 230,231,232,233,234
        400,401,402,403,404, 410,411,412,413,414, 420,421,422,423,424, 430,431,432,433,434
        ----------------------------------------------------------------------------------

        the cce code will like:
        for i1.outer in (0, 2)
            for ax0 in (0, 4) // 4 is in_shape[0]
                copy_gm_to_ub
            for i0 in (0,2)
                if i0 == 0
                    vmax()
                if i1 == 1
                    vmax()
                copy_ub_to_gm

        the dataflow in device will like:

        1) i1.outer loop 0

        copy_gm_to_ub, these data will in ub, this is continuous memory.
        |-------------------------------------------|
        | 100,101,102,103,104, 110,111,112,113,114, |
        | 200,201,202,203,204, 210,211,212,213,214, |  these is input data in ub
        | 300,301,302,303,304, 310,311,312,313,314, |
        | 400,401,402,403,404, 410,411,412,413,414, |
        |-------------------------------------------|
        |                                           |  these is for vmax result data
        |-------------------------------------------|

        1-1) inner loop 0, if i1 = 0,  do part_output_1 = vmax(part_row_1, part_row_2)
        |-------------------------------------------|
        | 100,101,102,103,104, 110,111,112,113,114, |
        | 200,201,202,203,204, 210,211,212,213,214, |  these is input data in ub
        | 300,301,302,303,304, 310,311,312,313,314, |
        | 400,401,402,403,404, 410,411,412,413,414, |
        |-------------------------------------------|
        | 200,201,202,203,204, 210,211,212,213,214, |  these is for vmax result data
        |-------------------------------------------|
                            ||
            copy_ub_to_gm   ||

        this out put data info
        |-------------------------------------------|  |------------------------------------------|
        | 200,201,202,203,204, 210,211,212,213,214, |  |                                          |
        |                                           |  |                                          |
        |-------------------------------------------|  |------------------------------------------|

        1-2) inner loop 1, if i1 = 1, do vmax
         |-------------------------------------------|
        | 100,101,102,103,104, 110,111,112,113,114, |
        | 200,201,202,203,204, 210,211,212,213,214, |  these is input data in ub
        | 300,301,302,303,304, 310,311,312,313,314, |
        | 400,401,402,403,404, 410,411,412,413,414, |
        |-------------------------------------------|
        | 400,401,402,403,404, 410,411,412,413,414, |  these is for vmax result data.(changed)
        |-------------------------------------------|
                            ||
            copy_ub_to_gm   ||
        this out put data info
        |-------------------------------------------|  |------------------------------------------|
        | 200,201,202,203,204, 210,211,212,213,214, |  |                                          |
        | 400,401,402,403,404, 410,411,412,413,414, |  |                                          |
        |-------------------------------------------|  |------------------------------------------|

        2) i1.outer loop 2
        copy these data to ub
        |------------------------------------------|
        | 120,121,122,123,124, 130,131,132,133,134 |
        | 220,221,222,223,224, 230,231,232,233,234 |  these is input data in ub
        | 320,321,322,323,324, 330,331,332,333,334 |
        | 420,421,422,423,424, 430,431,432,433,434 |
        |------------------------------------------|
        |                                          |  these is for vmax result data
        |------------------------------------------|
                            ||
                            ||
        2-1) inner loop 0, if i1=0, do vmax(part_row_1, part_row_2)
        |------------------------------------------|
        | 120,121,122,123,124, 130,131,132,133,134 |
        | 220,221,222,223,224, 230,231,232,233,234 |  these is input data in ub
        | 320,321,322,323,324, 330,331,332,333,334 |
        | 420,421,422,423,424, 430,431,432,433,434 |
        |------------------------------------------|
        | 220,221,222,223,224, 230,231,232,233,234 |  these is for vmax result data
        |------------------------------------------|
                            ||
            copy_ub_to_gm   ||
        this out put data info
        |-------------------------------------------|  |------------------------------------------|
        | 200,201,202,203,204, 210,211,212,213,214, |  | 220,221,222,223,224, 230,231,232,233,234 |
        | 400,401,402,403,404, 410,411,412,413,414, |  |                                          |
        |-------------------------------------------|  |------------------------------------------|

        2-2) inner loop 1, if i1=1, do vmax(part_row_3, part_row_4)
        |------------------------------------------|
        | 120,121,122,123,124, 130,131,132,133,134 |
        | 220,221,222,223,224, 230,231,232,233,234 |  these is input data in ub
        | 320,321,322,323,324, 330,331,332,333,334 |
        | 420,421,422,423,424, 430,431,432,433,434 |
        |------------------------------------------|
        | 420,421,422,423,424, 430,431,432,433,434 |  these is for vmax result data.(changed)
        |------------------------------------------|
                            ||
            copy_ub_to_gm   ||
        this out put data info
        |-------------------------------------------|  |------------------------------------------|
        | 200,201,202,203,204, 210,211,212,213,214, |  | 220,221,222,223,224, 230,231,232,233,234 |
        | 400,401,402,403,404, 410,411,412,413,414, |  | 420,421,422,423,424, 430,431,432,433,434 |
        |-------------------------------------------|  |------------------------------------------|

        last get the output data

        [Attention]
        here in device every command will handler a group of data.
        According to the max ub capacity and segment operation logic.
        should ensure: (in_shape[0] + 1) * group_data_size < max_ub_size

        according to device rule, the data in memory also need be aligned by align factor.
        so should ensure : (in_shape[0] + 1) * aligned_group_data_size < max_ub_size

        if the data is not aligned, like fp16 data, group_data_size % 16 != 0
        copy the result from ub to gm, will coverage the data in gm.
        example:
        in ub: |-----------------------|
               | 100,101,102,...114,   |
               |-----------------------|
               |    size = 16*n        |

        in gm  |------------------------------- |
               | 0,0.............0, 115,116,....|
               |--------------------------------|

        want copy these data from ub to gm, need copy from 115 (in gm) to ub
        after do copy_gm_ub(length=16)

        in ub:| 100,101,102,...114,   |  |115,....,131|
              |-------size 16*n-------|  |--size 16---|

        do 1) copy_ub_to_gm(length=16*n)
           2) copy_ub_to_gm(length=16)
        action as above can ensure the gm data won't be coveraged.
        should ensure ((in_shape[0] + 1) * aligned_group_data_size + (
                      align_factor if not_aligned else 0)) < max_ub_size

        according to the data flow and rules, find the greatest group_data_size
        """

        max_ptr = self.get_max_ptr()
        out_shape = [i.value for i in self._op_list[self._segment_op_pos]["dst_buffer"][0].shape]
        in_shape = [i.value for i in self._op_list[self._segment_op_pos]["src_buffer"][0].shape]
        if self._check_do_in_ai_cpu_better(in_shape):
            return 0, 0, False

        datatype = self._op_list[self._segment_op_pos]["dst_buffer"][0].dtype
        align_factor = self.get_align_factor(datatype)

        def fixed_split_info(split_axis, factor):
            """
            get split factor, and check speel or not
            """
            if factor == 1:
                # special case, last axis is a big speel number
                if split_axis == len(out_shape) - 1:
                    return split_axis, factor, False
                split_axis += 1
                factor = out_shape[split_axis]
            if (factor == out_shape[split_axis]) and split_axis == 1:
                return split_axis, factor, True

            # split not last axis, can't do speel(speel has function problem,
            # when fixed delete this check)
            if split_axis != (len(out_shape) - 1):
                return split_axis, factor, True
            # split last axis and factor == out_shape[-1]speel (speel has function problem,
            # when fixed delete this check)
            if (split_axis == (len(out_shape) - 1)) and factor == out_shape[-1]:
                return split_axis, factor, True

            return split_axis, factor, True

        split_axis, factor = self.get_greatest_split_axis_factor(in_shape, out_shape, max_ptr,
                                                                 align_factor)

        return fixed_split_info(split_axis, factor)

    def local_double_buffer(self, read_buffer):
        """
        double buffer operations
        read_buffer : the all read_cache for input in ub, type is list
        """
        for i in read_buffer:
            self._schedule[i].double_buffer()
