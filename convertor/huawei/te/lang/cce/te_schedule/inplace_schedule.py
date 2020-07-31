"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0. You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

inplace schedule, provide a schedule for inplace compute
"""
from te import tvm
from te import platform as cceconf
from .util import get_align_factor
from .util import shape_to_list

DOUBLE_BUFFER_THRESHOLD = 4
ALIGN_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256]


# pylint: disable=too-many-instance-attributes
class CceInplaceOp:
    """class of cce index op

    Parameters
    ----------
    scope : cal buffer like local.UB

    Returns
    -------
    CceOp_instance : instance of CceOp

    """
    def __init__(self, scope):
        self._scope = scope
        self._schedule = None
        self._op_list = []
        self._inplace_op_pos = None
        self._spec_node_list = []
        self._get_op_list_traversed_tensor = set()
        # input buffers in gm
        self._read_buffer = []
        # output buffers in gm
        self._write_buffer = []
        # cache_read in ub
        self._read_cache = []
        # cache_write in ub
        self._write_cache = []
        self._reshape_computeinline_buffer = []
        self._reshape_computeinline_cache = []

        # using for tiling
        if self._scope.lower().find('.ub') != -1:
            self._total_size = cceconf.get_soc_spec("UB_SIZE")
        else:
            raise RuntimeError("only support UB buffer now")
        # get device_core_num
        self._core_dim = cceconf.get_soc_spec("CORE_NUM")
        self._ub_tiling_max_size = self._total_size
        self._need_double_buffer = False
        self._need_block_tiling = False
        self._block_split_axis = None
        self._block_split_nparts = None
        self._block_split_factor = None
        self._ub_split_axis = None
        self._ub_split_nparts = None
        self._ub_split_factor = None
        self._ub_split_xo = None
        self._ub_split_xi = None
        self._first_axis = None

    def schedule(self, res, spec_node_list, sch_list):
        """
        schedule entry of inplace
        """
        self._schedule = sch_list[0]
        self._spec_node_list = spec_node_list

        # analyze op
        self._get_op_list(res)
        self._decode_buffer()
        self._locate_inplace_op_pos()
        self._locate_reshape_computeinline_op_pos()

        # data flow
        self._do_cache_read()
        self._do_cache_write()
        self._do_compute_inline()

        # data tiling
        self._calculate_tiling()
        self._do_ub_tiling(res)
        self._do_compute_at(res)

        # intrinsic
        self._do_emit_insn_dma_copy(res)
        self._do_emit_insn_inplace()

        if self._need_block_tiling:
            self._do_block_tiling(res)
        if self._need_double_buffer:
            self._do_double_buffer()

        sch_list[0] = self._schedule
        return True

    def _do_cache_read(self):
        def _get_input_reader_list(in_tensor):
            reader_list = []
            for op_i in self._op_list:
                if in_tensor in op_i['src_buffer']:
                    for reader_i in op_i['dst_buffer']:
                        reader_list.append(reader_i)
            return reader_list

        # PlaceholderOp, cache_read
        self._read_cache = []
        for in_tensor in self._read_buffer:
            reader_list = _get_input_reader_list(in_tensor)
            self._read_cache.append(self._schedule.cache_read(in_tensor, self._scope, reader_list))

    def _do_cache_write(self):
        # ComputeOp, cache_write
        self._write_cache = [self._schedule.cache_write(out, self._scope)
                             for out in self._write_buffer]

    def _do_compute_inline(self):
        # ComputeOp, compute_inline
        # res is _write_buffer[0], only compute_inline _write_buffer[1:]
        for out in self._write_buffer[1:]:
            self._schedule[out].compute_inline()

        for out in self._reshape_computeinline_buffer:
            idx = self._write_buffer.index(out)
            self._schedule[self._write_cache[idx]].compute_inline()

            self._reshape_computeinline_cache.append(self._write_cache[idx])
            del self._write_cache[idx]
            del self._write_buffer[idx]

    def _do_compute_at(self, res):
        """
        tvm.compute_at operations
        """
        lhs_related_cache, rhs_related_cache = self._get_inplace_input_related_cache()
        # placeholder and elewise computeOp before inplace
        for i in rhs_related_cache:
            self._schedule[i].compute_at(self._schedule[res], self._ub_split_xo)

        # rhTensor of InplaceOp need storage_align, except (xx,xx) reshape (1,xx,xx)
        if not self._reshape_computeinline_cache:
            for i in rhs_related_cache:
                align_factor, _ = get_align_factor(i.dtype)
                self._schedule[i].storage_align(self._schedule[i].op.axis[0], align_factor, 0)

        # placeholder and elewise computeOp before inplace
        for i in lhs_related_cache:
            self._schedule[i].compute_at(self._schedule[res], self._first_axis)

        # [res 0 ~ elewise ~, inplace_op_pos, reshapeComputeinlineOp ~ placeholder]
        # inplace and elewise computeOp after inplace
        inplace_buffer = self._op_list[self._inplace_op_pos]["dst_buffer"][0]
        for i in self._write_cache[0:self._write_buffer.index(inplace_buffer) + 1]:
            self._schedule[i].compute_at(self._schedule[res], self._first_axis)

    def _do_emit_insn_inplace(self):
        for op_index in range(len(self._op_list)):
            out_buffer = self._op_list[op_index]["dst_buffer"][0]
            op_name = self._op_list[op_index]["op"]

            # reshape computeOp
            if op_name == "":
                continue

            # InplaceOp
            elif op_index == self._inplace_op_pos:
                # for using in emit_insn callback in cce_intrin_md
                cceconf.cce_emitinsn_params.cceEmitParamsIns.insert_params(
                    {"inplace_ids": self._op_list[op_index]["inplace_ids"]})
                out_cache = self._write_cache[self._write_buffer.index(out_buffer)]
                self._schedule[out_cache].emit_insn(
                    self._schedule[out_cache].op.axis[self._ub_split_axis],
                    op_name)

            # EleWiseOp after inplace
            elif op_index < self._inplace_op_pos:
                out_cache = self._write_cache[self._write_buffer.index(out_buffer)]
                self._schedule[out_cache].emit_insn(
                    self._schedule[out_cache].op.axis[0],
                    op_name)

            # EleWiseOp before inplace
            else:
                out_cache = self._write_cache[self._write_buffer.index(out_buffer)]
                self._schedule[out_cache].emit_insn(
                    self._schedule[out_cache].op.axis[self._ub_split_axis],
                    op_name)

    def _do_emit_insn_dma_copy(self, res):
        for i in self._read_cache:
            self._schedule[i].emit_insn(
                self._schedule[i].op.axis[0],
                cceconf.dma_copy)

        self._schedule[res].emit_insn(self._ub_split_xi,
                                      cceconf.dma_copy, {"no_overlap": 1})

    def _do_double_buffer(self):
        """
        double buffer operations
        read_buffer : the all read_cache for input in ub, type is list
        """
        for i in self._read_cache:
            self._schedule[i].double_buffer()

    def _get_op_list(self, tmp):
        tmp_op = {"op": None,
                  "dst_buffer": [],
                  "src_buffer": [],
                  "args": None,
                  "inplace_ids": None}

        # ComputeOp: find the associated input_tensor, from the result_tensor of ComputeOp
        str_list = tmp.op.tag.split("|")
        tmp_op["op"] = str_list[0]
        tmp_op["dst_buffer"].append(tmp)
        for in_tensor in list(tmp.op.input_tensors):
            tmp_op['src_buffer'].append(in_tensor)

        # inplace_ids and other args
        if 'inplace' in str_list[0].split('_'):
            inplace_ids_str = str_list[1].split(",")
            inplace_ids = [int(i) for i in inplace_ids_str]
            tmp_op["inplace_ids"] = inplace_ids

        # traversing: input_tensors is ComputeOp result buffer (not PlaceholderOp)
        for in_tensor in list(tmp.op.input_tensors):
            if (not isinstance((in_tensor.op), tvm.tensor.PlaceholderOp)) \
                and (in_tensor not in self._spec_node_list) \
                and (in_tensor not in self._get_op_list_traversed_tensor):
                self._get_op_list_traversed_tensor.add(in_tensor)
                self._get_op_list(in_tensor)

        self._op_list.append(tmp_op)

    def _decode_buffer(self):
        read_buffer = set()
        write_buffer = []
        self._op_list.reverse()
        for op_i in self._op_list:
            read_buffer = read_buffer | set(op_i['src_buffer'])
            write_buffer = write_buffer + op_i['dst_buffer']
        for op_i in self._op_list:
            read_buffer = read_buffer - set(op_i['dst_buffer'])
        self._read_buffer = list(read_buffer)
        self._write_buffer = list(write_buffer)

    def _locate_inplace_op_pos(self):
        for idx in range(len(self._op_list)):
            if "inplace" in self._op_list[idx]["op"]:
                self._inplace_op_pos = idx
                break

    def _locate_reshape_computeinline_op_pos(self):
        for idx in range(len(self._op_list)):
            if self._op_list[idx]["op"] == "" and \
               self._op_list[idx]["dst_buffer"][0].name == "reshapeComputeinlineOp":
                self._reshape_computeinline_buffer.append(self._op_list[idx]["dst_buffer"][0])
                break

    def get_max_ub_count(self, total_size, dtype):
        """
        |-------------------------------------------/
        | 200,201,202,203,204, 210,211,212,213,214, /  lhs, these is input data in ub
        |-------------------------------------------/
        | 100,101,102,103,104, 110,111,112,113,114, /
        | 200,201,202,203,204, 210,211,212,213,214, /  rhs, these is input data in ub
        | 300,301,302,303,304, 310,311,312,313,314, /
        | 400,401,402,403,404, 410,411,412,413,414, /
        |-------------------------------------------/
        | 200,201,202,203,204, 210,211,212,213,214, /  res, these is for vmax result data
        |-------------------------------------------/
        """
        ids_len = len(self._op_list[self._inplace_op_pos]["inplace_ids"])
        inplace_buffer = self._op_list[self._inplace_op_pos]["dst_buffer"][0]
        lhs_related_cache, rhs_related_cache = self._get_inplace_input_related_cache()
        row_nums = len(self._write_cache[0:self._write_buffer.index(inplace_buffer) + 1]) + \
                    len(lhs_related_cache) + \
                    len(rhs_related_cache)*ids_len
        max_shape0 = (total_size // 32 - \
                      len(self._write_cache[0:self._write_buffer.index(inplace_buffer) + 1]) - \
                      len(lhs_related_cache)) // len(rhs_related_cache)

        # base on the diff data type, get the max count based on usable ub
        _, dtype_bytes = get_align_factor(dtype)
        _align = int(32 // dtype_bytes)
        max_ub_count = int(total_size // (dtype_bytes * row_nums))
        max_ub_count = int(max_ub_count // _align * _align)
        if max_ub_count < _align:
            raise RuntimeError(
                "inpalce rhTensor shape[0] is larger than %s, causing max_ub_count less than 32B"
                % max_shape0)
        return max_ub_count

    def _get_inplace_input_related_cache(self):
        related_buffer = set()

        def _get_storage_align_list(tmp_buffer):
            # ComputeOp: find the associated input_tensor, from the result_tensor of ComputeOp
            related_buffer.add(tmp_buffer)
            for in_tensor in list(tmp_buffer.op.input_tensors):
                related_buffer.add(in_tensor)

            # traversing: input_tensors is InplaceComputeOp result buffer (not PlaceholderOp)
            for in_tensor in list(tmp_buffer.op.input_tensors):
                if not isinstance((in_tensor.op), tvm.tensor.PlaceholderOp):
                    _get_storage_align_list(in_tensor)

        lhs = self._op_list[self._inplace_op_pos]["src_buffer"][0]
        rhs = self._op_list[self._inplace_op_pos]["src_buffer"][1]

        related_buffer.clear()
        _get_storage_align_list(rhs)
        rhs_related_buffer = list(related_buffer)

        related_buffer.clear()
        _get_storage_align_list(lhs)
        lhs_related_buffer = list(related_buffer)

        def _buffer_to_cache(buffer_data):
            _cache = []
            for i in buffer_data:
                if i in self._write_buffer:
                    _cache.append(self._write_cache[self._write_buffer.index(i)])
                elif i in self._read_buffer:
                    _cache.append(self._read_cache[self._read_buffer.index(i)])
                elif i in self._reshape_computeinline_buffer:
                    continue
                else:
                    raise RuntimeError("_buffer_to_cache error")
            return _cache

        return _buffer_to_cache(lhs_related_buffer), _buffer_to_cache(rhs_related_buffer)

    # find the block split axis, shape = (shape[0], ..., shape[split_axis], shape[n])
    # so that shape[0]*...*shape[split_axis-1] < core_dim
    # and shape[0]*...*shape[split_axis] > core_dim
    # pylint: disable=no-self-use
    def _calculate_block_tiling(self, shape, core_dim):
        # get block split axis
        bound_size = 1
        split_axis = 0
        for i, shape_i in enumerate(shape):
            bound_size = int(shape_i * bound_size)
            split_axis = i
            if bound_size >= core_dim:
                break

        tmp_size = int(bound_size // shape[split_axis])
        # get block outer, inner
        if bound_size <= core_dim:
            outer = shape[split_axis]
            inner = 1
        else:
            outer = int(core_dim // tmp_size)
            inner = int((shape[split_axis] + outer - 1) // outer)

        # split_axis != 0 means that it need to fuse, fuse can't be non-divisible split
        # Non-divisible split -> reorder -> bind multicore,
        # ir_pass IntSet bug: cloud ok, mini "Cannot match type handle64 vs int32"
        if split_axis != -1:
            tmp_outer = 1
            for i in range(outer, shape[split_axis]+1):
                if shape[split_axis] % i == 0:
                    tmp_outer = i
                    break
            outer = tmp_outer
            inner = int(shape[split_axis] // outer)

        return outer, inner, split_axis

    # find the ub split axis, shape = (shape[0], ..., shape[split_axis], shape[n])
    # so that shape[split_axis + 1]*...*shape[n] < ub_tiling_max_size
    # and shape[split_axis ]* shape[split_axis + 1]*...*shape[n] > ub_tiling_max_size
    # pylint: disable=no-self-use
    def _calculate_ub_tiling(self, shape, ub_tiling_max_size, align_factor=128):
        # get ub split axis
        bound_size = 1
        split_axis = len(shape) - 1
        for i in reversed(range(len(shape))):
            bound_size = int(shape[i] * bound_size)
            split_axis = i
            if bound_size >= ub_tiling_max_size:
                break

        tmp_size = int(bound_size // shape[split_axis])
        # get ub outer, inner
        if bound_size <= ub_tiling_max_size:
            inner = shape[split_axis]
            outer = 1
        else:
            inner = int(ub_tiling_max_size // tmp_size)
            outer = int((shape[split_axis] + inner - 1) // inner)

        # try 128 split
        mod2count = 0
        while tmp_size % 2 == 0:
            tmp_size = tmp_size // 2
            mod2count = mod2count + 1
        mod2need = ALIGN_LIST.index(align_factor) - mod2count
        if mod2need <= 0:
            return outer, inner, split_axis

        factor = ALIGN_LIST[mod2need]
        if inner >= factor:
            inner = int(inner // factor * factor)
            outer = int((shape[split_axis] + inner - 1) // inner)

        return outer, inner, split_axis

    # pylint: disable=no-self-use
    def _need_inpalce_block_tiling(self, shape, datatype):
        align_factor, _ = get_align_factor(datatype)
        mul_shape = 1
        for i in range(1, len(shape)):
            mul_shape = mul_shape * shape[i]
        if mul_shape < align_factor:
            need_block_tiling = False
        else:
            need_block_tiling = True

        return need_block_tiling

    def _need_inpalce_doublebuffer(self, shape, core_dim, ub_tiling_max_size):
        ub_outer, _, ub_split_axis = self._calculate_ub_tiling(shape[1:], ub_tiling_max_size)
        ub_split_axis = ub_split_axis + 1
        if ub_split_axis == 1:
            return False

        _, block_inner, block_split_axis = self._calculate_block_tiling(
            shape[1:ub_split_axis], core_dim)
        block_split_axis = block_split_axis + 1

        # (a_1,..,(a_bo,a_bi),...,(a_uo,a_ui),...,a_n)
        mul_shape = block_inner * ub_outer
        for i in range(block_split_axis+1, ub_split_axis):
            mul_shape = mul_shape * shape[i]

        need_double_buffer = False
        if mul_shape > DOUBLE_BUFFER_THRESHOLD:
            need_double_buffer = True

        return need_double_buffer


    def _calculate_tiling(self):
        """
        (a_0, .., a_t, ..., a_k, ..., a_n)
        the shape is split to:
        (a_0, .., [a_bo,a_bi], ..., [a_uo,a_ui], ..., a_n)
        and reorder to:
        (a_1,.., [a_bo,a_bi], ...,[a_uo], a_0, [a_ui], ..., a_n)

        If shape.dim == 1:
        ie. ub_split_axis=0, (a_0,) => (xo=1,first=a_0,xi=1)
            # Multicore is not enabled
            # double buffer is not enabled
            # no reorder

        Other shape.dim > 1, ub_split_axis >= 0
        Enable multi-core, single row data >32B, otherwise multi-core cannot be enabled

        If ub_split_axis == 1:
        ie ([a_uo], a_0, [a_ui], ..., a_n)
            # Enable multicore at a_uo
            # double buffer is not enabled

        If ub_split_axis >1 and block_split_axis = 1:
        ie ([a_bo], [a_bi], [a_uo], a_0, [a_ui], ..., a_n)
            # Enable multi-core at a_bo, not do fuse can be non-divisible
            # Determine a_bi*a_uo is greater than the threshold to enable double buffer

        If ub_split_axis >1 and block_split_axis > 1:
        ie (a_1,.., [a_bo], [a_bi], ...,[a_uo], a_0, [a_ui], ..., a_n)
            # Enable multi-core at a_bo, do fuse need to divide the segmentation
            # Determine a_bi*...*a_uo is greater than the threshold to enable double buffer
        """
        shape = shape_to_list(self._op_list[self._inplace_op_pos]["dst_buffer"][0].shape)
        datatype = self._op_list[self._inplace_op_pos]["dst_buffer"][0].dtype
        self._ub_tiling_max_size = self.get_max_ub_count(self._total_size, datatype)
        # double buffer
        try:
            self.get_max_ub_count(self._total_size // 2, datatype)
        except RuntimeError:
            # rhTensor shape[0] is to large, could not double buffer
            self._need_double_buffer = False
        else:
            self._need_double_buffer = True
            self._total_size = self._total_size // 2
            self._ub_tiling_max_size = self.get_max_ub_count(self._total_size, datatype)

        if len(shape) == 1:
            self._need_block_tiling = False
            self._ub_split_axis = 0
            self._ub_split_nparts = shape[0]
            self._ub_split_factor = 1
            return

        # Multi-core processing cannot be done with data less than 32B
        self._need_block_tiling = self._need_inpalce_block_tiling(shape, datatype)

        ub_outer, ub_inner, ub_split_axis = self._calculate_ub_tiling(
            shape[1:], self._ub_tiling_max_size)
        ub_split_axis = ub_split_axis + 1
        if ub_split_axis == 1:
            self._ub_split_axis = ub_split_axis
            self._ub_split_nparts = ub_outer
            self._ub_split_factor = ub_inner
            return

        # ub_split_axis is not axis0, becuase of multicore first priority
        # (a_1,..,(a_bo,a_bi),...,(a_uo,a_ui),...,a_n)
        block_outer, block_inner, block_split_axis = self._calculate_block_tiling(
            shape[1:ub_split_axis], self._core_dim)
        self._block_split_axis = block_split_axis + 1
        self._block_split_nparts = block_outer
        self._block_split_factor = block_inner
        self._ub_split_axis = ub_split_axis
        self._ub_split_nparts = ub_outer
        self._ub_split_factor = ub_inner
        return

    def _do_ub_tiling(self, res):
        if self._ub_split_axis == 0:
            axo, self._ub_split_xi = self._schedule[res].split(
                res.op.axis[self._ub_split_axis], factor=self._ub_split_factor)
            self._ub_split_xo, self._first_axis = self._schedule[res].split(axo, nparts=1)
        else:
            self._first_axis = self._schedule[res].op.axis[0]
            self._ub_split_xo, self._ub_split_xi = self._schedule[res].split(
                res.op.axis[self._ub_split_axis], factor=self._ub_split_factor)
            reorder_axis_list = self._schedule[res].op.axis[1:self._ub_split_axis] + \
                                [self._ub_split_xo, ] + \
                                [self._first_axis, ] + \
                                [self._ub_split_xi, ] + \
                                self._schedule[res].op.axis[self._ub_split_axis+1:]
            self._schedule[res].reorder(*reorder_axis_list)

    def _do_block_tiling(self, res):
        if self._ub_split_axis == 1:
            thread_block = tvm.thread_axis("blockIdx.x")
            self._schedule[res].bind(self._ub_split_xo, thread_block)
            return

        block_split_xo, _ = self._schedule[res].split(
            res.op.axis[self._block_split_axis], factor=self._block_split_factor)
        if self._block_split_axis == 1:
            bind_axis = block_split_xo
        else:
            fuse_axis_list = []
            for i in range(1, self._block_split_axis):
                fuse_axis_list.append(res.op.axis[i])
            fuse_axis_list.append(block_split_xo)
            bind_axis = self._schedule[res].fuse(*fuse_axis_list)

        thread_block = tvm.thread_axis("blockIdx.x")
        self._schedule[res].bind(bind_axis, thread_block)
        return
