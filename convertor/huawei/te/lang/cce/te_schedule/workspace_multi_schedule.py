#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

elewise mutil out schedule
"""
from te import platform as cceconf
from te import tvm
from . import util
from .cce_schedule_declarations import OpSpecTypes


# pylint: too-many-return-statements,too-few-public-methods,
# pylint: too-many-arguments,too-many-statements,no-self-use,too-many-lines,
# pylint: too-many-instance-attributes,too-many-branches,
class WorkspaceMultiSchedule(): # pylint: disable=too-many-instance-attributes
    """
    class of cce schedule for enable multi core in workspace

    Parameters
    ----------
    WorkspaceMultiSchedule: class of workspace multi core schedule

    Returns
    -------
    WorkspaceMultiSchedule_instance : instance of WorkspaceMultiSchedule
    """

    def __init__(self, pattern_limit):
        self._enable = True
        self._fake_axis = None
        self._black_list = []
        self._white_list = []
        self._spec_list = []
        self._tensor_list = []
        self._schedule = None
        self._info = None
        self._res = None
        self._tiling_axis = None
        self._tiling_factor = None
        self._multi_core_axis = None
        self._compute_at_list = []
        self._over_dma = []
        self._op_type = None
        if not pattern_limit:
            self._enable = False

    def set_op_type(self, op_type):
        """
        set operation type
        """
        self._op_type = op_type

    # pylint: disable=attribute-defined-outside-init, unused-argument, arguments-differ
    def update_schedule(self, spec_node_list, sch_list, spec_mid_list):
        """
        use fake node to fuse multi out in order to compute at
        :param spec_node_list: special node and outs
        :param sch_list: schedule list
        :param spec_mid_list: to append fake node when fusion is enable
        :return:
        """
        if not util.MULTI_WORKSPACE or not self._enable:
            return
        self._spec_list = spec_node_list
        self._calculate_limit_list()
        self._calculate_tensor_list(spec_node_list)
        self._check_enable()
        if not self._enable:
            return
        # one out can use last as compute_at target
        if len(spec_node_list) - len(spec_mid_list) == 1 and self._op_type != OpSpecTypes.MVN:
            workspace_node = spec_node_list[:-1]
            for tensor in workspace_node:
                if tensor not in spec_mid_list:
                    self._enable = False
                    return
            self._res = spec_node_list[-1]
            self._schedule = sch_list[0]
            if self._res.shape[-1].value < util.LOG_SOFTMAX_LIMIT and \
                self._res.shape[-1].value != util.LOG_SOFTMAX_MATCH:
                self._enable = False
            return
        # multi out need use fake node as compute_at target
        self._res = util.fake_node_fuse_fun(self._spec_list)
        if self._op_type != OpSpecTypes.MVN and \
                (self._res == util.FAKE_NODE_FUSE_FAILED or not util.MULTI_WORKSPACE_ALL):
            self._enable = False
            return
        self._fake_axis = self._res.op.axis[0]
        self._schedule = tvm.create_schedule([self._res.op])
        sch_list[0] = self._schedule

    # pylint: disable=attribute-defined-outside-init, unused-argument, arguments-differ
    def do_schedule(self, workspace_info):
        """
        do schedule for enable multi core in workspace
        :param workspace_info: single workspace node information collection
        :return:
        """
        if not self._enable or not util.MULTI_WORKSPACE:
            return

        self._info = workspace_info

        if self._check_for_limit() and self._calculate_basic_info():
            self._do_tiling()
            self._do_over_dma()
            self._do_compute_at()
            self._do_multi_core()

        self._handle_fake_node()

        return

    # pylint: disable=attribute-defined-outside-init, unused-argument, arguments-differ
    def _check_for_limit(self):
        if len(self._res.shape) != 2 and self._op_type != OpSpecTypes.MVN:
            return False
        visited = []
        next_list = list(self._info.keys())
        while next_list:
            cur_list = next_list[:]
            next_list = []
            for tensor in cur_list:
                if tensor in visited:
                    continue
                visited.append(tensor)
                if "div" in tensor.op.tag:
                    return False
                next_list += tensor.op.input_tensors
        return True

    def _calculate_basic_info(self): # pylint: disable=too-many-branches
        def __is_tiling_factor_right(block_cut_size):
            if self._op_type == OpSpecTypes.MVN:
                align_type = self._res.dtype
                # bool is represented by int8
                if align_type == 'bool':
                    align_type = 'int8'
                align_factor, _ = util.get_align_factor(align_type)
                if self._tiling_factor < align_factor:
                    return False
            else:
                if block_cut_size % self._tiling_factor:
                    self._tiling_factor = util.INIT_SIZE
            return True

        relax_axis = len(self._res.shape)
        for tensor in self._info.keys():
            if tensor != self._res:
                self._compute_at_list.append(tensor)
            relax_axis = min(self._info[tensor]["split_axis_index"], relax_axis)
        if relax_axis == 0:
            return False

        # reserve size for gm align
        reserve_size = util.INIT_SIZE
        for tensor in self._info.keys():
            cur_reserve = util.ceil(int(util.VECTOR_ONE_BLOCK_UNIT //
                                        util.DTYPE_WIDTH_MAP[tensor.dtype]),
                                    util.get_shape_size(
                                        util.shape_to_list(tensor.shape)[relax_axis:]))
            if cur_reserve != util.INIT_SIZE:
                self._over_dma.append(tensor)
                self._compute_at_list.append(self._info[tensor]["cache_buffer_map"][tensor])
                reserve_size = max(cur_reserve, reserve_size)
                while util.REDUCE_OP_TAG_LABEL not in tensor.op.name:
                    p_tensor = tensor.op.input_tensors[0]
                    self._compute_at_list.append(self._info[tensor]["cache_buffer_map"][p_tensor])
                    tensor = p_tensor

        multi_core_shape = []
        for i in range(relax_axis):
            target_dim_size = self._res.shape[i].value
            for tensor in self._spec_list:
                if tensor.shape[i].value != target_dim_size:
                    target_dim_size = util.DEFAULT_INDEX
                    break
            if target_dim_size == util.DEFAULT_INDEX:
                break
            multi_core_shape.append(target_dim_size)

        multi_core_size = util.get_shape_size(multi_core_shape)
        multi_core_num = min(int(multi_core_size // reserve_size),
                             cceconf.get_soc_spec("CORE_NUM"))
        if multi_core_size == util.INIT_SIZE or \
                multi_core_num <= util.INIT_SIZE:
            return False

        multi_core_barrier = [len(multi_core_shape)]
        multi_core_shape.append(util.INIT_SIZE)
        self._tiling_axis, self._tiling_factor = \
            util.get_block_factor_radical(multi_core_shape, multi_core_barrier, multi_core_num)
        # fix tiling, for reduce tensorize will meet error in irregularities
        block_cut_size = util.INIT_SIZE
        for i in self._tiling_axis:
            block_cut_size *= multi_core_shape[i]

        return __is_tiling_factor_right(block_cut_size)

    def _do_tiling(self):
        sch = self._schedule
        if len(self._tiling_axis) == 1:
            target_axis = sch[self._res].op.axis[self._tiling_axis[0]]
        else:
            target_axes = []
            for i in self._tiling_axis:
                target_axes.append(sch[self._res].op.axis[i])
            target_axis = sch[self._res].fuse(*target_axes)
        i_outer, i_inner = sch[self._res].split(target_axis, self._tiling_factor)
        if self._fake_axis is not None:
            self._fake_axis = i_inner
        self._multi_core_axis = i_outer

    def _do_over_dma(self):
        sch = self._schedule
        for tensor in self._over_dma:
            sch[tensor].emit_insn(tensor.op.axis[0], util.DMA_COPY_PRAGMA)

    def _do_compute_at(self):
        sch = self._schedule
        for tensor in self._compute_at_list:
            sch[tensor].compute_at(sch[self._res], self._multi_core_axis)

    def _do_multi_core(self):
        res = self._res
        block = tvm.thread_axis("blockIdx.x")
        self._schedule[res].bind(self._multi_core_axis, block)

    def _handle_fake_node(self):
        if self._fake_axis is not None and util.FAKE_NODE_TAG == self._res.op.tag:
            self._schedule[self._res].set_scope("")
            self._schedule[self._res].emit_insn(self._fake_axis,
                                                util.FAKE_NODE_PRAGMA)

    def _check_enable(self):
        # white list empty means all
        if self._white_list:
            for tensor in self._tensor_list:
                if tensor.op.tag not in self._white_list:
                    self._enable = False
                    return
        # black list empty means none
        if self._black_list:
            for tensor in self._tensor_list:
                if tensor.op.tag in self._black_list:
                    self._enable = False
                    return
        if len(self._spec_list) <= 1:
            self._enable = False

    def _calculate_tensor_list(self, tensor_list):
        for tensor in tensor_list:
            if tensor not in self._tensor_list:
                self._tensor_list.append(tensor)
                self._calculate_tensor_list(tensor.op.input_tensors)

    def _calculate_limit_list(self):
        self._white_list = []
        self._black_list = ["tuple_reduce_sum"]
