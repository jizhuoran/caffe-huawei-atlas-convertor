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

"""

#!/usr/bin/python
# -*- coding: UTF-8 -*-

from te import tvm
from .util import Compare
from .util import ceil_div


class AttachMap:
    """docstring for  AttachMap"""
    def __init__(self):
        # axis:stage
        self._parent_stages = dict()
        self._attached_path = dict()

    def record_attach(self, stage, scope):
        """
        record the attach situation of stage

        Parameters
        ----------
        stage : the processing of compute
        scope : means axis

        Returns
        -------
        None
        """
        if self._attached_path.get(scope) is None:
            self._attached_path[scope] = [stage]
        else:
            self._attached_path[scope].append(stage)

    def follow_with(self, stage, parent_stage, scope):
        """
        update the _parent_stages, record the attach

        Parameters
        ----------
        stage : the processing of compute
        parent_stage : the parent tensor
        scope : means axis

        Returns
        -------
        scope : the attached axis
        """
        self._parent_stages[scope] = parent_stage
        self.record_attach(stage, scope)
        return scope

    def record_same_attach(self, stage, ref_stage):
        """
        attach the stage same as ref_stage

        Parameters
        ----------
        stage : the processing of compute
        ref_stage : the reference tensor

        Returns
        -------
        None
        """
        for scope, s_list in self._attached_path.items():
            if ref_stage in s_list:
                self.record_attach(stage, scope)
                return scope
        return None

    def update_scope(self, scope, new_scope):
        """
        replace the scope by new_scope

        Parameters
        ----------
        scope : the old axis
        new_scope : the new axis

        Returns
        -------
        None
        """
        if scope == new_scope:
            return
        child_stages = self._attached_path.get(scope)
        if child_stages is None:
            return

        attached_path = self._attached_path
        if attached_path.get(new_scope) is None:
            attached_path[new_scope] = child_stages
        else:
            attached_path[new_scope].extend(child_stages)

        self._attached_path.pop(scope)

        self._parent_stages[new_scope] = self._parent_stages[scope]
        return

    def apply(self):
        """
        begin the compute_at operation
        according to _parent_stages and _attached_path

        Parameters
        ----------

        Returns
        -------
        None
        """
        for scope, array_stages in self._attached_path.items():
            parent = self._parent_stages[scope]
            pre_scope = None
            for axis in parent.leaf_iter_vars:
                if axis == scope:
                    break
                pre_scope = axis
            if pre_scope is not None:
                for stage in array_stages:
                    stage.compute_at(parent, pre_scope)

    @property
    def attached_path(self):
        """
        get the _attached_path

        Parameters
        ----------

        Returns
        -------
        None
        """
        return self._attached_path

    @property
    def parent_stages(self):
        """
        get the _parent_stages

        Parameters
        ----------

        Returns
        -------
        None
        """
        return self._parent_stages


class ScopeManager:
    """docstring for ScopeManager
    keep active tensor same in whole schedule
    """

    def __init__(self, stage):
        # super(ScopeManager, self).__init__()
        self._stage = stage
        self._axis_unit = dict()
        self._active_scopes = list()
        self._origin_axis = list()
        self._last_attached = None
        self._scope_intrinsic = None
        if len(stage.leaf_iter_vars) != len(stage.all_iter_vars):
            raise RuntimeError("Op should be init before schedule")
        for axis in stage.leaf_iter_vars:
            self._axis_unit[axis] = [1, axis.dom.extent.value]
            self._active_scopes.append(axis)
            self._origin_axis.append(axis)

    @property
    def op(self):
        """
        get stage's op
        """
        return self._stage.op

    @property
    def origin_axis(self):
        """
        get stage's _origin_axis
        """
        return self._origin_axis

    @property
    def scope_intrinsic(self):
        """
        get stage's _scope_intrinsic
        """
        return self._scope_intrinsic

    @property
    def last_attached(self):
        """
        get stage's _last_attached
        """
        return self._last_attached

    def reused_by(self, *args):
        """
        stage's original reused_by function
        """
        self._stage.reused_by(*args)

    def split(self, parent, factor=None, nparts=None):
        """
        apply split to scope

        Parameters
        ----------
        parent : Itervar
        factor : int
        nparts : int

        Returns
        -------
        outer : Itervar
        inner : Itervar

        """

        if factor is None and nparts is None:
            raise RuntimeError("factor nparts can not be None")

        if self._axis_unit.get(parent) is None:
            raise RuntimeError("parent scope can not be None")

        unit, extent = self._axis_unit[parent]
        if nparts is not None:
            outer, inner = self._stage.split(parent, nparts=nparts)
            factor = ceil_div(extent, nparts)
            self._axis_unit[inner] = [unit, factor]
            self._axis_unit[outer] = [factor * unit, nparts]
            if parent in self._active_scopes:  # not else
                self._update_active_scope(parent, inner)
        else:
            outer, inner = self._stage.split(parent, factor=factor)
            self._axis_unit[inner] = [unit, factor]
            self._axis_unit[outer] = [unit * factor, ceil_div(extent, factor)]
            if parent in self._active_scopes:  # not else
                self._update_active_scope(parent, outer)
        return outer, inner

    def reorder(self, *args):
        """
        stage's reorder function
        need check if axises in args is valid
        do not need all axised in this scope
        """
        visited_scope = set()
        scopes = list(args)
        leaf_ivars = self._stage.leaf_iter_vars
        valid_scopes = list()
        scopes.reverse()
        for axis in scopes:
            if axis is not None \
                and axis not in visited_scope \
                and axis in leaf_ivars:
                valid_scopes.append(axis)
                visited_scope.add(axis)
        if len(valid_scopes) <= 1:
            return
        valid_scopes.reverse()
        self._stage.reorder(*valid_scopes)

    def double_buffer(self):
        """
        stage's original double_buffer function
        """
        self._stage.double_buffer()

    def unroll(self, var):
        """
        stage's original unroll function
        """
        self._stage.unroll(var)

    def buffer_align(self, *arg):
        """
        stage's original buffer_align function
        """
        self._stage.buffer_align(*arg)

    def buffer_tile(self, *arg):
        """
        stage's original buffer_tile function
        """
        self._stage.buffer_tile(*arg)

    def pragma(self, var, pragma_type, pragma_value=None):
        """
        stage's original pragma function
        """
        self._stage.pragma(var, pragma_type, pragma_value)

    def storage_align(self, axis, factor, offset):
        """
        stage's original storage_align function
        """
        self._stage.pragma(axis, factor, offset)

    def get_active_scope_and_unit(self):
        """
        get _active_scopes and unit_list

        Returns
        -------
        _active_scopes : the scopes(axis) now used
        unit_list : the split part of axis
        """
        if not self._check_active_scopes(self._active_scopes):
            raise RuntimeError("active itervar should in leaf_iter_vars")
        unit_list = list()
        for axis in self._active_scopes:
            unit, _ = self._axis_unit[axis]
            unit_list.append(unit)

        return self._active_scopes, unit_list

    def get_active_scopes(self):
        """
        get _active_scopes
        _active_scopes: the scopes(axis) now used
        """
        if not self._check_active_scopes(self._active_scopes):
            raise RuntimeError("active itervar should in leaf_iter_vars")
        return self._active_scopes

    def _update_active_scope(self, ax_before, ax_after):
        active_scopess = self._active_scopes
        index = active_scopess.index(ax_before)
        active_scopess[index] = ax_after

    def nlast_scopes(self, n_scope):
        """
        get n last axises of this stage
        """
        if n_scope <= 0:
            raise ValueError("n_scope must >0")

        leaf_ivars = list(self._stage.leaf_iter_vars)
        if n_scope > len(leaf_ivars):
            raise ValueError("n_scope must less equal to leaf_ivars")
        return leaf_ivars[-n_scope::]

    def intrin_scopes(self, nlast=0):
        """
        this function developed for mmad
        split the axis of scope and reorder
        return the nlast axises for emit insn
        """
        n_scope_intrin = len(self._origin_axis)
        nlast = n_scope_intrin if nlast == 0 else nlast

        if nlast < 0 or nlast > len(self._origin_axis):
            raise RuntimeError("nlast must >0 and < %d" %
                               len(self._origin_axis))
        # find the first split parent-child scope
        axis_maping = dict()
        for relation in self._stage.relations:
            if not isinstance(relation, tvm.schedule.Split):
                continue
            if relation.parent in self._origin_axis:
                axis_maping[relation.inner] = relation.parent
        # get the order of outer and intrin scopes
        leaf_ivars = list(self._stage.leaf_iter_vars)
        outer_ivars = list()
        inner_ivars = list(self._origin_axis)
        for scope in leaf_ivars:
            if scope in inner_ivars:
                continue
            parent = axis_maping.get(scope)
            if parent is None:  # not a intrin sope
                outer_ivars.append(scope)
            else:  # scope is a intrin scope
                if parent not in inner_ivars:
                    raise RuntimeError("parent scope shound be in inner_ivars")
                offset = inner_ivars.index(parent)
                inner_ivars[offset] = scope
        order_keeped_axis = outer_ivars + inner_ivars
        self._stage.reorder(*order_keeped_axis)
        self._scope_intrinsic = inner_ivars[0]
        return order_keeped_axis[-nlast::]

    def bind_core(self, scope_list, core_num_list):
        """
        bind core : use the chip better
        finally fuse all the outter axises

        Parameters
        ----------
        scope_list : the list that axises to bind
        core_num_list : the list that core use

        Returns
        -------
        axis_to_bind : the axis that bind

        """
        if not isinstance(scope_list, (list, tuple)):
            scope_list = [scope_list]
        if not isinstance(core_num_list, (list, tuple)):
            core_num_list = [core_num_list]
        if not scope_list:  # len(scope_list) == 0
            raise RuntimeError("at least one axis is to bind")
        if len(scope_list) != len(core_num_list):
            raise RuntimeError(
                "len of scope_list and core_num_list should be same")

        if not self._check_active_scopes(scope_list):
            raise RuntimeError("axis should be in leaf_iter_vars")

        old_leaf_ivars = list(self._stage.leaf_iter_vars)
        axis_outers = list()
        max_index = 0
        for axis, core_num in zip(scope_list, core_num_list):
            axo, axi = self.split(axis, nparts=core_num)
            index = old_leaf_ivars.index(axis)
            old_leaf_ivars[index] = axi
            max_index = max(max_index, index)
            axis_outers.append(axo)

        reorder_list = axis_outers + old_leaf_ivars[0:max_index + 1:]
        self._stage.reorder(*reorder_list)

        block = tvm.thread_axis("blockIdx.x")
        if len(axis_outers) > 1:
            axis_to_bind = self._stage.fuse(*axis_outers)
        else:  # len(axis_outers) is 1
            axis_to_bind = axis_outers[0]
        self._stage.bind(axis_to_bind, block)
        return axis_to_bind

    def get_superkernel_axis_pragma(self):
        """
        get the axis that superkernel used to pragma
        """
        leaf_ivars = self._stage.leaf_iter_vars
        return leaf_ivars[1]

    def get_relate_scope(self, scope_key, scope_end=None):
        """
        get the axises whose name contain scope_key
        """
        scope_list = list()
        for scope in self._stage.leaf_iter_vars:
            if (scope_end is not None) and (scope == scope_end):
                break
            if scope.var.name.find('{}{}'\
                .format(scope_key.var.name, '.')) == 0:
                scope_list.append(scope)
        return scope_list

    def emit_insn(self, scope, value, attrs=None):
        """
        stage's original storage_align function
        difference is the default axis to emit insn
        """
        if self._scope_intrinsic is None:
            self._scope_intrinsic = scope
        self._stage.emit_insn(scope, value, attrs)

    def set_last_attached(self, scope):
        """
        set stage's _last_attached
        """
        self._last_attached = scope

    def _check_active_scopes(self, ax_list):
        """
        check if axis in stage's leaf_ivars
        """
        leaf_ivars = list(self._stage.leaf_iter_vars)
        for axis in ax_list:
            if axis not in leaf_ivars:
                return False
        return True


class ScheduleAgent:
    """docstring for ScheduleAgent"""

    def __init__(self, sch):
        self._sch = sch
        self._attach_map = AttachMap()
        # key=op,active op othen than origin_ops
        self._scope_managers = dict()

    def __getitem__(self, tensor):
        """
        get scope manager of input tensor

        Parameters
        ----------
        tensor : Tensor

        Returns
        -------
        scope_manager

        """
        if isinstance(tensor, tvm.tensor.Tensor):
            key = tensor.op
        else:
            key = tensor
        if self._scope_managers.get(key) is None:
            self._scope_managers[key] = ScopeManager(self._sch[key])
        return self._scope_managers[key]

    def same_attach(self, tensor_a, tensor_b):
        """
        attached tensor_a  at the scope that the scope tensor_b attached

        Parameters
        ----------
        tensor_a : Tensor
        tensor_b : Tenosr

        Returns
        -------

        """

        sch = self._sch
        return self._attach_map.record_same_attach(sch[tensor_a],
                                                   sch[tensor_b])

    def attach_at(self, tensor, parent, affine_shape):
        """
        attach tensor to parent according to the affine_shape

        Parameters
        ----------
        tensor : Tensor
        parent : Tenosr
        affine_shape: shape of tensor affine to parent

        Returns
        -------
        the scope that tensor follow with
        """
        # attach_map = self._attach_map
        scopes = self[parent]
        ax_list, unit = scopes.get_active_scope_and_unit()
        if len(affine_shape) != len(ax_list):
            raise RuntimeError("len(affine_shape) should be equal to "
                               "len(shape)+len(reduce_axis) of {} "\
                               .format(parent))
        factor_list = list(
            ceil_div(i, j) if i is not None else None
            for i, j in zip(affine_shape, unit)
        )
        axis_outer = list()
        axis_intrinsic = list()
        axis_ori_unrelate = list()

        def start_attach(factor_list, ax_list):
            origin_axis = scopes.origin_axis
            for factor, axis in zip(factor_list, ax_list):
                if factor is not None and (factor > 1 or axis in origin_axis):
                    axo, axi = scopes.split(axis, factor=factor)
                    self._attach_map.update_scope(axis, axi)
                    axis_outer.append(axo)
                    axis_intrinsic.append(axi)
                elif axis in origin_axis:
                    axis_ori_unrelate.append(axis)
                else:
                    axis_outer.append(axis)
        start_attach(factor_list, ax_list)
        scope_attach = None
        if axis_intrinsic:  # len(axis_intrinsic) > 0:
            reorder_list = axis_outer + axis_ori_unrelate + axis_intrinsic
            self._sch[parent].reorder(*reorder_list)
            scope_attach = axis_intrinsic[0]
            self._attach_map.follow_with(self._sch[tensor],
                                         self._sch[parent],
                                         scope_attach)
            self[parent].set_last_attached(scope_attach)
        elif axis_outer:  # len(axis_outer) > 0:
            scope_attach = self[parent].last_attached
            if scope_attach is not None:  # no else
                self._attach_map.record_attach(self._sch[tensor],
                                               scope_attach)
        else:
            pass
        return scope_attach

    def root_stage_at(self, parent, scope):
        """
        parent: parent stage
        scope: scope
        """
        stage_array = self._sch.stages
        parent_stage = self._sch[parent.op]
        for stage in stage_array:
            if stage == parent_stage:
                continue
            if stage.attach_type == 1:
                stage.compute_at(parent_stage, scope)

    def apply(self):
        '''
        apply the attach path
        '''

        attach_map = self._attach_map
        parent_stages = list(set(attach_map.parent_stages.values()))
        remain_scopes = set(attach_map.attached_path.keys())
        for parent in parent_stages:
            scope_intrinsic = self[parent.origin_op].scope_intrinsic
            if scope_intrinsic is None:
                continue
            leaf_ivars = list(parent.leaf_iter_vars)
            index = leaf_ivars.index(scope_intrinsic)
            un_attachable_scopes = leaf_ivars[index + 1:]
            for scope in list(remain_scopes):
                if scope in un_attachable_scopes:
                    attach_map.update_scope(scope, scope_intrinsic)
                    remain_scopes.remove(scope)
        self._attach_map.apply()

    def pattern_abc(self,
                    status,
                    tensor_a,
                    tensor_b,
                    affine_shape_to_b,
                    tensor_c,
                    affine_shape_to_c):
        """
        attach tensor_a to tensor_b
        or
        attach tensor_a to tensor_c
        according to the status
        """
        if status is None:
            return None
        if isinstance(status, (list, tuple)):
            return None

        attach = None
        if status == Compare.EQUAL:
            attach = self.same_attach(tensor_a, tensor_b)
        elif status == Compare.LESS_EQ:
            attach = self.attach_at(tensor_a, tensor_b, affine_shape_to_b)
        elif status == Compare.GREATE_EQ:
            attach = self.attach_at(tensor_a, tensor_c, affine_shape_to_c)
        else:
            raise RuntimeError("tiling shape of {} shouldn't be both less "
                               "and greater than {}"\
                               .format(tensor_a, tensor_b))
        return attach
