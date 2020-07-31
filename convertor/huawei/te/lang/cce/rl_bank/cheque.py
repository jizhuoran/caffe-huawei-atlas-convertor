#!/usr/bin/env python
# -*- coding:utf-8 -*-
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

generate the cheque from best result python file by rl search
"""
import os
import time
import pickle
import te
from te import tvm
from te.platform import cce_emitinsn_params
from te.lang.cce.rl_bank.bank_cfg import INTRIN_MAP
from te.lang.cce.rl_bank.bank_cfg import SCOPE_DICT
from te.lang.cce.rl_bank.bank_cfg import PRIMITIVE_DICT
from te.lang.cce.rl_bank.bank_cfg import ScheduleTarget
from te.lang.cce.rl_bank.bank_cfg import Axis


def get_stage_by_name(stage_name, sch_targets):
    '''
    get_stage_by_name
    :param stage_name:
    :param sch_targets:
    :return:
    '''
    stage_name = stage_name.strip()
    for stage_idx, sch_target in enumerate(sch_targets):
        if sch_target.name == stage_name:
            return stage_idx, sch_target.obj.op.output(0)
    raise RuntimeError("no stage named by ", stage_name)


def get_axis_by_name(axis_name, axes):
    '''
    get_stage_by_name
    :param stage_name:
    :param sch_targets:
    :return:
    '''
    axis_name = axis_name.strip()
    for axis_idx, axis in enumerate(axes):
        if (isinstance(axis.name, str) and axis_name == axis.name) or (isinstance(axis.name, list)
                                                                       and axis_name in axis.name):
            return axis_idx, axis.obj
    raise RuntimeError("no axis named by ", axis_name)


def get_primitive_id(primitive_name):
    '''
    get_stage_by_name
    :param stage_name:
    :param sch_targets:
    :return:
    '''
    primitive_name = primitive_name.strip()
    for primitive_id in PRIMITIVE_DICT:
        if PRIMITIVE_DICT[primitive_id] == primitive_name:
            return primitive_id
    raise RuntimeError("no primitive_name named by ", primitive_name)


def get_scope_id(scope):
    '''
    get_stage_by_name
    :param stage_name:
    :param sch_targets:
    :return:
    '''
    scope = scope.strip()
    for scope_id in SCOPE_DICT:
        if SCOPE_DICT[scope_id] == scope:
            return scope_id
    raise RuntimeError("no scope named by ", scope)


def get_insn_id(insn_name):
    '''
    get_stage_by_name
    :param stage_name:
    :param sch_targets:
    :return:
    '''
    insn_name = insn_name.strip()
    for insn_id in INTRIN_MAP:
        if INTRIN_MAP[insn_id] == insn_name:
            return insn_id
    raise RuntimeError("no insn named by ", insn_name)


def get_axis(sch, sch_targets, cheque_list, specify_axis_name_dict, split_stage_name):
    '''
    get_axis
    :param sch:
    :param sch_targets:
    :param cheque_list:
    :param specify_axis_name_dict:
    :param split_stage_name:
    :param split_stage_idx:
    :param split_stage_obj:
    :return:
    '''
    split_stage_idx, split_stage_obj = get_stage_by_name(split_stage_name, sch_targets)
    # get axis when have not got
    axis_num = len(split_stage_obj.op.axis)
    for i in range(axis_num):
        axis_name = '%s_axis_%d' % (split_stage_name, i)
        axis_obj = sch[split_stage_obj].op.axis[i]
        curr_axis = Axis(axis_name, axis_obj)
        if axis_name in specify_axis_name_dict:
            curr_axis.update_name(specify_axis_name_dict[axis_name])
        sch_targets[split_stage_idx].axes.append(curr_axis)

    cheque = [split_stage_idx, get_primitive_id("get_axis"), axis_num]
    cheque_list.append(cheque)

    # get reduce axis
    reduce_axis_num = len(sch[split_stage_obj].op.reduce_axis)
    if reduce_axis_num:
        for i in range(reduce_axis_num):
            axis_name = '%s_reduce_axis_%d' % (split_stage_name, i)
            axis_obj = sch[split_stage_obj].op.reduce_axis[i]
            curr_axis = Axis(axis_name, axis_obj)
            if axis_name in specify_axis_name_dict:
                curr_axis.update_name(specify_axis_name_dict[axis_name])
            sch_targets[split_stage_idx].axes.append(curr_axis)
        cheque = [split_stage_idx, get_primitive_id("get_reduce_axis"), reduce_axis_num]
        cheque_list.append(cheque)


def proc_cache_read(sch, sch_targets, cheque_list, code_line):  # pylint: disable=too-many-locals
    '''
    get_stage_by_name
    :param stage_name:
    :param sch_targets:
    :return:
    '''
    if "cache_read" in code_line:
        out_stage_name = code_line.split("=")[0].strip()
        read_stage_name = code_line.split("cache_read(")[1].split(",")[0]
        scope = code_line.split("cache_read(")[1].split(",")[1].replace("'", "").strip()
        read_stage_idx, read_stage_obj = get_stage_by_name(read_stage_name, sch_targets)
        consumer_list = code_line.split("[")[1].split("]")[0].split(",")
        consumer_stage_idx_list = []
        consumer_stage_obj_list = []
        for consumer in consumer_list:
            consumer = consumer.strip()
            consumer_stage_idx, consumer_stage_obj = get_stage_by_name(consumer, sch_targets)
            consumer_stage_idx_list.append(consumer_stage_idx)
            consumer_stage_obj_list.append(consumer_stage_obj)

        readed_tensor = sch.cache_read(read_stage_obj, scope, consumer_stage_obj_list)
        # Tensor shoule insert after orignal tensor  when cache_read
        sch_targets.insert(read_stage_idx + 1, ScheduleTarget(out_stage_name, readed_tensor, []))
        cheque = [
            read_stage_idx,
            get_primitive_id("cache_read"),
            get_scope_id(scope), consumer_stage_idx_list
        ]
        cheque_list.append(cheque)


def proc_cache_write(sch, sch_targets, cheque_list, code_line):  # pylint: disable=too-many-locals
    '''
    get_stage_by_name
    :param stage_name:
    :param sch_targets:
    :return:
    '''
    if "cache_write" in code_line:
        written_name = code_line.split("=")[0].strip()

        if ',' not in written_name:
            # one tensor do cache_write
            write_stage_name, scope = code_line.split("cache_write(")[1].replace(")", "").split(",")
            scope = scope.replace("'", "").strip()
            write_stage_idx, write_stage_obj = get_stage_by_name(write_stage_name, sch_targets)
            written_tensor = sch.cache_write(write_stage_obj, scope)
            # Tensor shoule insert before orignal tensor  when cache_write
            sch_targets.insert(write_stage_idx, ScheduleTarget(written_name, written_tensor, []))

            cheque = [write_stage_idx, get_primitive_id("cache_write"), get_scope_id(scope)]
            cheque_list.append(cheque)
        else:
            # more than one tensor do cache_write
            write_tensor_names, scope = code_line.split("cache_write(")[1].replace(")", "")\
                .split("],")
            write_tensor_names = write_tensor_names.lstrip('[').rstrip(']').split(', ')
            write_stage_name = write_tensor_names[0].split('_v')[0].strip()
            scope = scope.replace("'", "").strip()
            write_stage_idx, write_stage_obj = get_stage_by_name(
                write_stage_name, sch_targets)
            write_tensor_objs = []
            for idx in range(len(write_tensor_names)):
                write_tensor_objs.append(write_stage_obj.op.output(idx))
            written_tensors = sch.cache_write(write_tensor_objs, scope)

            written_name = '%s_l' % write_stage_name
            sch_targets.insert(write_stage_idx,
                               ScheduleTarget(written_name, written_tensors[0], []))

            cheque = [[write_stage_idx, len(write_tensor_names)],
                      get_primitive_id("cache_write"),
                      get_scope_id(scope)]
            cheque_list.append(cheque)


def proc_double_buffer(sch, sch_targets, cheque_list, code_line):  # pylint: disable=too-many-locals
    '''
    get_stage_by_name
    :param stage_name:
    :param sch_targets:
    :return:
    '''
    if "double_buffer" in code_line:
        stage_name = code_line.split("[")[1].split("]")[0]
        stage_idx, stage_obj = get_stage_by_name(stage_name, sch_targets)
        sch[stage_obj].double_buffer()

        cheque = [stage_idx, get_primitive_id("double_buffer")]
        cheque_list.append(cheque)


def proc_compute_inline(sch, sch_targets, cheque_list, code_line):  # pylint: disable=too-many-locals
    '''
    get_stage_by_name
    :param stage_name:
    :param sch_targets:
    :return:
    '''
    if "compute_inline" in code_line:
        stage_name = code_line.split("[")[1].split("]")[0]
        stage_idx, stage_obj = get_stage_by_name(stage_name, sch_targets)
        sch[stage_obj].compute_inline()

        cheque = [stage_idx, get_primitive_id("compute_inline")]
        cheque_list.append(cheque)


def proc_reduce_split(sch, sch_targets, cheque_list, stage_get_axis_dict, specify_axis_name_dict,
                      code_line):  # pylint: disable=too-many-locals
    '''
    get_stage_by_name
    :param stage_name:
    :param sch_targets:
    :return:
    '''
    if "split" in code_line and "factor" in code_line and "reduce_axis" in code_line:
        split_stage_name = code_line.split("].split")[0].split("[")[1]
        split_stage_idx, split_stage_obj = get_stage_by_name(split_stage_name, sch_targets)
        factor = int(code_line.split("factor=")[1].split(")")[0])

        split_axis = code_line.split(".split(")[1].split(",")[0]
        if "[" in split_axis and "]" in split_axis:
            split_axis = int(split_axis.split("axis[")[1].split("]")[0])

        # might be int or specify_axis_name: ub_split_reduce_axis
        if isinstance(split_axis, int):
            split_axis_idx = split_axis
            # get axis
            if not stage_get_axis_dict.get(split_stage_name, False):
                get_axis(sch, sch_targets, cheque_list, specify_axis_name_dict, split_stage_name)

                stage_get_axis_dict[split_stage_name] = True

            split_outer, split_inner = sch[split_stage_obj].split(
                sch[split_stage_obj].op.reduce_axis[split_axis_idx], factor=factor)

            # delete axis when axis is splited
            split_reduce_axis_name = '%s_reduce_axis_%d' % (split_stage_name, split_axis_idx)
            split_reduce_axis_idx, _ = get_axis_by_name(split_reduce_axis_name,
                                                        sch_targets[split_stage_idx].axes)
            sch_targets[split_stage_idx].axes.pop(split_reduce_axis_idx)

            # inset inner then insert outer
            split_outer_name, split_inner_name = code_line.split("=")[0].strip().split(",")
            sch_targets[split_stage_idx].axes.insert(split_reduce_axis_idx,
                                                     Axis(split_inner_name, split_inner))
            sch_targets[split_stage_idx].axes.insert(split_reduce_axis_idx,
                                                     Axis(split_outer_name, split_outer))
            # add cheque
            cheque = [split_stage_idx, get_primitive_id("split"), split_reduce_axis_idx, factor]
            cheque_list.append(cheque)
        elif split_stage_name.endswith('_rfactor') and \
            split_axis in specify_axis_name_dict.values():
            # split_axis is specify_axis_name: ub_split_reduce_axis
            if not stage_get_axis_dict.get(split_stage_name, False):
                # get comm axis and get reduce axis
                get_axis(sch, sch_targets, cheque_list, specify_axis_name_dict, split_stage_name)

                stage_get_axis_dict[split_stage_name] = True

            # get real split axis obj
            ori_stage_name = split_stage_name.rstrip('_rfactor')
            ori_stage_idx, _ = get_stage_by_name(
                ori_stage_name, sch_targets)
            _, ori_axis_obj = get_axis_by_name(split_axis,
                                               sch_targets[
                                                   ori_stage_idx].axes)
            for axis in sch_targets[split_stage_idx].axes:
                if axis.obj == ori_axis_obj:
                    split_axis = axis.name
                    break

            split_axis_idx, axis_obj = get_axis_by_name(split_axis,
                                                        sch_targets[split_stage_idx].axes)
            split_outer, split_inner = sch[split_stage_obj].split(axis_obj, factor=factor)

            # delete axis when axis is splited
            sch_targets[split_stage_idx].axes.pop(split_axis_idx)

            # inset inner then insert outer
            split_outer_name, split_inner_name = code_line.split("=")[
                0].strip().split(",")
            sch_targets[split_stage_idx].axes.insert(split_axis_idx,
                                                     Axis(split_inner_name,
                                                          split_inner))
            sch_targets[split_stage_idx].axes.insert(split_axis_idx,
                                                     Axis(split_outer_name,
                                                          split_outer))
            # add cheque
            cheque = [split_stage_idx, get_primitive_id("split"),
                      split_axis_idx, factor]
            cheque_list.append(cheque)


def proc_split(sch, sch_targets, cheque_list, stage_get_axis_dict, specify_axis_name_dict,
               code_line):  # pylint: disable=too-many-locals
    '''
    get_stage_by_name
    :param stage_name:
    :param sch_targets:
    :return:
    '''
    if "split" in code_line and "factor" in code_line and "reduce_axis" not in code_line:
        split_stage_name = code_line.split("].split")[0].split("[")[1]
        split_stage_idx, split_stage_obj = get_stage_by_name(split_stage_name, sch_targets)
        factor = int(code_line.split("factor=")[1].split(")")[0])

        split_axis = code_line.split(".split(")[1].split(",")[0]
        if "[" in split_axis and "]" in split_axis:
            split_axis = int(split_axis.split("axis[")[1].split("]")[0])

        if isinstance(split_axis, int):
            # get axis
            if not stage_get_axis_dict.get(split_stage_name, False):
                get_axis(sch, sch_targets, cheque_list, specify_axis_name_dict, split_stage_name)
                stage_get_axis_dict[split_stage_name] = True

            split_outer, split_inner = sch[split_stage_obj].split(
                sch[split_stage_obj].op.axis[split_axis], factor=factor)

            # get axis index in current axes by axis_name
            split_axis_name = '%s_axis_%d' % (split_stage_name, split_axis)
            split_axis_idx, _ = get_axis_by_name(split_axis_name, sch_targets[split_stage_idx].axes)
            # delete split axis
            sch_targets[split_stage_idx].axes.pop(split_axis_idx)
            # inset inner then outer
            split_outer_name, split_inner_name = code_line.split("=")[0].strip().split(",")
            sch_targets[split_stage_idx].axes.insert(split_axis_idx,
                                                     Axis(split_inner_name, split_inner))
            sch_targets[split_stage_idx].axes.insert(split_axis_idx,
                                                     Axis(split_outer_name, split_outer))
            cheque = [split_stage_idx, get_primitive_id("split"), split_axis_idx, factor]
            cheque_list.append(cheque)

        # if string mean axis has been split, get axis obj by axis name
        else:
            split_axis_name = split_axis

            # split_axis is specify_axis_name: ub_split_axis
            # eg: sch[reduce_39_rfactor].split(ub_split_axis, factor)
            if split_stage_name.endswith('_rfactor') and \
                    split_axis_name in specify_axis_name_dict.values():
                # get split axis name of curr stage
                if not stage_get_axis_dict.get(split_stage_name, False):
                    # get comm axis and get reduce axis
                    get_axis(sch, sch_targets, cheque_list, specify_axis_name_dict,
                             split_stage_name)
                    stage_get_axis_dict[split_stage_name] = True

                # get real split axis obj
                ori_stage_name = split_stage_name.rstrip('_rfactor')
                ori_stage_idx, _ = get_stage_by_name(
                    ori_stage_name, sch_targets)
                _, ori_axis_obj = get_axis_by_name(split_axis_name,
                                                   sch_targets[ori_stage_idx].axes)
                for axis in sch_targets[split_stage_idx].axes:
                    if axis.obj == ori_axis_obj:
                        split_axis_name = axis.name
                        break

            axis_idx, axis_obj = get_axis_by_name(split_axis_name,
                                                  sch_targets[split_stage_idx].axes)

            split_outer, split_inner = sch[split_stage_obj].split(axis_obj, factor=factor)
            # delete split axis
            sch_targets[split_stage_idx].axes.pop(axis_idx)
            # insert inner then outer
            split_outer_name, split_inner_name = code_line.split("=")[0].strip().split(",")
            sch_targets[split_stage_idx].axes.insert(axis_idx, Axis(split_inner_name, split_inner))
            sch_targets[split_stage_idx].axes.insert(axis_idx, Axis(split_outer_name, split_outer))
            cheque = [split_stage_idx, get_primitive_id("split"), axis_idx, factor]
            cheque_list.append(cheque)


def proc_reorder(sch, sch_targets, cheque_list, code_line):  # pylint: disable=too-many-locals
    '''
    get_stage_by_name
    :param stage_name:
    :param sch_targets:
    :return:
    '''
    if "reorder" in code_line:
        stage_name = code_line.split("[")[1].split("]")[0]
        stage_idx, stage_obj = get_stage_by_name(stage_name, sch_targets)
        reorder_axis_name_list = code_line.split("reorder(")[1].split(")")[0].split(",")
        reorder_axis_name_list = [x.strip() for x in reorder_axis_name_list if x.strip()]

        # axes is empty if have not get axis
        if not sch_targets[stage_idx].axes:
            # comm axis
            axis_num = len(stage_obj.op.axis)
            cheque = [stage_idx, get_primitive_id("get_axis"), axis_num]
            cheque_list.append(cheque)

            for i in range(axis_num):
                axis_name = '%s_axis_%d' % (stage_name, i)
                axis_obj = sch[stage_obj].op.axis[i]
                sch_targets[stage_idx].axes.append(Axis(axis_name, axis_obj))
            # reduce axis
            reduce_axis_num = len(stage_obj.op.reduce_axis)
            cheque = [stage_idx, get_primitive_id("get_reduce_axis"), reduce_axis_num]
            cheque_list.append(cheque)

            for i in range(reduce_axis_num):
                axis_name = '%s_reduce_axis_%d' % (stage_name, i)
                axis_obj = sch[stage_obj].op.reduce_axis[i]
                sch_targets[stage_idx].axes.append(Axis(axis_name, axis_obj))

        reorder_axis_idx_list = []
        reorder_axis_obj_list = []
        for axis_name in reorder_axis_name_list:
            axis_idx, axis_obj = get_axis_by_name(axis_name, sch_targets[stage_idx].axes)
            reorder_axis_idx_list.append(axis_idx)
            reorder_axis_obj_list.append(axis_obj)

        cheque = [stage_idx, get_primitive_id("reorder"), reorder_axis_idx_list]
        cheque_list.append(cheque)

        sch[stage_obj].reorder(*(reorder_axis_obj_list))
        # sort axis by reorder
        sch_targets[stage_idx].axes = [
            sch_targets[stage_idx].axes[i] for i in reorder_axis_idx_list
        ]


def proc_nparts(sch, sch_targets, cheque_list, code_line):  # pylint: disable=too-many-locals
    '''
    get_stage_by_name
    :param stage_name:
    :param sch_targets:
    :return:
    '''
    if "split" in code_line and "nparts" in code_line:
        split_stage_name = code_line.split("].split")[0].split("[")[1]
        split_stage_idx, split_stage_obj = get_stage_by_name(split_stage_name, sch_targets)
        nparts = int(code_line.split("nparts=")[1].split(")")[0])
        split_axis_name = code_line.split(".split(")[1].split(",")[0]

        axis_idx, axis_obj = get_axis_by_name(split_axis_name, sch_targets[split_stage_idx].axes)
        cheque = [split_stage_idx, get_primitive_id("split_nparts"), axis_idx, nparts]
        cheque_list.append(cheque)

        split_outer, split_inner = sch[split_stage_obj].split(axis_obj, nparts=nparts)
        # delete split axis
        sch_targets[split_stage_idx].axes.pop(axis_idx)
        split_outer_name, split_inner_name = code_line.split("=")[0].strip().split(",")
        # insert inner then outer
        sch_targets[split_stage_idx].axes.insert(axis_idx, Axis(split_inner_name, split_inner))
        sch_targets[split_stage_idx].axes.insert(axis_idx, Axis(split_outer_name, split_outer))


def proc_compute_at(sch, sch_targets, cheque_list, code_line):  # pylint: disable=too-many-locals
    '''
    get_stage_by_name
    :param stage_name:
    :param sch_targets:
    :return:
    '''
    if "compute_at" in code_line:
        stage_name = code_line.split("compute_at")[0].split("[")[1].split("]")[0]
        at_stage_name = code_line.split("compute_at")[1].split("[")[1].split("]")[0]
        at_axis_name = code_line.split(",")[1].split(")")[0]

        stage_idx, stage_obj = get_stage_by_name(stage_name, sch_targets)
        at_stage_idx, at_stage_obj = get_stage_by_name(at_stage_name, sch_targets)
        at_axis_idx, at_axis_obj = get_axis_by_name(at_axis_name, sch_targets[at_stage_idx].axes)

        cheque = [stage_idx, get_primitive_id("compute_at"), at_stage_idx, at_axis_idx]
        cheque_list.append(cheque)

        sch[stage_obj].compute_at(sch[at_stage_obj], at_axis_obj)


def proc_fuse(sch, sch_targets, cheque_list, specify_axis_name_dict, code_line):  # pylint: disable=too-many-locals
    '''
    get_stage_by_name
    :param stage_name:
    :param sch_targets:
    :return:
    '''
    if "fuse(" in code_line:
        fused_axis_name = code_line.split("=")[0].strip()
        stage_name = code_line.split("[")[1].split("]")[0]
        stgae_idx, stage_obj = get_stage_by_name(stage_name, sch_targets)

        fuse_axis_name_list = [
            x.strip() for x in code_line.split("fuse(")[1].split(")")[0].split(",") if x.strip()
        ]
        fuse_axis_idx_list = []
        fuse_axis_obj_list = []
        for fuse_axis_name in fuse_axis_name_list:
            if fuse_axis_name in specify_axis_name_dict.values():
                for local_axis_name, special_axis_name in specify_axis_name_dict.items():
                    if fuse_axis_name == special_axis_name:
                        fuse_axis_name = local_axis_name
                        break
            axis_idx, axis_obj = get_axis_by_name(fuse_axis_name, sch_targets[stgae_idx].axes)
            fuse_axis_idx_list.append(axis_idx)
            fuse_axis_obj_list.append(axis_obj)
        fused_axis_obj = sch[stage_obj].fuse(*(fuse_axis_obj_list))
        fuse_axis_start = min(fuse_axis_idx_list)
        # delete fuse axis
        for _ in fuse_axis_idx_list:
            sch_targets[stgae_idx].axes.pop(fuse_axis_start)
        # insert outer
        sch_targets[stgae_idx].axes.insert(fuse_axis_start, Axis(fused_axis_name, fused_axis_obj))

        cheque = [stgae_idx, get_primitive_id("fuse"), fuse_axis_idx_list]
        cheque_list.append(cheque)


def proc_rfactor(sch, sch_targets, cheque_list, code_line):  # pylint: disable=too-many-locals
    '''
    get_stage_by_name
    :param stage_name:
    :param sch_targets:
    :return:
    '''
    if "rfactor(" in code_line:
        out_tensor_name = code_line.split("=")[0].split(',')[0].strip()
        stage_name = code_line.split("rfactor(")[1].split(",")[0]
        rfactor_axis_name = code_line.split("rfactor(")[1].split(",")[1]
        factor_axis = int(code_line.split("factor_axis=")[1].replace(")", "").strip())
        stage_idx, stage_obj = get_stage_by_name(stage_name, sch_targets)
        axis_idx, axis_obj = get_axis_by_name(rfactor_axis_name, sch_targets[stage_idx].axes)
        out_tensor_obj = sch.rfactor(stage_obj, axis_obj, factor_axis=factor_axis)

        cheque = [stage_idx, get_primitive_id("rfactor"), axis_idx, factor_axis]
        cheque_list.append(cheque)

        if not isinstance(out_tensor_obj, tvm.tensor.Tensor):
            out_tensor_obj = out_tensor_obj[0]
        sch_targets.insert(stage_idx,
                           ScheduleTarget(out_tensor_name, out_tensor_obj.op.output(0), []))


def proc_set_scope(sch, sch_targets, cheque_list, code_line):  # pylint: disable=too-many-locals
    '''
    get_stage_by_name
    :param stage_name:
    :param sch_targets:
    :return:
    '''
    if "set_scope" in code_line:
        stage_name = code_line.split(".set_scope")[0].split("[")[1].split("]")[0]
        stage_idx, stage_obj = get_stage_by_name(stage_name, sch_targets)
        scope = code_line.split("'")[1]

        cheque = [stage_idx, get_primitive_id("set_scope"), get_scope_id(scope)]
        cheque_list.append(cheque)

        sch[stage_obj].set_scope(scope)


def proc_bind(sch, sch_targets, cheque_list, code_line):  # pylint: disable=too-many-locals
    '''
    get_stage_by_name
    :param stage_name:
    :param sch_targets:
    :return:
    '''
    if 'bind' in code_line:
        stage_name = code_line.split(".bind(")[0].split("[")[1].split("]")[0]
        stage_idx, stage_obj = get_stage_by_name(stage_name, sch_targets)

        cheque = [stage_idx, get_primitive_id("bind")]
        cheque_list.append(cheque)

        block = tvm.thread_axis('blockIdx.x')
        sch[stage_obj].bind(sch_targets[stage_idx].axes[0].obj, block)


def proc_pragma(sch, sch_targets, cheque_list, code_line):  # pylint: disable=too-many-locals
    '''
    get_stage_by_name
    :param stage_name:
    :param sch_targets:
    :return:
    '''
    if 'pragma' in code_line:
        stage_name = code_line.split(".pragma(")[0].split("[")[1].split("]")[0]
        stage_idx, stage_obj = get_stage_by_name(stage_name, sch_targets)
        pragma_axis = code_line.split(".pragma(")[1].split(",")[0]
        if "[" in pragma_axis and "]" in pragma_axis:
            pragma_axis = int(pragma_axis.split("axis[")[1].split("]")[0])
        pragma_name = code_line.split("'")[1]
        pragma_offset = int(code_line.split(",")[2].split(")")[0].strip())

        # int，get axis by index
        if isinstance(pragma_axis, int):
            cheque = [
                stage_idx,
                get_primitive_id("pragma"), [pragma_axis, -1],
                get_insn_id(pragma_name), pragma_offset
            ]
            sch[stage_obj].pragma(sch[stage_obj].op.axis[pragma_axis], pragma_name, pragma_offset)
        # str，axis name，get axis obj by name
        else:
            insn_axis_idx, insn_axis_obj = get_axis_by_name(pragma_axis,
                                                            sch_targets[stage_idx].axes)
            cheque = [
                stage_idx,
                get_primitive_id("pragma"), [-1, insn_axis_idx],
                get_insn_id(pragma_name), pragma_offset
            ]
            sch[stage_obj].pragma(insn_axis_obj, pragma_name, pragma_offset)

        cheque_list.append(cheque)


def proc_emit_insn(sch, sch_targets, cheque_list, code_line):  # pylint: disable=too-many-locals
    '''
    get_stage_by_name
    :param stage_name:
    :param sch_targets:
    :return:
    '''
    if 'emit_insn' in code_line:
        stage_name = code_line.split(".emit_insn(")[0].split("[")[1].split("]")[0]
        stage_idx, stage_obj = get_stage_by_name(stage_name, sch_targets)
        insn_axis = code_line.split(".emit_insn(")[1].split(",")[0]
        if "[" in insn_axis and "]" in insn_axis:
            insn_axis = int(insn_axis.split("axis[")[1].split("]")[0])
        insn_name = code_line.split("'")[1]

        # int，get axis by index
        if isinstance(insn_axis, int):
            cheque = [
                stage_idx,
                get_primitive_id("emit_insn"), [insn_axis, -1],
                get_insn_id(insn_name)
            ]
            sch[stage_obj].emit_insn(sch[stage_obj].op.axis[insn_axis], insn_name)
        # str，axis name，get axis obj by name
        else:
            insn_axis_idx, insn_axis_obj = get_axis_by_name(insn_axis, sch_targets[stage_idx].axes)
            cheque = [
                stage_idx,
                get_primitive_id("emit_insn"), [-1, insn_axis_idx],
                get_insn_id(insn_name)
            ]
            sch[stage_obj].emit_insn(insn_axis_obj, insn_name)

        cheque_list.append(cheque)


def proc_insert_param(cheque_list, code_line):  # pylint: disable=too-many-locals
    '''
    get_stage_by_name
    :param stage_name:
    :param sch_targets:
    :return:
    '''
    if "insert_param('broadcast_axis_offset'" in code_line:
        broadcast_axis_offset = int(code_line.split(",")[1].split(")")[0].strip())

        cce_emitinsn_params.cceEmitParamsIns.del_param('broadcast_axis_offset')
        cce_emitinsn_params.cceEmitParamsIns.insert_param('broadcast_axis_offset',
                                                          broadcast_axis_offset)

        cheque = [-1, get_primitive_id("broadcast_axis_offset"), broadcast_axis_offset]
        cheque_list.append(cheque)


def proc_storage_align(sch, sch_targets, cheque_list, code_line):  # pylint: disable=too-many-locals
    '''
    get_stage_by_name
    :param stage_name:
    :param sch_targets:
    :return:
    '''
    if 'storage_align' in code_line:
        stage_name = code_line.split(".storage_align(")[0].split("[")[1].split("]")[0]
        stage_idx, stage_obj = get_stage_by_name(stage_name, sch_targets)
        axis_idx = int(
            code_line.split(".storage_align(")[1].split(",")[0].split("axis[")[1].split("]")[0])
        block_num = int(code_line.split(",")[1].strip())

        cheque = [stage_idx, get_primitive_id("storage_align"), axis_idx, block_num]
        cheque_list.append(cheque)

        sch[stage_obj].storage_align(sch[stage_obj].op.axis[axis_idx], block_num, 0)


def proc_cce_special(sch, sch_targets, cheque_list, code_line, cce_special_cheque):  # pylint: disable=too-many-locals
    '''
    get_stage_by_name
    :param stage_name:
    :param sch_targets:
    :return:
    '''
    if "cce_special" in code_line:
        if sch.cce_special is None:
            sch.cce_special = dict()
            cce_special_cheque.extend([-1, get_primitive_id("cce_special")])
        else:
            cce_special_key = code_line.split("\"")[1]
            tensor_name_list = code_line.split("=")[1].strip().replace("[",
                                                                       "").replace("]",
                                                                                   "").split(",")
            tensor_name_list = [x.strip() for x in tensor_name_list if x.strip()]
            tensor_obj_list = []
            tensor_idx_list = []
            # tuple_reduce generate cce_special cheque
            spec_scene = False
            for cheque in cheque_list:
                if cheque[1] == 1 and isinstance(cheque[0], list):
                    spec_scene = True
                    break
            if spec_scene:
                tensor_nums = 0
                other_tensor_list = []
                stage_name = ''
                for idx, tensor_name in enumerate(tensor_name_list):
                    if '_v%s' % idx in tensor_name:
                        tensor_nums += 1
                        stage_name = tensor_name.split('_v%s' % idx)[0] + \
                                     tensor_name.split('_v%s' % idx)[1]
                    else:
                        other_tensor_list.append(tensor_name)
                if tensor_nums > 0:
                    stage_idx, stage_obj = get_stage_by_name(stage_name, sch_targets)
                    tensor_idx_list.append([stage_idx, tensor_nums])
                    for idx in range(tensor_nums):
                        tensor_obj_list.append(stage_obj.op.output(idx))
                tensor_name_list = other_tensor_list

            for tensor_name in tensor_name_list:
                stage_idx, stage_obj = get_stage_by_name(tensor_name, sch_targets)
                tensor_obj_list.extend([stage_obj.op.output(0)])
                tensor_idx_list.append(stage_idx)
            sch.cce_special[cce_special_key] = tensor_obj_list
            cce_special_cheque.append(tensor_idx_list)

        if len(cce_special_cheque) == 5:
            cheque_list.append(cce_special_cheque)


def proc_build(sch, sch_targets, cheque_list, code_line, kernel_name=""):  # pylint: disable=too-many-locals
    '''
    proc_build
    :param stage_name:
    :param sch_targets:
    :return:
    '''
    if 'config["tensor_list"]' in code_line:
        tensor_name_list = code_line.split("=")[1].replace("[", "").replace("]", "").split(",")
        tensor_name_list = [x.strip() for x in tensor_name_list if x.strip()]

        tensor_obj_list = []
        # tuple_reduce generate special config
        spec_scene = False
        for cheque in cheque_list:
            if cheque[1] == 1 and isinstance(cheque[0], list):
                spec_scene = True
                break
        if spec_scene:
            tensor_nums = 0
            stage_name = ''
            for tensor_name in tensor_name_list:
                if '_v' not in tensor_name:
                    _, stage_obj = get_stage_by_name(tensor_name, sch_targets)
                    tensor_obj_list.append(stage_obj.op.output(0))
                elif '_v' in tensor_name and '_l' in tensor_name:
                    tensor_nums += 1
                    stage_name = tensor_name.split('_v')[0] + '_l'
                elif '_v' in tensor_name and '_l' not in tensor_name:
                    tensor_nums += 1
                    stage_name = tensor_name.split('_v')[0]
            if tensor_nums > 0:
                _, stage_obj = get_stage_by_name(stage_name, sch_targets)
                for idx in range(tensor_nums):
                    tensor_obj_list.append(stage_obj.op.output(idx))

        # try to build ensure proc right
        config = dict()
        config["print_ir"] = False
        config["need_build"] = True
        if not kernel_name:
            kernel_name = "%s_%s" % ("default",
                                     str(os.getpid()) + "_" + str(int(time.time() * 1000)))
        config["name"] = kernel_name
        config["tensor_list"] = tensor_obj_list
        config["bool_storage_as_1bit"] = False
        te.lang.cce.cce_build_code(sch, config)


def proc_axis_name(code_line, specify_axis_name_dict):
    '''
    proc_axis_name
    :param code_line:
    :param specify_axis_name_dict:
    :return:
    '''
    if "=" in code_line and ',' not in code_line and 'axis' in code_line and 'op' in code_line:
        specify_axis_name = code_line.split("=")[0].strip()
        stage_name = code_line.split(".op")[0].split("[")[1].split("]")[0].strip()
        axis_type = "axis"
        if "reduce_axis" in code_line:
            axis_type = "reduce_axis"
        axis_idx = int(code_line.split("[")[-1].split("]")[0])
        local_axis_name = '%s_%s_%d' % (stage_name, axis_type, axis_idx)
        if local_axis_name != specify_axis_name:
            specify_axis_name_dict[local_axis_name] = specify_axis_name


def get_cheque_non_cce_special(other_cheque_str_list):
    '''
    get_cheque_non_cce_special
    :param other_cheque_str_list:
    :return:
    '''
    # get cheque of schedule operations other than cce_special
    non_cce_special_cheque_list = []
    for cheque_str in other_cheque_str_list:
        single_cheque_list = []
        if '[' and ']' in cheque_str:
            # cheque: [4, 11, [0, -1], 12]
            cheque_str1 = cheque_str.split('[')[0]
            cheque_str2 = cheque_str.split('[')[1].split(']')[0]
            cheque_str3 = cheque_str.split('[')[1].split(']')[1]
            if cheque_str1:
                single_cheque_list1 = [int(x.strip()) for x in cheque_str1.split(', ') if x.strip()]
                single_cheque_list.extend(single_cheque_list1)

            single_cheque_list2 = [int(x.strip()) for x in cheque_str2.split(', ') if x.strip()]
            single_cheque_list.append(single_cheque_list2)

            if cheque_str3:
                single_cheque_list3 = [int(x.strip()) for x in cheque_str3.split(', ') if x.strip()]
                single_cheque_list.extend(single_cheque_list3)
        else:
            single_cheque_list = [int(x.strip()) for x in cheque_str.split(', ') if x.strip()]
        non_cce_special_cheque_list.append(single_cheque_list)
    return non_cce_special_cheque_list


def get_cheque_cce_special(cce_special_cheque_str_list):
    '''
    get_cheque_cce_special
    :param cce_special_cheque_str_list:
    :return:
    '''
    # get cheque of sch.cce_special
    cce_special_cheque_list = [-1, 14]
    for cce_special_cheque_str in cce_special_cheque_str_list:
        if ']' not in cce_special_cheque_str:
            cce_special_cheque_list.append(int(cce_special_cheque_str.strip()))
            continue
        cce_special_cheque_str = cce_special_cheque_str.rstrip(']')
        if not cce_special_cheque_str:
            cce_special_cheque_list.append([])
            continue
        # cheque: [-1, 14, [], [[8, 2]], [[7, 2]]]
        cce_special_cheque = []
        if '[' in cce_special_cheque_str:
            cheque = [int(x.strip()) for x in
                      cce_special_cheque_str.replace('[', '').replace(']', '').split(',')
                      if x.strip()]
            cce_special_cheque.append(cheque)
        # cheque: [-1, 14, [], [8, 2], [7, 2]]
        else:
            cce_special_cheque = [int(x.strip()) for x in
                                  cce_special_cheque_str.split(', ') if x.strip()]
        cce_special_cheque_list.append(cce_special_cheque)
    return cce_special_cheque_list


def gen_cheque_by_code(code_line_list, kernel_name):
    '''
    gen_cheque_by_code
    :param code_line_list:
    :return:
    '''
    # get cheque_list by code of python file
    cheque_list = []
    cce_special_cheque = []
    sch = []
    sch_targets = []
    stage_get_axis_dict = {}
    specify_axis_name_dict = {}
    for code_line in code_line_list:
        if "pickle.loads(" in code_line:
            tensor_pickle_byte = code_line.split("pickle.loads(b'")[-1][:-2].encode('ISO-8859-1').\
                decode('unicode-escape').encode('ISO-8859-1')
            sch = pickle.loads(tensor_pickle_byte)
            sch.cce_special = None
            for stage in sch.stages:
                sch_targets.append(ScheduleTarget(stage.op.name, stage.op.output(0), []))
            continue
        if "    #" in code_line:
            continue

        proc_cache_read(sch, sch_targets, cheque_list, code_line)
        proc_cache_write(sch, sch_targets, cheque_list, code_line)
        proc_double_buffer(sch, sch_targets, cheque_list, code_line)
        proc_compute_inline(sch, sch_targets, cheque_list, code_line)
        proc_axis_name(code_line, specify_axis_name_dict)
        proc_reduce_split(sch, sch_targets, cheque_list, stage_get_axis_dict,
                          specify_axis_name_dict, code_line)
        proc_split(sch, sch_targets, cheque_list, stage_get_axis_dict,
                   specify_axis_name_dict,
                   code_line)
        proc_reorder(sch, sch_targets, cheque_list, code_line)
        proc_nparts(sch, sch_targets, cheque_list, code_line)
        proc_compute_at(sch, sch_targets, cheque_list, code_line)
        proc_fuse(sch, sch_targets, cheque_list, specify_axis_name_dict, code_line)
        proc_rfactor(sch, sch_targets, cheque_list, code_line)
        proc_set_scope(sch, sch_targets, cheque_list, code_line)
        proc_bind(sch, sch_targets, cheque_list, code_line)
        proc_pragma(sch, sch_targets, cheque_list, code_line)
        proc_emit_insn(sch, sch_targets, cheque_list, code_line)
        proc_insert_param(cheque_list, code_line)
        proc_storage_align(sch, sch_targets, cheque_list, code_line)
        proc_cce_special(sch, sch_targets, cheque_list, code_line,
                         cce_special_cheque)
        proc_build(sch, sch_targets, code_line, kernel_name)

    return cheque_list


def gen_cheque_by_comments(code_line_list):
    '''
    gen_cheque_by_comments
    :param code_line_list:
    :return:
    '''
    # get cheque_list by comments of python file
    new_cheque_list = []
    for code_line in code_line_list:
        if "# cheque_list: " in code_line:
            cheque_list_str = code_line.split("# cheque_list: ")[1].strip()
            # get cheque of schedule operations other than cce_special
            other_cheque_str_list = cheque_list_str.split("], [-1, 14, [")[
                0].lstrip('[').split('], [')
            other_cheque_list = get_cheque_non_cce_special(other_cheque_str_list)
            new_cheque_list.extend(other_cheque_list)

            # get cheque of sch.cce_special
            cce_special_cheque_str_list = cheque_list_str.split("], [-1, 14, [")[1].split(', [')
            cce_special_cheque_list = get_cheque_cce_special(cce_special_cheque_str_list)
            new_cheque_list.append(cce_special_cheque_list)

    return new_cheque_list


def judge_equal_or_not(cheque_list, new_cheque_list):
    '''
    judge_equal_or_not
    :param cheque_list:
    :param new_cheque_list:
    :return:
    '''
    for cheque in cheque_list:
        if cheque not in new_cheque_list:
            return False
    if len(cheque_list) == len(new_cheque_list):
        return True


def gen_cheque(py_path, kernel_name=""):
    '''
    gen_cheque
    :param py_path:
    :param kernel_name:
    :return:
    '''

    with open(py_path, 'r') as file_handler:
        schedule_code_str = file_handler.read()

    code_line_list = schedule_code_str.split("\n")
    # new_cheque_list by py should equal to cheque_list by code
    cheque_list = gen_cheque_by_code(code_line_list, kernel_name)
    new_cheque_list = gen_cheque_by_comments(code_line_list)
    ret = judge_equal_or_not(cheque_list, new_cheque_list)
    if ret:
        return cheque_list
    raise RuntimeError("new_cheque_list by comments not equals to cheque_list by code,"
                       " gen_cheque fail!!!")
