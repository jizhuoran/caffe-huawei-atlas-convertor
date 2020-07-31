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

Runtime function related hooks
"""
# pylint: disable=unused-import
from __future__ import absolute_import as _abs

from te import tvm
from te.tvm import build_module
from . import ir_pass

def get_pass_list():
    """
    Returns
    -------
    pass_list: list of function
        The pass to set to build_config(add_lower_pass=tpu.get_pass_list())
    """

    # the number in pass_list such as 0,1,2,3 represents the order of the pass
    # called
    pass_list = [
        (1, tvm.ir_pass.ReuseBuf),
        (1, tvm.ir_pass.ReuseCoAxisBuffer),
        (1, tvm.ir_pass.FullySplitLoopPartition),
        (1, lambda x: tvm.ir_pass.UnrollLoop(x, 0, 8, 0, True)),
        (1, tvm.ir_pass.RemoveNoOp),
        (1, tvm.ir_pass.CanonicalSimplify),
        (1, tvm.ir_pass.FissionLoop),
        (1, tvm.ir_pass.EmitInsn),
        (1, tvm.ir_pass.SetSPROptimizer),
        (1, tvm.ir_pass.TikDoubleBufferSupport),
        (1, tvm.ir_pass.InjectPipeBuffer),
        (2, tvm.ir_pass.AutoFuseBuffer),
        (2, tvm.ir_pass.Simplify),
        (2, tvm.ir_pass.InjectSync),
        (2, tvm.ir_pass.PackIntrinArgConfig),
        (3, tvm.ir_pass.DeviceMark),
    ]
    return pass_list

def build_config_update(current_config, attr, value):
    """
    update build config
    set attr to value in build_config
    :return:
    """
    new_list = {attr: value, }
    return build_config_update_list(current_config, new_list)


def build_config_update_list(current_config, config_list):
    """
    update build config
    set attr to value in build_config
    :return:
    """
    current_config_map = {
        "auto_unroll_max_step": current_config.auto_unroll_max_step,
        "auto_unroll_max_depth": current_config.auto_unroll_max_depth,
        "auto_unroll_max_extent": current_config.auto_unroll_max_extent,
        "unroll_explicit": current_config.unroll_explicit,
        "detect_global_barrier": current_config.detect_global_barrier,
        "partition_const_loop": current_config.partition_const_loop,
        "offset_factor": current_config.offset_factor,
        "data_alignment": current_config.data_alignment,
        "restricted_func": current_config.restricted_func,
        "double_buffer_split_loop": current_config.double_buffer_split_loop,
        "predicate_realize_bound": current_config.predicate_realize_bound,
        "constant_realize_extent_in_infer_bound":
            current_config.constant_realize_extent_in_infer_bound,
        "double_buffer_non_reuse": current_config.double_buffer_non_reuse,
        "apply_tbe_pass": current_config.apply_tbe_pass,
        "dump_cce_code": current_config.dump_cce_code,
        "bool_storage_as_1bit": current_config.bool_storage_as_1bit,
        "sync_mode": current_config.sync_mode,
        "debug_message": current_config.debug_message,
        "dump_pass_ir": current_config.dump_pass_ir,
        "instrument_bound_checkers": current_config.instrument_bound_checkers,
        "disable_select_rewriting": current_config.disable_select_rewriting,
        "disable_vectorize": current_config.disable_vectorize,
        "disable_assert": current_config.disable_assert,
        "save_temp_cce_file": current_config.save_temp_cce_file,
        "dump_pass_ir_to_file": current_config.dump_pass_ir_to_file,
        "out_of_bound_sync_check": current_config.out_of_bound_sync_check,
        "dummy_placeholder": current_config.dummy_placeholder,
        "read_write_bank_conflict": current_config.read_write_bank_conflict,
        "bind_reduction_using_block": current_config.bind_reduction_using_block,
        "use_realize_bound_simplify": current_config.use_realize_bound_simplify,
        "ir_location_enable": current_config.ir_location_enable,
        "l1_fusion_option": current_config.l1_fusion_option,
        "dynamic_shape": current_config.dynamic_shape,
        "enable_vector_hierarchical_emit":
            current_config.enable_vector_hierarchical_emit,
        "enable_branch_eliminator_else_case":
            current_config.enable_branch_eliminator_else_case,
        "let_stmt_not_inline": current_config.let_stmt_not_inline,
        "double_buffer_no_tail": current_config.double_buffer_no_tail}

    for attr in config_list:
        value = config_list[attr]
        if attr in current_config_map:
            current_config_map[attr] = value
        elif attr not in ["add_lower_pass", "dump_pass_ir_list"]:
            raise RuntimeError("build config has no parameter \"%s\"" % attr)

    config = tvm.make.node("BuildConfig", **current_config_map)

    if "add_lower_pass" in config_list:
        config.add_lower_pass = config_list["add_lower_pass"]
    else:
        config.add_lower_pass = current_config.add_lower_pass

    if "dump_pass_ir_list" in config_list:
        config.dump_pass_ir_list = config_list["dump_pass_ir_list"]
    else:
        config.dump_pass_ir_list = current_config.dump_pass_ir_list

    return config


def get_new_build_config(config, update_map):
    new_config = config
    for k, v in update_map.items():
        new_config = build_config_update(new_config, k, v)
    return new_config


# pylint: disable=invalid-name
# Add a lower pass to sync uop
build_config = build_module.build_config(
    apply_tbe_pass=True,
    dump_cce_code=False,
    add_lower_pass=get_pass_list(),
    predicate_realize_bound=True,
    constant_realize_extent_in_infer_bound=True,
    bool_storage_as_1bit=True,
    sync_mode=2,
    debug_message=False,
    out_of_bound_sync_check=False,
    save_temp_cce_file=True,
    double_buffer_non_reuse=False,
    bind_reduction_using_block=True,
    read_write_bank_conflict=False,
    dump_pass_ir_to_file=True,
    dump_pass_ir=False,
    dump_pass_ir_list=[],
    dummy_placeholder=False,
    use_realize_bound_simplify=False,
    ir_location_enable=False,
    l1_fusion_option=False,
    dynamic_shape=False,
    enable_vector_hierarchical_emit=False,
    enable_branch_eliminator_else_case=True,
    let_stmt_not_inline=False,
    double_buffer_no_tail=False,
)

# pylint: disable=invalid-name
# Add a lower pass option for dynamic shape
dynamic_build_config = get_new_build_config(
    build_config,
    {
        'dynamic_shape': True,
        'out_of_bound_sync_check': True,
        'dummy_placeholder': True,
        'enable_vector_hierarchical_emit': True,
        'enable_branch_eliminator_else_case': True,
        'let_stmt_not_inline': True,
        'double_buffer_no_tail': True,
    }
)
