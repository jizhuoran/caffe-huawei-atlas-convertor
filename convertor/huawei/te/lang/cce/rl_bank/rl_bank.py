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

rl bank, generate schedule from built-in bank or custom bank

"""
import os
import datetime
import json
from te import tvm
from te import platform as cceconf
from te.lang.cce.te_compute.util import shape_to_list
from te.lang.cce.te_schedule.util import gen_dfs_tensor_map
from te.lang.cce.te_schedule.util import get_reduce_axis_num
from te.lang.cce.rl_bank.withdraw import gen_sch_by_cheque
from te.lang.cce.rl_bank.bank_cfg import DTYPE_INDEX
from te.lang.cce.rl_bank.bank_cfg import TAG_INDEX

FILE_PATH = os.path.dirname(os.path.realpath(__file__))

RL_BANK_DICT = {}
# rl bank version should include te/pass/rl version
RL_BANK_VERSION = 'v001'


def gen_tensor_feature(dfs_tensor_list):
    '''
    gen_tensor_feature
    :param dfs_tensor_list:
    :return:
    '''
    feature_list = []
    depends_map = {}
    # get all tensor
    for tensor_idx, tensor in enumerate(dfs_tensor_list):
        if not isinstance(tensor, tvm.tensor.Tensor):
            return ""
        # olny support for PlaceholderOp and ComputeOp
        if not isinstance(tensor.op,
                          (tvm.tensor.PlaceholderOp, tvm.tensor.ComputeOp)):
            return ""
        curr_tensor_info_list = []
        # ===tag===
        op_tag = tensor.op.tag.split("|")[0] if tensor.op.tag else ""
        if op_tag not in TAG_INDEX:
            return ""
        curr_tensor_info_list.append(TAG_INDEX[op_tag])
        # ===axis===
        if isinstance(tensor.op, tvm.tensor.PlaceholderOp):
            # Placeholder has no axis, use shape
            curr_tensor_info_list.append(shape_to_list(tensor.op.shape))
            curr_tensor_info_list.append([])
        else:
            # output shape
            curr_tensor_info_list.append(
                [axis.dom.extent.value for axis in tensor.op.axis])
            # reduce axis idx
            curr_tensor_info_list.append(
                get_reduce_axis_num(tensor) if tensor.op.reduce_axis else [])
        # ===dtype===
        curr_tensor_info_list.append(DTYPE_INDEX[tensor.op.output(0).dtype])

        # ===depens===
        for input_tensor in tensor.op.input_tensors:
            depends_map.setdefault(input_tensor.op.name, []).append(tensor_idx)
        curr_tensor_info_list.append(depends_map.get(tensor.op.name, []))

        feature_list.append(curr_tensor_info_list)

    return str(feature_list)


def get_rl_bank_key(output_tensors, op_info=None):
    '''
    generate bank_key from tensor info
    tensor_info_list, one for one tensor:tag，output shape，
    output dtype，reduce_axis，depends
    :param output_tensors:
    :return:bank_key:
    '''
    # try to get dfs_tensor_list from op_info by auto_schedule
    dfs_tensor_list = []
    if op_info and op_info.get("dfs_tensor_list", None):
        dfs_tensor_list = op_info["dfs_tensor_list"]
    if not dfs_tensor_list:
        dfs_tensor_list = get_dfs_tensor_list(output_tensors)

    bank_key = gen_tensor_feature(dfs_tensor_list)

    return bank_key


def add_case(outputs, cheque, tick, bank_json_path):
    '''
    add cheque to specify bank path
    :param res:
    :param actions:
    :param bank_json_path:
    :return:
    '''
    bank_key = get_rl_bank_key(outputs)
    if not bank_key:
        return False
    if os.path.exists(bank_json_path):
        with open(bank_json_path) as json_file:
            base_key_actions_dict = json.load(json_file)
    else:
        base_key_actions_dict = {}
    # update
    base_key_actions_dict.update({bank_key: (cheque, tick)})
    if not os.path.exists(os.path.dirname(bank_json_path)):
        os.makedirs(os.path.dirname(bank_json_path))
    with open(bank_json_path, 'w') as outfile:
        json.dump(base_key_actions_dict, outfile, sort_keys=True)
    return True


def get_dfs_tensor_list(out_tensors):
    '''
    get_dfs_tensor_list
    :param out_tensors:
    :return:
    '''
    if not isinstance(out_tensors, list):
        out_tensors = [out_tensors]

    dfs_tensor_list, _, _, _ = gen_dfs_tensor_map(out_tensors)

    return dfs_tensor_list


def get_bank_path():
    '''
    get_bank_path
    :return:
    '''
    local_bank = os.getenv("LOCAL_BANK")  # pylint: disable=invalid-envvar-default
    spec_bank = os.getenv("SPEC_BANK")  # pylint: disable=invalid-envvar-default
    if local_bank:
        base_dir = FILE_PATH
    elif (spec_bank and os.path.isdir(spec_bank)):
        base_dir = spec_bank
    else:
        base_dir = ""
        # parse LD_LIBRARY_PATH to get bank install path
        ld_library_env = os.getenv("LD_LIBRARY_PATH", "")
        for env_item in ld_library_env.split(":"):
            env_item = env_item.strip()
            if env_item.endswith("/fwkacllib/lib64") \
                    or env_item.endswith("/atc/lib64"):
                base_dir = env_item[:-5]
                break
            elif env_item.endswith("/fwkacllib/lib64/") \
                    or env_item.endswith("/atc/lib64/"):
                base_dir = env_item[:-6]
                break
        if not base_dir:
            # fwkacllib or atc not in env LD_LIBRARY_PATH
            return ""
        base_dir = os.path.join(base_dir, "data/rl")
    return base_dir


def read_custom_bank(custom_bank_dir, bank_name):
    '''
    read custom bank, maybe there are more than 1 files.
    :param custom_bank_dir:
    :param bank_name:
    :return:
    '''
    custom_bank = {}
    bank_files = []
    for bank_json in os.listdir(custom_bank_dir):
        if bank_json.startswith(bank_name) and bank_json.endswith('.json'):
            bank_file = os.path.join(custom_bank_dir, bank_json)
            bank_files.append(bank_file)
            with open(bank_file) as fh_bank:
                tmp_bank = json.load(fh_bank)
            for key, (_, tick) in tmp_bank.items():
                if key in custom_bank and custom_bank[key][1] <= tick:
                    continue
                custom_bank[key] = tmp_bank[key]
    return custom_bank, bank_files


def merge_custom_bank(custom_bank_dir, bank_files, bank_name, custom_bank):
    '''
    if custom bank files > 1，merge them
    :param custom_bank_dir:
    :param bank_files: origin custom bank files
    :param bank_name: bank_name
    :param custom_bank: custom bank dict
    :return:
    '''
    if len(bank_files) > 1:
        for bank_file in bank_files:
            os.remove(bank_file)
        time_stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        custom_bank_file = '{}_{}.json'.format(bank_name, time_stamp)
        custom_bank_path = os.path.join(custom_bank_dir, custom_bank_file)
        with open(custom_bank_path, 'w') as bank_fh:
            json.dump(custom_bank, bank_fh)


def get_bank_name(soc_version):
    '''
    get bank name
    :return:
    '''
    aicore_type = cceconf.get_soc_spec(cceconf.AICORE_TYPE)
    core_num = cceconf.get_soc_spec(cceconf.CORE_NUM)
    bank_name = '{}_{}_{}_{}'.format(soc_version, aicore_type, core_num,
                                     RL_BANK_VERSION)
    return bank_name


def get_bank_dict(bank_name, soc_version):
    '''
    get_bank_dict
    :return:
    '''
    # if already get RL_BANK_DICT, just return
    if bank_name in RL_BANK_DICT:
        return True
    RL_BANK_DICT[bank_name] = {}

    bank_dir = get_bank_path()
    if not bank_dir:
        return False

    custom_bank_dir = os.path.join(bank_dir, soc_version, "custom")
    if os.path.isdir(custom_bank_dir):
        custom_bank, bank_files = read_custom_bank(custom_bank_dir, bank_name)
        RL_BANK_DICT[bank_name]["custom"] = custom_bank
        merge_custom_bank(custom_bank_dir, bank_files, bank_name, custom_bank)

    built_in_bank_path = os.path.join(bank_dir, soc_version,
                                      "built-in/%s.json" % bank_name)
    if os.path.exists(built_in_bank_path):
        with open(built_in_bank_path) as json_file:
            RL_BANK_DICT[bank_name]["built-in"] = json.load(json_file)
            return True
    # built-in bank not exists,disable bank
    return False


def query_rl_bank(out_tensors, op_info=None):
    '''
    query_rl_bank
    :param out_tensors:
    :param op_info:
    :return:
    '''
    try:
        if str(os.getenv("RL_BANK", True)).lower() != "true":  # pylint: disable=invalid-envvar-default
            return False, None
        soc_version = cceconf.get_soc_spec(cceconf.SOC_VERSION)
        bank_name = get_bank_name(soc_version)
        ret = get_bank_dict(bank_name, soc_version)
        if not ret:
            # read built-in bank fail, disable rl rank
            os.environ["RL_BANK"] = "False"
            return False, None

        # get_rl_bank_key by op_name
        rl_bank_key = get_rl_bank_key(out_tensors, op_info=op_info)
        if not rl_bank_key:
            return False, None

        # get cheque from bank
        cheque, _ = RL_BANK_DICT.get(bank_name,
                                     {}).get("custom",
                                             {}).get(rl_bank_key, ([], 0))
        if cheque:
            # hit custom bank
            ret, rl_schedule_obj = gen_sch_by_cheque(out_tensors, cheque)
            if ret:
                return True, rl_schedule_obj

        cheque, _ = RL_BANK_DICT.get(bank_name,
                                     {}).get("built-in",
                                             {}).get(rl_bank_key, ([], 0))
        if cheque:
            # hit built-in bank
            ret, rl_schedule_obj = gen_sch_by_cheque(out_tensors, cheque)
            if ret:
                return True, rl_schedule_obj
        return False, None
    except:  # pylint: disable=bare-except
        return False, None
