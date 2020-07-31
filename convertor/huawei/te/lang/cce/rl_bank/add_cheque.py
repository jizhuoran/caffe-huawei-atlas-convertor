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

add cheque to specify bank
"""
import os
import pickle
import traceback
from te import platform as cceconf
from te.lang.cce.rl_bank.rl_bank import add_case
from te.lang.cce.rl_bank.cheque import gen_cheque
from te.lang.cce.rl_bank.rl_bank import get_bank_path
from te.lang.cce.rl_bank.rl_bank import get_bank_name


def get_output_tensors(output_tensors, output_names, load_obj):
    """
    get real output tensors
    :param output_tensors:
    :param output_names:
    :param load_obj:
    :return:
    """
    for output_name in output_names:
        for i in range(len(load_obj.stages)):
            stage = load_obj.stages[i]
            # support for tuple_reduce_sum
            if output_name.startswith(stage.op.name + '_v'):
                out_idx = int(output_name.split('_v')[-1])
                output_tensors.append(stage.op.output(out_idx))
            elif output_name == stage.op.name:
                out_idx = 0
                output_tensors.append(stage.op.output(out_idx))


def add_cheque_to_bank(sch_py_path, bank_type, kernel_name=""):
    """
    add_cheque_to_bank
    :param sch_py_path:
    :param bank_type:
    :param kernel_name:
    :return:
    """

    if not os.path.exists(sch_py_path):
        raise RuntimeError("%s not exists" % sch_py_path)

    if bank_type not in ["custom", "built-in"]:
        raise RuntimeError("bank_type must be custom or built-in,while is %s" % bank_type)

    with open(sch_py_path, 'r') as file_handler:
        shcedule_code_str = file_handler.read()

    tick = int(os.path.basename(sch_py_path).split('_')[0])

    output_tensors = []
    output_names = []
    code_line_list = shcedule_code_str.split("\n")
    for code_line in code_line_list:
        if "#op_outputs:" in code_line:
            output_names = [
                output.strip() for output in code_line.split("#op_outputs:")[1].split(",")
            ]
        elif "pickle.loads(" in code_line:
            tensor_pickle_byte = code_line.split("pickle.loads(b'")[-1][:-2].encode(
                'ISO-8859-1').decode('unicode-escape').encode('ISO-8859-1')
            load_obj = pickle.loads(tensor_pickle_byte)
            get_output_tensors(output_tensors, output_names, load_obj)

    if not output_tensors:
        raise RuntimeError("get output_tensors from schedule py file fail!!!")

    cheque_list = gen_cheque(sch_py_path, kernel_name=kernel_name)

    bank_dir = get_bank_path()
    soc_version = cceconf.get_soc_spec("SOC_VERSION")
    bank_name = get_bank_name(soc_version)
    bank_json_path = os.path.join(bank_dir, soc_version, bank_type, "%s.json" % bank_name)
    ret = add_case(output_tensors, cheque_list, tick, bank_json_path)
    if ret:
        return True
    return False


def try_add_cheque(sch_py_path, bank_type, kernel_name=""):
    """
    try_add_cheque
    :param sch_py_path:
    :param bank_type:
    :param kernel_name:
    :return:
    """
    try:
        ret = add_cheque_to_bank(sch_py_path, bank_type, kernel_name=kernel_name)
        return ret, ""
    except Exception:  # pylint: disable=broad-except
        return False, "sch_py_path:%s add cheque to %s bank fail:%s" % (sch_py_path, bank_type,
                                                                        traceback.format_exc())
