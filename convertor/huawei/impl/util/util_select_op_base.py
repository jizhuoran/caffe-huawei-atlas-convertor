#!/usr/bin/env python
# encoding: utf-8
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

util_select_op_base
"""
import json


def get_dynamic_param_in_json(param_desc_list):
    param_dynamic = {}
    for item in param_desc_list:
        param_dict = {}
        param_dict["name"] = item.element.name
        param_dict["dtype"] = item.element.datatype
        param_dict["format"] = item.element.format
        param_dynamic[item.classify] = param_dict
    param_dynamic_in_json = json.dumps(param_dynamic, indent=4)
    return param_dynamic_in_json


def gen_param(classify, name, datatype, format):
    return ParamItem(classify=classify,
                     element=Element(name=name,
                                     datatype=datatype,
                                     format=format))


class Element:
    def __init__(self, name, datatype, format):
        self.name = name
        self.datatype = datatype
        self.format = format


class ParamItem:
    def __init__(self, classify, element):
        self.classify = classify
        self.element = element
