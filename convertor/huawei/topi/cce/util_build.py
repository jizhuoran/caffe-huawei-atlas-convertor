"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this
file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

common function for build
"""
# pylint: disable=invalid-name, import-error
from te.platform.cce_build import build_config_update
from te.platform.cce_build import build_config


def set_bool_storage_config():
    """
    update build config
    set is_bool_storage_as_1bit as false
    :return:
    """
    return build_config_update(build_config, "bool_storage_as_1bit", False)
