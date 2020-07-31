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

CceConv3dBackpropInputOp
"""
from te.platform import cce_params
from .conv3d_backprop_input_general_schedule import general_schedule


class CceConv3dBackpropInputOp():   # pylint: disable=R0903
    """
    The class of conv3d backprop input

    Parameters
    ----------
    scope : cal buffer like local.UB

    need_tensorize : if need to doing tensorize when using calculate

    need_pragma : if need to doing pragma when using calculate

    Returns
    -------
    CceConv2dBackpropInputOp_instance : instance of CceConv2dBackpropInputOp
    """
    def __init__(self, scope, need_tensorize=True, need_pragma=True):
        self._scope = scope
        self._need_tensorize = need_tensorize
        self._need_pragma = need_pragma
        self._res_tensor = None
        self._spec_node_list = None

    def schedule(self, res, spec_node_list, sch_list):
        """
        auto_schedule for cce AI-CORE.

        Parameters
        ----------
        res : tvm.tensor

        spec_node_list : same as other template in cce_schedule

        sch_list: use sch_list[0] to return conv schedule

        Returns
        -------
        True for sucess, False for no schedule
        """
        self._res_tensor = res
        self._spec_node_list = spec_node_list
        cce_params.jump_expand_flag = True
        sch = general_schedule(res, sch_list)
        return sch
