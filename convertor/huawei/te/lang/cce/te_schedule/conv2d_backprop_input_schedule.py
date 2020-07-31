"""
Copyright 2019 Huawei Technologies Co., Ltd

CceConv2dBackpropInputOp
"""
from te.platform import cce_params
from .conv2d_backprop_input_general_schedule import general_schedule
from .conv2d_backprop_input_opti_schedule import opti_schedule


class CceConv2dBackpropInputOp():  # pylint: disable=R0903
    """
    The class of conv2d backprop input

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

        schedule = sch_list[0]
        is_general = False
        for stage in schedule.stages:
            operation = stage.op
            if operation.tag == 'im2col_row_major':
                _, _, _, kernel_h, kernel_w, _ = operation.output(0).shape
                if kernel_h.value != 1 or kernel_w.value != 1:
                    is_general = True
                    break
        if is_general:
            sch = general_schedule(res, sch_list)
        else:
            sch = opti_schedule(res, sch_list)
        return sch
