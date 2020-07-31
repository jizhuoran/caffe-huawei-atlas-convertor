"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     __init__.py
DESC:     make debug a module
CREATED:  2019-7-04 20:12:13
MODIFIED: 2019-7-22 20:04:45
"""

import numpy as np

from te.tik.tik_lib.tik_check_util import TikCheckUtil
from te.tik.tik_lib.tik_source_info import source_info_decorator
from .context import Context
from ..debugger import TikDebugger
from .statement import PrintExpr, TikDebug
from .tikdbg.codemapping import get_caller_context
from .decorators import tensor_set_as_decorator, build_cce_decorator, \
    broadcast_ub_to_l0c_decorator, col2img_decorator, datamove_decorator, \
    dma_dquant_decorator, else_scope_decorator, for_range_decorator, \
    if_scope_decorator, list_list_elewise_decorator, rpn_cor_decorator, \
    mov_cmpmask_to_tensor_decorator, \
    mov_tensor_to_cmpmask_decorator, object_special_decorator, \
    rpn_cor_diag_decorator, scalar_conv_decorator, scalar_set_as_decorator, \
    scalar_single_decorator, scat_vcmp_elewise_func_dec, \
    scatter_vconv_decorator, scatter_vector_scalar_decorator, \
    scat_vec_bin_ternary_ele_dec, vcmp_decorator, \
    scat_vec_single_elewise_dec, scatter_vmulconv_decorator, \
    scatter_vsel_decorator, set_rpn_cor_ir_decorator, tensor_move_decorator, \
    set_rpn_offset_decorator, tik_return_decorator, vcmpv_decorator, \
    vconcate_decorator, vconv_decorator, vec_reduce_decorator, \
    vec_bin_elewise_func_dec, vextract_decorator, vms_decarator, \
    vec_scalar_elewise_func_dec, vmulconv_decorator, vsel_decorator, \
    vec_scalar_single_elewise_dec, vms4_to_scalar_decorator, \
    vec_single_elewise_func_dec, vnchwconv_decorator, \
    vtranspose_decorator, load2dv1_decorator, scalar_binary_decorator, \
    load3dv1_decorator, mmad_decorator, vadddeqrelu_decorator, \
    vec_reduce_wo_order_decorator, tensor_register, scalar_register, \
    set_l0_set_value_decorator, set_2d_decorator, set_ctrl_bits, \
    get_ctrl_bits, vcmpvs_elewise_func_decorator, v4dtrans_decorator, \
    vreduce_decorator, vpadding_decorator, load3dv2_decorator, \
    load2dv2_decorator, vscatter_decorator, vgather_decorator, \
    depthwise_conv_decorator, load_smask_decorator, load_image_decorator, \
    vec_sel_decorator, vec_conv_decorator, vec_trans_scatter_decorator, \
    vec_reduce_group_decorator, vec_trans_decorator, vec_reduce_add_decorator, \
    vec_all_reduce_decorator, mmad_brc_decorator, \
    winograd_fm_transform_decorator, winograd_weight_trans_decorator, \
    vbi_decorator, vrsqrt_high_preci_decorator, vrec_high_preci_decorator, \
    vec_ln_high_preci_decorator, vexpm1_high_preci_decorator


class NumpyPrintSet():
    """set numpy print options and error set"""
    def __init__(self, threshold):
        self.threshold = threshold
        # save default print option
        self.default_print = np.get_printoptions()
        # save default how floating-point errors are handled
        self.default_err = np.geterr()

    def __enter__(self):
        # set print options and errors handled
        np.set_printoptions(threshold=self.threshold)
        np.seterr(all='raise')

    def __exit__(self, exc_type, exc_val, exc_tb):
        # reset to default
        np.set_printoptions(**self.default_print)
        np.seterr(**self.default_err)


class Tikdb():
    """tik debug class"""

    def __init__(self, tik_instance):
        """init with tik_instance

        Parameters
        ----------
        tik_instance : an instance of tik
        """
        self.context = Context(tik_instance.d_profiling)
        self.tik_instance = tik_instance

    @source_info_decorator()
    def start_debug(self, feed_dict, interactive=True):
        """start debug

        Parameters
        ----------
        feed_dict : symbol dict
        interactive : if interactive , True/False

        Returns
        -------
        None
        """
        if self.tik_instance.debug_disabled:
            TikCheckUtil.raise_error(
                'Debug function is disabled,'
                ' please try to open it when you define the tik_instance.')

        # set is_debugging equal to True to not check life cycle
        self.tik_instance.is_debugging = True

        if self.context.tensor_list:
            for tensor in self.context.tensor_list:
                self.context.add_tensor(tensor)

        self.context.interactive = interactive
        if interactive:
            TikDebug.set_trace(TikDebugger())
        else:
            TikDebug.set_trace(None)

        # set threshold of numpy print equals to 100000000, bigger than default
        np_print_threshold = 100000000
        with NumpyPrintSet(np_print_threshold):
            return self.context.eval_(feed_dict)

    @source_info_decorator()
    def debug_print(self, expr):
        """print when start debug

        Parameters
        ----------
        expr : str, user input

        Returns
        -------
        None
        """
        TikCheckUtil.check_type_match(expr, str, 'expr must be a string')
        print_stmt = PrintExpr(get_caller_context(), expr)
        self.context.curr_scope().add_stmt(print_stmt)
