"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     tik_build.py
DESC:     tik build
CREATED:  2019-7-04 20:12:13
MODIFIED: 2019-8-06 15:04:45
"""
# disabling:
# C0103: invalid-name
# Tensor, Scalar, BuildCCE is the api of Tik, so disable them
# R0902: too-many-instance-attributes
# R0913: too-many-arguments
# R0901: too-many-ancestors
# pylint: disable=C0103, R0902, R0913, R0901

import os
from math import ceil
from resource import setrlimit, RLIMIT_STACK
import numpy as np

from te.platform.cce_params import scope_gm, TIK_WORKSPACE_SIZE_LIST,\
    TIK_ATOMIC_ADD_LIST, CCE_AXIS, GM_NAME_MAP_CLASS
from te.platform.cce_build import build_config
from te.platform.cce_build import build_config_update
from te.platform.cce_build import build_config_update_list
from te.tvm import make, ir_pass, call_extern
from te import tvm
from te.tik.tik_lib import Expr, TikVectorApi, \
    TikDataOpApi, TikProposalApi, TikCubeApi, TikSysControlApi, \
    TikCompareApi, TikReduceApi, TikVecScatterApi
from te.tik.api.cube.tik_cube_api import TikCubeOpenApi
from .tik_vector_api import TikVectorApiv1
from .tik_data_operation_api import TikDataOpApiv1
from .tik_reduce_api import TikReduceApiv1
from .tik_cmp_api import TikCompareApiv1
from .tik_scalar_api import TikScalarApi
from .tik_scalar import InputScalar, Scalar
from .tik_tensor import Tensor
from .tik_dprofile import Dprofile
from .. import debug
from ..tik_lib.tik_params import MAX_IR_STATEMENT_NUM, \
    ONE_IR, TWO_IR, FOUR_IR, THREE_IR, AI_CORE_INDICATE, VA_REG
from ..tik_lib.tik_profiling import start_profiling
from ..common.util import DTYPE_SIZE, reduce_mul, DTYPE_FOR_INPUT_SCALAR, \
    DTYPE_INT_VALUE, get_check_feed_dict, check_scope
from ..tik_lib.tik_check_util import TikCheckUtil, ERROR_MSG_LEVEL
from ..tik_lib.tik_source_info import TikSourceInfo, source_info_decorator
from ..common.tik_get_soc_name import get_soc_name

_MIN_OUTPUTS_LENGTH = 1
_MAX_INPUT_OUTPUT_NUM = 64
# we will change stack for each 2500 IR num
_MIN_IR_NUM_FOR_MODIFY = 2500
# each 2500 will get 8192k stack
_STACK_BYTES_PER_MIN_MODIFY_IR = 8192 * 1024


@debug.build_cce_decorator
def build_cce(kernel_name, tik_instance, inputs, outputs, global_scalar_list,
              workspace_tensor_list, config, flowtable_tmp):
    """to generate cce file

    Parameters
    ----------
    kernel_name : this operator name
    tik_instance : the instance of TIK
    inputs : operator input
    outputs : operator output
    global_scalar_list : store global information
    workspace_tensor_list: store all workspace tensor
    config: build-config key value
    flowtable_tmp: flowtable_values
    Returns
    -------
    None
    """
    for i in inputs:
        TikCheckUtil.check_equality(i.scope, scope_gm,
                                    "inputs' scope should be scope_gm")
    for i in outputs:
        TikCheckUtil.check_equality(i.scope, scope_gm,
                                    "outputs' scope should be scope_gm")

    body, body_ir_num = make_body(tik_instance, global_scalar_list)

    build_schedule(kernel_name, inputs, outputs, body, body_ir_num,
                   workspace_tensor_list, config, flowtable_tmp)


def make_body(tik_instance, global_scalar_list):
    """
    Call tvm.make api to make a body.
    :param tik_instance: an instance of Tik
    :param global_scalar_list: scalar list
    :return: body
    """
    # pylint: disable=E1101
    # disable E1101, because pylint can't recognize symbol from back-end so

    from ..tik_lib.tik_util import non_stmt_judge
    body = tik_instance.get()
    body_ir_num = tik_instance.total_ir_lines
    body_judge = ir_pass.RemoveNoOp(body)
    if non_stmt_judge(body_judge):
        tmp_node = call_extern("uint64", "return")
        body = make.Evaluate(tmp_node)
        tik_instance.source_info.set_node_loc(body)
        # ir num will be empty once through here, set to 1
        body_ir_num = ONE_IR
    for i in global_scalar_list:
        body = i.merge_scalar(body)
        # each scalar list will generate 2 ir, one for scope, one for allocate
        body_ir_num += TWO_IR

    body = make.AttrStmt(CCE_AXIS, "IR_platform", make.StringImm("TIK"), body)
    tik_instance.source_info.set_node_loc(body)
    # IR_platform will generate 1 ir
    body_ir_num += ONE_IR

    body = make.AttrStmt(CCE_AXIS, "comment",
                         make.StringImm(AI_CORE_INDICATE
                                        + get_soc_name()), body)
    tik_instance.source_info.set_node_loc(body)
    # comment will generate 1 ir
    body_ir_num += ONE_IR

    for i in VA_REG:
        body = make.AttrStmt(CCE_AXIS, "var_pre_def", i, body)
        tik_instance.source_info.set_node_loc(body)
        # each VA_REG will generate 1 ir
        body_ir_num += ONE_IR

    for id_list in tik_instance.buffer_no_reuse_list:
        body = make.AttrStmt(None, "pragma_buffer_non_reuse",
                             tvm.call_extern("int64", "buffer_non_reuse",
                                             *id_list), body)
        tik_instance.source_info.set_node_loc(body)
        # each pragma_buffer_non_reuse will generate 1 ir
        body_ir_num += ONE_IR

    for id_list in tik_instance.buffer_reuse_list:
        body = make.AttrStmt(None, "pragma_buffer_reuse",
                             tvm.call_extern("int64", "buffer_reuse", *id_list),
                             body)
        tik_instance.source_info.set_node_loc(body)
        # each pragma_buffer_reuse will generate 1 ir
        body_ir_num += ONE_IR

    # here is the final body
    TikSourceInfo.update_node_loc()
    # visit the stmt to check whether all vars used in the right scope
    ir_pass.VariableScopeCheckPass(body)

    return body, body_ir_num


def build_schedule(kernel_name, inputs, outputs, body, body_ir_num,
                   workspace_tensor_list, config_map, flowtable):
    """
    Run tvm.build to build schedule
    :param kernel_name: operator kernel name
    :param inputs: input data
    :param outputs: output data
    :param body: body from make_body
    :param body_ir_num: body ir number
    :param workspace_tensor_list: all workspace tensor list
    :param config_map: build-config
    :param flowtable: flowtable
    :return: mod
    """
    inputs_placeholder = []
    input_vars = []
    input_tensors = []
    output_vars = []
    output_tensors = []
    for i in inputs:
        if isinstance(i, InputScalar):
            input_vars.append(i.get())
        else:
            input_tensors.append(i)
    for i in outputs:
        if isinstance(i, InputScalar):
            output_vars.append(i.get())
        else:
            output_tensors.append(i)

    flowtable_vars = [i.get() for i in flowtable]

    res = compute_res(input_tensors, output_tensors, inputs_placeholder,
                      body, body_ir_num,
                      workspace_tensor_list, inputs, outputs, flowtable)

    schedule = tvm.create_schedule([r.op for r in res])
    # clear source info before end of tik
    TikSourceInfo.end_and_clear()
    config = build_config_update(build_config, "ir_location_enable", True)
    if config_map is not None:
        config = build_config_update_list(config, config_map)
    with config:
        return tvm.build(schedule,
                         inputs_placeholder + input_vars + res +
                         output_vars + flowtable_vars,
                         "cce",
                         name=kernel_name)


def compute_res(inputs, outputs, inputs_placeholder, body, body_ir_num,
                workspace_tensor_list, all_inputs, all_outputs, flowtable):
    """
    Compute the result of inputs and outputs
    :param inputs: input, tensor etc.
    :param outputs: output
    :param inputs_placeholder: placeholder
    :param body: body from make_body
    :param body_ir_num: body ir num
    :param all_inputs: save all inputs tensor and inputscalar
    :param all_outputs: save all outputs tensor and inputscalar
    :param flowtable: flowtable
    :return: res
    """
    from te.tvm._api_internal import _ExternOp

    input_buffer = []
    output_buffer = []

    # total remain ir include produce 2 ir, realize 1 ir, extern_scope 1 ir
    body_ir_num += FOUR_IR

    for i in inputs:
        inputs_placeholder.append(
            tvm.placeholder(i.buffer.shape, i.buffer.dtype, i.buffer.name))
        input_buffer.append(i.buffer)
        TikSourceInfo.set_node_loc(i.buffer, loc=i.source_loc)
        # each input has one ir
        body_ir_num += ONE_IR
        _add_stomic_list_from_tensor_info(i.is_atomic_add)

    for i in all_inputs:
        if isinstance(i, InputScalar):
            _add_stomic_list_from_tensor_info(False)

    for i in outputs:
        output_buffer.append(i.buffer)
        TikSourceInfo.set_node_loc(i.buffer, loc=i.source_loc)
        # each output has three ir
        body_ir_num += THREE_IR
        _add_stomic_list_from_tensor_info(i.is_atomic_add)

    for i in all_outputs:
        if isinstance(i, InputScalar):
            _add_stomic_list_from_tensor_info(False)

    for i in workspace_tensor_list:
        output_buffer.append(i.buffer)
        TikSourceInfo.set_node_loc(i.buffer, loc=i.source_loc)
        _add_stomic_list_from_tensor_info(i.is_atomic_add)

    for i in flowtable:
        if isinstance(i, InputScalar):
            _add_stomic_list_from_tensor_info(False)

    if not output_buffer:
        output_name = "__fake_tensor"
    else:
        output_name = output_buffer[0].name
    if len(outputs) > _MIN_OUTPUTS_LENGTH:
        for i, output in enumerate(outputs):
            GM_NAME_MAP_CLASS.gm_name_map[output_name + "_v" +
                                          str(i)] = output.name

    extern_op = _ExternOp(output_name, "", None, inputs_placeholder,
                          input_buffer, output_buffer, body)
    TikSourceInfo.set_node_loc(extern_op)
    TikSourceInfo.update_node_loc()
    TikCheckUtil.check_le(
        body_ir_num, MAX_IR_STATEMENT_NUM,
        "Total IR num is already more than " + str(MAX_IR_STATEMENT_NUM) + "!")
    Tik.TOTAL_IR_NUM = body_ir_num
    return [extern_op.output(i) for i in range(len(output_buffer))]


def _add_stomic_list_from_tensor_info(is_atomic_add):
    """use to add 1 for is atomic add tensor, else add 0"""
    if is_atomic_add:
        TIK_ATOMIC_ADD_LIST.local_list.append(1)
    else:
        TIK_ATOMIC_ADD_LIST.local_list.append(0)


def tik_dsl_gen():
    """generate DSL"""
    from te.tvm import ir_builder
    return ir_builder.create()


def _check_start_profiling_feed_dict(last_inputs, feed_dict, last_flowtable):
    """for startProfiling function
    once last_inputs not empty, we should check key is same"""
    if last_inputs or feed_dict or last_flowtable:
        build_cce_input_list_tensor = []
        build_cce_input_list_var = []
        last_inputs = last_inputs + last_flowtable
        for i in last_inputs:
            if isinstance(i, InputScalar):
                build_cce_input_list_var.append(i.name)
            else:
                build_cce_input_list_tensor.append(i.name)
        build_cce_input_names = " ".join(i.name for i in last_inputs)
        build_cce_input_tensor_names = " ".join(build_cce_input_list_tensor)
        build_cce_input_var_names = " ".join(build_cce_input_list_var)
        build_list_tensor = set(i.name for i in last_inputs)
        get_check_feed_dict(
            feed_dict, build_cce_input_list_tensor,
            build_cce_input_list_var, build_list_tensor,
            build_cce_input_names, build_cce_input_tensor_names,
            build_cce_input_var_names)


class Tik(TikCubeOpenApi, TikVectorApiv1, TikDataOpApiv1,
          TikReduceApiv1, TikCompareApiv1,
          TikScalarApi, TikVectorApi, TikCubeApi,
          TikDataOpApi, TikProposalApi, TikSysControlApi,
          TikCompareApi, TikReduceApi, TikVecScatterApi):
    # pylint: disable=R0901
    # disable R0901 because Tik inherit all api
    """tik instance class"""
    # @cond
    GLOBAL_SCALAR_COUNT = 0
    TOTAL_IR_NUM = 0
    # @endcond

    def __init__(self, profiling=None,  # pylint: disable=W0613
                 disable_debug=False,
                 err_msg_level=0):
        '''
        Creates a TIK DSL container by passing a tik.Dprofile instance.

        Description:
          Creates a TIK DSL container by passing a tik.Dprofile instance.

        Args:
            profiling:  Configuration information of the Ascend AI processor.
            Dprofile is supported.

        Kwargs:
            disable_debug:  An optional bool for disabling the debug function.
            Defaults to False (debug enabled).
            err_msg_level:  An optional int for specifying the level of error
            messages to be printed. Defaults
            to 0. Value range:
                - 0: user level. The error messages, error file paths and line
                numbers, error codes and contexts,
                and usercall stacks are printed.
                - 1: developer level. The error messages, error file paths and
                line numbers, error codes and contexts,
                and all call stacks are printed.

        Restrictions:
          - If the build duration is strictly limited, the debug function can
          be used during operator development. After
          the code is submitted, you can manually set the disable_debug
          parameter to True when constructing a TIK
          instance to disable the debug function. This reduces the build time.
          - If disable_debug is set to True and the debug API is called after
          BuildCCE is complete, the program exits
          abnormally and the debugging fails.

        Returns:
          Instance of class TIK

        Examples:
          #The following is an example of enabling the debug function:
              from te import tik
              tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            # Alternatively,
              tik_instance = tik.Tik(tik.Dprofile("v100", "mini"),
                                    disable_debug=False)

          #The following is an example of disabling the debug function:
              from te import tik
              tik_instance = tik.Tik(tik.Dprofile("v100", "mini"), True)
           # Alternatively,
              tik_instance = tik.Tik(tik.Dprofile("v100", "mini"),
                                    disable_debug=True)

          #The following is an example of setting err_msg_level to
          #the developer level:
              from te import tik
              tik_instance = tik.Tik(tik.Dprofile("v100", "mini"),
                                    err_msg_level=1)
        '''
        super(Tik, self).__init__()
        tvm.location_init()
        # @cond
        self.source_info.register_source_info()
        self.d_profiling = Dprofile()
        self.global_scalar_list = []
        # for storing global information
        self.global_dict = {}
        self.code_buffer_manager.inject_dprofile(self.d_profiling)

        # record for profiling
        self.last_output_path = None
        self.last_kernel_name = None
        self.last_outputs = []
        self.last_enable_l2 = True
        self.last_inputs = []
        self.build_done = False
        self.last_flowtable = []
        # BuildCCE default mode
        # if true then not check life cycle
        self._is_building_cce = False

        # debug mode for checking life cycle
        # True: in debug mode; False: not debug mode
        # if true then not check life cycle
        self._is_debugging = False

        # debug status, if false, then can not debug
        self.debug_disabled_ = disable_debug

        self.tikdb = debug.Tikdb(self)
        self.context = self.tikdb.context

        # use this list to collect reuse and no_reuse relationship
        self.buffer_reuse_list = []
        self.buffer_no_reuse_list = []
        self._buffer_reuse_dict = dict()

        # use this list to collect workspace list.
        self._workspace_tensor_list = []
        self._workspace_tensor_name_set = set()
        TIK_WORKSPACE_SIZE_LIST.local_list = []

        # use to save all atomic add list
        TIK_ATOMIC_ADD_LIST.local_list = []

        # set error message level
        TikCheckUtil.check_in_range(err_msg_level, (0, 1),
                                    "err_msg_level only support 0 and 1,"
                                    " input value is {}".format(err_msg_level))
        ERROR_MSG_LEVEL.err_msg_level = err_msg_level

        self.source_info.clear_source_info()
        # @endcond

    # @cond
    @source_info_decorator()
    def get_available_buffer_size(self, buffer_scope=None):
        """
        get the available buffer size.
        :param buffer_scope:
        :return:  the buffer size of scope
        """
        check_scope(buffer_scope)
        return self.code_buffer_manager.buffer_aviable()[buffer_scope]
    # @endcond

    @staticmethod
    def _get_in_and_out_tmp(inputs, outputs, flowtable):
        """get inputs and outputs"""
        if not isinstance(inputs, (list, tuple)):
            inputs_tmp = [inputs]
        else:
            inputs_tmp = list(inputs)
        if not isinstance(outputs, (list, tuple)):
            outputs_tmp = [outputs]
        else:
            outputs_tmp = list(outputs)
        if flowtable is not None:
            if not isinstance(flowtable, (list, tuple)):
                flowtable_tmp = [flowtable]
            else:
                flowtable_tmp = list(flowtable)
        else:
            flowtable_tmp = []
        return inputs_tmp, outputs_tmp, flowtable_tmp

    def _set_workspace_list(self):
        """set workspace list"""
        for tmp_tensor in self._workspace_tensor_list:
            TIK_WORKSPACE_SIZE_LIST.local_list.append(
                DTYPE_SIZE[tmp_tensor.dtype]
                * reduce_mul(tmp_tensor.shape))

    def _check_in_and_out_num(self, inputs_tmp, outputs_tmp, flowtable_tmp):
        """check inputs and outputs nums"""
        if (len(inputs_tmp) + len(flowtable_tmp) > _MAX_INPUT_OUTPUT_NUM) \
                or (len(outputs_tmp) + len(self._workspace_tensor_list)
                        > _MAX_INPUT_OUTPUT_NUM):
            TikCheckUtil.raise_error(
                "Input and (output + workspace) num should either <= " +
                str(_MAX_INPUT_OUTPUT_NUM) + "!")

    @source_info_decorator()
    def BuildCCE(self,  # pylint: disable=R0914
                 kernel_name,
                 inputs,
                 outputs,
                 output_files_path=None,
                 enable_l2=False,
                 config=None,
                 flowtable=None):
        '''
        Generates DSL defined on the target machine

        Description:
          Generates DSL defined on the target machine and compiles the DSL into
           binary code that is executable on the
          Ascend AI processorHiSilicon SoC and corresponding
          configuration files.

        Args:

          kernel_name : A string; Specifies the names of the generated binary
          file and CCE kernel function.
            - Example:
              If the string test is passed, the generated binary file is named
              test.o, and the generated CCE kernel
              function is named test__kernel0.
          inputs : A list or tuple of tensors (Tensor) and/or Input
          Scalars (InputScalar) whose scope is scope_gm.
            - Specifies the operator inputs in the generated CCE kernel
            function. The list or tuple length is up to 64.
          outputs : A list or tuple of tensors whose scope is scope_gm.
            - Specifies the operator outputs in the generated CCE kernel
            function. The list or tuple length is up to 64.
            - When outputs=[], a length 1 array is returned whose element size
            is 0, that is, [ [0] ] .

        Kwargs:

          output_files_path : A string specifying the path of the generated
          files after build. Defaults to None,
          indicating the path ./kernel_meta in the current directory.
          enable_l2 : A bool specifying whether to enable L2 buffer enable.
          Defaults to False.This argument does not
          take effect.
          config : A dictionary including a key string and its value, used to
          configure the operator build properties.
            - Format : config = {"key" : value}
            - The following keys are supported : double_buffer_non_reuse: If
            set to True, the ping and pong variables in
            double_buffer are not reused.
            - Example :  config = {"double_buffer_non_reuse" : True}
          flowtable : A list or tuple of InputScalars.
          A flow table of tiling parameters (computed by the operator selector
          in the dynamic-shape scenario).
          The flowtable length and inputs length adds up to less than or
          equal to 64.


        Restrictions:
          - inputs and outputs must not have the same tensor. Otherwise, the
          TIK reports an error.
          - All non-workspace tensors with the scope of scope_gm must be in
          inputs or outputs. Otherwise, a build error
          is reported.
          - When there is no output, BuildCCE specifies an array whose length
          is 1 and data is 0, that is, outputs=[].
          The return value is [[0]].
          - In inputs, tensors must be placed before InputScalars.

        Returns:
          None

        Examples:
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            data_A = tik_instance.Tensor("float16", (128,), name="data_A",
                                        scope=tik.scope_gm)
            data_B = tik_instance.Tensor("float16", (128,), name="data_B",
                                        scope=tik.scope_gm)
            data_C = tik_instance.Tensor("float16", (128,), name="data_C",
                                        scope=tik.scope_gm)
            tik_instance.BuildCCE(kernel_name="simple_add",
                                 inputs=[data_A,data_B],outputs=[data_C])
        '''
        from te.platform.cce_params import OUTPUT_PATH_CLASS
        # @cond
        self.is_building_cce = True
        # @endcond
        TikCheckUtil.check_name_str_valid(kernel_name)
        if output_files_path is not None:
            OUTPUT_PATH_CLASS.output_path = os.path.realpath(output_files_path)

        self.d_profiling.registe()
        inputs_tmp, outputs_tmp, flowtable_tmp = \
            self._get_in_and_out_tmp(inputs, outputs, flowtable)
        with self.context.freeze():
            tensor_shape_test = (1,)
            if not outputs_tmp:
                outputs_tmp = [Tensor(self, dtype="float16",
                                      shape=tensor_shape_test,
                                      scope=scope_gm,
                                      name="__fake_tensor")]

        # before build cce, set TIK_WORKSPACE_SIZE_LIST
        self._set_workspace_list()

        # check inputs and outputs num either less than 64
        self._check_in_and_out_num(inputs_tmp, outputs_tmp, flowtable_tmp)
        # use to set resource for large ir num
        _modify_resource_for_ir(self.total_ir_lines)
        input_output_name_set = set()
        input_output_tensor_name_set = set()

        input_scalar_name_set, input_scalar_num, input_scalar_index_list, \
        input_tensor_index_list = \
            _gen_inputscalar_num(inputs_tmp, input_output_name_set,
                                 input_output_tensor_name_set)
        output_scalar_name_set, output_scalar_num, output_scalar_index_list, \
        output_tensor_index_list = \
            _gen_inputscalar_num(outputs_tmp, input_output_name_set,
                                 input_output_tensor_name_set)
        flowtable_name_set, flowtable_num = \
            _gen_flowtable_scalar(flowtable_tmp)
        TikCheckUtil.check_equality(len((input_output_name_set |
                                         flowtable_name_set) &
                                        self._workspace_tensor_name_set),
                                    0, "Workspace's name is used in "
                                       "input or output!")
        # add check input name is all different
        TikCheckUtil.check_equality(
            len(input_scalar_name_set | output_scalar_name_set |
                flowtable_name_set),
            input_scalar_num + output_scalar_num + flowtable_num,
            "Duplicate name found in InputScalar! Please make all the "
            "InputScalar with different name!")

        # check tensor name and input scalar name has same!
        TikCheckUtil.check_equality(
            len(input_output_tensor_name_set & (input_scalar_name_set
                                                | output_scalar_name_set
                                                | flowtable_name_set)),
            0, "InputScalar's name is used in input or output Tensor!")

        # check if sequence of inputs is Tensor first and inputScalar second.
        if input_tensor_index_list and input_scalar_index_list:
            # only compare when two list are not empty
            TikCheckUtil.check_ge(
                input_scalar_index_list[0], input_tensor_index_list[-1],
                "All InputScalar in inputs should be put behind the Tensor!"
                " But find Tensor index: {}, InputScalar index: {}.".format(
                    input_tensor_index_list[-1], input_scalar_index_list[0]))
        if output_tensor_index_list and output_scalar_index_list:
            # only compare when two list are not empty
            TikCheckUtil.check_ge(
                output_scalar_index_list[0], output_tensor_index_list[-1],
                "All InputScalar in inputs should be put behind the Tensor!"
                " But find Tensor index: {}, InputScalar index: {}.".format(
                    output_tensor_index_list[-1], output_scalar_index_list[0]))

        # check config is not none
        if config is not None:
            TikCheckUtil.check_type_match(config, dict,
                                          "config should be dict "
                                          "for BuildCCE")
            for i in config:
                TikCheckUtil.check_type_match(i, str,
                                              "config's key should be string")
        build_cce(kernel_name, self, inputs_tmp, outputs_tmp,
                  self.global_scalar_list, self._workspace_tensor_list,
                  config, flowtable_tmp)

        # @cond
        self.last_outputs = outputs_tmp
        self.last_inputs = inputs_tmp
        self.last_output_path = output_files_path
        self.last_kernel_name = kernel_name
        self.last_enable_l2 = enable_l2
        self.build_done = True
        self.last_flowtable = flowtable_tmp
        TIK_ATOMIC_ADD_LIST.local_list = []
        TIK_WORKSPACE_SIZE_LIST.local_list = []
        # @endcond

    @source_info_decorator()
    def Tensor(self, dtype, shape, scope, name, enable_buffer_reuse=False,
               no_reuse_list=None, reuse_list=None, is_workspace=False,
               is_atomic_add=False):
        '''
        Defines a Tensor variable.
        Description:
          Defines a Tensor variable.

        Kwargs:
          dtype : Data type of the Tensor object. Must be one of the
          following data types: uint8, int8, uint16, int16, float16
          , uint32, int32, float32, uint64, int64
          shape : A list or tuple of ints, specifying the shape of the
          Tensor object.
            - NOTICE :
            In the current version, only a list or tuple of immediate
            integrals is supported.
          scope : Buffer scope of the Tensor object, that is, buffer space
          where the Tensor object is located:
            - scope_cbuf: L1 Buffer
            - scope_cbuf_out: L1OUT Buffer
            - scope_ubuf: Unified Buffer (UB)
            - scope_gm: Global Memory (GM)
          name : A string specifying the name of the Tensor object.
          Only digits (0-9), uppercase letters (A-Z),
          lowercase letters (a-z), and underscores (_) are allowed.
          However, the name cannot start with a digit.
          If set to None, the name auto_tensor_$(COUNT) is automatically
          used, with COUNT starting at zero.
            - NOTICE :
            When scope is set to scope_gm, the name must not be __fake_tensor.
          is_workspace : A bool. Defaults to False. If set to True, the
          current tensor is used for storing intermediate
          data only.If set to True, scope must be scope_gm and the tensor
          must not be included in the input and output
          tensors(that is, the names of the input and output tensors do
          not contain the workspace tensor).
          is_atomic_add : A bool. Defaults to False. This argument does
          not take effect.
          enable_buffer_reuse: An internal optional parameters
          no_reuse_list: An internal optional parameters
          reuse_list : An internal optional parameters

        Restrictions:
          - When the total size of the tensors exceeds the total size of the
          corresponding buffer type, a build error is
          reported.In the following example, the size of data_a is 1025 x 1024
          bytes, which is greater than the total
          size of L1 buffer by 1 MB.
            @code
              import numpy as np
              import sys
              from te import tik
              import tvm
              def buffer_allocate_test6():
                  tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
                  data_a = tik_instance.Tensor("int8", (1025 * 1024,),
                                        name="data_a", scope=tik.scope_cbuf)
                  tik_instance.BuildCCE(kernel_name="buffer_allocate_test",
                                        inputs=[],outputs=[])
                  return tik_instance
              if __name__ == "__main__":
                  tik_instance = buffer_allocate_test6()
            @endcode
          Build error :
              RuntimeError: Appiled buffer size(1049600B) more than avaiable
              buffer size(1048576B).
          - If a tensor is access beyond its defined scope, a build error will
          be reported.
          In the following example, data_a_l1 is defined only in
          new_stmt_scope. Beyond its defined scope, an error will
          be reported when the data_move API is called to access
          data_a_l1 again.
          @code
              import numpy as np
              import sys
              from te import tik
              import tvm
              def tensor_outrange_examine_test6():
                  tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
                  data_a = tik_instance.Tensor("float16", (128,),
                                            name="data_a", scope=tik.scope_gm)
                  data_b = tik_instance.Tensor("float16", (128,),
                                            name="data_b", scope=tik.scope_gm)
                  with tik_instance.new_stmt_scope():
                      data_a_ub = tik_instance.Tensor("float16", (128,),
                                        name="data_a_ub", scope=tik.scope_ubuf)
                      data_a_l1 = tik_instance.Tensor("float16", (128,),
                                        name="data_a_l1", scope=tik.scope_cbuf)
                  tik_instance.data_move(data_a_l1, data_a, 0, 1, 128 // 16,
                                        0, 0)
                  tik_instance.data_move(data_a_ub, data_a_l1, 0, 1, 128 // 16,
                                        0, 0)
                  tik_instance.data_move(data_b, data_a_ub, 0, 1, 128 // 16,
                                        0, 0)
                  tik_instance.BuildCCE(kernel_name="tensor_outrange_examine",
                                        inputs=[data_a], outputs=[data_b])
                  return tik_instance
          @endcode
          Build error :
              RuntimeError: This tensor is not defined in this scope.
          - If a tensor is beyond its defined scope, the buffer can be reused.
          In the following example, as data_a_ub1 and data_a_ub2 are beyond the
           defined scopes, the occupied buffer of
          size 126,976 bytes (62 x 2 x 1024 bytes) can be reused by data_b_ub.
          @code
              import numpy as np
              import sys
              from te import tik
              import tvm
              def double_buffer_test6():
                  tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
                  data_a = tik_instance.Tensor("int8", (124 *1024,),
                                        name="data_a", scope=tik.scope_ubuf)
                  with tik_instance.for_range(0, 2):
                      data_a_ub1 = tik_instance.Tensor("int8", (62 * 1024,),
                                    name="data_a_ub1", scope=tik.scope_ubuf)
                      data_a_ub2 = tik_instance.Tensor("int8", (62 * 1024,),
                                    name="data_a_ub2", scope=tik.scope_ubuf)
                  data_b_ub = tik_instance.Tensor("int8", (125 * 1024,),
                                    name="data_b_ub", scope=tik.scope_ubuf)
                  tik_instance.BuildCCE(kernel_name="tbe_double_buffer_no_loop"
                                    , inputs=[ ], outputs=[ ])
                  return tik_instance
              if __name__ == "__main__":
                  tik_instance = double_buffer_test6()
          @endcode
          If data_b_ub exceeds the Unified Buffer size, the following error is
          reported during the build:
              RuntimeError: Tensor data_b_ub appiles buffer size(128000B) more
              than avaiable buffer size(126976B).
          - shape does not support scalar arguments. It supports only
          immediates or Python variables.
          - For user-defined tensors, the starting address of the allocated
          buffer scope will be aligned according to
          the following rules:
            - UB: 32-byte aligned
            - GM: alignment not required
            If the total size of a buffer type is exceeded due to address
            alignment, a build error is reported.

        Returns:
          Tensor instance

        Examples:
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            data_A = tik_instance.Tensor("float16", (128,), name="data_A",
                                        scope=tik.scope_gm)
        '''
        return self.Tensor_(dtype, shape, scope, name,
                            enable_buffer_reuse=enable_buffer_reuse,
                            no_reuse_list=no_reuse_list, reuse_list=reuse_list,
                            is_workspace=is_workspace,
                            is_atomic_add=is_atomic_add)

    # @cond
    def Tensor_(self, dtype, shape, scope, name, enable_buffer_reuse=False,
                no_reuse_list=None, reuse_list=None, is_workspace=False,
                is_atomic_add=False):
        """create Tensor
        note: use this function to call tik.Tensor inside!!
        """
        TikCheckUtil.check_name_str_valid(name)
        if name[0].isdigit():
            TikCheckUtil.raise_error(
                "tensor's name should not begin with a digit,"
                " input name: {}".format(name))
        # if name is __fake_tensor, scope is gm, it would not appear in cce_args
        if name == "__fake_tensor" and scope == scope_gm:
            TikCheckUtil.raise_error(
                "gm tensor's name should not be '__fake_tensor'")
        TikCheckUtil.check_type_match(enable_buffer_reuse, bool,
                                      "enable_buffer_reuse should be bool type"
                                      " for Tensor:{}".format(name))
        TikCheckUtil.check_type_match(is_atomic_add, bool,
                                      "is_atomic_add should be bool type for "
                                      "Tensor:{}".format(name))
        if no_reuse_list is not None:
            TikCheckUtil.check_equality(
                enable_buffer_reuse, True,
                "please check enable_buffer_reuse param")
            TikCheckUtil.check_type_match(
                no_reuse_list, (list, tuple),
                "no_reuse_list for Tensor {} should be list or tuple, "
                "input type is {}".format(name, type(no_reuse_list)))
        if reuse_list is not None:
            TikCheckUtil.check_equality(
                enable_buffer_reuse, True,
                "please check enable_buffer_reuse param")
            TikCheckUtil.check_type_match(
                reuse_list, (list, tuple),
                "reuse_list for Tensor {} should be list or tuple, "
                "input type is {}".format(name, type(reuse_list)))

        tmp_tensor = Tensor(self, dtype, shape, scope, name,
                            enable_buffer_reuse=enable_buffer_reuse,
                            is_workspace=is_workspace,
                            is_atomic_add=is_atomic_add)
        TikCheckUtil.check_type_match(is_workspace, bool,
                                      "is_workspace should be bool type for "
                                      "Tensor:{}".format(name))
        if is_workspace:
            # workspace should be gm
            TikCheckUtil.check_equality(
                scope, scope_gm, "Workspace' scope should be scope_gm for"
                                 " Tensor:" + name + ", but get:" + scope)
            self._workspace_tensor_list.append(tmp_tensor)
            if tmp_tensor.name in self._workspace_tensor_name_set:
                TikCheckUtil.raise_error("Workspace's name: " +
                                         tmp_tensor.name +
                                         " is already used before")
            self._workspace_tensor_name_set.add(tmp_tensor.name)
        if is_atomic_add:
            # atomic should be gm
            TikCheckUtil.check_equality(scope, scope_gm,
                                        "Atomic' scope should be scope_gm for "
                                        "Tensor:" + name + ", but get:" + scope)
        if enable_buffer_reuse:
            if no_reuse_list:
                self._update_buffer_use_list(tmp_tensor, no_reuse_list,
                                             "no_reuse")
            if reuse_list:
                self._update_buffer_use_list(tmp_tensor, reuse_list, "reuse")
            self._buffer_reuse_dict.update(
                {tmp_tensor.name: tmp_tensor.buffer_reuse_id})
        return tmp_tensor

    # @endcond

    def _update_buffer_use_list(self, tmp_tensor, name_list, reuse_type):
        """update buffer_reuse_list and buffer_no_reuse_list
           according to name_list

        Parameters
        ----------
        tmp_tensor : tensor defined
        name_list : list of reuse tensor or no_reuse tensor
        reuse_type : string, "reuse" or "no_reuse"

        Returns
        -------
        None
        """
        id_list = []
        for tensor_name in name_list:
            if tensor_name is None:
                if len(name_list) == 1 and reuse_type == "no_reuse":
                    break
                TikCheckUtil.raise_error(
                    "{}_list not support, input value"
                    " is {}".format(reuse_type, name_list))
            if isinstance(tensor_name, str) and \
                    tensor_name in self._buffer_reuse_dict:
                id_list.append(self._buffer_reuse_dict.get(tensor_name))
            else:
                TikCheckUtil.raise_error(
                    "{}_list contains unspecified reuse tensor, please check"
                    " enable_buffer_reuse param".format(reuse_type))
        id_list.append(tmp_tensor.buffer_reuse_id)
        if reuse_type == "reuse":
            if id_list not in self.buffer_reuse_list:
                self.buffer_reuse_list.append(id_list)
        else:
            if id_list not in self.buffer_no_reuse_list:
                self.buffer_no_reuse_list.append(id_list)

    @source_info_decorator()
    def Scalar(self, dtype="int64", name="reg_buf", init_value=None):
        '''
        Defines a Scalar variable.

        Description:
          Defines a Scalar variable.

        Kwargs:
          dtype : Data type of the Scalar object. Must be one of the
          following data types: int8, uint8, int16, uint16, float16, int32
          , uint32, float32, int64, uint64
            - Defaults to int64.
          name : A string specifying the name of the Scalar object. Only
          digits (0-9), uppercase letters (A-Z),
          lowercase letters (a-z), and underscores (_) are allowed.
            - Defaults to reg_buf$(COUNT), with COUNT starting at zero.
          init_value : Initial value; An immediate of type int or float; A
          Scalar variable; A Tensor value; An Expr
          consisting of a Scalar variable, an immediate, and a Tensor value
            - NOTICE :
            If the argument is an Expr, the immediate cannot be of type float.

        Restrictions:
          When the initial value is an Expr, the immediate can only be of
          integer type instead of float, for example:
          @code
              from te import tik
              tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
              index_reg = tik_instance.Scalar(dtype="float32")
              index_reg.set_as(10.2)
              # Assign an initial value to the scalar using init_value.
              index_reg1 = tik_instance.Scalar(dtype="float32",
                                                init_value=10.2)
              index_reg2 = tik_instance.Scalar(dtype="float32")
              # Expr. The immediate is of float type. An error occurs with the
              #CCE compiler (CCEC), because the hardware does not support this
              #data type.    index_reg2.set_as(index_reg+2.2);
          @endcode

        Returns:
          Scalar instance

        Examples:
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))

            index_reg = tik_instance.Scalar(dtype = "int32")
            index_reg.set_as(10)

            index_reg2 = tik_instance.Scalar(dtype = "float16")
            index_reg2.set_as(10.2)

            # Scalar variable
            index_reg3 = tik_instance.Scalar(dtype = "float16")
            index_reg3.set_as(index_reg2)

            # Tensor value
            data_A = tik_instance.Tensor("float16", (128,), name="data_A",
                                        scope=tik.scope_gm)
            index_reg3.set_as(data_A[0])

            #Expr
            index_reg4 = tik_instance.Scalar(dtype = "int32")
            index_reg4.set_as(index_reg+20)
        '''
        return self.Scalar_(dtype=dtype, name=name, init_value=init_value)

    # @cond
    def Scalar_(self, dtype="int64", name="reg_buf", init_value=None):
        """create scalar
        note: use this function to call tik.scalar inside!!
        """
        TikCheckUtil.check_name_str_valid(name)
        return Scalar(self, dtype, name, init_value)
    # @endcond

    @source_info_decorator()
    def InputScalar(self, dtype="int64", name="input_scalar"):
        '''
        Defines an InputScalar variable.

        Description:
          Defines an InputScalar variable. An InputScalar serves as an inputs
          argument passed to the BuildCCE call.
          It supports a range of basic data types including int, uint, and
          float.

        Kwargs:
          dtype : Data type of the InputScalar object. Must be one of the
          following data types: int8, uint8, int16, uint16, int32, uint32
          , int64, uint64, float16, float32
            - Defaults to int64.
          name : A string specifying the name of the InputScalar object. Only
          digits (0-9), uppercase letters (A-Z),
          lowercase letters (a-z), and underscores (_) are allowed.
            - Defaults to input_scalar. Ensure that each InputScalar variable
            has a unique name.

        Restrictions:
          - Currently, InputScalar can be used in scenarios where a variable
          argument is an Expr.
            @code
            #For example, if the repeat_times argument in vec_abs is an Expr,
            #the code can be written as follows :
                from te import tik
                tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
                data_A = tik_instance.Tensor("float16", (128,), name="data_A",
                                            scope=tik.scope_gm)
                data_B = tik_instance.Tensor("float16", (128,), name="data_B",
                                            scope=tik.scope_gm)
                src_ub = tik_instance.Tensor("float16", (128,), name="src_ub",
                                            scope=tik.scope_ubuf)
                dst_ub = tik_instance.Tensor("float16", (128,), name="dst_ub",
                                            scope=tik.scope_ubuf)
                inputscalar = tik_instance.InputScalar(dtype="int16",
                                            name="inputscalar")
                tik_instance.vec_abs(128, dst_ub, src_ub, inputscalar, 8, 8)
                tik_instance.BuildCCE(kernel_name="simple_add",
                            inputs=[data_A,data_B,inputscalar],outputs=[])
            @endcode
          - Ensure that each InputScalar object has a unique name.

        Examples:
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            data_A = tik_instance.Tensor("float16", (128,), name="data_A",
                                        scope=tik.scope_gm)
            data_B = tik_instance.Tensor("float16", (128,), name="data_B",
                                        scope=tik.scope_gm)
            abc = tik_instance.InputScalar(dtype="int16", name="abc")
            src_ub = tik_instance.Tensor("float16", (128,), name="src_ub",
                                        scope=tik.scope_ubuf)
            dst_ub = tik_instance.Tensor("float16", (128,), name="dst_ub",
                                        scope=tik.scope_ubuf)
            tik_instance.vec_abs(128, dst_ub, src_ub, abc, 8, 8)
            tik_instance.BuildCCE(kernel_name="simple_add",
                                inputs=[data_A,data_B,abc],outputs=[])
        '''
        TikCheckUtil.check_name_str_valid(name)
        TikCheckUtil.check_type_match(
            dtype, str, "dtype should be str, but get " + str(type(dtype)))
        TikCheckUtil.check_in_range(
            dtype, DTYPE_FOR_INPUT_SCALAR,
            "dtype only support: " + " ".join(DTYPE_FOR_INPUT_SCALAR) +
            ", but get " + dtype)
        return InputScalar(self, dtype=dtype, name=name)

    # @cond
    @staticmethod
    def expr(expr_, dtype=None):
        """create expr

        Parameters
        ----------
        expr_ : str, the input str
        dtype : type

        Returns
        -------
        expr
        """
        return Expr(expr_, dtype)

    @property
    def debug_disabled(self):
        """
        close debug function

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        return self.debug_disabled_

    @property
    def is_debugging(self):
        """
        check status of debug
        when _is_debugging is true, the debug mode is running.
        when _is_debugging is false, the debug mode is closed.

        Parameters
        ----------
        None

        Returns:bool
        -------
        return is_debugging mode

        """
        return self._is_debugging

    @is_debugging.setter
    def is_debugging(self, debug_mode):
        """
        run debug mode

        Parameters
        ----------
        debug_mode: bool
            if True: debug is running
            if False: debug is not running

        Returns
        -------
        None
        """
        self._is_debugging = debug_mode

    @property
    def is_building_cce(self):
        """
        check status of debug
        when _is_build_cce is true, the program is running cceBuild function.
        when _is_build_cce is false, the program don't run cceBuild function.

        Parameters
        ----------
        None

        Returns:bool
        -------
        return ccebuild model

        """
        return self._is_building_cce

    @is_building_cce.setter
    def is_building_cce(self, buildcce_mode):
        """
        run cceBuild

        Parameters
        ----------
        buildcce_mode: bool
            if True: running cceBuild function
            if False: not running cceBuild function

        Returns
        -------
        None
        """
        self._is_building_cce = buildcce_mode

    @source_info_decorator()
    def global_scalar(self, dtype="int64", name="reg_buf", init_value=None):
        """create global scalar

        Parameters
        ----------
        dtype : the scalar's type
        init_value : the scalar's init value
        name : scalar's name

        Returns
        -------
        scalar
        """
        # name already has been checked in Scalar func
        new_scalar = Scalar(self, dtype, name, init_value, if_global_scope=True)
        self.global_scalar_list.append(new_scalar)
        return new_scalar

    @source_info_decorator()
    def StartProfiling(self, feed_dict, simulatorlog_path=None,
                       generate_html=False):
        """
        :param feed_dict: input for profiling
        :param simulatorlog_path: where to put log
        :param generate_html:whether generate html or not
        :return:
        """
        # pylint: disable=C1801
        # because input is list or tuple so could use len
        TikCheckUtil.check_type_match(feed_dict, dict,
                                      "feed_dict should be dict")
        if self.build_done is False:
            TikCheckUtil.raise_error("BuildCCE must be called before "
                                     "StartProfiling!")
        output_path = self.last_output_path
        if output_path is None:
            output_path = 'kernel_meta'
        output_spec = []
        for t in self.last_outputs:
            if isinstance(t, Tensor):
                d = {}
                d['size'] = np.zeros(t.shape, dtype=t.dtype).nbytes
                d['shape'] = t.shape
                d['dtype'] = t.dtype
                output_spec.append(d)

        # once last_inputs not empty, we should check key is same
        _check_start_profiling_feed_dict(self.last_inputs,
                                         feed_dict, self.last_flowtable)

        # check dtype and shape should be same
        feed_data = []
        input_scalar_value = []
        input_scalar_dtype_list = []
        is_tensor_or_var_list = []
        flowtable_scalar_dtype_list = []
        flowtable_scalar_value = []
        for t in self.last_inputs:
            if isinstance(t, Tensor):
                TikCheckUtil.check_equality(
                    [int(s) for s in t.shape],
                    [int(s) for s in feed_dict[t.name].shape],
                    "%s input shape mismatch %s vs feed_dict %s" % (
                        t.name, t.shape, feed_dict[t.name].shape))
                TikCheckUtil.check_equality(
                    t.dtype, feed_dict[t.name].dtype,
                    "%s input dtype mismatch %s vs %s" % (
                        t.name, t.dtype, feed_dict[t.name].dtype))
                feed_data.append(feed_dict[t.name])
                is_tensor_or_var_list.append(0)
        for t in self.last_inputs:
            if isinstance(t, InputScalar):
                _check_inputscalar_type_match(t, feed_dict)
                tmp_np = np.array((feed_dict[t.name],)).astype(dtype=t.dtype)
                feed_data.append(tmp_np)
                is_tensor_or_var_list.append(1)
                input_scalar_value.append(feed_dict[t.name])
                input_scalar_dtype_list.append(t.dtype)
        for t in self.last_flowtable:
            _check_inputscalar_type_match(t, feed_dict)
            flowtable_scalar_dtype_list.append(t.dtype)
            flowtable_scalar_value.append(feed_dict[t.name])
        TikCheckUtil.check_type_match(
            generate_html, bool, "generate_html should be bool")
        return start_profiling(output_path, self.last_kernel_name, feed_data,
                               output_spec, simulatorlog_path, generate_html,
                               is_tensor_or_var_list, input_scalar_value,
                               input_scalar_dtype_list, self.last_enable_l2,
                               flowtable_scalar_dtype_list,
                               flowtable_scalar_value)

    def get_ir_num(self):
        """Use to return total it num
        """
        return self.TOTAL_IR_NUM
    # @endcond


def _modify_resource_for_ir(total_ir_nums):
    """Modify stack for generate ir,
        each 2500 IR contains 8192KB
        :param total_ir_nums: ir num in body
    """
    if total_ir_nums >= _MIN_IR_NUM_FOR_MODIFY:
        setrlimit(RLIMIT_STACK,
                  (ceil(total_ir_nums / _MIN_IR_NUM_FOR_MODIFY) *
                   _STACK_BYTES_PER_MIN_MODIFY_IR, -1))


def _gen_inputscalar_num(inputs_tmp, input_output_name_set, input_output_tensor_name_set):
    """
    use to gen inputscalar number
    :param inputs_tmp: total inputs
    :param input_output_name_set: input output name set
    :param input_output_tensor_name_set: name set
    :return:
    """
    input_scalar_name_set = set()
    input_scalar_num = 0
    input_scalar_index_list = []
    input_tensor_index_list = []
    for i, tmp_tensor in enumerate(inputs_tmp):
        input_output_name_set.add(tmp_tensor.name)
        if isinstance(tmp_tensor, InputScalar):
            input_scalar_name_set.add(tmp_tensor.name)
            input_scalar_num += 1
            input_scalar_index_list.append(i)
        elif isinstance(tmp_tensor, Tensor):
            input_output_tensor_name_set.add(tmp_tensor.name)
            input_tensor_index_list.append(i)
        else:
            TikCheckUtil.raise_error(
                "BuildCCE's input can only be Tensor or InputScalar!")
    return input_scalar_name_set, input_scalar_num, \
           input_scalar_index_list, input_tensor_index_list


def _gen_flowtable_scalar(flowtable_tmp):
    """
    use to gen flowtable_scalar number
    :param flowtable_tmp: total flowtable_tmp
    :return:
    """
    input_scalar_name_set = set()
    input_scalar_num = 0
    for tmp_tensor in flowtable_tmp:
        if isinstance(tmp_tensor, InputScalar):
            if tmp_tensor.name in input_scalar_name_set:
                TikCheckUtil.raise_error(
                    "Duplicate flowtable name found! %s" % tmp_tensor.name)
            input_scalar_name_set.add(tmp_tensor.name)
            input_scalar_num += 1
        else:
            TikCheckUtil.raise_error(
                "BuildCCE's flowtable can only be InputScalar!")
    return input_scalar_name_set, input_scalar_num


def _check_inputscalar_type_match(input_scalar, feed_dict):
    if input_scalar.dtype.startswith("int") or \
            input_scalar.dtype.startswith("uint"):
        TikCheckUtil.check_type_match(
            feed_dict[input_scalar.name], int,
            input_scalar.name + " is " + input_scalar.dtype +
            ", but value is float!")
        TikCheckUtil.check_in_range(
            feed_dict[input_scalar.name],
            range(DTYPE_INT_VALUE[input_scalar.dtype][0],
                  DTYPE_INT_VALUE[input_scalar.dtype][1] + 1),
            "%s is %s type, should in [{%s, %s], but get %s" %
            (input_scalar.name, input_scalar.dtype,
             DTYPE_INT_VALUE[input_scalar.dtype][0],
             DTYPE_INT_VALUE[input_scalar.dtype][1],
             feed_dict[input_scalar.name]))
    if input_scalar.dtype.startswith("float"):
        TikCheckUtil.check_type_match(feed_dict[input_scalar.name], float,
                                      input_scalar.name + " is " +
                                      input_scalar.dtype +
                                      ", but value is int!")
