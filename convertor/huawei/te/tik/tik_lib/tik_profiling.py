"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     tik_profiling.py
DESC:
CREATED:  2019-10-22 18:53:42
MODIFIED: 2019-10-22 18:53:42
"""
import errno    # pylint: disable=C0302
import os
import shutil
import stat
import json
from ctypes import cdll, c_uint64, c_uint32, c_void_p, c_char_p,\
    c_int64, c_double, c_bool
import numpy as np

from .profiler.cce_perf_regression_tool_base import gen_html
from .profiler.gen_css_and_js import gen_css
from ..common.util import DTYPE_FOR_INPUT_SCALAR
from .tik_check_util import TikCheckUtil

# too many arguments, too many statements
_MAX_INPUT_OUTPUT_NUM = 64


class CAModel():    # pylint: disable=R0902
    """
    Represents camodel.
    """
    # too many artibutes, because model has so many things
    def __init__(self, input_num, output_num,  # pylint: disable=R0913
                 input_scalar_num, input_scalar_uint_num,
                 input_scalar_int_num, input_scalar_double_num,
                 ft_scalar_num, ft_scalar_uint_num,
                 ft_scalar_int_num, ft_scalar_double_num):
        self.dll = cdll.LoadLibrary('libtik_profiling.so')
        self.build_bin_path = self.dll.BuildBinPath
        self.build_bin_path.restype = None
        self.build_bin_path.argtypes = [c_char_p]

        # send inputNum, outputNum, input data to execute camodel
        self.execute = self.dll.InitAndReadAndExecute
        self.execute.restype = bool
        self.execute.argtypes = [c_uint64*input_num,
                                 c_uint64*output_num,
                                 c_uint32*2, c_void_p*input_num,
                                 c_void_p*output_num]

        self.save_tensor_var_list = self.dll.saveTensorVarList
        self.save_tensor_var_list.restype = None
        self.save_tensor_var_list.argtypes = [c_uint32*input_num, c_uint32]

        self.save_input_scalar_value_uint = self.dll.saveInputScalarValueUint
        self.save_input_scalar_value_uint.restype = None
        self.save_input_scalar_value_uint.argtypes = [c_uint64*input_scalar_uint_num, c_uint32]

        self.save_input_scalar_value_int = self.dll.saveInputScalarValueInt
        self.save_input_scalar_value_int.restype = None
        self.save_input_scalar_value_int.argtypes = [c_int64*input_scalar_int_num, c_uint32]

        self.save_input_scalar_value_double = self.dll.saveInputScalarValueDouble
        self.save_input_scalar_value_double.restype = None
        self.save_input_scalar_value_double.argtypes = [c_double*input_scalar_double_num, c_uint32]

        self.save_input_scalar_dtype_value = self.dll.saveInputScalarValue
        self.save_input_scalar_dtype_value.restype = None
        self.save_input_scalar_dtype_value.argtypes = [c_uint32*input_scalar_num, c_uint32]

        self.save_ft_value_uint = self.dll.saveFtValueUint
        self.save_ft_value_uint.restype = None
        self.save_ft_value_uint.argtypes = [c_uint64*ft_scalar_uint_num,
                                            c_uint32]

        self.save_ft_value_int = self.dll.saveFtValueInt
        self.save_ft_value_int.restype = None
        self.save_ft_value_int.argtypes = [c_int64*ft_scalar_int_num,
                                           c_uint32]

        self.save_ft_value_double = self.dll.saveFtValueDouble
        self.save_ft_value_double.restype = None
        self.save_ft_value_double.argtypes = [c_double*ft_scalar_double_num,
                                              c_uint32]

        self.save_ft_scalar_dtype_value = self.dll.saveFtValue
        self.save_ft_scalar_dtype_value.restype = None
        self.save_ft_scalar_dtype_value.argtypes = [c_uint32*ft_scalar_num,
                                                    c_uint32]

        self.save_enable_l2 = self.dll.saveEnableL2
        self.save_enable_l2.restype = None
        self.save_enable_l2.argtypes = [c_bool]

    def __str__(self):
        pass

    def __hash__(self):
        pass


def _try_mkdir(path):
    """
    use to make dir
    :param path: where to mkdir
    :return: no
    """
    try:
        os.makedirs(path, 0o750)
    except OSError as os_exception:
        # ignore error if it is making existing dir
        if os_exception.errno != errno.EEXIST:
            if os_exception.errno == errno.EACCES:
                TikCheckUtil.raise_error("No permission to create " + path)
            if os_exception.errno == errno.ENOSPC:
                TikCheckUtil.raise_error("No space left to create " + path)
            # raise other all error to runtime error.
            TikCheckUtil.raise_error("Failed to create " + path)


def start_profiling(out_path,  # pylint: disable=R0913, R0914, R0915
                    kernel_name, feed_data, output_spec,
                    simulatorlog_path, generate_html, is_tensor_or_var_list,
                    input_scalar_value, input_scalar_dtype_list, enable_l2,
                    flowtable_scalar_dtype_list, flowtable_scalar_value):
    """
    use to execute profiling
    :param out_path: where can be found the .o file
    :param kernel_name: which .o file name
    :param feed_data:  input data
    :param output_spec: output data type,size,shape
    :param simulatorlog_path: where to put log
    :param generate_html: whether generate html or not
    :param is_tensor_or_var_list: 0 represents tensor, 1 represents inputscalar
    :param input_scalar_value: contains value of inputscalar
    :param input_scalar_dtype_list: save inputscalar dtype to backend
    :param enable_l2: will set whether enable l2 or not
    :return: execute output value
    """

    if simulatorlog_path is None:
        simulatorlog_abs_path = os.path.join(os.path.realpath(out_path),
                                             'simulatorlog')
    else:
        simulatorlog_abs_path = os.path.join(
            os.path.realpath(simulatorlog_path), 'simulatorlog')
    _try_mkdir(simulatorlog_abs_path)
    os.environ["CAMODEL_LOG_PATH"] = simulatorlog_abs_path
    # calculate the size we need to transform
    feed_data_num = len(feed_data)
    output_data_num = len(output_spec)
    if (feed_data_num + len(flowtable_scalar_value) > _MAX_INPUT_OUTPUT_NUM)\
            or (output_data_num > _MAX_INPUT_OUTPUT_NUM):
        TikCheckUtil.raise_error(
            "Input and output num should either less than 64!")
    absolute_kernel_path = os.path.join(os.path.realpath(out_path),
                                        kernel_name)
    feed_data_array = []
    feed_data_nbytes_array = []
    output_data_nbytes_array = []
    output_data_to_c = []
    output_value_array = []
    is_tensor_or_var_list_array = []

    for tmp_feed_data in feed_data:
        feed_data_array.append(c_void_p(np.ascontiguousarray(tmp_feed_data).
                                        ctypes.data))
        feed_data_nbytes_array.append(c_uint64
                                      (np.ascontiguousarray(tmp_feed_data).
                                       nbytes))
    for tmp_output_data in output_spec:
        output_data_nbytes_array.append(c_uint64(tmp_output_data.get("size")))
        output_data_zero = np.zeros(tmp_output_data['shape'],
                                    dtype=tmp_output_data['dtype'])
        output_value_array.append(output_data_zero)
        output_data_to_c.append(c_void_p(output_data_zero.ctypes.data))
    for tmp_value in is_tensor_or_var_list:
        is_tensor_or_var_list_array.append((c_uint32(tmp_value)))

    input_scalar_value_array_uint64, input_scalar_value_array_int64, \
    input_scalar_value_array_double, input_scalar_dtype_number_list = \
        _gen_scalar_list(input_scalar_dtype_list, input_scalar_value)

    ft_scalar_value_array_uint64, ft_scalar_value_array_int64, \
    ft_scalar_value_array_double, ft_scalar_dtype_number_list = \
        _gen_scalar_list(flowtable_scalar_dtype_list, flowtable_scalar_value)

    _two_num_uint_array = c_uint32*2
    input_and_output_num_array = _two_num_uint_array\
        (feed_data_num, output_data_num)  # will send to executeApi
    input_data_nbytes_array_to_c = (c_uint64*feed_data_num)\
        (*feed_data_nbytes_array)
    output_data_nbytes_array_to_c = (c_uint64*output_data_num)\
        (*output_data_nbytes_array)
    input_data_array_to_c = (c_void_p * feed_data_num)(*feed_data_array)
    output_data_array_to_c = (c_void_p * output_data_num)(*output_data_to_c)
    is_tensor_or_var_list_to_c = (c_uint32*feed_data_num)(*is_tensor_or_var_list_array)
    input_scalar_value_uint64_to_c = (c_uint64*len(input_scalar_value_array_uint64))\
        (*input_scalar_value_array_uint64)
    input_scalar_value_int64_to_c = (c_int64*len(input_scalar_value_array_int64)) \
        (*input_scalar_value_array_int64)
    input_scalar_value_double_to_c = (c_double*len(input_scalar_value_array_double)) \
        (*input_scalar_value_array_double)
    input_scalar_dtype_list_to_c = (c_uint32*len(input_scalar_dtype_list)) \
        (*input_scalar_dtype_number_list)

    ft_value_uint64_to_c = (c_uint64*len(ft_scalar_value_array_uint64))(
        *ft_scalar_value_array_uint64)
    ft_value_int64_to_c = (c_int64*len(ft_scalar_value_array_int64))(
        *ft_scalar_value_array_int64)
    ft_value_double_to_c = (c_double*len(ft_scalar_value_array_double))(
        *ft_scalar_value_array_double)
    ft_dtype_list_to_c = (c_uint32*len(flowtable_scalar_dtype_list))(
        *ft_scalar_dtype_number_list)

    ca_model = CAModel(feed_data_num, output_data_num, len(input_scalar_value),
                       len(input_scalar_value_array_uint64),
                       len(input_scalar_value_array_int64),
                       len(input_scalar_value_array_double),
                       len(flowtable_scalar_value),
                       len(ft_scalar_value_array_uint64),
                       len(ft_scalar_value_array_int64),
                       len(ft_scalar_value_array_double))

    # step 1, use this kernel_path to build bin path.
    ca_model.build_bin_path(c_char_p(bytes(absolute_kernel_path,
                                           encoding="utf-8")))
    # step 1.1, save if enable_l2 or not
    ca_model.save_enable_l2(c_bool(enable_l2))

    # step2, save is tensor or var list
    ca_model.save_tensor_var_list(is_tensor_or_var_list_to_c, (c_uint32(feed_data_num)))

    # step3, save input scalar value of uint
    ca_model.save_input_scalar_value_uint(input_scalar_value_uint64_to_c,
                                          (c_uint32(len(input_scalar_value_array_uint64))))

    # step4, save input scalar value of int
    ca_model.save_input_scalar_value_int(input_scalar_value_int64_to_c,
                                         (c_uint32(len(input_scalar_value_array_int64))))

    # step5, save input scalar value of double
    ca_model.save_input_scalar_value_double(input_scalar_value_double_to_c,
                                            (c_uint32(len(input_scalar_value_array_double))))
    # step6, save input scalar value
    ca_model.save_input_scalar_dtype_value(input_scalar_dtype_list_to_c,
                                           (c_uint32(len(input_scalar_value))))

    # step7, save ft scalar value of uint
    ca_model.\
        save_ft_value_uint(ft_value_uint64_to_c,
                           (c_uint32(len(ft_scalar_value_array_uint64))))

    # step8, save input scalar value of int
    ca_model.\
        save_ft_value_int(ft_value_int64_to_c,
                          (c_uint32(len(ft_scalar_value_array_int64))))

    # step9, save input scalar value of double
    ca_model.\
        save_ft_value_double(ft_value_double_to_c,
                             (c_uint32(len(ft_scalar_value_array_double))))
    # step10, save input scalar value
    ca_model.\
        save_ft_scalar_dtype_value(ft_dtype_list_to_c,
                                   (c_uint32(len(flowtable_scalar_value))))

    # step11, use camodel to execute camodel.
    return_value = ca_model.\
        execute(input_data_nbytes_array_to_c,
                output_data_nbytes_array_to_c,
                input_and_output_num_array,
                input_data_array_to_c,
                output_data_array_to_c)
    if not return_value:
        TikCheckUtil.raise_error("Failed to run camodel!")
    # log to 440
    modify_log_permission(simulatorlog_abs_path)
    if generate_html:
        generate_report(_get_block_dim(os.path.realpath(out_path),
                                       kernel_name),
                        os.path.realpath(out_path),
                        simulatorlog_abs_path, kernel_name)
    return output_value_array


def modify_log_permission(simulatorlog_absolute_path):
    """
    use to modify log permission
    :param simulatorlog_absolute_path: where can be found simulator log
    :return: None
    """
    for i in os.listdir(simulatorlog_absolute_path):
        if not i.endswith(".dump"):
            continue
        os.chmod(os.path.join(simulatorlog_absolute_path, i), stat.S_IRUSR |
                 stat.S_IRGRP)


def _get_block_dim(path, kernel_name):
    """
    get block dim for generate html
    :param path: kernel_meta path
    :param kernel_name: kernel name
    :return: get block dim number
    """
    json_path = os.path.join(path, kernel_name + '.json')
    with open(json_path, 'r') as temp_file:
        jobj = json.load(temp_file)
        block_num = jobj['blockDim']
    if not isinstance(block_num, int):
        TikCheckUtil.raise_error(
            "blockDim in json should be int, but is: ", block_num)
    return block_num


def generate_report(block_num, kernel_meta, calog_abs_path, kernel_name):
    """
    generate report
    :param block_num: block dim
    :param kernel_meta: kernel_meta path
    :param calog_abs_path: log path
    :param kernel_name: name of kernel
    :return:
    """
    # generate report
    report_dir = os.path.join(kernel_meta, kernel_name+"_html")
    shutil.rmtree(report_dir, True)
    gen_css(report_dir)
    for blk_id in range(block_num):
        gen_html(['-c', 'core{}'.format(blk_id), '-t',
                  '{}_core{}'.format(kernel_name, blk_id), '-d', calog_abs_path,
                  '-k', kernel_meta])
        shutil.move(os.path.
                    join(kernel_meta,
                         '{}_core{}.html'.format(kernel_name, blk_id)),
                    report_dir)

   
def _gen_scalar_list(scalar_dtype_list, scalar_value):
    """
    generate each int value for scalar
    :param scalar_dtype_list: save scalar dtype list
    :param scalar_value: save scalar value
    :return:
    """
    scalar_value_array_uint64 = []
    scalar_value_array_int64 = []
    scalar_value_array_double = []
    scalar_dtype_number_list = []
    for scalar_dtype, scalar_value in zip(scalar_dtype_list,
                                          scalar_value):
        if scalar_dtype.startswith("uint"):
            scalar_value_array_uint64.append(c_uint64(scalar_value))
        elif scalar_dtype.startswith("int"):
            scalar_value_array_int64.append(c_int64(scalar_value))
        elif scalar_dtype.startswith("float"):
            scalar_value_array_double.append(c_double(scalar_value))
        scalar_dtype_number_list.\
            append(c_uint32(DTYPE_FOR_INPUT_SCALAR[scalar_dtype]))
    return scalar_value_array_uint64, scalar_value_array_int64, \
           scalar_value_array_double, scalar_dtype_number_list
