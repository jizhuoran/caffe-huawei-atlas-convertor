#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright 2019 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Schedule for cce depthwise_weight_6d_2_4d
"""

from __future__ import absolute_import
from __future__ import print_function
from te import tvm
from te import platform as cce
from te.platform.cce_build import build_config
import te.platform.cce_params as cce_params
from topi.cce import util


def _ceil_div(val, block):
    """
    :param val
    :param block
    :return: (val + block - 1) // block
    """
    return (val + block - 1) // block


def _ceil_fill(val, block):
    """
    :param val
    :param block
    :return: ((val + block - 1) // block)*block
    """
    return _ceil_div(val, block)*block


def _prod(values):
    """
    :param values: value list
    :return: _prod of values
    """
    res = 1
    for value in values:
        res *= value
    return res


def _apply_for_new_alloc(ib_, dtype, buf_len, align_size,
                         scope=cce_params.scope_ubuf):
    """
    :param ib_: ir builder
    :param dtype : the data type
    :param buf_len : apply buffer length
    :param align_size : for offset align
    :param scope : cce scope
    :return: new buffer
    """
    shape_x = (_ceil_fill(buf_len, align_size),)
    buf_var = ib_.allocate(dtype, shape_x, name="tmp_buf", scope=scope)
    tmp_buffer = tvm.decl_buffer(shape_x, buf_var.dtype,
                                 name="tmp_buf",
                                 scope=cce_params.scope_ubuf,
                                 data=buf_var)
    return tmp_buffer


# pylint: disable=locally-disabled,too-many-instance-attributes
class _BasicParams():
    """
    parameters for Segment
    """

    def __init__(self, ib_, dtype):
        self.ib_ = ib_
        self.dtype = dtype
        self.type_size = cce.cce_intrin.get_bit_len(dtype) // 8
        self.cp_align_len = cce_params.BLOCK_REDUCE_INT8 // self.type_size

        self.unified_buffer_len = \
            cce.CceProductParams().getParams("Unified_Buffer") // self.type_size
        self.vec_align_len = cce_params.VECTOR_INST_BLOCK_WIDTH // self.type_size
        self.uint8_max_value = 255
        self.last_block = \
            ib_.allocate("int32", (1,), name="last_block",
                         scope=cce_params.scope_reg)

        self.device_core_num = \
            cce.CceProductParams().getParams("Device_core_num")
        self.block = tvm.thread_axis("blockIdx.x")
        self.ib_.scope_attr(self.block, "thread_extent", self.device_core_num)

        self.input_ub = 0
        self.output_ub = 0

    def set_pipe_barrier(self, val):
        """
        :param val : "PIPE_ALL", "PIPE_MTE3", "PIPE_MTE2", "PIPE_MTE1",
        "PIPE_M", "PIPE_V", "PIPE_S"
        """
        args_str = tvm.call_pure_intrin("int32", "tvm_cce_string_print", val)
        self.ib_.emit(tvm.call_extern('int32', 'pipe_barrier', args_str))

    def apply_bufs(self, input_data_len, output_data_len):
        """
        :param input_len : input length
        :param output_len: output length
        """
        total_buf_len = _ceil_fill(input_data_len, self.vec_align_len) + \
                        _ceil_fill(output_data_len, self.vec_align_len)

        if total_buf_len > self.unified_buffer_len:
            return False

        self.input_ub = _apply_for_new_alloc(self.ib_, self.dtype,
                                             input_data_len, self.vec_align_len,
                                             cce_params.scope_ubuf)
        self.output_ub = _apply_for_new_alloc(self.ib_, self.dtype,
                                              output_data_len,
                                              self.vec_align_len,
                                              cce_params.scope_ubuf)
        return True


def _get_vec_align_len(dtype):
    """
    :param dtype: dtype
    :return: vec_align_len
    """
    type_size = cce.cce_intrin.get_bit_len(dtype) // 8
    return cce_params.VECTOR_INST_BLOCK_WIDTH // type_size


def _do_vector_dup(ubuf, dup_len, dtype, params, val=0):
    """
    :param ubuf: target ubuf
    :param dup_len : length to be dup
    :param dtype: target ubuf offset
    :param params : parameters
    :param val : val
    """
    buf, buf_offset = ubuf

    def _dump(data_len, cycle_offset):
        """
        :param data_len : length to dup
        :param cycle_offset : cycle_offset
        """
        params.ib_.emit(tvm.call_extern(
            dtype, 'vector_dup',
            buf.access_ptr("rw", offset=buf_offset + cycle_offset),
            tvm.const(val, dtype),
            _ceil_div(data_len, _get_vec_align_len(dtype)),
            1, 1, 8, 8))

    vec_buffer_max_len = params.uint8_max_value*params.vec_align_len
    num_cycle = dup_len // vec_buffer_max_len
    with params.ib_.for_range(0, num_cycle, for_type="serial", name="i") as i:
        _dump(vec_buffer_max_len, i*vec_buffer_max_len)
    tail_len = dup_len % vec_buffer_max_len
    with params.ib_.if_scope(tail_len > 0):
        _dump(tail_len, num_cycle*vec_buffer_max_len)

    params.set_pipe_barrier('PIPE_ALL')


def _do_cp_input_gm(input_gm, data_len, offset, params):
    """
    :param input_gm: gm input buf
    :param data_len : length to be add
    :param offset: gm offset
    :param params : parameters
    """
    params.ib_.emit(tvm.call_extern(
        params.dtype, 'copy_gm_to_ubuf',
        params.input_ub.access_ptr("rw", offset=0),
        input_gm.access_ptr("r", offset=offset),
        0, 1,
        _ceil_div(data_len, params.cp_align_len),
        0, 0))

    params.set_pipe_barrier('PIPE_ALL')


def _multi_core(_core_func, element_num, params):
    """
    :param _core_func : _core_func
    :param element_num : element num
    :param params : parameters
    """
    element_num_of_core = _ceil_div(element_num, params.device_core_num)
    out_begin = params.block.var*element_num_of_core
    out_end = params.block.var*element_num_of_core + element_num_of_core

    with params.ib_.if_scope(out_end <= element_num):
        _core_func(out_begin, out_end, element_num_of_core)
    with params.ib_.else_scope():
        with params.ib_.if_scope(out_begin < element_num):
            _core_func(out_begin, element_num, element_num_of_core)


# pylint: disable=locally-disabled,too-many-statements
def _all_in_fun(six2four, input_gm, output_gm, params):
    """
    :param six2four : six2four parameters
    :param input_gm : input gm
    :param output_gm : segment gm
    :param params : parameters
    """
    channel1, hight, weight, channel0 = six2four.get_hwc()
    channel = six2four.channel
    total_element = hight*weight*channel1

    block_num_of_core = _ceil_div(total_element, params.device_core_num)
    input_data_len = _prod(six2four.input_shape)
    output_data_len = block_num_of_core*channel

    if not params.apply_bufs(input_data_len, output_data_len):
        return False

    def _data_copy(c1_index, block_index, input_hw, channel0_num, offset):
        """
        :param c1_index : c1 index
        :param output_hw : output hw index
        :param input_hw : input hw index
        :param channel0_num : channel0 num
        :param offset : offset of data
        """
        with params.ib_.for_range(0, channel0_num, for_type="serial",
                                  name="c0_index") as c0_index:
            input_offset = (c1_index*hight*weight + input_hw)*channel0*channel0 \
                           + c0_index*channel0 + c0_index
            output_offset = block_index*channel0 + c0_index + offset

            # output_ub size >= _ceil_fill(output_data_len, params.cp_align_len)
            with params.ib_.if_scope(output_offset <
                                     _ceil_fill(output_data_len,
                                                params.cp_align_len)):
                value = params.input_ub.vload(input_offset)
                params.ib_.emit(params.output_ub.vstore(output_offset, value))

    def _core_func(out_begin, out_end, element_num_of_core):
        """
        :param out_begin : multi core output begin address
        :param out_end : multi core output end address
        :param element_num_of_core : element num of one core
        """
        if out_end != total_element or element_num_of_core == total_element:
            core_cal_num = element_num_of_core
        else:
            core_cal_num = total_element % element_num_of_core

        if core_cal_num == 0:
            return

        _do_vector_dup((params.output_ub, 0), output_data_len,
                       params.dtype, params)
        _do_cp_input_gm(input_gm, input_data_len, 0, params)

        c0_pad_len = _ceil_fill(channel, channel0) - channel

        def _do_data_copy(i, block_index):
            """
            :param block_index : block_index
            """
            i_hw = i // channel1
            c1_index = i % channel1

            if channel % channel0 == 0:
                _data_copy(c1_index, block_index, i_hw, channel0, 0)
            else:
                offset = ((i // channel1) - (out_begin // channel1))*c0_pad_len
                with params.ib_.if_scope(c1_index != channel1 - 1):
                    _data_copy(c1_index, block_index, i_hw, channel0, -offset)
                with params.ib_.else_scope():
                    _data_copy(c1_index, block_index, i_hw,
                               channel % channel0, -offset)

        with params.ib_.for_range(0, core_cal_num,
                                  for_type="serial", name="j") as j:
            i = out_begin + j
            _do_data_copy(i, j)

        core_out_len = core_cal_num*channel0

        if channel % channel0 != 0:
            out_begin_offset = params.ib_.allocate("int32", (1,),
                                                   name="out_begin_offset",
                                                   scope=cce_params.scope_reg)
            out_begin_offset[0] = (out_begin // channel1)*c0_pad_len
            core_out_len -= ((out_end // channel1)*c0_pad_len \
                             - out_begin_offset[0])
            pad_len = _ceil_fill(core_out_len, params.cp_align_len) \
                      - core_out_len
            with params.ib_.for_range(0, params.cp_align_len,
                                      for_type="serial", name="j") as j:
                i = out_begin + j + core_cal_num
                _do_data_copy(i, j + core_cal_num)

                real_pad_len = \
                    ((i + 1)*channel0 - ((i + 1) // channel1)*c0_pad_len) - \
                    ((out_begin + core_cal_num)*channel0 \
                     -((out_begin + core_cal_num) // channel1)*c0_pad_len)
                with params.ib_.if_scope(real_pad_len >= pad_len):
                    params.ib_.emit(tvm.call_extern(
                        params.dtype, 'break'))

        num_cp = _ceil_div(core_out_len, params.cp_align_len)
        output_offset = out_begin*channel0 - (out_begin // channel1)*c0_pad_len
        params.ib_.emit(tvm.call_extern(
            params.dtype, 'copy_ubuf_to_gm',
            output_gm.access_ptr("rw", offset=output_offset),
            params.output_ub.access_ptr("r", offset=0),
            0, 1,
            num_cp,
            0, 0))

    _multi_core(_core_func, total_element, params)
    return True


# pylint: disable=locally-disabled,too-many-statements
def _multi_in_multi_out_fun(six2four, input_gm, output_gm, params):
    """
    :param six2four : six2four parameters
    :param input_gm : input gm
    :param output_gm : segment gm
    :param params : parameters
    """
    channel1, hight, weight, channel0 = six2four.get_hwc()
    channel = six2four.channel
    len_params = {}
    len_params["total_element"] = hight*weight*channel1

    len_params["input_data_len"] = _prod(six2four.input_shape)
    len_params["input_block_len"] = _ceil_fill(params.unified_buffer_len // 3*2,
                                               params.vec_align_len)

    len_params["one_output_num"] = (params.unified_buffer_len -
                                    _ceil_fill(len_params["input_block_len"],
                                               params.vec_align_len)) // channel0
    if len_params["one_output_num"] <= 0:
        raise RuntimeError("len_params['one_output_num'] <= 0")

    len_params["output_data_len"] = len_params["one_output_num"]*channel0

    if not params.apply_bufs(len_params["input_block_len"], len_params["output_data_len"]):
        raise RuntimeError("apply buffers failed!")

    def _data_copy(c1_index, block_index, input_hw, channel0_num, offset):
        """
        :param c1_index : c1 index
        :param output_hw : output hw index
        :param input_hw : input hw index
        :param channel0_num : channel0 num
        :param offset : offset of data
        """
        with params.ib_.for_range(0, channel0_num,
                                  for_type="serial",
                                  name="c0_index") as c0_index:
            input_offset = (c1_index*hight*weight + \
                    input_hw)*channel0*channel0 + c0_index*channel0 + c0_index
            output_offset = block_index*channel0 + c0_index + offset

            with params.ib_.if_scope(params.last_block[0] != \
                                input_offset // len_params["input_block_len"]):
                params.last_block[0] = \
                    input_offset // len_params["input_block_len"]

                last_block = len_params["input_data_len"] \
                             // len_params["input_block_len"]
                with params.ib_.if_scope(params.last_block[0] == last_block):
                    _do_cp_input_gm(input_gm,
                                    len_params["input_data_len"] \
                                    % len_params["input_block_len"], \
                    params.last_block[0]*len_params["input_block_len"], params)
                with params.ib_.else_scope():
                    _do_cp_input_gm(input_gm, len_params["input_block_len"], \
                    params.last_block[0]*len_params["input_block_len"], params)

            with params.ib_.if_scope(output_offset <
                                     _ceil_fill(len_params["output_data_len"],
                                                params.cp_align_len)):
                value = params.input_ub.vload(input_offset \
                                              % len_params["input_block_len"])
                params.ib_.emit(params.output_ub.vstore(output_offset, value))

    def _do_data_copy(i, block_index, out_begin, c0_pad_len):
        i_hw = i // channel1
        c1_index = i % channel1
        if channel % channel0 == 0:
            _data_copy(c1_index, block_index, i_hw, channel0, 0)
        else:
            offset = ((i // channel1) - (out_begin // channel1))*c0_pad_len
            with params.ib_.if_scope(c1_index != channel1 - 1):
                _data_copy(c1_index, block_index, i_hw, channel0, -offset)
            with params.ib_.else_scope():
                _data_copy(c1_index, block_index, i_hw, channel % channel0,
                           -offset)

    def _out_put_one_time(out_begin, block_index, sub_num, tail_core=False):
        """
        :param out_begin : multi core output begin address
        :param block_index : block index
        :param sub_num : sub cycle num
        """
        c0_pad_len = _ceil_fill(channel, channel0) - channel
        _do_vector_dup((params.output_ub, 0), len_params["output_data_len"],
                       params.dtype, params)
        with params.ib_.for_range(0, sub_num, for_type="serial",
                                  name="i") as sub_i:
            i = out_begin + block_index*len_params["one_output_num"] + sub_i
            _do_data_copy(i, sub_i, out_begin, c0_pad_len)

        core_out_len = sub_num*channel0

        if channel % channel0 != 0 and tail_core:
            out_begin_offset = params.ib_.allocate("int32", (1,),
                                                   name="out_begin_offset",
                                                   scope=cce_params.scope_reg)
            out_begin_offset[0] = (out_begin // channel1)*c0_pad_len
            core_out_len -= (((out_begin + \
                            block_index*len_params["one_output_num"] + sub_num)
                              // channel1)*c0_pad_len - out_begin_offset[0])
            pad_len = \
                _ceil_fill(core_out_len, params.cp_align_len) - core_out_len
            with params.ib_.for_range(0, params.cp_align_len, for_type="serial",
                                      name="j") as sub_i:
                i = out_begin + block_index*len_params["one_output_num"] \
                    + sub_i + sub_num
                _do_data_copy(i, sub_i + sub_num, out_begin \
                        + block_index*len_params["one_output_num"], c0_pad_len)

                real_pad_len = ((i + 1)*channel0 - ((i + 1) \
                                // channel1)*c0_pad_len) - \
                               ((out_begin + \
                            block_index*len_params["one_output_num"] + sub_num)
                                *channel0 - \
            ((out_begin + block_index*len_params["one_output_num"] + sub_num)
                                 // channel1)*c0_pad_len)

                with params.ib_.if_scope(real_pad_len >= pad_len):
                    params.ib_.emit(tvm.call_extern(
                        params.dtype, 'break'))

        num_cp = _ceil_div(core_out_len, params.cp_align_len)
        out_gm_offset = (out_begin + \
                         block_index*len_params["one_output_num"])*channel0 - \
                        ((out_begin + block_index*len_params["one_output_num"]) \
                         // channel1)*c0_pad_len
        params.ib_.emit(tvm.call_extern(
            params.dtype, 'copy_ubuf_to_gm',
            output_gm.access_ptr("rw", offset=out_gm_offset),
            params.output_ub.access_ptr("r", offset=0),
            0, 1,
            num_cp,
            0, 0))

    def _core_func(out_begin, out_end, element_num_of_core):
        """
        :param out_begin : multi core output begin address
        :param out_end : multi core output end address
        :param element_num_of_core : element num of one core
        """
        if out_end != len_params["total_element"] or \
                element_num_of_core == len_params["total_element"]:
            core_cal_num = element_num_of_core
        else:
            core_cal_num = len_params["total_element"] % element_num_of_core

        if core_cal_num == 0:
            return

        params.last_block[0] = 0
        _do_cp_input_gm(input_gm, min(len_params["input_block_len"],
                                      len_params["input_data_len"]), 0, params)

        cycle_num = core_cal_num // len_params["one_output_num"]
        tail_num = core_cal_num % len_params["one_output_num"]
        if not isinstance(tail_num, int):
            raise RuntimeError("tail_num is not int")
        if tail_num > 0:
            core_inside_num = cycle_num
            tail_core_len = tail_num
        else:
            core_inside_num = cycle_num - 1
            tail_core_len = len_params["one_output_num"]

        with params.ib_.for_range(0, core_inside_num, for_type="serial",
                                  name="i") as i:
            _out_put_one_time(out_begin, i, len_params["one_output_num"])

        _out_put_one_time(out_begin, core_inside_num,
                          tail_core_len, tail_core=True)

    _multi_core(_core_func, len_params["total_element"], params)
    return True


def _intrin_factor(six2four, dtype, ins, outs):
    """
    :param six2four: six2four parameters
    :param dtype : the data type
    :param ins: input tensor
    :param outs: output tensor
    """

    input_data = ins[0]
    output_data = outs[0]
    ib_ = tvm.ir_builder.create()
    params = _BasicParams(ib_, dtype)
    with ib_.if_scope(params.block.var < params.device_core_num):
        if _all_in_fun(six2four, input_data, output_data, params):
            pass
        else:
            _multi_in_multi_out_fun(six2four, input_data, output_data, params)

    return ib_.get()


class _Six2FourParam():
    """
    parameters for Segment
    """

    def __init__(self, input_shape, channel_4d):
        self.input_shape = input_shape

        if len(input_shape) == 6:
            self.channel1, self.hight, self.weight, self.num,\
            self.channel0, channel0 = input_shape
            if self.num != 1:
                raise RuntimeError("N != 1.")
        else:
            raise RuntimeError("trans_depthwise_weight_6d_2_4d only"
                               "support 6D shape.")

        if self.channel0 != channel0 or self.channel0 != cce_params.C0_SIZE:
            raise RuntimeError("The last two dimensions of input_shape"
                               "must be 16.")

        if _ceil_div(channel_4d, self.channel0) != self.channel1:
            raise RuntimeError("ceil(channel_4d / C0) != C1.")
        self.channel = channel_4d

    def get_out_shape(self):
        """
        get 6D shape
        """
        return (self.hight, self.weight, self.channel, self.num)

    def get_hwc(self):
        """
        get h,w,c
        """
        return (self.channel1, self.hight, self.weight, self.channel0)


# pylint: disable=locally-disabled,invalid-name
def _check_parameters(x, y, src_format, dst_format):
    """
    check the parameters including src_shape, dst_shape,
    src_format, dst_format
    """
    if src_format.lower() != "c1hwncoc0":
        raise RuntimeError("src_format must be C1HWNCoC0 !")

    if dst_format.lower() != "hwcn":
        raise RuntimeError("dst_format must be HWCN!")

    if dst_format.lower() == "hwcn":
        src_shape = x.get("shape")
        dst_shape = y.get("shape")
        if src_shape[1] != dst_shape[0] or src_shape[2] != dst_shape[1]:
            raise RuntimeError("the shape of src and dst not match, "
                               "the 1st,2rd,dimension of shape "
                               "must be the same !")

# pylint: disable=locally-disabled,too-many-arguments,too-many-locals
@util.check_input_type(dict, dict, str, str, str)
def depthwise_weight_6d_2_4d(x, y, src_format, dst_format,
                             kernel_name="depthwise_weight_6d_2_4d"):
    """Operation and Schedule for depthwise_weight_6d_2_4d.

    Parameters
    ----------
    x: shape and dtype of input, the dtype support float16, float32,
    int32, uint16.

    y: the shape and dtype of outputs, the dtype same as input.

    src_format: the source data_format

    dst_format: the target data_format

    kernel_name : cce kernel name, default value is "depthwise_weight_6d_2_4d"

    Returns
    -------
        convert C1HWNCoC0 tp HWCN
    """
    _check_parameters(x, y, src_format, dst_format)
    output_shape = y.get("shape")
    channel_size = output_shape[2]
    input_shape = x.get("shape")
    dtype = x.get("dtype")
    channel_4d = channel_size
    util.check_shape_rule(input_shape)

    check_list = ("float16", "float32", "int32", "uint16")
    dtype = dtype.lower()
    util.check_dtype_rule(dtype, check_list)

    util.check_kernel_name(kernel_name)

    input_data = tvm.placeholder(input_shape, name="input_data", dtype=dtype)

    six2four = _Six2FourParam(input_shape, channel_4d)

    res = tvm.extern([six2four.get_out_shape()], [input_data],
                     lambda ins, outs: _intrin_factor(six2four, dtype,
                                                      ins, outs),
                     name="res", dtype=dtype)

    sch = tvm.create_schedule(res.op)
    build_list = [input_data, res]

    with build_config:
        tvm.build(sch, build_list, "cce", name=kernel_name)

# pylint: enable=too-many-arguments
