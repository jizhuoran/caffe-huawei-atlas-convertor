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

histogram_fixed_width
"""
from te import platform as tbe_platform
import te.lang.cce
import te.platform.cce_params as cce_params
from te.lang.cce.te_compute import irbuilder_api as kernel_api
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.platform.cce_build import build_config
from topi.cce import util

# segment num one time for copy_gm_to_ubuf
SEGMENT_SIZE_COPY_GM_TO_UB = 1024 * 10
# segment num one time for calcu_histogram,
# when do add use fp16, the value > 2048 will have a large error precision
SEGMENT_SIZE_CALCU_HISTOGRAM = 1024 * 10
SEGMENT_SIZE_CALCU_HISTOGRAM_MINI = 2048
# the max value int 64bit, 2**64 - 1
MAX_VALUE_UINT64 = 18446744073709551615
# the bit size of set_vector_mask
BIT_SIZE_OF_VECTOE_MASK = 128
# scalar -1
SCALAR_NEGATIVE_ONE = -1
# scalar 1
SCALAR_ONE = 1
# scalar 0
SCALAR_ZERO = 0


# pylint: disable=too-many-instance-attributes
class IrParams:
    """IR Params, include all params"""

    # pylint: disable=too-many-statements
    def __init__(self, ib_, dtype_list, shape_list, nbins):
        self.ir_builder = ib_
        self.input_dtype = dtype_list[0]
        self.input_range_dtype = dtype_list[1]
        self.output_dtype = dtype_list[2]
        self.nbins = nbins
        self.data_shape, self.data_range_shape, _ = shape_list

        # if data_type not fp32 will vconv input data
        self.is_need_mid_dtype = False
        if self.input_dtype not in ("float32", ):
            self.is_need_mid_dtype = True
            self.mid_dtype = "float32"
        else:
            self.mid_dtype = self.input_dtype

        # get dtype size, float16 size = 2 byte   / float32 size = 4 byte
        self.input_dtype_size = \
            tbe_platform.cce_intrin.get_bit_len(self.input_dtype) // \
            cce_params.VECTOR_INST_BLOCK_NUM
        self.output_dtype_size = \
            tbe_platform.cce_intrin.get_bit_len(self.output_dtype) // \
            cce_params.VECTOR_INST_BLOCK_NUM
        self.mid_dtype_size = \
            tbe_platform.cce_intrin.get_bit_len(self.mid_dtype) // \
            cce_params.VECTOR_INST_BLOCK_NUM

        # get one block data size, block align len, 1 block=16 fp16 and =8 fp32
        self.input_align_len = cce_params.BLOCK_REDUCE_INT8 // \
                               self.input_dtype_size
        self.output_align_len = cce_params.BLOCK_REDUCE_INT8 // \
                                self.output_dtype_size
        self.mid_align_len = cce_params.BLOCK_REDUCE_INT8 // \
                             self.mid_dtype_size

        # get vector data size, 8 block =16*8 fp16 and =8*8 when fp32
        self.input_vec_align_len = cce_params.VECTOR_INST_BLOCK_WIDTH // \
                                   self.input_dtype_size
        self.output_vec_align_len = cce_params.VECTOR_INST_BLOCK_WIDTH // \
                                    self.output_dtype_size
        self.mid_vec_align_len = cce_params.VECTOR_INST_BLOCK_WIDTH // \
                                 self.mid_dtype_size

        # for set_vec_mask
        self.uint64_all_one = tvm.const(MAX_VALUE_UINT64, "uint64")

        # get run plat, mini or cloud
        self.compile_plat = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
        self.segment_size_calcu_histogram = SEGMENT_SIZE_CALCU_HISTOGRAM
        if self.compile_plat in ("Ascend310",):
            self.segment_size_calcu_histogram = \
                SEGMENT_SIZE_CALCU_HISTOGRAM_MINI

        # cce pipe stri PIPE_ALL
        self.args_str = tvm.call_pure_intrin("int32", "tvm_cce_string_print",
                                             "PIPE_ALL")
        self.deqscale = tvm.call_pure_intrin("float16", "tvm_cce_string_print",
                                             "(half)1.000000e+00f")
        # tmp params for compute
        self.offset = 0
        self.out_begin = 0
        self.out_end = 0
        self.ub_size = 0

        self.src_ub = kernel_api.ib_new_alloc(self.ir_builder,
                                              self.input_dtype,
                                              [SEGMENT_SIZE_COPY_GM_TO_UB],
                                              "src_ub",
                                              scope=tbe_platform.scope_ubuf)
        self.get_ub_size(SEGMENT_SIZE_COPY_GM_TO_UB, self.input_dtype_size)
        self.range_src_ub = kernel_api.ib_new_alloc(
            self.ir_builder, self.input_range_dtype, [self.input_align_len],
            "range_src_ub", scope=tbe_platform.scope_ubuf)
        self.get_ub_size(self.input_align_len, self.mid_dtype_size)
        # offset: for vcadd
        self.vcadd_ub = kernel_api.ib_new_alloc(
            self.ir_builder,
            self.mid_dtype, [self.segment_size_calcu_histogram],
            "vcadd_ub",
            scope=tbe_platform.scope_ubuf)
        self.get_ub_size(self.segment_size_calcu_histogram,
                         self.mid_dtype_size)

        # offset:output in des ub
        if self.input_dtype != self.mid_dtype:
            self.src_mid_input_ub = \
                kernel_api.ib_new_alloc(self.ir_builder, self.mid_dtype,
                                        [SEGMENT_SIZE_COPY_GM_TO_UB],
                                        "src_mid_input_ub",
                                        scope=tbe_platform.scope_ubuf)
            self.get_ub_size(SEGMENT_SIZE_COPY_GM_TO_UB, self.mid_dtype_size)
            self.src_mid_input_range_ub = \
                kernel_api.ib_new_alloc(self.ir_builder, self.mid_dtype,
                                        [self.mid_vec_align_len],
                                        "src_mid_input_range_ub",
                                        scope=tbe_platform.scope_ubuf)
            self.get_ub_size(self.mid_vec_align_len, self.mid_dtype_size)
        else:
            self.src_mid_input_ub = self.src_ub
            self.src_mid_input_range_ub = self.range_src_ub

        self.reg = self.ir_builder.allocate(self.mid_dtype, (7, ),
                                            name="range_data",
                                            scope=cce_params.scope_reg)
        self.mask = \
            self.ir_builder.allocate("uint64", (4,), name="mask",
                                     scope=cce_params.scope_reg)
        self.set_mask_list = \
            self.ir_builder.allocate("uint64", (64,), name="mask",
                                     scope=cce_params.scope_reg)
        # for preprocess
        _shape = \
            [(((SEGMENT_SIZE_COPY_GM_TO_UB - 1) // self.mid_vec_align_len) +
              1) * self.mid_vec_align_len]
        self.range0_ub = kernel_api.ib_new_alloc(self.ir_builder,
                                                 self.mid_dtype,
                                                 [self.mid_vec_align_len * 2],
                                                 "range0_ub",
                                                 scope=tbe_platform.scope_ubuf)
        self.get_ub_size(self.mid_vec_align_len * 2, self.mid_dtype_size)

        # get output num per core
        self.device_core_num = \
            tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)

        self.ub_total_size = \
            tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)

        self.max_output_size = (self.ub_total_size * 0.8 -
                                self.ub_size) // self.mid_dtype_size // 6

        self.tmp_out_num_per_core = \
            ((self.nbins - 1) // self.device_core_num) + 1
        self.out_num_per_core = \
            ((self.tmp_out_num_per_core - 1) //
             self.output_align_len + 1)*self.output_align_len
        if self.out_num_per_core >= self.max_output_size:
            self.out_num_per_core = \
                int((self.max_output_size //
                     self.mid_vec_align_len)*self.mid_vec_align_len)
        self.is_same = 0 if self.nbins % self.out_num_per_core == 0 else 1
        self.core_num = self.nbins // self.out_num_per_core + self.is_same
        # offset:output in src ub
        _shape = \
            (((self.out_num_per_core + 1 - 1) // self.mid_vec_align_len) + 1) \
            * self.mid_vec_align_len
        self.src_output_ub = kernel_api.ib_new_alloc(
            self.ir_builder, self.mid_dtype, [_shape], "src_output_ub",
            scope=tbe_platform.scope_ubuf)
        self.src_output_ub_p1 = kernel_api.ib_new_alloc(
            self.ir_builder,
            self.mid_dtype,
            [_shape],
            "src_output_ub_p1",
            scope=tbe_platform.scope_ubuf)
        self.get_ub_size(_shape + _shape, self.mid_dtype_size)

        _shape = (((self.out_num_per_core - 1) // self.output_align_len) +
                  1) * self.output_align_len
        self.des_output_ub = kernel_api.ib_new_alloc(
            self.ir_builder, self.output_dtype, [_shape],
            "des_output_ub", scope=tbe_platform.scope_ubuf)
        # offset:tmp output in des ub for cast to des dtype
        self.des_tmp_output_ub = kernel_api.ib_new_alloc(
            self.ir_builder, self.output_dtype, [_shape],
            "des_tmp_output_ub", scope=tbe_platform.scope_ubuf)
        self.get_ub_size(_shape + _shape, self.output_dtype_size)
        if self.compile_plat in ("Ascend310",):
            self.src_fp16_output_ub = kernel_api.ib_new_alloc(
                self.ir_builder,
                "float16", [_shape],
                "src_fp16_output_ub",
                scope=tbe_platform.scope_ubuf)
            self.index_ub = kernel_api.ib_new_alloc(
                self.ir_builder, "float32", [self.mid_align_len], "index_ub",
                scope=tbe_platform.scope_ubuf)
            self.get_ub_size(_shape, 2)
            self.get_ub_size(self.mid_align_len, self.mid_dtype_size)
        # bind blockIdx.x
        self.block = tvm.thread_axis("blockIdx.x")
        self.ir_builder.scope_attr(self.block, "thread_extent", self.core_num)

    def get_ub_size(self, data_len, dtype_size):
        """add ub size in self.ub_size

        Parameters
        ----------
        data_len: new ub
        dtype_size: ub dtype size

        Returns
        -------
        None
        """
        self.ub_size = self.ub_size + data_len * dtype_size

    def get_block_offset_one_core(self):
        """get_block_offset_one_core

        Parameters
        ----------
        self : self

        Returns
        -------
        None
        """
        self.out_begin = self.block.var * tvm.const(self.out_num_per_core,
                                                    "int32")
        self.out_end = \
            self.block.var*tvm.const(self.out_num_per_core, "int32") \
            + tvm.const(self.out_num_per_core, "int32")
        # conv index of onecore to index ubvconv_deq
        if self.compile_plat in ("Ascend310",):
            kernel_api.kernel_vector_dup_fuc(self.ir_builder,
                                             [self.index_ub, 0], 1,
                                             [1, self.mid_vec_align_len])
            int_ub = kernel_api.ib_new_alloc(self.ir_builder,
                                             "int32", [8],
                                             "int_ub",
                                             scope=tbe_platform.scope_ubuf)
            fp16_ub = kernel_api.ib_new_alloc(self.ir_builder,
                                              "float16", [16],
                                              "fp16_ub",
                                              scope=tbe_platform.scope_ubuf)
            int_reg = self.ir_builder.allocate("int32", (1, ),
                                               name="int_data",
                                               scope=cce_params.scope_reg)
            self.ir_builder.emit(
                tvm.call_extern('int32', 'pipe_barrier', self.args_str))
            with self.ir_builder.if_scope(self.out_begin <= self.nbins):
                int_reg[0] = self.block.var
                self.ir_builder.emit(
                    tvm.call_extern('int32', 'pipe_barrier', self.args_str))
                kernel_api.kernel_vector_dup_fuc(self.ir_builder, [int_ub, 0],
                                                 int_reg[0], [1, 64])
                _addr_list = [[fp16_ub, 0], [int_ub, 0]]
                self.ir_builder.emit(
                    tvm.call_extern('int32', 'set_deqscale', self.deqscale))
                kernel_api.kernel_cast_to_fuc(self.ir_builder, _addr_list,
                                              [1, 64], "vconv_deq")
                _addr_list = [[self.index_ub, 0], [fp16_ub, 0]]
                # fp16 to s32
                kernel_api.kernel_cast_to_fuc(self.ir_builder, _addr_list,
                                              [1, self.mid_vec_align_len],
                                              "vconv_f162f32")
                kernel_api.kernel_scalar_to_one_fuc(
                    self.ir_builder, [[self.index_ub, 0], [self.index_ub, 0]],
                    [1, self.mid_vec_align_len],
                    ["vmuls", self.out_num_per_core])
                kernel_api.kernel_scalar_to_one_fuc(
                    self.ir_builder, [[self.index_ub, 0], [self.index_ub, 0]],
                    [1, self.mid_vec_align_len], ["vmuls", self.reg[3]])
                kernel_api.kernel_scalar_to_one_fuc(
                    self.ir_builder, [[self.index_ub, 0], [self.index_ub, 0]],
                    [1, self.mid_vec_align_len], ["vadds", self.reg[0]])
                self.ir_builder.emit(
                    tvm.call_extern('int32', 'pipe_barrier', self.args_str))
                self.reg[6] = self.index_ub.vload(0, self.mid_dtype)

    def get_mask_list(self, mask_list_num):
        """get_mask_list  for 0-64

        Parameters
        ----------
        mask_list_num: int
            the num of mask will be gen

        Returns
        -------
        None
        """
        with self.ir_builder.for_range(0, mask_list_num, for_type="serial",
                                       name="mask_index") as mask_index:
            with self.ir_builder.if_scope(mask_index == SCALAR_ZERO):
                self.set_mask_list[mask_index] = tvm.const(
                    SCALAR_ONE, "uint64")
            with self.ir_builder.else_scope():
                self.set_mask_list[mask_index] = \
                    tvm.const(SCALAR_ONE*2, "uint64") * \
                    self.set_mask_list[mask_index - 1]

    def get_mask_fuc(self, index, mask_index):
        """get_mask_fuc, calcu self.mask[4]
        get set_vec_mask when only one value(location is index)

        Parameters
        ----------
        index: int
            the value location
        mask_index:int
            0:self.mask[0]/self.mask[1]  1:self.mask[2]/self.mask[3]

        Returns
        -------
        None
        """
        self.mask[mask_index + 1] = tvm.const(SCALAR_ZERO, "uint64")
        self.mask[mask_index + 3] = tvm.const(SCALAR_ZERO, "uint64")
        mask_tail = index % self.mid_vec_align_len
        with self.ir_builder.if_scope(
                mask_tail <= (BIT_SIZE_OF_VECTOE_MASK // 2)):
            self.mask[mask_index + 2] = tvm.const(1, "uint64")
            with self.ir_builder.if_scope(mask_tail == SCALAR_ZERO):
                self.mask[mask_index] = tvm.const(1, "uint64")
            with self.ir_builder.else_scope():
                with self.ir_builder.for_range(0,
                                               mask_tail - 1,
                                               for_type="serial",
                                               name="mask_index"):
                    self.mask[
                        mask_index + 2] = \
                        self.mask[mask_index + 2] * tvm.const(2, "uint64")
                self.mask[mask_index] = \
                    self.mask[mask_index + 2] * tvm.const(2, "uint64")
        with self.ir_builder.else_scope():
            self.mask[mask_index + 1] = tvm.const(1, "uint64")
            mask_tail = mask_tail % (BIT_SIZE_OF_VECTOE_MASK // 2)
            with self.ir_builder.for_range(0,
                                           mask_tail,
                                           for_type="serial",
                                           name="mask_index"):
                self.mask[
                    mask_index + 1] = \
                    self.mask[mask_index + 1] * tvm.const(2, "uint64")


def _fuction_data_conv_ir(addr_info_list, data_info_list, params):
    """
    fuction_data_conv_ir

    Parameters
    ----------
    addr_info_list: list
        the addr info for input and output,
        [[out_ub,out_offset],[input_ub, input_offset]]
    data_info_list: list
        [data_len, data_align],data len of x and align len of type(x) 8block
    params: class
        all params

    Returns
    -------
    None
    """
    _src_addr, _ = addr_info_list[1]
    if params.is_need_mid_dtype is False:
        _des_addr, _ = addr_info_list[0]
        _des_addr = _src_addr
        return _des_addr
    _src_dtype = _src_addr.dtype

    def _run_vconv(fuc, cmd):
        fuc(params.ir_builder, addr_info_list, data_info_list, cmd)

    if _src_dtype == "int32":
        if params.compile_plat in ("Ascend310",):
            raise RuntimeError("Can not support int32 input in mini plat")
        cce_cmd = "vconv_s322f32"
        fuc_name = kernel_api.kernel_cast_to_fuc
        _run_vconv(fuc_name, cce_cmd)
    elif _src_dtype == "float16":
        cce_cmd = "vconv_f162f32"
        fuc_name = kernel_api.kernel_cast_to_fuc
        _run_vconv(fuc_name, cce_cmd)


# pylint: disable=too-many-statements
def _fuction_calcu_one_segment(calc_ub_info,
                               data_info_list,
                               segment_index,
                               params,
                               is_tail=False):
    """get histogram in one segment, result store in src_output_ub
    form : src_output_ub[i] = num++  if value < j
           src_output_ub[i] = src_output_ub[i + 1] - src_output_ub[i]

    Parameters
    ----------
    calc_ub_info: list
        include [calc_ub, calc_offset]
    data_info_list: list
        include [data_len, align_len]
    segment_index: int
        index segment of one pre process segment
    params: class
        include all params
    is_tail: bool
        is tail

    Returns
    -------
    None
    """
    def _do_cmp_calcu(repeat, cal_offset, nbins_index):
        if params.compile_plat in ("Ascend910", "Ascend610", "Ascend620"):
            params.ir_builder.emit(
                tvm.call_extern(
                    params.vcadd_ub.dtype, "vadds",
                    params.vcadd_ub.access_ptr("rw", offset=0),
                    calc_ub_info[0].access_ptr("r", offset=cal_offset),
                    nbins_index * params.reg[3] + params.reg[0], repeat, 1, 1,
                    8, 8))
        else:
            # scalar can not supprt int32 to float32 in mini
            params.ir_builder.emit(
                tvm.call_extern('int32', 'pipe_barrier', params.args_str))
            params.reg[5] = params.index_ub.vload(0, params.mid_dtype)
            kernel_api.kernel_scalar_to_one_fuc(
                params.ir_builder,
                [[params.index_ub, 0], [params.index_ub, 0]],
                [1, params.mid_vec_align_len], ["vadds", params.reg[3]])
            params.ir_builder.emit(
                tvm.call_extern(
                    params.vcadd_ub.dtype, "vadds",
                    params.vcadd_ub.access_ptr("rw", offset=0),
                    calc_ub_info[0].access_ptr("r", offset=cal_offset),
                    params.reg[5], repeat, 1, 1, 8, 8))
        params.ir_builder.emit(
            tvm.call_extern(params.vcadd_ub.dtype, "vmax",
                            params.vcadd_ub.access_ptr("rw", offset=0),
                            params.vcadd_ub.access_ptr("r", offset=0),
                            params.range0_ub.access_ptr("r", offset=0), repeat,
                            1, 1, 1, 8, 8, 0))
        params.ir_builder.emit(
            tvm.call_extern(
                params.vcadd_ub.dtype, "vmin",
                params.vcadd_ub.access_ptr("rw", offset=0),
                params.vcadd_ub.access_ptr("r", offset=0),
                params.range0_ub.access_ptr("r",
                                            offset=params.mid_vec_align_len),
                repeat, 1, 1, 1, 8, 8, 0))
        params.ir_builder.emit(
            tvm.call_extern(params.vcadd_ub.dtype, "vmuls",
                            params.vcadd_ub.access_ptr("rw", offset=0),
                            params.vcadd_ub.access_ptr("r", offset=0),
                            tvm.const(2**38, dtype=params.vcadd_ub.dtype),
                            repeat, 1, 1, 8, 8))
        params.ir_builder.emit(
            tvm.call_extern(params.vcadd_ub.dtype, "vmuls",
                            params.vcadd_ub.access_ptr("rw", offset=0),
                            params.vcadd_ub.access_ptr("r", offset=0),
                            tvm.const(2**44, dtype=params.vcadd_ub.dtype),
                            repeat, 1, 1, 8, 8))
        params.ir_builder.emit(
            tvm.call_extern(params.vcadd_ub.dtype, "vmuls",
                            params.vcadd_ub.access_ptr("rw", offset=0),
                            params.vcadd_ub.access_ptr("r", offset=0),
                            tvm.const(2**44, dtype=params.vcadd_ub.dtype),
                            repeat, 1, 1, 8, 8))

    def _do_calcu_f32_mini(cal_offset, nbins_index):
        with params.ir_builder.if_scope(
                tvm.any((nbins_index == SCALAR_ZERO),
                        (nbins_index == params.nbins))):
            with params.ir_builder.if_scope(nbins_index == 0):
                kernel_api.kernel_vector_dup_fuc(
                    params.ir_builder, [params.vcadd_ub, 0], 0,
                    [params.mid_vec_align_len, params.mid_vec_align_len])
                if params.compile_plat in ("Ascend310",):
                    kernel_api.kernel_scalar_to_one_fuc(
                        params.ir_builder,
                        [[params.index_ub, 0], [params.index_ub, 0]],
                        [1, params.mid_vec_align_len],
                        ["vadds", params.reg[3]])
            with params.ir_builder.else_scope():
                kernel_api.kernel_vector_dup_fuc(
                    params.ir_builder, [params.vcadd_ub, 0],
                    params.histogram_data_len,
                    [params.mid_vec_align_len, params.mid_vec_align_len])
        with params.ir_builder.else_scope():
            mask_paras = kernel_api.get_loopnum_and_masklist(
                params.histogram_data_len, params.mid_vec_align_len)
            _do_cmp_calcu(mask_paras[0] + mask_paras[1], cal_offset,
                          nbins_index)
            if mask_paras[0] > SCALAR_ONE:
                params.ir_builder.emit(
                    tvm.call_extern(
                        params.vcadd_ub.dtype, "vadd",
                        params.vcadd_ub.access_ptr("rw", offset=0),
                        params.vcadd_ub.access_ptr(
                            "r", offset=params.mid_vec_align_len),
                        params.vcadd_ub.access_ptr("r", offset=0),
                        mask_paras[0] - 1, 1, 1, 1, 0, 8, 0))
            if mask_paras[0] > SCALAR_ZERO:
                if mask_paras[1] == SCALAR_ONE:
                    params.ir_builder.emit(
                        tvm.call_extern("uint64", 'set_vector_mask',
                                        mask_paras[2][1], mask_paras[2][0]))
                    add_offset = mask_paras[0] * params.mid_vec_align_len
                    params.ir_builder.emit(
                        tvm.call_extern(
                            params.vcadd_ub.dtype, "vadd",
                            params.vcadd_ub.access_ptr("rw", offset=0),
                            params.vcadd_ub.access_ptr("r", offset=0),
                            params.vcadd_ub.access_ptr("r", offset=add_offset),
                            1, 1, 1, 1, 8, 8, 8))
                    params.ir_builder.emit(
                        tvm.call_extern("uint64", 'set_vector_mask',
                                        params.uint64_all_one,
                                        params.uint64_all_one))
            if mask_paras[0] == SCALAR_ZERO:
                params.ir_builder.emit(
                    tvm.call_extern("uint64", 'set_vector_mask',
                                    mask_paras[2][1], mask_paras[2][0]))
            params.ir_builder.emit(
                tvm.call_extern(params.vcadd_ub.dtype, "vcadd",
                                params.vcadd_ub.access_ptr("rw", offset=0),
                                params.vcadd_ub.access_ptr("r", offset=0), 1,
                                1, 1, 8))
            if mask_paras[0] == SCALAR_ZERO:
                params.ir_builder.emit(
                    tvm.call_extern("uint64", 'set_vector_mask',
                                    params.uint64_all_one,
                                    params.uint64_all_one))
            # bypass problem :addr not 32B align
            params.ir_builder.emit(
                tvm.call_extern('int32', 'pipe_barrier', params.args_str))
            params.reg[4] = params.vcadd_ub.vload(0, params.mid_dtype)
            kernel_api.kernel_vector_dup_fuc(
                params.ir_builder, [params.vcadd_ub, 0], params.reg[4],
                [params.mid_vec_align_len, params.mid_vec_align_len])
        # add num of index to src_output_ub
        nbins_index_core = nbins_index - params.block.var * \
                           params.out_num_per_core
        params.offset = \
            (nbins_index_core // params.mid_vec_align_len) * \
            params.mid_vec_align_len
        params.ir_builder.emit(
            tvm.call_extern("uint64", 'set_vector_mask', 0,
                            params.set_mask_list[nbins_index_core % 64]))
        params.ir_builder.emit(
            tvm.call_extern(
                params.src_output_ub.dtype, "vadd",
                params.src_output_ub.access_ptr("rw", offset=params.offset),
                params.vcadd_ub.access_ptr("r", offset=0),
                params.src_output_ub.access_ptr("r", offset=params.offset), 1,
                1, 1, 1, 8, 8, 8))
        # add num of index to src_output_ub_p1
        with params.ir_builder.if_scope(nbins_index_core != SCALAR_ZERO):
            with params.ir_builder.if_scope(nbins_index_core %
                                            64 == SCALAR_ZERO):
                params.ir_builder.emit(
                    tvm.call_extern("uint64", 'set_vector_mask', 0,
                                    params.set_mask_list[63]))
            with params.ir_builder.else_scope():
                params.ir_builder.emit(
                    tvm.call_extern(
                        "uint64", 'set_vector_mask', 0,
                        params.set_mask_list[nbins_index_core % 64 - 1]))
            params.offset = \
                ((nbins_index_core - 1) // params.mid_vec_align_len) * \
                params.mid_vec_align_len
            params.ir_builder.emit(
                tvm.call_extern(
                    params.src_output_ub_p1.dtype, "vadd",
                    params.src_output_ub_p1.access_ptr("rw",
                                                       offset=params.offset),
                    params.vcadd_ub.access_ptr("r", offset=0),
                    params.src_output_ub_p1.access_ptr("r",
                                                       offset=params.offset),
                    1, 1, 1, 1, 8, 8, 8))
        params.ir_builder.emit(
            tvm.call_extern("uint64", 'set_vector_mask', params.uint64_all_one,
                            params.uint64_all_one))

    if is_tail is False:
        params.histogram_data_len = params.segment_size_calcu_histogram
    else:
        params.histogram_data_len = \
            data_info_list[0] % params.segment_size_calcu_histogram
    # calcu the num if value < j in one segment,  0 <= j <=  nbins + 1
    # calcu result accumulate in src_out_ub
    if params.compile_plat in ("Ascend310",):
        kernel_api.kernel_vector_dup_fuc(params.ir_builder,
                                         [params.index_ub, 0], params.reg[6],
                                         [1, params.mid_vec_align_len])
    with params.ir_builder.for_range(0,
                                     params.out_num_per_core + SCALAR_ONE,
                                     name='out_index') as out_index:
        # one core only caluc one segment within (0, params.nbins + 1)
        core_index = params.block.var * params.out_num_per_core + out_index

        # process align data, calcu by params.segment_size_calcu_histogram
        offset = \
            calc_ub_info[1] + \
            segment_index * params.segment_size_calcu_histogram
        _do_calcu_f32_mini(offset, core_index)

    kernel_api.kernel_two_to_one_common_fuc(
        params.ir_builder,
        [[params.src_output_ub, 0], [params.src_output_ub_p1, 0],
         [params.src_output_ub, 0]],
        [params.out_num_per_core, params.mid_vec_align_len], "vsub")

    return True


def _fuction_accu_to_output(_ib, params):
    """accumulate des_output_ub in des_output_ub

    Parameters
    ----------
    _ib: tvm.ir_builder.IRBuilder
        Developer API of IR node builder make function.
    params: class
        include all params

    Returns
    -------
    None
    """
    _data_info = [params.out_num_per_core, params.output_vec_align_len]
    _addr_list = [[params.des_tmp_output_ub, 0], [params.src_output_ub, 0]]

    if params.compile_plat in ("Ascend310",):
        # mini do not support 32 to 32, first 32 to 16
        if params.src_output_ub.dtype == "float32":
            _addr_list = [[params.src_fp16_output_ub, 0],
                          [params.src_output_ub, 0]]
            kernel_api.kernel_cast_to_fuc(_ib, _addr_list, _data_info,
                                          "vconv_f322f16")
            _addr_list = [[params.des_tmp_output_ub, 0],
                          [params.src_fp16_output_ub, 0]]
        # fp16 to s32
        kernel_api.kernel_cast_to_fuc(_ib, _addr_list, _data_info,
                                      "vconv_f162s32r")
    elif params.compile_plat in ("Ascend910", "Ascend610", "Ascend620"):
        kernel_api.kernel_cast_to_fuc(_ib, _addr_list, _data_info,
                                      "vconv_f322s32r")

    _addr_list = [[params.des_output_ub, 0], [params.des_output_ub, 0],
                  [params.des_tmp_output_ub, 0]]
    kernel_api.kernel_two_to_one_common_fuc(_ib, _addr_list, _data_info,
                                            "vadd")


def _function_histogram_process_ir(calc_ub, calc_offset, data_info_list,
                                   params):
    """calcu histogram in a segment(2048) value, add result to final output ub

    Parameters
    ----------
    calc_ub: tvm.schedule.Buffer
        ub info will be used to calc histogram
    calc_offset: int
        the ipaddr offset of first element in calc_ub
    data_info_list: list
        data len and align_len
    params: class
        include all params

    Returns
    -------
    None
    """
    _data_len = data_info_list[0]
    _align_len = data_info_list[1]
    _data_info = [_data_len, _align_len]

    _addr_list = [[params.src_mid_input_ub, 0], [params.src_mid_input_ub, 0]]
    kernel_api.kernel_scalar_to_one_fuc(params.ir_builder, _addr_list,
                                        _data_info,
                                        ["vmuls", SCALAR_NEGATIVE_ONE])

    segmet_data_loop, tail_segment, _ = \
        kernel_api.get_loopnum_and_masklist(
            _data_len, params.segment_size_calcu_histogram)

    def _histogram_fuc(index, is_tail=False):
        # init tensor : tmp output tensor in src UB
        kernel_api.kernel_vector_dup_fuc(
            params.ir_builder, [params.src_output_ub, 0], SCALAR_ZERO,
            [(params.out_num_per_core // params.mid_vec_align_len + SCALAR_ONE)
             * params.mid_vec_align_len, params.mid_vec_align_len])
        kernel_api.kernel_vector_dup_fuc(
            params.ir_builder, [params.src_output_ub_p1, 0], SCALAR_ZERO,
            [(params.out_num_per_core // params.mid_vec_align_len + SCALAR_ONE)
             * params.mid_vec_align_len, params.mid_vec_align_len])
        # calu all num of one segment to src_output_ub
        _fuction_calcu_one_segment([calc_ub, calc_offset], _data_info, index,
                                   params, is_tail)
        # accumulate src_output_ub to output_ub
        _fuction_accu_to_output(params.ir_builder, params)

    with params.ir_builder.for_range(0, segmet_data_loop,
                                     name='n') as segment_index:
        _histogram_fuc(segment_index)

    if tail_segment == SCALAR_ONE:
        _histogram_fuc(segmet_data_loop, True)


def _histogram_fixed_width_ir(dst, src, nbins, shape_list):
    """
    IR node builder make function

    Parameters
    ----------
    dst: list
        the placeholders of dst
    src: list
        the placeholders of src, data, data_range = src
    nbins: int
        number of histogram bins.
    shape_list: list
        the shape list of srcs, data_shape, data_range_shape = shape_list

    Returns
    -------
    ib.get(): stmt
        The result statement.
    """
    data, data_range = src
    ib_create = tvm.ir_builder.create()
    # params init
    params = IrParams(ib_create, [data.dtype, data_range.dtype, dst[0].dtype],
                      [shape_list[0], shape_list[1], dst[0].shape], nbins)
    # calc out_begin and out_end  per core
    # init src_mid_input_ub
    kernel_api.kernel_vector_dup_fuc(
        params.ir_builder, [params.range0_ub, 0], SCALAR_ZERO,
        [params.mid_vec_align_len, params.mid_vec_align_len])
    kernel_api.kernel_vector_dup_fuc(
        params.ir_builder, [params.range0_ub, params.mid_vec_align_len],
        2**(-126), [params.mid_vec_align_len, params.mid_vec_align_len])
    # init tensor: output tensor, len=nbins
    kernel_api.kernel_vector_dup_fuc(
        params.ir_builder, [params.des_output_ub, 0], SCALAR_ZERO,
        [params.nbins, params.output_vec_align_len])
    kernel_api.kernel_vector_dup_fuc(
        params.ir_builder, [params.des_tmp_output_ub, 0], SCALAR_ZERO,
        [params.nbins, params.output_vec_align_len])
    # copy data_range from out to ub
    kernel_api.kernel_cp_fuc(
        params.ir_builder, [[params.range_src_ub, 0], [data_range, 0]],
        [params.data_range_shape[0], params.input_align_len],
        "copy_gm_to_ubuf")
    # vconv input to float32
    _fuction_data_conv_ir(
        [[params.src_mid_input_range_ub, 0], [params.range_src_ub, 0]], [
            params.data_range_shape[0],
            max(params.input_vec_align_len, params.mid_vec_align_len)
        ], params)
    params.ir_builder.emit(
        tvm.call_extern('int32', 'pipe_barrier', params.args_str))

    params.reg[0] = params.src_mid_input_range_ub.vload(0, params.mid_dtype)
    params.reg[1] = params.src_mid_input_range_ub.vload(1, params.mid_dtype)
    # range1 - range0
    kernel_api.kernel_vector_dup_fuc(
        params.ir_builder, [params.src_mid_input_ub, 0], params.reg[1],
        [params.mid_align_len, params.mid_align_len])
    _addr_list = [[params.src_mid_input_range_ub, 0],
                  [params.src_mid_input_ub, 0],
                  [params.src_mid_input_range_ub, 0]]
    kernel_api.kernel_two_to_one_common_fuc(params.ir_builder, _addr_list,
                                            [1, params.mid_align_len], "vsub")
    params.ir_builder.emit(
        tvm.call_extern('int32', 'pipe_barrier', params.args_str))

    params.reg[2] = params.src_mid_input_range_ub.vload(0, params.mid_dtype)

    _addr_list = [[params.src_mid_input_range_ub, 0],
                  [params.src_mid_input_range_ub, 0]]
    kernel_api.kernel_scalar_to_one_fuc(
        params.ir_builder, _addr_list, [1, params.mid_vec_align_len],
        ["vmuls", float(1.0 / params.nbins)])
    params.ir_builder.emit(
        tvm.call_extern('int32', 'pipe_barrier', params.args_str))
    params.reg[3] = params.src_mid_input_range_ub.vload(0, params.mid_dtype)
    # get 0-64 mask_value for set_vector_mask
    params.get_mask_list(64)
    params.get_block_offset_one_core()
    loop_and_mask_list = \
        kernel_api.get_loopnum_and_masklist(params.data_shape[0],
                                            SEGMENT_SIZE_COPY_GM_TO_UB)

    def _run_fuc(data_len, data_offset, copy_ub):
        # copy data(len=data_len,offset=data_offset) from out to ub
        kernel_api.kernel_cp_fuc(params.ir_builder,
                                 [[copy_ub, 0], [data, data_offset]],
                                 [data_len, params.input_align_len],
                                 "copy_gm_to_ubuf")
        _fuction_data_conv_ir(
            [[params.src_mid_input_ub, 0], [copy_ub, 0]],
            [data_len,
             max(params.input_align_len, params.mid_vec_align_len)], params)
        # fixed process:nbins*(values - value_range[0])/(value_range[1]
        # - value_range[0])
        _data_info_list = [data_len, params.mid_vec_align_len]

        # clac histogram in one SEGMENT_SIZE_COPY_GM_TO_UB and sum to output UB
        _function_histogram_process_ir(params.src_mid_input_ub, 0,
                                       _data_info_list, params)

    # data process SEGMENT_SIZE_COPY_GM_TO_UB by SEGMENT_SIZE_COPY_GM_TO_UB
    with params.ir_builder.for_range(0, loop_and_mask_list[0],
                                     name='m') as pre_index:
        _run_fuc(SEGMENT_SIZE_COPY_GM_TO_UB,
                 pre_index * SEGMENT_SIZE_COPY_GM_TO_UB, params.src_ub)
    # tail_data process; len = data_len % SEGMENT_SIZE_COPY_GM_TO_UB
    if loop_and_mask_list[1] == 1:
        _run_fuc(params.data_shape[0] % SEGMENT_SIZE_COPY_GM_TO_UB,
                 loop_and_mask_list[0] * SEGMENT_SIZE_COPY_GM_TO_UB,
                 params.src_ub)

    # copy result to out by mul cores
    def _copy_out(copy_data_len):
        kernel_api.kernel_cp_fuc(
            params.ir_builder,
            [[dst[0], params.out_begin], [params.des_output_ub, 0]],
            [copy_data_len, params.output_align_len], "copy_ubuf_to_gm")

    with params.ir_builder.if_scope(
            params.out_begin < tvm.const(params.nbins, "int32")):
        with params.ir_builder.if_scope(
                params.out_end <= tvm.const(params.nbins, "int32")):
            _copy_out(params.out_num_per_core)
        with params.ir_builder.else_scope():
            tail_core_num = params.nbins % params.out_num_per_core
            if tail_core_num != SCALAR_ZERO:
                _copy_out(tail_core_num)
    return params.ir_builder.get()


# pylint: disable=too-many-arguments,unused-argument
# pylint: disable=invalid-name,redefined-builtin
@fusion_manager.register("histogram_fixed_width_d")
def histogram_fixed_width_d_compute(x,
                                    range,
                                    y,
                                    nbins,
                                    kernel_name="histogram_fixed_width_d"):
    """TVM calculation process, used for fusion operation
    compute for histogram_fixed_width

    Parameters
    ----------
    x: TVM tensor
        the placeholders of x
    range: TVM tensor
        the placeholders of range
    y: dict
        dict info of output, not used
    nbins: int
        number of histogram bins.
    dtype: str
        data type for returned histogram.
    kernel_name: str
        cce kernel name, not used

    Returns
    -------
    res: TVM tensor
        the result histogram_fixed_width
    """
    dtype = "int32"
    input_values_shape = te.lang.cce.util.shape_to_list(x.shape)
    value_range_shape = te.lang.cce.util.shape_to_list(range.shape)
    res = tvm.extern(
        [nbins], [x, range],
        lambda ins, outs: _histogram_fixed_width_ir(
            outs, ins, nbins, [input_values_shape, value_range_shape]),
        name="res",
        dtype=dtype)
    return res


# pylint: disable=too-many-arguments,redefined-builtin
@util.check_input_type(dict, dict, dict, int, (str, int), str)
def histogram_fixed_width_d(x,
                            range,
                            y,
                            nbins,
                            dtype="int32",
                            kernel_name='histogram_fixed_width_d'):
    """this operation returns a rank 1 histogram counting
     the number of entries in `values` that fell into every bin.
      The bins are equal width and determined by the arguments
    `value_range` and `nbins`.

    Parameters
    ----------
    x: dict
        dict info of input value, must include the keys(shape and dtype).
    range: dict
        dict info of input value_range, must include the keys(shape and dtype).
                        the shape must be (2,) or [2]
    y: dict
        dict info of output
    nbins: int
        number of histogram bins.
    dtype: str
        data type for returned histogram.
    kernel_name: str
        cce kernel name, default value is "histogram_fixed_width"


    returns
    -------
    None
    """
    input_shape_list = [x.get("shape"), range.get("shape")]
    input_dtype = x.get("dtype")
    dtype_input = input_dtype.lower()

    util.check_shape_rule(input_shape_list[0])
    util.check_shape_rule(input_shape_list[1])
    util.check_kernel_name(kernel_name)
    util.compare_tensor_dict_key(x, range, "dtype")
    data_shape_size = util.check_tensor_shape_size(list(input_shape_list[0]))
    data_range_shape_size = util.check_tensor_shape_size(
        list(input_shape_list[1]))

    util.check_dtype_rule(dtype_input, ("float16", "float32", "int32"))

    if data_range_shape_size != 2:
        raise RuntimeError("the shape of range must be (2,) or [2]")

    if nbins <= 0:
        raise RuntimeError("the nbins must be > 0")

    data = tvm.placeholder([data_shape_size],
                           dtype=dtype_input,
                           name="input_data")
    range_data = tvm.placeholder([data_range_shape_size],
                                 dtype=dtype_input,
                                 name="input_range_data")

    res = histogram_fixed_width_d_compute(data, range_data, y, nbins,
                                          kernel_name)
    sch = tvm.create_schedule(res.op)
    with build_config:
        tvm.build(sch, [data, range_data, res], "cce", name=kernel_name)
