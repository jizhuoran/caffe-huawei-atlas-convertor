"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

arg_max_with_kd
"""

# pylint: disable=too-many-lines,import-error
from te import tik, platform as tbe_platform
from topi.cce import util
from impl.util.util_select_op_base import gen_param, get_dynamic_param_in_json

# define a scalar for fp16 minimal
SCALAR_MIN_FP16 = -65500
# define a scalar for fp32 minimal
SCALAR_MIN_FP32 = -(2 ** 31 - 1)
# max set_mask_int64 value
MAX_MASK_INT64 = 2 ** 64 - 1
# the segment len to split tasks in a core
CORE_SEGMENT_LEN = 2048
# int32 num in 8*block
OUT_MASK = 64
# 0101 mask value
MASK_0_1 = 6148914691236517205
# default axis
AXIS_DEFAULT = 10000
# the elements each vec_trans_scatter repeat takes
DATA_EACH_VNCHWCONV = 16
# the axis size maximal, int32, because use int32 to store indices, add this constraint
AXIS_SIZE_MAX = 2 ** 31
# large task case threshold
LARGE_TASK_NUM_PER_CORE = 32


# return ceil(int1 / int2)
def _get_div_ceil_int(int1, int2):
    """Get Ceil Int

    Parameters
    ----------
    int1: int
        input int 1
    int2: int
        input int 2

    Returns
    -------
    ceil_int: int
    """
    _result = int1 // int2
    if int1 % int2 == 0:
        ceil_int = _result
    else:
        ceil_int = _result + 1

    return ceil_int


def _get_round_ceil_int(int1, int2):
    """Round To Ceil Int

    Parameters
    ----------
    int1: int
        input int 1
    int2: int
        input int 2

    Returns
    -------
    ceil_int: int
    """
    return _get_div_ceil_int(int1, int2) * int2


def _get_round_floor_int(int1, int2):
    """Round To Floor Int

    Parameters
    ----------
    int1: int
        input int 1
    int2: int
        input int 2

    Returns
    -------
    floor_int: int
    """
    return int1 // int2 * int2


def get_tiling_info_for_axis(input_dim, task_segment_len):
    """
    get_tiling_info_for_axis when arg with last dim

    Parameters
    ----------
    None

    Returns
    -------
    result : list
        buf_size, loop_times, over_size, align_flag
    """
    segment_size = task_segment_len
    align_flag = ((input_dim % segment_size) != 0)
    if segment_size <= input_dim:
        buf_size = segment_size
        loop_times = input_dim // buf_size
        over_size = input_dim - (loop_times * buf_size)
    else:
        loop_times = 0
        buf_size = input_dim
        over_size = buf_size
    return buf_size, loop_times, over_size, align_flag


# get the mask for range(0, 2*_len, step = 2), which contains the values in vcmax result
def _calu_mask_by_one_zero(_len):
    _mask_h, _mask_l = 0, 0
    if _len > 32:
        _mask_l = MASK_0_1
        for i in range(_len - 32):
            _mask_h = _mask_h + 2 ** (2 * i)
    else:
        _mask_h = 0
        for i in range(_len):
            _mask_l = _mask_l + 2 ** (2 * i)
    return _mask_h, _mask_l


# get the mask for region of [tail_len, self.data_each_block]
def _get_tail_mask_for_b16(tail_len):
    if tail_len <= OUT_MASK:
        mask = 2 ** tail_len - 1
        mask_h = MAX_MASK_INT64
        mask_l = MAX_MASK_INT64 - mask
    else:
        mask_l = 0
        mask = 2 ** (tail_len - OUT_MASK) - 1
        mask_h = MAX_MASK_INT64 - mask
    return mask_h, mask_l


# get the mask for region of [tail_len, self.data_each_block]
def _get_tail_mask_for_b32(tail_len):
    mask = 2 ** tail_len - 1
    mask_h = 0
    mask_l = MAX_MASK_INT64 - mask
    return mask_h, mask_l


# pylint: disable=invalid-name,too-many-boolean-expressions,unused-argument,too-many-arguments
# pylint: disable=too-many-locals,too-many-branches
def op_select_format(x, indices, values, axis=AXIS_DEFAULT, out_max_val=False, topk=1,
                     kernel_name="arg_max_with_kd"):
    """
    select format dynamically
    """
    ori_format = x.get("ori_format")
    ori_shape = x.get("ori_shape")

    # set out_max_index, according to the caffe logic
    out_max_index = True
    if out_max_val and axis != AXIS_DEFAULT:
        out_max_index = False

    # check topk
    if axis != AXIS_DEFAULT:
        # make axis in -len(shape_x) <= axis < len(shape_x)
        axis = util.axis_check(len(ori_shape), axis)

    # for 5hd, axis is only valid for n,h,w
    if ((ori_format == "NHWC" and axis != 3) or (ori_format == "NCHW" and axis != 1)) and \
            len(ori_shape) == 4 and axis != AXIS_DEFAULT:
        # NC1HWC0+ND
        if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") not in \
                ("Ascend310", "Ascend910", "Hi3796CV300ES", "Hi3796CV300CS"):
            # fp16/fp32
            input0 = gen_param(classify="input0", name="x",
                               datatype="float16,float,float16,float",
                               format="NC1HWC0,NC1HWC0,ND,ND")
            if out_max_index:
                output0 = gen_param(classify="output0", name="indices",
                                    datatype="int32,int32,int32,int32",
                                    format="NC1HWC0,NC1HWC0,ND,ND")
            else:
                output0 = gen_param(classify="output0", name="indices",
                                    datatype="float16,float,float16,float",
                                    format="NC1HWC0,NC1HWC0,ND,ND")
            output1 = gen_param(classify="output1", name="values",
                                datatype="float16,float,float16,float",
                                format="NC1HWC0,NC1HWC0,ND,ND")
        else:
            # fp16
            input0 = gen_param(classify="input0", name="x",
                               datatype="float16,float16",
                               format="NC1HWC0,ND")
            if out_max_index:
                output0 = gen_param(classify="output0", name="indices",
                                    datatype="int32,int32",
                                    format="NC1HWC0,ND")
            else:
                output0 = gen_param(classify="output0", name="indices",
                                    datatype="float16,float16",
                                    format="NC1HWC0,ND")
            output1 = gen_param(classify="output1", name="values",
                                datatype="float16,float16",
                                format="NC1HWC0,ND")
    else:
        # ND
        if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") not in \
                ("Ascend310", "Ascend910", "Hi3796CV300ES", "Hi3796CV300CS"):
            # fp16/fp32
            input0 = gen_param(classify="input0", name="x",
                               datatype="float16,float",
                               format="ND,ND")
            if out_max_index:
                output0 = gen_param(classify="output0", name="indices",
                                    datatype="int32,int32",
                                    format="ND,ND")
            else:
                output0 = gen_param(classify="output0", name="indices",
                                    datatype="float16,float",
                                    format="ND,ND")
            output1 = gen_param(classify="output1", name="values",
                                datatype="float16,float",
                                format="ND,ND")
        else:
            # fp16
            input0 = gen_param(classify="input0", name="x",
                               datatype="float16",
                               format="ND")
            if out_max_index:
                output0 = gen_param(classify="output0", name="indices",
                                    datatype="int32",
                                    format="ND")
            else:
                output0 = gen_param(classify="output0", name="indices",
                                    datatype="float16",
                                    format="ND")
            output1 = gen_param(classify="output1", name="values",
                                datatype="float16",
                                format="ND")

    param_list = [input0, output0, output1]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-branches
# pylint: disable=too-many-statements
@util.check_input_type(dict, dict, dict, int, bool, int, str)
def arg_max_with_kd(x, indices, values, axis=AXIS_DEFAULT, out_max_val=False, topk=1,
                    kernel_name="arg_max_with_kd"):
    """
    Generate arg_max_with_kd operator use arg_max_with_kd

    Parameters
    ----------
    x: dict
        data of input.
        source data type, support "float16", "float32"
    index: dict
        index of output.
    value: dict
        value of output.
    axis: int
        the axis value for reverse
    kernel_name: str
        kernel name, default value is "reverse_ext2"

    Returns
    -------
    tik_instance
    """
    shape_x = x.get("shape")
    dtype_x = x.get("dtype").lower()

    # check for 5HD
    input_format = x.get("format")
    shape_x_ori = x.get("ori_shape")
    ori_format = x.get("ori_format")
    if input_format == "NC1HWC0":
        length_x_ori = len(shape_x_ori)
        length_x = len(shape_x)

        if ori_format not in ("NCHW", "NHWC"):
            raise RuntimeError("x's ori_format is invalid for 5D Tensor")
        if length_x != 5:
            raise RuntimeError("x's shape is invalid for 5D Tensor")
        if length_x_ori != 4:
            raise RuntimeError("x's ori_shape is invalid for 5D Tensor")
        # only N,H,W axis is valid for 5hd
        if axis != AXIS_DEFAULT:
            axis = util.axis_check(length_x_ori, axis)
            axis = util.axis_transfrom_5d(axis, ori_format)
            if axis in (1, 4):
                raise RuntimeError("axis is invalid for 5D Tensor")
        else:
            raise RuntimeError("axis is invalid for 5D Tensor")

    # the element size of axis
    axis_size = 1

    # check topk
    if axis == AXIS_DEFAULT:
        # check shape len
        util.check_shape_rule(shape_x, min_dim=2, max_dim=8)

        for dim in shape_x[1:]:
            axis_size *= dim
    else:
        # check shape len
        util.check_shape_rule(shape_x, min_dim=1, max_dim=8)

        # check axis
        # constraints: -len(shape_x) <= axis < len(shape_x)
        axis = util.axis_check(len(shape_x), axis)

        axis_size = shape_x[axis]

    if topk != 1 or topk > axis_size:
        raise RuntimeError("topk is out of range")

    # check axis size
    if axis_size > AXIS_SIZE_MAX:
        raise RuntimeError("axis_size is larger than int limit, fail")

    util.check_tensor_shape_size(shape_x)
    util.check_dtype_rule(dtype_x, ("float16", "float32"))
    util.check_kernel_name(kernel_name)

    # set out_max_index, according to the caffe logic
    out_max_index = True
    if out_max_val and axis != AXIS_DEFAULT:
        out_max_index = False

    max_index = ArgMax(shape_x, dtype_x, axis, out_max_val, out_max_index, topk, kernel_name)
    return max_index.argmax_compute()


# pylint: disable=too-many-public-methods
class ArgMax():
    """
       Function: use to implement arg_max_with_kd functionality
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, shape_x, dtype_x, axis, out_max_val, out_max_index, topk, kernel_name):
        """
        init parameters

        Parameters
        ----------
        shape_x: list
            shape of input x
        dtype_x: str
            dtype_x of input x
        axis: int
            process axis
        kernel_name: str
            kernel_name

        Returns
        -------
        None
        """
        self.shape_x = list(shape_x)
        self.dtype_x = dtype_x
        self.axis = axis
        self.out_max_val = out_max_val
        self.out_max_index = out_max_index
        self.topk = topk
        self.kernel_name = kernel_name

        self.tik_instance = None
        self.product_core_num = 0
        self.set_tik_product()

        # result indicies in gm
        self.result_gm = None
        # result values in gm
        self.result_gm_value = None

        # the bytes each element takes
        dtype_bytes_size = 2 if dtype_x == "float16" else 4
        # the elements each block(32 bytes) takes
        self.data_each_block = 32 // dtype_bytes_size
        # the elements each vector repeat(256 bytes) takes
        self.data_each_vector = self.data_each_block * 8
        # the index elements each block(32 bytes) takes
        self.index_each_block = 8
        # the index elements each repeat(256 bytes) takes
        self.index_each_vector = self.index_each_block * 8
        # [0, axis) shapes multiply
        self.first_dim_size = 1
        # (axis, -1] shapes multiply
        self.last_dim_size = 1
        # the shape of axis dim
        self.axis_size = 1
        # the elements in result index/value tensor
        self.gm_result_size = 0

        # the segment len to split works in a task
        self.task_segment_len = CORE_SEGMENT_LEN
        # the segment len to split the last dim in a task
        self.task_segment_len_last_dim = DATA_EACH_VNCHWCONV

        if axis < len(self.shape_x) - 1:
            # not last axis
            for dim in self.shape_x[:axis]:
                self.first_dim_size *= dim
            self.axis_size = self.shape_x[axis]
            for dim in self.shape_x[axis + 1:]:
                self.last_dim_size *= dim
            # there only handles top1, so just multiply first and last
            self.gm_result_size = self.first_dim_size * self.topk * self.last_dim_size
        elif axis == AXIS_DEFAULT:
            # the [1,-1] axis, can be convert to the last axis scenario
            self.first_dim_size = self.shape_x[0]
            for dim in self.shape_x[1:]:
                self.axis_size *= dim
            self.gm_result_size = self.first_dim_size * self.topk
        else:
            # last axis
            for dim in self.shape_x[:-1]:
                self.first_dim_size *= dim
            self.axis_size = self.shape_x[axis]
            self.gm_result_size = self.first_dim_size * self.topk

        # the tensor for input data in gm
        shape_size = self.first_dim_size * self.axis_size * self.last_dim_size
        self.data_gm = self.tik_instance.Tensor(
            self.dtype_x,
            (_get_round_ceil_int(shape_size, self.data_each_block),),
            name="data_gm",
            scope=tik.scope_gm)

    def set_tik_product(self):
        """
        init arg_max_with_kd  parameters

        Parameters
        ----------
        None
        Returns
        -------
        tik_instance: tik_instance
        """
        self.profile = tik.Dprofile()
        self.tik_instance = tik.Tik(self.profile, True)
        #self.version = self.profile.get_product_name()
        self.version = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
        #self.product_core_num = self.profile.get_aicore_num()
        self.product_core_num = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
        #self.product_ub_size = self.profile.get_unified_buffer_size()
        self.product_ub_size = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)

        if self.version in ("Ascend310", "Ascend910", "Hi3796CV300ES", "Hi3796CV300CS") and \
                self.dtype_x == "float32":
            raise RuntimeError("Only support float16 in mini/cloud/hisi-es/hisi-cs")

    def argmax_compute(self):
        """
        argmax_compute

        Parameters
        ----------

        Returns
        -------
        result : tik_instance
            self.tik_instance
        """
        # the result index tensor in gm
        if self.out_max_index:
            self.result_gm = self.tik_instance.Tensor(
                "int32",
                (_get_round_ceil_int(self.gm_result_size, self.index_each_block),),
                name="result_gm",
                scope=tik.scope_gm)
        else:
            self.result_gm = None

        # the result value tensor in gm
        if self.out_max_val:
            self.result_gm_value = self.tik_instance.Tensor(
                self.dtype_x,
                (_get_round_ceil_int(self.gm_result_size, self.data_each_block),),
                name="result_gm_value",
                scope=tik.scope_gm)
        else:
            self.result_gm_value = None

        if self.axis_size == 1:
            # special case for axis_size == 1
            self.argmax_axis_size_one()
        elif self.check_large_task_case():
            # for large task case, two conditions required:
            # 1. self.first_dim_size is large enough
            # 2. self.axis_size * self.last_dim_size is small enough to transpose
            self.argmax_large_task_case()
        else:
            if self.axis < len(self.shape_x) - 1:
                if self.last_dim_size == 1:
                    # last axis
                    self.argmax_last_axis()
                else:
                    # not last axis
                    self.argmax_not_last_axis()
            elif self.axis == AXIS_DEFAULT:
                # the [1,-1] axis, can be convert to the last axis scenario
                self.argmax_last_axis()
            else:
                # last axis
                self.argmax_last_axis()

        inputs = [self.data_gm]
        outputs = []
        if self.result_gm is not None:
            outputs.append(self.result_gm)
        if self.result_gm_value is not None:
            outputs.append(self.result_gm_value)

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=inputs,
            outputs=outputs)

        return self.tik_instance

    def check_large_task_case(self):
        """
        large task case

        Parameters
        ----------

        Returns
        -------
        Bool
        """
        result = False
        if self.first_dim_size >= LARGE_TASK_NUM_PER_CORE * self.product_core_num:
            # the segment len to split works in a task
            if self.dtype_x == "float16":
                ub_capacity = 1024 * 2
            else:
                ub_capacity = 1024

            tmp_size = self.axis_size * self.last_dim_size
            if tmp_size % DATA_EACH_VNCHWCONV == 0:
                if self.last_dim_size == 1 and tmp_size <= ub_capacity:
                    result = True
                elif self.last_dim_size % DATA_EACH_VNCHWCONV == 0 and tmp_size <= ub_capacity:
                    result = True
                elif DATA_EACH_VNCHWCONV % self.last_dim_size == 0 and \
                        tmp_size * DATA_EACH_VNCHWCONV // self.last_dim_size <= ub_capacity:
                    result = True
                elif tmp_size * DATA_EACH_VNCHWCONV <= ub_capacity:
                    result = True
            elif DATA_EACH_VNCHWCONV % self.last_dim_size == 0 and \
                    tmp_size * DATA_EACH_VNCHWCONV // self.last_dim_size <= ub_capacity:
                result = True
            elif tmp_size * DATA_EACH_VNCHWCONV <= ub_capacity:
                result = True

        return result

    def argmax_axis_size_one(self):
        """
        special case for axis_size == 1

        Parameters
        ----------

        Returns
        -------
        None
        """
        core_number = self.product_core_num
        # the result size less than a block, only one core to avoid result override between cores
        block_factor = self.data_each_block if self.out_max_val else self.index_each_block
        if self.gm_result_size <= block_factor:
            core_number = 1
        core_number_all = self.gm_result_size
        # the segments each core assigned
        core_segment = _get_div_ceil_int(core_number_all, core_number)
        # the elements each core handles, each core at least handle a result block
        core_segment = _get_round_ceil_int(core_segment, block_factor)
        # the core count used
        core_num_used = _get_div_ceil_int(core_number_all, core_segment)
        # the elements the last core handle
        core_segment_tail = core_number_all % core_segment

        # use block_num to parallel handling
        argmax_func = self.compute_argmax_axis_size_one
        with self.tik_instance.for_range(
                0, core_num_used, block_num=core_num_used) as n_i:
            if core_segment_tail == 0:
                argmax_func(n_i, core_segment, core_segment)
            else:
                with self.tik_instance.if_scope(n_i < (core_num_used - 1)):
                    argmax_func(n_i, core_segment, core_segment)
                # the last core handle less elements than other cores
                with self.tik_instance.else_scope():
                    argmax_func(n_i, core_segment_tail, core_segment)

    def compute_argmax_axis_size_one(self, n_i, core_segment, segment_core):
        """
        special case for axis_size == 1

        Parameters
        ----------

        Returns
        -------
        None
        """
        if self.out_max_val:
            if self.dtype_x == "float16":
                ub_size = 1024 * 32
            else:
                ub_size = 1024 * 16

            start_offset = n_i * segment_core
            loops = _get_div_ceil_int(core_segment, ub_size)
            # the elements the last core handle
            tail = core_segment % ub_size
            if tail == 0:
                tail = ub_size
            if loops != 0:
                if loops > 1:
                    thread_num = 2
                else:
                    thread_num = 1
                with self.tik_instance.for_range(0, loops, thread_num=thread_num) as loop:
                    # the original input data
                    ub_data = self.tik_instance.Tensor(
                        self.dtype_x, (ub_size,),
                        name="ub_data",
                        scope=tik.scope_ubuf)

                    with self.tik_instance.if_scope(loop < (loops - 1)):
                        # data move from gm to ub
                        self.tik_instance.data_move(
                            ub_data, self.data_gm[start_offset + loop * ub_size], 0, 1,
                            _get_div_ceil_int(ub_size, self.data_each_block), 0, 0)

                        # data move from ub to gm
                        self.tik_instance.data_move(
                            self.result_gm_value[start_offset + loop * ub_size], ub_data, 0, 1,
                            _get_div_ceil_int(ub_size, self.data_each_block), 0, 0)
                    with self.tik_instance.else_scope():
                        # data move from gm to ub
                        self.tik_instance.data_move(
                            ub_data, self.data_gm[start_offset + loop * ub_size], 0, 1,
                            _get_div_ceil_int(tail, self.data_each_block), 0, 0)

                        # data move from ub to gm
                        self.tik_instance.data_move(
                            self.result_gm_value[start_offset + loop * ub_size], ub_data, 0, 1,
                            _get_div_ceil_int(tail, self.data_each_block), 0, 0)

        if self.out_max_index:
            ub_size = 1024 * 32
            # the indices data
            ub_data = self.tik_instance.Tensor(
                "uint32", (ub_size,),
                name="ub_data",
                scope=tik.scope_ubuf)

            start_offset = 0
            repeat_times = _get_div_ceil_int(ub_size, self.index_each_vector)
            while repeat_times > 0:
                repeat = min(repeat_times, 255)
                self.tik_instance.vec_dup(self.index_each_vector, ub_data[start_offset],
                                          0, repeat, 8)
                start_offset += repeat * self.index_each_vector
                repeat_times -= repeat

            start_offset = n_i * segment_core
            loops = core_segment // ub_size
            # the elements the last core handle
            tail = core_segment % ub_size
            if loops != 0:
                with self.tik_instance.for_range(0, loops) as loop:
                    # data move from ub to gm
                    self.tik_instance.data_move(
                        self.result_gm[start_offset + loop * ub_size], ub_data, 0, 1,
                        _get_div_ceil_int(ub_size, self.index_each_block), 0, 0)
            if tail != 0:
                # data move from ub to gm
                self.tik_instance.data_move(
                    self.result_gm[start_offset + loops * ub_size], ub_data, 0, 1,
                    _get_div_ceil_int(tail, self.index_each_block), 0, 0)

    def argmax_large_task_case(self):
        """
        scedule for argmax_large_task_case

        Parameters
        ----------

        Returns
        -------
        None
        """
        core_number = self.product_core_num
        # the result size less than a block, only one core to avoid result override between cores
        block_factor = self.data_each_block if self.out_max_val else self.index_each_block
        if self.gm_result_size <= block_factor:
            core_number = 1
        core_number_all = self.first_dim_size
        # the segments each core assigned
        core_segment = _get_div_ceil_int(core_number_all, core_number)
        # the elements each core handles, each core at least handle a result block
        factor = self.topk * self.last_dim_size
        if core_segment * factor < block_factor:
            core_segment = _get_round_ceil_int(core_segment * factor, block_factor)
            core_segment = _get_div_ceil_int(core_segment, factor)
        # the core count used
        core_num_used = _get_div_ceil_int(core_number_all, core_segment)
        # the elements the last core handle
        core_segment_tail = core_number_all % core_segment

        self.task_segment_len = 16
        if self.dtype_x == "float16":
            self.task_segment_len_last_dim = 1024 * 16
        else:
            self.task_segment_len_last_dim = 1024 * 8
        if self.out_max_index and self.out_max_val:
            self.task_segment_len_last_dim = self.task_segment_len_last_dim // 4
        elif self.out_max_index:
            self.task_segment_len_last_dim = self.task_segment_len_last_dim // 2

        # use vec_max to get the top_1 result
        # use block_num to parallel handling
        argmax_func = self.compute_argmax_large_task_case
        with self.tik_instance.for_range(
                0, core_num_used, block_num=core_num_used) as n_i:
            if core_segment_tail == 0:
                argmax_func(n_i, core_segment, core_segment)
            else:
                with self.tik_instance.if_scope(n_i < (core_num_used - 1)):
                    argmax_func(n_i, core_segment, core_segment)
                # the last core handle less elements than other cores
                with self.tik_instance.else_scope():
                    argmax_func(n_i, core_segment_tail, core_segment)

    def argmax_not_last_axis(self):
        """
        scedule for argmax_not_last_axis

        Parameters
        ----------

        Returns
        -------
        None
        """
        core_number = self.product_core_num
        # the result size less than a block, only one core to avoid result override between cores
        block_factor = self.data_each_block if self.out_max_val else self.index_each_block
        if self.gm_result_size <= block_factor:
            core_number = 1
        core_number_all = self.first_dim_size
        # the segments each core assigned
        core_segment = _get_div_ceil_int(core_number_all, core_number)
        # the elements each core handles, each core at least handle a result block
        factor = self.topk * self.last_dim_size
        if core_segment * factor < block_factor:
            core_segment = _get_round_ceil_int(core_segment * factor, block_factor)
            core_segment = _get_div_ceil_int(core_segment, factor)
        # the core count used
        core_num_used = _get_div_ceil_int(core_number_all, core_segment)
        # the elements the last core handle
        core_segment_tail = core_number_all % core_segment

        # vec_max branch selection
        use_vmax_branch = False
        if self.axis_size <= 64 and self.last_dim_size >= 256:
            use_vmax_branch = True

        if use_vmax_branch:
            self.task_segment_len = 16
            if self.dtype_x == "float16":
                self.task_segment_len_last_dim = 1024 * 32
                if self.out_max_index:
                    self.task_segment_len_last_dim = self.task_segment_len_last_dim // 2
            else:
                self.task_segment_len_last_dim = 1024 * 16
                if self.out_max_index:
                    self.task_segment_len_last_dim = self.task_segment_len_last_dim // 2

            # use vec_max to get the top_1 result
            # use block_num to parallel handling
            argmax_func = self.compute_argmax_not_last_axis_using_vmax
            with self.tik_instance.for_range(
                    0, core_num_used, block_num=core_num_used) as n_i:
                if core_segment_tail == 0:
                    argmax_func(n_i, core_segment, core_segment)
                else:
                    with self.tik_instance.if_scope(n_i < (core_num_used - 1)):
                        argmax_func(n_i, core_segment, core_segment)
                    # the last core handle less elements than other cores
                    with self.tik_instance.else_scope():
                        argmax_func(n_i, core_segment_tail, core_segment)
        else:
            # the segment len to split works in a task
            if self.dtype_x == "float16":
                self.task_segment_len = 1024 * 2
            else:
                self.task_segment_len = 1024

            # the segment len to split the last dim in a task
            self.task_segment_len_last_dim = DATA_EACH_VNCHWCONV

            # adjust the self.task_segment_len_last_dim appropriately
            if self.last_dim_size > 32:
                self.task_segment_len = self.task_segment_len // 4
                self.task_segment_len_last_dim = self.task_segment_len_last_dim * 4
            elif self.last_dim_size > 16:
                self.task_segment_len = self.task_segment_len // 2
                self.task_segment_len_last_dim = self.task_segment_len_last_dim * 2

            # use vcmax to get the top_1 result
            # use block_num to parallel handling
            argmax_func = self.compute_argmax_not_last_axis
            with self.tik_instance.for_range(
                    0, core_num_used, block_num=core_num_used) as n_i:
                if core_segment_tail == 0:
                    argmax_func(n_i, core_segment, core_segment)
                else:
                    with self.tik_instance.if_scope(n_i < (core_num_used - 1)):
                        argmax_func(n_i, core_segment, core_segment)
                    # the last core handle less elements than other cores
                    with self.tik_instance.else_scope():
                        argmax_func(n_i, core_segment_tail, core_segment)

    # each core handle several segments
    def compute_argmax_large_task_case(self, n_i, core_segment, segment_core):
        """
        compute arg when do last axis

        Parameters
        ----------
        n_i : int
            the first loop index
        core_segment : int
            segment process len
        segment_core : int
            the total segment index

        Returns
        -------
        None
        """
        # the segment len to split works in a task
        if self.dtype_x == "float16":
            ub_capacity = 1024 * 2
        else:
            ub_capacity = 1024

        tmp_size = self.axis_size * self.last_dim_size
        second_dim = 1
        if tmp_size % DATA_EACH_VNCHWCONV == 0:
            if self.last_dim_size == 1 and tmp_size <= ub_capacity:
                second_dim = 1
            elif self.last_dim_size % DATA_EACH_VNCHWCONV == 0 and tmp_size <= ub_capacity:
                second_dim = 1
            elif DATA_EACH_VNCHWCONV % self.last_dim_size == 0 and \
                    tmp_size * DATA_EACH_VNCHWCONV // self.last_dim_size <= ub_capacity:
                second_dim = DATA_EACH_VNCHWCONV // self.last_dim_size
            elif tmp_size * DATA_EACH_VNCHWCONV <= ub_capacity:
                second_dim = DATA_EACH_VNCHWCONV
        elif DATA_EACH_VNCHWCONV % self.last_dim_size == 0 and \
                tmp_size * DATA_EACH_VNCHWCONV // self.last_dim_size <= ub_capacity:
            second_dim = DATA_EACH_VNCHWCONV // self.last_dim_size
        elif tmp_size * DATA_EACH_VNCHWCONV <= ub_capacity:
            second_dim = DATA_EACH_VNCHWCONV
        first_dim = ub_capacity // tmp_size // second_dim * DATA_EACH_VNCHWCONV

        CORE_SEGMENT_LEN_LOCAL = first_dim * second_dim

        # the number of elements this core to write, for 32B alignment purpose
        element_end_address_to_write = (n_i * segment_core + core_segment) * \
                                       self.topk * self.last_dim_size
        elements_to_write = core_segment * self.topk * self.last_dim_size

        def _run(segment_len, segment_index):
            first_dim_local = _get_round_ceil_int(_get_div_ceil_int(segment_len, second_dim),
                                                  DATA_EACH_VNCHWCONV)

            task_segment_len_last_dim = self.task_segment_len_last_dim
            if self.dtype_x == "float16" and second_dim * self.last_dim_size > 1 and \
                    self.last_dim_size * first_dim_local > task_segment_len_last_dim:
                # the range of first_dim is [16, 2K]
                # the range of task_segment_len_last_dim is [4k/8K/16K,]
                # make task_segment_len_last_dim integer multiple of first_dim_local
                task_segment_len_last_dim = \
                    _get_round_floor_int(task_segment_len_last_dim, first_dim_local)

            ub_buf_size, loop_times, over_size, align_flag = \
                get_tiling_info_for_axis(self.last_dim_size * first_dim_local,
                                         task_segment_len_last_dim)

            # the schema is: first_dim_local, second_dim, self.axis_size, self.last_dim_size
            index = CORE_SEGMENT_LEN_LOCAL * segment_index
            offset = n_i * segment_core + index

            # 1. transpose
            # before transpose: [first_dim_local, second_dim * tmp_size]
            # after transpose: [second_dim * tmp_size, first_dim_local]
            dim_before_1 = first_dim_local
            dim_before_2 = second_dim * tmp_size
            # use vec_trans_scatter for transpose
            ub_data_transpose = self.tik_instance.Tensor(
                self.dtype_x, (dim_before_1 * dim_before_2,),
                name="ub_data_transpose",
                scope=tik.scope_ubuf)

            with self.tik_instance.new_stmt_scope():
                # the original input data
                ub_data = self.tik_instance.Tensor(
                    self.dtype_x, (dim_before_1 * dim_before_2,),
                    name="ub_data",
                    scope=tik.scope_ubuf)

                # the last dim is small, move all continuous data in batches
                # the offset in the original input data
                gm_offset = offset * self.axis_size * self.last_dim_size
                # data move from gm to ub
                self.tik_instance.data_move(
                    ub_data, self.data_gm[gm_offset], 0, 1,
                    _get_div_ceil_int(self.axis_size * self.last_dim_size * segment_len,
                                      self.data_each_block),
                    0, 0)

                # first do the transpose, src is ub_data_transpose, dest is ub_data
                row_loop_count = _get_div_ceil_int(dim_before_1, DATA_EACH_VNCHWCONV)
                col_loop_count = _get_div_ceil_int(dim_before_2, DATA_EACH_VNCHWCONV)
                if col_loop_count > row_loop_count:
                    if self.dtype_x == "float16":
                        # float16, 16*16
                        with self.tik_instance.for_range(0, row_loop_count) as row_idx:
                            # should of length 16, but the elements can be duplicate,
                            # the region of each elements at most not overlap
                            dst_list = []
                            for i in range(DATA_EACH_VNCHWCONV):
                                dst_pos = dim_before_1 * i + DATA_EACH_VNCHWCONV * row_idx
                                dst_list.append(ub_data_transpose[dst_pos])
                            # should of length 16, but the elements can be duplicate
                            src_list = []
                            for i in range(DATA_EACH_VNCHWCONV):
                                src_pos = dim_before_2 * (DATA_EACH_VNCHWCONV * row_idx + i)
                                src_list.append(ub_data[src_pos])

                            if col_loop_count == 1:
                                self.tik_instance.vec_trans_scatter(False, False,
                                                                    dst_list, src_list,
                                                                    col_loop_count, 0, 0)
                            else:
                                self.tik_instance.vec_trans_scatter(
                                    False, False, dst_list, src_list, col_loop_count,
                                    DATA_EACH_VNCHWCONV * dim_before_1 // self.data_each_block,
                                    DATA_EACH_VNCHWCONV // self.data_each_block)
                    else:
                        # float32, 16*8
                        with self.tik_instance.for_range(0, row_loop_count) as row_idx:
                            # should of length 16, but the elements can be duplicate,
                            # the region of each elements at most not overlap
                            dst_list = []
                            for i in range(DATA_EACH_VNCHWCONV):
                                dst_pos = dim_before_1 * (i // 2) + \
                                          (i % 2) * DATA_EACH_VNCHWCONV // 2 + \
                                          DATA_EACH_VNCHWCONV * row_idx
                                dst_list.append(ub_data_transpose[dst_pos])
                            # should of length 16, but the elements can be duplicate
                            src_list = []
                            for i in range(DATA_EACH_VNCHWCONV):
                                src_pos = dim_before_2 * (DATA_EACH_VNCHWCONV * row_idx + i)
                                src_list.append(ub_data[src_pos])

                            self.tik_instance.vec_trans_scatter(
                                False, False, dst_list, src_list, 2 * col_loop_count,
                                DATA_EACH_VNCHWCONV // 2 * dim_before_1 // self.data_each_block,
                                DATA_EACH_VNCHWCONV // 2 // self.data_each_block)
                else:
                    if self.dtype_x == "float16":
                        # float16, 16*16
                        with self.tik_instance.for_range(0, col_loop_count) as col_idx:
                            # should of length 16, but the elements can be duplicate,
                            # the region of each elements at most not overlap
                            dst_list = []
                            for i in range(DATA_EACH_VNCHWCONV):
                                dst_pos = dim_before_1 * (DATA_EACH_VNCHWCONV * col_idx + i)
                                dst_list.append(ub_data_transpose[dst_pos])
                            # should of length 16, but the elements can be duplicate
                            src_list = []
                            for i in range(DATA_EACH_VNCHWCONV):
                                src_pos = dim_before_2 * i + DATA_EACH_VNCHWCONV * col_idx
                                src_list.append(ub_data[src_pos])

                            if row_loop_count == 1:
                                self.tik_instance.vec_trans_scatter(False, False,
                                                                    dst_list, src_list,
                                                                    row_loop_count, 0, 0)
                            else:
                                self.tik_instance.vec_trans_scatter(
                                    False, False, dst_list, src_list, row_loop_count,
                                    DATA_EACH_VNCHWCONV // self.data_each_block,
                                    DATA_EACH_VNCHWCONV * dim_before_2 // self.data_each_block)
                    else:
                        # float32, 16*8
                        with self.tik_instance.for_range(0, 2 * col_loop_count) as col_idx:
                            # should of length 16, but the elements can be duplicate,
                            # the region of each elements at most not overlap
                            dst_list = []
                            for i in range(DATA_EACH_VNCHWCONV):
                                dst_pos = dim_before_1 * \
                                          (DATA_EACH_VNCHWCONV // 2 * col_idx + i // 2) + \
                                          (i % 2) * DATA_EACH_VNCHWCONV // 2
                                dst_list.append(ub_data_transpose[dst_pos])
                            # should of length 16, but the elements can be duplicate
                            src_list = []
                            for i in range(DATA_EACH_VNCHWCONV):
                                src_pos = dim_before_2 * i + DATA_EACH_VNCHWCONV // 2 * col_idx
                                src_list.append(ub_data[src_pos])

                            if row_loop_count == 1:
                                self.tik_instance.vec_trans_scatter(False, False,
                                                                    dst_list, src_list,
                                                                    row_loop_count, 0, 0)
                            else:
                                self.tik_instance.vec_trans_scatter(
                                    False, False, dst_list, src_list, row_loop_count,
                                    DATA_EACH_VNCHWCONV // self.data_each_block,
                                    DATA_EACH_VNCHWCONV * dim_before_2 // self.data_each_block)

            # 2. vec_max
            src_row_len = self.axis_size * self.last_dim_size * first_dim_local
            dst_row_len = self.last_dim_size * first_dim_local

            if self.out_max_val:
                ub_data_values = self.tik_instance.Tensor(
                    self.dtype_x, (second_dim * dst_row_len,),
                    name="ub_data_values",
                    scope=tik.scope_ubuf)
            else:
                ub_data_values = None

            if self.out_max_index:
                ub_data_indices_int32 = self.tik_instance.Tensor(
                    "int32", (second_dim * dst_row_len,),
                    name="ub_data_indices_int32",
                    scope=tik.scope_ubuf)

                if self.dtype_x == "float16" and second_dim * self.last_dim_size > 1:
                    start_offset = 0
                    repeat_times = second_dim * dst_row_len // self.index_each_vector
                    remaining = second_dim * dst_row_len % self.index_each_vector
                    while repeat_times > 0:
                        repeat = min(repeat_times, 255)
                        self.tik_instance.vec_dup(self.index_each_vector,
                                                  ub_data_indices_int32[start_offset],
                                                  0, repeat, 8)
                        start_offset += repeat * self.index_each_vector
                        repeat_times -= repeat
                    if remaining > 0:
                        self.tik_instance.vec_dup(remaining, ub_data_indices_int32[start_offset],
                                                  0, 1, 8)
                    ub_data_indices = ub_data_indices_int32.reinterpret_cast_to("uint16")
                else:
                    ub_data_indices = ub_data_indices_int32
            else:
                ub_data_indices = None

            argmax_func = self.do_argmax_large_task_case
            with self.tik_instance.for_range(0, second_dim) as loop_ex:
                if ub_data_values is None:
                    val_ub = None
                else:
                    val_ub = ub_data_values[dst_row_len * loop_ex]

                if ub_data_indices is None:
                    ind_ub = None
                elif self.dtype_x == "float16" and second_dim * self.last_dim_size > 1:
                    ind_ub = ub_data_indices[dst_row_len * loop_ex * 2]
                else:
                    ind_ub = ub_data_indices[dst_row_len * loop_ex]

                if loop_times != 0:
                    with self.tik_instance.for_range(0, loop_times) as loop_in:
                        argmax_func(first_dim_local, second_dim,
                                    ub_buf_size, loop_in, task_segment_len_last_dim,
                                    ub_data_transpose[src_row_len * loop_ex], val_ub, ind_ub)
                if align_flag:
                    with self.tik_instance.new_stmt_scope():
                        argmax_func(first_dim_local, second_dim,
                                    over_size, loop_times, task_segment_len_last_dim,
                                    ub_data_transpose[src_row_len * loop_ex], val_ub, ind_ub)

            # 3. transpose
            # vec_max result is in ub_data
            # before transpose: [second_dim * self.last_dim_size, first_dim_local]
            # after transpose: [first_dim_local, second_dim * self.last_dim_size]
            dim_before_1 = second_dim * self.last_dim_size
            dim_before_2 = first_dim_local

            # data move to gm
            gm_offset = offset * self.topk * self.last_dim_size
            result_size = self.topk * self.last_dim_size * segment_len

            # store back the max value to gm
            if self.out_max_val:
                if dim_before_1 == 1:
                    # no need to transpose
                    values_result = ub_data_values
                else:
                    # first do the transpose, src is ub_data_transpose, dest is ub_data
                    row_loop_count = _get_div_ceil_int(dim_before_1, DATA_EACH_VNCHWCONV)
                    col_loop_count = _get_div_ceil_int(dim_before_2, DATA_EACH_VNCHWCONV)
                    if col_loop_count > row_loop_count:
                        if self.dtype_x == "float16":
                            # float16, 16*16
                            with self.tik_instance.for_range(0, row_loop_count) as row_idx:
                                # should of length 16, but the elements can be duplicate,
                                # the region of each elements at most not overlap
                                dst_list = []
                                for i in range(DATA_EACH_VNCHWCONV):
                                    dst_pos = dim_before_1 * i + DATA_EACH_VNCHWCONV * row_idx
                                    dst_list.append(ub_data_transpose[dst_pos])
                                # should of length 16, but the elements can be duplicate
                                src_list = []
                                for i in range(DATA_EACH_VNCHWCONV):
                                    src_pos = dim_before_2 * (DATA_EACH_VNCHWCONV * row_idx + i)
                                    src_list.append(ub_data_values[src_pos])

                                if col_loop_count == 1:
                                    self.tik_instance.vec_trans_scatter(False, False,
                                                                        dst_list, src_list,
                                                                        col_loop_count, 0, 0)
                                else:
                                    self.tik_instance.vec_trans_scatter(
                                        False, False, dst_list, src_list, col_loop_count,
                                        DATA_EACH_VNCHWCONV * dim_before_1 // self.data_each_block,
                                        DATA_EACH_VNCHWCONV // self.data_each_block)
                        else:
                            # float32, 16*8
                            with self.tik_instance.for_range(0, row_loop_count) as row_idx:
                                # should of length 16, but the elements can be duplicate,
                                # the region of each elements at most not overlap
                                dst_list = []
                                for i in range(DATA_EACH_VNCHWCONV):
                                    dst_pos = dim_before_1 * (i // 2) + \
                                              (i % 2) * DATA_EACH_VNCHWCONV // 2 + \
                                              DATA_EACH_VNCHWCONV * row_idx
                                    dst_list.append(ub_data_transpose[dst_pos])
                                # should of length 16, but the elements can be duplicate
                                src_list = []
                                for i in range(DATA_EACH_VNCHWCONV):
                                    src_pos = dim_before_2 * (DATA_EACH_VNCHWCONV * row_idx + i)
                                    src_list.append(ub_data_values[src_pos])

                                self.tik_instance.vec_trans_scatter(
                                    False, False, dst_list, src_list, 2 * col_loop_count,
                                    DATA_EACH_VNCHWCONV // 2 * dim_before_1 // self.data_each_block,
                                    DATA_EACH_VNCHWCONV // 2 // self.data_each_block)
                    else:
                        if self.dtype_x == "float16":
                            # float16, 16*16
                            with self.tik_instance.for_range(0, col_loop_count) as col_idx:
                                # should of length 16, but the elements can be duplicate,
                                # the region of each elements at most not overlap
                                dst_list = []
                                for i in range(DATA_EACH_VNCHWCONV):
                                    dst_pos = dim_before_1 * (DATA_EACH_VNCHWCONV * col_idx + i)
                                    dst_list.append(ub_data_transpose[dst_pos])
                                # should of length 16, but the elements can be duplicate
                                src_list = []
                                for i in range(DATA_EACH_VNCHWCONV):
                                    src_pos = dim_before_2 * i + DATA_EACH_VNCHWCONV * col_idx
                                    src_list.append(ub_data_values[src_pos])

                                if row_loop_count == 1:
                                    self.tik_instance.vec_trans_scatter(False, False,
                                                                        dst_list, src_list,
                                                                        row_loop_count, 0, 0)
                                else:
                                    self.tik_instance.vec_trans_scatter(
                                        False, False, dst_list, src_list, row_loop_count,
                                        DATA_EACH_VNCHWCONV // self.data_each_block,
                                        DATA_EACH_VNCHWCONV * dim_before_2 // self.data_each_block)
                        else:
                            # float32, 16*8
                            with self.tik_instance.for_range(0, 2 * col_loop_count) as col_idx:
                                # should of length 16, but the elements can be duplicate,
                                # the region of each elements at most not overlap
                                dst_list = []
                                for i in range(DATA_EACH_VNCHWCONV):
                                    dst_pos = dim_before_1 * \
                                              (DATA_EACH_VNCHWCONV // 2 * col_idx + i // 2) + \
                                              (i % 2) * DATA_EACH_VNCHWCONV // 2
                                    dst_list.append(ub_data_transpose[dst_pos])
                                # should of length 16, but the elements can be duplicate
                                src_list = []
                                for i in range(DATA_EACH_VNCHWCONV):
                                    src_pos = dim_before_2 * i + DATA_EACH_VNCHWCONV // 2 * col_idx
                                    src_list.append(ub_data_values[src_pos])

                                if row_loop_count == 1:
                                    self.tik_instance.vec_trans_scatter(False, False,
                                                                        dst_list, src_list,
                                                                        row_loop_count, 0, 0)
                                else:
                                    self.tik_instance.vec_trans_scatter(
                                        False, False, dst_list, src_list, row_loop_count,
                                        DATA_EACH_VNCHWCONV // self.data_each_block,
                                        DATA_EACH_VNCHWCONV * dim_before_2 // self.data_each_block)

                    values_result = ub_data_transpose

                if elements_to_write < self.data_each_block:
                    self.tik_instance.data_move(
                        self.result_gm_value[gm_offset], values_result, 0, 1,
                        _get_div_ceil_int(result_size, self.data_each_block), 0, 0)
                else:
                    bias = element_end_address_to_write - gm_offset - \
                           _get_round_ceil_int(result_size, self.data_each_block)
                    with self.tik_instance.if_scope(bias >= 0):
                        self.tik_instance.data_move(
                            self.result_gm_value[gm_offset], values_result, 0, 1,
                            _get_div_ceil_int(result_size, self.data_each_block), 0, 0)
                    with self.tik_instance.else_scope():
                        burst_len_floor = result_size // self.data_each_block
                        ub_elem_left = result_size % self.data_each_block
                        ub_elem_left_start = burst_len_floor * self.data_each_block
                        if burst_len_floor > 0:
                            self.tik_instance.data_move(self.result_gm_value[gm_offset],
                                                        values_result, 0, 1,
                                                        burst_len_floor, 0, 0)
                        result_out_scalar_tmp = self.tik_instance.Tensor(
                            self.dtype_x, (self.data_each_block,),
                            name="result_out_scalar_tmp",
                            scope=tik.scope_ubuf)
                        self.tik_instance.data_move(
                            result_out_scalar_tmp,
                            self.result_gm_value[element_end_address_to_write -
                                                 self.data_each_block],
                            0, 1, 1, 0, 0)
                        value_reg = self.tik_instance.Scalar(dtype=self.dtype_x)
                        with self.tik_instance.for_range(0, ub_elem_left) as i:
                            value_reg.set_as(values_result[ub_elem_left_start + i])
                            result_out_scalar_tmp[i - bias].set_as(value_reg)
                        self.tik_instance.data_move(
                            self.result_gm_value[element_end_address_to_write -
                                                 self.data_each_block],
                            result_out_scalar_tmp, 0, 1, 1, 0, 0)

            # store back the max index to gm
            if self.out_max_index:
                if dim_before_1 == 1:
                    # no need to transpose
                    indices_result = ub_data_indices
                else:
                    # first do the transpose, src is ub_data_transpose, dest is ub_data
                    if self.dtype_x == "float16":
                        dim_before_1 *= 2
                        row_loop_count = _get_div_ceil_int(dim_before_1, DATA_EACH_VNCHWCONV)
                    else:
                        row_loop_count = _get_div_ceil_int(dim_before_1, DATA_EACH_VNCHWCONV)
                    col_loop_count = _get_div_ceil_int(dim_before_2, DATA_EACH_VNCHWCONV)
                    if col_loop_count > row_loop_count:
                        if self.dtype_x == "float16":
                            ub_data_transpose = ub_data_transpose.reinterpret_cast_to("uint16")
                            # float16, 16*16
                            with self.tik_instance.for_range(0, row_loop_count) as row_idx:
                                # should of length 16, but the elements can be duplicate,
                                # the region of each elements at most not overlap
                                dst_list = []
                                for i in range(DATA_EACH_VNCHWCONV):
                                    dst_pos = dim_before_1 * i + DATA_EACH_VNCHWCONV * row_idx
                                    dst_list.append(ub_data_transpose[dst_pos])
                                # should of length 16, but the elements can be duplicate
                                src_list = []
                                for i in range(DATA_EACH_VNCHWCONV):
                                    src_pos = dim_before_2 * (DATA_EACH_VNCHWCONV * row_idx + i)
                                    src_list.append(ub_data_indices[src_pos])

                                if col_loop_count == 1:
                                    self.tik_instance.vec_trans_scatter(False, False,
                                                                        dst_list, src_list,
                                                                        col_loop_count, 0, 0)
                                else:
                                    self.tik_instance.vec_trans_scatter(
                                        False, False, dst_list, src_list, col_loop_count,
                                        DATA_EACH_VNCHWCONV * dim_before_1 // self.data_each_block,
                                        DATA_EACH_VNCHWCONV // self.data_each_block)
                        else:
                            ub_data_transpose = ub_data_transpose.reinterpret_cast_to("int32")
                            # float32, 16*8
                            with self.tik_instance.for_range(0, row_loop_count) as row_idx:
                                # should of length 16, but the elements can be duplicate,
                                # the region of each elements at most not overlap
                                dst_list = []
                                for i in range(DATA_EACH_VNCHWCONV):
                                    dst_pos = dim_before_1 * (i // 2) + \
                                              (i % 2) * DATA_EACH_VNCHWCONV // 2 + \
                                              DATA_EACH_VNCHWCONV * row_idx
                                    dst_list.append(ub_data_transpose[dst_pos])
                                # should of length 16, but the elements can be duplicate
                                src_list = []
                                for i in range(DATA_EACH_VNCHWCONV):
                                    src_pos = dim_before_2 * (DATA_EACH_VNCHWCONV * row_idx + i)
                                    src_list.append(ub_data_indices[src_pos])

                                self.tik_instance.vec_trans_scatter(
                                    False, False, dst_list, src_list, 2 * col_loop_count,
                                    DATA_EACH_VNCHWCONV // 2 * dim_before_1 // self.data_each_block,
                                    DATA_EACH_VNCHWCONV // 2 // self.data_each_block)
                    else:
                        if self.dtype_x == "float16":
                            ub_data_transpose = ub_data_transpose.reinterpret_cast_to("uint16")
                            # float16, 16*16
                            with self.tik_instance.for_range(0, col_loop_count) as col_idx:
                                # should of length 16, but the elements can be duplicate,
                                # the region of each elements at most not overlap
                                dst_list = []
                                for i in range(DATA_EACH_VNCHWCONV):
                                    dst_pos = dim_before_1 * (DATA_EACH_VNCHWCONV * col_idx + i)
                                    dst_list.append(ub_data_transpose[dst_pos])
                                # should of length 16, but the elements can be duplicate
                                src_list = []
                                for i in range(DATA_EACH_VNCHWCONV):
                                    src_pos = dim_before_2 * i + DATA_EACH_VNCHWCONV * col_idx
                                    src_list.append(ub_data_indices[src_pos])

                                if row_loop_count == 1:
                                    self.tik_instance.vec_trans_scatter(False, False,
                                                                        dst_list, src_list,
                                                                        row_loop_count, 0, 0)
                                else:
                                    self.tik_instance.vec_trans_scatter(
                                        False, False, dst_list, src_list, row_loop_count,
                                        DATA_EACH_VNCHWCONV // self.data_each_block,
                                        DATA_EACH_VNCHWCONV * dim_before_2 // self.data_each_block)
                        else:
                            ub_data_transpose = ub_data_transpose.reinterpret_cast_to("int32")
                            # float32, 16*8
                            with self.tik_instance.for_range(0, 2 * col_loop_count) as col_idx:
                                # should of length 16, but the elements can be duplicate,
                                # the region of each elements at most not overlap
                                dst_list = []
                                for i in range(DATA_EACH_VNCHWCONV):
                                    dst_pos = dim_before_1 * \
                                              (DATA_EACH_VNCHWCONV // 2 * col_idx + i // 2) + \
                                              (i % 2) * DATA_EACH_VNCHWCONV // 2
                                    dst_list.append(ub_data_transpose[dst_pos])
                                # should of length 16, but the elements can be duplicate
                                src_list = []
                                for i in range(DATA_EACH_VNCHWCONV):
                                    src_pos = dim_before_2 * i + DATA_EACH_VNCHWCONV // 2 * col_idx
                                    src_list.append(ub_data_indices[src_pos])

                                if row_loop_count == 1:
                                    self.tik_instance.vec_trans_scatter(False, False,
                                                                        dst_list, src_list,
                                                                        row_loop_count, 0, 0)
                                else:
                                    self.tik_instance.vec_trans_scatter(
                                        False, False, dst_list, src_list, row_loop_count,
                                        DATA_EACH_VNCHWCONV // self.data_each_block,
                                        DATA_EACH_VNCHWCONV * dim_before_2 // self.data_each_block)

                    indices_result = ub_data_transpose.reinterpret_cast_to("int32")

                if elements_to_write < self.index_each_block:
                    self.tik_instance.data_move(
                        self.result_gm[gm_offset], indices_result, 0, 1,
                        _get_div_ceil_int(result_size, self.index_each_block), 0, 0)
                else:
                    bias = element_end_address_to_write - gm_offset - \
                           _get_round_ceil_int(result_size, self.index_each_block)
                    with self.tik_instance.if_scope(bias >= 0):
                        self.tik_instance.data_move(
                            self.result_gm[gm_offset], indices_result, 0, 1,
                            _get_div_ceil_int(result_size, self.index_each_block), 0, 0)
                    with self.tik_instance.else_scope():
                        burst_len_floor = result_size // self.index_each_block
                        ub_elem_left = result_size % self.index_each_block
                        ub_elem_left_start = burst_len_floor * self.index_each_block
                        if burst_len_floor > 0:
                            self.tik_instance.data_move(self.result_gm[gm_offset],
                                                        indices_result, 0, 1,
                                                        burst_len_floor, 0, 0)
                        result_int32_tmp = self.tik_instance.Tensor(
                            "int32", (self.index_each_block,),
                            name="result_int32_tmp",
                            scope=tik.scope_ubuf)
                        self.tik_instance.data_move(
                            result_int32_tmp,
                            self.result_gm[element_end_address_to_write - self.index_each_block],
                            0, 1, 1, 0, 0)
                        index_reg = self.tik_instance.Scalar(dtype="int32")
                        with self.tik_instance.for_range(0, ub_elem_left) as i:
                            index_reg.set_as(indices_result[ub_elem_left_start + i])
                            result_int32_tmp[i - bias].set_as(index_reg)
                        self.tik_instance.data_move(
                            self.result_gm[element_end_address_to_write - self.index_each_block],
                            result_int32_tmp, 0, 1, 1, 0, 0)

        _loop_segment = core_segment // CORE_SEGMENT_LEN_LOCAL
        _loop_segment_tail = core_segment % CORE_SEGMENT_LEN_LOCAL
        if _loop_segment != 0:
            with self.tik_instance.for_range(0, _loop_segment) as _loop:
                _run(CORE_SEGMENT_LEN_LOCAL, _loop)
        if _loop_segment_tail != 0:
            with self.tik_instance.new_stmt_scope():
                _run(_loop_segment_tail, _loop_segment)

    # each core handle several segments
    def compute_argmax_not_last_axis_using_vmax(self, n_i, core_segment, segment_core):
        """
        compute arg when do last axis

        Parameters
        ----------
        n_i : int
            the first loop index
        core_segment : int
            segment process len
        segment_core : int
            the total segment index

        Returns
        -------
        None
        """
        # split the last dim into loops, the split step is DATA_EACH_VNCHWCONV(16) for simplicity
        ub_buf_size, loop_times, over_size, align_flag = \
            get_tiling_info_for_axis(self.last_dim_size, self.task_segment_len_last_dim)

        # the number of elements this core to write, for 32B alignment purpose
        element_end_address_to_write = (n_i * segment_core + core_segment) * \
                                       self.topk * self.last_dim_size
        elements_to_write = core_segment * self.topk * self.last_dim_size

        def _run(segment_len, segment_index):
            # more then 1 segments, enable double buffer
            with self.tik_instance.for_range(0, segment_len) as core_i:
                index = core_i + CORE_SEGMENT_LEN * segment_index
                offset = n_i * segment_core + index

                argmax_func = self.do_argmax_not_last_axis_using_vmax
                if loop_times != 0:
                    with self.tik_instance.for_range(0, loop_times) as loop:
                        argmax_func(ub_buf_size, loop, offset,
                                    loop_times, element_end_address_to_write, elements_to_write)
                if align_flag:
                    with self.tik_instance.new_stmt_scope():
                        argmax_func(over_size, loop_times, offset,
                                    loop_times, element_end_address_to_write, elements_to_write)

        _loop_segment = core_segment // CORE_SEGMENT_LEN
        _loop_segment_tail = core_segment % CORE_SEGMENT_LEN
        if _loop_segment != 0:
            with self.tik_instance.for_range(
                    0, _loop_segment) as _loop:
                _run(CORE_SEGMENT_LEN, _loop)
        if _loop_segment_tail != 0:
            _run(_loop_segment_tail, _loop_segment)

    # each core handle several segments
    def compute_argmax_not_last_axis(self, n_i, core_segment, segment_core):
        """
        compute arg when do last axis

        Parameters
        ----------
        n_i : int
            the first loop index
        core_segment : int
            segment process len
        segment_core : int
            the total segment index

        Returns
        -------
        None
        """
        # split the last dim into loops, the split step is DATA_EACH_VNCHWCONV(16) for simplicity
        ub_buf_size, loop_times, over_size, align_flag = \
            get_tiling_info_for_axis(self.last_dim_size, self.task_segment_len_last_dim)

        # the number of elements this core to write, for 32B alignment purpose
        element_end_address_to_write = (n_i * segment_core + core_segment) * \
                                       self.topk * self.last_dim_size
        elements_to_write = core_segment * self.topk * self.last_dim_size

        def _run(segment_len, segment_index):
            with self.tik_instance.for_range(0, segment_len) as core_i:
                index = core_i + CORE_SEGMENT_LEN * segment_index
                offset = n_i * segment_core + index

                argmax_func = self.do_argmax_not_last_axis
                if loop_times != 0:
                    with self.tik_instance.for_range(0, loop_times) as loop:
                        argmax_func(ub_buf_size, loop, offset,
                                    loop_times, element_end_address_to_write, elements_to_write)
                if align_flag:
                    with self.tik_instance.new_stmt_scope():
                        argmax_func(over_size, loop_times, offset,
                                    loop_times, element_end_address_to_write, elements_to_write)

        _loop_segment = core_segment // CORE_SEGMENT_LEN
        _loop_segment_tail = core_segment % CORE_SEGMENT_LEN
        if _loop_segment != 0:
            with self.tik_instance.for_range(
                    0, _loop_segment) as _loop:
                _run(CORE_SEGMENT_LEN, _loop)
        if _loop_segment_tail != 0:
            _run(_loop_segment_tail, _loop_segment)

    def do_argmax_large_task_case(self, first_dim, second_dim, ub_buf_size_last_dim, loop_last_dim,
                                  task_segment_len_last_dim, src_ub_data, val_ub, ind_ub):
        """
        do arg in one segment for float16

        Parameters
        ----------
        ub_buf_size : int
            process len
        loop : int
            segment index in one core
        n_i : int
            the first loop index

        Returns
        -------
        None
        """
        # split the axis size into loops
        ub_buf_size_axis, loop_times_axis, over_size_axis, align_flag_axis = \
            get_tiling_info_for_axis(self.axis_size, self.task_segment_len)
        last_dim_fetch_count = _get_round_ceil_int(ub_buf_size_last_dim, self.data_each_vector)

        ub_result_value = self.tik_instance.Tensor(
            self.dtype_x, (last_dim_fetch_count,),
            name="ub_result_value",
            scope=tik.scope_ubuf)

        dup_value = SCALAR_MIN_FP16 if self.dtype_x == "float16" else SCALAR_MIN_FP32
        repeat_times = _get_div_ceil_int(ub_buf_size_last_dim, self.data_each_vector)
        self.tik_instance.vec_dup(self.data_each_vector, ub_result_value,
                                  dup_value, repeat_times, 8)

        if self.out_max_index:
            # the ub buffer used to store the final results of segments assigned to this core
            if self.dtype_x == "float16" and second_dim * self.last_dim_size > 1:
                ub_result_indices = self.tik_instance.Tensor(
                    "uint16", (_get_round_ceil_int(ub_buf_size_last_dim, self.data_each_vector),),
                    name="ub_result_indices",
                    scope=tik.scope_ubuf)
            else:
                ub_result_indices = self.tik_instance.Tensor(
                    "int32", (_get_round_ceil_int(ub_buf_size_last_dim, self.index_each_vector),),
                    name="ub_result_indices",
                    scope=tik.scope_ubuf)

        def inner_func(axis_count, axis_loop):
            if self.out_max_index:
                # the ub buffer used to store the final results of segments assigned to this core
                ub_vcmpv_lt_result = self.tik_instance.Tensor(
                    "uint64",
                    (self.task_segment_len, _get_round_ceil_int(last_dim_fetch_count // 64, 32),),
                    name="ub_vcmpv_lt_result",
                    scope=tik.scope_ubuf)

            thread_num = 2 if axis_count > 1 else 1
            with self.tik_instance.for_range(0, axis_count, thread_num=thread_num) as idx:
                # the offset in the input data
                ub_offset = self.last_dim_size * first_dim * \
                            (axis_loop * self.task_segment_len + idx) + \
                            loop_last_dim * task_segment_len_last_dim
                if self.out_max_index:
                    # get repeat times
                    repeat_times = _get_div_ceil_int(ub_buf_size_last_dim, self.data_each_vector)
                    self.tik_instance.vcmpv_lt(ub_vcmpv_lt_result[idx, 0], ub_result_value,
                                               src_ub_data[ub_offset], repeat_times, 1, 1, 8, 8)
                    self.tik_instance.vec_max(self.data_each_vector, ub_result_value,
                                              ub_result_value, src_ub_data[ub_offset],
                                              repeat_times, 8, 8, 8)
                else:
                    # get repeat times
                    repeat_times = _get_div_ceil_int(ub_buf_size_last_dim, self.data_each_vector)
                    self.tik_instance.vec_max(self.data_each_vector, ub_result_value,
                                              ub_result_value, src_ub_data[ub_offset],
                                              repeat_times, 8, 8, 8)

            if self.out_max_index:
                # first process the vcmpv_lt result
                ub_tmp_not = self.tik_instance.Tensor(
                    "uint64", (_get_round_ceil_int(last_dim_fetch_count // 64, 32),),
                    name="ub_tmp_not",
                    scope=tik.scope_ubuf)

                ub_tmp_or = self.tik_instance.Tensor(
                    "uint64", (_get_round_ceil_int(last_dim_fetch_count // 64, 32),),
                    name="ub_tmp_or",
                    scope=tik.scope_ubuf)

                int64_row_repeat = _get_div_ceil_int(last_dim_fetch_count // 64, 32)
                int16_row_len = int64_row_repeat * 32 * 4
                ub_vcmpv_lt_uint16 = ub_vcmpv_lt_result.reinterpret_cast_to("uint16")
                ub_tmp_not_uint16 = ub_tmp_not.reinterpret_cast_to("uint16")
                ub_tmp_or_uint16 = ub_tmp_or.reinterpret_cast_to("uint16")

                with self.tik_instance.for_range(0, axis_count) as idx:
                    row_id = axis_count - 1 - idx

                    with self.tik_instance.if_scope(idx == 0):
                        self.tik_instance.vnot(128, ub_tmp_not_uint16,
                                               ub_vcmpv_lt_uint16[int16_row_len * row_id],
                                               int64_row_repeat, 1, 1, 8, 8)
                        self.tik_instance.vor(128, ub_tmp_or_uint16,
                                              ub_vcmpv_lt_uint16[int16_row_len * row_id],
                                              ub_vcmpv_lt_uint16[int16_row_len * row_id],
                                              int64_row_repeat, 1, 1, 1, 8, 8, 8)
                    with self.tik_instance.if_scope(row_id > 0):
                        self.tik_instance.vand(128,
                                               ub_vcmpv_lt_uint16[int16_row_len * (row_id - 1)],
                                               ub_vcmpv_lt_uint16[int16_row_len * (row_id - 1)],
                                               ub_tmp_not_uint16,
                                               int64_row_repeat, 1, 1, 1, 8, 8, 8)

                        self.tik_instance.vor(128, ub_tmp_or_uint16,
                                              ub_vcmpv_lt_uint16[int16_row_len * (row_id - 1)],
                                              ub_tmp_or_uint16,
                                              int64_row_repeat, 1, 1, 1, 8, 8, 8)

                        self.tik_instance.vnot(128, ub_tmp_not_uint16,
                                               ub_tmp_or_uint16,
                                               int64_row_repeat, 1, 1, 8, 8)

                # then do the normal flow
                if self.dtype_x == "float16" and second_dim * self.last_dim_size > 1:
                    int64_num = _get_div_ceil_int(last_dim_fetch_count, 64)
                    with self.tik_instance.for_range(0, axis_count) as idx:
                        axis_id = self.task_segment_len * axis_loop + idx

                        with self.tik_instance.for_range(0, int64_num // 2) as i:
                            mask_h = self.tik_instance.Scalar("uint64")
                            mask_h.set_as(ub_vcmpv_lt_result[idx, 2 * i + 1])
                            mask_l = self.tik_instance.Scalar("uint64")
                            mask_l.set_as(ub_vcmpv_lt_result[idx, 2 * i])
                            with self.tik_instance.if_scope(tik.any(mask_h != 0, mask_l != 0)):
                                self.tik_instance.vec_dup(
                                    [mask_h, mask_l],
                                    ub_result_indices[i * self.data_each_vector],
                                    axis_id, 1, 8)
                else:
                    int64_num = _get_div_ceil_int(last_dim_fetch_count, 64)
                    mask_h = self.tik_instance.Scalar("uint64")
                    mask_h.set_as(0)
                    with self.tik_instance.for_range(0, axis_count) as idx:
                        axis_id = self.task_segment_len * axis_loop + idx

                        with self.tik_instance.for_range(0, int64_num) as i:
                            mask_l = self.tik_instance.Scalar("uint64")
                            mask_l.set_as(ub_vcmpv_lt_result[idx, i])
                            with self.tik_instance.if_scope(mask_l != 0):
                                self.tik_instance.vec_dup(
                                    [mask_h, mask_l],
                                    ub_result_indices[i * self.index_each_vector],
                                    axis_id, 1, 8)

        if loop_times_axis != 0:
            with self.tik_instance.for_range(0, loop_times_axis) as axis_loop:
                inner_func(ub_buf_size_axis, axis_loop)
        if align_flag_axis:
            with self.tik_instance.new_stmt_scope():
                inner_func(over_size_axis, loop_times_axis)

        ub_offset = loop_last_dim * task_segment_len_last_dim

        if self.out_max_val:
            burst = _get_div_ceil_int(ub_buf_size_last_dim, self.data_each_block)
            self.tik_instance.data_move(val_ub[ub_offset], ub_result_value, 0, 1, burst, 0, 0)

        if self.out_max_index:
            if self.dtype_x == "float16" and second_dim * self.last_dim_size > 1:
                burst = _get_div_ceil_int(first_dim, self.data_each_block)
                self.tik_instance.data_move(ind_ub[2 * ub_offset], ub_result_indices,
                                            0, ub_buf_size_last_dim // first_dim,
                                            burst, 0, burst)
            else:
                burst = _get_div_ceil_int(ub_buf_size_last_dim, self.index_each_block)
                self.tik_instance.data_move(ind_ub[ub_offset], ub_result_indices, 0, 1, burst, 0, 0)

    def do_argmax_not_last_axis_using_vmax(self, ub_buf_size_last_dim, loop_last_dim, n_i,
                                           loop_last_dim_max, element_end_address_to_write,
                                           elements_to_write):
        """
        do arg in one segment for float16

        Parameters
        ----------
        ub_buf_size : int
            process len
        loop : int
            segment index in one core
        n_i : int
            the first loop index

        Returns
        -------
        None
        """
        # split the axis size into loops
        ub_buf_size_axis, loop_times_axis, over_size_axis, align_flag_axis = \
            get_tiling_info_for_axis(self.axis_size, self.task_segment_len)
        last_dim_fetch_count = _get_round_ceil_int(ub_buf_size_last_dim, self.data_each_vector)

        ub_result_value = self.tik_instance.Tensor(
            self.dtype_x, (last_dim_fetch_count,),
            name="ub_result_value",
            scope=tik.scope_ubuf)

        dup_value = SCALAR_MIN_FP16 if self.dtype_x == "float16" else SCALAR_MIN_FP32
        repeat_times = _get_div_ceil_int(ub_buf_size_last_dim, self.data_each_vector)
        start_offset = 0
        while repeat_times > 0:
            repeat = min(255, repeat_times)
            self.tik_instance.vec_dup(self.data_each_vector, ub_result_value[start_offset],
                                      dup_value, repeat, 8)
            repeat_times -= repeat
            start_offset += self.data_each_vector * repeat

        if self.out_max_index:
            # the ub buffer used to store the final results of segments assigned to this core
            ub_result_indices = self.tik_instance.Tensor(
                "int32", (_get_round_ceil_int(ub_buf_size_last_dim, self.index_each_vector),),
                name="ub_result_indices",
                scope=tik.scope_ubuf)

        def inner_func(axis_count, axis_loop):
            if self.out_max_index:
                # the ub buffer used to store the final results of segments assigned to this core
                ub_vcmpv_lt_result = self.tik_instance.Tensor(
                    "uint64",
                    (self.task_segment_len, _get_round_ceil_int(last_dim_fetch_count // 64, 32),),
                    name="ub_vcmpv_lt_result",
                    scope=tik.scope_ubuf)

            thread_num = 2 if axis_count > 1 else 1
            with self.tik_instance.for_range(0, axis_count, thread_num=thread_num) as idx:
                ub_data = self.tik_instance.Tensor(
                    self.dtype_x, (last_dim_fetch_count,),
                    name="ub_data",
                    scope=tik.scope_ubuf)

                # the offset in the original input data
                offset = n_i * self.axis_size * self.last_dim_size + \
                         self.last_dim_size * (axis_loop * self.task_segment_len + idx) + \
                         loop_last_dim * self.task_segment_len_last_dim
                # data move from gm to ub
                self.tik_instance.data_move(
                    ub_data, self.data_gm[offset], 0, 1,
                    _get_div_ceil_int(ub_buf_size_last_dim, self.data_each_block), 0, 0)

                if self.out_max_index:
                    # get repeat times
                    repeat_times = _get_div_ceil_int(ub_buf_size_last_dim, self.data_each_vector)

                    self.tik_instance.vcmpv_lt(ub_vcmpv_lt_result[idx, 0], ub_result_value,
                                               ub_data, repeat_times, 1, 1, 8, 8)

                    self.tik_instance.vec_max(self.data_each_vector, ub_result_value,
                                              ub_result_value, ub_data, repeat_times, 8, 8, 8)
                else:
                    # get repeat times
                    repeat_times = _get_div_ceil_int(ub_buf_size_last_dim, self.data_each_vector)
                    start_offset = 0
                    while repeat_times > 0:
                        repeat = min(repeat_times, 255)
                        self.tik_instance.vec_max(self.data_each_vector,
                                                  ub_result_value[start_offset],
                                                  ub_result_value[start_offset],
                                                  ub_data[start_offset],
                                                  repeat, 8, 8, 8)
                        repeat_times -= repeat
                        start_offset += self.data_each_vector * repeat

            if self.out_max_index:
                # first process the vcmpv_lt result
                ub_tmp_not = self.tik_instance.Tensor(
                    "uint64", (_get_round_ceil_int(last_dim_fetch_count // 64, 32),),
                    name="ub_tmp_not",
                    scope=tik.scope_ubuf)

                ub_tmp_or = self.tik_instance.Tensor(
                    "uint64", (_get_round_ceil_int(last_dim_fetch_count // 64, 32),),
                    name="ub_tmp_or",
                    scope=tik.scope_ubuf)

                int64_row_repeat = _get_div_ceil_int(last_dim_fetch_count // 64, 32)
                int16_row_len = int64_row_repeat * 32 * 4
                ub_vcmpv_lt_uint16 = ub_vcmpv_lt_result.reinterpret_cast_to("uint16")
                ub_tmp_not_uint16 = ub_tmp_not.reinterpret_cast_to("uint16")
                ub_tmp_or_uint16 = ub_tmp_or.reinterpret_cast_to("uint16")

                with self.tik_instance.for_range(0, axis_count) as idx:
                    row_id = axis_count - 1 - idx

                    with self.tik_instance.if_scope(idx == 0):
                        self.tik_instance.vnot(128, ub_tmp_not_uint16,
                                               ub_vcmpv_lt_uint16[int16_row_len * row_id],
                                               int64_row_repeat, 1, 1, 8, 8)
                        self.tik_instance.vor(128, ub_tmp_or_uint16,
                                              ub_vcmpv_lt_uint16[int16_row_len * row_id],
                                              ub_vcmpv_lt_uint16[int16_row_len * row_id],
                                              int64_row_repeat, 1, 1, 1, 8, 8, 8)
                    with self.tik_instance.if_scope(row_id > 0):
                        self.tik_instance.vand(128,
                                               ub_vcmpv_lt_uint16[int16_row_len * (row_id - 1)],
                                               ub_vcmpv_lt_uint16[int16_row_len * (row_id - 1)],
                                               ub_tmp_not_uint16,
                                               int64_row_repeat, 1, 1, 1, 8, 8, 8)

                        self.tik_instance.vor(128, ub_tmp_or_uint16,
                                              ub_vcmpv_lt_uint16[int16_row_len * (row_id - 1)],
                                              ub_tmp_or_uint16,
                                              int64_row_repeat, 1, 1, 1, 8, 8, 8)

                        self.tik_instance.vnot(128, ub_tmp_not_uint16,
                                               ub_tmp_or_uint16,
                                               int64_row_repeat, 1, 1, 8, 8)

                # then do the normal flow
                int64_num = _get_div_ceil_int(last_dim_fetch_count, 64)
                mask_h = self.tik_instance.Scalar("uint64")
                mask_h.set_as(0)
                with self.tik_instance.for_range(0, axis_count) as idx:
                    axis_id = self.task_segment_len * axis_loop + idx

                    with self.tik_instance.for_range(0, int64_num) as i:
                        mask_l = self.tik_instance.Scalar("uint64")
                        mask_l.set_as(ub_vcmpv_lt_result[idx, i])
                        with self.tik_instance.if_scope(mask_l != 0):
                            self.tik_instance.vec_dup(
                                [mask_h, mask_l],
                                ub_result_indices[i * self.index_each_vector],
                                axis_id, 1, 8)

        if loop_times_axis != 0:
            with self.tik_instance.for_range(0, loop_times_axis) as axis_loop:
                inner_func(ub_buf_size_axis, axis_loop)
        if align_flag_axis:
            with self.tik_instance.new_stmt_scope():
                inner_func(over_size_axis, loop_times_axis)

        gm_offset = n_i * self.last_dim_size + loop_last_dim * self.task_segment_len_last_dim

        # store back the max value to gm
        if self.out_max_val:
            if elements_to_write < self.data_each_block:
                self.tik_instance.data_move(
                    self.result_gm_value[gm_offset], ub_result_value, 0, 1,
                    _get_div_ceil_int(ub_buf_size_last_dim, self.data_each_block), 0, 0)
            else:
                with self.tik_instance.if_scope(loop_last_dim < loop_last_dim_max):
                    self.tik_instance.data_move(
                        self.result_gm_value[gm_offset], ub_result_value, 0, 1,
                        _get_div_ceil_int(ub_buf_size_last_dim, self.data_each_block), 0, 0)
                with self.tik_instance.else_scope():
                    bias = element_end_address_to_write - gm_offset - \
                           _get_round_ceil_int(ub_buf_size_last_dim, self.data_each_block)
                    with self.tik_instance.if_scope(bias >= 0):
                        self.tik_instance.data_move(
                            self.result_gm_value[gm_offset], ub_result_value, 0, 1,
                            _get_div_ceil_int(ub_buf_size_last_dim, self.data_each_block), 0, 0)
                    with self.tik_instance.else_scope():
                        burst_len_floor = ub_buf_size_last_dim // self.data_each_block
                        ub_elem_left = ub_buf_size_last_dim % self.data_each_block
                        ub_elem_left_start = burst_len_floor * self.data_each_block
                        if burst_len_floor > 0:
                            self.tik_instance.data_move(self.result_gm_value[gm_offset],
                                                        ub_result_value, 0, 1,
                                                        burst_len_floor, 0, 0)
                        result_out_scalar_tmp = self.tik_instance.Tensor(
                            self.dtype_x, (self.data_each_block,),
                            name="result_out_scalar_tmp",
                            scope=tik.scope_ubuf)
                        self.tik_instance.data_move(
                            result_out_scalar_tmp,
                            self.result_gm_value[element_end_address_to_write -
                                                 self.data_each_block],
                            0, 1, 1, 0, 0)
                        value_reg = self.tik_instance.Scalar(dtype=self.dtype_x)
                        with self.tik_instance.for_range(0, ub_elem_left) as i:
                            value_reg.set_as(ub_result_value[ub_elem_left_start + i])
                            result_out_scalar_tmp[i - bias].set_as(value_reg)
                        self.tik_instance.data_move(
                            self.result_gm_value[element_end_address_to_write -
                                                 self.data_each_block],
                            result_out_scalar_tmp, 0, 1, 1, 0, 0)

        # store back the max index to gm
        if self.out_max_index:
            if elements_to_write < self.index_each_block:
                self.tik_instance.data_move(
                    self.result_gm[gm_offset], ub_result_indices, 0, 1,
                    _get_div_ceil_int(ub_buf_size_last_dim, self.index_each_block), 0, 0)
            else:
                with self.tik_instance.if_scope(loop_last_dim < loop_last_dim_max):
                    self.tik_instance.data_move(
                        self.result_gm[gm_offset], ub_result_indices, 0, 1,
                        _get_div_ceil_int(ub_buf_size_last_dim, self.index_each_block), 0, 0)
                with self.tik_instance.else_scope():
                    bias = element_end_address_to_write - gm_offset - \
                           _get_round_ceil_int(ub_buf_size_last_dim, self.index_each_block)
                    with self.tik_instance.if_scope(bias >= 0):
                        self.tik_instance.data_move(
                            self.result_gm[gm_offset], ub_result_indices, 0, 1,
                            _get_div_ceil_int(ub_buf_size_last_dim, self.index_each_block), 0, 0)
                    with self.tik_instance.else_scope():
                        burst_len_floor = ub_buf_size_last_dim // self.index_each_block
                        ub_elem_left = ub_buf_size_last_dim % self.index_each_block
                        ub_elem_left_start = burst_len_floor * self.index_each_block
                        if burst_len_floor > 0:
                            self.tik_instance.data_move(self.result_gm[gm_offset],
                                                        ub_result_indices, 0, 1,
                                                        burst_len_floor, 0, 0)
                        result_int32_tmp = self.tik_instance.Tensor(
                            "int32", (self.index_each_block,),
                            name="result_int32_tmp",
                            scope=tik.scope_ubuf)
                        self.tik_instance.data_move(
                            result_int32_tmp,
                            self.result_gm[element_end_address_to_write - self.index_each_block],
                            0, 1, 1, 0, 0)
                        index_reg = self.tik_instance.Scalar(dtype="int32")
                        with self.tik_instance.for_range(0, ub_elem_left) as i:
                            index_reg.set_as(ub_result_indices[ub_elem_left_start + i])
                            result_int32_tmp[i - bias].set_as(index_reg)
                        self.tik_instance.data_move(
                            self.result_gm[element_end_address_to_write - self.index_each_block],
                            result_int32_tmp, 0, 1, 1, 0, 0)

    def do_argmax_not_last_axis(self, ub_buf_size_last_dim, loop_last_dim, n_i,
                                loop_last_dim_max, element_end_address_to_write, elements_to_write):
        """
        do arg in one segment for float16

        Parameters
        ----------
        ub_buf_size : int
            process len
        loop : int
            segment index in one core
        n_i : int
            the first loop index

        Returns
        -------
        None
        """
        # split the axis size into loops
        ub_buf_size_axis, loop_times_axis, over_size_axis, align_flag_axis = \
            get_tiling_info_for_axis(self.axis_size, self.task_segment_len)
        last_dim_fetch_count = _get_round_ceil_int(ub_buf_size_last_dim, DATA_EACH_VNCHWCONV)

        # the ub buffer used to store the final results of segments assigned to this core
        ub_result_indices = self.tik_instance.Tensor(
            "int32", (_get_round_ceil_int(ub_buf_size_last_dim, self.index_each_block),),
            name="ub_result_indices",
            scope=tik.scope_ubuf)
        ub_result_value = self.tik_instance.Tensor(
            self.dtype_x, (last_dim_fetch_count, DATA_EACH_VNCHWCONV),
            name="ub_result_value",
            scope=tik.scope_ubuf)

        def inner_func(axis_count, axis_loop):
            # use to store the first vcmax result, the self.task_segment_len limit to 16K
            repeat_num_1 = _get_div_ceil_int(axis_count, self.data_each_vector)
            repeat_num_2 = _get_div_ceil_int(repeat_num_1, self.data_each_vector // 2)

            # the ub buffer used to store the first vcmax results
            ub_result_first = self.tik_instance.Tensor(
                self.dtype_x, (ub_buf_size_last_dim, repeat_num_2 * self.data_each_vector,),
                name="ub_result_first",
                scope=tik.scope_ubuf)

            # the ub buffer used to store the second vcmax results
            ub_result_second = self.tik_instance.Tensor(
                self.dtype_x, (ub_buf_size_last_dim, self.data_each_block,),
                name="ub_result_second",
                scope=tik.scope_ubuf)

            # the ub buffer used to store the third vcmax results
            ub_result_third = self.tik_instance.Tensor(
                self.dtype_x, (ub_buf_size_last_dim, self.data_each_block,),
                name="ub_result_third",
                scope=tik.scope_ubuf)

            axis_fetch_count = _get_round_ceil_int(self.task_segment_len, self.data_each_vector)

            # use vec_trans_scatter for transpose
            ub_data_transpose = self.tik_instance.Tensor(
                self.dtype_x, (last_dim_fetch_count, axis_fetch_count),
                name="ub_data_transpose",
                scope=tik.scope_ubuf)

            with self.tik_instance.new_stmt_scope():
                # the original input data
                ub_data = self.tik_instance.Tensor(
                    self.dtype_x, (axis_fetch_count, last_dim_fetch_count),
                    name="ub_data",
                    scope=tik.scope_ubuf)

                if self.last_dim_size % self.data_each_block == 0:
                    # 32B alignment case
                    min_len = ub_buf_size_last_dim
                    # a special case for 32B alignment
                    src_gap = _get_div_ceil_int(
                        self.last_dim_size - min_len,
                        self.data_each_block)
                    dst_gap = _get_div_ceil_int(last_dim_fetch_count - min_len,
                                                self.data_each_block)
                    # the offset in the original input data
                    offset = n_i * self.axis_size * self.last_dim_size + \
                             self.last_dim_size * (axis_loop * self.task_segment_len) + \
                             loop_last_dim * self.task_segment_len_last_dim
                    # data move from gm to ub
                    if src_gap == 0 and dst_gap == 0:
                        self.tik_instance.data_move(
                            ub_data, self.data_gm[offset], 0, 1,
                            _get_div_ceil_int(min_len * axis_count,
                                              self.data_each_block),
                            src_gap, dst_gap)
                    else:
                        self.tik_instance.data_move(
                            ub_data, self.data_gm[offset], 0, axis_count,
                            _get_div_ceil_int(min_len, self.data_each_block),
                            src_gap, dst_gap)
                elif self.last_dim_size < self.task_segment_len_last_dim:
                    # the last dim is small, move all continuous data in batches
                    # the offset in the original input data
                    offset = n_i * self.axis_size * self.last_dim_size + \
                             self.last_dim_size * (axis_loop * self.task_segment_len) + \
                             loop_last_dim * self.task_segment_len_last_dim
                    # data move from gm to ub
                    self.tik_instance.data_move(
                        ub_data_transpose, self.data_gm[offset], 0, 1,
                        _get_div_ceil_int(self.last_dim_size * axis_count,
                                          self.data_each_block),
                        0, 0)

                    first_dim = DATA_EACH_VNCHWCONV
                    second_dim = self.task_segment_len // first_dim

                    # first do the transpose, src is ub_data_transpose, dest is ub_data
                    row_loop_count = _get_div_ceil_int(first_dim, DATA_EACH_VNCHWCONV)
                    col_loop_count = _get_div_ceil_int(second_dim * self.last_dim_size,
                                                       DATA_EACH_VNCHWCONV)
                    if self.dtype_x == "float16":
                        # float16, 16*16
                        with self.tik_instance.for_range(0, row_loop_count) as row_idx:
                            # should of length 16, but the elements can be duplicate,
                            # the region of each elements at most not overlap
                            dst_list = []
                            for i in range(DATA_EACH_VNCHWCONV):
                                dst_pos = first_dim * i + DATA_EACH_VNCHWCONV * row_idx
                                dst_list.append(ub_data[dst_pos])
                            # should of length 16, but the elements can be duplicate
                            src_list = []
                            for i in range(DATA_EACH_VNCHWCONV):
                                src_pos = self.last_dim_size * second_dim * \
                                          (DATA_EACH_VNCHWCONV * row_idx + i)
                                src_list.append(ub_data_transpose[src_pos])

                            if col_loop_count == 1:
                                self.tik_instance.vec_trans_scatter(False, False,
                                                                    dst_list, src_list,
                                                                    col_loop_count, 0, 0)
                            else:
                                self.tik_instance.vec_trans_scatter(
                                    False, False, dst_list, src_list, col_loop_count,
                                    DATA_EACH_VNCHWCONV * first_dim // self.data_each_block,
                                    DATA_EACH_VNCHWCONV // self.data_each_block)
                    else:
                        # float32, 16*8
                        with self.tik_instance.for_range(0, row_loop_count) as row_idx:
                            # should of length 16, but the elements can be duplicate,
                            # the region of each elements at most not overlap
                            dst_list = []
                            for i in range(DATA_EACH_VNCHWCONV):
                                dst_pos = first_dim * (i // 2) + \
                                          (i % 2) * DATA_EACH_VNCHWCONV // 2 + \
                                          DATA_EACH_VNCHWCONV * row_idx
                                dst_list.append(ub_data[dst_pos])
                            # should of length 16, but the elements can be duplicate
                            src_list = []
                            for i in range(DATA_EACH_VNCHWCONV):
                                src_pos = self.last_dim_size * second_dim * \
                                          (DATA_EACH_VNCHWCONV * row_idx + i)
                                src_list.append(ub_data_transpose[src_pos])

                            self.tik_instance.vec_trans_scatter(
                                False, False, dst_list, src_list, 2 * col_loop_count,
                                DATA_EACH_VNCHWCONV // 2 * first_dim // self.data_each_block,
                                DATA_EACH_VNCHWCONV // 2 // self.data_each_block)

                    # second do the padding, src is ub_data, dest is ub_data_transpose
                    with self.tik_instance.for_range(0, second_dim) as idx:
                        src_offset = self.last_dim_size * first_dim * idx
                        dest_offset = last_dim_fetch_count * first_dim * idx
                        self.tik_instance.vec_adds(self.data_each_vector,
                                                   ub_data_transpose[dest_offset],
                                                   ub_data[src_offset], 0,
                                                   _get_div_ceil_int(self.last_dim_size * first_dim,
                                                                     self.data_each_vector),
                                                   8, 8)

                    # third do the transpose again, src is ub_data_transpose, dest is ub_data
                    row_loop_count = _get_div_ceil_int(second_dim * last_dim_fetch_count,
                                                       DATA_EACH_VNCHWCONV)
                    col_loop_count = _get_div_ceil_int(first_dim, DATA_EACH_VNCHWCONV)

                    if self.dtype_x == "float16":
                        # float16, 16*16
                        with self.tik_instance.for_range(0, col_loop_count) as col_idx:
                            # should of length 16, but the elements can be duplicate,
                            # the region of each elements at most not overlap
                            dst_list = []
                            for i in range(DATA_EACH_VNCHWCONV):
                                dst_pos = last_dim_fetch_count * second_dim * \
                                          (DATA_EACH_VNCHWCONV * col_idx + i)
                                dst_list.append(ub_data[dst_pos])
                            # should of length 16, but the elements can be duplicate
                            src_list = []
                            for i in range(DATA_EACH_VNCHWCONV):
                                src_pos = first_dim * i + DATA_EACH_VNCHWCONV * col_idx
                                src_list.append(ub_data_transpose[src_pos])

                            if row_loop_count == 1:
                                self.tik_instance.vec_trans_scatter(False, False,
                                                                    dst_list, src_list,
                                                                    row_loop_count, 0, 0)
                            else:
                                self.tik_instance.vec_trans_scatter(
                                    False, False, dst_list, src_list, row_loop_count,
                                    DATA_EACH_VNCHWCONV // self.data_each_block,
                                    DATA_EACH_VNCHWCONV * first_dim // self.data_each_block)
                    else:
                        # float32, 16*8
                        with self.tik_instance.for_range(0, 2 * col_loop_count) as col_idx:
                            # should of length 16, but the elements can be duplicate,
                            # the region of each elements at most not overlap
                            dst_list = []
                            for i in range(DATA_EACH_VNCHWCONV):
                                dst_pos = last_dim_fetch_count * second_dim * \
                                          (DATA_EACH_VNCHWCONV // 2 * col_idx + i // 2) + \
                                          (i % 2) * DATA_EACH_VNCHWCONV // 2
                                dst_list.append(ub_data[dst_pos])
                            # should of length 16, but the elements can be duplicate
                            src_list = []
                            for i in range(DATA_EACH_VNCHWCONV):
                                src_pos = first_dim * i + DATA_EACH_VNCHWCONV // 2 * col_idx
                                src_list.append(ub_data_transpose[src_pos])

                            if row_loop_count == 1:
                                self.tik_instance.vec_trans_scatter(False, False,
                                                                    dst_list, src_list,
                                                                    row_loop_count, 0, 0)
                            else:
                                self.tik_instance.vec_trans_scatter(
                                    False, False, dst_list, src_list, row_loop_count,
                                    DATA_EACH_VNCHWCONV // self.data_each_block,
                                    DATA_EACH_VNCHWCONV * first_dim // self.data_each_block)
                else:
                    with self.tik_instance.for_range(0, axis_count) as idx:
                        # the offset in the original input data
                        offset = n_i * self.axis_size * self.last_dim_size + \
                                 self.last_dim_size * (axis_loop * self.task_segment_len + idx) + \
                                 loop_last_dim * self.task_segment_len_last_dim
                        # data move from gm to ub
                        self.tik_instance.data_move(
                            ub_data[last_dim_fetch_count * idx], self.data_gm[offset], 0, 1,
                            _get_div_ceil_int(last_dim_fetch_count, self.data_each_block), 0, 0)

                row_loop_count = _get_div_ceil_int(axis_count, DATA_EACH_VNCHWCONV)
                col_loop_count = _get_div_ceil_int(ub_buf_size_last_dim, DATA_EACH_VNCHWCONV)

                if col_loop_count > row_loop_count:
                    if self.dtype_x == "float16":
                        # float16, 16*16
                        with self.tik_instance.for_range(0, row_loop_count) as row_idx:
                            # should of length 16, but the elements can be duplicate,
                            # the region of each elements at most not overlap
                            dst_list = []
                            for i in range(DATA_EACH_VNCHWCONV):
                                dst_pos = axis_fetch_count * i + DATA_EACH_VNCHWCONV * row_idx
                                dst_list.append(ub_data_transpose[dst_pos])
                            # should of length 16, but the elements can be duplicate
                            src_list = []
                            for i in range(DATA_EACH_VNCHWCONV):
                                src_pos = last_dim_fetch_count * (DATA_EACH_VNCHWCONV * row_idx + i)
                                src_list.append(ub_data[src_pos])

                            if col_loop_count == 1:
                                self.tik_instance.vec_trans_scatter(False, False,
                                                                    dst_list, src_list,
                                                                    col_loop_count, 0, 0)
                            else:
                                self.tik_instance.vec_trans_scatter(
                                    False, False, dst_list, src_list, col_loop_count,
                                    DATA_EACH_VNCHWCONV * axis_fetch_count // self.data_each_block,
                                    DATA_EACH_VNCHWCONV // self.data_each_block)
                    else:
                        # float32, 16*8
                        with self.tik_instance.for_range(0, row_loop_count) as row_idx:
                            # should of length 16, but the elements can be duplicate,
                            # the region of each elements at most not overlap
                            dst_list = []
                            for i in range(DATA_EACH_VNCHWCONV):
                                dst_pos = axis_fetch_count * (i // 2) + \
                                          (i % 2) * DATA_EACH_VNCHWCONV // 2 + \
                                          DATA_EACH_VNCHWCONV * row_idx
                                dst_list.append(ub_data_transpose[dst_pos])
                            # should of length 16, but the elements can be duplicate
                            src_list = []
                            for i in range(DATA_EACH_VNCHWCONV):
                                src_pos = last_dim_fetch_count * (DATA_EACH_VNCHWCONV * row_idx + i)
                                src_list.append(ub_data[src_pos])

                            self.tik_instance.vec_trans_scatter(
                                False, False, dst_list, src_list, 2 * col_loop_count,
                                DATA_EACH_VNCHWCONV // 2 * axis_fetch_count // self.data_each_block,
                                DATA_EACH_VNCHWCONV // 2 // self.data_each_block)
                else:
                    if self.dtype_x == "float16":
                        # float16, 16*16
                        with self.tik_instance.for_range(0, col_loop_count) as col_idx:
                            # should of length 16, but the elements can be duplicate,
                            # the region of each elements at most not overlap
                            dst_list = []
                            for i in range(DATA_EACH_VNCHWCONV):
                                dst_pos = axis_fetch_count * (DATA_EACH_VNCHWCONV * col_idx + i)
                                dst_list.append(ub_data_transpose[dst_pos])
                            # should of length 16, but the elements can be duplicate
                            src_list = []
                            for i in range(DATA_EACH_VNCHWCONV):
                                src_pos = last_dim_fetch_count * i + DATA_EACH_VNCHWCONV * col_idx
                                src_list.append(ub_data[src_pos])

                            if row_loop_count == 1:
                                self.tik_instance.vec_trans_scatter(False, False,
                                                                    dst_list, src_list,
                                                                    row_loop_count, 0, 0)
                            else:
                                self.tik_instance.vec_trans_scatter(
                                    False, False, dst_list, src_list, row_loop_count,
                                    DATA_EACH_VNCHWCONV // self.data_each_block,
                                    DATA_EACH_VNCHWCONV * last_dim_fetch_count //
                                    self.data_each_block)
                    else:
                        # float32, 16*8
                        with self.tik_instance.for_range(0, 2 * col_loop_count) as col_idx:
                            # should of length 16, but the elements can be duplicate,
                            # the region of each elements at most not overlap
                            dst_list = []
                            for i in range(DATA_EACH_VNCHWCONV):
                                dst_pos = axis_fetch_count * \
                                          (DATA_EACH_VNCHWCONV // 2 * col_idx + i // 2) + \
                                          (i % 2) * DATA_EACH_VNCHWCONV // 2
                                dst_list.append(ub_data_transpose[dst_pos])
                            # should of length 16, but the elements can be duplicate
                            src_list = []
                            for i in range(DATA_EACH_VNCHWCONV):
                                src_pos = last_dim_fetch_count * i + \
                                          DATA_EACH_VNCHWCONV // 2 * col_idx
                                src_list.append(ub_data[src_pos])

                            if row_loop_count == 1:
                                self.tik_instance.vec_trans_scatter(False, False,
                                                                    dst_list, src_list,
                                                                    row_loop_count, 0, 0)
                            else:
                                self.tik_instance.vec_trans_scatter(
                                    False, False, dst_list, src_list, row_loop_count,
                                    DATA_EACH_VNCHWCONV // self.data_each_block,
                                    DATA_EACH_VNCHWCONV * last_dim_fetch_count //
                                    self.data_each_block)

            with self.tik_instance.for_range(0, ub_buf_size_last_dim) as idx:
                # if has tail, set the data in the tail part as minimal(SCALAR_MIN_FP16)
                tail = axis_count % self.data_each_vector
                if tail != 0:
                    _offset = axis_count // self.data_each_vector
                    if self.dtype_x == "float16":
                        mask_h, mask_l = _get_tail_mask_for_b16(tail)
                        self.tik_instance.vec_dup(
                            [mask_h, mask_l], ub_data_transpose[
                                axis_fetch_count * idx + _offset * self.data_each_vector],
                            SCALAR_MIN_FP16, 1, 8)
                    else:
                        mask_h, mask_l = _get_tail_mask_for_b32(tail)
                        self.tik_instance.vec_dup(
                            [mask_h, mask_l], ub_data_transpose[
                                axis_fetch_count * idx + _offset * self.data_each_vector],
                            SCALAR_MIN_FP32, 1, 8)

                # get repeat times
                repeat_times = _get_div_ceil_int(axis_count, self.data_each_vector)

                # find max value and index in each group
                self.tik_instance.vcmax(self.data_each_vector, ub_result_first[idx, 0],
                                        ub_data_transpose[axis_fetch_count * idx],
                                        repeat_times, 1, 1, 8)

                if self.dtype_x == "float16":
                    # for fp16
                    if repeat_times > 64:
                        # large than 64, use two more vcmax get the largest value and index
                        _repeat_times = _get_div_ceil_int(repeat_times, 64)
                        _repeat_tail = (repeat_times * 2) % self.data_each_vector
                        if _repeat_tail != 0:
                            mask_h, mask_l = _get_tail_mask_for_b16(_repeat_tail)
                            _offset = repeat_times * 2 // self.data_each_vector
                            self.tik_instance.vec_dup(
                                [mask_h, mask_l],
                                ub_result_first[idx, _offset * self.data_each_vector],
                                SCALAR_MIN_FP16, 1, 8)
                        self.tik_instance.vcmax([MASK_0_1,
                                                 MASK_0_1],
                                                ub_result_second[idx, 0], ub_result_first[idx, 0],
                                                _repeat_times, 1, 1, 8)

                        _mask = _calu_mask_by_one_zero(_repeat_times % 64)
                        self.tik_instance.vcmax(_mask,
                                                ub_result_third[idx, 0], ub_result_second[idx, 0],
                                                1, 1, 1, 8)
                        if self.out_max_val:
                            with self.tik_instance.if_scope(axis_loop == 0):
                                self.tik_instance.vec_adds(1, ub_result_value[idx, 0],
                                                           ub_result_third[idx, 0], 0, 1, 8, 8)
                            with self.tik_instance.else_scope():
                                self.tik_instance.vec_max(1, ub_result_value[idx, 0],
                                                          ub_result_value[idx, 0],
                                                          ub_result_third[idx, 0],
                                                          1, 8, 8, 8)

                    elif repeat_times > 1:
                        # less than 64, use one more vcmax get the largest value and index
                        _repeat_tail = repeat_times % 64
                        _mask = _calu_mask_by_one_zero(_repeat_tail)
                        if _repeat_tail == 0:
                            _mask = [MASK_0_1, MASK_0_1]
                        self.tik_instance.vcmax(_mask,
                                                ub_result_second[idx, 0], ub_result_first[idx, 0],
                                                1, 1, 1, 8)
                        if self.out_max_val:
                            with self.tik_instance.if_scope(axis_loop == 0):
                                self.tik_instance.vec_adds(1, ub_result_value[idx, 0],
                                                           ub_result_second[idx, 0], 0, 1, 8, 8)
                            with self.tik_instance.else_scope():
                                self.tik_instance.vec_max(1, ub_result_value[idx, 0],
                                                          ub_result_value[idx, 0],
                                                          ub_result_second[idx, 0],
                                                          1, 8, 8, 8)
                    else:
                        if self.out_max_val:
                            with self.tik_instance.if_scope(axis_loop == 0):
                                self.tik_instance.vec_adds(1, ub_result_value[idx, 0],
                                                           ub_result_first[idx, 0], 0, 1, 8, 8)
                            with self.tik_instance.else_scope():
                                self.tik_instance.vec_max(1, ub_result_value[idx, 0],
                                                          ub_result_value[idx, 0],
                                                          ub_result_first[idx, 0],
                                                          1, 8, 8, 8)
                else:
                    # for fp32
                    if repeat_times > 32:
                        # large than 32, use two more vcmax get the largest value and index
                        _repeat_times = _get_div_ceil_int(repeat_times, 32)
                        _repeat_tail = (repeat_times * 2) % self.data_each_vector
                        if _repeat_tail != 0:
                            mask_h, mask_l = _get_tail_mask_for_b32(_repeat_tail)
                            _offset = repeat_times * 2 // self.data_each_vector
                            self.tik_instance.vec_dup(
                                [mask_h, mask_l],
                                ub_result_first[idx, _offset * self.data_each_vector],
                                SCALAR_MIN_FP32, 1, 8)
                        self.tik_instance.vcmax([0,
                                                 MASK_0_1],
                                                ub_result_second[idx, 0], ub_result_first[idx, 0],
                                                _repeat_times, 1, 1, 8)

                        _mask = _calu_mask_by_one_zero(_repeat_times % 32)
                        self.tik_instance.vcmax(_mask,
                                                ub_result_third[idx, 0], ub_result_second[idx, 0],
                                                1, 1, 1, 8)
                        if self.out_max_val:
                            with self.tik_instance.if_scope(axis_loop == 0):
                                self.tik_instance.vec_adds(1, ub_result_value[idx, 0],
                                                           ub_result_third[idx, 0], 0, 1, 8, 8)
                            with self.tik_instance.else_scope():
                                self.tik_instance.vec_max(1, ub_result_value[idx, 0],
                                                          ub_result_value[idx, 0],
                                                          ub_result_third[idx, 0],
                                                          1, 8, 8, 8)

                    elif repeat_times > 1:
                        # less than 32, use one more vcmax get the largest value and index
                        _repeat_tail = repeat_times % 32
                        _mask = _calu_mask_by_one_zero(_repeat_tail)
                        if _repeat_tail == 0:
                            _mask = [0, MASK_0_1]
                        self.tik_instance.vcmax(_mask,
                                                ub_result_second[idx, 0], ub_result_first[idx, 0],
                                                1, 1, 1, 8)
                        if self.out_max_val:
                            with self.tik_instance.if_scope(axis_loop == 0):
                                self.tik_instance.vec_adds(1, ub_result_value[idx, 0],
                                                           ub_result_second[idx, 0], 0, 1, 8, 8)
                            with self.tik_instance.else_scope():
                                self.tik_instance.vec_max(1, ub_result_value[idx, 0],
                                                          ub_result_value[idx, 0],
                                                          ub_result_second[idx, 0],
                                                          1, 8, 8, 8)
                    else:
                        if self.out_max_val:
                            with self.tik_instance.if_scope(axis_loop == 0):
                                self.tik_instance.vec_adds(1, ub_result_value[idx, 0],
                                                           ub_result_first[idx, 0], 0, 1, 8, 8)
                            with self.tik_instance.else_scope():
                                self.tik_instance.vec_max(1, ub_result_value[idx, 0],
                                                          ub_result_value[idx, 0],
                                                          ub_result_first[idx, 0],
                                                          1, 8, 8, 8)

            # process the scalar in batch to improve performance
            if self.out_max_index:
                with self.tik_instance.for_range(0, ub_buf_size_last_dim) as idx:
                    scalar_type = "uint16" if self.dtype_x == "float16" else "uint32"
                    step_size = self.data_each_vector // 2

                    if repeat_num_2 > 1:
                        # three vcmax
                        # the third vcmax max index
                        third_max_index = self.tik_instance.Scalar(scalar_type)
                        third_max_index.set_as(ub_result_third[idx, 1])
                        # the second vcmax max index
                        second_max_index = self.tik_instance.Scalar(scalar_type)
                        second_max_index.set_as(ub_result_second[idx, third_max_index + 1])
                        # the first vcmax max index
                        last_max_index = self.tik_instance.Scalar(scalar_type)
                        last_max_index.set_as(
                            ub_result_first[idx, third_max_index * step_size +
                                            second_max_index + 1])
                        # the final vcmax max index in original data
                        max_index = self.tik_instance.Scalar(scalar_type)
                        max_index.set_as(
                            third_max_index * step_size * step_size +
                            second_max_index * step_size + last_max_index)
                    elif repeat_num_1 > 1:
                        # two vcmax
                        second_max_index = self.tik_instance.Scalar(scalar_type)
                        second_max_index.set_as(ub_result_second[idx, 1])
                        last_max_index = self.tik_instance.Scalar(scalar_type)
                        last_max_index.set_as(ub_result_first[idx, second_max_index + 1])
                        max_index = self.tik_instance.Scalar(scalar_type)
                        max_index.set_as(second_max_index * step_size + last_max_index)
                    else:
                        # one vcmax
                        # get the largest value and index directly
                        max_index = self.tik_instance.Scalar(scalar_type)
                        max_index.set_as(ub_result_first[idx, 1])

                    max_index_int32 = self.tik_instance.Scalar("int32")
                    max_index_int32.set_as(max_index)
                    with self.tik_instance.if_scope(axis_loop == 0):
                        ub_result_indices[idx].set_as(max_index_int32)
                        ub_result_value[idx, 0].set_as(
                            ub_data_transpose[axis_fetch_count * idx + max_index_int32])
                    with self.tik_instance.else_scope():
                        ub_result_cmp = self.tik_instance.Tensor(
                            self.dtype_x, (self.data_each_block,),
                            name="ub_result_cmp",
                            scope=tik.scope_ubuf)
                        ub_result_int32 = self.tik_instance.Tensor(
                            "int32", (self.index_each_block,),
                            name="ub_result_int32", scope=tik.scope_ubuf)
                        ub_result_cmp[0].set_as(ub_result_value[idx, 0])
                        ub_result_cmp[1].set_as(
                            ub_data_transpose[axis_fetch_count * idx + max_index_int32])
                        ub_result_int32[0].set_as(ub_result_indices[idx])
                        ub_result_int32[1].set_as(
                            max_index_int32 + self.task_segment_len * axis_loop)
                        self.tik_instance.vcmax(2, ub_result_cmp, ub_result_cmp,
                                                1, 1, 1, 8)
                        if self.dtype_x == "float16":
                            max_index1 = self.tik_instance.Scalar("uint16")
                        else:
                            max_index1 = self.tik_instance.Scalar("uint32")
                        max_index1.set_as(ub_result_cmp[1])
                        ub_result_indices[idx].set_as(ub_result_int32[max_index1])
                        ub_result_value[idx, 0].set_as(ub_result_cmp[0])

        if loop_times_axis != 0:
            with self.tik_instance.for_range(0, loop_times_axis) as axis_loop:
                inner_func(ub_buf_size_axis, axis_loop)
        if align_flag_axis:
            with self.tik_instance.new_stmt_scope():
                inner_func(over_size_axis, loop_times_axis)

        gm_offset = n_i * self.last_dim_size + loop_last_dim * self.task_segment_len_last_dim

        # store back the max value to gm
        if self.out_max_val:
            # the ub buffer to store the transposed results of segments assigned to this core
            ub_result_value_transposed = self.tik_instance.Tensor(
                self.dtype_x, (DATA_EACH_VNCHWCONV, last_dim_fetch_count),
                name="ub_result_value_transposed",
                scope=tik.scope_ubuf)

            # do the transpose
            row_loop_count = _get_div_ceil_int(last_dim_fetch_count, DATA_EACH_VNCHWCONV)
            col_loop_count = _get_div_ceil_int(DATA_EACH_VNCHWCONV, DATA_EACH_VNCHWCONV)

            # for max value transpose
            if self.dtype_x == "float16":
                # float16, 16*16
                with self.tik_instance.for_range(0, row_loop_count) as row_idx:
                    # should of length 16, but the elements can be duplicate,
                    # the region of each elements at most not overlap
                    dst_list = []
                    for i in range(DATA_EACH_VNCHWCONV):
                        dst_pos = last_dim_fetch_count * i + DATA_EACH_VNCHWCONV * row_idx
                        dst_list.append(ub_result_value_transposed[dst_pos])
                    # should of length 16, but the elements can be duplicate
                    src_list = []
                    for i in range(DATA_EACH_VNCHWCONV):
                        src_pos = DATA_EACH_VNCHWCONV * (DATA_EACH_VNCHWCONV * row_idx + i)
                        src_list.append(ub_result_value[src_pos])

                    if col_loop_count == 1:
                        self.tik_instance.vec_trans_scatter(False, False,
                                                            dst_list, src_list,
                                                            col_loop_count, 0, 0)
                    else:
                        self.tik_instance.vec_trans_scatter(
                            False, False, dst_list, src_list, col_loop_count,
                            DATA_EACH_VNCHWCONV * DATA_EACH_VNCHWCONV // self.data_each_block,
                            DATA_EACH_VNCHWCONV // self.data_each_block)
            else:
                # float32, 16*8
                with self.tik_instance.for_range(0, row_loop_count) as row_idx:
                    # should of length 16, but the elements can be duplicate,
                    # the region of each elements at most not overlap
                    dst_list = []
                    for i in range(DATA_EACH_VNCHWCONV):
                        dst_pos = last_dim_fetch_count * (i // 2) + \
                                  (i % 2) * DATA_EACH_VNCHWCONV // 2 + \
                                  DATA_EACH_VNCHWCONV * row_idx
                        dst_list.append(ub_result_value_transposed[dst_pos])
                    # should of length 16, but the elements can be duplicate
                    src_list = []
                    for i in range(DATA_EACH_VNCHWCONV):
                        src_pos = DATA_EACH_VNCHWCONV * (DATA_EACH_VNCHWCONV * row_idx + i)
                        src_list.append(ub_result_value[src_pos])

                    self.tik_instance.vec_trans_scatter(
                        False, False, dst_list, src_list, 2 * col_loop_count,
                        DATA_EACH_VNCHWCONV // 2 * DATA_EACH_VNCHWCONV // self.data_each_block,
                        DATA_EACH_VNCHWCONV // 2 // self.data_each_block)

            if elements_to_write < self.data_each_block:
                self.tik_instance.data_move(
                    self.result_gm_value[gm_offset], ub_result_value_transposed, 0, 1,
                    _get_div_ceil_int(ub_buf_size_last_dim, self.data_each_block), 0, 0)
            else:
                with self.tik_instance.if_scope(loop_last_dim < loop_last_dim_max):
                    self.tik_instance.data_move(
                        self.result_gm_value[gm_offset], ub_result_value_transposed, 0, 1,
                        _get_div_ceil_int(ub_buf_size_last_dim, self.data_each_block), 0, 0)
                with self.tik_instance.else_scope():
                    bias = element_end_address_to_write - gm_offset - \
                           _get_round_ceil_int(ub_buf_size_last_dim, self.data_each_block)
                    with self.tik_instance.if_scope(bias >= 0):
                        self.tik_instance.data_move(
                            self.result_gm_value[gm_offset], ub_result_value_transposed, 0, 1,
                            _get_div_ceil_int(ub_buf_size_last_dim, self.data_each_block), 0, 0)
                    with self.tik_instance.else_scope():
                        burst_len_floor = ub_buf_size_last_dim // self.data_each_block
                        ub_elem_left = ub_buf_size_last_dim % self.data_each_block
                        ub_elem_left_start = burst_len_floor * self.data_each_block
                        if burst_len_floor > 0:
                            self.tik_instance.data_move(self.result_gm_value[gm_offset],
                                                        ub_result_value_transposed, 0, 1,
                                                        burst_len_floor, 0, 0)
                        result_out_scalar_tmp = self.tik_instance.Tensor(
                            self.dtype_x, (self.data_each_block,),
                            name="result_out_scalar_tmp",
                            scope=tik.scope_ubuf)
                        self.tik_instance.data_move(
                            result_out_scalar_tmp,
                            self.result_gm_value[element_end_address_to_write -
                                                 self.data_each_block],
                            0, 1, 1, 0, 0)
                        value_reg = self.tik_instance.Scalar(dtype=self.dtype_x)
                        with self.tik_instance.for_range(0, ub_elem_left) as i:
                            value_reg.set_as(ub_result_value_transposed[ub_elem_left_start + i])
                            result_out_scalar_tmp[i - bias].set_as(value_reg)
                        self.tik_instance.data_move(
                            self.result_gm_value[element_end_address_to_write -
                                                 self.data_each_block],
                            result_out_scalar_tmp, 0, 1, 1, 0, 0)

        # store back the max index to gm
        if self.out_max_index:
            if elements_to_write < self.index_each_block:
                self.tik_instance.data_move(
                    self.result_gm[gm_offset], ub_result_indices, 0, 1,
                    _get_div_ceil_int(ub_buf_size_last_dim, self.index_each_block), 0, 0)
            else:
                with self.tik_instance.if_scope(loop_last_dim < loop_last_dim_max):
                    self.tik_instance.data_move(
                        self.result_gm[gm_offset], ub_result_indices, 0, 1,
                        _get_div_ceil_int(ub_buf_size_last_dim, self.index_each_block), 0, 0)
                with self.tik_instance.else_scope():
                    bias = element_end_address_to_write - gm_offset - \
                           _get_round_ceil_int(ub_buf_size_last_dim, self.index_each_block)
                    with self.tik_instance.if_scope(bias >= 0):
                        self.tik_instance.data_move(
                            self.result_gm[gm_offset], ub_result_indices, 0, 1,
                            _get_div_ceil_int(ub_buf_size_last_dim, self.index_each_block), 0, 0)
                    with self.tik_instance.else_scope():
                        burst_len_floor = ub_buf_size_last_dim // self.index_each_block
                        ub_elem_left = ub_buf_size_last_dim % self.index_each_block
                        ub_elem_left_start = burst_len_floor * self.index_each_block
                        if burst_len_floor > 0:
                            self.tik_instance.data_move(self.result_gm[gm_offset],
                                                        ub_result_indices, 0, 1,
                                                        burst_len_floor, 0, 0)
                        result_int32_tmp = self.tik_instance.Tensor(
                            "int32", (self.index_each_block,),
                            name="result_int32_tmp",
                            scope=tik.scope_ubuf)
                        self.tik_instance.data_move(
                            result_int32_tmp,
                            self.result_gm[element_end_address_to_write - self.index_each_block],
                            0, 1, 1, 0, 0)
                        index_reg = self.tik_instance.Scalar(dtype="int32")
                        with self.tik_instance.for_range(0, ub_elem_left) as i:
                            index_reg.set_as(ub_result_indices[ub_elem_left_start + i])
                            result_int32_tmp[i - bias].set_as(index_reg)
                        self.tik_instance.data_move(
                            self.result_gm[element_end_address_to_write - self.index_each_block],
                            result_int32_tmp, 0, 1, 1, 0, 0)

    # split into multiple cores, each core handles multiple segments
    def argmax_last_axis(self):
        """
        scedule then do last axis

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        core_number = self.product_core_num
        # the result size less than a block, only one core to avoid result override between cores
        block_factor = self.data_each_block if self.out_max_val else self.index_each_block
        if self.gm_result_size <= block_factor:
            core_number = 1
        core_number_all = self.first_dim_size
        # the segments each core assigned
        core_segment = _get_div_ceil_int(core_number_all, core_number)
        # the elements each core handles, each core at least handle a result block
        factor = self.topk
        if core_segment * factor < block_factor:
            core_segment = _get_round_ceil_int(core_segment * factor, block_factor)
            core_segment = _get_div_ceil_int(core_segment, factor)
        # the core count used
        core_num_used = _get_div_ceil_int(core_number_all, core_segment)
        # the elements the last core handle
        core_segment_tail = core_number_all % core_segment

        # the segment len to split works in a task
        if self.dtype_x == "float16":
            self.task_segment_len = 1024 * 32
        else:
            self.task_segment_len = 1024 * 16

        # the segment len to split the last dim in a task
        self.task_segment_len_last_dim = DATA_EACH_VNCHWCONV

        # use vcmax to get the top_1 result
        # use block_num to parallel handling
        with self.tik_instance.for_range(
                0, core_num_used, block_num=core_num_used) as n_i:
            if core_segment_tail == 0:
                self.compute_argmax_last_axis(n_i, core_segment, core_segment)
            else:
                with self.tik_instance.if_scope(n_i < (core_num_used - 1)):
                    self.compute_argmax_last_axis(n_i, core_segment,
                                                  core_segment)
                # the last core handle less elements than other cores
                with self.tik_instance.else_scope():
                    self.compute_argmax_last_axis(n_i, core_segment_tail,
                                                  core_segment)

    # each core handle several segments
    def compute_argmax_last_axis(self, n_i, core_segment, segment_core):
        """
        compute arg when do last axis

        Parameters
        ----------
        n_i : int
            the first loop index
        core_segment : int
            segment process len
        segment_core : int
            the total segment index

        Returns
        -------
        None
        """
        # split the last axis dim into loops
        ub_buf_size, loop_times, over_size, align_flag = \
            get_tiling_info_for_axis(self.axis_size, self.task_segment_len)
        # use 16 for vec_trans_scatter
        CORE_SEGMENT_LEN_DB = DATA_EACH_VNCHWCONV

        # use to store the first vcmax result, the self.task_segment_len limit to 16K
        seg_len = min(self.task_segment_len, self.axis_size)
        repeat_num_1 = _get_div_ceil_int(seg_len, self.data_each_vector)
        repeat_num_2 = _get_div_ceil_int(repeat_num_1, self.data_each_vector // 2)

        def _run(segment_len, segment_index):
            # the ub buffer used to store the first vcmax results
            ub_result_first = self.tik_instance.Tensor(
                self.dtype_x, (CORE_SEGMENT_LEN_DB, repeat_num_2 * self.data_each_vector,),
                name="ub_result_first",
                scope=tik.scope_ubuf)

            # the ub buffer used to store the second vcmax results
            ub_result_second = self.tik_instance.Tensor(
                self.dtype_x, (CORE_SEGMENT_LEN_DB, 2 * self.data_each_block,),
                name="ub_result_second",
                scope=tik.scope_ubuf)

            # the ub buffer used to store the third vcmax results
            ub_result_third = self.tik_instance.Tensor(
                self.dtype_x, (CORE_SEGMENT_LEN_DB, self.data_each_block,),
                name="ub_result_third",
                scope=tik.scope_ubuf)

            # the ub buffer used to store the final results of segments assigned to this core
            ub_result_indices = self.tik_instance.Tensor(
                "int32", (CORE_SEGMENT_LEN_DB,),
                name="ub_result_indices",
                scope=tik.scope_ubuf)
            ub_result_value = self.tik_instance.Tensor(
                self.dtype_x, (CORE_SEGMENT_LEN_DB, DATA_EACH_VNCHWCONV),
                name="ub_result_value",
                scope=tik.scope_ubuf)

            if loop_times > 1:
                thread_2 = 2
            else:
                thread_2 = 1
            if thread_2 == 1 and segment_len > 1:
                thread_1 = 2
            else:
                thread_1 = 1

            if loop_times == 0 or (loop_times == 1 and not align_flag):
                need_merge = False
            else:
                need_merge = True

            with self.tik_instance.for_range(0, segment_len, thread_num=thread_1) as core_i:
                index = core_i + CORE_SEGMENT_LEN_DB * segment_index
                offset = n_i * segment_core + index
                argmax_func = self.do_argmax_last_axis
                if loop_times != 0:
                    with self.tik_instance.for_range(0, loop_times, thread_num=thread_2) as loop:
                        argmax_func(ub_buf_size, loop, offset, need_merge,
                                    ub_result_first[core_i, 0],
                                    ub_result_second[core_i, 0],
                                    ub_result_third[core_i, 0],
                                    ub_result_indices[core_i],
                                    ub_result_value[core_i, 0])
                if align_flag:
                    with self.tik_instance.new_stmt_scope():
                        argmax_func(over_size, loop_times, offset, need_merge,
                                    ub_result_first[core_i, 0],
                                    ub_result_second[core_i, 0],
                                    ub_result_third[core_i, 0],
                                    ub_result_indices[core_i],
                                    ub_result_value[core_i, 0])

            gm_out_offset = n_i * segment_core + CORE_SEGMENT_LEN_DB * segment_index
            # store the indices
            if self.out_max_index:
                if not need_merge:
                    scalar_type = "uint16" if self.dtype_x == "float16" else "uint32"
                    step_size = self.data_each_vector // 2
                    with self.tik_instance.for_range(0, segment_len) as core_i:
                        if repeat_num_2 > 1:
                            # three vcmax
                            # the third vcmax max index
                            third_max_index = self.tik_instance.Scalar(scalar_type)
                            third_max_index.set_as(ub_result_third[core_i, 1])
                            # the second vcmax max index
                            second_max_index = self.tik_instance.Scalar(scalar_type)
                            second_max_index.set_as(ub_result_second[core_i, third_max_index + 1])
                            # the first vcmax max index
                            last_max_index = self.tik_instance.Scalar(scalar_type)
                            last_max_index.set_as(
                                ub_result_first[core_i,
                                                third_max_index * step_size + second_max_index + 1])
                            # the final vcmax max index in original data
                            ub_result_indices[core_i].set_as(
                                third_max_index * step_size * step_size +
                                second_max_index * step_size + last_max_index)
                            ub_result_value[core_i, 0].set_as(ub_result_third[core_i, 0])
                        elif repeat_num_1 > 1:
                            # two vcmax
                            second_max_index = self.tik_instance.Scalar(scalar_type)
                            second_max_index.set_as(ub_result_second[core_i, 1])
                            last_max_index = self.tik_instance.Scalar(scalar_type)
                            last_max_index.set_as(ub_result_first[core_i, second_max_index + 1])
                            ub_result_indices[core_i].set_as(second_max_index * step_size +
                                                             last_max_index)
                            ub_result_value[core_i, 0].set_as(ub_result_second[core_i, 0])
                        else:
                            # one vcmax
                            # get the largest value and index directly
                            max_index = self.tik_instance.Scalar(scalar_type)
                            max_index.set_as(ub_result_first[core_i, 1])
                            ub_result_indices[core_i].set_as(max_index)
                            ub_result_value[core_i, 0].set_as(ub_result_first[core_i, 0])

                if segment_len % self.index_each_block == 0:
                    out_nbust = _get_div_ceil_int(segment_len, self.index_each_block)
                    self.tik_instance.data_move(self.result_gm[gm_out_offset],
                                                ub_result_indices, 0, 1,
                                                out_nbust, 0, 0)
                elif core_segment < self.index_each_block:
                    # this is the last core
                    out_nbust = _get_div_ceil_int(segment_len, self.index_each_block)
                    self.tik_instance.data_move(self.result_gm[gm_out_offset],
                                                ub_result_indices, 0, 1,
                                                out_nbust, 0, 0)
                else:
                    burst_len_floor = segment_len // self.index_each_block
                    ub_elem_left = segment_len % self.index_each_block
                    ub_elem_left_start = burst_len_floor * self.index_each_block
                    bias = self.index_each_block - ub_elem_left
                    if burst_len_floor > 0:
                        self.tik_instance.data_move(self.result_gm[gm_out_offset],
                                                    ub_result_indices, 0, 1, burst_len_floor, 0, 0)
                    ub_result_value_tmp = self.tik_instance.Tensor(
                        "int32", (self.index_each_block,),
                        name="ub_result_value_tmp",
                        scope=tik.scope_ubuf)
                    self.tik_instance.data_move(
                        ub_result_value_tmp,
                        self.result_gm[gm_out_offset + segment_len - self.index_each_block],
                        0, 1, 1, 0, 0)
                    index_reg = self.tik_instance.Scalar(dtype="int32")
                    with self.tik_instance.for_range(0, ub_elem_left) as i:
                        index_reg.set_as(ub_result_indices[ub_elem_left_start + i])
                        ub_result_value_tmp[i + bias].set_as(index_reg)
                    self.tik_instance.data_move(
                        self.result_gm[gm_out_offset + segment_len - self.index_each_block],
                        ub_result_value_tmp, 0, 1, 1, 0, 0)

            # store the values
            if self.out_max_val:
                # the ub buffer to store the transposed results of segments assigned to this core
                ub_result_value_transposed = self.tik_instance.Tensor(
                    self.dtype_x, (DATA_EACH_VNCHWCONV, CORE_SEGMENT_LEN_DB),
                    name="ub_result_value_transposed",
                    scope=tik.scope_ubuf)

                # do the transpose
                row_loop_count = _get_div_ceil_int(CORE_SEGMENT_LEN_DB, DATA_EACH_VNCHWCONV)
                col_loop_count = _get_div_ceil_int(DATA_EACH_VNCHWCONV, DATA_EACH_VNCHWCONV)

                # for max value transpose
                if self.dtype_x == "float16":
                    # float16, 16*16
                    with self.tik_instance.for_range(0, row_loop_count) as row_idx:
                        # should of length 16, but the elements can be duplicate,
                        # the region of each elements at most not overlap
                        dst_list = []
                        for i in range(DATA_EACH_VNCHWCONV):
                            dst_pos = CORE_SEGMENT_LEN_DB * i + DATA_EACH_VNCHWCONV * row_idx
                            dst_list.append(ub_result_value_transposed[dst_pos])
                        # should of length 16, but the elements can be duplicate
                        src_list = []
                        for i in range(DATA_EACH_VNCHWCONV):
                            src_pos = DATA_EACH_VNCHWCONV * (DATA_EACH_VNCHWCONV * row_idx + i)
                            src_list.append(ub_result_value[src_pos])

                        if col_loop_count == 1:
                            self.tik_instance.vec_trans_scatter(False, False,
                                                                dst_list, src_list,
                                                                col_loop_count, 0, 0)
                        else:
                            self.tik_instance.vec_trans_scatter(
                                False, False, dst_list, src_list, col_loop_count,
                                DATA_EACH_VNCHWCONV * CORE_SEGMENT_LEN_DB // self.data_each_block,
                                DATA_EACH_VNCHWCONV // self.data_each_block)
                else:
                    # float32, 16*8
                    with self.tik_instance.for_range(0, row_loop_count) as row_idx:
                        # should of length 16, but the elements can be duplicate,
                        # the region of each elements at most not overlap
                        dst_list = []
                        for i in range(DATA_EACH_VNCHWCONV):
                            dst_pos = CORE_SEGMENT_LEN_DB * (i // 2) + \
                                      (i % 2) * DATA_EACH_VNCHWCONV // 2 + \
                                      DATA_EACH_VNCHWCONV * row_idx
                            dst_list.append(ub_result_value_transposed[dst_pos])
                        # should of length 16, but the elements can be duplicate
                        src_list = []
                        for i in range(DATA_EACH_VNCHWCONV):
                            src_pos = DATA_EACH_VNCHWCONV * (DATA_EACH_VNCHWCONV * row_idx + i)
                            src_list.append(ub_result_value[src_pos])

                        self.tik_instance.vec_trans_scatter(
                            False, False, dst_list, src_list, 2 *col_loop_count,
                            DATA_EACH_VNCHWCONV // 2 * CORE_SEGMENT_LEN_DB // self.data_each_block,
                            DATA_EACH_VNCHWCONV // 2 // self.data_each_block)

                if segment_len % self.data_each_block == 0:
                    out_nbust = _get_div_ceil_int(segment_len, self.data_each_block)
                    self.tik_instance.data_move(self.result_gm_value[gm_out_offset],
                                                ub_result_value_transposed, 0, 1,
                                                out_nbust, 0, 0)
                elif core_segment < self.data_each_block:
                    # this is the last core
                    out_nbust = _get_div_ceil_int(segment_len, self.data_each_block)
                    self.tik_instance.data_move(self.result_gm_value[gm_out_offset],
                                                ub_result_value_transposed, 0, 1,
                                                out_nbust, 0, 0)
                else:
                    burst_len_floor = segment_len // self.data_each_block
                    ub_elem_left = segment_len % self.data_each_block
                    ub_elem_left_start = burst_len_floor * self.data_each_block
                    bias = self.data_each_block - ub_elem_left
                    if burst_len_floor > 0:
                        self.tik_instance.data_move(self.result_gm_value[gm_out_offset],
                                                    ub_result_value_transposed, 0, 1,
                                                    burst_len_floor, 0, 0)
                    ub_result_value_tmp = self.tik_instance.Tensor(
                        self.dtype_x, (self.data_each_block,),
                        name="ub_result_value_tmp",
                        scope=tik.scope_ubuf)
                    self.tik_instance.data_move(
                        ub_result_value_tmp,
                        self.result_gm_value[gm_out_offset + segment_len - self.data_each_block],
                        0, 1, 1, 0, 0)
                    value_reg = self.tik_instance.Scalar(dtype=self.dtype_x)
                    with self.tik_instance.for_range(0, ub_elem_left) as i:
                        value_reg.set_as(ub_result_value_transposed[ub_elem_left_start + i])
                        ub_result_value_tmp[i + bias].set_as(value_reg)
                    self.tik_instance.data_move(
                        self.result_gm_value[gm_out_offset + segment_len - self.data_each_block],
                        ub_result_value_tmp, 0, 1, 1, 0, 0)

        _loop_segment = core_segment // CORE_SEGMENT_LEN_DB
        _loop_segment_tail = core_segment % CORE_SEGMENT_LEN_DB
        if _loop_segment != 0:
            with self.tik_instance.for_range(0, _loop_segment) as _loop:
                _run(CORE_SEGMENT_LEN_DB, _loop)
        if _loop_segment_tail != 0:
            _run(_loop_segment_tail, _loop_segment)

    def do_argmax_last_axis(self, ub_buf_size, loop, n_i, need_merge,
                            ub_result_first, ub_result_second, ub_result_third,
                            result_int32, result_out_scalar):
        """
        do arg in one segment for float16

        Parameters
        ----------
        ub_buf_size : int
            process len
        loop : int
            segment index in one core
        n_i : int
            the first loop index

        Returns
        -------
        None
        """
        if need_merge:
            # use to store the first vcmax result
            repeat_num = _get_div_ceil_int(ub_buf_size, self.data_each_vector)
            step_num = self.data_each_vector // 2
            repeat_num = _get_div_ceil_int(repeat_num, step_num)
            ub_result = self.tik_instance.Tensor(
                self.dtype_x, (repeat_num * self.data_each_vector,),
                name="ub_result",
                scope=tik.scope_ubuf)

            ub_second_result = self.tik_instance.Tensor(
                self.dtype_x, (2 * self.data_each_block,),
                name="ub_second_result",
                scope=tik.scope_ubuf)

            ub_third_result = self.tik_instance.Tensor(
                self.dtype_x, (self.data_each_block,),
                name="ub_third_result",
                scope=tik.scope_ubuf)
        else:
            ub_result = ub_result_first
            ub_second_result = ub_result_second
            ub_third_result = ub_result_third

        # the original input data
        ub_data = self.tik_instance.Tensor(
            self.dtype_x, (_get_round_ceil_int(ub_buf_size, self.data_each_vector),),
            name="ub_data",
            scope=tik.scope_ubuf)
        # the offset in the original input data
        offset = loop * self.task_segment_len + n_i * self.axis_size
        # data move from gm to ub
        self.tik_instance.data_move(ub_data, self.data_gm[offset], 0, 1,
                                    _get_div_ceil_int(ub_buf_size, self.data_each_block), 0, 0)

        # if has tail, set the data in the tail part as minimal(SCALAR_MIN_FP16)
        tail = ub_buf_size % self.data_each_vector
        if tail != 0:
            _offset = ub_buf_size // (self.data_each_vector)
            if self.dtype_x == "float16":
                mask_h, mask_l = _get_tail_mask_for_b16(tail)
                self.tik_instance.vec_dup(
                    [mask_h, mask_l], ub_data[_offset * self.data_each_vector],
                    SCALAR_MIN_FP16, 1, 8)
            else:
                mask_h, mask_l = _get_tail_mask_for_b32(tail)
                self.tik_instance.vec_dup(
                    [mask_h, mask_l], ub_data[_offset * self.data_each_vector],
                    SCALAR_MIN_FP32, 1, 8)

        # get repeat times
        repeat_times = _get_div_ceil_int(ub_buf_size, self.data_each_vector)
        if repeat_times < 256:
            # find max value and index in each group
            self.tik_instance.vcmax(self.data_each_vector, ub_result, ub_data,
                                    repeat_times, 1, 1, 8)
        else:
            # find max value and index in each group
            self.tik_instance.vcmax(self.data_each_vector, ub_result, ub_data,
                                    128, 1, 1, 8)
            # find max value and index in each group
            self.tik_instance.vcmax(self.data_each_vector, ub_result[256],
                                    ub_data[128 * self.data_each_vector],
                                    repeat_times - 128, 1, 1, 8)

        if self.dtype_x == "float16":
            # for fp16
            if repeat_times > 64:
                # large than 64, use two more vcmax get the largest value and index
                _repeat_times = _get_div_ceil_int(repeat_times, 64)
                _repeat_tail = (repeat_times * 2) % self.data_each_vector
                if _repeat_tail != 0:
                    mask_h, mask_l = _get_tail_mask_for_b16(_repeat_tail)
                    _offset = repeat_times * 2 // self.data_each_vector
                    self.tik_instance.vec_dup(
                        [mask_h, mask_l],
                        ub_result[_offset * self.data_each_vector],
                        SCALAR_MIN_FP16, 1, 8)

                self.tik_instance.vcmax([MASK_0_1,
                                         MASK_0_1],
                                        ub_second_result, ub_result,
                                        _repeat_times, 1, 1, 8)

                _mask = _calu_mask_by_one_zero(_repeat_times % 64)
                self.tik_instance.vcmax(_mask,
                                        ub_third_result, ub_second_result,
                                        1, 1, 1, 8)
                if self.out_max_index:
                    if need_merge:
                        # the third vcmax max index
                        third_max_index = self.tik_instance.Scalar("uint16")
                        third_max_index.set_as(ub_third_result[1])
                        # the second vcmax max index
                        second_max_index = self.tik_instance.Scalar("uint16")
                        second_max_index.set_as(ub_second_result[third_max_index + 1])
                        # the first vcmax max index
                        last_max_index = self.tik_instance.Scalar("uint16")
                        last_max_index.set_as(
                            ub_result[third_max_index * 64 + second_max_index + 1])
                        # the final vcmax max index in original data
                        max_index = self.tik_instance.Scalar("uint16")
                        max_index.set_as(
                            third_max_index * 64 * 64 + second_max_index * 64 + last_max_index)
                else:
                    if need_merge:
                        with self.tik_instance.if_scope(loop == 0):
                            self.tik_instance.vec_adds(1, result_out_scalar, ub_third_result,
                                                       0, 1, 8, 8)
                        with self.tik_instance.else_scope():
                            self.tik_instance.vec_max(1, result_out_scalar, result_out_scalar,
                                                      ub_third_result, 1, 8, 8, 8)
                    else:
                        self.tik_instance.vec_adds(1, result_out_scalar, ub_third_result,
                                                   0, 1, 8, 8)

            elif repeat_times > 1:
                # less than 64, use one more vcmax get the largest value and index
                _repeat_tail = repeat_times % 64
                _mask = _calu_mask_by_one_zero(_repeat_tail)
                if _repeat_tail == 0:
                    _mask = [MASK_0_1, MASK_0_1]
                self.tik_instance.vcmax(_mask,
                                        ub_second_result, ub_result,
                                        1, 1, 1, 8)
                if self.out_max_index:
                    if need_merge:
                        second_max_index = self.tik_instance.Scalar("uint16")
                        second_max_index.set_as(ub_second_result[1])
                        last_max_index = self.tik_instance.Scalar("uint16")
                        last_max_index.set_as(ub_result[second_max_index + 1])
                        max_index = self.tik_instance.Scalar("uint16")
                        # aka (second_max_index / 2 * 128 + last_max_index)
                        max_index.set_as(second_max_index * 64 + last_max_index)
                else:
                    if need_merge:
                        with self.tik_instance.if_scope(loop == 0):
                            self.tik_instance.vec_adds(1, result_out_scalar, ub_second_result,
                                                       0, 1, 8, 8)
                        with self.tik_instance.else_scope():
                            self.tik_instance.vec_max(1, result_out_scalar, result_out_scalar,
                                                      ub_second_result, 1, 8, 8, 8)
                    else:
                        self.tik_instance.vec_adds(1, result_out_scalar, ub_second_result,
                                                   0, 1, 8, 8)

            else:
                if self.out_max_index:
                    if need_merge:
                        # get the largest value and index directly
                        max_index = self.tik_instance.Scalar("uint16")
                        max_index.set_as(ub_result[1])
                else:
                    if need_merge:
                        with self.tik_instance.if_scope(loop == 0):
                            self.tik_instance.vec_adds(1, result_out_scalar, ub_result, 0, 1, 8, 8)
                        with self.tik_instance.else_scope():
                            self.tik_instance.vec_max(1, result_out_scalar, result_out_scalar,
                                                      ub_result, 1, 8, 8, 8)
                    else:
                        self.tik_instance.vec_adds(1, result_out_scalar, ub_result, 0, 1, 8, 8)
        else:
            # for fp32
            if repeat_times > 32:
                # large than 32, use two more vcmax get the largest value and index
                _repeat_times = _get_div_ceil_int(repeat_times, 32)
                _repeat_tail = (repeat_times * 2) % self.data_each_vector
                if _repeat_tail != 0:
                    mask_h, mask_l = _get_tail_mask_for_b32(_repeat_tail)
                    _offset = repeat_times * 2 // self.data_each_vector
                    self.tik_instance.vec_dup(
                        [mask_h, mask_l],
                        ub_result[_offset * self.data_each_vector],
                        SCALAR_MIN_FP32, 1, 8)
                self.tik_instance.vcmax([0,
                                         MASK_0_1],
                                        ub_second_result, ub_result,
                                        _repeat_times, 1, 1, 8)

                _mask = _calu_mask_by_one_zero(_repeat_times % 32)
                self.tik_instance.vcmax(_mask,
                                        ub_third_result, ub_second_result,
                                        1, 1, 1, 8)
                if self.out_max_index:
                    if need_merge:
                        # the third vcmax max index
                        third_max_index = self.tik_instance.Scalar("uint32")
                        third_max_index.set_as(ub_third_result[1])
                        # the second vcmax max index
                        second_max_index = self.tik_instance.Scalar("uint32")
                        second_max_index.set_as(ub_second_result[third_max_index + 1])
                        # the first vcmax max index
                        last_max_index = self.tik_instance.Scalar("uint32")
                        last_max_index.set_as(
                            ub_result[third_max_index * 32 + second_max_index + 1])
                        # the final vcmax max index in original data
                        max_index = self.tik_instance.Scalar("uint32")
                        max_index.set_as(
                            third_max_index * 32 * 32 + second_max_index * 32 + last_max_index)
                else:
                    if need_merge:
                        with self.tik_instance.if_scope(loop == 0):
                            self.tik_instance.vec_adds(1, result_out_scalar, ub_third_result,
                                                       0, 1, 8, 8)
                        with self.tik_instance.else_scope():
                            self.tik_instance.vec_max(1, result_out_scalar, result_out_scalar,
                                                      ub_third_result, 1, 8, 8, 8)
                    else:
                        self.tik_instance.vec_adds(1, result_out_scalar, ub_third_result,
                                                   0, 1, 8, 8)

            elif repeat_times > 1:
                # less than 32, use one more vcmax get the largest value and index
                _repeat_tail = repeat_times % 32
                _mask = _calu_mask_by_one_zero(_repeat_tail)
                if _repeat_tail == 0:
                    _mask = [0, MASK_0_1]
                self.tik_instance.vcmax(_mask,
                                        ub_second_result, ub_result,
                                        1, 1, 1, 8)
                if self.out_max_index:
                    if need_merge:
                        second_max_index = self.tik_instance.Scalar("uint32")
                        second_max_index.set_as(ub_second_result[1])
                        last_max_index = self.tik_instance.Scalar("uint32")
                        last_max_index.set_as(ub_result[second_max_index + 1])
                        max_index = self.tik_instance.Scalar("uint32")
                        # aka (second_max_index / 2 * 64 + last_max_index)
                        max_index.set_as(second_max_index * 32 + last_max_index)
                else:
                    if need_merge:
                        with self.tik_instance.if_scope(loop == 0):
                            self.tik_instance.vec_adds(1, result_out_scalar, ub_second_result,
                                                       0, 1, 8, 8)
                        with self.tik_instance.else_scope():
                            self.tik_instance.vec_max(1, result_out_scalar, result_out_scalar,
                                                      ub_second_result, 1, 8, 8, 8)
                    else:
                        self.tik_instance.vec_adds(1, result_out_scalar, ub_second_result,
                                                   0, 1, 8, 8)

            else:
                if self.out_max_index:
                    if need_merge:
                        # get the largest value and index directly
                        max_index = self.tik_instance.Scalar("uint32")
                        max_index.set_as(ub_result[1])
                else:
                    if need_merge:
                        with self.tik_instance.if_scope(loop == 0):
                            self.tik_instance.vec_adds(1, result_out_scalar, ub_result, 0, 1, 8, 8)
                        with self.tik_instance.else_scope():
                            self.tik_instance.vec_max(1, result_out_scalar, result_out_scalar,
                                                      ub_result, 1, 8, 8, 8)
                    else:
                        self.tik_instance.vec_adds(1, result_out_scalar, ub_result, 0, 1, 8, 8)

        if self.out_max_index:
            if need_merge:
                max_index_int32 = self.tik_instance.Scalar("int32")
                max_index_int32.set_as(max_index)

                with self.tik_instance.if_scope(loop == 0):
                    result_int32.set_as(max_index_int32)
                    result_out_scalar.set_as(ub_data[max_index_int32])
                with self.tik_instance.else_scope():
                    ub_result_cmp = self.tik_instance.Tensor(
                        self.dtype_x, (self.data_each_block,),
                        name="ub_result_cmp",
                        scope=tik.scope_ubuf)
                    # use to store the max index in this loop
                    ub_result_int32 = self.tik_instance.Tensor(
                        "int32", (self.index_each_block,),
                        name="ub_result_int32", scope=tik.scope_ubuf)
                    ub_result_cmp[0].set_as(result_out_scalar)
                    ub_result_cmp[1].set_as(ub_data[max_index_int32])
                    ub_result_int32[0].set_as(result_int32)
                    ub_result_int32[1].set_as(max_index_int32 + loop * self.task_segment_len)
                    # ub_result_cmp[0] is max value, ub_result_cmp[1] is max index
                    self.tik_instance.vcmax(2, ub_result_cmp, ub_result_cmp,
                                            1, 1, 1, 8)
                    if self.dtype_x == "float16":
                        max_index1 = self.tik_instance.Scalar("uint16")
                    else:
                        max_index1 = self.tik_instance.Scalar("uint32")
                    max_index1.set_as(ub_result_cmp[1])
                    result_int32.set_as(ub_result_int32[max_index1])
                    result_out_scalar.set_as(ub_result_cmp[0])
