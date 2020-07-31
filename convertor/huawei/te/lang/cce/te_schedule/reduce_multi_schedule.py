#!/usr/bin/env python
# -*- coding: UTF-8 -*-
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

elewise mutil out schedule
"""
from te import platform as cceconf
from te import tvm
from . import util
from .elewise_schedule_new import ElewiseSchedule
from .cce_schedule_declarations import OpSpecTypes

BLOCK_TILING_PRIME_THRESHOLD = 67


# pylint: disable=too-many-return-statements,too-few-public-methods,too-many-arguments,too-many-statements,no-self-use,too-many-lines,too-many-instance-attributes,too-many-branches,
class ReduceMultiSchedule(ElewiseSchedule):
    """
    class of cce elewise schedule

    Parameters
    ----------
    VectorSchedule: base class of elewise schedule

    Returns
    -------
    ElewiseSchedule_instance : instance of ElewiseSchedule
    """

    # pylint: disable=attribute-defined-outside-init, unused-argument, arguments-differ
    def do_schedule(self, out_tensors, sch, spec_node_list):
        """
        auto_schedule for cce AI-CORE

        Parameters
        ----------
        outTensors : the out tvm.tensor

        sch : schedule, the computation schedule for the op

        spec_node_list : special node list

        Returns
        -------
        Bool, now is true

        """
        # use this attr to identify itself
        self._unique_name_reduce_multi = None

        if (not util.MULTI_REDUCE) or self.__pre_complement_tensors_map(out_tensors):
            return False, None

        self._construct_compute_graph(out_tensors, spec_node_list)

        if not self._calculate_tiling():
            return False, None

        self._calculate_cache_read()
        self._do_cache_read()

        self._calculate_cache_write()
        self._do_cache_write()

        self._do_tiling()

        self._calculate_compute_inline()
        self._do_compute_inline()

        self._calculate_compute_at()
        self._do_compute_at()

        self._do_multi_core()
        self._do_storage_align()
        self._do_buffer_reuse()

        self._calculate_emit_insn()
        self._do_emit_insn()

        self._do_double_buffer()

        return True, self._schedule

    def __pre_complement_tensors_map(self, out_tensors):
        # pylint: disable=attribute-defined-outside-init
        """
        pre handle syntax tree by replace compute node
        use fake node to make it into one out schedule

        Parameters
        ----------
        outTensors : the out tvm.tensor

        Returns
        -------
        Schedule, mock schedule

        """
        # temp mid output tensors dst tensor map
        dst_tensor_map = {}
        temp_mid_output_tensors_in_ub = []
        self._mid_output_tensors_in_gm = []
        # travel syntax tree into map
        util.get_dst_tensor_map(out_tensors, dst_tensor_map)

        # tell difference between pure out and mid out
        for out in out_tensors:
            if out in dst_tensor_map.keys():
                temp_mid_output_tensors_in_ub.append(out)

        # make mid output tensors copy itself to out
        for out in temp_mid_output_tensors_in_ub:
            # pylint: disable=unnecessary-lambda
            with tvm.tag_scope(util.SET_GM_SCOPE_TAG):
                out_gm = tvm.compute(out.shape, lambda *i: out(*i), name=out.name + "_gm")
            index = out_tensors.index(out)
            out_tensors[index] = out_gm
            self._mid_output_tensors_in_gm.append(out_gm)

        # use fake node to intercept schedule
        res = util.fake_node_fuse_fun(out_tensors)
        if res == util.FAKE_NODE_FUSE_FAILED:
            return True
        self._schedule = tvm.create_schedule([res.op])
        out_tensors.append(res)
        return False

    def _is_broadcast_last_axis_tensor(self, tensor):
        if tensor.op.tag == "broadcast_for_tensor":
            # broadcast not last axis
            if list(tensor.op.input_tensors):
                original_tensor = tensor.op.input_tensors[0]
                original_shape = self._shape_to_list(original_tensor.shape)
                broadcast_shape = self._shape_to_list(tensor.shape)
                if original_shape[-1] == 1 and broadcast_shape[-1] != 1:
                    return True
                # include (3,1,1) -> (3,2,1)
                # include (1,1,1,1,1,1,1)->(10, 10, 5, 2, 3, 9, 1)
                for i in reversed(range(len(original_shape))):
                    if original_shape[i] != 1 and broadcast_shape[i] != 1:
                        return False
                    if original_shape[i] == 1 and broadcast_shape[i] != 1:
                        return True
        return False

    def _calculate_cache_read(self):
        """
        cache read operations

        Parameters
        ----------
        None

        Returns
        -------
        list : read buffers
        """
        for i in self._input_tensors:
            self._map_apend(self._cache_read_tensors_and_readers_map, i,
                            self._input_tensor_dst_tensor_map[i])

    def _calculate_cache_write(self):
        # pylint: disable=attribute-defined-outside-init
        """
        cache read operations

        Parameters
        ----------
        None

        Returns
        -------
        list : read buffers
        """
        # modify for reduce -> broadcast -> reduce
        temp_list = []
        self._multi_broadcast_for_reduce = []
        for tensor in self._cache_write_exclude_tensors:
            if tensor not in self._mid_output_tensors_dst_tensor_map.keys():
                continue
            suf_label = False
            for suf_tensor in self._mid_output_tensors_dst_tensor_map[tensor]:
                if util.REDUCE_OP_TAG_LABEL in suf_tensor.op.tag:
                    suf_label = True
                    break
            if not suf_label:
                temp_list.append(tensor)
            elif tensor not in self._multi_broadcast_for_reduce:
                self._multi_broadcast_for_reduce.append(tensor)
        self._cache_write_exclude_tensors = temp_list
        exclude_tensors = self._cache_write_exclude_tensors + self._mid_output_tensors_in_gm
        for i in self._mid_tensors:
            if i not in exclude_tensors:
                self._cache_write_tensors.append(i)

    def _is_bert_nz_broadcast_tanspose(self, tensor):
        return len(tensor.shape) > 4 and tensor.dtype == "float16" and \
               util.shape_to_list(tensor.shape)[-4:] == [1, 16, 16, 16] and \
               self._tiling_para["ub_tiling"]["axis"] < len(self._default_shape) - 2

    def _calculate_emit_insn(self):
        # pylint: disable=too-many-locals
        """
        Calculate the instruction map of tensor

        Parameters:
        ----------
        None

        Returns
        -------
        None
        """
        res = self._last_output_tensor
        ub_tiling_result = self._tiling_result["ub_tiling"]
        ub_split_axis = ub_tiling_result["axis"]
        res_ub_inner = ub_tiling_result["inner_itervar"]

        self._get_emit_insn_map()
        self._get_reg_emit_insn_map()

        def ge_broadcast_emit_insn_axis():
            axis_idx = self._last_axis_index
            cur_shape = util.shape_to_list(i.shape)
            input_shape = util.shape_to_list(i.op.input_tensors[0].shape)
            # ignore useless `1` at tail
            # case like (3,1,1,1) -> (3,2,1,1) pragma axis in (1,)
            # case like (3,1,1,1,1,1) -> (3,1,2,2,1,1) pragma axis in (2,)
            # case like (3,1,1,1,1,1) -> (3,2,1,2,1,1) pragma axis in (3,)
            while axis_idx > 0:
                if cur_shape[axis_idx] != util.INIT_SIZE:
                    break
                axis_idx -= 1
            if self._op_type == OpSpecTypes.MVN:
                # find pragma axis pos
                # case like (3,1,1,1) -> (3,2,2,2) pragma axis in (1,)
                # case like (3,1,1,1) -> (3,2,1,2) pragma axis in (1,)
                # case like (3,1,2,1) -> (3,2,2,2) pragma axis in (1,)
                axis_idx = self._tiling_barrier[0]
            else:
                # ignore useless `1` at tail
                # find pragma axis pos
                # case like (3,1,1,1) -> (3,2,2,2) pragma axis in (1,)
                # case like (3,1,1,1) -> (3,2,1,2) pragma axis in (3,)
                # case like (3,1,2,1) -> (3,2,2,2) pragma axis in (3,)
                while axis_idx >= 0:
                    if input_shape[axis_idx] == cur_shape[axis_idx]:
                        axis_idx += 1
                        break
                    axis_idx -= 1

            axis_idx = max(0, axis_idx)
            return axis_idx


        for i in self._cache_read_tensors_and_buffer_map:
            read_buffer = self._cache_read_tensors_and_buffer_map[i]
            para = {"scope": read_buffer.op.axis[ub_split_axis],
                    "instruction": util.DMA_COPY_PRAGMA}
            self._emit_insn_map[read_buffer] = para

        for i in self._cache_write_tensors_and_buffer_map:
            write_buffer = self._cache_write_tensors_and_buffer_map[i]
            insn = self._calculate_emit_insn_map(write_buffer)
            pragma_axis_idx = ub_split_axis
            if util.BROADCAST_TAG_LABEL in i.op.tag:
                if not self._is_broadcast_last_axis_tensor(i):
                    # for reduce -> broadcast -> reduce
                    if i in self._multi_broadcast_for_reduce:
                        insn = util.VECTOR_AUTO_PRAGMA
                    else:
                        continue
                elif self._is_bert_nz_broadcast_tanspose(i):
                    # for bert Nz
                    pragma_axis_idx = -2
                    insn = util.BROADCAST_TRANSPOSE
                else:
                    pragma_axis_idx = ge_broadcast_emit_insn_axis()
                    insn = util.BROADCAST_ALIGN_PRAGMA

            # special process for sub (1,1,1,32) (32,224,224,3)
            if insn == "vector_sub":
                if self._special_non_last_broadcast_scene:
                    if self._is_special_tensor_of_broadcast_not_last_axis(i):
                        insn = "vector_sub_with_multi_broadcast"

            para = {"scope": write_buffer.op.axis[pragma_axis_idx],
                    "instruction": insn}
            self._emit_insn_map[write_buffer] = para

        for out_tensor in self._mid_output_tensors:
            para = {"scope": out_tensor.op.axis[ub_split_axis],
                    "instruction": util.DMA_COPY_PRAGMA}
            self._emit_insn_map[out_tensor] = para

        # eliminate mid out tensor from gm to ub by fake node
        for tensor in self._mid_output_tensors_in_gm:
            para = {"scope": tensor.op.axis[ub_split_axis],
                    "instruction": util.DMA_COPY_PRAGMA}
            self._emit_insn_map[tensor] = para

        self._emit_insn_map[res] = {"scope": res_ub_inner,
                                    "instruction": util.FAKE_NODE_PRAGMA}
        self._schedule[res].set_scope("")

    def _calculate_tiling(self):
        """
        calculate tiling strategy

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.__tiling_init()

        if self.__check_for_op_limit() or \
           self.__caculate_reduce_axes_map() or \
           self.__caculate_basic_info() or \
           self.__caculate_align_in_ub() or \
           self.__caculate_align_in_gm():
            return False

        self.__consider_double_buffer()

        if self.__caculate_all_tiling():
            return False

        return True

    def __tiling_init(self):
        # pylint: disable=attribute-defined-outside-init
        # shape info
        self._reduce_include_last_axis = False
        self._last_axis_index = util.DEFAULT_INDEX
        self._default_shape = []
        # reduce info
        self._reduce_tensor_count = util.INIT_COUNT
        self._reduce_axes_map = {}
        self._reduce_tensor_map = {}
        self._reduce_axes_can_be_multi_core = []
        # ub size info
        self._core_num = util.INIT_COUNT
        self._max_ub_count = util.INIT_COUNT
        self._limit_ub_count = util.INIT_COUNT
        self._align_count = util.INIT_COUNT
        self._align_gm_axis_idx = util.DEFAULT_INDEX
        self._align_gm_factor = util.INIT_COUNT
        self._need_db = False
        self._pattern = util.pattern.P_NONE
        # type info
        self._type_list = []
        self._max_type_bitsize = util.MIN_TYPE_SIZE_UNIT
        self._min_type_bitsize = util.MAX_TYPE_SIZE_UNIT
        # tiling info
        self._tiling_strategy = None
        self._tiling_cur_shape = []
        self._tiling_barrier = []
        self._all_align_axis_idx = util.DEFAULT_INDEX
        self._multi_core_fused_axis = None
        self._multi_core_bind_tensor = None

    def __check_for_op_limit(self):
        # use white list to check all compute
        white_list = ["elewise_single_cast",
                      "elewise_single_log",
                      "elewise_single_exp",
                      "elewise_single_relu",
                      "elewise_single_abs",
                      "elewise_single_sqrt",
                      "elewise_single_rsqrt",
                      "elewise_single_rec",
                      "elewise_single_abs",
                      "elewise_binary_add",
                      "elewise_binary_sub",
                      "elewise_binary_mul",
                      "elewise_binary_div",
                      "elewise_binary_max",
                      "elewise_binary_min",
                      "elewise_single_VS_add",
                      "elewise_single_VS_mul",
                      "elewise_single_VS_max",
                      "elewise_single_VS_min",
                      "elewise_single_not",
                      "elewise_binary_and",
                      "elewise_binary_or",
                      "reduce_sum",
                      "reduce_max",
                      "reduce_min",
                      "broadcast",
                      "broadcast_for_tensor",
                      util.SET_GM_SCOPE_TAG,
                      util.FAKE_NODE_TAG]

        black_list_in_mini = ["elewise_binary_div"]

        for tensor in self._mid_tensors:
            if tensor.op.tag.split('|')[0] not in white_list:
                return True
        soc_version = cceconf.get_soc_spec("SOC_VERSION")
        if soc_version in ("Ascend310",):
            for tensor in self._mid_tensors:
                if tensor.op.tag.split('|')[0] in black_list_in_mini:
                    return True
        return False

    def __consider_double_buffer(self):
        # pylint: disable=attribute-defined-outside-init
        double_buffer_size = self._limit_ub_count * 2
        if double_buffer_size > self._max_ub_count:
            self._need_db = False
        else:
            self._need_db = True
            self._limit_ub_count = double_buffer_size

    def __caculate_reduce_axes_map(self):
        # pylint: disable=attribute-defined-outside-init
        self._default_shape = self._shape_to_list(self._last_output_tensor.shape)
        for tensor in self._mid_tensors:
            if util.REDUCE_OP_TAG_LABEL not in tensor.op.tag:
                continue
            shape1 = util.shape_to_list(tensor.op.input_tensors[0].shape)
            shape2 = util.shape_to_list(tensor.shape)
            # unsupport different dim shape
            if len(shape1) != len(shape2):
                return True
            # one axis may reduce multi times, and each time with different length
            # we record the max one, to calculate shape max size for tiling
            reduce_axes = []
            for d_i, _ in enumerate(shape1):
                if shape1[d_i] != shape2[d_i]:
                    # broadcast shape must the same with original shape
                    if shape2[d_i] == util.REDUCE_AXIS_SIZE:
                        reduce_axes.append(d_i)
                        self._reduce_axes_map[d_i] = shape1[d_i]
                    else:
                        return True
            # unsuport reduce 1
            if len(reduce_axes) != len(tensor.op.reduce_axis):
                return True
            if reduce_axes:
                self._reduce_tensor_map[tensor] = reduce_axes
                self._reduce_tensor_count += 1

        # calculate axes which support atomic sum in multi core
        version_info = util.get_atomic_reduce_info()
        for d_i in self._reduce_axes_map:
            temp_res = True
            for tensor in self._reduce_tensor_map:
                if d_i in self._reduce_tensor_map[tensor]:
                    temp_res = temp_res and \
                               tensor in self._mid_output_tensors and \
                               util.is_support_atomic_reduce(tensor, version_info)
            if temp_res:
                self._reduce_axes_can_be_multi_core.append(d_i)

        # ignore for 1 axis at tail
        # case like (2,3,5,1,1) -> (2,3,5)
        count = util.INIT_COUNT
        for i in reversed(range(len(self._default_shape))):
            if self._default_shape[i] == 1 and i not in self._reduce_axes_map.keys():
                count += 1
            else:
                break
        if count:
            self._default_shape = self._default_shape[:(0 - count)]
            self._last_axis_index -= count
        return False

    def __is_reduce_count_right(self):
        if (self._op_type != OpSpecTypes.MVN and self._reduce_tensor_count < 2) or \
                (self._op_type == OpSpecTypes.MVN and len(self._default_shape) > 5):
            return True
        return False

    def __caculate_basic_info(self):
        # pylint: disable=attribute-defined-outside-init
        ### reduce info
        self._last_axis_index = len(self._default_shape) - 1
        if self._last_axis_index in self._reduce_axes_map.keys():
            self._reduce_include_last_axis = True
        ### type list
        for tensor in self._mid_output_tensors_dst_tensor_map:
            temp_dtype = tensor.dtype.lower()
            if temp_dtype not in self._type_list:
                self._type_list.append(temp_dtype)
        ### max type and min type
        for dtype in self._type_list:
            if dtype not in util.DTYPE_WIDTH_MAP.keys():
                return True
            if util.DTYPE_WIDTH_MAP[dtype] < self._min_type_bitsize:
                self._min_type_bitsize = util.DTYPE_WIDTH_MAP[dtype]
            if util.DTYPE_WIDTH_MAP[dtype] > self._max_type_bitsize:
                self._max_type_bitsize = util.DTYPE_WIDTH_MAP[dtype]
        ### ub block
        self._core_num = cceconf.get_soc_spec("CORE_NUM")
        ### ub size
        # div 2 for align to fp16
        total_size = cceconf.get_soc_spec("UB_SIZE") // 2
        self._pattern = util.pattern_identify(self._mid_tensors)
        if self._pattern in util.pattern.width.keys() and util.PATTERN_OPTIMAZE:
            # custom for layernorm
            temp_size = util.INIT_SIZE
            for dim in self._reduce_axes_map:
                temp_size *= self._reduce_axes_map[dim]
            if self._pattern in util.pattern.layernorm_width.keys() and \
                    temp_size == 1024 and self._last_axis_index in self._reduce_axes_map:
                total_width = util.pattern.layernorm_width[self._pattern]
            else:
                total_width = util.pattern.width[self._pattern]
        else:
            total_width = self._get_total_width() + 1
        if not total_width:
            raise RuntimeError("Can not calculate with no compute")
        self._max_ub_count = int(total_size / total_width)
        # align by repeat
        coef = util.VECTOR_ONE_REPEAT_UNIT
        self._max_ub_count = int(self._max_ub_count // coef) * coef
        # pattern limit
        if self._pattern not in util.pattern.width.keys() and util.PATTERN_LIMIT:
            return True
        # must have reduce tensor, and more than 1, must have elemwise axis to compute at with
        if self._pattern == util.pattern.P_NONE and \
                (self.__is_reduce_count_right() or
                 len(self._reduce_axes_map.keys()) == len(self._default_shape)):
            return True
        return False

    def __calculate_align_in_reduce_last(self):
        # pylint: disable=attribute-defined-outside-init
        ### reduce last, need consider series axes
        # case like 'fp16' (5, 2, 8), align cout is 16
        # if reduce in (1, 2) can not be aligned
        # if reduce in (2), it must be align
        reduce_axes_touch_last = list(range(len(self._default_shape)))
        ## find reduce last axis series axes, which all reduce have them
        for tensor in self._reduce_tensor_map:
            if self._last_axis_index in self._reduce_tensor_map[tensor]:
                # for remove element in list, should use copy
                temp_axes = reduce_axes_touch_last[:]
                for axis in reduce_axes_touch_last:
                    if axis not in self._reduce_tensor_map[tensor]:
                        temp_axes.remove(axis)
                reduce_axes_touch_last = temp_axes
        ## calculate align, until touch non reduce axis
        cur_size = util.INIT_SIZE
        for idx in reversed(range(len(self._default_shape))):
            if idx not in reduce_axes_touch_last:
                break
            cur_size *= self._default_shape[idx]
            if cur_size % self._align_count == 0:
                self._all_align_axis_idx = util.DEFAULT_INDEX
                return False
        ## reduce not align, recalculate align count
        align_axis_ori_size = self._default_shape[self._last_axis_index]
        ori_coef = util.gcd(align_axis_ori_size, self._align_count)
        new_coef = util.gcd(cur_size, self._align_count)
        align_count = self._align_count
        self._align_count = int(self._align_count / (new_coef / ori_coef))
        # recalculate ub size
        align_axis_new_size = util.align(align_axis_ori_size, self._align_count)

        if self._op_type == OpSpecTypes.MVN:
            self._align_count = align_count

        self._limit_ub_count = self._limit_ub_count / align_axis_ori_size * align_axis_new_size
        if self._limit_ub_count > self._max_ub_count:
            return True
        self._all_align_axis_idx = self._last_axis_index
        self._tiling_cur_shape[self._last_axis_index] = align_axis_new_size
        return False

    def __calculate_align_in_reduce_non_last(self):
        # pylint: disable=attribute-defined-outside-init
        ### reduce non last
        ## all reduce axes size, if not reduce last, need to add least one block size to be aligned
        ## this strategy allow it to fallback 3 times,
        ## 1. use one block size to align, in ub size limit
        ## 2. use one repeat size to align, in ub size limit
        ## 3. use whole consecutive axes to align, in ub size limit
        # this will consider from last reduce last backward to pick all axis size to fuse or split
        # for discussion above, consider fuse all axis after final reduce axis, and cut one block
        # calculate last axis to align
        self._limit_ub_count *= self._align_count
        if self._limit_ub_count > self._max_ub_count:
            return True
        ## fake split for align and limit block tiling
        # case like (6, 160) reduce in (0,)
        # if not barrier, block tiling into (32, 6, 5), lead to not align
        self._tiling_cur_shape[-1] = util.ceil(self._tiling_cur_shape[-1], self._align_count)
        self._tiling_cur_shape.append(self._align_count)
        self._tiling_barrier.append(len(self._default_shape))
        if self._default_shape[-1] % self._align_count:
            self._all_align_axis_idx = self._last_axis_index
        return False

    def __caculate_align_in_ub(self):
        # pylint: disable=attribute-defined-outside-init
        self._limit_ub_count = util.INIT_SIZE
        for idx in self._reduce_axes_map:
            self._limit_ub_count *= self._default_shape[idx]
            self._tiling_barrier.append(idx)
        if self._limit_ub_count > self._max_ub_count:
            return True
        self._align_count = int(util.VECTOR_ONE_BLOCK_UNIT // self._min_type_bitsize)
        self._tiling_cur_shape = self._default_shape[:]

        if self._reduce_include_last_axis:
            return self.__calculate_align_in_reduce_last()
        return self.__calculate_align_in_reduce_non_last()

    def __is_last_axis_align(self):
        if self._op_type != OpSpecTypes.MVN and self._default_shape[-1] % self._align_count:
            return True
        return False

    def __caculate_align_in_gm(self):
        # pylint: disable=attribute-defined-outside-init
        # pylint: disable-msg=too-many-locals
        # unsupport last axis not align
        if self.__is_last_axis_align():
            return True
        # gm align focus on out put date, it must more than one block
        # or multi core pipe will be a problem
        res_align_dim = util.DEFAULT_INDEX
        res_align_factor = util.DEFAULT_INDEX
        for tensor in self._mid_output_tensors:
            cur_factor = int(util.VECTOR_ONE_BLOCK_UNIT // util.DTYPE_WIDTH_MAP[tensor.dtype])
            cur_shape = util.shape_to_list(tensor.shape)
            cur_align_status = False
            for i in reversed(range(len(cur_shape))):
                # this tensor find align dim and factor
                if cur_factor <= cur_shape[i]:
                    cur_align_status = True
                    if (res_align_dim == util.DEFAULT_INDEX or res_align_dim > i or \
                        (res_align_dim == i and cur_factor > res_align_factor)) and \
                            i not in self._tiling_barrier:
                        res_align_dim = i
                        res_align_factor = cur_factor
                    break
                cur_factor = util.ceil(cur_factor, cur_shape[i])
            # not fit gm align, unsupport
            if not cur_align_status:
                return True
        # all align default by current barrier
        if res_align_dim == util.DEFAULT_INDEX or \
           res_align_factor == util.DEFAULT_INDEX or \
           res_align_dim == self._last_axis_index:
            return False
        # make align axis as barrier
        cur_ub_limit_size = self._limit_ub_count
        cur_barrier = []
        for i in range(res_align_dim + 1, len(self._default_shape)):
            if i not in self._tiling_barrier:
                cur_ub_limit_size *= self._default_shape[i]
                cur_barrier.append(i)
        rest_size = int(self._max_ub_count // cur_ub_limit_size)
        # not enough space to align gm
        if rest_size < res_align_factor:
            self._core_num = util.INIT_SIZE
            return False
        self._tiling_barrier += cur_barrier
        self._limit_ub_count = cur_ub_limit_size
        # consider multi core
        free_size_after_block_tiling = util.INIT_SIZE
        for i in range(res_align_dim + 1):
            if i not in self._tiling_barrier:
                free_size_after_block_tiling *= self._default_shape[i]
        free_size_after_block_tiling = util.ceil(free_size_after_block_tiling, self._core_num)
        # use barrier to make sure align
        cur_align_size = res_align_factor
        upper_size = min(rest_size, self._default_shape[res_align_dim])
        # find perfect cut
        for align_size in range(res_align_factor, upper_size + 1):
            if free_size_after_block_tiling % align_size == 0:
                cur_align_size = align_size
                break
        misalign_size = self._default_shape[res_align_dim] % cur_align_size
        if misalign_size and misalign_size < res_align_factor:
            self._core_num = util.INIT_SIZE
            return False
        self._align_gm_axis_idx = res_align_dim
        self._align_gm_factor = cur_align_size
        self._tiling_cur_shape[res_align_dim] //= cur_align_size
        self._limit_ub_count *= cur_align_size
        return False

    def __caculate_all_tiling(self):
        # pylint: disable=attribute-defined-outside-init
        ### note: 1.fuse axis can not be splited twice, or compute at will be wrong
        ### note: 2.ub tiling can be front at block tiling when use fused axis to block tiling,
        ### or compute at will be wrong
        if len(self._tiling_barrier) == len(self._tiling_cur_shape):
            return True
        # block tiling
        ## use greedy, chiefly, make it efficiency
        tiling_shape = self._tiling_cur_shape[:]
        tiling_barrier = self._tiling_barrier[:]
        block_tiling_axes, block_factor = util.get_block_factor_radical(tiling_shape,
                                                                        tiling_barrier,
                                                                        self._core_num)
        # set barrier base on block tiling result
        tiling_shape[block_tiling_axes[0]] = block_factor
        tiling_barrier = tiling_barrier + list(range(block_tiling_axes[0]))
        tiling_barrier = tiling_barrier + block_tiling_axes[1:]
        # ub tiling
        rest_size = int(self._max_ub_count // self._limit_ub_count)
        ub_axis_idx, ub_factor = util.get_ub_factor(tiling_shape, tiling_barrier, rest_size)
        # check result
        if ub_axis_idx not in block_tiling_axes or len(block_tiling_axes) == 1:
            self._tiling_strategy = util.TILING_RADICAL
            self._tiling_cur_shape = tiling_shape
            self._tiling_barrier = tiling_barrier
        else:
            tiling_shape = self._tiling_cur_shape[:]
            tiling_barrier = self._tiling_barrier[:]
            ## use fill, at last, make it successful
            block_tiling_axes, block_factor = util.get_block_factor_conservative(tiling_shape,
                                                                                 tiling_barrier,
                                                                                 self._core_num)
            # set barrier base on block tiling result
            tiling_barrier = tiling_barrier + list(range(block_tiling_axes[-1]))
            tiling_shape[block_tiling_axes[-1]] = block_factor[-1]
            # ub tiling
            rest_size = self._max_ub_count // self._limit_ub_count
            ub_axis_idx, ub_factor = util.get_ub_factor(tiling_shape, tiling_barrier, rest_size)
            self._tiling_strategy = util.TILING_CONSERVATIVE
            self._tiling_cur_shape = tiling_shape
            self._tiling_barrier = tiling_barrier

        # modify cut factor as barrier last axis to ub align
        if self._last_axis_index in block_tiling_axes:
            if self._tiling_strategy == util.TILING_RADICAL:
                block_factor *= self._align_count
            if self._tiling_strategy == util.TILING_CONSERVATIVE:
                block_factor[-1] = block_factor[-1] * self._align_count
        if ub_axis_idx == self._last_axis_index:
            ub_factor *= self._align_count
        # modify cut factor as barrier some axes to gm align
        if self._align_gm_axis_idx in block_tiling_axes:
            if self._tiling_strategy == util.TILING_RADICAL:
                block_factor *= self._align_gm_factor
            if self._tiling_strategy == util.TILING_CONSERVATIVE:
                block_factor[-1] = block_factor[-1] * self._align_gm_factor
        if ub_axis_idx == self._align_gm_axis_idx:
            ub_factor *= self._align_gm_factor
        # set result
        block_tiling_para = {"axes": block_tiling_axes, "factor": block_factor}
        ub_tiling_para = {"axis": ub_axis_idx, "factor": ub_factor}
        self._tiling_para["block_tiling"] = block_tiling_para
        self._tiling_para["ub_tiling"] = ub_tiling_para
        self._tiling_tensor = self._last_output_tensor
        return False

    # pylint: disable=too-many-locals
    def _do_tiling(self):
        sch = self._schedule
        res = self._tiling_tensor
        # align tiling
        res_axes = []
        leave_in_ub_axes = []
        front_ub_axis_idx = util.DEFAULT_INDEX
        for idx in self._reduce_axes_map:
            leave_in_ub_axes.append(res.op.axis[idx])
            if idx < front_ub_axis_idx or front_ub_axis_idx == util.DEFAULT_INDEX:
                front_ub_axis_idx = idx
        for d_i in range(len(self._default_shape)):
            res_axes.append(res.op.axis[d_i])
        # get params
        block_tiling_para = self._tiling_para["block_tiling"]
        block_tiling_axes = block_tiling_para["axes"]
        block_tiling_factor = block_tiling_para["factor"]
        ub_tiling_para = self._tiling_para["ub_tiling"]
        ub_tiling_axis = ub_tiling_para["axis"]
        ub_tiling_factor = ub_tiling_para["factor"]
        # make init
        if self._tiling_strategy == util.TILING_RADICAL:
            # block tiling
            block_target_axis = res_axes[block_tiling_axes[0]]
            fuse_axes_idx = block_tiling_axes[1:]
            for d_i in fuse_axes_idx:
                block_target_axis = sch[res].fuse(block_target_axis, res_axes[d_i])
                res_axes[d_i] = util.DEFAULT_INDEX
            res_block_outer, res_block_inner = sch[res].split(block_target_axis,
                                                              factor=block_tiling_factor)
            block_target_axis = res_block_outer
            res_axes[block_tiling_axes[0]] = res_block_inner
            # ub tiling
            ub_target_axis = res_axes[ub_tiling_axis]
            res_ub_outer, res_ub_inner = sch[res].split(ub_target_axis, factor=ub_tiling_factor)
        elif self._tiling_strategy == util.TILING_CONSERVATIVE:
            # block tiling
            # pylint: disable=unsubscriptable-object
            if len(block_tiling_axes) == 1:
                res_block_outer, res_block_inner = sch[res].split(res_axes[block_tiling_axes[0]],
                                                                  factor=block_tiling_factor[0])
                res_axes[block_tiling_axes[0]] = res_block_inner
                block_target_axis = res_block_outer
            else:
                pre_outer, pre_inner = sch[res].split(res_axes[block_tiling_axes[0]],
                                                      factor=block_tiling_factor[0])
                suf_outer, suf_inner = sch[res].split(res_axes[block_tiling_axes[-1]],
                                                      factor=block_tiling_factor[-1])
                block_target_axis = pre_inner
                block_fuse_axes_idx = block_tiling_axes[1:-1]
                for d_i in block_fuse_axes_idx:
                    block_target_axis = sch[res].fuse(block_target_axis, res_axes[d_i])
                    res_axes[d_i] = util.DEFAULT_INDEX
                block_target_axis = sch[res].fuse(block_target_axis, suf_outer)
                res_axes[block_tiling_axes[0]] = pre_outer
                res_axes[block_tiling_axes[-1]] = suf_inner
            # ub tiling
            ub_target_axis = res_axes[ub_tiling_axis]
            res_ub_outer, res_ub_inner = sch[res].split(ub_target_axis, factor=ub_tiling_factor)
        else:
            raise RuntimeError("Tiling Falied!!")

        # reorder
        axes_order = [block_target_axis]
        # modify it, case like (2048,1024) reduce in (0,)
        # cut into (2048,32,4,8), should reorder as (32,4,2048,8), pragma in (2,)
        ub_tiling_result_axis_idx = ub_tiling_axis
        if front_ub_axis_idx < ub_tiling_axis and front_ub_axis_idx != util.DEFAULT_INDEX:
            ub_tiling_result_axis_idx = front_ub_axis_idx
        for idx, _ in enumerate(res_axes):
            if res_axes[idx] == util.DEFAULT_INDEX:
                continue
            if idx == ub_tiling_axis:
                axes_order.append(res_ub_outer)
                axes_order.append(res_ub_inner)
            elif res_axes[idx] not in leave_in_ub_axes:
                axes_order.append(res_axes[idx])
        axes_order = axes_order + leave_in_ub_axes
        sch[res].reorder(*axes_order)
        # save params
        self._multi_core_fused_axis = block_target_axis
        self._multi_core_bind_tensor = self._last_output_tensor
        block_tiling_result = {"axes": block_tiling_axes,
                               "parent_itervar": block_target_axis,
                               "outer_itervar": block_target_axis,
                               "inner_itervar": block_target_axis}
        ub_tiling_result = {"axis": ub_tiling_result_axis_idx,
                            "parent_itervar": ub_target_axis,
                            "outer_itervar": res_ub_outer,
                            "inner_itervar": res_ub_inner}
        self._tiling_result = {"block_tiling": block_tiling_result,
                               "ub_tiling": ub_tiling_result}

    def _do_storage_align(self):
        sch = self._schedule
        if self._all_align_axis_idx == util.DEFAULT_INDEX:
            return
        axis_index = -2
        if self._op_type == OpSpecTypes.MVN:
            axis_index = self._tiling_barrier[0] - 1
        # if allow non last axis, align axis should be reconsidered
        for i in self._cache_read_tensors_and_buffer_map:
            read_buffer = self._cache_read_tensors_and_buffer_map[i]
            if read_buffer.shape[self._last_axis_index].value != util.REDUCE_AXIS_SIZE:
                sch[read_buffer].storage_align(read_buffer.op.axis[axis_index],
                                               self._align_count, 0)

        for i in self._cache_write_tensors_and_buffer_map:
            write_buffer = self._cache_write_tensors_and_buffer_map[i]
            if write_buffer.shape[self._last_axis_index].value != util.REDUCE_AXIS_SIZE:
                sch[write_buffer].storage_align(write_buffer.op.axis[axis_index],
                                                self._align_count, 0)

    def _calculate_compute_at(self):
        """
        Calculate the tensor that needs compute at

        Parameters:
        ----------
        None

        Returns
        -------
        None
        """
        res = self._last_output_tensor
        ub_tiling_result = self._tiling_result["ub_tiling"]
        res_ub_outer = ub_tiling_result["outer_itervar"]

        for i in self._cache_read_tensors_and_buffer_map:
            read_buffer = self._cache_read_tensors_and_buffer_map[i]
            para = {"parent": self._schedule[res], "scope": res_ub_outer}
            self._compute_at_map[read_buffer] = para

        for i in self._cache_write_tensors_and_buffer_map:
            write_buffer = self._cache_write_tensors_and_buffer_map[i]
            para = {"parent": self._schedule[res], "scope": res_ub_outer}
            self._compute_at_map[write_buffer] = para

        for i in self._mid_output_tensors:
            para = {"parent": self._schedule[res], "scope": res_ub_outer}
            self._compute_at_map[i] = para

    def _get_emit_insn_map(self):
        self._insn_map = {"elewise_single_cast": "vector_conv",
                          "elewise_single_VS_max": "vector_maxs",
                          "elewise_single_VS_min": "vector_mins",
                          "elewise_single_log": "vector_ln",
                          "elewise_single_exp": "vector_exp",
                          "elewise_single_rec": "vector_rec",
                          "elewise_single_relu": "vector_relu",
                          "elewise_single_abs": "vector_abs",
                          "elewise_single_not": "vector_not",
                          "elewise_single_sqrt": "vector_sqrt",
                          "elewise_single_rsqrt": "vector_rsqrt",
                          "elewise_binary_mul": "vector_mul",
                          "elewise_single_VS_mul": "vector_muls",
                          "elewise_binary_div": "vector_div",
                          "elewise_binary_add": "vector_add",
                          "elewise_single_VS_add": "vector_adds",
                          "elewise_binary_min": "vector_min",
                          "elewise_binary_max": "vector_max",
                          "elewise_binary_or": "vector_or",
                          "elewise_binary_and": "vector_and",
                          "elewise_binary_sub": "vector_sub",
                          "reduce_sum": "vector_reduce_sum",
                          "reduce_max": "vector_reduce_max",
                          "reduce_min": "vector_reduce_min"}
        # dichotomy reduce check
        if self._pattern == util.pattern.P_NONE:
            return

        reduce_size = util.INIT_SIZE
        for cur_dim in self._reduce_axes_map:
            reduce_size *= self._reduce_axes_map[cur_dim]
        dichotomy_reduce_times = reduce_size // util.VECTOR_ONE_REPEAT_UNIT

        reduce_matched = len(self._default_shape) > 2 and \
             len(self._reduce_axes_map) == 2 and \
             self._last_axis_index in self._reduce_axes_map.keys() and \
             self._last_axis_index - 1 not in self._reduce_axes_map.keys() and \
             self._default_shape[-1] % util.VECTOR_ONE_BLOCK_UNIT == 0

        if reduce_matched and reduce_size % util.VECTOR_ONE_REPEAT_UNIT == 0 and \
                dichotomy_reduce_times & (dichotomy_reduce_times - 1) == 0:
            const_size = util.INIT_SIZE
            reduce_axes = list(self._reduce_axes_map.keys())
            for i in range(reduce_axes[0] + 1, reduce_axes[1]):
                if i not in self._tiling_barrier:
                    const_size *= self._tiling_cur_shape[i]
            ub_tiling_axis = self._tiling_para["ub_tiling"]["axis"]
            ub_tiling_factor = self._tiling_para["ub_tiling"]["factor"]
            pre_reduce_axis_index = len(self._default_shape)
            for i in self._reduce_axes_map:
                pre_reduce_axis_index = min(i, pre_reduce_axis_index)
            tiling_limit = (ub_tiling_axis == pre_reduce_axis_index - 1 and \
                            ub_tiling_factor == util.INIT_SIZE) or \
                           ub_tiling_axis > pre_reduce_axis_index
            size_limit = const_size == util.INIT_SIZE and \
                         self._align_gm_axis_idx == util.DEFAULT_INDEX
            if not tiling_limit or size_limit:
                return
            self._insn_map["reduce_sum"] = "vector_dichotomy_reduce"
            self._insn_map["reduce_max"] = "vector_dichotomy_reduce"
            self._insn_map["reduce_min"] = "vector_dichotomy_reduce"

    def _do_buffer_reuse(self):
        if self._pattern not in util.pattern.width.keys() or not util.PATTERN_OPTIMAZE or \
                self._pattern in util.pattern.no_buffer_reused:
            return
        # common reuse
        reused_relation = {}
        used = []
        for tensor in self._mid_tensors:
            if len(self._mid_tensor_dst_tensor_map[tensor]) != 1:
                continue
            dst_tensor = self._mid_tensor_dst_tensor_map[tensor][0]
            # 1.type wide the same
            # 2.shape the same
            if dst_tensor in used or dst_tensor.op.tag == util.FAKE_NODE_TAG or \
               util.DTYPE_WIDTH_MAP[tensor.dtype] != util.DTYPE_WIDTH_MAP[dst_tensor.dtype] or \
               util.shape_to_list(tensor.shape) != util.shape_to_list(dst_tensor.shape) or \
               tensor in self._cache_write_exclude_tensors:
                continue
            src_buffer = self._cache_write_tensors_and_buffer_map[tensor]
            dst_buffer = self._cache_write_tensors_and_buffer_map[dst_tensor]
            self._schedule[src_buffer].reused_by(dst_buffer)
            used.append(dst_tensor)
            reused_relation[src_buffer] = dst_buffer
