#!/usr/bin/env python
# -*- coding:utf-8 -*-
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

atomic_addr_clean
"""

from te import tik
from topi.cce import util
from te import platform as tbe_platform

MEMORY_CLEAR_BLOCK_SIZE = 32
MEMORY_CLEAR_VECTOR_SIZE = 256
MEMORY_CLEAR_KB_NUM = 48
MEMORY_CLEAR_ONE_KB = 1024
MEMORY_CLEAR_UB_ZERO_BUFF = MEMORY_CLEAR_KB_NUM * MEMORY_CLEAR_ONE_KB


# pylint: disable=locally-disabled,unused-argument,invalid-name,too-many-locals,too-many-arguments
def ceil_align(ori_num, divider):
    '''
    dst_num = (ori_num + divider -1) / divider * divider
    :param ori_num: original number
    :param divider:
    :return:
    '''
    return (ori_num + divider - 1) // divider * divider


def ceil_divide(ori_num, divider):
    '''
    dst_num = (ori_num + divider -1) / divider
    :param ori_num:
    :param divider:
    :return:
    '''
    return (ori_num + divider - 1) // divider


class AtomicCleaner(object):
    '''
    AtmoicAddrCleaner Class implementing the function of operator AtomicClean
    '''

    def __init__(self, size_list, str_dtype="float16"):
        '''
        :param size_list: size list of workspaces
        :param addr_list: address list of workspaces
        '''
        self.tik_instance = tik.Tik()
        self.core_num = tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.CORE_NUM)
        self.workspace_sizes = size_list
        self.workspace_addrs = []
        self.workspace_num = len(size_list)
        self.data_type = str_dtype

    def gen_zero_ub(self, ub_size, str_dtype="float16"):
        '''
        generate an ub tensor of which size is ub_size and set it to be zero
        :param ub_size: size of ub tensor in byte
        :return: ub zero tensor
        '''

        div = 2
        if str_dtype == "float32":
            div = 4

        repeat_times = ceil_divide(ub_size, MEMORY_CLEAR_VECTOR_SIZE)
        common_mask = MEMORY_CLEAR_VECTOR_SIZE // div
        zero_ub = self.tik_instance.Tensor(str_dtype, (ub_size // div,),
                                           tik.scope_ubuf, "zero_ub")
        self.tik_instance.vector_dup(common_mask, zero_ub, 0, repeat_times, 1,
                                     8)
        return zero_ub

    def clean_workspace_single_core(self, data_size, data_addr, zero_ub):
        '''
        clean the memory of a workspace
        :param data_size: memory size of the workspace in byte
        :param data_addr: memory address of workspace
        :return:
        '''
        div = 2
        if self.data_type == "float32":
            div = 4

        loop_num = ceil_divide(data_size, MEMORY_CLEAR_UB_ZERO_BUFF)
        loop_offset = MEMORY_CLEAR_UB_ZERO_BUFF // div
        remain_load_size = data_size % MEMORY_CLEAR_UB_ZERO_BUFF
        burst_len = ceil_divide(MEMORY_CLEAR_UB_ZERO_BUFF,
                                MEMORY_CLEAR_BLOCK_SIZE)
        if remain_load_size > 0:
            burst_len_last = ceil_divide(remain_load_size,
                                         MEMORY_CLEAR_BLOCK_SIZE)
            with self.tik_instance.for_range(0, loop_num) as idx:
                with self.tik_instance.if_scope(idx + 1 == loop_num):
                    with self.tik_instance.if_scope(idx == 0):
                        self.tik_instance.data_move(data_addr, zero_ub, 0, 1,
                                                    burst_len_last, 0, 0)
                    with self.tik_instance.else_scope():
                        self.tik_instance.data_move(
                            data_addr[idx * loop_offset], zero_ub, 0, 1,
                            burst_len_last, 0, 0)

                with self.tik_instance.else_scope():
                    with self.tik_instance.if_scope(idx == 0):
                        self.tik_instance.data_move(data_addr, zero_ub, 0, 1,
                                                    burst_len, 0, 0)
                    with self.tik_instance.else_scope():
                        self.tik_instance.data_move(
                            data_addr[idx * loop_offset], zero_ub, 0, 1,
                            burst_len, 0, 0)
        else:
            with self.tik_instance.for_range(0, loop_num) as idx:
                self.tik_instance.data_move(data_addr[idx * loop_offset],
                                            zero_ub, 0, 1, burst_len, 0, 0)

    def workspaces_distribute_to_cores(self, data_size_list):
        '''
        distribute the workspaces to all cores
        :param data_size_list:
        :return: workspace segments list of all cores
        '''
        div = 2
        if self.data_type == "float32":
            div = 4
        total_size = 0
        for ds in data_size_list:
            total_size = total_size + ds

        total_blockes = total_size // MEMORY_CLEAR_BLOCK_SIZE
        aver_blocks = ceil_divide(total_blockes, self.core_num)
        aver_size = aver_blocks * MEMORY_CLEAR_BLOCK_SIZE
        aver_size_aligned = ceil_align(aver_size, MEMORY_CLEAR_UB_ZERO_BUFF)

        # list of workspace segment, cell format of which is [data_addr_index, offset, seg_len]
        workspace_seg_list = []
        for ix, ds in enumerate(data_size_list):
            num = ceil_divide(ds, aver_size_aligned)
            rear_size = ds % aver_size_aligned
            if num == 1:
                workspace_seg_list.append([ix, 0, ds])
                continue
            for sn in range(num):
                # data_addr offset in count of fp16 or fp32
                offset = sn * aver_size_aligned // div
                seg_len = aver_size_aligned
                if rear_size > 0 and sn == num - 1:
                    seg_len = rear_size
                workspace_seg_list.append([ix, offset, seg_len])

        # sort the workspace segs by length in descending order
        def take_third(elem_list):
            return elem_list[2]

        workspace_seg_list.sort(key=take_third, reverse=True)

        core_segs_list = []
        for cidx in range(self.core_num):
            if not workspace_seg_list:
                break

            # if this is the last core, all the rest segs consign to this core
            if cidx == self.core_num - 1:
                core_segs_list.append(workspace_seg_list)
                break

            # for each core, construct a list segs, consign the first seg of seglist to it
            # until the total length of segs is greater than aver_size_aligned
            segs = []
            segs.append(workspace_seg_list[0])
            workspace_seg_list.pop(0)
            core_seg_len = segs[-1][2]
            for ix in range(len(workspace_seg_list) - 1, -1, -1):
                if core_seg_len + workspace_seg_list[ix][
                        2] <= aver_size_aligned:
                    core_seg_len = core_seg_len + workspace_seg_list[ix][2]
                    segs.append(workspace_seg_list[ix])
                    workspace_seg_list.pop(ix)
                else:
                    break

            core_segs_list.append(segs)

        return core_segs_list

    def workspaces_clean_multi_core_integrally(self, data_size_list,
                                               data_addr_list):
        '''
        there is an restriction that in one operator we can apply core loop only one time
        it means that if we want to process more than one workspace
        we should take the core loop as the outer one
        ATTENSION:
        in this function, each workspace is cleaned in one core as possible
        :param data_size_list:
        :param data_addr_list:
        :return:
        '''

        core_segs = self.workspaces_distribute_to_cores(data_size_list)
        core_used = len(core_segs)

        with self.tik_instance.for_range(0, core_used,
                                         block_num=core_used) as block_idx:
            zero_ub = self.gen_zero_ub(MEMORY_CLEAR_UB_ZERO_BUFF,
                                       self.data_type)
            for core_index in range(core_used):
                with self.tik_instance.if_scope(core_index == block_idx):
                    current_segs = core_segs[core_index]
                    for addr_index, offset, data_size in current_segs:
                        data_addr = data_addr_list[addr_index]
                        self.clean_workspace_single_core(data_size,
                                                         data_addr[offset],
                                                         zero_ub)

    def tik_instance_fun(self, kernel_name):
        for idx in range(0, self.workspace_num):
            addr_gm = self.tik_instance.Tensor("float16",
                                               (self.workspace_sizes[idx] / 2,),
                                               tik.scope_gm,
                                               "".join(["gm", str(idx)]))
            self.workspace_addrs.append(addr_gm)

        # clean workspaces using multi core, each workspace is processed integrally as possible
        self.workspaces_clean_multi_core_integrally(self.workspace_sizes,
                                                    self.workspace_addrs)
        self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                   inputs=self.workspace_addrs,
                                   outputs=(), enable_l2=False)
        return self.tik_instance


@util.check_input_type((list, tuple), str)
def atomic_addr_clean(size_list, kernel_name="atomic_clean"):
    """
    clean memory of workspace list
    Parameters
    ----------
    size_list :  list
        sizes of workspaces
    kernel_name : str
        kernel name, default value is "atomic_addr_clean"

    Returns
    -------
    tik_instance
    """

    for msize in size_list:
        if msize <= 0:
            raise RuntimeError("workspace size must be greater than 0!")
        if msize % 32 != 0:
            raise RuntimeError(
                "workspace size must be able to be divided by 32!")

    atomic_clean = AtomicCleaner(size_list)
    return atomic_clean.tik_instance_fun(kernel_name)
