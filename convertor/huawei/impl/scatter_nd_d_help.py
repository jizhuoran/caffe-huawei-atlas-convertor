#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

scatter_nd_d
"""
from te import tik
from impl import common_util
from impl import constant_util as constant

# max ub size(unit: element number)
MAX_UB_ELEMENT_NUMBER = 10240


# pylint: disable = locally-disabled,invalid-name,useless-object-inheritance
# pylint: disable = unused-argument,no-member,consider-using-in
class ScatterNdBase(object):
    """
       Function: use to store scatter_nd base parameters
       Modify : 2019-09-27
    """

    def __init__(self, input_param, tik_instance):
        """
        init scatter_nd base parameters

        Parameters
        ----------
        input_param: a tuple with indices,updates,output_y,shape
               input_param[0]is indices,dict,shape and datatype,
               datatype supports int32
               input_param[1] is updates,dict,shape and datatype,
               datatype supports float32,float16,int32,int8,uint8
               input_param[2] is output_y, dict,shape and datatype,
               datatype supports float32,float16,int32,int8,uint8
               input_param[3] is shape,out put shape
        tik_instance: tik_instance

        Returns
        -------
        None
        """
        self.tik_instance = tik_instance
        self.indices = input_param[0]
        self.updates = input_param[1]
        self.output_y = input_param[2]
        self.shape = input_param[3]
        self.data_size = common_util.get_data_size(
            self.updates.get("dtype").lower())

    def get_instance(self):
        """
        init scatter_nd  parameters

        Parameters
        ----------
        None

        Returns
        -------
        tik_instance: tik_instance
        """
        return self.tik_instance

    def get_data_size(self):
        """
        get data size

        Parameters
        ----------
        None

        Returns
        -------
        data_size: data_size
        """
        return self.data_size


class ScatterNd(ScatterNdBase):
    """
       Function: use to store scatter_nd  parameters
       Modify : 2019-09-27
    """

    def __init__(self, input_param, tik_instance):
        """
        init scatter_nd  parameters

        Parameters
        ----------
        input_param: a tuple with indices,updates,output_y,shape
               input_param[0]is indices,dict,shape and datatype,
               datatype supports int32
               input_param[1] is updates,dict,shape and datatype,
               datatype supports float32,float16,int32,int8,uint8
               input_param[2] is output_y, dict,shape and datatype,
               datatype supports float32,float16,int32,int8,uint8
               input_param[3] is shape,out put shape
        tik_instance: tik_instance

        Returns
        -------
        None
        """
        super(ScatterNd, self).__init__(input_param, tik_instance)
        updates = input_param[1]
        oneburst_num = constant.BLOCK_SIZE // common_util.get_data_size(
            input_param[0].get("dtype").lower())
        ind_shape = input_param[0].get("shape")
        indices_gm_num = get_gm_number(oneburst_num, ind_shape)
        indices_dtype = input_param[0].get("dtype").lower()

        if indices_gm_num > MAX_UB_ELEMENT_NUMBER and MAX_UB_ELEMENT_NUMBER % \
                ind_shape[-1] != 0:
            ind_ub_size = (MAX_UB_ELEMENT_NUMBER // ind_shape[-1]) * ind_shape[
                -1]
            last_ub = indices_gm_num % ind_ub_size

            if last_ub % oneburst_num != 0:
                last_size = (last_ub // oneburst_num + 1) * oneburst_num
                indices_gm_num = indices_gm_num - last_ub + last_size

        self.input_indices_gm = \
            self.tik_instance.Tensor(indices_dtype, (indices_gm_num,),
                                     name="input_indices_gm",
                                     scope=tik.scope_gm)
        oneburst_num = constant.BLOCK_SIZE // common_util.get_data_size(
            updates.get("dtype").lower())
        update_gm_num = self.get_last_alignment_gm_num(
            self.updates.get("shape"), oneburst_num)

        updates_dtype = updates.get("dtype").lower()
        self.input_updates_gm = \
            self.tik_instance.Tensor(updates_dtype, (update_gm_num,),
                                     name="input_updates_gm",
                                     scope=tik.scope_gm)

        out_gm_num = self.get_last_alignment_gm_num(input_param[3],
                                                    oneburst_num)
        self.output_y_gm = self.tik_instance.Tensor(updates_dtype,
                                                    (out_gm_num,),
                                                    name="output_y_gm",
                                                    scope=tik.scope_gm)

    def get_last_alignment_gm_num(self, shape, oneburst_num):
        """
        get last alignment gm number

        Parameters
        ----------
        shape: out put shape
        oneburst_num: data move oneburst can move
                                 the number of the elements

        Returns
        -------
        total_num: total gm element number
        """
        indice_len = get_indice_len(self.indices.get("shape"))
        update_each_size = get_shape_total_number(
            self.updates.get("shape")) // indice_len
        shape_len = get_shape_total_number(shape)
        move_times = shape_len // update_each_size
        total_num = update_each_size * (move_times - 1)

        # must 32b alignment
        if update_each_size % oneburst_num != 0:
            update_each_size = (update_each_size // oneburst_num + 1) \
                               * oneburst_num
        total_num = total_num + update_each_size

        return total_num

    def initial_output(self, process, output_offset, output_size):
        """
        init output as 0

        Parameters
        ----------
        process: ScatterProcess class,which used to store scatter nd parameters
        output_offset: the offset of output data
        output_size: each cycle process the data size

        Returns
        -------
        None
        """
        loop_times = self.tik_instance.Scalar("int32")
        loop_times.set_as(output_size / MAX_UB_ELEMENT_NUMBER)
        last_ub_size = self.tik_instance.Scalar("int32")
        last_ub_size.set_as(output_size % MAX_UB_ELEMENT_NUMBER)
        with self.tik_instance.if_scope(last_ub_size != 0):
            loop_times.set_as(loop_times + 1)
        with self.tik_instance.else_scope():
            last_ub_size.set_as(MAX_UB_ELEMENT_NUMBER)
        offset = self.tik_instance.Scalar("int32")
        offset.set_as(output_offset)
        total_size = self.tik_instance.Scalar("int32")
        total_size.set_as(MAX_UB_ELEMENT_NUMBER * self.data_size)

        with self.tik_instance.for_range(0, loop_times) as times:
            with self.tik_instance.if_scope(times == loop_times - 1):
                total_size.set_as(last_ub_size * self.data_size)
            repeats = common_util.get_vector_repeat_times(self.tik_instance,
                                                          total_size)
            nburst = common_util.get_datamove_nburst(self.tik_instance,
                                                     total_size)
            self.generate_input_data(process, repeats)
            with self.tik_instance.if_scope(total_size % \
                                            constant.BLOCK_SIZE != 0):
                self.move_out_non32_alignment(process, offset,
                                              total_size // self.data_size)
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(self.output_y_gm[offset],
                                            process.input_ub,
                                            constant.SID,
                                            constant.DEFAULT_NBURST, nburst,
                                            constant.STRIDE_ZERO,
                                            constant.STRIDE_ZERO)
            offset.set_as(offset + total_size // self.data_size)

    def update_data(self, process, loop_cycle, output_offset):
        """
        update output data

        Parameters
        ----------
        process: ScatterProcess class,which used to store scatter nd parameters
        loop_cycle: each block's loop cycle
        output_offset: the offset of output

        Returns
        -------
        None
        """
        loop_size = MAX_UB_ELEMENT_NUMBER
        if MAX_UB_ELEMENT_NUMBER % process.get_each_size() != 0:
            loop_size = (MAX_UB_ELEMENT_NUMBER // process.get_each_size()) * \
                        process.get_each_size()
        total = get_shape_total_number(self.indices.get("shape"))
        if total <= MAX_UB_ELEMENT_NUMBER:
            loop_size = total
        loop_times, last_ub_size = get_loop_param(total, loop_size)
        total_size = self.tik_instance.Scalar("int32")
        total_size.set_as(loop_size * constant.DATA_SIZE_FOUR)
        offset = self.tik_instance.Scalar("int32")
        offset.set_as(0)

        with self.tik_instance.for_range(0, loop_times) as times:
            with self.tik_instance.if_scope(times == loop_times - 1):
                total_size.set_as(last_ub_size * constant.DATA_SIZE_FOUR)
            nburst = common_util.get_datamove_nburst(self.tik_instance,
                                                     total_size)
            self.tik_instance.data_move(process.input_indices_ub,
                                        self.input_indices_gm[offset],
                                        constant.SID, constant.DEFAULT_NBURST,
                                        nburst, constant.STRIDE_ZERO,
                                        constant.STRIDE_ZERO)
            offset.set_as(offset + loop_size)
            inner_loop_times = self.tik_instance.Scalar("int32")
            inner_loop_times.set_as(total_size // process.get_each_size() // \
                                    constant.DATA_SIZE_FOUR)
            loop_param = (times, loop_size, inner_loop_times, loop_cycle)
            self.process_each_indices(process, loop_param, output_offset)

    def process_each_indices(self, process, loop_param, output_offset):
        """
        process each indices

        Parameters
        ----------
        process: ScatterProcess class,which used to store scatter nd parameters
        loop_param: a tupe keep the loop params
        output_offset: the offset of output data

        Returns
        -------
        None
        """
        update_offset = self.tik_instance.Scalar("int32")
        update_offset.set_as(0)
        with self.tik_instance.for_range(0, loop_param[2]) as ind_cycle:
            start_address = self.tik_instance.Scalar("int32")
            start_address.set_as(0)
            for k in range(process.get_each_size()):
                indices_saclar = self.tik_instance.Scalar("int32")
                indices_saclar.set_as(0)
                indices_saclar.set_as(ind_cycle * process.get_each_size() + k)
                indices_saclar.set_as(process.input_indices_ub[indices_saclar])
                start_address.set_as(start_address + \
                                     indices_saclar * \
                                     process.elem_of_each_dim[k])
            end_address = self.tik_instance.Scalar("int32")
            end_address.set_as(
                output_offset + (loop_param[3] - 1) * process.update_each_size)
            with self.tik_instance.if_scope(tik.all(start_address \
                                                    >= output_offset,
                                                    start_address \
                                                    <= end_address)):
                update_offset.set_as((loop_param[0] * loop_param[1] // \
                                      process.get_each_size() + \
                                      ind_cycle) * process.update_each_size)
                self.update_each_slice(process, update_offset, start_address)

    def update_each_slice(self, process, update_offset, start_address):
        """
        update each update slice

        Parameters
        ----------
        process: ScatterProcess class,which used to store scatter nd parameters
        update_offset: the offset of gm update data
        start_address: the start_address of output data

        Returns
        -------
        None
        """
        output_offset = self.tik_instance.Scalar("int32")
        output_offset.set_as(start_address)
        total_size = self.tik_instance.Scalar("int32")
        total_size.set_as(MAX_UB_ELEMENT_NUMBER * self.data_size)
        with self.tik_instance.for_range(0,
                                         process.loop_update) as update_cycle:
            with self.tik_instance.if_scope(update_cycle == \
                                            process.loop_update - 1):
                total_size.set_as(process.last_update_ub_size * self.data_size)
            nburst = common_util.get_datamove_nburst(self.tik_instance,
                                                     total_size)
            repeats = common_util.get_vector_repeat_times(self.tik_instance,
                                                          total_size)
            self.tik_instance.data_move(process.input_update_ub,
                                        self.input_updates_gm[update_offset],
                                        constant.SID,
                                        constant.DEFAULT_NBURST, nburst,
                                        constant.STRIDE_ZERO,
                                        constant.STRIDE_ZERO)

            self.tik_instance.data_move(process.input_ub,
                                        self.output_y_gm[output_offset],
                                        constant.SID,
                                        constant.DEFAULT_NBURST, nburst,
                                        constant.STRIDE_ZERO,
                                        constant.STRIDE_ZERO)

            dtype = process.input_ub.dtype
            input_ub_fp16 = None
            if dtype == constant.DATA_TYPE_INT8 \
                    or dtype == constant.DATA_TYPE_UINT8:
                input_shape = process.input_update_ub.shape
                total_number = constant.DATA_SIZE_TWO * \
                               get_shape_total_number(input_shape)
                input_ub_fp16 = self.tik_instance.Tensor(
                    constant.DATA_TYPE_FP16,
                    (total_number,),
                    name="input_ub_fp16",
                    scope=tik.scope_ubuf)
                self.tik_instance.vconv(constant.MASK128, "", input_ub_fp16,
                                        process.input_ub,
                                        repeats * constant.DATA_SIZE_TWO,
                                        constant.STRIDE_ONE,
                                        constant.STRIDE_ONE,
                                        constant.REPEAT_STRIDE_EIGHT,
                                        constant.REPEAT_STRIDE_FOUR)
            element_num = self.tik_instance.Scalar("int32")
            element_num.set_as(total_size // self.data_size)

            self.add_same_indices(process, repeats, input_ub_fp16, element_num)
            with self.tik_instance.if_scope(tik.all(total_size % \
                                                    constant.BLOCK_SIZE != 0, \
                                                    process.update_each_size * \
                                                    self.data_size >= \
                                                    constant.BLOCK_SIZE)):
                self.move_out_non32_alignment(process, output_offset,
                                              element_num)
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(self.output_y_gm[output_offset],
                                            process.input_ub,
                                            constant.SID,
                                            constant.DEFAULT_NBURST, nburst,
                                            constant.STRIDE_ZERO,
                                            constant.STRIDE_ZERO)
            output_offset.set_as(output_offset + element_num)
            update_offset.set_as(update_offset + element_num)

    def move_out_non32_alignment(self, process, output_address, total_number):
        """
        move out non32 alignment

        Parameters
        ----------
        process: ScatterProcess class,which used to store scatter nd parameters
        output_address: output offset of output data
        total_number: total element number

        Returns
        -------
        None
        """
        elements_number = constant.BLOCK_SIZE // self.data_size
        out_ub = self.tik_instance.Tensor(process.input_ub.dtype,
                                          (elements_number,),
                                          name="out_ub",
                                          scope=tik.scope_ubuf)

        nbursts = (total_number * self.data_size) // constant.BLOCK_SIZE
        scalar = self.tik_instance.Scalar(process.input_ub.dtype)
        with self.tik_instance.if_scope(nbursts == 0):
            offset = elements_number - total_number
            self.tik_instance.data_move(out_ub,
                                        self.output_y_gm[
                                            output_address - offset],
                                        constant.SID, constant.DEFAULT_NBURST,
                                        constant.DEFAULT_BURST_LEN,
                                        constant.STRIDE_ZERO,
                                        constant.STRIDE_ZERO)

            with self.tik_instance.for_range(0, total_number) as out_cycle:
                scalar.set_as(process.input_ub[out_cycle])
                out_ub[offset + out_cycle].set_as(scalar)
            self.tik_instance.data_move(
                self.output_y_gm[output_address - offset],
                out_ub,
                constant.SID, constant.DEFAULT_NBURST,
                constant.DEFAULT_BURST_LEN,
                constant.STRIDE_ZERO, constant.STRIDE_ZERO)

        with self.tik_instance.else_scope():
            self.tik_instance.data_move(self.output_y_gm[output_address],
                                        process.input_ub,
                                        constant.SID, constant.DEFAULT_NBURST,
                                        nbursts,
                                        constant.STRIDE_ZERO,
                                        constant.STRIDE_ZERO)
            offset = total_number - elements_number
            scalar = self.tik_instance.Scalar(process.input_ub.dtype)
            with self.tik_instance.for_range(0, elements_number) as time:
                scalar.set_as(process.input_ub[offset + time])
                out_ub[time].set_as(scalar)
            self.tik_instance.data_move(
                self.output_y_gm[output_address + offset], out_ub,
                constant.SID, constant.DEFAULT_NBURST,
                constant.DEFAULT_BURST_LEN,
                constant.STRIDE_ZERO, constant.STRIDE_ZERO)

    def generate_input_data(self, process, repeats):
        """
        generate data according to different datatype

        Parameters
        ----------
        process: ScatterProcess class,which used to store scatter nd parameters
        repeats: instructions repeats

        Returns
        -------
        input_ub_fp16: a ub tensor when process.input_ub.dtype is int8 \
                       or uint8 is effective
        """
        dtype = process.input_ub.dtype
        input_ub_fp16 = None
        if dtype == constant.DATA_TYPE_INT8 or dtype == \
                constant.DATA_TYPE_UINT8:
            total_number = constant.DATA_SIZE_TWO * \
                           get_shape_total_number(process.input_ub.shape)
            input_ub_fp16 = self.tik_instance.Tensor(constant.DATA_TYPE_FP16,
                                                     (total_number,),
                                                     name="input_ub_fp16",
                                                     scope=tik.scope_ubuf)
            self.tik_instance.vector_dup(constant.MASK128, input_ub_fp16,
                                         constant.INT_DEFAULT_ZERO,
                                         repeats * constant.DATA_SIZE_TWO,
                                         constant.STRIDE_ONE,
                                         constant.REPEAT_STRIDE_EIGHT)
            self.tik_instance.vconv(constant.MASK128, "", process.input_ub,
                                    input_ub_fp16,
                                    repeats * constant.DATA_SIZE_TWO,
                                    constant.STRIDE_ONE,
                                    constant.STRIDE_ONE,
                                    constant.REPEAT_STRIDE_FOUR,
                                    constant.REPEAT_STRIDE_EIGHT)
        else:
            self.tik_instance.vector_dup(256//self.data_size, process.input_ub,
                                         constant.INT_DEFAULT_ZERO,
                                         repeats, constant.STRIDE_ONE,
                                         constant.REPEAT_STRIDE_EIGHT)

        return input_ub_fp16

    def add_same_indices(self, process, repeats, input_ub_fp16, element_num):
        """
        add the data with the same indices

        Parameters
        ----------
        process: ScatterProcess class,which used to store scatter nd parameters
        repeats: instructions repeats
        input_ub_fp16: a ub tensor when process.input_ub.dtype is int8 or
                       uint8 is effective
        element_num: the number of elements

        Returns
        -------
        None
        """
        mask = constant.MASK128
        if process.update_each_size * self.data_size < constant.BLOCK_SIZE:
            mask = element_num
        dtype = process.input_ub.dtype
        if dtype == constant.DATA_TYPE_INT8 or dtype == \
                constant.DATA_TYPE_UINT8:
            total_number = constant.DATA_SIZE_TWO * \
                           get_shape_total_number(process.input_update_ub.shape)
            input_update_ub_fp16 = self.tik_instance.Tensor(
                constant.DATA_TYPE_FP16,
                (total_number,),
                name="input_update_ub_fp16",
                scope=tik.scope_ubuf)

            self.tik_instance.vconv(constant.MASK128, "", input_update_ub_fp16,
                                    process.input_update_ub,
                                    repeats * constant.DATA_SIZE_TWO,
                                    constant.STRIDE_ONE,
                                    constant.STRIDE_ONE,
                                    constant.REPEAT_STRIDE_EIGHT,
                                    constant.REPEAT_STRIDE_FOUR)
            self.tik_instance.vadd(mask, input_ub_fp16, input_ub_fp16,
                                   input_update_ub_fp16,
                                   repeats * constant.DATA_SIZE_TWO,
                                   constant.STRIDE_ONE,
                                   constant.STRIDE_ONE,
                                   constant.STRIDE_ONE,
                                   constant.REPEAT_STRIDE_EIGHT,
                                   constant.REPEAT_STRIDE_EIGHT,
                                   constant.REPEAT_STRIDE_EIGHT)
            self.tik_instance.vconv(constant.MASK128, "", process.input_ub,
                                    input_ub_fp16,
                                    repeats * constant.DATA_SIZE_TWO,
                                    constant.STRIDE_ONE,
                                    constant.STRIDE_ONE,
                                    constant.REPEAT_STRIDE_FOUR,
                                    constant.REPEAT_STRIDE_EIGHT)
        else:
            mask = 256//self.data_size
            if process.update_each_size * self.data_size < constant.BLOCK_SIZE:
                mask = element_num
            self.tik_instance.vadd(mask, process.input_ub,
                                   process.input_update_ub,
                                   process.input_ub,
                                   repeats, constant.STRIDE_ONE,
                                   constant.STRIDE_ONE,
                                   constant.STRIDE_ONE,
                                   constant.REPEAT_STRIDE_EIGHT,
                                   constant.REPEAT_STRIDE_EIGHT,
                                   constant.REPEAT_STRIDE_EIGHT)


class ScatterProcessBase(object):
    """
    Function: use to store scatter_nd  process parameters
    Modify : 2019-09-27
    """

    def __init__(self, tik_instance, updates, indices):
        """
        init scatter_nd  process parameters

        Parameters
        ----------
        tik_instance: tik_instance
        updates: dict,shape and datatype,datatype supports float32,
                 float16,int32,int8,uint8
        indices: dict,shape and datatype,datatype supports int32

        Returns
        -------
        None
        """
        self.input_update_ub = tik_instance.Tensor(updates.get("dtype").lower(),
                                                   (MAX_UB_ELEMENT_NUMBER,),
                                                   name="input_update_ub",
                                                   scope=tik.scope_ubuf)
        self.input_ub = tik_instance.Tensor(updates.get("dtype").lower(),
                                            (MAX_UB_ELEMENT_NUMBER,),
                                            name="input_ub",
                                            scope=tik.scope_ubuf)
        self.input_ub_tmp = tik_instance.Tensor(updates.get("dtype").lower(),
                                                (MAX_UB_ELEMENT_NUMBER,),
                                                name="input_ub_tmp",
                                                scope=tik.scope_ubuf)

        self.input_indices_ub = tik_instance.Tensor(
            indices.get("dtype").lower(),
            (MAX_UB_ELEMENT_NUMBER,),
            name="input_indices_ub",
            scope=tik.scope_ubuf)
        indices_shape = indices.get("shape")
        indice_len = get_indice_len(indices_shape)
        each_size = indices_shape[-1]
        self.ind_params = [indice_len, each_size]
        self.update_each_size = \
            get_shape_total_number(updates.get("shape")) // self.ind_params[0]

    def get_each_size(self):
        """
        get each size

        Parameters
        ----------
        None:

        Returns
        -------
        each_size: indices each_size
        """
        return self.ind_params[1]

    def get_indice_len(self):
        """
        get indices data size

        Parameters
        ----------
        None:

        Returns
        -------
        indices_data_size: indices data size
        """
        return self.ind_params[0]


class ScatterProcess(ScatterProcessBase):
    """
    Function: use to store scatter_nd  process parameters
    Modify : 2019-09-27
    """

    def __init__(self, tik_instance, updates, indices, shape):
        """
        init scatter_nd  process parameters

        Parameters
        ----------
        tik_instance: tik_instance
        updates: dict,shape and datatype,datatype supports float32,float16,
                 int32,int8,uint8
        indices: dict,shape and datatype,datatype supports int32
        shape: out put shape

        Returns
        -------
        None
        """
        super(ScatterProcess, self).__init__(tik_instance, updates, indices)
        self.loop_indices_cycle, self.last_ind_ub_size = get_loop_param(
            get_shape_total_number(indices.get("shape")),
            MAX_UB_ELEMENT_NUMBER)
        self.loop_update, self.last_update_ub_size = get_loop_param(
            self.update_each_size,
            MAX_UB_ELEMENT_NUMBER)
        self.loop_out_cycle, self.last_out_ub_size = get_loop_param(
            get_shape_total_number(shape),
            self.update_each_size)
        self.elem_of_each_dim = get_elem_of_each_dim(shape)

    def get_loop_out_cycle(self):
        """
        get loop out cycle

        Parameters
        ----------
        None

        Returns
        -------
        loop_out_cycle: loop out cycle
        """
        return self.loop_out_cycle

    def get_last_out_ub_size(self):
        """
        get last out ub size

        Parameters
        ----------
        None

        Returns
        -------
        last_out_ub_size: last out ub size
        """

        return self.last_out_ub_size


def get_loop_param(length, max_ub_num):
    """
    get loop parameters

    Parameters
    ----------
    length: total number
    max_ub_num: max of ub num

    Returns
    -------
    loop_cycle: loop cycle
    last_ub_num: the last data needs ub num
    """
    loop_cycle = length // max_ub_num
    last_ub_num = length % max_ub_num
    if last_ub_num != 0:
        loop_cycle = loop_cycle + 1
    else:
        last_ub_num = max_ub_num

    return loop_cycle, last_ub_num


def get_indice_len(indices_shape):
    """
    get indice len

    Parameters
    ----------
    indices_shape: indices shape

    Returns
    -------
    indice_len: the length of each slice
    """
    indice_len = 1
    for i in indices_shape[:-1]:
        indice_len = indice_len * i

    return indice_len


def get_elem_of_each_dim(shape):
    """
    get element of each dim

    Parameters
    ----------
    shape: out put shape

    Returns
    -------
    None
    """
    elem_of_each_dim = [1] * len(shape)
    for i in range(len(shape) - 1):
        j = i + 1
        while j < len(shape):
            elem_of_each_dim[i] = elem_of_each_dim[i] * shape[j]
            j = j + 1

    return elem_of_each_dim


def get_gm_number(oneburst_element_number, shape):
    """
    get gm number

    Parameters
    ----------
    oneburst_element_number: data move one burst element number
    shape: out put shape

    Returns
    -------
    None
    """
    gm_element_number = get_shape_total_number(shape)

    # must 32b alignment
    if gm_element_number % oneburst_element_number != 0:
        gm_element_number = (gm_element_number // oneburst_element_number + 1) \
                            * oneburst_element_number
    return gm_element_number


def get_shape_total_number(shape):
    """
    get the number of element from the shape

    Parameters
    ----------
    shape: out put shape

    Returns
    -------
    total_number: the number of element of the shape
    """
    total_number = len(shape)
    if total_number == 0:
        return 0
    total_number = 1
    for i in shape:
        total_number = total_number * i

    return total_number
