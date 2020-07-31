#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

segment speel schedule
"""
from .segment_schedule import CceSegmentOp

SPEEL_FACTOR = 128


class CceSegmentSpeelOp(CceSegmentOp):
    """
    segment speel schedule
    """
    def split(self, res):
        self._split_axis, self._factor_, tiling_result = self.segment_op_tiling()
        if not tiling_result:
            return None, None, False
        out_shape = [i.value for i in self._op_list[-1]["dst_buffer"][0].shape]
        split_outer, split_inner = self._schedule[res].split(res.op.axis[self._split_axis],
                                                             out_shape[self._split_axis])
        self._schedule[res].speel(split_inner, self._factor_)
        return split_outer, split_inner, True

    def get_greatest_speel_axis_factor(self, in_shape, out_shape, max_ptr, align_factor):
        """
        in_shape, the segment op input shape
        out_shape, the segment op output shape
        max ptr, max "max ub use size"
        return speel_axis, factor

        in_shape[speel_axis] % factor != 0
        """
        # first try split last axis
        speel_axis = len(out_shape) - 1
        # first try split factor = 1, then factor+=1,
        # until use the this factor the mov data is bigger then ub size,
        # then the previous factor and speel_axis is best
        # when the factor reach the shape[speel_axis], speel_axis -=1, and factor reset to 1
        # that means we can try split the earlier axis, this can decrease the outer loop
        factor = 1

        while speel_axis > 0:
            tmp_factor = 2
            while tmp_factor <= out_shape[speel_axis]:
                _, valid_ub_use = self._calc_and_check_max_ub_use(speel_axis, tmp_factor, in_shape,
                                                                  out_shape, max_ptr, align_factor)
                if not valid_ub_use:
                    return speel_axis, factor
                factor = tmp_factor
                tmp_factor += 1
            speel_axis -= 1
            factor = 1
        return speel_axis, factor

    def segment_op_tiling(self):
        """
        a simple segment max example as follow:
        input data like (4,4,5)
        |------------------------------------------------------------------------------------|
        | 100,101,102,103,104, 110,111,112,113,114, 120,121,122,123,124, 130,131,132,133,134 |
        | 200,201,202,203,204, 210,211,212,213,214, 220,221,222,223,224, 230,231,232,233,234 |
        | 300,301,302,303,304, 310,311,312,313,314, 320,321,322,323,324, 330,331,332,333,334 |
        | 400,401,402,403,404, 410,411,412,413,414, 420,421,422,423,424, 430,431,432,433,434 |
        |------------------------------------------------------------------------------------|
        as the input data above, it's like the data in memory,
        in the memory it's a continuous memory.
        In order to understand the segment operation,
        show the data as 4 rows, 4 is the input_shape[0] dim size.
        segment max mean do operation with the data in the same position of each row.
        how to do the operation is refer to the segment_ids:

        output example 1:
        if the segment id [0,0,1,1], the segment id lenght is same with the the input_shape[0]
        the output is : the output shape will be (2,4,5)
        max(100,200) => 200, max(101,201) => 201, max(102,202) => 202 ...
        max(300,400) => 300, max(301,401) => 401, max(302,402) => 402 ...
        then all the output data as follow
        ----------------------------------------------------------------------------------
        200,201,202,203,204, 210,211,212,213,214, 220,221,222,223,224, 230,231,232,233,234
        400,401,402,403,404, 410,411,412,413,414, 420,421,422,423,424, 430,431,432,433,434
        ----------------------------------------------------------------------------------

        the cce code will like:
        for i1.outer in (0, 2)
            for ax0 in (0, 4) // 4 is in_shape[0]
                copy_gm_to_ub
            for i0 in (0,2)
                if i0 == 0
                    vmax()
                if i1 == 1
                    vmax()
                copy_ub_to_gm

        the dataflow in device will like:

        1) i1.outer loop 0

        copy_gm_to_ub, these data will in ub, this is continuous memory.
        |-------------------------------------------|
        | 100,101,102,103,104, 110,111,112,113,114, |
        | 200,201,202,203,204, 210,211,212,213,214, |  these is input data in ub
        | 300,301,302,303,304, 310,311,312,313,314, |
        | 400,401,402,403,404, 410,411,412,413,414, |
        |-------------------------------------------|
        |                                           |  these is for vmax result data
        |-------------------------------------------|

        1-1) inner loop 0, if i1 = 0,  do part_output_1 = vmax(part_row_1, part_row_2)
        |-------------------------------------------|
        | 100,101,102,103,104, 110,111,112,113,114, |
        | 200,201,202,203,204, 210,211,212,213,214, |  these is input data in ub
        | 300,301,302,303,304, 310,311,312,313,314, |
        | 400,401,402,403,404, 410,411,412,413,414, |
        |-------------------------------------------|
        | 200,201,202,203,204, 210,211,212,213,214, |  these is for vmax result data
        |-------------------------------------------|
                            ||
            copy_ub_to_gm   ||
        this out put data info
        |-------------------------------------------|  |------------------------------------------|
        | 200,201,202,203,204, 210,211,212,213,214, |  |                                          |
        |                                           |  |                                          |
        |-------------------------------------------|  |------------------------------------------|

        1-2) inner loop 1, if i1 = 1, do vmax
         |-------------------------------------------|
        | 100,101,102,103,104, 110,111,112,113,114, |
        | 200,201,202,203,204, 210,211,212,213,214, |  these is input data in ub
        | 300,301,302,303,304, 310,311,312,313,314, |
        | 400,401,402,403,404, 410,411,412,413,414, |
        |-------------------------------------------|
        | 400,401,402,403,404, 410,411,412,413,414, |  these is for vmax result data.(changed)
        |-------------------------------------------|
                            ||
            copy_ub_to_gm   ||
        this out put data info
        |-------------------------------------------|  |------------------------------------------|
        | 200,201,202,203,204, 210,211,212,213,214, |  |                                          |
        | 400,401,402,403,404, 410,411,412,413,414, |  |                                          |
        |-------------------------------------------|  |------------------------------------------|

        2) i1.outer loop 2
        copy these data to ub
        |------------------------------------------|
        | 120,121,122,123,124, 130,131,132,133,134 |
        | 220,221,222,223,224, 230,231,232,233,234 |  these is input data in ub
        | 320,321,322,323,324, 330,331,332,333,334 |
        | 420,421,422,423,424, 430,431,432,433,434 |
        |------------------------------------------|
        |                                          |  these is for vmax result data
        |------------------------------------------|
                            ||
                            ||
        2-1) inner loop 0, if i1=0, do vmax(part_row_1, part_row_2)
        |------------------------------------------|
        | 120,121,122,123,124, 130,131,132,133,134 |
        | 220,221,222,223,224, 230,231,232,233,234 |  these is input data in ub
        | 320,321,322,323,324, 330,331,332,333,334 |
        | 420,421,422,423,424, 430,431,432,433,434 |
        |------------------------------------------|
        | 220,221,222,223,224, 230,231,232,233,234 |  these is for vmax result data
        |------------------------------------------|
                            ||
            copy_ub_to_gm   ||
        this out put data info
        |-------------------------------------------|  |------------------------------------------|
        | 200,201,202,203,204, 210,211,212,213,214, |  | 220,221,222,223,224, 230,231,232,233,234 |
        | 400,401,402,403,404, 410,411,412,413,414, |  |                                          |
        |-------------------------------------------|  |------------------------------------------|

        2-2) inner loop 1, if i1=1, do vmax(part_row_3, part_row_4)
        |------------------------------------------|
        | 120,121,122,123,124, 130,131,132,133,134 |
        | 220,221,222,223,224, 230,231,232,233,234 |  these is input data in ub
        | 320,321,322,323,324, 330,331,332,333,334 |
        | 420,421,422,423,424, 430,431,432,433,434 |
        |------------------------------------------|
        | 420,421,422,423,424, 430,431,432,433,434 |  these is for vmax result data.(changed)
        |------------------------------------------|
                            ||
            copy_ub_to_gm   ||
        this out put data info
        |-------------------------------------------|  |------------------------------------------|
        | 200,201,202,203,204, 210,211,212,213,214, |  | 220,221,222,223,224, 230,231,232,233,234 |
        | 400,401,402,403,404, 410,411,412,413,414, |  | 420,421,422,423,424, 430,431,432,433,434 |
        |-------------------------------------------|  |------------------------------------------|

        last get the output data

        [Attention]
        here in device every command will handler a group of data.
        According to the max ub capacity and segment operation logic.
        should ensure: (in_shape[0] + 1) * group_data_size < max_ub_size

        according to device rule, the data in memory also need be aligned by align factor.
        so should ensure : (in_shape[0] + 1) * aligned_group_data_size < max_ub_size

        if the data is not aligned, like fp16 data, group_data_size % 16 != 0
        copy the result from ub to gm, will coverage the data in gm.
        example:
        in ub: |-----------------------|
               | 100,101,102,...114,   |
               |-----------------------|
               |    size = 16*n        |

        in gm  |------------------------------- |
               | 0,0.............0, 115,116,....|
               |--------------------------------|

        want copy these data from ub to gm, need copy from 115 (in gm) to ub
        after do copy_gm_ub(length=16)

        in ub:| 100,101,102,...114,   |  |115,....,131|
              |-------size 16*n-------|  |--size 16---|

        do 1) copy_ub_to_gm(length=16*n)
           2) copy_ub_to_gm(length=16)
        action as above can ensure the gm data won't be coveraged.
        should ensure ((in_shape[0] + 1) * aligned_group_data_size + (
                      align_factor if not_aligned else 0)) < max_ub_size

        according to the data flow and rules, find the greatest group_data_size
        """
        max_ptr = self.get_max_ptr()
        out_shape = [i.value for i in self._op_list[self._segment_op_pos]["dst_buffer"][0].shape]
        in_shape = [i.value for i in self._op_list[self._segment_op_pos]["src_buffer"][0].shape]
        # if the shape dim length is 1, build an AI CPU op is better, so return False
        if self._check_do_in_ai_cpu_better(in_shape):
            return 0, 0, False

        datatype = self._op_list[self._segment_op_pos]["dst_buffer"][0].dtype
        align_factor = self.get_align_factor(datatype)

        def fixed_split_info(speel_axis, factor):
            """
            if factor == 1 means the axis after the speel_axis can do tensorize
            change it to speel_axis + 1, and set the factor = the axis size after the split axis
            """
            if factor == 1:
                # if speel_axis is last axis and factor is 1, mean the last axis is a big speel
                if speel_axis == len(out_shape) - 1:
                    return speel_axis, factor, False
                speel_axis += 1
                factor = out_shape[speel_axis]
            return speel_axis, out_shape[speel_axis] // factor, True

        speel_axis, factor = self.get_greatest_speel_axis_factor(in_shape, out_shape, max_ptr,
                                                                 align_factor)
        return fixed_split_info(speel_axis, factor)
