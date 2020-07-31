#!/usr/bin/env python
# -*- coding:utf-8 -*-
# pylint: disable=R0912, R0914, R0913, R0915, C0302
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

roi_align
"""
from te import tik
from te import platform as tbe_platform
from topi.cce import util
from impl import roi_align_vbi
import math


NoneType = type(None)

# 16K size
UB_30K_SIZE = 150*1024

# CCE param
ROINUM = 128
C0SIZE = 16
ONEVECTOR = 128

NEG_ONE = -1.0
ZERO = 0.0
POINT_FIVE = 0.5
NEG_POINT_FIVE = -0.5
BLOCKNUM = 8
MASK_FP32 = 64
MASK_FP16 = 128

#FlowTabParam
ROIDIMOFFSET = 1
ONE = 1.0

ES_UB_SIZE = 192


class RoiAlign():
    def __init__(self, feature_map_dict,
                 rois_dict,
                 spatial_scale,
                 output_h,
                 output_w,
                 sample_ratio=2,
                 roi_end_mode=0,
                 kernel_name="roi_align"):

        # self.tik_inst = tik.Tik(tik.Dprofile(), False)
        self.tik_inst = tik.Tik(tik.Dprofile(), disable_debug=False)
        self.roi_end_mode = roi_end_mode
        self.input_shape = feature_map_dict.get("shape")
        self.input_dtype = feature_map_dict.get("dtype")
        self.rois_shape = rois_dict.get("shape")
        self.rois_dtype = rois_dict.get("dtype")

        self.input_n = self.input_shape[0]
        self.input_c1 = self.input_shape[1]
        self.input_h = self.input_shape[2]
        self.input_w = self.input_shape[3]
        self.input_c0 = self.input_shape[4]

        rois_total_num = self.rois_shape[0]
        self.rois_block_num = rois_total_num // ROINUM

        self.spatial_scale = spatial_scale
        self.pooled_h = output_h
        self.pooled_w = output_w
        self.out_offset = output_h * output_w * C0SIZE
        self.zeros_buf_dup_repeat = int(
            math.ceil(float(output_h * output_w * C0SIZE) / 128))

        if self.input_c0 * self.input_c1 > 1280:
            raise RuntimeError("feature map channel should less than 1280")

        if (self.input_h * self.input_w > 5248) and \
           (self.input_c0 * self.input_c1 * self.input_w >= 40960):
            raise RuntimeError("feature map H*W <= 5248 or C*W < 40960")

        if self.pooled_h > self.input_h or self.pooled_w > self.input_w:
            raise RuntimeError("pooled_h should less than feature map height \
                and pooled_w should less than feature map width")


        feature_map_h = feature_map_dict["shape"][2]
        feature_map_w = feature_map_dict["shape"][3]
        self.kernel_name = kernel_name


        self.feature_map = self.tik_inst.Tensor(self.input_dtype,
                                                self.input_shape,
                                                name="feature_map",
                                                scope=tik.scope_gm)
        self.rois = self.tik_inst.Tensor(self.input_dtype, self.rois_shape,
                                       name="rois", scope=tik.scope_gm)
        self.output_gm = self.tik_inst.Tensor(self.input_dtype,
          (rois_total_num, self.input_shape[1],  output_h, output_w,
           self.input_shape[4]), name="output_gm",
           scope=tik.scope_gm)

        self.batch_offset = self.input_c1 * self.input_c0 * \
                            self.input_h * self.input_w


        self.sample_ratio = sample_ratio
        if sample_ratio < 0:
            self.sample_ratio = -1

        self.repeat = int(math.ceil(float(self.input_c1) / 8))

        #feature map ub size  mini cloud : 164KB es:108KB
        ub_size_bytes = \
         tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE) // 1024
        self.max_fmap_ub_size = 164 * 32
        if ub_size_bytes != ES_UB_SIZE:
            if self.sample_ratio * self.pooled_h > 128 or \
               self.sample_ratio * self.pooled_w > 128:
                raise RuntimeError("sample_ratio * pooled_h should \
                                    no more than 128, \
                                    sample_ratio * pooled_w should \
                                    no more than 128")
            self.max_fmap_ub_size = 164 * 32
        else:
            if self.sample_ratio * self.pooled_h > 58 or \
               self.sample_ratio * self.pooled_w > 58:
                raise RuntimeError("sample_ratio * pooled_h should \
                                    no more than 58, \
                                    sample_ratio * pooled_w should \
                                    no more than 58")
            self.max_fmap_ub_size = 108 * 32

        self.flowtable_scale = [0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625]
        self.flowtable_scale_fp32 = \
                               [0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625]

        self.update_flowtable()

        #multi core param
        self.aicore_num = tik.Dprofile().get_aicore_num()
        self.total_rois_num = self.rois_shape[0]

        #first core
        self.each_core_rois_num = self.total_rois_num // self.aicore_num
        self.each_core_block_left = self.each_core_rois_num % ROINUM
        self.each_core_block_num = \
         int(math.ceil(float(self.each_core_rois_num) / ROINUM))
        #second core
        self.last_core_rois_num = \
         self.total_rois_num - self.each_core_rois_num * (self.aicore_num - 1)
        self.last_core_block_left = self.last_core_rois_num % ROINUM
        self.last_core_block_num = \
         int(math.ceil(float(self.last_core_rois_num) / ROINUM))

    def update_flowtable(self):
        data = self.spatial_scale  #0.0625(1 / 16)
        self.flowtable_scale[0] = data
        self.flowtable_scale_fp32[0] = data

        data = 1.0 / self.pooled_h  # 1 / 14
        self.flowtable_scale[1] = data
        self.flowtable_scale_fp32[1] = data

        data = 1.0 / self.pooled_w  # 1 / 14
        self.flowtable_scale[2] = data
        self.flowtable_scale_fp32[2] = data

        scale = self.pooled_h * self.sample_ratio
        data = 1.0 / scale
        self.flowtable_scale[3] = data
        self.flowtable_scale_fp32[3] = data

        scale = self.pooled_w* self.sample_ratio
        data = 1.0 / scale
        self.flowtable_scale[4] = data
        self.flowtable_scale_fp32[4] = data

        scale = self.sample_ratio * self.sample_ratio
        data = 1.0 / scale
        self.flowtable_scale[5] = data
        self.flowtable_scale_fp32[5] = data

        data = -1.0 / self.spatial_scale
        self.flowtable_threshold = data

    def roialign_compute(self):
        last_core_index = self.aicore_num - 1
        with self.tik_inst.for_range(0, self.aicore_num, block_num = self.aicore_num) as i:
            block_offset = self.tik_inst.Scalar("int32", init_value = 0)
            block_offset.set_as(i * self.each_core_rois_num)
            with self.tik_inst.if_scope(i != last_core_index):
                self.roialign_compute_each_core(self.each_core_block_num, block_offset, \
                 self.each_core_block_left)
            with self.tik_inst.else_scope():
                self.roialign_compute_each_core(self.last_core_block_num, block_offset, \
                 self.last_core_block_left)

        self.tik_inst.BuildCCE(kernel_name=self.kernel_name, \
         inputs=[self.feature_map, self.rois], outputs=[self.output_gm])
        return self.tik_inst

    def roialign_compute_each_core(self, block_num, block_offset, block_left):

        rois_index = self.tik_inst.Scalar("int32")
        rois_process_num = self.tik_inst.Scalar("int32")

        scale = self.tik_inst.Scalar("float16")
        scale.set_as((1 / math.pow(self.sample_ratio, 2)))

        hstart = self.tik_inst.Scalar("int32")
        hend = self.tik_inst.Scalar("int32")
        wstart = self.tik_inst.Scalar("int32")
        wend = self.tik_inst.Scalar("int32")

        max_length = self.tik_inst.Scalar("int32")
        max_length.set_as(self.max_fmap_ub_size // self.input_c1)

        width = self.tik_inst.Scalar("int32")
        height = self.tik_inst.Scalar("int32")

        src_stride = self.tik_inst.Scalar("int32")
        dst_stride = self.tik_inst.Scalar("int32", init_value = 0)
        cont_scalar = self.tik_inst.Scalar("int32")

        grid_h = self.tik_inst.Scalar("int32", init_value = self.sample_ratio)
        grid_w = self.tik_inst.Scalar("int32", init_value = self.sample_ratio)

        stride_h = self.tik_inst.Scalar("int32")
        stride_h.set_as(self.pooled_h * grid_h)
        stride_w = self.tik_inst.Scalar("int32")
        stride_w.set_as(self.pooled_w * grid_w)

        tile_height = self.tik_inst.Scalar("int32")
        s = self.tik_inst.Scalar("int32")
        block_tile = self.tik_inst.Scalar("int32")

        tile_h = self.tik_inst.Scalar("int32")

        tile_start = self.tik_inst.Scalar("int32")
        tile_end = self.tik_inst.Scalar("int32")

        h_dim = self.tik_inst.Scalar("int32")
        w_dim = self.tik_inst.Scalar("int32")

        curr_grid = self.tik_inst.Scalar("int32")
        curr_pos = self.tik_inst.Scalar("int32")
        grid_start = self.tik_inst.Scalar("int32")
        grid_end = self.tik_inst.Scalar("int32")

        curr_y = self.tik_inst.Scalar("int32")
        curr_yh = self.tik_inst.Scalar("int32")
        ly0 = self.tik_inst.Scalar("float16")
        hy0 = self.tik_inst.Scalar("float16")

        py = self.tik_inst.Scalar("int32")
        pyh = self.tik_inst.Scalar("int32")
        widthl = self.tik_inst.Scalar("int32")
        widthh = self.tik_inst.Scalar("int32")
        gh_left = self.tik_inst.Scalar("int32")

        curr_x = self.tik_inst.Scalar("int32")
        curr_xh = self.tik_inst.Scalar("int32")
        lx0 = self.tik_inst.Scalar("float16")
        hx0 = self.tik_inst.Scalar("float16")

        px = self.tik_inst.Scalar("int32")
        pxh = self.tik_inst.Scalar("int32")
        pw_c0 = self.tik_inst.Scalar("int32")
        n_grid_w_scalar = self.tik_inst.Scalar("int32")

        grid_len = self.tik_inst.Scalar("int32")
        t = self.tik_inst.Scalar("int32")
        id1 = self.tik_inst.Scalar("int32")
        id2 = self.tik_inst.Scalar("int32")
        id3 = self.tik_inst.Scalar("int32")
        id4 = self.tik_inst.Scalar("int32")

        batch_index = self.tik_inst.Scalar("int32")

        len_c1 = self.tik_inst.Scalar("int32")
        len_c1.set_as(self.repeat * 8 * C0SIZE)
        len_c2 = self.tik_inst.Scalar("int32")
        len_c2.set_as(2 * len_c1)
        len_c3 = self.tik_inst.Scalar("int32")
        len_c3.set_as(3 * len_c1)

        out_index = self.tik_inst.Scalar("int32")

        valid_rois_num = self.tik_inst.Scalar("int32", init_value = ROINUM)
        invalid_rois_num = self.tik_inst.Scalar("int32")

        last_block_index = self.tik_inst.Scalar("int32", init_value = block_num - 1)

        rois_block_ub = self.tik_inst.Tensor("float16", (128, 16), \
         scope = tik.scope_ubuf, name = "rois_block_ub")
        feature_map_ub = self.tik_inst.Tensor("float16", (self.max_fmap_ub_size * 16, ), \
         scope = tik.scope_ubuf, name = "feature_map")
        output_ub = self.tik_inst.Tensor("float16", \
         (self.repeat * self.pooled_w * 8 * C0SIZE, ), \
          scope = tik.scope_ubuf, name = "output_ub")
        val_ub = self.tik_inst.Tensor("float16", (self.repeat * ONEVECTOR * 4, ), \
         scope = tik.scope_ubuf, name = "val_ub")

        batch_index_int32 = self.tik_inst.Tensor("int32", (128, ), \
         scope = tik.scope_ubuf, name = "batch_index_int32")

        y_low_int = self.tik_inst.Tensor("int32", (128, ), \
         scope = tik.scope_ubuf, name = "y_low_int")
        y_high_int = self.tik_inst.Tensor("int32", (128, ), \
         scope = tik.scope_ubuf, name = "y_high_int")
        x_low_int = self.tik_inst.Tensor("int32", (128, ), \
         scope = tik.scope_ubuf, name = "x_low_int")
        x_high_int = self.tik_inst.Tensor("int32", (128, ), \
         scope = tik.scope_ubuf, name = "x_high_int")

        ly = self.tik_inst.Tensor("float16", (128, ), \
         scope = tik.scope_ubuf, name = "ly")
        lx = self.tik_inst.Tensor("float16", (128, ), \
         scope = tik.scope_ubuf, name = "lx")
        hy = self.tik_inst.Tensor("float16", (128, ), \
         scope = tik.scope_ubuf, name = "hy")
        hx = self.tik_inst.Tensor("float16", (128, ), \
         scope = tik.scope_ubuf, name = "hx")

        n_grid_h = self.tik_inst.Tensor("int32", (128, ), \
         scope =tik.scope_ubuf, name = "n_grid_h")
        n_grid_w = self.tik_inst.Tensor("int32", (128, ), \
         scope =tik.scope_ubuf, name = "n_grid_w")

        roi_gridh_fp32 = self.tik_inst.Tensor("float32", (128, ), \
         scope =tik.scope_ubuf, name = "roi_gridh_fp32")
        roi_gridw_fp32 = self.tik_inst.Tensor("float32", (128, ), \
         scope =tik.scope_ubuf, name = "roi_gridw_fp32")
        bin_scale = self.tik_inst.Tensor("float16", (128, ), \
         scope =tik.scope_ubuf, name = "bin_scale")

        x_start_ub = self.tik_inst.Tensor("float16", (ROINUM,), \
         name="x_start_ub", scope=tik.scope_ubuf)
        x_start_fp32_ub = self.tik_inst.Tensor("float32", (ROINUM,), \
         name="x_start_fp32_ub", scope=tik.scope_ubuf)
        y_start_fp32_ub = self.tik_inst.Tensor("float32", (ROINUM,), \
         name="y_start_fp32_ub", scope=tik.scope_ubuf)

        index_arr = self.tik_inst.Tensor("float16", (ROINUM, ), \
         name="index_arr", scope=tik.scope_ubuf)

        cont = self.tik_inst.Tensor("int32", (128, ), \
         name="cont", scope = tik.scope_ubuf)

        with self.tik_inst.new_stmt_scope():
            tmp = self.tik_inst.Scalar("float16", init_value = 0.5)
            index_arr_int32 = self.tik_inst.Tensor("int32", (ROINUM, ), \
             name="index_arr_int32", scope=tik.scope_ubuf)
            with self.tik_inst.for_range(0, 128) as i:
                index_arr_int32[i].set_as(i)
            self.tik_inst.vec_conv(64, '', index_arr, index_arr_int32, \
             2, 4, 8, 1.0)
            self.tik_inst.vec_adds(128, index_arr, index_arr, tmp, 1, 8, 8)
        #get block
        with self.tik_inst.for_range(0, block_num) as loop_b:
            rois_index.set_as(block_offset + loop_b * ROINUM)
            if block_left == 0:
                rois_process_num.set_as(ROINUM)
                with self.tik_inst.for_range(0, 128) as i:
                    self.tik_inst.data_move(rois_block_ub[i, 0], \
                     self.rois[rois_index + i, 0], 0, 1, 1, 0, 0)
            else:
                with self.tik_inst.if_scope(loop_b != last_block_index):
                    rois_process_num.set_as(ROINUM)
                    with self.tik_inst.for_range(0, 128) as i:
                        self.tik_inst.data_move(rois_block_ub[i, 0], \
                         self.rois[rois_index + i, 0], 0, 1, 1, 0, 0)
                with self.tik_inst.else_scope():
                    rois_process_num.set_as(block_left)
                    with self.tik_inst.for_range(0, block_left) as i:
                        self.tik_inst.data_move(rois_block_ub[i, 0], \
                         self.rois[rois_index + i, 0], 0, 1, 1, 0, 0)

            #transe rois coordinate
            self.tik_inst.vec_dup(64, cont, 1, 2, 8)
            valid_rois_num = self.roi_align_perf_scale(rois_block_ub, x_start_ub,rois_process_num, \
                                                       self.sample_ratio, n_grid_w, n_grid_h, \
                                                       x_start_fp32_ub, y_start_fp32_ub, \
                                                       roi_gridw_fp32, roi_gridh_fp32, batch_index_int32, \
                                                       bin_scale, cont, self.roi_end_mode)
            invalid_rois_num.set_as(rois_process_num - valid_rois_num)
            #get ROINUM
            with self.tik_inst.for_range(0, rois_process_num) as loop_n:
                cont_scalar.set_as(cont[loop_n])
                with self.tik_inst.if_scope(cont_scalar == 0):
                    batch_index.set_as(batch_index_int32[loop_n])
                    with self.tik_inst.new_stmt_scope():
                        self.roialign_perf_gen_grid(
                            index_arr,
                            n_grid_w, n_grid_h, loop_n,
                            lx, ly, hx, hy,
                            x_low_int, y_low_int, x_high_int, y_high_int,
                            x_start_fp32_ub, y_start_fp32_ub,
                            roi_gridh_fp32, roi_gridw_fp32,
                            self.input_w, self.input_h)

                    if self.sample_ratio <= 0:
                        grid_h.set_as(n_grid_h[loop_n])
                        grid_w.set_as(n_grid_w[loop_n])
                        scale.set_as(bin_scale[loop_n])

                        stride_h.set_as(self.pooled_h * grid_h - 1)
                        stride_w.set_as(self.pooled_w * grid_w - 1)

                    hstart.set_as(y_low_int[0])
                    hend.set_as(y_high_int[stride_h])
                    wstart.set_as(x_low_int[0])
                    wend.set_as(x_high_int[stride_w])

                    #cce : wend = (wend == W) ? W : wend + 1
                    with self.tik_inst.if_scope(wend != self.input_w):
                        wend.set_as(wend + 1)
                    with self.tik_inst.else_scope():
                        pass
                    #cce : hend = (hend == H) ? H : hend + 1
                    with self.tik_inst.if_scope(hend != self.input_h):
                        hend.set_as(hend + 1)
                    with self.tik_inst.else_scope():
                        pass

                    width.set_as(wend - wstart)
                    height.set_as(hend - hstart)
                    src_stride.set_as(self.input_w - width)
                    tile_h.set_as(max_length // width)

                    grid_start.set_as(0)
                    grid_end.set_as(0)
                    curr_grid.set_as(-1)
                    curr_pos.set_as(0)

                    h_dim.set_as(self.pooled_h * grid_h)
                    w_dim.set_as(self.pooled_w * grid_w)
                    gh_left.set_as(0)

                    tile_start.set_as(y_low_int[0])
                    tile_end.set_as(tile_start + 1)

                    #cce : for (ph = 0; ph <= hDim; ph++)
                    with self.tik_inst.for_range(0, h_dim + 1) as loop_ph:
                        with self.tik_inst.if_scope(loop_ph < h_dim):
                            curr_grid.set_as(curr_grid + 1)
                            curr_pos.set_as(y_high_int[loop_ph])
                        with self.tik_inst.else_scope():
                            pass

                        with self.tik_inst.if_scope(tik.all(((curr_pos - tile_start) < tile_h), \
                                                    (loop_ph < h_dim))):
                            tile_end.set_as(curr_pos)
                            grid_end.set_as(curr_grid)
                        with self.tik_inst.else_scope():
                            tile_height.set_as(tile_end - tile_start + 1)
                            s.set_as(tile_height * width)
                            #cce : for (uint64_t currC1 = 0; currC1 < C1; currC1++) {
                            with self.tik_inst.for_range(0, self.input_c1) as loop_c1:
                                self.tik_inst.data_move(feature_map_ub[loop_c1 * s * C0SIZE], \
                                                        self.feature_map[batch_index, loop_c1, tile_start, wstart, 0], \
                                                        0, tile_height, width, src_stride, dst_stride)


                            #cce : for (gh = gridStart; gh < (gridEnd + 1); gh++)
                            grid_len.set_as(grid_end - grid_start + 1)
                            with self.tik_inst.for_range(0, grid_len) as loop_gh:
                                #loop_gh.set_as(loop_gh + grid_start)
                                loop_gh = loop_gh + grid_start
                                curr_y.set_as(y_low_int[loop_gh])
                                curr_yh.set_as(y_high_int[loop_gh])
                                ly0.set_as(ly[loop_gh])
                                hy0.set_as(hy[loop_gh])

                                py.set_as(curr_y - tile_start)
                                pyh.set_as(curr_yh - tile_start)

                                widthl.set_as(py * width)
                                widthh.set_as(pyh * width)

                                gh_left.set_as(loop_gh % grid_h)
                                #cce : if (ghLeft == 0)
                                with self.tik_inst.if_scope(gh_left == 0):
                                    self.tik_inst.vec_dup(128, output_ub, 0, \
                                             self.repeat * self.pooled_w, 8)
                                with self.tik_inst.else_scope():
                                    pass

                                n_grid_w_scalar.set_as(-1)
                                with self.tik_inst.for_range(0, w_dim) as loop_gw:

                                    pw_c0.set_as(loop_gw // grid_w * C0SIZE)
                                    #cce : ngridW++
                                    n_grid_w_scalar.set_as(n_grid_w_scalar + 1)

                                    curr_x.set_as(x_low_int[n_grid_w_scalar])
                                    curr_xh.set_as(x_high_int[n_grid_w_scalar])

                                    lx0.set_as(lx[n_grid_w_scalar])
                                    hx0.set_as(hx[n_grid_w_scalar])

                                    px.set_as(curr_x - wstart)
                                    pxh.set_as(curr_xh - wstart)

                                    #m1*hx, m2*lx, m3*hx, m4*lx
                                    id1.set_as((widthl + px) * C0SIZE)
                                    id2.set_as((widthl + pxh) * C0SIZE)
                                    id3.set_as((widthh + px) * C0SIZE)
                                    id4.set_as((widthh + pxh) * C0SIZE)

                                    if self.repeat == 1:
                                        mask_tmp = self.input_c1 * 16
                                        self.tik_inst.vmuls(mask_tmp, val_ub[0], \
                                                            feature_map_ub[id1], \
                                                            hx0, 1, 1, s, 8, 8)
                                        self.tik_inst.vmuls(mask_tmp, val_ub[len_c1], \
                                                            feature_map_ub[id2], \
                                                            lx0, 1, 1, s, 8, 8)
                                        self.tik_inst.vmuls(mask_tmp, val_ub[len_c2], \
                                                            feature_map_ub[id3], \
                                                            hx0, 1, 1, s, 8, 8)
                                        self.tik_inst.vmuls(mask_tmp, val_ub[len_c3], \
                                                            feature_map_ub[id4], \
                                                            lx0, 1, 1, s, 8, 8)
                                    else :
                                        loop_repeat = self.input_c1 // 8
                                        left_mask = (self.input_c1 % 8) * 16
                                        with self.tik_inst.for_range(0, loop_repeat) as i:
                                            self.tik_inst.vmuls(128, val_ub[i * ONEVECTOR], \
                                                                feature_map_ub[i*s*ONEVECTOR + id1], \
                                                                hx0, 1, 1, s, 8, 8)
                                            self.tik_inst.vmuls(128, val_ub[i * ONEVECTOR + len_c1], \
                                                                feature_map_ub[i*s*ONEVECTOR + id2], \
                                                                lx0, 1, 1, s, 8, 8)
                                            self.tik_inst.vmuls(128, val_ub[i * ONEVECTOR + len_c2], \
                                                                feature_map_ub[i*s*ONEVECTOR + id3], \
                                                                hx0, 1, 1, s, 8, 8)
                                            self.tik_inst.vmuls(128, val_ub[i * ONEVECTOR + len_c3], \
                                                                feature_map_ub[i*s*ONEVECTOR + id4], \
                                                                lx0, 1, 1, s, 8, 8)
                                        if left_mask != 0:
                                            self.tik_inst.vmuls(left_mask, val_ub[(loop_repeat - 1) * ONEVECTOR], \
                                                                feature_map_ub[(loop_repeat - 1) * s * ONEVECTOR + id1], \
                                                                hx0, 1, 1, s, 8, 8)
                                            self.tik_inst.vmuls(left_mask, val_ub[(loop_repeat - 1) * ONEVECTOR + len_c1], \
                                                                feature_map_ub[(loop_repeat - 1) * s * ONEVECTOR + id2], \
                                                                lx0, 1, 1, s, 8, 8)
                                            self.tik_inst.vmuls(left_mask, val_ub[(loop_repeat - 1) * ONEVECTOR + len_c2], \
                                                                feature_map_ub[(loop_repeat - 1) * s * ONEVECTOR + id3], \
                                                                hx0, 1, 1, s, 8, 8)
                                            self.tik_inst.vmuls(left_mask, val_ub[(loop_repeat - 1) * ONEVECTOR + len_c3], \
                                                                feature_map_ub[(loop_repeat - 1) * s * ONEVECTOR + id4], \
                                                                lx0, 1, 1, s, 8, 8)

                                    #m1*hx + m2*lx
                                    self.tik_inst.vec_add(128, val_ub, val_ub, val_ub[len_c1], \
                                                       self.repeat, 8, 8, 8)
                                    #m3*hx + m4*lx
                                    self.tik_inst.vec_add(128, val_ub[len_c2], val_ub[len_c2], \
                                                       val_ub[len_c3], self.repeat, 8, 8, 8)
                                    #(m1*hx + m2*lx) * hy
                                    self.tik_inst.vec_muls(128, val_ub, val_ub, hy0, \
                                                        self.repeat, 8, 8)
                                    #(m3*hx + m4*lx) * ly
                                    self.tik_inst.vec_muls(128, val_ub[len_c2], val_ub[len_c2], \
                                                        ly0, self.repeat, 8, 8)
                                    #add(m1*hx + m2*lx) * hy to vout
                                    self.tik_inst.vadd(128, output_ub[pw_c0], output_ub[pw_c0], \
                                                       val_ub, self.repeat, \
                                                       self.pooled_w, self.pooled_w, 1, \
                                                       8 * self.pooled_w, 8 * self.pooled_w, 8)
                                    #add(m3*hx + m4*lx) * ly to vout
                                    self.tik_inst.vadd(128, output_ub[pw_c0], output_ub[pw_c0], \
                                                       val_ub[len_c2], self.repeat, \
                                                       self.pooled_w, self.pooled_w, 1, \
                                                       8 * self.pooled_w, 8 * self.pooled_w, 8)

                                #end of loop_gw
                                #if is the last grid in pooling bin,
                                #do average pooling and move data from ub to out
                                t.set_as(grid_h - 1)
                                if self.sample_ratio != 1:
                                    with self.tik_inst.if_scope(gh_left == t):
                                        #cce : vmuls scale  scale = 1 / pow(sample_ratio, 2)
                                        self.tik_inst.vec_muls(128, output_ub, output_ub, scale, \
                                                            self.repeat * self.pooled_w, 8, 8)

                                with self.tik_inst.if_scope(gh_left == (grid_h - 1)):
                                    #mov ub to gm
                                    out_index.set_as(loop_gh // grid_h)
                                    #out_index.set_as(1)
                                    self.tik_inst.data_move(self.output_gm[rois_index + loop_n, 0, \
                                                            out_index, 0, 0], output_ub, 0, \
                                                            self.input_c1, self.pooled_w, 0, \
                                                            (self.pooled_h - 1) * self.pooled_w)
                            #end of loop_gh
                            with self.tik_inst.if_scope(loop_ph < h_dim):
                                tile_start.set_as(y_low_int[loop_ph])
                                tile_end.set_as(curr_pos)
                                grid_start.set_as(curr_grid)
                                grid_end.set_as(curr_grid)
                        #end of else
                    #end of loop_ph
                with self.tik_inst.else_scope():
                    #for the redundant Rois, write zero to output
                    self.tik_inst.vec_dup(128, feature_map_ub, 0, self.zeros_buf_dup_repeat, 8)
                    with self.tik_inst.for_range(0, self.input_c1) as loop_j:
                        #mov ub to out
                        self.tik_inst.data_move(self.output_gm[block_offset + loop_b * ROINUM + loop_n, loop_j, 0, 0, 0], \
                                                feature_map_ub, 0, 1, self.pooled_h * self.pooled_w, 0, 0)
                        # end of loop_j
                    # end of loop_n1
            #end of loop_n
        # end of loop_b

    def roi_align_perf_scale(self, rois, xstart, roisnum, sample_ratio,
                             roi_gridw, roi_gridh, x_start_fp32_ub,
                             y_start_fp32_ub, roi_gridw_fp32,
                             roi_gridh_fp32, batch_index_int32,
                             binscale, cont, roi_end_mode = 0):
        """
        calc roiAlgin scale
        :param rois: rois tensor. shape:(n, c1, 1, 1, 16) dtype: float16
        :param xstart: the startx tensor. shape:(128, ) dtype: float16
        :param roisnum: roi numbers. shape(1,), dtype: int32
        :param roi_gridw: grid width tensor. shape(128,), dtype:int32, output
        :param roi_gridh: grid height tensor. shape(128,), dtype:float16, output
        :param roi_gridw_fp32: roi gridw tensor. shape(128,), dtype:float32, output
        :param roi_gridh_fp32: roi gridh tensor. shape(128,), dtype:float32, output
        :param binscale: output tensor. shape(128,), dtype:float16, output
        :return:
            count: valid roi number
        """
        count = self.tik_inst.Scalar("int32")
        count.set_as(0)

        with self.tik_inst.new_stmt_scope():
            cmp_buf = self.tik_inst.Tensor("float16", (ROINUM,), \
             name="cmp_buf_ub", scope=tik.scope_ubuf)
            cmp_res = self.tik_inst.Tensor("uint16", (ROINUM,), \
             name="cmp_res_ub", scope=tik.scope_ubuf)
            temp_ub = self.tik_inst.Tensor("uint16", (ROINUM,), \
             name="temp_ub", scope=tik.scope_ubuf)
            cmp16_ub = self.tik_inst.Tensor("uint16", (ROINUM,), \
             name="cmp16_ub", scope=tik.scope_ubuf)
            roi_buf1_ub = self.tik_inst.Tensor("float16", (4, ROINUM), \
             name="roi_buf2_ub", scope=tik.scope_ubuf)
            transpose_ub = self.tik_inst.Tensor("float16", (16, 16), \
             name="transpose_ub", scope=tik.scope_ubuf)

            x_end_fp32_ub = self.tik_inst.Tensor("float32", (ROINUM,), \
             name="x_end_fp32_ub", scope=tik.scope_ubuf)
            y_end_fp32_ub = self.tik_inst.Tensor("float32", (ROINUM,), \
             name="y_end_fp32_ub", scope=tik.scope_ubuf)
            roi_w_ub = self.tik_inst.Tensor("float16", (ROINUM,), \
             name="roi_w_ub", scope=tik.scope_ubuf)
            roi_h_ub = self.tik_inst.Tensor("float16", (ROINUM,), \
             name="roi_h_ub", scope=tik.scope_ubuf)
            gird_h_fp32_ub = self.tik_inst.Tensor("float32", (ROINUM,), \
             name="gird_h_fp32_ub", scope=tik.scope_ubuf)
            gird_w_fp32_ub = self.tik_inst.Tensor("float32", (ROINUM,), \
             name="gird_w_fp32_ub", scope=tik.scope_ubuf)
            roi_bin_h_ub = self.tik_inst.Tensor("float16", (ROINUM,), \
             name="roi_bin_h_ub", scope=tik.scope_ubuf)
            roi_bin_w_ub = self.tik_inst.Tensor("float16", (ROINUM,), \
             name="roi_bin_w_ub", scope=tik.scope_ubuf)
            n_grid_h_int_ub = self.tik_inst.Tensor("int32", (ROINUM,), \
             name="n_grid_h_int_ub", scope=tik.scope_ubuf)
            n_grid_w_int_ub = self.tik_inst.Tensor("int32", (ROINUM,), \
             name="n_grid_w_int_ub", scope=tik.scope_ubuf)
            n_grid_h_ub = self.tik_inst.Tensor("float16", (ROINUM,), \
             name="n_grid_h_ub", scope=tik.scope_ubuf)
            n_grid_w_ub = self.tik_inst.Tensor("float16", (ROINUM,), \
             name="n_grid_w_ub", scope=tik.scope_ubuf)
            n_grid_rec_w_ub = self.tik_inst.Tensor("float16", (ROINUM,), \
             name="n_grid_rec_w_ub", scope=tik.scope_ubuf)
            n_grid_rec_h_ub = self.tik_inst.Tensor("float16", (ROINUM,), \
             name="n_grid_rec_h_ub", scope=tik.scope_ubuf)
            roi_temp_ub = self.tik_inst.Tensor("float16", (16, ROINUM), \
             name="roi_temp_ub", scope=tik.scope_ubuf)
            roi_buf2_ub = self.tik_inst.Tensor("float16", (16 * ROINUM,), \
             name="roi_buf2_ub", scope=tik.scope_ubuf)
            batch_index_fp16 = self.tik_inst.Tensor("float16", (ROINUM, ), \
             name="batch_index_fp16", scope=tik.scope_ubuf)
            tmp_value = self.tik_inst.Scalar("int32", init_value = 0)

            self.tik_inst.vec_dup(128, cmp_buf, self.flowtable_threshold, 1, 8)
            self.tik_inst.vcmpv_ge(cmp_res, rois, cmp_buf, 1, 1, 1, 8, 8)

            self.tik_inst.vec_dup(128, temp_ub, -1, 1, 8)
            self.tik_inst.vec_and([0, 2 ** 64 - 1], cmp16_ub, temp_ub, cmp_res, \
             1, 8, 8, 8)

            with self.tik_inst.for_range(0, roisnum) as i:
                with self.tik_inst.if_scope(cmp16_ub[i] != 0):
                    count.set_as(count + 1)
                    cont[i].set_as(tmp_value)
                with self.tik_inst.else_scope():
                    pass

            # transpose and reshape RoI coordinate
            with self.tik_inst.for_range(0, BLOCKNUM) as i:
                self.tik_inst.vec_trans(transpose_ub[0, 0], rois[16 * i, 0], 1, 1, 1)
                self.tik_inst.data_move(rois[16 * i, 0], transpose_ub[0, 0], \
                 0, 1, 256 // 16, 0, 0)

            self.tik_inst.data_move(roi_buf2_ub, rois, 0, 1, ROINUM, 0, 0)
            with self.tik_inst.for_range(0, 16) as i:
                self.tik_inst.data_move(roi_temp_ub[i, 0], roi_buf2_ub[16 * i], \
                 0, ROINUM // 16, 1, 15, 0)

            self.tik_inst.vec_dup(128, xstart, 0.0, 1, 8)

            if(ROIDIMOFFSET > 0):
                self.tik_inst.data_move(batch_index_fp16, roi_temp_ub[0, 0], \
                 0, 1, ROINUM // 16, 0, 0)
                self.tik_inst.vec_conv(64, "floor", batch_index_int32, \
                 batch_index_fp16, 2, 8, 4)
            # reorder ROI coordinate
            with self.tik_inst.for_range(0, 4) as i:
                self.tik_inst.vec_add(128, roi_buf1_ub[i, 0], \
                                   roi_temp_ub[(i + ROIDIMOFFSET), 0], \
                                   xstart, 1, 8, 8, 8)
            # fp16->fp32
            self.tik_inst.vec_conv(64, 'none', x_start_fp32_ub, roi_buf1_ub[0, 0], \
             2, 8, 4)
            self.tik_inst.vec_conv(64, 'none', y_start_fp32_ub, roi_buf1_ub[1, 0], \
             2, 8, 4)
            self.tik_inst.vec_conv(64, 'none', x_end_fp32_ub, roi_buf1_ub[2, 0], \
             2, 8, 4)
            self.tik_inst.vec_conv(64, 'none', y_end_fp32_ub, roi_buf1_ub[3, 0], \
             2, 8, 4)

            # scale RoIs, from raw picture to feature map
            spatialScale_fp32 = self.flowtable_scale_fp32[0]
            self.tik_inst.vec_muls(64, x_start_fp32_ub, x_start_fp32_ub, \
             spatialScale_fp32, 2, 8, 8)
            self.tik_inst.vec_muls(64, y_start_fp32_ub, y_start_fp32_ub, \
             spatialScale_fp32, 2, 8, 8)
            #add roi_end_mode
            if roi_end_mode > 0 :
                self.tik_inst.vec_adds(64, x_end_fp32_ub, x_end_fp32_ub, \
                 roi_end_mode, 2, 8, 8)
                self.tik_inst.vec_adds(64, y_end_fp32_ub, y_end_fp32_ub, \
                 roi_end_mode, 2, 8, 8)

            self.tik_inst.vec_muls(64, x_end_fp32_ub, x_end_fp32_ub, \
             spatialScale_fp32, 2, 8, 8)
            self.tik_inst.vec_muls(64, y_end_fp32_ub, y_end_fp32_ub, \
             spatialScale_fp32, 2, 8, 8)

            # height and width of each RoI
            self.tik_inst.vec_sub(64, y_end_fp32_ub, y_end_fp32_ub, \
             y_start_fp32_ub, 2, 8, 8, 8)  # h == yEndFp32
            self.tik_inst.vec_sub(64, x_end_fp32_ub, x_end_fp32_ub, \
            x_start_fp32_ub, 2, 8, 8, 8)  # w == xEndFp32

            # fp32 -> fp16
            self.tik_inst.vec_conv(64, 'none', roi_w_ub, x_end_fp32_ub, \
             2, 4, 8)
            self.tik_inst.vec_conv(64, 'none', roi_h_ub, y_end_fp32_ub, \
             2, 4, 8)


            sel = self.tik_inst.Tensor("uint16", (8, ),
                                   name="sel", scope=tik.scope_ubuf)
            self.tik_inst.vec_dup(8, sel, 0, 1, 8) 
            # roi_height = max(roi_end_h - roi_start_h, 1.0)
            self.tik_inst.vec_dup(128, cmp_buf, ONE, 1, 8)
            self.tik_inst.vec_cmpv_lt(sel, roi_h_ub, cmp_buf, 1, 8, 8)
            self.tik_inst.vec_sel(128, 0, roi_h_ub, sel, cmp_buf, roi_h_ub, \
             1, 8, 8, 8)

            # roi_width = max(roi_end_w - roi_start_w, 1.0)
            self.tik_inst.vec_cmpv_lt(sel, roi_w_ub, cmp_buf, 1, 8, 8)
            self.tik_inst.vec_sel(128, 0, roi_w_ub, sel, cmp_buf, roi_w_ub, \
             1, 8, 8, 8)

            # fp16->fp32
            self.tik_inst.vec_conv(64, 'none', gird_w_fp32_ub, roi_w_ub, \
             2, 8, 4)
            self.tik_inst.vec_conv(64, 'none', gird_h_fp32_ub, roi_h_ub, \
             2, 8, 4)

            scaleH_fp32 = self.flowtable_scale_fp32[1]
            scaleW_fp32 = self.flowtable_scale_fp32[2]
            scaleGH_fp32 = self.flowtable_scale_fp32[3]
            scaleGW_fp32 = self.flowtable_scale_fp32[4]

            if (sample_ratio > 0):
                self.tik_inst.vec_muls(64, roi_gridw_fp32, gird_w_fp32_ub, \
                 scaleGW_fp32, 2, 8, 8)
                self.tik_inst.vec_muls(64, roi_gridh_fp32, gird_h_fp32_ub, \
                 scaleGH_fp32, 2, 8, 8)
            else:
                scaleH = self.flowtable_scale[1]
                scaleW = self.flowtable_scale[2]
                scaleGH = self.flowtable_scale[3]
                scaleGW = self.flowtable_scale[4]

                # roi_grid_h = ceil(roiH / pooledH) , roi_grid_w = ceil(roiW / pooledW)
                self.tik_inst.vec_muls(128, roi_bin_w_ub, roi_w_ub, scaleW, \
                 1, 8, 8)
                self.tik_inst.vec_muls(128, roi_bin_h_ub, roi_h_ub, scaleH, \
                 1, 8, 8)
                self.tik_inst.vec_conv(64, 'ceil', n_grid_w_int_ub, roi_bin_w_ub, \
                 2, 8, 4)
                self.tik_inst.vec_conv(64, 'ceil', n_grid_h_int_ub, roi_bin_h_ub, \
                 2, 8, 4)
                self.tik_inst.vec_conv(64, 'none', n_grid_h_ub, n_grid_w_int_ub, \
                 2, 4, 8, 1.0)
                self.tik_inst.vec_conv(64, 'none', n_grid_w_ub, n_grid_h_int_ub, \
                 2, 4, 8, 1.0)
                self.tik_inst.vec_mul(128, binscale, n_grid_w_ub, n_grid_h_ub, \
                 1, 8, 8, 8)
                self.tik_inst.vec_rec(128, binscale, binscale, 1, 8, 8)
                self.tik_inst.vec_rec(128, n_grid_rec_w_ub, n_grid_w_ub, \
                 1, 8, 8)
                self.tik_inst.vec_rec(128, n_grid_rec_h_ub, n_grid_h_ub, \
                 1, 8, 8)
                self.tik_inst.vec_mul(128, roi_gridw, roi_bin_w_ub, \
                 n_grid_rec_w_ub, 1, 8, 8, 8)
                self.tik_inst.vec_mul(128, roi_gridh, roi_bin_h_ub, \
                 n_grid_rec_h_ub, 1, 8, 8, 8)
                # fp16->fp32
                self.tik_inst.vec_conv(64, 'none', roi_gridw_fp32, \
                 roi_gridw, 2, 8, 4)
                self.tik_inst.vec_conv(64, 'none', roi_gridh_fp32, \
                 roi_gridh, 2, 8, 4)
            return count

    def roialign_perf_gen_grid(self,
                               index_arr,
                               roi_grid_w, roi_grid_h, curr_roi,
                               lx, ly, hx, hy,
                               xlow_int, ylow_int, xhigh_int, yhigh_int,
                               x_start_fp32, y_start_fp32,
                               roi_grid_h_fp32, roi_grid_w_fp32,
                               width, height):
        tik_instance = self.tik_inst
        index_arr_fp32 = tik_instance.Tensor("float32", (ROINUM, ), \
         name="index_arr", scope=tik.scope_ubuf)
        xpos = tik_instance.Tensor("float16", (ROINUM, ), \
         name="xpos", scope=tik.scope_ubuf)
        ypos = tik_instance.Tensor("float16", (ROINUM, ), \
         name="ypos", scope=tik.scope_ubuf)
        xpos_fp32 = tik_instance.Tensor("float32", (ROINUM, ), \
         name="xpos_fp32", scope=tik.scope_ubuf)
        ypos_fp32 = tik_instance.Tensor("float32", (ROINUM, ), \
         name="ypos_fp32", scope=tik.scope_ubuf)

        xlow = tik_instance.Tensor("float16", (ROINUM, ), \
         name="xlow", scope=tik.scope_ubuf)
        ylow = tik_instance.Tensor("float16", (ROINUM, ), \
         name="ylow", scope=tik.scope_ubuf)
        xhigh = tik_instance.Tensor("float16", (ROINUM, ), \
         name="xhigh", scope=tik.scope_ubuf)
        yhigh = tik_instance.Tensor("float16", (ROINUM, ), \
         name="yhigh", scope=tik.scope_ubuf)

        xlow_fp32 = tik_instance.Tensor("float32", (ROINUM, ), \
         name="xlow_fp32", scope=tik.scope_ubuf)
        ylow_fp32 = tik_instance.Tensor("float32", (ROINUM, ), \
         name="ylow_fp32", scope=tik.scope_ubuf)
        xhigh_fp32 = tik_instance.Tensor("float32", (ROINUM, ), \
         name="xhigh_fp32", scope=tik.scope_ubuf)
        yhigh_fp32 = tik_instance.Tensor("float32", (ROINUM, ), \
         name="yhigh_fp32", scope=tik.scope_ubuf)

        lx_fp32 = tik_instance.Tensor("float32", (ROINUM, ), \
         name="lx_fp32", scope=tik.scope_ubuf)
        ly_fp32 = tik_instance.Tensor("float32", (ROINUM, ), \
         name="ly_fp32", scope=tik.scope_ubuf)
        hx_fp32 = tik_instance.Tensor("float32", (ROINUM, ), \
         name="hx_fp32", scope=tik.scope_ubuf)
        hy_fp32 = tik_instance.Tensor("float32", (ROINUM, ), \
         name="hy_fp32", scope=tik.scope_ubuf)

        cmp_buf = tik_instance.Tensor("float16", (ROINUM, ), \
         name="cmp_buf", scope=tik.scope_ubuf)
        cmp_xy = tik_instance.Tensor("float16", (ROINUM, ), \
         name="cmp_xy", scope=tik.scope_ubuf)
        cmp_const = tik_instance.Tensor("float16", (ROINUM, ), \
         name="cmp_const", scope=tik.scope_ubuf)

        lx_pos_cmp = tik_instance.Tensor("float16", (ROINUM, ), \
         name="lx_pos_cmp", scope=tik.scope_ubuf)
        ly_pos_cmp = tik_instance.Tensor("float16", (ROINUM, ), \
         name="lx_pos_cmp", scope=tik.scope_ubuf)
        hx_pos_cmp = tik_instance.Tensor("float16", (ROINUM, ), \
         name="lx_pos_cmp", scope=tik.scope_ubuf)
        hy_pos_cmp = tik_instance.Tensor("float16", (ROINUM, ), \
         name="hy_pos_cmp", scope=tik.scope_ubuf)
        lx_neg_cmp = tik_instance.Tensor("float16", (ROINUM, ), \
         name="lx_neg_cmp", scope=tik.scope_ubuf)
        ly_neg_cmp = tik_instance.Tensor("float16", (ROINUM, ), \
         name="ly_neg_cmp", scope=tik.scope_ubuf)
        hx_neg_cmp = tik_instance.Tensor("float16", (ROINUM, ), \
         name="hx_neg_cmp", scope=tik.scope_ubuf)
        hy_neg_cmp = tik_instance.Tensor("float16", (ROINUM, ), \
         name="hy_neg_cmp", scope=tik.scope_ubuf)
        xlow_pos_cmp = tik_instance.Tensor("float16", (ROINUM, ), \
         name="xlow_pos_cmp", scope=tik.scope_ubuf)
        ylow_pos_cmp = tik_instance.Tensor("float16", (ROINUM, ), \
         name="ylow_pos_cmp", scope=tik.scope_ubuf)
        xhigh_pos_cmp = tik_instance.Tensor("float16", (ROINUM, ), \
         name="xhigh_pos_cmp", scope=tik.scope_ubuf)
        yhigh_pos_cmp = tik_instance.Tensor("float16", (ROINUM, ), \
         name="yhigh_pos_cmp", scope=tik.scope_ubuf)
        xlow_neg_cmp = tik_instance.Tensor("float16", (ROINUM, ), \
         name="xlow_neg_cmp", scope=tik.scope_ubuf)
        ylow_neg_cmp = tik_instance.Tensor("float16", (ROINUM, ), \
         name="ylow_neg_cmp", scope=tik.scope_ubuf)
        xhigh_neg_cmp = tik_instance.Tensor("float16", (ROINUM, ), \
         name="xhigh_neg_cmp", scope=tik.scope_ubuf)
        yhigh_neg_cmp = tik_instance.Tensor("float16", (ROINUM, ), \
         name="xhigh_neg_cmp", scope=tik.scope_ubuf)

        delta_w = tik_instance.Scalar("float32", name="delta_w")
        delta_h = tik_instance.Scalar("float32", name="delta_h")
        delta_w.set_as(roi_grid_w_fp32[curr_roi])
        delta_h.set_as(roi_grid_h_fp32[curr_roi])

        w0 = tik_instance.Scalar("float32", name="w0")
        h0= tik_instance.Scalar("float32", name="h0")
        w0.set_as(x_start_fp32[curr_roi])
        h0.set_as(y_start_fp32[curr_roi])

        # RoI's coordinate for interpolation,
        # maxium grid number is 128,
        # as the indexAddr is set to a vector
        # which contains 128 elements.
        # convXt = CalcXtForOneSrcVectorOP(1, 1, 8, 4, 2)
        tik_instance.vec_conv(MASK_FP32, 'none', index_arr_fp32, \
         index_arr, 2, 8, 4)   # fp16 -> fp32

        # fp32Xt = CalcXtForOneSrcVectorOP(1, 1, 8, 8, 2)
        tik_instance.vec_muls(MASK_FP32, xpos_fp32, index_arr_fp32, delta_w, \
         2, 8, 8)
        tik_instance.vec_muls(MASK_FP32, ypos_fp32, index_arr_fp32, delta_h, \
         2, 8, 8)

        # fp32Xt = CalcXtForOneSrcVectorOP(1, 1, 8, 8, 2)
        tik_instance.vec_adds(MASK_FP32, xpos_fp32, xpos_fp32, w0, \
         2, 8, 8)
        tik_instance.vec_adds(MASK_FP32, ypos_fp32, ypos_fp32, h0, \
         2, 8, 8)

        # deqXt = CalcXtForOneSrcVectorOP(1, 1, 4, 8, 2)
        tik_instance.vec_conv(MASK_FP32, 'none', xpos, xpos_fp32, \
         2, 4, 8)  # fp32 -> fp16
        tik_instance.vec_conv(MASK_FP32, 'none', ypos, ypos_fp32, \
         2, 4, 8)  # fp32 -> fp16

        # convXt = CalcXtForOneSrcVectorOP(1, 1, 8, 4, 2)
        tik_instance.vec_conv(MASK_FP32, 'floor', xlow_int, xpos, \
         2, 8, 4)   # fp16  -> int32
        tik_instance.vec_conv(MASK_FP32, 'floor', ylow_int, ypos, \
         2, 8, 4)   # fp16  -> int32

        # deqXt = CalcXtForOneSrcVectorOP(1, 1, 4, 8, 2)
        tik_instance.vec_conv(MASK_FP32, 'none', xlow, xlow_int, \
         2, 4, 8, 1.0)  # int32 -> fp16
        tik_instance.vec_conv(MASK_FP32, 'none', ylow, ylow_int, \
         2, 4, 8, 1.0)  # int32 -> fp16

        # add 0.5 to avoid incorrect ceiling
        # binXt = CalcXtForOneSrcVectorOP(1, 1, 8, 8, 1)
        tik_instance.vec_adds(MASK_FP16, xhigh, xlow, POINT_FIVE, \
         1, 8, 8)
        tik_instance.vec_adds(MASK_FP16, yhigh, ylow, POINT_FIVE, \
         1, 8, 8)

        # convXt = CalcXtForOneSrcVectorOP(1, 1, 8, 4, 2)
        tik_instance.vec_conv(MASK_FP32, 'ceil', xhigh_int, xhigh, \
         2, 8, 4)   # fp16  -> int32
        tik_instance.vec_conv(MASK_FP32, 'ceil', yhigh_int, yhigh, \
         2, 8, 4)   # fp16  -> int32

        # binXt = CalcXtForOneSrcVectorOP(1, 1, 8, 8, 1)
        tik_instance.vec_adds(MASK_FP16, xhigh, xlow, ONE, 1, 8, 8)
        tik_instance.vec_adds(MASK_FP16, yhigh, ylow, ONE, 1, 8, 8)

        # lx, ly, hx, hy are the weights for interpolation
        # convXt = CalcXtForOneSrcVectorOP(1, 1, 8, 4, 2)
        tik_instance.vec_conv(MASK_FP32, 'none', xlow_fp32, xlow, \
         2, 8, 4)   # fp16 -> fp32
        tik_instance.vec_conv(MASK_FP32, 'none', ylow_fp32, ylow, \
         2, 8, 4)   # fp16 -> fp32
        tik_instance.vec_conv(MASK_FP32, 'none', xhigh_fp32, xhigh, \
         2, 8, 4)   # fp16 -> fp32
        tik_instance.vec_conv(MASK_FP32, 'none', yhigh_fp32, yhigh, \
         2, 8, 4)   # fp16 -> fp32

        # fp32TwoXt = CalcXtForTwoSrcVectorOP(1, 1, 1, 8, 8, 8, 2)
        tik_instance.vec_sub(MASK_FP32, lx_fp32, xpos_fp32, xlow_fp32, \
         2, 8, 8, 8)
        tik_instance.vec_sub(MASK_FP32, ly_fp32, ypos_fp32, ylow_fp32, \
         2, 8, 8, 8)
        tik_instance.vec_sub(MASK_FP32, hx_fp32, xhigh_fp32, xpos_fp32, \
         2, 8, 8, 8)
        tik_instance.vec_sub(MASK_FP32, hy_fp32, yhigh_fp32, ypos_fp32, \
         2, 8, 8, 8)

        # deqXt = CalcXtForOneSrcVectorOP(1, 1, 4, 8, 2)
        tik_instance.vec_conv(MASK_FP32, 'none', lx, lx_fp32, \
         2, 4, 8)  # fp32 -> fp16
        tik_instance.vec_conv(MASK_FP32, 'none', ly, ly_fp32, \
         2, 4, 8)  # fp32 -> fp16
        tik_instance.vec_conv(MASK_FP32, 'none', hx, hx_fp32, \
         2, 4, 8)  # fp32 -> fp16
        tik_instance.vec_conv(MASK_FP32, 'none', hy, hy_fp32, \
         2, 4, 8)  # fp32 -> fp16

        # temporary data for comparision
        # binXt = CalcXtForOneSrcVectorOP(1, 1, 8, 8, 1)
        tik_instance.vec_adds(MASK_FP16, lx_pos_cmp, lx, ONE, \
         1, 8, 8)
        tik_instance.vec_adds(MASK_FP16, ly_pos_cmp, ly, ONE, \
         1, 8, 8)
        tik_instance.vec_adds(MASK_FP16, hx_pos_cmp, hx, ONE, \
         1, 8, 8)
        tik_instance.vec_adds(MASK_FP16, hy_pos_cmp, hy, ONE, \
         1, 8, 8)

        tik_instance.vec_adds(MASK_FP16, lx_neg_cmp, lx, NEG_ONE, \
         1, 8, 8)
        tik_instance.vec_adds(MASK_FP16, ly_neg_cmp, ly, NEG_ONE, \
         1, 8, 8)
        tik_instance.vec_adds(MASK_FP16, hx_neg_cmp, hx, NEG_ONE, \
         1, 8, 8)
        tik_instance.vec_adds(MASK_FP16, hy_neg_cmp, hy, NEG_ONE, \
         1, 8, 8)

        tik_instance.vec_adds(MASK_FP16, xlow_pos_cmp, xlow, ONE, \
         1, 8, 8)
        tik_instance.vec_adds(MASK_FP16, ylow_pos_cmp, ylow, ONE, \
         1, 8, 8)
        tik_instance.vec_adds(MASK_FP16, xhigh_pos_cmp, xhigh, ONE, \
         1, 8, 8)
        tik_instance.vec_adds(MASK_FP16, yhigh_pos_cmp, yhigh, ONE, \
         1, 8, 8)

        tik_instance.vec_adds(MASK_FP16, xlow_neg_cmp, xlow, NEG_ONE, \
         1, 8, 8)
        tik_instance.vec_adds(MASK_FP16, ylow_neg_cmp, ylow, NEG_ONE, \
         1, 8, 8)
        tik_instance.vec_adds(MASK_FP16, xhigh_neg_cmp, xhigh, NEG_ONE, \
         1, 8, 8)
        tik_instance.vec_adds(MASK_FP16, xhigh_neg_cmp, yhigh, NEG_ONE, \
         1, 8, 8)

        sel = tik_instance.Tensor("uint16", (8, ),
                                   name="sel", scope=tik.scope_ubuf)
        tik_instance.vec_dup(8, sel, 0, 1, 8) 
        # compare lx ly with 0 and 1
        # if lx > 1:
        #   lx = lx - 1
        #   hx = hx + 1
        #   xlow = xlow + 1
        #   xhigh = xhigh + 1
        # binXt = CalcXtForOneSrcVectorOP(1, 1, 8, 8, 1)
        tik_instance.vec_dup(MASK_FP16, cmp_const, ONE, 1, 8)

        # twoXt = CalcXtForTwoSrcVectorOP(1, 1, 1, 8, 8, 8, 1)
        tik_instance.vec_cmpv_gt(sel, lx, cmp_const, 1, 8, 8)
        tik_instance.vec_sel(MASK_FP16, 0, lx, sel, lx_neg_cmp, lx, \
                              1, 8, 8, 8)
        tik_instance.vec_sel(MASK_FP16, 0, hx, sel, hx_pos_cmp, hx, \
                              1, 8, 8, 8)
        tik_instance.vec_sel(MASK_FP16, 0, xlow, sel, xlow_pos_cmp, xlow, \
                              1, 8, 8, 8)
        tik_instance.vec_sel(MASK_FP16, 0, xhigh, sel, xhigh_pos_cmp, xhigh, \
                              1, 8, 8, 8)

        # if lx < 0:
        #   lx = lx + 1
        #   hx = hx - 1
        #   xlow = xlow - 1
        #   xhigh = xhigh - 1
        # binXt = CalcXtForOneSrcVectorOP(1, 1, 8, 8, 1)
        tik_instance.vec_dup(MASK_FP16, cmp_const, ZERO, 1, 8)

        # twoXt = CalcXtForTwoSrcVectorOP(1, 1, 1, 8, 8, 8, 1)
        tik_instance.vec_cmpv_lt(sel, lx, cmp_const, 1, 8, 8)
        tik_instance.vec_sel(MASK_FP16, 0, lx, sel, lx_pos_cmp, lx, \
                              1, 8, 8, 8)
        tik_instance.vec_sel(MASK_FP16, 0, hx, sel, hx_neg_cmp, hx, \
                              1, 8, 8, 8)
        tik_instance.vec_sel(MASK_FP16, 0, xlow, sel, xlow_neg_cmp, xlow, \
                              1, 8, 8, 8)
        tik_instance.vec_sel(MASK_FP16, 0, xhigh, sel, xhigh_neg_cmp, xhigh, \
                              1, 8, 8, 8)

        # if ly > 1:
        #   ly = ly - 1
        #   hy = hy + 1
        #   ylow = ylow + 1
        #   yhigh = yhigh + 1
        # binXt = CalcXtForOneSrcVectorOP(1, 1, 8, 8, 1)
        tik_instance.vec_dup(MASK_FP16, cmp_const, ONE, 1, 8)

        # twoXt = CalcXtForTwoSrcVectorOP(1, 1, 1, 8, 8, 8, 1)
        tik_instance.vec_cmpv_gt(sel, ly, cmp_const, 1, 8, 8)
        tik_instance.vec_sel(MASK_FP16, 0, ly, sel, ly_neg_cmp, ly, \
                              1, 8, 8, 8)
        tik_instance.vec_sel(MASK_FP16, 0, hy, sel, hy_pos_cmp, hy, \
                              1, 8, 8, 8)
        tik_instance.vec_sel(MASK_FP16, 0, ylow, sel, ylow_pos_cmp, ylow, \
                              1, 8, 8, 8)
        tik_instance.vec_sel(MASK_FP16, 0, yhigh, sel, yhigh_pos_cmp, yhigh, \
                              1, 8, 8, 8)

        # if ly < 0:
        #   ly = ly + 1
        #   hy = hy - 1
        #   ylow = ylow - 1
        #   yhigh = yhigh - 1
        # binXt = CalcXtForOneSrcVectorOP(1, 1, 8, 8, 1)
        tik_instance.vec_dup(MASK_FP16, cmp_const, ZERO, 1, 8)

        # twoXt = CalcXtForTwoSrcVectorOP(1, 1, 1, 8, 8, 8, 1)
        tik_instance.vec_cmpv_lt(sel, ly, cmp_const, 1, 8, 8)
        tik_instance.vec_sel(MASK_FP16, 0, ly, sel, ly_pos_cmp, ly, \
                              1, 8, 8, 8)
        tik_instance.vec_sel(MASK_FP16, 0, hy, sel, hy_neg_cmp, hy, \
                              1, 8, 8, 8)
        tik_instance.vec_sel(MASK_FP16, 0, ylow, sel, ylow_neg_cmp, ylow, \
                              1, 8, 8, 8)
        tik_instance.vec_sel(MASK_FP16, 0, yhigh, sel, yhigh_neg_cmp, yhigh, \
                              1, 8, 8, 8)

        # update xlowInt, ylowInt, xhighInt, yhighInt
        # binXt = CalcXtForOneSrcVectorOP(1, 1, 8, 8, 1)
        tik_instance.vec_adds(MASK_FP16, cmp_buf, xlow, POINT_FIVE, 1, 8, 8)

        # convXt = CalcXtForOneSrcVectorOP(1, 1, 8, 4, 2)
        tik_instance.vec_conv(MASK_FP32, 'floor', xlow_int, cmp_buf, \
         2, 8, 4)   # fp16  -> int32
        tik_instance.vec_conv(MASK_FP32, 'ceil', xhigh_int, cmp_buf, \
         2, 8, 4)   # fp16  -> int32

        # binXt = CalcXtForOneSrcVectorOP(1, 1, 8, 8, 1)
        tik_instance.vec_adds(MASK_FP16, cmp_buf, ylow, POINT_FIVE, \
         1, 8, 8)

        # convXt = CalcXtForOneSrcVectorOP(1, 1, 8, 4, 2)
        tik_instance.vec_conv(MASK_FP32, 'floor', ylow_int, cmp_buf, \
         2, 8, 4)   # fp16  -> int32
        tik_instance.vec_conv(MASK_FP32, 'ceil', yhigh_int, cmp_buf, \
         2, 8, 4)   # fp16  -> int32

        # if x_low >= W-1:
        #   x_high = x_low = W-1
        #   hx = 1
        #   lx = 0
        # binXt = CalcXtForOneSrcVectorOP(1, 1, 8, 8, 1)
        tik_instance.vec_dup(MASK_FP16, cmp_buf, width, 1, 8)
        tik_instance.vec_adds(MASK_FP16, cmp_buf, cmp_buf, NEG_ONE, \
         1, 8, 8)
        tik_instance.vec_adds(MASK_FP16, cmp_xy, xlow, POINT_FIVE, \
         1, 8, 8)

        # binXt = CalcXtForOneSrcVectorOP(1, 1, 8, 8, 1)
        tik_instance.vec_dup(MASK_FP16, cmp_const, ONE, 1, 8)
        # twoXt = CalcXtForTwoSrcVectorOP(1, 1, 1, 8, 8, 8, 1)

        tik_instance.vec_cmpv_gt(sel, cmp_xy, cmp_buf, 1, 8, 8)
        tik_instance.vec_sel(MASK_FP16, 0, hx, sel, cmp_const, hx, \
                              1, 8, 8, 8)

        # binXt = CalcXtForOneSrcVectorOP(1, 1, 8, 8, 1)
        tik_instance.vec_dup(MASK_FP16, cmp_const, ZERO, 1, 8)
        # twoXt = CalcXtForTwoSrcVectorOP(1, 1, 1, 8, 8, 8, 1)
        tik_instance.vec_sel(MASK_FP16, 0, lx, sel, cmp_const, lx, \
         1, 8, 8, 8)

        # set x_low = W-1
        # binXt = CalcXtForOneSrcVectorOP(1, 1, 8, 8, 1)
        tik_instance.vec_adds(MASK_FP16, cmp_buf, cmp_buf, POINT_FIVE, \
         1, 8, 8)
        # twoXt = CalcXtForTwoSrcVectorOP(1, 1, 1, 8, 8, 8, 1)
        tik_instance.vec_sel(MASK_FP16, 0, cmp_xy, sel, cmp_buf, cmp_xy, \
         1, 8, 8, 8)
        # convXt = CalcXtForOneSrcVectorOP(1, 1, 8, 4, 2)
        tik_instance.vec_conv(MASK_FP32, 'floor', xlow_int, cmp_xy, \
         2, 8, 4)   # fp16  -> int32


        # set x_high = W-1
        # binXt = CalcXtForOneSrcVectorOP(1, 1, 8, 8, 1)
        tik_instance.vec_adds(MASK_FP16, cmp_buf, cmp_buf, NEG_ONE, 1, 8, 8)
        tik_instance.vec_adds(MASK_FP16, cmp_xy, xlow, POINT_FIVE, 1, 8, 8)
        # twoXt = CalcXtForTwoSrcVectorOP(1, 1, 1, 8, 8, 8, 1)

        tik_instance.vec_sel(MASK_FP16, 0, cmp_xy, sel, cmp_buf, cmp_xy, \
         1, 8, 8, 8)
        # convXt = CalcXtForOneSrcVectorOP(1, 1, 8, 4, 2)
        tik_instance.vec_conv(MASK_FP32, 'ceil', xhigh_int, cmp_xy, \
         2, 8, 4)   # fp16  -> int32


        # if y_low >= H-1:
        #   y_high = y_low = H-1
        #   hy = 1
        #   ly = 0
        # binXt = CalcXtForOneSrcVectorOP(1, 1, 8, 8, 1)
        tik_instance.vec_dup(MASK_FP16, cmp_buf, height, 1, 8)
        tik_instance.vec_adds(MASK_FP16, cmp_buf, cmp_buf, NEG_ONE, 1, 8, 8)
        tik_instance.vec_adds(MASK_FP16, cmp_xy, ylow, POINT_FIVE, 1, 8, 8)

        # binXt = CalcXtForOneSrcVectorOP(1, 1, 8, 8, 1)
        tik_instance.vec_dup(MASK_FP16, cmp_const, ONE, 1, 8)
        # twoXt = CalcXtForTwoSrcVectorOP(1, 1, 1, 8, 8, 8, 1)
        tik_instance.vec_cmpv_gt(sel, cmp_xy, cmp_buf, 1, 8, 8)
        tik_instance.vec_sel(MASK_FP16, 0, hy, sel, cmp_const, hy, \
                              1, 8, 8, 8)

        # binXt = CalcXtForOneSrcVectorOP(1, 1, 8, 8, 1)
        tik_instance.vec_dup(MASK_FP16, cmp_const, ZERO, 1, 8)
        # twoXt = CalcXtForTwoSrcVectorOP(1, 1, 1, 8, 8, 8, 1)
        tik_instance.vec_sel(MASK_FP16, 0, ly, sel, cmp_const, ly, \
         1, 8, 8, 8)

        # set y_low = H-1
        tik_instance.vec_adds(MASK_FP16, cmp_buf, cmp_buf, POINT_FIVE, \
         1, 8, 8)
        # twoXt = CalcXtForTwoSrcVectorOP(1, 1, 1, 8, 8, 8, 1)
        tik_instance.vec_sel(MASK_FP16, 0, cmp_xy, sel, cmp_buf, cmp_xy, \
         1, 8, 8, 8)
        # convXt = CalcXtForOneSrcVectorOP(1, 1, 8, 4, 2)
        tik_instance.vec_conv(MASK_FP32, 'floor', ylow_int, cmp_xy, \
         2, 8, 4)   # fp16  -> int32


        # set y_high = H-1
        tik_instance.vec_adds(MASK_FP16, cmp_buf, cmp_buf, NEG_ONE, 1, 8, 8)
        tik_instance.vec_adds(MASK_FP16, cmp_xy, ylow, POINT_FIVE, 1, 8, 8)
        # twoXt = CalcXtForTwoSrcVectorOP(1, 1, 1, 8, 8, 8, 1)

        tik_instance.vec_sel(MASK_FP16, 0, cmp_xy, sel, cmp_buf, cmp_xy, \
         1, 8, 8, 8)
        # convXt = CalcXtForOneSrcVectorOP(1, 1, 8, 4, 2)
        tik_instance.vec_conv(MASK_FP32, 'ceil', yhigh_int, cmp_xy, \
         2, 8, 4)   # fp16  -> int32

        # if x_low >= W:
        #   hx = 0
        #   lx = 0
        # binXt = CalcXtForOneSrcVectorOP(1, 1, 8, 8, 1)
        tik_instance.vec_dup(MASK_FP16, cmp_buf, width, 1, 8)
        tik_instance.vec_dup(MASK_FP16, cmp_const, ZERO, 1, 8)

        tik_instance.vec_adds(MASK_FP16, cmp_xy, xlow, POINT_FIVE, \
         1, 8, 8)
        # twoXt = CalcXtForTwoSrcVectorOP(1, 1, 1, 8, 8, 8, 1)
        tik_instance.vec_cmpv_gt(sel, cmp_xy, cmp_buf, 1, 8, 8)
        tik_instance.vec_sel(MASK_FP16, 0, hx, sel, cmp_const, hx, \
         1, 8, 8, 8)
        tik_instance.vec_sel(MASK_FP16, 0, lx, sel, cmp_const, lx, \
         1, 8, 8, 8)

        # if y_low >= H:
        #   hy = 0
        #   ly = 0
        # binXt = CalcXtForOneSrcVectorOP(1, 1, 8, 8, 1)
        tik_instance.vec_dup(MASK_FP16, cmp_buf, height, 1, 8)
        tik_instance.vec_dup(MASK_FP16, cmp_const, ZERO, 1, 8)

        tik_instance.vec_adds(MASK_FP16, cmp_xy, ylow, POINT_FIVE, \
         1, 8, 8)
        # twoXt = CalcXtForTwoSrcVectorOP(1, 1, 1, 8, 8, 8, 1)
        tik_instance.vec_cmpv_gt(sel, cmp_xy, cmp_buf, 1, 8, 8)
        tik_instance.vec_sel(MASK_FP16, 0, hy, sel, cmp_const, hy, \
         1, 8, 8, 8)
        tik_instance.vec_sel(MASK_FP16, 0, ly, sel, cmp_const, ly, \
         1, 8, 8, 8)

        # if x_low <= -1:
        #   hx = 0
        #   lx = 0
        # binXt = CalcXtForOneSrcVectorOP(1, 1, 8, 8, 1)
        tik_instance.vec_dup(MASK_FP16, cmp_buf, NEG_POINT_FIVE, 1, 8)
        tik_instance.vec_dup(MASK_FP16, cmp_const, ZERO, 1, 8)
        # twoXt = CalcXtForTwoSrcVectorOP(1, 1, 1, 8, 8, 8, 1)
        tik_instance.vec_cmpv_lt(sel, xlow, cmp_buf, 1, 8, 8)

        tik_instance.vec_sel(MASK_FP16, 0, hx, sel, cmp_const, hx, \
         1, 8, 8, 8)
        tik_instance.vec_sel(MASK_FP16, 0, lx, sel, cmp_const, lx, \
         1, 8, 8, 8)

        # if y_low <= -1:
        #   hy = 0
        #   ly = 0
        # binXt = CalcXtForOneSrcVectorOP(1, 1, 8, 8, 1)
        tik_instance.vec_dup(MASK_FP16, cmp_buf, NEG_POINT_FIVE, 1, 8)
        tik_instance.vec_dup(MASK_FP16, cmp_const, ZERO, 1, 8)
        # twoXt = CalcXtForTwoSrcVectorOP(1, 1, 1, 8, 8, 8, 1)
        tik_instance.vec_cmpv_lt(sel, ylow, cmp_buf, 1, 8, 8)
        tik_instance.vec_sel(MASK_FP16, 0, hy, sel, cmp_const, hy, \
         1, 8, 8, 8)
        tik_instance.vec_sel(MASK_FP16, 0, ly, sel, cmp_const, ly, \
         1, 8, 8, 8)


def roi_align_cce(feature_map_dict, rois_dict, output_dict, \
                  spatialScale, output_h, output_w, sample_ratio, \
                  roi_end_mode, kernel_name_val):
    obj = RoiAlign(feature_map_dict,
                   rois_dict,
                   spatialScale,
                   output_h,
                   output_w,
                   sample_ratio,
                   roi_end_mode,
                   kernel_name_val)
    return obj.roialign_compute()



def get_roi_align_perf_scale_for_zero(tik_instance, proposal, proposals_ub_x0,
                                      proposals_ub_y0, proposals_ub_x1,
                                      proposals_ub_y1, scale, pool_h, pool_w,
                                      sample_num, roi_end_mode, dtype):
    """
    get satart point, bin_size and sample number
    """
    proposal_num_128 = 128
    if dtype == "float32":
        dtype_num = 1
    else:
        dtype_num = 2

    roi_h_fp32 = tik_instance.Tensor(
        dtype, [128], name="roi_h_fp32", scope=tik.scope_ubuf)
    roi_w_fp32 = tik_instance.Tensor(
        dtype, [128], name="roi_w_fp32", scope=tik.scope_ubuf)

    roi_fp16_pos = tik_instance.Tensor(
        "float16", proposal.shape, name="roi_fp16_pos", scope=tik.scope_ubuf)
    roi_fp16_fm_index = tik_instance.Tensor(
        "float16", [128], name="roi_fp16_fm_index", scope=tik.scope_ubuf)
    roi_fp32_fm_index = tik_instance.Tensor(
        dtype, [128], name="roi_fp32_fm_index", scope=tik.scope_ubuf)
    roi_int32_fm_index = tik_instance.Tensor(
        "int32", [128], name="roi_int32_fm_index", scope=tik.scope_ubuf)
    support_vextract = \
            tbe_platform.cce_conf.api_check_support("tik.vextract", "float32")
    if support_vextract is False and dtype == "float32":
        tik_instance.vec_conv(64, "", roi_fp16_pos[0, 0], proposal[0, 0],
                           (128 * 8) // 64, 4, 8)

        tik_instance.vextract(roi_fp16_fm_index[0], roi_fp16_pos, 8, 0)
        tik_instance.vec_conv(64, "ceil", roi_int32_fm_index[0],
                           roi_fp16_fm_index[0], 2, 8, 4)
    else:
        tik_instance.vextract(roi_fp32_fm_index[0], proposal[0, 0], 8, 0)
        tik_instance.vec_conv(64, "ceil", roi_int32_fm_index[0],
                           roi_fp32_fm_index[0], 2 , 8, 8 // dtype_num)

    tik_instance.vec_muls(64 * dtype_num, proposals_ub_x0[0, 0],
                       proposals_ub_x0[0, 0],
                       scale, 128 * 2 // 128 // dtype_num, 8, 8)
    tik_instance.vec_muls(64 * dtype_num, proposals_ub_y0[0, 0],
                       proposals_ub_y0[0, 0],
                       scale, 128 * 2 // 128 // dtype_num, 8, 8)

    if roi_end_mode == 1:
        tik_instance.vec_adds(64 * dtype_num, proposals_ub_x1[0, 0],
                           proposals_ub_x1[0, 0], 1,
                           128 * 2 // 128 // dtype_num, 8, 8)
        tik_instance.vec_adds(64 * dtype_num, proposals_ub_y1[0, 0],
                           proposals_ub_y1[0, 0], 1,
                           128 * 2 // 128 // dtype_num, 8, 8)

    tik_instance.vec_muls(64 * dtype_num, proposals_ub_x1[0, 0],
                       proposals_ub_x1[0, 0],
                       scale, 128 * 2 // 128 // dtype_num, 8, 8)
    tik_instance.vec_muls(64 * dtype_num, proposals_ub_y1[0, 0],
                       proposals_ub_y1[0, 0],
                       scale, 128 * 2 // 128 // dtype_num, 8, 8)

    roi_h_1to8 = tik_instance.Tensor(
        dtype, [128, 1], name="roi_h_1to8", scope=tik.scope_ubuf)
    roi_w_1to8 = tik_instance.Tensor(
        dtype, [128, 1], name="roi_w_1to8", scope=tik.scope_ubuf)

    tik_instance.vec_sub(64 * dtype_num, roi_h_1to8, proposals_ub_y1[0, 0],
                      proposals_ub_y0[0, 0], 128 * 2 // 128 // dtype_num,
                      8, 8, 8)
    tik_instance.vec_sub(64 * dtype_num, roi_w_1to8, proposals_ub_x1[0, 0],
                      proposals_ub_x0[0, 0], 128 * 2 // 128 // dtype_num,
                      8, 8, 8)


    const_mode = tik_instance.Tensor(
        dtype, [128, 1], name="const_mode", scope=tik.scope_ubuf)
    tik_instance.vec_dup(64 * dtype_num, const_mode, 1-roi_end_mode,
                            2 // dtype_num, 8)

    # compare roi_width adn roi_height to 1-mode (1 or 0)_

    tik_instance.vec_max(64 * dtype_num, roi_w_1to8, roi_w_1to8, const_mode,
                      128 * 2 // 128 // dtype_num, 8, 8, 0)
    tik_instance.vec_max(64 * dtype_num, roi_h_1to8, roi_h_1to8, const_mode,
                      128 * 2 // 128 // dtype_num, 8, 8, 0)

    with tik_instance.for_range(0, roi_w_fp32.shape[0]) as i:
        roi_w_fp32[i].set_as(roi_w_1to8[i, 0])
        roi_h_fp32[i].set_as(roi_h_1to8[i, 0])

    # Declare roi_bin_size tik_instance.Tensor
    roi_bin_h_fp32_value = tik_instance.Tensor(
        dtype, [128], name="roi_bin_h_fp32_value", scope=tik.scope_ubuf)
    roi_bin_w_fp32_value = tik_instance.Tensor(
        dtype, [128], name="roi_bin_w_fp32_value", scope=tik.scope_ubuf)

    grid_w_fp32 = tik_instance.Tensor(
        dtype, [proposal_num_128], name="grid_w_fp32", scope=tik.scope_ubuf)
    grid_h_fp32 = tik_instance.Tensor(
        dtype, [proposal_num_128], name="grid_h_fp32", scope=tik.scope_ubuf)

    grid_w_fp16 = tik_instance.Tensor(
        "float16", [proposal_num_128], name="grid_w_fp16", scope=tik.scope_ubuf)
    grid_h_fp16 = tik_instance.Tensor(
        "float16", [proposal_num_128], name="grid_h_fp16", scope=tik.scope_ubuf)

    grid_w_int32 = tik_instance.Tensor(
        "int32", [proposal_num_128], name="grid_w_int32", scope=tik.scope_ubuf)
    grid_h_int32 = tik_instance.Tensor(
        "int32", [proposal_num_128], name="grid_h_int32", scope=tik.scope_ubuf)

    # bin size
    tik_instance.vec_muls(64 * dtype_num, roi_bin_h_fp32_value[:],
                       roi_h_fp32[:], 1.0 / pool_h,
                       proposal_num_128 * 2 // dtype_num // 128, 8, 8)
    tik_instance.vec_muls(64 * dtype_num, roi_bin_w_fp32_value[:],
                       roi_w_fp32[:], 1.0 / pool_w,
                       proposal_num_128 * 2 // dtype_num // 128, 8, 8)
    suppot_vconv = tbe_platform.cce_conf.intrinsic_check_support( \
        "Intrinsic_vconv", "f322s32c")
    if sample_num <= 0:
        if suppot_vconv is False and dtype == "float32":
            roi_bin_w_fp16_value = tik_instance.Tensor(
                "float16", [128],
                name="roi_bin_w_fp16_value",
                scope=tik.scope_ubuf)
            roi_bin_h_fp16_value = tik_instance.Tensor(
                "float16", [128],
                name="roi_bin_h_fp16_value",
                scope=tik.scope_ubuf)
            tik_instance.vec_conv(64 * dtype_num, "", roi_bin_w_fp16_value,
                               roi_bin_w_fp32_value, 2, 4, 8)
            tik_instance.vec_conv(64 * dtype_num, "", roi_bin_h_fp16_value,
                               roi_bin_h_fp32_value, 2, 4, 8)

            tik_instance.vec_conv(64, "ceiling", grid_w_int32,
                               roi_bin_w_fp16_value, 2, 8, 4)
            tik_instance.vec_conv(64, "ceiling", grid_h_int32,
                               roi_bin_h_fp16_value, 2, 8, 4)
        else:
            tik_instance.vec_conv(64, "ceiling", grid_w_int32,
                               roi_bin_w_fp32_value, 2, 8,
                               8 // dtype_num)
            tik_instance.vec_conv(64, "ceiling", grid_h_int32,
                               roi_bin_h_fp32_value, 2, 8,
                               8 // dtype_num)

        if suppot_vconv is False and dtype == "float32":
            tik_instance.vec_conv(64 * dtype_num, "", grid_w_fp16, grid_w_int32,
                               2 // dtype_num, 4, 8, 1.0)
            tik_instance.vec_conv(64 * dtype_num, "", grid_h_fp16, grid_h_int32,
                               2 // dtype_num, 4, 8, 1.0)
            tik_instance.vec_conv(64 * dtype_num, "", grid_w_fp32, grid_w_fp16,
                               2 // dtype_num, 8, 4)
            tik_instance.vec_conv(64 * dtype_num, "", grid_h_fp32, grid_h_fp16,
                               2 // dtype_num, 8, 4)
        else:
            if dtype == "float32":
                tik_instance.vec_conv(64, "", grid_w_fp32, grid_w_int32, 2,
                                      8 // dtype_num, 8)
                tik_instance.vec_conv(64, "", grid_h_fp32, grid_h_int32, 2,
                                      8 // dtype_num, 8)
            else:
                tik_instance.vec_conv(64, "", grid_w_fp32, grid_w_int32, 2,
                                   8 // dtype_num, 8, 1.0)
                tik_instance.vec_conv(64, "", grid_h_fp32, grid_h_int32, 2,
                                   8 // dtype_num, 8, 1.0)

    else:
        tik_instance.vec_dup(64, grid_w_int32, sample_num, 2, 8)
        tik_instance.vec_dup(64, grid_h_int32, sample_num, 2, 8)
        tik_instance.vec_dup(64 * dtype_num, grid_w_fp32,
                                sample_num, 2 // dtype_num, 8)
        tik_instance.vec_dup(64 * dtype_num, grid_h_fp32,
                                sample_num, 2 // dtype_num, 8)


    return tik_instance, roi_bin_h_fp32_value, \
           roi_bin_w_fp32_value, \
           proposals_ub_x0, proposals_ub_y0, \
           grid_w_int32, grid_h_int32, grid_w_fp32, \
           grid_h_fp32, roi_int32_fm_index


def newton(tik_instance, mask, dst_ub, src1, src2, repeat, dtype):
    """
    for div and usr newton when in mini
    """
    rec_2 = tik_instance.Tensor(dtype, src2.shape, name="rec_1",
                                scope=tik.scope_ubuf)
    reciprocal(tik_instance, mask, rec_2, src2, repeat, dtype)
    tik_instance.vec_mul(mask, dst_ub, rec_2, src1, repeat, 8, 8, 8)

def reciprocal(tik_instance, mask, dest_ub, src1, repeat, dtype):
    """
    get reciprocal when in mini
    """
    rec_1 = tik_instance.Tensor(dtype, src1.shape, name="rec_1",
                                scope=tik.scope_ubuf)
    rec_2 = tik_instance.Tensor(dtype, src1.shape, name="rec_2",
                                scope=tik.scope_ubuf)
    tik_instance.vec_rec(mask, rec_1, src1, repeat, 8, 8)
    tik_instance.vec_mul(mask, rec_2, rec_1, src1, repeat, 8, 8, 8)
    tik_instance.vec_muls(mask, rec_2, rec_2, -1, repeat, 8, 8)
    tik_instance.vec_adds(mask, rec_2, rec_2, 2, repeat, 8, 8)
    tik_instance.vec_mul(mask, rec_2, rec_2, rec_1, repeat, 8, 8, 8)
    tik_instance.vec_mul(mask, rec_1, rec_2, src1, repeat, 8, 8, 8)
    tik_instance.vec_muls(mask, rec_1, rec_1, -1, repeat, 8, 8)
    tik_instance.vec_adds(mask, rec_1, rec_1, 2, repeat, 8, 8)
    tik_instance.vec_mul(mask, dest_ub, rec_1, rec_2, repeat, 8, 8, 8)


# pylint: disable=unused-argument
def get_grid_weight_per_roi(tik_instance, roi_bin_h_fp32_value,
                            proposals_ub_y0, grid_h_fp32, pool_n, grid_n, fm_h,
                            curr_roi, dtype, verify, w_h):
    """
    get grid size and coordinate in feature
    """
    if dtype == "float32":
        dtype_num = 1
    else:
        dtype_num = 2
    roi_bin_ph_gh_hy_int32 = tik_instance.Tensor(
        "int32", [1], name="roi_bin_ph_gh_hy_int32", scope=tik.scope_ubuf)
    roi_bin_ph_gh_ly_int32 = tik_instance.Tensor(
        "int32", [1], name="roi_bin_ph_gh_ly_int32", scope=tik.scope_ubuf)
    roi_bin_ph_gh_pos_fp32 = tik_instance.Tensor(
        dtype, [1], name="roi_bin_ph_gh_pos_fp32", scope=tik.scope_ubuf)
    roi_bin_ph_gh_ly_weight = tik_instance.Tensor(
        dtype, [1], name="roi_bin_ph_gh_ly_weight", scope=tik.scope_ubuf)
    roi_bin_ph_gh_hy_weight = tik_instance.Tensor(
        dtype, [1], name="roi_bin_ph_gh_hy_weight", scope=tik.scope_ubuf)
    tmp_float16 = tik_instance.Tensor(
        "float16", [1], name="tmp_float16", scope=tik.scope_ubuf)

    const_one = tik_instance.Tensor(
        "int32", [1], name="const_one", scope=tik.scope_ubuf)
    tik_instance.vec_dup(1, const_one, 1, 1, 0)

    tmp_bin_h_fp32 = tik_instance.Tensor(
        dtype, [1], name="tmp_bin_h_fp32", scope=tik.scope_ubuf)
    tmp_bin_h_fp32[0].set_as(roi_bin_h_fp32_value[curr_roi])
    vconv_suppot = tbe_platform.cce_conf.intrinsic_check_support( \
        "Intrinsic_vconv", "f322s32c")
    if vconv_suppot is False or dtype == "float16":
        tmp_int32 = tik_instance.Tensor(
            "int32", [1], name="tmp_int32", scope=tik.scope_ubuf)
        tmp_float32 = tik_instance.Tensor(
            dtype, [1], name="tmp_float32", scope=tik.scope_ubuf)
        tmp_int32[0].set_as(pool_n)
        tik_instance.vec_conv(1, "", tmp_float16, tmp_int32, 1, 4, 8, 1.0)
        if dtype == "float32":
            tik_instance.vec_conv(1, "", tmp_float32, tmp_float16, 1,
                               8, 4)
            tik_instance.vec_mul(1, roi_bin_ph_gh_pos_fp32, tmp_bin_h_fp32,
                              tmp_float32,
                              1, 8, 8, 8)
        else:
            tik_instance.vec_mul(1, roi_bin_ph_gh_pos_fp32, tmp_bin_h_fp32,
                              tmp_float16,
                              1, 8, 8, 8)
    else:
        tik_instance.vec_muls(1, roi_bin_ph_gh_pos_fp32, tmp_bin_h_fp32, pool_n,
                           1, 8, 8)

    const_value_fp32 = tik_instance.Tensor(
        dtype, [1], name="const_value_fp32", scope=tik.scope_ubuf)

    const_value_fp32[0].set_as(proposals_ub_y0[curr_roi, 0])

    # get every bin_start_pose
    tik_instance.vec_add(1, roi_bin_ph_gh_pos_fp32[0], roi_bin_ph_gh_pos_fp32[0],
                      const_value_fp32, 1, 8, 8, 8)
    const_value_fp32[0].set_as(grid_h_fp32[curr_roi])
    support_div = tbe_platform.cce_conf.api_check_support( \
        "tik.vdiv", "float32")
    if support_div is False:
        newton(tik_instance, 1, tmp_bin_h_fp32, tmp_bin_h_fp32,
               const_value_fp32, 1, dtype)

    else:
        tik_instance.vdiv(1, tmp_bin_h_fp32, tmp_bin_h_fp32, const_value_fp32,
                          1, 1, 1, 1, 8, 8, 8)

    # i * bin_size_h /sample_num_h;
    if vconv_suppot is False or dtype == "float16":
        tmp_int32 = tik_instance.Tensor(
            "int32", [1], name="tmp_int32", scope=tik.scope_ubuf)
        tmp_int32[0].set_as(grid_n)
        if dtype == "float32":
            tik_instance.vec_conv(1, "", tmp_float16, tmp_int32,
                               1, 4, 8, 1.0)
            tik_instance.vec_conv(1, "", const_value_fp32, tmp_float16,
                               1, 8, 4)
        else:
            tik_instance.vec_conv(1, "", const_value_fp32, tmp_int32, 1,
                                  4, 8, 1.0)
    else:
        tik_instance.vec_dup(1, const_value_fp32, grid_n, 1, 0)

    tik_instance.vec_adds(1, const_value_fp32, const_value_fp32, 0.5, 1, 8,
                       8)
    tik_instance.vec_mul(1, tmp_bin_h_fp32, tmp_bin_h_fp32, const_value_fp32, 1,
                       8, 8, 8)
    tik_instance.vec_add(1, roi_bin_ph_gh_pos_fp32[0], tmp_bin_h_fp32,
                      roi_bin_ph_gh_pos_fp32[0], 1, 8, 8, 8)

    roi_y_floor = tik_instance.Tensor(
        "int32", [1], name="roi_y_floor", scope=tik.scope_ubuf)
    vconvf_suppot = tbe_platform.cce_conf.intrinsic_check_support( \
        "Intrinsic_vconv", "f322s32f")
    if vconvf_suppot is False and dtype == "float32":
        tik_instance.vec_conv(1, "", tmp_float16, roi_bin_ph_gh_pos_fp32, 1,
                           4, 8)
        tik_instance.vec_conv(1, "floor", roi_y_floor[0], tmp_float16, 1, 8,
                           4)
    else:
        tik_instance.vec_conv(1, "floor", roi_y_floor[0], roi_bin_ph_gh_pos_fp32,
                           1, 8, 8 // dtype_num)

    tmp_verify = tik_instance.Scalar(dtype="int32")
    tmp_verify.set_as(roi_y_floor[0])
    with tik_instance.if_scope(tmp_verify < -1):
        verify.set_as(1)
    with tik_instance.if_scope(tmp_verify >= fm_h):
        verify.set_as(1)

    # if (y <= 0) y = 0
    tik_instance.vec_dup(1, const_value_fp32, 0, 1, 0)
    tik_instance.vec_max(1, roi_bin_ph_gh_pos_fp32, roi_bin_ph_gh_pos_fp32,
                      const_value_fp32, 1, 8, 8, 0)

    # int y_low = (int)y
    if vconvf_suppot is False and dtype == "float32":
        tik_instance.vec_conv(1, "", tmp_float16, roi_bin_ph_gh_pos_fp32, 1,
                           4, 8)
        tik_instance.vec_conv(1, "floor", roi_bin_ph_gh_ly_int32[0], tmp_float16,
                           1, 8, 4)
    else:
        tik_instance.vec_conv(1, "floor", roi_bin_ph_gh_ly_int32[0],
                           roi_bin_ph_gh_pos_fp32, 1, 8, 4)

    # y_high = y_low + 1
    tik_instance.vec_add(1, roi_bin_ph_gh_hy_int32[0], roi_bin_ph_gh_ly_int32[0],
                      const_one, 1, 8, 8, 8)

    tik_instance.vec_dup(1, const_one, fm_h - 1, 1, 0)
    tik_instance.vec_dup(1, const_value_fp32, fm_h - 1, 1, 0)
    tik_instance.vec_min(1, roi_bin_ph_gh_ly_int32, roi_bin_ph_gh_ly_int32,
                      const_one, 1, 8, 8, 0)
    tik_instance.vec_min(1, roi_bin_ph_gh_hy_int32, roi_bin_ph_gh_hy_int32,
                      const_one, 1, 8, 8, 0)
    tik_instance.vec_min(1, roi_bin_ph_gh_pos_fp32, roi_bin_ph_gh_pos_fp32,
                      const_value_fp32, 1, 8, 8, 0)

    if vconvf_suppot is False and dtype == "float32":
        tik_instance.vec_conv(1, "", tmp_float16, roi_bin_ph_gh_ly_int32, 1,
                           4, 8, 1.0)
        # low level
        tik_instance.vec_conv(1, "", tmp_bin_h_fp32[0], tmp_float16[0], 1,
                           8, 4)
    else:
        # low level
        if dtype == "float32":
            tik_instance.vec_conv(1, "", tmp_bin_h_fp32[0],
                               roi_bin_ph_gh_ly_int32[0],
                               1, 8 // dtype_num, 8)
        else:
            tik_instance.vec_conv(1, "", tmp_bin_h_fp32[0],
                               roi_bin_ph_gh_ly_int32[0],
                               1, 8 // dtype_num, 8, 1.0)

    # ly = y - y_low
    tik_instance.vec_sub(1, roi_bin_ph_gh_ly_weight[0], roi_bin_ph_gh_pos_fp32,
                      tmp_bin_h_fp32, 1, 8, 0, 8)
    # hy = 1. - ly
    tik_instance.vec_dup(1, const_value_fp32, 1, 1, 0)
    tik_instance.vec_sub(1, roi_bin_ph_gh_hy_weight[0], const_value_fp32[0],
                      roi_bin_ph_gh_ly_weight[0], 1, 8, 0, 8)

    low_y = tik_instance.Scalar(
        dtype=dtype, init_value=roi_bin_ph_gh_ly_weight[0])
    high_y = tik_instance.Scalar(
        dtype=dtype, init_value=roi_bin_ph_gh_hy_weight[0])
    return roi_bin_ph_gh_ly_int32, low_y, roi_bin_ph_gh_hy_int32, high_y, verify


def compute_w1234(tik_instance, h_y, l_y, h_x, l_x, w1_lt, w2_rt, w3_lb, w4_rb,
                  fm_grid, c_valid, n_bust):
    """
    get weight 1, 2, 3 and 4
    """
    if n_bust == 2:
        dtype = "float32"
    else:
        dtype = "float16"

    vconvf_suppot = tbe_platform.cce_conf.intrinsic_check_support( \
        "Intrinsic_vconv", "f322s32f")

    if dtype == "float32" and vconvf_suppot is True:
        hy_tensor = tik_instance.Scalar(dtype=dtype, init_value=h_y)
        ly_tensor = tik_instance.Scalar(dtype=dtype, init_value=l_y)
        hx_tensor = tik_instance.Scalar(dtype=dtype, init_value=h_x)
        lx_tensor = tik_instance.Scalar(dtype=dtype, init_value=l_x)

        w_1 = tik_instance.Scalar(dtype=dtype, init_value=hy_tensor*hx_tensor)
        w_2 = tik_instance.Scalar(dtype=dtype, init_value=hy_tensor*lx_tensor)
        w_3 = tik_instance.Scalar(dtype=dtype, init_value=hx_tensor*ly_tensor)
        w_4 = tik_instance.Scalar(dtype=dtype, init_value=ly_tensor*lx_tensor)
    else:
        hy_tensor = tik_instance.Tensor( \
            dtype, [1], name="hy_tensor", scope=tik.scope_ubuf)
        hy_tensor[0].set_as(h_y)
        ly_tensor = tik_instance.Tensor( \
            dtype, [1], name="ly_tensor", scope=tik.scope_ubuf)
        ly_tensor[0].set_as(l_y)
        hx_tensor = tik_instance.Tensor( \
            dtype, [1], name="hx_tensor", scope=tik.scope_ubuf)
        hx_tensor[0].set_as(h_x)
        lx_tensor = tik_instance.Tensor( \
            dtype, [1], name="lx_tensor", scope=tik.scope_ubuf)
        lx_tensor[0].set_as(l_x)

        w1_tensor = tik_instance.Tensor( \
            dtype, [1], name="w1_tensor", scope=tik.scope_ubuf)
        w2_tensor = tik_instance.Tensor( \
            dtype, [1], name="w2_tensor", scope=tik.scope_ubuf)
        w3_tensor = tik_instance.Tensor( \
            dtype, [1], name="w3_tensor", scope=tik.scope_ubuf)
        w4_tensor = tik_instance.Tensor( \
            dtype, [1], name="w4_tensor", scope=tik.scope_ubuf)

        tik_instance.vec_mul(1, w1_tensor, hy_tensor, hx_tensor, 1, 8, 8, \
                          8)
        tik_instance.vec_mul(1, w2_tensor, hy_tensor, lx_tensor, 1, 8, 8, \
                          8)
        tik_instance.vec_mul(1, w3_tensor, hx_tensor, ly_tensor, 1, 8, 8, \
                          8)
        tik_instance.vec_mul(1, w4_tensor, ly_tensor, lx_tensor, 1, 8, 8, \
                          8)
        w_1 = tik_instance.Scalar(dtype=dtype)
        w_1.set_as(w1_tensor[0])
        w_2 = tik_instance.Scalar(dtype=dtype)
        w_2.set_as(w2_tensor[0])
        w_3 = tik_instance.Scalar(dtype=dtype)
        w_3.set_as(w3_tensor[0])
        w_4 = tik_instance.Scalar(dtype=dtype)
        w_4.set_as(w4_tensor[0])


    tik_instance.vec_muls(16, w1_lt[0, 0], fm_grid[0, 0, 0, 0], w_1, \
                       c_valid, n_bust, 4 * n_bust)
    tik_instance.vec_muls(16, w2_rt[0, 0], fm_grid[0, 0, 1, 0], w_2, \
                       c_valid, n_bust, 4 * n_bust)
    tik_instance.vec_muls(16, w3_lb[0, 0], fm_grid[0, 1, 0, 0], w_3, \
                       c_valid, n_bust, 4 * n_bust)
    tik_instance.vec_muls(16, w4_rb[0, 0], fm_grid[0, 1, 1, 0], w_4, \
                       c_valid, n_bust, 4 * n_bust)

    tik_instance.vec_add(16, w1_lt[0, 0], w1_lt[0, 0], w2_rt[0, 0], \
                      c_valid, n_bust, n_bust, n_bust)
    tik_instance.vec_add(16, w1_lt[0, 0], w1_lt[0, 0], w3_lb[0, 0], \
                      c_valid, n_bust, n_bust, n_bust)
    tik_instance.vec_add(16, w1_lt[0, 0], w1_lt[0, 0], w4_rb[0, 0], \
                      c_valid, n_bust, n_bust, n_bust)


def get_average(tik_instance, grid_curr_h_f32, grid_curr_w_f32, val, c_valid,
                p_w, n_bust):
    """
    get average
    """
    if n_bust == 2:
        dtype = "float32"
    else:
        dtype = "float16"
    vconvs32_suppot = tbe_platform.cce_conf.intrinsic_check_support( \
        "Intrinsic_vconv", "f322s32f")
    if vconvs32_suppot is False or dtype == "float16":
        grid_h_num_f_tensor = tik_instance.Tensor( \
            dtype, [1], name="grid_h_num_f_tensor", scope=tik.scope_ubuf)
        grid_h_num_f_tensor[0].set_as(grid_curr_h_f32)
        grid_w_num_f_tensor = tik_instance.Tensor( \
            dtype, [1], name="grid_w_num_f_tensor", scope=tik.scope_ubuf)
        grid_w_num_f_tensor[0].set_as(grid_curr_w_f32)
        h_w_tensor = tik_instance.Tensor( \
            dtype, [1], name="h_w_tensor", scope=tik.scope_ubuf)
        tik_instance.vec_mul(1, h_w_tensor, grid_w_num_f_tensor, \
                          grid_h_num_f_tensor, 1, 8, 8, 8)
        h_w = tik_instance.Scalar(dtype=dtype)

        reciprocal(tik_instance, 1, h_w_tensor, h_w_tensor, 1, dtype)
        h_w.set_as(h_w_tensor[0])

        tik_instance.vec_muls(16, val, val, h_w, c_valid * p_w, \
                           n_bust, n_bust)

    else:
        wh_tmp = tik_instance.Scalar(dtype=dtype)
        wh_tmp.set_as(grid_curr_h_f32 * grid_curr_w_f32)

        tik_instance.vec_muls(16, val, val, 1.0 / wh_tmp, c_valid * p_w, \
                              n_bust, n_bust)


def compute_roi_with_single_point(tik_instance, feature_shape, dtype,
                                  fm_to_l1, fm_c1,
                                  block_i, index, roi_128_number, block_num,
                                  w_number, pool_h, pool_w, n_bust, grid_curr_h,
                                  roi_bin_h_fp32_value, ret, grid_curr_w_f32,
                                  grid_w_fp32, grid_curr_h_f32,
                                  proposals_ub_y0, grid_h_fp32, roi_bin_w_fp32_value,
                                  proposals_ub_x0, feature_map, curr_roi, grid_curr_w,
                                  cache_fm, cache_index, fm_to_ub,
                                  w_number_ub, feature_map_ub):

    """
    compute roi with single point
    """
    fm_h = feature_shape[2]
    fm_w = feature_shape[3]
    if fm_to_ub >= 1:
        with tik_instance.if_scope(cache_index != index):
            tik_instance.data_move( \
                feature_map_ub, feature_map[index, 0, 0, 0, 0], \
                0, 1, fm_c1 * fm_h * fm_w * n_bust, \
                0, 0)
            cache_index.set_as(index)
    elif fm_to_l1 >= 1:
        with tik_instance.if_scope(cache_index != index):
            tik_instance.data_move( \
                cache_fm, feature_map[index, 0, 0, 0, 0], \
                0, 1, fm_c1 * fm_h * fm_w * n_bust, \
                0, 0)
            cache_index.set_as(index)
    elif w_number_ub >= 2:
        cache_ub = tik_instance.Tensor( \
            dtype, [2, fm_c1, fm_w, 16], \
            name="cache_ub", \
            scope=tik.scope_ubuf)
        cache_table = tik_instance.Tensor( \
            "int32", [2, 2], name="cache_table", scope=tik.scope_ubuf)
        with tik_instance.for_range(0, 2) as j:
            one = tik_instance.Scalar(dtype="int32", init_value=-1)
            cache_table[j, 0].set_as(one)
    elif w_number >= 2:
        cache_l1, cache_table = init_l1(tik_instance, dtype, 2,
                                        fm_c1, fm_w, 16)
        with tik_instance.for_range(0, 2) as j:
            one = tik_instance.Scalar(dtype="int32", init_value=-1)
            cache_table[j, 0].set_as(one)
    with tik_instance.for_range(0, pool_h) as p_h:
        with tik_instance.for_range(0, pool_w, thread_num=2) as p_w:
            # less 255
            c_block = 110
            c_number = (fm_c1 + (c_block - 1)) // c_block
            val = tik_instance.Tensor(
                dtype, [c_block, 16], name="val", scope=tik.scope_ubuf)
            for current_cb in range(c_number):
                c_valid = c_block
                if current_cb == c_number - 1:
                    c_valid = fm_c1 - c_block * current_cb

                tik_instance.vec_dup(16, val, 0.0, c_block,
                                     n_bust)
                with tik_instance.if_scope(c_valid != 0):
                    w1_lt = tik_instance.Tensor(
                        dtype, [c_block, 16], name="w1_lt",
                        scope=tik.scope_ubuf)
                    w2_rt = tik_instance.Tensor(
                        dtype, [c_block, 16], name="w2_rt",
                        scope=tik.scope_ubuf)
                    w3_lb = tik_instance.Tensor(
                        dtype, [c_block, 16], name="w3_lb",
                        scope=tik.scope_ubuf)
                    w4_rb = tik_instance.Tensor(
                        dtype, [c_block, 16], name="w4_rb",
                        scope=tik.scope_ubuf)
                    with tik_instance.for_range(0, grid_curr_h) as g_h:
                        verify = tik_instance.Scalar(
                            dtype="int32", init_value=0)

                        roi_bin_ph_gh_ly_int32, l_y, \
                        roi_bin_ph_gh_hy_int32, h_y, verify = \
                            get_grid_weight_per_roi(
                                tik_instance, roi_bin_h_fp32_value,
                                proposals_ub_y0,
                                grid_h_fp32, p_h, g_h, fm_h,
                                curr_roi, dtype, verify, 1)

                        with tik_instance.if_scope(verify == 0):
                            y_low = tik_instance.Scalar(
                                dtype="int32",
                                init_value=roi_bin_ph_gh_ly_int32[0])
                            y_high = tik_instance.Scalar(
                                dtype="int32", init_value=roi_bin_ph_gh_hy_int32[0])
                            if w_number_ub >= 2:
                                load_a_w_to_l1(tik_instance, cache_table, cache_ub, feature_map,
                                               index, y_low, n_bust, 0)
                                load_a_w_to_l1(tik_instance, cache_table, cache_ub, feature_map,
                                               index, y_high, n_bust, 1)
                            elif w_number >= 2:
                                load_a_w_to_l1(
                                    tik_instance, cache_table, cache_l1,
                                    feature_map,
                                    index, y_low, n_bust, 0)
                                load_a_w_to_l1(
                                    tik_instance, cache_table, cache_l1,
                                    feature_map,
                                    index, y_high, n_bust, 1)
                            with tik_instance.for_range(0, grid_curr_w) as g_w:
                                roi_bin_pw_gw_lx_int32, l_x, \
                                roi_bin_pw_gw_hx_int32, h_x, verify = \
                                    get_grid_weight_per_roi(
                                        tik_instance, roi_bin_w_fp32_value,
                                        proposals_ub_x0,
                                        grid_w_fp32, p_w, g_w,
                                        fm_w, curr_roi, dtype,
                                        verify, 0)

                                with tik_instance.if_scope(verify == 0):
                                    x_low = tik_instance.Scalar(
                                        dtype="int32",
                                        init_value=roi_bin_pw_gw_lx_int32[0])
                                    x_high = tik_instance.Scalar(
                                        dtype="int32",
                                        init_value=roi_bin_pw_gw_hx_int32[0])
                                    fm_grid = tik_instance.Tensor(
                                        dtype, (c_block, 2, 2, 16),
                                        name="fm_grid", scope=tik.scope_ubuf)
                                    if fm_to_ub >= 1:
                                        load_feature_map_to_ub(
                                            tik_instance, fm_grid, feature_shape,
                                            c_block, c_valid, feature_map_ub, index, 0,
                                            y_low, x_low, x_high, y_high, n_bust, 1)
                                    elif fm_to_l1 >= 1:
                                        load_feature_map_to_ub(
                                            tik_instance, fm_grid, feature_shape,
                                            c_block, c_valid, cache_fm, index, 0,
                                            y_low, x_low, x_high, y_high, n_bust, 1)
                                    elif w_number_ub >= 2:
                                        load_from_l1_cache(
                                            tik_instance, feature_map,
                                            fm_grid, cache_ub, 0,
                                            current_cb, c_block, x_low,
                                            x_high, c_valid, n_bust)
                                        load_from_l1_cache(
                                            tik_instance, feature_map, fm_grid,
                                            cache_ub, 1, current_cb, c_block,
                                            x_low, x_high, c_valid, n_bust)
                                    elif w_number >= 2:
                                        load_from_l1_cache(
                                            tik_instance, feature_map,
                                            fm_grid, cache_l1, 0,
                                            current_cb, c_block, x_low,
                                            x_high, c_valid, n_bust)
                                        load_from_l1_cache(
                                            tik_instance, feature_map, fm_grid,
                                            cache_l1, 1, current_cb, c_block,
                                            x_low, x_high, c_valid, n_bust)
                                    else:
                                        load_feature_map_to_ub(
                                            tik_instance, fm_grid,
                                            feature_shape,
                                            c_block,
                                            c_valid, feature_map, index,
                                            current_cb, y_low, x_low, x_high,
                                            y_high, n_bust, 0)

                                    compute_w1234(
                                        tik_instance, h_y, l_y, h_x, l_x,
                                        w1_lt,
                                        w2_rt, w3_lb, w4_rb, fm_grid, c_valid,
                                        n_bust)

                                    tik_instance.vec_add(
                                        16, val, val, w1_lt, c_valid,
                                        n_bust, n_bust, n_bust)

                    get_average(tik_instance, grid_curr_h_f32,
                                grid_curr_w_f32, val, c_valid, 1, n_bust)

                    with tik_instance.if_scope((pool_h * pool_w - 1) * n_bust
                                               <= 65535):
                        tik_instance.data_move(
                            ret[block_i * block_num +
                                128 * roi_128_number + curr_roi,
                                current_cb * c_block, p_h, p_w, 0],
                            val, 0, c_valid, n_bust, 0,
                            (pool_h * pool_w - 1) * n_bust)
                    with tik_instance.else_scope():
                        with tik_instance.for_range(0, c_valid) as c_iter_i:
                            tik_instance.data_move(
                                ret[block_i * block_num +
                                    128 * roi_128_number + curr_roi,
                                    current_cb * c_block +
                                    c_iter_i, p_h, p_w, 0],
                                val[c_iter_i, 0], 0, 1, n_bust, 0, 0)

def bilinear_interpolate(tik_instance, x_lo_w, x_hi_w, y_lo_w, y_hi_w, x_lo,
                         x_hi, y_lo, y_hi, raw_x, raw_y, sample_num_w,
                         sample_num_h, grid_h_num_f,
                         grid_w_num_f, fm_h, fm_w,
                         fm_c1, dtype, n_bust, pw_s,
                         pw_int, ph_int, block_i, block_num,
                         index, curr_roi, feature_map, ret, roi_128_number,
                         w_number, fm_to_l1, cache_fm, cache_index, fm_to_ub,
                         w_number_ub, feature_map_ub):
    """
    bilinear_interpolate
    """
    if dtype == "float32":
        dtype_num = 1
    else:
        dtype_num = 2

    if fm_to_ub >= 1:
        with tik_instance.if_scope(cache_index != index):
            tik_instance.data_move( \
                feature_map_ub, feature_map[index, 0, 0, 0, 0], \
                0, 1, fm_c1 * fm_h * fm_w * n_bust, \
                0, 0)
            cache_index.set_as(index)
    elif fm_to_l1 >= 1:
        with tik_instance.if_scope(cache_index != index):
            tik_instance.data_move( \
                cache_fm, feature_map[index, 0, 0, 0, 0], \
                0, 1, fm_c1 * fm_h * fm_w * n_bust, \
                0, 0)
            cache_index.set_as(index)
    elif w_number_ub >= 2:
        cache_ub = tik_instance.Tensor( \
            dtype, [2, fm_c1, fm_w, 16], \
            name="cache_ub", \
            scope=tik.scope_ubuf)
        cache_table = tik_instance.Tensor( \
            "int32", [2, 2], name="cache_table", scope=tik.scope_ubuf)
        with tik_instance.for_range(0, 2) as j:
            one = tik_instance.Scalar(dtype="int32", init_value=-1)
            cache_table[j, 0].set_as(one)
    elif w_number >= 2:
        cache_l1, cache_table = init_l1(tik_instance, dtype, 2, \
                                        fm_c1, fm_w, 16)
        with tik_instance.for_range(0, 2) as j:
            one = tik_instance.Scalar(dtype="int32", init_value=-1)
            cache_table[j, 0].set_as(one)

    val = tik_instance.Tensor(
        dtype, [fm_c1, pw_int, 16], name="val", scope=tik.scope_ubuf)

    tik_instance.vec_dup(16, val, 0.0, fm_c1 * pw_s, n_bust)

    roi_y_floor = tik_instance.Tensor(
        "int32", [128], name="roi_y_floor", scope=tik.scope_ubuf)
    roi_x_floor = tik_instance.Tensor(
        "int32", [128], name="roi_x_floor", scope=tik.scope_ubuf)

    if dtype == "float32":
        raw_fp16 = tik_instance.Tensor(
            "float16", [128], name="raw_fp16", scope=tik.scope_ubuf)
        tik_instance.vec_conv(64, "", raw_fp16[0], raw_y[0], 2, 4, 8)
        # maybe repeattimes 1
        tik_instance.vec_conv(64, "floor", roi_y_floor[0], raw_fp16[0], 2,
                              8, 4)

        tik_instance.vec_conv(64, "", raw_fp16[0], raw_x[0], 2, 4, 8)
        tik_instance.vec_conv(64, "floor", roi_x_floor[0], raw_fp16[0], 2,
                              8, 4)
    else:
        tik_instance.vec_conv(64, "floor", roi_y_floor[0], raw_y[0], 2,
                              8, 4)
        tik_instance.vec_conv(64, "floor", roi_x_floor[0], raw_x[0], 2,
                              8, 4)

    with tik_instance.for_range(0, ph_int * sample_num_h) as grid_num_h:
        verify = tik_instance.Scalar(dtype="int32", init_value=0)
        y_tmp = tik_instance.Scalar(dtype="int32")
        y_tmp.set_as(roi_y_floor[grid_num_h])
        with tik_instance.if_scope(y_tmp < -1):
            verify.set_as(1)
        with tik_instance.if_scope(y_tmp >= fm_h):
            verify.set_as(1)

        with tik_instance.if_scope(verify == 0):
            y_low = tik_instance.Scalar(
                dtype="int32", init_value=y_lo[grid_num_h])
            y_high = tik_instance.Scalar(
                dtype="int32", init_value=y_hi[grid_num_h])
            if w_number_ub >= 2:
                load_a_w_to_l1(tik_instance, cache_table, cache_ub, feature_map,
                               index, y_low, n_bust, 0)
                load_a_w_to_l1(tik_instance, cache_table, cache_ub, feature_map,
                               index, y_high, n_bust, 1)
            elif w_number >= 2:
                load_a_w_to_l1(tik_instance, cache_table, cache_l1, feature_map,
                               index, y_low, n_bust, 0)
                load_a_w_to_l1(tik_instance, cache_table, cache_l1, feature_map,
                               index, y_high, n_bust, 1)

            with tik_instance.for_range(0, pw_int * sample_num_w) as grid_num_w:
                x_tmp = tik_instance.Scalar(dtype="int32")
                x_tmp.set_as(roi_x_floor[grid_num_w])
                with tik_instance.if_scope(x_tmp < -1):
                    verify.set_as(1)
                with tik_instance.if_scope(x_tmp >= fm_w):
                    verify.set_as(1)

                with tik_instance.if_scope(verify == 0):

                    w1_lt = tik_instance.Tensor(
                        dtype, [fm_c1, 16], name="w1_lt", scope=tik.scope_ubuf)
                    w2_rt = tik_instance.Tensor(
                        dtype, [fm_c1, 16], name="w2_rt", scope=tik.scope_ubuf)
                    w3_lb = tik_instance.Tensor(
                        dtype, [fm_c1, 16], name="w3_lb", scope=tik.scope_ubuf)
                    w4_rb = tik_instance.Tensor(
                        dtype, [fm_c1, 16], name="w4_rb", scope=tik.scope_ubuf)
                    x_low = tik_instance.Scalar(
                        dtype="int32", init_value=x_lo[grid_num_w])
                    x_high = tik_instance.Scalar(
                        dtype="int32", init_value=x_hi[grid_num_w])
                    feature_shape = [0, 0, fm_h, fm_w]
                    fm_grid = tik_instance.Tensor(
                        dtype, (fm_c1, 2, 2, 16 * 2 // n_bust),
                        name="fm_grid",
                        scope=tik.scope_ubuf)

                    if fm_to_ub >= 1:
                        load_feature_map_to_ub(tik_instance, fm_grid, feature_shape,
                                               fm_c1, fm_c1, feature_map_ub, index, 0,
                                               y_low, x_low, x_high, y_high, n_bust, 1)
                    elif fm_to_l1 >= 1:
                        load_feature_map_to_ub(tik_instance, fm_grid, feature_shape,
                                               fm_c1, fm_c1, cache_fm, index, 0,
                                               y_low, x_low, x_high, y_high, n_bust, 1)
                    elif w_number_ub >= 2:
                        load_from_l1_cache(
                            tik_instance, feature_map, fm_grid, cache_ub,
                            0, 0, fm_c1, x_low, x_high, fm_c1, n_bust)
                        load_from_l1_cache(
                            tik_instance, feature_map, fm_grid, cache_ub,
                            1, 0, fm_c1, x_low, x_high, fm_c1, n_bust)
                    elif w_number >= 2:
                        load_from_l1_cache(
                            tik_instance, feature_map, fm_grid, cache_l1,
                            0, 0, fm_c1, x_low, x_high, fm_c1, n_bust)
                        load_from_l1_cache(
                            tik_instance, feature_map, fm_grid, cache_l1,
                            1, 0, fm_c1, x_low, x_high, fm_c1, n_bust)
                    else:
                        load_feature_map_to_ub(tik_instance, fm_grid, feature_shape,
                                               fm_c1, fm_c1, feature_map,
                                               index, 0, y_low, x_low,
                                               x_high, y_high, n_bust, 0)

                    h_y = tik_instance.Scalar(
                        dtype, init_value=y_hi_w[grid_num_h])
                    l_y = tik_instance.Scalar(
                        dtype, init_value=y_lo_w[grid_num_h])
                    h_x = tik_instance.Scalar(
                        dtype, init_value=x_hi_w[grid_num_w])
                    l_x = tik_instance.Scalar(
                        dtype, init_value=x_lo_w[grid_num_w])

                    compute_w1234(tik_instance, h_y, l_y, h_x, l_x, w1_lt,
                                  w2_rt, w3_lb, w4_rb, fm_grid, fm_c1, n_bust)

                    with tik_instance.for_range(0, fm_c1) as c_iter_i:
                        tik_instance.vec_add(
                            16,
                            val[c_iter_i, grid_num_w // sample_num_w, 0],
                            val[c_iter_i, grid_num_w // sample_num_w, 0],
                            w1_lt[c_iter_i, 0], 1, n_bust, n_bust, n_bust)

        with tik_instance.if_scope((grid_num_h + 1) % sample_num_h == 0):
            get_average(tik_instance, grid_h_num_f, grid_w_num_f, val, fm_c1,
                        pw_s, n_bust)

            with tik_instance.if_scope(
                    (pw_int * ph_int - pw_int) * n_bust <= 65535):
                tik_instance.data_move(
                    ret[block_i * block_num + 128 * roi_128_number + curr_roi,
                        0, grid_num_h // sample_num_h, 0, 0], val[0, 0, 0], 0,
                    fm_c1,
                    pw_int * n_bust, 0, (pw_int * ph_int - pw_int)*n_bust)

            with tik_instance.else_scope():
                with tik_instance.for_range(0, fm_c1) as c_iter_i:
                    tik_instance.data_move(
                        ret[block_i * block_num + \
                            128 * roi_128_number + curr_roi,
                            c_iter_i, grid_num_h // sample_num_h, 0, 0],
                        val[c_iter_i, 0, 0], 0, 1, pw_int * n_bust, 0, 0)

            tik_instance.vec_dup(16, val, 0.0, fm_c1 * pw_s, n_bust)


def get_grid_weight(tik_instance, grid_w, grid_h, rois_start_w, rois_start_h,
                    height, width, dtype):
    """
    get grid size and coordinate in feature
    """
    x_lo_w = tik_instance.Tensor(
        dtype, [128], name="x_lo_w", scope=tik.scope_ubuf)
    x_hi_w = tik_instance.Tensor(
        dtype, [128], name="x_hi_w", scope=tik.scope_ubuf)
    y_lo_w = tik_instance.Tensor(
        dtype, [128], name="y_lo_w", scope=tik.scope_ubuf)
    y_hi_w = tik_instance.Tensor(
        dtype, [128], name="_lo_w", scope=tik.scope_ubuf)
    x_lo = tik_instance.Tensor(
        "int32", [128], name="x_lo", scope=tik.scope_ubuf)
    x_hi = tik_instance.Tensor(
        "int32", [128], name="x_hi", scope=tik.scope_ubuf)
    y_lo = tik_instance.Tensor(
        "int32", [128], name="y_lo", scope=tik.scope_ubuf)
    y_hi = tik_instance.Tensor(
        "int32", [128], name="y_hi", scope=tik.scope_ubuf)

    raw_x = tik_instance.Tensor(
        dtype, [128], name="raw_x", scope=tik.scope_ubuf)
    raw_y = tik_instance.Tensor(
        dtype, [128], name="raw_y", scope=tik.scope_ubuf)
    x_output = tik_instance.Tensor(
        dtype, [128], name="x_output", scope=tik.scope_ubuf)
    y_output = tik_instance.Tensor(
        dtype, [128], name="y_output", scope=tik.scope_ubuf)
    tmp_fp16 = tik_instance.Tensor(
        "float16", [128], name="tmp_fp16", scope=tik.scope_ubuf)


    const_value_0_127 = tik_instance.Tensor(
        dtype, (128,), name="const_value_0_127", scope=tik.scope_ubuf)
    if dtype == "float32":
        dtype_num = 1
    else:
        dtype_num = 2
    vconv_f322s32f_suppot = tbe_platform.cce_conf.intrinsic_check_support( \
        "Intrinsic_vconv", "f322s32f")
    if vconv_f322s32f_suppot is False or dtype == "float16":
        const_value_0_127_int = tik_instance.Tensor(
            "int32", (128,), name="const_value_0_127_int",
            scope=tik.scope_ubuf)
        with tik_instance.for_range(0, 128) as i:
            const_value_0_127_int[i].set_as(i)
        if dtype == "float32":
            const_value_0_127_float = tik_instance.Tensor(
                "float16", (128,),
                name="const_value_0_127_float",
                scope=tik.scope_ubuf)
            tik_instance.vec_conv(64, "", const_value_0_127_float,
                               const_value_0_127_int, 2, 4, 8, 1.0)
            tik_instance.vec_conv(64, "", const_value_0_127,
                               const_value_0_127_float,
                               2, 8, 4)
        else:
            tik_instance.vec_conv(64, "", const_value_0_127,
                               const_value_0_127_int,
                               2, 4, 8, 1.0)

    else:
        with tik_instance.for_range(0, 128) as i:
            const_value_0_127[i] = i


    grid_w_vector = tik_instance.Tensor(
        dtype, [128], name="grid_w_vector", scope=tik.scope_ubuf)
    grid_h_vector = tik_instance.Tensor(
        dtype, [128], name="grid_h_vector", scope=tik.scope_ubuf)

    tik_instance.vec_muls(64 * dtype_num, grid_w_vector, const_value_0_127,
                       grid_w, 2 // dtype_num, 8, 8)
    tik_instance.vec_muls(64 * dtype_num, grid_h_vector, const_value_0_127,
                       grid_h, 2 // dtype_num, 8, 8)
    # fp16 scalar floating-point operation is not allowed in aicore fucntion
    # fl32 scalar floating-point operation is not allowed on mini
    if vconv_f322s32f_suppot is False or dtype == "float16":
        point_05 = tik_instance.Scalar(dtype, init_value=0.5)
        point_05_tensor = tik_instance.Tensor(
            dtype, [1], name="point_05_tensor", scope=tik.scope_ubuf)
        tik_instance.vec_dup(1, point_05_tensor, 0.5, 1, 0)
        tik_instance.vec_muls(1, point_05_tensor, point_05_tensor, grid_w,
                              1, 8, 8)
        tik_instance.vec_adds(1, point_05_tensor, point_05_tensor, rois_start_w,
                              1, 8, 8)
        point_05.set_as(point_05_tensor[0])
        tik_instance.vec_adds(64 * dtype_num, raw_x, grid_w_vector, point_05,
                           2 // dtype_num, 8, 8)
        tik_instance.vec_dup(1, point_05_tensor, 0.5, 1, 0)

        tik_instance.vec_muls(1, point_05_tensor, point_05_tensor, grid_h,
                              1, 8, 8)
        tik_instance.vec_adds(1, point_05_tensor, point_05_tensor, rois_start_h,
                              1, 8, 8)
        point_05.set_as(point_05_tensor[0])
        tik_instance.vec_adds(64 * dtype_num, raw_y, grid_h_vector,
                           point_05, 2 // dtype_num, 8, 8)
    # fp32 besides mini
    else:
        half_grid = 0.5 * grid_w + rois_start_w
        tik_instance.vec_adds(64 * dtype_num, raw_x, grid_w_vector,
                           half_grid, 2 // dtype_num, 8, 8)
        half_grid = 0.5 * grid_h + rois_start_h
        tik_instance.vec_adds(64 * dtype_num, raw_y, grid_h_vector,
                           half_grid, 2 // dtype_num, 8, 8)

    const_zero = tik_instance.Tensor(
        dtype, [64 * dtype_num], name="const_zero", scope=tik.scope_ubuf)

    tik_instance.vec_dup(64 * dtype_num, const_zero, 0, 1, 0)

    # if (y <= 0) y = 0;
    # if (x <= 0) x = 0;
    tik_instance.vec_max(64 * dtype_num, x_output, raw_x, const_zero,
                      2 // dtype_num, 8, 8, 0)
    tik_instance.vec_max(64 * dtype_num, y_output, raw_y, const_zero,
                      2 // dtype_num, 8, 8, 0)

    # y_low = (int)y;
    # x_low = (int)x;
    if vconv_f322s32f_suppot is False and dtype == "float32":
        tik_instance.vec_conv(64, "", tmp_fp16, x_output, 2, 4, 8)
        tik_instance.vec_conv(64, "floor", x_lo, tmp_fp16, 2, 8, 4)
        tik_instance.vec_conv(64, "", tmp_fp16, y_output, 2, 4, 8)
        tik_instance.vec_conv(64, "floor", y_lo, tmp_fp16, 2, 8, 4)
    else:
        tik_instance.vec_conv(64, "floor", x_lo, x_output, 2,
                              8, 8 // dtype_num)
        tik_instance.vec_conv(64, "floor", y_lo, y_output, 2,
                              8, 8 // dtype_num)

    # y_high = y_low + 1;
    # x_high = x_low + 1;
    const_one = tik_instance.Tensor(
        "int32", [64], name="const_one", scope=tik.scope_ubuf)
    tik_instance.vec_dup(64, const_one, 1, 1, 0)
    # 128 int32 4B /256
    tik_instance.vec_add(64, x_hi, x_lo, const_one, 2, 8, 8, 0)
    tik_instance.vec_add(64, y_hi, y_lo, const_one, 2, 8, 8, 0)

    const_value_fp32 = tik_instance.Tensor(
        dtype, [64 * dtype_num], name="const_value_fp32", scope=tik.scope_ubuf)
    const_value_int32 = tik_instance.Tensor(
        "int32", [64], name="const_value_int32", scope=tik.scope_ubuf)

    tik_instance.vec_dup(64 * dtype_num, const_value_fp32, width - 1, 1, 0)
    tik_instance.vec_dup(64, const_value_int32, width - 1, 1, 0)
    tik_instance.vec_min(64, x_lo, x_lo, const_value_int32, 2, 8, 8, 0)
    tik_instance.vec_min(64, x_hi, x_hi, const_value_int32, 2, 8, 8, 0)
    tik_instance.vec_min(64 * dtype_num, x_output, x_output, const_value_fp32,
                      2 // dtype_num, 8, 8, 0)

    tik_instance.vec_dup(64, const_value_int32, height - 1, 1, 0)
    tik_instance.vec_dup(64 * dtype_num, const_value_fp32, height - 1, 1, 0)
    tik_instance.vec_min(64, y_lo, y_lo, const_value_int32, 2, 8, 8, 0)
    tik_instance.vec_min(64, y_hi, y_hi, const_value_int32, 2, 8, 8, 0)
    tik_instance.vec_min(64 * dtype_num, y_output, y_output, const_value_fp32,
                      2 // dtype_num, 8, 8, 0)

    # ly = y - y_low;
    # lx = x - x_low;
    tmp_fp32 = tik_instance.Tensor(
        dtype, [128], name="tmp_fp32", scope=tik.scope_ubuf)

    if vconv_f322s32f_suppot is False and dtype == "float32":
        tik_instance.vec_conv(64, "", tmp_fp16, x_lo, 2, 4, 8, 1.0)
        tik_instance.vec_conv(64, "", tmp_fp32, tmp_fp16, 2, 8, 4)
    else:
        # float16 add 1.0 float32 can not add 1.0
        if dtype == "float32":
            tik_instance.vec_conv(64, "", tmp_fp32, x_lo, 2, 8, 8)
        else:
            tik_instance.vec_conv(64, "", tmp_fp32, x_lo, 2,
                               8 // dtype_num, 8, 1.0)

    tik_instance.vec_sub(64 * dtype_num, x_lo_w, x_output, tmp_fp32,
                         2 // dtype_num, 8, 8, 8)

    if vconv_f322s32f_suppot is False and dtype == "float32":
        tik_instance.vec_conv(64 * dtype_num, "", tmp_fp16, y_lo, 2,
                              4, 8, 1.0)
        tik_instance.vec_conv(64 * dtype_num, "", tmp_fp32, tmp_fp16, 2,
                              8, 4)
    else:
        if dtype == "float32":
            tik_instance.vec_conv(64, "", tmp_fp32, y_lo, 2, 8, 8)
        else:
            tik_instance.vec_conv(64, "", tmp_fp32, y_lo, 2,
                               8 // dtype_num, 8, 1.0)

    tik_instance.vec_sub(64 * dtype_num, y_lo_w, y_output, tmp_fp32,
                      2 // dtype_num, 8, 8, 8)

    # hy = 1. - ly;
    # hx = 1. - lx;
    tik_instance.vec_dup(64 * dtype_num, const_value_fp32, 1.0, 1, 0)
    tik_instance.vec_sub(64 * dtype_num, x_hi_w, const_value_fp32, x_lo_w,
                      2 // dtype_num, 8, 0, 8)
    tik_instance.vec_sub(64 * dtype_num, y_hi_w, const_value_fp32, y_lo_w,
                      2 // dtype_num, 8, 0, 8)

    return x_lo_w, x_hi_w, y_lo_w, y_hi_w, x_lo, x_hi, y_lo, y_hi, raw_x, raw_y


def get_input(tik_instance, dtype, grid_h, grid_w, proposals_ub_y0,
              proposals_ub_x0, grid_h_int32, grid_w_int32,
              grid_h_fp32, grid_w_fp32, curr_roi):
    """
    :param tik_instance:
    :param dtype:
    :param grid_h:
    :param grid_w:
    :param proposals_ub_y0:
    :param proposals_ub_x0:
    :param roi_fp16_pos_16:
    :param grid_h_int32:
    :param grid_w_int32:
    :param grid_h_fp32:
    :param grid_w_fp32:
    :param curr_roi:
    :return:list
    """
    grid_h_roi = tik_instance.Scalar(dtype=dtype)
    grid_h_roi.set_as(grid_h[curr_roi])

    grid_w_roi = tik_instance.Scalar(dtype=dtype)
    grid_w_roi.set_as(grid_w[curr_roi])

    rois_start_h = tik_instance.Scalar(dtype=dtype)
    rois_start_h.set_as(proposals_ub_y0[curr_roi, 0])
    rois_start_w = tik_instance.Scalar(dtype=dtype)
    rois_start_w.set_as(proposals_ub_x0[curr_roi, 0])

    grid_h_num = tik_instance.Scalar(dtype="int32")
    grid_h_num.set_as(grid_h_int32[curr_roi])
    grid_w_num = tik_instance.Scalar(dtype="int32")
    grid_w_num.set_as(grid_w_int32[curr_roi])

    grid_h_num_f = tik_instance.Scalar(dtype=dtype)
    grid_h_num_f.set_as(grid_h_fp32[curr_roi])
    grid_w_num_f = tik_instance.Scalar(dtype=dtype)
    grid_w_num_f.set_as(grid_w_fp32[curr_roi])

    return grid_w_roi, grid_h_roi, grid_w_num, \
           grid_h_num, rois_start_w, rois_start_h, \
           grid_h_num_f, grid_w_num_f


def load_a_w_to_l1(tik_instance, cache_table, cache_l1, feature_map,
                   index, y_low, n_bust, point):
    """
    load a width of feature map to l1
    """
    stride = (feature_map.shape[2] * feature_map.shape[3] - \
              feature_map.shape[3]) * n_bust
    c_iter_c1 = feature_map.shape[1]
    fm_h = feature_map.shape[2]
    fm_w = feature_map.shape[3]

    if stride > 65535:
        with tik_instance.for_range(0, c_iter_c1) as c_iter_i:
            tik_instance.data_move(
                cache_l1[point, c_iter_i, 0, 0],
                feature_map[index, c_iter_i, y_low, 0, 0],
                0, 1, fm_w * n_bust, 1, 1)
    else:
        tik_instance.data_move(cache_l1[point, 0, 0, 0],
                               feature_map[index, 0, y_low, 0, 0],
                               0, c_iter_c1, fm_w * n_bust,
                               (fm_h * fm_w - fm_w) * n_bust,
                               0)
    # ylow:
    cache_table[point, 0].set_as(index)
    cache_table[point, 1].set_as(y_low)


def load_from_l1_cache(tik_instance, feature_map, fm_grid, cache_l1, point,
                       current_cb, c_block, x_low, x_high, c_valid, n_bust):
    """
    load feature map from l1 cache
    """

    tik_instance.data_move(fm_grid[0, point, 0, 0],
                           cache_l1[point, current_cb * c_block, x_low, 0], 0,
                           c_valid, n_bust, (feature_map.shape[3] - 1) * n_bust,
                           (4 - 1) * n_bust)
    tik_instance.data_move(fm_grid[0, point, 1, 0],
                           cache_l1[point, current_cb * c_block, x_high, 0], 0,
                           c_valid, n_bust, (feature_map.shape[3] - 1) * n_bust,
                           (4 - 1) * n_bust)


def init_l1(tik_instance, dtype, w_number, fm_c1, fm_w, fm_c0):
    """
    initialize L1 cache
    """
    cache_l1 = tik_instance.Tensor(
        dtype, [w_number, fm_c1, fm_w, fm_c0],
        name="cache_l1",
        scope=tik.scope_cbuf)
    cache_table = tik_instance.Tensor(
        "int32", [w_number, 2], name="cache_table", scope=tik.scope_ubuf)

    return cache_l1, cache_table


def tf_n52n8(tik_instance, rois_ub, rois_n5, block_num):
    """
    transform ROIS form N5 to N8
    """
    with tik_instance.for_range(0, block_num) as rois_num:
        rois_ub[rois_num, 0].set_as(rois_n5[rois_num, 0])
        rois_ub[rois_num, 1].set_as(rois_n5[rois_num, 1])
        rois_ub[rois_num, 2].set_as(rois_n5[rois_num, 2])
        rois_ub[rois_num, 3].set_as(rois_n5[rois_num, 3])
        rois_ub[rois_num, 4].set_as(rois_n5[rois_num, 4])


def load_feature_map_to_ub(tik_instance, fm_grid, feature_shape,
                           c_block, c_valid,
                           feature_map, index, current_cb,
                           y_low, x_low, x_high,
                           y_high, n_bust, cache_flag):
    """
    load feature map from ddr to ub
    """

    stride = (feature_shape[2] * feature_shape[3] - 1) * n_bust
    stride_s = tik_instance.Scalar(dtype="int32", init_value=stride)

    with tik_instance.if_scope(cache_flag == 1):
        index.set_as(0)

    with tik_instance.if_scope(stride <= 65535):
        tik_instance.data_move(
            fm_grid[0, 0, 0, 0],
            feature_map[index, current_cb * c_block, y_low, x_low, 0], 0,
            c_valid, n_bust, stride_s, (4 - 1) * n_bust)
        tik_instance.data_move(
            fm_grid[0, 0, 1, 0],
            feature_map[index, current_cb * c_block, y_low, x_high, 0], 0,
            c_valid, n_bust, stride_s, (4 - 1) * n_bust)
        tik_instance.data_move(
            fm_grid[0, 1, 1, 0],
            feature_map[index, current_cb * c_block, y_high, x_high, 0], 0,
            c_valid, n_bust, stride_s, (4 - 1) * n_bust)
        tik_instance.data_move(
            fm_grid[0, 1, 0, 0],
            feature_map[index, current_cb * c_block, y_high, x_low, 0], 0,
            c_valid, n_bust, stride_s, (4 - 1) * n_bust)

    with tik_instance.else_scope():
        with tik_instance.for_range(0, c_valid) as c_iter_i:
            tik_instance.data_move(fm_grid[c_iter_i, 0, 0, 0], feature_map[
                index, current_cb * c_block + c_iter_i, y_low, x_low, 0], 0,
                                   1, n_bust, 1, 1)
            tik_instance.data_move(fm_grid[c_iter_i, 0, 1, 0], feature_map[
                index, current_cb * c_block + c_iter_i, y_low, x_high, 0], 0,
                                   1, n_bust, 1, 1)
            tik_instance.data_move(fm_grid[c_iter_i, 1, 1, 0], feature_map[
                index, current_cb * c_block + c_iter_i, y_high, x_high, 0], 0,
                                   1, n_bust, 1, 1)
            tik_instance.data_move(fm_grid[c_iter_i, 1, 0, 0], feature_map[
                index, current_cb * c_block + c_iter_i, y_high, x_low, 0], 0,
                                   1, n_bust, 1, 1)


def roi_align_compute(tik_instance, feature_map, ret, proposals_ub_x0,
                      proposals_ub_y0, pool_h, pool_w, dtype, roi_128_number,
                      rois_valid_in_block,
                      feature_shape, grid_curr_h, grid_curr_w, fm_c1, n_bust,
                      block_i, block_num, roi_int32_fm_index, grid_h_int32,
                      grid_w_int32, grid_h_fp32, grid_w_fp32,
                      roi_bin_h_fp32_value,
                      roi_bin_w_fp32_value, w_number, fm_to_l1, fm_to_ub,
                      w_number_ub):
    """
    get ret without L1
    """
    grid_h = tik_instance.Tensor(
        dtype, [128], name="grid_h", scope=tik.scope_ubuf)
    grid_w = tik_instance.Tensor(
        dtype, [128], name="grid_w", scope=tik.scope_ubuf)
    if dtype == "float32":
        dtype_num = 1
    else:
        dtype_num = 2
    vdiv_suppot = tbe_platform.cce_conf.api_check_support( \
        "tik.vdiv", "float32")
    if vdiv_suppot is False:
        newton(tik_instance, 64 * dtype_num, grid_h, roi_bin_h_fp32_value,
               grid_h_fp32,
               2 // dtype_num, dtype)
        newton(tik_instance, 64 * dtype_num, grid_w, roi_bin_w_fp32_value,
               grid_w_fp32,
               2 // dtype_num, dtype)

    else:
        tik_instance.vdiv(64 * dtype_num, grid_h, roi_bin_h_fp32_value,
                          grid_h_fp32, 2 // dtype_num, 1,
                          1, 1, 8, 8, 8)
        tik_instance.vdiv(64 * dtype_num, grid_w, roi_bin_w_fp32_value,
                          grid_w_fp32, 2 // dtype_num, 1,
                          1, 1, 8, 8, 8)

    if fm_to_ub >= 1:
        feature_map_ub = tik_instance.Tensor(
            dtype, [1, fm_c1, feature_shape[2], feature_shape[3], 16],
            name="feature_map_ub",
            scope=tik.scope_ubuf)
        cache_index = tik_instance.Scalar(dtype="int32", init_value=-1)
    else:
        feature_map_ub = tik_instance.Tensor(
            dtype, [1], name="feature_map_ub", scope=tik.scope_ubuf)
        cache_index = tik_instance.Scalar(dtype="int32", init_value=-1)

    if fm_to_l1 >= 1:
        cache_fm = tik_instance.Tensor(
            dtype, [1, fm_c1, feature_shape[2], feature_shape[3], 16],
            name="cache_fm",
            scope=tik.scope_cbuf)
        cache_index = tik_instance.Scalar(dtype="int32", init_value=-1)
    else:
        cache_fm = tik_instance.Tensor(
            dtype, [1], name="cache_fm", scope=tik.scope_ubuf)
        cache_index = tik_instance.Scalar(dtype="int32", init_value=-1)

    with tik_instance.for_range(0, rois_valid_in_block) as curr_roi:
        index = tik_instance.Scalar(dtype="int32")

        index.set_as(roi_int32_fm_index[curr_roi])
        grid_curr_h.set_as(grid_h_int32[curr_roi])
        grid_curr_w.set_as(grid_w_int32[curr_roi])

        w_num = tik_instance.Scalar(dtype="int32")
        h_num = tik_instance.Scalar(dtype="int32")
        w_num.set_as((grid_curr_w * pool_w + 127) // 128)
        h_num.set_as((grid_curr_h * pool_h + 127) // 128)
        grid_curr_h_f32 = tik_instance.Scalar(
            dtype=dtype, init_value=grid_h_fp32[curr_roi])
        grid_curr_w_f32 = tik_instance.Scalar(
            dtype=dtype, init_value=grid_w_fp32[curr_roi])

        flag_para = tik_instance.Scalar(dtype="int32", init_value=0)
        with tik_instance.if_scope(w_num > 1):
            flag_para.set_as(1)
        with tik_instance.if_scope(h_num > 1):
            flag_para.set_as(1)
        with tik_instance.if_scope(fm_c1 * pool_w > 255):
            flag_para.set_as(1)
        pool_w_s = tik_instance.Scalar(dtype="int32", init_value=pool_w)

        with tik_instance.if_scope(flag_para == 0):
            if fm_c1 * pool_w <= 255:
                grid_w_roi, grid_h_roi, grid_w_num, \
                grid_h_num, rois_start_w, rois_start_h, \
                grid_h_num_f, grid_w_num_f = \
                    get_input(tik_instance, dtype, grid_h, grid_w,
                              proposals_ub_y0,
                              proposals_ub_x0,
                              grid_h_int32, grid_w_int32,
                              grid_h_fp32, grid_w_fp32, curr_roi)

                x_lo_w, x_hi_w, y_lo_w, y_hi_w, \
                x_lo, x_hi, y_lo, y_hi, raw_x, raw_y = \
                    get_grid_weight(tik_instance, grid_w_roi,
                                    grid_h_roi, rois_start_w, rois_start_h,
                                    feature_shape[2], feature_shape[3], dtype)

                bilinear_interpolate(
                    tik_instance, x_lo_w, x_hi_w, y_lo_w, y_hi_w, x_lo, x_hi, y_lo,
                    y_hi, raw_x, raw_y, grid_w_num, grid_h_num, grid_h_num_f,
                    grid_w_num_f, feature_shape[2], feature_shape[3], fm_c1, dtype,
                    n_bust, pool_w_s, pool_w, pool_h, block_i, block_num, index,
                    curr_roi, feature_map, ret, roi_128_number, w_number,
                    fm_to_l1, cache_fm, cache_index, fm_to_ub, w_number_ub,
                    feature_map_ub)

        with tik_instance.else_scope():
            compute_roi_with_single_point(tik_instance, feature_shape, dtype,
                                          fm_to_l1,
                                          fm_c1, block_i, index,
                                          roi_128_number,
                                          block_num, w_number, pool_h, pool_w,
                                          n_bust,
                                          grid_curr_h, roi_bin_h_fp32_value,
                                          ret, grid_curr_w_f32,
                                          grid_w_fp32, grid_curr_h_f32,
                                          proposals_ub_y0,
                                          grid_h_fp32, roi_bin_w_fp32_value,
                                          proposals_ub_x0, feature_map, curr_roi,
                                          grid_curr_w, cache_fm, cache_index,
                                          fm_to_ub, w_number_ub, feature_map_ub)


def roi_align_tik(feature_map_dict, rois_dict, output, \
                  scale, pool_h, pool_w, sample_ratio, \
                  roi_end_mode, kernel_name):

    tik_instance = tik.Tik(tik.Dprofile(), True)
    rois_shape = rois_dict.get("shape")
    dtype = feature_map_dict.get("dtype")
    feature_shape = feature_map_dict.get("shape")
    feature_map = tik_instance.Tensor(
        dtype, feature_shape, name="feature_map", scope=tik.scope_gm)
    rois = tik_instance.Tensor(
        dtype, rois_shape, name="rois", scope=tik.scope_gm)
    fm_c1 = feature_shape[1]
    fm_c0 = 16
    proposal_num = rois_shape[0]
    ret = tik_instance.Tensor(
        dtype, [rois_shape[0], fm_c1, pool_h, pool_w, fm_c0],
        name="ret",
        scope=tik.scope_gm)
    grid_curr_h = tik_instance.Scalar(dtype="int32")
    grid_curr_w = tik_instance.Scalar(dtype="int32")
    core_num = \
        tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
    block_num = rois_shape[0] // core_num
    if block_num == 0:
        block_num = 1
    if dtype == "float32":
        n_bust = 2
    else:
        n_bust = 1
    l1_size = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.L1_SIZE)
    # every block, process 128 rois
    with tik_instance.for_range(
            0, (proposal_num + (block_num - 1)) // block_num,
            block_num=(proposal_num + (block_num - 1)) // block_num) \
            as block_i:

        rois_ub = tik_instance.Tensor(
            dtype, [128, 8], name="rois_ub", scope=tik.scope_ubuf)
        proposals_ub_x0 = tik_instance.Tensor(
            dtype, [128, 1], name="proposals_ub_x0", scope=tik.scope_ubuf)
        proposals_ub_y0 = tik_instance.Tensor(
            dtype, [128, 1], name="proposals_ub_y0", scope=tik.scope_ubuf)
        proposals_ub_x1 = tik_instance.Tensor(
            dtype, [128, 1], name="proposals_ub_x1", scope=tik.scope_ubuf)
        proposals_ub_y1 = tik_instance.Tensor(
            dtype, [128, 1], name="proposals_ub_y1", scope=tik.scope_ubuf)

        rois_valid = tik_instance.Scalar(dtype="int32", init_value=block_num)
        with tik_instance.if_scope(
                block_i == ((proposal_num + (block_num - 1)) // block_num - 1)):
            rois_valid.set_as(proposal_num - block_i * block_num)
        with tik_instance.if_scope(rois_valid != 0):
            with tik_instance.for_range(
                    0, (rois_valid + (128 - 1))//128) as roi_128_number:
                rois_valid_in_block = \
                    tik_instance.Scalar(dtype="int32", init_value=128)
                with tik_instance.if_scope(
                        roi_128_number == ((rois_valid + (128 - 1))//128 - 1)):
                    rois_valid_in_block.set_as(
                        rois_valid - roi_128_number * 128)

                if rois_shape[1] == 5:
                    rois_ub_n5 = tik_instance.Tensor(
                        dtype, [128, 5], name="rois_ub_n5",
                        scope=tik.scope_ubuf)
                    tik_instance.data_move(rois_ub_n5[0, 0],
                                           rois[block_i * block_num +
                                                roi_128_number * 128, 0],
                                           0, 1,
                                           40 * n_bust, 0, 0)
                    tf_n52n8(tik_instance, rois_ub, rois_ub_n5, 128)
                else:
                    tik_instance.data_move(rois_ub[0, 0],
                                           rois[block_i * block_num +
                                                roi_128_number * 128, 0],
                                           0, 1,
                                           64 * n_bust, 0, 0)

                support_vextract = \
            tbe_platform.cce_conf.api_check_support("tik.vextract", "float32")
                if dtype == "float16":
                    cce_product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
                    if cce_product == "Ascend310":
                        j_value = tik_instance.Scalar(dtype=dtype)
                        with tik_instance.for_range(0, 128) as j:
                            j_value.set_as(rois_ub[j, 4])
                            proposals_ub_y1[j, 0].set_as(j_value)
                    else:
                        tik_instance.vextract(proposals_ub_y1[0, 0], rois_ub[0], 8, 4)
                    tik_instance.vextract(proposals_ub_x0[0, 0], rois_ub[0], 8, 1)
                    tik_instance.vextract(proposals_ub_y0[0, 0], rois_ub[0], 8, 2)
                    tik_instance.vextract(proposals_ub_x1[0, 0], rois_ub[0], 8, 3)

                else:
                    if support_vextract is False:
                        j_value = tik_instance.Scalar(dtype=dtype)
                        with tik_instance.for_range(0, 128) as j:
                            j_value.set_as(rois_ub[j, 1])
                            proposals_ub_x0[j, 0].set_as(j_value)
                            j_value.set_as(rois_ub[j, 2])
                            proposals_ub_y0[j, 0].set_as(j_value)
                            j_value.set_as(rois_ub[j, 3])
                            proposals_ub_x1[j, 0].set_as(j_value)
                            j_value.set_as(rois_ub[j, 4])
                            proposals_ub_y1[j, 0].set_as(j_value)
                    else:
                        tik_instance.vextract(proposals_ub_x0[0, 0], rois_ub[0], 8, 1)
                        tik_instance.vextract(proposals_ub_y0[0, 0], rois_ub[0], 8, 2)
                        tik_instance.vextract(proposals_ub_x1[0, 0], rois_ub[0], 8, 3)
                        tik_instance.vextract(proposals_ub_y1[0, 0], rois_ub[0], 8, 4)


                tik_instance, roi_bin_h_fp32_value, \
                roi_bin_w_fp32_value, \
                proposals_ub_x0, proposals_ub_y0, \
                grid_w_int32, grid_h_int32, grid_w_fp32, \
                grid_h_fp32, roi_int32_fm_index = \
                    get_roi_align_perf_scale_for_zero(tik_instance, rois_ub,
                                                      proposals_ub_x0,
                                                      proposals_ub_y0,
                                                      proposals_ub_x1,
                                                      proposals_ub_y1,
                                                      scale, pool_h, pool_w,
                                                      sample_ratio,
                                                      roi_end_mode,
                                                      dtype)
                w_number = 0
                feature_map_to_l1_verify = 0
                w_number_ub = 0
                ub_size_bytes = \
                 tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE) - UB_30K_SIZE
                feature_map_to_ub_verify = ub_size_bytes // \
                                (fm_c1 * feature_shape[2] * feature_shape[3] * 16 * n_bust * 2)
                feature_map_to_l1_verify = \
                                l1_size // (fm_c1 * feature_shape[2] * \
                                feature_shape[3] * 16 * n_bust * 2)
                if feature_map_to_ub_verify == 0 and feature_map_to_l1_verify == 0:
                    w_number_ub = ub_size_bytes // \
                                  (feature_shape[1] * feature_shape[3] *
                                   feature_shape[4] * n_bust * 2)
                if feature_map_to_ub_verify == 0 and \
                   feature_map_to_l1_verify == 0 and w_number_ub == 0:
                    if (feature_shape[3] - 1) * n_bust < 65535:
                        w_number = l1_size // (feature_shape[1] * \
                                               feature_shape[3] * \
                                               feature_shape[4] * n_bust * 2)

                roi_align_compute(
                    tik_instance, feature_map, ret, proposals_ub_x0,
                    proposals_ub_y0, pool_h, pool_w, dtype, roi_128_number,
                    rois_valid_in_block,
                    feature_shape, grid_curr_h, grid_curr_w, fm_c1, n_bust,
                    block_i, block_num, roi_int32_fm_index, grid_h_int32,
                    grid_w_int32, grid_h_fp32, grid_w_fp32,
                    roi_bin_h_fp32_value,
                    roi_bin_w_fp32_value, w_number,
                    feature_map_to_l1_verify, feature_map_to_ub_verify,
                    w_number_ub)

    tik_instance.BuildCCE(
        kernel_name=kernel_name, inputs=[feature_map, rois], outputs=[ret])
    return tik_instance

# pylint: disable=unused-argument
@util.check_input_type(dict, dict, (dict, NoneType),
                       dict, float, int, int, int, int, str)
def roi_align(feature_map_dict,
              rois_dict,
              roisn,
              output,
              scale,
              pool_h,
              pool_w,
              sample_ratio = 2,
              roi_end_mode = 1,
              kernel_name="roi_align"):
    """
    ROIAlign operator
    """
    dtype = feature_map_dict.get("dtype")

    if ((tik.Dprofile().get_product_name() in ("aic", )) \
        and (dtype == "float16") and \
        (pool_h == 7) and (pool_w == 7) and (roi_end_mode == 1)):
        return roi_align_vbi.roi_align_vbi(feature_map_dict, \
            rois_dict, kernel_name)
    elif dtype == "float16":
        return roi_align_cce(feature_map_dict, rois_dict, output,
                      scale, pool_h, pool_w, sample_ratio,
                      roi_end_mode, kernel_name)
    else:
        return roi_align_tik(feature_map_dict, rois_dict, output,
                      scale, pool_h, pool_w, sample_ratio,
                      roi_end_mode, kernel_name)
