"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

conv2d backprop input general compute.
"""

# !/usr/bin/python
# -*- coding: UTF-8 -*-
from te import tvm
from te.platform import CUBE_MKN
from te.lang.cce.te_compute.conv3d_backprop_input_cube_util \
    import CubeDslPattern
from te.lang.cce.te_compute.conv3d_backprop_input_cube_util \
    import ConvDslPattern
from topi.cce.util import is_lhisi_version


class DeConvPattern(CubeDslPattern): # pylint: disable=R0902
    """
    class of convolution back propagation

    Parameters
    ----------
    kernel_sizes : shape of weight, [N, D, H, W, C]

    strides :list of strides,
             [stridebatch, strided, strideh, stridew, stridechannel]

    pad: list of padding, [pad_front, pad_tail, pad_top,
                           pad_bottom, pad_left, pad_right]

    output_shape : shape of dE/dX, [N, D, H, W, C]

    Returns
    -------
    deconv_pattern_instance : instance of deconv pattern
    """

    def __init__(self, # pylint: disable=R0913
                 kernel_sizes, strides, pad, output_shape, dilations):
        super(DeConvPattern, self).__init__()
        _, _, kernel_d, kernel_h, kernel_w = kernel_sizes
        self._kernel_d = kernel_d
        self._kernel_h = kernel_h
        self._kernel_w = kernel_w
        _, self._stride_d, self._stride_h, self._stride_w, _ = strides
        self._pad_head, self._pad_tail, self._pad_up, self._pad_down,\
        self._pad_left, self._pad_right = pad
        self.output_shape = output_shape
        self.m_0, _, _ = CUBE_MKN["float16"]['mac']
        self.dy_d = 0
        _, _, self._dilate_h, self._dilate_w, _ = dilations

    def generate_a(self, dy_ddr): # pylint: disable=R0914
        """
        generate dy_col tensor for mad

        Parameters
        ----------
        dy_ddr : 5D dE/dY tensor in ddr

        Returns
        ----------
        dy_col: dE/dY tensor of fractal shape in L0A
        """
        dy_batch, dy_deep, kernel_cout1, dy_h, dy_w, kernel_cout0 = \
            list(i.value for i in dy_ddr.shape)
        stride_d, stride_h, stride_w = \
            self._stride_d, self._stride_h, self._stride_w
        shape_dy_filling = (dy_batch, dy_deep, kernel_cout1, dy_h * stride_h,
                            dy_w * stride_w, kernel_cout0)
        self.dy_d = dy_deep
        if stride_h == 1 and stride_w == 1:
            dy_filling = dy_ddr
        else:
            dy_zero = tvm.compute(
                shape_dy_filling,
                lambda *indice: tvm.convert(0).astype(dy_ddr.dtype),
                name="dy_filling_i",
                tag="init_zero")
            dy_filling = tvm.compute(
                shape_dy_filling,
                lambda batch_idx,
                       dy_deep_idx,
                       kernel_cout1_idx,
                       ho_idx,
                       wo_idx,
                       kernel_cout0_idx:
                tvm.select(tvm.all(ho_idx % stride_h == 0,
                                   wo_idx % stride_w == 0),
                           dy_ddr(batch_idx,
                                  dy_deep_idx,
                                  kernel_cout1_idx,
                                  ho_idx // stride_h,
                                  wo_idx // stride_w,
                                  kernel_cout0_idx),
                           dy_zero(batch_idx,
                                   dy_deep_idx,
                                   kernel_cout1_idx,
                                   ho_idx,
                                   wo_idx,
                                   kernel_cout0_idx)),
                name="dy_filling",
                tag="stride_filling",
                attrs={"stride_expand": (self._stride_h, self._stride_w)})

        kernel_d, kernel_h, kernel_w =\
            self._kernel_d, self._kernel_h, self._kernel_w
        new_stride = (1, 1, 1)
        new_hw = (dy_deep * stride_d, dy_h * stride_h, dy_w * stride_w)
        new_pad_before = (kernel_d - 1 - self._pad_head,
                          kernel_h - 1 - self._pad_up,
                          kernel_w - 1 - self._pad_left)
        _, dx_d, _, dx_h, dx_w, _ = self.output_shape
        new_pad_after = tuple((o - 1) * s + k - i - pb for i, pb, k, s, o in \
                              zip(new_hw,
                                  new_pad_before,
                                  (kernel_d, kernel_h, kernel_w),
                                  new_stride,
                                  (dx_d, dx_h, dx_w)))
        pad_tail_after, pad_down_after, pad_right_after = new_pad_after

        # stride > 1 ub->l1 may cut
        if stride_h > 1 or stride_w > 1:
            if pad_down_after < 0 or pad_right_after < 0 or pad_tail_after < 0:
                shape_down_modify = (pad_down_after - abs(pad_down_after)) // 2
                shape_right_modify = \
                    (pad_right_after - abs(pad_right_after)) // 2
                shape_dy_filling_cut = (dy_batch, dy_deep,
                                        kernel_cout1,
                                        dy_h * stride_h + shape_down_modify,
                                        dy_w * stride_w + shape_right_modify,
                                        kernel_cout0)
                # cut dy_filling
                dy_filling_l1 = tvm.compute(
                    shape_dy_filling_cut,
                    lambda batch_idx,
                           dy_deep,
                           kernel_cout1_idx,
                           ho_idx,
                           wo_idx,
                           kernel_cout0_idx:
                    dy_filling[batch_idx,\
                               dy_deep,
                               kernel_cout1_idx,\
                               ho_idx,\
                               wo_idx,\
                               kernel_cout0_idx],
                    name="dy_l1_cut",
                    tag="dy_l1_cut")

                pad_down_after = (pad_down_after + abs(pad_down_after)) // 2
                pad_right_after = (pad_right_after + abs(pad_right_after)) // 2
                pad_tail_after = (pad_tail_after + abs(pad_tail_after)) // 2
            else:
                dy_filling_l1 = tvm.compute(
                    shape_dy_filling,
                    lambda batch_idx,
                           dy_deep,
                           kernel_cout1_idx,
                           ho_idx,
                           wo_idx,
                           kernel_cout0_idx:
                    dy_filling[batch_idx,
                               dy_deep,
                               kernel_cout1_idx,
                               ho_idx,
                               wo_idx,
                               kernel_cout0_idx],
                    name="dy_l1",
                    tag="dy_l1")

        new_pad = (kernel_d - 1 - self._pad_head,
                   pad_tail_after,
                   kernel_h - 1 - self._pad_up,
                   pad_down_after,
                   kernel_w - 1 - self._pad_left,
                   pad_right_after)
        pat_conv = ConvDslPattern(kernel_h, kernel_w, new_stride, new_pad)

        if stride_h > 1 or stride_w > 1:
            dy_col = pat_conv.generate_a(dy_filling_l1)
        else:
            dy_col = pat_conv.generate_a(dy_filling)

        return dy_col

    def generate_b(self, kernels):
        """
        generate w_col tensor for mad

        Parameters
        ----------
        kernels : weight tensor of fractal shape before transformation in ddr

        Returns
        ----------
        w_col: w tensor of fractal shape after transformation in L0B
        """
        _, w_k1, kernel_cout1, kernel_cout0, w_k0 = \
            list(i.value for i in kernels.shape)
        kernel_h, kernel_w, kernel_d = \
            self._kernel_h, self._kernel_w, self._kernel_d
        if w_k1 % (kernel_h * kernel_w) != 0:
            raise RuntimeError(
                "could not be divided %d, %d*%d " % (w_k1, kernel_h, kernel_w))
        kernel_cin1 = w_k1 // (kernel_w * kernel_h)
        shape_w_l0b = (kernel_d,
                       kernel_cout1 * kernel_h * kernel_w,
                       kernel_cin1,
                       w_k0,
                       kernel_cout0)
        w_col = tvm.compute(
            shape_w_l0b,
            lambda w_d, w_k1_idx, kernel_cin1_idx, w_k0_idx, kernel_cout0_idx:
            kernels[w_d,
                    kernel_cin1_idx * kernel_h * kernel_w + \
                    (kernel_h * kernel_w - 1 -
                     w_k1_idx % (kernel_h * kernel_w)), \
                    w_k1_idx // (kernel_h * kernel_w), \
                    kernel_cout0_idx, w_k0_idx],
            name="w_col",
            tag="inverse_trans_dma")
        return w_col

    def generate_c(self, dy_col, w_col):  # pylint: disable=W0221,R0914
        """
        generate dx_ddr

        Parameters
        ----------
        dy_col : dE/dY tensor of fractal shape in L0A

        w_col : w tensor of fractal shape after transformation in L0B

        Returns
        ----------
        dx_ddr: dx tensor in ddr
        """
        if is_lhisi_version():
            dx_col = super(DeConvPattern,
                           self).generate_c(dy_col, w_col, c_type="float16")
        else:
            dx_col = super(DeConvPattern,
                           self).generate_c(dy_col, w_col, c_type="float32")

        # mad dx shape
        dx_batch, dx_deep, dx_c1, dx_hw, dx_c0 =\
            list(i.value for i in dx_col.shape)
        # real dx shape
        _, _, dx_cin1, dx_h, dx_w, dx_cin0 = self.output_shape
        out_shape = (dx_batch, dx_deep, dx_cin1, dx_h * dx_w, dx_cin0)

        # float32->float16
        dx_ub = tvm.compute(
            (dx_batch, dx_deep, dx_c1, dx_hw, dx_c0),
            lambda dx_batch_idx, dx_deep_idx,
                   dx_cin1_idx, dx_hw_idx, dx_cin0_idx:
            dx_col[dx_batch_idx, dx_deep_idx, dx_cin1_idx,
                   dx_hw_idx, dx_cin0_idx].astype("float16"),
            name="c_ub")

        # sd>kd,add0
        if self._stride_d >= self._kernel_d:
            dx_ub = tvm.compute(
                (dx_batch, dx_deep, dx_c1, dx_hw, dx_c0),
                lambda dx_batch_idx, dx_deep_idx,
                       dx_cin1_idx, dx_hw_idx, dx_cin0_idx:
                tvm.select(
                    tvm.any(dx_deep_idx + self._pad_head -
                            (dx_deep_idx + self._pad_head) //
                            self._stride_d * self._stride_d -
                            self._kernel_d >= 0,
                        dx_deep_idx + \
                        self._pad_head >= self._stride_d * self.dy_d),
                    tvm.const(0, dtype="float16"),
                    dx_ub[dx_batch_idx, dx_deep_idx,
                          dx_cin1_idx, dx_hw_idx, dx_cin0_idx]),
                name="dx_filing_zero"
            )
        else:
            dx_ub = tvm.compute(
                (dx_batch, dx_deep, dx_c1, dx_hw, dx_c0),
                lambda dx_batch_idx, dx_deep_idx,
                       dx_cin1_idx, dx_hw_idx, dx_cin0_idx:
                tvm.select(
                    dx_deep_idx + self._pad_head >=
                    self._stride_d * (self.dy_d - 1) + self._kernel_d,
                    tvm.const(0, dtype="float16"),
                    dx_ub[dx_batch_idx, dx_deep_idx,
                          dx_cin1_idx, dx_hw_idx, dx_cin0_idx]),
                name="dx_filing_zero"
            )

        dx_ddr = tvm.compute(
            out_shape,
            lambda dx_batch_idx, dx_deep_idx,
                   dx_cin1_idx, dx_hw_idx, dx_cin0_idx:
            dx_ub[dx_batch_idx, dx_deep_idx,
                  dx_cin1_idx, dx_hw_idx, dx_cin0_idx],
            name="c_ddr",
            tag="conv3d_backprop_input",
            attrs={"output_shape": self.output_shape,
                   "stride_d": self._stride_d,
                   'depth_pad': (self._pad_head, self._pad_tail),
                   'kernels': (self._kernel_d,
                               self._kernel_h, self._kernel_w)})
        return dx_ddr
