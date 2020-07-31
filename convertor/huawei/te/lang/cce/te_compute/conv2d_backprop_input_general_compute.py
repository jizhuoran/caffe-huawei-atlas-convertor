"""
Copyright (C) 2016. Huawei Technologies Co., Ltd.

conv2d backprop input general compute.
"""

#!/usr/bin/python
# -*- coding: UTF-8 -*-
from te import tvm
from te.platform import CUBE_MKN
from te.lang.cce.te_compute.cube_util import CubeDslPattern
from te.lang.cce.te_compute.cube_util import ConvDslPattern
from te.platform import intrinsic_check_support


class DeConvPattern(CubeDslPattern):  # pylint: disable=R0902
    """
    class of convolution back propagation

    Parameters
    ----------
    kernel_sizes : shape of weight, [N, C, H, W]

    strides : list of strides, [strideh, stridew]

    pad: list of padding, [pad_up, pad_down, pad_left, pad_right]

    output_shape : shape of dE/dX, [N, C, H, W]

    dilations: list of dilations, [dilate_n, dilate_c, dilate_h, dilate_w]

    Returns
    -------
    deconv_pattern_instance : instance of deconv pattern
    """

    def __init__(self,  # pylint: disable=R0913
                 kernel_sizes, strides, pad, output_shape, dilations):
        super(DeConvPattern, self).__init__()
        _, _, kernel_h, kernel_w = kernel_sizes
        self._kernel_h = kernel_h
        self._kernel_w = kernel_w
        self._stride_h, self._stride_w = strides
        self._pad_up, self._pad_down, self._pad_left, self._pad_right = pad
        self._output_shape = output_shape
        _, _, self._dilate_h, self._dilate_w = dilations
        self.m_0, _, _ = CUBE_MKN["float16"]['mac']

    def generate_a(self, dy_ddr):  # pylint: disable=R0914
        """
        generate dy_col tensor for mad

        Parameters
        ----------
        dy_ddr: 5D dE/dY tensor in ddr

        Returns
        ----------
        dy_col: dE/dY tensor of fractal shape in L0A
        """
        dy_batch, kernel_cout1, dy_h, dy_w, kernel_cout0 = \
                            list(i.value for i in dy_ddr.shape)
        stride_h, stride_w = self._stride_h, self._stride_w
        shape_dy_filling = (dy_batch, kernel_cout1, dy_h*stride_h,
                            dy_w*stride_w, kernel_cout0)
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
                       kernel_cout1_idx,
                       ho_idx,
                       wo_idx,
                       kernel_cout0_idx:
                tvm.select(tvm.all(ho_idx % stride_h == 0,
                                   wo_idx % stride_w == 0),
                           dy_ddr[batch_idx,
                                  kernel_cout1_idx,
                                  ho_idx // stride_h,
                                  wo_idx // stride_w,
                                  kernel_cout0_idx],
                           dy_zero[batch_idx,
                                   kernel_cout1_idx,
                                   ho_idx,
                                   wo_idx,
                                   kernel_cout0_idx]),
                name="dy_filling",
                tag="stride_filling",
                attrs={"stride_expand": (self._stride_h, self._stride_w)})

        kernel_h, kernel_w = self._kernel_h, self._kernel_w
        dilate_h, dilate_w = self._dilate_h, self._dilate_w
        new_stride = (1, 1)
        new_hw = (dy_h*stride_h, dy_w*stride_w)

        new_pad_before = ((kernel_h - 1) * dilate_h - self._pad_up,
                          (kernel_w - 1) * dilate_w - self._pad_left)
        _, _, dx_h, dx_w, _ = self._output_shape
        new_pad_after = tuple(i - o - pb + (k - 1) * d for i, o, pb, k, d in
                              zip((dx_h, dx_w),
                                  new_hw,
                                  new_pad_before,
                                  (kernel_h, kernel_w),
                                  (dilate_h, dilate_w)))
        pad_down_after, pad_right_after = new_pad_after

        # stride > 1 ub->l1 may cut
        if stride_h > 1 or stride_w > 1:
            if pad_down_after < 0 or pad_right_after < 0:
                shape_down_modify = (pad_down_after - abs(pad_down_after)) // 2
                shape_right_modify = (pad_right_after - abs(pad_right_after))\
                                        // 2
                shape_dy_filling_cut = (dy_batch, kernel_cout1,
                                        dy_h*stride_h + shape_down_modify,
                                        dy_w*stride_w + shape_right_modify,
                                        kernel_cout0)

                # cut dy_filling
                dy_filling_l1 = tvm.compute(
                    shape_dy_filling_cut,
                    lambda batch_idx, kernel_cout1_idx,
                           ho_idx, wo_idx, kernel_cout0_idx:
                    dy_filling[batch_idx, kernel_cout1_idx,
                               ho_idx, wo_idx, kernel_cout0_idx],
                    name="dy_l1_cut",
                    tag="dy_l1_cut")

                pad_down_after = (pad_down_after + abs(pad_down_after)) // 2
                pad_right_after = (pad_right_after + abs(pad_right_after)) // 2
            else:
                dy_filling_l1 = tvm.compute(
                    shape_dy_filling,
                    lambda batch_idx, kernel_cout1_idx,
                           ho_idx, wo_idx, kernel_cout0_idx:
                    dy_filling[batch_idx, kernel_cout1_idx,
                               ho_idx, wo_idx, kernel_cout0_idx],
                    name="dy_l1",
                    tag="dy_l1")

        new_pad = ((kernel_h - 1) * dilate_h - self._pad_up,
                   pad_down_after,
                   (kernel_w - 1) * dilate_w - self._pad_left,
                   pad_right_after)
        pat_conv = ConvDslPattern(kernel_h, kernel_w, new_stride, new_pad,
                                  (dilate_h, dilate_w))

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
        kernels: weight tensor of fractal shape before transformation in ddr

        Returns
        ----------
        w_col: w tensor of fractal shape after transformation in L0B
        """
        if kernels.dtype == "int8":
            def _kernel_elem_func(*index):
                return kernels(*index)
            w_col = tvm.compute(kernels.shape, _kernel_elem_func,
                                name="w_col", tag="inverse_trans_dma")
        else:
            w_k1, kernel_cout1, kernel_cout0, w_k0 = \
                list(i.value for i in kernels.shape)
            kernel_h, kernel_w = self._kernel_h, self._kernel_w
            if w_k1 % (kernel_h*kernel_w) != 0:
                raise RuntimeError("w_k1 could not "
                                   "be divided by kernel_h*kernel_w ")
            kernel_cin1 = w_k1 / (kernel_w*kernel_h)
            shape_w_l0b = (kernel_cout1*kernel_h*kernel_w,
                           kernel_cin1,
                           w_k0,
                           kernel_cout0)
            w_col = tvm.compute(
                shape_w_l0b,
                lambda w_k1_idx, kernel_cin1_idx, w_k0_idx, kernel_cout0_idx:
                kernels[kernel_cin1_idx*kernel_h*kernel_w + \
                        (kernel_h*kernel_w - 1 - w_k1_idx
                         % (kernel_h*kernel_w)), \
                        w_k1_idx // (kernel_h*kernel_w), \
                        kernel_cout0_idx, w_k0_idx],
                name="w_col",
                tag="inverse_trans_dma")
        return w_col

    def generate_c(self,  # pylint: disable=R0914
                   tensor_a, tensor_b, tensor_bias=None, c_type=None):
        """
        generate dx_ddr

        Parameters
        ----------
        tensor_a : dE/dY tensor of fractal shape in L0A

        tensor_b : w tensor of fractal shape after transformation in L0B

        tensor_bias : same as that in Class->CubeDslPattern

        c_type : same as that in Class->CubeDslPattern

        Returns
        ----------
        dx_ddr: dx tensor in ddr
        """

        def _add_bias_in_ub(in_tensor0, in_tensor1):
            c_add_vector = tvm.compute(
                in_tensor0.shape,
                lambda *indice:
                in_tensor0(*indice) +
                in_tensor1(
                    indice[1]*CUBE_MKN[in_tensor0.dtype]['mac'][2] + indice[3]
                ),
                name="bias_add_vector",
            )
            return c_add_vector

        dy_col = tensor_a
        w_col = tensor_b

        res_c_type = "float32"
        if not intrinsic_check_support("Intrinsic_mmad", "f162f32"):
            res_c_type = "float16"
        if w_col.dtype == "int8" and dy_col.dtype == "int8":
            res_c_type = "int32"

        if tensor_bias is not None and tensor_bias.dtype == "int32":
            bias = tensor_bias
        else:
            bias = None
        dx_col = super(DeConvPattern, self).generate_c(dy_col,
                                                       w_col,
                                                       c_type=res_c_type,
                                                       tensor_bias=bias)
        # mad dx shape
        dx_batch, dx_c1, dx_hw, dx_c0 = list(i.value for i in dx_col.shape)

        # real dx shape
        _, dx_cin1, dx_h, dx_w, dx_cin0 = self._output_shape
        out_shape = (dx_batch, dx_cin1, dx_h*dx_w, dx_cin0)

        # float32->float16
        dx_ub_type = "float16"
        if w_col.dtype == "int8" and dy_col.dtype == "int8":
            dx_ub_type = "int32"

        dx_ub = tvm.compute(
            (dx_batch, dx_c1, dx_hw, dx_c0),
            lambda dx_batch_idx, dx_cin1_idx, dx_hw_idx, dx_cin0_idx:
            dx_col[dx_batch_idx, dx_cin1_idx, dx_hw_idx, dx_cin0_idx]\
                .astype(dx_ub_type), name="c_ub")

        if tensor_bias is not None and tensor_bias.dtype == "float16":
            dx_ub = _add_bias_in_ub(dx_ub, tensor_bias)

        dx_ddr = tvm.compute(
            out_shape,
            lambda dx_batch_idx, dx_cin1_idx, dx_hw_idx, dx_cin0_idx:
            dx_ub[dx_batch_idx, dx_cin1_idx, dx_hw_idx, dx_cin0_idx],
            name="c_ddr",
            tag="conv2d_backprop_input",
            attrs={"output_shape": (dx_batch, dx_cin1, dx_h, dx_w, dx_cin0)})
        return dx_ddr
