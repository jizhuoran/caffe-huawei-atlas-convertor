"""
Copyright 2018 Huawei Technologies Co., Ltd

conv2d_backprop_input_opti_compute
"""
from te import tvm
from te.platform import CUBE_MKN
from te.platform import intrinsic_check_support
from .cube_util import CubeDslPattern

# broadcast should be 16
BRC_STANDARD_BLOCK_SIZE = 16


class DeConvKernelSize1Pattern(CubeDslPattern):  # pylint:disable=R0902
    """
    class of convolution back propagation for kernelsize1 pattern

    Parameters
    ----------
    kernel_sizes : shape of weight, [N, C, H, W]

    strides : list of strides, [strideh, stridew]

    pad: list of padding, [pad_up, pad_down, pad_left, pad_right]

    output_shape : shape of dE/dX, [N, C, H, W]

    Returns
    -------
    deconv_pattern_instance : instance of deconv pattern
    """

    def __init__(self, kernel_size, strides, pad, output_shape):
        super(DeConvKernelSize1Pattern, self).__init__()
        _, _, kernel_h, kernel_w = kernel_size
        stride_h, stride_w = strides
        if not (kernel_h == 1 and kernel_w == 1):
            raise RuntimeError("not match pattern kernel HW:[{},{}]"
                               .format(kernel_h, kernel_w))
        if not (pad[0] == 0 and pad[1] == 0 and pad[2] == 0 and pad[3] == 0):
            raise RuntimeError("not match pattern pad[{},{},{},{}]".format(
                pad[0], pad[1], pad[2], pad[3]))
        if not (kernel_h <= stride_h and kernel_w <= stride_w):
            raise RuntimeError("not match stride HW[{}, {}],kernel HW[{}, {}]"
                               .format(stride_h, stride_w, kernel_h, kernel_w))
        self._m0 = CUBE_MKN["float16"]["mac"][0]
        self._output_shape = output_shape
        self._stride_h, self._stride_w = strides
        _, _, self._kernel_h, self._kernel_w = kernel_size
        self._img_h, self._img_w = [0, 0]

    def _dilate_tensor(self, raw_tensor,  # pylint:disable=R0913,R0914
                       out_shape_h, out_shape_w, dilate_h, dilate_w):
        (shape_n, shape_c1, _, shape_c0) = raw_tensor.shape
        new_h = out_shape_h  # dilate_h * (shape_h - 1) + 1
        new_w = out_shape_w  # dilate_w * (shape_w - 1) + 1
        dilate_shape = (shape_n, shape_c1, new_h * new_w, shape_c0)
        dx_zero = tvm.compute(
            dilate_shape,
            lambda *indice: tvm.convert(0).astype(raw_tensor.dtype),
            name=raw_tensor.name + "_dx_zero",
            tag="init_zero")

        dilate_tensor = tvm.compute(
            dilate_shape,
            lambda n, c1, hw, c0:
            tvm.select(tvm.all((hw // new_w) % dilate_h == 0,
                               (hw % new_w) % dilate_w == 0),
                       raw_tensor[n,
                                  c1,
                                  ((hw // new_w) // dilate_h)
                                  * self._img_w + (hw % new_w // dilate_w),
                                  c0]+dx_zero[n, c1, hw, c0],
                       dx_zero[n, c1, hw, c0]),
            name=raw_tensor.name + "_dilation",
            tag="conv2d_backprop_input_opti",
            attrs={'dilate': [dilate_h, dilate_w],
                   'out_hw': [out_shape_h, out_shape_w],
                   "img_w": self._img_w})
        return dilate_tensor

    def generate_a(self, dedy):
        """
        generate dedy_col_fractal tensor for mad

        Parameters
        ----------
        dedy: 5D dE/dY tensor in ddr

        Returns
        ----------
        dedy_col_fractal: dE/dY tensor of fractal shape in L0A
        """

        def _int_ceil_div(a_factor, b_factor):
            if b_factor == 0:
                raise RuntimeError("division by zero")
            return (a_factor + b_factor - 1) // b_factor

        batch_dim, co1_dim, ho_dim, wo_dim, co0_dim = list(
            i.value for i in dedy.shape)

        self._img_h = ho_dim
        self._img_w = wo_dim

        hw_dim = _int_ceil_div(wo_dim * ho_dim, self._m0)
        real_hwdim = ho_dim * wo_dim

        shape = (batch_dim, co1_dim, real_hwdim, co0_dim)
        dedy_col = tvm.compute(shape, lambda n, co1, m1, co0: dedy(
            n, co1, tvm.floordiv(m1, wo_dim), tvm.floormod(m1, wo_dim), co0),
                               name=dedy.name + "_col")

        shape = (batch_dim, hw_dim, co1_dim, self._m0, co0_dim)
        dedy_col = tvm.compute(shape, lambda n, m1, co1, m0, co0: dedy_col(
            n, co1, m0 + self._m0 * m1, co0), name=dedy.name + "_col_fractal")

        return dedy_col

    def generate_b(self, kernels):
        """
        generate b_l0b tensor for mad

        Parameters
        ----------
        kernels : weight tensor of fractal shape before transformation in ddr

        Returns
        ----------
        b_l0b: w tensor of fractal shape after transformation in L0B
        """
        if kernels.dtype == "int8":
            def _bl1_elem_func(*index):
                return kernels(*index)

            def _bl0_elem_func(*index):
                return b_l1(*index)
            b_l1 = tvm.compute(kernels.shape, _bl1_elem_func,
                               name=kernels.name + "_B_l1")
            b_l0b = tvm.compute(kernels.shape, _bl0_elem_func,
                                name=kernels.name + "_B_l0b",
                                attrs={"kernel_hw": (self._kernel_h,
                                                     self._kernel_w)})
        else:
            k1_dim, co1_dim, co0_dim, k0_dim \
                = list(i.value for i in kernels.shape)
            shape = (k1_dim, co1_dim, co0_dim, k0_dim)

            def _bl1_elem_func(*index):
                return kernels(*index)
            b_l1 = tvm.compute(shape, _bl1_elem_func,
                               name=kernels.name + "_B_l1")
            shape = (co1_dim, k1_dim, k0_dim, co0_dim)
            b_l0b = tvm.compute(shape, lambda co1, k1, k0, co0: b_l1[
                k1, co1, co0, k0], name=kernels.name + "_B_l0b")
        return b_l0b

    def generate_c(self,  # pylint:disable=R0914
                   tensor_a, tensor_b, tensor_bias=None, c_type=None):
        """
        generate img_c

        Parameters
        ----------
        dedy_col_fractal : dE/dY tensor of fractal shape in L0A

        b_l0b : w tensor of fractal shape after transformation in L0B

        Returns
        ----------
        img_c: dx tensor in ddr
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

        def _inner_generate_mmad(matrix_a, matrix_b):  # pylint:disable=R0914
            n_dim, hw_dim, k1_dim, m0_dim, k0_dim = list(
                i.value for i in matrix_a.shape)
            bk1_dim, c01_dim, co0_dim, bk0_dim = list(
                i.value for i in matrix_b.shape)
            if bk1_dim != k1_dim or bk0_dim != k0_dim:
                raise RuntimeError("invaild shape", bk1_dim, k1_dim, bk0_dim,
                                   k0_dim)

            shape_c = (n_dim, c01_dim, hw_dim * m0_dim, co0_dim)
            k0_axis = tvm.reduce_axis([0, k0_dim], name='k0')
            k1_axis = tvm.reduce_axis([0, k1_dim], name='k1')
            if matrix_a.dtype == "int8" and matrix_b.dtype == "int8":
                c_type = "int32"
                mode = "s8"
            elif intrinsic_check_support("Intrinsic_mmad", "f162f32"):
                c_type = "float32"
                mode = "f162f32"
            else:
                c_type = "float16"
                mode = "f162f16"

            mmad = tvm.compute(
                shape_c,
                lambda n, co1, m, co0:
                tvm.sum(
                    (matrix_a[n, m // self._m0, k1_axis,
                              m % self._m0, k0_axis]
                     * matrix_b[k1_axis, co1, co0, k0_axis]).astype(c_type),
                    axis=[k1_axis, k0_axis]),
                name="C",
                attrs={'mode': mode})
            return mmad

        def _add_bias_in_l0c(mmad, tensor_bias):
            if tensor_bias is not None and tensor_bias.dtype == "int32" and \
                    self._stride_h == 1 and self._stride_w == 1:
                shape_c = mmad.shape
                bias_ub_brc_shape = list(shape_c)
                bias_ub_brc_shape[2] = bias_ub_brc_shape[2] // \
                                       BRC_STANDARD_BLOCK_SIZE
                bias_ub_brc = tvm.compute(
                    bias_ub_brc_shape,
                    lambda *indices:
                    tensor_bias(
                        indices[1] * CUBE_MKN[tensor_bias.dtype]['mac'][2] +
                        indices[3]
                    ),
                    name="bias_ub_brc",
                )
                bias_l0c = tvm.compute(
                    shape_c,
                    lambda i, j, k, l: bias_ub_brc(
                        i, j, k // BRC_STANDARD_BLOCK_SIZE, l),
                    name="bias_l0c",
                )
                mmad = tvm.compute(
                    shape_c,
                    lambda *indices: bias_l0c(*indices) + mmad(*indices),
                    name="c_add_bias",
                )
            return mmad

        res_c = _inner_generate_mmad(tensor_a, tensor_b)
        res_c = _add_bias_in_l0c(res_c, tensor_bias)

        n_dim, c1_dim, h_dim, w_dim, c0_dim = self._output_shape
        batch_dim, k1_dim, m1_dim, k0_dim = list(i.value for i in res_c.shape)
        if not (n_dim == batch_dim and c0_dim == k0_dim and c1_dim == k1_dim):
            raise RuntimeError("ouput shape inlegal")
        conv_shape = [n_dim, c1_dim, m1_dim, c0_dim]
        res_c_dtype = "float16"
        if tensor_a.dtype == "int8" and tensor_b.dtype == "int8":
            res_c_dtype = "int32"
        res_cub = tvm.compute(
            conv_shape,
            lambda *indice: res_c[indice].astype(res_c_dtype),
            name='CUB')

        if self._stride_h > 1 or self._stride_w > 1:
            res_cub = self._dilate_tensor(res_cub, h_dim, w_dim,
                                          self._stride_h, self._stride_w)

        if tensor_bias is not None and \
                (tensor_bias.dtype == "float16" or \
                 (self._stride_h > 1 or self._stride_w > 1)):
            res_cub = _add_bias_in_ub(res_cub, tensor_bias)

        output_shape = [n_dim, c1_dim, h_dim * w_dim, c0_dim]
        img_c = tvm.compute(output_shape,
                            lambda n, c1, hw, c0: res_cub(n, c1, hw, c0),
                            tag="conv2d_backprop_input_opti",
                            name=res_cub.name + "_img",
                            attrs={"hw_dim": h_dim * w_dim})
        return img_c
