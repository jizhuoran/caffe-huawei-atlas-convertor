"""
cube util.
"""

#!/usr/bin/python
# -*- coding: UTF-8 -*-
from te import tvm


def im2col_row_major(a_im2col_vm_shape,  # pylint: disable=R0913, E1101
                     tensor_a,
                     kernel_w,
                     padding,
                     stride,
                     compute_dtype,
                     tag=''):
    """
    calculate im2col_row_major tensor
    Parameters
    ----------
    a_im2col_vm_shape : shape of a_im2col_row_major

    tensor_a : feature map

    kernel_w: width of filter

    padding: the padding shape

    stride: the stride value

    compute_dtype: dtype of compute result
    -------
    Returns : a_im2col_row_major tensor
    :param tag:
    """
    def __im2col_row_major_indices(indices,  # pylint: disable=R0913,R0914
                                   tensor_a,
                                   kernel_w,
                                   padding,
                                   stride):
        """
        calculate im2col_row_major tvm lambda function
        Parameters
        ----------
        indices : indices in lambda function

        tensor_a : feature map

        kernel_w: width of filter

        padding: the padding shape

        stride: the stride value

        -------
        Returns : im2col_row_major tensor
        """
        _, _, _, a_height, a_width, _ = tensor_a.shape
        n_index, deep_index, hw_index,\
        c1_index, kh_index, kw_index, c0_index = indices
        stride_h, stride_w = stride
        padding_up, _, padding_left, padding_right = padding
        width_out = (a_width.value + padding_left + padding_right - kernel_w) \
                    // stride_w + 1

        h_index = (hw_index // width_out) * stride_h + kh_index
        w_index = (hw_index % width_out) * stride_w + kw_index

        return tvm.select(tvm.any(h_index < padding_up,
                                  h_index > a_height.value + padding_up - 1,
                                  w_index < padding_left,
                                  w_index > a_width.value + padding_left - 1),
                          tvm.const(0.0, compute_dtype),
                          tensor_a(n_index,
                                   deep_index,
                                   c1_index,
                                   h_index - padding_up,
                                   w_index - padding_left,
                                   c0_index))

    return tvm.compute(a_im2col_vm_shape,
                       lambda *indices: __im2col_row_major_indices(
                           indices, tensor_a, kernel_w, padding, stride),
                       name='im2col_row_major',
                       tag=tag + 'im2col_row_major',
                       attrs={'padding': padding})


def im2col_fractal(a_im2col_shape, tensor_a_row_major, tag=''):
    """
    calculate im2col_fractal tensor
    Parameters
    ----------
    a_im2col_shape : shape of a_im2col

    tensor_a_row_major : feature map after row major

    config: the config of cube

    compute_dtype: dtype of compute result
    -------
    Returns : a_im2col_fractal tensor
    """
    def __im2col_fractal_indices(indices,  # pylint: disable=R0914
                                 tensor_a_row_major):
        """
        calculate im2col_fractal tvm lambda function
        Parameters
        ----------
        indices : indices in lambda function

        a : feature map

        -------
        Returns : im2col_fractal tvm lambda function
        """
        _, _, _, _, a_col_m0, a_col_k0 = a_im2col_shape
        _, _, a_row_major_hw, _, kernel_h, kernel_w, _ =\
            tensor_a_row_major.shape
        n_index, deep_index, m1_index, k1_index, m0_index, k0_index = indices

        hw_index = m1_index*a_col_m0 + m0_index

        c1_index = (((k1_index*a_col_k0 + k0_index) // a_col_k0) //
                    kernel_w.value) // kernel_h.value

        kh_index = (((k1_index*a_col_k0 + k0_index) // a_col_k0) //
                    kernel_w.value) % kernel_h.value

        kw_index = ((k1_index*a_col_k0
                     + k0_index) // a_col_k0) % kernel_w.value

        c0_index = (k1_index*a_col_k0 + k0_index) % a_col_k0

        return tvm.select(tvm.any(hw_index < 0, hw_index >
                                  a_row_major_hw.value - 1),
                          tvm.const(0.0, tensor_a_row_major.dtype),
                          tensor_a_row_major(n_index,
                                             deep_index,
                                             hw_index,
                                             c1_index,
                                             kh_index,
                                             kw_index,
                                             c0_index))

    return tvm.compute(a_im2col_shape,
                       lambda *indices:
                       __im2col_fractal_indices(indices, tensor_a_row_major),
                       name='im2col_fractal',
                       tag=tag+'im2col_fractal')


class CubeDslPattern:
    """
    class of cube mmad calculation

    Parameters
    ----------
    None

    Returns
    -------
    cube_pattern_instance : instance of cube mmad pattern
    """

    type_c_map = dict()

    def __init__(self):
        pass

    @staticmethod
    def get_type_c(type_a, type_b, type_bias=None):
        """
        get the data type of mad result tensor

        Parameters
        ----------
        type_a : data type of tensor a

        type_b : data type of tensor b

        type_bias : data type of bias

        Returns
        ----------
        type_c : data type of tensor c
        """
        cal_hash = lambda tp_a, tp_b, tp_bias: \
            hash(str(tp_a) + str(tp_b) + str(tp_bias))

        if CubeDslPattern.type_c_map == {}:
            CubeDslPattern.type_c_map[
                cal_hash("uint8", "uint8", None)] = "int32"
            CubeDslPattern.type_c_map[cal_hash("int8", "int8", None)] = "int32"
            CubeDslPattern.type_c_map[
                cal_hash("float16", "float16", None)] = "float16"
            CubeDslPattern.type_c_map[
                cal_hash("float16", "float16", "float32")] = "float32"
            CubeDslPattern.type_c_map[
                cal_hash("float16", "float16", "float16")] = "float16"

        type_c_key = cal_hash(type_a, type_b, type_bias)
        type_c = CubeDslPattern.type_c_map.get(type_c_key)

        return type_c

    def generate_c(self, # pylint: disable=R0914
                   tensor_a, tensor_b, c_type=None):
        """
        calculate the mad result tensor

        Parameters
        ----------
        tensor_a : tensor a

        tensor_b : tensor b

        tensor_bias : bias tensor

        Returns
        ----------
        tensor_c : mad result tensor
        """
        def __mad_condition(indices, # pylint: disable=R0914,R0913
                            axis_kd, axis_k1, axis_k0, tensor_a, tensor_b):
            n_index, deep_index, co1_index, m_index, co0_index = indices
            tensor_c = tvm.select(
                tvm.all(
                    (deep_index - axis_kd + pad_head) >= 0,
                    (deep_index - axis_kd + pad_head) % stride_d == 0,
                    (deep_index - axis_kd
                     + pad_head) // stride_d < tensor_a.shape[1]),
                tensor_a(n_index,
                         (deep_index - axis_kd + pad_head) // stride_d,
                         m_index // a_m0, axis_k1, m_index % a_m0,
                         axis_k0).astype(type_c) *
                tensor_b(axis_kd, axis_k1, co1_index, co0_index,
                         axis_k0).astype(type_c),
                tvm.const(0.0, type_c))
            return tensor_c

        def __mad_condition_stride1(indices, # pylint: disable=R0914,R0913
                                    axis_kd, axis_k1,
                                    axis_k0, tensor_a, tensor_b):
            n_index, deep_index, co1_index, m_index, co0_index = indices
            tensor_c = tvm.select(
                tvm.all((deep_index - axis_kd + pad_head) >= 0,
                        (deep_index - axis_kd + pad_head) < tensor_a.shape[1]),
                tensor_a(n_index, (deep_index - axis_kd + pad_head),
                         m_index // a_m0, axis_k1, m_index % a_m0,
                         axis_k0).astype(type_c) *
                tensor_b(axis_kd, axis_k1, co1_index, co0_index,
                         axis_k0).astype(type_c),
                tvm.const(0.0, type_c))
            return tensor_c

        def __conv3d_backprop_input_mad(indices, tensor_a, tensor_b):
            tensor_c = tvm.sum(__mad_condition(indices, axis_kd, axis_k1,
                                               axis_k0, tensor_a, tensor_b),
                               axis=[axis_kd, axis_k1, axis_k0])
            return tensor_c

        def __conv3d_backprop_input_mad_stride1(indices, tensor_a, tensor_b):
            tensor_c = tvm.sum(__mad_condition_stride1(indices,
                                                       axis_kd, axis_k1,
                                                       axis_k0, tensor_a,
                                                       tensor_b),
                               axis=[axis_kd, axis_k1, axis_k0])
            return tensor_c
        a_batch, a_deep, a_m1, a_k1, a_m0, a_k0 = \
            list(i.value for i in tensor_a.shape)
        axis_k0 = tvm.reduce_axis([0, a_k0], name='axis_k0')
        axis_k1 = tvm.reduce_axis([0, a_k1], name='axis_k1')
        b_kd, _, b_n1, b_n0, _ = list(i.value for i in tensor_b.shape)
        axis_kd = tvm.reduce_axis([0, b_kd], name='axis_kd')
        pad_head, pad_tail = \
            self._pad_head, self._pad_tail # pylint: disable=E1101
        stride_d = self._stride_d # pylint: disable=E1101
        output_depth = self.output_shape[1] # pylint: disable=E1101
        kernel_d = self._kernel_d # pylint: disable=E1101
        shape_c = (a_batch, output_depth, b_n1, a_m1*a_m0, b_n0)
        type_c = c_type if c_type is not None else CubeDslPattern.get_type_c(
            tensor_a.dtype, tensor_b.dtype)
        if stride_d == kernel_d and (output_depth + pad_head
                                     + pad_tail) == a_deep*stride_d:
            tensor_c = tvm.compute(
                shape_c,
                lambda n_index, deep_index, co1_index, m_index, co0_index:
                tvm.sum(
                    (tensor_a(n_index, deep_index//stride_d, m_index//a_m0,
                              axis_k1, m_index%a_m0, axis_k0) *
                     tensor_b(deep_index%stride_d, axis_k1, co1_index,
                              co0_index, axis_k0)).astype(type_c),
                    axis=[axis_k1, axis_k0]),
                name="C",
                tag="mad")
        elif stride_d == 1:
            tensor_c = tvm.compute(
                shape_c,
                lambda *indices: __conv3d_backprop_input_mad_stride1(
                    indices, tensor_a, tensor_b),
                name="C",
                tag="mad")
        else:
            tensor_c = tvm.compute(
                shape_c,
                lambda *indices: __conv3d_backprop_input_mad(
                    indices, tensor_a, tensor_b),
                name="C",
                tag="mad")
        return tensor_c

class ConvDslPattern(CubeDslPattern): # pylint: disable=R0902
    """
    class of convolution

    Parameters
    ----------
    kernel_h: height of filter

    kernel_w: width of filter

    stride : list of strides, [strideh, stridew]

    pad: list of padding, [pad_up, pad_down, pad_left, pad_right]

    Returns
    -------
    conv_pattern_instance : instance of conv pattern
    """

    def __init__(self, kernel_h, kernel_w, stride, pad):
        super(ConvDslPattern, self).__init__()
        self._kernel_h = kernel_h
        self._kernel_w = kernel_w
        self._stride_d, self._stride_h, self._stride_w = stride
        self._pad_head, self._pad_tail, self._pad_up, self._pad_down,\
        self._pad_left, self._pad_right = pad
        self._m0 = 16

    def cal_howo(self, height_in, width_in):
        """
        calculate the height and width of convolution output tensor

        Parameters
        ----------
        height_in : height of input tensor

        width_in : width of input tensor

        Returns
        ----------
        height_out : height of output tensor

        width_out : width of output tensor
        """
        new_hw = [height_in, width_in]
        kernel_h, kernel_w = self._kernel_h, self._kernel_w
        new_pad_before = (self._pad_up, self._pad_left)
        new_pad_after = (self._pad_down, self._pad_right)
        stride = [self._stride_h, self._stride_w]

        height_out, width_out = list(
            ((i + p_before + p_after - kernel) // s + 1)
            for i, p_before, p_after, kernel, s in
            zip(new_hw,
                new_pad_before,
                new_pad_after,
                [kernel_h, kernel_w],
                stride))
        return height_out, width_out

    def generate_a(self, feature_map): # pylint: disable=R0914
        """
        calculate im2col_fractal tensor

        Parameters
        ----------
        feature_map : feature map tensor in the shape of NC1HWC0

        Returns
        -------
        a_col : a_im2col_fractal tensor
        """
        a_batch, a_deep, a_c1, a_h, a_w, a_c0 =\
            list(i.value for i in feature_map.shape)
        kernel_h, kernel_w = self._kernel_h, self._kernel_w

        new_pad = [self._pad_up, self._pad_down,
                   self._pad_left, self._pad_right]
        stride = [self._stride_h, self._stride_w]

        height_out, width_out = self.cal_howo(a_h, a_w)

        a_im2col_row_major_shape = (a_batch,
                                    a_deep,
                                    height_out * width_out,
                                    a_c1,
                                    kernel_h,
                                    kernel_w,
                                    a_c0)
        a_row_major = im2col_row_major(a_im2col_row_major_shape,
                                       feature_map,
                                       kernel_w,
                                       padding=new_pad,
                                       stride=stride,
                                       compute_dtype=feature_map.dtype)
        a_im2col_fractal_shape = (a_batch,
                                  a_deep,
                                  (height_out*width_out + self._m0 - 1) \
                                  // self._m0,
                                  a_c1 * kernel_h * kernel_w,
                                  self._m0,
                                  a_c0)
        a_col = im2col_fractal(a_im2col_fractal_shape, a_row_major)
        return a_col

    def generate_c(self, tensor_a, tensor_b): # pylint: disable=W0221
        """
        calculate convolution output tensor

        Parameters
        ----------
        tensor_a : a_im2col_fractal tensor
        tensor_b : b_fractal tensor

        Returns
        ----------
        tensor_c: convolution output tensor
        """
        tensor_c = super(ConvDslPattern, self).generate_c(tensor_a, tensor_b)
        row_major = tensor_a.op.input_tensors[0]
        ho_wo = row_major.shape[1].value
        _, _, c_m, _ = list(i.value for i in tensor_c.shape)
        m_0 = self._m0
        m_1 = c_m // m_0
        if not ((m_1 - 1)*m_0) < ho_wo <= c_m:
            raise RuntimeError("HoWo param error!")
        return tensor_c
