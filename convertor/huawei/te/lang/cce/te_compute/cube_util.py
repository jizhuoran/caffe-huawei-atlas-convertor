"""
cube util.
"""

#!/usr/bin/python
# -*- coding: UTF-8 -*-
from te import tvm
from te.platform import CUBE_MKN

# broadcast should be 16
BRC_STANDARD_BLOCK_SIZE = 16


def im2col_row_major(a_im2col_vm_shape,  # pylint: disable=R0913
                     tensor_a,
                     kernel_w,
                     padding,
                     stride,
                     compute_dtype,
                     opti_h_flag=False,
                     tag='',
                     dilation=(1, 1)):
    """
    calculate im2col_row_major tensor
    Parameters
    ----------
    a_im2col_vm_shape : shape of a_im2col_row_major

    tensor_a : feature map

    kernel_w: width of filter

    padding: the padding shape

    stride: the stride value

    dilation: the dilation value

    compute_dtype: dtype of compute result
    -------
    Returns : a_im2col_row_major tensor
    """
    def __im2col_row_major_indices(indices,  # pylint: disable=R0913,R0914
                                   tensor_a,
                                   kernel_w,
                                   padding,
                                   stride,
                                   dilation):
        """
        calculate im2col_row_major tvm lambda function
        Parameters
        ----------
        indices : indices in lambda function

        tensor_a : feature map

        kernel_w: width of filter

        padding: the padding shape

        stride: the stride value

        dilation: the dilation value
        -------
        Returns : im2col_row_major tensor
        """
        _, _, a_height, a_width, _ = tensor_a.shape
        n_index, hw_index, c1_index, kh_index, kw_index, c0_index = indices
        stride_h, stride_w = stride
        if opti_h_flag:
            stride_h = 1
        dilate_h, dilate_w = dilation
        padding_up, _, padding_left, padding_right = padding
        width_out = (a_width.value + padding_left + padding_right \
            - ((kernel_w - 1) * dilate_w + 1)) // (stride_w) + 1

        h_index = (hw_index // width_out) * stride_h + kh_index * dilate_h
        w_index = (hw_index % width_out) * stride_w + kw_index * dilate_w

        return tvm.select(tvm.any(h_index < padding_up, h_index > \
                                    a_height.value + padding_up - 1,
                                  w_index < padding_left, w_index > \
                                    a_width.value + padding_left - 1),
                          tvm.const(0.0, compute_dtype),
                          tensor_a(n_index, \
                                c1_index, \
                                h_index - padding_up, \
                                w_index - padding_left, \
                                c0_index))

    return tvm.compute(a_im2col_vm_shape,
                       lambda *indices: __im2col_row_major_indices( \
                       indices, tensor_a, kernel_w,
                       padding, stride, dilation),
                       name='im2col_row_major',
                       tag=tag + 'im2col_row_major',
                       attrs={'padding': padding, 'dilation': dilation})

def im2col_fractal(a_im2col_shape, tensor_a_row_major):
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
        _, _, _, a_col_m0, a_col_k0 = a_im2col_shape
        _, a_row_major_hw, _, kernel_h, kernel_w, _ = tensor_a_row_major.shape
        n_index, m1_index, k1_index, m0_index, k0_index = indices

        hw_index = m1_index * a_col_m0 + m0_index

        c1_index = (((k1_index * a_col_k0 + k0_index) // a_col_k0) //
                    kernel_w.value) // kernel_h.value

        kh_index = (((k1_index * a_col_k0 + k0_index) // a_col_k0) //
                    kernel_w.value) % kernel_h.value

        kw_index = ((k1_index*a_col_k0 + k0_index) // a_col_k0) \
                    % kernel_w.value

        c0_index = (k1_index * a_col_k0 + k0_index) % a_col_k0

        # dtype is compute_dtype
        return tvm.select(tvm.any(hw_index < 0, hw_index > \
                                    a_row_major_hw.value - 1),
                          tvm.const(0.0, tensor_a_row_major.dtype),
                          tensor_a_row_major(n_index, \
                                            hw_index, \
                                            c1_index, \
                                            kh_index, \
                                            kw_index, \
                                            c0_index))

    return tvm.compute(a_im2col_shape,
                       lambda *indices: __im2col_fractal_indices(indices, \
                            tensor_a_row_major),
                       name='im2col_fractal',
                       tag='im2col_fractal')


def im2col_fractal_3d(a_im2col_shape,  # pylint: disable=R0913
                      tensor_a_row_major,
                      fmap_c1,
                      d_out,
                      filter_d,
                      stride_d,
                      cyclebuffer_flag,
                      tag=''):
    """
    calculate 3d im2col_fractal tensor
    Parameters
    ----------
    a_im2col_shape : shape of a_im2col

    tensor_a_row_major : feature map after row major

    fmap_c1 : channel c1

    d_out : output d

    filter_d : kernel_d

    strided : stride d

    cyclebuffer_flag : whether to do  cyclebuffer
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
        _, _, _, a_col_m0, a_col_k0 = a_im2col_shape
        _, a_row_major_hw, _, kernel_h, kernel_w, _ = tensor_a_row_major.shape
        n_index, m1_index, k1_index, m0_index, k0_index = indices

        hw_index = m1_index * a_col_m0 + m0_index

        if cyclebuffer_flag:
            c1_index = (((k1_index * a_col_k0 + k0_index) // a_col_k0) //
                        kernel_w.value) // kernel_h.value
            c1_index = ((c1_index // fmap_c1 + (n_index % d_out) * stride_d) %
                        filter_d) * fmap_c1 + c1_index % fmap_c1
        else:
            c1_index = (((k1_index * a_col_k0 + k0_index) // a_col_k0) //
                        kernel_w.value) // kernel_h.value

        kh_index = (((k1_index * a_col_k0 + k0_index) // a_col_k0) //
                    kernel_w.value) % kernel_h.value

        kw_index = ((k1_index*a_col_k0 + k0_index) // a_col_k0) \
                    % kernel_w.value

        c0_index = (k1_index * a_col_k0 + k0_index) % a_col_k0

        # dtype is compute_dtype
        return tvm.select(tvm.any(hw_index < 0, hw_index > \
                                    a_row_major_hw.value - 1),
                          tvm.const(0.0, tensor_a_row_major.dtype),
                          tensor_a_row_major(n_index, \
                                            hw_index, \
                                            c1_index, \
                                            kh_index, \
                                            kw_index, \
                                            c0_index))

    return tvm.compute(a_im2col_shape,
                       lambda *indices: __im2col_fractal_indices(indices, \
                            tensor_a_row_major),
                       name='im2col_fractal',
                       tag=tag + 'im2col_fractal')


class CubeDslPattern():
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
        self._tensor_c = None

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
            CubeDslPattern.type_c_map[cal_hash("uint8", "uint8",
                                               None)] = "int32"
            CubeDslPattern.type_c_map[cal_hash("int8", "int8", None)] = "int32"
            CubeDslPattern.type_c_map[cal_hash("float16", "float16",
                                               None)] = "float16"
            CubeDslPattern.type_c_map[cal_hash("float16", "float16",
                                               "float32")] = "float32"
            CubeDslPattern.type_c_map[cal_hash("float16", "float16",
                                               "float16")] = "float16"

        type_c_key = cal_hash(type_a, type_b, type_bias)
        type_c = CubeDslPattern.type_c_map.get(type_c_key)

        return type_c

    def generate_c(self,  # pylint: disable=R0914
                   tensor_a, tensor_b, tensor_bias=None, c_type=None):
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
        is_conv3d_backprop_input = False
        a_batch, a_m1, a_k1, a_m0, a_k0 = list(i.value for i in tensor_a.shape)
        axis_k0 = tvm.reduce_axis([0, a_k0], name='axis_k0')
        axis_k1 = tvm.reduce_axis([0, a_k1], name='axis_k1')
        if len(tensor_b.shape) == 5:
            is_conv3d_backprop_input = True
        if is_conv3d_backprop_input:
            _, _, b_n1, b_n0, _ = list(i.value for i in tensor_b.shape)
            shape_c = (a_batch*2, b_n1, a_m1*a_m0, b_n0)
            type_c = c_type \
            if c_type is not None else CubeDslPattern.get_type_c(
                tensor_a.dtype, tensor_b.dtype)
            tensor_c = tvm.compute(\
                shape_c,
                lambda n_index, co1_index, m_index, co0_index: \
                   tvm.sum((tensor_a(n_index // 2, m_index // a_m0, \
                            axis_k1, m_index % a_m0, axis_k0) * \
                            tensor_b(n_index % 2, axis_k1, co1_index, \
                            co0_index, axis_k0)).astype(type_c), \
                            axis=[axis_k1, axis_k0]),
                name="C",
                tag="mad")
        else:
            _, b_n1, b_n0, _ = list(i.value for i in tensor_b.shape)

            shape_c = (a_batch, b_n1, a_m1*a_m0, b_n0)
            type_c = c_type if c_type is not None \
                            else CubeDslPattern.get_type_c(tensor_a.dtype,
                                                           tensor_b.dtype)
            tensor_c = tvm.compute( \
                    shape_c,
                    lambda n_index, co1_index, m_index, co0_index: \
                    tvm.sum((tensor_a(n_index,
                                      m_index // a_m0,
                                      axis_k1,
                                      m_index % a_m0,
                                      axis_k0) * \
                            tensor_b(axis_k1,
                                     co1_index,
                                     co0_index,
                                     axis_k0)).astype(type_c), \
                    axis=[axis_k1, axis_k0]),
                    name="C",
                    tag="mad")
            if tensor_bias is not None:
                bias_ub_brc_shape = list(shape_c)
                bias_ub_brc_shape[2] = bias_ub_brc_shape[2] // \
                                       BRC_STANDARD_BLOCK_SIZE
                bias_ub_brc = tvm.compute(
                    bias_ub_brc_shape,
                    lambda *indices:
                    tensor_bias(
                        indices[1]*CUBE_MKN[tensor_bias.dtype]['mac'][2] +
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
                tensor_c = tvm.compute(
                    shape_c,
                    lambda *indices: bias_l0c(*indices) + tensor_c(*indices),
                    name="c_add_bias",
                )
        self._tensor_c = tensor_c
        return tensor_c


class ConvDslPattern(CubeDslPattern):  # pylint: disable=R0902
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

    def __init__(self,  # pylint: disable=R0913
                 kernel_h, kernel_w, stride, pad, dilations):
        super(ConvDslPattern, self).__init__()
        self._kernel_h = kernel_h
        self._kernel_w = kernel_w
        self._stride_h, self._stride_w = stride
        self._pad_up, self._pad_down, self._pad_left, self._pad_right = pad
        self._dilate_h, self._dilate_w = dilations
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
            ((i + p_before + p_after - (kernel - 1) * d - 1) // s + 1)
            for i, p_before, p_after, kernel, d, s in
            zip(new_hw,
                new_pad_before,
                new_pad_after,
                [kernel_h, kernel_w],
                [self._dilate_h, self._dilate_w],
                stride))

        return height_out, width_out

    def generate_a(self, feature_map):  # pylint: disable=R0914
        """
        calculate im2col_fractal tensor

        Parameters
        ----------
        feature_map : feature map tensor in the shape of NC1HWC0

        Returns
        -------
        a_col : a_im2col_fractal tensor
        """
        a_batch, a_c1, a_h, a_w, a_c0 = list(i.value \
                                             for i in feature_map.shape)
        kernel_h, kernel_w = self._kernel_h, self._kernel_w

        new_pad = [
            self._pad_up, self._pad_down, self._pad_left, self._pad_right
        ]
        stride = [self._stride_h, self._stride_w]

        height_out, width_out = self.cal_howo(a_h, a_w)

        a_im2col_row_major_shape = (a_batch, height_out * width_out, a_c1,
                                    kernel_h, kernel_w, a_c0)
        a_row_major = im2col_row_major(a_im2col_row_major_shape, \
                                    feature_map, \
                                    kernel_w, \
                                    padding=new_pad, \
                                    stride=stride, \
                                    compute_dtype=feature_map.dtype, \
                                    dilation=(self._dilate_h,
                                              self._dilate_w))
        a_im2col_fractal_shape = (a_batch,
                                  (height_out * width_out + self._m0 - 1) \
                                  // self._m0,
                                  a_c1 * kernel_h * kernel_w, \
                                  self._m0, \
                                  a_c0)
        a_col = im2col_fractal(a_im2col_fractal_shape, a_row_major)
        return a_col


    def generate_c(self, tensor_a, tensor_b, tensor_bias=None, c_type=None):
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
        tensor_c = super(ConvDslPattern, self).generate_c(tensor_a,
                                                          tensor_b,
                                                          tensor_bias,
                                                          c_type)
        row_major = tensor_a.op.input_tensors[0]
        ho_wo = row_major.shape[1].value
        _, _, c_m, _ = list(i.value for i in tensor_c.shape)
        m_0 = self._m0
        m_1 = c_m // m_0
        if not ((m_1 - 1) * m_0) < ho_wo <= c_m:
            raise RuntimeError("HoWo param error!")
        return tensor_c
