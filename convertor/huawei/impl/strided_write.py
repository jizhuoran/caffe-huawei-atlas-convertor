"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
strided write
"""
from te import tvm
from topi import generic
from te.platform.fusion_manager import fusion_manager
from topi.cce import util


STRIDED_WRITE_TAG = "strided_write"


def check_params(x, y, axis):
    """
    check the parameters including x, y, axis.
    """
    if len(x.get("shape")) != 5:
        raise RuntimeError("x's length must be 5 while length is "
                           "{}.".format(len(x.get("shape"))))
    if len(y.get("shape")) != 5:
        raise RuntimeError("y's length must be 5 while length is "
                           "{}.".format(len(y.get("shape"))))
    if x.get("dtype") not in ("float16", "int8"):
        raise RuntimeError("x dtype shoule be float16 or int8 while dtype is "
                           "{}.".format(x.get("dtype")))
    if y.get("dtype") not in ("float16", "int8"):
        raise RuntimeError("y dtype shoule be float16 or int8 while dtype is "
                           "{}.".format(y.get("dtype")))
    if x.get("format") != "NC1HWC0":
        raise RuntimeError("input x's format must be NC1HWC0 while format is "
                           "{}.".format(x.get("format")))
    if y.get("format") != "NC1HWC0":
        raise RuntimeError("output y's format must be NC1HWC0 while format is "
                           "{}.".format(y.get("format")))
    if x.get("dtype") != y.get("dtype"):
        raise RuntimeError("x's dtype must be equal to y's dtype.")
    if axis != 1:
        raise RuntimeError("Only support axis = 1 now.")

@fusion_manager.register("strided_write")
def strided_write_compute(x, y, axis, stride, kernel_name='strided_write'):
    """
    write data to tensor by stride.

    Parameters:
    ----------
    x: placeholder of input tesnor.

    y: dict of output tensor.

    axis: which axis to write data by stride.

    stride: data write stride.

    kernel_name: cce kernel name, default value is "strided_write".

    Returns:
    ----------
    output_y: result tensor.
    """
    shape_y = tuple(i.value for i in x.shape)
    output_y = tvm.compute(
        shape_y,
        lambda batch_idx, c1_idx, hw_idx, c0_idx:
        x[batch_idx, c1_idx, hw_idx, c0_idx],
        name=kernel_name,
        tag=STRIDED_WRITE_TAG,
        attrs={"stride": stride})

    return output_y


@util.check_input_type(dict, dict, int, int, str)
def strided_write(x, y, axis, stride, kernel_name='strided_write'):
    """
    write data to tensor by stride.

    Parameters:
    ----------
    x: dict of input.

    y: dict of output.

    axis: which axis to write data by stride.

    stride: data write stride.

    kernel_name: cce kernel name, default value is "strided_write".

    Returns:
    -------
    None
    """

    check_params(x, y, axis)
    dtype_x = x.get("dtype")
    n_i, c1_i, h_i, w_i, c0_i = x.get("shape")
    shape_x = n_i, c1_i, h_i*w_i, c0_i
    input_x = tvm.placeholder(shape_x, name="input_x", dtype=dtype_x)
    res = strided_write_compute(input_x, y, axis, stride, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)
