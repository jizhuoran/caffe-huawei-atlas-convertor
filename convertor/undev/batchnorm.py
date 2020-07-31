import te.lang.cce
from te import tvm
from topi import generic


def bn_fw_op(shape, input_dtype, kernel_name):



def batch_norm(x, scale, offset, mean, variance, y, batch_mean,
               batch_variance, reserve_space_1, reserve_space_2,
               epsilon=0.0001, data_format="NHWC",
               is_training=True, kernel_name="batch_norm"):
    """
    algorithm: fused_batch_norm
    Batch normalization.
    Note that the size of 5D Tensors are defined by "NC1HWC0".
    The input tensor's dimension C should be equal.

    Parameters
    ----------
    x: dict
        dict of input, A Tensor for input data.
    scale: dict
        dict of scale,
        A Tensor for scaling factor, to scale the normalized x.
    offset: dict
        dict of offset, A Tensor for offset, to shift to the normalized x.
    mean: dict
        dict of mean, A Tensor for population mean.
        Used for inference only, must be empty for training.
    variance: dict
        dict of variance, A Tensor for population variance.
        Used for inference only, must be empty for training.
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    batch_mean: dict
        dict of batch_mean, A `Tensor`. Has the same type as `mean`.
    batch_variance: dict
        dict of batch_var, A `Tensor`. Has the same type as `variance`.
    reserve_space_1: dict
        dict of reserve_space_1, A `Tensor`.
    reserve_space_2: dict
        dict of reserve_space_2, A `Tensor`.
    epsilon: float
        A small float number added to the variance of x. Defaults to `0.001`.
    data_format: str
        The data format for x and y. Support "NC1HWC0" only.
    is_training: bool
        A bool value indicates the operation for train (default) or inference.
    kernel_name: str
        kernel name, default value is "batch_norm"


def bn_bw_op(shape, input_dtype, kernel_name):

    data = tvm.placeholder(shape, name="data", dtype=input_dtype)
    diff = tvm.placeholder(shape, name="diff", dtype=input_dtype) 
    

    with tvm.target.cce():
        mask = te.lang.cce.vcmpsel(data, 
                                  tvm.const(0, dtype=input_dtype),
                                  'gt', 
                                  tvm.const(1, dtype =input_dtype), 
                                  tvm.const(0, dtype =input_dtype))
        res = te.lang.cce.vmul(mask, diff)
        sch = generic.auto_schedule(res)

    config = {"print_ir": False,
              "need_build": True,
              "name": kernel_name,
              "tensor_list": [data, diff, res]}
    
    te.lang.cce.cce_build_code(sch, config)


if __name__ == '__main__':
    relu_fw_op((5000, 1), "float16", "ReLU_fw_5000")
    relu_bw_op((5000, 1), "float16", "ReLU_bw_5000")
