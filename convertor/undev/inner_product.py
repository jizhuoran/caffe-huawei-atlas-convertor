import te.lang.cce
from te import tvm
from topi import generic
import impl as ops








def _matmul_op(M, K, N, trans_a, trans_b, has_bias, input_dtype):

    M = 1 if M == 1 else (M + 15 ) // 16 * 16;
    K = 1 if K == 1 else (K + 15 ) // 16 * 16;
    N = 1 if N == 1 else (N + 15 ) // 16 * 16;

    a_shape = (K, M) if trans_a else (M, K)
    b_shape = (N, K) if trans_b else (K, N)
    bias_shape = (N, )


    tensor_a = tvm.placeholder(a_shape, name='tensor_a', dtype=input_dtype)
    tensor_b = tvm.placeholder(b_shape, name='tensor_b', dtype=input_dtype)
    tensor_blas = tvm.placeholder(bias_shape, name='tensor_blas', dtype=input_dtype) if has_bias else None

    kernel_name = "matmul_op_{}_{}_{}_{}_{}_{}".format(M, K, N, 
                                                    "TA" if trans_a else "NTA", 
                                                    "TB" if trans_b else "NTB", 
                                                    "bias" if has_bias else "nobias")

    res = te.lang.cce.matmul(tensor_a, tensor_b, trans_a=trans_a, trans_b=trans_b, dst_dtype=input_dtype, tensor_bias=tensor_blas)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"print_ir": False,
                "need_build": True,
                "name": kernel_name,
                "tensor_list": [tensor_a, tensor_b, tensor_blas, res] if has_bias else [tensor_a, tensor_b, res]}

    te.lang.cce.cce_build_code(sch, config)
    return [kernel_name]


             
def inner_fw_bw_op(M, K, N, transpose, has_bias, input_dtype):


    fw_kernels = _matmul_op(M, K, N, False, (not transpose), has_bias, input_dtype)

    bw_kernels = _matmul_op(K if transpose else N, 
        M, 
        N if transpose else K, 
        True, False, 
        False, 
        input_dtype)
    bw_kernels += _matmul_op(M, N, K, False, transpose, False, input_dtype)
    
    return fw_kernels + bw_kernels



if __name__ == '__main__':
    inner_fw_bw_op(64, 32, 128, False, True, "float32")
