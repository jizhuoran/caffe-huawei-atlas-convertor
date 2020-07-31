import te.lang.cce
from te import tvm
from topi import generic
#import impl as ops


def matmul1_op(M, K, N, trans_a, trans_b, has_bias, input_dtype):

  K = (K + 15 ) // 16 * 16;
  N = (N + 15 ) // 16 * 16;

  a_shape = (M, K)
  b_shape = (K, N)
  bias_shape = (N, )

  print(M, K, N, trans_a, trans_b, has_bias)

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

  config = {"print_ir": True,
             "need_build": True,
             "name": kernel_name,
             "tensor_list": [tensor_a, tensor_b, tensor_blas, res] if has_bias else [tensor_a, tensor_b, res]}
    
  te.lang.cce.cce_build_code(sch, config)


def matmul_op(M, K, N, trans_a, trans_b, has_bias, input_dtype, print_ir = False):

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
  print(kernel_name)
 
  res = te.lang.cce.matmul(tensor_a, tensor_b, trans_a=trans_a, trans_b=trans_b, dst_dtype=input_dtype, tensor_bias=tensor_blas)

  with tvm.target.cce():
    sch = generic.auto_schedule(res)

  config = {"print_ir": print_ir,
             "need_build": True,
             "name": kernel_name,
             "tensor_list": [tensor_a, tensor_b, tensor_blas, res] if has_bias else [tensor_a, tensor_b, res]}
    
  te.lang.cce.cce_build_code(sch, config)


def reduce_max(shape, axis, input_dtype):

  data = tvm.placeholder(shape, name="data", dtype=input_dtype)
  res = te.lang.cce.reduce_max(data, axis=axis)
  
  kernel_name = "reduce_max_{}_{}".format("_".join(map(str, shape)), axis)

  print(res)

  with tvm.target.cce():
    sch = generic.auto_schedule(res)

  config = {"print_ir": False,
             "need_build": True,
             "name": kernel_name,
             "tensor_list": [data, res]}
    
  te.lang.cce.cce_build_code(sch, config)

def vadd(shape, input_dtype):

  data1 = tvm.placeholder(shape, name="data1", dtype=input_dtype)
  data2 = tvm.placeholder(shape, name="data2", dtype=input_dtype)
  res = te.lang.cce.vadd(data1, data2)
  
  kernel_name = "vadd_{}".format("_".join(map(str, shape)))

  print(res)

  with tvm.target.cce():
    sch = generic.auto_schedule(res)

  config = {"print_ir": False,
             "need_build": True,
             "name": kernel_name,
             "tensor_list": [data1, data2, res]}
    
  te.lang.cce.cce_build_code(sch, config)

def vmul(shape, input_dtype):

  data1 = tvm.placeholder(shape, name="data1", dtype=input_dtype)
  data2 = tvm.placeholder(shape, name="data2", dtype=input_dtype)
  res = te.lang.cce.vmul(data1, data2)
  
  kernel_name = "vmul_{}".format("_".join(map(str, shape)))

  print(res)

  with tvm.target.cce():
    sch = generic.auto_schedule(res)

  config = {"print_ir": False,
             "need_build": True,
             "name": kernel_name,
             "tensor_list": [data1, data2, res]}
    
  te.lang.cce.cce_build_code(sch, config)


def broadcast(shape, axis, broadcast_size, input_dtype):

  outshape = list(shape)
  outshape[axis] = broadcast_size
  outshape = tuple(outshape)

  data = tvm.placeholder(shape, name="data", dtype=input_dtype)
  res = te.lang.cce.broadcast(data, outshape)
 
  
  kernel_name = "reduce_max_{}_{}_{}".format("_".join(map(str, shape)), axis, broadcast_size)

  print(res)

  with tvm.target.cce():
    sch = generic.auto_schedule(res)

  config = {"print_ir": False,
             "need_build": True,
             "name": kernel_name,
             "tensor_list": [data, res]}
    
  te.lang.cce.cce_build_code(sch, config)


def reduce_sum(shape, axis, input_dtype):

  data = tvm.placeholder(shape, name="data", dtype=input_dtype)
  res = te.lang.cce.sum(data, axis)

  kernel_name = "sum_{}_{}".format("_".join(map(str, shape)), axis)

  print(res)

  with tvm.target.cce():
    sch = generic.auto_schedule(res)

  config = {"print_ir": False,
             "need_build": True,
             "name": kernel_name,
             "tensor_list": [data, res]}
    
  te.lang.cce.cce_build_code(sch, config)

def softmax1_op(shape, input_dtype):

  outshape = list(shape)
  outshape[-1] = 1


  data = tvm.placeholder(shape, name="data", dtype=input_dtype)
  res = te.lang.cce.reduce_max(data, len(shape) - 1, keepdims=True)
  
  kernel_name = "softmax_fw1_{}".format("_".join(map(str, shape)))
  with tvm.target.cce():
    sch = generic.auto_schedule(res)

  config = {"print_ir": False,
             "need_build": True,
             "name": kernel_name,
             "tensor_list": [data, res]}
    
  te.lang.cce.cce_build_code(sch, config)
  
  ##########

  data = tvm.placeholder(shape, name="data", dtype=input_dtype)
  max4sub = tvm.placeholder(tuple(outshape), name="max4sub", dtype=input_dtype)

  res = te.lang.cce.sum(te.lang.cce.vexp(te.lang.cce.vsub(data, 
                                                          te.lang.cce.broadcast(max4sub, shape))), 
                        len(shape) - 1, 
                        keepdims=True)


  kernel_name = "softmax_fw2_{}".format("_".join(map(str, shape)))
  with tvm.target.cce():
    sch = generic.auto_schedule(res)

  config = {"print_ir": False,
             "need_build": True,
             "name": kernel_name,
             "tensor_list": [data, max4sub, res]}
    
  te.lang.cce.cce_build_code(sch, config)
  
  ############
  data = tvm.placeholder(shape, name="data", dtype=input_dtype)
  sum4div = tvm.placeholder(tuple(outshape), name="sum4div", dtype=input_dtype)

  res = te.lang.cce.vdiv(data, te.lang.cce.broadcast(sum4div, shape))

  kernel_name = "softmax_fw3_{}".format("_".join(map(str, shape)))
  with tvm.target.cce():
    sch = generic.auto_schedule(res)

  config = {"print_ir": False,
             "need_build": True,
             "name": kernel_name,
             "tensor_list": [data, sum4div, res]}
    
  te.lang.cce.cce_build_code(sch, config)


def softmax_op(shape, input_dtype):

  data = tvm.placeholder(shape, name="data", dtype=input_dtype)
  data1 = te.lang.cce.vsub(data, 
                           te.lang.cce.broadcast(te.lang.cce.reduce_max(data, len(shape) - 1, keepdims=True), shape))
  # data3 = te.lang.cce.vsub(data, data1)
  data4 = te.lang.cce.vexp(data1)
  data5 = te.lang.cce.sum(data4, len(shape) - 1, keepdims=True)
  data6 = te.lang.cce.broadcast(data5, shape)
  res = te.lang.cce.vdiv(data4, data6)

  kernel_name = "softmax_fw_{}".format("_".join(map(str, shape)))

  print(res)

  with tvm.target.cce():
    sch = generic.auto_schedule(res)

  config = {"print_ir": False,
             "need_build": True,
             "name": kernel_name,
             "tensor_list": [data, res]}
    
  te.lang.cce.cce_build_code(sch, config)


if __name__ == '__main__':
  #softmax_op((64, 10), "float16")
  # broadcast((1024, 1), 1, 1024, "float16")
  
     matmul_op(32, 32, 16, False, False, False, "float16")
    # matmul_op(64, 512, 16, False, True, True, "float16")

  # matmul_op(64, 800, 512, False, True, True, "float16")
  # matmul_op(16, 64, 512, True, False, False, "float16")
  # matmul_op(512, 64, 800, True, False, False, "float16")
  # matmul_op(1, 64, 10, False, False, False, "float16")
    # pooling_bw_op((32000, 1), "float16", "pooling_bw_5")
