import te.lang.cce
from te import tvm
from topi import generic

def pooling_fw_op(pooling_mode, padding_mode,
                  batch_size, channel, height, width, 
                  kernel_h, kernel_w, stride_h, stride_w, 
                  input_dtype):
  
  
  data_five_shape  = (batch_size, (channel + 15)//16, height, width, 16)
  data_five = tvm.placeholder(data_five_shape, name="data", dtype=input_dtype)

  kernel_name = "pooling_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(pooling_mode, padding_mode,
                                       batch_size, channel, height, width, 
                                       kernel_h, kernel_w, stride_h, stride_w)

  res = te.lang.cce.pooling2d(data_five, (kernel_h, kernel_w), (stride_h, stride_w), pooling_mode, padding_mode)

  with tvm.target.cce():
    sch = generic.auto_schedule(res)

  config = {"print_ir": False,
             "need_build": True,
             "name": kernel_name,
             "tensor_list": [data_five, res]}
    
  te.lang.cce.cce_build_code(sch, config)


# def pooling_bw_op(shape, input_dtype, kernel_name):

#     data = tvm.placeholder(shape, name="data", dtype=input_dtype)
#     diff = tvm.placeholder(shape, name="diff", dtype=input_dtype) 
    

#     with tvm.target.cce():
#         mask = te.lang.cce.vcmpsel(data, 
#                                   tvm.const(0, dtype=input_dtype),
#                                   'gt', 
#                                   tvm.const(1, dtype =input_dtype), 
#                                   tvm.const(0, dtype =input_dtype))
#         res = te.lang.cce.vmul(mask, diff)
#         sch = generic.auto_schedule(res)

#     config = {"print_ir": False,
#               "need_build": True,
#               "name": kernel_name,
#               "tensor_list": [data, diff, res]}
    
#     te.lang.cce.cce_build_code(sch, config)


if __name__ == '__main__':
    pooling_fw_op("MAX", "SAME", 64, 20, 28, 28, 2, 2, 2, 2, "float16")
    # pooling_bw_op((32000, 1), "float16", "pooling_bw_5")
