import te.lang.cce
from te import tvm
from topi import generic
import impl as ops

def four2five(batch_size, channel_in, height, width, input_dtype="float16"):
    raw_shape = (batch_size, channel_in, height, width)
    kernel_name = 'four2five_{}_{}_{}_{}'.format(*raw_shape)
    data = tvm.placeholder(raw_shape, name='data', dtype=input_dtype)
    res = te.lang.cce.compute_four2five(data, raw_shape) #res.shape = (batch_size,(channel_in+15)//16,height,width,16)    
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"print_ir": True,
              "need_build": True,
              "name": kernel_name,
              "tensor_list": [data, res]}
    
    te.lang.cce.cce_build_code(sch, config)

def five2four(batch_size, channel_in, height, width, input_dtype="float16"):
    raw_shape = (batch_size, channel_in, height, width)
    shape = (batch_size, (channel_in+15)//16, height, width, 16)
    kernel_name = 'five2four_{}_{}_{}_{}'.format(*raw_shape)
    data = tvm.placeholder(shape, name='data', dtype=input_dtype)
    res = te.lang.cce.compute_five2four(data, raw_shape) #res.shape = (batch_size,(channel_in+15)//16,height,width,16)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"print_ir": False,
              "need_build": True,
              "name": kernel_name,
              "tensor_list": [data, res]}
    
    te.lang.cce.cce_build_code(sch, config)