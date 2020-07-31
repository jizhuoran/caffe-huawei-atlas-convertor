import impl as ops



def relu_fw_bw_op(shape, input_dtype):

    fw_kernel_name = "ReLU_fw_{}".format(shape)
    bw_kernel_name = "ReLU_bw_{}".format(shape)

    ops.relu({"shape":(shape, 1), "dtype":input_dtype},
             {"shape":(shape, 1), "dtype":input_dtype},
             kernel_name = fw_kernel_name)
    ops.relu_grad({"shape":(shape, 1), "dtype":input_dtype},
             {"shape":(shape, 1), "dtype":input_dtype},
             {"shape":(shape, 1), "dtype":input_dtype},
             kernel_name = bw_kernel_name)
 
    return [fw_kernel_name, bw_kernel_name]


if __name__ == '__main__':
    relu_fw_bw_op(5000, "float32")
