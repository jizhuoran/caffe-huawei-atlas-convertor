# import te.lang.cce
# from te import tvm
# from topi import generic
import impl as ops


def _conv_fw_op(batch_size, channel_in, channel_out,
               in_height, in_width, out_height, out_width, has_bias,
               filter_h, filter_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
               input_dtype):

    kernel_name = "conv_fw_op_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(batch_size, 
                                                        channel_in, channel_out,
                                                        in_height, in_width, "bias" if has_bias else "nobias",
                                                        filter_h, filter_w,
                                                        pad_h, pad_w,
                                                        stride_h, stride_w)

    ops.conv2d(
        {"ori_shape":[batch_size, channel_in, in_height, in_width], "dtype":input_dtype, "ori_format":"NCHW"},
        {"ori_shape":[channel_out, channel_in, filter_h, filter_w], "dtype":input_dtype, "ori_format":"NCHW"},
        {"ori_shape":[channel_out, ], "dtype":input_dtype} if has_bias else None,
        None,
        {"ori_shape":(batch_size, channel_out, out_height, out_width), "dtype":input_dtype},
        (stride_h, stride_h, stride_w, stride_w),
        (pad_h, pad_h, pad_w, pad_w),
        (dilation_h, dilation_h, dilation_w, dilation_w),
        kernel_name = kernel_name)
    
    return [kernel_name]
             
def _conv_bw_op(batch_size, channel_in, channel_out,
               in_height, in_width, out_height, out_width, has_bias,
               filter_h, filter_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
               input_dtype):

                   
    def _conv_bw_weight_op():
        kernel_name = "conv_bw_weight_op_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(batch_size, 
                                                                        channel_in, channel_out,
                                                                        in_height, in_width,
                                                                        filter_h, filter_w,
                                                                        pad_h, pad_w,
                                                                        stride_h, stride_w)


        ops.conv2d_backprop_filter_d(
            {"ori_shape":(batch_size, channel_in, in_height, in_width), "dtype":input_dtype, "ori_format":"NCHW"},
            {"ori_shape":(batch_size, channel_out, out_height, out_width), "dtype":input_dtype, "ori_format":"NCHW"},
            {"ori_shape":(channel_out, channel_in, filter_h, filter_w), "dtype":"float32", "ori_format":"NCHW"},
            (channel_out, channel_in, filter_h, filter_w),
            (stride_h, stride_h, stride_w, stride_w),
            (pad_h, pad_h, pad_w, pad_w),
            (dilation_h, dilation_h, dilation_w, dilation_w),
            kernel_name = kernel_name)

        return kernel_name

    def _conv_bw_input_op():

        kernel_name = "conv_bw_input_op_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(batch_size, 
                                                                        channel_in, channel_out,
                                                                        in_height, in_width,
                                                                        filter_h, filter_w,
                                                                        pad_h, pad_w,
                                                                        stride_h, stride_w)

        ops.conv2d_backprop_input_d(
            {"ori_shape":(channel_out, channel_in, filter_h, filter_w), "dtype":input_dtype, "ori_format":"NCHW"},
            {"ori_shape":(batch_size, channel_out, out_height, out_width), "dtype":input_dtype, "ori_format":"NCHW"},
            {"ori_shape":(batch_size, channel_in, in_height, in_width), "dtype":input_dtype, "ori_format":"NCHW"},
            (batch_size, channel_in, in_height, in_width),
            (stride_h, stride_h, stride_w, stride_w),
            (pad_h, pad_h, pad_w, pad_w),
            (dilation_h, dilation_h, dilation_w, dilation_w),
            kernel_name = kernel_name)
       
        return kernel_name

    weight_kernel_name = _conv_bw_weight_op()
    input_kernel_name = _conv_bw_input_op()
    return [weight_kernel_name, input_kernel_name]

def conv_fw_bw_op(batch_size, channel_in, channel_out,
    in_height, in_width, out_height, out_width, has_bias,
    filter_h, filter_w, pad_h, pad_w, stride_h, stride_w,
    input_dtype="float16", dilation_h = 1, dilation_w = 1):

    fw_kernels = _conv_fw_op(batch_size, channel_in, channel_out, in_height, in_width, out_height, out_width, has_bias, 
        filter_h, filter_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, input_dtype)
    bw_kernels = _conv_bw_op(batch_size, channel_in, channel_out, in_height, in_width, out_height, out_width, has_bias, 
        filter_h, filter_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, input_dtype)
    return fw_kernels + bw_kernels

if __name__ == '__main__':
    conv_fw_bw_op(64, 32, 32, 16, 16, 16, 16, False, 3, 3, 1, 1, 1, 1, "float16")
