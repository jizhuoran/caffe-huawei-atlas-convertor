"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

strided_slice
"""
from functools import reduce as functools_reduce
from te import tik
from topi.cce import util
from te import platform as tbe_platform

SHAPE_SIZE_LIMIT = 2147483648


# pylint: disable=too-many-instance-attributes,too-few-public-methods
# pylint: disable=too-many-locals,too-many-statements,unused-variable
def get_tensor_size_in_fp16(data_shape, data_type="float16"):
    data_size = functools_reduce(lambda x, y: x  * y, data_shape)
    fp16_size = data_size
    if data_type == "float32":
        fp16_size = fp16_size * 2
    return fp16_size


class StridedSliceLastDim:
    '''
    strided_slice applied on the last dim of the input tensor
    '''
    def __init__(self, input_data, output_data, kernel_name):
        self.input_shape = input_data.get('shape')
        self.output_shape = output_data.get('shape')
        self.data_type = input_data.get('dtype').lower()
        self.kernel_name = kernel_name
        self.tik_instance = tik.Tik()
        self.product_core_num = \
            tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)

        self.gm_input = self.tik_instance.Tensor(self.data_type, self.input_shape,
                                                 scope=tik.scope_gm, name="gm_input")

        self.gm_output = self.tik_instance.Tensor(self.data_type, self.input_shape,
                                                  scope=tik.scope_gm, name="gm_output")

    def strided_slice_compute(self):
        # 计算过程分32核进行多核实现
        input_size = get_tensor_size_in_fp16(self.input_shape, self.data_type)
        output_size = get_tensor_size_in_fp16(self.output_shape, self.data_type)

        # 总共要搬运的次数
        total_loop = (input_size + 511) // 512

        # 需要的核数（默认情形是总的循环次数小于32）
        core_used = total_loop
        # 核内循环次数
        loop_num = 1
        less_core_end = core_used

        # 这里除2的原因在于，GM地址是FP32类型
        common_src_size = 256
        more_src_size = 256
        last_src_size = (input_size % 512) // 2


        if total_loop > 32:
            core_used = 32
            src_size = input_size // 4 // 32 * 4
            dst_size = src_size // 2
            loop_num = (src_size + 511) // 512

            with self.tik_instance.for_range(0, core_used, block_num=core_used) as block_index:
                src_ub = self.tik_instance.Tensor("float16", (512,), tik.scope_ubuf, "src_ub")
                dst_ub = self.tik_instance.Tensor("float16", (512,), tik.scope_ubuf, "dst_ub")

                with self.tik_instance.if_scope(block_index == core_used - 1):
                    # 尾部数据处理
                    self.tik_instance.data_move(src_ub,
                                                self.gm_input[input_size // 2 - 256],
                                                0, 4, 8, 0, 0)


                    # 对偶数行做VNCHWCONV并通过重排实现数据抽取
                    src_addr = [src_ub[i * 32] for i in range(16)]
                    dst_addr = []
                    for i in range(4):
                        dst_addr = dst_addr + [dst_ub[i * 32], dst_ub[i * 32 + 16],
                                               dst_ub[i * 32 + 256], dst_ub[i * 32 + 256 + 16]]
                    self.tik_instance.vnchwconv(False, False, dst_addr, src_addr, 1, 0, 0)

                    # 对奇数行做VNCHWCONV并通过重排实现数据抽取
                    src_addr = [src_ub[i * 32 + 16] for i in range(16)]
                    dst_addr = []
                    for i in range(4):
                        dst_addr = dst_addr + [dst_ub[i * 32 + 128], dst_ub[i * 32 + 128 + 16],
                                               dst_ub[i * 32 + 384], dst_ub[i * 32 + 384 + 16]]
                    self.tik_instance.vnchwconv(False, False, dst_addr, src_addr, 1, 0, 0)

                    # 将上述两次重拍结果合并并再次VNCHWCONV最终实现STRIDED_SLICE
                    src_addr = [dst_ub[i * 16] for i in range(16)]
                    dst_addr = [src_ub[i * 16] for i in range(16)]
                    self.tik_instance.vnchwconv(False, False, dst_addr, src_addr, 1, 0, 0)

                    # 将处理结果128个FP32复制到GM里去
                    self.tik_instance.data_move(
                        self.gm_output[output_size // 2 - 128], src_ub, 0, 2, 8, 0, 0)

                with self.tik_instance.for_range(0, loop_num) as loop_index:
                    with self.tik_instance.if_scope(loop_index != loop_num - 1):
                        self.tik_instance.data_move(
                            src_ub, self.gm_input[(src_size // 2 * block_index)
                                                  + loop_index * 256], 0, 4, 8, 0, 0)
                    with self.tik_instance.else_scope():
                        self.tik_instance.data_move(
                            src_ub, self.gm_input[(src_size // 2 * block_index)
                                                  + src_size // 2 - 256], 0, 4, 8, 0, 0)


                    # 对偶数行做VNCHWCONV并通过重排实现数据抽取
                    src_addr = [src_ub[i * 32] for i in range(16)]
                    dst_addr = []
                    for i in range(4):
                        dst_addr = dst_addr + [dst_ub[i * 32], dst_ub[i * 32 + 16],
                                               dst_ub[i * 32 + 256], dst_ub[i * 32 + 256 + 16]]
                    self.tik_instance.vnchwconv(False, False, dst_addr, src_addr, 1, 0, 0)

                    # 对奇数行做VNCHWCONV并通过重排实现数据抽取
                    src_addr = [src_ub[i * 32 + 16] for i in range(16)]
                    dst_addr = []
                    for i in range(4):
                        dst_addr = dst_addr + [dst_ub[i * 32 + 128], dst_ub[i * 32 + 128 + 16],
                                               dst_ub[i * 32 + 384], dst_ub[i * 32 + 384 + 16]]
                    self.tik_instance.vnchwconv(False, False, dst_addr, src_addr, 1, 0, 0)

                    # 将上述两次重拍结果合并并再次VNCHWCONV最终实现STRIDED_SLICE
                    src_addr = [dst_ub[i * 16] for i in range(16)]
                    dst_addr = [src_ub[i * 16] for i in range(16)]
                    self.tik_instance.vnchwconv(False, False, dst_addr, src_addr, 1, 0, 0)

                    # 将处理结果128个FP32复制到GM里去
                    with self.tik_instance.if_scope(loop_index != loop_num - 1):
                        self.tik_instance.data_move(self.gm_output[(dst_size // 2 * block_index)
                                                                   + loop_index * 128],
                                                    src_ub, 0, 2, 8, 0, 0)
                    with self.tik_instance.else_scope():
                        self.tik_instance.data_move(self.gm_output[(dst_size // 2 * block_index)
                                                                   + dst_size // 2 - 128],
                                                    src_ub, 0, 2, 8, 0, 0)

        else:
            with self.tik_instance.for_range(0, core_used, block_num=core_used) as block_index:
                src_ub = self.tik_instance.Tensor("float16", (512,), tik.scope_ubuf, "src_ub")
                dst_ub = self.tik_instance.Tensor("float16", (512,), tik.scope_ubuf, "dst_ub")

                cur_src_size = 256
                with self.tik_instance.if_scope(block_index == core_used - 1):
                    cur_src_size = last_src_size

                #当前核的GM起始地址
                src_start_addr = 256 * loop_num * block_index
                dst_start_addr = 128 * loop_num * block_index

                with self.tik_instance.for_range(0, loop_num) as loop_index:
                    with self.tik_instance.if_scope(block_index == core_used - 1):
                        self.tik_instance.data_move(src_ub,
                                                    self.gm_input[src_start_addr
                                                                  + loop_index * 256 +
                                                                  last_src_size - 256],
                                                    0, 4, 8, 0, 0)
                    with self.tik_instance.else_scope():
                        self.tik_instance.data_move(
                            src_ub, self.gm_input[src_start_addr + loop_index * 256],
                            0, 4, 8, 0, 0)

                    # 对偶数行做VNCHWCONV并通过重排实现数据抽取
                    src_addr = [src_ub[i * 32] for i in range(16)]
                    dst_addr = []
                    for i in range(4):
                        dst_addr = dst_addr + [dst_ub[i * 32], dst_ub[i * 32 + 16],
                                               dst_ub[i * 32 + 256], dst_ub[i * 32 + 256 + 16]]
                    self.tik_instance.vnchwconv(False, False, dst_addr, src_addr, 1, 0, 0)

                    # 对奇数行做VNCHWCONV并通过重排实现数据抽取
                    src_addr = [src_ub[i * 32 + 16] for i in range(16)]
                    dst_addr = []
                    for i in range(4):
                        dst_addr = dst_addr + [dst_ub[i * 32 + 128], dst_ub[i * 32 + 128 + 16],
                                               dst_ub[i * 32 + 384], dst_ub[i * 32 + 384 + 16]]
                    self.tik_instance.vnchwconv(False, False, dst_addr, src_addr, 1, 0, 0)

                    # 将上述两次重拍结果合并并再次VNCHWCONV最终实现STRIDED_SLICE
                    src_addr = [dst_ub[i * 16] for i in range(16)]
                    dst_addr = [src_ub[i * 16] for i in range(16)]
                    self.tik_instance.vnchwconv(False, False, dst_addr, src_addr, 1, 0, 0)

                    # 将处理结果128个FP32复制到GM里去
                    with self.tik_instance.if_scope(block_index == core_used - 1):
                        self.tik_instance.data_move(
                            self.gm_output[dst_start_addr +
                                           loop_index * 128 + last_src_size // 2 - 128],
                            src_ub, 0, 2, 8, 0, 0)
                    with self.tik_instance.else_scope():
                        self.tik_instance.data_move(
                            self.gm_output[dst_start_addr + loop_index * 128],
                            src_ub, 0, 2, 8, 0, 0)

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=[self.gm_input],
                                   outputs=[self.gm_output], enable_l2=False)
        return self.tik_instance


def strided_slice_two_turn_one(input_x, output_x, kernel_name):
    """

    Returns
    -------
    None
    """
    input_shape = input_x.get("shape")
    input_dtype = input_x.get("dtype").lower()
    check_list = ("float16", "float32")

    util.check_dtype_rule(input_dtype, check_list)
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(input_shape)
    util.check_shape_size(input_shape, SHAPE_SIZE_LIMIT)

    ss_last_dim = StridedSliceLastDim(input_x, output_x, kernel_name)

    return ss_last_dim.strided_slice_compute()
