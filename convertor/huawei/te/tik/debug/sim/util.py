"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     util.py
DESC:     sim's util file
CREATED:  2019-7-04 20:12:13
MODIFIED: 2019-7-24 14:04:45
"""
from __future__ import print_function
import math

from te.tik.common.util import get_bit_len, DTYPE_SIZE, ceil_div
from te.tik.common.common_util import vector_max_offset_cal
from te.tik.tik_lib.tik_params import ONE_REP_BYTE_SIZE, MASK_VALUE_ZERO
from te.platform.cce_params import scope_gm
from te.tik.tik_lib.tik_check_util import TikCheckUtil
from ..util import get_dtype_size, get_flatten_idx


class TempEnv():
    """the temp environment"""

    def __init__(self):
        self.memory_space = {}
        self.x_register_alloc = 0
        self.va_register_alloc = 0
        self.tensor_cache = {}
        self.buffer_cache = {}

    def alloc_memory(self, size, scope, align):
        """allocate memory

        Parameters
        ----------
        size : memory size
        scope : ub global and so on.
        align: align address

        Returns
        -------
        the buffer address
        """
        if scope not in self.memory_space:
            self.memory_space[scope] = 0

        cur_addr = self.memory_space[scope]
        aligned_addr = int(math.ceil(cur_addr / float(align))*align)
        self.memory_space[scope] = aligned_addr + size
        return aligned_addr

    def alloc_register(self):
        """allocate register

        Returns
        -------
        the result code
        """
        from te.tik.tik_lib.tik_params import MAX_XREG_ALLOCATED
        ret = self.x_register_alloc
        if ret >= MAX_XREG_ALLOCATED:
            TikCheckUtil.raise_error("all register exhausted")
        self.x_register_alloc += 1
        return ret

    def alloc_va_register(self, num=1):
        """allocate va register.
        If 'num' equals to 2, user is applying 2 successive VA registers

        Parameters
        ----------
        num: the number of VA register

        Returns
        -------
        the result code
        """
        if num not in (1, 2):
            TikCheckUtil.raise_error("Only support 1 or 2 VA for allocation.")

        from te.tik.tik_lib.tik_params import MAX_VAREG_ALLOCATED
        ret = self.va_register_alloc
        if ret >= MAX_VAREG_ALLOCATED or (num == 2 and
                                          (ret + 1 >= MAX_VAREG_ALLOCATED)):
            TikCheckUtil.raise_error("all va register exhausted")

        self.va_register_alloc += num
        return ret if (num == 1) else (ret, ret + 1)

    def get_tensor_addr(self, context, tensor, access_mode):
        """get tensor's address

        Parameters
        ----------
        context : stack context
        tensor : the tensor
        access_mode: buffer read or write mode

        Returns
        -------
        the tensor address
        """
        # ATTENTION: you must copy tensor to model before you query the addr
        key = str(tensor.buffer) + access_mode
        dtype_size_ = get_dtype_size(tensor.dtype)
        flatten_idx = get_flatten_idx(tensor.indice, context)
        buffer_info = self.tensor_cache[key]
        # 1 is index of buffer addr in buffer info list
        return buffer_info[1] + flatten_idx*dtype_size_

    def get_buffer_info(self, context, tensor,
                        dtype_size, align, access_mode):
        """get buffer info:splited from function copy_tensor_to_model

        Parameters
        ----------
        context : stack context
        tensor : the tensor
        dtype_size: tensor dtype's size
        align :align address
        access_mode: buffer rw mode

        Returns
        -------
        info of buffer
        """
        # pylint: disable=R0913
        key = str(tensor.buffer) + access_mode
        flatten_np = context.get_value(tensor).buffer.reshape(-1)
        if key in self.tensor_cache:
            buffer_info = self.tensor_cache[key]
        else:
            if tensor.buffer not in self.buffer_cache:
                # need to allocate new memory
                alloc_size = len(flatten_np)*dtype_size
                addr = self.alloc_memory(alloc_size, tensor.scope, align)
                context.model.write_memory(
                    addr, tensor.scope, flatten_np.ctypes.data, alloc_size)
                # last buffer info is access_valid for check mem access,
                # default False
                buffer_info = [tensor.buffer, addr,
                               alloc_size, flatten_np.ctypes.data,
                               access_mode, tensor.name, False]
                self.tensor_cache[key] = buffer_info
                self.buffer_cache[tensor.buffer] = key
            else:
                # don't need allocate memory,
                # just change buffer_info of access_mode
                buffer_info = self.tensor_cache[
                    self.buffer_cache[tensor.buffer]][:]
                # 4 is index of access_mode,
                # if you change code of line 137, you should change this index
                buffer_info[4] = access_mode
                self.tensor_cache[key] = buffer_info

        return buffer_info

    @staticmethod
    def _get_xn_addr(context, tensor, dtype_size, buffer_addr, offset=0):
        """get xn register align addr

        Parameters
        ----------
        context : stack context
        tensor : the tensor
        dtype_size: tensor dtype size
        buffer_addr: addr of buffer
        offset: buffer offset

        Returns
        -------
        x_n: xn register align addr
        """
        flatten_idx = get_flatten_idx(tensor.indice, context)
        x_n = buffer_addr + (flatten_idx + offset)*dtype_size
        x_n = context.evaluate_expr(x_n)
        return x_n

    def copy_tensor_to_model(self, context, tensor,  # pylint: disable=R0914
                             align, check_align,
                             require_xt, access_mode, offset=0):
        """copy tensor to context.model

        Parameters
        ----------
        context : stack context
        tensor : the tensor
        align : align address
        check_align : True or False
        require_xt: True or False
        access_mode: buffer rw mode
        offset: tensor offset

        Returns
        -------
        tuple , info of buffer
        """
        # pylint: disable=R0913
        dtype_size_ = get_dtype_size(tensor.dtype)
        _, buffer_addr, buffer_size, buffer_ptr, _, _, _ = \
            self.get_buffer_info(context, tensor, dtype_size_,
                                 align, access_mode)

        x_n = self._get_xn_addr(context, tensor, dtype_size_,
                                buffer_addr, offset)
        if check_align:
            # is not divisible, then raise error
            if x_n % align != 0:
                TikCheckUtil.raise_error(
                    'Address align error! {} is not {} align'.format(
                        x_n, align))
        xn_idx = None
        if require_xt:
            xn_idx = self.alloc_register()
            context.model.write_gpr(xn_idx, x_n)

        #  ATTENTION: here we return the the info of buffer
        #  we are not expected to manipulate the partial buffer directly
        return xn_idx, buffer_addr, buffer_size, buffer_ptr

    def check_mem_access(self, model, is_vector=False):
        """check memory if access

        Parameters
        ----------
        model : stack context.model
        is_vector : if it's vector
        """
        access_list = model.get_memory_access()
        for mem_access in access_list:
            # the current model does not respect mask in vector instr read mem.
            if is_vector and mem_access.mode == 'r':
                continue

            valid = False

            for buf, buf_info in self.tensor_cache.items():
                # set access_valid to False
                self.tensor_cache[buf][-1] = False
                # 0 is index of tensor buffer
                if buf_info[0].scope == mem_access.scope:
                    _, buf_addr, buffer_size, _, rw_mode, _, _ = buf_info
                    buf_upper_bound = buf_addr + buffer_size
                    if buf_info[0].scope == scope_gm:
                        # 32 is addr align byte nums.
                        buf_size_mod = buffer_size % 32
                        if buf_size_mod != 0:
                            # if not 32B align, increase bound to 32B aligned.
                            buf_upper_bound += (32 - buf_size_mod)
                    if rw_mode == 'rw':
                        if buf_addr <= mem_access.addr <= buf_upper_bound and \
                                buf_addr <= mem_access.addr + mem_access.size \
                                <= buf_upper_bound:
                            valid = True
                            # set access_valid to True
                            self.tensor_cache[buf][-1] = True
                            break
                    else:
                        if buf_addr <= mem_access.addr <= buf_upper_bound and \
                                buf_addr <= mem_access.addr + mem_access.size \
                                <= buf_upper_bound and \
                                rw_mode == mem_access.mode:
                            valid = True
                            # set access_valid to True
                            self.tensor_cache[buf][-1] = True
                            break

            if not valid:
                print('Memory out of range occur in following tensor buffer:')
                for _, buf_info in self.tensor_cache.items():
                    # check buffer info of access_valid param
                    if not buf_info[-1]:
                        # buffer info: buffer, buffer addr,
                        #              buffer size, buffer ptr, access mode,
                        #              tensor name, access valid flag
                        print('aviliable buffer:{} '
                              '[buf_addr:{} buf_size:{} '
                              'tensor_name:{} '
                              'access_mode:{}]'.format(buf_info[0].scope,
                                                       buf_info[1],
                                                       buf_info[2],
                                                       buf_info[5],
                                                       buf_info[4]))
                print('------------')
                TikCheckUtil.raise_error(
                    'AccessViolation:\n{}'.format(mem_access))


def check_read_mem_out_of_bounds(context, src_buffer_size, mask, tensor,
                                 repeat_time, blk_stride, rep_stride,
                                 stride_unit=0, mask_mode="normal",
                                 ori_offset=0):
    """check if out-of-bounds visit when it's read mode for a tensor

    Parameters
    ----------
    context: debug context
    src_buffer_size: tensor's buffer size
    mask: instruction's mask
    tensor: the tensor to be read
    repeat_time: the times of instruction run
    blk_stride: the stride between each block
    rep_stride: the stride between each repeat
    stride_unit : address and offset unit both affect it. default = 0
    mask_mode : mode of mask, normal/counter, default value = normal
    ori_offset: tensor offset

    Return
    ----------
    None
    """
    # pylint: disable=R0913
    if isinstance(mask, (list, tuple)):
        mask_value = [context.evaluate_expr(value) for value in mask]
    else:
        mask_value = context.evaluate_expr(mask)

    if mask_mode == "counter":
        repeat_time = ceil_div(mask,
                               ONE_REP_BYTE_SIZE // DTYPE_SIZE[tensor.dtype])
        mask_value = mask_value % (ONE_REP_BYTE_SIZE //
                                   DTYPE_SIZE[tensor.dtype])
        if mask_value == MASK_VALUE_ZERO:
            mask_value = ONE_REP_BYTE_SIZE // DTYPE_SIZE[tensor.dtype]

    extend_offset = vector_max_offset_cal(
        mask_value, tensor.dtype,
        ONE_REP_BYTE_SIZE // get_bit_len(tensor.dtype),
        context.evaluate_expr(repeat_time),
        context.evaluate_expr(blk_stride),
        context.evaluate_expr(rep_stride), stride_unit)

    offset = get_flatten_idx(tensor.indice, context)
    expected_size = extend_offset + offset + ori_offset
    total_size = src_buffer_size // get_dtype_size(tensor.dtype)

    if expected_size > total_size:
        TikCheckUtil.raise_error(
            "AccessViolation: tensor {} need read {} elements, "
            "but only {} elements space".format(
                tensor.name, expected_size, total_size))
