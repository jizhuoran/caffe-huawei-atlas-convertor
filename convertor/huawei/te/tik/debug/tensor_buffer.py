"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     context.py
DESC:     debug context
CREATED:  2019-10-26 20:12:13
MODIFIED: 2019-10-28 09:12:23
"""
# disabling:
# R0913: too-many-arguments

from te.tik.tik_lib.tik_check_util import TikCheckUtil
from .npbuffer import NumpyBuffer, NumpyBufferProxy


class TensorBuffer():
    """Class tensor buffer define"""
    def __init__(self):
        super(TensorBuffer, self).__init__()
        # keep a ref of all tensor,
        # so they won't be GCed and our id is not changed
        self.tensor_holder = {}
        # id(tensor) -> NumpyBuffer
        self.tensor2buffer = {}
        # tvm buffer to it's owner
        self.tvmbuffer2tensor = {}
        self.init_mode = 'random'
        self.init_value = None

    def add_tensor(self, tensor):
        """add tensor to buffer
        Parameters
        ----------
        tensor:source tensor

        Returns
        ----------
        No returns
        """
        tensor_key = id(tensor)
        # allocate buffer if necessary
        if tensor_key in self.tensor2buffer:
            TikCheckUtil.raise_error('we should not add a tensor buffer twice')
        npbuf = NumpyBuffer(tensor.shape, tensor.dtype,
                            self.init_mode, self.init_value)
        self.tvmbuffer2tensor[tensor.buffer] = tensor
        self.tensor2buffer[tensor_key] = npbuf
        self.tensor_holder[tensor_key] = tensor

    def add_tensor_proxy(self, context, proxy_tensor, buffer_, indice, dtype):
        """add tensor to buffer proxy
        Parameters
        ----------
        context: class Context

        proxy_tensor: tensor of proxy

        buffer_: tensor buffer

        indice: tensor indice

        dtype: tensor dtype

        Returns
        ----------
        No returns
        """
        # pylint: disable=R0913
        # too many arguments, so disable it.
        np_buf = NumpyBufferProxy(context, buffer_, indice, dtype)
        self.tensor2buffer[id(proxy_tensor)] = np_buf
        self.tensor_holder[id(proxy_tensor)] = proxy_tensor

    def get_npbuffer_by_tvmbuffer(self, tvm_buffer):
        """get numpy buffer by tvm buffer
        Parameters
        ----------
        tvm_buffer: tvm buffer

        Returns
        ----------
        return: numpy buffer
        """
        tensor = self.tvmbuffer2tensor[tvm_buffer]
        return self.tensor2buffer[id(tensor)]

    def get_npbuffer_by_tensor(self, tensor):
        """get numpy buffer by tensor
        Parameters
        ----------
        tensor:source tensor

        Returns
        ----------
        return: numpy buffer
        """
        return self.tensor2buffer[id(tensor)]
