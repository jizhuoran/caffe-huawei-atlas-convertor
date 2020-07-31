"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     reindex.py
DESC:     matmul function
CREATED:  2020-4-23 21:12:13
MODIFIED: 2020-4-23 21:12:45
"""
from te import tik
from te.tik.api import tik_tensor
from te.tik.tik_lib.tik_util import type_convert

# @cond
class ReIndexProxy():
    """
     tensor help proxy class
    """
    def __init__(self, tensor, sub_dims):
        """
        init the ReIndexProxy class
        :param tensor:  the tensor
        :param sub_dims:  the dim of tensor.
        """
        self.tensor = tensor
        self.dtype = tensor.dtype
        self.sub_dims = sub_dims
        self.tensor_shape = tensor.indice.origin_shape
        self.tensor_start_idx = [i.start for i in tensor.indice.indice]

    def boudary_check(self):
        """
        check the dim and shape
        :return: None
        """
        assert len(self.sub_dims) == len(self.tensor_shape)
        assert len(self.tensor_shape) == len(self.tensor_start_idx)

        for shape_n, sub_dim_n, idx_n in zip(self.tensor_shape, self.sub_dims,
                                             self.tensor_start_idx):
            assert sub_dim_n + idx_n <= shape_n
            assert idx_n >= 0

    def flat_access(self, index):
        """
         flat access the tensor .
        :param index: the index of tensor.
        :return: new tensor .
        """

        new_indices = []
        offset = 0
        accumulate_dim = 1
        for (start_idx, origin_shape) \
                        in reversed(list(zip(self.tensor_start_idx, self.tensor_shape))):
            offset += start_idx * accumulate_dim
            accumulate_dim *= origin_shape
        offset = tik.Expr(offset)

        start_ = type_convert(offset+index)
        end_ = type_convert(offset+index+1)
        step_ = type_convert(1)
        new_indices.append(slice(start_, end_, step_))

        # simulation of tik_tensor.TensorSlice.__init__
        proxy_indice = tik_tensor.TensorSlice.__new__(tik_tensor.TensorSlice)
        proxy_indice.origin_shape = (accumulate_dim,)
        proxy_indice.indice = new_indices

        proxy_tensor = self.tensor.access_wrap(self.tensor.ir_generator, self.tensor.buffer,
                                               self.tensor.dtype)
        proxy_tensor = proxy_tensor.flatten()
        proxy_tensor.indice = proxy_indice

        return proxy_tensor
    def __getitem__(self, index):
        new_indices = []
        if index is None:
            for i in self.tensor_shape:
                new_indices.append(slice(0, i, 1))
        else:
            if isinstance(index, (tuple, list)):
                for (start_idx, idx, _) \
                    in zip(self.tensor_start_idx, index, self.tensor_shape):
                    new_indices.append(start_idx + idx)
            else:
                new_indices.append(self.tensor_start_idx[0] + index)

        # simulation of tik_tensor.TensorSlice.__init__
        proxy_indice = tik_tensor.TensorSlice.__new__(tik_tensor.TensorSlice)
        proxy_indice.origin_shape = self.tensor_shape
        proxy_indice.indice = new_indices

        proxy_tensor = tik.Tensor(self.tensor.ir_generator, buffer_=self.tensor.buffer,
                                  indice=proxy_indice, dtype=self.tensor.dtype)
        return proxy_tensor
# @endcond
