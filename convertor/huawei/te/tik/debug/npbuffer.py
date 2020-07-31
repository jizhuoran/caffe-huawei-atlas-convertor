"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     npbuffer.py
DESC:     To store numpy buffer
CREATED:  2019-7-04 20:12:13
MODIFIED: 2019-7-16 21.46.13
"""
# disabling:
# R0903: too-few-public-methods
import numpy as np

from te.tik.common.util import reduce_mul
from ..tik_lib.tik_check_util import TikCheckUtil

_MAX_INIT_VALUE = 256
_MAX_RANDINT = 255


def get_uninited_buffer(shape, dtype, init_mode='random', init_value=None):
    """get uninited buffer

    Parameters
    ----------
    shape: data shape
    dtype: data type
    init_mode: init mode
    init_value: init value

    Returns
    ----------
    _buffer:NumpyBuffer
    """
    # 255 is max value of uint8
    # check init_mode is str
    TikCheckUtil.check_type_match(init_mode, str)
    if init_mode == 'random':
        shape = [int(s) for s in shape]
        _buffer = np.zeros(shape=shape, dtype=dtype).view('uint8')
        tmp_shape = _buffer.shape
        rand_buffer = np.random.randint(_MAX_RANDINT, size=tmp_shape,
                                        dtype='uint8').view(dtype)
        _buffer = rand_buffer
        return _buffer
    if init_mode == 'constant':
        TikCheckUtil.check_in_range(
            init_value, range(_MAX_INIT_VALUE),
            'init value out of range,value must be [0:255], but got %s'
            % init_value)
        _buffer = np.zeros(shape=shape, dtype=dtype).view('uint8')
        rand_buffer = (_buffer + init_value).view(dtype)
        _buffer = rand_buffer
        return _buffer
    return TikCheckUtil.raise_error(
        'unknown init mode %s init node only can be '
        'random or constant' % init_mode)


class NumpyBuffer():
    """ Store numpy buffer"""
    # pylint: disable=R0903
    # Class doesn't have public method, so disable it.
    def __init__(self, shape, dtype, init_mode='random', init_value=None):
        self.dtype = dtype
        self.shape = shape
        self.buffer = get_uninited_buffer(shape, dtype, init_mode, init_value)


class NumpyBufferProxy():
    """Store numpy buffer proxy"""
    # pylint: disable=R0903
    # Class has too few public methods, so disable it.
    def __init__(self, context, buffer_, indice, dtype):
        self.context = context
        self.tvm_buffer = buffer_
        self.indice = indice.indice
        self.dtype = dtype

    @staticmethod
    def _simplify_slice(old_slice, context):
        """simplify slice"""
        start = old_slice.start
        step = old_slice.step
        stop = old_slice.stop

        if start is not None:
            start = context.evaluate_expr(start)

        if step is not None:
            step = context.evaluate_expr(step)

        if stop is not None:
            stop = context.evaluate_expr(stop)

        return slice(start, stop, step)


    def shape(self):
        """
        get the shape

        Parameters
        ----------
        Returns
        ----------
        the list of shape
        """
        static_indice = [
            self._simplify_slice(s, self.context) for s in self.indice]
        tmp_shape = []
        for i in static_indice:
            tmp = (i.stop - i.start) // i.step
            tmp_shape.append(tmp)
        return tmp_shape

    @staticmethod
    def is_same_shape(origin_shape, new_shape):
        """
        judge the same shape

        Parameters
        ----------
        origin_shape: the origin  shape
        new_shape: the new shape
        Returns
        ----------
        True ; mean it is the same shape.
        False: mean it is not the same shape.
        """
        if len(origin_shape) != len(new_shape):
            return False
        for i, item in enumerate(origin_shape):
            if item != new_shape[i]:
                return False
        return True

    @property
    def buffer(self):
        """
        get _buffer attribute

        Returns
        ----------
        buffer:NumpyBuffer
        """
        static_indice = [
            self._simplify_slice(s, self.context) for s in self.indice]
        np_buffer = self.context.tensor_buffer.get_npbuffer_by_tvmbuffer(
            self.tvm_buffer)

        orign_tensor = \
            self.context.tensor_buffer.tvmbuffer2tensor[self.tvm_buffer]
        proxy_shape = self.shape()
        buffer = np_buffer.buffer
        # numpy reshape's function need
        # the total new shape equal to the total old shape.
        if (not self.is_same_shape(orign_tensor.shape, proxy_shape)) and (
                reduce_mul(proxy_shape) == reduce_mul(orign_tensor.shape)):
            buffer = buffer.reshape(proxy_shape)
        # for reinterpret_cast_to case.
        if self.dtype != orign_tensor.dtype:
            # at first view the buffer for new dtype.
            # and then call the numpy.resize to new shape.
            buffer = buffer.view(self.dtype)
            np.resize(buffer, proxy_shape)
            return buffer
        # other case .
        return buffer[tuple(static_indice)].view(self.dtype)
