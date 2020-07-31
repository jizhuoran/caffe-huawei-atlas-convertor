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

"""

#!/usr/bin/python
# -*- coding: UTF-8 -*-

from .util import check_common_tiling
from .util import Compare


class TilingBufferID:
    '''
    enum buffer id
    '''
    AL1 = 0
    BL1 = 1
    AL0 = 2
    BL0 = 3
    AUB = 4
    BUB = 5
    CL0 = 6
    CUB = 7
    OUT = 8


class TilingToAffineShape:
    """docstring for TilingToAffineShape"""

    def __init__(self, tiling, f_affine_to_l0c=None,
                 f_affine_to_out=None, need_check_tiling=True):
        if need_check_tiling:
            check_common_tiling(tiling)
        self._tiling = tiling
        _, _, fractal_m, fractal_k, _ = tiling["AL0_matrix"]
        _, _, fractal_n, fractal_k, _ = tiling["BL0_matrix"]
        self._fractal_m = fractal_m
        self._fractal_k = fractal_k
        self._fractal_n = fractal_n
        # self._extent_kc = extent_kb if extent_kb > extent_ka else extent_ka
        default_affine = lambda *indice: indice
        if f_affine_to_l0c is None:
            self._affine_to_l0c = default_affine
        else:
            if f_affine_to_l0c.__code__.co_argcount != 7:
                raise RuntimeError("narg must be 7")
            self._affine_to_l0c = f_affine_to_l0c

        if f_affine_to_out is None:
            self._affine_to_out = default_affine
        else:
            if f_affine_to_out.__code__.co_argcount != 5:
                raise RuntimeError("narg must be 5")
            self._affine_to_out = f_affine_to_out

    def compare_input_to_l0c(self, tiling_buffer_id, shape_k):
        '''
        Function: compare_input_to_l0c
        Summary: compare affine shape to l0c
        Examples:
        Attributes:
            @param (self):self
            @param (tiling_buffer_id):TilingBufferID
            @param (shape_k): buffer shape
        Returns: status
        '''
        tiling = self._tiling
        extent_nc, extent_mc, fractal_m, fractal_n, batch = tiling.get(
            "CL0_matrix")

        refs_shape = self._affine_to_l0c(batch, extent_nc, extent_mc,
                                         fractal_m, fractal_n,
                                         shape_k, self._fractal_k)
        affine_shape = None
        if tiling_buffer_id == TilingBufferID.AL0:
            affine_shape = self.al0_to(TilingBufferID.CL0)
        elif tiling_buffer_id == TilingBufferID.BL0:
            affine_shape = self.bl0_to(TilingBufferID.CL0)
        elif tiling_buffer_id == TilingBufferID.AL1:
            affine_shape = self.al1_to(TilingBufferID.CL0)
        elif tiling_buffer_id == TilingBufferID.BL1:
            affine_shape = self.bl1_to(TilingBufferID.CL0)
        else:
            raise RuntimeError("unsupport this buffer id %d"
                               % (tiling_buffer_id))

        if not affine_shape:
            # raise RuntimeError
            return None
        status = Compare.compare(affine_shape, refs_shape)
        return status

    def cub_to(self, dst=TilingBufferID.OUT):
        '''
        Function: cub_to
        Summary: cal affine shape from cub to dst
        Examples:
        Attributes:
            @param (self): self
            @param (dst) default=TilingBufferID.OUT: the dst tiling buffer id
        Returns: affine shape
        '''
        if dst != TilingBufferID.OUT:
            raise RuntimeError("error")
        tiling = self._tiling
        nc_part, mc_part, fractal_m, fractal_n, batch = tiling["CUB_matrix"]
        return self._affine_to_out(batch, nc_part, mc_part,
                                   fractal_m, fractal_n)

    def cl0_to(self, dst=TilingBufferID.OUT):
        '''
        Function: cl0_to
        Summary: cal affine shape from cl0 to dst
        Examples:
        Attributes:
            @param (self):self
            @param (dst) default=TilingBufferID.OUT: the dst buffer id
        Returns:  affine shape
        '''
        if dst != TilingBufferID.OUT:
            raise RuntimeError("error")
        tiling = self._tiling
        extent_nc, extent_mc, fractal_m, fractal_n, batch = tiling[
            "CL0_matrix"]
        return self._affine_to_out(batch, extent_nc, extent_mc,
                                   fractal_m, fractal_n)

    def al0_to(self, dst=TilingBufferID.OUT):
        '''
        Function: al0_to
        Summary: cal affine shape from al0 to dst
        Examples:
        Attributes:
            @param (self):self
            @param (dst) default=TilingBufferID.OUT: the dst buffer id
        Returns: affine shape
        '''
        tiling = self._tiling
        extent_ma, extent_ka, fractal_m, fractal_k, batch = tiling[
            "AL0_matrix"]

        _, extent_nb, fractal_n, _, _ = tiling.get("BL0_matrix")
        affine_shape = None
        if dst == TilingBufferID.OUT:
            affine_shape = self._affine_to_out(batch, None, extent_ma,
                                               fractal_m, fractal_n)
        elif dst == TilingBufferID.CL0:
            affine_shape = self._affine_to_l0c(batch, extent_nb, extent_ma,
                                               fractal_m, fractal_n,
                                               extent_ka, fractal_k)
        else:
            raise RuntimeError("Invalid")
        return affine_shape

    def bl0_to(self, dst=TilingBufferID.OUT):
        '''
        Function: bl0_to
        Summary:  cal affine shape from bl0 to dst
        Examples:
        Attributes:
            @param (self):self
            @param (dst) default=TilingBufferID.OUT:  the dst buffer id
        Returns: affine shape
        '''
        tiling = self._tiling
        bl0_feature = tiling.get("BL0_matrix")
        if not bl0_feature:
            return bl0_feature

        extent_kb, extent_nb, fractal_n, fractal_k, batch = bl0_feature
        extent_ma, _, fractal_m, _, _ = tiling["AL0_matrix"]
        affine_shape = None
        if dst == TilingBufferID.OUT:
            affine_shape = self._affine_to_out(batch, extent_nb, None,
                                               fractal_m, fractal_n)
        elif dst == TilingBufferID.CL0:
            affine_shape = self._affine_to_l0c(batch, extent_nb, extent_ma,
                                               fractal_m, fractal_n,
                                               extent_kb, fractal_k)
        else:
            raise RuntimeError("Invalid")
        return affine_shape

    def al1_to(self, dst=TilingBufferID.OUT):
        '''
        Function: al1_to
        Summary:  cal affine shape from al1 to dst
        Examples:
        Attributes:
            @param (self):self
            @param (dst) default=TilingBufferID.OUT:  the dst buffer id
        Returns: affine shape
        '''
        tiling = self._tiling
        k_al1, m_al1, batch = tiling["AL1_shape"]
        _, extent_mc, fractal_m, fractal_n, _ = tiling["CL0_matrix"]
        fractal_k = self._fractal_k
        affine_shape = None
        if dst == TilingBufferID.OUT:
            affine_shape = self._affine_to_out(batch, None, m_al1 * extent_mc,
                                               fractal_m, fractal_n)
        elif dst == TilingBufferID.CL0:
            affine_shape = self._affine_to_l0c(batch, None, m_al1 * extent_mc,
                                               fractal_m, fractal_n,
                                               k_al1, fractal_k)
        else:
            raise RuntimeError("Invalid")
        return affine_shape

    def bl1_to(self, dst=TilingBufferID.OUT):
        '''
        Function: bl1_to
        Summary:  cal affine shape from bl1 to dst
        Examples:
        Attributes:
            @param (self):self
            @param (dst) default=TilingBufferID.OUT:  the dst buffer id
        Returns: affine shape
        '''
        tiling = self._tiling
        bl1_feature = tiling.get("BL1_shape")
        if not bl1_feature:
            return bl1_feature
        k_bl1, n_bl1, batch_bl1 = bl1_feature
        extent_nc, _, fractal_m, fractal_n, _ = tiling["CL0_matrix"]
        fractal_k = self._fractal_k
        affine_shape = None
        if dst == TilingBufferID.OUT:
            affine_shape = self._affine_to_out(batch_bl1, n_bl1 * extent_nc,
                                               None, fractal_m, fractal_n)
        elif dst == TilingBufferID.CL0:
            affine_shape = self._affine_to_l0c(batch_bl1, n_bl1 * extent_nc,
                                               None, fractal_m, fractal_n,
                                               k_bl1, fractal_k)
        else:
            raise RuntimeError("Invalid")
        return affine_shape
