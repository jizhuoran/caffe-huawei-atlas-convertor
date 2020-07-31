#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright 2018 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Instructions Strings
"""

EXP = "vector_exp"
RELU = "vector_relu"
REC = "vector_rec"
LN = "vector_ln"
ABS = "vector_abs"
SQRT = "vector_sqrt"
RSQRT = "vector_rsqrt"
NOT = "vector_not"
DUP = "vector_dup"
MUL = "vector_mul"
ADD = "vector_add"
SUB = "vector_sub"
DIV = "vector_div"
MAX = "vector_max"
MIN = "vector_min"
MULVS = "vector_muls"
ADDVS = "vector_adds"
MAXVS = "vector_maxs"
MINVS = "vector_mins"
LRELU = "vector_lrelu"
EQ = "vector_eq"
NE = "vector_ne"
GE = "vector_ge"
LE = "vector_le"
GT = "vector_gt"
LT = "vector_lt"
EQVS = "vector_eqs"
NEVS = "vector_nes"
GEVS = "vector_ges"
LEVS = "vector_les"
GTVS = "vector_gts"
LTVS = "vector_lts"
AND = "vector_and"
OR = "vector_or"
MULCONV = "vector_mul_conv"
ADDRELU = "vector_addrelu"
SUBRELU = "vector_subrelu"
ADDRELUCONV = "vector_addrelu_conv"
SUBRELUCONV = "vector_subrelu_conv"
SHR = "vector_shr"
SHR_ROUND = "vector_shr_round"
SHL = "vector_shl"
MADD = "vector_madd"
MADDRELU = "vector_maddrelu"
MLA = "vector_mla"
AXPY = "vector_axpy"
CAST = "vector_conv"
CAST_VDEQ = "vector_conv_vdeq"
CAST_RINT = "vector_conv_rint"
CAST_ROUND = "vector_conv_round"
CAST_FLOOR = "vector_conv_floor"
CAST_CEIL = "vector_conv_ceil"
CAST_TRUNC = "vector_conv_trunc"
CAST_ROUNDING = "vector_conv_rounding"
TCAST = "vector_tconv"
TCAST_RINT = "vector_tconv_rint"
TCAST_ROUND = "vector_tconv_round"
TCAST_FLOOR = "vector_tconv_floor"
TCAST_CEIL = "vector_tconv_ceil"
TCAST_TRUNC = "vector_tconv_trunc"
TCAST_ROUNDING = "vector_tconv_rounding"
REDUCE_INIT = "vector_reduce_init"
REDUCE_SUM = "vector_reduce_sum"
REDUCE_MIN = "vector_reduce_min"
REDUCE_MAX = "vector_reduce_max"
REDUCE_ARGMIN = "vector_reduce_argmin"
REDUCE_ARGMAX = "vector_reduce_argmax"
REDUCE = "vector_reduce"
SELECT_EQ = "vector_select_eq"
SELECT_NE = "vector_select_ne"
SELECT_GT = "vector_select_gt"
SELECT_GE = "vector_select_ge"
SELECT_LT = "vector_select_lt"
SELECT_LE = "vector_select_le"
SELECT = "vector_select_bool"
SELECTVS = "vector_selects_bool"
AUTO = "vector_auto"
PHONY_INSN = "phony_insn"
IM2COL = "im2col"
SET_FMATRIX = "set_fmatrix"
MAD = "mad"
DEPTHWISE_CONV = "depthwise_conv"
DMA_COPY = "dma_copy"
DMA_PADDING = "dma_padding"
DATA_MOV = "data_mov"
SCALAR = "scalar"
SCALAR_SQRT = "scalar_sqrt"
VSCSPLIT = "vscsplit"


def get_insn_cmd():
    support_insn_cmd = (EXP,
                        RELU,
                        REC,
                        LN,
                        ABS,
                        SQRT,
                        RSQRT,
                        NOT,
                        DUP,
                        MUL,
                        ADD,
                        SUB,
                        DIV,
                        MAX,
                        MIN,
                        MULVS,
                        ADDVS,
                        MAXVS,
                        MINVS,
                        LRELU,
                        EQ,
                        NE,
                        GE,
                        LE,
                        GT,
                        LT,
                        EQVS,
                        NEVS,
                        GEVS,
                        LEVS,
                        GTVS,
                        LTVS,
                        AND,
                        OR,
                        MULCONV,
                        ADDRELU,
                        SUBRELU,
                        ADDRELUCONV,
                        SUBRELUCONV,
                        SHR,
                        SHR_ROUND,
                        SHL,
                        MADD,
                        MADDRELU,
                        MLA,
                        AXPY,
                        CAST,
                        CAST_VDEQ,
                        CAST_RINT,
                        CAST_ROUND,
                        CAST_FLOOR,
                        CAST_CEIL,
                        CAST_TRUNC,
                        CAST_ROUNDING,
                        TCAST,
                        TCAST_RINT,
                        TCAST_ROUND,
                        TCAST_FLOOR,
                        TCAST_CEIL,
                        TCAST_TRUNC,
                        TCAST_ROUNDING,
                        REDUCE_INIT,
                        REDUCE_SUM,
                        REDUCE_MIN,
                        REDUCE_MAX,
                        REDUCE_ARGMIN,
                        REDUCE_ARGMAX,
                        REDUCE,
                        SELECT_EQ,
                        SELECT_NE,
                        SELECT_GT,
                        SELECT_GE,
                        SELECT_LT,
                        SELECT_LE,
                        SELECT,
                        SELECTVS,
                        AUTO,
                        PHONY_INSN,
                        IM2COL,
                        SET_FMATRIX,
                        MAD,
                        DEPTHWISE_CONV,
                        DMA_COPY,
                        DMA_PADDING,
                        DATA_MOV,
                        SCALAR,
                        SCALAR_SQRT,
                        VSCSPLIT)
    return support_insn_cmd