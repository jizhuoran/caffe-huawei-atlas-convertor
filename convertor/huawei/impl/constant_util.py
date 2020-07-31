#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

constant_util
"""
# datatype of fp32
DATA_TYPE_FP32 = "float32"

# datatype of fp16
DATA_TYPE_FP16 = "float16"

# datatype of int32
DATA_TYPE_INT32 = "int32"

# datatype of int16
DATA_TYPE_INT16 = "int16"

# datatype of int8
DATA_TYPE_INT8 = "int8"

# datatype of uint64 is uint64
DATA_TYPE_UINT64 = "uint64"

# datatype of int64 is int64
DATA_TYPE_INT64 = "int64"

# datatype of uint32 is uint32
DATA_TYPE_UINT32 = "uint32"

# datatype of uint16 is uint16
DATA_TYPE_UINT16 = "uint16"

# datatype of uint8 is uint8
DATA_TYPE_UINT8 = "uint8"

# one element takes up 4b
DATA_SIZE_FOUR = 4

# one element takes up 2b
DATA_SIZE_TWO = 2

# one element takes up 1b
DATA_SIZE_ONE = 1

# one element takes up 8b
DATA_SIZE_EIGHT = 8

# instruction's default sid is 0
SID = 0

# int zero constant
INT_DEFAULT_ZERO = 0

# float zero constant
FLOAT_DEFAULT_ZERO = 0.0

# default Nburst
DEFAULT_NBURST = 1

# stride zero
STRIDE_ZERO = 0

# default burst length
DEFAULT_BURST_LEN = 1

# default repeat time
DEFAULT_REPEAT_TIME = 1
# instruction mask
MASK128 = 128

# instruction mask 64
MASK64 = 64

# stride one
STRIDE_ONE = 1

# default repeat stride length
REPEAT_STRIDE_EIGHT = 8

# repeat stride length
REPEAT_STRIDE_FOUR = 4

# default repeat time one
REPEAT_TIME_ONCE = 1

# default block stride one
BLOCK_STRIDE_ONE = 1

# one block size takes up 32b
BLOCK_SIZE = 32

# the vector size is 256B
VECTOR_BYTE_SIZE = 256

# scalar shape size=1
SCALAR_SIZE = 1

# max block number
MAX_BLOCK_NUMBER = 32

# product is hisi-es
HISI_ES = "hisi-es"

# product is v200
AIC = "aic"

# product is mini
MINI = "mini"

# product is cloud
CLOUD = "cloud"
