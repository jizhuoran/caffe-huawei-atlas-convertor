"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     tik_api_constants.py
DESC:     place constants
CREATED:  2019-08-14 15:45:18
MODIFIED: 2019-08-14 15:45:18
"""
# disabling:
# C0302: too-many-lines (this file is full of data operation instructions)

from te.platform.cce_params import scope_cc, scope_cbuf, scope_cb, scope_ca,\
    scope_ubuf, scope_gm
from te.platform.cce_params import AIC
from te.platform.cce_params import HI3796CV300CSAIC

# data operation
DTYPE_MAP = {
    "uint4": "u4",
    "int4": "s4",
    "uint8": "u8",
    "int8": "s8",
    "uint16": "u16",
    "int16": "s16",
    "float16": "f16",
    "uint32": "u32",
    "int32": "s32",
    "float32": "f32",
    "uint64": "u64",
    "int64": "s64",
    "float64": "f64"
}

VNCHWCONV_INSTR_APPENDIX_MAP = {
    "f32f32": "b32",
    "u32u32": "b32",
    "s32s32": "b32",
    "f16f16": "b16",
    "u16u16": "b16",
    "s16s16": "b16",
    "u8u8": "b8",
    "s8s8": "b8"
}

SCOPE_MAP = {
    scope_ca: "ca",
    scope_cb: "cb",
    scope_cc: "cc",
    scope_cbuf: "cbuf",
    scope_ubuf: "ubuf",
    scope_gm: "gm"
}


LOAD2D_DMA_LIST = {
    ('cbuf', 'cb'): (4, "load_cbuf_to_cb"),
    ('cbuf', 'ca'): (4, "load_cbuf_to_ca"),
    ('gm', 'ca'): (5, "load_gm_to_ca"),
    ('gm', 'cb'): (5, "load_gm_to_cb"),
    ('gm', 'cbuf'): (5, "load_gm_to_cbuf")
}

VTYPE_T_MAP = {
    "u8u8": "u8",
    "u8u8u8": "u8",
    "s8s8": "s8",
    "s8s8s8": "s8",
    "u16u16": "u16",
    "u16u16u16": "u16",
    "s16s16": "s16",
    "s16s16s16": "s16",
    "u32u32": "u32",
    "u32u32u32": "u32",
    "s32s32": "s32",
    "s32s32s32": "s32",
    "f16f16": "f16",
    "f16f16f16": "f16",
    "f32f16": "fmix",
    "f32f16f16": "fmix",
    "f32f32": "f32",
    "f32f32f32": "f32"
}

ROUND_MODE_MAP = {
    "": "",
    "none": "",
    "round": "r",
    "floor": "f",
    "ceil": "c",
    "ceiling": "c",
    "away-zero": "a",
    "to-zero": "z",
    "odd": "o"
}

LOAD3DV2_FUNC_MAP = {
    'sk': []
}

ARCHVERSION_ONTHEFLY = {HI3796CV300CSAIC: ['s16', 'f16'],
                        AIC: ['s16', 'f16', 'f32']}

CR_MODE_MAP = {
    "f16f16": 0,
    "f32f32": 0,
    "s32s32": 0,
    "u32u32": 0,
    "f32f16": 1,
    "f32f16relu": 2,
    "s32f16deq": 3,
    "f16f32": 4,
    "f16f16relu": 5,
    "f32f32relu": 5,
    "s32s32relu": 5,
    "f16f16deq": 6,
    "s32f16vdeq": 7,
    "s32f16vdeqrelu": 7,
    "s32s8vdeq8": 8,
    "s32u8vdeq8": 8,
    "s32s8vdeq8relu": 8,
    "s32u8vdeq8relu": 8,
    "s32s8deq8": 9,
    "s32u8deq8": 9,
    "s32s8deq8relu": 9,
    "s32u8deq8relu": 9,
    "s32f16vdeq16": 10,
    "s32f16vdeq16relu": 10,
    "s32f16deq16": 11,
    "s32f16deq16relu": 11,
    "s32s16vdeqs16": 12,
    "s32s16vdeqs16relu": 12,
    "s32s16deqs16": 13,
    "s32s16deqs16relu": 13
}

WINO_PAD_MAP = {(0, 0): 0, (0, 1): 1, (0, 2): 2, (1, 0): 3, (1, 1): 4,
                (2, 0): 5}
