"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     tik_params.py
DESC:     provide params
CREATED:  2019-04-18 18:53:42
MODIFIED: 2019-07-18 21:29:18
"""

from te import tvm
from te.platform.cce_params import AIC
from te.platform.cce_params import ASCEND_310AIC
from te.platform.cce_params import ASCEND_910AIC
from te.platform.cce_params import HI3796CV300ESAIC
from te.platform.cce_params import VEC
# def the cce thread axis for sync
CCE_GLOBAL_AXIS = tvm.thread_axis("cce_global")
CCE_MASK_AXIS = tvm.thread_axis("cce_mask")

CMPMASK_VAR = tvm.var("CMPMASK")
FMATRIX_VAR = tvm.var("FMATRIX")
FCOL2IMG_VAR = tvm.var("FCOL2IMG")
MASK_VAR = tvm.var("MASK")
DEQ_VAR = tvm.var("DEQ")
RPN_COR_IR = tvm.var("RPN_COR_IR")
RPN_OFFSET = tvm.var("RPN_OFFSET")
VMS4_SR = tvm.var("VMS4_SR")
CTRL_SPR = tvm.var("CTRL_SPR")
PADDING = tvm.var("PADDING")
RSVD_CNT = tvm.var("RSVD_CNT")
MAX_MIN_CNT = tvm.var("MAX_MIN_CNT")
L0_SET_VALUE = tvm.var("L0_SET_VALUE")
VA_REG = [tvm.var("VA0"), tvm.var("VA1"), tvm.var("VA2"), tvm.var("VA3"),
          tvm.var("VA4"), tvm.var("VA5"), tvm.var("VA6"), tvm.var("VA7")]

AI_CORE_INDICATE = "aicore arch: "

PYTHON_VERSION3 = 3
PYTHON_VERSION_IDX = 0
ATTR_VALUE = 0
FOR_TYPE_ZERO = 0

# default 0 for strides
DEFAULT_STRIDE = 0

MAX_BLK_STRIDE_DOUBLE_BYTE = 65536
MAX_BLK_STRIDE_SINGLE_BYTE = 256
MAX_REP_STRIDE_DOUBLE_BYTE = 65536
MAX_REP_STRIDE_SINGLE_BYTE = 256
MIN_REPEAT_TIMES = 1
MAX_REPEAT_TIMES = 256
MAX_MATRIX = 4096
MAX_N_SMALL_CHANNEL = 5

# strides idx
SRC_BLK_STRIDE_IDX = 0
DST_BLK_STRIDE_IDX = 1

PIPE_S = 1
PIPE_V = 2
PIPE_M = 3
PIPE_MTE1 = 4
PIPE_MTE2 = 5
PIPE_MTE3 = 6
PIPE_MTE4 = 7
PIPE_MTE5 = 8
PIPE_V2 = 9
PIPE_ALL = 10

ONE_BYTE_BIT_LEN = 8
ONE_REP_BYTE_SIZE = 256
ONE_BLK_SIZE = 32

BLK_NUM_PER_REP = 8

KB_SIZE = 1024

PAD_LENGTH = 4
MAX_PADDING = 256
MAX_TENSOR_WIDTH = 32768
MAX_TENSOR_HEIGHT = 32768
MIN_TENSOR_WIDTH = 1
MIN_TENSOR_HEIGHT = 1
MAX_FETCH_POS = 256
MAX_START_POINT = 32768
MIN_START_POINT = -255
MAX_STRIDE = 64
MIN_STRIDE = 1
MAX_FILTER_WIDTH = 256
MIN_FILTER_WIDTH = 1
MAX_FILTER_HEIGHT = 256
MIN_FILTER_HEIGHT = 1
MAX_DILATION = 256
MIN_DILATION = 1

MAX_VA_ADDR_NUM = 8
VA0_INDEX = 0
VA1_INDEX = 1
VA2_INDEX = 2
VA3_INDEX = 3
VA4_INDEX = 4
VA5_INDEX = 5
VA6_INDEX = 6
VA7_INDEX = 7
ONE_VA_ADDR_NUM = 4
VA_ADDR_BIT_LEN = 16
VA_ADDR_BYTE_SIZE = 32
MAX_ADDR = 2**16 - 1
MAX_ADDR_HEX = 0xffff

MIN_NBURST = 1
MAX_NBURST_DOUBLE_BYTE = 4096
MAX_NBURST_SINGLE_BYTE = 256
MIN_BURST_LEN = 1
MAX_BURST_LEN_DOUBLE_BYTE = 65536
MAX_BURST_LEN_SINGLE_BYTE = 256
MAX_START_INDEX = 65536
MAX_SID = 16
MAX_DST_GAP_DOUBLE_BYTE = 65536
MAX_DST_GAP_SINGLE_BYTE = 256
MAX_SRC_GAP = 256
MIN_BURST_REPEAT = 1
MAX_BURST_REPEAT = 256

HAS_PARAM_CONCAT = 1
NEED_PARAM_CONCAT = 3

# v4dtrans
MIN_M_LEN = 1
MAX_M_LEN = 4096
MIN_CHANNELS = 1
MAX_CHANNELS = 4096

# vpadding
MAX_PAD_MODE = 3

# vreduce
MIN_SRC1_PATTERN = 1
MAX_SRC1_PATTERN = 7

# load3dv2
MIN_CHANNEL_SIZE = 1
MAX_CHANNEL_SIZE = 64
MIN_EXTENSION = 1
MAX_EXTENSION = 65536
MAX_START_PT = 65536
ELE_PER_FRACTAL_EDGE = 16

# vcmax, vcmin
MAXMIN_CNT_INDEX_LEN_1 = 1
MAXMIN_CNT_INDEX_LEN_3 = 3
FOUR_BYTE_VALUE = 0xffffffff
TWO_BYTE_VALUE = 0xffff
CNT_SHIFT_POS = 32
INDEX_SHIFT_POS = 48

# vconv
VDEQ_BLK_STRIDE = 0
VDEQ_REP_STRIDE = 0


# offset and segment list for concat param
FMATRIX_OFFSET_LIST = [0, 16, 32, 40, 48, 56]
FMATRIX_SEGMENT_LIST = [16, 16, 8, 8, 8, 8]
REG_FCOL2IMG_OFFSET_LIST = [0, 16, 32, 40, 48, 56]
REG_FCOL2IMG_SEGMENT_LIST = [16, 16, 8, 8, 8, 8]
PADDING_ONE_BYTE_OFFSET_LIST = [0, 8]
PADDING_ONE_BYTE_SEGMENT_LIST = [8, 8]
PADDING_TWO_BYTE_OFFSET_LIST = [0]
PADDING_TWO_BYTE_SEGMENT_LIST = [16]
LOAD3DV1_REG_XM_OFFSET_LIST = [0, 16, 24, 32, 48]
LOAD3DV1_REG_XM_SEGMENT_LIST = [12, 8, 8, 16, 16]
LOAD3DV1_REG_XT_OFFSET_LIST = [0, 6, 12, 20, 28, 36, 44, 52, 56]
LOAD3DV1_REG_XT_SEGMENT_LIST = [6, 6, 8, 8, 8, 8, 8, 1, 8]
LOAD3DV2_REG_XM_OFFSET_LIST = [0, 16, 32, 48]
LOAD3DV2_REG_XM_SEGMENT_LIST = [16, 16, 16, 16]
LOAD3DV2_REG_XT_OFFSET_LIST = [0, 6, 12, 20, 28, 36, 44, 48]
LOAD3DV2_REG_XT_SEGMENT_LIST = [6, 6, 8, 8, 8, 8, 1, 16]
VSCATTER_VGATHER_XT_SEGMENT_LIST = [32, 8, 8, 2, 8]
VSCATTER_VGATHER_XT_OFFSET_LIST = [0, 32, 40, 54, 56]
COL2IMG_REG_XT_OFFSET_LIST = [0, 6, 12, 20, 28, 36, 52, 56]
COL2IMG_REG_XT_SEGMENT_LIST = [6, 6, 8, 8, 8, 8, 1, 8]
COL2IMG_REG_XM_OFFSET_LIST = [16, 24, 32, 48]
COL2IMG_REG_XM_SEGMENT_LIST = [8, 8, 16, 16]
TENSOR_PADDING_OFFSET_LIST = [0]
TENSOR_PADDING_SEGMENT_LIST = [8]
OBJECT_SPECIAL_OFFSET_LIST = [56]
OBJECT_SPECIAL_SEGMENT_LIST = [8]
VEXTRACT_OFFSET_LIST = [16, 56]
VEXTRACT_SEGMENT_LIST = [3, 8]
VCONCAT_OFFSET_LIST = [16, 56]
VCONCAT_SEGMENT_LIST = [3, 8]
VMRGSORT4_OFFSET_LIST = [0, 8, 20, 32, 44, 59, 60]
VMRGSORT4_SEGMENT_LIST = [8, 12, 12, 12, 12, 1, 4]
RPN_COR_OFFSET_LIST = [16, 32, 56]
RPN_COR_SEGMENT_LIST = [16, 16, 8]
VECTOR_PAIR_OFFSET_LIST = [0, 16, 32, 56]
VECTOR_PAIR_SEGMENT_LIST = [16, 16, 16, 8]
SIX_STRIDE_OFFSET_LIST = [0, 8, 16, 24, 32, 40, 56]
SIX_STRIDE_SEGMENT_LIST = [8, 8, 8, 8, 8, 8, 8]
FOUR_STRIDE_OFFSET_LIST = [0, 16, 32, 40, 56]
FOUR_STRIDE_SEGMENT_LIST = [16, 16, 8, 8, 8]
THREE_STRIDE_OFFSET_LIST = [0, 16, 32, 56]
THREE_STRIDE_SEGMENT_LIST = [16, 16, 16, 8]
TWO_STRIDE_OFFSET_LIST = [0, 16, 56]
TWO_STRIDE_SEGMENT_LIST = [16, 16, 8]
VCMPVS_OFFSET_LIST = [16, 40, 56]
VCMPVS_SEGMENT_LIST = [16, 8, 8]
VREDUCE_OFFSET_LIST = [8, 16, 32, 40, 54, 56]
VREDUCE_SEGMENT_LIST = [8, 8, 8, 8, 2, 8]
V4DTRANS_OFFSET_LIST = [0, 12, 63]
V4DTRANS_SEGMENT_LIST = [12, 12, 1]
VPADDING_OFFSET_LIST = [0, 16, 32, 40, 48, 50, 54, 56]
VPADDING_SEGMENT_LIST = [16, 16, 8, 8, 2, 1, 2, 8]
VSEL_OFFSET_LIST = [0, 8, 16, 24, 32, 40, 48, 56]
VSEL_SEGMENT_LIST = [8, 8, 8, 8, 8, 8, 2, 8]
VEC_SCALAR_OFFSET_LIST = [56, 0, 16, 32, 40, 54]
VEC_SCALAR_SEGMENT_LIST = [8, 16, 16, 8, 8, 2]


DEQSCALE_SHIFT_POS = 48
SCALE_SHIFT_POS = 47
SCALE_ADDR_BIT_POS = 15
DEQSCALE_46BIT_SHIFT_POS = 46
DEQSCALE_46BIT_MASK = 0x0000400000000000

MAX_C1_INDEX = 4096
MAX_JUMP_OFFSET = 128
MIN_JUMP_OFFSET = 1

DST_TYPE_LEN = 16
DST_TYPE_LEN_MAX = 32
MAX_REPEAT_MODE = 2
STRIDES_LEN = 2
PADMODE_NO_PADDING = 0
MAX_PADMODE = 6
PAD_MASK = 0xff
PADDING_SHIFT_POS = 8
PADDING_LEFT_IDX = 0
PADDING_RIGHT_IDX = 1
PADDING_TOP_IDX = 2
PADDING_BOT_IDX = 3

DEFAULT_VALUE_INDEX = 0
LEN_SHAPE_ONE = 1

MAX_MODE_NUMBER_VEXTRACT_V100 = 4
MAX_MODE_NUMBER = 6
ELEMENTS_MULTIPLE = 16
ELEMENTS_MULTIPLE_EIGHT = 16*8
SRC_LIST_LEN = 4
MAX_ELEMENTS_LEN = 4096
VALID_BIT_TUPLE = (15, 7, 3)
VALID_BIT_TUPLE_V200 = (15, 7, 3, 1)
VMS4_SR_ARRAY_LEN = 4
VMS4_REG_BIT_ALL_ONE = 2**13 - 1
VMS4_REGION_LIST0_POS = 39
VMS4_REGION_LIST1_POS = 26
VMS4_REGION_LIST2_POS = 13
SCALAR_LIST_LEN = 4
MAX_NUMBER = 65536
MIN_STRIDE_UNIT = 0
MAX_STRIDE_UNIT = 4

SCALAR_EXTENTS = (1,)

MASK_LEN_CONTINOUS_MODE = 1
MASK_LEN_FULL_MODE = 2
MASK_LEN_64 = 64
MASK_LEN_128 = 128
MASK_LOW_SHIFT = 32
MASK_VALUE_ZERO = 0
SCALAR_MASK_VALUE_IDX = 0
MASK_VALUE_64 = 64
MASK_VALUE_128 = 128
MASK_HIGH_IDX = 0
MASK_LOW_IDX = 1
MAX_MASK_LOW_VALUE = 2**32 - 1
MIN_MASK = 1
MAX_MASK = 129
MAX_MASK_64 = 65
MIN_MASK_HALF = 0
MAX_MASK_HALF = 2**64 - 1
MAX_COUNTER_MASK = 2**32

MAX_ATOMIC_ADD_MODE = 3
ATOMIC_ADD_MODE_SHIFT_POS = 60
MAX_SMALL_CHANNEL_MODE = 2
MAX_FP2INT_MODE = 2
SMALL_CHANNEL_ENABLE_SHIFT_POS = 63
FP2INT_SHIFT_POS = 59
MAX_SYSTEM_CACHE_MODE = 4
SYSTEM_CACHE_MODE_SHIFT_POS = 57
MAX_TWO_BITS_VALUE = 0b11
MAX_ONE_BIT_VALUE = 0b1
ATOMIC_ADD_MASK = 0xcfffffffffffffff
SMALL_CHANNEL_MASK = 0x7fffffffffffffff
SYSTEM_CACHE_MASK = 0xf9ffffffffffffff
FP2INT_MASK = 0xf7ffffffffffffff
MASK_MODE_MASK = 0xfeffffffffffffff
MASK_COUNTER_MODE_ENABLE_SHIFT_POS = 56

BLK_32_LIST = [0xff, 0xff00, 0xff0000, 0xff000000, 0xff00000000,
               0xff0000000000, 0xff000000000000, 0xff00000000000000]
BLK_16_LIST = [0xffff, 0xffff0000, 0xffff00000000, 0xffff000000000000]

INSTR_UNIT = 32
CACHE_LINE_SIZE = 128

INDEX_IN_STOP = 9223372036854775807
INDEX_IN_START = 0

MIN_INDEX = 0
MAX_INDEX = 64
CONST_MASK_VALUE = 0x8000000000000000
MAX_LOW_MASK_LEN = 64

SIX_STRIDE = 6
FOUR_STRIDE = 4
THREE_STRIDE = 3
TWO_STRIDE = 2

MAX_RET = 2**64
BIT_LEN_32 = 32
BIT_LEN_8 = 8
BIT_LEN_16 = 16

MAX_STORE_MODE = 2
VSEL_PARALLEL_BIT = 2048
VSEL_BLK_PARALLEL_BIT = 256
VCONV_BLK_PARALLEL = 16

MAX_XREG_ALLOCATED = 32
MAX_VAREG_ALLOCATED = 8

INC_MODE = 0
DEC_MODE = 1

MAX_VSEL_MODE = 3
VSEL_MODE_TENSOR_SCALAR = 1
VSEL_MODE_DOUBLE_TENSOR_ONE_IT = 0
VSEL_MODE_DOUBLE_TENSOR_MANY_IT = 2

CONV_F322F16_IS_RELU = 0b0010
CONV_F322F16_NO_RELU = 0b0001
CONV_S322F16_VECTOR_QUANT = 0b0111
CONV_RELU_VECTOR_QUANT = 7
CONV_RELU_QUANT = 3
BYTE_SIZE = 32
SPR_CONFIG_BIT_LEN = 15
CONV_S322F16_QUANT = 0b0011
CONV_F162F32_NO_RELU = 0b0100
NO_CONV_IS_RELU = 0b0101
CONV_L0C16_DEQ = 0b0110
CONV_S322B8_DEQ = 0b1000
VALUE_BI_1001 = 0b1001
VALUE_BI_1010 = 0b1010
VALUE_BI_1011 = 0b1011
VALUE_BI_1100 = 0b1100
VALUE_BI_1101 = 0b1101

ALIGN_TENSOR = 512
ALIGN_SRC = 32
ALIGN_SRC_EVEN = 2
ALIGN_DST = 1024
ALIGNED_ADDR = 32


REPEAT_MODE_SHIFT_BIT = 52

MMAD_MATRIX_K_POS = 12
MMAD_MATRIX_N_POS = 24
MMAD_EN_WINOGRAD_A_POS = 58
MMAD_EN_WINOGRAD_B_POS = 59
MMAD_EN_WEIGHT_OFFSET_POS = 60
MMAD_EN_SSPARSE_POS = 61
MMAD_L0C_BIT_POS = 63

# object_detect
ADDR_BIT_LEN = 16
EXHAUSTED_SUSPENSION_POS = 59
SRC_LIST_BIT_POS = 60
LENGTH_BIT_LEN = 12
LENGTH_BIAS = 8

# simd
SRC_BLOCK_STRIDE_SHIFT_POS = 16
STRIDE_UNIT_SHIFT_POS = 54
REPEAT_SHIFT_POS = 56
DST_REPEAT_STRIDE_SHIFT_POS = 32
SRC_REPEAT_STRIDE_SHIFT_POS = 40

# tik_proposal
REPEAT_UNIT_ELEMENTS = 16

UINT64_BIT = 64

# buffer scope
L1_BUFFER = "L1_Buffer"
UB_BUFFER = "Unified_Buffer"
L0A_BUFFER = "L0A_Buffer"
L0B_BUFFER = "L0B_Buffer"
LOC_BUFFER = "L0C_Buffer"
SMASK_BUFFER = "SMASK_Buffer"

# new SCOPE_CBUF_OUT, actual scope_cc
scope_cbuf_out = "local.L0C"  # pylint: disable=invalid-name

# current limit max ir num
MAX_IR_STATEMENT_NUM = 80000

ONE_IR = 1
TWO_IR = 2
THREE_IR = 3
FOUR_IR = 4
# product version to product name
AI_CORE_VERSION_MAP_TO_PRODUCT = {ASCEND_310AIC: "mini",
                                  ASCEND_910AIC: "cloud",
                                  AIC: "aic",
                                  HI3796CV300ESAIC: "hisi-es",
                                  VEC: "vec"}

VNCHWCONV_LIST_LEN = 16
MAX_C_SIZE = 2

#winograd
MIN_L1_H_W_C = 1
MAX_L1_H_W = 65536
MAX_L1_C = 4096
MAX_DST_GAP_WINO = 64
MAX_K_WINO = 4096
MAX_COL_INDIC = 4
WINO_FM_XM_OFFSET_LIST = [0, 16, 32, 48, 54, 56, 59]
WINO_FM_XM_SEGMENT_LIST = [16, 16, 12, 6, 2, 3, 3]
WINO_FM_XT_OFFSET_LIST = [8, 20, 32, 48]
WINO_FM_XT_SEGMENT_LIST = [12, 12, 16, 16]
MAX_REP_DIR = 2
WINO_WGT_OFFSET_LIST = [8, 16, 32, 40, 52, 54, 55, 56]
WINO_WGT_SEGMENT_LIST = [8, 16, 8, 7, 2, 1, 1, 8]

SHIFT_BIT_POS_7 = 7
SHIFT_BIT_POS_8 = 8
SHIFT_BIT_POS_11 = 11
SHIFT_BIT_POS_12 = 12
SHIFT_BIT_POS_13 = 13
SHIFT_BIT_POS_16 = 16
SHIFT_BIT_POS_20 = 20
SHIFT_BIT_POS_24 = 24
SHIFT_BIT_POS_32 = 32
SHIFT_BIT_POS_48 = 48
SHIFT_BIT_POS_52 = 52
SHIFT_BIT_POS_54 = 54
SHIFT_BIT_POS_55 = 55
SHIFT_BIT_POS_56 = 56
SHIFT_BIT_POS_59 = 59
SHIFT_BIT_POS_62 = 62

# depthwise_conv
VALUE_4096 = 4096
VALUE_3 = 3
VALUE_1 = 1
VALUE_4 = 4
VALUE_256 = 256
VALUE_128 = 128
VALUE_65504 = 65504

BYTE_PER_FRACTAL = 512
VALUE_127 = 127
LOAD_SMASK_OFFSET_LIST = [11, 0, 7]
LOAD_SMASK_SEGMENT_LIST = [1, 7, 4]
# DATA MAX MIN
INT8_MIN = -128
INT8_MAX = 127
UINT_MIN = 0
UINT8_MAX = 255
INT16_MIN = -32768
INT16_MAX = 32767
UINT16_MAX = 65535

# AIPP SPR
AIPP0_OFFSET_LIST = [0, 48, 56]
AIPP0_SEGMENT_LIST = [48, 8, 8]
AIPP1_OFFSET_LIST = [0, 63]
AIPP1_SEGMENT_LIST = [48, 1]
AIPP2_OFFSET_LIST = [0, 16, 32, 48]
AIPP2_SEGMENT_LIST = [16, 16, 16, 16]
AIPP3_OFFSET_LIST = [0, 16, 32, 48]
AIPP3_SEGMENT_LIST = [16, 16, 16, 16]
AIPP4_OFFSET_LIST = [0, 16, 24, 32, 40, 48, 56]
AIPP4_SEGMENT_LIST = [16, 8, 8, 8, 8, 8, 8]
AIPP5_OFFSET_LIST = [0, 16, 32, 48]
AIPP5_SEGMENT_LIST = [16, 16, 16, 16]
AIPP6_OFFSET_LIST = [0, 16, 32, 48]
AIPP6_SEGMENT_LIST = [16, 16, 16, 16]
AIPP7_OFFSET_LIST = [0, 16, 32, 48]
AIPP7_SEGMENT_LIST = [16, 16, 16, 16]
AIPP8_OFFSET_LIST = [0, 16, 32, 48]
AIPP8_SEGMENT_LIST = [16, 16, 16, 16]
AIPP9_OFFSET_LIST = [0, 16, 17, 18, 19, 24, 25, 26,
                     27, 29, 30, 35, 36, 38, 40, 48, 56]
AIPP9_SEGMENT_LIST = [16, 1, 1, 1, 4, 1, 1, 1, 2, 1, 5, 1, 2, 2, 1, 8, 8]
AIPP10_OFFSET_LIST = [48]
AIPP10_SEGMENT_LIST = [16]
AIPP11_OFFSET_LIST = [2, 8]
AIPP11_SEGMENT_LIST = [6, 6]
AIPP12_OFFSET_LIST = [0, 16]
AIPP12_SEGMENT_LIST = [12, 12]

AIPP13_OFFSET_LIST = [0, 2, 7, 8, 9, 10, 11]
AIPP13_SEGMENT_LIST = [1, 1, 1, 1, 1, 1, 1]
AIPP15_OFFSET_LIST = [0, 32]
AIPP15_SEGMENT_LIST = [30, 30]
AIPP16_OFFSET_LIST = [0, 32]
AIPP16_SEGMENT_LIST = [24, 24]
AIPP17_OFFSET_LIST = [0, 8, 16, 24, 61]
AIPP17_SEGMENT_LIST = [6, 6, 6, 6, 1]
AIPP_XS_OFFSET_LIST = [0, 16, 32, 48]
AIPP_XT_OFFSET_LIST = [0, 16, 24, 32, 45, 58]

# BIT SEG
BIT_0 = 2
BIT16 = 16
BIT_16 = 655536

# aipp input format
YUV420 = 0
XRGB8888 = 1
NC1HWC0DI_FP16 = 2
NC1HWC0DI_INT8 = 3
RGB888 = 4
ARGB8888 = 5
YUYV = 6
YUV422 = 7
AYUV444 = 8
YUV400 = 9
RAW10 = 10
RAW12 = 11
RAW16 = 12
RAW24 = 15

# function format
CROP = 1
PRE_CLIP = 2
SWAP = 4
CSC = 8
SCF = 16
POST_CLIP = 32
DTC = 64
FLIP = 128
AERA_PADDING = 256
CPADDING = 512
STRETCH = 1024
RAW = 2048

AIPP_INPUT_VERSON_AND_FUNCTION = {
    ASCEND_310AIC: {
        YUV420: {CROP, SWAP, CSC, DTC, AERA_PADDING, CPADDING},
        XRGB8888: {CROP, SWAP, CSC, DTC, AERA_PADDING, CPADDING},
        NC1HWC0DI_INT8: {CROP, SWAP, CPADDING},
        NC1HWC0DI_FP16: {CROP, SWAP, CPADDING},
        RGB888: {CROP, SWAP, CSC, DTC, AERA_PADDING, CPADDING},
        YUV400: {CROP, DTC, AERA_PADDING, CPADDING},
    },
    ASCEND_910AIC: {
        YUV420: {CROP, SWAP, CSC, DTC, AERA_PADDING, CPADDING},
        XRGB8888: {CROP, SWAP, CSC, DTC, AERA_PADDING, CPADDING},
        RGB888: {CROP, SWAP, CSC, DTC, AERA_PADDING, CPADDING},
        YUV400: {CROP, DTC, AERA_PADDING, CPADDING},
    },
    HI3796CV300ESAIC: {
        YUV420: {CROP, SWAP, CSC, DTC, AERA_PADDING, CPADDING, PRE_CLIP, SCF,
                 POST_CLIP, FLIP, STRETCH},
        XRGB8888: {CROP, SWAP, CSC, DTC, AERA_PADDING, CPADDING, PRE_CLIP, SCF,
                   POST_CLIP, FLIP, STRETCH},
        RGB888: {CROP, SWAP, CSC, DTC, AERA_PADDING, CPADDING, PRE_CLIP, SCF,
                 POST_CLIP, FLIP, STRETCH},
        ARGB8888: {CROP, SWAP, CSC, DTC, AERA_PADDING, CPADDING, PRE_CLIP, SCF,
                   POST_CLIP, FLIP, STRETCH},
        YUYV: {CROP, SWAP, CSC, DTC, AERA_PADDING, CPADDING, PRE_CLIP, SCF,
               POST_CLIP, FLIP, STRETCH},
        YUV422: {CROP, SWAP, CSC, DTC, AERA_PADDING, CPADDING, PRE_CLIP, SCF,
                 POST_CLIP, FLIP, STRETCH},
        AYUV444: {CROP, SWAP, CSC, DTC, AERA_PADDING, CPADDING, PRE_CLIP, SCF,
                  POST_CLIP, FLIP, STRETCH},
        YUV400: {CROP, DTC, AERA_PADDING, CPADDING, PRE_CLIP, SCF, POST_CLIP, FLIP,
                 STRETCH},
        RAW10: {CROP, DTC, CPADDING, STRETCH},
        RAW12: {CROP, DTC, CPADDING, STRETCH},
        RAW16: {CROP, DTC, CPADDING, STRETCH},
    },
    AIC: {
        YUV420: {CROP, SWAP, CSC, DTC, AERA_PADDING, CPADDING},
        XRGB8888: {CROP, SWAP, CSC, DTC, AERA_PADDING, CPADDING},
        NC1HWC0DI_INT8: {CROP, SWAP, CPADDING},
        NC1HWC0DI_FP16: {CROP, SWAP, CPADDING},
        RGB888: {CROP, SWAP, CSC, DTC, AERA_PADDING, CPADDING},
        YUV400: {CROP, DTC, AERA_PADDING, CPADDING},
        RAW16: {CROP, DTC, CPADDING, RAW},
        RAW24: {CROP, DTC, CPADDING, RAW},
    }
}

AIPP_INPUT_TYPE_SWAP_ALIGN = {
    YUV420: {'type': 'uint8', 'channels': 3,
             'swap': [[0], [0, 1], [0]], 'addr1_align': 2, 'addr2_align': 2,
             '1src_align': 2, 'size_channel': 1.5, 'src0_size_channel': 1,
             'src1_size_channel': 0.5, 'src0_size_bytes': 1},
    XRGB8888: {'type': 'uint8', 'channels': 3,
               'swap': [[0, 1], [0], [0, 1]],
               '1src_align': 4, 'size_channel': 4},
    NC1HWC0DI_INT8: {'type': 'int8', 'channels': 5,
                     'swap': [[0, 1], [0, 1], [0]],
                     'addr1_align': 4, 'addr2_align': 2,
                     '1src_align': 2, 'size_channel': 6, 'src0_size_channel': 4,
                     'src1_size_channel': 2, 'src0_size_bytes': 4},
    NC1HWC0DI_FP16: {'type': 'float16', 'channels': 5,
                     'swap': [[0, 1], [0, 1], [0]],
                     'addr1_align': 8, 'addr2_align': 4,
                     '1src_align': 4, 'size_channel': 12,
                     'src0_size_channel': 8, 'src1_size_channel': 4,
                     'src0_size_bytes': 8},
    RGB888: {'type': 'uint8', 'channels': 3, 'swap': [[0, 1], [0], [0]],
             '1src_align': 2, 'size_channel': 3},
    ARGB8888: {'type': 'uint8', 'channels': 4,
               'swap': [[0, 1], [0], [0, 1]],
               '1src_align': 4, 'size_channel': 4},
    YUYV: {'type': ['uint8'], 'channels': 3,
           'swap': [[0], [0, 1], [0]], '1src_align': 2, 'size_channel': 2},
    YUV422: {'type': ['uint8'], 'channels': 3,
             'swap': [[0], [0, 1], [0]], 'addr1_align': 2,
             'addr2_align': 2, '1src_align': 2,
             'size_channel': 2, 'src0_size_channel': 1,
             'src1_size_channel': 1, 'src0_size_bytes': 1},
    AYUV444: {'type': ['uint8'], 'channels': 4,
              'swap': [[0], [0, 1], [0, 1]],
              '1src_align': 4, 'size_channel': 4},
    YUV400: {'type': ['uint8'], 'channels': 1,
             '1src_align': 2, 'addr1_align': 2, 'size_channel': 1},
    RAW10: {'type': ['uint16'], 'channels': 1,
            '1src_align': 2, 'size_channel': 2},
    RAW12: {'type': ['uint16'], 'channels': 1,
            '1src_align': 2, 'size_channel': 2},
    RAW16: {'type': ['uint16'], 'channels': 1,
            '1src_align': 2, 'size_channel': 2},
    RAW24: {'type': ['uint8'], 'channels': 1,
            '1src_align': 3, 'size_channel': 3}
}

# format_convert
AIPP_FORMAT_CONVERT = {

    # YUV420
    1: {
        'csc_matrix': [[298, 0, 409], [298, -100, -208], [298, 516, 0]],
        'csc_out_bias': [0, 0, 0],
        'csc_in_bias': [16, 128, 128],
    },

    2: {
        'csc_matrix': [[298, 516, 0], [298, -100, -208], [298, 0, 409]],
        'csc_out_bias': [0, 0, 0],
        'csc_in_bias': [16, 128, 128],
    },

    3: {
        'csc_matrix': [[256, 0, 0], [0, 0, 0], [0, 0, 0]],
        'csc_out_bias': [0, 0, 0],
        'csc_in_bias': [0, 0, 0],
    },

    # xrgb
    4: {
        'csc_matrix': [[66, 129, 25], [-38, -74, 112], [112, -94, -18]],
        'csc_out_bias': [16, 128, 128],
        'csc_in_bias': [0, 0, 0],
    },

    5: {
        'csc_matrix': [[66, 129, 25], [112, -94, -18], [-38, -74, 112]],
        'csc_out_bias': [16, 128, 128],
        'csc_in_bias': [0, 0, 0],
    },

    6: {
        'csc_matrix': [[76, 150, 30], [0, 0, 0], [0, 0, 0]],
        'csc_out_bias': [0, 0, 0],
        'csc_in_bias': [0, 0, 0],
    },

    # rgb
    7: {
        'csc_matrix': [[66, 129, 25], [-38, -74, 112], [112, -94, -18]],
        'csc_out_bias': [16, 128, 128],
        'csc_in_bias': [0, 0, 0],
    },

    8: {
        'csc_matrix': [[66, 129, 25], [112, -94, -18], [-38, -74, 112]],
        'csc_out_bias': [16, 128, 128],
        'csc_in_bias': [0, 0, 0],
    },

    9: {
        'csc_matrix': [[76, 150, 30], [0, 0, 0], [0, 0, 0]],
        'csc_out_bias': [0, 0, 0],
        'csc_in_bias': [0, 0, 0],
    },

    # hisi-es
    # rgb2yuv_601_narrow
    10: {
        'csc_matrix': [[1052, 2065, 401], [-607, -1192, 1799],
                       [1799, -1506, -293]],
        'csc_out_bias': [16, 128, 128],
        'csc_in_bias': [0, 0, 0],
    },

    # rgb2yuv_601_wide
    12: {
        'csc_matrix': [[1225, 2404, 467], [-691, -1357, 2048],
                       [2048, -1715, -333]],
        'csc_out_bias': [0, 128, 128],
        'csc_in_bias': [0, 0, 0],
    },

    # rgb2yuv_709_narrow
    14: {
        'csc_matrix': [[748, 2516, 254], [-421, -1389, 1799],
                       [1799, -1634, -165]],
        'csc_out_bias': [16, 128, 128],
        'csc_in_bias': [0, 0, 0]
    },

    # rgb2yuv_709_narrow
    16: {
        'csc_matrix': [[871, 2929, 296], [-469, -1579, 2048],
                       [2048, -1860, -188]],
        'csc_out_bias': [0, 128, 128],
        'csc_in_bias': [0, 0, 0]
    },

    # yuv2rgb_601_narrow
    11: {
        'csc_matrix': [[298, 0, 409], [298, -100, -208],
                       [298, 517, 0]],
        'csc_out_bias': [0, 0, 0],
        'csc_in_bias': [16, 128, 128]
    },

    # yuv2rgb_601_wide
    13: {
        'csc_matrix': [[256, 0, 359], [256, -88, -183],
                       [256, 454, 0]],
        'csc_out_bias': [0, 0, 0],
        'csc_in_bias': [0, 128, 128]
    },

    # yuv2rgb_601_narrow
    15: {
        'csc_matrix': [[298, 0, 459], [298, -55, -137],
                       [298, 541, 0]],
        'csc_out_bias': [0, 0, 0],
        'csc_in_bias': [16, 128, 128]
    },

    # yuv2rgb_601_wide
    17: {
        'csc_matrix': [[256, 0, 403], [256, -48, -120],
                       [256, 475, 0]],
        'csc_out_bias': [0, 0, 0],
        'csc_in_bias': [0, 128, 128]
    }
}

AIPP_INIT_VALUE = 0
AIPP_ENABLE = 1
AIPP_DISABLE = 0
AIPP_INIT_FLOAT_VALUE_ZERO = 0.0
AIPP_INIT_FLOAT_VALUE_ONE = 1.0
DTC_MEAN_TYPE_UINT = 0
DTC_MEAN_TYPE_FLOAT = 1

INSPECT_RANGE = 7
MIN_START_LINE_NO = 1
CUR_FRAME_IDX = 0

CROP_BIT = 1
SWAP_BIT = 4
CSC_BIT = 8
DTC_BIT = 64
AREA_PAD_BIT = 256
CPAD_BIT = 512
PRE_CLIP_BIT = 2
SCF_BIT = 16
POST_CLIP_BIT = 32
FLIP_BIT = 128
STRETCH = 1024
RAW_BIT = 2048

REAL_SCALE_MIN = 1.0
REAL_SCALE_MAX = 256.0

SCALE_COF = 262144

RAW_TO_16_N = [0, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# vtranspose command data unit 512B, data type B16
PER_TRANSPOSE_DATA_SIZE = 256
MIN_VNCHWTRANS_STRIDE = 0
MAX_VNCHWTRANS_STRIDE = 4096
MAX_VNCHWTRANS_REPEAT_TIMES = 4096

VTRANSPOSE_REQUIRED_ELEMENT = 256

# for vec_reduce_max vec_reduce_min vec_reduce_add
MAX_VREDUCE_REPEAT_TIMES = 4096
VREDUCE_MIN_REPEAT_TIMES = 1
VREDUCE_DEFAULT_DST_REP_STRIDE = 1
VREDUCE_DEFAULT_DST_BLK_STRIDE = 1
VREDUCE_DEFAULT_SRC_BLK_STRIDE = 1
VREDUCE_DEFAULT_SRC_REP_STRIDE = 8

# vreduce per rep output count
VREDUCE_PER_REP_OUTPUT = 2

MIN_L1_H_W_C = 1
MAX_L1_H_W = 65536
MAX_L1_C = 4096
MAX_DST_GAP_WINO = 64
MAX_K_WINO = 4096
MAX_COL_INDIC = 4
WINO_FM_XM_OFFSET_LIST = [0, 16, 32, 48, 54, 56, 59]
WINO_FM_XM_SEGMENT_LIST = [16, 16, 12, 6, 2, 3, 3]
WINO_FM_XT_OFFSET_LIST = [8, 20, 32, 48]
WINO_FM_XT_SEGMENT_LIST = [12, 12, 16, 16]
MAX_REP_DIR = 2
WINO_WGT_OFFSET_LIST = [8, 16, 32, 40, 52, 54, 55, 56]
WINO_WGT_SEGMENT_LIST = [8, 16, 8, 7, 2, 1, 1, 8]

SHIFT_BIT_POS_2 = 2
SHIFT_BIT_POS_3 = 3
SHIFT_BIT_POS_7 = 7
SHIFT_BIT_POS_8 = 8
SHIFT_BIT_POS_11 = 11
SHIFT_BIT_POS_12 = 12
SHIFT_BIT_POS_16 = 16
SHIFT_BIT_POS_20 = 20
SHIFT_BIT_POS_24 = 24
SHIFT_BIT_POS_32 = 32
SHIFT_BIT_POS_47 = 47
SHIFT_BIT_POS_48 = 48
SHIFT_BIT_POS_52 = 52
SHIFT_BIT_POS_54 = 54
SHIFT_BIT_POS_55 = 55
SHIFT_BIT_POS_56 = 56
SHIFT_BIT_POS_59 = 59
SHIFT_BIT_POS_62 = 62
SHIFT_BIT_POS_63 = 63

MIN_ONTHEFLY_MODE = 1
MAX_ONTHEFLY_MODE = 3
# use to display chip support some  dtype for instr
INSTR_DTYPE_SUPPORT_STATEMENT = "current chip not support {} for Intrinsic {}"

CONST_FIVE_THREE = 1.6666667
CONST_ONE_THREE = 0.3333333
CONST_NEG_FOUR_THREE = -1.3333333
# Log value
LOG_FOUR_THREE = 0.28768207
# const value
CONST_NEG_ONE = -1.0000000
CONST_ONE = 1.0000000
CONST_HALF = 0.5000000
CONST_THREE_FOUR = 0.7500000
CONST_ONE_FIVE = 0.2000000
CONST_NEG_ONE_FOUR = -0.2500000
CONST_NEG_HALF = -0.5000000
CONST_ONE_NINE = 0.1111111
CONST_NEG_ONE_EIGHT = -0.1250000
CONST_ONE_SEVEN = 0.1428571
CONST_NEG_ONE_SIX = -0.1666667
