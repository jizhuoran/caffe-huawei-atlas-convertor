"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     instr_encoder_const.py
DESC:     define encode instruction constant .
CREATED:  2020-3-05 14:12:13
MODIFIED: 2020-3-05 14:12:13
"""
from ctypes import Structure, c_uint
from enum import Enum


class XRegs(Enum):
    """define the x register."""
    X0_REG = 0
    X1_REG = 1
    X2_REG = 2
    X3_REG = 3
    X4_REG = 4
    X5_REG = 5
    X6_REG = 6
    X7_REG = 7
    X8_REG = 8
    X9_REG = 9
    X10 = 10
    X11 = 11
    X12 = 12
    X13 = 13
    X14 = 14
    X15 = 15
    X16 = 16
    X17 = 17
    X18 = 18
    X19 = 19
    X20 = 20
    X21 = 21
    X22 = 22
    X23 = 23
    X24 = 24
    X25 = 25
    X26 = 26
    X27 = 27
    X28 = 28
    X29 = 29
    X30 = 30
    X31 = 31
    X32 = 32
    X33 = 33
    X34 = 34
    X35 = 35
    X36 = 36
    X37 = 37
    X38 = 38
    X39 = 39
    X40 = 40
    X41 = 41
    X42 = 42
    X43 = 43
    X44 = 44
    X45 = 45
    X46 = 46
    X47 = 47
    X48 = 48
    X49 = 49
    X50 = 50
    X51 = 51
    X52 = 52
    X53 = 53
    X54 = 54
    X55 = 55
    X56 = 56
    X57 = 57
    X58 = 58
    X59 = 59
    X60 = 60
    X61 = 61
    X62 = 62
    X63 = 63
    X_REGS_NUM_64 = 64
    INV_X_REG = 255


class VRegs(Enum):
    """define the va register."""
    VA0 = 0
    VA1 = 1
    VA2 = 2
    VA3 = 3
    VA4 = 4
    VA5 = 5
    VA6 = 6
    VA7 = 7
    V_REG_NUM = 8
    INV_V_REG = 255


class SRegs(Enum):
    """define the Spec Register."""
    PC_SPR = 0
    BLOCKID = 1
    STATUS = 2
    CTRL = 3
    PARA_BASE = 4
    DATA_MAIN_BASE = 5
    DATA_UB_BASE = 6
    DATA_SIZE = 7
    LPCNT = 8
    BLOCKDIM = 9
    FMATRIX = 10
    CONDITION_FLAG = 11
    DEQSCALE = 12
    PADDING = 13
    L2_VADDR_BASE = 14
    L0_SET_VALUE = 15
    COREID = 16
    VMS4_SR = 17
    L2_IN_MAIN = 18
    RPN_COR_IR = 19
    ARCH_VER = 20
    SMMU_TAG_VER = 21
    L1_3D_SIZE = 22
    AIPP_SPR_0 = 23
    AIPP_SPR_1 = 24
    AIPP_SPR_2 = 25
    AIPP_SPR_3 = 26
    AIPP_SPR_4 = 27
    AIPP_SPR_5 = 28
    AIPP_SPR_6 = 29
    AIPP_SPR_7 = 30
    AIPP_SPR_8 = 31
    AIPP_SPR_9 = 32
    AIPP_SPR_10 = 33
    AIPP_SPR_11 = 34
    AIPP_SPR_12 = 35
    AIPP_SPR_13 = 36
    AIPP_SPR_14 = 37
    AIPP_SPR_15 = 38
    AIPP_SPR_16 = 39
    AIPP_SPR_17 = 40
    AIPP_SPR_18 = 41
    AIPP_SPR_19 = 42
    AIPP_SPR_20 = 43
    AIPP_SPR_21 = 44
    AIPP_SPR_22 = 45
    AIPP_SPR_23 = 46
    AIPP_SPR_24 = 47
    DATA_EXP_0 = 48
    DATA_EXP_1 = 49
    DATA_EXP_2 = 50
    DATA_EXP_3 = 51
    LOW_PRE_TBL = 52
    SMASK_INDEX = 53
    K_NUM = 54
    RPN_OFFSET = 55
    FCOL2IMG = 56
    RSVD_CNT = 57
    TID = 58
    COND = 59
    PNT_COE = 60
    S_REG_NUM = 61


class GetRegMode(Enum):
    """define the Reg mode."""
    DEF = 0
    ECL = 1
    CONT = 2
    GP0 = 3
    GP1 = 4
    GP2 = 5
    GP3 = 6


class ISGMemType(Enum):
    """define ISG memory type ."""
    LOA = 0
    LOB = 1
    LOC = 2
    L1_MEM = 3
    UB_MEM = 4
    OUT_I = 5
    OUT_O = 6
    SB_MEM = 7
    SPARSETABLE = 8
    WINO = 9
    WINO_L0A = 10


class PipeType(Enum):
    """define the pipe type."""
    S_PIPE = 0
    V_PIPE = 1
    M_PIPE = 2
    MTE1 = 3
    MTE2 = 4
    MTE3 = 5
    ALL = 6
    MTE4 = 7
    MTE5 = 8


class CondOp(Enum):
    """define the Cond OP."""
    EQ_OP = 0
    NE_OP = 1
    LT_OP = 2
    GT_OP = 3
    GE_OP = 4
    LE_OP = 5


class TypeV(Enum):
    """define the data type ."""
    B8_TYPE = 0
    B16 = 1
    B32 = 2
    S8_TYPE = 3
    S32 = 4
    F16 = 5
    FMIX = 6
    F32 = 7


class ConvType(Enum):
    """define the conv type."""
    VDEQ16 = 16
    F322S32A = 32
    F322S32F = 33
    F322S32C = 34
    F322S32Z = 35
    F322S32R = 36
    F162U8A = 37
    F162U8F = 38
    F162U8C = 39
    F162U8Z = 40
    S322F32 = 41
    F322F16O = 42
    F162S4 = 43
    S42F16 = 44
    DEQ8 = 45
    VDEQ8 = 46
    DEQ16 = 47
    F322F16 = 48
    F162F32 = 49
    F162S8 = 50
    F162U8 = 51
    DEQ = 52
    F162S32F = 53
    F162S32C = 54
    F162S32R = 55
    U82F16 = 56
    S82F16 = 57
    F162S32A = 58
    F162S32Z = 59
    F162S8A = 60
    F162S8F = 61
    F162S8C = 62
    F162S8Z = 63


class IMMType(Enum):
    """define the IMM type."""
    ALL_ZEROS = 0
    ALL_ONES = 1
    ALL_FFS = 2
    ALMOST_ZERO = 3
    ALMOST_ONE = 4
    ALL_NORMAL = 5
    SEPERATE = 6


class ConvDirection(Enum):
    """define the conv direction."""
    NCHW_TO_NHWC = 0
    NHWC_TO_NCHW = 1


class InstrGenParam(Structure):  # pylint: disable=R0902
    """define instruct param struct."""
    _fields_ = [
        ('type', c_uint, 32),
        ("sub_op", c_uint, 32),
        ("xd", c_uint, 32),
        ("xt", c_uint, 32),
        ("xm", c_uint, 32),
        ("xn", c_uint, 32),
        ("xi", c_uint, 32),
        ("xj", c_uint, 32),
        ("xs", c_uint, 32),
        ("xx", c_uint, 32),
        ("xm2", c_uint, 32),
        ("xm3", c_uint, 32),
        ("spr", c_uint, 32),
        ("vad", c_uint, 32),
        ("van", c_uint, 32),
        ("vam", c_uint, 32),
        ("cond_op", c_uint, 32),
        ("post", c_uint, 32),
        ("ext", c_uint, 32),
        ("position", c_uint, 32),
        ("imm2", c_uint, 32),
        ("imm3", c_uint, 32),
        ("imm5", c_uint, 32),
        ("imm6", c_uint, 32),
        ("imm8", c_uint, 32),
        ("imm10", c_uint, 32),
        ("imm12", c_uint, 32),
        ("imm16", c_uint, 32),
        ("validreg", c_uint, 32),
        ("groupID", c_uint, 32),
        ("imm_type", c_uint, 32),
        ("pipe", c_uint, 32),
        ("t_pipe", c_uint, 32),
        ("src_mem_id", c_uint, 32),
        ("dst_mem_id", c_uint, 32),
        ("trans", c_uint, 32),
        ("conv", c_uint, 32),
        ("conv_relu", c_uint, 32),
        ("pad", c_uint, 32),
        ("csize", c_uint, 32),
        ("addrmode", c_uint, 32),
        ("ele_id", c_uint, 32),
        ("ldv_h", c_uint, 32),
        ("conv_type", c_uint, 32),
        ("srcH", c_uint, 32),
        ("dstH", c_uint, 32),
        ("h", c_uint, 32),
        ("order", c_uint, 32),
        ("out", c_uint, 32),
        ("entire", c_uint, 32),
        ("cmp_dtype", c_uint, 32),
        ("cmp_optype", c_uint, 32),
        ("StVal0", c_uint, 32),
        ("StVal1", c_uint, 32),
        ("StVal2", c_uint, 32),
        ("StVal3", c_uint, 32),
        ("REG_64BIT_EN", c_uint, 32)
    ]

    @staticmethod
    def get():
        """get the value."""
        return

    @staticmethod
    def set():
        """set the value."""
        return
