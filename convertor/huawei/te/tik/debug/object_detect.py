"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     object_detect.py
DESC:     debug object detect
CREATED:  2019-7-04 20:12:13
MODIFIED: 2019-7-24 14:04:45
"""
# disabling:
# R0913: too-many-arguments(__init__ function, so disable them)
from te.platform.cce_params import ASCEND_310
from te.platform.cce_params import ASCEND_910
from .statement import STMT
from .util import copy_tensor_to_model, _VEC_DATA_TYPE_ENCODING, \
    cvt_float_to_uint
from .sim.util import TempEnv
from .sim.instr_encoder import Encoder
from ..common import DTYPE_SIZE
from ..common.common_util import check_vms4_repeat_times
from ..common.tik_get_soc_name import get_soc_name
from ..tik_lib.tik_params import ADDR_BIT_LEN, EXHAUSTED_SUSPENSION_POS, \
    SRC_LIST_BIT_POS, LENGTH_BIT_LEN, LENGTH_BIAS, ALIGNED_ADDR, \
    REPEAT_SHIFT_POS, MAX_REPEAT_TIMES, MAX_MODE_NUMBER_VEXTRACT_V100, \
    MAX_MODE_NUMBER, VALID_BIT_TUPLE, VALID_BIT_TUPLE_V200,\
    MAX_REP_STRIDE_DOUBLE_BYTE, SHIFT_BIT_POS_13
from ..tik_lib.tik_check_util import TikCheckUtil

_ENCODER = Encoder()

_XN_STRIDE_SHIFT_BIT_POS = 16
_XM_STRIDE_SHIFT_BIT_POS = 32


class VBS16(STMT):
    """VBS16 instruction"""
    def __init__(self, source_info, dst, src, repeat_times):
        super(VBS16, self).__init__(source_info)
        self.dst = dst
        self.src = src[0]
        self.repeat_times = repeat_times

    def eval_(self, context):
        """run the instruction"""
        TikCheckUtil.check_equality(self.src.dtype, self.dst.dtype)
        TikCheckUtil.check_in_range(self.src.dtype, ('float16', 'float32'))
        temp_env = TempEnv()
        xn_idx, _, _, _ = copy_tensor_to_model(
            context, temp_env, self.src, ALIGNED_ADDR, access_mode='r')
        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, ALIGNED_ADDR, access_mode='w')

        repeat_time = context.evaluate_expr(self.repeat_times)
        TikCheckUtil.check_in_range(
            repeat_time, range(MAX_REPEAT_TIMES),
            "repeat_times should be in the range of [0, 255],"
            " input value is %s" % str(repeat_time))

        xt_idx = temp_env.alloc_register()
        x_t = 0
        x_t |= repeat_time << REPEAT_SHIFT_POS

        context.model.write_gpr(xt_idx, x_t)

        param = _ENCODER.new_param()
        param.type = _VEC_DATA_TYPE_ENCODING[self.src.dtype]
        param.xd = xd_idx
        param.xn = xn_idx
        param.xt = xt_idx

        instr = _ENCODER.gen_vbs16(param)
        context.model.step(instr)
        temp_env.check_mem_access(context.model)
        context.model.read_memory(
            dst_addr, self.dst.scope, dst_ptr, dst_alloc_size)


class VMS4(STMT):
    """VMS4 instruction"""
    def __init__(self, source_info, dst, src_list, element_lengths,
                 if_exhausted_suspension, valid_bit, repeat_times=0,
                 scalar_list=None):
        # pylint: disable=R0913
        super(VMS4, self).__init__(source_info)
        self.dst = dst
        self.src_list = src_list
        self.element_lengths = element_lengths
        self.exhausted_suspension = if_exhausted_suspension
        self.valid_bit = valid_bit
        self.repeat_times = repeat_times
        self.scalar_list = scalar_list

    def eval_(self, context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        TikCheckUtil.check_in_range(self.dst.dtype, ('float16', 'float32'))
        if not isinstance(self.element_lengths, (tuple, list)):
            self.element_lengths = [self.element_lengths]*4
        TikCheckUtil.check_equality(
            len(self.src_list), len(self.element_lengths))

        # check repeat_times
        element_lengths = [context.evaluate_expr(value) for value in
                           self.element_lengths]
        if isinstance(self.valid_bit, str):
            # binary dtype -> int dtype
            self.valid_bit = int(self.valid_bit, 2)

        check_vms4_repeat_times(context.evaluate_expr(self.repeat_times),
                                element_lengths,
                                context.evaluate_expr(self.valid_bit),
                                self.exhausted_suspension)

        temp_env = TempEnv()

        param, dst_addr, dst_ptr, dst_alloc_size = self.get_param(
            context, temp_env)

        instr = _ENCODER.gen_vms4(param)

        context.model.step(instr)

        temp_env.check_mem_access(context.model)
        context.model.read_memory(
            dst_addr, self.dst.scope, dst_ptr, dst_alloc_size)
        if self.scalar_list is not None:
            sr_value = context.model.read_spr('VMS4_SR')
            # vms4 bit shift
            value = 2**13 - 1
            for i, scalar in enumerate(self.scalar_list):
                temp_value = (sr_value >> (i*SHIFT_BIT_POS_13)) & value
                context.update_var(scalar.debug_var, temp_value)

    def get_param(self, context, temp_env):
        """get the gen_param and buffer information

        Parameters
        ----------
        context : the stack context
        temp_env : the temp environment

        Returns
        -------
        param : gen_param
        dst_addr, dst_ptr, dst_alloc_size : info of buffer
        """
        align = 16
        if self.dst.dtype == 'float16':
            proposal_size = 16
        else:
            proposal_size = 32
            align = 32
        xn_idx, xt_idx = self.set_register(context,
                                           temp_env, align, proposal_size)
        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, align, access_mode='w')
        param = _ENCODER.new_param()
        param.type = _VEC_DATA_TYPE_ENCODING[self.dst.dtype]
        param.xd = xd_idx
        param.xn = xn_idx
        param.xt = xt_idx

        return param, dst_addr, dst_ptr, dst_alloc_size

    def set_register(self, context, temp_env, align, proposal_size):
        """set register and get xn_idx and xt_idx

        Parameters
        ----------
        context : the stack context
        temp_env : the temp environment
        align : align address
        proposal_size : dst's dtpe size, 16 or 32

        Returns
        -------
        xn_idx : the result code
        xt_idx : the result code
        """
        xn_idx = temp_env.alloc_register()
        x_n = 0

        # check repeat_times
        TikCheckUtil.check_in_range(
            context.evaluate_expr(self.repeat_times),
            range(MAX_REPEAT_TIMES),
            "repeat_times should be in the range of [0, 255],"
            " input value is %s" % str(
                context.evaluate_expr(self.repeat_times)))

        # check valid_bit
        if get_soc_name() in (ASCEND_310, ASCEND_910):
            TikCheckUtil.check_in_range(
                context.evaluate_expr(self.valid_bit), VALID_BIT_TUPLE,
                "valid bits only support 1111, 0111 or 0011 // binary, "
                "input valid bits: {} // "
                "decimal".format(context.evaluate_expr(self.valid_bit)))
        else:
            TikCheckUtil.check_in_range(
                context.evaluate_expr(self.valid_bit), VALID_BIT_TUPLE_V200,
                "valid bits only support 1111, 0111, 0011 or 0001 // binary, "
                "input valid_bit: {} // "
                "decimal".format(context.evaluate_expr(self.valid_bit)))

        xt_idx = temp_env.alloc_register()
        x_t = context.evaluate_expr(self.repeat_times)

        for i, src in enumerate(self.src_list):
            # this will waste some gpr, but don't worry
            copy_tensor_to_model(context, temp_env,
                                 src, align, access_mode='r')
            src_addr = temp_env.get_tensor_addr(
                context, src, access_mode='r') // proposal_size
            x_n |= src_addr << (i*ADDR_BIT_LEN)
            length = self.element_lengths[i]
            length = context.evaluate_expr(length)
            x_t |= length << (i*LENGTH_BIT_LEN + LENGTH_BIAS)
        context.model.write_gpr(xn_idx, x_n)
        x_t |= self.exhausted_suspension << EXHAUSTED_SUSPENSION_POS
        x_t |= context.evaluate_expr(self.valid_bit) << SRC_LIST_BIT_POS
        context.model.write_gpr(xt_idx, x_t)

        return xn_idx, xt_idx


class VEXTRACT(STMT):
    """
    Vector Region Proposal Coordination Extraction
    """

    def __init__(self, source_info, dst, src, repeat_times, mode):
        # pylint: disable=R0913
        super(VEXTRACT, self).__init__(source_info)
        self.dst = dst
        self.src = src
        self.repeat_times = repeat_times
        self.mode = mode

    def eval_(self, context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        TikCheckUtil.check_equality(self.src.dtype, self.dst.dtype)
        TikCheckUtil.check_in_range(self.src.dtype, ('float16', 'float32'))
        temp_env = TempEnv()

        xn_idx, _, src_alloc_size, _ = copy_tensor_to_model(
            context, temp_env, self.src, ALIGNED_ADDR, access_mode='r')
        # check src overflow
        src_expected_ele = context.evaluate_expr(self.repeat_times*16*8 +
                                                 self.src.offset)
        TikCheckUtil.check_ge(
            src_alloc_size // DTYPE_SIZE[self.src.dtype], src_expected_ele,
            "src tensor overflow, expected src shape: {}, actual src "
            "shape: {}".format(src_expected_ele,
                               src_alloc_size // DTYPE_SIZE[self.src.dtype]))

        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, ALIGNED_ADDR, access_mode='w')

        param = _ENCODER.new_param()
        param.type = _VEC_DATA_TYPE_ENCODING[self.src.dtype]
        param.xn = xn_idx
        param.xd = xd_idx
        param.xt = self.create_gpr_x_t(context, temp_env)

        context.model.step(_ENCODER.gen_vextract(param))
        temp_env.check_mem_access(context.model)
        context.model.read_memory(
            dst_addr, self.dst.scope, dst_ptr, dst_alloc_size)

    def create_gpr_x_t(self, context, temp_env):
        """create general purpose register x_t

        Parameters
        ----------
        context : the stack context
        temp_env : the temp environment

        Returns
        -------
        xt_idx
        """
        repeat_time = context.evaluate_expr(self.repeat_times)
        mode = context.evaluate_expr(self.mode)
        # check repeat_time and mode
        TikCheckUtil.check_in_range(
            repeat_time, range(MAX_REPEAT_TIMES),
            "repeat_times should be in the range of [0, 255],"
            " input value is %s" % str(repeat_time))
        if get_soc_name() == ASCEND_310:
            TikCheckUtil.check_in_range(
                mode, range(MAX_MODE_NUMBER_VEXTRACT_V100),
                "mode_number should be in the range of [0, 3],"
                " input value is %s" % str(mode))
        else:
            TikCheckUtil.check_in_range(
                mode, range(MAX_MODE_NUMBER),
                "mode_number should be in the range of [0, 5],"
                " input value is %s" % str(mode))

        mode_bit_pos = 16

        xt_idx = temp_env.alloc_register()
        x_t = 0
        x_t |= repeat_time << REPEAT_SHIFT_POS
        x_t |= mode << mode_bit_pos

        context.model.write_gpr(xt_idx, x_t)
        return xt_idx


class VCONCAT(STMT):
    """
    Vector Region Proposal Coordination Concatenation
    """

    def __init__(self, source_info, dst, src, repeat_times, mode):
        # pylint: disable=R0913
        super(VCONCAT, self).__init__(source_info)
        self.dst = dst
        self.src = src
        self.repeat_times = repeat_times
        self.mode = mode

    def eval_(self, context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        TikCheckUtil.check_equality(self.src.dtype, self.dst.dtype)
        TikCheckUtil.check_in_range(self.src.dtype, ('float16', 'float32'))
        temp_env = TempEnv()
        xn_idx, _, _, _ = copy_tensor_to_model(
            context, temp_env, self.src, ALIGNED_ADDR, access_mode='r')
        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, ALIGNED_ADDR, access_mode='rw')

        param = _ENCODER.new_param()
        param.type = _VEC_DATA_TYPE_ENCODING[self.src.dtype]
        param.xd = xd_idx
        param.xn = xn_idx
        param.xt = self.create_gpr_x_t(context, temp_env)

        context.model.step(_ENCODER.gen_vconcat(param))
        temp_env.check_mem_access(context.model)
        context.model.read_memory(
            dst_addr, self.dst.scope, dst_ptr, dst_alloc_size)

    def create_gpr_x_t(self, context, temp_env):
        """create general purpose register x_t

        Parameters
        ----------
        context : the stack context
        temp_env : the temp environment

        Returns
        -------
        xt_idx
        """
        repeat_time = context.evaluate_expr(self.repeat_times)
        TikCheckUtil.check_in_range(
            repeat_time, range(MAX_REPEAT_TIMES),
            "repeat_times should be in range of [1, 255], "
            "input value is %s" % str(repeat_time))
        mode = context.evaluate_expr(self.mode)
        TikCheckUtil.check_in_range(
            mode, range(MAX_MODE_NUMBER),
            "mode_number should be in the range of [0, 5], "
            "input value is %s" % str(mode))

        mode_bit_pos = 16

        xt_idx = temp_env.alloc_register()
        x_t = 0
        x_t |= repeat_time << REPEAT_SHIFT_POS
        x_t |= mode << mode_bit_pos

        context.model.write_gpr(xt_idx, x_t)
        return xt_idx


def _set_rpn_offset(dtype, rpn_offset, context):
    """Set rpn offset"""
    rpn_offset = context.evaluate_expr(rpn_offset)
    bin_value = cvt_float_to_uint(dtype, rpn_offset)
    if dtype == 'float32':
        bin_value = bin_value << 16
    context.model.write_spr('RPN_OFFSET', bin_value)


class SetRpnOffset(STMT):
    """
    Set rpn offset debug evaluate
    """
    def __init__(self, source_info, rpn_offset):
        super(SetRpnOffset, self).__init__(source_info)
        self.rpn_offset = rpn_offset

    def eval_(self, context):
        # we only support float16 now
        _set_rpn_offset('float16', self.rpn_offset, context)


class VRPAC(STMT):
    """
    Vector Region Proposal Area Calculation
    """

    def __init__(self, source_info, dst, src_list, repeat_times):
        super(VRPAC, self).__init__(source_info)
        self.dst = dst
        self.src = src_list[0]
        self.repeat_times = repeat_times

    def eval_(self, context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        TikCheckUtil.check_equality(self.src.dtype, self.dst.dtype)
        TikCheckUtil.check_in_range(self.src.dtype, ('float16', 'float32'))
        temp_env = TempEnv()
        xn_idx, _, _, _ = copy_tensor_to_model(
            context, temp_env, self.src, ALIGNED_ADDR, access_mode='r')
        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, ALIGNED_ADDR, access_mode='w')

        repeat_time = context.evaluate_expr(self.repeat_times)
        TikCheckUtil.check_in_range(
            repeat_time, range(MAX_REPEAT_TIMES),
            "repeat_times should be in range of [1, 255],"
            " input value is %s" % str(repeat_time))

        xt_idx = temp_env.alloc_register()
        x_t = 0
        x_t |= repeat_time << REPEAT_SHIFT_POS

        context.model.write_gpr(xt_idx, x_t)

        param = _ENCODER.new_param()
        param.type = _VEC_DATA_TYPE_ENCODING[self.src.dtype]
        param.xd = xd_idx
        param.xn = xn_idx
        param.xt = xt_idx

        instr = _ENCODER.gen_vrpac(param)
        context.model.step(instr)
        temp_env.check_mem_access(context.model)
        context.model.read_memory(
            dst_addr, self.dst.scope, dst_ptr, dst_alloc_size)


class VAADD(STMT):
    """VAADD instruction"""
    def __init__(self, source_info, dst, src_list, repeat_times):
        super(VAADD, self).__init__(source_info)
        self.dst = dst
        self.src1 = src_list[0]
        self.src2 = src_list[1]
        self.repeat_times = repeat_times

    def eval_(self, context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        TikCheckUtil.check_equality(self.src1.dtype, self.src2.dtype)
        TikCheckUtil.check_equality(self.src1.dtype, self.dst.dtype)
        TikCheckUtil.check_in_range(self.src1.dtype, ('float16', 'float32'))
        temp_env = TempEnv()
        xn_idx, _, _, _ = copy_tensor_to_model(
            context, temp_env, self.src1, ALIGNED_ADDR, access_mode='r')
        xm_idx, _, _, _ = copy_tensor_to_model(
            context, temp_env, self.src2, ALIGNED_ADDR, access_mode='r')
        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, ALIGNED_ADDR, access_mode='w')

        repeat_time = context.evaluate_expr(self.repeat_times)
        TikCheckUtil.check_in_range(
            repeat_time, range(MAX_REPEAT_TIMES),
            "repeat_times should be in range of [1, 255], "
            "input value is %s" % str(repeat_time))

        xt_idx = temp_env.alloc_register()
        x_t = 0
        x_t |= repeat_time << REPEAT_SHIFT_POS

        context.model.write_gpr(xt_idx, x_t)

        param = _ENCODER.new_param()
        param.type = _VEC_DATA_TYPE_ENCODING[self.src1.dtype]
        param.xd = xd_idx
        param.xn = xn_idx
        param.xm = xm_idx
        param.xt = xt_idx

        context.model.step(_ENCODER.gen_vaadd(param))
        temp_env.check_mem_access(context.model)
        context.model.read_memory(
            dst_addr, self.dst.scope, dst_ptr, dst_alloc_size)


class VIOU(STMT):
    """VIOU instruction"""
    def __init__(self, source_info, dst, src_list, repeat_times):
        super(VIOU, self).__init__(source_info)
        self.dst = dst
        self.src1 = src_list[0]
        self.src2 = src_list[1]
        self.repeat_times = repeat_times

    def eval_(self, context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        TikCheckUtil.check_equality(self.src1.dtype, self.src2.dtype)
        TikCheckUtil.check_equality(self.src1.dtype, self.dst.dtype)
        TikCheckUtil.check_in_range(self.src1.dtype, ('float16', 'float32'))
        temp_env = TempEnv()
        xn_idx, _, _, _ = copy_tensor_to_model(
            context, temp_env, self.src1, ALIGNED_ADDR, access_mode='r')
        xm_idx, _, _, _ = copy_tensor_to_model(
            context, temp_env, self.src2, ALIGNED_ADDR, access_mode='r')
        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, ALIGNED_ADDR, access_mode='w')

        repeat_time = context.evaluate_expr(self.repeat_times)
        TikCheckUtil.check_in_range(
            repeat_time, range(MAX_REPEAT_TIMES),
            "repeat_times should be in range of [1, 255], "
            "input value is %s" % str(repeat_time))

        xt_idx = temp_env.alloc_register()
        x_t = 0
        x_t |= repeat_time << REPEAT_SHIFT_POS

        context.model.write_gpr(xt_idx, x_t)

        param = _ENCODER.new_param()
        param.type = _VEC_DATA_TYPE_ENCODING[self.src1.dtype]
        param.xd = xd_idx
        param.xn = xn_idx
        param.xm = xm_idx
        param.xt = xt_idx

        context.model.step(_ENCODER.gen_viou(param))
        temp_env.check_mem_access(context.model)
        context.model.read_memory(
            dst_addr, self.dst.scope, dst_ptr, dst_alloc_size)


class VMergeCH(STMT):
    """debug module for vmrgch instruction"""
    def __init__(self, source_info, dst, src_list, repeat_times):
        super(VMergeCH, self).__init__(source_info)
        self.dst = dst
        self.src = src_list[0]
        self.repeat_times = repeat_times

    def eval_(self, context):
        """
        run instruction
        :param context:
        :return:
        """
        temp_env = TempEnv()
        xn_idx, _, _, _ = copy_tensor_to_model(
            context, temp_env, self.src, ALIGNED_ADDR, access_mode='r')
        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, ALIGNED_ADDR, access_mode='w')

        repeat_time = context.evaluate_expr(self.repeat_times)
        TikCheckUtil.check_in_range(
            repeat_time, range(MAX_REPEAT_TIMES),
            "repeat_times should be in the range of [0, 255],"
            " input value is %s" % str(repeat_time))

        xt_idx = temp_env.alloc_register()
        x_t = 0
        x_t |= repeat_time << REPEAT_SHIFT_POS

        context.model.write_gpr(xt_idx, x_t)

        param = _ENCODER.new_param()
        param.type = _VEC_DATA_TYPE_ENCODING[self.src.dtype]
        param.xd = xd_idx
        param.xn = xn_idx
        param.xt = xt_idx

        instr = _ENCODER.gen_vmergech(param)
        context.model.step(instr)
        context.model.read_memory(
            dst_addr, self.dst.scope, dst_ptr, dst_alloc_size)


class RpnCor(STMT):
    """
    rpn condition OR for non-diagonal suppression matrix
    """

    def __init__(self, source_info, src0, src1, src0_stride, src1_stride,
                 repeat_times):
        # pylint: disable=R0913
        super(RpnCor, self).__init__(source_info)
        self.src = src0
        self.src1 = src1
        self.src_stride = src0_stride
        self.src1_stride = src1_stride
        self.repeat_times = repeat_times

    def eval_(self, context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        temp_env = TempEnv()

        xn_idx, _, _, _ = copy_tensor_to_model(
            context, temp_env, self.src, ALIGNED_ADDR, access_mode='r')
        xm_idx, _, _, _ = copy_tensor_to_model(
            context, temp_env, self.src1, ALIGNED_ADDR, access_mode='r')

        xn_stride = context.evaluate_expr(self.src_stride)
        xm_stride = context.evaluate_expr(self.src1_stride)
        repeat = context.evaluate_expr(self.repeat_times)
        # check repeat_times
        TikCheckUtil.check_in_range(
            repeat, range(MAX_REPEAT_TIMES),
            "repeat_times should in the range of [0, 255],"
            " input value is %s" % str(repeat))
        # check stride
        TikCheckUtil.check_in_range(
            xn_stride, range(MAX_REP_STRIDE_DOUBLE_BYTE),
            "src0_rep_stride should be in the range of [0,65535], "
            "input value is %s" % str(xn_stride))
        TikCheckUtil.check_in_range(
            xm_stride, range(MAX_REP_STRIDE_DOUBLE_BYTE),
            "src1_rep_stride should be in the range of [0,65535], "
            "input value is %s" % str(xm_stride))

        xt_idx = temp_env.alloc_register()
        x_t = 0
        x_t |= xn_stride << _XN_STRIDE_SHIFT_BIT_POS
        x_t |= xm_stride << _XM_STRIDE_SHIFT_BIT_POS
        x_t |= repeat << REPEAT_SHIFT_POS

        context.model.write_gpr(xt_idx, x_t)

        param = _ENCODER.new_param()
        param.xn = xn_idx
        param.xm = xm_idx
        param.xt = xt_idx

        instr = _ENCODER.gen_rpn_cor(param)
        context.model.step(instr)
        temp_env.check_mem_access(context.model)


class RpnCorDiag(STMT):
    """
    rpn condition OR for diagnal suppression matrix
    """

    def __init__(self, source_info, dst, src):
        super(RpnCorDiag, self).__init__(source_info)
        self.dst = dst
        self.src = src

    def eval_(self, context):
        """run the instruction

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        None
        """
        temp_env = TempEnv()

        xd_idx, dst_addr, dst_alloc_size, dst_ptr = copy_tensor_to_model(
            context, temp_env, self.dst, ALIGNED_ADDR, access_mode='w')
        xn_idx, _, _, _ = copy_tensor_to_model(
            context, temp_env, self.src, ALIGNED_ADDR, access_mode='r')

        param = _ENCODER.new_param()
        param.xd = xd_idx
        param.xn = xn_idx

        instr = _ENCODER.gen_rpn_cor_diag(param)

        context.model.step(instr)
        temp_env.check_mem_access(context.model)

        context.model.read_memory(
            dst_addr, self.dst.scope, dst_ptr, dst_alloc_size)
