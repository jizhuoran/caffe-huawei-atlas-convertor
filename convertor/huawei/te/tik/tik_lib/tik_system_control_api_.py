"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     tik_system_control_api_.py
DESC:     provide system control instructions
CREATED:  2019-08-12 18:53:42
MODIFIED: 2019-08-12 21:29:18
"""
# disabling:
# E1101: no-member

from te import tvm
from te.platform.cce_params import VEC
from te.platform.cce_params import AIC
from te.platform.cce_params import ASCEND_910
from te.platform.cce_params import ASCEND_910AIC
from te.platform.cce_params import HI3796CV300ESAIC
from te.platform.cce_params import HI3796CV300CSAIC
from te.platform.cce_params import ASCEND_610
from te.platform.cce_params import ASCEND_620
from te.platform.cce_conf import api_check_support
from .tik_expr import Expr
from ..api.tik_scalar import Scalar
from .. import debug
from .tik_util import type_convert
from ..api.tik_ir_builder import TikIRBuilder
from .tik_params import MAX_ATOMIC_ADD_MODE, ATOMIC_ADD_MODE_SHIFT_POS,\
    MAX_SMALL_CHANNEL_MODE, SMALL_CHANNEL_ENABLE_SHIFT_POS, FP2INT_SHIFT_POS, \
    MAX_SYSTEM_CACHE_MODE, SYSTEM_CACHE_MODE_SHIFT_POS, MAX_FP2INT_MODE, \
    MAX_TWO_BITS_VALUE, MAX_ONE_BIT_VALUE, ATOMIC_ADD_MASK, \
    SMALL_CHANNEL_MASK, SYSTEM_CACHE_MASK, FP2INT_MASK, ONE_IR, \
    INSTR_DTYPE_SUPPORT_STATEMENT
from .tik_check_util import TikCheckUtil
from .tik_source_info import source_info_decorator
from ..common.tik_get_soc_name import get_soc_name
from ..common.tik_get_soc_name import get_soc_core_type
_MAX_OVERFLOW_STATUS = 2


class TikSysControlApi(TikIRBuilder):
    """
    Proposal Operation Api
    """
    def __init__(self):
        super(TikSysControlApi, self).__init__()

    def _mov_ctrl_spr_to_scalar(self):
        """mov all bits of CTRL register to scalar

        Parameters
        ----------
        Returns
        -------
        ctrl : CTRL register
        """
        # disable it because calls subclass method
        ctrl = self.Scalar_(dtype="uint64")  # pylint: disable=E1101
        with self.new_scope():
            self.emit(
                tvm.call_extern(ctrl.dtype, "reg_set", ctrl.get(),
                                tvm.call_extern(ctrl.dtype, "get_ctrl")),
                ONE_IR)
        return ctrl

    @source_info_decorator()
    @debug.get_ctrl_bits(ATOMIC_ADD_MODE_SHIFT_POS,
                         ATOMIC_ADD_MODE_SHIFT_POS + 2)
    def mov_atomic_add_to_scalar(self, scalar):
        """mov the atomic_add bits of CTRL register to scalar

        Parameters
        ----------
        scalar : destination scalar operation

        Returns
        -------
        scalar : scalar with the atomic_add bits
        """
        TikCheckUtil.check_equality(
            get_soc_name(), ASCEND_910,
            "this api doesn't support version: %s" % get_soc_name())
        TikCheckUtil.check_type_match(
            scalar, Scalar, "input scalar should be Scalar")
        TikCheckUtil.check_equality(
            scalar.dtype, "uint64", "scalar must be uint64")
        ctrl = self._mov_ctrl_spr_to_scalar()
        ctrl.set_as(ctrl & (MAX_TWO_BITS_VALUE << ATOMIC_ADD_MODE_SHIFT_POS))
        ctrl.set_as(ctrl >> ATOMIC_ADD_MODE_SHIFT_POS)
        scalar.set_as(ctrl)
        return scalar

    @source_info_decorator()
    @debug.get_ctrl_bits(SMALL_CHANNEL_ENABLE_SHIFT_POS)
    def mov_small_channel_to_scalar(self, scalar):
        """mov the small_channel bits of CTRL register to scalar

        Parameters
        ----------
        scalar : destination scalar operation

        Returns
        -------
        scalar : scalar with the small_channel bits
        """
        TikCheckUtil.check_equality(
            get_soc_name(), ASCEND_910,
            "this api doesn't support version: %s" % get_soc_name())
        TikCheckUtil.check_type_match(
            scalar, Scalar, "input scalar should be Scalar")
        TikCheckUtil.check_equality(
            scalar.dtype, "uint64", "scalar must be uint64")
        ctrl = self._mov_ctrl_spr_to_scalar()
        ctrl.set_as(ctrl & (MAX_ONE_BIT_VALUE <<
                            SMALL_CHANNEL_ENABLE_SHIFT_POS))
        ctrl.set_as(ctrl >> SMALL_CHANNEL_ENABLE_SHIFT_POS)
        scalar.set_as(ctrl)
        return scalar

    @source_info_decorator()
    @debug.get_ctrl_bits(SYSTEM_CACHE_MODE_SHIFT_POS,
                         SYSTEM_CACHE_MODE_SHIFT_POS + 2)
    def mov_system_cache_to_scalar(self, scalar):
        """mov the system_cache bits of CTRL register to scalar

        Parameters
        ----------
        scalar : destination scalar operation

        Returns
        -------
        scalar : scalar with the system_cache bits
        """
        TikCheckUtil.check_equality(
            get_soc_name(), ASCEND_910,
            "this api doesn't support version: %s" % get_soc_name())
        TikCheckUtil.check_type_match(
            scalar, Scalar, "input scalar should be Scalar")
        TikCheckUtil.check_equality(
            scalar.dtype, "uint64", "scalar must be uint64")
        ctrl = self._mov_ctrl_spr_to_scalar()
        ctrl.set_as(ctrl & (MAX_TWO_BITS_VALUE <<
                            SYSTEM_CACHE_MODE_SHIFT_POS))
        ctrl.set_as(ctrl >> SYSTEM_CACHE_MODE_SHIFT_POS)
        scalar.set_as(ctrl)
        return scalar

    @source_info_decorator()
    @debug.get_ctrl_bits(FP2INT_SHIFT_POS)
    def mov_fp2int_mode_to_scalar(self, scalar):
        """mov the fp2int_mode bits of CTRL register to scalar

        Parameters
        ----------
        scalar : destination scalar operation

        Returns
        -------
        scalar : scalar with the fp2int_mode bits
        """
        TikCheckUtil.check_equality(
            get_soc_name(), ASCEND_910,
            "this api doesn't support version: %s" % get_soc_name())
        TikCheckUtil.check_type_match(
            scalar, Scalar, "input scalar should be Scalar")
        TikCheckUtil.check_equality(
            scalar.dtype, "uint64", "scalar must be uint64")
        ctrl = self._mov_ctrl_spr_to_scalar()
        ctrl.set_as(ctrl & (MAX_ONE_BIT_VALUE << FP2INT_SHIFT_POS))
        ctrl.set_as(ctrl >> FP2INT_SHIFT_POS)
        scalar.set_as(ctrl)
        return scalar

    @source_info_decorator()
    @debug.set_ctrl_bits(ATOMIC_ADD_MODE_SHIFT_POS,
                         ATOMIC_ADD_MODE_SHIFT_POS + 2)
    def set_atomic_add(self, mode):
        """set mode to the atomic_add bits of CTRL register

        Parameters
        ----------
        mode : atomic add mode, 0 - disable, 1 - float32, 2 - float16

        Returns
        -------
        None
        """
        arch_version_mode = {ASCEND_910: (0, 1),
                             ASCEND_610: (0, 1, 2, 3),
                             ASCEND_620: (0, 1, 2, 3)}
        TikCheckUtil.check_type_match(
            mode, (int, Scalar), "mode should be int or Scalar")
        TikCheckUtil.check_equality(api_check_support("tik." +
                                                      "set_atomic_add",
                                                      ""), True,
                                    INSTR_DTYPE_SUPPORT_STATEMENT.
                                    format("", "set_atomic_add"))
        if isinstance(mode, int):
            TikCheckUtil.check_in_range(
                mode, arch_version_mode[get_soc_name()],
                "mode should be in the range of %s, input mode: %s" % (
                    arch_version_mode[get_soc_name()], mode))
        if isinstance(mode, Scalar):
            TikCheckUtil.check_equality(
                mode.dtype, "uint64",
                "scalar_mode should be a scalar of uint64")
        ctrl = self._mov_ctrl_spr_to_scalar()
        ctrl.set_as(ctrl & ATOMIC_ADD_MASK)
        ctrl.set_as(ctrl | (mode << ATOMIC_ADD_MODE_SHIFT_POS))

        with self.new_scope():
            self.emit(tvm.call_extern("uint64", "set_ctrl", ctrl.get()),
                      ONE_IR)

    @source_info_decorator()
    @debug.set_ctrl_bits(SMALL_CHANNEL_ENABLE_SHIFT_POS)
    def set_small_channel(self, enable):
        """set enable mode to the small_channel bit of CTRL register

        Parameters
        ----------
        enable : small_channel mode, 0 - disable, 1 - enable

        Returns
        -------
        None
        """
        TikCheckUtil.check_type_match(
            enable, (int, Scalar), "enable should be int or Scalar")
        TikCheckUtil.check_equality(
            get_soc_name(), ASCEND_910,
            "this api doesn't support version: %s" % get_soc_name())
        if isinstance(enable, int):
            TikCheckUtil.check_in_range(enable, range(MAX_SMALL_CHANNEL_MODE),
                                        "enable should be 0 or 1")
        if isinstance(enable, Scalar):
            TikCheckUtil.check_equality(
                enable.dtype, "uint64",
                "scalar_enable should be a scalar of uint64")
        ctrl = self._mov_ctrl_spr_to_scalar()
        ctrl.set_as(ctrl & SMALL_CHANNEL_MASK)
        ctrl.set_as(ctrl | (enable << SMALL_CHANNEL_ENABLE_SHIFT_POS))
        with self.new_scope():
            self.emit(tvm.call_extern("uint64", "set_ctrl", ctrl.get()),
                      ONE_IR)

    @source_info_decorator()
    @debug.set_ctrl_bits(SYSTEM_CACHE_MODE_SHIFT_POS,
                         SYSTEM_CACHE_MODE_SHIFT_POS + 2)
    def set_system_cache(self, mode):
        """set mode to the system_cache bits of CTRL register

        Parameters
        ----------
        mode : system_cache mode, 0 - Normalread, 1 - ReadLast,
               2 - ReadInvalid, 3 - NotNeedWriteBack

        Returns
        -------
        None
        """
        TikCheckUtil.check_equality(
            get_soc_name(), ASCEND_910,
            "this api doesn't support version: %s" % get_soc_name())
        TikCheckUtil.check_type_match(
            mode, (int, Scalar), "mode should be int or Scalar")
        if isinstance(mode, int):
            TikCheckUtil.check_in_range(
                mode, range(MAX_SYSTEM_CACHE_MODE),
                "mode should be in the range of [0, 3]")
        if isinstance(mode, Scalar):
            TikCheckUtil.check_equality(
                mode.dtype, "uint64",
                "scalar_mode should be a scalar of uint64")
        ctrl = self._mov_ctrl_spr_to_scalar()
        ctrl.set_as(ctrl & SYSTEM_CACHE_MASK)
        ctrl.set_as(ctrl | (mode << SYSTEM_CACHE_MODE_SHIFT_POS))
        with self.new_scope():
            self.emit(tvm.call_extern("uint64", "set_ctrl", ctrl.get()),
                      ONE_IR)

    @source_info_decorator()
    @debug.set_ctrl_bits(FP2INT_SHIFT_POS)
    def set_fp2int_mode(self, mode):
        """set mode to the fp2int_mode bit of CTRL register

        Parameters
        ----------
        mode : fp2int_mode mode, 0 - Saturated, 1 - Truncted

        Returns
        -------
        None
        """
        TikCheckUtil.check_equality(
            get_soc_name(), ASCEND_910,
            "this api doesn't support version: %s" % get_soc_name())
        TikCheckUtil.check_type_match(
            mode, (int, Scalar), "mode should be int or Scalar")
        if isinstance(mode, int):
            TikCheckUtil.check_in_range(
                mode, range(MAX_FP2INT_MODE), "mode should be 0 or 1")
        if isinstance(mode, Scalar):
            TikCheckUtil.check_equality(
                mode.dtype, "uint64",
                "scalar_mode should be a scalar of uint64")
        ctrl = self._mov_ctrl_spr_to_scalar()
        ctrl.set_as(ctrl & FP2INT_MASK)
        ctrl.set_as(ctrl | (mode << FP2INT_SHIFT_POS))
        with self.new_scope():
            self.emit(tvm.call_extern("uint64", "set_ctrl", ctrl.get()),
                      ONE_IR)

    @source_info_decorator()
    def instr_preload(self, icache_base_addr, instr_num):
        """
            instr_num must be multiple of 128/4=32
        """
        # disable it because pylint can't recognize symbols from back-end
        arch_version_str = get_soc_name() + get_soc_core_type()
        TikCheckUtil.check_in_range(
            arch_version_str, [ASCEND_910AIC, AIC, VEC, HI3796CV300ESAIC,
                               HI3796CV300CSAIC],
            "%s doesn't support instr_preload." % (arch_version_str))
        TikCheckUtil.check_type_match(
            instr_num, (int, Scalar, Expr),
            "instr_num should be int, Scalar or Expr")
        from .tik_params import INSTR_UNIT, CACHE_LINE_SIZE
        if isinstance(instr_num, int):
            TikCheckUtil.check_equality(instr_num % INSTR_UNIT, 0,
                                        "instr_num must be multiple of 32.")
        with self.new_scope():
            with self.context.freeze():  # pylint: disable=E1101
                icache_addr = self.Scalar_("uint64")  # pylint: disable=E1101
                with self.for_range_(0, instr_num // INSTR_UNIT) as i:
                    icache_addr.set_as(icache_base_addr + i*CACHE_LINE_SIZE)
                    self.emit(
                        tvm.call_extern("uint64", "preload",
                                        icache_addr.get()), ONE_IR)

    @source_info_decorator()
    def get_overflow_status(self, status):
        """get overflow status stored in scalar buffer

        Parameters
        ----------
        status : destination scalar operation

        Returns
        -------
        None
        """
        TikCheckUtil.check_equality(
            get_soc_name(), ASCEND_910,
            "this api doesn't support version: %s" % get_soc_name())
        TikCheckUtil.check_type_match(
            status, Scalar, "variable should be Scalar, invalid type "
                            "for status: {}".format(type(status)))
        TikCheckUtil.check_equality(
            status.dtype, "uint64",
            "variable should be a Scalar of uint64, invalid "
            "dtype for status: {}".format(status.dtype))
        with self.new_scope():
            # overflow status is stored on scalar buffer, the address is 0x40000
            self.emit(
                tvm.call_extern("uint64_t", "reg_set",
                                status.get(),
                                tvm.call_pure_intrin(
                                    "uint64", "tvm_cce_string_print",
                                    "*(uint64_t *) 0x40000")),
                ONE_IR)

    @source_info_decorator()
    def set_overflow_status(self, value):
        """set overflow status to scalar buffer

        Parameters
        ----------
        value : overflow status

        Returns
        -------
        None
        """
        TikCheckUtil.check_equality(
            get_soc_name(), ASCEND_910,
            "this api doesn't support version: %s" % get_soc_name())
        TikCheckUtil.check_type_match(
            value, int, "value should be Int, "
                        "invalid type: {}".format(type(value)))
        TikCheckUtil.check_in_range(
            value, range(_MAX_OVERFLOW_STATUS),
            "value should be 0 or 1, invalid value: {}".format(value))
        with self.new_scope():
            self.emit(
                tvm.call_extern("uint64_t", "set_overflow",
                                type_convert(Expr(value, dtype="uint64"))),
                ONE_IR)
