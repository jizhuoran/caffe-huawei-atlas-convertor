"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     tik_scalar_api.py
DESC:     provide scalar instructions
CREATED:  2019-08-12 18:53:42
MODIFIED: 2019-08-12 21:29:18
"""
import sys

from te import tvm
from te.platform.cce_conf import intrinsic_check_support
from te.platform.cce_conf import api_check_support
from .tik_scalar import Scalar
from .. import debug
from ..tik_lib.tik_util import dtype_convert
from .tik_ir_builder import TikIRBuilder
from ..tik_lib.tik_check_util import TikCheckUtil
from ..tik_lib.tik_params import RPN_COR_IR
from ..tik_lib.tik_params import ONE_IR
from ..tik_lib.tik_params import INSTR_DTYPE_SUPPORT_STATEMENT
from ..tik_lib.tik_api_constants import DTYPE_MAP, ROUND_MODE_MAP
from ..tik_lib.tik_source_info import source_info_decorator


class TikScalarApi(TikIRBuilder):
    """
    Scalar Operation Api

    Single operand instruction: abs sqrt countbit0 countbit1 countleading0 conv
    Double operand instruction: max min conv
    """

    # @cond
    def __init__(self):
        super(TikScalarApi, self).__init__()
    # @endcond

    def scalar_sqrt(self, dst, src):
        r"""
        Extracts the square root of a scalar
        Description:
          Extracts the square root of a scalar:\f$dst=\sqrt{src}\f$
        Args:
          dst : Destination operand, which must be the same as the source
          operand.The following data types are supported:
            - Ascend 310 AI processor: scalar(int64)
            - Ascend 910 AI processor: scalar(int64, float32)
          src : Source operand. The following data types are supported:
            - Ascend 310 AI processor: scalar(int64) and immediate(int64)
            - Ascend 910 AI processor: scalar(int64, float32)
            and immediate(int64, float32)
        Returns:
          None
        Precautions:
          A negative value is supported. The absolute value must be obtained
          before the square root.
        Example:
            @code
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            src_scalar = tik_instance.Scalar(dtype = "int64")
            src_scalar.set_as(10)
            dst_scalar = tik_instance.Scalar(dtype = "int64")
            tik_instance.scalar_sqrt(dst_scalar, src_scalar)
            @endcode
        """
        return self._scalar_single_func('sqrt', dst, src)

    def scalar_abs(self, dst, src):
        r"""
        Obtains the absolute value of a scalar.
        Description:
          Obtains the absolute value of a scalar:
          \f$dst=\left | src \right |\f$
        Args:
          dst : Destination operand.The following data types are supported:
            - Ascend 310 AI processor: scalar(int64)
            - Ascend 910 AI processor: scalar(int64)
            - HiSilicon SoC (ES): scalar(int64/float32)
            - Ascend 610 AI processor (AI Core): scalar(int64/float32)
            - Ascend 610 AI processor (Vector Core): scalar(int64/float32)
          src : Source operand. When src is a scalar, the data types of src
          and dst must be the same.The following
          data types are supported:
            - Ascend 310 AI processor: scalar(int64) and immediate(int64)
            - Ascend 910 AI processor: scalar(int64) and immediate(int64)
            - HiSilicon SoC (ES): scalar(int64/float32) and immediate(int64)
            - Ascend 610 AI processor (AI Core): scalar(int64/float32)
            and immediate(int64)
            - Ascend 610 AI processor (Vector Core): scalar(int64/float32)
            and immediate(int64)
        Returns:
          None
        Precautions:
          None
        Examples:
            @code
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            src_scalar = tik_instance.Scalar(dtype = "int64")
            src_scalar.set_as(10)
            dst_scalar = tik_instance.Scalar(dtype = "int64")
            tik_instance.scalar_abs(dst_scalar, src_scalar)
            @endcode
        """
        return self._scalar_single_func('abs', dst, src)

    def scalar_countbit0(self, dst, src):
        """
        Counts the number of bits whose values are 0 in the 64-bit binary ...
        Description:
          Counts the number of bits whose values are 0 in the 64-bit binary
          format of the source operand bitwise.
        Args:
          dst : A scalar of type uint64, for the destination operand
          src : A scalar or an immediate of type uint64, for the source operand
        Returns:
          None
        Restrictions:
          None
        Example:
            @code
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            src_scalar = tik_instance.Scalar(dtype = "uint64")
            src_scalar.set_as(10)
            dst_scalar = tik_instance.Scalar(dtype = "uint64")
            tik_instance.scalar_countbit0(dst_scalar, src_scalar)
            @endcode
        """
        return self._scalar_single_func('bcnt0', dst, src)

    def scalar_countbit1(self, dst, src):
        """
        Counts the number of bits whose values are 1 in the 64-bit binary ...
        Description:
          Counts the number of bits whose values are 1 in the 64-bit binary
          format of the source operand bitwise.
        Args:
          dst : A scalar of type uint64, specifying the destination operand.
          src : A scalar or an immediate of type uint64, specifying
          the source operand.
        Returns:
          None
        Restrictions:
          None
        Example:
            @code
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            src_scalar = tik_instance.Scalar(dtype = "uint64")
            src_scalar.set_as(10)
            dst_scalar = tik_instance.Scalar(dtype = "uint64")
            tik_instance.scalar_countbit0(dst_scalar, src_scalar)
            @endcode
        """
        return self._scalar_single_func('bcnt1', dst, src)

    def scalar_countleading0(self, dst, src):
        """
        Counts the number of consecutive bits whose values are 0 in the 64-bit
        Description:
          Counts the number of consecutive bits whose values are 0
          in the 64-bit binary format of the source operand.
        Args:
          dst : A scalar of type uint64, specifying the destination operand.
          src : A scalar or an immediate of type uint64, specifying the
          source operand.
        Returns:
          None
        Restrictions:
          None
        Example:
            @code
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            src_scalar = tik_instance.Scalar(dtype = "uint64")
            src_scalar.set_as(10)
            dst_scalar = tik_instance.Scalar(dtype = "uint64")
            tik_instance.scalar_countbit0(dst_scalar, src_scalar)
            @endcode
        """
        return self._scalar_single_func('clz', dst, src)

    def scalar_max(self, dst, src0, src1):
        """
        Compares two source operands and returns the maximum
        Description:
          Compares two source operands and returns the maximum:
          \f$dst = max(src0, src1)\f$
        Args:
          dst : A scalar of type int64, specifying the destination operand.
          src0 : A scalar or an immediate of type int64, specifying the source
          operand 0.
          src1 : A scalar or an immediate of type int64, specifying the source
          operand 1.
        Returns:
          None
        Precautions:
          The operands must have the same data type.
        Example:
            @code
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            src0_scalar = tik_instance.Scalar(dtype = "int64",
                                        name='src0_scalar', init_value=3)
            src1_scalar = tik_instance.Scalar(dtype = "int64",
                                        name='src1_scalar', init_value=2)
            dst_scalar = tik_instance.Scalar(dtype = "int64",
                                        name='dst_scalar')
            tik_instance.scalar_max(dst_scalar, src0_scalar, src1_scalar)
            @endcode
        """
        return self._scalar_binary_func('max', dst, src0, src1)

    def scalar_min(self, dst, src0, src1):
        """
        Compares two source operands and returns the minimum
        Description:
          Compares two source operands and returns the minimum:
          \f$dst = min(src0, src1)\f$
        Args:
          dst : A scalar of type int64, specifying the destination operand.
          src0 : A scalar or an immediate of type int64, specifying
          the source operand 0.
          src1 : A scalar or an immediate of type int64, specifying
          the source operand 1.
        Returns:
          None
        Precautions:
          The operands must have the same data type.
        Example:
            @code
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            src0_scalar = tik_instance.Scalar(dtype = "int64",
                                        name='src0_scalar', init_value=3)
            src1_scalar = tik_instance.Scalar(dtype = "int64",
                                        name='src1_scalar', init_value=2)
            dst_scalar = tik_instance.Scalar(dtype = "int64",
                                        name='dst_scalar')
            tik_instance.scalar_min(dst_scalar, src0_scalar, src1_scalar)
            @endcode
        """
        return self._scalar_binary_func('min', dst, src0, src1)

    @source_info_decorator()
    @debug.scalar_conv_decorator
    def scalar_conv(self, round_mode, dst, src):
        """
        Converts the scalar precision
        Description:
          Converts the scalar precision (value) as follows:
            - int32 to float32
            - float32 to int32
            - float32 to float16
            - float16 to float32
        Args:
          dst : A scalar of type float32/float16/int32,
          for the destination operand
          round_mode : Rounding mode
            - '' or 'none': no rounding
            - 'round': rounding to the nearest even number.
            For x.5, the value gets rounded to an even number, and
            other values are rounded off.
            - 'floor': rounding down
            - 'ceil' or 'ceiling': rounding up
            - 'away-zero': rounding away from 0. For a positive number, x.x
            gets rounded to (x + 1). For a negative
            number, -x.x gets rounded to -(x - 1)
            - 'to-zero': rounding to 0
            - 'odd': rounding to the nearest odd number<br>
          Description of round_mode describes the precision conversion and the
          corresponding round_mode.
          src : A scalar of type float32/float16/int32, for the source operand
          NOTICE :  Description of round_mode:
          |  Source    |      OperandDestination   |   OperandConversion Mode|
          |  ----      |        ----               |   ----                  |
        |float32 |int32 |'round','away-zero',to-zero','floor',ceil','ceiling'|
          |  int32     |      float32              |       '', 'none'|
          |  float16   |      float32              |       '', 'none'|
          |  float32   |      float16              |       '', 'none', 'odd' |
        Returns:
          None
        Precautions:
          During the conversion, precision loss may occur.
          See the following Round_mode examples:
          |  Value| round|  floor|  ceil/ceiling|  away-zero|  to-zero|    odd|
          |  ---- |  ---- |    ---- |  ----        |  ----     |  ----  | ----|
          |   1.8 |     2   |   1   |     2    |     2       |     1     | 2 |
          |   1.5 |     2   |     1   |     2  |     2       |     1     | 1 |
          |   1.2 |     1   |     1   |     2  |     1       |     1     | 1 |
          |   0.8 |     1   |     0   |     1  |     1       |     0     | 1 |
          |   0.5 |     0   |     0   |     1  |     1       |     0     | 1 |
          |   0.2 |     0   |     0   |     1  |     0       |     0     | 0 |
          |  -0.2 |     0   |    -1   |     0  |     0       |     0     | 0 |
          |  -0.5 |     0   |    -1   |     0  |    -1       |     0     | -1 |
          |  -0.8 |    -1   |    -1   |     0  |    -1       |     0     | -1 |
          |  -1.2 |    -1   |    -2   |    -1  |    -1       |    -1     | -1 |
          |  -1.5 |    -2   |    -2   |    -1  |    -2       |    -1     | -1 |
          |  -1.8 |    -2   |    -2   |    -1  |    -2       |    -1     | 2 |
        Example:
            @code
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            src_scalar = tik_instance.Scalar(dtype="float32", init_value=10.2)
            dst_scalar = tik_instance.Scalar(dtype="int32")
            tik_instance.scalar_conv('round', dst_scalar, src_scalar)
            @endcode
        """
        # check Scalar
        TikCheckUtil.check_type_match(
            dst, Scalar, 'scalar conv dst must be a scalar')
        TikCheckUtil.check_type_match(
            src, Scalar, 'scalar conv src must be a scalar')
        # check dtype
        dtype_str = DTYPE_MAP[src.dtype] + '2' + DTYPE_MAP[
            dst.dtype] + ROUND_MODE_MAP[round_mode]
        TikCheckUtil.check_equality(api_check_support("tik." +
                                                      "scalar_conv",
                                                      dtype_str), True,
                                    INSTR_DTYPE_SUPPORT_STATEMENT.
                                    format(dtype_str, "scalar_conv"))
        # code gen
        with self.new_scope():
            # f322s32z: convert f32 to s32, any number out of s32 range
            # will be +/- s32 max number.
            # round mode = Z. round to zero(c language trunc)
            if dtype_str in ('s322f32', 'f322f16', 'f162f32', 'f322s32z'):
                self.emit(
                    tvm.call_extern(dst.dtype, "reg_set", dst.get(),
                                    dtype_convert(src, dst.dtype)), ONE_IR)
            else:
                self.emit(
                    tvm.call_extern(
                        dst.dtype, "reg_set", dst.get(),
                        tvm.call_extern(src.dtype, 'conv_' + dtype_str,
                                        src.get())), ONE_IR)

    @source_info_decorator(depth=2)
    @debug.scalar_single_decorator
    def _scalar_single_func(self, name, dst, src):
        # check Scalar
        TikCheckUtil.check_type_match(dst, Scalar, "dst should be Scalar")
        # check dtype
        # pylint: disable=W0601, C0103, W0622
        # disable it because it's to support python3
        global long
        if sys.version_info[0] >= 3:
            long = int
        if isinstance(src, Scalar):
            TikCheckUtil.check_equality(dst.dtype, src.dtype,
                                        "Intrinsic {}'s src's dtype"
                                        " should be equal to dst's dtype".
                                        format(name))
        elif not isinstance(src, (int, float, long)):
            TikCheckUtil.raise_error("not support this type of src now")

        TikCheckUtil.check_equality(intrinsic_check_support("Intrinsic_" +
                                                            name, dst.dtype),
                                    True,
                                    INSTR_DTYPE_SUPPORT_STATEMENT.
                                    format(dst.dtype, name))
        # code gen
        scalar_tmp = dtype_convert(src, dst.dtype)
        with self.new_scope():
            self.emit(
                tvm.call_extern(dst.dtype, "reg_set", dst.get(),
                                tvm.call_extern(dst.dtype, name, scalar_tmp)),
                ONE_IR)

    @source_info_decorator(depth=2)
    @debug.scalar_binary_decorator
    def _scalar_binary_func(self, name, dst, src0, src1):
        # check Scalar
        TikCheckUtil.check_type_match(dst, Scalar, "dst should be Scalar")
        # pylint: disable=W0601, C0103, W0622
        # disable it because it's to support python3
        global long
        if sys.version_info[0] >= 3:
            long = int
        # check dtype
        if isinstance(src0, Scalar):
            TikCheckUtil.check_equality(dst.dtype, src0.dtype,
                                        "Intrinsic {}'s src0's "
                                        "dtype should be equal to dst's "
                                        "dtype".format(name))
        elif not isinstance(src0, (int, float, long)):
            TikCheckUtil.raise_error("not support this type of src0 now")
        if isinstance(src1, Scalar):
            TikCheckUtil.check_equality(dst.dtype, src1.dtype,
                                        "Intrinsic {}'s src1's dtype should "
                                        "be equal to dst's dtype".
                                        format(name))
        elif not isinstance(src1, (int, float, long)):
            TikCheckUtil.raise_error("not support this type of src1 now")
        TikCheckUtil.check_equality(intrinsic_check_support("Intrinsic_" +
                                                            name, dst.dtype),
                                    True,
                                    INSTR_DTYPE_SUPPORT_STATEMENT.
                                    format(dst.dtype, name))
# code gen
        scalar_tmp0 = dtype_convert(src0, dst.dtype)
        scalar_tmp1 = dtype_convert(src1, dst.dtype)
        with self.new_scope():
            self.emit(
                tvm.call_extern(
                    dst.dtype, "reg_set", dst.get(),
                    tvm.call_extern(dst.dtype, name, scalar_tmp0,
                                    scalar_tmp1)), ONE_IR)

    #@cond
    @source_info_decorator()
    def mov_rpn_cor_ir_to_scalar(self, scalar, rpn_cor_ir):
        """
        Used to move rpn cor ir to scalar
        :param scalar:
        :param rpn_cor_ir:
        :return: None
        """
        TikCheckUtil.check_type_match(
            scalar, Scalar, "input scalar should be Scalar")
        TikCheckUtil.check_equality(
            scalar.dtype, "int64", "input scalar's dtype should be int64")
        TikCheckUtil.check_is(rpn_cor_ir, RPN_COR_IR,
                              "rpn_cor_ir should be tvm varibale RPN_COR_IR")
        with self.new_scope():
            self.emit(tvm.call_extern(
                scalar.dtype, "reg_set",
                scalar.get(), tvm.call_extern(scalar.dtype, "get_rpn_cor_ir")
                ), ONE_IR)
    #@endcond
