"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     tik_proposal_api_.py
DESC:     provide proposal instructions
CREATED:  2019-08-12 18:53:42
MODIFIED: 2019-08-12 21:29:18
"""
# disabling:
# E1101: no-member(inherited from super class, so disable them)
# R0913: too-many-arguments
# C0302: too-many-lines (this file is full of proposal instructions)

from te import tvm  # pylint: disable=C0302
from te.platform import cce_params
from te.platform.cce_params import ASCEND_310
from te.platform.cce_params import ASCEND_910
from te.platform.cce_params import scope_ubuf
from te.platform.cce_conf import intrinsic_check_support
from te.tik.common.util import DTYPE_SIZE, check_integer_in_range, \
    reduce_mul, get_bit_len, check_scalar_dtype, is_basic_expr
from te.tik.common.common_util import check_vector_stride
from te.tik.common.common_util import check_vms4_repeat_times
from ..api.tik_tensor import Tensor
from ..api.tik_scalar import Scalar
from .. import debug
from .tik_expr import Expr
from ..api.tik_ir_builder import TikIRBuilder
from .tik_util import type_convert, concat_params
from .tik_api_util import check_tensor_list, check_repeat_times
from .tik_params import RPN_COR_IR, VMS4_SR, RPN_OFFSET, PIPE_V, \
    MAX_MODE_NUMBER, VMS4_REGION_LIST0_POS, \
    VMS4_REGION_LIST1_POS, VMS4_REGION_LIST2_POS, VMS4_REG_BIT_ALL_ONE, \
    SRC_LIST_LEN, MAX_ELEMENTS_LEN, VALID_BIT_TUPLE, VMS4_SR_ARRAY_LEN, \
    MAX_REP_STRIDE_DOUBLE_BYTE, VEXTRACT_OFFSET_LIST, VEXTRACT_SEGMENT_LIST, \
    VCONCAT_OFFSET_LIST, VCONCAT_SEGMENT_LIST, VMRGSORT4_OFFSET_LIST, \
    VMRGSORT4_SEGMENT_LIST, RPN_COR_OFFSET_LIST, RPN_COR_SEGMENT_LIST, \
    MAX_MODE_NUMBER_VEXTRACT_V100, TWO_IR, ONE_IR, VALID_BIT_TUPLE_V200, \
    INSTR_DTYPE_SUPPORT_STATEMENT
from .tik_check_util import TikCheckUtil
from .tik_source_info import source_info_decorator
from ..common.tik_get_soc_name import get_soc_name

# one proposal contanis 8 elements
_ELEMENTS_COUNT = 8
# each repeat computes 16 proposals
_PROPOSAL_NUM = 16
# element number per proposal
_ELEMENTS_PER_PROPOSAL = 8


def _count_bit_1(num):
    """count number of bit 1 in integer num

    Parameters
    ----------
    num: input integer

    Returns
    -------
    ret: number of bit 1 in num
    """
    ret = 0
    while num > 0:
        ret += num & 1
        num = num >> 1
    return ret

def _check_vms4_dst_tensor_overflow(dst, element_count_list, valid_bit,
                                    repeat_times, if_exhausted_suspension):
    """check vms dst tensor overflow, cannot check repeat condition and
    is_exhausted_suspension = True

    Parameters
    ----------
    dst: dst tensor
    element_count_list : length of the proposal list
    valid_bit : 0001 one lines are valid
                0011 two lines are valid
                0111 three lines are valid
                1111 four lines are valid
    repeat_times: repeat_times: times of invoke this instrction
    if_exhausted_suspension : 0 not stop, 1 stop

    Returns
    -------
    None
    """
    valid_bit = Expr(valid_bit).eval_value()
    if valid_bit is None:
        return
    for value in element_count_list:
        if Expr(value).eval_value() is None:
            return
    element_count_list = [Expr(value).eval_value() for value in
                          element_count_list]
    if if_exhausted_suspension is True or valid_bit != 15 or \
            len(set(element_count_list)) != 1:
        repeat_times = 1

    index = _count_bit_1(valid_bit)
    if if_exhausted_suspension is True:
        least_expected_dst_ele = min(element_count_list[0:index])*\
                                 _ELEMENTS_PER_PROPOSAL + dst.offset
    else:
        least_expected_dst_ele = (sum(element_count_list[0:index]) +
                                  (repeat_times - 1)*sum(element_count_list))*\
                                 _ELEMENTS_PER_PROPOSAL + dst.offset

    actual_dst_ele = reduce_mul(dst.indice.origin_shape)
    least_expected_dst_ele = Expr(least_expected_dst_ele).eval_value()
    actual_dst_ele = Expr(actual_dst_ele).eval_value()
    if all(value is not None for value in
           (actual_dst_ele, least_expected_dst_ele)):
        TikCheckUtil.check_ge(
            actual_dst_ele, least_expected_dst_ele,
            "dst tensor overflow, expected elements at least: {}, actual "
            "elements: {}".format(least_expected_dst_ele, actual_dst_ele))


def _check_vms4_src_tensor_overflow(src_list, element_count_list, valid_bit,
                                    repeat_times, if_exhausted_suspension):
    """check vms src tensor overflow, cannot check repeat condition

    Parameters
    ----------
    src_list : source operation list
    element_count_list : length of the proposal list
    valid_bit : 0001 one lines are valid
                0011 two lines are valid
                0111 three lines are valid
                1111 four lines are valid
    repeat_times: repeat_times: times of invoke this instrction
    if_exhausted_suspension : 0 not stop, 1 stop

    Returns
    -------
    None
    """
    valid_bit = Expr(valid_bit).eval_value()
    if valid_bit is None:
        return
    for value in element_count_list:
        if Expr(value).eval_value() is None:
            return
    element_count_list = [Expr(value).eval_value() for value in
                          element_count_list]
    if if_exhausted_suspension is True or valid_bit != 15 or \
            len(set(element_count_list)) != 1:
        repeat_times = 1
    # valid_bit(binary) contains 1/2/3/4 bit1, check 1/2/3/4 src tensor
    index = 0
    while valid_bit > 0:
        actual_src_ele = reduce_mul(src_list[index].indice.origin_shape)
        expected_src_ele = (element_count_list[index] + (repeat_times - 1)*
                            sum(element_count_list))*_ELEMENTS_PER_PROPOSAL + \
                           src_list[index].offset

        actual_src_ele = Expr(actual_src_ele).eval_value()
        expected_src_ele = Expr(expected_src_ele).eval_value()
        if all(value is not None for value in
               (actual_src_ele, expected_src_ele)):
            TikCheckUtil.check_ge(
                actual_src_ele, expected_src_ele,
                "src_list[{}] tensor overflow, expected elements: {}, "
                "actual elements: {}".format(index, expected_src_ele,
                                             actual_src_ele))
        valid_bit = valid_bit >> 1
        index += 1


def _check_overflow(tensor, extent, tensor_name):
    """check tensor overflow

    Parameters
    ----------
    tensor : tensor operation
    extent : max offset of calculate tensor
    tensor_name : operation name

    Returns
    -------
    None
    """
    total_size = reduce_mul(tensor.indice.origin_shape)
    offset = tensor.offset
    total_offset = Expr(offset + extent).eval_value()
    if total_offset is not None:
        TikCheckUtil.check_ge(
            total_size, total_offset,
            "{} tensor overflow, expected elements: {}, actual elements: {}"
            .format(tensor_name, total_offset, total_size))


def _check_special_intrin_func_overflow(name, dst, src_list, repeat_times):
    """for proposal api, check tensor overflow

    Parameters
    ----------
    name : instruction name
    dst : destination operator
    src_list : the list of source operation
    repeat_times : Repeated iterations times

    Returns
    -------
    None
    """
    if Expr(repeat_times).eval_value() is None:
        return
    if name == "vaadd":
        _check_overflow(dst, repeat_times*_PROPOSAL_NUM*_PROPOSAL_NUM,
                        "dst")
        _check_overflow(src_list[0], repeat_times*_PROPOSAL_NUM, "src0")
        _check_overflow(src_list[1], _PROPOSAL_NUM, "src1")
    elif name == "viou":
        _check_overflow(dst, repeat_times*_PROPOSAL_NUM*_PROPOSAL_NUM,
                        "dst")
        _check_overflow(src_list[0],
                        repeat_times*_PROPOSAL_NUM*_ELEMENTS_COUNT, "src0")
        _check_overflow(src_list[1], _PROPOSAL_NUM*_ELEMENTS_COUNT, "src1")
    elif name == "vconcat":
        _check_overflow(dst, repeat_times*_PROPOSAL_NUM*_ELEMENTS_COUNT,
                        "dst")
        _check_overflow(src_list[0], repeat_times*_PROPOSAL_NUM, "src")
    elif name == "vrpac":
        _check_overflow(dst, repeat_times*_PROPOSAL_NUM, "dst")
        _check_overflow(src_list[0],
                        repeat_times*_PROPOSAL_NUM*_ELEMENTS_COUNT, "src")
    elif name == "vbitsort":
        _check_overflow(dst, repeat_times*_PROPOSAL_NUM*_ELEMENTS_COUNT, "dst")
        _check_overflow(src_list[0],
                        repeat_times*_PROPOSAL_NUM*_ELEMENTS_COUNT, "src")


def addr_array_make(tik_instance, tensor_list):
    """help generate the input array in VA mode.
    tensor_list -> address_list

    Parameters
    ----------
    tik_instance : tik
    tensor_list : the list of tensor

    Returns
    -------
    list
    """
    TikCheckUtil.check_type_match(tensor_list, (list, tuple),
                                  "tensor_list shoulde be list or tuple")
    addr_list = []
    scope = None
    dtype = None
    for i in tensor_list:
        TikCheckUtil.check_type_match(i, Tensor,
                                      "element of tensor_list must be Tensor")
        if scope is None:
            scope = i.scope
        else:
            TikCheckUtil.check_equality(scope, i.scope,
                                        "scope should be equal to each other")
        if dtype is None:
            dtype = i.dtype
        else:
            TikCheckUtil.check_equality(dtype, i.dtype,
                                        "dtype should be equal to each other")
        addr_list.append(i.access_ptr("r"))

    tmp_node = tvm.make.Evaluate(tvm.call_extern(dtype, "addr_array", scope,
                                                 "addr_array_" +
                                                 TikProposalApi.
                                                 get_vm4_value(True),
                                                 *addr_list))
    tik_instance.source_info.set_node_loc(tmp_node)
    return tmp_node


class TikProposalApi(TikIRBuilder):
    """
    Proposal Operation Api
    """

    # used to define attr vmrgsort4 value
    vmrgsort4_attr_value = 0

    def __init__(self):
        super(TikProposalApi, self).__init__()

    @staticmethod
    def get_vm4_value(is_addr_array_make):
        """
        if is_addr_array_make, value should add 1
        used for vmrgsort4
        :param is_addr_array_make:
        :return: current vmrgsort4 value
        """
        if is_addr_array_make:
            TikProposalApi.vmrgsort4_attr_value += 1
            return str(TikProposalApi.vmrgsort4_attr_value)
        return str(TikProposalApi.vmrgsort4_attr_value)

    @source_info_decorator(depth=2)
    @debug.object_special_decorator
    def _special_intrin_func(self, name, dst, src_list, repeat_times):
        # check repeat_times
        check_repeat_times(repeat_times)
        # check dst tensor info
        TikCheckUtil.check_type_match(dst, Tensor, "dst should be tensor")
        # check tensor scope
        TikCheckUtil.check_equality(dst.scope, scope_ubuf,
                                    "dst's scope must be UB")
        # check tensor dtype
        TikCheckUtil.check_type_match(src_list, (list, tuple),
                                      "src_list should be list or tuple")
        for src in src_list:
            TikCheckUtil.check_type_match(src, Tensor, "src should be tensor")
            TikCheckUtil.check_equality(src.scope, scope_ubuf,
                                        "src's scope must be UB")
            TikCheckUtil.check_equality(dst.dtype, src.dtype,
                                        "Intrinsic {}'s src's dtype should "
                                        "be equal to dst's dtype".format(name))

        TikCheckUtil.check_equality(intrinsic_check_support("Intrinsic_" +
                                                            name, dst.dtype),
                                    True,
                                    INSTR_DTYPE_SUPPORT_STATEMENT.
                                    format(dst.dtype, name))
        # check tensor overflow
        _check_special_intrin_func_overflow(name, dst, src_list, repeat_times)

        self._generate_code_intrin_func(name, dst, src_list, repeat_times)

    def _generate_code_intrin_func(self, name, dst, src_list, repeat_times):
        """generate code

        Parameters
        ----------
        name : The function name.
        dst : destination operator
        src : source operation list
        repeat_times : Repeated iterations times

        Returns
        -------
        None
        """
        from .tik_params import OBJECT_SPECIAL_OFFSET_LIST, \
            OBJECT_SPECIAL_SEGMENT_LIST
        if name == "vmergech":
            if get_bit_len(dst.dtype) == 16:
                dst = dst.reinterpret_cast_to("float16")
                src_list = [src.reinterpret_cast_to("float16")
                            for src in src_list]
            else:
                dst = dst.reinterpret_cast_to("uint8")
                src_list = [src.reinterpret_cast_to("uint8")
                            for src in src_list]
        # code gen
        if name == "vrpac":
            # dst_extent: repeat_times * 16 elements/proposal * dtype_size
            # src_extent: repeat_time * 16 proposal * 8 element/proposal *
            #             dtype_size
            dst_extent = Expr(repeat_times*16*DTYPE_SIZE[dst.dtype])
            src_extent = [Expr(repeat_times*16*8*DTYPE_SIZE[src_list[0].dtype])]
        elif name == "vbitsort":
            dst_extent = Expr(repeat_times*16*8*DTYPE_SIZE[dst.dtype])
            src_extent = [Expr(repeat_times*16*8*DTYPE_SIZE[src_list[0].dtype])]
        elif name == "viou":
            # dst_extent: repeat_times * 256 intersection area * dtype_size
            # src_extent: 16 region proposals are continous in unified buffer
            #             repeat_times * 16 region proposals *
            #             8 element/proposal * dtype_size
            dst_extent = Expr(repeat_times*256*DTYPE_SIZE[dst.dtype])
            src_extent = [Expr(repeat_times*16*8*DTYPE_SIZE[src_list[0].dtype]),
                          Expr(16*8*DTYPE_SIZE[src_list[1].dtype])]
        elif name == "vaadd":
            # dst_extent, src_extent same as "viou"
            src_extent = [Expr(repeat_times*16*DTYPE_SIZE[src_list[0].dtype]),
                          Expr(16*DTYPE_SIZE[src_list[1].dtype])]
            dst_extent = Expr(repeat_times*256*DTYPE_SIZE[dst.dtype])
        else:  # name == "vmergech":
            if dst.dtype in ["float16", "uint16", "int16"]:
                # extract valid 8B from each 32B for b16
                dst_extent = Expr(repeat_times*8)
            else: # uint8 int8
                # extract valid 4B from each 32B for b8
                dst_extent = Expr(repeat_times*4)
            src_extent = [Expr(repeat_times*32)]
        tensor_addr = [dst.access_ptr("w", extent=dst_extent.get())]
        TikCheckUtil.check_equality(
            len(src_list), len(src_extent),
            "length of src_list should be equal to length of src_extent")
        for index, tmp_src in enumerate(src_list):
            tensor_addr.append(tmp_src.access_ptr(
                "r", extent=src_extent[index].get()))
        config = concat_params([repeat_times],
                               OBJECT_SPECIAL_OFFSET_LIST,
                               OBJECT_SPECIAL_SEGMENT_LIST)
        with self.new_scope():
            instr = tvm.call_extern(dst.dtype, name,
                                    *type_convert(tensor_addr + [config]))
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            self.emit(instr, ONE_IR)

    def vmrgch(self, dst, src, repeat_times):
        """Keep the first 4 data, the rest of the data is removed

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst : destination operator
        src : source operation
        repeat_times : Repeated iterations times

        Returns
        -------
        None
        """
        return self._special_intrin_func('vmergech', dst, [src], repeat_times)

    def vrpac(self, dst, src, repeat_times):
        """Calculate the area of the proposal

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst : destination operator
        src : source operation
        repeat_times : Repeated iterations times

        Returns
        -------
        None
        """
        return self._special_intrin_func('vrpac', dst, [src], repeat_times)

    def vaadd(self, dst, src0, src1, repeat_times):
        """Find the sum of the two proposal areas

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst : destination operator
        src : source operation
        repeat_times : Repeated iterations times

        Returns
        -------
        None
        """
        return self._special_intrin_func('vaadd', dst, [src0, src1],
                                         repeat_times)

    def viou(self, dst, src0, src1, repeat_times):
        """Find the intersection area of two proposals

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst : destination operator
        src : source operation
        repeat_times : Repeated iterations times

        Returns
        -------
        None
        """
        return self._special_intrin_func('viou', dst, [src0, src1],
                                         repeat_times)

    @source_info_decorator()
    @debug.vextract_decorator
    def vextract(self, dst, src, repeat_times, mode_number):
        """Extract the corresponding element from the proposal

        Parameters
        ----------
        dst:destination tensor
        src:source tensor
        mode_number: 0: x1, 1: y1, 2: x2, 3: y2, 4: score, 5:label
        repeat_times:[1,255]

        Returns
        -------
        None
        """
        # check dst src
        TikCheckUtil.check_type_match(dst, Tensor, "dst should be tensor")
        TikCheckUtil.check_equality(
            dst.scope, scope_ubuf,
            "dst scope should be ub, input scope: {}".format(dst.scope))
        TikCheckUtil.check_type_match(src, Tensor, "src should be tensor")
        TikCheckUtil.check_equality(
            src.scope, scope_ubuf,
            "src scope should be ub, input scope: {}".format(src.scope))
        # check repeat_times
        check_repeat_times(repeat_times)
        TikCheckUtil.check_type_match(
            mode_number, (int, Scalar, Expr),
            "mode_number should be int, Scalar or Expr")
        check_scalar_dtype(mode_number,
                           "scalar_mode_number should be a scalar of int/uint")
        if get_soc_name() == ASCEND_310:
            check_integer_in_range(
                mode_number, range(MAX_MODE_NUMBER_VEXTRACT_V100),
                "mode_number should be in the range of [0, 3],"
                " input value is %s" % str(mode_number))
        else:
            check_integer_in_range(
                mode_number, range(MAX_MODE_NUMBER),
                "mode_number should be in the range of [0, 5],"
                " input value is %s" % str(mode_number))

        # check dtype
        TikCheckUtil.check_equality(dst.dtype, src.dtype,
                                    "Intrinsic {}'s src's dtype should "
                                    "be equal to dst's dtype".
                                    format("vextract"))
        TikCheckUtil.check_equality(intrinsic_check_support("Intrinsic_" +
                                                            "vextract",
                                                            dst.dtype), True,
                                    INSTR_DTYPE_SUPPORT_STATEMENT.
                                    format(dst.dtype, "vextract"))
    # check dst src tensor overflow
        # in 1 repeat, read 16 proposals in which there're 8 elements.
        # write 16 elements
        if all(Expr(value).eval_value() is not None for value
               in (repeat_times, src.offset)):
            src_expected_size = Expr(repeat_times*16*8 +
                                     src.offset).eval_value()
            TikCheckUtil.check_ge(
                reduce_mul(src.indice.origin_shape), src_expected_size,
                "src tensor overflow, expected src shape: {}, actual src "
                "shape: {}".format(src_expected_size,
                                   reduce_mul(src.indice.origin_shape)))
        if all(Expr(value).eval_value() is not None for value
               in (repeat_times, dst.offset)):
            dst_expected_size = Expr(repeat_times*16 + dst.offset).eval_value()
            TikCheckUtil.check_ge(
                reduce_mul(dst.indice.origin_shape),
                dst_expected_size,
                "dst tensor overflow, expected dst shape: {}, actual dst "
                "shape: {}".format(dst_expected_size + dst.offset,
                                   reduce_mul(dst.indice.origin_shape)))
        # code gen
        params = [mode_number, repeat_times]
        config = concat_params(params, VEXTRACT_OFFSET_LIST,
                               VEXTRACT_SEGMENT_LIST)
        # extracts 16 region proposals coordination,
        # and merge result into one 32B, each result occupies dtype_size Byte
        # 8 elements/proposal, dtype_size
        dst_extent = Expr(repeat_times*16*DTYPE_SIZE[dst.dtype])
        src_extent = Expr(repeat_times*16*8*DTYPE_SIZE[src.dtype])
        with self.new_scope():
            instr = tvm.call_extern(dst.dtype, "vextract",
                                    dst.access_ptr("w",
                                                   extent=dst_extent.get()),
                                    src.access_ptr("r",
                                                   extent=src_extent.get()),
                                    type_convert(config))
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            self.emit(instr, ONE_IR)

    # special intrin of NMS
    @source_info_decorator()
    @debug.vconcate_decorator
    def vconcat(self, dst, src, repeat_times, mode_number):
        """Contrary to vextract, the elements are merged into the
        corresponding position of the proposal

        Parameters
        ----------
        dst:destination tensor
        src:source tensor
        mode_number: 0: x1, 1: y1, 2: x2, 3: y2, 4: score, 5:label
        repeat_times:[1,255]

        Returns
        -------
        None
        """
        # check dst tensor and scope
        TikCheckUtil.check_type_match(dst, Tensor,
                                      "dst should be tensor, input type: {}"
                                      .format(type(dst)))
        TikCheckUtil.check_equality(
            dst.scope, scope_ubuf, "dst's scope must be UB, input scope: {}"
            .format(dst.scope))
        # check src tensor and scope
        TikCheckUtil.check_type_match(
            src, Tensor, "src should be tensor, input type: {}"
            .format(type(src)))
        TikCheckUtil.check_equality(
            src.scope, scope_ubuf, "src's scope must be UB, input scope: {}"
            .format(src.scope))
        # check repeat_times
        check_repeat_times(repeat_times)
        TikCheckUtil.check_type_match(
            mode_number, (int, Scalar, Expr),
            "mode_number should be int, Scalar or Expr, input type: {}"
            .format(type(mode_number)))
        check_scalar_dtype(mode_number,
                           "scalar_mode_number should be a scalar of int/uint")
        check_integer_in_range(mode_number, range(MAX_MODE_NUMBER),
                               "mode_number should be in the range of [0, 5], "
                               "input value is %s" % str(mode_number))
        # check dtype
        TikCheckUtil.check_equality(dst.dtype, src.dtype,
                                    "Intrinsic {}'s src's dtype "
                                    "should be equal to dst's dtype".
                                    format("vconcat"))
        TikCheckUtil.check_equality(intrinsic_check_support("Intrinsic_"
                                                            + "vconcat",
                                                            dst.dtype), True,
                                    INSTR_DTYPE_SUPPORT_STATEMENT.
                                    format(dst.dtype, "vconcat"))
        # check tensor overflow
        _check_special_intrin_func_overflow("vconcat", dst, [src], repeat_times)
        # code gen
        params = [mode_number, repeat_times]
        offset_list = VCONCAT_OFFSET_LIST
        segment_list = VCONCAT_SEGMENT_LIST
        config = concat_params(params, offset_list, segment_list)
        # splits 16 numbers into 16 region proposals
        # 8 elements/proposal, dtype_size
        src_extent = Expr(repeat_times*16*DTYPE_SIZE[src.dtype])
        dst_extent = Expr(repeat_times*16*8*DTYPE_SIZE[dst.dtype])
        with self.new_scope():
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            instr = tvm.call_extern(dst.dtype, "vconcat",
                                    dst.access_ptr("w",
                                                   extent=dst_extent.get()),
                                    src.access_ptr("r",
                                                   extent=src_extent.get()),
                                    type_convert(config))
            self.emit(instr, ONE_IR)

    # special intrin of SORT
    @source_info_decorator()
    @debug.vms_decarator
    def vmrgsort4(self,  # pylint: disable=R0913
                  dst,
                  src_list,
                  element_count_list,
                  if_exhausted_suspension,
                  valid_bit,
                  repeat_times=1,
                  vms4_sr_scalar_array=None):
        """Arrange and merge multiple (up to four) proposal queues
                that have been queued into one queue

        Parameters
        ----------
        dst : destination operation
        src_list : source operation list
        element_count_list : length of the proposal list
        if_exhausted_suspension : 0 not stop, 1 stop
        valid_bit : 0011 two lines are valid
                    0111 three lines are valid
                    1111 four lines are valid
        repeat_times: times of invoke this instrction
        vms4_sr_scalar_array: list consist of 4 scalar, as a return value

        Returns
        -------
        None or scalar_list
        """
        # check dtype
        TikCheckUtil.check_type_match(
            dst, Tensor, "dst should be tensor, input type of dst: {}".format(
                type(dst)))
        TikCheckUtil.check_equality(
            dst.scope, scope_ubuf, "dst scope should be ub, "
                                   "input scope: {}".format(dst.scope))
        # check src_list type, scope, dtype
        check_tensor_list([src_list], ["src_list"])
        # check src_list
        TikCheckUtil.check_type_match(
            src_list, (list, tuple),
            "src_list should be list or tuple, input type of src_list: {}"
            .format(type(src_list)))
        TikCheckUtil.check_equality(
            len(src_list), SRC_LIST_LEN,
            "the length of src_list should be 4, length of input src_list: %s."
            % len(src_list))
        # src_list dtype are same, choose src_list[0].dtype as src_dtype
        TikCheckUtil.check_equality(dst.dtype, src_list[0].dtype,
                                    "Intrinsic {}'s src's dtype should"
                                    " be equal to dst's dtype".
                                    format("vmrgsort4"))
        TikCheckUtil.check_equality(intrinsic_check_support("Intrinsic_"
                                                            + "vmrgsort4",
                                                            dst.dtype), True,
                                    INSTR_DTYPE_SUPPORT_STATEMENT.
                                    format(dst.dtype, "vmrgsort4"))

        # check element_lengths
        TikCheckUtil.check_type_match(
            element_count_list, (tuple, list, int, Scalar, Expr),
            "element_count_list should be tuple, list, int, Scalar or Expr, "
            "input type of element_count_list: {}"
            .format(type(element_count_list)))
        if isinstance(element_count_list, (tuple, list)):
            TikCheckUtil.check_equality(
                len(element_count_list), len(src_list),
                "length of input element_count_list(length: {}) should be equal"
                " to length of src_list(length: {})"
                .format(len(element_count_list), len(src_list)))
            for index in range(SRC_LIST_LEN):
                TikCheckUtil.check_type_match(
                    element_count_list[index], (int, Scalar, Expr),
                    "element_count_list[{}] should be int, Scalar or Expr, "
                    "input type: {}".format(index,
                                            type(element_count_list[index])))
        else:
            element_count_list = [element_count_list]*len(src_list)

        for index in range(len(src_list)):
            check_integer_in_range(
                element_count_list[index], range(MAX_ELEMENTS_LEN),
                "element_count[{}] should be in the range of [0, {}]"
                .format(index, MAX_ELEMENTS_LEN - 1))
        # check valid_bit
        TikCheckUtil.check_type_match(
            valid_bit, (int, str, Scalar),
            "valid_bit should be int or str or Scalar, input type of valid_bit:"
            " {}".format(type(valid_bit)))
        check_scalar_dtype(valid_bit,
                           "scalar_valid_bit should be a scalar of int/uint")
        if not isinstance(valid_bit, Scalar):
            if isinstance(valid_bit, str):
                # binary dtype -> int dtype
                valid_bit = int(valid_bit, 2)
            if get_soc_name() in (ASCEND_310, ASCEND_910):
                TikCheckUtil.check_in_range(
                    valid_bit, VALID_BIT_TUPLE,
                    "valid bits only support 1111, 0111 or 0011 // binary, "
                    "input valid bits: {} // decimal".format(valid_bit))
            TikCheckUtil.check_in_range(
                valid_bit, VALID_BIT_TUPLE_V200,
                "valid bits only support 1111, 0111, 0011 or 0001 // binary, "
                "input valid_bit: {} // decimal".format(valid_bit))
        # check if_exhausted_suspension
        TikCheckUtil.check_type_match(
            if_exhausted_suspension, bool,
            "if_exhausted_suspension should be bool, input type of "
            "if_exhausted_suspension: {}".format(type(if_exhausted_suspension)))

        # check repeat_times
        check_repeat_times(repeat_times)
        check_vms4_repeat_times(repeat_times, element_count_list, valid_bit,
                                if_exhausted_suspension)

        # 1 need suspend; 0 don't need suspend
        exh_susp_number = 1 if if_exhausted_suspension else 0

        # check tensor overflow
        _check_vms4_src_tensor_overflow(src_list, element_count_list, valid_bit,
                                        repeat_times, if_exhausted_suspension)
        _check_vms4_dst_tensor_overflow(dst, element_count_list, valid_bit,
                                        repeat_times, if_exhausted_suspension)

        # code gen
        # element_count_list has four proposal, 0-3 is idx of four proposal
        params = [
            repeat_times,
            element_count_list[0], element_count_list[1],
            element_count_list[2], element_count_list[3],
            exh_susp_number, valid_bit
        ]
        self._generate_code_vmrgsort4(params, dst, src_list)
        # read vms4_sr
        if vms4_sr_scalar_array is None:
            return VMS4_SR
        TikCheckUtil.check_type_match(
            vms4_sr_scalar_array, (list, tuple),
            "vms4_sr_scalar_array should be list or tuple, input type of "
            "vms4_sr_scalar_array: {}".format(type(vms4_sr_scalar_array)))
        TikCheckUtil.check_ge(
            len(vms4_sr_scalar_array), VMS4_SR_ARRAY_LEN,
            "length of vms4_sr_scalar_array should contain 4 scalars at least, "
            "input length of vms4_sr_scalar_array: {}"
            .format(len(vms4_sr_scalar_array)))
        if not is_basic_expr(vms4_sr_scalar_array):
            TikCheckUtil.raise_error("vms4_sr_scalar_array should be scalar")
        if not if_exhausted_suspension:
            TikCheckUtil.raise_error("vms4_sr can't be read in "
                                     "non-exhausted mode.")
        with self.new_scope():
            scalar_tmp = self.Scalar_("int64",  # pylint: disable=E1101
                                      init_value=0)
            self.emit(
                tvm.call_extern(
                    scalar_tmp.dtype, "reg_set", scalar_tmp.get(),
                    tvm.call_extern(scalar_tmp.dtype, "get_vms4_sr")), ONE_IR)
            # vms4_sr_scalar_array has four scalar element,
            # 0-3 is idx of scalar array
            with self.context.freeze():  # pylint: disable=E1101
                vms4_sr_scalar_array[3].set_as(
                    (scalar_tmp >> VMS4_REGION_LIST0_POS) &
                    VMS4_REG_BIT_ALL_ONE)
                vms4_sr_scalar_array[2].set_as(
                    (scalar_tmp >> VMS4_REGION_LIST1_POS) &
                    VMS4_REG_BIT_ALL_ONE)
                vms4_sr_scalar_array[1].set_as(
                    (scalar_tmp >> VMS4_REGION_LIST2_POS) &
                    VMS4_REG_BIT_ALL_ONE)
                vms4_sr_scalar_array[0].set_as(
                    scalar_tmp & VMS4_REG_BIT_ALL_ONE)
        return None

    def _generate_code_vmrgsort4(self, params, dst, src_list):
        """generate code

        Parameters
        ----------
        params: list of param
        dst : destination operation
        src_list : source operation list

        Returns
        -------
        None
        """
        offset_list = VMRGSORT4_OFFSET_LIST
        segment_list = VMRGSORT4_SEGMENT_LIST
        config = concat_params(params, offset_list, segment_list)
        src_array = addr_array_make(self, src_list)
        with self.new_scope():
            instr = tvm.make.Evaluate(
                tvm.call_extern(dst.dtype,
                                "vmrgsort4",
                                dst.access_ptr("w"),
                                tvm.call_pure_intrin("uint64",
                                                     "tvm_cce_string_print",
                                                     "addr_array_" +
                                                     TikProposalApi.
                                                     get_vm4_value(False)),
                                type_convert(config)))
            self.source_info.set_node_loc(instr)
            instr_block = tvm.make.Block(src_array, instr)
            self.source_info.set_node_loc(instr_block)
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            self.scope_attr(cce_params.CCE_AXIS, "mem_access_scope",
                            tvm.call_extern("int64", "__vmrgsort4__",
                                            "addr_array"))
            self.emit(instr_block, TWO_IR)

    @source_info_decorator()
    @debug.vms4_to_scalar_decorator
    def mov_vmrgsort4_sr_to_scalar(self, scalar_list, vms4_sr):
        """get finished elements of 4 lists a scalar-array.

        Parameters
        ----------
        scalar_list: a list of Scalar

        vms4_sr: tik_params's VMS4 SR

        Returns
        ----------
        the list which have 4scalar
        """
        from .tik_params import SCALAR_LIST_LEN
        TikCheckUtil.check_is(
            vms4_sr, VMS4_SR,
            "Please assure the vms4_sr is the return of vmrgsort4.")
        TikCheckUtil.check_type_match(scalar_list, list,
                                      "scalar_list should be list")
        TikCheckUtil.check_ge(
            len(scalar_list), SCALAR_LIST_LEN,
            "Please specify a list of scalar at least 4 elements.")
        for i in range(SCALAR_LIST_LEN):
            TikCheckUtil.check_type_match(
                scalar_list[i], Scalar,
                "scalar_list[%d] should be scalar" % (i))
            TikCheckUtil.check_in_range(
                scalar_list[i].dtype, ('int64',),
                "scalar_list[%d] should be a Scalar of int64." % (i))
        with self.new_scope():
            self.emit(tvm.call_extern(
                scalar_list[0].dtype, "reg_set",
                scalar_list[0].get(),
                tvm.call_extern(scalar_list[0].dtype, "get_vms4_sr")
            ), ONE_IR)
            with self.context.freeze():  # pylint: disable=E1101
                scalar_list[3].set_as((scalar_list[0] >> VMS4_REGION_LIST0_POS)
                                      & VMS4_REG_BIT_ALL_ONE)
                scalar_list[2].set_as((scalar_list[0] >> VMS4_REGION_LIST1_POS)
                                      & VMS4_REG_BIT_ALL_ONE)
                scalar_list[1].set_as((scalar_list[0] >> VMS4_REGION_LIST2_POS)
                                      & VMS4_REG_BIT_ALL_ONE)
                scalar_list[0].set_as(scalar_list[0] & VMS4_REG_BIT_ALL_ONE)

    @source_info_decorator()
    @debug.set_rpn_cor_ir_decorator
    def set_rpn_cor_ir(self, number):
        """Set special register variables for intermediate
        suppression vector storage

        Parameters
        ----------
        number : the set number

        Returns
        -------
        None
        """
        from .tik_params import MAX_NUMBER
        # check number
        TikCheckUtil.check_type_match(
            number, (int, Scalar), "number should be int or Scalar")
        check_integer_in_range(number, range(MAX_NUMBER),
                               "set value should in the range of [0, 65535]")
        if isinstance(number, Scalar):
            TikCheckUtil.check_equality(
                number.dtype, "uint16", "scalar_number should be uint16")
        with self.new_scope():
            if isinstance(number, Scalar):
                self.emit(tvm.call_extern("int64", "set_rpn_cor_ir",
                                          number.get()), ONE_IR)
            else:
                self.emit(tvm.call_extern("int64", "set_rpn_cor_ir", number),
                          ONE_IR)
        return RPN_COR_IR

    @source_info_decorator()
    @debug.set_rpn_offset_decorator
    def set_rpn_offset(self, number):
        """Set the offset of area and join for computing rpn_proposal

        Parameters
        ----------
        number : offset

        Returns
        -------
        None
        """
        # check core_version
        TikCheckUtil.check_not_equality(
            get_soc_name(), ASCEND_310,
            "instr set_rpn_offset doesn't support ASCEND_310")
        # check number
        TikCheckUtil.check_type_match(number, (int, float, Scalar),
                                      "number should be int, float or Scalar")
        num_upper_limit = 65505
        num_lower_limit = -65504
        check_integer_in_range(
            number, range(num_lower_limit, num_upper_limit),
            "set value should be in the range of [-65504, 65504]")
        if isinstance(number, Scalar):
            TikCheckUtil.check_equality(
                number.dtype, "float16", "scalar_number should be float16")

        with self.new_scope():
            if isinstance(number, Scalar):
                self.emit(tvm.call_extern("float16", "set_rpn_offset",
                                          number.get()), ONE_IR)
            else:
                self.emit(tvm.call_extern("float16", "set_rpn_offset", number),
                          ONE_IR)
        return RPN_OFFSET

    @source_info_decorator()
    @debug.rpn_cor_decorator
    def rpn_cor(self, src0, src1,  # pylint: disable=R0913
                src0_rep_stride, src1_rep_stride,
                repeat_times):
        """Find a new 16 proposal suppresion vector

        Parameters
        ----------
        src0 : source operation
        src1 : source operation
        repeat_times : Repeated iterations times
        src0_rep_stride : offset of src operator in the same block
                          between adjacent iterations
        src1_rep_stride : offset of src operator in the same block
                        between adjacent iterations

        Returns
        -------
        The median value of the suppression vector
        """
        # check tensor
        TikCheckUtil.check_type_match(src0, Tensor, "src0 should be tensor")
        TikCheckUtil.check_type_match(src1, Tensor, "src1 should be tensor")
        TikCheckUtil.check_equality(
            src0.dtype, 'uint16', "suppression matrix should use uint16")
        TikCheckUtil.check_equality(
            src1.dtype, 'uint16', "suppression Vector should use uint16")
        # check repeat
        check_repeat_times(repeat_times)
        # check stride
        check_vector_stride(None, [src0_rep_stride, src1_rep_stride],
                            None, MAX_REP_STRIDE_DOUBLE_BYTE, ["src0", "src1"])

        # check tensor scope
        TikCheckUtil.check_equality(
            src1.scope, scope_ubuf, "src1's scope must be UB")
        TikCheckUtil.check_equality(
            src0.scope, scope_ubuf, "src0's scope must be UB")

        params = [src0_rep_stride, src1_rep_stride, repeat_times]
        offset_list = RPN_COR_OFFSET_LIST
        segment_list = RPN_COR_SEGMENT_LIST
        config = concat_params(params, offset_list, segment_list)
        # get 16-bit suppression vector intermediate result,
        # only update RPN_COR_IR special register
        # extent: ((repeat_times - 1) * src0_rep_stride + 1) * 32Byte/Block
        src0_extent = Expr(((repeat_times - 1)*src0_rep_stride + 1)*32)
        src1_extent = Expr(((repeat_times - 1)*src1_rep_stride + 1)*32)
        with self.new_scope():
            instr = tvm.call_extern(src0.dtype, "rpn_cor",
                                    src0.reinterpret_cast_to("float16").
                                    access_ptr("r", extent=src0_extent.get()),
                                    src1.reinterpret_cast_to("float16").
                                    access_ptr("r", extent=src1_extent.get()),
                                    type_convert(config))
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            self.emit(instr, ONE_IR)
        return RPN_COR_IR

    @source_info_decorator()
    @debug.rpn_cor_diag_decorator
    def rpn_cor_diag(self, dst, src, src_register):
        """Find a new 16 proposal suppresion vector

        Parameters
        ----------
        dst : destination operation
        src : source operation
        src_register : specieal register

        Returns
        -------
        The median value of the suppression vector
        """
        # check tensor
        TikCheckUtil.check_type_match(src, Tensor, "src should be tensor")
        TikCheckUtil.check_type_match(dst, Tensor, "dst should be tensor")
        TikCheckUtil.check_equality(
            src.dtype, 'uint16', "suppression matrix should use uint16")
        TikCheckUtil.check_equality(
            dst.dtype, 'uint16', "suppression Vector should use uint16")
        # check rpn_cor_ir
        TikCheckUtil.check_equality(
            src_register, RPN_COR_IR, "src_register not equal to RPN_COR_IR")
        # check tensor scope
        TikCheckUtil.check_equality(
            src.scope, scope_ubuf, "src's scope must be UB")
        TikCheckUtil.check_equality(
            dst.scope, scope_ubuf, "dst's scope must be UB")
        # 16 elements * each elements occupies 2B
        extent = Expr(16*2)
        with self.new_scope():
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            self.emit(
                tvm.call_extern(
                    dst.dtype, "rpn_cor_diag",
                    dst.reinterpret_cast_to("float16").access_ptr(
                        "w", extent=extent.get()),
                    src.reinterpret_cast_to("float16").access_ptr("r",
                                                                  extent=
                                                                  extent.get())
                    ), ONE_IR)

    def vrpsort16(self, dst, src, repeat_times):
        """Sort them according to the score field in proposal

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst : destination operator
        src : source operation
        repeat_times : Repeated iterations times

        Returns
        -------
        None
        """
        return self._special_intrin_func('vbitsort', dst, [src], repeat_times)
