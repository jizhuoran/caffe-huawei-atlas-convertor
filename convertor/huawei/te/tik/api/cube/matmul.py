"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     matmul.py
DESC:     matmul function
CREATED:  2020-4-23 21:12:13
MODIFIED: 2020-4-23 21:12:45
"""
import math
from te.tik.tik_lib.tik_params import scope_cbuf_out
from te.tik.tik_lib.tik_api_constants import DTYPE_MAP
from te.tik.tik_lib.tik_params import INSTR_DTYPE_SUPPORT_STATEMENT
from te.tik.tik_lib.tik_expr import Expr
from te.platform.cce_conf import api_check_support
from te.platform.cce_params import ASCEND_310
from te.platform.cce_params import ASCEND_910
from .reindex import ReIndexProxy
from ....platform.cce_params import scope_ca
from ....platform.cce_params import scope_cb
from ....platform.cce_params import scope_cbuf
from ...common.util import ceil_div
from ...common.util import reduce_mul
from ...common.util import DTYPE_SIZE
from ...common.util import TikCheckUtil
from ...common.tik_get_soc_name import get_soc_name
from ..tik_tensor import Tensor


# @cond
def _get_for_loop_param(tiling, mode):
    if mode == 'k':
        iter_num = tiling.k_iter_num
        thread_num = tiling.k_thread_num
        has_tail = tiling.k_has_tail_blk
    elif mode == 'm':
        iter_num = tiling.m_iter_num
        thread_num = tiling.m_thread_num
        has_tail = tiling.m_has_tail_blk
    elif mode == 'n':
        iter_num = tiling.n_iter_num
        thread_num = tiling.n_thread_num
        has_tail = tiling.n_has_tail_blk
    else:
        raise RuntimeError('error for_loop mode!')

    if has_tail:
        iter_num = iter_num - 1
        if iter_num < thread_num:
            thread_num = 1

    return iter_num, thread_num


class MatmulTileInfo():  # pylint: disable=R0902
    """
     matmul tile info class
    """
    def __init__(self,  # pylint: disable=R0913
                 m, n, k, c0,
                 m_tile_block,
                 m_thread_num,
                 n_tile_block,
                 n_thread_num,
                 k_tile_block,
                 k_thread_num,
                 loop_mode):
        """
         init the matmul tile info.
        :param m:   the number of m.
        :param k:   the number of k.
        :param n:   the number of n.
        :param c0:  the number of c0.
        :param m_tile_block:  the block  number of m.
        :param m_thread_num:  the thread number of m.
        :param n_tile_block:  the block  number of n.
        :param n_thread_num:  the thread number of n.
        :param k_tile_block:  the block number of k.
        :param k_thread_num:  the thread number of k.
        :param loop_mode:   the mode of loop.
        """
        self.m_num = m
        self.n_num = n
        self.k_num = k
        self.c0_num = c0
        self.k_thread_num = k_thread_num
        self.k_tile_block = k_tile_block
        self.m_tile_block = m_tile_block
        self.m_thread_num = m_thread_num
        self.n_tile_block = n_tile_block
        self.n_thread_num = n_thread_num
        self.loop_mode = loop_mode
        self.block_size = 16
        self.m_iter_num = math.ceil(
            self.m_num / (self.m_tile_block * self.block_size))
        self.n_iter_num = math.ceil(
            self.n_num / (self.n_tile_block * self.block_size))
        self.k_iter_num = math.ceil(
            self.k_num / (self.k_tile_block * self.c0_num))

        self.m_blocks = ceil_div(m, self.block_size)
        self.k_blocks = ceil_div(k, self.c0_num)
        self.n_blocks = ceil_div(n, self.block_size)



        self.m_tail_block = self.m_blocks - (self.m_iter_num - 1) * \
                            m_tile_block
        self.k_tail_block = self.k_blocks - (self.k_iter_num - 1) * \
                            k_tile_block
        self.n_tail_block = self.n_blocks - (self.n_iter_num - 1) * \
                            n_tile_block

        self.m_has_tail_blk = self.m_tail_block != self.m_tile_block
        self.k_has_tail_blk = self.k_tail_block != self.k_tile_block
        self.n_has_tail_blk = self.n_tail_block != self.n_tile_block


    def __str__(self):
        return "MatmulTileInfo\n" \
                "m iter {} block {} thread {}\n" \
                "n iter {} block {} thread {}\n" \
                "k iter {} block {} thread {}\n" \
                "loop mode {} MTE1 count {}".format(
                    self.m_iter_num, self.m_tile_block, self.m_thread_num,
                    self.n_iter_num, self.n_tile_block, self.n_thread_num,
                    self.k_iter_num, self.k_tile_block, self.k_thread_num,
                    self.loop_mode, self.get_met1_issue_count())

    def get_l0a_met1_issue_count(self):
        """
        get the l0a met1 issue count.
        :return: the count of l0a met1.
        """

        if self.k_tile_block == 1:
            # Load one column at a time
            return self.k_iter_num

        # Load one row
        return self.m_tile_block * self.k_iter_num

    def get_l0b_met1_issue_count(self):
        """
         get the l0b met1 issue count.
        :return:  the count of l0b met1.
        """
        if self.n_tile_block == 1:
            # Load one column
            return self.k_iter_num
        if self.n_iter_num == 1:
            # load whole tiling l0b
            return self.k_iter_num
        # Load one row
        return self.k_iter_num * self.k_tile_block

    def get_met1_issue_count(self):
        """
        get the met1 issue count.
        :return:  the count of met1 issue.
        """
        l0a_count = self.get_l0a_met1_issue_count()
        l0b_count = self.get_l0b_met1_issue_count()

        if self.loop_mode in "NM":
            mte1_count = self.n_iter_num * l0b_count + \
                         self.n_iter_num * l0a_count * self.m_iter_num
        elif self.loop_mode == "MN":
            mte1_count = self.m_iter_num * l0a_count + \
                         self.m_iter_num * l0b_count * self.n_iter_num
        else:
            raise RuntimeError("only support NM or MN mode but get %s" %
                               self.loop_mode)
        return mte1_count
# @endcond


def gen_best_tiling(m_num, k_num,  # pylint: disable=R0913, R0914
                    n_num, c0_num, block_size, a_dtype, b_dtype,
                    tik_instance):
    """
    generator the best of tiling.
    src0's shape : [m, k], src1's shape : [k, n]

    Parameters
    ----------
    m_num: int, the 16-based upward aligned value of m
    k_num: int, the 16(b16)/32(b8) upward aligned value of k
    n_num: int, the 16-based upward aligned value of n
    c0_num: int, elements number, 16
    block_size: int, number of element in one block
    a_dtype: str, the type of src0
    b_dtype: str, the type of src1
    c_dtype: str, the type of dst

    Returns
    ----------
    """
    valid_tiling = []

    a_dtype_size = DTYPE_SIZE[a_dtype]
    b_dtype_size = DTYPE_SIZE[b_dtype]

    m_blocks = math.ceil(m_num / block_size)
    n_blocks = math.ceil(n_num / block_size)
    k_blocks = math.ceil(k_num / c0_num)

    loop_modes = [(2, 1, 1, "NM"),
                  (1, 2, 1, "NM"),
                  (1, 2, 1, "MN"),
                  (2, 1, 1, "MN"),
                  (1, 1, 2, "NM"),
                  (1, 1, 2, "MN")]

    def _gen_valid_tiling(loop_modes):
        for m_thread, n_thread, k_thread, loop_mode in loop_modes:
            for k_tile_block in range(1, k_blocks + 1):
                for m_tile_block in range(1, m_blocks + 1):
                    for n_tile_block in range(1, n_blocks + 1):
                        m_iter_num = math.ceil(m_blocks / m_tile_block)
                        n_iter_num = math.ceil(n_blocks / n_tile_block)
                        k_iter_num = math.ceil(k_blocks / k_tile_block)

                        # if cut m then we must load l0b one column at a time
                        if m_iter_num > 1 and n_tile_block > 1:
                            continue

                        if m_thread > m_iter_num:
                            continue

                        if k_thread > k_iter_num:
                            continue

                        if n_thread > n_iter_num:
                            continue

                        l0a_mat_size = m_tile_block * k_tile_block * \
                                       block_size * c0_num * a_dtype_size * 2
                        l0b_mat_size = n_tile_block * k_tile_block * \
                                       block_size * c0_num * b_dtype_size * 2

                        if l0a_mat_size > \
                                tik_instance.d_profiling.get_l0a_buffer_size():
                            continue
                        if l0b_mat_size > \
                                tik_instance.d_profiling.get_l0b_buffer_size():
                            continue

                        tile_info = MatmulTileInfo(m_num, n_num, k_num, c0_num,
                                                   m_tile_block, m_thread,
                                                   n_tile_block, n_thread,
                                                   k_tile_block, k_thread,
                                                   loop_mode)
                        valid_tiling.append(tile_info)

    _gen_valid_tiling(loop_modes)

    if not valid_tiling:
        loop_modes = [(1, 1, 1, "NM"),
                      (1, 1, 1, "MN")]
        _gen_valid_tiling(loop_modes)

    valid_tiling.sort(key=lambda x: x.get_met1_issue_count())
    best_tiling = valid_tiling[0]
    return best_tiling

# @cond
class MatMulImpl():  # pylint: disable=R0902
    """
    the implement of matmul
    """
    def __init__(self, tik_instance, l1_out_dst, init_l1out=True):
        """
         init of matmul.
        :param tik_instance: the instance of tik.
        :param l1_out_dst:   the result tensor of mmad.
        :param init_l1out:   mark whether to init on l1out tensor mat_c
                         if False, will accumulate on l1out tensor mat_c
        """
        self.tik_instance = tik_instance
        self.l1_out_dst = l1_out_dst
        self.round_m = 0
        self.round_k = 0
        self.round_n = 0
        self.block_size = 16
        self.a_dtype = None
        self.b_dtype = None
        self.c_dtype = None
        self.mat_a = None
        self.mat_b = None
        self.m_num = 0
        self.k_num = 0
        self.c0_num = 0
        self.n_num = 0
        self.k_has_tail_ele = False
        self.k_tail_ele = 0
        self.init_l1out = init_l1out

    def _check_params(self, mat_a, mat_b):
        TikCheckUtil.check_type_match(self.l1_out_dst, Tensor,
                                      "dst should be Tensor, but dst's type "
                                      "is %s" % type(self.l1_out_dst))
        TikCheckUtil.check_type_match(mat_a, Tensor,
                                      "a should be Tensor, but a's type is %s" %
                                      type(mat_a))
        TikCheckUtil.check_type_match(mat_b, Tensor,
                                      "b should be Tensor, but b's type is %s" %
                                      type(mat_b))
        TikCheckUtil.check_equality(self.l1_out_dst.scope, scope_cbuf_out,
                                    "dst's scope should be L1_OUT, but dst's "
                                    "scope is %s" % self.l1_out_dst.scope)
        TikCheckUtil.check_equality(mat_a.scope, scope_cbuf,
                                    "a's scope should be L1, but a's scope is "
                                    "%s" % mat_a.scope)
        TikCheckUtil.check_equality(mat_b.scope, scope_cbuf,
                                    "b's scope should be L1, but b's scope is "
                                    "%s" % mat_b.scope)
        # check dtype
        dtype_str = DTYPE_MAP[mat_a.dtype] + DTYPE_MAP[mat_b.dtype] + \
                    DTYPE_MAP[self.l1_out_dst.dtype]
        TikCheckUtil.check_equality(api_check_support("tik.matmul",
                                                      dtype_str), True,
                                    INSTR_DTYPE_SUPPORT_STATEMENT.
                                    format(dtype_str, "matmul"))
        TikCheckUtil.check_type_match(self.m_num, int,
                                      "m should be int, but m's type is %s" %
                                      type(self.m_num))
        TikCheckUtil.check_type_match(self.k_num, int,
                                      "k should be int, but k's type is %s" %
                                      type(self.k_num))
        TikCheckUtil.check_type_match(self.n_num, int,
                                      "n should be int, but n's type is %s" %
                                      type(self.n_num))
        m_range = range(1, 4097)
        k_range = range(1, 16385) if mat_a.dtype == "float16" \
            else range(1, 32769)
        n_range = m_range
        TikCheckUtil.check_in_range(
            self.m_num, m_range,
            "m should be in range of [%s, %s], but m is %s" %
            (m_range.start, m_range.stop - 1, self.m_num))
        TikCheckUtil.check_in_range(
            self.k_num, k_range,
            "k should be in range of [%s, %s], but k is %s" %
            (k_range.start, k_range.stop - 1, self.k_num))
        TikCheckUtil.check_in_range(
            self.n_num, n_range,
            "n should be in range of [%s, %s], but n is %s" %
            (n_range.start, n_range.stop - 1, self.n_num))
        # check init_l1out
        TikCheckUtil.check_type_match(
            self.init_l1out, bool, "init_l1out should be bool type.")

    def _check_overflow(self, mat_a, mat_b):
        # check dst
        need_element = self.round_m*self.round_n
        if self.l1_out_dst.is_single_point():
            total_element = 1
        else:
            total_element = Expr(
                reduce_mul(self.l1_out_dst.indice.origin_shape) -
                self.l1_out_dst.offset).eval_value()
        if total_element is not None:
            TikCheckUtil.check_ge(total_element, need_element,
                                  "dst tensor overflow, expected elements: %s, "
                                  "actual elements: %s" % (need_element,
                                                           total_element))
        # check a
        need_element = self.round_m*self.round_k
        if mat_a.is_single_point():
            total_element = 1
        else:
            total_element = Expr(reduce_mul(mat_a.indice.origin_shape) -
                                 mat_a.offset).eval_value()
        if total_element is not None:
            TikCheckUtil.check_ge(total_element, need_element,
                                  "a tensor overflow, expected elements: %s, "
                                  "actual elements: %s" % (need_element,
                                                           total_element))
        # check b
        need_element = self.round_k * self.round_n
        if mat_b.is_single_point():
            total_element = 1
        else:
            total_element = Expr(reduce_mul(mat_b.indice.origin_shape) -
                                 mat_b.offset).eval_value()
        if total_element is not None:
            TikCheckUtil.check_ge(total_element, need_element,
                                  "b tensor overflow, expected elements: %s, "
                                  "actual elements: %s" % (need_element,
                                                           total_element))

    def _check_operator_align(self, mat_a, mat_b):
        tensor_offset = Expr(self.l1_out_dst.offset).eval_value()
        if tensor_offset is not None:
            TikCheckUtil.check_equality(
                tensor_offset * DTYPE_SIZE[self.l1_out_dst.dtype] % 1024, 0,
                "dst address should be 1024B aligned")
        if get_soc_name() in (ASCEND_310, ASCEND_910):
            align = 512
        else:
            align = 32
        for cur_tensor in (mat_a, mat_b):
            tensor_offset = Expr(cur_tensor.offset).eval_value()
            if tensor_offset is not None:
                TikCheckUtil.check_equality(
                    tensor_offset*DTYPE_SIZE[cur_tensor.dtype] % align, 0,
                    "source a and b address should be %sB aligned" % align)

    def execute(self,  # pylint: disable=R0913
                mat_a, mat_b, m_num, k_num, n_num):
        """
         execute of matmul.
        :param mat_a:  the a tensor
        :param mat_b:  the b tensor.
        :param m_num:      the number of m.
        :param k_num:      the number of k.
        :param n_num:      the number of n.
        :return:    None.
        """
        self.m_num = m_num
        self.k_num = k_num
        self.n_num = n_num

        self._check_params(mat_a, mat_b)

        self.mat_a = ReIndexProxy(mat_a, (m_num, k_num))
        self.mat_b = ReIndexProxy(mat_b, (k_num, n_num))

        self.a_dtype = self.mat_a.dtype
        self.b_dtype = self.mat_b.dtype
        self.c_dtype = self.l1_out_dst.dtype

        # means 16*16 or 16*32
        self.c0_num = 16 if self.a_dtype == "float16" else 32

        # shape cut
        self.round_m = math.ceil(m_num / self.block_size) * self.block_size
        self.round_k = math.ceil(k_num / self.c0_num) * self.c0_num
        self.round_n = math.ceil(n_num / self.block_size) * self.block_size

        self._check_overflow(mat_a, mat_b)
        self._check_operator_align(mat_a, mat_b)

        self.l1_out_dst = ReIndexProxy(
            self.l1_out_dst, (math.ceil(n_num / self.block_size),
                              self.round_m, self.block_size))

        tiling = gen_best_tiling(self.round_m, self.round_k,
                                 self.round_n, self.c0_num,
                                 self.block_size, self.a_dtype,
                                 self.b_dtype, self.tik_instance)

        self.k_has_tail_ele = self.round_k != self.k_num

        self.k_tail_ele = self.k_num % (tiling.k_tile_block*self.c0_num)

        self.mmad_codegen(tiling)

    def mmad_codegen(self, tiling):
        """
         the code generator of mmad .
        :param tiling:  the tiling of mmad mn
        :return:  None
        """
        # make k loop the outer most loop
        inst = self.tik_instance
        k_iter_num, k_thread_num = _get_for_loop_param(tiling, 'k')
        with inst.for_range(0, k_iter_num, thread_num=k_thread_num) as k_idx:
            if tiling.loop_mode == 'NM':
                self._do_nm_loop(tiling, k_idx, tiling.k_tile_block)
            elif tiling.loop_mode == 'MN':
                self._do_mn_loop(tiling, k_idx, tiling.k_tile_block)

        if tiling.k_has_tail_blk:
            if tiling.loop_mode == 'NM':
                self._do_nm_loop(tiling, k_iter_num, tiling.k_tail_block)
            elif tiling.loop_mode == 'MN':
                self._do_mn_loop(tiling, k_iter_num, tiling.k_tail_block)



    def _do_nm_loop(self, tiling, k_idx, k_actual_blk):

        def _do_m_loop(n_idx, n_actual_blk):
            data_l0b = self._make_l0b_code(tiling, k_idx, k_actual_blk,
                                           n_idx, n_actual_blk)
            m_iter_num, m_thread_num = _get_for_loop_param(tiling, 'm')
            with inst.for_range(0, m_iter_num, thread_num=m_thread_num) as \
                    m_idx:
                data_l0a = self._make_l0a_code(
                    tiling, m_idx, tiling.m_tile_block, k_idx,
                    k_actual_blk)
                self._make_mmad_code(data_l0a, data_l0b, tiling, m_idx, k_idx,
                                     n_idx, tiling.m_tile_block, k_actual_blk,
                                     n_actual_blk)

            if tiling.m_has_tail_blk:
                data_l0a = self._make_l0a_code(
                    tiling, m_iter_num, tiling.m_tail_block,
                    k_idx, k_actual_blk)
                self._make_mmad_code(data_l0a, data_l0b, tiling, m_iter_num,
                                     k_idx, n_idx, tiling.m_tail_block,
                                     k_actual_blk, n_actual_blk)

        inst = self.tik_instance
        n_iter_num, n_thread_num = _get_for_loop_param(tiling, 'n')
        with inst.for_range(0, n_iter_num, thread_num=n_thread_num) as n_idx:
            _do_m_loop(n_idx, tiling.n_tile_block)
        if tiling.n_has_tail_blk:
            _do_m_loop(n_iter_num, tiling.n_tail_block)


    def _do_mn_loop(self, tiling, k_idx, k_actual_blk):
        def _do_n_loop(m_idx, m_actual_blk):
            data_l0a = self._make_l0a_code(tiling, m_idx, m_actual_blk,
                                           k_idx, k_actual_blk)
            n_iter_num, n_thread_num = _get_for_loop_param(tiling, 'n')
            with inst.for_range(0, n_iter_num, thread_num=n_thread_num) \
                    as n_idx:
                data_l0b = self._make_l0b_code(tiling, k_idx, k_actual_blk,
                                               n_idx, tiling.n_tile_block)
                self._make_mmad_code(data_l0a, data_l0b, tiling, m_idx, k_idx,
                                     n_idx, m_actual_blk, k_actual_blk,
                                     tiling.n_tile_block)
            if tiling.n_has_tail_blk:
                data_l0b = self._make_l0b_code(tiling, k_idx, k_actual_blk,
                                               n_iter_num, tiling.n_tail_block)
                self._make_mmad_code(data_l0a, data_l0b, tiling, m_idx, k_idx,
                                     n_iter_num, m_actual_blk, k_actual_blk,
                                     tiling.n_tail_block)

        inst = self.tik_instance
        m_iter_num, m_thread_num = _get_for_loop_param(tiling, 'm')
        with inst.for_range(0, m_iter_num, thread_num=m_thread_num) as m_idx:
            _do_n_loop(m_idx, tiling.m_tile_block)
        if tiling.m_has_tail_blk:
            _do_n_loop(m_iter_num, tiling.m_tail_block)

    def _make_l0a_code(self, tiling, m_idx, m_actual, # pylint: disable=R0913
                       k_idx, k_actual):
        inst = self.tik_instance
        data_l0a = inst.Tensor(
            self.a_dtype,
            (m_actual * self.block_size * k_actual * self.c0_num,),
            name='_cubeapi_data_l0a', scope=scope_ca)
        if k_actual == 1:
            # Load one column
            l1a_offset = m_idx * tiling.m_tile_block * \
                         self.block_size * self.c0_num + k_idx * \
                         tiling.k_tile_block * tiling.m_blocks * \
                         self.block_size * self.c0_num
            if get_soc_name() in (ASCEND_310, ASCEND_910):
                inst.load2dv1(data_l0a,
                              self.mat_a.flat_access(l1a_offset),
                              0, m_actual, 1, 0)
            else:
                inst.load2dv2(data_l0a,
                              self.mat_a.flat_access(l1a_offset),
                              0, m_actual, 0, 1, 0)
        else:
            # Load row by row
            with inst.for_range(0, m_actual) as m_tile_idx:
                l0a_offset = m_tile_idx * k_actual * \
                             self.block_size * self.c0_num
                l1a_offset = \
                    (m_idx * tiling.m_tile_block + m_tile_idx) * \
                    self.block_size * self.c0_num + k_idx * \
                    tiling.k_tile_block * tiling.m_blocks * \
                    self.block_size * self.c0_num
                if get_soc_name() in (ASCEND_310, ASCEND_910):
                    inst.load2dv1(data_l0a[l0a_offset],
                                  self.mat_a.flat_access(l1a_offset),
                                  0, k_actual, tiling.m_blocks, 0)
                else:
                    inst.load2dv2(data_l0a[l0a_offset],
                                  self.mat_a.flat_access(l1a_offset),
                                  0, k_actual, 0, tiling.m_blocks, 0)
        return data_l0a

    def _make_l0b_code(self, tiling, k_idx, k_actual, # pylint: disable=R0913
                       n_idx, n_actual):
        inst = self.tik_instance
        data_l0b = inst.Tensor(
            self.b_dtype,
            (n_actual * k_actual * self.block_size * self.c0_num,),
            name='_cubeapi_data_l0b', scope=scope_cb)
        if n_actual == 1:
            # Load one column
            l1b_idx = n_idx * tiling.n_tile_block * self.block_size * \
                      self.c0_num + k_idx * tiling.k_tile_block * \
                      tiling.n_blocks * self.block_size * self.c0_num
            if get_soc_name() in (ASCEND_310, ASCEND_910):
                inst.load2dv1(data_l0b,
                              self.mat_b.flat_access(l1b_idx),
                              0, k_actual, tiling.n_blocks, 0)
            else:
                inst.load2dv2(data_l0b,
                              self.mat_b.flat_access(l1b_idx),
                              0, k_actual, 0, tiling.n_blocks, 0)
        elif tiling.n_iter_num == 1:
            # Load all once
            l0b_idx = 0
            l1b_idx = n_idx * tiling.n_tile_block * self.block_size * \
                      self.c0_num + \
                      (k_idx * tiling.k_tile_block) * \
                      tiling.n_blocks * self.block_size * self.c0_num
            if get_soc_name() in (ASCEND_310, ASCEND_910):
                inst.load2dv1(data_l0b[l0b_idx],
                              self.mat_b.flat_access(l1b_idx),
                              0, n_actual * k_actual, 1, 0)
            else:
                inst.load2dv2(data_l0b[l0b_idx],
                              self.mat_b.flat_access(l1b_idx),
                              0, n_actual * k_actual, 0, 1, 0)
        else:
            # Load row by row
            with inst.for_range(0, k_actual) as k_sub_idx:
                l0b_idx = k_sub_idx * n_actual * \
                          self.block_size * self.c0_num
                l1b_idx = n_idx * tiling.n_tile_block * self.block_size * \
                          self.c0_num + \
                          (k_idx * tiling.k_tile_block + k_sub_idx) * \
                          tiling.n_blocks * self.block_size * self.c0_num
                if get_soc_name() in (ASCEND_310, ASCEND_910):
                    inst.load2dv1(data_l0b[l0b_idx],
                                  self.mat_b.flat_access(l1b_idx),
                                  0, n_actual, 1, 0)
                else:
                    inst.load2dv2(data_l0b[l0b_idx],
                                  self.mat_b.flat_access(l1b_idx),
                                  0, n_actual, 0, 1, 0)
        return data_l0b

    def _make_mmad_code(self, data_l0a, data_l0b, # pylint: disable=R0913
                        tiling, m_idx, k_idx, n_idx,
                        m_actual, k_actual, n_actual):
        inst = self.tik_instance
        l0c_offset = n_idx * tiling.n_tile_block * tiling.m_blocks * \
                     self.block_size * self.block_size + m_idx * \
                     tiling.m_tile_block * self.block_size * self.block_size

        if tiling.k_iter_num == 1:
            inst.mmad(self.l1_out_dst.flat_access(l0c_offset),
                      data_l0a, data_l0b, m_actual * self.block_size,
                      self.k_num, n_actual * self.block_size,
                      not self.init_l1out)
        elif self.init_l1out and self.k_has_tail_ele:
            with inst.if_scope(k_idx == 0):
                inst.mmad(self.l1_out_dst.flat_access(l0c_offset),
                          data_l0a, data_l0b, m_actual * self.block_size,
                          k_actual * self.c0_num, n_actual * self.block_size,
                          0)
            with inst.else_scope():
                self._mmad_has_k_tail(data_l0a, data_l0b, k_idx,
                                      m_actual, k_actual, n_actual,
                                      l0c_offset, tiling)

        elif not self.init_l1out and self.k_has_tail_ele:
            self._mmad_has_k_tail(data_l0a, data_l0b, k_idx,
                                  m_actual, k_actual, n_actual,
                                  l0c_offset, tiling)
        elif self.init_l1out and not self.k_has_tail_ele:
            with inst.if_scope(k_idx == 0):
                inst.mmad(self.l1_out_dst.flat_access(l0c_offset),
                          data_l0a, data_l0b, m_actual * self.block_size,
                          k_actual * self.c0_num,
                          n_actual * self.block_size,
                          0)
            with inst.else_scope():
                inst.mmad(self.l1_out_dst.flat_access(l0c_offset),
                          data_l0a, data_l0b, m_actual * self.block_size,
                          k_actual * self.c0_num,
                          n_actual * self.block_size,
                          1)
        else:
            inst.mmad(self.l1_out_dst.flat_access(l0c_offset),
                      data_l0a, data_l0b, m_actual * self.block_size,
                      k_actual * self.c0_num,
                      n_actual * self.block_size,
                      1)

    def _mmad_has_k_tail(self, data_l0a, data_l0b,  # pylint: disable=R0913
                         k_idx, m_actual, k_actual, n_actual,
                         l0c_offset, tiling):
        inst = self.tik_instance
        with inst.if_scope(k_idx == tiling.k_iter_num - 1):
            inst.mmad(self.l1_out_dst.flat_access(l0c_offset),
                      data_l0a, data_l0b, m_actual * self.block_size,
                      self.k_tail_ele,
                      n_actual * self.block_size,
                      1)
        with inst.else_scope():
            inst.mmad(self.l1_out_dst.flat_access(l0c_offset),
                      data_l0a, data_l0b, m_actual * self.block_size,
                      k_actual * self.c0_num,
                      n_actual * self.block_size,
                      1)
# @endcond
