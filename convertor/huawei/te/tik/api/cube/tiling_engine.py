"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     tiling_engine.py
DESC:     provide tiling method
CREATED:  2020-4-23 21:12:13
MODIFIED: 2020-4-23 21:12:45
"""
import math
from te.tik.api.tik_tensor import Tensor

DTYPE_SIZE = {
    'uint8': 1,
    'int8': 1,
    'uint16': 2,
    'int16': 2,
    'uint32': 4,
    'int32': 4,
    'float16': 2,
    'float32': 4,
    'int64': 8,
    'uint64': 8,
    'float64': 8
}


def get_bytes(dtype):
    """get bit size"""
    return DTYPE_SIZE[dtype]


# @cond
class FMDesc():  # pylint: disable=R0903, R0902
    """feature map info"""
    def __init__(self,  # pylint: disable=R0913
                 height, width, kh, kw, cin, input_dtype, output_dtype,
                 stride_h, stride_w, dilation_h, dilation_w, pad_list=None):
        if pad_list is None:
            pad_list = [0, 0, 0, 0]
        self.pad_list = pad_list
        self.pad_left = pad_list[0]
        self.pad_right = pad_list[1]
        self.pad_top = pad_list[2]
        self.pad_bottom = pad_list[3]

        self.stride_h = stride_h
        self.stride_w = stride_w

        self.dilation_h = dilation_h
        self.dilation_w = dilation_w

        self.h_i = height
        self.w_i = width
        self.h_o = math.floor((height + self.pad_top + self.pad_bottom -
                               self.dilation_h * (kh - 1) - 1) / stride_h + 1)
        self.w_o = math.floor((width + self.pad_left + self.pad_right -
                               self.dilation_w * (kw - 1) - 1) / stride_w + 1)
        self.cin = cin
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
# @endcond


# @cond
class FilterDesc():  # pylint: disable=R0903
    """filter info"""
    def __init__(self,  # pylint: disable=R0913
                 c1, kh, kw, cout, c0, dtype):
        self.height = kh
        self.width = kw
        self.cin = c1 * c0
        self.cout = cout
        self.dtype = dtype
# @endcond


# @cond
class ConvTileInfo():  # pylint: disable=R0902
    """Conv Tiling Info"""
    def __init__(self,  # pylint: disable=R0913
                 fm_desc,
                 filter_desc,
                 howo_tile_block,
                 howo_tile_thread,
                 k_tile_block,
                 k_tile_thread,
                 cout_tile_block,
                 cout_tile_thread,
                 c_0, loop_mode, tik_instance):
        self.fm_desc = fm_desc
        self.filter_desc = filter_desc
        self.tik_instance = tik_instance
        self.howo_tile_block = howo_tile_block
        self.howo_tile_thread = howo_tile_thread
        self.k_tile_block = k_tile_block
        self.k_tile_thread = k_tile_thread
        self.cout_tile_block = cout_tile_block
        self.cout_tile_thread = cout_tile_thread
        self.block_size = 16
        self.c_0 = c_0
        self.total_howo = self.fm_desc.h_o * self.fm_desc.w_o
        self.howo_block_num = math.ceil(self.total_howo / self.block_size)
        self.howo_iter_num = math.ceil(
            self.total_howo / (self.block_size * self.howo_tile_block))
        self.cin_block_num = math.ceil(
            self.filter_desc.cin * self.filter_desc.height *
            self.filter_desc.width / self.c_0)
        self.k_iter_num = math.ceil(
            self.cin_block_num / self.k_tile_block)
        self.cout_block_num = math.ceil(
            self.filter_desc.cout / self.block_size)
        self.cout_iter_num = math.ceil(
            self.cout_block_num / self.cout_tile_block)
        self.l0a_mte1_count = None
        self.l0b_mte1_count = None
        self.l0a_mode = None
        self.l0b_mode = None
        self.loop_mode = loop_mode  # KN or NK
        self.hw_has_tail = int(self.howo_block_num < self.howo_iter_num *
                               self.howo_tile_block)
        self.cin_has_tail = int(self.cin_block_num < self.k_iter_num *
                                self.k_tile_block)
        self.cout_has_tail = int(self.cout_block_num < self.cout_iter_num *
                                 self.cout_tile_block)

        self.total_l0b_cycles = 0
        self.total_l0a_cycles = 0
        self.total_cube_cycles = 0
        self.hw_tail_cycle = 0
        self.cin_tail_cycle = 0
        self.cout_tail_cycle = 0
        self.total_cycles = 0
        self.set_load_mode()
        self.cal_cycles()

    def _get_cube_cycles(self):
        # M * K * N
        cube_cycles = self.howo_tile_block * self.k_tile_block * \
                      self.cout_tile_block

        # M or N is odd, cube conflict cycles is (K - 2)
        if self.cout_tile_block % 2 != 0 or \
                self.howo_tile_block % 2 != 0:
            cube_cycles += self.k_tile_block - 2

        # load cost
        head_cost = 16
        return cube_cycles + head_cost

    @staticmethod
    def get_load_l0a_time_unit(h_o, w_o):
        """cal once l0a load cycles"""
        iter_num = math.ceil(h_o * w_o / 16)
        current_position = 0
        old_wo = 1
        total_cycle_count = 0
        for _ in range(iter_num):
            current_position += 16
            current_wo = min(math.ceil(current_position / w_o), w_o)
            total_cycle_count += (current_wo - old_wo) + 1
            old_wo = current_wo
        return total_cycle_count / iter_num

    def _get_load_l0a_cycles(self):
        stride_w = self.fm_desc.stride_w
        if self.fm_desc.w_o >= 16:
            if self.fm_desc.w_o % 16 == 0:
                one_instr_times = stride_w
            else:
                one_instr_times = self.get_load_l0a_time_unit(
                    self.fm_desc.h_o, self.fm_desc.w_o) * stride_w
        elif self.fm_desc.w_o > 8:
            one_instr_times = math.ceil(16 / self.fm_desc.w_o) * stride_w
        # wo LE 8
        else:
            one_instr_times = math.ceil(16 / self.fm_desc.w_o)

        head_cost = 16
        if self.l0a_mode == 0:
            # l0a_mode is 0
            cal_times = self.howo_tile_block * (self.k_tile_block *
                                                one_instr_times + head_cost)
        else:
            # l0a_mode is 1
            cal_times = self.k_tile_block * (self.howo_tile_block *
                                             one_instr_times + head_cost)

        load_l0a_cycles = cal_times
        return load_l0a_cycles

    def _get_load_l0b_cycles(self):
        head_cost = 12
        if self.l0b_mode == 0 and self.cout_iter_num != 1:
            # l0b mode is 0
            load_l0b_cycles = self.k_tile_block*self.cout_tile_block*2 + \
                              self.k_tile_block*head_cost
        else:
            # l0b mode is 1 or (l0b mode is 0 and cout_iter_num is 1)
            load_l0b_cycles = self.k_tile_block * self.cout_tile_block * 2 +\
                              head_cost
        return load_l0b_cycles

    @staticmethod
    def _deal_with_double_buffer_cycles(a_cycles, b_cycles, iter_nums):
        if iter_nums > 2 and iter_nums != math.ceil(iter_nums / 2) * 2:
            if a_cycles > b_cycles:
                cycles = a_cycles * (iter_nums - 1) + b_cycles + \
                         a_cycles + b_cycles
            else:
                cycles = b_cycles * (iter_nums - 1) + a_cycles + \
                         a_cycles + b_cycles
            return cycles
        if a_cycles > b_cycles:
            return a_cycles * iter_nums + b_cycles
        return b_cycles * iter_nums + a_cycles

    @staticmethod
    def cal_double_db_value(a_cycles, b_cycles,  # pylint: disable=R0913
                            c_cycles, inner_iter, outer_iter, tail_cycle):
        """cal inner and outer all enable double buffer cycles"""
        if a_cycles + 2 * b_cycles <= 2 * c_cycles:
            db_value1 = c_cycles * inner_iter + tail_cycle
            db_value2 = a_cycles + b_cycles + outer_iter * db_value1
        elif c_cycles >= b_cycles and a_cycles + 2 * b_cycles > 2 * c_cycles:
            db_value1 = c_cycles * inner_iter + a_cycles + 2 * b_cycles - \
                        2 * c_cycles + tail_cycle
            db_value2 = a_cycles + b_cycles + inner_iter * c_cycles + \
                        (outer_iter - 1) * db_value1
        else:
            db_value1 = b_cycles * inner_iter + c_cycles + tail_cycle
            db_value2 = a_cycles + (a_cycles - c_cycles + db_value1) * \
                        (outer_iter - 1) + db_value1
        return db_value1, db_value2

    def _cal_one_db_value(self, outer_tile_thread,  # pylint: disable=R0913
                          inner_tile_thread, outer_load_cycles,
                          inner_load_cycles, cube_cycles,
                          outer_iter_num, inner_iter_num, tail_cycles):
        if inner_tile_thread == 2:
            inner_db_cycles = self._deal_with_double_buffer_cycles(
                inner_load_cycles, cube_cycles, inner_iter_num)
        else:
            inner_db_cycles = (inner_load_cycles + cube_cycles) * \
                              inner_iter_num
        inner_db_cycles += tail_cycles

        if outer_tile_thread == 2:
            if outer_load_cycles <= cube_cycles:
                outer_db_cycles = outer_load_cycles + \
                                  outer_iter_num*inner_db_cycles
            else:
                outer_db_cycles = outer_load_cycles + \
                                  outer_iter_num*inner_db_cycles + \
                                  (outer_iter_num - 1) * \
                                  (outer_load_cycles - cube_cycles)
        else:
            outer_db_cycles = (inner_db_cycles + outer_load_cycles) * \
                        outer_iter_num
        return inner_db_cycles, outer_db_cycles

    @staticmethod
    def _cal_loop_info(has_tail, tile_thread, iter_num, tail_cycles=0):
        """accroding the for loop info to cal iter num,
        tail cycles, thread num
        """
        tmp_iter_num = iter_num
        db_sub = False
        tmp_tail_cycles = 0
        tmp_tile_thread = tile_thread
        if has_tail or (tile_thread == 2 and iter_num % 2 == 1):
            tmp_iter_num -= 1
            tmp_tail_cycles = tail_cycles
            if tmp_iter_num > 1 and tmp_iter_num % 2 == 1:
                db_sub = True
                tmp_iter_num -= 1
                tmp_tail_cycles += tail_cycles
        if tmp_iter_num == 1:
            tmp_tile_thread = 1
        return db_sub, tmp_iter_num, tmp_tile_thread, tmp_tail_cycles

    def _cal_nk_cycles(self, load_l0a_cycles, load_l0b_cycles, cube_cycles):
        db_cin_iter_num = self.k_iter_num
        if self.cin_has_tail or (self.k_tile_thread == 2 and
                                 self.k_iter_num % 2 == 1):
            db_cin_iter_num -= 1
            self.cin_tail_cycle = load_l0a_cycles + load_l0b_cycles + \
                                  cube_cycles
            if db_cin_iter_num > 1 and db_cin_iter_num % 2 == 1:
                db_cin_iter_num -= 1
                self.cin_tail_cycle += self.cin_tail_cycle
            if db_cin_iter_num == 1:
                self.k_tile_thread = 1

        if self.k_tile_thread == 2 or (self.cout_tile_thread == 2 and
                                       db_cin_iter_num > 1):
            db_cycles = self._deal_with_double_buffer_cycles(
                load_l0b_cycles + load_l0a_cycles,
                cube_cycles, db_cin_iter_num)
        else:
            db_cycles = (load_l0a_cycles + load_l0b_cycles +
                         cube_cycles) * db_cin_iter_num

        db_cycles += self.cin_tail_cycle
        self.total_l0a_cycles = load_l0a_cycles * self.k_iter_num * \
                                self.cout_iter_num
        self.total_l0b_cycles = load_l0b_cycles * self.k_iter_num * \
                                self.cout_iter_num
        self.total_cycles = self.cout_iter_num * db_cycles

    def _cal_kn_cycles(self, load_l0a_cycles, load_l0b_cycles, cube_cycles):
        db_cin_sub, db_cin_iter_num, self.k_tile_thread, _ = \
            self._cal_loop_info(self.cin_has_tail, self.k_tile_thread,
                                self.k_iter_num)

        tmp_cout_tile_thread = max(self.k_tile_thread, self.cout_tile_thread)
        _, db_cout_iter_num, self.cout_tile_thread, self.cout_tail_cycle = \
            self._cal_loop_info(self.cout_has_tail, tmp_cout_tile_thread,
                                self.cout_iter_num,
                                load_l0b_cycles + cube_cycles)

        if (self.k_tile_thread == 2 and db_cout_iter_num >= 2) or \
                (self.cout_tile_thread == 2 and self.k_tile_thread == 2):
            cout_db_cycles, db_cycles = self.cal_double_db_value(
                load_l0a_cycles, load_l0b_cycles,
                cube_cycles, db_cout_iter_num, db_cin_iter_num,
                self.cout_tail_cycle)
        else:
            cout_db_cycles, db_cycles = \
                self._cal_one_db_value(self.k_tile_thread,
                                       self.cout_tile_thread, load_l0a_cycles,
                                       load_l0b_cycles, cube_cycles,
                                       db_cin_iter_num, db_cout_iter_num,
                                       self.cout_tail_cycle)

        if db_cin_iter_num != self.k_iter_num:
            self.cin_tail_cycle = load_l0a_cycles + cout_db_cycles
        if db_cin_sub:
            self.cin_tail_cycle += load_l0a_cycles + cout_db_cycles
        self.total_cycles = db_cycles + self.cin_tail_cycle

        self.total_l0a_cycles = load_l0a_cycles * self.k_iter_num
        self.total_l0b_cycles = load_l0b_cycles * self.k_iter_num * \
                                self.cout_iter_num

    def _cal_cout_hw_cycles(self,  # pylint: disable=R0913, R0914
                            outer_tile_thread, inner_tile_thread,
                            ori_outer_iter_num, outer_iter_num,
                            inner_iter_num, outer_load_cycles,
                            inner_load_cycles, cube_cycles, tail_cycles):
        no_outer_loop = False
        db_sub_cin = False
        db_cin_iter_num = self.k_iter_num
        if (self.k_tile_thread == 2 and  # pylint: disable=R0916
                outer_iter_num >= 2 and
                inner_iter_num >= 2) or \
                (outer_tile_thread == 2 and inner_tile_thread == 2) or \
                (outer_tile_thread == 2 and inner_iter_num > 1):
            inner_db_cycles, outer_db_cycles = self.cal_double_db_value(
                outer_load_cycles, inner_load_cycles, cube_cycles,
                inner_iter_num, outer_iter_num, tail_cycles)
        elif self.k_tile_thread == 2 and ori_outer_iter_num == 1 and \
                inner_iter_num > 1:
            db_sub_cin, db_cin_iter_num, self.k_tile_thread, _ = \
                self._cal_loop_info(self.cin_has_tail, self.k_tile_thread,
                                    self.k_iter_num)
            inner_db_cycles, outer_db_cycles = self.cal_double_db_value(
                outer_load_cycles, inner_load_cycles,
                cube_cycles, outer_iter_num, db_cin_iter_num,
                self.cout_tail_cycle)
            no_outer_loop = True
        else:
            inner_db_cycles, outer_db_cycles = \
                self._cal_one_db_value(outer_tile_thread,
                                       inner_tile_thread, outer_load_cycles,
                                       inner_load_cycles, cube_cycles,
                                       outer_iter_num, inner_iter_num,
                                       tail_cycles)
        return no_outer_loop, db_sub_cin, db_cin_iter_num, \
               inner_db_cycles, outer_db_cycles

    def _cal_mn_cycles(self, load_l0a_cycles,  # pylint: disable=R0914
                       load_l0b_cycles, cube_cycles):
        tmp_hw_tile_thread = max(self.k_tile_thread, self.howo_tile_thread)
        db_sub_hw, db_hw_iter_num, self.howo_tile_thread, _ = \
            self._cal_loop_info(self.hw_has_tail, tmp_hw_tile_thread,
                                self.howo_iter_num)

        tmp_cout_tile_thread = max(self.cout_tile_thread, tmp_hw_tile_thread)
        _, db_cout_iter_num, self.cout_tile_thread, self.cout_tail_cycle = \
            self._cal_loop_info(self.cout_has_tail, tmp_cout_tile_thread,
                                self.cout_iter_num,
                                load_l0b_cycles + cube_cycles)

        no_hw_loop, db_sub_cin, db_cin_iter_num, cout_db_cycles,\
        hw_db_cycles = self._cal_cout_hw_cycles(self.howo_tile_thread,
                                                self.cout_tile_thread,
                                                self.howo_iter_num,
                                                db_hw_iter_num,
                                                db_cout_iter_num,
                                                load_l0a_cycles,
                                                load_l0b_cycles, cube_cycles,
                                                self.cout_tail_cycle)
        if db_hw_iter_num != self.howo_iter_num:
            self.hw_tail_cycle = load_l0a_cycles + cout_db_cycles
            if db_sub_hw:
                self.hw_tail_cycle += self.hw_tail_cycle

        hw_db_cycles += self.hw_tail_cycle
        self.total_l0a_cycles = load_l0a_cycles * self.k_iter_num * \
                                self.howo_iter_num
        self.total_l0b_cycles = load_l0b_cycles * self.k_iter_num * \
                                self.howo_iter_num * self.cout_iter_num

        if no_hw_loop:
            if db_cin_iter_num != self.k_iter_num:
                self.cin_tail_cycle = load_l0a_cycles + cout_db_cycles
                if db_sub_cin:
                    self.cin_tail_cycle += self.cin_tail_cycle
            self.total_cycles = hw_db_cycles + self.cin_tail_cycle
        else:
            self.total_cycles = self.k_iter_num * hw_db_cycles

    def _cal_nm_cycles(self, load_l0a_cycles,  # pylint: disable=R0914
                       load_l0b_cycles, cube_cycles):
        tmp_howo_tile_thread = max(self.howo_tile_thread,
                                   self.cout_tile_thread, self.k_tile_thread)

        _, db_hw_iter_num, self.howo_tile_thread, self.hw_tail_cycle = \
            self._cal_loop_info(self.hw_tail_cycle, tmp_howo_tile_thread,
                                self.howo_iter_num,
                                load_l0a_cycles + cube_cycles)

        tmp_cout_tile_thread = max(self.cout_tile_thread, self.k_tile_thread)
        db_sub_cout, db_cout_iter_num, self.cout_tile_thread, _ = \
            self._cal_loop_info(self.cout_has_tail, tmp_cout_tile_thread,
                                self.cout_iter_num)

        no_cout_loop, db_sub_cin, db_cin_iter_num, hw_db_cycles, \
        cout_db_cycles = self._cal_cout_hw_cycles(self.cout_tile_thread,
                                                  self.howo_tile_thread,
                                                  self.cout_iter_num,
                                                  db_cout_iter_num,
                                                  db_hw_iter_num,
                                                  load_l0b_cycles,
                                                  load_l0a_cycles,
                                                  cube_cycles,
                                                  self.hw_tail_cycle)

        if db_cout_iter_num != self.cout_iter_num:
            self.cout_tail_cycle = load_l0b_cycles + hw_db_cycles
            if db_sub_cout:
                self.cout_tail_cycle += self.cout_tail_cycle

        cout_db_cycles += self.cout_tail_cycle
        self.total_l0a_cycles = load_l0a_cycles * self.k_iter_num * \
                                self.howo_iter_num * self.cout_iter_num
        self.total_l0b_cycles = load_l0b_cycles * self.k_iter_num * \
                                self.cout_iter_num
        if no_cout_loop:
            if db_cin_iter_num != self.k_iter_num:
                self.cin_tail_cycle = load_l0b_cycles + hw_db_cycles
                if db_sub_cin:
                    self.cin_tail_cycle += self.cin_tail_cycle
            self.total_cycles = cout_db_cycles + self.cin_tail_cycle
        else:
            self.total_cycles = self.k_iter_num * cout_db_cycles

    def cal_cycles(self):
        """calulate the cycles of different tiling"""
        load_l0a_cycles = self._get_load_l0a_cycles()
        load_l0b_cycles = self._get_load_l0b_cycles()
        cube_cycles = self._get_cube_cycles()
        if self.loop_mode == "nk":
            self._cal_nk_cycles(load_l0a_cycles, load_l0b_cycles, cube_cycles)
        elif self.loop_mode == "kn":
            self._cal_kn_cycles(load_l0a_cycles, load_l0b_cycles, cube_cycles)
        elif self.loop_mode == "mn":
            self._cal_mn_cycles(load_l0a_cycles, load_l0b_cycles, cube_cycles)
        else:
            self._cal_nm_cycles(load_l0a_cycles, load_l0b_cycles, cube_cycles)

        self.total_cube_cycles = cube_cycles * self.k_iter_num * \
                                 self.howo_iter_num * self.cout_iter_num

    def set_load_mode(self):
        """set l0a and l0b load mode"""
        if self.howo_tile_block // 2 < self.k_tile_block:
            self.l0a_mode = 0
        else:
            self.l0a_mode = 1

        if self.cout_tile_block == 1 and \
                self.k_tile_block * self.block_size * self.c_0 * 2 * \
                get_bytes(self.fm_desc.input_dtype) < \
                self.tik_instance.d_profiling.get_l0b_buffer_size():
            self.l0b_mode = 1
        else:
            self.l0b_mode = 0

    def __str__(self):
        return "HoWo: tile_block {} thread {} iter {} has tail {}\n" \
               "Cin: tile_block {} thread {} iter {} has tail {}\n" \
               "Cout tile_block {} thread {} iter {} has tail {}\n" \
               "loop mode {} l0a mode {} l0b mode {}\n"\
               "total_cycles {} total cube {} total l0a {} total l0b {}\n"\
            .format(
                self.howo_tile_block, self.howo_tile_thread,
                self.howo_iter_num, self.hw_has_tail, self.k_tile_block,
                self.k_tile_thread, self.k_iter_num, self.cin_has_tail,
                self.cout_tile_block, self.cout_tile_thread,
                self.cout_iter_num, self.cout_has_tail, self.loop_mode,
                self.l0a_mode, self.l0b_mode, self.total_cycles,
                self.total_cube_cycles, self.total_l0a_cycles,
                self.total_l0b_cycles)

    def __lt__(self, other):
        return self.total_cycles < other.total_cycles
# @endcond


# @cond
class ValidTilingInfo():  # pylint: disable=R0902
    """valid tiling info"""
    def __init__(self, fm_desc, filter_desc, is_db_for_loop_flag,
                 tik_instance):
        self.fm_desc = fm_desc
        self.filter_desc = filter_desc
        self.tik_instance = tik_instance
        self.ho_wo = fm_desc.h_o * fm_desc.w_o
        self.block_size = 16
        self.howo_block_num = math.ceil(self.ho_wo / self.block_size)
        self.c_0 = 16 if fm_desc.input_dtype == "float16" else 32

        self.cin_block_num = math.ceil(filter_desc.cin / self.c_0)
        self.cout_block_num = math.ceil(filter_desc.cout / self.block_size)

        self.hk_wk = filter_desc.height * filter_desc.width
        self.fm_input_dtype_size = get_bytes(fm_desc.input_dtype)
        self.fm_output_dtype_size = get_bytes(fm_desc.output_dtype)
        self.filter_dtype_size = get_bytes(filter_desc.dtype)

        self.valid_solution = []

        # only Cin loop need double buffer
        self.hw_thread_num = 1

        # flag indicating whether current instr in double buffer for range,
        # if not, set thread_num of the tiling outermost for loop as 2
        self.is_db_for_loop_flag = is_db_for_loop_flag

        # we don't tile HoWo to keep L0C format
        self.howo_tile_block_num = self.howo_block_num
        self.k_block_num = self.hk_wk * self.cin_block_num

    def gen_nk_tiling(self):
        """generate nk tiling"""
        loop_modes = [(2, 1, "kn"), (1, 2, "nk")]
        if not self.is_db_for_loop_flag:
            loop_modes = [(2, 2, "kn"), (1, 2, "kn"),
                          (1, 2, "nk")]

        for cout_thread_num, cin_thread_num, loop_mode in loop_modes:
            for k_tile_block in range(1, self.k_block_num + 1):
                for cout_tile_block_num in range(1, self.cout_block_num + 1):
                    cin_iter_num = math.ceil(
                        self.k_block_num / k_tile_block)
                    cout_iter_num = math.ceil(
                        self.cout_block_num / cout_tile_block_num)

                    if cin_thread_num > cin_iter_num:
                        continue
                    if cout_thread_num > cout_iter_num:
                        continue

                    # L0A size: (HoWo, Cin, Hk, Wk)
                    if loop_mode == "kn":
                        l0a_fm_size = self.howo_tile_block_num * \
                                      k_tile_block * self.block_size * \
                                      self.c_0 * self.fm_input_dtype_size * \
                                      cin_thread_num
                    else:
                        l0a_fm_size = self.howo_tile_block_num * \
                                      k_tile_block * self.block_size * \
                                      self.c_0 * self.fm_input_dtype_size * \
                                      max(cout_thread_num, cin_thread_num)
                    if l0a_fm_size > \
                            self.tik_instance.d_profiling.get_l0a_buffer_size():
                        continue
                    # L0B size: (Cin, HkWk, Co)
                    l0b_filter_size = k_tile_block * cout_tile_block_num * \
                                      self.block_size * self.c_0 * \
                                      self.filter_dtype_size * \
                                      max(cout_thread_num, cin_thread_num)
                    if l0b_filter_size > \
                            self.tik_instance.d_profiling.get_l0b_buffer_size():
                        continue

                    # L0C size: (HoWo, Cout)
                    l0c_ps_size = \
                        self.howo_tile_block_num * cout_tile_block_num * \
                        self.block_size * self.c_0 * \
                        self.fm_output_dtype_size * \
                        max(cout_thread_num, cin_thread_num)
                    if l0c_ps_size > \
                            self.tik_instance.d_profiling.get_l0c_buffer_size():
                        continue

                    tile_info0 = ConvTileInfo(
                        self.fm_desc, self.filter_desc,
                        self.howo_tile_block_num,
                        self.hw_thread_num, k_tile_block,
                        cin_thread_num, cout_tile_block_num,
                        cout_thread_num, self.c_0, loop_mode,
                        self.tik_instance)
                    self.valid_solution.append(tile_info0)

    def gen_mn_tiling(self):  # pylint: disable=R0914
        """generate mn tiling"""
        cin_thread_num = 1
        if not self.is_db_for_loop_flag:
            cin_thread_num = 2
        # MN mode
        cout_tile_block_num = 1

        loop_modes = [(2, 2, "mn"), (2, 1, "mn"), (1, 2, "mn"),
                      (2, 2, "nm"), (2, 1, "nm"), (1, 2, "nm")]

        for hw_thread_num, cout_thread_num, loop_mode in loop_modes:
            for k_tile_block in range(1, self.k_block_num + 1):
                for howo_tile_block_num in range(1, self.howo_block_num + 1):
                    hw_iter_num = math.ceil(
                        self.howo_block_num / howo_tile_block_num)
                    cin_iter_num = math.ceil(
                        self.hk_wk * self.cin_block_num / k_tile_block)
                    cout_iter_num = math.ceil(
                        self.cout_block_num / cout_tile_block_num)

                    if hw_thread_num > hw_iter_num:
                        continue
                    if cin_thread_num > cin_iter_num:
                        continue
                    if cout_thread_num > cout_iter_num:
                        continue

                    # L0A size: (HoWo, Cin, Hk, Wk)
                    if loop_mode == "mn":
                        l0a_fm_size = \
                            howo_tile_block_num * k_tile_block * \
                            self.block_size * self.c_0 * \
                            self.fm_input_dtype_size * \
                            max(cin_thread_num, hw_thread_num)
                    else:
                        l0a_fm_size = \
                            howo_tile_block_num * k_tile_block * \
                            self.block_size * self.c_0 * \
                            self.fm_input_dtype_size * \
                            max(cin_thread_num, cout_thread_num,
                                hw_thread_num)
                    if l0a_fm_size > \
                            self.tik_instance.d_profiling.get_l0a_buffer_size():
                        continue
                    # L0B size: (Cin, HkWk, Co)
                    if loop_mode == "mn":
                        l0b_filter_size = \
                            k_tile_block * cout_tile_block_num * \
                            self.block_size * self.c_0 * \
                            self.filter_dtype_size * \
                            max(cin_thread_num, cout_thread_num,
                                hw_thread_num)
                    else:
                        l0b_filter_size = \
                            k_tile_block * cout_tile_block_num * \
                            self.block_size * self.c_0 * \
                            self.filter_dtype_size * \
                            max(cin_thread_num, cout_thread_num)
                    if l0b_filter_size > \
                            self.tik_instance.d_profiling.get_l0b_buffer_size():
                        continue

                    # L0C size: (HoWo, Cout)
                    l0c_ps_size = \
                        howo_tile_block_num * cout_tile_block_num * \
                        self.block_size * self.c_0 * \
                        self.fm_output_dtype_size * \
                        max(cin_thread_num, cout_thread_num, hw_thread_num)
                    if l0c_ps_size > \
                            self.tik_instance.d_profiling.get_l0c_buffer_size():
                        continue

                    tile_info0 = ConvTileInfo(
                        self.fm_desc, self.filter_desc, howo_tile_block_num,
                        hw_thread_num, k_tile_block, cin_thread_num,
                        cout_tile_block_num, cout_thread_num,
                        self.c_0, loop_mode, self.tik_instance)
                    self.valid_solution.append(tile_info0)
# @endcond


def gen_best_tiling(fm_desc, filter_desc, is_db_for_loop_flag, tik_instance):
    """generate best tiling"""
    valid_tiling = ValidTilingInfo(fm_desc, filter_desc, is_db_for_loop_flag,
                                   tik_instance)
    valid_tiling.gen_nk_tiling()
    valid_tiling.gen_mn_tiling()
    # add one fractal tile info
    if not valid_tiling.valid_solution:
        howo_tile_block_num = 1
        hw_thread_num = 1
        c_0 = 16 if fm_desc.input_dtype == "float16" else 32
        k_tile_block = 1
        cin_thread_num = 1
        loop_mode = "nk"
        cout_tile_block_num = 1
        cout_thread_num = 1
        tile_info0 = ConvTileInfo(
            fm_desc, filter_desc, howo_tile_block_num,
            hw_thread_num, k_tile_block, cin_thread_num,
            cout_tile_block_num, cout_thread_num,
            c_0, loop_mode, tik_instance)
        valid_tiling.valid_solution.append(tile_info0)
    valid_tiling.valid_solution.sort()
    return valid_tiling.valid_solution[0]


# @cond
class FixpipeTileInfo(): # pylint: disable=R0902
    """fixpipe tiling info"""
    vector_issue_heuristic = 24
    max_repeat = 255
    def __init__(self, l1out_blocks, # pylint: disable=R0913
                 l1out_tile_blocks,
                 l1out_thread_num,
                 howo_blocks,
                 howo_tile_blocks,
                 howo_thread_num,
                 bias_blocks,
                 bias_repeat,
                 bias_tail_repeat,
                 has_bias,
                 has_deq,  # if no bias, deq does not need vector
                 vconv_merge_channel):
        self.has_bias = has_bias
        self.has_deq = has_deq
        self.vconv_merge_channel = vconv_merge_channel
        self.l1out_iter_num = math.ceil(l1out_blocks / l1out_tile_blocks)
        self.howo_iter_num = math.ceil(howo_blocks / howo_tile_blocks)
        self.l1out_blocks = l1out_blocks
        self.l1out_tile_blocks = l1out_tile_blocks
        self.l1out_thread_num = l1out_thread_num
        self.howo_blocks = howo_blocks
        self.howo_tile_blocks = howo_tile_blocks
        self.howo_thread_num = howo_thread_num

        self.repeat_per_block = 16 * 16 // 128

        self.bias_blocks = bias_blocks
        self.bias_tail_repeat = bias_tail_repeat

        self.l0c_to_ub_count = 1
        self.bias_repeat = bias_repeat

        # issue count for 8 block bias
        self.bias_issue_count = math.ceil(
            howo_tile_blocks * self.repeat_per_block / self.max_repeat)
        self.bias_count = bias_repeat * self.bias_issue_count

        if bias_tail_repeat > 0:
            self.bias_tail_issue_count = math.ceil(
                howo_tile_blocks * self.repeat_per_block / self.max_repeat)
            self.bias_count += self.bias_tail_issue_count

        self.deq_count = math.ceil(howo_tile_blocks * l1out_tile_blocks *
                                   self.repeat_per_block / self.max_repeat)
        self.vector_issue_num = \
            (self.l0c_to_ub_count + self.bias_count + self.deq_count) * \
            self.l1out_iter_num * self.howo_iter_num

        partial_utilization = self.l1out_tile_blocks / \
                              (math.ceil(self.l1out_tile_blocks/8)*8)

        self.vector_bias_utilization = 1
        if has_bias:
            self.vector_bias_utilization = partial_utilization
        self.vector_fp16_to_b8_utilization = 1
        if vconv_merge_channel:
            self.vector_fp16_to_b8_utilization = partial_utilization

        self.vector_utilization = (self.vector_bias_utilization +
                                   self.vector_fp16_to_b8_utilization) / 2

        self.howo_has_tail = howo_blocks != howo_tile_blocks * \
                             self.howo_iter_num
        self.l1out_has_tail = l1out_blocks != l1out_tile_blocks * \
                              self.l1out_iter_num

        self.howo_tail_blk = howo_blocks - (self.howo_iter_num - 1) * \
                             howo_tile_blocks
        self.l1out_tail_blk = l1out_blocks - (self.l1out_iter_num - 1) * \
                              l1out_tile_blocks


    def get_vector_issue_num(self):
        """get vector issue num"""
        return self.vector_issue_num

    def get_vector_utilization(self):
        """get vector utilization"""
        return self.vector_utilization

    def __lt__(self, other):
        # maximize vector utilization and minimize vector issue count
        if self.get_vector_utilization() > other.get_vector_utilization():
            return True
        elif self.get_vector_utilization() < other.get_vector_utilization():
            return False

        if self.get_vector_issue_num() < other.get_vector_issue_num():
            return True
        elif self.get_vector_issue_num() > other.get_vector_issue_num():
            return False

        if self.l1out_thread_num > other.l1out_thread_num:
            return True
        elif self.l1out_thread_num < other.l1out_thread_num:
            return False

        if self.howo_thread_num > other.howo_thread_num:
            return True
        elif self.howo_thread_num < other.howo_thread_num:
            return False

        if self.l1out_tile_blocks < other.l1out_tile_blocks:
            return True
        elif self.l1out_tile_blocks > other.l1out_tile_blocks:
            return False

        return self.howo_tile_blocks < other.howo_tile_blocks

    def __str__(self):
        return "FixpipeTileInfo:\n" \
               "l1out blocks:{} tile:{} thread:{}\n" \
               "hw blocks:{} tile:{} thread:{}\n" \
               "vector issue num {} " \
               "vector utilization {}\n" \
               "has bias {} " \
               "has vconv deq {} " \
               "has vconv merge channel {}"\
            .format(self.l1out_blocks, self.l1out_tile_blocks,
                    self.l1out_thread_num, self.howo_blocks,
                    self.howo_tile_blocks, self.howo_thread_num,
                    self.vector_issue_num, self.vector_utilization,
                    self.has_bias, self.has_deq, self.vconv_merge_channel)
# @endcond


def gen_fixpipe_tiling( # pylint: disable=R0914, R0913
        l1out_column, howo_blocks, l1out_dtype, out_dtype, has_bias,
        bias_size, has_ele_wise_bias, tik_instance, deq_value):
    """get fixpipe tiling"""
    tilings = []
    l1out_dtype_size = get_bytes(l1out_dtype)
    out_dtype_size = get_bytes(out_dtype)
    vconv_deq = l1out_dtype != out_dtype
    vconv_merge_channel = False
    l1out_min_column = 1

    if l1out_dtype == "int32" and out_dtype in ("uint8", "int8"):
        vconv_merge_channel = True
        # at least 2 column in s32 s8 case
        l1out_min_column = 2

    src_element_512_byte = 512 // l1out_dtype_size

    block_size = 16
    bias_vec_block_size = 16 if l1out_dtype_size == "float16" else 8
    vec_blocks = 8

    bias_block = bias_size // bias_vec_block_size

    bias_repeats = bias_block // vec_blocks
    bias_tail_blocks = bias_block % vec_blocks

    real_bias_size = bias_repeats * 256
    bias_tail_duplicate_time = 0

    if bias_tail_blocks > 0:
        # we should broadcast bias to make use of vector
        bias_tail_duplicate_time = 8 // bias_block
        real_bias_size += 256

    user_db_thread_num = 2 if tik_instance.is_double_buffer_for_loop else 1

    for l1out_thread_num in (1, 2):
        for howo_thread_num in (1, 2):
            for l1out_tile_blocks in range(l1out_min_column, l1out_column + 1):
                for howo_tile_blocks in range(1, howo_blocks + 1):
                    l1out_loop_iter = math.ceil(l1out_column / l1out_tile_blocks)
                    howo_loop_iter = math.ceil(howo_blocks / howo_tile_blocks)

                    if l1out_thread_num > l1out_loop_iter:
                        continue
                    if howo_thread_num > howo_loop_iter:
                        continue

                    actual_db_thread_num = max(l1out_thread_num,
                                               howo_thread_num,
                                               user_db_thread_num)

                    tile_element_count = \
                        l1out_tile_blocks * (block_size + howo_tile_blocks *
                                             block_size * block_size)
                    # round to 512 byte, thus it's safe to vector simd instr
                    tile_element_count = \
                        math.ceil(tile_element_count / src_element_512_byte) * \
                        src_element_512_byte
                    l1out_tile_size = tile_element_count * l1out_dtype_size * \
                                      actual_db_thread_num

                    ub_used = _get_ub_used(
                        l1out_tile_size, has_bias, vconv_deq,
                        vconv_merge_channel, real_bias_size, tile_element_count,
                        out_dtype_size, l1out_dtype, has_ele_wise_bias,
                        deq_value, actual_db_thread_num)

                    if ub_used > \
                            tik_instance.d_profiling.get_unified_buffer_size():
                        continue
                    tile_info = FixpipeTileInfo(
                        l1out_column, l1out_tile_blocks, l1out_thread_num,
                        howo_blocks, howo_tile_blocks, howo_thread_num,
                        bias_block, bias_repeats, bias_tail_duplicate_time,
                        has_bias, vconv_deq, vconv_merge_channel)
                    tilings.append(tile_info)
    best_tiling = sorted(tilings)[0]
    return best_tiling


def _get_ub_used( # pylint: disable=R0913
        l1out_tile_size, has_bias, vconv_deq, vconv_merge_channel,
        real_bias_size, tile_element_count, out_dtype_size,
        l1_out_dtype, has_ele_wise_bias, deq_value, actual_db_thread_num):
    ub_used = l1out_tile_size
    if has_bias:
        # don't reuse space to avoid rw bankconflit
        ub_used += real_bias_size + l1out_tile_size * actual_db_thread_num
    if has_ele_wise_bias:
        # element-wise-add apply two tensor(shape is tile_element_count)
        ub_used += tile_element_count*get_bytes(l1_out_dtype)*2 * \
                   actual_db_thread_num
    if vconv_deq:
        deq_dtype_size = get_bytes("float16")
        ub_used += tile_element_count * deq_dtype_size * actual_db_thread_num
        if isinstance(deq_value, Tensor):
            # apply tensor to move deqscale(float16) from l1 to ub
            ub_used += 16*deq_dtype_size
    if vconv_merge_channel:
        ub_used += tile_element_count * out_dtype_size * actual_db_thread_num

    return ub_used
