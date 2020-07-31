"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

spp_pooling
"""

from te import tik
from topi.cce import util
from te import platform as tbe_platform

FP16_MINI = -65504
FP32_MINI = -3.4 * (10**38)
POOLING_CEIL = 0
POOLING_FLOOR = 1
MAX_POOLING = 0
AVG_POOLING = 1
UB_RESERVED = 2048
DBL_BUF_SW = True


def check_param(x_dic, y_dic, param_dic, kernel_name):
    """
    check ops param interface

    Parameters
    ----------
    x_dic : shape/dtype/fmt of input
    y_dic : shape/dtype/fmt of output
    param_dic : pooling param
    kernel_name : kernel name

    Returns
    -------
    None
    """
    x_shape_val = x_dic['shape']
    y_shape_val = y_dic['shape']
    dtype_val = x_dic['dtype']

    util.check_shape_rule(x_shape_val, min_dim=5, max_dim=5,
                          max_shape_num=10**8)
    util.check_shape_rule(y_shape_val, min_dim=5, max_dim=5,
                          max_shape_num=10**8)
    util.check_shape_size(x_shape_val, 2**31)
    util.check_shape_size(y_shape_val, 2**31)


    tik_name = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
    if tik_name == "Hi3796CV300ES":
        util.check_dtype_rule(dtype_val.lower(), ["float16"])
    else:
        util.check_dtype_rule(dtype_val.lower(), ["float16", "float32"])

    util.check_kernel_name(kernel_name)

    if param_dic['window'][0] < 1 or param_dic['window'][1] < 1:
        raise ValueError("window value must be Greater than 1")
    if param_dic['stride'][0] < 1 or param_dic['stride'][1] < 1:
        raise ValueError("stride value must be Greater than 1")

    if param_dic['pad'][0] > param_dic['window'][0] or \
            param_dic['pad'][2] > param_dic['window'][1]:
        raise ValueError("pad value must not be grater than window value")

    if param_dic["mode"] != AVG_POOLING and \
            param_dic["mode"] != MAX_POOLING:
        raise ValueError("Now Pooling mode only support AVG and MAX")

    if param_dic["ceil_mode"] != POOLING_CEIL and \
            param_dic["ceil_mode"] != POOLING_FLOOR:
        raise ValueError("Now Ceil mode only support CEIL and FLOOR")


class BaseParam:
    """
    Define Base Param
    """
    def __init__(self, x_param):
        self.dtype = x_param['dtype']
        self.dtype_size = 2 if (x_param['dtype'] == "float16") else 4
        self.shape = x_param['shape']
        self.in_size = {'n': 0, 'c1': 0, 'h': 0, 'w': 0}
        self.out_size = {'n': 0, 'c1': 0, 'h': 0, 'w': 0}
        self.ubuf = {'total': 0, 'avail': 0}
        self.mask = 256 // self.dtype_size

    def set_size(self, direction, shape):
        """
        set in/out size of per n/c1/h/w
        Parameters
        ----------
        direction: input or output
        shape: input or output shape
        Returns
        -------
        None
        """
        if direction == "IN":
            self.in_size['n'] = shape[1] * shape[2] * shape[3] * shape[4]
            self.in_size['c1'] = shape[2] * shape[3] * shape[4]
            self.in_size['h'] = shape[3] * shape[4]
            self.in_size['w'] = shape[4]
        else:
            self.out_size['n'] = shape[1] * shape[2] * shape[3] * shape[4]
            self.out_size['c1'] = shape[2] * shape[3] * shape[4]
            self.out_size['h'] = shape[3] * shape[4]
            self.out_size['w'] = shape[4]

    def set_ub_param(self, double_buf_switch, double_buf_cond):
        """
        set ub param
        Parameters
        ----------
        double_buf_switch: double buffer switch
        double_buf_cond: double buffer condition
        Returns
        -------
        None
        """
        self.ubuf['total'] = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
        if double_buf_switch is True and double_buf_cond > 1:
            self.ubuf['avail'] = (self.ubuf['total'] - 2 * UB_RESERVED) // 2
        else:
            self.ubuf['avail'] = self.ubuf['total'] - UB_RESERVED


class CurParam(BaseParam):
    """
    Define Current Param
    """
    def __init__(self, tik_inst, x_param):
        super(CurParam, self).__init__(x_param)
        self.cur_h = tik_inst.Scalar(dtype="int32", name="cur_h")
        self.cur_n_c1 = tik_inst.Scalar(dtype="int32", name="cur_n_c1")
        self.cur_ph = tik_inst.Scalar(dtype="int32", name="cur_ph")
        self.cur_pw = tik_inst.Scalar(dtype="int32", name="cur_pw")

    def set_cur_n_c1(self, cur_n_c1):
        """
        set current n*c1
        Parameters
        ----------
        cur_n_c1: current n*c1 value
        Returns
        -------
        None
        """
        self.cur_n_c1.set_as(cur_n_c1)

    def set_cur_h(self, cur_h):
        """
        set current height position
        Parameters
        ----------
        cur_h: current h value
        Returns
        -------
        None
        """
        self.cur_h.set_as(cur_h)


class PoolingParam(CurParam):
    """
    Define Pooling Param
    """
    def __init__(self, tik_inst, x_param, attr_param):
        super(PoolingParam, self).__init__(tik_inst, x_param)
        self.window = attr_param['window']
        self.stride = attr_param['stride']
        self.pad = attr_param['pad']
        self.mode = attr_param['mode']
        self.global_pooling = attr_param['global_pooling']

        if attr_param['global_pooling'] is True:
            shape = x_param['shape']
            self.window = (shape[2], shape[3])
            self.stride = (1, 1)
            self.pad = (0, 0, 0, 0)

    def set_pooling_param(self, attr_param):
        """
        set pooling attr_param
        Parameters
        ----------
        attr_param: pooling attr_param
        Returns
        -------
        None
        """
        self.window = attr_param['window']
        self.stride = attr_param['stride']
        self.pad = attr_param['pad']
        self.mode = attr_param['mode']
        self.global_pooling = attr_param['global_pooling']


class PoolingCommon(PoolingParam):
    """
    Define Pooling common option
    """
    def __init__(self, x_param, attr_param, base_param):
        self.tik_inst = base_param['tik_inst']
        super(PoolingCommon, self).__init__(self.tik_inst, x_param, attr_param)
        self.out = base_param['out']

        self.h_start = self.tik_inst.Scalar(dtype="int64", name="h_start")
        self.w_start = self.tik_inst.Scalar(dtype="int64", name="w_start")
        self.h_end = self.tik_inst.Scalar(dtype="int64", name="h_end")
        self.w_end = self.tik_inst.Scalar(dtype="int64", name="w_end")

        double_buf_switch = DBL_BUF_SW
        self.thread_num = 2 if (double_buf_switch and self.out['h'] > 1) else 1
        self.set_ub_param(double_buf_switch=double_buf_switch,
                          double_buf_cond=self.out['h'])

        if self.window[1] * self.shape[4] > \
                (self.ubuf['avail'] - 256) // self.dtype_size:
            RuntimeError("(window_w*c0) must be less than Available UB")

    def set_pooling_pos(self, cur_ph, cur_pw):
        """
        set current pooling position
        Parameters
        ----------
        cur_ph: current pooling height
        cur_pw: current pooling width
        Returns
        -------
        NONE
        """
        self.cur_ph.set_as(cur_ph)
        self.cur_pw.set_as(cur_pw)
        self.h_start.set_as(cur_ph * self.stride[0] - self.pad[0])
        self.w_start.set_as(cur_pw * self.stride[1] - self.pad[2])
        calc_h = self.tik_inst.Scalar(dtype="int64", name="calc_h")
        calc_w = self.tik_inst.Scalar(dtype="int64", name="calc_w")
        calc_h.set_as(self.h_start + self.window[0])
        calc_w.set_as(self.w_start + self.window[1])
        self.tik_inst.scalar_min(self.h_end, calc_h, self.shape[2])
        self.tik_inst.scalar_min(self.w_end, calc_w, self.shape[3])
        self.tik_inst.scalar_max(self.h_start, self.h_start, 0)
        self.tik_inst.scalar_max(self.w_start, self.w_start, 0)


class PoolingCompute(PoolingCommon):
    """
    Define Pooling compute option
    """
    def __init__(self, x_param, attr_param, base_param):
        super(PoolingCompute, self).__init__(x_param, attr_param, base_param)
        shape_in = x_param['shape']
        shape_out = (shape_in[0], shape_in[1],
                     self.out['h'], self.out['w'], shape_in[4])

        self.set_size("IN", shape_in)
        self.set_size("OUT", shape_out)

        self.src_gm = base_param['src_gm']
        self.dst_gm = base_param['dst_gm']

        self.src_ub = self.tik_inst.Tensor(self.dtype, (self.ubuf['avail'] //
                                                        self.dtype_size, ),
                                           scope=tik.scope_ubuf, name="src_ub")
        self.dst_ub = self.tik_inst.Tensor(self.dtype, (512//self.dtype_size, ),
                                           scope=tik.scope_ubuf, name="dst_ub")
        self.res_ub = self.tik_inst.Tensor(self.dtype, (16, ),
                                           scope=tik.scope_ubuf, name="res_ub")

    def data_load(self, load_size):
        """
        load data
        Parameters
        ----------
        load_size: load size
        Returns
        -------
        None
        """
        pooling_w = self.w_end - self.w_start
        burst_len = pooling_w * self.shape[4] * self.dtype_size // 32
        nburst = load_size // pooling_w
        src_stride = (self.shape[3] - pooling_w) * \
                     self.in_size['w'] * self.dtype_size // 32
        dst_stride = 0

        nburst_loop = load_size // pooling_w // 4095
        nburst_tail = load_size // pooling_w % 4095
        with self.tik_inst.for_range(0, nburst_loop) as nburst_loopi:
            src_ofst = self.cur_n_c1 * self.in_size['c1'] + \
                       self.cur_h * self.in_size['h'] + \
                       self.w_start * self.in_size['w'] + \
                       nburst_loopi * 4095 * pooling_w * self.shape[4]
            dst_ofst = nburst_loopi * 4095 * pooling_w * self.shape[4]
            self.tik_inst.data_move(self.src_ub[dst_ofst],
                                    self.src_gm[src_ofst],
                                    0, 4095, burst_len,
                                    src_stride, dst_stride)

        with self.tik_inst.if_scope(nburst_tail > 0):
            src_ofst = self.cur_n_c1 * self.in_size['c1'] + \
                       self.cur_h * self.in_size['h'] + \
                       self.w_start * self.in_size['w'] + \
                       nburst_loop * 4095 * pooling_w * self.shape[4]
            dst_ofst = nburst_loop * 4095 * pooling_w * self.shape[4]
            self.tik_inst.data_move(self.src_ub[dst_ofst],
                                    self.src_gm[src_ofst],
                                    0, nburst_tail, burst_len,
                                    src_stride, dst_stride)

        self.set_cur_h(self.cur_h + nburst)

    def data_store(self):
        """
        load store
        Parameters
        ----------

        Returns
        -------
        None
        """
        dst_ofst = self.cur_n_c1 * self.out_size['c1'] + \
                   self.cur_ph * self.out_size['h'] + \
                   self.cur_pw * self.out_size['w']
        self.tik_inst.data_move(self.dst_gm[dst_ofst], self.res_ub, 0, 1,
                                self.shape[4] * self.dtype_size // 32, 0, 0)

    def pooling_vop_1(self, repeats, ofst):
        """
        pooling compute
        Parameters
        ----------
        repeats: vector op repeat_times
        ofst : src_ub offset
        Returns
        -------
        None
        """
        if self.mode == AVG_POOLING:
            self.tik_inst.vadd(self.mask, self.dst_ub, self.src_ub[ofst],
                               self.dst_ub, repeats, 1, 1, 1, 8, 8, 8)
        else:
            self.tik_inst.vmax(self.mask, self.dst_ub, self.src_ub[ofst],
                               self.dst_ub, repeats, 1, 1, 1, 8, 8, 8)

    def pooling_vop_2(self, repeats, ofst):
        """
        pooling compute
        Parameters
        ----------
        repeats: vector op repeat_times
        ofst : src_ub offset
        Returns
        -------
        None
        """
        if self.mode == AVG_POOLING:

            self.tik_inst.vadd(self.mask, self.dst_ub, self.src_ub[ofst],
                               self.dst_ub, repeats, 1, 1, 1, 0, 8, 0)
        else:
            self.tik_inst.vmax(self.mask, self.dst_ub, self.src_ub[ofst],
                               self.dst_ub, repeats, 1, 1, 1, 0, 8, 0)

    def run_compute(self, compute_size):
        """
        pooling compute
        Parameters
        ----------
        compute_size : size of per compute
        Returns
        -------
        None
        """
        self.data_load(compute_size)
        init_val = 0 if (self.mode == AVG_POOLING) else \
            (FP16_MINI if (self.dtype == "float16") else FP32_MINI)
        self.tik_inst.vector_dup(self.mask,
                                 self.src_ub[compute_size*self.shape[4]],
                                 init_val, 1, 1, 8)

        if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") == "Ascend310" \
                and self.dtype == "float32":
            run_loop = (compute_size * self.shape[4] *
                        self.dtype_size + 255) // 256
            with self.tik_inst.for_range(0, run_loop) as run_i:
                self.pooling_vop_1(1, run_i * 256 // self.dtype_size)
        else:
            repeats_loop = compute_size * self.shape[4] * \
                           self.dtype_size // 65280
            repeats_tail = compute_size * self.shape[4] * \
                           self.dtype_size % 65280

            with self.tik_inst.for_range(0, repeats_loop) as loopi:
                self.pooling_vop_2(255, loopi * 65280 // self.dtype_size)

            with self.tik_inst.if_scope(repeats_tail > 0):
                self.pooling_vop_2((repeats_tail + 255) // 256,
                                   repeats_loop * 65280 // self.dtype_size)

        tmp_ub = self.tik_inst.Tensor(self.dtype, (256 // self.dtype_size, ),
                                      scope=tik.scope_ubuf, name="tmp_ub")
        self.tik_inst.vector_dup(self.mask, tmp_ub, init_val, 1, 1, 8)

        cycle = 3 if (self.dtype == "float16") else 2
        for i in range(0, cycle):
            if self.mode == AVG_POOLING:
                self.tik_inst.data_move(tmp_ub, self.dst_ub[256 //
                                                            self.dtype_size //
                                                            2**(i+1)],
                                        0, 1, 8, 0, 0)
                self.tik_inst.vadd(self.mask, self.dst_ub, self.dst_ub,
                                   tmp_ub, 1, 1, 1, 1, 8, 8, 8)

                self.tik_inst.vector_dup(self.mask,
                                         self.dst_ub[256//self.dtype_size //
                                                     2**(i+1)],
                                         0, 1, 1, 8)
                self.tik_inst.vector_dup(self.mask, tmp_ub, 0, 1, 1, 8)
            else:
                self.tik_inst.data_move(tmp_ub,
                                        self.dst_ub[256 // self.dtype_size //
                                                    2**(i+1)], 0, 1, 8, 0, 0)
                self.tik_inst.vmax(self.mask, self.dst_ub, self.dst_ub,
                                   tmp_ub, 1, 1, 1, 1, 8, 8, 8)
                self.tik_inst.vector_dup(self.mask, tmp_ub, init_val, 1, 1, 8)

        if self.mode == AVG_POOLING:
            self.tik_inst.vadd(16, self.res_ub, self.res_ub, self.dst_ub,
                               1, 1, 1, 1, 8, 8, 8)
        else:
            self.tik_inst.vmax(16, self.res_ub, self.res_ub, self.dst_ub,
                               1, 1, 1, 1, 8, 8, 8)

    def avg_calc(self):
        """
        calc avg
        Parameters
        ----------
        Returns
        -------
        None
        """
        if self.mode == AVG_POOLING:
            pooling_size = self.window[0] * self.window[1]
            coef = 1 / pooling_size
            self.tik_inst.vmuls(16, self.res_ub, self.res_ub,
                                coef, 1, 1, 1, 8, 8)

    def run_pooling(self, compute_id):
        """
        start pooling
        Parameters
        ----------
        compute_id : compute id, n*c1 id
        Returns
        -------
        None
        """
        with self.tik_inst.for_range(0, self.out['h'],
                                     thread_num=self.thread_num) as ph_i:
            with self.tik_inst.for_range(0, self.out['w']) as pw_i:
                self.set_pooling_pos(ph_i, pw_i)
                self.set_cur_n_c1(compute_id)
                self.set_cur_h(self.h_start)
                pooling_h = self.h_end - self.h_start
                pooling_w = self.w_end - self.w_start
                pool_real_size = pooling_h * pooling_w

                init_val = 0 if (self.mode == AVG_POOLING) else \
                    (FP16_MINI if (self.dtype == "float16") else FP32_MINI)
                self.tik_inst.vector_dup(self.mask, self.dst_ub,
                                         init_val, 2, 1, 8)
                self.tik_inst.vector_dup(16, self.res_ub, init_val, 1, 1, 8)

                compute_size = self.tik_inst.Scalar(dtype="int32",
                                                    name="compute_size")
                compute_size.set_as((self.ubuf['avail'] - 256) //
                                    (self.shape[4] * self.dtype_size) //
                                    pooling_w * pooling_w)
                compute_loop = pool_real_size // compute_size
                compute_tail = pool_real_size % compute_size

                with self.tik_inst.for_range(0, compute_loop):
                    self.run_compute(compute_size)
                    self.tik_inst.vector_dup(self.mask, self.dst_ub,
                                             init_val, 2, 1, 8)
                with self.tik_inst.if_scope(compute_tail > 0):
                    self.run_compute(compute_tail)

                self.avg_calc()
                self.data_store()


def compute_in_core(x_param, pooling_attr, base_param):
    """
    pooling compute core distribute
    Parameters
    ----------
    x_param : input shape, type param
    pooling_attr : pooling param
    base_param : tik_inst, src/dst gm, out size
    Returns
    -------
    None
    """
    compute_num = x_param['shape'][0] * x_param['shape'][1]
    tik_inst = base_param['tik_inst']

    block_num = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
    if compute_num > block_num:
        core_div_num = block_num
        core_div_loops = compute_num // block_num
        core_div_tail = compute_num % block_num
    else:
        core_div_num = compute_num
        core_div_loops = 0
        core_div_tail = compute_num

    with tik_inst.for_range(0, core_div_num, block_num=core_div_num) as block_i:
        pooling = PoolingCompute(x_param, pooling_attr, base_param)
        with tik_inst.for_range(0, core_div_loops) as loop_i:
            pooling.run_pooling(loop_i * core_div_num + block_i)
        if core_div_tail > 0:
            with tik_inst.if_scope(block_i < core_div_tail):
                pooling.run_pooling(core_div_loops * core_div_num + block_i)


def pooling_compute(x_param, pooling_attr):
    """
    start pooling compute
    Parameters
    ----------
    x_param : input shape, type param
    pooling_attr : pooling param
    Returns
    -------
    tik_inst, src/dst gm
    """
    tik_inst = tik.Tik(tik.Dprofile())
    shape_in = x_param['shape']

    window = pooling_attr['window']
    pad = pooling_attr['pad']
    stride = pooling_attr['stride']
    if pooling_attr['ceil_mode'] == POOLING_CEIL:
        out = {'h': (shape_in[2]+pad[0]*2-window[0]+stride[0]-1)//stride[0]+1,
               'w': (shape_in[3]+pad[2]*2-window[1]+stride[1]-1)//stride[1]+1}
    else:
        out = {'h': (shape_in[2]+pad[0]*2-window[0])//stride[0]+1,
               'w': (shape_in[3]+pad[2]*2-window[1])//stride[1]+1}

    if pad[0] > 0 or pad[2] > 0:
        if (out['h'] - 1) * stride[0] >= shape_in[2] + pad[0]:
            out['h'] = out['h'] - 1
        if (out['w'] - 1) * stride[1] >= shape_in[3] + pad[2]:
            out['w'] = out['w'] - 1
        if (out['h'] - 1) * stride[0] >= shape_in[2] + pad[0]:
            RuntimeError("Pooling out height out of range")
        if (out['w'] - 1) * stride[1] >= shape_in[3] + pad[2]:
            RuntimeError("Pooling out width out of range")
    if pooling_attr['global_pooling'] is True:
        out = {'h': 1, 'w': 1}

    vec_in_size = shape_in[0]*shape_in[1]*shape_in[2]*shape_in[3]*shape_in[4]
    vec_out_size = shape_in[0]*shape_in[1]*out['h']*out['w']*shape_in[4]

    src_gm = tik_inst.Tensor(x_param['dtype'], (vec_in_size, ),
                             scope=tik.scope_gm, name="src_gm")
    dst_gm = tik_inst.Tensor(x_param['dtype'], (vec_out_size, ),
                             scope=tik.scope_gm, name="dst_gm")

    base_param = {'tik_inst': tik_inst, 'out': out,
                  'src_gm': src_gm, 'dst_gm': dst_gm}

    compute_in_core(x_param, pooling_attr, base_param)

    return tik_inst, src_gm, dst_gm


# pylint: disable=too-many-arguments
def spp_pooling(x_dic, y_dic, global_pooling, mode, window, pad,
                stride, ceil_mode, kernel_name="spp_pooling"):
    """
    SPPPooling interface
    Parameters
    ----------
    x_dic : input dic, shape, type, format
    y_dic : output dic, shape, type, format
    global_pooling : global pooling flag, bool
    mode : pooling mode, 0-max, 1-avg
    window : window size, (window_h, window_w)
    pad : (pad_t, pad_b, pad_l, pad_r)
    stride : (stride_h, stride_w)
    ceil_mode : 0-ceil, 1-floor
    kernel_name : kernel name, default "spp_pooling"
    Returns
    -------
    tik_inst, src/dst gm
    """
    x_param = {'shape': x_dic['shape'], 'dtype': x_dic['dtype']}
    pooling_attr = {'global_pooling': global_pooling,
                    'mode': mode, 'window': window, 'pad': pad,
                    'stride': stride, 'ceil_mode': ceil_mode}

    check_param(x_dic, y_dic, pooling_attr, kernel_name)

    tik_instance, src_gm, dst_gm = pooling_compute(x_param, pooling_attr)

    tik_instance.BuildCCE(kernel_name=kernel_name,
                          inputs=src_gm, outputs=dst_gm)

    return tik_instance

