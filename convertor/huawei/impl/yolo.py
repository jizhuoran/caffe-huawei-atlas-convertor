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

yolo
"""
# pylint: disable=too-many-lines
# pylint: disable=too-many-arguments
from te import tik
from topi.cce import util
from te import platform as tbe_platform

RESV_UB = 512
FP16_MINI = -65504
FP32_MINI = -3.4 * (10**38)


def ceil_x(total_len, align_value):
    """
    ceil align
    Parameters
    ----------
    total_len: len before align
    align_value: align byte
    Returns
    -------
    len after ceil align
    """
    align_len = (total_len + align_value - 1) // align_value * align_value
    return align_len


def check_yolo_param(check_dic_dic, param_dic, kernel_name_check):
    """
    check yolo param error
    Parameters
    ----------
    check_dic_dic: include input_dic, output_dic
    param_dic: include boxes, coords, classes
    kernel_name_check: check name
    Returns
    -------
    NONE
    """
    #tik_name_check = tik.Dprofile().get_product_name()

    in_shape = check_dic_dic.get("in_dic").get("shape")
    out1_shape = check_dic_dic.get("out1_dic").get("shape")
    out2_shape = check_dic_dic.get("out2_dic").get("shape")
    out3_shape = check_dic_dic.get("out3_dic").get("shape")
    dtype = check_dic_dic.get("in_dic").get("dtype")

    if param_dic['boxes'] <= 0:
        raise RuntimeError("boxes value should be greater than 0")
    if param_dic['coords'] != 4:
        raise RuntimeError("coords value should be equal with 4")
    if param_dic['classes'] <= 0:
        raise RuntimeError("classes value should be greater than 0")

    util.check_shape_rule(in_shape, min_dim=4, max_dim=4, max_shape_num=10**8)
    util.check_shape_size(in_shape, 2**31)
    util.check_shape_rule(out1_shape, min_dim=3, max_dim=3, max_shape_num=10**8)
    util.check_shape_size(out1_shape, 2**31)
    util.check_shape_rule(out2_shape, min_dim=2, max_dim=2, max_shape_num=10**8)
    util.check_shape_size(out2_shape, 2**31)
    util.check_shape_rule(out3_shape, min_dim=3, max_dim=3, max_shape_num=10**8)
    util.check_shape_size(out3_shape, 2**31)

    project_name = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
    #if tik_name_check == "mini":
    if project_name in ("Ascend310",):
        util.check_dtype_rule(dtype.lower(), ["float16"])
    elif project_name in ("Ascend910",):
    #elif tik_name_check == "cloud":
        util.check_dtype_rule(dtype.lower(), ["float16"])
    elif project_name in ("Hi3796CV300ES",):
    #elif tik_name_check == "hisi-es":
        util.check_dtype_rule(dtype.lower(), ["float16"])
    elif project_name in ("Ascend610","Ascend620"):
    #elif tik_name_check == "aic":
        util.check_dtype_rule(dtype.lower(), ["float16", "float32"])
    else:
        util.check_dtype_rule(dtype.lower(), ["float16", "float32"])

    util.check_kernel_name(kernel_name_check)


class ShapeInfo:
    """
    Define Shape Info
    """

    def __init__(self, shape_info, param_info):
        self.batch = shape_info['batch']
        self.height = shape_info['height']
        self.width = shape_info['width']
        self.hw_size = shape_info['height'] * shape_info['width']
        self.dtype_size = 2 if (param_info['dtype'] == "float16") else 4

    def set_shape(self, shape_info):
        """
        set shape
        Parameters
        ----------
        shape_info: shape info
        Returns
        -------
        NONE
        """
        self.batch = shape_info['batch']
        self.height = shape_info['height']
        self.width = shape_info['width']
        self.hw_size = shape_info['height'] * shape_info['width']

    def set_type_size(self, param_info):
        """
        set type_size
        Parameters
        ----------
        param_info: set type size due to param_info['dtype']
        Returns
        -------
        NONE
        """
        self.dtype_size = 2 if (param_info['dtype'] == "float16") else 4


class ComputerCommon(ShapeInfo):
    """
    Define Compute Common Option
    """
    def __init__(self, shape_info, param_info, gm_info):
        self.param = param_info
        self.gm_addr = gm_info
        self.gm_dst = 'crd_dout'
        super(ComputerCommon, self).__init__(shape_info, param_info)

    def update_dout(self, gm_out):
        """
        update dout info, crd_dout/obj_dout/cls_dout
        Parameters
        ----------
        gm_out: type of gm_out
        Returns
        -------
        NONE
        """
        self.gm_dst = gm_out

    def newton_iteration(self, tik_inst, mask, rec, ori, cycle):
        """
        newton_raphson iteration
        Parameters
        ----------
        tik_inst: tik instance
        mask: vector calc mask
        rec: original 1/C and return 1/C result
        ori: original C
        cycle: newton_iteration times
        Returns
        -------
        NONE
        """
        tmp = tik_inst.Tensor(rec.dtype, rec.shape,
                              scope=tik.scope_ubuf, name="tmp")
        repeats = (rec.size * self.dtype_size + 255) // 256

        with tik_inst.for_range(0, cycle):
            tik_inst.vec_mul(mask, tmp, rec, ori, repeats, 8, 8, 8)
            tik_inst.vec_adds(mask, tmp, tmp, -2.0, repeats, 8, 8)
            tik_inst.vec_mul(mask, rec, tmp, rec, repeats, 8, 8, 8)
            tik_inst.vec_muls(mask, rec, rec, -1.0, repeats, 8, 8)

    def self_div(self, tik_inst, mask, dividend, divisor, dlen):
        """
        self define divide
        Parameters
        ----------
        tik_inst: tik instance
        mask: vector calc mask
        dividend: dividend tensor
        divisor: divisor tensor
        dlen: div tensor size
        Returns
        -------
        None
        """
        repeats = (dlen + 255) // 256
        #if tik.Dprofile().get_product_name() != "mini":
        if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") not in ("Ascend310",):
            tik_inst.vdiv(mask, dividend, dividend, divisor,
                          repeats, 1, 1, 1, 8, 8, 8)
        else:
            rec = tik_inst.Tensor(divisor.dtype, (dlen // self.dtype_size, ),
                                  scope=tik.scope_ubuf, name="rec")
            tik_inst.vec_rec(mask, rec, divisor, repeats, 8, 8)
            self.newton_iteration(tik_inst, mask, rec, divisor, 2)
            tik_inst.vec_mul(mask, dividend, dividend, rec, repeats, 8, 8, 8)


class SigMovCommon(ComputerCommon):
    """
    Define Sigmoid or Move Compute Common Option
    """
    def __init__(self, tik_inst, shape_info, param_info, gm_info):
        super(SigMovCommon, self).__init__(shape_info, param_info, gm_info)
        self.tik_inst = tik_inst
        self.tiling = {'mode': "TILING_BOX", 'box_num': 0,
                       'box_loop': 0, 'chn_loop': 0,
                       'data': 0, 'tail': 0}

    def sig_mov_tiling(self, available_ub):
        """
        sigmoid or move tiling
        Parameters
        ----------
        available_ub: tiling available buffer
        Returns
        -------
        double buf thread_num
        """
        chn_ceil_len = ceil_x(self.hw_size * self.dtype_size, 32)
        if available_ub - 256 >= chn_ceil_len:
            self.tiling['mode'] = "TILING_BOX"
            box_num = (available_ub - 256) // chn_ceil_len
            if box_num > self.param['boxes']:
                box_num = self.param['boxes']
            self.tiling['box_num'] = box_num
            self.tiling['data'] = box_num * chn_ceil_len
            self.tiling['box_loop'] = self.param['boxes'] // box_num
            self.tiling['tail'] = (self.param['boxes'] % box_num) * chn_ceil_len
            if self.tiling['tail'] > 0:
                self.tiling['box_loop'] = self.tiling['box_loop'] + 1
        else:
            self.tiling['mode'] = "TILING_CHN"
            self.tiling['box_num'] = 1
            self.tiling['box_loop'] = self.param['boxes']
            self.tiling['data'] = ceil_x(available_ub - 256, 256)
            self.tiling['chn_loop'] = chn_ceil_len // self.tiling['data']
            self.tiling['tail'] = chn_ceil_len % self.tiling['data']
            if self.tiling['tail'] > 0:
                self.tiling['chn_loop'] = self.tiling['chn_loop'] + 1

        return 2 if self.tiling['box_loop'] > 1 else 1

    def sig_mov_create_ub(self, ub_len):
        """
        create load ub tensor
        Parameters
        ----------
        ub_len: create ub length, need align to 256
        Returns
        -------
        load ub Tensor
        """
        size = ceil_x(ub_len, 256) // self.dtype_size
        sig_mov_ub = self.tik_inst.Tensor(self.param['dtype'], (size, ),
                                          scope=tik.scope_ubuf,
                                          name="sig_mov_ub")
        return sig_mov_ub

    def sig_mov_data_load(self, sgmd_mov_ub, idx, load_len):
        """
        load data from gm to ub
        Parameters
        ----------
        sgmd_mov_ub: sigmoid ubuf
        idx: cur loop value, include 'batch', 'elem', 'box', 'chn'
        load_len: load length, unit: byte
        Returns
        -------
        NONE
        """
        if self.tiling['mode'] == "TILING_CHN":
            cin_idx = idx['box'] * (self.param['coords'] + 1 +
                                    self.param['classes']) + idx['elem']
            hin_idx = self.tiling['data'] * idx['chn'] // \
                      self.dtype_size // self.width
            win_idx = self.tiling['data'] * idx['chn'] // \
                      self.dtype_size % self.width
            burst = load_len // 32
            self.tik_inst.data_move(sgmd_mov_ub[0],
                                    self.gm_addr['yolo_din'][idx['batch'],
                                                             cin_idx,
                                                             hin_idx,
                                                             win_idx],
                                    0, 1, burst, 0, 0)
        else:
            burst = ceil_x(self.hw_size * self.dtype_size, 32) // 32
            nburst = load_len // (burst * 32)
            with self.tik_inst.for_range(0, nburst) as burst_i:
                cin_idx = (idx['box'] * self.tiling['box_num'] + burst_i) * \
                          (self.param['coords'] + 1 + self.param['classes']) + \
                          idx['elem']
                self.tik_inst.data_move(
                    sgmd_mov_ub[burst_i * burst * 32 // self.dtype_size],
                    self.gm_addr['yolo_din'][idx['batch'], cin_idx, 0, 0],
                    0, 1, burst, 0, 0)

    def sig_mov_data_store(self, sgmd_mov_ub, idx, store_len):
        """
        store data from ub to gm
        Parameters
        ----------
        sgmd_mov_ub: sigmoid ubuf
        idx: cur loop value, include 'batch', 'elem', 'box', 'chn'
        store_len: store length, unit: byte
        Returns
        -------
        NONE
        """
        ofst_idx = self.tik_inst.Scalar(dtype="int32", name="ofst_idx")
        with self.tik_inst.if_scope(idx['elem'] < 4):
            with self.tik_inst.if_scope(idx['elem'] < 2):
                ofst_idx.set_as(idx['box'] * self.tiling['box_num'] +
                                idx['elem'] * self.param['boxes'])
            with self.tik_inst.if_scope(idx['elem'] == 2):
                ofst_idx.set_as(idx['box'] * self.tiling['box_num'] +
                                3 * self.param['boxes'])
            with self.tik_inst.if_scope(idx['elem'] == 3):
                ofst_idx.set_as(idx['box'] * self.tiling['box_num'] +
                                2 * self.param['boxes'])
        with self.tik_inst.if_scope(idx['elem'] == 4):
            ofst_idx.set_as(idx['box'] * self.tiling['box_num'] * self.hw_size)
        with self.tik_inst.if_scope(idx['elem'] > 4):
            ofst_idx.set_as(idx['box'] * self.tiling['box_num'] * self.hw_size)

        if self.tiling['mode'] == "TILING_CHN":
            out_idx = idx['chn'] * self.tiling['data'] // self.dtype_size
            burst = store_len // 32
            if self.gm_dst == 'crd_dout':
                self.tik_inst.data_move(
                    self.gm_addr['crd_dout'][idx['batch'], ofst_idx, out_idx],
                    sgmd_mov_ub[0], 0, 1, burst, 0, 0)
            elif self.gm_dst == 'obj_dout':
                self.tik_inst.data_move(
                    self.gm_addr['obj_dout'][idx['batch'], ofst_idx + out_idx],
                    sgmd_mov_ub[0], 0, 1, burst, 0, 0)
            else:
                self.tik_inst.data_move(
                    self.gm_addr['cls_dout'][idx['batch'], idx['elem'] - 5,
                                             ofst_idx + out_idx],
                    sgmd_mov_ub[0], 0, 1, burst, 0, 0)
        else:
            burst = ceil_x(self.hw_size * self.dtype_size, 32) // 32
            nburst = store_len // (burst * 32)
            with self.tik_inst.for_range(0, nburst) as burst_i:
                if self.gm_dst == 'crd_dout':
                    out_idx = burst_i + ofst_idx
                    self.tik_inst.data_move(
                        self.gm_addr['crd_dout'][idx['batch'], out_idx, 0],
                        sgmd_mov_ub[burst_i * burst * 32 // self.dtype_size],
                        0, 1, burst, 0, 0)
                elif self.gm_dst == 'obj_dout':
                    out_idx = ofst_idx + burst_i * self.hw_size
                    self.tik_inst.data_move(
                        self.gm_addr['obj_dout'][idx['batch'], out_idx],
                        sgmd_mov_ub[burst_i * burst * 32 // self.dtype_size],
                        0, 1, burst, 0, 0)
                else:
                    out_idx = ofst_idx + burst_i * self.hw_size
                    self.tik_inst.data_move(
                        self.gm_addr['cls_dout'][idx['batch'],
                                                 idx['elem'] - 5, out_idx],
                        sgmd_mov_ub[burst_i * burst * 32 // self.dtype_size],
                        0, 1, burst, 0, 0)


class MoveComputer(SigMovCommon):
    """
    Define move option calculation
    """

    def __init__(self, tik_inst, shape_info, param_info, gm_info):
        super(MoveComputer, self).__init__(tik_inst, shape_info,
                                           param_info, gm_info)
        self.thread_num = 1

    def run_compute(self, idx_dic, move_len):
        """
        execute move step
        Parameters
        ----------
        idx_dic: cur loop value, include 'batch', 'elem', 'box', 'chn'
        move_len: move data length, unit: byte
        Returns
        -------
        NONE
        """
        with self.tik_inst.new_stmt_scope():
            mov_ub = self.sig_mov_create_ub(move_len)
            self.sig_mov_data_load(mov_ub, idx_dic, move_len)
            self.sig_mov_data_store(mov_ub, idx_dic, move_len)


class SigmoidComputer(SigMovCommon):
    """
    Define sigmoid option calculation
    """

    def __init__(self, tik_inst, shape_info, param_info, gm_info):
        super(SigmoidComputer, self).__init__(tik_inst, shape_info,
                                              param_info, gm_info)
        self.thread_num = 1

    def sigmoid_computer(self, cmpt_ub, data_len):
        """
        execute sigmoid option
        Parameters
        ----------
        cmpt_ub: compute ub tensor
        data_len: compute data length, unit: byte
        Returns
        -------
        NONE
        """
        dlen = ceil_x(data_len, 256)
        mask = 256 // self.dtype_size
        rpts = dlen // 256
        tmp_ub_1 = self.tik_inst.Tensor(self.param['dtype'],
                                        (dlen // self.dtype_size, ),
                                        name="tmp_ub_1",
                                        scope=tik.scope_ubuf)
        tmp_ub_2 = self.tik_inst.Tensor(self.param['dtype'],
                                        (dlen // self.dtype_size, ),
                                        name="tmp_ub_2",
                                        scope=tik.scope_ubuf)

        self.tik_inst.vec_dup(mask, tmp_ub_1, 0.0, rpts, 8)
        self.tik_inst.vec_max(mask, tmp_ub_1, cmpt_ub, tmp_ub_1, rpts, 8, 8, 8)
        self.tik_inst.vec_muls(mask, tmp_ub_1, tmp_ub_1, -1.0, rpts, 8, 8)
        self.tik_inst.vec_exp(mask, tmp_ub_1, tmp_ub_1, rpts, 8, 8)
        self.tik_inst.vec_adds(mask, tmp_ub_1, tmp_ub_1, 1, rpts, 8, 8)
        self.tik_inst.vec_dup(mask, tmp_ub_2, 1.0, rpts, 8)
        with self.tik_inst.new_stmt_scope():
            self.self_div(self.tik_inst, mask, tmp_ub_2, tmp_ub_1, dlen)

        tmp_ub_3 = self.tik_inst.Tensor(self.param['dtype'],
                                        (dlen // self.dtype_size, ),
                                        name="tmp_ub_3",
                                        scope=tik.scope_ubuf)
        self.tik_inst.vec_dup(mask, tmp_ub_3, 0.0, rpts, 8)
        self.tik_inst.vec_min(mask, tmp_ub_3, cmpt_ub, tmp_ub_3, rpts, 8, 8, 8)
        self.tik_inst.vec_exp(mask, tmp_ub_1, tmp_ub_3, rpts, 8, 8)
        self.tik_inst.vec_adds(mask, tmp_ub_3, tmp_ub_1, 1, rpts, 8, 8)
        with self.tik_inst.new_stmt_scope():
            self.self_div(self.tik_inst, mask, tmp_ub_1, tmp_ub_3, dlen)

        self.tik_inst.vec_add(mask, cmpt_ub, tmp_ub_1, tmp_ub_2, rpts, 8, 8, 8)
        self.tik_inst.vec_adds(mask, cmpt_ub, cmpt_ub, -0.5, rpts, 8, 8)

    def run_compute(self, idx_dic, sigmoid_len):
        """
        execute sigmoid step
        Parameters
        ----------
        idx_dic: cur loop value, include 'batch', 'elem', 'box', 'chn'
        sigmoid_len: sigmoid data length, unit: byte
        Returns
        -------
        NONE
        """
        sgmd_ub = self.sig_mov_create_ub(sigmoid_len)
        self.sig_mov_data_load(sgmd_ub, idx_dic, sigmoid_len)

        loop = sigmoid_len // 65280
        tail = sigmoid_len % 65280
        if tail > 0:
            loop = loop + 1

        for i in range(0, loop):
            with self.tik_inst.new_stmt_scope():
                if i == loop-1 and tail > 0:
                    self.sigmoid_computer(sgmd_ub[i * 65280 //
                                                  self.dtype_size], tail)
                else:
                    self.sigmoid_computer(sgmd_ub[i * 65280 //
                                                  self.dtype_size], 65280)

        self.sig_mov_data_store(sgmd_ub, idx_dic, sigmoid_len)


class SoftmaxComputer(ComputerCommon):
    """
    Define softmax option calculation
    """

    def __init__(self, tik_inst, shape_info, param_info, gm_info):

        # softmax need 2 partition
        super(SoftmaxComputer, self).__init__(shape_info, param_info, gm_info)
        self.tik_inst = tik_inst
        self.thread_num = 1
        self.sfmx_num = 0
        self.is_bc_sfmx = False
        self.tiling = {'data': 1, 'chn_loop': 1, 'tail': 1}

    def softmax_tiling(self, available_ub, is_bc_sfmx):
        """
        softmax tiling
        Parameters
        ----------
        available_ub: tiling available buffer
        is_bc_sfmx: is obj and class do softmax at the same time or not
        Returns
        -------
        double buf thread_num
        """
        self.is_bc_sfmx = is_bc_sfmx
        self.sfmx_num = self.param['classes'] + is_bc_sfmx
        # consider calc buf
        dlen = available_ub // (self.sfmx_num + 4)
        if dlen < 32:
            raise RuntimeError("Classes num overflow")
        chn_ceil_len = ceil_x(self.hw_size * self.dtype_size, 32)
        if dlen > chn_ceil_len:
            dlen = chn_ceil_len+31
        self.tiling['data'] = ceil_x(dlen-32, 32)
        self.tiling['chn_loop'] = chn_ceil_len // self.tiling['data']
        self.tiling['tail'] = chn_ceil_len % self.tiling['data']
        if self.tiling['tail'] > 0:
            self.tiling['chn_loop'] = self.tiling['chn_loop'] + 1

        return 2 if self.tiling['chn_loop'] > 1 else 1

    def softmax_create_ub(self, data_len):
        """
        create load ub tensor
        Parameters
        ----------
        data_len: create ub length, need align to 32
        Returns
        -------
        load ub Tensor and compute middle ub
        """
        ub_len = ceil_x(data_len, 32)
        sfmx_size = ub_len // self.dtype_size * self.sfmx_num
        mid_size = ub_len // self.dtype_size
        sfmx_ub = self.tik_inst.Tensor(self.param['dtype'], (sfmx_size, ),
                                       scope=tik.scope_ubuf, name="sfmx_ub")
        mid_ub = self.tik_inst.Tensor(self.param['dtype'], (mid_size, ),
                                      scope=tik.scope_ubuf, name="mid_ub")
        return sfmx_ub, mid_ub

    def softmax_data_load(self, sfmx_ub, idx, load_len):
        """
        load data from gm to ub
        Parameters
        ----------
        sfmx_ub: softmax load ub
        idx: cur loop value, include 'batch', 'box', 'chn'
        load_len: load length, unit:byte
        Returns
        -------
        NONE
        """
        burst = load_len // 32
        with self.tik_inst.for_range(0, self.sfmx_num) as sfmx_i:
            cin_idx = idx['box'] * \
                      (self.param['coords'] + 1 + self.param['classes']) + \
                      4 + (not self.is_bc_sfmx) + sfmx_i
            hin_idx = self.tiling['data'] * idx['chn'] // \
                      self.dtype_size // self.width
            win_idx = self.tiling['data'] * idx['chn'] // \
                      self.dtype_size % self.width

            self.tik_inst.data_move(sfmx_ub[sfmx_i*burst*32//self.dtype_size],
                                    self.gm_addr['yolo_din'][idx['batch'],
                                                             cin_idx,
                                                             hin_idx,
                                                             win_idx],
                                    0, 1, burst, 0, 0)

    def softmax_data_store(self, sfmx_ub, idx, store_len):
        """
        store data from ub to gm
        Parameters
        ----------
        sfmx_ub: softmax store ub
        idx: cur loop value, include 'batch', 'box', 'chn'
        store_len: store length, unit:byte
        Returns
        -------
        NONE
        """
        burst = store_len // 32
        ofst_idx = idx['box'] * self.hw_size
        if self.is_bc_sfmx is True:
            out_idx = ofst_idx + idx['chn'] * \
                      self.tiling['data'] // self.dtype_size
            self.tik_inst.data_move(
                self.gm_addr['obj_dout'][idx['batch'], out_idx],
                sfmx_ub[0], 0, 1, burst, 0, 0)

        nburst = self.param['classes']
        gm_out_idx = ofst_idx+idx['chn']*self.tiling['data']//self.dtype_size
        ub_out_idx = (self.is_bc_sfmx * store_len) // self.dtype_size
        src_stride = 0
        dst_stride = (ceil_x(self.param['boxes'] * self.hw_size * 2 + 32, 32) //
                      2 * self.dtype_size - ceil_x(store_len, 32)) // 32

        self.tik_inst.data_move(
            self.gm_addr['cls_dout'][idx['batch'], 0, gm_out_idx],
            sfmx_ub[ub_out_idx], 0, nburst, burst, src_stride, dst_stride)

    def softmax_vop(self, sfmx_ub, mid_ub, stride_len, mask, op_type, ofst):
        """
        softmax vector option 1, usage for stride_len <= 255 * 32
        Parameters
        ----------
        sfmx_ub: softmax store ub
        mid_ub: compute middle ub
        stride_len: compute stride offset, unit: byte
        mask: vector compute mask
        op_type: vector option type
        ofst: sfmx_ub and mid_ub offset
        Returns
        -------
        NONE
        """
        repeats = 255
        repeats_loop = self.sfmx_num // 255
        repeats_tail = self.sfmx_num % 255
        if repeats_tail > 0:
            repeats_loop = repeats_loop + 1

        for repeats_i in range(0, repeats_loop):
            if repeats_i == repeats_loop-1 and repeats_tail > 0:
                repeats = repeats_tail
            cur_pos = ofst + repeats_i * 255 * stride_len // self.dtype_size
            if op_type == 'max':
                self.tik_inst.vec_max(mask, mid_ub[ofst], sfmx_ub[cur_pos],
                                      mid_ub[ofst], repeats,
                                      0, stride_len // 32, 0)
            elif op_type == 'sub':
                self.tik_inst.vec_sub(mask, sfmx_ub[cur_pos], sfmx_ub[cur_pos],
                                      mid_ub[ofst], repeats,
                                      stride_len // 32, stride_len // 32, 0)
            elif op_type == 'exp':
                self.tik_inst.vec_exp(mask, sfmx_ub[cur_pos], sfmx_ub[cur_pos],
                                      repeats, stride_len//32, stride_len//32)
            elif op_type == 'add':
                self.tik_inst.vec_add(mask, mid_ub[ofst], sfmx_ub[cur_pos],
                                      mid_ub[ofst], repeats,
                                      0, stride_len // 32, 0)

    def softmax_compute(self, sfmx_ub, mid_ub, stride_len, proc_len, ofst):
        """
        execute softmax option, usage for stride_len <= 255 * 32
        Parameters
        ----------
        sfmx_ub: softmax store ub
        mid_ub: compute middle ub
        stride_len: compute stride offset, unit: byte
        proc_len: process data len in one time
        ofst: sfmx_ub and mid_ub offset
        Returns
        -------
        NONE
        """
        fp_mini = FP32_MINI if self.param['dtype'] == "float32" else FP16_MINI

        mask = proc_len // self.dtype_size
        self.tik_inst.vec_dup(mask, mid_ub[ofst], fp_mini, 1, 8)
        self.softmax_vop(sfmx_ub, mid_ub, stride_len, mask, 'max', ofst)
        self.softmax_vop(sfmx_ub, mid_ub, stride_len, mask, 'sub', ofst)
        self.softmax_vop(sfmx_ub, mid_ub, stride_len, mask, 'exp', ofst)
        self.tik_inst.vec_dup(mask, mid_ub[ofst], 0, 1, 8)
        self.softmax_vop(sfmx_ub, mid_ub, stride_len, mask, 'add', ofst)

        with self.tik_inst.for_range(0, self.sfmx_num) as sfmx_i:
            self.self_div(self.tik_inst, mask,
                          sfmx_ub[ofst+sfmx_i*stride_len//self.dtype_size],
                          mid_ub[ofst], proc_len)

    def softmax_vop_2(self, sfmx_ub, mid_ub, mask, ofst,
                      stride_len, proc_len, op_type):
        """
        softmax vector option 2, usage for stride_len > 255 * 32
        Parameters
        ----------
        sfmx_ub: softmax store ub
        mid_ub: compute middle ub
        stride_len: compute stride offset, unit: byte
        proc_len: process data len in one time
        mask: vector compute mask
        op_type: vector option type
        ofst: sfmx_ub and mid_ub offset
        Returns
        -------
        NONE
        """
        repeats = (proc_len + 255) // 256
        for i in range(0, self.sfmx_num):
            cur_pos = i * stride_len // self.dtype_size + ofst
            if op_type == 'max':
                self.tik_inst.vec_max(mask, mid_ub[ofst], sfmx_ub[cur_pos],
                                      mid_ub[ofst], repeats, 8, 8, 8)
            elif op_type == 'sub':
                self.tik_inst.vec_sub(mask, sfmx_ub[cur_pos], sfmx_ub[cur_pos],
                                      mid_ub[ofst], repeats, 8, 8, 8)
            elif op_type == 'exp':
                self.tik_inst.vec_exp(mask, sfmx_ub[cur_pos], sfmx_ub[cur_pos],
                                      repeats, 8, 8)
            elif op_type == 'add':
                self.tik_inst.vec_add(mask, mid_ub[ofst], sfmx_ub[cur_pos],
                                      mid_ub[ofst], repeats, 8, 8, 8)

    def softmax_compute_2(self, sfmx_ub, mid_ub, mask,
                          stride_len, proc_len, ofst):
        """
        execute softmax option, usage for stride_len <= 255 * 32
        Parameters
        ----------
        sfmx_ub: softmax store ub
        mid_ub: compute middle ub
        mask: vector option mask
        stride_len: compute stride offset, unit: byte
        proc_len: process data len in one time
        ofst: sfmx_ub and mid_ub offset
        Returns
        -------
        NONE
        """
        fp_mini = FP32_MINI if self.param['dtype'] == "float32" else FP16_MINI
        repeats = (proc_len + 255) // 256
        self.tik_inst.vec_dup(mask, mid_ub[ofst], fp_mini, repeats, 8)
        self.softmax_vop_2(sfmx_ub, mid_ub, mask, ofst,
                           stride_len, proc_len, 'max')
        self.softmax_vop_2(sfmx_ub, mid_ub, mask, ofst,
                           stride_len, proc_len, 'sub')
        self.softmax_vop_2(sfmx_ub, mid_ub, mask, ofst,
                           stride_len, proc_len, 'exp')
        self.tik_inst.vec_dup(mask, mid_ub[ofst], 0, repeats, 8)
        self.softmax_vop_2(sfmx_ub, mid_ub, mask, ofst,
                           stride_len, proc_len, 'add')

        with self.tik_inst.for_range(0, self.sfmx_num) as sfmx_i:
            self.self_div(self.tik_inst, mask,
                          sfmx_ub[ofst+sfmx_i*stride_len//self.dtype_size],
                          mid_ub[ofst], proc_len)

    def run_compute(self, idx_dic, sfmx_len):
        """
        execute softmax step
        Parameters
        ----------
        idx_dic: cur loop value, include 'batch', 'elem', 'box', 'chn'
        sfmx_len: softmax compute data length, unit: byte
        Returns
        -------
        NONE
        """
        sfmx_ub, mid_ub = self.softmax_create_ub(sfmx_len)
        self.softmax_data_load(sfmx_ub, idx_dic, sfmx_len)
        # dst_rep_stride and src_rep_stride in range[0, 255], 8160=255*32
        if sfmx_len <= 8160:
            loop = sfmx_len // 256
            tail = sfmx_len % 256
            if tail > 0:
                loop = loop + 1
            for i in range(0, loop):
                if i == loop-1 and tail > 0:
                    self.softmax_compute(sfmx_ub, mid_ub, sfmx_len, tail,
                                         i * 256 // self.dtype_size)
                else:
                    self.softmax_compute(sfmx_ub, mid_ub, sfmx_len, 256,
                                         i * 256 // self.dtype_size)
        else:
            loop = sfmx_len // 65280
            tail = sfmx_len % 65280
            if tail > 0:
                loop = loop + 1
            for i in range(0, loop):
                if i == loop-1 and tail > 0:
                    tail_len = tail // 256 * 256
                    tail_tail = tail % 256
                    if tail_len > 0:
                        self.softmax_compute_2(sfmx_ub, mid_ub, 128,
                                               sfmx_len, tail_len,
                                               i * 65280 // self.dtype_size)
                    if tail_tail > 0:
                        self.softmax_compute_2(sfmx_ub, mid_ub,
                                               tail_tail // self.dtype_size,
                                               sfmx_len, tail_tail,
                                               (i * 65280 + tail_len) //
                                               self.dtype_size)
                else:
                    self.softmax_compute_2(sfmx_ub, mid_ub, 128, sfmx_len,
                                           65280, i * 65280 // self.dtype_size)

        self.softmax_data_store(sfmx_ub, idx_dic, sfmx_len)


class InitTikAndTensor:
    """
    Define IN OUT Tensor in gm
    """
    def __init__(self, shape_info, param_info):
        classes = param_info['classes']
        coords = param_info['coords']
        boxes = param_info['boxes']
        dtype = param_info['dtype']
        batch = shape_info['batch']
        height = shape_info['height']
        width = shape_info['width']
        dtype_size = 2 if (param_info['dtype'] == "float16") else 4
        self.product_name = ""
        self.total_ub_size = 0

        self.tik_inst = tik.Tik(tik.Dprofile())

        # in order to solve 32B not enough when do the last data_mov
        batch_padding = 32//((boxes*(coords+1+classes)) *
                             height*width*dtype_size) + 1
        self.yolo_din = \
            self.tik_inst.Tensor(dtype, (batch+batch_padding,
                                         boxes*(coords+1+classes),
                                         height, width),
                                 scope=tik.scope_gm, name="yolo_din")
        # shape defined by fp16 for infershape dtype can not be determined
        self.crd_dout = \
            self.tik_inst.Tensor(dtype, (batch, boxes*coords,
                                         ceil_x(height*width*2+32, 32)//2),
                                 scope=tik.scope_gm, name="crd_dout")

        self.obj_dout = \
            self.tik_inst.Tensor(dtype, (batch, ceil_x(boxes*height*width *
                                                       2+32, 32)//2),
                                 scope=tik.scope_gm, name="obj_dout")

        self.cls_dout = \
            self.tik_inst.Tensor(dtype, (batch, classes,
                                         ceil_x(boxes*height*width *
                                                2+32, 32)//2),
                                 scope=tik.scope_gm, name="cls_dout")

    def set_product_name(self):
        """
        set product name
        Parameters
        ----------
        Returns
        -------
        NONE
        """
        self.product_name = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
        #tik.Dprofile().get_product_name()

    def set_ub_buf(self):
        """
        set ub buffer size
        Parameters
        ----------
        Returns
        -------
        tNONE
        """
        #self.total_ub_size = tik.Dprofile().get_unified_buffer_size()
        self.total_ub_size = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)


class YoloOp(InitTikAndTensor):
    """
    Define yolo op node class
    do different computer in accord to yolo_version and softmax/background
    """

    def __init__(self, shape_info, param_info):
        super(YoloOp, self).__init__(shape_info, param_info)
        self.shape = shape_info
        self.param = param_info
        self.gm_addr = {'yolo_din': self.yolo_din,
                        'crd_dout': self.crd_dout,
                        'obj_dout': self.obj_dout,
                        'cls_dout': self.cls_dout}

    def get_yolo_mode(self, yolo_ver):
        """
        get yolo mode
        Parameters
        ----------
        yolo_ver: yolo version, "V2" or "V3"
        Returns
        -------
        yolo compute mode
        """
        if yolo_ver.lower() == "v2":
            if self.param['background'] is False \
                    and self.param['softmax'] is False:
                yolo_mode = 'YOLO_MODE_1'
            elif self.param['background'] is False \
                    and self.param['softmax'] is True:
                yolo_mode = 'YOLO_MODE_2'
            elif self.param['background'] is True \
                    and self.param['softmax'] is False:
                yolo_mode = 'YOLO_MODE_3'
            else:
                yolo_mode = 'YOLO_MODE_4'
        elif yolo_ver.lower() == "v3":
            yolo_mode = 'YOLO_MODE_1'
        else:
            raise RuntimeError("Now only support other yolo_version V2&V3")

        return yolo_mode

    def run_op_compute(self, common_op, idx_dic, loop_mode):
        """
        run option, select option with op
        Parameters
        ----------
        common_op: move, sigmoid or softmax op
        idx_dic: cur loop value, include 'batch', 'elem', 'box', 'chn'
        loop_mode: 'chn_loop' or 'box_loop'
        Returns
        -------
        NONE
        """
        loop = idx_dic['chn'] if loop_mode == 'chn_loop' else idx_dic['box']
        with self.tik_inst.if_scope(loop == common_op.tiling[loop_mode]-1):
            if common_op.tiling['tail'] > 0:
                common_op.run_compute(idx_dic, common_op.tiling['tail'])
            else:
                common_op.run_compute(idx_dic, common_op.tiling['data'])
        with self.tik_inst.else_scope():
            common_op.run_compute(idx_dic, common_op.tiling['data'])

    def run_sig_mov(self, sig_mov_op, batch, elem_loop, dbbuf_inside):
        """
        run sigmoid or move option
        Parameters
        ----------
        sig_mov_op: move or sigmoid op
        batch: current batch
        elem_loop: current x,y,w,h,obj,c1,c2...,cn loop value
        dbbuf_inside: is do double buffer in this function or not
        Returns
        -------
        NONE
        """
        thread_num = sig_mov_op.thread_num if dbbuf_inside is True else 1
        with self.tik_inst.for_range(0, sig_mov_op.tiling['box_loop'],
                                     thread_num=thread_num) as box_loop:
            if sig_mov_op.tiling['mode'] == "TILING_CHN":
                with self.tik_inst.for_range(
                        0, sig_mov_op.tiling['chn_loop']) as chn_loop:
                    idx_dic = {'batch': batch, 'elem': elem_loop,
                               'box': box_loop, 'chn': chn_loop}
                    self.run_op_compute(sig_mov_op, idx_dic, 'chn_loop')
            else:
                idx_dic = {'batch': batch, 'elem': elem_loop,
                           'box': box_loop, 'chn': 0}
                self.run_op_compute(sig_mov_op, idx_dic, 'box_loop')

    def run_softmax(self, sfmx_op, batch, box_loop, dbbuf_inside):
        """
        run sigmoid or move option
        Parameters
        ----------
        sfmx_op: softmax op
        batch: current batch
        box_loop: box loop value
        dbbuf_inside: is do double buffer in this function or not
        Returns
        -------
        NONE
        """
        thread_num = sfmx_op.thread_num if dbbuf_inside is True else 1
        with self.tik_inst.for_range(0, sfmx_op.tiling['chn_loop'],
                                     thread_num=thread_num) as chn_loop:
            idx_dic = {'batch': batch, 'box': box_loop, 'chn': chn_loop}
            self.run_op_compute(sfmx_op, idx_dic, 'chn_loop')

    def run_yolo_computer(self, batch, yolo_ver):
        """
        active yolo computer
        Parameters
        ----------
        batch: batch index
        yolo_ver: yolo version, "V2" or "V3"
        Returns
        -------
        NONE
        """
        self.set_product_name()
        self.set_ub_buf()
        op_move = MoveComputer(self.tik_inst, self.shape,
                               self.param, self.gm_addr)
        op_sgmd = SigmoidComputer(self.tik_inst, self.shape,
                                  self.param, self.gm_addr)
        op_sfmx = SoftmaxComputer(self.tik_inst, self.shape,
                                  self.param, self.gm_addr)

        sigmoid_ub_part = 4 if self.product_name not in ("Ascend310",) else 6

        available_ub = (self.total_ub_size // 2 - RESV_UB) // sigmoid_ub_part
        op_sgmd.thread_num = op_sgmd.sig_mov_tiling(available_ub)
        op_sgmd.update_dout('crd_dout')
        with self.tik_inst.for_range(0, 2, thread_num=2) as xy_loop:
            self.run_sig_mov(op_sgmd, batch, xy_loop, dbbuf_inside=False)

        available_ub = self.total_ub_size // 2
        op_move.thread_num = op_move.sig_mov_tiling(available_ub)
        with self.tik_inst.for_range(0, 2, thread_num=2) as wh_loop:
            self.run_sig_mov(op_move, batch, wh_loop + 2, dbbuf_inside=False)

        yolo_mode = self.get_yolo_mode(yolo_ver)

        if yolo_mode == 'YOLO_MODE_1':
            op_sgmd.update_dout('obj_dout')
            self.run_sig_mov(op_sgmd, batch, 4, dbbuf_inside=True)

            op_sgmd.update_dout('cls_dout')
            db_inside = self.param['classes'] <= 1
            db_outside = 2 if self.param['classes'] > 1 else 1
            with self.tik_inst.for_range(0, self.param['classes'],
                                         thread_num=db_outside) as cls_loop:
                self.run_sig_mov(op_sgmd, batch, cls_loop + 5,
                                 dbbuf_inside=db_inside)

        elif yolo_mode == 'YOLO_MODE_2':
            op_sgmd.update_dout('obj_dout')
            self.run_sig_mov(op_sgmd, batch, 4, dbbuf_inside=True)

            available_ub = self.total_ub_size // 2 - RESV_UB
            op_sfmx.thread_num = op_sfmx.softmax_tiling(available_ub,
                                                        is_bc_sfmx=False)
            db_inside = self.param['boxes'] <= 1
            db_outside = 2 if self.param['boxes'] > 1 else 1
            with self.tik_inst.for_range(0, self.param['boxes'],
                                         thread_num=db_outside) as box_loop:
                self.run_softmax(op_sfmx, batch, box_loop,
                                 dbbuf_inside=db_inside)

        elif yolo_mode == 'YOLO_MODE_3':
            op_move.update_dout('obj_dout')
            self.run_sig_mov(op_move, batch, 4, dbbuf_inside=True)

            op_sgmd.update_dout('cls_dout')
            db_inside = self.param['classes'] <= 1
            db_outside = 2 if self.param['classes'] > 1 else 1
            with self.tik_inst.for_range(0, self.param['classes'],
                                         thread_num=db_outside) as cls_loop:
                self.run_sig_mov(op_sgmd, batch, cls_loop + 5,
                                 dbbuf_inside=db_inside)

        else:
            available_ub = self.total_ub_size // 2 - RESV_UB
            op_sfmx.thread_num = op_sfmx.softmax_tiling(available_ub,
                                                        is_bc_sfmx=True)
            db_inside = self.param['boxes'] <= 1
            db_outside = 2 if self.param['boxes'] > 1 else 1
            with self.tik_inst.for_range(0, self.param['boxes'],
                                         thread_num=db_outside) as box_loop:
                self.run_softmax(op_sfmx, batch, box_loop,
                                 dbbuf_inside=db_inside)


def yolo_computer(yolo_op, shape_info, yolo_version):
    """
    divide core and run yolo computer
    Parameters
    ----------
    yolo_op: yolo op node
    shape_info: shape info
    yolo_version: yolo version, "V2" or "V3"
    Returns
    -------
    NONE
    """
    #block_num = tik.Dprofile().get_aicore_num()
    block_num = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
    if shape_info['batch'] > block_num:
        core_div_num = block_num
        core_div_loops = shape_info['batch'] // block_num
        core_div_tail = shape_info['batch'] % block_num
    else:
        core_div_num = shape_info['batch']
        core_div_loops = 0
        core_div_tail = shape_info['batch']

    with yolo_op.tik_inst.for_range(0, core_div_num,
                                    block_num=core_div_num) as block_i:
        with yolo_op.tik_inst.for_range(0, core_div_loops) as loop_i:
            yolo_op.run_yolo_computer(loop_i * core_div_num + block_i,
                                      yolo_version)
        if core_div_tail > 0:
            with yolo_op.tik_inst.if_scope(block_i < core_div_tail):
                yolo_op.run_yolo_computer(
                    core_div_loops * core_div_num + block_i, yolo_version)


def yolo(input_dic, coord_out_dic, obj_out_dic, class_out_dic,
         boxes, coords, classes, yolo_version="V3",
         softmax=False, background=False, softmaxtree=False,
         kernel_name="yolo"):
    """
    yolo interface, v2: region, v3:yolo
    divide core and run yolo computer
    Parameters
    ----------
    input_dic: input feature dictionary, include shape, dtype
    coord_out_dic: output feature dictionary, include shape, dtype
    obj_out_dic: output feature dictionary, include shape, dtype
    class_out_dic: output feature dictionary, include shape, dtype
    boxes: anchor box num, "V2" default 5, "V3" default 3
    coords: x, y, w, h, always 4
    classes: class num, default 80
    yolo_version: "V2" or "V3"
    softmax: decide yolo calc mode with background
    background: decide yolo calc mode with softmax
    softmaxtree: softmax tree, no use in this op
    kernel_name: kernel name
    Returns
    -------
    tik instance
    """

    check_yolo_param({'in_dic': input_dic, 'out1_dic': coord_out_dic,
                      'out2_dic': obj_out_dic, 'out3_dic': class_out_dic},
                     {'boxes': boxes, 'coords': coords, 'classes': classes},
                     kernel_name)

    shape_info = {'batch': input_dic['shape'][0],
                  'height': input_dic['shape'][2],
                  'width': input_dic['shape'][3]}
    param_info = {'classes': classes, 'coords': coords, 'boxes': boxes,
                  'softmax': softmax, 'background': background,
                  'softmaxtree': softmaxtree, 'dtype': input_dic['dtype']}

    yolo_op = YoloOp(shape_info, param_info)

    yolo_computer(yolo_op, shape_info, yolo_version)

    yolo_op.tik_inst.BuildCCE(kernel_name=kernel_name,
                              inputs=yolo_op.yolo_din,
                              outputs=(yolo_op.crd_dout,
                                       yolo_op.obj_dout,
                                       yolo_op.cls_dout))

    return yolo_op.tik_inst
