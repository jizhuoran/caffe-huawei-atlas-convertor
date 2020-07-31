# pylint: disable=too-many-lines
# pylint: disable=import-error,too-many-instance-attributes,no-self-use
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

yolo_v3_correct_region_box
"""

import math
from te import tik
from impl.constant_util import BLOCK_SIZE, VECTOR_BYTE_SIZE, STRIDE_ONE, AIC,\
    CLOUD

# value zero
VALUE_ZERO = 0

# value two
VALUE_TWO = 2

# value three
VALUE_THREE = 3

# neg two
NEG_TWO = -2

# neg one
NEG_ONE = -1

# value half
VALUE_HALF = 0.5

# repeat one
REPEAT_ONE = 1

# value one
VALUE_ONE = 1


class GetCorrectBoxBase():
    """
     class for correct box compute
    """
    def __init__(self, input_dict):
        """
        init function
        Parameters
        ----------
        input_dict: input dict
        """
        self.instance = tik.Tik(tik.Dprofile())
        self.batch = input_dict['batch']
        self.dtype = input_dict['dtype']
        self.height = input_dict['height']
        self.width = input_dict['width']
        self.biases = input_dict['biases']
        self.boxes = input_dict['boxes']
        self.classes = input_dict['classes']
        self.relative = input_dict['relative']
        self.obj_threshold = input_dict['obj_threshold']
        self.post_nms_topn = input_dict['post_nms_topn']
        self.score_threshold = input_dict['score_threshold']
        self.iou_threshold = input_dict['iou_threshold']
        self.max_box_number_per_batch = input_dict['max_box_number_per_batch']
        self.pre_nms_topn = input_dict['pre_nms_topn']
        self.kernel_name = input_dict['kernel_name']
        self.obj_num = self.boxes * self.height * self.width

        self.dsize = 2 if self.dtype == "float16" else 4
        self.adj_hw = self.get_adj_hw(self.height*self.width)
        self.mask = 256 // self.dsize
        self.len_32b = BLOCK_SIZE // self.dsize
        self.hw_len = self.width * self.height
        self.hwtail_len = self.hw_len % self.len_32b
        self.one_max_size = (((tik.Dprofile().get_unified_buffer_size() -
                               16*1024) // 8) // 256) * 256

        adj_hw = self.get_adj_hw(self.height*self.width)
        self.coord_data = self.instance.Tensor(self.dtype,
                                               (self.batch, self.boxes * 4,
                                                adj_hw),
                                               scope=tik.scope_gm,
                                               name="coord_data")
        self.windex = self.instance.Tensor(self.dtype, (adj_hw,),
                                           scope=tik.scope_gm,
                                           name="windex")
        self.hindex = self.instance.Tensor(self.dtype, (adj_hw,),
                                           scope=tik.scope_gm,
                                           name="hindex")

        adj_hw = self.get_adj_hw(self.boxes * self.height*self.width)
        self.obj_prob = self.instance.Tensor(self.dtype, (self.batch, adj_hw),
                                             scope=tik.scope_gm,
                                             name="obj_prob")
        self.classes_prob = self.instance.Tensor(self.dtype,
                                                 (self.batch, self.classes,
                                                  adj_hw),
                                                 scope=tik.scope_gm,
                                                 name="classes_prob")
        self.img_info = self.instance.Tensor(self.dtype,
                                             (self.batch * 4 + BLOCK_SIZE,),
                                             scope=tik.scope_gm,
                                             name="img_info")
        # Intermediate Output
        self.inter_coords = self.instance.Tensor(self.dtype, self.get_shape(
            (self.batch, 4, self.boxes * self.width * self.height), True),
                                                 scope=tik.scope_gm,
                                                 name="inter_coords",
                                                 is_workspace=True)
        self.inter_classes = self.instance.Tensor(self.dtype, self.get_shape(
            (self.batch, self.classes, self.boxes * self.width * self.height),
            True), scope=tik.scope_gm, name="inter_classes", is_workspace=True)


    def get_shape(self, old_shape, need_low_dim=False):
        """
          compute shape

          Parameters
          ----------
           old_shape: shape before compute
           need_low_dim: whether need low dim,true or false

          Returns
          -------
          None
          """
        old_shape = list(old_shape)
        length = len(old_shape)

        if length == 1:
            old_shape[0] += BLOCK_SIZE
            return tuple(old_shape)

        if not need_low_dim:
            size = self.dsize
            for i in range(0, length):
                size *= old_shape[i]
            unit_rev = self.dsize
            for i in range(1, length):
                unit_rev *= old_shape[i]
        else:
            size = self.dsize * old_shape[length - 1]
            unit_rev = size

        if size % BLOCK_SIZE == 0:
            rev = 0
        else:
            rev = BLOCK_SIZE // unit_rev + 1
        old_shape[0] += rev

        return tuple(old_shape)


    def get_adj_hw(self, total_len):
        """
          compute height and weight with 32 alignment

          Parameters
          ----------
           total_len: total length

          Returns
          -------
          None
        """
        return math.ceil((total_len + 16) / 16) * 16


    def get_burlen(self, length):
        """
          compute data move nburst

          Parameters
          ----------
           length: the number of elements need data move

          Returns
          -------
          burlen: data move nburst
          """
        if (length * self.dsize) % BLOCK_SIZE == 0:
            return (length * self.dsize) // BLOCK_SIZE

        return (length * self.dsize) // BLOCK_SIZE + 1

    def get_repeat(self, length):
        """
          compute vector instructs repeat times

          Parameters
          ----------
           length: the number of elements need data move
          Returns
          -------
          repeats: vector instructs repeat times
          """
        if (length * self.dsize) % VECTOR_BYTE_SIZE == 0:
            return (length * self.dsize) // VECTOR_BYTE_SIZE

        return (length * self.dsize) // VECTOR_BYTE_SIZE + 1

    def convert_biases_data(self):
        """
        convert biases data
        Returns
        -------

        """
        ub_bias = self.instance.Tensor(self.dtype, (VECTOR_BYTE_SIZE,),
                                       scope=tik.scope_ubuf,
                                       name="ub_bias")
        t_scalar = self.instance.Scalar(self.dtype)
        # set bias to ub
        for i in range(0, self.boxes):
            t_scalar.set_as(self.biases[2 * i])
            ub_bias[2 * i].set_as(t_scalar)
            t_scalar.set_as(self.biases[2 * i + 1])
            ub_bias[2 * i + 1].set_as(t_scalar)

        return ub_bias



class CommonInstruct(GetCorrectBoxBase):
    """
     for simplify instruction writing
    """
    def __init__(self, input_dict):
        """
        init function
        Parameters
        ----------
        input_dict: some input param
        """
        super(CommonInstruct, self).__init__(input_dict)
        self.input_di = input_dict

    def t_data_move(self, dst, src, burlen, nburst=1):
        """
        dma instruction
        Parameters
        ----------
        dst: dst addr
        src: src addr
        burlen: burlen param
        nburst: nburst

        Returns
        -------

        """
        self.instance.data_move(dst, src, 0, nburst, burlen, 0, 0)

    def t_vector_dup(self, dst, number, repeat=1):
        """
        vector_dup instruction
        Parameters
        ----------
        dst: dst addr
        number: number for instruction
        repeat: repeat time

        Returns
        -------

        """
        self.instance.vec_dup(self.mask, dst, number, repeat, 8)

    def t_vdiv(self, dst, src0, src1, repeat=1):
        """
        vdiv instruction
        Parameters
        ----------
        dst: dst addr
        src0: src0 addr
        src1: src1 addr
        repeat: repeat times

        Returns
        -------

        """
        self.instance.vdiv(self.mask, dst, src0, src1, repeat, 1, 1, 1, 8, 8, 8)

    def t_vrec(self, dst, src, repeat):
        """
        vrec instruction
        Parameters
        ----------
        dst: dst addr
        src: src addr
        repeat: repeat times

        Returns
        -------

        """
        self.instance.vec_rec(self.mask, dst, src, repeat, 8, 8)

    def t_vmul(self, dst, src0, src1, repeat=1):
        """
        vmul instruction
        Parameters
        ----------
        dst: dst addr
        src0: src0 addr
        src1: src1 addr
        repeat: repeat times

        Returns
        -------

        """
        self.instance.vec_mul(self.mask, dst, src0, src1, repeat, 8, 8, 8)

    def t_vadds(self, dst, src, scalar, repeat=1):
        """
        vadds instruction
        Parameters
        ----------
        dst: dst addr
        src: src addr
        scalar: scalar for compute
        repeat: repeat times

        Returns
        -------

        """
        self.instance.vec_adds(self.mask, dst, src, scalar, repeat, 8, 8)

    def t_vmuls(self, dst, src, scalar, repeat=1):
        """
        vmuls instruction
        Parameters
        ----------
        dst: dst addr
        src: src addr
        scalar: scalar for compute
        repeat: repeat time

        Returns
        -------

        """
        self.instance.vec_muls(self.mask, dst, src, scalar, repeat, 8, 8)

    def t_vadd(self, dst, src0, src1, repeat=1):
        """
        vadd instruction
        Parameters
        ----------
        dst: dst addr
        src0: src0 addr
        src1: src1 addr
        repeat: repeat times

        Returns
        -------

        """
        self.instance.vec_add(self.mask, dst, src0, src1, repeat, 8, 8, 8)

    def t_vexp(self, dst, src, repeat):
        """
        vexp instruction
        Parameters
        ----------
        dst: dst addr
        src: src addr
        repeat: repeat times

        Returns
        -------

        """
        self.instance.vec_exp(self.mask, dst, src, repeat, 8, 8)


class CorrectBoxComputer(CommonInstruct):
    """
    class for correct box
    """
    def __init__(self, input_dict):
        """
        init function
        Parameters
        ----------
        input_dict: some param
        """
        super(CorrectBoxComputer, self).__init__(input_dict)
        self.block_num, self.outer_loop, self.outer_tail = self.get_block_param()

    def get_block_param(self):
        """
        compute block param
        Returns
        -------

        """
        block_num = tik.Dprofile().get_aicore_num()
        if block_num > self.batch:
            outer_loop = 1
            block_num = self.batch
            outer_tail = 0
        else:
            outer_loop = self.batch // block_num
            outer_tail = self.batch - block_num * outer_loop

        return block_num, outer_loop, outer_tail

    def get_faces_params(self, adj_hw, c_num):
        """
          data move of small shape

          Parameters
          ----------
           adj_hw: the lenth of boxes
           c_num: the length of boxes's c dim
          Returns
          -------
          None
          """
        if self.one_max_size // (adj_hw * self.dsize) > c_num:
            loop = 1
            last_loop = c_num
            faces_in_one_loop = 0
        else:
            faces_in_one_loop = self.one_max_size // (adj_hw * self.dsize)
            loop = c_num // faces_in_one_loop
            faces_tail = c_num - faces_in_one_loop * loop
            loop = loop if faces_tail == 0 else loop + 1
            last_loop = faces_in_one_loop if faces_tail == 0 else faces_tail

        return faces_in_one_loop, last_loop, loop

    def get_tiling_param(self, height, weight):
        """
           compute tilling param

           Parameters
           ----------
            height: box's height
            weight: box's width
           Returns
           -------
           mov_len: the number of elements of each loop
           mov_loop: loop times
           last_len: the number of elements of last loop
           """
        max_size = self.one_max_size
        # max_size = 256

        mov_len = max_size // self.dsize
        mov_loop = (height * weight) // (max_size // self.dsize)
        mov_tail = height * weight - mov_len * mov_loop

        mov_loop = mov_loop if mov_tail == 0 else mov_loop + 1
        last_len = mov_len if mov_tail == 0 else mov_tail

        return mov_len, mov_loop, last_len

    def newton_div(self, dst, divisor, dividend, repeat):
        """
          use newton_div to improve performance

          Parameters
          ----------
           dst: vdiv's dest tensor
           divisor: vdiv's src0 tensor
           dividend: vdiv's src1 tensor
           repeat: vdiv's needs repeat times
          Returns
          -------
          None
          """
        if tik.Dprofile().get_product_name() in (CLOUD, AIC):
            self.t_vdiv(dst, divisor, dividend, repeat)

        else:
            with self.instance.new_stmt_scope():
                tmp_tensor = self.instance.Tensor(self.dtype,
                                                  (repeat*VECTOR_BYTE_SIZE, ),
                                                  scope=tik.scope_ubuf,
                                                  name="tmp_tensor")
                # 1/dividend
                self.t_vrec(tmp_tensor, dividend, repeat)

                self.t_vmul(dividend, dividend, tmp_tensor, repeat)
                self.t_vadds(dividend, dividend, NEG_TWO, repeat)
                self.t_vmul(dividend, dividend, tmp_tensor, repeat)
                self.t_vmuls(dividend, dividend, NEG_ONE, repeat)

                # divisor * (1/dividend)
                self.t_vmul(dst, divisor, dividend, repeat)


    def get_new_h_w(self, img_info, param):
        """
          compute boxes's height and width

          Parameters
          ----------
           img_info: a tensor,store image's width and height
           param: a dict,the keys as follow:
                  ub_d: a middle tensor,used to compute boxes's height and width
                  ub_e:a middle tensor,used to compute boxes's height and width
                  ub_f:a middle tensor,used to compute boxes's height and width
                  ub_g:a middle tensor,used to compute boxes's height and width
                  ret_tensor:a middle tensor,used to compute boxes's
                             height and width
                  lgt_tensor:a middle tensor,used to compute boxes's
                             height and width

          Returns
          -------
          new_h: a scalar,store new_h
          new_w: a scalar,store new_w
          """
        tmp_scalar = self.instance.Scalar(self.dtype)
        new_h = self.instance.Scalar(self.dtype)
        new_w = self.instance.Scalar(self.dtype)
        # if netw/w < neth/h
        # vdup neth/h
        tmp_scalar.set_as(img_info[0])
        self.t_vector_dup(param['ub_d'], tmp_scalar)
        tmp_scalar.set_as(img_info[2])
        self.t_vector_dup(param['ub_g'], tmp_scalar)

        self.newton_div(param['ub_d'], param['ub_d'], param['ub_g'], REPEAT_ONE)

        # vdup netw/w
        tmp_scalar.set_as(img_info[1])
        self.t_vector_dup(param['ub_e'], tmp_scalar)
        tmp_scalar.set_as(img_info[3])
        self.t_vector_dup(param['ub_g'], tmp_scalar)
        self.newton_div(param['ub_e'], param['ub_e'], param['ub_g'], REPEAT_ONE)

        sel = self.instance.Tensor("uint16", (8, ),
                                   name="sel",
                                   scope=tik.scope_ubuf)
        self.instance.vec_dup(8, sel, 0, 1, 8)
        self.instance.vec_cmpv_lt(sel, param['ub_e'], param['ub_d'], 1, 8, 8)

        # get new w
        tmp_scalar.set_as(img_info[1])
        param['lgt_tensor'][0].set_as(tmp_scalar)
        tmp_scalar.set_as(img_info[3])
        self.t_vmuls(param['ub_d'], param['ub_d'], tmp_scalar, REPEAT_ONE)

        self.instance.vec_sel(self.mask, VALUE_ZERO, param['ret_tensor'], sel,
                           param['lgt_tensor'],
                           param['ub_d'], STRIDE_ONE)
        new_w.set_as(param['ret_tensor'][0])

        # get new h
        tmp_scalar.set_as(img_info[2])
        self.t_vmuls(param['ub_e'], param['ub_e'], tmp_scalar, REPEAT_ONE)

        tmp_scalar.set_as(img_info[0])
        param['lgt_tensor'][0].set_as(tmp_scalar)
        self.instance.vec_sel(self.mask, VALUE_ZERO, param['ret_tensor'], sel,
                           param['ub_e'],
                           param['lgt_tensor'], STRIDE_ONE)
        new_h.set_as(param['ret_tensor'][0])

        return new_h, new_w

    def get_x_y_params(self, img_info):
        """
          compute x,y parameters

          Parameters
          ----------
           img_info: a tensor,store image's width and height

          Returns
          -------
          x_vmuls_val: a scalar,store x_vmuls_val
          y_vmuls_val: a scalar,store y_vmuls_val
          x_vadds_val: a scalar,store x_vadds_val
          y_vadds_val: a scalar,store y_vadds_val
          """
        x_vmuls_val = self.instance.Scalar(self.dtype)
        y_vmuls_val = self.instance.Scalar(self.dtype)
        x_vadds_val = self.instance.Scalar(self.dtype)
        y_vadds_val = self.instance.Scalar(self.dtype)

        param = {}

        with self.instance.new_stmt_scope():
            param['ub_d'] = self.instance.Tensor(self.dtype,
                                                 (VECTOR_BYTE_SIZE,),
                                                 scope=tik.scope_ubuf,
                                                 name="ub_d")
            param['ub_e'] = self.instance.Tensor(self.dtype,
                                                 (VECTOR_BYTE_SIZE,),
                                                 scope=tik.scope_ubuf,
                                                 name="ub_e")
            param['ub_f'] = self.instance.Tensor(self.dtype,
                                                 (VECTOR_BYTE_SIZE,),
                                                 scope=tik.scope_ubuf,
                                                 name="ub_f")
            param['ub_g'] = self.instance.Tensor(self.dtype,
                                                 (VECTOR_BYTE_SIZE,),
                                                 scope=tik.scope_ubuf,
                                                 name="ub_g")
            param['lgt_tensor'] = self.instance.Tensor(self.dtype,
                                                       (VECTOR_BYTE_SIZE,),
                                                       scope=tik.scope_ubuf,
                                                       name="lgt_tensor")
            param['ret_tensor'] = self.instance.Tensor(self.dtype,
                                                       (VECTOR_BYTE_SIZE,),
                                                       scope=tik.scope_ubuf,
                                                       name="ret_tensor")

            new_h, new_w = self.get_new_h_w(img_info, param)

            tmp_scalar = self.instance.Scalar(self.dtype)

            # x vmuls param --> netw / new_w
            tmp_scalar.set_as(img_info[1])
            # ub_d netw
            self.t_vector_dup(param['ub_d'], tmp_scalar, REPEAT_ONE)

            # ub_e new_w
            self.t_vector_dup(param['ub_e'], new_w, REPEAT_ONE)

            # netw / new_w
            self.newton_div(param['ub_d'], param['ub_d'], param['ub_e'],
                            REPEAT_ONE)

            x_vmuls_val.set_as(param['ub_d'][0])

            # x vadds param --> ((new_w - netw)/2.0/netw) * (netw / new_w)
            # --> ((-1)*(netw / new_w) + 1)* 0.5
            self.t_vmuls(param['ub_d'], param['ub_d'], NEG_ONE, REPEAT_ONE)
            self.t_vadds(param['ub_d'], param['ub_d'], VALUE_ONE, REPEAT_ONE)
            self.t_vmuls(param['ub_d'], param['ub_d'], VALUE_HALF, REPEAT_ONE)

            x_vadds_val.set_as(param['ub_d'][0])

            # y vmuls param --> neth / new_h
            tmp_scalar.set_as(img_info[0])
            # ub_d neth
            self.t_vector_dup(param['ub_d'], tmp_scalar, REPEAT_ONE)

            # ub_e new_h
            self.t_vector_dup(param['ub_e'], new_h, REPEAT_ONE)

            # neth / new_h
            self.newton_div(param['ub_d'], param['ub_d'], param['ub_e'],
                            REPEAT_ONE)

            y_vmuls_val.set_as(param['ub_d'][0])

            # y vadds param --> ((-1)*(neth / new_h) + 1)* 0.5
            self.t_vmuls(param['ub_d'], param['ub_d'], NEG_ONE, REPEAT_ONE)
            self.t_vadds(param['ub_d'], param['ub_d'], VALUE_ONE, REPEAT_ONE)
            self.t_vmuls(param['ub_d'], param['ub_d'], VALUE_HALF, REPEAT_ONE)

            y_vadds_val.set_as(param['ub_d'][0])

        return x_vmuls_val, x_vadds_val, y_vmuls_val, y_vadds_val

    def correct_box(self, batch_idx, img_ub):
        """
        Computing entry

        Parameters
        ----------
        batch_idx: batch index
        img_ub: image size

        Returns
        -------

        """
        with self.instance.new_stmt_scope():
            if self.width * self.height * self.dsize < self.one_max_size // 2:
                self.small_surface_template(batch_idx, img_ub)
            else:
                self.big_surface_template(batch_idx, img_ub)

    def big_surface_template(self, b_idx, img_ub):
        """
        big size data template
        Parameters
        ----------
        b_idx: batch index
        img_ub: image size

        Returns
        -------

        """
        param = {"img_ub": img_ub}
        param['mov_len'], param['mov_loop'], param[
            'last_len'] = self.get_tiling_param(self.height, self.width)
        param['ub_bias'] = self.convert_biases_data()
        param['x_vmuls_val'], param['x_vadds_val'], param['y_vmuls_val'], param[
            'y_vadds_val'] = self.get_x_y_params(param['img_ub'])

        shape = self.one_max_size // self.dsize
        with self.instance.for_range(0, self.boxes * 4) as cycle:


            param['co_id'] = self.instance.Scalar()
            param['box_id'] = self.instance.Scalar()
            param['co_id'].set_as(cycle // self.boxes)
            param['box_id'].set_as(cycle % self.boxes)
            thread_num = 1 if param['mov_loop'] == 1 else 2
            with self.instance.for_range(0, param['mov_loop'],
                                         thread_num=thread_num) as loop:
                param['ub_a'] = self.instance.Tensor(self.dtype, (shape,),
                                                     scope=tik.scope_ubuf,
                                                     name="ub_a")
                param['ub_b'] = self.instance.Tensor(self.dtype, (shape,),
                                                     scope=tik.scope_ubuf,
                                                     name="ub_b")
                param['ub_c'] = self.instance.Tensor(self.dtype, (shape,),
                                                     scope=tik.scope_ubuf,
                                                     name="ub_c")
                param['last_32b'] = self.instance.Tensor(self.dtype,
                                                         (BLOCK_SIZE,),
                                                         scope=tik.scope_ubuf,
                                                         name="last_32b")
                param['burlen'] = self.instance.Scalar("int32")
                repeat = self.instance.Scalar("int32")

                with self.instance.if_scope(loop == param['mov_loop'] - 1):
                    param['burlen'].set_as(self.get_burlen(param['last_len']))
                    repeat.set_as(self.get_repeat(param['last_len']))
                with self.instance.else_scope():
                    param['burlen'].set_as(self.get_burlen(param['mov_len']))
                    repeat.set_as(self.get_repeat(param['mov_len']))

                # move coords data to ub a
                self.t_data_move(param['ub_a'],
                                 self.coord_data[b_idx, cycle,
                                                 param['mov_len'] * loop],
                                 param['burlen'])

                self.compute_big_xy(b_idx, loop, param, repeat)

                self.compute_big_hw(b_idx, loop, param, repeat)

    def compute_big_xy(self, batch, loop, param, repeat):
        """
           compute big shape of x,y

           Parameters
           ----------
            batch: the number of picture
            loop: loop times
            param: a dict,the keys as fllow:
                   mov_len: the number of elements of each data move
                   mov_loop: data move loop times
                   last_len: the number of elements of last_len data move
                   ub_bias: a tensor,store bias
                   x_vmuls_val: a scalar
                   x_vadds_val: a scalar
                   y_vmuls_val: a scalar
                   ub_a: a tensor,store middle compute data
                   ub_b: a tensor,store middle compute data
                   ub_c: a tensor,store middle compute data
                   last_32b: a tensor,store last_32b data
                   co_id: a scalar,store co_id
                   box_id:  a scalar,store box_id
                   in_data: a tensor
            repeat: vector repeat times

           Returns
           -------
           None
           """
        tmp_scalar = self.instance.Scalar(self.dtype)
        offset_scalar = self.instance.Scalar("int32")
        # x
        with self.instance.if_scope(param['co_id'] == 0):
            tmp_scalar.set_as(param['img_ub'][3])
            # move windex to ub b
            self.t_data_move(param['ub_b'],
                             self.windex[loop * param['mov_len']],
                             param['burlen'])

            # a = x + windex
            self.t_vadd(param['ub_a'], param['ub_a'], param['ub_b'], repeat)
            # a = (x + windex)*(1/w)
            self.t_vmuls(param['ub_b'], param['ub_a'], 1.0 / self.width, repeat)

            self.t_vmuls(param['ub_b'], param['ub_b'], param['x_vmuls_val'],
                         repeat)
            self.t_vadds(param['ub_b'], param['ub_b'], param['x_vadds_val'],
                         repeat)

            if not self.relative:
                self.t_vmuls(param['ub_b'], param['ub_b'], tmp_scalar, repeat)

            offset_scalar.set_as(
                self.hw_len*param['box_id'] + param['mov_len'] * loop)

            self.t_data_move(self.inter_coords[batch, 0, offset_scalar],
                             param['ub_b'], param['burlen'])

        # y
        with self.instance.if_scope(param['co_id'] == 1):
            tmp_scalar.set_as(param['img_ub'][2])
            # move hindex to ub b
            self.t_data_move(param['ub_b'],
                             self.hindex[loop * param['mov_len']],
                             param['burlen'])
            self.t_vadd(param['ub_b'], param['ub_a'], param['ub_b'], repeat)
            self.t_vmuls(param['ub_b'], param['ub_b'], 1.0 / self.height, repeat)

            self.t_vmuls(param['ub_b'], param['ub_b'], param['y_vmuls_val'],
                         repeat)
            self.t_vadds(param['ub_b'], param['ub_b'], param['y_vadds_val'],
                         repeat)

            if not self.relative:
                self.t_vmuls(param['ub_b'], param['ub_b'], tmp_scalar, repeat)

            offset_scalar.set_as(
                self.hw_len*param['box_id'] + param['mov_len'] * loop)

            self.t_data_move(self.inter_coords[batch, 1, offset_scalar],
                             param['ub_b'], param['burlen'])

    def compute_big_hw(self, batch, loop, param, repeat):
        """
           compute big shape height and weight

           Parameters
           ----------
            batch: the number of picture
            loop: loop times
            param: a dict,the keys as fllow:
                   mov_len: the number of elements of each data move
                   mov_loop: data move loop times
                   last_len: the number of elements of last_len data move
                   ub_bias: a tensor,store bias
                   x_vmuls_val: a scalar
                   x_vadds_val: a scalar
                   y_vmuls_val: a scalar
                   ub_a: a tensor,store middle compute data
                   ub_b: a tensor,store middle compute data
                   ub_c: a tensor,store middle compute data
                   last_32b: a tensor,store last_32b data
                   co_id: a scalar,store co_id
                   box_id:  a scalar,store box_id
                   in_data: a tensor
            repeat: vector repeat times

           Returns
           -------
           None
           """
        tmp_scalar = self.instance.Scalar(self.dtype)
        bias_value = self.instance.Scalar(self.dtype)
        offset_scalar = self.instance.Scalar("int32")
        # h
        with self.instance.if_scope(param['co_id'] == 2):
            bias_value.set_as(
                param['ub_bias'][VALUE_TWO * param['box_id'] + VALUE_ONE])
            tmp_scalar.set_as(param['img_ub'][0])
            self.t_vexp(param['ub_c'], param['ub_a'], repeat)
            self.t_vmuls(param['ub_c'], param['ub_c'], bias_value, repeat)

            self.t_vmuls(param['ub_b'], param['ub_c'], 1.0 / self.height,
                         repeat)
            self.t_vmuls(param['ub_b'], param['ub_b'], param['y_vmuls_val'],
                         repeat)

            if not self.relative:
                tmp_scalar.set_as(param['img_ub'][2])
                self.t_vmuls(param['ub_b'], param['ub_b'], tmp_scalar, repeat)

            offset_scalar.set_as(
                self.hw_len*param['box_id'] + param['mov_len'] * loop)

            self.t_data_move(self.inter_coords[batch, 2, offset_scalar],
                             param['ub_b'], param['burlen'])

        # w
        with self.instance.if_scope(param['co_id'] == 3):
            bias_value.set_as(param['ub_bias'][VALUE_TWO * param['box_id']])

            # img ub: neth,netw,scaleh,scalew
            tmp_scalar.set_as(param['img_ub'][1])
            self.t_vexp(param['ub_c'], param['ub_a'], repeat)
            self.t_vmuls(param['ub_c'], param['ub_c'], bias_value, repeat)
            self.t_vmuls(param['ub_b'], param['ub_c'], 1.0 / self.width,
                         repeat)
            self.t_vmuls(param['ub_b'], param['ub_b'], param['x_vmuls_val'],
                         repeat)

            if not self.relative:
                tmp_scalar.set_as(param['img_ub'][3])
                self.t_vmuls(param['ub_b'], param['ub_b'], tmp_scalar, repeat)
            offset_scalar.set_as(
                self.hw_len*param['box_id'] + param['mov_len']*loop)
            if self.hw_len % BLOCK_SIZE != 0:
                with self.instance.if_scope(
                        tik.all(param['box_id'] == self.boxes - 1,
                                loop == param['mov_len'] - 1)):
                    param['burlen'].set_as(param['burlen'] - 1)
                    with self.instance.if_scope(param['burlen'] > 0):
                        self.t_data_move(
                            self.inter_coords[batch, 3, offset_scalar],
                            param['ub_b'], param['burlen'])
                    param['burlen'].set_as(param['burlen'] + 1)

                    tail_idx = self.instance.Scalar("int32")
                    tail_idx.set_as(param['last_len'] - self.len_32b)
                    self.t_data_move(
                        param['last_32b'],
                        self.inter_coords[batch, 3,
                                          offset_scalar + tail_idx], 1)

                    with self.instance.for_range(0, self.hwtail_len) as cycle:
                        tmp_scalar = self.instance.Scalar(self.dtype)
                        tmp_scalar.set_as(param['ub_b'][param['last_len'] - \
                                                        self.hwtail_len +
                                                        cycle])
                        param['last_32b'][self.len_32b - \
                                          self.hwtail_len + \
                                          cycle].set_as(tmp_scalar)
                    self.t_data_move(
                        self.inter_coords[batch, 3, offset_scalar + tail_idx],
                        param['last_32b'], 1)

                with self.instance.else_scope():
                    self.t_data_move(
                        self.inter_coords[batch, 3, offset_scalar],
                        param['ub_b'], param['burlen'])
            else:
                self.t_data_move(
                    self.inter_coords[batch, 3, offset_scalar],
                    param['ub_b'], param['burlen'])

    def small_surface_template(self, b_idx, img_ub):
        """
        Small data calculation template
        Parameters
        ----------
        b_idx; batch index
        img_ub: image size

        Returns
        -------

        """
        param = {"img_ub": img_ub}
        param['x_vmuls_val'], param['x_vadds_val'], param['y_vmuls_val'], param[
            'y_vadds_val'] = self.get_x_y_params(param['img_ub'])
        param['faces_one_loop'], param['last_faces'], param['loop'] = \
            self.get_faces_params(self.adj_hw, 4 * self.boxes)

        with self.instance.for_range(0, param['loop']) as loop_idx:
            param['ub_a'] =\
                self.instance.Tensor(self.dtype,
                                     (self.one_max_size // self.dsize,),
                                     scope=tik.scope_ubuf, name="ub_a")

            param['mov_in_faces'] = self.instance.Scalar("int32")
            with self.instance.if_scope(loop_idx != param['loop'] - 1):
                param['mov_in_faces'].set_as(param['faces_one_loop'])
            with self.instance.else_scope():
                param['mov_in_faces'].set_as(param['last_faces'])

            param['burlen'] = self.instance.Scalar("int32")
            param['burlen'].\
                set_as(param['mov_in_faces']*self.adj_hw*self.dsize // BLOCK_SIZE)

            self.t_data_move(param['ub_a'],
                             self.coord_data[b_idx, param['faces_one_loop'] * loop_idx, 0],
                             param['burlen'])

            with self.instance.for_range(0, param['mov_in_faces']) as cycle:
                param['ub_b'] =\
                    self.instance.Tensor(self.dtype,
                                         (self.one_max_size // self.dsize,),
                                         scope=tik.scope_ubuf, name="ub_b")
                param['ub_c'] =\
                    self.instance.Tensor(self.dtype,
                                         (self.one_max_size // self.dsize,),
                                         scope=tik.scope_ubuf, name="ub_c")
                param['last_32b'] =\
                    self.instance.Tensor(self.dtype,
                                         (self.one_max_size // self.dsize,),
                                         scope=tik.scope_ubuf, name="last_32b")

                # Calculate the cindex.
                start_idx = self.instance.Scalar("int32")
                start_idx.set_as(cycle * self.adj_hw)

                # Indicates the number of the box.
                param['box_id'] = self.instance.Scalar("int32")
                param['box_id'].set_as(
                    (param['faces_one_loop'] * loop_idx + cycle) % self.boxes)

                param['co_id'] = self.instance.Scalar("int32")
                param['co_id'].set_as(
                    (param['faces_one_loop'] * loop_idx + cycle) // self.boxes)

                param['burlen'].set_as(
                    self.get_burlen(self.height*self.width))
                repeat = self.get_repeat(self.height*self.width)

                self.compute_small_xy(b_idx, param, repeat, start_idx)
                self.compute_small_hw(b_idx, param, repeat, start_idx)

    def compute_small_hw(self, batch, param, repeat, start_idx):
        """
          compute small shape of height and weight

           Parameters
           ----------
            batch: the number of picture
            param: a dict,the keys as fllow:
                   ub_a: a tensor,store middle compute data
                   ub_b: a tensor,store middle compute data
                   ub_c: a tensor,store middle compute data
                   last_32b: a tensor,store last_32b data
                   co_id: a scalar,store co_id
                   box_id: a scalar,store box_id
                   img_ub: a tensor,store img data
                   x_vmuls_val: a scalar,store x_vmuls_val
                   y_vmuls_val: a scalar,store y_vmuls_val
                   ub_bias:  a tensor,store bias data

            repeat: vector repeat times
            start_idx: a scalar,store start_idx

           Returns
           -------
           None
           """
        param['ub_bias'] = self.convert_biases_data()
        tmp_scalar = self.instance.Scalar(self.dtype)
        bias_value = self.instance.Scalar(self.dtype)
        with self.instance.if_scope(param['co_id'] == VALUE_TWO):
            bias_value.set_as(
                param['ub_bias'][VALUE_TWO * param['box_id'] + VALUE_ONE])
            tmp_scalar.set_as(param['img_ub'][0])

            self.t_vexp(param['ub_c'], param['ub_a'][start_idx], repeat)
            self.t_vmuls(param['ub_c'], param['ub_c'], bias_value, repeat)

            self.t_vmuls(param['ub_b'], param['ub_c'], 1.0 / self.height,
                         repeat)

            self.t_vmuls(param['ub_b'], param['ub_b'], param['y_vmuls_val'],
                         repeat)

            if not self.relative:
                tmp_scalar.set_as(param['img_ub'][2])
                self.t_vmuls(param['ub_b'], param['ub_b'], tmp_scalar, repeat)

            self.t_data_move(
                self.inter_coords[
                    batch, 2, self.height*self.width*param['box_id']],
                param['ub_b'], param['burlen'])

        with self.instance.if_scope(param['co_id'] == VALUE_THREE):
            bias_value.set_as(param['ub_bias'][VALUE_TWO * param['box_id']])
            tmp_scalar.set_as(param['img_ub'][1])

            self.t_vexp(param['ub_c'], param['ub_a'][start_idx], repeat)
            self.t_vmuls(param['ub_c'], param['ub_c'], bias_value, repeat)

            self.t_vmuls(param['ub_b'], param['ub_c'], 1.0 / self.width,
                         repeat)

            self.t_vmuls(param['ub_b'], param['ub_b'], param['x_vmuls_val'],
                         repeat)

            if not self.relative:
                tmp_scalar.set_as(param['img_ub'][3])
                self.t_vmuls(param['ub_b'], param['ub_b'], tmp_scalar, repeat)

            if (self.hw_len*self.dsize) % BLOCK_SIZE != 0 and self.block_num > 1:
                with self.instance.if_scope(param['box_id'] == self.boxes - 1):
                    # Each side is at least 32B, so burlen is at least 2.
                    param['burlen'].set_as(param['burlen'] - 1)

                    self.t_data_move(
                        self.inter_coords[
                            batch, 3, self.height*self.width*param['box_id']],
                        param['ub_b'], param['burlen'])
                    param['burlen'].set_as(param['burlen'] + 1)
                    tail_idx = self.instance.Scalar(name="tail_idx")
                    offset_idx = self.instance.Scalar("int32")
                    tail_idx.set_as(self.hw_len - self.len_32b)
                    offset_idx.set_as(self.hw_len*param['box_id'] + self.hw_len
                                      - self.len_32b)

                    self.t_data_move(param['last_32b'],
                                     self.inter_coords[batch, 3, offset_idx], 1)

                    with self.instance.for_range(0, self.hwtail_len) as index:
                        tmp_scalar = self.instance.Scalar(self.dtype)
                        tmp_scalar.set_as(
                            param['ub_b'][self.hw_len -self.hwtail_len + index])
                        param['last_32b'][self.len_32b - \
                                          self.hwtail_len + \
                                          index].set_as(tmp_scalar)

                    self.t_data_move(self.inter_coords[batch, 3, offset_idx],
                                     param['last_32b'], 1)

                with self.instance.else_scope():
                    self.t_data_move(
                        self.inter_coords[
                            batch, 3, self.hw_len*param['box_id']],
                        param['ub_b'], param['burlen'])
            else:
                self.t_data_move(
                    self.inter_coords[
                        batch, 3, self.hw_len*param['box_id']],
                    param['ub_b'], param['burlen'])


    def compute_small_xy(self, batch, param, repeat, start_idx):
        """
          compute small shape of x,y

           Parameters
           ----------
            batch: the number of picture
            param: a dict,the keys as fllow:
                   ub_a: a tensor,store middle compute data
                   ub_b: a tensor,store middle compute data
                   ub_c: a tensor,store middle compute data
                   last_32b: a tensor,store last_32b data
                   co_id: a scalar,store co_id
                   box_id: a scalar,store box_id
                   img_ub: a tensor,store img data
                   x_vmuls_val: a scalar,store x_vmuls_val
                   y_vmuls_val: a scalar,store y_vmuls_val
                   ub_bias:  a tensor,store bias data

            repeat: vector repeat times
            start_idx: a scalar,store start_idx

           Returns
           -------
           None
           """
        tmp_scalar = self.instance.Scalar(self.dtype)
        with self.instance.if_scope(param['co_id'] == VALUE_ZERO):
            self.t_data_move(param['ub_b'], self.windex, param['burlen'])
            self.t_vadd(param['ub_b'], param['ub_a'][start_idx], param['ub_b'],
                        repeat)
            self.t_vmuls(param['ub_b'], param['ub_b'], 1.0 / self.width, repeat)

            self.t_vmuls(param['ub_b'], param['ub_b'], param['x_vmuls_val'],
                         repeat)
            self.t_vadds(param['ub_b'], param['ub_b'], param['x_vadds_val'],
                         repeat)
            if not self.relative:
                self.t_vmuls(param['ub_b'], param['ub_b'], tmp_scalar, repeat)

            self.t_data_move(
                self.inter_coords[
                    batch, 0, self.height*self.width*param['box_id']],
                param['ub_b'], param['burlen'])

        with self.instance.if_scope(param['co_id'] == VALUE_ONE):
            self.t_data_move(param['ub_b'], self.hindex, param['burlen'])

            # a = y + hindex
            self.t_vadd(param['ub_b'], param['ub_a'][start_idx], param['ub_b'],
                        repeat)
            # a = (y + hindex)*(1/lh)
            self.t_vmuls(param['ub_b'], param['ub_b'], 1.0 / self.height,
                         repeat)

            self.t_vmuls(param['ub_b'], param['ub_b'], param['y_vmuls_val'],
                         repeat)
            self.t_vadds(param['ub_b'], param['ub_b'], param['y_vadds_val'],
                         repeat)

            if not self.relative:
                self.t_vmuls(param['ub_b'], param['ub_b'], tmp_scalar, repeat)

            self.t_data_move(
                self.inter_coords[
                    batch, 1, self.height*self.width*param['box_id']],
                param['ub_b'], param['burlen'])
