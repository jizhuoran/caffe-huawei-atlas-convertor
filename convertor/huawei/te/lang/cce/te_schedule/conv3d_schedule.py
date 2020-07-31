# -*- coding:utf-8 -*-
"""
Copyright 2020 Huawei Technologies Co., Ltd

Schedule of conv3d.
"""
from te import tvm
from te import platform as cce
from te.platform import CUBE_MKN
from te.platform import cce_params
from te.platform import get_soc_spec
from te.lang.cce.te_compute import util as te_util
from te.lang.cce.te_compute import Conv3DParam

# tiling check
TILING_L1_SHAPWE_DIM = 4
TILING_AL0_MATRIX_DIM = [6]
TILING_BL0_MATRIX_DIM = [6, 0]
TILING_CL0_MATRIX_DIM = [6]
TILING_CUB_MATRIX_DIM = [6]
TILING_BLOCK_DIM_DIM = [4]
TILING_FLOAT16_MKN = 16
VALID_TILING_NUM = 32


class CceConv3dOp:
    """class of cce index op

    Parameters
    ----------
    scope : cal buffer like local.UB

    need_tensorize : if need to doing tensorize when using calculate

    need_pragma : if need to doing paagma when using calculate

    Returns
    -------
    cceop_instance : instance of cceop

    """
    def __init__(self, scope, need_tensorize=True, need_pragma=True):
        self._need_tensorize = need_tensorize
        self._need_pragma = need_pragma
        self._scope = scope
        self._schedule = None
        self._tensor_map = Conv3DParam.TENSOR_MAP
        self._dim_map = Conv3DParam.dim_map
        self._tiling = Conv3DParam.tiling
        self._res_tensor = None
        self.body_ops = []
        self.input_ops = []
        self.output_ops = []
        self._has_vector_flag = False
        self._in_quant_sqrt_flag = False

    def weight_to_bl1(self, tiling, filter_matrix, weight, c_col):
        """
        weight to bl1.

        Parameters
        ----------
        tiling : case tiling

        filter_matrix : full load

        weight: input weight tensor

        c_col: stage loc

        Returns
        -------
        bl1
        """
        sch = self._schedule
        if tiling["BL0_matrix"] == filter_matrix:
            tiling["BL0_matrix"] = []

        if tiling["BL0_matrix"] == []:
            tiling["BL1_shape"] = None

        if tiling["BL1_shape"] is not None:
            bl1 = sch.cache_read(weight, cce.scope_cbuf, [c_col])
        else:
            # tiling["BL1_shape"] != None ---> weigt from OUT To l0b directly
            bl1 = weight
        return bl1

    def factor_al1_bl1(self, tiling, c_factor):
        """
        get al1_factor and bl1_factor.

        Parameters
        ----------
        tiling : case tiling

        c_factor : [cout//nc//n0, howo//mc//mo]

        Returns
        -------
        al1_factor, bl1_factor

        """
        if len(tiling["AL1_shape"]) == 1:
            tiling["AL1_shape"] = tiling["AL1_shape"] + [1]

        if tiling["AL1_shape"]:
            al1_factor = [
                self._dim_map["img_shape"][1] // tiling["AL1_shape"][0],
                te_util.int_ceil_div(c_factor[1], tiling["AL1_shape"][1])
            ]
        else:
            al1_factor = [1, 1]

        if tiling["BL1_shape"]:
            if len(tiling["BL1_shape"]) > 1:
                if c_factor[0] % tiling["BL1_shape"][1] != 0:
                    raise RuntimeError("second value of BL1_shape should be"
                                       " factor of n block num")
                if tiling["BL1_shape"][
                        1] > 1 and tiling["BL1_shape"][1] % 2 != 0:
                    raise RuntimeError(
                        "second value of BL1_shape better to be even number")
            if len(tiling["BL1_shape"]) == 1:
                tiling["BL1_shape"] = tiling["BL1_shape"] + [1]
            bl1_factor = [
                (self._dim_map["img_shape"][1] + tiling["BL1_shape"][0] - 1) //
                tiling["BL1_shape"][0],
                (c_factor[0] + tiling["BL1_shape"][1] - 1) //
                tiling["BL1_shape"][1]
            ]
        else:
            bl1_factor = [1, tiling["block_dim"][1]]

        outer_factor = max(al1_factor[0], bl1_factor[0])
        inner_factor = min(al1_factor[0], bl1_factor[0])
        if outer_factor % inner_factor != 0:
            raise RuntimeError("illegal value of AL1_shape & BL1_shape")

        return al1_factor, bl1_factor

    def reorder_axis(self, tiling, al1_factor, bl1_factor, double_buffer_flag,
                     reorder_axis_dict, res_c):
        """
        reorder axis.

        Parameters
        ----------
        tiling : case tiling

        al1_factor : al1 split factor [c1//kAl1, howo//mc//m0//m_Al1]

        bl1_factor : bl1 split factor [c1//kBl1, howo//nc//n0//n_Bl1]

        double_buffer_flag : flag of double buffer

        reorder_axis_dict : axis to reorder

        res_c : c ddr

        Returns
        -------
        reorder flag

        """
        sch = self._schedule
        reorder_flag = False
        noi = reorder_axis_dict["noi"]
        m_outer_outer_outer_inner = reorder_axis_dict[
            "m_outer_outer_outer_inner"]
        c_outer_outer_outer_inner = reorder_axis_dict[
            "c_outer_outer_outer_inner"]

        if not tiling["BL1_shape"]:
            reorder_flag = True
        elif double_buffer_flag["AL1_pbuffer"] == double_buffer_flag[
                "BL1_pbuffer"]:
            if bl1_factor[1] >= al1_factor[1]:
                reorder_flag = True
        elif double_buffer_flag["BL1_pbuffer"] == 2:
            reorder_flag = True

        if reorder_flag:
            sch[res_c].reorder(m_outer_outer_outer_inner, noi,
                               c_outer_outer_outer_inner)
        else:
            sch[res_c].reorder(c_outer_outer_outer_inner,
                               m_outer_outer_outer_inner, noi)
        return reorder_flag

    def attach_bl0(self, tiling, stage_dict, bl0, coo, noo):
        """
        bl0 compute at.

        Parameters
        ----------
        tiling : case tiling

        stage_dict : c_col res_c

        bl0 : loc axis

        coo : loc axis

        noo : res axis

        Returns
        -------

        """
        sch = self._schedule
        res_c = stage_dict["res_c"]
        c_col = stage_dict["c_col"]
        if tiling["BL0_matrix"]:
            sch[bl0].compute_at(sch[c_col], coo)
        else:
            sch[bl0].compute_at(sch[res_c], noo)

        return True

    def al1_bl1_axis(self, stage_dict, al1_factor, bl1_factor, k_outer_outer):
        """
        splite al1 and bl1 k_axis.

        Parameters
        ----------
        stage_dict : c_col res_c

        al1_factor : al1 split factor [c1//kAl1, howo//mc//m0//m_Al1]

        bl1_factor : bl1 split factor [c1//kBl1, howo//nc//n0//n_Bl1]

        k_outer_outer : loc axis

        Returns
        -------
        al1_at_ccol_axis, bl1_at_ccol_axis, k_axis_dict

        """
        c_col = stage_dict["c_col"]
        sch = self._schedule
        if al1_factor[0] > bl1_factor[0]:
            k_outer_outer_outer, k_outer_outer_inner = sch[c_col].split(
                k_outer_outer, nparts=al1_factor[0])
            k_outer_outer_outer_outer, k_outer_outer_outer_inner = sch[
                c_col].split(k_outer_outer_outer, nparts=(bl1_factor[0]))
            al1_at_ccol_axis = k_outer_outer_outer_inner
            bl1_at_ccol_axis = k_outer_outer_outer_outer
        else:
            k_outer_outer_outer, k_outer_outer_inner = sch[c_col].split(
                k_outer_outer, nparts=bl1_factor[0])
            k_outer_outer_outer_outer, k_outer_outer_outer_inner = sch[
                c_col].split(k_outer_outer_outer, nparts=(al1_factor[0]))
            al1_at_ccol_axis = k_outer_outer_outer_outer
            bl1_at_ccol_axis = k_outer_outer_outer_inner

        k_axis_dict = {
            "k_outer_outer_outer_outer": k_outer_outer_outer_outer,
            "k_outer_outer_outer_inner": k_outer_outer_outer_inner,
            "k_outer_outer_inner": k_outer_outer_inner
        }

        return al1_at_ccol_axis, bl1_at_ccol_axis, k_axis_dict

    def get_nbuffer_al1_flag(self, tiling, compute_al1_axis, buffer_dict,
                             k_outer_outer_inner, k_outer_outer_inner_size,
                             shape_w):
        """
        get al1 nbuffer flag.

        Parameters
        ----------
        tiling : case tiling

        compute_al1_axis : al1 compute at axis

        buffer_dict : al1/bl1 al0/bl0 c_col c_ub

        k_outer_outer_inner : loc axis

        k_outer_outer_inner_size : k_outer_outer_inner size

        shape_w : weight shape

        Returns
        -------
        nbuffer_flag_al1, compute_al1_axis, nbuffer_axis

        """
        sch = self._schedule
        c_col = buffer_dict["c_col"]
        nbuffer_flag_al1 = False
        nbuffer_axis = {}
        if tiling["A_overhead_opt_flag"]:
            if (shape_w[-3] * shape_w[-2]) % tiling["AL0_matrix"][1] == 0:
                nbuffer_size = \
                    shape_w[-3] * shape_w[-2] // tiling["AL0_matrix"][1]
            else:
                nbuffer_size = shape_w[-3] * shape_w[-2]
            if int(k_outer_outer_inner_size % nbuffer_size\
                   ) == 0 and k_outer_outer_inner_size > nbuffer_size:
                k_outer_outer_inner_outer, k_outer_outer_inner_inner = sch[
                    c_col].split(k_outer_outer_inner, nbuffer_size)
                nbuffer_flag_al1 = True
                compute_al1_axis[
                    "k_outer_outer_inner_outer"] = k_outer_outer_inner_outer
                nbuffer_axis = {
                    "k_outer_outer_inner_outer": k_outer_outer_inner_outer,
                    "k_outer_outer_inner_inner": k_outer_outer_inner_inner
                }

        return nbuffer_flag_al1, compute_al1_axis, nbuffer_axis

    def cachebuffer(self, spec_node_list):
        """
        tensor not for conv set scope.

        Parameters
        ----------
        bodyops : body dict

        inputops : input dict

        spec_node_list : res tensor

        Returns
        -------

        """
        for lop in self.body_ops:
            if ("conv3d" not in lop["op"]) and \
                    lop['dst_buffer'] not in spec_node_list:
                self._schedule[lop["dst_buffer"]].set_scope(cce.scope_ubuf)
        for lop in self.input_ops:  # not for A, B, DeqScale, ReqScale,
            if "conv3d" in lop["op"]:
                continue
            if ("bias_tensor" in lop["dst_buffer"].name) and (
                    "bias" in self._tensor_map.keys()):
                continue
            tmp_read_map = []
            for nop in lop["next_op"]:
                tmp_read_map.append(nop["dst_buffer"])
            tmp_cache_buffer = self._schedule.cache_read(
                lop["dst_buffer"], cce.scope_ubuf, list(set(tmp_read_map)))
            lop["cache_buffer"] = tmp_cache_buffer

        return True

    def tiling_l0a_l0b(self, partial_ab, full_c, instr):
        """
        reduce factor.

        Parameters
        ----------
        partial_ab : tiling["AL0_matrix"] or tiling["BL0_matrix"]

        full_c : tiling["CL0_matrix"]

        instr: "A" or "B"

        Returns
        -------
        axis_factor, reduce factor
        """
        reduce_dim = [
            self._dim_map["fmap_matrix_dim"][-3],
            self._dim_map["fmap_matrix_dim"][-1]
        ]

        if instr == 'A':
            full_ab = [full_c[-3], reduce_dim[-2], full_c[-2], reduce_dim[-1]]
        elif instr == 'B':
            full_ab = [reduce_dim[-2], full_c[-4], full_c[-1], reduce_dim[-1]]

        partial_ab = list(partial_ab) if partial_ab else full_ab
        i_axis = 0
        for i_axis in range(len(partial_ab))[::-1]:
            if partial_ab[i_axis] != full_ab[i_axis]:
                break

        axis_factor = {}
        reduce_factor = {}
        red_axis = 0

        if instr == 'A':
            axis_map_a2c = {0: 1, 2: 2}
            axis_factor = {axis_map_a2c[0]: full_ab[0]}
            reduce_factor[0] = full_ab[1]
            for i in range(i_axis + 1):
                if i in [0, 2]:
                    axis_factor[axis_map_a2c[i]] = partial_ab[i]
                else:
                    reduce_factor[red_axis] = partial_ab[i]
                    red_axis += 1
        elif instr == 'B':
            axis_map_b2c = {1: 0, 2: 3}
            axis_factor = {axis_map_b2c[1]: full_ab[1]}
            reduce_factor[0] = full_ab[0]
            for i in range(i_axis + 1):
                reduce_factor[red_axis] = partial_ab[i]
                red_axis += 1
        axis_factor_for_batch = {}
        for i in axis_factor:
            axis_factor_for_batch[i + 1] = axis_factor[i]

        return {
            "axis_factor": axis_factor_for_batch,
            "reduce_factor": reduce_factor
        }

    def check_tiling(self, tiling, w_dtype):
        """
        default tiling check

        Returns
        -------
        true for auto tiling, false for default tiling
        """
        if tiling["AL0_matrix"][2] == VALID_TILING_NUM:
            return False

        l1_shape = ["AL1_shape", "BL1_shape"]
        for shape in l1_shape:
            if tiling[shape] and len(tiling[shape]) != TILING_L1_SHAPWE_DIM:
                raise RuntimeError("wrong tiling: %s dim must be %d" %
                                   (shape, TILING_L1_SHAPWE_DIM))

        matrix_list = [
            "AL0_matrix", "BL0_matrix", "CL0_matrix", "CUB_matrix", "block_dim"
        ]
        matrix_dim = [
            TILING_AL0_MATRIX_DIM, TILING_BL0_MATRIX_DIM,
            TILING_CL0_MATRIX_DIM, TILING_CUB_MATRIX_DIM, TILING_BLOCK_DIM_DIM
        ]
        for matrix, dim in zip(matrix_list, matrix_dim):
            if len(tiling[matrix]) not in dim:
                raise RuntimeError("wrong tiling: %s dim must be %d" %
                                   (matrix, dim[0]))

        matrix_cab = ["CL0_matrix", "AL0_matrix", "BL0_matrix"]
        for index0, index1 in zip(matrix_list[0:3], matrix_cab):
            if te_util.get_and_res(tiling[index0], tiling[index1]):
                if tiling[index0][0] != tiling[index1][1]:
                    raise RuntimeError(
                        "wrong tiling: tiling['%s'][0] must equal to"
                        " tiling['%s'][1]" % (index0, index1))

        if w_dtype != "float16":
            raise RuntimeError("wrong w_dtype")
        for matrix in matrix_list[0:3]:
            if tiling[matrix] != []:
                if tiling[matrix][2] != TILING_FLOAT16_MKN:
                    raise RuntimeError(
                        "wrong tiling: tiling['%s'][2] must be equal to %d"
                        " when w_dtype is float16" %
                        (matrix, TILING_FLOAT16_MKN))
                if tiling[matrix][3] != TILING_FLOAT16_MKN:
                    raise RuntimeError(
                        "wrong tiling: tiling['%s'][3] must be equal to %d"
                        " when w_dtype is float16" %
                        (matrix, TILING_FLOAT16_MKN))

        return True

    def tiling_fetch(self):
        """
        get tiling.

        Returns
        -------
        tiling
        """
        fmap_shape_ndc1hwc0 = Conv3DParam.tiling_query_param[
            "fmap_shape_ndc1hwc0"]
        shape_w_ndc1hwc0 = Conv3DParam.tiling_query_param["shape_w_ndc1hwc0"]
        in_dtype = Conv3DParam.tiling_query_param["in_dtype"]
        w_dtype = Conv3DParam.tiling_query_param["w_dtype"]
        padw = Conv3DParam.tiling_query_param["padw"]
        strideh = Conv3DParam.tiling_query_param["strideh"]
        stridew = Conv3DParam.tiling_query_param["stridew"]
        batch_size = fmap_shape_ndc1hwc0[0]
        in_size_w = fmap_shape_ndc1hwc0[-2]
        kernel_h = shape_w_ndc1hwc0[-3]
        kernel_w = shape_w_ndc1hwc0[-2]

        tiling_new = self._tensor_map["tiling_new"]
        tiling_ok_flag = self.check_tiling(tiling_new, w_dtype)

        tiling = {}
        tiling["AL0_matrix"] = tiling_new["AL0_matrix"][0:4]
        tiling["AL0_matrix"][
            1] = tiling["AL0_matrix"][1] * tiling_new["AL0_matrix"][-1]
        tiling["CL0_matrix"] = tiling_new["CL0_matrix"][0:4]
        tiling["CUB_matrix"] = tiling_new["CUB_matrix"][0:4]
        tiling["A_overhead_opt_flag"] = tiling_new["A_overhead_opt_flag"]
        tiling["B_overhead_opt_flag"] = tiling_new["B_overhead_opt_flag"]

        tiling["BL0_matrix"] = []
        if tiling_new["BL0_matrix"]:
            tiling["BL0_matrix"] = tiling_new["BL0_matrix"][0:4]
            tiling["BL0_matrix"][
                1] = tiling["BL0_matrix"][1] * tiling_new["BL0_matrix"][-1]

        tiling["manual_pingpong_buffer"] = tiling_new["manual_pingpong_buffer"]
        tiling["n_bef_batch_flag"] = tiling_new["n_bef_batch_flag"]

        tiling["AL1_shape"] = []
        if tiling_new["AL1_shape"]:
            tiling["AL1_shape"] = tiling_new["AL1_shape"][0:2]
            tiling["AL1_shape"][0] = int(
                tiling["AL1_shape"][0] * tiling_new["AL1_shape"][-1] /
                (kernel_h * kernel_w * CUBE_MKN[w_dtype]['mac'][1]))

        if te_util.get_or_res(tiling_new["BL1_shape"] == [],
                              tiling_new["BL1_shape"] is None):
            tiling["BL1_shape"] = tiling_new["BL1_shape"]
        else:
            tiling["BL1_shape"] = tiling_new["BL1_shape"][0:2]
            tiling["BL1_shape"][0] = int(
                tiling["BL1_shape"][0] * tiling_new["BL1_shape"][-1] /
                (kernel_h * kernel_w * CUBE_MKN[w_dtype]['mac'][1]))

        tiling["block_dim"] = tiling_new["block_dim"]
        tiling["block_dim"][
            0] = tiling["block_dim"][0] * tiling["block_dim"][-1]
        tiling["scale_drq_split_flag"] = False
        tiling["bias_split_flag"] = False

        if te_util.get_or_res(tiling_ok_flag is False,
            Conv3DParam.tiling_query_param["default_tiling"]):
            tiling = {}
            config = CUBE_MKN[w_dtype]
            ci0 = config['mac'][1]
            l1_buffer_size = get_soc_spec("L1_SIZE")
            m_bit_length = {
                "float32": 32,
                "float16": 16,
                "uint8": 8,
                "int8": 8,
                "uint4": 4,
                "int4": 4
            }
            m_bit_ratio = {
                "int32": 4,
                "float32": 4,
                "float16": 2,
                "uint8": 1,
                "int8": 1,
                "uint4": 1.0 / 2,
                "int4": 1.0 / 2
            }
            input_data_type = in_dtype
            w_out = (in_size_w + (padw[0] + padw[1] - kernel_w)) // stridew + 1

            for m_target in range(32, 0, -1):
                tmp1 = (
                    (m_target * m_bit_length['float16']) + w_out - 1) // w_out
                tmp2 = ((tmp1 * strideh) + kernel_h) * in_size_w
                max_feature_map = 1 * ci0 * tmp2 * \
                                    2 * m_bit_ratio[input_data_type]
                if max_feature_map < l1_buffer_size:
                    break

            tiling_m = m_target
            tiling_k = 1
            tiling_n = 2
            tiling["AL1_shape"] = [1]
            tiling["BL1_shape"] = None
            tiling["AL0_matrix"] = [tiling_m, tiling_k, 16, 16]
            tiling["BL0_matrix"] = [tiling_k, tiling_n, 16, 16]
            tiling["CL0_matrix"] = [tiling_n, tiling_m, 16, 16]
            tiling["CUB_matrix"] = [tiling_n, tiling_m, 16, 16]
            tiling["manual_pingpong_buffer"] = {
                'AL1_pbuffer': 1,
                'BL1_pbuffer': 1,
                'AL0_pbuffer': 1,
                'BL0_pbuffer': 1,
                'CL0_pbuffer': 1,
                'CUB_pbuffer': 1,
                'UBG_pbuffer': 1,
            }
            tiling["block_dim"] = [1, 1, 1]
            device_core_num = get_soc_spec("CORE_NUM")
            if te_util.get_and_res(batch_size > 1, device_core_num > 1):
                if batch_size <= device_core_num:
                    tiling["block_dim"][0] = batch_size
                else:
                    for i in range(device_core_num, 0, -1):
                        if batch_size % i == 0:
                            break
                    tiling["block_dim"][0] = i
            else:
                tiling["block_dim"][0] = 1
            tiling["scale_drq_split_flag"] = True
            tiling["bias_split_flag"] = True
            tiling["A_overhead_opt_flag"] = 0
            tiling["B_overhead_opt_flag"] = 0
            tiling["n_bef_batch_flag"] = 0

        return tiling

    def double_buffer(self, buffer_dict, double_buffer_flag):
        """
        double buffer.

        Parameters
        ----------
        buffer_dict : al1/bl1 al0/bl0 c_col c_ub

        double_buffer_flag : flag for double buffer

        Returns
        -------

        """
        sch = self._schedule
        cyclebuffer_flag = self._tensor_map["cyclebuffer_flag"]
        # al1
        if double_buffer_flag["AL1_pbuffer"] == 2:
            if cyclebuffer_flag:
                sch[buffer_dict["al1"]].cycle_double_buffer()
            else:
                sch[buffer_dict["al1"]].double_buffer()
        # bl1
        if double_buffer_flag["BL1_pbuffer"] == 2:
            sch[buffer_dict["bl1"]].double_buffer()
        # l0a
        if double_buffer_flag["AL0_pbuffer"] == 2:
            sch[buffer_dict["fmap_col"]].double_buffer()
        # l0b
        if double_buffer_flag["BL0_pbuffer"] == 2:
            sch[buffer_dict["bl0"]].double_buffer()
        # L0C
        if double_buffer_flag["CL0_pbuffer"] == 2:
            sch[buffer_dict["c_col"]].double_buffer()
        # CUB
        if double_buffer_flag["CUB_pbuffer"] == 2:
            sch[buffer_dict["c_ub"]].double_buffer()

    def intrin_mapping(self, fmap, mad_dict, buffer_dict, new_fmap_col_axis,
                       tiling, cn_axis, l0a_load2d_flag):
        """
        intrin_mapping.

        Parameters
        ----------
        famp : input tensor

        mad_dict : for mad pragma

        buffer_dict : al1/bl1 al0/bl0 c_col c_ub

        new_fmap_col_axis : fmap_col axis

        tiling : case tiling

        cn_axis : loc axis

        l0a_load2d_flag : true or false

        Returns
        -------

        """
        sch = self._schedule
        al1 = buffer_dict["al1"]
        bl1 = buffer_dict["bl1"]
        fmap_col = buffer_dict["fmap_col"]
        bl0 = buffer_dict["bl0"]
        c_col = buffer_dict["c_col"]
        c_ub = buffer_dict["c_ub"]

        setfmatrix_dict = {
            "conv_kernel_h": c_ub.op.attrs['kernel_h'],
            "conv_kernel_w": c_ub.op.attrs['kernel_w'],
            "conv_padding_top": c_ub.op.attrs['padding'][0],
            "conv_padding_bottom": c_ub.op.attrs['padding'][1],
            "conv_padding_left": c_ub.op.attrs['padding'][2],
            "conv_padding_right": c_ub.op.attrs['padding'][3],
            "conv_stride_h": c_ub.op.attrs['stride'][0],
            "conv_stride_w": c_ub.op.attrs['stride'][1],
        }

        setfmatrix_dict["conv_fm_c"] = fmap.op.shape[2] * fmap.op.shape[5]
        setfmatrix_dict["conv_fm_h"] = fmap.op.shape[3]
        setfmatrix_dict["conv_fm_w"] = fmap.op.shape[4]
        cyclebuffer_flag = self._tensor_map["cyclebuffer_flag"]
        if cyclebuffer_flag:
            sch[al1].emit_insn(al1.op.axis[1], 'dma_copy')
        else:
            sch[al1].emit_insn(al1.op.axis[0], 'dma_copy')

        if l0a_load2d_flag:
            sch[fmap_col].emit_insn(new_fmap_col_axis[3], 'dma_copy')
        else:
            if self._tensor_map["opti_h_flag"]:
                setfmatrix_dict["conv_stride_h"] = 1
            fmap_col_before = buffer_dict["fmap_col_before"]
            sch[fmap_col_before].emit_insn(fmap_col_before.op.axis[0],
                                           'set_fmatrix', setfmatrix_dict)
            sch[fmap_col].emit_insn(new_fmap_col_axis[-5], 'im2col')

        if tiling["BL1_shape"] is not None:
            sch[bl1].emit_insn(bl1.op.axis[0], 'dma_copy')
        sch[bl0].emit_insn(bl0.op.axis[0], 'dma_copy')
        sch[c_ub].emit_insn(c_ub.op.axis[0], 'dma_copy')
        sch[c_col].emit_insn(cn_axis, 'mad', mad_dict)

    def attach_at(self, bodyops, inputops, compute_at_buffer, compute_at_axis,
                  tiling):
        """
        tensor not for conv compute at.

        Parameters
        ----------
        bodyops : body dict

        inputops : input dict

        compute_at_buffer : col res_c

        compute_at_axis : axis for compute at

        tiling : case tiling

        Returns
        -------

        """
        for lop in bodyops:
            if "conv3d" not in lop["op"] or "convolution_A" in lop["op"]:
                if lop["op"] == "conv_vector_remove_pad":
                    continue
                if "Before" in lop["op"]:
                    self._schedule[lop["dst_buffer"]].compute_at(
                        self._schedule[compute_at_buffer[2]],
                        compute_at_axis[2])
                else:
                    self._schedule[lop["dst_buffer"]].compute_at(
                        self._schedule[compute_at_buffer[1]],
                        compute_at_axis[1])
                    self._schedule[lop["dst_buffer"]].buffer_align(
                        (1, 1), (1, 1),
                        (1, tiling["CL0_matrix"][1] * tiling["CL0_matrix"][2]),
                        (1, 1))

        for lop in inputops:
            if "conv3d" in lop["op"]:
                continue
            if ("bias_tensor" in lop["op"]) and (
                    "bias" in self._tensor_map.keys()):
                continue
            if "Before" in lop["op"]:
                self._schedule[lop["cache_buffer"]].compute_at(
                    self._schedule[compute_at_buffer[2]], compute_at_axis[2])
            else:
                self._schedule[lop["cache_buffer"]].compute_at(
                    self._schedule[compute_at_buffer[0]], compute_at_axis[0])

    def to_pragma(self, bodyops, inputops, c_outer_inner_inner):
        """
        tensor not for conv to pragma.

        Parameters
        ----------
        bodyops : body dict

        inputops : input dict

        fmap : input tensor

        c_ub : conv res in ub

        c_outer_inner_inner : res axis

        Returns
        -------

        """
        for lop in bodyops:
            if "conv3d" not in lop["op"] or "conv3d_A" in lop["op"]:
                lop["tensorize_axis"] = self._schedule[
                    lop["dst_buffer"]].op.axis[0]
                if "Before" in lop["op"]:
                    lop["op"] = lop["op"].replace("_Before", "")
                if "_conv3d_A" in lop["op"]:
                    lop["op"] = lop["op"].replace("_conv3d_A", "")
                self.__pragma_for_op(lop, c_outer_inner_inner)

        for lop in inputops:
            if "conv3d" in lop["op"]:
                continue
            if ("bias_tensor" in lop["op"]) and (
                    "bias" in self._tensor_map.keys()):
                continue
            self._schedule[lop["cache_buffer"]].emit_insn(
                lop["cache_buffer"].op.axis[0], 'dma_copy')

    def set_al1_at_axis(self, l0a_load2d_flag, nbuffer_flag_al1, reorder_flag,
                        tiling, al1_factor, compute_axis, run_once_axis,
                        allocate_axis, index_axis, buffer_dict, stage):
        """
        al1 compute_at.

        Parameters
        ----------
        l0a_load2d_flag : true or false

        nbuffer_flag_al1 : true or false

        reorder_flag : true or false

        tiling : case tiling

        al1_factor : al1 split factor [c1//kAl1, howo//mc//m0//m_Al1]

        compute_axis : al1 axis to compute at

        run_once_axis : al1 axis to run once

        allocate_axis : al1 axis to allocate at

        index_axis : al1 index to stage

        buffer_dict : al1/bl1 al0/bl0 c_col c_ub

        stage : c_col res_c

        Returns
        -------

        """
        index = int(al1_factor[0] == 1) if tiling["AL1_shape"] else 2
        cyclebuffer_flag = self._tensor_map["cyclebuffer_flag"]
        sch = self._schedule
        al1_allocate_axis = None
        al1_run_once_axis = []
        compute_stage = None
        allocate_stage = None
        al1 = buffer_dict["al1"]
        if not l0a_load2d_flag:
            fmap_col_before = buffer_dict["fmap_col_before"]
        if te_util.get_and_res(l0a_load2d_flag, nbuffer_flag_al1):
            al1_compute_axis = compute_axis["k_outer_outer_inner_outer"]
            compute_stage = stage[0]
            al1_allocate_axis = allocate_axis[index_axis[index]]
            allocate_stage = stage[index]
            run_flag = te_util.get_and_res(index == 1, reorder_flag)
            if te_util.get_or_res(run_flag, index == 2):
                al1_run_once_axis = [
                    run_once_axis["c_outer_outer_inner"],
                    run_once_axis["c_outer_outer_outer_inner"]
                ]
        elif nbuffer_flag_al1:
            if index == 0:
                al1_compute_axis = compute_axis["k_outer_outer_inner_outer"]
                compute_stage = stage[0]
                al1_allocate_axis = allocate_axis[index_axis[0]]
                allocate_stage = stage[0]
            else:
                al1_compute_axis = compute_axis[index_axis[index]]
                compute_stage = stage[index]
        else:
            al1_compute_axis = compute_axis[index_axis[index]]
            compute_stage = stage[index]

        if l0a_load2d_flag:
            sch[al1].compute_at(sch[compute_stage], al1_compute_axis)
        else:
            sch[al1].compute_at(sch[compute_stage], al1_compute_axis)
            sch[fmap_col_before].compute_at(sch[compute_stage],
                                            al1_compute_axis)
        if cyclebuffer_flag:
            cyclebuffer_factor = self._tensor_map["d_out"] // \
                        self._tensor_map["d_dim"]
            expr = tvm.select(tvm.convert(cyclebuffer_factor) == 1,
                al1.op.axis[0].var, \
                tvm.floormod(al1.op.axis[0].var, cyclebuffer_factor))
            sch[al1].pragma(al1.op.axis[0],
                            "cyclebuffer",
                            (expr == 0).asnode())

        if al1_run_once_axis:
            sch[al1].allocate_at(sch[allocate_stage],
                                 al1_allocate_axis,
                                 run_once_axes=al1_run_once_axis)
        elif al1_allocate_axis is not None:
            sch[al1].allocate_at(sch[allocate_stage], al1_allocate_axis)

        return True

    def set_bl1_at_axis(self, reorder_flag, tiling, bl1_factor,
                        compute_bl1_axis, run_once_bl1_axis, allocate_bl1_axis,
                        bl1_index_dict, buffer_dict, stage):
        """
        bl1 compute_at.

        Parameters
        ----------
        reorder_flag : true or false

        tiling : case tiling

        bl1_factor : bl1 split factor [c1//kBl1, cout//nc//n0//m_Bl1]

        compute_bl1_axis : bl1 axis to compute at

        run_once_bl1_axis : bl1 axis to run once

        allocate_bl1_axis : bl1 axis to allocate at

        bl1_index_axis : bl1 index to stage

        buffer_dict : al1/bl1 al0/bl0 c_col c_ub

        stage : c_col res_c

        Returns
        -------

        """
        index = 2 if (tiling["BL1_shape"] == []
                      or tiling["BL1_shape"] is None) else int(
                          bl1_factor[0] == 1)
        sch = self._schedule
        bl1_compute_axis = None
        bl1_allocate_axis = None
        bl1_run_once_axis = []
        bl1_compute_stage = None
        bl1_allocate_stage = None
        bl1 = buffer_dict["bl1"]
        if tiling["B_overhead_opt_flag"]:
            if index == 0 or (index == 1 and reorder_flag):
                bl1_compute_axis = compute_bl1_axis["coo"]
                bl1_compute_stage = stage[0]
                bl1_allocate_axis = allocate_bl1_axis[bl1_index_dict[index]]
                bl1_allocate_stage = stage[index]
                if index == 1 and reorder_flag:
                    bl1_run_once_axis = [
                        run_once_bl1_axis["m_outer_outer_inner"]
                    ]
            else:
                bl1_compute_axis = compute_bl1_axis[bl1_index_dict[index]]
                bl1_compute_stage = stage[index]
        else:
            bl1_compute_axis = compute_bl1_axis[bl1_index_dict[index]]
            bl1_compute_stage = stage[index]

        sch[bl1].compute_at(sch[bl1_compute_stage], bl1_compute_axis)
        if bl1_run_once_axis:
            sch[bl1].allocate_at(sch[bl1_allocate_stage],
                                 bl1_allocate_axis,
                                 run_once_axes=bl1_run_once_axis)
        elif bl1_allocate_axis is not None:
            sch[bl1].allocate_at(sch[bl1_allocate_stage], bl1_allocate_axis)

        return True

    def do_schedule(self, res, spec_node_list, sch_list):
        """
        auto_schedule for cce AI-CORE.
        For now, only one convolution operation is supported.

        Parameters
        ----------
        res : tvm.tensor

        spec_node_list : same as other template in cce_schedule

        sch_list: use sch_list[0] to return conv schedule

        Returns
        -------
        True for sucess, False for no schedule
        """
        cce_params.jump_expand_flag = True

        tensor_map = self._tensor_map
        dim_map = self._dim_map

        c_ub = tensor_map["c_ub"]

        sch = sch_list[0]
        self._schedule = sch

        color_op = AutoScheduleOp(res)
        self.body_ops = color_op.body_ops
        self.input_ops = color_op.input_ops
        self.output_ops = color_op.output_ops

        self._res_tensor = res
        res_c = self._res_tensor
        self.cachebuffer(spec_node_list)

        l0a_load2d_flag = tensor_map["l0a_load2d_flag"]

        fmap = tensor_map["fmap"]
        weight = tensor_map["filter"]
        c_col = tensor_map["c_col"]
        stage_dict = {"res_c": res_c, "c_col": c_col}

        config = CUBE_MKN[weight.dtype]

        pad_right = c_ub.op.attrs['padding'][2]
        pad_left = c_ub.op.attrs['padding'][3]
        kernel_w = c_ub.op.attrs['kernel_w']

        fmap_w = fmap.shape[-2] if fmap.op.input_tensors else fmap.op.shape[-2]

        stride_w = c_ub.op.attrs['stride'][1]
        w_out = (fmap_w + pad_left + pad_right - kernel_w) // stride_w + 1

        if l0a_load2d_flag:
            al1 = tensor_map["al1_load2d"]
            al0 = tensor_map["al0_load2d"]
            sch[al1].storage_align(sch[al1].op.axis[1], 256, 0)
            fmap_col = al0
            sch[al1].set_scope(cce.scope_cbuf)
        else:
            fuse_fmap_tensor = tensor_map["fmap_do_tensor"]
            fmap_col_before = tensor_map["fmap_im2col_row_major_res"]
            fmap_col = tensor_map["fmap_im2col_fractal_res"]
            sch[fmap_col_before].buffer_align(
                (1, 1), (w_out, w_out), (1, 1), (1, 1), (1, 1),
                (1, CUBE_MKN[fmap_col_before.dtype]["mac"][1]))
            al1 = fuse_fmap_tensor
            sch[fuse_fmap_tensor].set_scope(cce.scope_cbuf)
            sch[fmap_col_before].set_scope(cce.scope_cbuf)

        sch[c_ub].buffer_align((1, 1), (1, 1),
                               (1, CUBE_MKN[c_ub.dtype]["mac"][0]),
                               (1, CUBE_MKN[c_ub.dtype]["mac"][2]))

        tiling = self.tiling_fetch()

        filter_matrix = list(dim_map["filter_matrix_dim"])
        filter_matrix[1] = filter_matrix[1] // tiling["block_dim"][1]

        bl1 = self.weight_to_bl1(tiling, filter_matrix, weight, c_col)
        bl0 = sch.cache_read(bl1, cce.scope_cb, [c_col])

        sch[c_col].set_scope(cce.scope_cc)
        sch[c_ub].set_scope(cce.scope_ubuf)

        compute_at_buffer = []
        compute_at_axis = []

        sch[fmap_col].set_scope(cce.scope_ca)

        factor_m = tiling["AL0_matrix"][0]
        factor_k = tiling["AL0_matrix"][1]

        a1_axis, a3_axis = sch[fmap_col].split(sch[fmap_col].op.axis[1],
                                               factor_m)
        a2_axis, a4_axis = sch[fmap_col].split(sch[fmap_col].op.axis[2],
                                               factor_k)
        # split N begin
        fmap_col_no, fmap_col_ni = sch[fmap_col].split(
            sch[fmap_col].op.axis[0], 1)
        sch[fmap_col].reorder(fmap_col_no, a1_axis, a2_axis, fmap_col_ni,
                              a3_axis, a4_axis, sch[fmap_col].op.axis[3],
                              sch[fmap_col].op.axis[4])
        new_fmap_col_axis = [
            fmap_col_no, a1_axis, a2_axis, fmap_col_ni, a3_axis, a4_axis,
            sch[fmap_col].op.axis[3], sch[fmap_col].op.axis[4]
        ]

        new_c_col_axis = [
            sch[c_col].op.axis[0], sch[c_col].op.axis[1],
            sch[c_col].op.axis[2], sch[c_col].op.axis[3]
        ]

        _, _, _, nn_axis = new_c_col_axis

        c_tiling_factor = [
            tiling["CL0_matrix"][0],
            tiling["CL0_matrix"][1] * tiling["CL0_matrix"][2]
        ]

        c_factor = [
            te_util.int_ceil_div(dim_map["out_img_shape"][1],
                                 c_tiling_factor[0]),
            te_util.int_ceil_div(dim_map["out_img_shape"][-2],
                                 c_tiling_factor[1])
        ]

        c_ub_tiling_factor = tiling["CUB_matrix"]
        c_ub_factor = [
            te_util.int_ceil_div(c_tiling_factor[0], c_ub_tiling_factor[0]),
            te_util.int_ceil_div(c_tiling_factor[1],
                                 c_ub_tiling_factor[1] * c_ub_tiling_factor[2])
        ]

        al1_factor, bl1_factor = self.factor_al1_bl1(tiling, c_factor)

        al0_axis_factor = self.tiling_l0a_l0b(tiling["AL0_matrix"],
                                              tiling["CL0_matrix"], 'A')

        bl0_axis_factor = self.tiling_l0a_l0b(tiling["BL0_matrix"],
                                              tiling["CL0_matrix"], 'B')

        # --------------------------double buffer------------------------
        double_buffer_flag = {
            'AL1_pbuffer': False,
            'BL1_pbuffer': False,
            'AL0_pbuffer': False,
            'BL0_pbuffer': False,
            'CL0_pbuffer': False,
            'CUB_pbuffer': False,
            'UBG_pbuffer': False,
        }

        double_buffer_flag = tiling["manual_pingpong_buffer"]

        #--------------------------tile res_c------------------------
        c_outer_outer, c_outer_inner = sch[res_c].split(
            res_c.op.axis[1], (res_c.shape[1].value // c_factor[0]))
        m_outer_outer, m_outer_inner = sch[res_c].split(
            res_c.op.axis[2], c_tiling_factor[1])
        sch[res_c].reorder(c_outer_outer, m_outer_outer, c_outer_inner,
                           m_outer_inner)
        m_outer_outer_outer, m_outer_outer_inner = sch[res_c].split(
            m_outer_outer, nparts=al1_factor[1])
        c_outer_outer_outer, c_outer_outer_inner = sch[res_c].split(
            c_outer_outer, nparts=bl1_factor[1])
        c_slice_axis = m_outer_outer_inner

        block_dim = tiling["block_dim"] if "block_dim" in tiling else [1, 1, 1]

        # split batch of res_c
        batch_outer, batch_inner = sch[res_c].split(res_c.op.axis[0],
                                                    nparts=int(block_dim[0]))
        # split cout of res_c
        c_outer_outer_outer_outer, c_outer_outer_outer_inner = sch[
            res_c].split(c_outer_outer_outer, nparts=block_dim[1])
        bl1_at_c_axis = c_outer_outer_outer_inner

        m_outer_outer_outer_outer, m_outer_outer_outer_inner = sch[
            res_c].split(m_outer_outer_outer, nparts=block_dim[2])
        al1_at_c_axis = m_outer_outer_outer_inner
        if tensor_map["cyclebuffer_flag"]:
            batch_inner_outer, batch_inner_inner = sch[res_c].split(
                batch_inner, nparts=1)
            al1_at_c_axis = batch_inner_inner
            sch[res_c].reorder(batch_outer, c_outer_outer_outer_outer,
                               m_outer_outer_outer_outer, batch_inner_outer,
                               c_outer_outer_outer_inner,
                               m_outer_outer_outer_inner, batch_inner_inner)
            cycbuf_axis = batch_inner_outer
        else:
            sch[res_c].reorder(batch_outer, c_outer_outer_outer_outer,
                               m_outer_outer_outer_outer, batch_inner,
                               c_outer_outer_outer_inner,
                               m_outer_outer_outer_inner)
            cycbuf_axis = batch_inner
        mc_flag = False
        blocks = block_dim[0] * block_dim[1] * block_dim[2]

        if blocks != 1:
            batch_cout_fused = sch[res_c].fuse(batch_outer,
                                               c_outer_outer_outer_outer,
                                               m_outer_outer_outer_outer)
            noo_true, _ = sch[res_c].split(batch_cout_fused, nparts=blocks)
            bido, _ = sch[res_c].split(noo_true, 1)
            block = tvm.thread_axis("blockIdx.x")
            sch[res_c].bind(bido, block)
            mc_flag = True
        else:
            noo_true = cycbuf_axis
            block = 1

        noi_tree = cycbuf_axis

        noo, noi = sch[res_c].split(noi_tree, factor=1)

        if not mc_flag:
            bido = noo
        reorder_axis_dict = {
            "m_outer_outer_outer_inner": m_outer_outer_outer_inner,
            "c_outer_outer_outer_inner": c_outer_outer_outer_inner,
            "noi": noi
        }

        reorder_flag = self.reorder_axis(tiling, al1_factor, bl1_factor,
                                         double_buffer_flag, reorder_axis_dict,
                                         res_c)
        m_outer_inner_outer, m_outer_inner_inner = sch[res_c].split(
            m_outer_inner, nparts=1)

        if te_util.get_and_res(tiling["n_bef_batch_flag"], not reorder_flag):
            sch[res_c].reorder(c_outer_outer_outer_inner, noo)

        # ============ tile CUB ========================
        c_outer_inner_outer, c_outer_inner_inner = sch[res_c].split(
            c_outer_inner, nparts=c_ub_factor[0])

        sch[res_c].reorder(c_outer_inner_outer, m_outer_inner_outer,
                           c_outer_inner_inner, m_outer_inner_inner)
        sch[c_ub].compute_at(sch[res_c], m_outer_inner_outer)

        # ============ tile c_col =======================
        compute_at_buffer.append(res_c)
        compute_at_axis.append(c_slice_axis)
        compute_at_buffer.append(res_c)
        compute_at_axis.append(m_outer_inner_outer)

        sch[c_col].compute_at(sch[res_c], c_slice_axis)

        _, reduce_kk = sch[c_col].op.reduce_axis

        axis_factor = list(al0_axis_factor["axis_factor"].items())
        boo, boi = sch[c_col].split(new_c_col_axis[axis_factor[0][0]],
                                    axis_factor[0][1] * config["mac"][0])

        axis_factor = list(bl0_axis_factor["axis_factor"].items())
        coo, coi = sch[c_col].split(new_c_col_axis[axis_factor[0][0]],
                                    axis_factor[0][1])

        # for reduce axis, al0 and bl0 should be the same
        reduce_axis_factor = list(al0_axis_factor["reduce_factor"].items())

        # k_outer_outer should be no less than kd
        k_outer_outer, k_outer_inner = sch[c_col].split(
            c_col.op.reduce_axis[reduce_axis_factor[0][0]],
            reduce_axis_factor[0][1])
        k_outer_outer_size = c_col.op.reduce_axis[
            reduce_axis_factor[0][0]].dom.extent // reduce_axis_factor[0][1]

        # split N begin
        _, cn_axis = sch[c_col].split(c_col.op.axis[0], 1)
        sch[c_col].reorder(k_outer_outer, coo, boo, cn_axis, coi, boi, nn_axis,
                           k_outer_inner, reduce_kk)
        sch[fmap_col].compute_at(sch[c_col], boo)
        self.attach_bl0(tiling, stage_dict, bl0, coo, noo)

        #  ============ al1 and bl1 slice can be different with CUB & CL0 =====
        al1_at_ccol_axis, bl1_at_ccol_axis, k_axis_dict = self.al1_bl1_axis(
            stage_dict, al1_factor, bl1_factor, k_outer_outer)

        k_outer_outer_outer_outer = k_axis_dict["k_outer_outer_outer_outer"]
        k_outer_outer_outer_inner = k_axis_dict["k_outer_outer_outer_inner"]
        k_outer_outer_inner = k_axis_dict["k_outer_outer_inner"]

        buffer_dict = {
            "al1": al1,
            "bl1": bl1,
            "fmap_col": fmap_col,
            "bl0": bl0,
            "c_col": c_col,
            "c_ub": c_ub,
            "res_c": res_c
        }
        if not l0a_load2d_flag:
            buffer_dict["fmap_col_before"] = fmap_col_before
        # al1 compute_at
        compute_al1_axis = {
            "al1_at_ccol_axis": al1_at_ccol_axis,
            "al1_at_c_axis": al1_at_c_axis,
            "noo": noo
        }
        shape_w = Conv3DParam.tiling_query_param["shape_w_ndc1hwc0"]
        k_outer_outer_inner_size = int(k_outer_outer_size //
                                       max(al1_factor[0], bl1_factor[0]))
        nbuffer_flag_al1, compute_al1_axis, _ = self.get_nbuffer_al1_flag(
            tiling, compute_al1_axis, buffer_dict, k_outer_outer_inner,
            k_outer_outer_inner_size, shape_w)
        run_once_al1_axis = {
            "c_outer_outer_inner": c_outer_outer_inner,
            "c_outer_outer_outer_inner": c_outer_outer_outer_inner
        }
        allocate_al1_axis = {
            "al1_at_ccol_axis": al1_at_ccol_axis,
            "al1_at_c_axis": al1_at_c_axis,
            "noo": noo
        }
        index_al1_dict = {0: "al1_at_ccol_axis", 1: "al1_at_c_axis", 2: "noo"}
        stage = {0: c_col, 1: res_c, 2: res_c}
        self.set_al1_at_axis(l0a_load2d_flag, nbuffer_flag_al1, reorder_flag,
                             tiling, al1_factor, compute_al1_axis,
                             run_once_al1_axis, allocate_al1_axis,
                             index_al1_dict, buffer_dict, stage)

        # bl1 compute_at
        compute_bl1_axis = {
            "coo": coo,
            "bl1_at_ccol_axis": bl1_at_ccol_axis,
            "bl1_at_c_axis": bl1_at_c_axis,
            "bido": bido
        }
        allocate_bl1_axis = {
            "bl1_at_ccol_axis": bl1_at_ccol_axis,
            "bl1_at_c_axis": bl1_at_c_axis,
            "bido": bido
        }
        run_once_bl1_axis = {"m_outer_outer_inner": m_outer_outer_inner}
        bl1_index_dict = {0: "bl1_at_ccol_axis", 1: "bl1_at_c_axis", 2: "bido"}
        self.set_bl1_at_axis(reorder_flag, tiling, bl1_factor,
                             compute_bl1_axis, run_once_bl1_axis,
                             allocate_bl1_axis, bl1_index_dict, buffer_dict,
                             stage)
        ############################ double buffer ###########################
        self.double_buffer(buffer_dict, double_buffer_flag)
        ############################ intrin mapping ###########################
        stride_d = c_col.op.attrs['stride_d']
        pad_head = c_col.op.attrs['pad_head']
        fmap_d = c_col.op.attrs['fmap_d']
        d_out = c_col.op.attrs['d_out']
        w_h = shape_w[-3]
        w_w = shape_w[-2]
        batch_axis = tvm.floordiv(
            block, block_dim[1] *
            block_dim[2]) * fmap_col.shape[0].value // block_dim[0] + noo
        if tensor_map["cyclebuffer_flag"]:
            batch_axis = tvm.floordiv(
                block, block_dim[1] * block_dim[2]) * fmap_col.shape[
                    0].value // block_dim[0] + noo + batch_inner_inner

        outer_factor = max(al1_factor[0], bl1_factor[0])
        inner_factor = min(al1_factor[0], bl1_factor[0])
        mad_dict = {
            "mad_pattern":
            2,
            "k_outer": [
                k_outer_outer_outer_outer, k_outer_outer_outer_inner,
                k_outer_outer_inner
            ],
            "k_coeff":
            tvm.all(
                (batch_axis % d_out * stride_d +
                 (((k_outer_outer_outer_outer *
                    (outer_factor // inner_factor) + k_outer_outer_outer_inner)
                   * k_outer_outer_inner_size + k_outer_outer_inner) *
                  reduce_axis_factor[0][1] //
                  (w_h * w_w)) // fmap.op.shape[2] >= pad_head),
                (batch_axis % d_out * stride_d +
                 (((k_outer_outer_outer_outer *
                    (outer_factor // inner_factor) + k_outer_outer_outer_inner)
                   * k_outer_outer_inner_size + k_outer_outer_inner) *
                  reduce_axis_factor[0][1] //
                  (w_h * w_w)) // fmap.op.shape[2] < fmap_d + pad_head)),
            "k_cond":
            tvm.any(
                tvm.all(
                    (batch_axis % d_out * stride_d +
                     (((k_outer_outer_outer_outer *
                        (outer_factor // inner_factor) +
                        k_outer_outer_outer_inner) * k_outer_outer_inner_size +
                       k_outer_outer_inner) * reduce_axis_factor[0][1] //
                      (w_h * w_w)) // fmap.op.shape[2] == pad_head),
                    (((k_outer_outer_outer_outer *
                       (outer_factor // inner_factor) +
                       k_outer_outer_outer_inner) * k_outer_outer_inner_size +
                      k_outer_outer_inner) * reduce_axis_factor[0][1] %
                     (w_h * w_w * fmap.op.shape[2]) <= 0)),
                ((k_outer_outer_outer_outer *
                  (outer_factor // inner_factor) + k_outer_outer_outer_inner) *
                 k_outer_outer_inner_size + k_outer_outer_inner) == 0),
        }
        self.intrin_mapping(fmap, mad_dict, buffer_dict, new_fmap_col_axis,
                            tiling, cn_axis, l0a_load2d_flag)
        ########################### cube schedule end #########################
        self.attach_at(self.body_ops, self.input_ops, compute_at_buffer,
                       compute_at_axis, tiling)
        self.to_pragma(self.body_ops, self.input_ops, c_outer_inner_inner)

        tensor_map.clear()
        dim_map.clear()
        tiling.clear()
        return True

    def __pragma_for_op(self, lop, c_outer_inner_inner=None):
        # for not in conv op pragma
        op_cmd = lop["op"].split("_")
        cache_buffer = lop["dst_buffer"]
        tensorize_axis = lop["tensorize_axis"]
        if op_cmd[0].lower() == "elewise":
            self._schedule[cache_buffer].emit_insn(tensorize_axis, lop["op"])
        elif lop["op"] == 'convolution_C':
            self._schedule[cache_buffer].emit_insn(
                self._schedule[cache_buffer].op.axis[0], 'dma_copy')
        elif lop["op"] == 'conv_vector_remove_pad':
            self._schedule[cache_buffer].emit_insn(c_outer_inner_inner,
                                                   'dma_copy')
        elif lop["op"] == 'conv_vector_bias_add':
            self._schedule[cache_buffer].emit_insn(tensorize_axis,
                                                   "vector_add")
        else:
            pass


class AutoScheduleDict(dict):
    """
    class of AutoScheduleDict
    """
    def __init__(self, **kwargs):
        super(AutoScheduleDict, self).__init__(**kwargs)
        self.read_only = False


class AutoScheduleOp:
    """
    class of AutoScheduleOp
    """
    def __init__(self, *init_args):
        if len(init_args) == 1 and isinstance(init_args[0], tvm.tensor.Tensor):
            res_tensor = init_args[0]
            self._color_count = 0
            self._op = []
            self.body_ops = []
            self.input_ops = []
            self.output_ops = []
            self._res_tensor = res_tensor
            self._before_conv_flag = False
            self.__scrapy_tensor_graph(self._res_tensor)
            self.__connect_op()
            self._end_op = self.get_op_by_tensor(self._res_tensor)
            self._end_op["color"] = self._color_count
            self.__init_color(self._end_op)
            self.__analyse_input_output()

    def __split_tensor(self, tensor):
        tmp_op = AutoScheduleDict()
        operator = tensor.op
        if hasattr(operator, "tag"):
            if operator.tag == "":
                tmp_op["op"] = operator.name
            else:
                tmp_op["op"] = operator.tag
        if tmp_op["op"].find("|") != -1:
            str_list = operator.tag.split("|")
            tmp_op["op"] = str_list[0]
        if hasattr(tensor, "tag"):
            tmp_op["op"] = tmp_op["op"] + "_" + tensor.tag
        tmp_op["dst_buffer"] = tensor
        tmp_op["src_buffer"] = list(operator.input_tensors)

        if "conv3d_A" in tmp_op["op"]:
            self._before_conv_flag = True
        if self._before_conv_flag:
            tmp_op["op"] = tmp_op["op"] + "_Before"

        return tmp_op

    def __scrapy_tensor_graph(self, res_tensor):
        operation_list = [res_tensor]
        while operation_list:
            tmp_operation_list = []
            for operation in operation_list:
                tmp_op = self.__split_tensor(operation)
                self._op.append(tmp_op)
                for i in tmp_op["src_buffer"]:
                    i.next = operation
                    operation.prev = i
                    tmp_operation_list.append(i)
                    if tmp_op["op"] == "conv3d_c_col":
                        i.tag = "conv3d_Input"
                    if tmp_op["op"] == "conv3d_fuse_fmap_tensor":
                        i.tag = "conv3d_A"
                    if tmp_op["op"] == "conv3d_al1_load2d":
                        i.tag = "conv3d_A"
                    if tmp_op["op"] == "conv3d_bias_l0c":
                        i.tag = "conv3d_bias_tensor"
            operation_list = list(set(tmp_operation_list))

    def __connect_op(self):
        for lop in self._op:
            lop["prev_op"] = []
            lop["next_op"] = []

        for lop in self._op:
            for src_tensor in lop["src_buffer"]:
                tmp_op = self.get_op_by_tensor(src_tensor)
                lop["prev_op"].append(tmp_op)
                tmp_op["next_op"].append(lop)

    def __init_color(self, start_op):
        for p_op in start_op["prev_op"]:
            p_op["color"] = start_op["color"]
            self.__init_color(p_op)

    def get_op_by_tensor(self, tensor):
        """
        get op by tensor

        Parameters
        ----------
        tensor: the source tensor

        Returns
        -------
        tensor : op
        """
        for i in self._op:
            if i["dst_buffer"].same_as(tensor):
                return i
        return None

    def __analyse_input_output(self):
        input_ops = []
        output_ops = []
        body_ops = []
        input_tensor_name = []
        body_tensor_name = []
        for lop in self._op:
            if not lop["prev_op"]:
                lop["color"] = -1
                if lop["dst_buffer"].name not in input_tensor_name:
                    input_ops.append(lop)
                    input_tensor_name.append(lop["dst_buffer"].name)
                else:
                    continue
            else:
                if lop["dst_buffer"].name not in body_tensor_name:
                    body_ops.append(lop)
                    body_tensor_name.append(lop["dst_buffer"].name)
                else:
                    continue
                if not lop["next_op"]:
                    output_ops.append(lop)

        for i in input_ops:
            i["color"] = -1
        self.input_ops = input_ops
        self.output_ops = output_ops
        self.body_ops = body_ops
