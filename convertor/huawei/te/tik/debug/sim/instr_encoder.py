"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     instr_encoder.py
DESC:     encode instruction
CREATED:  2019-7-04 20:12:13
MODIFIED: 2019-7-24 15:09:45
"""
from ctypes import cdll, c_uint32, POINTER
from .instr_encoder_const import InstrGenParam

ENCODE_LIBRARY_PATH_NAME = 'libtvm.so'
_ENCODE_DLL = cdll.LoadLibrary(ENCODE_LIBRARY_PATH_NAME)


class Encoder():  # pylint: disable=R0902
    """instruct encode for model."""
    def __init__(self):  # pylint: disable=R0915
        self.gen_conv = _ENCODE_DLL.GenConv
        self.gen_conv.restype = c_uint32
        self.gen_conv.argtypes = [POINTER(InstrGenParam)]

        self.gen_dma_ld_2d = _ENCODE_DLL.GenDmald2d
        self.gen_dma_ld_2d.restype = c_uint32
        self.gen_dma_ld_2d.argtypes = [POINTER(InstrGenParam)]

        self.gen_dma_ld_smask = _ENCODE_DLL.GenDmaldsmask
        self.gen_dma_ld_smask.restype = c_uint32
        self.gen_dma_ld_smask.argtypes = [POINTER(InstrGenParam)]

        self.gen_dma_set_2d = _ENCODE_DLL.GenDmaset2d
        self.gen_dma_set_2d.restype = c_uint32
        self.gen_dma_set_2d.argtypes = [POINTER(InstrGenParam)]

        self.gen_dma_brc = _ENCODE_DLL.GenDmabrc
        self.gen_dma_brc.restype = c_uint32
        self.gen_dma_brc.argtypes = [POINTER(InstrGenParam)]

        self.gen_dma_ld_img = _ENCODE_DLL.GenDmaldimg
        self.gen_dma_ld_img.restype = c_uint32
        self.gen_dma_ld_img.argtypes = [POINTER(InstrGenParam)]

        self.gen_dma_ld_3d = _ENCODE_DLL.GenDmald3d
        self.gen_dma_ld_3d.restype = c_uint32
        self.gen_dma_ld_3d.argtypes = [POINTER(InstrGenParam)]

        self.gen_dma_ld_3dv2 = _ENCODE_DLL.GenDmald3dv2
        self.gen_dma_ld_3dv2.restype = c_uint32
        self.gen_dma_ld_3dv2.argtypes = [POINTER(InstrGenParam)]

        self.gen_dma_winograd_l0b = _ENCODE_DLL.GenDmawinogradL0b
        self.gen_dma_winograd_l0b.restype = c_uint32
        self.gen_dma_winograd_l0b.argtypes = [POINTER(InstrGenParam)]

        self.gen_dma_winograd_l0a = _ENCODE_DLL.GenDmawinogradL0a
        self.gen_dma_winograd_l0a.restype = c_uint32
        self.gen_dma_winograd_l0a.argtypes = [POINTER(InstrGenParam)]

        self.gen_dma_mov = _ENCODE_DLL.GenDmamov
        self.gen_dma_mov.restype = c_uint32
        self.gen_dma_mov.argtypes = [POINTER(InstrGenParam)]

        self.gen_dma_col2_img = _ENCODE_DLL.GenDmacol2img
        self.gen_dma_col2_img.restype = c_uint32
        self.gen_dma_col2_img.argtypes = [POINTER(InstrGenParam)]

        self.gen_vgather = _ENCODE_DLL.GenVgather
        self.gen_vgather.restype = c_uint32
        self.gen_vgather.argtypes = [POINTER(InstrGenParam)]

        self.gen_vscatter = _ENCODE_DLL.GenVscatter
        self.gen_vscatter.restype = c_uint32
        self.gen_vscatter.argtypes = [POINTER(InstrGenParam)]

        self.gen_vpadding = _ENCODE_DLL.GenVpadding
        self.gen_vpadding.restype = c_uint32
        self.gen_vpadding.argtypes = [POINTER(InstrGenParam)]

        self.gen_vaddva = _ENCODE_DLL.GenVaddva
        self.gen_vaddva.restype = c_uint32
        self.gen_vaddva.argtypes = [POINTER(InstrGenParam)]

        self.gen_vmulva = _ENCODE_DLL.GenVmulva
        self.gen_vmulva.restype = c_uint32
        self.gen_vmulva.argtypes = [POINTER(InstrGenParam)]

        self.gen_move_vx = _ENCODE_DLL.GenMovevx
        self.gen_move_vx.restype = c_uint32
        self.gen_move_vx.argtypes = [POINTER(InstrGenParam)]

        self.gen_vexpx = _ENCODE_DLL.GenVexpx
        self.gen_vexpx.restype = c_uint32
        self.gen_vexpx.argtypes = [POINTER(InstrGenParam)]

        self.gen_vexpv = _ENCODE_DLL.GenVexpv
        self.gen_vexpv.restype = c_uint32
        self.gen_vexpv.argtypes = [POINTER(InstrGenParam)]

        self.gen_vrsqrtx = _ENCODE_DLL.GenVrsqrtx
        self.gen_vrsqrtx.restype = c_uint32
        self.gen_vrsqrtx.argtypes = [POINTER(InstrGenParam)]

        self.gen_vrsqrtv = _ENCODE_DLL.GenVrsqrtv
        self.gen_vrsqrtv.restype = c_uint32
        self.gen_vrsqrtv.argtypes = [POINTER(InstrGenParam)]

        self.gen_vrelux = _ENCODE_DLL.GenVrelux
        self.gen_vrelux.restype = c_uint32
        self.gen_vrelux.argtypes = [POINTER(InstrGenParam)]

        self.gen_vreluv = _ENCODE_DLL.GenVreluv
        self.gen_vreluv.restype = c_uint32
        self.gen_vreluv.argtypes = [POINTER(InstrGenParam)]

        self.gen_vrecx = _ENCODE_DLL.GenVrecx
        self.gen_vrecx.restype = c_uint32
        self.gen_vrecx.argtypes = [POINTER(InstrGenParam)]

        self.gen_vrecv = _ENCODE_DLL.GenVrecv
        self.gen_vrecv.restype = c_uint32
        self.gen_vrecv.argtypes = [POINTER(InstrGenParam)]

        self.gen_vlnx = _ENCODE_DLL.GenVlnx
        self.gen_vlnx.restype = c_uint32
        self.gen_vlnx.argtypes = [POINTER(InstrGenParam)]

        self.gen_vlnv = _ENCODE_DLL.GenVlnv
        self.gen_vlnv.restype = c_uint32
        self.gen_vlnv.argtypes = [POINTER(InstrGenParam)]

        self.gen_vabsx = _ENCODE_DLL.GenVabsx
        self.gen_vabsx.restype = c_uint32
        self.gen_vabsx.argtypes = [POINTER(InstrGenParam)]

        self.gen_vabsv = _ENCODE_DLL.GenVabsv
        self.gen_vabsv.restype = c_uint32
        self.gen_vabsv.argtypes = [POINTER(InstrGenParam)]

        self.gen_vcadd = _ENCODE_DLL.GenVcadd
        self.gen_vcadd.restype = c_uint32
        self.gen_vcadd.argtypes = [POINTER(InstrGenParam)]

        self.gen_vcmax = _ENCODE_DLL.GenVcmax
        self.gen_vcmax.restype = c_uint32
        self.gen_vcmax.argtypes = [POINTER(InstrGenParam)]

        self.gen_vcmin = _ENCODE_DLL.GenVcmin
        self.gen_vcmin.restype = c_uint32
        self.gen_vcmin.argtypes = [POINTER(InstrGenParam)]

        self.gen_vcgmax = _ENCODE_DLL.GenVcgmax
        self.gen_vcgmax.restype = c_uint32
        self.gen_vcgmax.argtypes = [POINTER(InstrGenParam)]

        self.gen_vcgmin = _ENCODE_DLL.GenVcgmin
        self.gen_vcgmin.restype = c_uint32
        self.gen_vcgmin.argtypes = [POINTER(InstrGenParam)]

        self.gen_vcgadd = _ENCODE_DLL.GenVcgadd
        self.gen_vcgadd.restype = c_uint32
        self.gen_vcgadd.argtypes = [POINTER(InstrGenParam)]

        self.gen_vnchwconv = _ENCODE_DLL.GenVnchwconv
        self.gen_vnchwconv.restype = c_uint32
        self.gen_vnchwconv.argtypes = [POINTER(InstrGenParam)]

        self.gen_vmovv = _ENCODE_DLL.GenVmovv
        self.gen_vmovv.restype = c_uint32
        self.gen_vmovv.argtypes = [POINTER(InstrGenParam)]

        self.gen_vcpadd = _ENCODE_DLL.GenVcpadd
        self.gen_vcpadd.restype = c_uint32
        self.gen_vcpadd.argtypes = [POINTER(InstrGenParam)]

        self.gen_vnotx = _ENCODE_DLL.GenVnotx
        self.gen_vnotx.restype = c_uint32
        self.gen_vnotx.argtypes = [POINTER(InstrGenParam)]

        self.gen_vrpac = _ENCODE_DLL.GenVrpac
        self.gen_vrpac.restype = c_uint32
        self.gen_vrpac.argtypes = [POINTER(InstrGenParam)]

        self.gen_vbs16 = _ENCODE_DLL.GenVbs16
        self.gen_vbs16.restype = c_uint32
        self.gen_vbs16.argtypes = [POINTER(InstrGenParam)]

        self.gen_vms4 = _ENCODE_DLL.GenVms4
        self.gen_vms4.restype = c_uint32
        self.gen_vms4.argtypes = [POINTER(InstrGenParam)]

        self.gen_vextract = _ENCODE_DLL.GenVextract
        self.gen_vextract.restype = c_uint32
        self.gen_vextract.argtypes = [POINTER(InstrGenParam)]

        self.gen_vconcat = _ENCODE_DLL.GenVconcat
        self.gen_vconcat.restype = c_uint32
        self.gen_vconcat.argtypes = [POINTER(InstrGenParam)]

        self.gen_vmergech = _ENCODE_DLL.GenVmergech
        self.gen_vmergech.restype = c_uint32
        self.gen_vmergech.argtypes = [POINTER(InstrGenParam)]

        self.gen_rpn_cor_diag = _ENCODE_DLL.GenRpncordiag
        self.gen_rpn_cor_diag.restype = c_uint32
        self.gen_rpn_cor_diag.argtypes = [POINTER(InstrGenParam)]

        self.gen_vtranspose = _ENCODE_DLL.GenVtranspose
        self.gen_vtranspose.restype = c_uint32
        self.gen_vtranspose.argtypes = [POINTER(InstrGenParam)]

        self.gen_v4dtrans = _ENCODE_DLL.GenV4dtrans
        self.gen_v4dtrans.restype = c_uint32
        self.gen_v4dtrans.argtypes = [POINTER(InstrGenParam)]

        self.gen_vconvv = _ENCODE_DLL.GenVconvv
        self.gen_vconvv.restype = c_uint32
        self.gen_vconvv.argtypes = [POINTER(InstrGenParam)]

        self.gen_vconvx = _ENCODE_DLL.GenVconvx
        self.gen_vconvx.restype = c_uint32
        self.gen_vconvx.argtypes = [POINTER(InstrGenParam)]

        self.gen_vaddx = _ENCODE_DLL.GenVaddx
        self.gen_vaddx.restype = c_uint32
        self.gen_vaddx.argtypes = [POINTER(InstrGenParam)]

        self.gen_vaddv = _ENCODE_DLL.GenVaddv
        self.gen_vaddv.restype = c_uint32
        self.gen_vaddv.argtypes = [POINTER(InstrGenParam)]

        self.gen_vsubx = _ENCODE_DLL.GenVsubx
        self.gen_vsubx.restype = c_uint32
        self.gen_vsubx.argtypes = [POINTER(InstrGenParam)]

        self.gen_vsubv = _ENCODE_DLL.GenVsubv
        self.gen_vsubv.restype = c_uint32
        self.gen_vsubv.argtypes = [POINTER(InstrGenParam)]

        self.gen_vmaxx = _ENCODE_DLL.GenVmaxx
        self.gen_vmaxx.restype = c_uint32
        self.gen_vmaxx.argtypes = [POINTER(InstrGenParam)]

        self.gen_vmaxv = _ENCODE_DLL.GenVmaxv
        self.gen_vmaxv.restype = c_uint32
        self.gen_vmaxv.argtypes = [POINTER(InstrGenParam)]

        self.gen_vminx = _ENCODE_DLL.GenVminx
        self.gen_vminx.restype = c_uint32
        self.gen_vminx.argtypes = [POINTER(InstrGenParam)]

        self.gen_vminv = _ENCODE_DLL.GenVminv
        self.gen_vminv.restype = c_uint32
        self.gen_vminv.argtypes = [POINTER(InstrGenParam)]

        self.gen_vmulx = _ENCODE_DLL.GenVmulx
        self.gen_vmulx.restype = c_uint32
        self.gen_vmulx.argtypes = [POINTER(InstrGenParam)]

        self.gen_vmulv = _ENCODE_DLL.GenVmulv
        self.gen_vmulv.restype = c_uint32
        self.gen_vmulv.argtypes = [POINTER(InstrGenParam)]

        self.gen_vmlax = _ENCODE_DLL.GenVmlax
        self.gen_vmlax.restype = c_uint32
        self.gen_vmlax.argtypes = [POINTER(InstrGenParam)]

        self.gen_vmlav = _ENCODE_DLL.GenVmlav
        self.gen_vmlav.restype = c_uint32
        self.gen_vmlav.argtypes = [POINTER(InstrGenParam)]

        self.gen_vmaddx = _ENCODE_DLL.GenVmaddx
        self.gen_vmaddx.restype = c_uint32
        self.gen_vmaddx.argtypes = [POINTER(InstrGenParam)]

        self.gen_vmaddv = _ENCODE_DLL.GenVmaddv
        self.gen_vmaddv.restype = c_uint32
        self.gen_vmaddv.argtypes = [POINTER(InstrGenParam)]

        self.gen_vmaddrelux = _ENCODE_DLL.GenVmaddrelux
        self.gen_vmaddrelux.restype = c_uint32
        self.gen_vmaddrelux.argtypes = [POINTER(InstrGenParam)]

        self.gen_vmaddreluv = _ENCODE_DLL.GenVmaddreluv
        self.gen_vmaddreluv.restype = c_uint32
        self.gen_vmaddreluv.argtypes = [POINTER(InstrGenParam)]

        self.gen_vaxpyx = _ENCODE_DLL.GenVaxpyx
        self.gen_vaxpyx.restype = c_uint32
        self.gen_vaxpyx.argtypes = [POINTER(InstrGenParam)]

        self.gen_vaxpyv = _ENCODE_DLL.GenVaxpyv
        self.gen_vaxpyv.restype = c_uint32
        self.gen_vaxpyv.argtypes = [POINTER(InstrGenParam)]

        self.gen_vaddsx = _ENCODE_DLL.GenVaddsx
        self.gen_vaddsx.restype = c_uint32
        self.gen_vaddsx.argtypes = [POINTER(InstrGenParam)]

        self.gen_vaddsv = _ENCODE_DLL.GenVaddsv
        self.gen_vaddsv.restype = c_uint32
        self.gen_vaddsv.argtypes = [POINTER(InstrGenParam)]

        self.gen_vmulsx = _ENCODE_DLL.GenVmulsx
        self.gen_vmulsx.restype = c_uint32
        self.gen_vmulsx.argtypes = [POINTER(InstrGenParam)]

        self.gen_vmulsv = _ENCODE_DLL.GenVmulsv
        self.gen_vmulsv.restype = c_uint32
        self.gen_vmulsv.argtypes = [POINTER(InstrGenParam)]

        self.gen_vmaxsx = _ENCODE_DLL.GenVmaxsx
        self.gen_vmaxsx.restype = c_uint32
        self.gen_vmaxsx.argtypes = [POINTER(InstrGenParam)]

        self.gen_vmaxsv = _ENCODE_DLL.GenVmaxsv
        self.gen_vmaxsv.restype = c_uint32
        self.gen_vmaxsv.argtypes = [POINTER(InstrGenParam)]

        self.gen_vminsx = _ENCODE_DLL.GenVminsx
        self.gen_vminsx.restype = c_uint32
        self.gen_vminsx.argtypes = [POINTER(InstrGenParam)]

        self.gen_vminsv = _ENCODE_DLL.GenVminsv
        self.gen_vminsv.restype = c_uint32
        self.gen_vminsv.argtypes = [POINTER(InstrGenParam)]

        self.gen_vcmpx = _ENCODE_DLL.GenVcmpx
        self.gen_vcmpx.restype = c_uint32
        self.gen_vcmpx.argtypes = [POINTER(InstrGenParam)]

        self.gen_vcmpv = _ENCODE_DLL.GenVcmpv
        self.gen_vcmpv.restype = c_uint32
        self.gen_vcmpv.argtypes = [POINTER(InstrGenParam)]

        self.gen_vcmpvx = _ENCODE_DLL.GenVcmpvx
        self.gen_vcmpvx.restype = c_uint32
        self.gen_vcmpvx.argtypes = [POINTER(InstrGenParam)]

        self.gen_vcmpvsx = _ENCODE_DLL.GenVcmpvsx
        self.gen_vcmpvsx.restype = c_uint32
        self.gen_vcmpvsx.argtypes = [POINTER(InstrGenParam)]

        self.gen_vorx = _ENCODE_DLL.GenVorx
        self.gen_vorx.restype = c_uint32
        self.gen_vorx.argtypes = [POINTER(InstrGenParam)]

        self.gen_vandx = _ENCODE_DLL.GenVandx
        self.gen_vandx.restype = c_uint32
        self.gen_vandx.argtypes = [POINTER(InstrGenParam)]

        self.gen_vshrx = _ENCODE_DLL.GenVshrx
        self.gen_vshrx.restype = c_uint32
        self.gen_vshrx.argtypes = [POINTER(InstrGenParam)]

        self.gen_vshlx = _ENCODE_DLL.GenVshlx
        self.gen_vshlx.restype = c_uint32
        self.gen_vshlx.argtypes = [POINTER(InstrGenParam)]

        self.gen_vselx = _ENCODE_DLL.GenVselx
        self.gen_vselx.restype = c_uint32
        self.gen_vselx.argtypes = [POINTER(InstrGenParam)]

        self.gen_vselv = _ENCODE_DLL.GenVselv
        self.gen_vselv.restype = c_uint32
        self.gen_vselv.argtypes = [POINTER(InstrGenParam)]

        self.gen_viou = _ENCODE_DLL.GenViou
        self.gen_viou.restype = c_uint32
        self.gen_viou.argtypes = [POINTER(InstrGenParam)]

        self.gen_vaadd = _ENCODE_DLL.GenVaadd
        self.gen_vaadd.restype = c_uint32
        self.gen_vaadd.argtypes = [POINTER(InstrGenParam)]

        self.gen_vreduce = _ENCODE_DLL.GenVreduce
        self.gen_vreduce.restype = c_uint32
        self.gen_vreduce.argtypes = [POINTER(InstrGenParam)]

        self.gen_rpn_cor = _ENCODE_DLL.GenRpncor
        self.gen_rpn_cor.restype = c_uint32
        self.gen_rpn_cor.argtypes = [POINTER(InstrGenParam)]

        self.gen_vdivx = _ENCODE_DLL.GenVdivx
        self.gen_vdivx.restype = c_uint32
        self.gen_vdivx.argtypes = [POINTER(InstrGenParam)]

        self.gen_vdivv = _ENCODE_DLL.GenVdivv
        self.gen_vdivv.restype = c_uint32
        self.gen_vdivv.argtypes = [POINTER(InstrGenParam)]

        self.gen_vmulconvx = _ENCODE_DLL.GenVmulconvx
        self.gen_vmulconvx.restype = c_uint32
        self.gen_vmulconvx.argtypes = [POINTER(InstrGenParam)]

        self.gen_vmulconvv = _ENCODE_DLL.GenVmulconvv
        self.gen_vmulconvv.restype = c_uint32
        self.gen_vmulconvv.argtypes = [POINTER(InstrGenParam)]

        self.gen_vadd_deq_relux = _ENCODE_DLL.GenVadddeqrelux
        self.gen_vadd_deq_relux.restype = c_uint32
        self.gen_vadd_deq_relux.argtypes = [POINTER(InstrGenParam)]

        self.gen_vsqrtx = _ENCODE_DLL.GenVsqrtx
        self.gen_vsqrtx.restype = c_uint32
        self.gen_vsqrtx.argtypes = [POINTER(InstrGenParam)]

        self.gen_vsqrtv = _ENCODE_DLL.GenVsqrtv
        self.gen_vsqrtv.restype = c_uint32
        self.gen_vsqrtv.argtypes = [POINTER(InstrGenParam)]

        self.gen_mmad = _ENCODE_DLL.GenMmad
        self.gen_mmad.restype = c_uint32
        self.gen_mmad.argtypes = [POINTER(InstrGenParam)]

        self.gen_dp = _ENCODE_DLL.GenDp
        self.gen_dp.restype = c_uint32
        self.gen_dp.argtypes = [POINTER(InstrGenParam)]

        self.gen_vaddrelux = _ENCODE_DLL.GenVaddrelux
        self.gen_vaddrelux.restype = c_uint32
        self.gen_vaddrelux.argtypes = [POINTER(InstrGenParam)]

        self.gen_vsubrelux = _ENCODE_DLL.GenVsubrelux
        self.gen_vsubrelux.restype = c_uint32
        self.gen_vsubrelux.argtypes = [POINTER(InstrGenParam)]

        self.gen_vbi = _ENCODE_DLL.GenVbi
        self.gen_vbi.restype = c_uint32
        self.gen_vbi.argtypes = [POINTER(InstrGenParam)]

    @staticmethod
    def new_param():
        """new  param for function call."""
        return InstrGenParam()

    def __str__(self):
        """
        use to represent instr encoder
        :return:
        """
        return "represent instr encoder"
