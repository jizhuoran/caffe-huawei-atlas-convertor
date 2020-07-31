"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     pvmodel.py
DESC:     import C APIs of PVModel
CREATED:  2019-7-04 20:12:13
MODIFIED: 2019-7-17 15.06.16
"""
# disabling:
# R0903: too-few-public-methods
# R0902: too-many-instance-attributes

# import C APIs of PVModel. Avoid using from ctypes import *
from ctypes import Structure, c_int, c_bool, c_uint64, POINTER, cdll, \
    c_void_p, c_uint32, c_char_p, c_int64, c_size_t, byref

from te.tik.tik_lib.tik_params import ONE_VA_ADDR_NUM, VA_ADDR_BIT_LEN, \
    VA_ADDR_BYTE_SIZE, MAX_ADDR, MAX_ADDR_HEX, MAX_VA_ADDR_NUM
from te.platform.cce_params import scope_cc, scope_cbuf, scope_cb, scope_ca,\
    scope_ubuf, scope_gm, HI3796CV300ESAIC, \
    HI3796CV300CSAIC, AIC, VEC, ASCEND_310AIC, ASCEND_910AIC
from te.tik.tik_lib.tik_check_util import TikCheckUtil
from te.tik.common.tik_get_soc_name import get_soc_name
from te.tik.common.tik_get_soc_name import get_soc_core_type
_RETURN_CODE_ZERO = 0


class PvMemAccess(Structure):
    """memory object"""
    # pylint: disable=R0903
    _fields_ = [
        ('mem_scope', c_int),
        ('is_read', c_bool),
        ('addr', c_uint64),
        ('size', c_uint64)
    ]


class PVMemAccessList(Structure):
    """memory object list"""
    # pylint: disable=R0903
    _fields_ = [
        ('list', POINTER(PvMemAccess)),
        ('len', c_int)
    ]


class PyMemAccess():
    """python memory object"""
    # pylint: disable=R0903
    _r_scope_mapping = {
        0: scope_gm,
        1: scope_cbuf,
        2: scope_ca,
        3: scope_cb,
        4: scope_cc,
        5: scope_ubuf,
        6: 'local.SB',
        7: 'local.L0A_WINO',
        8: 'local.L0B_WINO'
    }

    def __init__(self, scope, mode, addr, size):

        self.scope = self._r_scope_mapping[scope]
        self.mode = 'r' if mode else 'w'
        self.addr = addr
        self.size = size

    def __str__(self):
        return '[MemAccess]scope:{} mode:{} addr:{} size:{}'\
            .format(self.scope, self.mode, self.addr, self.size)


class PVModel():
    """Encapsulation of pvmodel"""
    # pylint: disable=R0902

    _scope_mapping = {
        scope_gm: 0,
        scope_cbuf: 1,
        scope_ca: 2,
        scope_cb: 3,
        scope_cc: 4,
        scope_ubuf: 5,
    }

    _version2id = {
        AIC: 0,
        ASCEND_310AIC: 0,
        ASCEND_910AIC: 0,
        VEC: 3,
        HI3796CV300ESAIC: 1,
        HI3796CV300CSAIC: 1
    }

    def __init__(self, dprofile):
        self.dprofile = dprofile
        self._load_pvmodel()

        self._pv_create = self._dll.pv_create
        self._pv_create.restype = c_void_p

        self._pv_destroy = self._dll.pv_destroy
        self._pv_destroy.argtypes = [c_void_p]

        self._pv_step = self._dll.pv_step
        self._pv_step.restype = c_int
        self._pv_step.argtypes = [c_void_p, c_uint32]

        self._pv_read_gpr_register = self._dll.pv_read_gpr_register
        self._pv_read_gpr_register.restype = c_int
        self._pv_read_gpr_register.argtypes = [c_void_p, c_uint64,
                                               POINTER(c_uint64)]

        self._pv_write_gpr_register = self._dll.pv_write_gpr_register
        self._pv_write_gpr_register.restype = c_int
        self._pv_write_gpr_register.argtypes = [c_void_p,
                                                c_uint64, c_uint64]

        self._pv_read_spr_register = self._dll.pv_read_spr_register
        self._pv_read_spr_register.restype = c_int
        self._pv_read_spr_register.argtypes = [c_void_p, c_char_p,
                                               POINTER(c_uint64)]

        self._pv_write_spr_register = self._dll.pv_write_spr_register
        self._pv_write_spr_register.restype = c_int
        self._pv_write_spr_register.argtypes = [c_void_p,
                                                c_char_p, c_uint64]

        self._pv_read_va_register = self._dll.pv_read_va_register
        self._pv_read_va_register.restype = c_int
        self._pv_read_va_register.argtypes = \
            [c_void_p, c_int, POINTER(c_uint64), POINTER(c_uint64)]

        self._pv_write_va_register = self._dll.pv_write_va_register
        self._pv_write_va_register.restype = c_int
        self._pv_write_va_register.argtypes = \
            [c_void_p, c_int, POINTER(c_uint64), POINTER(c_uint64)]

        self._pv_read_memory = self._dll.pv_read_memory
        self._pv_read_memory.restype = c_int
        self._pv_read_memory.argtypes = [c_void_p, c_int64, c_int,
                                         c_void_p, c_int]

        self._pv_write_memory = self._dll.pv_write_memory
        self._pv_write_memory.restype = c_int
        self._pv_write_memory.argtypes = [c_void_p, c_int64, c_int,
                                          c_void_p, c_int]

        self._pv_get_memory_capacity = self._dll.pv_get_memory_capacity
        self._pv_get_memory_capacity.restype = c_int
        self._pv_get_memory_capacity.argtypes = [c_void_p, c_int,
                                                 POINTER(c_int)]

        self._pv_malloc = self._dll.malloc
        self._pv_malloc.restype = c_void_p
        self._pv_malloc.argtypes = [c_size_t]

        self._pv_free = self._dll.free
        self._pv_free.argtypes = [c_void_p]

        self._pv_get_mem_acc_list = self._dll.pv_get_mem_access
        self._pv_get_mem_acc_list.restype = POINTER(PVMemAccessList)
        self._pv_get_mem_acc_list.argtypes = []

        # c_void_p
        self.model = self._pv_create(self._version2id[get_soc_name()
                                                      + get_soc_core_type()],
                                     '', 0)

    def _load_pvmodel(self):
        """load pvmodel"""
        pvmodel_name = 'lib_pvmodel.so'
        llt_mini_pvmodel_name = 'lib_pvmodel_mini.so'
        llt_cloud_pvmodel_name = 'lib_pvmodel_cloud.so'

        for name in (pvmodel_name,
                     llt_mini_pvmodel_name, llt_cloud_pvmodel_name):
            try:
                self._dll = cdll.LoadLibrary(name)
            except OSError:
                # load current pvmodel_so fail, continue to next
                continue
            else:
                # load pvmodel_so success, end
                break
        else:
            # all load pvmodel_so fail, raise error
            TikCheckUtil.raise_error(
                'lib_pvmodel.so: cannot open shared object file:'
                ' No such file or directory', exception_type=OSError)

    def step(self, inst):
        """run a instruction"""
        return_code = self._pv_step(self.model, c_uint32(inst))
        TikCheckUtil.check_equality(
            return_code, _RETURN_CODE_ZERO, "return_code not 0")

    def read_gpr(self, x_i):
        """read from general purpose register"""
        ret = c_uint64(0)
        return_code = self._pv_read_gpr_register(self.model,
                                                 c_uint64(x_i),
                                                 byref(ret))
        TikCheckUtil.check_equality(
            return_code, _RETURN_CODE_ZERO, "return_code not 0")
        return ret.value

    def write_gpr(self, x_i, value):
        """write to general purpose register"""
        return_code = self._pv_write_gpr_register(self.model,
                                                  c_uint64(x_i),
                                                  c_uint64(value))
        TikCheckUtil.check_equality(
            return_code, _RETURN_CODE_ZERO, "return_code not 0")

    def read_spr(self, name):
        """read from special register"""
        ret = c_uint64(0)
        return_code = self._pv_read_spr_register(self.model,
                                                 name.encode("utf-8"),
                                                 byref(ret))
        TikCheckUtil.check_equality(
            return_code, _RETURN_CODE_ZERO,
            "read spr %s failed, maybe spr name is wrong" % name)
        return ret.value

    def write_spr(self, name, value):
        """write to special register"""
        return_code = self._pv_write_spr_register(self.model,
                                                  name.encode("utf-8"),
                                                  c_uint64(value))
        TikCheckUtil.check_equality(
            return_code, _RETURN_CODE_ZERO, "return_code not 0")

    def read_va(self, va_id):
        """read from VA register"""
        val0 = c_uint64(0)
        val1 = c_uint64(0)
        return_code = self._pv_read_va_register(self.model, c_int(va_id),
                                                byref(val0), byref(val1))
        TikCheckUtil.check_equality(
            return_code, _RETURN_CODE_ZERO, "return_code not 0")
        val0 = val0.value
        val1 = val1.value
        addr_list = []
        for i in range(ONE_VA_ADDR_NUM):
            addr_list.append((val0 >> (i*VA_ADDR_BIT_LEN)) & MAX_ADDR_HEX)

        for i in range(ONE_VA_ADDR_NUM):
            addr_list.append((val1 >> (i*VA_ADDR_BIT_LEN)) & MAX_ADDR_HEX)

        return addr_list

    def write_va(self, va_id, addr_list):
        """write to VA register"""
        val0 = 0
        val1 = 0
        TikCheckUtil.check_equality(
            len(addr_list), MAX_VA_ADDR_NUM,
            "Expecting 8 addresses in addr list, but get %d" % len(addr_list))
        if va_id < 0 or va_id >= MAX_VA_ADDR_NUM:
            TikCheckUtil.raise_error("Invalid VA-id: %d" % va_id)

        i = 0
        for addr in addr_list:
            addr = (addr // VA_ADDR_BYTE_SIZE)
            TikCheckUtil.check_le(
                addr, MAX_ADDR,
                "Invalid adress setting to VA: 0x%x. Too large for 16bit" %
                addr*VA_ADDR_BYTE_SIZE)
            if i < ONE_VA_ADDR_NUM:
                val0 |= (addr << (i*VA_ADDR_BIT_LEN))
            else:
                val1 |= (addr << ((i - ONE_VA_ADDR_NUM)*VA_ADDR_BIT_LEN))
            i += 1

        val0 = c_uint64(val0)
        val1 = c_uint64(val1)

        return_code = self._pv_write_va_register(self.model, c_int(va_id),
                                                 byref(val0), byref(val1))
        TikCheckUtil.check_equality(
            return_code, _RETURN_CODE_ZERO,
            'write va {} failed'.format(va_id))

    def _scope2int(self, scope):
        """convert the scope name to the specified encoding"""
        return c_int(self._scope_mapping[scope])

    def read_memory(self, addr, scope, buffer_addr, buffer_len):
        """read a segment of memory"""
        return_code = self._pv_read_memory(self.model, c_int64(addr),
                                           self._scope2int(scope),
                                           buffer_addr, c_int(buffer_len))
        TikCheckUtil.check_equality(
            return_code, _RETURN_CODE_ZERO, "return_code not 0")

    def write_memory(self, addr, scope, buffer_addr, buffer_len):
        """write a segment of memory"""
        return_code = self._pv_write_memory(self.model, c_int64(addr),
                                            self._scope2int(scope),
                                            buffer_addr, c_int(buffer_len))
        TikCheckUtil.check_equality(
            return_code, _RETURN_CODE_ZERO, "return_code not 0")

    def get_memory_capacity(self, scope):
        """get the memory size of the specified scope"""
        ret = c_int(0)
        return_code = self._pv_get_memory_capacity(
            self.model, self._scope2int(scope), byref(ret))
        TikCheckUtil.check_equality(
            return_code, _RETURN_CODE_ZERO, "return_code not 0")
        return ret

    def get_memory_access(self):
        """get the memory access of the previous instruction"""
        mem_access_list = self._pv_get_mem_acc_list()
        ret = mem_access_list.contents
        num = ret.len
        access_list = []
        for i in range(num):
            c_ptr = ret.list[i]
            access_list.append(PyMemAccess(c_ptr.mem_scope, c_ptr.is_read,
                                           c_ptr.addr, c_ptr.size))
        return access_list

    def __del__(self):
        self._pv_destroy(self.model)
