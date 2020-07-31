"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     tik_conf_.py
DESC:     configuration of tik
CREATED:  2019-04-18 18:53:42
MODIFIED: 2019-07-25 09:59:30
"""
from te.platform.cce_conf import te_set_version
from te.platform.cce_conf import cceProduct
from te.platform.cce_conf import get_soc_spec
from te.platform.cce_params import scope_cbuf
from te.platform.cce_params import scope_ubuf
from te.platform.cce_params import scope_ca
from te.platform.cce_params import scope_cb
from te.platform.cce_params import scope_cc
from te.platform.cce_params import scope_smask
from te.platform.cce_params import AIC
from te.platform.cce_params import HI3796CV300ESAIC
from te.platform.cce_params import VEC
from te.platform.cce_params import ASCEND_310AIC
from te.platform.cce_params import ASCEND_910AIC
from .tik_params import KB_SIZE, L1_BUFFER, UB_BUFFER,\
    L0A_BUFFER, L0B_BUFFER, LOC_BUFFER, SMASK_BUFFER, \
    AI_CORE_VERSION_MAP_TO_PRODUCT
from .tik_source_info import TikSourceInfo, source_info_decorator
from .tik_check_util import TikCheckUtil
from ..common.tik_get_soc_name import get_soc_name
from ..common.tik_get_soc_name import get_soc_core_type

def _get_buffers_parameter(buffer_arch_list):
    """
    According  D core information,buffers parameters can be known.

    Parameters
    ----------
    buffer_arch_list:D aicore information

    Returns
    ----------
    return:buffers_map
    buffers_map: L0A/B/C,L1 and UB buffer memory size

    """
    buffer_names = [L1_BUFFER, UB_BUFFER,
                    L0A_BUFFER, L0B_BUFFER, LOC_BUFFER, SMASK_BUFFER]
    buffer_arch = {}
    for buffer_params in buffer_arch_list:
        for buffer_name in buffer_names:
            if buffer_params.find(buffer_name) == 0:
                buffer_paras = buffer_params.split(' ')
                if buffer_name == SMASK_BUFFER:
                    buffer_size = buffer_paras[-1].split("B")[0]
                    buffer_arch[buffer_name] = int(buffer_size)
                else:
                    buffer_size = buffer_paras[-1].split("KB")[0]
                    buffer_arch[buffer_name] = int(buffer_size)*KB_SIZE
    return buffer_arch


class Dprofile1():
    """
    ai_core profile explanation
    """
    _SMASK_MAP = {
        AIC: 256,
        VEC: 0,
        HI3796CV300ESAIC: 256,
        ASCEND_310AIC: 0,
        ASCEND_910AIC: 0
    }

    def __init__(self, ai_core_arch=None, ai_core_version=None, ddk_version=None):
        """
        ai_core profile initialization
        Parameters
        ----------
        ai_core_arch : ai_core architecture
        ai_core_version: ai_core version

        Returns
        -------
        """
        # need TikSourceInfo() init function to set source_info None first
        self.source_info = TikSourceInfo()
        self.source_info.register_source_info(depth=2)
        if ddk_version is not None:
            self.ddk_version = ddk_version
            cceProduct(self.ddk_version)
        elif SetProductFlag.is_set_product_version or \
                (ai_core_arch is None or ai_core_version is None):
            # ge do the init!
            pass
        else:
            TikCheckUtil.check_type_match(ai_core_arch, str,
                                          "ai_core_arch should be str")
            TikCheckUtil.check_type_match(ai_core_version, str,
                                          "ai_core_version should be str")
            self.ai_core_arch = ai_core_arch.lower()
            self.ai_core_version = ai_core_version.lower()
            # version
            self.ddk_version = _gen_version(self.ai_core_arch,
                                            self.ai_core_version)
            cceProduct(self.ddk_version)
        # we will use cce product params to represent ai core arch and version.
        # map for save device buffer info
        self.ai_core_buffer = {}
        self.registe()

        self.source_info.clear_source_info()

    def registe(self):
        """
        ai_core register configure
        Parameters
        ----------

        Returns
        -------
        return:no return
        """
        # get the device buffer info
        _smask_map = {
            AIC: 256,
            VEC: 0,
            HI3796CV300ESAIC: 256,
            ASCEND_310AIC: 0,
            ASCEND_910AIC: 0
        }
        l1_buffer = _get_l1_size()
        ub_buffer = _get_ub_size()
        l0a_buffer = _get_l0a_size()
        l0b_buffer = _get_l0b_size()
        l0c_buffer = _get_l0c_size()
        smask_buffer = _smask_map[get_soc_name() + get_soc_core_type()]

        key_name = get_soc_name() + get_soc_core_type()
        # save the device buffer info into the map
        self.ai_core_buffer[key_name] = ["L1_Buffer: " +
                                         str(l1_buffer // KB_SIZE) + "KB",
                                         "Unified_Buffer: " +
                                         str(ub_buffer // KB_SIZE) + "KB",
                                         "L0A_Buffer: " +
                                         str(l0a_buffer // KB_SIZE) + "KB",
                                         "L0B_Buffer: " +
                                         str(l0b_buffer // KB_SIZE) + "KB",
                                         "L0C_Buffer: " +
                                         str(l0c_buffer // KB_SIZE) + "KB",
                                         "SMASK_Buffer: " +
                                         str(smask_buffer) + "B"
                                        ]

    @source_info_decorator()
    def get_aicore_num(self):
        """
        return ai_core number for specify ai core
        Parameters
        ----------

        Returns
        -------
        return:ai_core number
        """
        return get_soc_spec("CORE_NUM")

    def buffer_size_query(self, buffer_scope=None):
        """
        according the AI core params, get the buffer params
        for_example:LOA/B/C,L01,UB buffer
        Parameters
        ----------
        buffer_scope:value is scope_cbuf, scope_ubuf, scope_ca
                      scope_cb, scope_cc

        Returns
        -------
        return:ai_core buffer params
        """
        key_map = {scope_cbuf: L1_BUFFER,
                   scope_ubuf: UB_BUFFER,
                   scope_ca: L0A_BUFFER,
                   scope_cb: L0B_BUFFER,
                   scope_cc: LOC_BUFFER,
                   scope_smask: SMASK_BUFFER}
        key_name = get_soc_name() + get_soc_core_type()
        buffer_arch_list = self.ai_core_buffer[key_name]
        buffer_arch = _get_buffers_parameter(buffer_arch_list)
        if buffer_scope is None:
            buffer_map = {scope_cbuf: 0, scope_ubuf: 0, scope_ca: 0,
                          scope_cb: 0, scope_cc: 0, scope_smask: 0}
            for scope in key_map:
                buffer_map[scope] = buffer_arch[key_map[scope]]
            return buffer_map
        TikCheckUtil.check_in_range(
            buffer_scope, key_map.keys(), "buffer_scope value is not correct!")
        return buffer_arch[key_map[buffer_scope]]

    @source_info_decorator()
    def get_l1_buffer_size(self):
        """
        return l1 buffer size for specify ai core
        Parameters
        ----------

        Returns
        -------
        return:l1_buffer_size
        """
        return _get_l1_size()

    @source_info_decorator()
    def get_l0a_buffer_size(self):
        """
        return l0a_buffer buffer size for specify ai core
        Parameters
        ----------

        Returns
        -------
        return:l0a_buffer_size
        """
        return _get_l0a_size()

    @source_info_decorator()
    def get_l0b_buffer_size(self):
        """
        return l0b_buffer buffer size for specify ai core
        Parameters
        ----------

        Returns
        -------
        return:l0a_buffer_size
        """
        return _get_l0b_size()

    @source_info_decorator()
    def get_l0c_buffer_size(self):
        """
        return l0c_buffer buffer size for specify ai core
        Parameters
        ----------

        Returns
        -------
        return:l0c_buffer_size
        """
        return _get_l0c_size()

    @source_info_decorator()
    def get_product_name(self):
        """
        return product_name for specify ai core
        Parameters
        ----------

        Returns
        -------
        return:product_name
        """
        return _get_product_name()


def _get_product_name():
    """
    return a tuple, include version and product name
    Parameters
    ----------
    product:ddk version, like 1.1.xxx.xxx

    Returns
    -------
    return:a tuple represents current product, like v100, mini
    """
    product_name = get_soc_name() + get_soc_core_type()
    if product_name in AI_CORE_VERSION_MAP_TO_PRODUCT:
        return AI_CORE_VERSION_MAP_TO_PRODUCT[product_name]
    return TikCheckUtil.raise_error(
        "Not valid product version for tik:" + product_name)


def set_product_version(version):
    """set version info

    Parameters
    ----------
    version : str
        product name
        format: <major>.<middle>.<minor>.<point>
        major: 1xx or 2xx or 3xx
    """
    # need TikSourceInfo() init function to set source_info None first
    TikSourceInfo().register_source_info()
    te_set_version(version)
    SetProductFlag.is_set_product_version = True
    TikSourceInfo.clear_source_info()

def unset_product_version():
    """unset SetProductFlag to false

    """
    # need TikSourceInfo() init function to set source_info None first
    TikSourceInfo().register_source_info()
    SetProductFlag.is_set_product_version = False
    TikSourceInfo.clear_source_info()

class SetProductFlag():
    """
    use to show whether has set product version
    """
    is_set_product_version = False

    def __str__(self):
        pass

    def __hash__(self):
        pass

def _get_ub_size():
    return get_soc_spec("UB_SIZE")

def _get_l0a_size():
    return get_soc_spec("L0A_SIZE")

def _get_l0b_size():
    return get_soc_spec("L0B_SIZE")

def _get_l0c_size():
    return get_soc_spec("L0C_SIZE")

def _get_l1_size():
    return get_soc_spec("L1_SIZE")

def _gen_version(arch, version):
    """
    gen version
    :param arch:
    :param version:
    :return: string ddk version
    """
    if arch == "v100":
        if version == "mini":
            return "1.1.xxx.xxx"
        if version == "cloud":
            return "1.60.xxx.xxx"
        TikCheckUtil.raise_error("Error chip!")
    elif arch == "v200":
        if version == "aic":
            return "2.10.xxx.xxx"
        if version in ("hisi-es", "hisi-cs"):
            return "5.10.xxx.xxx"
        if version == "vec":
            return "3.20.xxx.xxx"
        TikCheckUtil.raise_error("Error chip!")
