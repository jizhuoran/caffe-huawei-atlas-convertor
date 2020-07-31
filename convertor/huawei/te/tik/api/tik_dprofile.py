"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     tik_dprofile.py
DESC:     configuration of tik
CREATED:  2020-3-16 21:12:13
MODIFIED: 2020-3-16 21:12:45
"""
from te.platform.cce_conf import get_soc_spec
from ..tik_lib.tik_conf_ import Dprofile1
from ..tik_lib.tik_source_info import source_info_decorator


class Dprofile(Dprofile1):
    """
    ai_core profile explanation
    """
    def __init__(self, ai_core_arch=None,
                 ai_core_version=None, ddk_version=None):
        """
        Manages and configures the information about the Ascend AI processor
        Description:
          Manages and configures the information about the Ascend AI processor,
          such as the product architecture,product form, and buffer sizes at
          all levels.Since the buffer size and instructions vary according to
          the Ascend AI processor version, the class Dprofile constructor is
          used to define the target machine of the Ascend AI processor and
          specify the hardware environment for programming.

        Args:
          ai_core_arch: Product architecture. The value is a string. The
          options are as follows:
            1. v100(Ascend310;Ascend910)
            2. v200(Ascend610-vec;Ascend610-aic;Hi3796)
          ai_core_version: Chip version. The value is a string. The options
          are as follows:
            1. mini(Ascend310)
            2. cloud(Ascend910)
            3. hisi-es(Hi3796)
            4. vec(Ascend610-vec)
            5. aic(Ascend610-aic)
          ddk_version: An internal optional parameters

        Returns:
          Instance of class Dprofile

        Restrictions:
          - Select instructions carefully since they are related to the product
          architecture and product form.
          - If the product architecture and product form are not specified in
          the Dprofile, the default product form is used.

        Example:
            from te import tik
            tik_dprofile = tik.Dprofile("v100","mini")
        """
        super(Dprofile, self).__init__(ai_core_arch=ai_core_arch,
                                       ai_core_version=ai_core_version,
                                       ddk_version=ddk_version)

    @source_info_decorator()
    def get_unified_buffer_size(self):
        """
        Obtains the UB size (in bytes) of the corresponding product form.
        Description:
          Obtains the UB size (in bytes) of the corresponding product form.

        Kwargs:
          None

        Returns:
          UB size (in bytes) of the corresponding product form

        Restrictions:
          None

        Example:
            from te import tik
            tik_dprofile = tik.Dprofile("v100","mini")
            unified_buffer_size = tik_dprofile.get_unified_buffer_size()
        """
        unified_buffer_size = get_soc_spec("UB_SIZE")
        if unified_buffer_size in (-1, 0):
            return None
        return unified_buffer_size

