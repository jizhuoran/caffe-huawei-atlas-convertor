"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

Tiling helper, set and get a fixed tiling
"""
import os
import copy
import json
from te import tvm


class Singleton():
    """Singleton base class
    """
    __instance = None
    __tiling = None
    __tiling_type = None
    __input_params = None

    def __init__(self):
        self._singleton__tiling = None
        self._singleton__tiling_type = None
        self._singleton__input_params = None

        get_config_path_fun = tvm.get_global_func("_query_config_path")
        config_path = get_config_path_fun()
        config_path = os.path.realpath(config_path)
        if os.path.exists(config_path):
            with open(config_path, 'r') as handler:
                config = json.load(handler)
                self._singleton__tiling_type = config.get("tiling_type")
        if not self._singleton__tiling_type:
            self._singleton__tiling_type = "auto_tiling"

    def __new__(cls, *args, **kw):
        if not cls.__instance:
            cls.__instance = super(Singleton, cls).__new__(cls, *args, **kw)
        return cls.__instance

    def get_params(self):
        """Get tiling input params

        Notice
        ------
        this function is create for auto tune tool to get tiling input params

        Returns
        ----------
        input_params: dict
            set by tiling query or get tiling
        """
        return copy.deepcopy(self._singleton__input_params)

    def set_params(self, inputs):
        """Set get tiling or tiling query input params

        Parameters
        ----------
        inputs: dict
            build up by schedule

        Notice
        ------
        this function is create for auto tune tool to get tiling input params,
            params set last time should be same to get_tiling inputs, usually
            called under non-tuning_tiling mode by schedule when building
            executable binary file

        """
        self._singleton__input_params = copy.deepcopy(inputs)

    def get_tiling(self, inputs):
        """Get the tiling from Singleton object
        Parameters
        ----------

        Notice
        ------
        this function is work under tuning tiling mode together with
            set tiling, input params used is set by set_params last
            time, should be exaclty same to inputs

        some list value given is tvm.expr.Int, so compare use string
            and not original dict

        Returns
        -------
        tiling: dict
            The tiling saved in Singleton object
        """
        if not isinstance(inputs, dict) or not inputs:
            raise RuntimeError("illegal inputs: %s" % str(inputs))

        pre = copy.copy(self._singleton__input_params)
        if not isinstance(inputs, dict) or not pre:
            raise RuntimeError("set params when tuning tiling, like: %s"
                               % str(inputs))
        cur = copy.copy(inputs)
        ignore_list = ["reserved_ub"]
        for item in ignore_list:
            if item in pre:
                pre.pop(item)
            if item in cur:
                cur.pop(item)
        if str(cur) != str(pre):
            raise RuntimeError("tiling params is changed, previous is: %s, "
                               "input is %s"
                               % (str(pre), str(cur)))

        return copy.deepcopy(self._singleton__tiling)

    def set_tiling(self, tiling):
        """Set the tiling to private member variable of Singleton object
        Parameters
        ----------
        tiling: dict
            The setting tiling

        Returns
        -------
        """
        type_list = ["tuning_tiling", "atc_tuning_tiling"]
        if self._singleton__tiling_type not in type_list:
            raise RuntimeError("tiling mode is not tuning tiling, "
                               "current is %s"
                               % str(self._singleton__tiling_type))

        if isinstance(tiling, dict):
            self._singleton__tiling = copy.deepcopy(tiling)
        else:
            raise TypeError('tiling is not a dict.')

    def get_tiling_type(self):
        """Get the tiling type from Singleton object
        Parameters
        ----------

        Returns
        -------
        tiling_type: string
            The tiling type saved in Singleton object
        """
        return copy.deepcopy(self._singleton__tiling_type)

    def set_tiling_type(self, tiling_type):
        """Set the tiling type to private member variable of Singleton object
        Parameters
        ----------
        tiling_type: string
            The setting tiling type

        Returns
        -------
        """
        if isinstance(tiling_type, str):
            self._singleton__tiling_type = copy.deepcopy(tiling_type)
        else:
            raise TypeError('tiling is not a str.')

TILING_INSTANCE = Singleton()
