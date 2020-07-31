#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright 2018 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Compute Manager for fusion
"""

import importlib
from te import tvm
from .cce_build import build_config as default_build_config


# pylint: disable=useless-object-inheritance, too-many-instance-attributes
class FusionManager(object):
    """Manage computes which are registered
    Save and call compute function by their registered key
    """

    def __init__(self):
        """init"""
        self._build_cfg = "enable"
        self._current_op_name = ""
        self._current_op_func_name = ""
        self._current_op_pattern = "Opaque"
        self._op_build_type = {}
        self._op_args = {}
        self._op_kwds = {}
        self._op_compute = {}
        self.fusion_build_config = default_build_config
        self.res_map = {}

    def register(self, register_name):
        """Register compute

        Parameters
        ----------
        register_name : string
            register_name to call compute.

        Returns
        -------
        decorator : decorator
            decorator to register compute.
        """

        def decorator(func):
            """Save op function name for compute name

            Parameters
            ----------
            func : string
                op function name

            Returns
            -------
                op function name.
            """
            self._op_compute[register_name] = func
            return func

        return decorator

    def has_op_compute(self, register_name):
        """Whether compute_manager has this compute

        Parameters
        ----------
        register_name : string
            compute_name to call compute.

        Returns
        -------
        has_op_compute : bool
            Whether compute_manager has this compute.
        """
        return register_name in self._op_compute

    def get_op_compute(self, register_name, *args, **kwds):
        """Get op compute

        Parameters
        ----------
        register_name : string
            register_name to call func.
        *args, **kwds
            op_params.

        Returns
        -------
        op_compute : compute
            compute corresponding to the compute_name.
        """
        return self._op_compute[register_name](*args, **kwds)

    def set_current_op_name(self, op_name):
        """Set current op_name

        Parameters
        ----------
        op_name : string
            update current op_name to save op_params.
        """
        self._current_op_name = op_name

    def set_op_params(self, *args, **kwds):
        """Set current op_name's op_params

        Parameters
        ----------
        *args, **kwds
            save current op_name's op_params.
        """
        self._op_args[self._current_op_name] = args
        self._op_kwds[self._current_op_name] = kwds

    def set_op_build_type(self, args_type):
        """Get current op_name's build type, it is singlebuild or prebuild

        Parameters
        ----------
        args_type : string
            singlebuild or prebuild
        """
        self._op_build_type[self._current_op_name] = args_type

    def get_op_build_type(self, register_name):
        """Get current op_name's build type

        Parameters
        ----------
        register_name : string
            key to get build type

        Returns
        -------
        args
            current op_name's build type.
        """
        if register_name in self._op_build_type:
            return self._op_build_type[register_name]
        return ""

    def set_op_res(self, res):
        """Get current op_name's build type, it is singlebuild or prebuild

        Parameters
        ----------
        args_type : string
            singlebuild or prebuild
        """
        if get_build_cfg() == "enable" and self._current_op_name not in self.res_map:
            res_op = []
            op_outputs = []
            if isinstance(res, list):
                for single_res in res:
                    res_op.append(single_res.op)
                    op_outputs.append(single_res.op.name)
            else:
                res_op.append(res.op)
                op_outputs.append(res.op.name)
            sch = tvm.create_schedule(res_op)
            sch.cce_special = {"op_outputs": op_outputs}
            self.res_map[self._current_op_name] = sch

    def get_op_res(self, key):
        """Get current op_name's build type

        Parameters
        ----------
        register_name : string
            key to get build type

        Returns
        -------
        args
            current op_name's build type.
        """
        return self.res_map.get(key, None)

    def get_op_args(self, op_name):
        """Get current op_name's op_args

        Parameters
        ----------
        op_name : string
            key to get op_args

        Returns
        -------
        args
            save current op_name's op_args.
        """
        return self._op_args[op_name]

    def get_op_kwds(self, op_name):
        """Get current op_name's op_kwds

        Parameters
        ----------
        op_name : string
            key to get op_kwds

        Returns
        -------
        kwds
            save current op_name's op_kwds.
        """
        return self._op_kwds[op_name]

    def init_current_op_pattern(self):
        """Init current op's pattern"""

        self._current_op_pattern = "Opaque"

    def set_current_op_pattern(self, op_pattern):
        """Set current op's pattern

        Parameters
        ----------
        op_pattern : string
            current single op's pattern.
        """
        self._current_op_pattern = op_pattern

    def get_current_op_pattern(self):
        """Get current single op's pattern

        Returns
        ----------
        op_pattern : string
            current single op's pattern.
        """
        if self.has_op_compute(self.get_current_op_func_name()):
            return self._current_op_pattern
        return "Opaque"

    def set_current_op_func_name(self, op_func_name):
        """Set current op's func name

        Parameters
        ----------
        op_func_name : string
            current single op's func name.
        """
        self._current_op_func_name = op_func_name

    def get_current_op_func_name(self):
        """Get current single op's func name

        Returns
        ----------
        op_func_name : string
            current single op's func name.
        """
        return self._current_op_func_name

    def get_fuse_info(self):
        """Check whether this op will be fused

        Returns
        ----------
        True : bool
            this op will be fused
        False : bool
            this op will not be fused
        """
        if self._current_op_name \
                and self._current_op_func_name \
                and (self._current_op_name in self._op_args
                     or self._current_op_name in self._op_kwds):
            return True
        return False

    def set_build_cfg(self, op_build_cfg):
        """Set current op's build switch

        Parameters
        ----------
        op_pattern : string
            current single op's switch.
        """
        self._build_cfg = op_build_cfg

    def get_build_cfg(self):
        """Get current single op's switch

        Returns
        ----------
        op_pattern : string
            current single op's switch.
        """
        return self._build_cfg


# pylint: disable=invalid-name
# Singleton for managing all registered compute
fusion_manager = FusionManager()


def set_current_op_name(op_name):
    """Set current op_name, external interface for C call python

    Parameters
    ----------
    op_name : string
        update current op_name to save op_params.

    Returns
    -------
    succ_flag : boolean
        end of execution
    """
    fusion_manager.set_current_op_name(op_name)


def set_current_op_func_name(op_func_name):
    """Set current op's func name, external interface for C call python

    Parameters
    ----------
    op_func_name : string
        current single op's func name.

    Returns
    -------
    succ_flag : boolean
        end of execution
    """
    fusion_manager.set_current_op_func_name(op_func_name)


def set_op_params(*args, **kwds):
    """Set current name's op_params, external interface for C call python

    Parameters
    ----------
    *args, **kwds
        save current op_name's op_params.

    Returns
    -------
    succ_flag : boolean
        end of execution
    """
    fusion_manager.set_op_params(*args, **kwds)


def set_op_build_type(args_type):
    """Get current op_name's build type

    Parameters
    ----------
    args_type : string
        build type need to be set
    """
    fusion_manager.set_op_build_type(args_type)


def get_op_build_type(register_name):
    """Get current op_name's op_args

    Parameters
    ----------
    register_name : string
        key to get build type

    Returns
    -------
    args
        return current op_name's build type.
    """
    return fusion_manager.get_op_build_type(register_name)


def set_op_res(res_val):
    """Get current op_name's build type

    Parameters
    ----------
    args_type : string
        build type need to be set
    """
    fusion_manager.set_op_res(res_val)


def get_op_res(key):
    """Get current op_name's op_args

    Parameters
    ----------
    register_name : string
        key to get build type

    Returns
    -------
    args
        return current op_name's build type.
    """
    return fusion_manager.get_op_res(key)


def op_build_cfg_en():
    """Set current name's op_params, enable  build .o

    Parameters
    ----------

    Returns
    -------
    succ_flag : boolean
        end of execution
    """
    fusion_manager.set_build_cfg("enable")


def op_build_cfg_dis():
    """Set current name's op_params, disable build .o

    Parameters
    ----------

    Returns
    -------
    succ_flag : boolean
        end of execution
    """
    fusion_manager.set_build_cfg("disable")


def get_build_cfg():
    """Get current name's op_params, build .o or not

    Parameters
    ----------

    Returns
    -------
    succ_flag : boolean
        end of execution
    """
    return fusion_manager.get_build_cfg()


def init_op_pattern():
    """init current name's pattern

    Parameters
    ----------

    Returns
    -------
    op pattern value
        end of execution
    """
    fusion_manager.init_current_op_pattern()


def get_op_pattern():
    """Get current name's pattern

    Parameters
    ----------

    Returns
    -------
    op pattern value
        end of execution
    """
    return fusion_manager.get_current_op_pattern()


def prebuild_op(op_module, op_func_name, op_args):
    """Prebuild Op

    Parameters
    ----------
    op_module: op module name
    op_args: op args

    Returns
    -------
    op pattern value
        end of execution
    """
    opm = importlib.import_module(op_module)
    opfunc = getattr(opm, op_func_name)

    set_current_op_func_name(op_func_name)  # for pattern
    init_op_pattern()  # for init pattern to Opaque
    op_build_cfg_dis()  # for cce build
    opfunc(*op_args)
    op_build_cfg_en()
    pattern = get_op_pattern()
    return pattern


def save_op_params(op_name, op_build_type, op_args):
    """Save op params

    Parameters
    ----------
    op_name: op name
    op_func_name: op function name
    op_args: op args

    Returns
    -------
    """
    set_current_op_name(op_name)
    set_op_build_type(op_build_type)  # for fusion
    set_op_params(*op_args)


def get_fusion_build_cfg():
    """get build_config used by fusion manager

    Returns
    -------
    fusion_manger build_config:
    """
    return fusion_manager.fusion_build_config


def set_fusion_build_cfg(cfg):
    """set build_config used by fusion manager

    Parameters
    ----------
    cfg : build_config

    """
    fusion_manager.fusion_build_config = cfg


def reset_fusion_build_cfg():
    """reset build_config used by fusion manager
    """
    fusion_manager.fusion_build_config = default_build_config
