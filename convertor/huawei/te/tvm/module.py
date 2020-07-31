# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Container of compiled functions of TVM."""
from __future__ import absolute_import as _abs

import struct
from collections import namedtuple

from ._ffi.function import ModuleBase, _set_class_module
from ._ffi.function import _init_api
from ._ffi.libinfo import find_include_path

ProfileResult = namedtuple("ProfileResult", ["mean", "results"])


class Module(ModuleBase):
    """Module container of all TVM generated functions"""

    def __repr__(self):
        return "Module(%s, %x)" % (self.type_key, self.handle.value)

    @property
    def type_key(self):
        """Get type key of the module."""
        return _GetTypeKey(self)

    def get_source(self, fmt=""):
        """Get source code from module, if available.

        Parameters
        ----------
        fmt : str, optional
            The specified format.

        Returns
        -------
        source : str
            The result source code.
        """
        return _GetSource(self, fmt)

    @property
    def imported_modules(self):
        """Get imported modules

        Returns
        ----------
        modules : list of Module
            The module
        """
        nmod = _ImportsSize(self)
        return [_GetImport(self, i) for i in range(nmod)]

    def save(self, file_name, fmt=""):
        """Save the module to file.

        This do not save the dependent device modules.
        See also export_shared

        Parameters
        ----------
        file_name : str
            The name of the file.
        fmt : str
            The format of the file.

        See Also
        --------
        Module.export_library : export the module to shared library.
        """
        _SaveToFile(self, file_name, fmt)


    def time_evaluator(self, func_name, ctx, number=10, repeat=1, min_repeat_ms=0):
        """Get an evaluator that measures time cost of running function.

        Parameters
        ----------
        func_name: str
            The name of the function in the module.

        ctx: TVMContext
            The context we should run this function on.

        number: int
            The number of times to run this function for taking average.
            We call these runs as one `repeat` of measurement.

        repeat: int, optional
            The number of times to repeat the measurement.
            In total, the function will be invoked (1 + number x repeat) times,
            where the first one is warm up and will be discarded.
            The returned result contains `repeat` costs,
            each of which is an average of `number` costs.

        min_repeat_ms: int, optional
            The minimum duration of one `repeat` in milliseconds.
            By default, one `repeat` contains `number` runs. If this parameter is set,
            the parameters `number` will be dynamically adjusted to meet the
            minimum duration requirement of one `repeat`.
            i.e., When the run time of one `repeat` falls below this time, the `number` parameter
            will be automatically increased.

        Note
        ----
        The function will be invoked  (1 + number x repeat) times,
        with the first call discarded in case there is lazy initialization.

        Returns
        -------
        ftimer : Function
            The function that takes same argument as func and returns a ProfileResult.
            The ProfileResult reports `repeat` time costs in seconds.
        """
        try:
            feval = _RPCTimeEvaluator(
                self, func_name, ctx.device_type, ctx.device_id, number, repeat, min_repeat_ms)

            def evaluator(*args):
                """Internal wrapped evaluator."""
                # Wrap feval so we can add more stats in future.
                blob = feval(*args)
                fmt = "@" + ("d" * repeat)
                results = struct.unpack(fmt, blob)
                mean = sum(results) / float(repeat)
                return ProfileResult(mean=mean, results=results)

            return evaluator
        except NameError:
            raise NameError("time_evaluate is only supported when RPC is enabled")


def system_lib():
    """Get system-wide library module singleton.

    System lib is a global module that contains self register functions in startup.
    Unlike normal dso modules which need to be loaded explicitly.
    It is useful in environments where dynamic loading api like dlopen is banned.

    To build system lib function, simply specify target option ```llvm --system-lib```
    The system lib will be available as long as the result code is linked by the program.

    The system lib is intended to be linked and loaded during the entire life-cyle of the program.
    If you want dynamic loading features, use dso modules instead.

    Returns
    -------
    module : Module
        The system-wide library module.
    """
    return _GetSystemLib()


def enabled(target):
    """Whether module runtime is enabled for target

    Parameters
    ----------
    target : str
        The target device type.

    Returns
    -------
    enabled : bool
        Whether runtime is enabled.

    Examples
    --------
    The following code checks if gpu is enabled.

    >>> tvm.module.enabled("gpu")
    """
    return _Enabled(target)


_init_api("tvm.module")
_set_class_module(Module)
