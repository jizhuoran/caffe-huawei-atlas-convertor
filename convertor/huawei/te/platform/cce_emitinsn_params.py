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

CCE EmitInsn params
"""


# pylint: disable=useless-object-inheritance
class CceEmitInsnParams(object):
    '''
    when you use emit_insn dsl, you can put the args you need into this class,
    and use it in cce_intrin_md.py
    '''

    def __init__(self):
        self._params_dict = {}
        self._penetration_dict = {}

    def insert_params(self, params_dict):
        """
        from params_dict, insert params in params dict
        """
        for key in params_dict:
            if key in self._params_dict.keys():
                raise RuntimeError("the emit_insn params exits")

            self._params_dict[key] = params_dict[key]

    def insert_param(self, arg_name, arg_value):
        """
        using arg_name and arg_value to insert params in params dict
        """
        if arg_name in self._params_dict.keys():
            raise RuntimeError("the emit_insn params exits")

        self._params_dict[arg_name] = arg_value

    def get_param(self, arg_name):
        """
        get params from params_dict
        """
        if arg_name in self._params_dict.keys():
            return self._params_dict[arg_name]

        return None

    def clear_param(self):
        """
        clear the params_dict
        """
        # Prepare for penetration enabled params
        temp_dict = {}
        for param in self._penetration_dict:
            if param in self._params_dict and self._penetration_dict[param] > 0:
                temp_dict[param] = self._params_dict[param]
                self._penetration_dict[param] -= 1
            elif param not in self._params_dict:
                raise RuntimeError("Penetration param not found: " + str(param))
        # Clear the dict
        self._params_dict.clear()
        # Recover penetration enabled params
        for param in temp_dict:
            self._params_dict[param] = temp_dict[param]
            if self._penetration_dict[param] <= 0:
                del self._penetration_dict[param]

    def del_param(self, arg_name):
        """
        delete arg_name in params_dict
        """
        if arg_name in self._params_dict.keys():
            del self._params_dict[arg_name]

    def insert_param_with_penetration(self, arg_name, arg_value, pen_value=1):
        """
        insert an arg_name and arg_value that can penetrate clear() for x time(s)
        """
        self._penetration_dict[arg_name] = pen_value
        self.insert_param(arg_name, arg_value)

# pylint: disable=invalid-name
# cceEmitParamsIns is a single instance
cceEmitParamsIns = CceEmitInsnParams()
