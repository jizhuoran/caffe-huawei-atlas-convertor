#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# pylint: disable=invalid-name
"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Utility to invoke ccec compiler in the system
"""
from __future__ import absolute_import as _abs

import sys
import os
import subprocess
import stat
from te.platform import cce_conf as cceconf
from te.platform import cce_build
from te.platform import cce_runtime
from te.platform.cce_params import OUTPUT_PATH_CLASS
from te.platform import cce_params
from te.tvm import _api_internal


def current_build_config():
    """Get the current build configuration."""
    return _api_internal._GetCurrentBuildConfig()


def _get_temp_dir(dirpath, target):
    """Getting the temporary files used in compile and link.

    Parameters
    ----------
    temp : TempDirectory
        instance of class TempDirectory

    target : str
        The target format

    Return
    ------
    temp_code : str
        The temporary code file.

    temp_target : str
        The temporary object file.

    temp_linked_target : str
        The temporary target file.
    """

    if not os.path.isdir(dirpath):
        raise RuntimeError(
            "%s do not exits" % (dirpath))
    temp_code = None
    temp_target = None

    temp_linked_target = None
    if target == "cce_core":
        temp_code = os.path.realpath(
            os.path.join(dirpath, "my_kernel_core.cce"))
        temp_target = os.path.realpath(
            os.path.join(dirpath, "my_kernel_core.o"))
    elif target == "cce_cpu":
        temp_code = os.path.realpath(os.path.join(dirpath, "my_kernel_cpu.cce"))
        temp_target = os.path.realpath(
            os.path.join(dirpath, "my_kernel_cpu_prelink.o"))
        temp_linked_target = os.path.realpath(
            os.path.join(dirpath, "my_kernel_cpu.o"))
    elif target == "cce_cpu_llvm":
        temp_code = os.path.realpath(os.path.join(dirpath, "my_kernel_cpu.ll"))
        temp_target = os.path.realpath(
            os.path.join(dirpath, "my_kernel_cpu_prelink.o"))
        temp_linked_target = os.path.realpath(
            os.path.join(dirpath, "my_kernel_cpu.o"))
    return temp_code, temp_target, temp_linked_target


_ccec_path = None          # pylint: disable=invalid-name


def _get_ccec_path():
    global _ccec_path  # pylint: disable=global-statement, invalid-name
    if _ccec_path is None:
        ccec = "ccec"
        _ccec_path = ccec
        cmd = ["which", ccec]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        _, _ = proc.communicate()
        if proc.returncode != 0:
            _ccec_path = "/usr/local/HiAI/runtime/ccec_compiler/bin/ccec"
    return _ccec_path


def _build_aicore_compile_cmd(cce_product_params, src_file, dst_file):
    """Build the compile command for aicore op.

    Parameters
    ----------
    cce_product_params : TempDirectory
        Instance of class CceProductParams.

    src_file : str
        The file of source code used in compile.

    dst_file : str
        The object file.

    Return
    ------
    cmd : list
        The compile command.

    """
    cce_arch = cce_product_params.getParams("Compiler_arch")

    arch = "cce-aicore-only"
    cce_arch_prefix = "cce-aicore-arch"
    ccec = _get_ccec_path()
    cmd = [ccec,
           "-c",
           "-O2",
           src_file,
           "--%s=%s" % (cce_arch_prefix, cce_arch),
           "--%s" % arch,
           "-o",
           dst_file]
    if cce_product_params.cce_product == "1.60":
        cmd += ["-mllvm", "-cce-aicore-function-stack-size=16000"]
    elif cce_product_params.cce_product == "5.10":
        cmd += ["-mllvm", "-cce-aicore-sk-transform"]
    skt_env = os.getenv('SKT_ENABLE')
    if skt_env == "1":
        if cce_runtime.CceFlag.BatchBindOnly is True:
            cmd += ["-mllvm", "-cce-aicore-sk-transform"]
            cce_runtime.CceFlag.BatchBindOnly = False

    return cmd


def _build_aicpu_compile_cmd(cce_product_params, target, src_file, dst_file):
    """Build the compile command for aicpu op.

    Parameters
    ----------
    cce_product_params : TempDirectory
        Instance of class CceProductParams.

    target : str
        Types of aicpu, cce_cpu or cce_cpu_llvm

    src_file : str
        The file of source code used in compile.

    dst_file : str
        The object file.

    Return
    ------
    cmd : list
        The compile command.

    """
    if cce_product_params.getParams("is_cloud"):
        raise RuntimeError(
            "this platform could not support AICPU")

    env_path = os.getenv('TVM_AICPU_INCLUDE_PATH')
    if not env_path:
        raise RuntimeError(
            "can not find the TVM_AICPU_INCLUDE_PATH environment variable, please config it")

    if not check_env_variable(env_path):
        raise RuntimeError(
            "the TVM_AICPU_INCLUDE_PATH environment variable contains sepcial "
            "code: '|, ;, &, $, &&, ||, >, >>, <' please config it")

    env_path = get_env_real_path(env_path)
    aicpu_support_os = cce_product_params.getParams("Compiler_aicpu_support_os")
    arch = "cce-aicpu-only"
    cce_arch_prefix = "cce-aicpu-arch"
    ccec = _get_ccec_path()
    cmd = [ccec,
           "-c",
           "-O2",
           src_file,
           "--%s" % arch,
           "--%s=%s" % (cce_arch_prefix, "cortex-a55"),
           "-mcpu=cortex-a55", ]
    if target == "cce_cpu_llvm":
        cmd += ["--target=aarch64-hisilicon-cce", "-fPIC"]
    if aicpu_support_os:
        if target == "cce_cpu":
            cmd += ['--cce-aicpu-no-firmware']
            # Safety_checks
            cmd += ['--cce-aicpu-fstack-protector-all', '-fPIC']
        elif target == "cce_cpu_llvm":
            # must specify -mllvm first
            cmd += ['-mllvm', '-cce-aicpu-no-firmware=true']
    for inc_path in env_path.split(':'):
        if inc_path:
            cmd += ["-I%s" % inc_path]
    cmd += ["-o",
            dst_file, ]
    return cmd

# pylint: disable=too-many-branches, unused-argument
def _build_aicpu_link_cmd(cce_product_params, target, src_file, dst_file,
                          lib_name):
    """Build the link command for aicpu op.

    Parameters
    ----------
    cce_product_params : TempDirectory
        Instance of class CceProductParams.

    target : str
        Types of aicpu, cce_cpu or cce_cpu_llvm

    src_file : str
        The file of source code used in compile.

    dst_file : str
        The object file.

    lib_name : str
        lib name of each op, using in soname.

    Return
    ------
    cmd : list
        The link command.

    """
    env_path = os.getenv('TVM_AICPU_LIBRARY_PATH')
    if not env_path:
        raise RuntimeError(
            "can not find the TVM_AICPU_LIBRARY_PATH environment variable, please config it")

    if not check_env_variable(env_path):
        raise RuntimeError(
            "the TVM_AICPU_INCLUDE_PATH environment variable contains sepcial "
            "code: '|, ;, &, $, &&, ||, >, >>, <' please config it")

    env_path = get_env_real_path(env_path)
    # bool to indicate aicpu support os in a specified product
    aicpu_support_os = cce_product_params.getParams("Compiler_aicpu_support_os")
    # -static: do not link against shared libraries
    # -m :Set target emulation, aicpulinux,aicorelinux,aarch64linux
    #          (see ld.lld --help for detail)
    lib_type = ["-static", "-m", "aicpulinux", "-Ttext", "0"]
    if aicpu_support_os:
        # create a shared library
        lib_type = ["-shared", "-m", "aarch64linux"]

    cmd = ["ld.lld"] + lib_type + [src_file, "-ltvm_aicpu", "-lm", "-lc", ]
    # add library search path
    if aicpu_support_os:
        for lib_path in env_path.split(PATH_DELIMITER):
            if lib_path and "aicpu_lib" not in lib_path:
                cmd += ["-L%s" % lib_path]
    else:
        for lib_path in env_path.split(':'):
            if lib_path:
                cmd += ["-L%s" % lib_path,
                        "-L%s/../../../../../toolchain/artifacts/aicpu_lib" % lib_path, ]

    # add include seach path
    env_path = os.getenv('TVM_AICPU_INCLUDE_PATH')
    if not env_path:
        raise RuntimeError(
            "can not find the TVM_AICPU_LIBRARY_PATH environment variable, please config it")

    # if aicpu has deployed OS
    if aicpu_support_os:
        env_path = os.getenv('TVM_AICPU_OS_SYSROOT')
        if not env_path:
            raise RuntimeError(
                "can not find the TVM_AICPU_OS_SYSROOT environment variable, please config it")
        if not check_env_variable(env_path):
            raise RuntimeError(
                "the TVM_AICPU_OS_SYSROOT environment variable contains sepcial "
                "code: '|, ;, &, $, &&, ||, >, >>, <' please config it")

        env_path = get_env_real_path(env_path)
        lib_includes = env_path
        if not lib_includes:
            raise RuntimeError(
                "can not find the TVM_AICPU_OS_SYSROOT environment variable, please config it")

        if lib_includes:
            if lib_includes == "/usr/aarch64-linux-gnu":  # developerkit
                cmd += ['-L%s/lib' % lib_includes.strip()]
            else:
                cmd += ['--sysroot=%s' % lib_includes.strip()]
                cmd += ['-L%s/usr/lib64' % lib_includes.strip()]
        cmd += ['-soname=%s.so' % lib_name]
        # Safety_checks
        cmd += ['-z', 'relro', '-z', 'now', '-z', 'noexecstack']

    cmd += ["-o", dst_file]
    return cmd


def _run_cmd_stackoverflow_case(cmd, cmd_type):
    """Run a shell commond (only support linux) for stack overflow case in
     aicore.
    The current compiler's hardware instructions are not friendly to large
    immediate Numbers.
    Add compile options "-mllvm", "-disable-machine-licm" can avoid this
    problem, but it can degrade performance.

    Parameters
    ----------
    cmd : list
        Command to be run.

    cmd_type : str
        Comand type uesd in running.

    Return
    ------
    out : str
        Standard output.

    """
    cmd.insert(3, "-mllvm")
    cmd.insert(4, "-disable-machine-licm")
    cmd.insert(5, "-mllvm")
    cmd.insert(6, "--cce-aicore-jump-expand=true")

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    if proc.returncode != 0:
        msg = "%s error:\n" % (cmd_type)
        raise RuntimeError(msg)
    return out


def _run_cmd(cmd, cmd_type):
    """Run a shell commond (only support linux).

    Parameters
    ----------
    cmd : list
        Command to be run.

    cmd_type : str
        Comand type uesd in running.

    Return
    ------
    out : str
        Standard output.

    """
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    if proc.returncode != 0:
        # stack overflow case for aicore compile
        if cmd_type == "compile" and "--cce-aicore-only" in cmd:
            _run_cmd_stackoverflow_case(cmd, cmd_type)
        else:
            msg = "%s error:\n" % (cmd_type)
            raise RuntimeError(msg)
    return out


# pylint: disable=unused-argument
def _get_kernel_name(kernel_name, target, path_target):
    """Getting the lib name of op.

    Parameters
    ----------
    code : str
        Source code.

    target : str
        Target format.

    path_target : str
        The target path of op.

    Return
    ------
    kernel name : str
        kernel name of op

    """
    kernel_name = ''
    try:
        if path_target:
            kernel_name = path_target.split(".")[0].split('/')[-1]
        else:
            return kernel_name
    except IndexError as err:
        raise RuntimeError("Getting kernel name failed", err)
    return kernel_name

def _check_tmpdir(dirpath):
    if sys.platform.startswith('linux'):
        if not (dirpath.startswith('/tmp') \
                or dirpath.startswith('/var/tmp') \
                or dirpath.startswith('/usr/tmp') \
                or dirpath.startswith(os.path.realpath('/tmp')) \
                or dirpath.startswith(os.path.realpath('/var/tmp')) \
                or dirpath.startswith(os.path.realpath('/usr/tmp'))):
            raise ValueError("dir path is invalid")
    elif sys.platform.startswith("win32"):
        # pylint: disable=anomalous-backslash-in-string
        if not dirpath.find("AppData\Local\Temp"):
            raise ValueError("dir path is invalid")
    else:
        raise ValueError("Platform %s is not support now" % sys.platform)

def _save_cce_file(kernel_name, temp_file):
    """save the temp cce file

    Parameters
    ----------
    kernel_name : str
        the cce file name is kernel_name.cce

    temp_file: str
        the temp cce file

    Returns
    -------
    None
    """
    cce_path = OUTPUT_PATH_CLASS.output_path
    if cce_path is not None:
        cce_file_path = os.path.realpath("%s/%s%s" %
                                         (os.path.normpath(cce_path),
                                          kernel_name, ".cce"))
        dir_path = os.path.dirname(cce_file_path)
        if not os.path.isdir(dir_path):
            raise RuntimeError("cce file path not exist")
        copy_cmd = ["cp", temp_file, cce_file_path]
        os.chmod(dir_path, stat.S_IRWXU + stat.S_IRGRP)
        _run_cmd(copy_cmd, "copy")

# pylint: disable=too-many-locals, too-many-arguments, unused-argument
def compile_cce(dirpath,
                kernel_name,
                target="cce_core",
                arch=None,
                options=None,
                path_target=None):
    """Compile cce code with ccec from env.

    Parameters
    ----------
    code : str
        The cce code.

    target : str
        The target format

    arch : str
        The architecture

    options : str
        The additional options

    path_target : str, optional
        Output file.

    Return
    ------
    binary : bytearray
        The bytearray of the binary
    """
    if target not in ["cce_core", "cce_cpu", "cce_cpu_llvm"]:
        raise ValueError("Unknown architecture, does not support now")

    dirpath = os.path.realpath(dirpath)
    _check_tmpdir(dirpath)

    temp_code, temp_target, temp_linked_target = _get_temp_dir(dirpath, target)
    if not os.path.exists(temp_code):
        raise RuntimeError("tmp code file not exits")

    if current_build_config().save_temp_cce_file:
        _save_cce_file(kernel_name, temp_code)

    if current_build_config().dump_cce_code:
        fi = open(temp_code)
        print(fi.read())
        fi.close()

    # get instance of CceProductParams used in compile and link
    cce_product_params = cceconf.CceProductParams()

    # compile step, both aicore and aicpu
    file_target = path_target if path_target else temp_target
    file_target = file_target if target == "cce_core" else temp_target

    compile_cmd = []
    if target == "cce_core":
        compile_cmd = _build_aicore_compile_cmd(cce_product_params, temp_code,
                                                file_target)
    elif target in ('cce_cpu', 'cce_cpu_llvm'):
        compile_cmd = _build_aicpu_compile_cmd(cce_product_params, target,
                                               temp_code, file_target)

    _run_cmd(compile_cmd, "compile")

    # link step, for aicpu only
    if target in ('cce_cpu', 'cce_cpu_llvm'):
        path_target = path_target if path_target else temp_linked_target
        kernel_name = _get_kernel_name(kernel_name, target, path_target)
        link_cmd = _build_aicpu_link_cmd(cce_product_params,
                                         target,
                                         file_target,
                                         path_target,
                                         kernel_name)
        _run_cmd(link_cmd, "link")

    # pylint: disable=pointless-statement
    with open(file_target, "rb") as f:
        try:
            return bytearray(f.read())
        except Exception as e:
            raise RuntimeError("get byte array error", e)
        finally:
            f.close


def check_env_variable(env_variable):
    '''
    check if the enviroment variable contains special characters.
    :param env_variable: enviroment variable
    :return:True or False
    '''
    if sys.platform.startswith('linux'):
        spe_code = ['|', ';', '&', '$', '&&', '||', '>', '>>', '<', '`', '\\',
                    '!']
    else:
        spe_code = ['|', '&', '$', '&&', '||', '>', '>>', '<', '`', '!']

    for i in spe_code:
        if i in repr(env_variable):
            return False

    return True


def get_env_real_path(env_variable):
    """
    realpath the env_variable
    :param env_variable: enviroment variable
    :return:real path env
    """
    env_variable_real_path = ""
    for env_path in env_variable.split(PATH_DELIMITER):
        env_real_path = os.path.realpath(env_path)
        if env_real_path is not None:
            env_variable_real_path += env_real_path + PATH_DELIMITER

    if env_variable_real_path == "":
        raise RuntimeError(
            "can not find the environment variable, please config it")

    return env_variable_real_path[0:len(env_variable_real_path) - 1]


# default delimiter for env path
PATH_DELIMITER = ':'
if sys.platform.startswith('linux'):
    PATH_DELIMITER = ':'
elif sys.platform.startswith('win32'):
    PATH_DELIMITER = ';'
else:
    raise ValueError('Platform % is not support now' % sys.platform)
