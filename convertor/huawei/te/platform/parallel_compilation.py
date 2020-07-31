#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

support parallel compilation
"""

import multiprocessing as mp
import queue
import time
import os
import importlib
import signal
import datetime
import zlib
import pickle
import sys
import subprocess
import logging
import json
import threading
from configparser import ConfigParser
import te.platform.log_util as telog
import te.platform.fusion_manager as fusion_manager
import te.platform.cce_policy as cce_policy


def init_logger():
    """
    init logger module
    """
    logging.raiseExceptions = False
    logmap = {
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG
    }
    loglevel = os.getenv('TE_LOGLEVEL', 'WARNING')
    loglevel = logmap[loglevel] if loglevel in logmap else logging.WARNING

    newlogger = logging.getLogger('PCOMPILE')
    newlogger.propagate = False
    newlogger.setLevel(loglevel)
    log_ch = logging.StreamHandler()
    log_fmt = logging.Formatter('%(asctime)s %(process)d %(name)s %(message)s')
    log_ch.setFormatter(log_fmt)
    newlogger.addHandler(log_ch)
    return newlogger


logger = init_logger()          # pylint: disable=invalid-name


# 'pylint: disable=too-few-public-methods
class Counter:
    """
    Atomic counter
    """
    counter = 0
    locker = threading.Lock()

    @staticmethod
    def next():
        """
        get next counter
        :return: next counter
        """
        with Counter.locker:
            Counter.counter += 1
        return Counter.counter


def mygetattr(obj, name):
    """
    get object attr recursively
    :param obj: python object
    :param name: attr name
    :return: attr
    """
    name_list = name.split(".")
    while name_list:
        obj = getattr(obj, name_list[0])
        name_list = name_list[1:]
    return obj


def mysetattr(obj, name, value):
    """
    get object attr recursively
    :param obj:  python object
    :param name: attr name
    :param value: attr value
    :return: None
    """
    name_list = name.split(".")
    target_obj = mygetattr(obj, ".".join(name_list[:-1]))
    target_attr = name_list[-1]
    setattr(target_obj, target_attr, value)


def excepthook_silent(etype, value, tback):  # pylint: disable=unused-argument
    """
    excepthook to print nothing
    """


def worker_sigint_handler(signum, frame):  # pylint: disable=unused-argument
    """
    worker process just quit when Ctrl-C pressed
    """
    # logging module uses reentrant threading.Rlock,
    # can be safely used in signal handler
    logger.warning('Ctrl-C pressed, worker process quiting...')
    os._exit(1)  # pylint: disable=protected-access


def exec_compilation_task(worker_env, task_env):
    """
    compilation task worker entry
    :param socinfo: soc_version, core_type, core_num, l1size
    :param task_env: tuple of task queue, pipe, etc...
    :return: None
    """
    # just quit, avoid KeyboardInterrupt traceback printing mess
    signal.signal(signal.SIGINT, worker_sigint_handler)

    socver, coretype, corenum, l1fusion, l2mode, l2fusion = worker_env
    cce = importlib.import_module("te.platform.cce_conf")

    cce.te_set_version(socver, coretype, corenum, l1fusion, l2mode, l2fusion)
    logger.info("socinfo:%s, l2mode:%s", worker_env, l2mode)

    importlib.import_module("te.platform.fusion_manager")
    importlib.import_module("te.platform.fusion_util")
    worker = TaskWorker(task_env)
    worker.loop()


def get_multi_process_count(tune_mode):
    """
    get compilation worker number from ini conf file
    :return: compilation worker number
    """
    try:
        if 'TE_PARALLEL_COMPILER' in os.environ:
            count = int(os.getenv('TE_PARALLEL_COMPILER'))
        else:
            home_path = os.getenv('HOME')
            config_file_path = os.path.join(home_path, ".tbe_build.ini")
            config_file = ConfigParser()
            config_file.read(config_file_path)
            count = config_file.getint('compilation', 'max_parallel_jobs')
        count = max(0, min(count, len(os.sched_getaffinity(0))))
        # always enable async when RL Tune
        if tune_mode == "RL" and count == 0:
            count = 1
        return count
    except:  # pylint: disable=bare-except
        return 8


def set_main_info():
    """
    set __file__ and name of main to None
    :return: (orignal main module name, path)
    """
    main_module = sys.modules['__main__']
    main_mod_name = getattr(main_module.__spec__, "name", None)
    main_path = getattr(main_module, '__file__', None)
    if main_mod_name is not None:
        setattr(main_module.__spec__, "name", None)

    if main_path is not None:
        setattr(main_module, '__file__', None)
    return (main_mod_name, main_path)


def restore_main_info(name, path):
    """
    restor main module name and path
    """
    main_module = sys.modules['__main__']
    if name is not None:
        setattr(main_module.__spec__, "name", name)
    if path is not None:
        setattr(main_module, '__file__', path)


def guess_pyexe_path(mp_ctx):
    """
    search for a suitable python exe, should be called before any multiprocessing calls
    :param mp_ctx: multiprocessing module
    """
    pylibver = sys.version_info
    pyver = subprocess.run([sys.executable, '-V'], stderr=subprocess.DEVNULL,
                           stdout=subprocess.PIPE).stdout.decode().split()[1].split('.')
    if pyver[0] == str(pylibver.major) and pyver[1] == str(pylibver.minor):
        return

    targetpy = "python" + str(pylibver.major) + "." + str(pylibver.minor)
    binpath = [os.path.join(path, targetpy) for path in
               os.environ['PATH'].split(os.pathsep) + ['/usr/bin', '/usr/local/bin']]

    for path in binpath:
        if os.path.isfile(path):
            mp_ctx.set_executable(path)
            logger.info("guessed python path:%s", path)
            return


class OpCompiler:
    """
    OpCompiler
    """
    compiler = None

    def __init__(self, embedding, worker_num, worker_env):
        """
        init
        :param task_env:
        :param worker_list:
        """
        self._task_dispatcher = None
        # '{graphid: {taskid: desc}}
        self._task_running = {}

        # '{graphid: {taskid: result}}
        self._task_finished = {}
        self._worker_num = worker_num
        self._worker_env = worker_env
        self._embedding = embedding
        self.finished_task_queue = None
        self.live_checker = None
        self.termination_event = None
        OpCompiler.compiler = self

    def start(self):
        """
        start worker compiler process
        """
        if self._task_dispatcher is not None:
            return self._worker_num, self.finished_task_queue, \
                    self.live_checker, self.termination_event

        if self._embedding:
            guess_pyexe_path(mp)

        ctx = mp.get_context("forkserver")
        task_queue = ctx.Queue()
        self.finished_task_queue = ctx.Queue()
        worker_list = []
        self.termination_event = ctx.Event()
        self.live_checker = ctx.Pipe()
        data_queue = []

        # multiprocessing will access sys.argv, if sys.argv not exist,
        # exception raised and can not be caught here
        if not hasattr(sys, "argv"):
            sys.argv = ['']

        # Child process of py multiprocessing will import all modules imported
        # by parent, which is unnecessary and problematic, here is a hack to
        # bypass it.
        main_mod_name, main_path = set_main_info()

        for _ in range(0, self._worker_num):
            new_queue = ctx.Queue()
            worker = \
                ctx.Process(target=exec_compilation_task,
                            args=(self._worker_env,
                                  (task_queue, self.finished_task_queue,
                                   new_queue, self.termination_event,
                                   self.live_checker[0])),
                            daemon=True)
            worker.start()
            worker_list.append(worker)
            data_queue.append(new_queue)
            self._task_dispatcher = \
                TaskDispatcher((task_queue, self.finished_task_queue,
                                data_queue, self.termination_event,
                                self.live_checker), worker_list)

        restore_main_info(main_mod_name, main_path)
        return self._worker_num, self.finished_task_queue, self.live_checker, \
                self.termination_event

    def destory(self):
        """
        deinit multi compilation process
        :return: None
        """
        dispatcher = self._task_dispatcher
        worker_list = dispatcher.worker_list
        dispatcher.term_event.set()

        time.sleep(0.2)
        for worker in worker_list:
            if worker.is_alive():
                worker.terminate()
        time.sleep(0.1)
        for worker in worker_list:
            if worker.is_alive():
                os.kill(worker.pid, signal.SIGKILL)
        self._task_dispatcher = None
        self._task_running = {}

    def is_worker_alive(self):
        """
        check wether all worker processes are alive
        :return:
        """
        worker_list = self._task_dispatcher.worker_list
        all_alive = True
        for worker in worker_list:
            if not worker.is_alive():
                logger.warning("worker process %s died. exitcode %s",
                               worker.pid, worker.exitcode)
                all_alive = False
        return all_alive

    def get_finished_task(self, graphid=None, taskids=None):
        """
        return finished compilation task
        :return:
        """
        try:
            while True:
                task_res = self._task_dispatcher.get_result(False)
                gid = task_res['graph_id']
                tid = task_res['task_id']
                self.save_finished_task(gid, tid, task_res)
        except queue.Empty:
            pass

        # if any worker process dead, all task will be markded as failed
        if not self.is_worker_alive():
            for gid, tasks in list(self._task_running.items()):
                for tid, task_desc in list(tasks.items()):
                    errmsg = "compiler process died"
                    task_res = gen_task_res(0, gid, tid, 1,
                                            'FatalError', errmsg,
                                            err_args=task_desc)
                    self.save_finished_task(gid, tid, task_res)
            self._task_running.clear()

        res = []
        if graphid is not None:
            task_res = self._task_finished.get(graphid)
            if task_res is None:
                return res

            if taskids is None:
                res = list(task_res.values())
                del self._task_finished[graphid]
            else:
                res = [task_res.pop(tid, None) for tid in taskids]
                if task_res == {}:
                    del self._task_finished[graphid]
        else:
            for gid, tasks in self._task_finished.items():
                res.extend(list(tasks.values()))
            self._task_finished.clear()

        return res

    def update_running_task(self, task):
        """
        update task to _task_running
        :param task:
        """
        runnings = self._task_running.setdefault(task.graph_id, {})
        running = runnings.get(task.task_id)
        if running is not None:
            logger.warning("task already exist, dispatch failed. %d:%d",
                           task.graph_id, task.task_id)
            return
        runnings[task.task_id] = task.desc()

    def dispatch_task(self, task):
        """
        dispatch task to workers
        :param task:
        """
        self._task_dispatcher.dispatch(task)
        self.update_running_task(task)

    def sync_data(self, data):
        """
        sync data to all workers
        :param data:
        """
        self._task_dispatcher.sync_data(data)


    def clear_running_task(self, gid, tid):
        """
        clear running task
        :param gid: task graphid
        :param tid: task taskid
        """
        tasks_in_gid = self._task_running.get(gid)
        if tasks_in_gid is None:
            logger.warning("task finished, but graphid not found. %d:%d",
                           gid, tid)
            return False

        running = tasks_in_gid.get(tid)
        if running is None:
            logger.warning("task finished, but taskid not found. %d:%d",
                           gid, tid)
            return False

        del tasks_in_gid[tid]
        return True


    def save_finished_task(self, gid, tid, res):
        """
        save finished task
        :param gid: task graphid
        :param tid: task taskid
        :param res: task result
        """

        if not self.clear_running_task(gid, tid):
            return

        finished_task_in_gid = self._task_finished.setdefault(gid, {})
        finished_task_in_gid[tid] = res


# 'pylint: disable=too-few-public-methods
class DeferredOpRes:
    """
    DeferredOpRes
    """
    def __init__(self, gid, tid):
        """
        init DeferredOpRes
        :param gid
        :param tid
        """
        self._gid = gid
        self._tid = tid
        self._res = None

    def get(self):
        """
        get Op compilation result
        :return: None if still runing, others if finished
        """
        if self._res is not None:
            return self._res

        res = OpCompiler.compiler.get_finished_task(self._gid, [self._tid])
        res = res[0]
        if res is not None:
            self._res = res

        return self._res


def init_multi_process_env(embedding, socinfo, tune_mode):
    """
    init multi compilation process
    :param embedding: if is embedding python
    :param socinfo:
    :param l2mode:
    :return: compilation worker number
    """
    process_count = get_multi_process_count(tune_mode)
    if process_count <= 0:
        return 0, None, None, None

    compiler = OpCompiler(embedding, process_count, socinfo)
    return compiler.start()


def deinit_multi_process_env():
    """
    deinit multi compilation process
    :return: None
    """
    compiler = OpCompiler.compiler
    compiler.destory()


def get_finished_compilation_task(graph_id):
    """
    return finished compilation task
    :return:
    """
    compiler = OpCompiler.compiler
    return compiler.get_finished_task(graph_id)


# 'pylint: disable=too-many-arguments
def gen_task_res(ttype, gid, tid, status_code, result, msg, **kwds):
    """
    gen_task_res
    :return: task result
    """
    res = {
        'type': ttype,
        'graph_id': gid,
        'task_id': tid,
        'status_code': status_code,
        'result': result,
        'info_msg': msg
    }

    for key, value in kwds.items():
        res[key] = value

    return res


# pylint: disable=too-many-instance-attributes
class TaskDispatcher:
    """
    Task Dispatcher
    """
    def __init__(self, task_env, worker_list):
        """
        init
        :param task_env:
        :param worker_list:
        """
        self._task_queue, \
            self._fin_task_queue, \
            self._data_queue, \
            self.term_event, \
            self._live_checker = task_env
        self.worker_list = worker_list
        self._data_sync_count = 0
        self._concurrent = 0

    def get_result(self, block=True):
        """
        get result form finished task queue
        :param block:
        :return:
        """
        task = self._fin_task_queue.get(block)
        self._concurrent -= 1
        return task

    def dispatch(self, task):
        """
        dispatch task to compilation worker
        :param task:
        :return:
        """
        tqueue = self._task_queue
        task.set_data_sync_count(self._data_sync_count)
        tqueue.put(task, True)
        self._concurrent += 1

    def sync_data(self, data_task):
        """
        sync data to compilation worker
        :param data_task:
        :return:
        """
        for dqueue in self._data_queue:
            dqueue.put(data_task, True)
        self._data_sync_count += 1


# pylint: disable=too-many-instance-attributes
class TaskWorker:
    """
    Task runner
    """
    def __init__(self, task_env):
        """
        init
        :param task_env:
        """
        self._task_queue, \
            self._fin_task_queue, \
            self._data_queue, \
            self.term_event, \
            self._live_checker = task_env
        self._block_timeout = 2
        self._data_synced = 0
        self._start = None
        self._end = None
        self._delta = datetime.timedelta()
        self._count = 0

    def do_sync_data(self, block=False, timeout=2):
        """
        load synced data from dispatcher process
        :param block:
        :param timeout:
        :return:
        """
        data_task = self._data_queue.get(block, timeout)
        data_task.run()
        self._data_synced += 1

    def try_sync_data(self):
        """
        try sync data non-blocking
        :return:
        """
        # sync as much as possible
        try:
            while True:
                self.do_sync_data()
        except queue.Empty:
            return

    def mandatory_sync_data(self, count):
        """
        sync exactly count data
        :param count:
        :return:
        """
        # sync exactly 'count' data
        # if there's no enough data, raise exception
        try:
            for _ in range(0, count):
                self.do_sync_data(True, 60)
        except queue.Empty:
            logger.warning("syncing mandatory data failed. count: %d/%d",
                           count, self._data_synced)


    def loop(self):
        """
        main loop
        :return:
        """
        while not self.term_event.is_set():
            try:
                # check dispatcher process is alive
                if self._live_checker.poll():
                    self._live_checker.recv()

                # CAUTION: task dispatcher MUST dispatch data_sync task first
                try:
                    task = self._task_queue.get(True, self._block_timeout)
                except queue.Empty:
                    task = None

                if task is None:
                    self.try_sync_data()
                    continue

                if self._start is None:
                    self._start = datetime.datetime.now()

                count = task.check_need_sync()
                self.mandatory_sync_data(count - self._data_synced)

                if task.l1size > 0:
                    cce_policy.set_L1_info("op_L1_space", task.l1size)
                res = task.run()
                if task.l1size > 0:
                    cce_policy.set_L1_info("op_L1_space", -1)  # reset l1 space

                self._count += 1
                if res is not None:
                    self._fin_task_queue.put(res)

                self._end = datetime.datetime.now()

            except EOFError:
                logger.warning("Master process dead. worker process quiting..")
                # Avoid 'Broken PIPE' exception msg of multiprocessing module,
                # we are quiting anyway.
                sys.excepthook = excepthook_silent
                break


class OpTask:
    """
    Base class of various parallel task
    """
    def __init__(self, timeout_ms=2000):
        self._timeout_ms = timeout_ms
        self._data_sync_count = 0
        self.l1size = -1
        self.res = []

    def check_need_sync(self):
        """
        check if need to sync data before do this op task
        :return:
        """
        return self._data_sync_count

    def set_data_sync_count(self, count):
        """
        set the exactly number of data need to sync
        :param count:
        :return:
        """
        self._data_sync_count = count

    def set_l1size(self, l1size):
        """
        set l1 size when compile op
        :param count:
        :return:
        """
        self.l1size = l1size

    def run(self):
        """
        should overide in sub class
        """


class PySysPathTask(OpTask):
    """
    task to add directories to sys.path
    """
    def __init__(self, syspath):
        """
        init
        :param syspath: path needed to add to sys.path
        """
        super().__init__()
        self._syspath = syspath

    def run(self):
        """
        add directory to sys.path
        :return:
        """
        if self._syspath not in sys.path:
            sys.path.append(self._syspath)


class PyImportTask(OpTask):
    """
    task to import py modules
    """
    def __init__(self, module_list):
        """
        init
        :param module_list:
        """
        super().__init__()
        self._module_list = module_list.split(",")

    def run(self):
        """
        do python module import
        :return:
        """
        for mlist in self._module_list:
            if mlist:
                importlib.import_module(mlist)


class ObjSyncTask(OpTask):
    """
    Task to sync module objects from parent to child process.
    """
    def __init__(self, module_name, obj_name, obj_value):
        """
        init
        :param module_name:
        :param obj_name:
        :param obj_value:
        """
        super().__init__()
        self._module_name = module_name
        self._obj_name = obj_name
        self._obj_value = obj_value

    def run(self):
        """
        do the data sync
        :return:
        """
        pymodule = importlib.import_module(self._module_name)
        obj = pickle.loads(zlib.decompress(self._obj_value))
        mysetattr(pymodule, self._obj_name, obj)


class PrebuildTask(OpTask):
    """
    Task to prebuild tbe op
    """
    def __init__(self, graph_id, task_id, op_module, op_func, *op_args):
        """
        init
        :param graph_id:
        :param task_id:
        :param op_module:
        :param op_func:
        :param op_args:
        """
        super().__init__()
        self.graph_id = graph_id
        self.task_id = task_id
        self._op_module = op_module
        self._op_func = op_func
        self._op_args = op_args
        self.build_type = 0

    def __str__(self):
        """
        string representation
        :return:
        """
        return "taskID[{}.{}]".format(self.graph_id, self.task_id)

    def run(self):
        """
        do prebuild
        :return:
        """
        try:
            start = datetime.datetime.now()
            opm = importlib.import_module(self._op_module)
            opfunc = getattr(opm, self._op_func)
            fusion_manager.op_build_cfg_dis()
            fusion_manager.set_op_build_type("prebuild")
            fusion_manager.set_current_op_func_name(self._op_func)
            fusion_manager.init_op_pattern()
            opfunc(*self._op_args)
            fusion_manager.op_build_cfg_en()
            pattern = fusion_manager.get_op_pattern()
            end = datetime.datetime.now()
            infomsg = "prebuild success. pattern[{}] module[{}] "\
                "func[{}], time:{}/{}".format(pattern, self._op_module,
                                              self._op_func, start, end-start)
            return gen_task_res(self.build_type, self.graph_id, self.task_id,
                                0, pattern, infomsg)
        except:                 # pylint: disable=bare-except
            except_msg = telog.except_msg()
            errmsg = "prebuild failed. module[{}] func[{}]"\
                .format(self._op_module, self._op_func)
            logger.info("%s, args:%s\n%s", errmsg, self._op_args, except_msg)
            return gen_task_res(self.build_type, self.graph_id, self.task_id,
                                1, 'None', errmsg,
                                err_args="args:{}".format(self._op_args),
                                except_msg=except_msg)

    def desc(self):
        """
        task description in json format
        """
        op_desc = {
            "type:": "prebuild",
            "module": self._op_module,
            "args": self._op_args
        }
        return json.dumps(op_desc)


class SingleOpTask(OpTask):
    """
    Task to compile single tbe op
    """
    # pylint: disable=too-many-arguments
    def __init__(self, graph_id, task_id, op_module, op_func, kernel_name, *op_args):
        """
        init
        :param graph_id:
        :param task_id:
        :param op_module:
        :param op_func:
        :param kernel_name:
        :param op_args:
        """
        super().__init__()
        self.graph_id = graph_id
        self.task_id = task_id
        self._op_module = op_module
        self._op_func = op_func
        self._kernel_name = kernel_name
        self._op_args = op_args
        self.build_type = 1

    def __str__(self):
        """
        string representation
        :return:
        """
        return "taskID[{}.{}]".format(self.graph_id, self.task_id)

    def run(self):
        """
        do single op compilation
        :return:
        """
        try:
            start = datetime.datetime.now()
            opm = importlib.import_module(self._op_module)
            opfunc = getattr(opm, self._op_func)
            fusion_manager.op_build_cfg_en()
            opfunc(*self._op_args)
            end = datetime.datetime.now()
            infomsg = "single op compile success. kernel[{}] "\
                "module[{}] func[{}], time:{}/{}"\
                .format(self._kernel_name, self._op_module,
                        self._op_func, start, end-start)
            return gen_task_res(self.build_type, self.graph_id, self.task_id,
                                0, self._kernel_name, infomsg)
        except:                 # pylint: disable=bare-except
            except_msg = telog.except_msg()
            errmsg = "single op compile failed. kernel[{}] "\
                "module[{}] func[{}]"\
                .format(self._kernel_name, self._op_module,
                        self._op_func)
            logger.info("%s, args:%s\n%s", errmsg, self._op_args, except_msg)
            return gen_task_res(self.build_type, self.graph_id, self.task_id,
                                1, self._kernel_name, errmsg,
                                err_args="args:{}".format(self._op_args),
                                except_msg=except_msg)

    def desc(self):
        """
        task description in json format
        """
        op_desc = {
            "type:": "single build",
            "module": self._op_module,
            "kernel_name": self._kernel_name,
            "args": self._op_args
        }
        return json.dumps(op_desc)


class FusionOpTask(OpTask):
    """
    Task to compile fusion op
    """
    def __init__(self, graph_id, task_id, json_str, kernel_name):
        """
        init
        :param graph_id:
        :param task_id:
        :param json_str:
        :param kernel_name:
        """
        super().__init__(self)
        self.graph_id = graph_id
        self.task_id = task_id
        self._json_str = json_str
        self._kernel_name = kernel_name
        self.build_type = 2

    def __str__(self):
        """
        string representation
        :return:
        """
        return "taskID[{}.{}]".format(self.graph_id, self.task_id)

    def run(self):
        """
        do fusion op compilation
        :return:
        """
        try:
            start = datetime.datetime.now()
            opm = importlib.import_module("te.platform.fusion_util")
            opfunc = getattr(opm, "fusion_op")
            fusion_manager.op_build_cfg_en()
            opfunc(self._json_str)
            end = datetime.datetime.now()
            infomsg = "fusion op compile success. "\
                "kernel[{}], time:{}/{}"\
                .format(self._kernel_name, start, end-start)
            return gen_task_res(self.build_type, self.graph_id, self.task_id,
                                0, self._kernel_name, infomsg)
        except:                 # pylint: disable=bare-except
            except_msg = telog.except_msg()
            errmsg = "fusion op compile fail. kernel_name[{}]"\
                .format(self._kernel_name)
            logger.info("%s. json:%s\n%s", errmsg, self._json_str, except_msg)
            return gen_task_res(self.build_type, self.graph_id, self.task_id,
                                1, self._kernel_name, errmsg,
                                err_args="json_str:{}".format(self._json_str),
                                except_msg=except_msg)

    def desc(self):
        """
        task description in json format
        """
        op_desc = {
            "type:": "fusion build",
            "kernel_name": self._kernel_name,
        }
        return json.dumps(op_desc)


# 'pylint: disable=too-many-arguments
def dispatch_prebuild_task(graph_id, task_id, l1size,
                           op_module, op_func, op_args):
    """
    prebuild task
    :param graph_id:
    :param task_id:
    :param op_module:
    :param op_func:
    :param op_args:
    """
    task = PrebuildTask(graph_id, task_id, op_module, op_func, *op_args)
    task.set_l1size(l1size)
    OpCompiler.compiler.dispatch_task(task)


# 'pylint: disable=too-many-arguments
def dispatch_single_op_compile_task(graph_id, task_id, l1size,
                                    op_module, op_func, kernel_name, op_args):
    """
    single op build task
    :param graph_id:
    :param task_id:
    :param op_module:
    :param op_func:
    :param kernel_name:
    :param op_args:
    """
    task = SingleOpTask(graph_id, task_id,
                        op_module, op_func, kernel_name, *op_args)
    task.set_l1size(l1size)
    OpCompiler.compiler.dispatch_task(task)


def dispatch_fusion_op_compile_task(graph_id, task_id, l1size,
                                    json_str, kernel_name):
    """
    fusion op build task
    :param graph_id:
    :param task_id:
    :param json_str:
    :param kernel_name:
    """
    task = FusionOpTask(graph_id, task_id, json_str, kernel_name)
    task.set_l1size(l1size)
    OpCompiler.compiler.dispatch_task(task)


def import_py_module(module_list):
    """
    import py module task
    :param module_list:
    """
    task = PyImportTask(module_list)
    OpCompiler.compiler.sync_data(task)


def sync_py_object(module_name, obj_name):
    """
    sync python object to worker process
    :param module_name:
    :param obj_name:
    """
    opm = importlib.import_module(module_name)
    obj = mygetattr(opm, obj_name)
    obj = zlib.compress(pickle.dumps(obj))
    task = ObjSyncTask(module_name, obj_name, obj)
    OpCompiler.compiler.sync_data(task)


def sync_syspath(syspath):
    """
    sync syspath to worker process
    :param syspath: the path needed to add to sys.path of worker process
    """
    task = PySysPathTask(syspath)
    OpCompiler.compiler.sync_data(task)


def compile_op(json_str):
    """
    compile op parallelly
    """
    op_desc = json.loads(json_str)
    gid = threading.get_ident()
    tid = Counter.next()
    task = FusionOpTask(gid, tid, json_str, op_desc['fusion_op_name'])
    OpCompiler.compiler.dispatch_task(task)
    return DeferredOpRes(gid, tid)


def compile_op_sync(json_str):
    defer = compile_op(json_str)
    while True:
        time.sleep(0.01)
        res = defer.get()
        if res is not None:
            return


def update_running_task(task):
    OpCompiler.compiler.update_running_task(task)
