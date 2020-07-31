"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     debugger.py
DESC:     TODO (some explanation)
CREATED:  2019-04-18 18:53:42
MODIFIED: 2019-07-16 20:14:42
"""
# disabling:
# R0902: too-many-instance-attributes
# C0103: invalid-name

from __future__ import print_function
import sys
import os
import re
import copy
from cmd import Cmd
from te.tik.tik_lib.tik_check_util import TikCheckUtil, read_context_from_file
from te.tik.tik_lib.tik_params import CUR_FRAME_IDX
from .debug.statement import TikDebug

_MIN_LINE_NO = 0
_MIN_ID_EXPECTED = 0


class TikQuitDebugLoop(BaseException):
    """ The Exception Object used for quiting Interactive mode """

    def __init__(self, msg=""):
        """Initalize class TikQuitDebugLoop.

        Parameters
        ----------
        msg:The message of TikQuitDebugLoop exception.

        Returns
        ----------
        No returns
        """
        super(TikQuitDebugLoop, self).__init__(msg)
        self.msg = msg
        self.quit_tik = False


class TikDebugger(Cmd):
    """ Class TikDebugger instance"""
    # pylint: disable=R0902
    # filename : [(source_info, ast_node), ...]
    debug_info = {}

    def __init__(self):
        """Initialize class TikDebugger"""
        Cmd.__init__(self, completekey='tab', stdin=None, stdout=None)
        # record of event
        self.event = ""
        # cmd is running or not running
        self.running = True
        # cache of file name
        self.fncache = {}
        self.current_ast_node = None
        self.current_ast_context = None
        self.num_trace_event = 0
        self.args = None

    def register_debug_info(self, source_info, ast_node):
        """registe debugger information

        Parameters
        -----------
        source_info:file's information.
            This is a CallerContext Class instance.

        ast_node:Node of Abstract Syntax Trees.
            This is a STMT Class instance.

        Returns
        -----------
        No returns
        """
        if not source_info:
            return

        cano_filename = self.canonic(source_info[CUR_FRAME_IDX].get("filename"))
        if cano_filename in self.debug_info.keys():
            self.debug_info[cano_filename].append((source_info, ast_node))
        else:
            self.debug_info[cano_filename] = [(source_info, ast_node)]

    def force_interactive(self, prompt="[TIK]>"):
        """ Force stop interactive

        Parameters
        ----------
        prompt:Display in console with "[TIK]>"

        Returns
        ----------
        No returns
        """
        success_code = 0
        self.prompt = prompt
        try:
            self.cmdloop()
        except TikQuitDebugLoop as tik_except:
            if tik_except.quit_tik:
                print("[INFO]: `quit' command stops TIK process")
                sys.exit(success_code)
        except KeyboardInterrupt:
            print("[INFO]: keyboard interrupt")
            sys.exit(success_code)

    def trace(self, ast_context, ast_node, event):
        """ The trace function provided to AST evaluator

        Parameters
        ----------
        ast_context:Context of abstract syntax trees.
            This is a Class Context instance.

        ast_node:Node of Abstract Syntax Trees.
            This is a STMT Class instance.

        event:Record of event
            This is an argument for exception.

        Returns
        ----------
        None
        """
        self.event = event
        self.num_trace_event += 1
        self.current_ast_node = ast_node
        self.current_ast_context = ast_context

        if ast_node.break_point and ast_node.break_point.enabled:
            self.force_interactive()
            return

        if TikDebug.force_interactive:
            self.force_interactive()
            return

        if event == 'exception':
            self.force_interactive()
            return

    # Interactive commands:
    def do_break(self, arg):
        """ Debug command break for printing infomation of break_points

        Parameters
        ----------
        arg:Argument of break
            Input an argument of break command on console.

        Returns
        ----------
        None
        """
        if not arg:
            print("Num Type         Enb   Where")
            for break_point in TikBreakPoint.break_points:
                print(break_point)
            return
        re_break_command = re.compile(r"^\s*(\S+)\s*:\s*(\d+)\s*$")
        if not self.debug_info:
            print("[WARNING]: no debug info registed during building TIK AST.")
            return

        # 1) parse the argument
        match_result = re_break_command.match(arg)
        if not match_result:
            print("[ERROR]: Invalid argument(%s), "
                  "expected form '<filename>:<lineno>'" % arg)
            return
        filename = match_result.group(1)
        lineno = int(match_result.group(2))

        # 2) Validate the parsing result
        # invalid line number and filename
        if not filename or lineno < _MIN_LINE_NO:
            print("[ERROR]: Invalid arugment( filename=%s lineno=%d )" %
                  (filename, lineno))
            return

        cano_filename = self.canonic(filename)
        if cano_filename not in self.debug_info.keys():
            print("[ERROR]: Failed to set break point to file, "
                  "as it contains no TIK primitives. filename=%s" % filename)
            return

        set_bp = False
        # 3) go through TIK primitives in the file and set the break point
        tik_primitive_list = self.debug_info.get(cano_filename)
        for source_info, ast_node in tik_primitive_list:
            if source_info[CUR_FRAME_IDX].get("line_no") == lineno and \
                    ast_node.traceable:
                ast_node.break_point = TikBreakPoint(ast_node)
                TikBreakPoint.break_points.append(ast_node.break_point)
                set_bp = True
                print("[INFO]: Break point set to %s:%d" %
                      (filename, lineno))

        if not set_bp:
            print("[ERROR]: Failed to set breakpoint to %s:%d, "
                  "because it's not a TIK primitive" % (filename, lineno))

    do_b = do_break

    def do_continue(self, args):
        """ Continue the TIK execution

        Parameters
        ----------
        args:Argument of continue
            Input an argument of continue command on console.

        Returns
        ----------
        None
        """
        self.args = args
        if not self.running:
            print("[ERROR]: invalid command `continue` "
                  "when an exception occured")
            return

        TikDebug.force_interactive = False
        raise TikQuitDebugLoop("continue command")

    do_c = do_continue

    def do_next(self, args):
        """ Stop at the next TIK primitives evaluation

        Parameters
        ----------
        args : Argument of next
            Input an argument of next command on console.

        Returns
        ----------
        None
        """
        self.args = args
        if not self.running:
            print("[ERROR]: invalid command `next` execute "
                  "when an exception occured")
            return
        TikDebug.force_interactive = True
        raise TikQuitDebugLoop("User next")

    do_n = do_next

    def do_list(self, args):
        """ Print current code and context

        Parameters
        ----------
        args:Argument of list
            Input an argument of list on console.

        Returns
        ----------
        None
        """
        self.do_where(args)
    do_l = do_list

    def do_enable(self, args):
        """ Debug command for enabling a breakpoint

        Parameters
        ----------
        args:Argument of enable
            Input an argument of enable on console.

        Returns
        ----------
        None
        """
        break_point = self.manipulate_break_point(args, "enable")
        if break_point:
            break_point.debug_print("[INFO]: Enabling ")

    def do_disable(self, args):
        """ Debug command to disable a breakpoint by id

        Parameters
        ----------
        args:Argument of disable
            Input an argument of disable on console.

        Returns
        ----------
        None
        """
        break_point = self.manipulate_break_point(args, "disable")
        if break_point:
            break_point.debug_print("[INFO]: Disabling ")

    def do_clear(self, args):
        """ Debug command to delete a breakpoint by id

        Parameters
        ----------
        args:Argument of clear
            Input an argument of clear on console, to clear break points.

        Returns
        ----------
        None
        """
        if args is None or args == '':
            bp_temp = TikBreakPoint.break_points[:]
            for target_bp in bp_temp:
                target_bp.ast_node.break_point = None
                TikBreakPoint.break_points.remove(target_bp)
                TikBreakPoint.num_break_points -= 1
            return
        self.manipulate_break_point(args, "delete")

        return

    def do_print(self, args):
        """ Debug command to print tensor

        Parameters
        ----------
        args:Argument of print
            Input an argument of print on console, to print tensor.

        Returns
        ----------
        None
        """
        if not args:
            print("[ERROR]: Invalid syntax. Expected usage:")
            print("    print <tensor_name> ")
            return

        if not self.current_ast_node:
            print("[ERROR]: Invalid current AST node!")
            return

        TikCheckUtil.check_not_is(self.current_ast_node, None)
        TikCheckUtil.check_not_is(self.current_ast_context, None)
        TikCheckUtil.check_not_is(
            self.current_ast_node.source_info[CUR_FRAME_IDX].get("sym_table"),
            None)

        print_obj = args

        symtable = copy.copy(
            self.current_ast_node.source_info[CUR_FRAME_IDX].get("sym_table"))

        from .debug.statement import PrintExpr

        PrintExpr.print_expr(print_obj, symtable, self.current_ast_context)

    do_p = do_print

    def do_where(self, args):
        """ Debug command to query call stack

        Parameters
        args:Argument of where
            Input an argument of querying call stack.

        Returns
        ----------
        None
        """
        self.args = args
        self.dump_cal_stack()

    do_w = do_where

    def do_quit(self, args):
        """ Debug command to quit tik

        Parameters
        ----------
        args:Argument of quit
            Input an argument of quit on console.

        Returns
        ----------
        None
        """
        self.args = args
        exception = TikQuitDebugLoop("User Quit")
        exception.quit_tik = True
        raise exception

    do_q = do_quit

    def do_dump_debug_info(self, args):
        """ Debug command to show debug info

        Parameters
        ----------
        args:Argument of show debug information

        Returns
        ----------
        None
        """
        self.args = args
        if not self.debug_info:
            print("No debug info registed!")
            return

        print("Dumping debug info...")
        num_ast_node = 0
        num_user_file = 0
        for filename, debug_info_list in self.debug_info.items():
            print("  filename: %s" % filename)
            for source_info, ast_node in debug_info_list:
                print("    line:%d ast_node:%s" %
                      (source_info[CUR_FRAME_IDX].get("line_no"),
                       type(ast_node)))
                num_ast_node += 1
            num_user_file += 1
            print("")
        print("# of AST node  =  %d" % num_ast_node)
        print("# of AST files =  %d" % num_user_file)
        print("Dumping debug info... Completed")

    @staticmethod
    def manipulate_break_point(id_string, action):
        """ Manipuate breakpoints

        Parameters
        ----------
        id_string:The break point's id.

        action:Argument of action
            This is the action of manipulating break point.

        Returns
        ----------
        None
        """
        try:
            id_expected = int(id_string)
        except ValueError as exception:
            print(str(exception))
            return None

        if id_expected < _MIN_ID_EXPECTED:
            print("[ERROR]: Invalid break point id: %d" % id_expected)

        supported_action = ("enable", "disable", "delete")
        if action not in supported_action:
            print("[INFO]:supported actions are:")
            for act in supported_action:
                print(act)
            print("[ERROR]: Unsupported action to break points: %s" % action)
            return None

        target_bp = None
        for break_point in TikBreakPoint.break_points:
            if break_point.id == id_expected:
                target_bp = break_point
                break

        if target_bp is None:
            print("[ERROR]: Failed to find breakpoint with id=%d" % id_expected)
            return None

        if action == "enable":
            target_bp.enabled = True
        elif action == "disable":
            target_bp.enabled = False
        elif action == "delete":
            TikCheckUtil.check_is(target_bp.ast_node.break_point, target_bp)
            target_bp.ast_node.break_point = None
            TikBreakPoint.break_points.remove(target_bp)
            TikBreakPoint.num_break_points -= 1
        return target_bp

    def canonic(self, filename):
        """ Return canonical form of filename.
        For real filenames, the canonical form is a case-normalized (on
        case insenstive filesystems) absolute path.  'Filenames' with
        angle brackets, such as "<stdin>", generated in interactive
        mode, are returned unchanged.

        Parameters
        ----------
        filename:Name of file that you select.
            The expect form is "<filename>".

        Returns
        ----------
        filename:The expect filename.
        canonic:absolute path of real filenames, is not the expect filename.

        """
        if filename is None:
            return None

        if filename == ("<" + filename[1:-1] + ">"):
            return filename
        canonic = self.fncache.get(filename)
        if not canonic:
            canonic = os.path.realpath(filename)
            canonic = os.path.normcase(canonic)
            self.fncache[filename] = canonic
        return canonic

    def dump_cal_stack(self):
        """ Used as a utlity to dump call stack from other functionality code

        Parameters
        ----------
        No parameters

        Returns
        ----------
        No returns
        """
        # Called within the debugger session
        TikCheckUtil.check_not_is(self.current_ast_node, None)

        ast_node = self.current_ast_node
        while ast_node:
            source_info = ast_node.source_info
            if not source_info:
                break

            if isinstance(source_info, list):
                print(read_context_from_file(
                    source_info[CUR_FRAME_IDX].get("filename"),
                    source_info[CUR_FRAME_IDX].get("line_no")))
                return  # only print current stack now
            ast_node = ast_node.parent


class TikBreakPoint():
    """ Class TikeBreakPoint"""
    num_break_points = 0
    break_points = []
    next_id = 0

    def __init__(self, ast_node):
        """Initialize class TikBreakPoint

        Parameters
        ----------
        ast_node: Node of Abstract Syntax Trees
            This is a Class STMT instance.

        enable:A parameter for enabling break point

        Returns
        ----------
        No returns
        """
        # pylint: disable=C0103
        # self.id must corresponds to Varresolver.visit_Name.node.id
        # in statement.py
        self.enabled = True
        self.id = TikBreakPoint.next_id
        self.ast_node = ast_node
        TikBreakPoint.next_id += 1
        TikBreakPoint.num_break_points += 1

    def __eq__(self, other):
        """Judge to is other break point's id equal to self's id.

        Parameters
        ----------
        other:Other break point instances

        Returns
        ----------
        False or True
        """
        if not other:
            return False
        return self.id == other.id

    def __str__(self):
        """Display the break point.

        Parameters
        ----------
        No parameters

        Returns
        ----------
        id:break point id.

        disp:yes or no to display

        source_file:file name of source_info

        source_line_no:line number of source_info
        """
        source_info = self.ast_node.source_info
        if self.enabled:
            disp = 'yes  '
        else:
            disp = 'no   '
        return '%-4dbreakpoint   %s at %s:%d' % (
            self.id, disp, source_info[CUR_FRAME_IDX].get("filename"),
            source_info[CUR_FRAME_IDX].get("line_no"))

    def debug_print(self, prefix=""):
        """ Print debug break point

        Parameters
        ----------
        prefix:This is a empty parameter.

        Returns
        ----------
        No returns
        """
        print(prefix, end='')
        source_info = self.ast_node.source_info
        print("breakpoint id=%d %s:%d ast_node=%s" %
              (self.id, source_info[CUR_FRAME_IDX].get("filename"),
               source_info[CUR_FRAME_IDX].get("line_no"),
               type(self.ast_node).__name__))


# set up trace function for AST evaluation
TikDebug.set_trace(TikDebugger())
