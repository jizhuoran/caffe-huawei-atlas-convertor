# -*- coding: utf-8 -*-

"""
  Desc:   Internal tools for extracting performance data from CAModel
"""

import sys  # pylint: disable=C0302
import getopt
import os
import re


# too many attributes, too many lines
# a global config variable
class ConfigInfo:
    """
    use to save configInfo
    """
    dumpVersion = 2  # 2 for 1980 new version
    testName = "NoTestname"
    coreId = "core0"  # default for v2
    instrPoppedDump = ""
    instrRetiredDump = ""
    mteBiuReqDump = ""
    icacheDump = ""
    cubeDump = ""
    vectorDump = ""
    scalarDump = ""

    @classmethod
    def display_all_config_info(cls):
        """
        show config info
        :return: none
        """
        print("Config.Version: %s" % ConfigInfo.dumpVersion)
        print("Config.Testname: %s" % ConfigInfo.testName)
        print("Config.coreId: %s" % ConfigInfo.coreId)
        print("Config.instrPoppedDump: %s" % ConfigInfo.instrPoppedDump)
        print("Config.instrRetiredDump: %s" % ConfigInfo.instrRetiredDump)
        print("Config.mteBiuReqDump: %s" % ConfigInfo.mteBiuReqDump)
        print("Config.icacheDump: %s" % ConfigInfo.icacheDump)
        print("Config.cubeDump: %s" % ConfigInfo.cubeDump)
        print("Config.vectorDump: %s" % ConfigInfo.vectorDump)
        print("Config.scalarDump: %s" % ConfigInfo.scalarDump)

    def __str__(self):
        return "Config for generate html"


def is_event_inst(inst_name):
    """
    is event inst or not
    :param inst_name: instr name
    :return: True if isEvent
    """
    return inst_name in ('set_flag', 'wait_flag', 'barrier')


class IcacheMissRecord:
    """
    save icache miss info
    """
    # too many arguments, so disable R0913
    def __init__(self, start, end, addr, req_id, s):  # pylint: disable=R0913
        self.start = int(start)
        self.end = int(end)
        self.addr = addr
        self.req_id = req_id
        self.raw_str = s

    def __str__(self):
        return "IcacheMissRecord"

    def __hash__(self):
        return "IcacheMissRecord"


def get_icache_req_id_key(obj):
    """
    get icachereqid
    :param obj: object of instr
    :return: object's key
    """
    return obj.req_id


class IcacheMissList:
    """
    save icache miss list
    """
    def __init__(self, filename, cache_name):
        self.filename = filename
        self.cache_name = cache_name
        self.num_miss = 0
        self.accum_clk = 0
        self.piped_clk = 0
        self.records = self.parse_icache_file()

    def display_stat(self, test_name):
        """
        display stat
        :param test_name: testname
        :return: print stat
        """
        ic_missformat_str_dict = "%-14s #I: %8d, Accum: %8d, " \
                              "Piped: %8d (  0.00%%), " \
                              "Idle:        0 \t=>    NA" \
            % ("ICMiss", self.num_miss, self.accum_clk, self.piped_clk)
        print(ic_missformat_str_dict)
        with open("./" + test_name + ".data", "a") as temp_file:
            temp_file.writelines(ic_missformat_str_dict + "\n")

    def parse_icache_file(self):  # pylint: disable=R0914
        """
        parse icache file
        :return: nothing
        """
        record_list = []
        req_list = []
        ack_list = []
        with open(self.filename, "r") as temp_file:
            for line in temp_file.readlines():
                if line.find("cache refill request") != -1:
                    clk = re.search(r"\[([0-9]{8})\]: " + self.cache_name +
                                    " refill request", line)
                    req_id = re.search("id is (0x[0-9a-z]{8}), address", line)
                    addr = re.search("address is (0x[0-9a-z]{8})", line)
                    req_list.append(IcacheMissRecord(clk.group(1), 0,
                                                     addr.group(1),
                                                     req_id.group(1), line))
                elif line.find("cache refill acknowledge") != -1:
                    clk = re.search(r"\[([0-9]{8})\]: " + self.cache_name +
                                    " refill acknowledge", line)
                    req_id = re.search("id is (0x[0-9a-z]{8}), address", line)
                    addr = re.search("address is (0x[0-9a-z]{8})", line)
                    ack_list.append(IcacheMissRecord(0, clk.group(1), addr.group(1),
                                                     req_id.group(1), line))
                continue
        # merge requests
        if len(req_list) != len(ack_list):
            # Just print a err msg, we'll skip the non-match req, CAModel not handle
            print("Error: icache req not equal ack")

        # Sort to Merge
        sorted_ack_list = sorted(ack_list, key=get_icache_req_id_key)
        sorted_req_list = sorted(req_list, key=get_icache_req_id_key)

        prev_end = 0
        list_len = len(sorted_ack_list)
        i = 0
        while i < list_len:
            if (sorted_req_list[i].req_id != sorted_ack_list[i].req_id) or \
               (sorted_req_list[i].addr != sorted_ack_list[i].addr):
                print("Error: Icache req does not match [%s:%s] [%s:%s]" \
                      % (sorted_req_list[i].req_id, sorted_ack_list[i].req_id,
                         sorted_req_list[i].addr, sorted_ack_list[i].addr))
                return record_list
            rec = IcacheMissRecord(sorted_req_list[i].start,
                                   sorted_ack_list[i].end,
                                   sorted_req_list[i].addr,
                                   sorted_req_list[i].req_id,
                                   sorted_req_list[i].raw_str + sorted_ack_list[i].raw_str)
            record_list.append(rec)
            self.num_miss += 1
            self.accum_clk += sorted_ack_list[i].end - sorted_req_list[i].start + 1
            if sorted_req_list[i].start <= prev_end:
                if sorted_ack_list[i].end > prev_end:
                    self.piped_clk += sorted_ack_list[i].end - prev_end
            else:
                self.piped_clk += sorted_ack_list[i].end - sorted_req_list[i].start + 1
            prev_end = sorted_ack_list[i].end
            i += 1

        return record_list


class InstDumpRecord:  # pylint: disable=R0902
    """
    save inst dump
    """
    unresolvedRepeat0InstList = []

    def __init__(self, raw_record):
        self.raw_record = raw_record
        self.parse()
        self.merged = False

    def parse(self):
        """
        parse dump file
        :return: nothing
        """
        reg_tick = re.search(r"\[([0-9]{8})\]", self.raw_record)
        reg_pc = re.search(r"\(PC: (0x[0-9a-z]{8})\)", self.raw_record)
        reg_core_id = re.search("@CORE([0-9]{1,2})", self.raw_record)  # 1980 optional
        reg_inst_pipe = re.search(r"([A-Za-z0-3 ]*) : \(Binary",
                                  self.raw_record)
        reg_binary = re.search(r"\(Binary: (0x[0-9a-z]{8})\)", self.raw_record)
        reg_info = re.search(r"\(Binary: 0x[0-9a-z]{8}\) ([0-9a-z_]*)"
                             r"([\(]{0,1}.*[\)]{0,1})",
                             self.raw_record)
        reg_instr_id = re.search("instr ID is: ([0-9]*)", self.raw_record)
        reg_bank_conflict = re.search(r"Bank conflict RD\(([0-9]*)\) "
                                      r"WR\(([0-9]*)\)",
                                      self.raw_record)

        # CAModel will replace a Repeat0 instr to a nop and send.
        self.is_repeat0 = False
        self.is_repeat0_nop = False

        if reg_core_id:
            self.coreid = reg_core_id.group(1)   # ISA 5.6 format
        else:
            self.coreid = 0    # TOFIX: ISA 6.x format

        if reg_tick and reg_pc and reg_inst_pipe and reg_binary and reg_info:
            self.tick = reg_tick.group(1)
            self.program_counter = reg_pc.group(1)
            self.inst_pipe = reg_inst_pipe.group(1).lstrip()
            self.inst_binary = reg_binary.group(1)
            self.inst_name = reg_info.group(1)
            self.inst_detail = reg_info.group(2)

            if reg_core_id:
                #strip 'CORE0 ', we don't have more than 10 core now.
                self.inst_pipe = self.inst_pipe[6:]

            if self.inst_pipe.find("Repeat0") != -1:
                self.is_repeat0 = True

            if self.inst_pipe.find("ISSUE") != -1 and \
               self.inst_name == "nop_pipe":
                self.is_repeat0_nop = True
                # find a Nop replacement of repeat0, must have a repeat0 prev
                if not InstDumpRecord.unresolvedRepeat0InstList:
                    print("Error: a repeat Nop does not have a previous Repeat0 ist")
                for prev_rec in InstDumpRecord.unresolvedRepeat0InstList:
                    if self.program_counter != prev_rec.program_counter:
                        continue
                    self.tick = prev_rec.tick
                    r_pos = prev_rec.inst_pipe.find("Repeat0")
                    self.inst_pipe = prev_rec.inst_pipe[0:r_pos - 1]
                    self.inst_name = prev_rec.inst_name + " Repeat0"
                    self.inst_binary = prev_rec.inst_binary
                    self.inst_detail = prev_rec.inst_detail
                    InstDumpRecord.unresolvedRepeat0InstList.remove(prev_rec)
                    break
        else:
            print("Parse a instr dump record failed:", self.raw_record)

        self._set_id_bank(reg_instr_id, reg_bank_conflict)

    def _set_id_bank(self, reg_instr_id, reg_bank_conflict):
        if reg_instr_id:
            self.instr_id = reg_instr_id.group(1)  # pylint: disable=W0201
            # not set a default "0" ID here, since most dump in instr.dump
            # does not have a ID , but does in intr_poped,dump
        if reg_bank_conflict:
            self.bank_conflict_rd = reg_bank_conflict.group(1)  # pylint: disable=W0201
            self.bank_conflict_wr = reg_bank_conflict.group(2)  # pylint: disable=W0201

    def display(self):
        """
        display thing
        :return: nothing
        """
        print(self.tick, self.program_counter, self.coreid, self.inst_pipe,
              self.inst_binary, self.inst_name, self.inst_detail,
              self.instr_id, self.bank_conflict_rd, self.bank_conflict_wr)

    def get_repeat0_info(self):
        """
        get repeat0 info
        :return: nothing
        """
        if self.is_repeat0_nop:
            return "Repeat0"
        return ""

    def get_parsed_instr_detail_info(self):
        """
        use to get parsed instr detail info
        :return: info msg
        """
        if self.inst_name == "mmad":
            reg_mnk = re.search("M:([0-9]*),K:([0-9]*),N:([0-9]*),",
                                self.inst_detail)
            reg_m = reg_mnk.group(1)
            reg_n = reg_mnk.group(2)
            reg_k = reg_mnk.group(3)
            mnk_str = "M:%s:N:%s:K:%s" % (reg_m, reg_n, reg_k)
            nfract = "%3d" %((int(reg_m) * int(reg_n) * int(reg_k))/(16*16*16))
            return mnk_str + " " + nfract
        return ""

    def get_scalar_ld_st_addr_info(self):
        """
        use to get scalar ld st addr info
        :return: info msg
        """
        if self.inst_name.find("scalar_ld_imm") != -1 or \
                self.inst_name.find("scalar_st_imm") != -1 \
           or self.inst_name.find("scalar_ld4_imm") != -1 or \
                self.inst_name.find("scalar_st4_imm") != -1:
            reg_addr = re.search(r"\(dType:0x([0-3]{1}),.*, x\[[0-9]{1,2}\]= "
                                 "0x([0-9a-f]*), imm12= 0x([0-9a-f]*)",
                                 self.inst_detail)
            d_type = reg_addr.group(1)
            base = reg_addr.group(2)
            offset = reg_addr.group(3)
            addr = int(base, 16)
            addr_str = "dType:%s base:%s offset:%s" % (d_type, base, offset)
            return addr_str, addr
        if self.inst_name.find("scalar_ld") != -1 or \
                self.inst_name.find("scalar_st") != -1:
            reg_addr = re.search(r"\(dType:0x([0-3]{1}),.*, x\[[0-9]{1,2}\]="
                                 r" 0x([0-9a-f]*), x\[[0-9]{1,2}\]= 0x([0-9a-f]*)",
                                 self.inst_detail)
            d_type = reg_addr.group(1)
            base = reg_addr.group(2)
            offset = reg_addr.group(3)
            addr = int(base, 16)
            addr_str = "dType:%s base:%s offset:%s" % (d_type, base, offset)
            return addr_str, addr
        return "", 0

# Parse instr_popped.dump file
# This file is a little different formatted with instr.dump,
# secondary pipe instr popped twice.


class InstPopDumpRecord:  # pylint: disable=R0902
    """
    save inst pop dump
    """
    unresolvedPartialInstList = []

    def __init__(self, raw_record):
        self.raw_record = raw_record
        self.parse()

    def _set_pop(self, reg_pop_info0, reg_pop_info1):
        # lsu_mov_special_xn inst is in MTE but not issue twice.
        if reg_pop_info0 and (self.inst_name != "lsu_mov_special_xn"):
            self.is_partial_issue = True  # pylint: disable=W0201
        if reg_pop_info1:
            self.is_pop_from_ex = True  # pylint: disable=W0201
            self.instr_id = reg_pop_info1.group(1)  # pylint: disable=W0201

    def parse(self):
        """
        parse InstPopDumpRecord info
        :return: nothing
        """
        reg_tick = re.search(r"\[([0-9]{8})\]", self.raw_record)
        reg_pc = re.search(r"\(PC: (0x[0-9a-z]{8})\)", self.raw_record)
        reg_core_id = re.search("@CORE([0-9]{1,2})", self.raw_record)  # 1980 optional
        reg_inst_pipe = re.search(r"([A-Z1-3 ]*) : \(Binary",
                                  self.raw_record)
        reg_binary = re.search(r"\(Binary: (0x[0-9a-z]{8})\)", self.raw_record)
        reg_info = re.search(r"\(Binary: 0x[0-9a-z]{8}\) ([0-9a-z_]*)([\(]{0,1}.*[\)]{0,1})",
                             self.raw_record)
        reg_pop_info0 = re.search("poped from IQ", self.raw_record)
        reg_pop_info1 = re.search("instr ID is: ([0-9]*)", self.raw_record)

        if reg_core_id:
            self.coreid = reg_core_id.group(1)   # ISA 5.6 format
        else:
            self.coreid = 0  # TOFIX: ISA 6.x format, to support core id from logfile

        if reg_tick and reg_pc and reg_inst_pipe and reg_binary and reg_info:
            self.tick = reg_tick.group(1)
            self.program_counter = reg_pc.group(1)
            self.inst_pipe = reg_inst_pipe.group(1)
            self.inst_binary = reg_binary.group(1)
            self.inst_name = reg_info.group(1)
            self.inst_detail = reg_info.group(2)
        else:
            print("Parse a instr_popped dump record failed:", self.raw_record)

        # This inst just issued from the main pipe, and is partial done
        self.is_partial_issue = False
        # inst now is issued from its own queue, complete issue
        self.is_pop_from_ex = False

        self._set_pop(reg_pop_info0, reg_pop_info1)

        # inst that poped from EX, must have a previous partialIssue
        if self.is_pop_from_ex:
            if not InstPopDumpRecord.unresolvedPartialInstList:
                print("Error: a pop from Ex inst must have a "
                      "previous partial issue inst")
            for prev_record in InstPopDumpRecord.unresolvedPartialInstList:
                if self.program_counter != prev_record.program_counter:
                    # try to find next
                    continue
                self.tick_pipe = self.tick
                self.tick = prev_record.tick
                InstPopDumpRecord.unresolvedPartialInstList.remove(prev_record)
                break
                # Does not find a matched previous partial Issue, must be error
        else:
            self.tick_pipe = '        '


    def display(self):
        """
        InstPopDumpRecord info display
        :return:
        """
        print(self.tick, self.tick_pipe, self.program_counter, self.coreid, self.inst_pipe,
              self.inst_binary, self.inst_name, self.inst_detail,
              "popIQ ", self.is_partial_issue, " popEx ", self.is_pop_from_ex)

# Parse dump file
def get_tick_key(obj):
    """
    get tick key value
    :param obj:
    :return: get tick key
    """
    return obj.tick


def parse_inst_dump_file(filename):
    """
    use to parse inst dump file
    :param filename:
    :return:
    """
    num_scalar_inst = 0
    num_event_inst = 0
    num_mte_inst = 0

    record_list = []
    if not os.path.exists(filename):
        print("Error: %s file does not exists!!!ABORT" % filename)
        sys.exit(2)
    with open(filename, "r") as temp_file:
        for line in temp_file.readlines():
            if line == '\n':
                continue
            if ConfigInfo.dumpVersion == 2 and (not line[1:5] == "info"):
                # Skip lines that not started with [info]
                continue
            if "++++++++++++++++++++++" in line:
                # Currently skip the line
                # "++++++++++++++++++++++tick: 0, kernal0 done++++++++++++++++++++++."
                continue

            rec = InstDumpRecord(line)
            if not rec.is_repeat0:
                record_list.append(rec)
            else:
                InstDumpRecord.unresolvedRepeat0InstList.append(rec)

            if rec.inst_pipe == "SCALAR":
                num_scalar_inst += 1
            elif is_event_inst(rec.inst_name):
                num_event_inst += 1
            elif rec.inst_pipe[0:3] == "MTE":
                num_mte_inst += 1

    # Sort by Tick
    sorted_list_by_tick = sorted(record_list, key=get_tick_key)

    return sorted_list_by_tick


# Parse instr_popped.dump file
def parse_inst_popped_dump_file(filename):
    """
    use to parse inst popped dump file
    :param filename:
    :return: None
    """
    record_list = []
    with open(filename, "r") as temp_file:
        for line in temp_file.readlines():
            if line == '\n':
                continue
            if "++++++++++++++++++++++" in line:
                continue

            rec = InstPopDumpRecord(line)
            if not rec.is_partial_issue:
                record_list.append(rec)
            else:
                InstPopDumpRecord.unresolvedPartialInstList.append(rec)

    # Sort by Tick
    sorted_list_by_tick = sorted(record_list, key=get_tick_key)
    return sorted_list_by_tick


# Merge issue and retire inst records
RENDER_NAME_DICT = {"SCALAR": "A",
                    ".scalar_ldst": "B",
                    "FLOWCTRL": "C",
                    "VECTOR": "D",
                    "CUBE": "E",
                    "MTE2": "F",
                    "MTE1": "G",
                    "MTE3": "H",
                    ".event": "I"}


class InstrExecutionRecord:  # pylint: disable=R0902
    """
    save instr execution
    """
    # Min start cycle in all pipes
    start_point = 0
    # Max end cycle in all pipes
    end_point = 0

    def __init__(self, start, end, instr_rec):
        self.start = start
        self.end = end
        self.pipe = instr_rec.inst_pipe
        self.instr_name = instr_rec.inst_name
        self.instr_binary = instr_rec.inst_binary
        self.instr_pc = instr_rec.program_counter
        self.instr_detail = instr_rec.inst_detail
        self.bank_conflict = ""
        if hasattr(instr_rec, 'bankConflictRD'):
            self.bank_conflict = "BC:R%s W%s" % (instr_rec.bankConflictRD, instr_rec.bankConflictWR)
        self.ldst_addr_info, self.ldst_addr = instr_rec.get_scalar_ld_st_addr_info()

        # change the catagory StatisticPipe, just for statistic
        self.rename_instr_pipe()
        self.render_pipe_name = RENDER_NAME_DICT[self.pipe]

    def rename_instr_pipe(self):
        """
        use to rename instr pipe
        :return:
        """
        if self.instr_name.find("scalar_ld") != -1 or self.instr_name.find("scalar_st") != -1:
            self.pipe = ".scalar_ldst"
        if self.pipe.find("ISSUE") != -1:
            self.pipe = ".event"

    def display(self):
        """
        use to display latency info
        :return:
        """
        print("[%d, %d) Latency %d %s" % (self.start, self.end, self.end-self.start+1, self.pipe))


def _retire_inst(retire_inst, issue_inst, instr_exec_list, show_detail):
    """func for retire inst"""
    if (hasattr(retire_inst, 'instrId') and
            hasattr(issue_inst, 'instrId') and
            (retire_inst.instrId != issue_inst.instrId)):
        print("Error: instr ID does not match")
    # Very Luck, find
    find_match = True
    tail_str = ""
    if hasattr(retire_inst, 'bankConflictRD'):
        tail_str = "Bank Conflict RD(%s) WR(%s)" % \
                   (retire_inst.bankConflictRD,
                    retire_inst.bankConflictWR)

    retire_inst.merged = True
    instr_exec_list.append(InstrExecutionRecord(int(issue_inst.tick),
                                                int(retire_inst.tick),
                                                retire_inst))
    if show_detail:
        print("Inst Latency : ", issue_inst.program_counter,
              issue_inst.tick, " ",
              issue_inst.tick_pipe, " ",
              retire_inst.tick, " ",
              int(retire_inst.tick) - int(issue_inst.tick),
              "\t", retire_inst.inst_pipe,
              "\t", retire_inst.inst_name,
              retire_inst.get_parsed_instr_detail_info(), tail_str,
              InstrExecutionRecord(int(issue_inst.tick),
                                   int(retire_inst.tick),
                                   retire_inst).ldst_addr_info)
    return find_match


def _retire_inst_tick(retire_inst, issue_inst, show_detail, instr_exec_list):
    """retire inst tick"""
    if int(retire_inst.tick) < int(issue_inst.tick):
        if show_detail:
            print("Inst Latency : ", retire_inst.program_counter,
                  retire_inst.tick, "          ",
                  '---', "\t", retire_inst.inst_name,
                  retire_inst.inst_detail,
                  retire_inst.get_repeat0_info())

        retire_inst.merged = True
        instr_exec_list.append(InstrExecutionRecord(int(retire_inst.tick),
                                                    int(retire_inst.tick),
                                                    retire_inst))


def _rest_inst(rest_start_idx, retire_list_len,  # pylint: disable=R0913
               show_detail, retire_list, instr_exec_list):
    """rest inst"""
    if show_detail:
        print('\nRest instr in retire inst list:')
    while rest_start_idx < retire_list_len:
        rest = retire_list[rest_start_idx]
        rest_start_idx += 1
        if rest.merged:
            continue
        if show_detail:
            print("Inst Latency : ", rest.program_counter, rest.tick, "          ",
                  '---', "\t", rest.inst_name)
        if int(rest.tick) > InstrExecutionRecord.end_point:
            InstrExecutionRecord.end_point = int(rest.tick)
        instr_exec_list.append(InstrExecutionRecord(int(rest.tick),
                                                    int(rest.tick),
                                                    rest))


def _check_find_match(find_match, issue_inst, retire_inst):
    """check find match"""
    if not find_match:
        print("Error: issue & retire instr does not match")
        issue_inst.display()
        retire_inst.display()


def merge_issue_retire_info(issue_list, retire_list, show_detail):
    """
    use to merge issue retire info
    :param issue_list: issue list to merge
    :param retire_list: retire list to merge
    :param show_detail: whether show or not
    :return:
    """
    # too many attributes
    instr_exec_list = []

    if show_detail:
        print("Statistic of Instruction Latency:")
        print("Inst Latency : ", "   InstrPC", "IssueClk", "Issue2Clk",
              "FinshClk", "Latency", "\tPipe", "\tInstName")

    issue_list_len = len(issue_list)
    retire_list_len = len(retire_list)

    i = 0
    remainder_retire_start_idx = 0
    while i < issue_list_len:
        issue_inst = issue_list[i]
        i += 1
        find_match = False

        idx = remainder_retire_start_idx

        while idx < retire_list_len:
            retire_inst = retire_list[idx]
            idx += 1
            if retire_inst.merged:
                continue

            if int(retire_inst.tick) > InstrExecutionRecord.end_point:
                InstrExecutionRecord.end_point = int(retire_inst.tick)

            if retire_inst.program_counter == issue_inst.program_counter \
                    and not retire_inst.is_repeat0_nop:
                find_match = _retire_inst(retire_inst, issue_inst,
                                          instr_exec_list, show_detail)

                break
            if is_event_inst(retire_inst.inst_name) or \
                retire_inst.inst_pipe == 'FLOWCTRL' or \
                retire_inst.is_repeat0_nop:
                _retire_inst_tick(retire_inst, issue_inst, show_detail,
                                  instr_exec_list)
                continue


        # while retireList end
        idx = remainder_retire_start_idx
        while idx < retire_list_len:
            idx += 1
            if retire_list[idx - 1].merged:
                remainder_retire_start_idx = idx
            else:
                break

        _check_find_match(find_match, issue_inst, retire_inst)

    #While issueList End

    # if still something in retire issue queue
    rest_start_idx = remainder_retire_start_idx
    _rest_inst(rest_start_idx, retire_list_len,
               show_detail, retire_list, instr_exec_list)
    #end of merge
    return instr_exec_list

#----------------------
HTML_HEADER_STR = '''
<!DOCTYPE html>
<html>
	<head>
		<meta http-equiv="Content-Type" content="text/html;charset=utf-8">
		<title>InstrTraceViewer 0.1</title>
		<script type="text/javascript" src="js/d3.v3.min.js" charset="utf-8"></script>
		<script type="text/javascript" src="js/moment.min.js"></script>
		<script type="text/javascript" src="js/timeline.d3.min.js"></script>
		<link href="http://fonts.googleapis.com/css?family=Cantata+One" rel="stylesheet" type="text/css">
		<link href="http://fonts.googleapis.com/css?family=Imprima" rel="stylesheet" type="text/css">
		<link rel="stylesheet" href="css/timeline.d3.css" type="text/css" media="screen" />
		<style>
		h1 {
			font-family: 'Cantata One', Georgia, serif;
			font-size: 38px;
			line-height: 38px;
			color: #999;
		}
		p {
			font-family: 'Imprima', Verdana, Helvetica, sans-serif;
			font-size: 16px;
			line-height: 12px;
		}
		div.timeline {
			width: 960px;
			margin: 0 auto;
		}
		#phpTimeline {
			width: 960px;
			margin: 1em auto;
		}
		a {
			text-decoration: none;
		}
		</style>
	</head>
	<body>
		<div class="timeline">
			<h1>InstrTraceViewer 0.1</h1>
                        <h3>      By Compiler Lab</h3>
			<div id="phpTimeline"></div>
			<p>Click above to zoom in the timeline.</p>
			<p>Drag left and right border of the selected zone to decrease or increase the zoom, or drag the selected zone to travel in time.</p>
		</div>
		<script type="text/javascript">
		var data = [
'''

HTML_TAIL_STR = '''
		];

		var phpTimeline = d3.timeline.build(data, '#phpTimeline');
		</script>
	</body>
</html>

'''

def get_render_pipe_key(obj):
    """
    get render pipe key
    :param obj:
    :return: render pipe key
    """
    return obj.render_pipe_name

def _render_instr_when_render_item(sorted_instr_m, scalar_only, lane_idx,
                                   temp_file):
    """render instruction when render_item is True"""
    id_str = "%s.%s.%s" % \
             (sorted_instr_m.instr_pc,
              sorted_instr_m.instr_name, sorted_instr_m.bank_conflict)
    lane_str = sorted_instr_m.pipe

    if scalar_only.find("unfold") != -1:
        if lane_idx == 4:
            lane_idx = 0
        lane_str += ".%d" % lane_idx
        lane_idx += 1
    if is_event_inst(sorted_instr_m.instr_name):
        id_str += sorted_instr_m.instr_detail
    temp_file.writelines("{\"lane\":\"%s\", \"id\":\"%s\","
                         " \"start\":%d, \"end\":%d},\n"
                         % (lane_str, id_str, sorted_instr_m.start,
                            sorted_instr_m.end + 1))


def _render_execution(sorted_instr_exe_list, has_window,  # pylint: disable=R0913
                      window_start, window_end, scalar_only, temp_file):
    """render instruction execution"""
    # render instruction execution
    lane_idx = 0
    for sorted_instr_m in sorted_instr_exe_list:
        render_item = False
        if not has_window:
            render_item = True
        elif has_window and sorted_instr_m.start >= window_start and \
                sorted_instr_m.start < window_end:
            render_item = True

        if scalar_only.find("yes") != -1 and \
                (sorted_instr_m.pipe != "SCALAR" and
                 sorted_instr_m.pipe != ".scalar_ldst"):
            render_item = False

        if render_item:
            _render_instr_when_render_item(sorted_instr_m, scalar_only,
                                           lane_idx,
                                           temp_file)


def render_instruction_trace(test_name, instr_exe_list,  # pylint: disable=R0913
                             icache_list, window_start, window_end,
                             scalar_only, kernel_path):
    """
    use to render instruction trace
    :param test_name: case name
    :param instr_exe_list: instr list
    :param icache_list: icache list
    :param window_start: window start
    :param window_end:  window end
    :param scalar_only: whether scalar only
    :param kernel_path: where to put path
    :return:
    """
    # too many arguments, so disable R0913
    has_window = False
    if window_start is not None and window_end is not None:
        if window_end < window_start:
            print("Warning -s > -e arg! Default to no window.")
            has_window = False
        else:
            has_window = True

    sorted_instr_exe_list = sorted(instr_exe_list, key=get_render_pipe_key)

    with open(os.path.join(kernel_path, test_name + ".html"), "w") as temp_file:
        temp_file.writelines(HTML_HEADER_STR)

        # render instruction execution
        _render_execution(sorted_instr_exe_list, has_window,
                          window_start, window_end, scalar_only, temp_file)

        # render icache req
        for icache in icache_list:
            render_item = False
            if not has_window:
                render_item = True
            elif has_window and icache.start >= window_start and \
                    icache.start < window_end:
                render_item = True

            if render_item:
                temp_file.writelines("{\"lane\":\"ICmiss\", \"id\":\"%s-%s\","
                                     " \"start\":%d, \"end\":%d},\n"
                                     % (icache.addr, icache.req_id,
                                        icache.start, icache.end))

        temp_file.writelines(HTML_TAIL_STR)


def get_macc_usage(dump_file_name):
    """
    use to get macc usage
    :param dump_file_name:
    :return: cycle number
    """
    if not os.path.exists(dump_file_name):
        return 0

    exe_cmd = 'grep -c "Mac_Usage: 1.000000" %s ' % dump_file_name
    read_obj = os.popen(exe_cmd)
    cycle = int(read_obj.readlines()[0])
    read_obj.close()
    return cycle


def set_format_perf_dict(  # pylint: disable=R0913
        instr_num_dict, format_str_dict,
        perf_regression_inst_str_dict,
        instr_cycle_accum_dict, instr_cycle_piped_dict,
        instr_cycle_idle_dict, test_name, _start_point, _end_point):
    """set format_str_dict and perf_regression_inst_str_dict"""
    for key in sorted(instr_num_dict.keys()):
        format_str_dict[key] = "%-14s #I: %8d, " % (key, instr_num_dict[key])
        perf_regression_inst_str_dict[key] = "%s, %s, #I, %d\n" % \
                                         (test_name, key, instr_num_dict[key])
    for key in sorted(instr_cycle_accum_dict.keys()):
        format_str_dict[key] += "Accum: %8d, " % instr_cycle_accum_dict[key]
        perf_regression_inst_str_dict[key] += "%s, %s, Accum, %d\n" \
                                          % (test_name, key,
                                             instr_cycle_accum_dict[key])
    for key in sorted(instr_cycle_piped_dict.keys()):
        value = instr_cycle_piped_dict[key]
        format_str_dict[key] += \
            "Piped: %8d (%s), " % (value, "{:7.2%}".
                                   format(float(value)/
                                          (_end_point - _start_point)))
        perf_regression_inst_str_dict[key] += \
            "%s, %s, Piped, %d\n" % (test_name, key, value)
    for key in sorted(instr_cycle_idle_dict.keys()):
        format_str_dict[key] += "Idle: %8d \t=> %8d" % \
                              (instr_cycle_idle_dict[key], instr_cycle_idle_dict[key]
                               + instr_cycle_piped_dict[key])


def _get_stack_addr(subtarget):
    """get stack addr"""
    if subtarget == "dav-m100":
        stack_addr = int('0x3c000', 16)
    elif subtarget == "dav-c100" or "dav-l100" or "dav-t100":
        stack_addr = int('0x40000', 16)
    else:
        stack_addr = int('0x3c000', 16)
    return stack_addr


def _per_func(instr_exe_m, prev_node_dict,  # pylint: disable=R0913
              current_max_end_dict, instr_num_dict, instr_cycle_accum_dict,
              instr_cycle_piped_dict, instr_cycle_idle_dict,
              accum_delta, _start_point):
    """per function"""
    piped_delta = 0
    if instr_exe_m.pipe not in prev_node_dict:
        # first inst for the pipe
        prev_node_dict[instr_exe_m.pipe] = None

    if (prev_node_dict[instr_exe_m.pipe] is not None) and \
            (instr_exe_m.start <= current_max_end_dict[instr_exe_m.pipe]):
        # if: instr overlapped in the same pipe, no plus 1
        if instr_exe_m.end > current_max_end_dict[instr_exe_m.pipe]:
            piped_delta = instr_exe_m.end - \
                          current_max_end_dict[instr_exe_m.pipe]
    else:
        # 1. instr does not overlapped with previous instr in same pipe
        # 2. it is the first instr for new pipe
        # 3. have plus 1
        piped_delta = instr_exe_m.end - instr_exe_m.start + 1

    # Update per pipe previous instrution and cycle info
    if prev_node_dict[instr_exe_m.pipe] is None:
        instr_num_dict[instr_exe_m.pipe] = 1
        instr_cycle_accum_dict[instr_exe_m.pipe] = accum_delta
        instr_cycle_piped_dict[instr_exe_m.pipe] = piped_delta
        instr_cycle_idle_dict[
            instr_exe_m.pipe] = instr_exe_m.start - _start_point
        current_max_end_dict[instr_exe_m.pipe] = instr_exe_m.end
    else:
        instr_num_dict[instr_exe_m.pipe] += 1
        instr_cycle_accum_dict[instr_exe_m.pipe] += accum_delta
        instr_cycle_piped_dict[instr_exe_m.pipe] += piped_delta
        if (instr_exe_m.start - current_max_end_dict[instr_exe_m.pipe]) > 1:
            instr_cycle_idle_dict[instr_exe_m.pipe] += \
                (instr_exe_m.start -
                 current_max_end_dict[instr_exe_m.pipe]) - 1
        if instr_exe_m.end > current_max_end_dict[instr_exe_m.pipe]:
            current_max_end_dict[instr_exe_m.pipe] = instr_exe_m.end


def _set_access_list(instr_exe_m, stack_addr, stack_access, flow_table_access):
    """set access list"""
    if instr_exe_m.ldst_addr >= stack_addr:
        if instr_exe_m.instr_name.find("scalar_ld") != -1:
            stack_access[0] += 1
        else:
            stack_access[1] += 1
    elif instr_exe_m.ldst_addr != 0:
        if instr_exe_m.instr_name.find("scalar_ld") != -1:
            flow_table_access[0] += 1
        else:
            flow_table_access[1] += 1


def gather_instr_exe_statistics(test_name,  # pylint: disable=R0914, R0915
                                instr_exe_list, subtarget):
    """
    gather instr execution info
    :param test_name: case name
    :param instr_exe_list: instr exe list
    :param subtarget: target
    :return:
    """

    # Assume The Cycle Start from cycle 0
    # +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
    # | 0 | 1 | 2 | 3 | 4 | 5 | 6 | ....
    # +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
    # One Cycle Instruction:
    #     +---+
    #     | 1 |
    #     +---+
    # Each Cycle there's no instruction, will count as an idle cycle.
    # Here is a two cycle Idle case (cycle 0 and 2 is idling)
    # +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
    # |   | 1 |   | 3 | 4 | 5 | 6 | MMAD Inst
    # +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
    # too many attributes


    _start_point = InstrExecutionRecord.start_point
    _end_point = InstrExecutionRecord.end_point

    instr_num_dict = {"SCALAR": 0,
                      ".scalar_ldst": 0,
                      "FLOWCTRL": 0,
                      "VECTOR": 0,
                      "CUBE": 0,
                      "MTE2": 0,
                      "MTE1": 0,
                      "MTE3": 0,
                      ".event": 0}

    instr_cycle_accum_dict = {"SCALAR":0,
                              ".scalar_ldst":0,
                              "FLOWCTRL":0,
                              "VECTOR":0,
                              "CUBE":0,
                              "MTE2":0,
                              "MTE1":0,
                              "MTE3":0,
                              ".event":0}

    instr_cycle_piped_dict = {"SCALAR": 0,
                              ".scalar_ldst": 0,
                              "FLOWCTRL": 0,
                              "VECTOR": 0,
                              "CUBE": 0,
                              "MTE2": 0,
                              "MTE1": 0,
                              "MTE3": 0,
                              ".event": 0}

    instr_cycle_idle_dict = {"SCALAR": 0,
                             ".scalar_ldst": 0,
                             "FLOWCTRL": 0,
                             "VECTOR": 0,
                             "CUBE": 0,
                             "MTE2": 0,
                             "MTE1": 0,
                             "MTE3": 0,
                             ".event": 0}

    instr_end_tick_dict = {"SCALAR": 0,
                           ".scalar_ldst": 0,
                           "FLOWCTRL": 0,
                           "VECTOR": 0,
                           "CUBE": 0,
                           "MTE2": 0,
                           "MTE1": 0,
                           "MTE3": 0,
                           ".event": 0}

    current_max_end_dict = {"SCALAR": 0,
                            "VECTOR": 0,
                            "CUBE": 0,
                            "MTE2": 0,
                            "MTE1": 0,
                            "MTE3": 0,
                            ".event": 0}

    prev_node_dict = {}

    total_cycle_idle = 0
    current_max_cycle = 0  # record the max retire cycle of reviewed instr
    flow_table_access = [0, 0]
    stack_access = [0, 0]

    for instr_exe_m in instr_exe_list:
        # Keep track of the final ending tick for each pipe
        if (instr_end_tick_dict[instr_exe_m.pipe] is None) or\
                (instr_exe_m.end > instr_end_tick_dict[instr_exe_m.pipe]):
            instr_end_tick_dict[instr_exe_m.pipe] = instr_exe_m.end

        accum_delta = instr_exe_m.end - instr_exe_m.start + 1
        stack_addr = _get_stack_addr(subtarget)
        # LdSt address statistic
        _set_access_list(instr_exe_m, stack_addr,
                         stack_access, flow_table_access)

        # per pipe statistic
        _per_func(instr_exe_m, prev_node_dict, current_max_end_dict,
                  instr_num_dict, instr_cycle_accum_dict,
                  instr_cycle_piped_dict, instr_cycle_idle_dict,
                  accum_delta, _start_point)

        prev_node_dict[instr_exe_m.pipe] = instr_exe_m
        # Assume issue rate is 1 inst/cycle
        if (instr_exe_m.start - current_max_cycle) > 1:
            total_cycle_idle += (instr_exe_m.start - current_max_cycle) - 1
        if instr_exe_m.end > current_max_cycle:
            current_max_cycle = instr_exe_m.end


    # Calcuate the last portion of idle cycles which is from the last instruction end in
    # the pipe to the max ending cycle in all pipes (i.e: _endPoint), this is caused by
    # different pipe could retire at different time (this is usually the case).
    for key, value in instr_end_tick_dict.items():
        if value < _end_point:
            # Need to adjust by one, if no instructions in a pipe, in which case value is 0
            instr_cycle_idle_dict[key] += _end_point - value + (1 if value == 0 else 0)

    perf_regression_record_str = ""

    totalformat_str_dict = "%s %s TotalLatency: %d, TotalIdleCycle: %d, " \
                         "FlowTableLdSt: %d, %d, StackAccess: %d, %d"  \
        % (test_name, ConfigInfo.coreId, _end_point - _start_point + 1, total_cycle_idle,
           flow_table_access[0], flow_table_access[1],
           stack_access[0], stack_access[1])
    print(totalformat_str_dict)

    format_str_dict = {}
    perf_regression_inst_str_dict = {}

    set_format_perf_dict(
        instr_num_dict, format_str_dict,
        perf_regression_inst_str_dict,
        instr_cycle_accum_dict, instr_cycle_piped_dict,
        instr_cycle_idle_dict, test_name, _start_point, _end_point)

    for key in sorted(format_str_dict.keys()):
        print(format_str_dict[key])
        totalformat_str_dict += ("\n" + format_str_dict[key])
        perf_regression_record_str += perf_regression_inst_str_dict[key]

    perf_regression_record_str += "%s, total, TotalLatency, %d\n" %\
                               (test_name, _end_point - _start_point + 1)
    perf_regression_record_str += "%s, total, TotalIdleCycle, %d\n" % \
                               (test_name, total_cycle_idle)

    # Get real MACC usage, tmp simple implement
    cube_cycle = get_macc_usage(ConfigInfo.cubeDump)
    vector_cycle = get_macc_usage(ConfigInfo.vectorDump)
    scalar_cycle = get_macc_usage(ConfigInfo.scalarDump)
    cubeformat_str_dict = "%-14s #I: %8d, Accum: %8d, Piped: %8d (%s)" % \
                        ("CubeMACC", cube_cycle, cube_cycle, cube_cycle, "{:7.2%}"
                         .format(float(cube_cycle)/(_end_point - _start_point)))
    vectorformat_str_dict = "%-14s #I: %8d, Accum: %8d, Piped: %8d (%s)" % \
                          ("VectorMACC", vector_cycle, vector_cycle, vector_cycle, "{:7.2%}"
                           .format(float(vector_cycle)/(_end_point - _start_point)))
    scalarformat_str_dict = "%-14s #I: %8d, Accum: %8d, Piped: %8d (%s)" % \
                          ("ScalarMACC", scalar_cycle, scalar_cycle, scalar_cycle,
                           "{:7.2%}"
                           .format(float(scalar_cycle)/(_end_point - _start_point)))
    print(cubeformat_str_dict)
    print(vectorformat_str_dict)
    print(scalarformat_str_dict)
    totalformat_str_dict += "\n" + cubeformat_str_dict + "\n" +\
                          vectorformat_str_dict\
                          + "\n" + scalarformat_str_dict + "\n"
    with open("./" + test_name + ".data", "w") as temp_file:
        temp_file.writelines(totalformat_str_dict)
    # save the line to file for test regression
    with open("./" + test_name + ".perf.csv", "w") as temp_file:
        temp_file.writelines(perf_regression_record_str)

class LdstInstFreq:
    """
    save ldst inst
    """
    def __init__(self, instr_name, instr_binary):
        self.instr_name = instr_name
        self.cnt = 1
        self.binary = instr_binary

    def __str__(self):
        return "LdstInstFreq"

    def __hash__(self):
        return "LdstInstFreq"

def get_scalar_ldst_hot_spot(test_name, instr_exe_list, base_pc):
    """
    use to get scalar ldst hot spot
    :param test_name:
    :param instr_exe_list:
    :param base_pc:
    :return:
    """
    ldst_hot_spot_statistic = {}

    data_str = ""
    for instr_exe_m in instr_exe_list:
        if instr_exe_m.instr_pc not in ldst_hot_spot_statistic:
            ldst_hot_spot_statistic[instr_exe_m.instr_pc] = LdstInstFreq(
                instr_exe_m.instr_name, instr_exe_m.instr_binary)
        else:
            ldst_hot_spot_statistic[instr_exe_m.instr_pc].cnt += 1

    for key in sorted(ldst_hot_spot_statistic.keys()):
        base = int(base_pc, 16)
        offset = int(key, 16)
        offset_in_asm = offset - base
        data_str += "%s : %s : %s : %d : %s\n" % \
                    (key, hex(offset_in_asm), ldst_hot_spot_statistic[key].binary,
                     ldst_hot_spot_statistic[key].cnt,
                     ldst_hot_spot_statistic[key].instr_name)

    with open("./" + test_name + ".hotSpot.txt", "w") as temp_file:
        temp_file.writelines(data_str)


def get_critical_path_stats(test_name,  # pylint: disable=R0914, R0912
                            instr_exe_list):
    """
    use to get critical path stats
    :param test_name:
    :param instr_exe_list:
    :return:
    """
    # too many attributes
    # Create instruction cycle status dictionary according to pipe type.
    cycle_to_status_dict = {"ALL_SCALAR": {},
                            "VECTOR": {},
                            "CUBE": {},
                            "ALL_MTE": {}}

    for instr_exe_m in instr_exe_list:
        # SCALAR, .scalar_ldst, .event and FLOWCTRL all belongs to the scalar pipe.
        scalar_keys = {"SCALAR", ".scalar_ldst", "FLOWCTRL", ".event"}
        mte_keys = {"MTE1", "MTE2", "MTE3"}
        for cycle in range(instr_exe_m.start, instr_exe_m.end + 1):
            if instr_exe_m.pipe in scalar_keys:
                cycle_to_status_dict["ALL_SCALAR"][cycle] = "BUSY"
            elif instr_exe_m.pipe in mte_keys:
                cycle_to_status_dict["ALL_MTE"][cycle] = "BUSY"
            else:  # CUBE or VECTOR
                cycle_to_status_dict[instr_exe_m.pipe][cycle] = "BUSY"

    critical_cycle_dict = {"ALL_SCALAR": 0,
                           "VECTOR": 0,
                           "CUBE": 0,
                           "ALL_MTE": 0}

    total_cycle_dict = {"ALL_SCALAR": 0,
                        "VECTOR": 0,
                        "CUBE": 0,
                        "ALL_MTE": 0}

    for cycle in range(InstrExecutionRecord.start_point, InstrExecutionRecord.end_point + 1):
        for pipe in critical_cycle_dict:
            other_pipes = set(critical_cycle_dict.keys()) - {pipe}
            if cycle in cycle_to_status_dict[pipe]:
                total_cycle_dict[pipe] += 1
            other_pipes_busy = False
            for other_pipe in other_pipes:
                if cycle in cycle_to_status_dict[other_pipe]:
                    other_pipes_busy = True
                    break

            # Other pipes are all idling
            if not other_pipes_busy:
                critical_cycle_dict[pipe] += 1

    data_str = ""
    for pipe in critical_cycle_dict:
        critical_cycle_perecent = 0.0
        if total_cycle_dict[pipe] != 0:
            critical_cycle_perecent = float(critical_cycle_dict[pipe]) / total_cycle_dict[pipe]
        global_critical_cycle_perecent = float(critical_cycle_dict[pipe]) / \
                                         (InstrExecutionRecord.end_point -
                                          InstrExecutionRecord.start_point + 1)
        data_str += "Current pipe name : " + pipe + "\n"
        data_str += "total cycles                                            " \
                   + "{:7}".format(total_cycle_dict[pipe]) + "\n"
        data_str += "total cycles on critical path                           " \
                   + "{:7}".format(critical_cycle_dict[pipe]) + "\n"
        data_str += "critical path cycles / current pipe total piped cycles  " \
                   + "{:7.2%}".format(critical_cycle_perecent) + "\n"
        data_str += "critical path cycles / global total latency             " \
                   + "{:7.2%}".format(global_critical_cycle_perecent) + "\n\n\n"
    with open("./" + test_name + ".criticalCycleStats.txt", "w") as temp_file:
        temp_file.writelines(data_str)


def _core0_core1(coreid, inst_dump_file, inst_popped_file, ca_log_path):
    """gen for core0 and core1"""
    if coreid in ('core0', 'core1'):
        ConfigInfo.dumpVersion = 2
        if inst_dump_file == "" or inst_popped_file == "":
            ConfigInfo.instrRetiredDump = os.path.join(ca_log_path, coreid +
                                                       "_instr_log.dump")
            ConfigInfo.instrPoppedDump = os.path.join(ca_log_path, coreid +
                                                      "_instr_popped_log.dump")
    elif coreid == "none":
        ConfigInfo.dumpVersion = 1
        if inst_dump_file == "" or inst_popped_file == "":
            ConfigInfo.instrRetiredDump = os.path.join(ca_log_path, "instr.dump")
            ConfigInfo.instrPoppedDump = os.path.join(ca_log_path, "instr_popped.dump")

    else:
        print("Error: invalid arg for coreid")
        sys.exit(2)


def _dump_version(dump_dir):
    """dump by version to get cachename"""
    if ConfigInfo.dumpVersion == 2:
        ConfigInfo.mteBiuReqDump = \
            dump_dir + "/%s_mte_biu_req_log.dump" % ConfigInfo.coreId
        ConfigInfo.icacheDump = \
            dump_dir + "/%s_icache_log.dump" % ConfigInfo.coreId
        ConfigInfo.cubeDump = dump_dir + "/%s_cube_log.dump" % ConfigInfo.coreId
        ConfigInfo.vectorDump = dump_dir + "/%s_vector_log.dump" % \
                                ConfigInfo.coreId
        ConfigInfo.scalarDump = dump_dir + "/%s_scalar_log.dump" % \
                                ConfigInfo.coreId
        cache_name = "icache"
    elif ConfigInfo.dumpVersion == 1:
        ConfigInfo.mteBiuReqDump = dump_dir + "/mte_biu_req.dump"
        ConfigInfo.icacheDump = dump_dir + "/icache.dump"
        ConfigInfo.cubeDump = dump_dir + "/cube.dump"
        ConfigInfo.vectorDump = dump_dir + "/vector.dump"
        ConfigInfo.scalarDump = dump_dir + "/scalar.dump"
        cache_name = "cache"
    return cache_name


def _check_config(show_config):
    """check config"""
    if show_config:
        ConfigInfo.display_all_config_info()
        print("\n")

    if not os.path.exists(ConfigInfo.mteBiuReqDump):
        print("Error: No %s file, cannot tell if it is CAModel or not!!! ABORT"
              % ConfigInfo.mteBiuReqDump)
        sys.exit(2)

    with open(ConfigInfo.mteBiuReqDump, "r") as temp_file:
        print("mtebiu is", ConfigInfo.mteBiuReqDump)
        line = temp_file.readlines()
        if not line:
            print("Error: This is not run on CAModel!!! ABORT!")
            sys.exit(2)


def _get_inst_list():
    """get retire inst"""
    # Parse instr.dump file
    retire_list = None
    if ConfigInfo.instrRetiredDump != "":
        retire_list = parse_inst_dump_file(ConfigInfo.instrRetiredDump)

    # if do more anaysis
    issue_list = None
    if ConfigInfo.instrPoppedDump != "":
        issue_list = parse_inst_popped_dump_file(ConfigInfo.instrPoppedDump)
    return retire_list, issue_list


def _get_opt_arg_list(opt, arg, param_dict):  # pylint: disable=R0912
    """get param by opt"""
    if opt in ("-i", "--instr_dump"):
        param_dict["inst_dump_file"] = arg
    elif opt in ("-t", "--testname"):
        ConfigInfo.testName = arg
    elif opt in ("-p", "--instr_popped"):
        param_dict["inst_popped_file"] = arg
    elif opt in ("-s", "--start"):
        param_dict["start_window"] = arg
    elif opt in ("-e", "--end"):
        param_dict["end_window"] = int(arg)
    elif opt == '-v':
        param_dict["showinstr_detail"] = True
    elif opt in ("-b", "--base_pc"):
        param_dict["base_pc"] = arg
    elif opt in ("-r", "--render_mode"):
        param_dict["only_render_scalar_trace"] = arg
    elif opt in ("-c", "--coreid"):
        param_dict["coreid"] = arg
    elif opt in ("-S", "--subtarget"):
        param_dict["subtarget"] = arg
    elif opt in ("", "--show_config"):
        param_dict["show_config"] = True
    elif opt in ("-d", "--ca_log_path"):
        param_dict["ca_log_path"] = arg
    elif opt in ("-k", "--kernel_path"):
        param_dict["kernel_path"] = arg


def gen_html(argv):  # pylint: disable=R0914
    """
    use to run case
    :param argv:
    :return:
    """
    # too many attributes
    # Help Msg:
    help_msg = '''
    Must Options:
         -c <coreid, eg core0, core1; specify *none* for the very old CAModel dump format>
         -t <test name>
    Optional Options:
         -i <specify the instr.dump file, when coreid=none>
         -p <specify the instr_popped.dump file, when coreid=none>
         -r <render mode>
         -s <start cycle for visualization>
         -e <end cycle for visualization>
         -b <specify base_pc that device binary loaded>
         -v <show detailed instr items>
         -S <add the information of subtarget, such as dav-m100/dav-c100>
    '''
    cache_name = ""
    try:
        opts, _ = getopt.getopt(argv, "hi:p:t:s:e:b:vr:c:S:d:k:",
                                ["instr_dump=", "instr_popped=", "testname=",
                                 "start=", "end=", "base_pc=", "render_mode=",
                                 "subtarget=", "coreid=", "show_config",
                                 "ca_log_path=", "kernel_path="])
    except getopt.GetoptError:
        print(help_msg)
        sys.exit(2)

    param_dict = {}
    for opt, arg in opts:
        if opt == '-h':
            print(help_msg)
            sys.exit()
        else:
            _get_opt_arg_list(opt, arg, param_dict)

    # Parse command line args
    inst_dump_file = param_dict.get("inst_dump_file", "")
    inst_popped_file = param_dict.get("inst_popped_file", "")
    start_window = param_dict.get("start_window", None)
    end_window = param_dict.get("end_window", None)
    showinstr_detail = param_dict.get("showinstr_detail", False)
    only_render_scalar_trace = param_dict.get("only_render_scalar_trace", "no")
    show_config = param_dict.get("show_config", False)
    base_pc = param_dict.get("base_pc", "0x3000000")
    # default coreid is setting to core0
    coreid = param_dict.get("coreid", "core0")
    # default subtarget is setting to dav-100
    subtarget = param_dict.get("subtarget", "dav-m100")
    ca_log_path = param_dict.get("ca_log_path", "")
    kernel_path = param_dict.get("kernel_path", "")

    ConfigInfo.coreId = coreid
    ConfigInfo.instrPoppedDump = inst_popped_file
    ConfigInfo.instrRetiredDump = inst_dump_file
    _core0_core1(coreid, inst_dump_file, inst_popped_file, ca_log_path)

    dump_dir = os.path.dirname(os.path.realpath(ConfigInfo.instrRetiredDump))
    cache_name = _dump_version(dump_dir)

    _check_config(show_config)

    retire_list, issue_list = _get_inst_list()

    m_list = merge_issue_retire_info(issue_list, retire_list, showinstr_detail)

    i_cache_info = IcacheMissList(ConfigInfo.icacheDump, cache_name)
    render_instruction_trace(ConfigInfo.testName, m_list, i_cache_info.records,
                             start_window, end_window, only_render_scalar_trace,
                             kernel_path)

    gather_instr_exe_statistics(ConfigInfo.testName, m_list, subtarget)
    i_cache_info.display_stat(ConfigInfo.testName)
    get_scalar_ldst_hot_spot(ConfigInfo.testName, m_list, base_pc)
    get_critical_path_stats(ConfigInfo.testName, m_list)
