#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright 2018 Huawei Technologies Co., Ltd

auto_schedule distribution rules

Add your rules here so cce auto_schedule can distribute your operators
to corresponding schedule correctly
"""
from te import platform
from te.platform import get_soc_spec
from .cce_schedule_declarations import OpFlags
from .util import _check_pattern_matched
from .util import get_reduce_axes
from .util import get_all_axes
from .op_pattern import softmax_dfs_tag_list
from .op_pattern import bn_reduce_pattern_list
from .op_pattern import bn_update_grad_pattern_list
from .op_pattern import bn_update_pattern_list
from .op_pattern import bn_grad_reduce_pattern_list
from .op_pattern import layer_norm_grad_pattern_list
from .op_pattern import softmax_cross_entropy_with_logits_pattern_list
from .op_pattern import l2_loss_pattern_list
from .op_pattern import l2loss_mul_addn_pattern_list
from .op_pattern import l2_normalize_dfs_tag_list
from .op_pattern import relu_v2_tag_list
from .op_pattern import reduce_mean_2d_aligned_mid_reduce_no_cast_list


class OpPatternRules:
    """Rules for matching patterns"""
    @staticmethod
    def simple_pattern_rule(flags: dict, pattern, mapping, *args) -> bool:
        """Simple pattern rule"""
        list(args).clear()  # Use it once to avoid static checks
        result = False
        if not isinstance(mapping[pattern], tuple):
            if flags[mapping[pattern].value]["count"] > 0:
                result = True
        else:
            for flag in mapping[pattern]:
                if flags[flag.value]["count"] > 0:
                    result = True
                    break
        return result

    @staticmethod
    def simple_pattern_rule_conv_phase(flags: dict, pattern, mapping, *args) -> bool:
        """Simple pattern rule for convolution phase"""
        return OpPatternRules.simple_pattern_rule(flags, pattern, mapping, *args)

    @staticmethod
    def simple_pattern_rule_quant_phase(flags: dict, pattern, mapping, *args) -> bool:
        """Simple pattern rule for quant&dequant phase"""
        return OpPatternRules.simple_pattern_rule(flags, pattern, mapping, *args)

    @staticmethod
    def single_only_pattern_rule(flags: dict, pattern, mapping, *args) -> bool:
        """Single action pattern rule"""
        list(args).clear()  # Use it once to avoid static checks
        result = False
        if flags[mapping[pattern].value]["count"] == 1:
            if flags["total"]["count"] - flags[None]["count"] \
                    - flags[mapping[pattern].value]["count"] == 0:
                result = True
        return result

    @staticmethod
    def softmax_pattern_rule(flags: dict, input_tensors: list,
                             output_tensors: list, dfs_tensor_list: list,
                             *args) -> bool:
        """softmax pattern rule"""
        [flags, input_tensors, args].clear()  # Use it once to avoid static checks
        is_match = False
        out_len = 1
        if len(output_tensors) == out_len:
            for tag_list in softmax_dfs_tag_list:
                if _check_pattern_matched(dfs_tensor_list, tag_list):
                    is_match = True
                    break
        return is_match

    @staticmethod
    def reduce_pattern_rule(flags: dict, *args) -> bool:
        """Reduce pattern rule"""
        list(args).clear()  # Use it once to avoid static checks
        result = False
        if flags[OpFlags.reduce_flag.value]["count"] == 1:
            if flags["total"]["count"] - flags[None]["count"] \
                    - flags[OpFlags.reduce_flag.value]["count"] \
                    - flags[OpFlags.elewise_flag.value]["count"] == 0:
                reduce_tensor = flags[OpFlags.reduce_flag.value]["tensors"][0]
                reduce_tensor_body = reduce_tensor.op.body
                reduce_tensor_axes = reduce_tensor_body[0].axis
                reduce_axes = []
                for reduce_axis in reduce_tensor_axes:
                    reduce_axes.append(reduce_axis.var.name)
                last_axis = reduce_tensor_body[0].source[0].args[
                    len(reduce_tensor_body[0].source[0].args) - 1
                ]
                if last_axis.name not in reduce_axes:
                    result = True
        return result

    @staticmethod
    def elewise_pattern_rule(flags: dict, *args) -> bool:
        """Elewise pattern rule"""
        list(args).clear()  # Use it once to avoid static checks
        result = False
        expected_flag_num = flags[OpFlags.elewise_flag.value]["count"] \
            + flags[OpFlags.broadcast_flag.value]["count"]
        if expected_flag_num > 0:
            if flags["total"]["count"] - flags[None]["count"] - expected_flag_num == 0:
                result = True
        return result

    @staticmethod
    def bn_update_pattern_rule(flags: dict, input_tensors: list, output_tensors: dict,
                               dfs_tensor_list: list, *args):
        """Bn update pattern rule"""
        list([flags, input_tensors, args]).clear()  # Use it once to avoid static checks
        out_len = 5
        input_data_len = 5
        if (len(output_tensors) == out_len or len(output_tensors) == out_len + 1) \
                and len(output_tensors[0].shape) == input_data_len:
            is_match = False
            for tag_liat in bn_update_pattern_list:
                if _check_pattern_matched(dfs_tensor_list, tag_liat):
                    is_match = True
                    break

            return is_match

        return False

    @staticmethod
    def bn_grad_reduce_pattern_rule(flags: dict, input_tensors: list, output_tensors: dict,
                                    dfs_tensor_list: list, *args):
        """
        check bn grad reduce pattern
        """
        [flags, args].clear()  # Use it once to avoid static checks

        in_len = 7
        out_len = 1
        input_shape_len = 5
        if len(output_tensors) == out_len and len(input_tensors) == in_len \
                and len(output_tensors[0].shape) == input_shape_len:
            is_match = False
            for tag_list in bn_grad_reduce_pattern_list:
                if _check_pattern_matched(dfs_tensor_list, tag_list):
                    is_match = True
                    break

            return is_match

        return False

    @staticmethod
    def bn_update_grad_pattern_rule(flags: dict, input_tensors: list, output_tensors: dict,
                                    dfs_tensor_list: list, tensor_map: dict, *args):
        """
        check bn update grad pattern
        """
        list([flags, args]).clear()  # Use it once to avoid static checks
        requirements = {
            "out_len": 2,
            "in_len": 4,
            "input_data_len": 5,
            "nd_len": 4
        }
        flag = False
        if len(output_tensors) == requirements["out_len"] and \
                len(input_tensors) == requirements["in_len"]:
            is_out_all_tuple_reduce = True
            for i in range(0, requirements["out_len"]):
                if output_tensors[i].op.tag.find("tuple_reduce") == -1:
                    if output_tensors[i] not in tensor_map.keys():
                        is_out_all_tuple_reduce = False
            if is_out_all_tuple_reduce:
                flag = True

        if flag:
            if len(input_tensors[0].shape) == requirements["input_data_len"] \
                    or len(input_tensors[0].shape) == requirements["nd_len"]:
                is_match = False
                for tag_list in bn_update_grad_pattern_list:
                    if _check_pattern_matched(dfs_tensor_list, tag_list):
                        is_match = True
                        break

                return is_match

        return False

    @staticmethod
    def layer_norm_grad_pattern_rule(flags: dict, input_tensors: list, output_tensors: list,
                                     dfs_tensor_list: list, tensor_map: dict, *args):
        """
        check layer norm grad pattern
        """
        list([flags, args]).clear()  # Use it once to avoid static checks
        out_len = 2
        in_len = 4
        input_data_len = 2

        flag = False
        if len(output_tensors) == out_len and len(input_tensors) == in_len:
            is_out_all_tuple_reduce = True
            for i in range(0, out_len):
                if output_tensors[i].op.tag.find("tuple_reduce") == -1:
                    if output_tensors[i] not in tensor_map.keys():
                        is_out_all_tuple_reduce = False
            if is_out_all_tuple_reduce:
                flag = True

        if flag:
            if len(input_tensors[0].shape) >= input_data_len:
                is_match = False
                for tag_list in layer_norm_grad_pattern_list:
                    if _check_pattern_matched(dfs_tensor_list, tag_list):
                        is_match = True
                        break

                return is_match
        return False

    @staticmethod
    def l2loss_mul_addn_pattern_rule(flags: dict, input_tensors: list, output_tensors: list,
                                     dfs_tensor_list: list, *args):
        """
        check l2loss mul addn pattern
        """
        list([flags, args]).clear()  # Use it once to avoid static checks
        out_len = 2
        in_len = 3

        if not (len(output_tensors) == out_len and len(input_tensors) == in_len):
            return False

        is_match = False
        for tag_list in l2loss_mul_addn_pattern_list:
            if _check_pattern_matched(dfs_tensor_list, tag_list):
                is_match = True
                break

        return is_match

    @staticmethod
    def bn_reduce_pattern_rule(flags: dict, input_tensors: list, output_tensors: list,
                               dfs_tensor_list: list, *args):
        """
        check bn reduce pattern
        """
        list([flags, args]).clear()  # Use it once to avoid static checks
        out_len = 2
        in_len = 1
        input_data_len = 5
        flag = False
        if len(output_tensors) == out_len and len(input_tensors) == in_len:
            is_out_all_tuple_reduce = True
            for i in range(0, out_len):
                if output_tensors[i].op.tag.find("tuple_reduce") == -1:
                    is_out_all_tuple_reduce = False
            if is_out_all_tuple_reduce:
                flag = True

        if flag:
            if len(input_tensors[0].shape) == input_data_len:
                is_match = False
                for tag_list in bn_reduce_pattern_list:
                    if _check_pattern_matched(dfs_tensor_list, tag_list):
                        is_match = True
                        break

                return is_match

        return False

    @staticmethod
    def softmax_cross_entropy_with_logits_pattern_rule(flags: dict, input_tensors: list,
                                                       output_tensors: list, dfs_tensor_list: list,
                                                       *args):
        """
        check softmax cross entropy with logits pattern
        """
        list([flags, input_tensors, args]).clear()  # Use it once to avoid static checks
        out_len = 2
        input_data_len_2d = 2
        input_data_len_4d = 4

        if len(output_tensors) == out_len and (
                len(output_tensors[1].shape) == input_data_len_2d or
                len(output_tensors[1].shape) == input_data_len_4d):
            is_match = False
            for tag_list in softmax_cross_entropy_with_logits_pattern_list:
                if _check_pattern_matched(dfs_tensor_list, tag_list):
                    is_match = True
                    break

            return is_match

        return False

    @staticmethod
    def l2_normalize_pattern_rule(flags: dict, input_tensors: list, output_tensors: list,
                                  dfs_tensor_list: list, tensor_map: dict, *args):
        """
        check l2 normalize pattern
        """
        list([flags, tensor_map, args]).clear()  # Use it once to avoid static checks
        out_len = 1
        in_len = 1

        if len(output_tensors) == out_len and len(input_tensors) == in_len:
            is_match = False
            for tag_list in l2_normalize_dfs_tag_list:
                if _check_pattern_matched(dfs_tensor_list, tag_list):
                    is_match = True
                    break

            return is_match

        return False

    @staticmethod
    def l2_loss_pattern_rule(flags: dict, input_tensors: list, output_tensors: list,
                             dfs_tensor_list: list, *args):
        """
        check l2 loss pattern
        """
        list([flags, args]).clear()  # Use it once to avoid static checks
        input_len = 1
        output_len = 1
        output_dim_cnt = 1
        output_dim_val = 1

        product = platform.cce_conf.get_product()
        # atomic add does't support fp16 now.
        check_dtype = (product.startswith("1.60") and
                       output_tensors[0].dtype == input_tensors[0].dtype and
                       input_tensors[0].dtype == "float32")
        check_shape = (len(output_tensors) == output_len and
                       len(output_tensors[0].shape) == output_dim_cnt and
                       int(output_tensors[0].shape[0].value) == output_dim_val and
                       len(input_tensors) == input_len)
        if check_dtype and check_shape:
            return _check_pattern_matched(dfs_tensor_list, l2_loss_pattern_list)

        return False


class OpSubPatternRules:  # pylint: disable=R0903
    """Rules for matching subpatterns"""
    @staticmethod
    def simple_pattern_rule(flags: dict, pattern, mapping, *args) -> bool:
        """Simple pattern rule"""
        list(args).clear()  # Use it once to avoid static checks
        result = False
        if not isinstance(mapping[pattern], tuple):
            if flags[mapping[pattern].value]["count"] > 0:
                result = True
        else:
            for flag in mapping[pattern]:
                if flags[flag.value]["count"] > 0:
                    result = True
                    break
        return result

    @staticmethod
    def reduce_atomic_sub_pattern_rule(flags: dict, input_tensors: list,
                                       output_tensors: list, *args) -> bool:
        """
        check reduce atomic pattern
        """
        list([input_tensors, args]).clear()  # Use it once to avoid static checks
        flags.copy()  # Use it once to avoid static checks
        result = False
        # Only cloud has atomic function
        product = platform.cce_conf.get_product()
        if product.startswith("1.60"):
            # Use atomic only when there is more than one reduce output tensors
            for output_tensor in output_tensors:
                if OpFlags.reduce_flag.value in output_tensor.op.tag:
                    result = True
                    break
        return result

    @staticmethod
    def reduce_5hdc_sub_pattern_rule(flags: dict, input_tensors: list, output_tensors: list,
                                     dfs_tensor_list: list, *args) -> bool:
        """
        check 5hd reduce including c axis pattern except reduce_mean
        """
        list([input_tensors, output_tensors,
              dfs_tensor_list, args]).clear()  # Use it once to avoid static checks
        # Is Reduce + elewise_single pattern, Elewise_single == 1
        reduce_flag_num = flags[OpFlags.reduce_flag.value]["count"]
        elewise_single_flag_num = flags[OpFlags.elewise_single_flag.value]["count"]
        expected_flag_num = reduce_flag_num + \
            elewise_single_flag_num + \
            flags[None]["count"]
        if flags["total"]["count"] - expected_flag_num == 0 and \
                reduce_flag_num == 1:
            # Is one in one out, 1 elewise or no elewise
            if len(input_tensors) == 1 and len(output_tensors) == 1 and \
                    elewise_single_flag_num in (0, 1):
                # Shape length is 5 and last axis length is 16
                input_shape = list(map(int, input_tensors[0].shape))
                input_len = len(input_shape)
                if input_shape[-1] == 16 and \
                        input_len == 5 and input_tensors[0].dtype == "float16":
                    # reduce axes
                    reduce_tensor = flags[OpFlags.reduce_flag.value]["tensors"][0]
                    reduce_tensor_axes = get_reduce_axes(reduce_tensor)
                    all_axes = get_all_axes(reduce_tensor)
                    c0_axis = all_axes[-1]
                    c1_axis = all_axes[1]
                    if c0_axis in reduce_tensor_axes and \
                            (input_shape[1] == 1 or c1_axis in reduce_tensor_axes):
                        return True
        return False


class OpSpecialRules:
    """Rules for matching special types"""
    @staticmethod
    def relu_grad_v2_operation_rule(flags: dict, input_tensors: list, output_tensors: list,
                                    dfs_tensor_list: list, *args):
        """
        check relu_grad_v2 operation
        """
        list([flags, args]).clear()  # Use it once to avoid static checks
        out_len = 1
        in_len = 2
        relu_grad_v2_tag_list = [
            # fp16
            [
                "emit_insn_elewise_multiple_sel|bit",
                "placeholder",
                "placeholder",
            ],

            # fp32
            [
                "emit_insn_elewise_multiple_sel|bit",
                "placeholder",
                "elewise_single_cast",
                "placeholder",
            ],
        ]

        if len(output_tensors) == out_len and len(input_tensors) == in_len:
            is_match = False
            for tag_list in relu_grad_v2_tag_list:
                if _check_pattern_matched(dfs_tensor_list, tag_list):
                    is_match = True
                    break

            return is_match

        return False

    @staticmethod
    def relu_v2_operation_rule(flags: dict, input_tensors: list, output_tensors: list,
                               dfs_tensor_list: list, *args):
        """
        check relu_grad_v2 operation
        """
        list([flags, args]).clear()  # Use it once to avoid static checks
        out_len = 2
        in_len = 1
        if len(output_tensors) == out_len and len(input_tensors) == in_len:
            is_match = False
            for tag_list in relu_v2_tag_list:
                if _check_pattern_matched(dfs_tensor_list, tag_list):
                    is_match = True
                    break

            return is_match
        return False

    @staticmethod
    def normalize_scale_operation_rule(flags: dict, input_tensors: list,
                                       output_tensors: list,
                                       dfs_tensor_list: list, *args):
        """
        check normalize scale operation
        """
        list([flags, args]).clear()
        out_len = 1
        in_len = 3

        if get_soc_spec("SOC_VERSION") != "Ascend310":
            return False

        normalize_scale_tag_list = [
            # mini fp16
            [
                "elewise_single_cast|not_auto_cast",
                "elewise_binary_mul",
                "elewise_binary_mul",
                "elewise_single_cast|not_auto_cast",
                "placeholder",
                "broadcast_for_tensor",
                "elewise_single_cast|not_auto_cast",
                "placeholder",
                "broadcast_for_tensor",
                "elewise_binary_mul",
                "elewise_single_VS_add",
                "elewise_single_VS_mul",
                "elewise_binary_mul",
                "elewise_single_VS_mul",
                "elewise_binary_add",
                "elewise_binary_mul",
                "elewise_single_VS_add",
                "placeholder",
                "elewise_binary_mul",
                "elewise_single_VS_add",
                "elewise_single_VS_mul",
                "elewise_binary_mul",
                "elewise_binary_mul",
                "elewise_single_VS_add",
                "elewise_single_VS_mul",
                "elewise_binary_mul",
                "elewise_single_rsqrt",
                "elewise_single_rec",
                "elewise_single_rec",
                "elewise_single_rec",
            ],
        ]

        if len(output_tensors) == out_len and len(input_tensors) == in_len:
            is_match = False
            for tag_list in normalize_scale_tag_list:
                if _check_pattern_matched(dfs_tensor_list, tag_list):
                    is_match = True
                    break

            return is_match
        return False

    @staticmethod
    def only_broadcast_operation_rule(flags: dict, input_tensors: list, output_tensors: list,
                                      dfs_tensor_list: list, *args):
        """
        check the compute only have broadcast operation
        """
        list([flags, args]).clear()  # Use it once to avoid static checks
        out_len = 1
        in_len = 1
        only_broadcast_tag_list = [
            [
                "broadcast_for_tensor",
                "placeholder",
            ],
        ]
        if len(output_tensors) == out_len and len(input_tensors) == in_len:
            is_match = False
            for tag_list in only_broadcast_tag_list:
                if _check_pattern_matched(dfs_tensor_list, tag_list):
                    is_match = True
                    break

            return is_match
        return False

    @staticmethod
    def mvn_operation_rule(flags: dict, input_tensors: list, output_tensors: list,
                           dfs_tensor_list: list, *args):
        """
        check mvn operation
        """
        list([flags, args]).clear()  # Use it once to avoid static checks
        out_len = 1
        in_len = 1

        mvn_tag_list = [
            # fp16
            [
                "elewise_single_cast|not_auto_cast",
                "elewise_binary_sub",
                "elewise_single_cast|not_auto_cast",
                "placeholder",
                "broadcast_for_tensor",
                "elewise_single_VS_mul",
                "reduce_sum",
            ],
            [
                "elewise_single_cast|not_auto_cast",
                "elewise_binary_mul",
                "elewise_binary_sub",
                "elewise_single_cast|not_auto_cast",
                "placeholder",
                "broadcast_for_tensor",
                "elewise_single_VS_mul",
                "reduce_sum",
                "elewise_binary_mul",
                "elewise_single_VS_add",
                "elewise_single_VS_mul",
                "elewise_binary_mul",
                "elewise_single_VS_add",
                "elewise_single_VS_mul",
                "elewise_binary_add",
                "elewise_binary_mul",
                "broadcast_for_tensor",
                "elewise_single_VS_mul",
                "reduce_sum",
                "elewise_binary_mul",
                "elewise_binary_mul",
                "elewise_single_VS_add",
                "elewise_single_VS_mul",
                "elewise_binary_mul",
                "elewise_single_VS_mul",
                "elewise_binary_add",
                "elewise_binary_mul",
                "elewise_binary_mul",
                "elewise_single_VS_add",
                "elewise_single_VS_mul",
                "elewise_binary_mul",
                "elewise_single_VS_mul",
                "elewise_binary_add",
                "elewise_binary_mul",
                "elewise_binary_mul",
                "elewise_single_VS_add",
                "elewise_single_VS_mul",
                "elewise_binary_mul",
                "elewise_binary_mul",
                "elewise_single_VS_add",
                "elewise_single_VS_mul",
                "elewise_binary_mul",
                "elewise_single_rsqrt",
                "elewise_single_rec",
                "elewise_single_rec",
                "elewise_single_rec",
                "elewise_single_rec",
                "elewise_single_rec",
            ],

            # fp32
            [
                "elewise_binary_sub",
                "placeholder",
                "broadcast_for_tensor",
                "elewise_single_VS_mul",
                "reduce_sum",
            ],
            [
                "elewise_binary_mul",
                "elewise_binary_sub",
                "placeholder",
                "broadcast_for_tensor",
                "elewise_single_VS_mul",
                "reduce_sum",
                "elewise_binary_mul",
                "elewise_single_VS_add",
                "elewise_single_VS_mul",
                "elewise_binary_mul",
                "elewise_single_VS_add",
                "elewise_single_VS_mul",
                "elewise_binary_add",
                "elewise_binary_mul",
                "broadcast_for_tensor",
                "elewise_single_VS_mul",
                "reduce_sum",
                "elewise_binary_mul",
                "elewise_binary_mul",
                "elewise_single_VS_add",
                "elewise_single_VS_mul",
                "elewise_binary_mul",
                "elewise_single_VS_mul",
                "elewise_binary_add",
                "elewise_binary_mul",
                "elewise_binary_mul",
                "elewise_single_VS_add",
                "elewise_single_VS_mul",
                "elewise_binary_mul",
                "elewise_single_VS_mul",
                "elewise_binary_add",
                "elewise_binary_mul",
                "elewise_binary_mul",
                "elewise_single_VS_add",
                "elewise_single_VS_mul",
                "elewise_binary_mul",
                "elewise_binary_mul",
                "elewise_single_VS_add",
                "elewise_single_VS_mul",
                "elewise_binary_mul",
                "elewise_single_rsqrt",
                "elewise_single_rec",
                "elewise_single_rec",
                "elewise_single_rec",
                "elewise_single_rec",
                "elewise_single_rec",
            ],
        ]

        if len(output_tensors) == out_len and len(input_tensors) == in_len:
            is_match = False
            for tag_list in mvn_tag_list:
                if _check_pattern_matched(dfs_tensor_list, tag_list):
                    is_match = True
                    break

            return is_match

        return False

    @staticmethod
    def bn_ext2_operation_rule(flags: dict, input_tensors: list, output_tensors: list,
                               dfs_tensor_list: list, *args):
        """
        check bn_ext2 operation
        """
        list([flags, args]).clear()  # Use it once to avoid static checks
        out_len = 5
        in_len = 3

        bn_ext2_tag_list = [
            # fp32
            [
                "elewise_binary_add",
                "elewise_binary_mul",
                "placeholder",
                "elewise_binary_mul",
                "elewise_binary_sub",
                "placeholder",
                "elewise_single_VS_mul",
                "reduce_sum",
                "elewise_binary_mul",
                "elewise_single_VS_add",
                "elewise_single_VS_mul",
                "elewise_binary_mul",
                "elewise_binary_mul",
                "elewise_single_VS_add",
                "elewise_single_VS_mul",
                "elewise_binary_mul",
                "elewise_single_rsqrt",
                "elewise_single_VS_add",
                "elewise_single_VS_mul",
                "reduce_sum",
                "elewise_binary_mul",
                "elewise_binary_sub",
                "elewise_single_rec",
                "elewise_single_rec",
                "placeholder",
                "reduce_sum",
                "elewise_single_VS_mul",
                "reduce_sum",
                "elewise_single_VS_mul",
                "reduce_sum",
                "elewise_single_VS_mul",
                "reduce_sum",
                "elewise_single_VS_mul",
            ],
        ]

        if len(output_tensors) == out_len and len(input_tensors) == in_len:
            is_match = False
            for tag_list in bn_ext2_tag_list:
                if _check_pattern_matched(dfs_tensor_list, tag_list):
                    is_match = True
                    break

            return is_match

        return False

    @staticmethod
    def reduce_mean_2d_aligned_mid_reduce_no_cast_pattern_rule(flags: dict,
                                                               input_tensors: list,
                                                               output_tensors: list,
                                                               dfs_tensor_list: list,
                                                               *args):
        """
        check reduce_mean_d 2D aligned middle reduce no cast pattern
        """
        list([flags, input_tensors, args]).clear()  # Use it once to avoid static checks
        out_len = 1
        input_data_len_2d = 2
        input_data_len_3d = 3
        input_shape = list(map(int, input_tensors[0].shape))
        input_dtype = str(input_tensors[0].dtype)

        reduce_mean_requirement_fulfilled = OpSpecialRules.check_reduce_mean_requirements(
            input_data_len_2d, input_data_len_3d, input_dtype, input_shape, out_len, output_tensors)

        if reduce_mean_requirement_fulfilled:
            for tag_list in reduce_mean_2d_aligned_mid_reduce_no_cast_list:
                if _check_pattern_matched(dfs_tensor_list, tag_list):
                    if len(dfs_tensor_list) != len(tag_list):
                        continue
                    reduce_tensor = flags[OpFlags.reduce_flag.value]["tensors"][0]
                    reduce_tensor_body = reduce_tensor.op.body
                    reduce_tensor_axes = reduce_tensor_body[0].axis
                    reduce_axes = []
                    for reduce_axis in reduce_tensor_axes:
                        reduce_axes.append(reduce_axis.var.name)
                    if len(reduce_axes) == 1 and \
                            reduce_axes[0] == str(reduce_tensor_body[0].source[0].args[-2]) and \
                            (get_soc_spec("SOC_VERSION") not in ["Ascend910"]):
                        return True
                    break
        return False

    @staticmethod
    def check_reduce_mean_requirements(input_data_len_2d, input_data_len_3d,
                                       input_dtype, input_shape, out_len,
                                       output_tensors):
        """Extracted to avoid static checks"""
        reduce_mean_requirement_fulfilled = len(output_tensors) == out_len and \
                                            len(input_shape) in [input_data_len_2d,
                                                                 input_data_len_3d] and \
                                            int(input_shape[-1] % 16) == 0 and \
                                            int(input_shape[-1]) < 48 and \
                                            int(input_shape[-2]) > 8 and \
                                            (len(input_shape) != 3 or input_shape[-3] < 32) and \
                                            input_dtype in ["float16", "float32"]
        return reduce_mean_requirement_fulfilled
