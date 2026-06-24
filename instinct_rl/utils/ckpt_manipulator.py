"""
# A python module that manipulates torch checkpoint file in a hacky way.
Each function should be used with caution and should be used only when thoughtfully considered.
---
Args:
    source_state_dict: the state_dict loaded using torch.load
    algo_state_dict: the algorithm state_dict summarized from algorithm as an example
---
Returns:
    new_state_dict: the state_dict that has been manipulated or directly saved as a checkpoint file.
"""

from collections import OrderedDict
from typing import Literal

import regex as re
import torch


def replace_encoder0(source_state_dict, algo_state_dict):
    print("\033[1;36m Replacing encoder.0 weights with untrained weights and avoid critic_encoder.0 \033[0m")
    new_model_state_dict = OrderedDict()
    for key in algo_state_dict["model_state_dict"].keys():
        if "critic_encoders.0" in key:
            new_model_state_dict[key] = source_state_dict["model_state_dict"][key]
        elif "encoders.0" in key:
            print(
                "key:", key, "shape:", algo_state_dict["model_state_dict"][key].shape, "using untrained module weights."
            )
            new_model_state_dict[key] = algo_state_dict["model_state_dict"][key]
        else:
            new_model_state_dict[key] = source_state_dict["model_state_dict"][key]
    new_state_dict = dict(
        model_state_dict=new_model_state_dict,
        # No optimizer_state_dict
        iter=source_state_dict["iter"],
        infos=source_state_dict["infos"],
    )
    return new_state_dict


def append_GRU_weights(source_state_dict, algo_state_dict):
    print("\033[1;36m Appending GRU weights to fit the new model \033[0m")
    print("\033[1;36m Operating on both actor and critic \033[0m")
    new_model_state_dict = OrderedDict()
    for key in algo_state_dict["model_state_dict"].keys():
        if ("memory_a" in key or "memory_c" in key) and "rnn" in key and "weight_ih" in key:
            print(
                "key:",
                key,
                "shape:",
                source_state_dict["model_state_dict"][key].shape,
                "is updated to shape:",
                algo_state_dict["model_state_dict"][key].shape,
            )
            new_model_state_dict[key] = algo_state_dict["model_state_dict"][key]
            new_model_state_dict[key][:, : source_state_dict["model_state_dict"][key].shape[1]] = source_state_dict[
                "model_state_dict"
            ][key]
            new_model_state_dict[key][:, source_state_dict["model_state_dict"][key].shape[1] :] /= 10
        else:
            new_model_state_dict[key] = source_state_dict["model_state_dict"][key]
    new_state_dict = dict(
        model_state_dict=new_model_state_dict,
        # No optimizer_state_dict
        iter=source_state_dict["iter"],
        infos=source_state_dict["infos"],
    )
    return new_state_dict


def append_GRU_weights_newStd(source_state_dict, algo_state_dict):
    return_ = append_GRU_weights(source_state_dict, algo_state_dict)
    print(
        "\033[1;36m Setting the std of the new actor to {} \033[0m".format(
            algo_state_dict["model_state_dict"]["std"].mean().cpu().item()
        )
    )
    return_["model_state_dict"]["std"][:] = algo_state_dict["model_state_dict"]["std"][:]
    return return_


def reinitialize_actor_critic_backbone(source_state_dict, algo_state_dict):
    print("\033[1;36m Reinitializing the actor and critic backbone \033[0m")
    new_model_state_dict = OrderedDict()
    for key in algo_state_dict["model_state_dict"].keys():
        if (
            "actor." in key
            or "critic." in key
            or "critics." in key
            or "memory_a" in key
            or "memory_c" in key
            or "std" in key
        ):
            if not key in source_state_dict["model_state_dict"]:
                print(
                    "key:",
                    key,
                    "shape:",
                    algo_state_dict["model_state_dict"][key].shape,
                    "using untrained module weights.",
                )
            else:
                print(
                    "key:",
                    key,
                    "shape:",
                    source_state_dict["model_state_dict"][key].shape,
                    "is updated to shape:",
                    algo_state_dict["model_state_dict"][key].shape,
                )
            new_model_state_dict[key] = algo_state_dict["model_state_dict"][key]
        else:
            new_model_state_dict[key] = source_state_dict["model_state_dict"][key]
    new_state_dict = dict(
        model_state_dict=new_model_state_dict,
        # No optimizer_state_dict
        iter=source_state_dict["iter"],
        infos=source_state_dict["infos"],
    )
    return new_state_dict


def ignore_missing_key(source_state_dict, algo_state_dict):
    """Ignore the missing critic mlp weights and use the initialized ones."""
    print("\033[1;36m Ignoring missing key and using the initialized weights \033[0m")
    new_model_state_dict = OrderedDict()
    missing_keys = []
    for key in algo_state_dict["model_state_dict"].keys():
        if key in source_state_dict["model_state_dict"]:
            new_model_state_dict[key] = source_state_dict["model_state_dict"][key]
        else:
            new_model_state_dict[key] = algo_state_dict["model_state_dict"][key]
            missing_keys.append(key)
    new_state_dict = dict(
        model_state_dict=new_model_state_dict,
        # No optimizer_state_dict
        iter=source_state_dict["iter"],
        infos=source_state_dict["infos"],
    )
    print("\033[1;36m Missing keys: \033[0m", missing_keys)
    return new_state_dict


def merge_matching_model_state(source_state_dict, algo_state_dict, drop_optimizer: bool = False):
    """Merge same-shape model weights and keep initialized values for incompatible keys."""
    source_model = source_state_dict["model_state_dict"]
    target_model = algo_state_dict["model_state_dict"]
    all_target_keys_match = all(
        key in source_model and source_model[key].shape == value.shape for key, value in target_model.items()
    )
    if (
        all_target_keys_match
        and "optimizer_state_dict" in source_state_dict
        and "iter" in source_state_dict
        and not drop_optimizer
    ):
        print("\033[1;36m Checkpoint is already compatible; loading it unchanged. \033[0m")
        return source_state_dict

    print("\033[1;36m Merging same-shape model weights and keeping initialized incompatible keys. \033[0m")
    new_model_state_dict = OrderedDict()
    copied_keys = []
    skipped_keys = []
    for key, value in target_model.items():
        if key in source_model and source_model[key].shape == value.shape:
            new_model_state_dict[key] = source_model[key]
            copied_keys.append(key)
        else:
            new_model_state_dict[key] = value
            skipped_keys.append(key)
    new_state_dict = dict(model_state_dict=new_model_state_dict)
    for key in ("discriminator", "discriminator_optimizer"):
        if key == "discriminator_optimizer" and drop_optimizer:
            continue
        if key not in source_state_dict or key not in algo_state_dict:
            continue
        source_substate = source_state_dict[key]
        target_substate = algo_state_dict[key]
        if isinstance(source_substate, dict) and isinstance(target_substate, dict):
            matching = True
            for subkey, value in target_substate.items():
                if subkey not in source_substate:
                    matching = False
                    break
                source_value = source_substate[subkey]
                if torch.is_tensor(value) and torch.is_tensor(source_value) and value.shape != source_value.shape:
                    matching = False
                    break
            if matching:
                new_state_dict[key] = source_substate
    new_state_dict["iter"] = source_state_dict.get("iter", source_state_dict.get("offline_iteration", 0))
    new_state_dict["infos"] = source_state_dict.get("infos", {})
    print(f"\033[1;36m Copied {len(copied_keys)} keys; kept initialized {len(skipped_keys)} keys. \033[0m")
    if skipped_keys:
        print("\033[1;36m Incompatible/missing keys: \033[0m", skipped_keys)
    return new_state_dict


def merge_matching_model_state_append_critic_obs(
    source_state_dict,
    algo_state_dict,
    drop_optimizer: bool = False,
    encoded_tail_dim: int = 128,
    critic_first_layer_regex: str = r"critic\.(gate\.0|experts\.\d+\.0)\.weight",
):
    """Merge a checkpoint while preserving critic first-layer weights when critic obs grows.

    This is useful when extra train-only critic observations are inserted before the encoded
    depth tail.  For widened critic first layers, copy the old proprio prefix and the old
    encoded tail into the matching new locations, and zero the newly inserted columns.
    """

    source_model = source_state_dict["model_state_dict"]
    target_model = algo_state_dict["model_state_dict"]
    new_model_state_dict = OrderedDict()
    copied_keys = []
    expanded_keys = []
    skipped_keys = []

    for key, target_value in target_model.items():
        source_value = source_model.get(key)
        if source_value is not None and source_value.shape == target_value.shape:
            new_model_state_dict[key] = source_value
            copied_keys.append(key)
            continue

        can_expand_critic = (
            source_value is not None
            and torch.is_tensor(source_value)
            and torch.is_tensor(target_value)
            and source_value.ndim == 2
            and target_value.ndim == 2
            and source_value.shape[0] == target_value.shape[0]
            and source_value.shape[1] < target_value.shape[1]
            and encoded_tail_dim > 0
            and source_value.shape[1] > encoded_tail_dim
            and target_value.shape[1] > encoded_tail_dim
            and re.match(critic_first_layer_regex, key)
        )
        if can_expand_critic:
            old_prefix = source_value.shape[1] - encoded_tail_dim
            new_prefix = target_value.shape[1] - encoded_tail_dim
            value = target_value.clone()
            value[:, :old_prefix] = source_value[:, :old_prefix]
            value[:, old_prefix:new_prefix] = 0.0
            value[:, new_prefix:] = source_value[:, old_prefix:]
            new_model_state_dict[key] = value
            expanded_keys.append(key)
            continue

        new_model_state_dict[key] = target_value
        skipped_keys.append(key)

    new_state_dict = dict(model_state_dict=new_model_state_dict)
    for key in ("discriminator", "discriminator_optimizer"):
        if key == "discriminator_optimizer" and drop_optimizer:
            continue
        if key not in source_state_dict or key not in algo_state_dict:
            continue
        source_substate = source_state_dict[key]
        target_substate = algo_state_dict[key]
        if isinstance(source_substate, dict) and isinstance(target_substate, dict):
            matching = True
            for subkey, value in target_substate.items():
                if subkey not in source_substate:
                    matching = False
                    break
                source_subvalue = source_substate[subkey]
                if torch.is_tensor(value) and torch.is_tensor(source_subvalue) and value.shape != source_subvalue.shape:
                    matching = False
                    break
            if matching:
                new_state_dict[key] = source_substate

    new_state_dict["iter"] = source_state_dict.get("iter", source_state_dict.get("offline_iteration", 0))
    new_state_dict["infos"] = source_state_dict.get("infos", {})
    print(
        f"\033[1;36m Copied {len(copied_keys)} keys; expanded critic obs for {len(expanded_keys)} keys; "
        f"kept initialized {len(skipped_keys)} keys. \033[0m"
    )
    if expanded_keys:
        print("\033[1;36m Expanded critic first-layer keys: \033[0m", expanded_keys)
    if skipped_keys:
        print("\033[1;36m Incompatible/missing keys: \033[0m", skipped_keys)
    return new_state_dict


def fit_smaller_weight(
    source_state_dict: dict,
    algo_state_dict: dict,
    weight_name_regex: str = ".*",
    weight_match_mode: Literal["start", "end"] = "start",
):
    """To fix the weight matrix in algo_state_dict which is smaller than the one in source_state_dict,
    we will copy the part of the weight matrix from source_state_dict to algo_state_dict.
    ## Args:
        weight_name_regex: str
            The regex to match the weight name in algo_state_dict.
        weight_match_mode: Literal["start", "end"]
            If "start", weight_algo = weight_source[:weight_algo.shape[0], :weight_algo.shape[1]]
            If "end", weight_algo = weight_source[-weight_algo.shape[0]:, -weight_algo.shape[1]:]
    """
    print("\033[1;36m Fitting smaller weight matrix, matching \033[0m")
    new_model_state_dict = OrderedDict()
    for key in algo_state_dict["model_state_dict"].keys():
        if re.match(weight_name_regex, key):
            weight_algo = algo_state_dict["model_state_dict"][key]
            weight_source = source_state_dict["model_state_dict"][key]
            if weight_match_mode == "start":
                new_model_state_dict[key] = weight_source[: weight_algo.shape[0], : weight_algo.shape[1]]
            elif weight_match_mode == "end":
                new_model_state_dict[key] = weight_source[-weight_algo.shape[0] :, -weight_algo.shape[1] :]
            else:
                raise ValueError(f"Invalid weight_match_mode: {weight_match_mode}. Must be one of ['start', 'end'].")
        else:
            new_model_state_dict[key] = source_state_dict["model_state_dict"][key]
    new_state_dict = dict(
        model_state_dict=new_model_state_dict,
    )
    for k in source_state_dict.keys():
        if k not in new_state_dict and not k.startswith("optimizer_state_dict"):
            new_state_dict[k] = source_state_dict[k]
    return new_state_dict


def newStd(
    source_state_dict: dict,
    algo_state_dict: dict,
):
    """Replicate everything except for policy std"""
    print(
        "\033[1;36m Setting the std of the new actor to {} \033[0m".format(
            algo_state_dict["model_state_dict"]["std"].mean().cpu().item()
        )
    )
    new_state_dict = OrderedDict()
    for state_dict_key in source_state_dict.keys():
        if state_dict_key == "model_state_dict":
            new_state_dict[state_dict_key] = OrderedDict()
            for model_state_dict_key in source_state_dict[state_dict_key].keys():
                if "std" == model_state_dict_key:
                    new_state_dict[state_dict_key][model_state_dict_key] = algo_state_dict["model_state_dict"][
                        model_state_dict_key
                    ]
                else:
                    new_state_dict[state_dict_key][model_state_dict_key] = source_state_dict[state_dict_key][
                        model_state_dict_key
                    ]
        else:
            new_state_dict[state_dict_key] = source_state_dict[state_dict_key]
    return new_state_dict
