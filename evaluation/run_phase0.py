#!/usr/bin/env python3
"""Phase 0 diagnostic: does NaVILA respond to time-budget language appended to
R2R val_unseen instructions, zero-shot?

Runs 20 fixed episodes under three instruction conditions (original / +"You have
15 seconds." / +"You have 3 minutes.") and logs per-episode action outputs and
metrics. The ONLY thing that changes between conditions is the instruction text
fed to the model; the model, simulator config, and decoding params are untouched.

The rollout loop mirrors NaVILATrainer._eval_checkpoint exactly (same prompt,
same regex action parsing, same 25cm/15deg discretization) so Phase 0 behavior is
directly comparable to the standard eval.
"""

import argparse
import gzip
import json
import os
import random
import re
import time

import numpy as np
import torch
import tqdm
from habitat import logger
from habitat.utils.visualizations.utils import append_text_to_image
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    get_active_obs_transforms,
)
from habitat_baselines.utils.common import batch_obs
from PIL import Image

# Importing these registers the dataset ("VLN-CE-v1"), envs, measures, and trainers.
import habitat_extensions  # noqa: F401
import vlnce_baselines  # noqa: F401
from habitat_extensions.utils import generate_video, observations_to_image
from vlnce_baselines.common.env_utils import construct_envs_auto_reset_false
from vlnce_baselines.common.utils import extract_instruction_tokens
from vlnce_baselines.config.default import get_config
from vlnce_baselines.navila_trainer import sample_and_pad_images

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model

ACTION_NAME = {0: "stop", 1: "forward", 2: "turn_left", 3: "turn_right"}

CONDITIONS = [
    ("original", ""),
    ("short_budget", "You have 15 seconds."),
    ("long_budget", "You have 3 minutes."),
]


def load_succeeded_ids(success_from):
    """Return the set of episode_ids with success==1.0 across the given result files.

    success_from may be a directory containing per-chunk JSONs (val_unseen_*-*.json)
    or a single JSON file mapping episode_id -> metrics dict.
    """
    paths = []
    if os.path.isdir(success_from):
        for name in sorted(os.listdir(success_from)):
            if name.endswith(".json"):
                paths.append(os.path.join(success_from, name))
    else:
        paths.append(success_from)

    succeeded = set()
    for p in paths:
        with open(p) as f:
            results = json.load(f)
        for ep_id, m in results.items():
            if m.get("success") == 1.0:
                succeeded.add(str(ep_id))
    return succeeded


def sample_episode_ids(data_path, split, num_episodes, seed, allowed_ids=None):
    """Deterministically pick `num_episodes` episode_ids from a split.

    If `allowed_ids` is provided, sampling is restricted to that set (e.g. only
    episodes where the baseline eval succeeded). Returns (sampled_ids,
    original_instructions) where original_instructions maps episode_id (str)
    -> unmodified instruction_text.
    """
    path = data_path.format(split=split)
    with gzip.open(path, "rt") as f:
        deserialized = json.load(f)
    episodes = deserialized["episodes"]

    # episode_ids are cast to str everywhere downstream (see habitat_extensions.task)
    all_ids = [str(ep["episode_id"]) for ep in episodes]
    if allowed_ids is not None:
        pool = [ep_id for ep_id in all_ids if ep_id in allowed_ids]
        if len(pool) < num_episodes:
            raise ValueError(
                f"Only {len(pool)} episodes match the success filter; need {num_episodes}."
            )
    else:
        pool = all_ids

    rng = random.Random(seed)
    sampled = sorted(rng.sample(pool, num_episodes), key=lambda x: int(x))

    sampled_set = set(sampled)
    original_instructions = {
        str(ep["episode_id"]): ep["instruction"]["instruction_text"]
        for ep in episodes
        if str(ep["episode_id"]) in sampled_set
    }
    return sampled, original_instructions


def apply_budget(instruction, budget_text):
    """Append the budget sentence to an instruction. No-op for the original condition."""
    if not budget_text:
        return instruction
    return instruction.rstrip() + " " + budget_text


def build_eval_config(base_config, episode_ids, video_dir=None):
    """Clone the base config and restrict it to the sampled episodes.

    If video_dir is set, enable disk video output (one MP4 per episode) and add
    the TOP_DOWN_MAP_VLNCE measure used by habitat_extensions.generate_video.
    """
    config = base_config.clone()
    config.defrost()
    split = config.EVAL.SPLIT
    config.TASK_CONFIG.DATASET.SPLIT = split
    config.TASK_CONFIG.DATASET.ROLES = ["guide"]
    config.TASK_CONFIG.DATASET.LANGUAGES = config.EVAL.LANGUAGES
    config.TASK_CONFIG.TASK.NDTW.SPLIT = split
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    config.TASK_CONFIG.DATASET.NUM_CHUNKS = 1
    config.TASK_CONFIG.DATASET.CHUNK_IDX = 0
    config.TASK_CONFIG.DATASET.EPISODES_ALLOWED = list(episode_ids)
    if video_dir is not None:
        config.VIDEO_OPTION = ["disk"]
        config.VIDEO_DIR = video_dir
        if "TOP_DOWN_MAP_VLNCE" not in config.TASK_CONFIG.TASK.MEASUREMENTS:
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
    else:
        config.VIDEO_OPTION = []
    config.freeze()
    return config


def final_agent_position(envs, index):
    """Read the agent's world position from the (thread-shared) habitat Env.

    ThreadedVectorEnv passes objects by reference, so call_at returns the live Env.
    Must be called before reset_at() advances to the next episode.
    """
    try:
        habitat_env = envs.call_at(index, "habitat_env")
        return habitat_env.sim.get_agent_state().position.tolist()
    except Exception as e:  # noqa: BLE001
        logger.warn(f"could not read final position: {e}")
        return None


def run_condition(model, tokenizer, image_processor, base_config, episode_ids, budget_text, device, video_dir=None, on_episode_done=None):
    """Run all sampled episodes for one instruction condition.

    Returns ep_id -> dict(info, all_action_outputs, action_histogram,
    final_position, modified_instruction, trajectory, start_position).
    If video_dir is provided, one MP4 per episode is written there with the
    modified instruction overlaid on each frame.
    on_episode_done(ep_id, condition_result) is called each time an episode
    finishes — used by main() to flush results to disk incrementally so a
    crash/reboot doesn't lose all data.
    """
    config = build_eval_config(base_config, episode_ids, video_dir=video_dir)
    num_video_frames = model.config.num_video_frames

    if video_dir is not None:
        os.makedirs(video_dir, exist_ok=True)

    envs = construct_envs_auto_reset_false(config, get_env_class(config.ENV_NAME))
    obs_transforms = get_active_obs_transforms(config)

    observations = envs.reset()
    observations = extract_instruction_tokens(
        observations, config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
    )
    batch = batch_obs(observations, device)
    batch = apply_obs_transforms_batch(batch, obs_transforms)

    assert envs.num_envs == 1, "Phase 0 assumes a single environment."
    num_eps = sum(envs.number_of_episodes)

    results = {}
    past_rgbs = [[]]
    queue_actions = []

    def fresh_hist():
        return {"forward": 0, "turn_left": 0, "turn_right": 0, "stop": 0}

    cur_outputs = []
    cur_hist = fresh_hist()
    cur_modified_instruction = None
    cur_trajectory = []  # list of {"action": str, "position": [x,y,z]} per env.step
    cur_start_position = None
    rgb_frames = [[]]  # one frame buffer per env, used only when video_dir is set

    pbar = tqdm.tqdm(total=num_eps, desc=f"[{budget_text or 'original'}]")

    while envs.num_envs > 0 and len(results) < num_eps:
        current_episodes = envs.current_episodes()

        if cur_start_position is None:
            try:
                henv = envs.call_at(0, "habitat_env")
                cur_start_position = henv.sim.get_agent_state().position.tolist()
            except Exception:  # noqa: BLE001
                cur_start_position = None

        if len(queue_actions) > 0:
            act = queue_actions[0]
            step_result = envs.step([act])
            queue_actions.pop(0)
            cur_hist[ACTION_NAME.get(act, "stop")] += 1
            last_action_int = act
        else:
            with torch.no_grad():
                curr_rgb = Image.fromarray(np.uint8(batch[0]["rgb"].cpu().numpy())).convert("RGB")
                past_and_current_rgbs = past_rgbs[0] + [curr_rgb]
                past_and_current_rgbs = sample_and_pad_images(
                    past_and_current_rgbs, num_frames=num_video_frames
                )

                instruction = current_episodes[0].instruction.instruction_text
                instruction = apply_budget(instruction, budget_text)
                cur_modified_instruction = instruction

                interleaved_images = "<image>\n" * (len(past_and_current_rgbs) - 1)
                question = (
                    f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
                    f'of historical observations {interleaved_images}, and current observation <image>\n. Your assigned task is: "{instruction}" '
                    f"Analyze this series of images to decide your next action, which could be turning left or right by a specific "
                    f"degree, moving forward a certain distance, or stop if the task is completed."
                )

                conv = conv_templates["llama_3"].copy()
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                images_tensor = process_images(
                    past_and_current_rgbs, image_processor, model.config
                ).to(model.device, dtype=torch.float16)
                input_ids = (
                    tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                    .unsqueeze(0)
                    .cuda()
                )

                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=images_tensor.half().cuda(),
                        do_sample=False,
                        temperature=0.0,
                        max_new_tokens=32,
                        use_cache=True,
                        stopping_criteria=[stopping_criteria],
                        pad_token_id=tokenizer.eos_token_id,
                    )

                outputs_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                if outputs_text.endswith(stop_str):
                    outputs_text = outputs_text[: -len(stop_str)]
                outputs_text = outputs_text.strip()

                cur_outputs.append(outputs_text)

                patterns = {
                    0: re.compile(r"\bstop\b", re.IGNORECASE),
                    1: re.compile(r"\bis move forward\b", re.IGNORECASE),
                    2: re.compile(r"\bis turn left\b", re.IGNORECASE),
                    3: re.compile(r"\bis turn right\b", re.IGNORECASE),
                }

                def map_string_to_action(s):
                    for action, pattern in patterns.items():
                        if pattern.search(s):
                            return action
                    return None

                try:
                    actions = [map_string_to_action(outputs_text)]
                except Exception:  # noqa: BLE001
                    actions = [1]

            if actions[0] == 1:
                try:
                    match = re.search(r"move forward (\d+) cm", outputs_text)
                    distance = int(match.group(1))
                except Exception:  # noqa: BLE001
                    distance = 25
                if (distance % 25) != 0:
                    distance = min([25, 50, 75], key=lambda x: abs(x - distance))
                step_result = envs.step([1])
                cur_hist["forward"] += 1
                last_action_int = 1
                for _ in range(int(distance // 25) - 1):
                    queue_actions.append(1)

            elif actions[0] == 2:
                try:
                    match = re.search(r"turn left (\d+) degree", outputs_text)
                    degree = int(match.group(1))
                except Exception:  # noqa: BLE001
                    degree = 15
                if (degree % 15) != 0:
                    degree = min([15, 30, 45], key=lambda x: abs(x - degree))
                step_result = envs.step([2])
                cur_hist["turn_left"] += 1
                last_action_int = 2
                for _ in range(int(degree // 15) - 1):
                    queue_actions.append(2)

            elif actions[0] == 3:
                try:
                    match = re.search(r"turn right (\d+) degree", outputs_text)
                    degree = int(match.group(1))
                except Exception:  # noqa: BLE001
                    degree = 15
                if (degree % 15) != 0:
                    degree = min([15, 30, 45], key=lambda x: abs(x - degree))
                step_result = envs.step([3])
                cur_hist["turn_right"] += 1
                last_action_int = 3
                for _ in range(int(degree // 15) - 1):
                    queue_actions.append(3)

            else:  # stop (0) or unmatched (None)
                step_result = envs.step([0])
                cur_hist["stop"] += 1
                last_action_int = 0

        observations, _, dones, infos = [list(x) for x in zip(*step_result)]

        # Per-step trajectory: position AFTER taking the action.
        try:
            henv = envs.call_at(0, "habitat_env")
            cur_pos = henv.sim.get_agent_state().position.tolist()
        except Exception:  # noqa: BLE001
            cur_pos = None
        cur_trajectory.append({"action": ACTION_NAME.get(last_action_int, "stop"), "position": cur_pos})

        for i in range(envs.num_envs):
            past_rgbs[i].append(Image.fromarray(batch[0]["rgb"].cpu().numpy()).convert("RGB"))

            if video_dir is not None:
                frame = observations_to_image(observations[i], infos[i])
                # Overlay the modified instruction so each video frame is self-labeling.
                frame = append_text_to_image(
                    frame, cur_modified_instruction or current_episodes[i].instruction.instruction_text
                )
                rgb_frames[i].append(frame)

            if not dones[i]:
                continue

            ep_id = current_episodes[i].episode_id
            final_pos = final_agent_position(envs, i)
            results[ep_id] = {
                "info": infos[i],
                "all_action_outputs": list(cur_outputs),
                "action_histogram": dict(cur_hist),
                "final_position": final_pos,
                "start_position": cur_start_position,
                "modified_instruction": cur_modified_instruction,
                "trajectory": list(cur_trajectory),
            }

            if video_dir is not None:
                generate_video(
                    video_option=["disk"],
                    video_dir=video_dir,
                    images=rgb_frames[i],
                    episode_id=ep_id,
                    checkpoint_idx="0",
                    metrics={"spl": infos[i].get("spl", 0.0)},
                    tb_writer=None,
                )
                # top_down_map_vlnce is a huge nested dict; drop before JSON serialization.
                results[ep_id]["info"].pop("top_down_map_vlnce", None)
                rgb_frames[i] = []

            # Incremental save: notify caller so partial results survive a crash/reboot.
            if on_episode_done is not None:
                try:
                    on_episode_done(ep_id, results[ep_id])
                except Exception as e:  # noqa: BLE001
                    logger.warn(f"on_episode_done failed for {ep_id}: {e}")

            observations[i] = envs.reset_at(i)[0]
            past_rgbs[i] = []
            queue_actions = []
            cur_outputs = []
            cur_hist = fresh_hist()
            cur_modified_instruction = None
            cur_trajectory = []
            cur_start_position = None
            pbar.update()

        observations = extract_instruction_tokens(
            observations, config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        )
        batch = batch_obs(observations, device)
        batch = apply_obs_transforms_batch(batch, obs_transforms)

        envs_to_pause = []
        next_episodes = envs.current_episodes()
        for i in range(envs.num_envs):
            if next_episodes[i].episode_id in results:
                envs_to_pause.append(i)

        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)
            for k, v in batch.items():
                batch[k] = v[state_index]
            past_rgbs = [past_rgbs[i] for i in state_index]
            rgb_frames = [rgb_frames[i] for i in state_index]

    pbar.close()
    envs.close()
    return results


def assemble_record(ep_id, condition_name, original_instruction, condition_result):
    info = condition_result["info"]
    hist = condition_result["action_histogram"]
    num_steps = sum(hist.values())
    return {
        "episode_id": ep_id,
        "condition": condition_name,
        "original_instruction": original_instruction,
        "modified_instruction": condition_result["modified_instruction"],
        "all_action_outputs": condition_result["all_action_outputs"],
        "trajectory_length": info.get("path_length"),
        "num_steps": num_steps,
        "success": bool(info.get("success", 0.0)),
        "spl": info.get("spl"),
        "distance_to_goal": info.get("distance_to_goal"),
        "start_position": condition_result.get("start_position"),
        "final_position": condition_result["final_position"],
        "action_histogram": hist,
        "trajectory": condition_result.get("trajectory", []),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-config",
        default="vlnce_baselines/config/r2r_baselines/navila.yaml",
        help="Base eval config (kept identical to the standard NaVILA eval).",
    )
    parser.add_argument(
        "--model-path",
        default="/home/maitree-tiamat/models/navila-llama3-8b-8f",
        help="Checkpoint dir or HF id (a8cheng/navila-llama3-8b-8f).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", default="phase0_results.json")
    parser.add_argument(
        "--success-from",
        default=None,
        help="Path (file or dir) to a baseline-eval results JSON; restricts "
        "sampling to episodes with success==1.0 there. Recommended: "
        "eval_out/<model>/VLN-CE-v1/val_unseen/  so Phase 0 measures budget "
        "response only on episodes NaVILA can solve under the original instruction.",
    )
    parser.add_argument(
        "--video-dir",
        default="phase0_videos",
        help="Directory to write per-episode MP4s (one subfolder per condition). "
        "Set to empty string to disable video output.",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    config = get_config(args.exp_config, ["EVAL.SPLIT", args.split])
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

    data_path = config.TASK_CONFIG.DATASET.DATA_PATH
    allowed_ids = None
    if args.success_from:
        allowed_ids = load_succeeded_ids(args.success_from)
        logger.info(f"Restricting sample pool to {len(allowed_ids)} success-only episodes from {args.success_from}")
    episode_ids, original_instructions = sample_episode_ids(
        data_path, args.split, args.num_episodes, args.seed, allowed_ids=allowed_ids
    )
    logger.info(f"Sampled {len(episode_ids)} episodes (seed={args.seed}): {episode_ids}")

    model_name = os.path.basename(os.path.normpath(args.model_path))
    logger.info(f"Loading model {args.model_path} ...")
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, model_name)
    model = model.cuda()

    metadata = {
        "seed": args.seed,
        "num_episodes": args.num_episodes,
        "conditions": [name for name, _ in CONDITIONS],
        "short_budget_text": dict(CONDITIONS)["short_budget"],
        "long_budget_text": dict(CONDITIONS)["long_budget"],
        "model": model_name,
        "split": args.split,
        "episode_ids": episode_ids,
        "success_filter_source": args.success_from,
    }

    # Resume support: if the output file already exists, load and skip episodes
    # already done for each condition. Lets us recover from a crash/reboot.
    results = []
    done_keys = set()  # (condition, episode_id) tuples
    if os.path.exists(args.output):
        try:
            prior = json.load(open(args.output))
            results = list(prior.get("results", []))
            done_keys = {(r["condition"], r["episode_id"]) for r in results}
            logger.info(f"Resuming: loaded {len(results)} prior records from {args.output}")
        except Exception as e:  # noqa: BLE001
            logger.warn(f"Could not parse prior {args.output} ({e}); starting fresh.")
            results = []
            done_keys = set()

    def flush_to_disk():
        tmp = args.output + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"metadata": metadata, "results": results}, f, indent=2)
        os.replace(tmp, args.output)

    base_video_dir = args.video_dir if args.video_dir else None
    for condition_name, budget_text in CONDITIONS:
        already_done = {ep_id for (c, ep_id) in done_keys if c == condition_name}
        remaining = [ep for ep in episode_ids if ep not in already_done]
        if not remaining:
            logger.info(f"=== Condition: {condition_name} (already complete, skipping) ===")
            continue
        logger.info(
            f"=== Condition: {condition_name} (suffix={budget_text!r}) — "
            f"{len(remaining)}/{len(episode_ids)} to run ==="
        )
        start = time.time()
        condition_video_dir = (
            os.path.join(base_video_dir, condition_name) if base_video_dir else None
        )

        def on_episode_done(ep_id, cond_result, _cond=condition_name):
            results.append(
                assemble_record(ep_id, _cond, original_instructions[ep_id], cond_result)
            )
            flush_to_disk()

        run_condition(
            model,
            tokenizer,
            image_processor,
            config,
            remaining,  # only run the not-yet-done episodes
            budget_text,
            device,
            video_dir=condition_video_dir,
            on_episode_done=on_episode_done,
        )
        logger.info(
            f"Condition {condition_name} done in {round(time.time() - start)}s "
            f"({len(results)} total records on disk)"
        )

    flush_to_disk()
    logger.info(f"Wrote {len(results)} records to {args.output}")


if __name__ == "__main__":
    main()
