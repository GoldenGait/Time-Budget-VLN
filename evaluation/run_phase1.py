#!/usr/bin/env python3
"""Phase 1 probe: can NaVILA do goal-only search, zero-shot?

For a fixed set of R2R val_unseen episodes (evaluation/phase1_instructions.json),
run two instruction conditions:
  - original   : the unmodified route instruction (control)
  - goal_only  : route stripped, only the goal/landmark retained

under two prompt-scaffold pipelines:
  - phase1_baseline : NaVILA's original route-following scaffold (unchanged)
  - phase1_explore  : an exploration-nudged scaffold (search framing)

So each episode is run 4 times: {baseline, explore} x {original, goal_only}.
The rollout loop, regex action parsing, and 25cm/15deg discretization are byte
-identical to run_phase0.py / the standard NaVILA eval. The ONLY things that vary
are (a) the instruction text and (b) the surrounding scaffold. The action-menu
wording inside each scaffold is kept identical so the action parser is unaffected.
"""

import argparse
import gzip
import json
import os
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

CONDITIONS = ["original", "goal_only"]


# --- Scaffolds: framing x stop-clause factorial ----------------------------
# A scaffold = FRAMING + ACTION_MENU + STOP_CLAUSE. The ACTION_MENU wording is
# byte-identical across all scaffolds, so the regex action parser is unaffected.
# Decomposing framing and stop clause lets us disentangle their effects (the
# original explore pipeline flipped BOTH at once vs baseline).

# Shared, identical across every scaffold (DO NOT change — parser depends on it).
ACTION_MENU = (
    "Analyze this series of images to decide your next action, which could be turning left or right by a specific "
    "degree, moving forward a certain distance, "
)

FRAMINGS = {
    # route-following framing (original NaVILA / phase-0)
    "nav": lambda instruction, imgs: (
        f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
        f'of historical observations {imgs}, and current observation <image>\n. Your assigned task is: "{instruction}" '
    ),
    # search/exploration framing
    "explore": lambda instruction, imgs: (
        f"Imagine you are a robot programmed for search and navigation tasks. You have been given a video "
        f'of historical observations {imgs}, and current observation <image>\n. Your assigned task is: "{instruction}" '
        f"You have not been given step-by-step directions — you must actively explore the environment to locate the target. "
        f"Move through the space to reveal areas you have not seen yet, head toward the kinds of rooms where the target is most "
        f"likely to be, and avoid going back over places you have already visited. "
    ),
}

STOP_CLAUSES = {
    "task_complete": "or stop if the task is completed.",
    "target_visible": "or stop only once the target is clearly visible in front of you.",
}

# pipeline name -> (framing_key, stop_key)
PIPELINE_SPEC = {
    "phase1_baseline": ("nav", "task_complete"),
    "phase1_explore": ("explore", "target_visible"),
    # disentangler: explore framing with the ORIGINAL stop clause
    "phase1_explore_origstop": ("explore", "task_complete"),
    # optional 4th cell to complete the 2x2 (not run by default)
    "phase1_baseline_tgtstop": ("nav", "target_visible"),
}


def make_scaffold(framing_key, stop_key):
    def scaffold(instruction, interleaved_images):
        return FRAMINGS[framing_key](instruction, interleaved_images) + ACTION_MENU + STOP_CLAUSES[stop_key]
    return scaffold


PIPELINES = {name: make_scaffold(*spec) for name, spec in PIPELINE_SPEC.items()}


def _selfcheck_scaffolds():
    """Guarantee the refactor did NOT perturb the two already-run cells."""
    instr, imgs = "TEST", "<image>\n<image>\n"
    orig_baseline = (
        f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
        f'of historical observations {imgs}, and current observation <image>\n. Your assigned task is: "{instr}" '
        f"Analyze this series of images to decide your next action, which could be turning left or right by a specific "
        f"degree, moving forward a certain distance, or stop if the task is completed."
    )
    orig_explore = (
        f"Imagine you are a robot programmed for search and navigation tasks. You have been given a video "
        f'of historical observations {imgs}, and current observation <image>\n. Your assigned task is: "{instr}" '
        f"You have not been given step-by-step directions — you must actively explore the environment to locate the target. "
        f"Move through the space to reveal areas you have not seen yet, head toward the kinds of rooms where the target is most "
        f"likely to be, and avoid going back over places you have already visited. "
        f"Analyze this series of images to decide your next action, which could be turning left or right by a specific "
        f"degree, moving forward a certain distance, or stop only once the target is clearly visible in front of you."
    )
    assert PIPELINES["phase1_baseline"](instr, imgs) == orig_baseline, "baseline scaffold drifted from the run version!"
    assert PIPELINES["phase1_explore"](instr, imgs) == orig_explore, "explore scaffold drifted from the run version!"


_selfcheck_scaffolds()


def load_phase1_instructions(path):
    """Load the phase-1 config. Returns (episode_ids, instr_map) where
    instr_map[condition][episode_id] -> instruction text."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    eps = d["episodes"]
    episode_ids = [str(e["episode_id"]) for e in eps]
    instr_map = {c: {} for c in CONDITIONS}
    for e in eps:
        eid = str(e["episode_id"])
        for c in CONDITIONS:
            if c not in e:
                raise ValueError(f"episode {eid} missing condition '{c}'")
            instr_map[c][eid] = e[c]
    return episode_ids, instr_map


def build_eval_config(base_config, episode_ids, video_dir=None, max_steps=None):
    """Clone the base config and restrict it to the given episodes."""
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
    # Search legitimately needs more steps than route-following; allow override.
    if max_steps is not None:
        config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = int(max_steps)
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
    try:
        habitat_env = envs.call_at(index, "habitat_env")
        return habitat_env.sim.get_agent_state().position.tolist()
    except Exception as e:  # noqa: BLE001
        logger.warn(f"could not read final position: {e}")
        return None


def run_condition(model, tokenizer, image_processor, base_config, episode_ids,
                  scaffold_fn, instruction_override, device, video_dir=None,
                  max_steps=None, tag="", on_episode_done=None):
    """Run all given episodes for one (pipeline, condition) cell.

    scaffold_fn(instruction, interleaved_images) -> prompt string.
    instruction_override: dict episode_id -> instruction text (replaces the
    instruction fed to the model; the goal location used for success is the
    episode's, so this does not affect success measurement).
    """
    config = build_eval_config(base_config, episode_ids, video_dir=video_dir, max_steps=max_steps)
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

    assert envs.num_envs == 1, "Phase 1 assumes a single environment."
    num_eps = sum(envs.number_of_episodes)

    results = {}
    past_rgbs = [[]]
    queue_actions = []

    def fresh_hist():
        return {"forward": 0, "turn_left": 0, "turn_right": 0, "stop": 0}

    cur_outputs = []
    cur_hist = fresh_hist()
    cur_used_instruction = None
    cur_trajectory = []
    cur_start_position = None
    rgb_frames = [[]]

    pbar = tqdm.tqdm(total=num_eps, desc=f"[{tag}]")

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

                ep_id = str(current_episodes[0].episode_id)
                instruction = instruction_override.get(
                    ep_id, current_episodes[0].instruction.instruction_text
                )
                cur_used_instruction = instruction

                interleaved_images = "<image>\n" * (len(past_and_current_rgbs) - 1)
                question = scaffold_fn(instruction, interleaved_images)

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
                frame = append_text_to_image(
                    frame, cur_used_instruction or current_episodes[i].instruction.instruction_text
                )
                rgb_frames[i].append(frame)

            if not dones[i]:
                continue

            ep_id = str(current_episodes[i].episode_id)
            final_pos = final_agent_position(envs, i)
            results[ep_id] = {
                "info": infos[i],
                "all_action_outputs": list(cur_outputs),
                "action_histogram": dict(cur_hist),
                "final_position": final_pos,
                "start_position": cur_start_position,
                "used_instruction": cur_used_instruction,
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
                results[ep_id]["info"].pop("top_down_map_vlnce", None)
                rgb_frames[i] = []

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
            cur_used_instruction = None
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
            if str(next_episodes[i].episode_id) in results:
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


def assemble_record(ep_id, pipeline, condition, condition_result):
    info = condition_result["info"]
    hist = condition_result["action_histogram"]
    num_steps = sum(hist.values())
    return {
        "episode_id": ep_id,
        "pipeline": pipeline,
        "condition": condition,
        "used_instruction": condition_result["used_instruction"],
        "all_action_outputs": condition_result["all_action_outputs"],
        "trajectory_length": info.get("path_length"),
        "num_steps": num_steps,
        "success": bool(info.get("success", 0.0)),
        "spl": info.get("spl"),
        "distance_to_goal": info.get("distance_to_goal"),
        "oracle_success": info.get("oracle_success"),
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
        help="Base eval config (identical to the standard NaVILA eval).",
    )
    parser.add_argument(
        "--model-path",
        default="/home/maitree-tiamat/models/navila-llama3-8b-8f",
    )
    parser.add_argument(
        "--instructions",
        default="phase1_instructions.json",
        help="Phase-1 instruction config (episode_ids + original/goal_only text).",
    )
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", default="phase1_results.json")
    parser.add_argument("--video-dir", default="phase1_videos",
                        help="Root for per-episode MP4s; subfoldered by pipeline/condition. Empty to disable.")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override MAX_EPISODE_STEPS (search needs more than route-following; "
                             "e.g. 2-3x the default). None = use config default.")
    parser.add_argument("--pipelines", default="phase1_baseline,phase1_explore",
                        help="Comma list of pipelines to run (subset of PIPELINES keys).")
    parser.add_argument("--conditions", default="original,goal_only",
                        help="Comma list of conditions to run.")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Run only the first N episodes (smoke test).")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    pipelines = [p.strip() for p in args.pipelines.split(",") if p.strip()]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for p in pipelines:
        assert p in PIPELINES, f"unknown pipeline {p}; choices: {list(PIPELINES)}"
    for c in conditions:
        assert c in CONDITIONS, f"unknown condition {c}; choices: {CONDITIONS}"

    config = get_config(args.exp_config, ["EVAL.SPLIT", args.split])
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

    episode_ids, instr_map = load_phase1_instructions(args.instructions)
    if args.max_episodes is not None:
        episode_ids = episode_ids[: args.max_episodes]
    logger.info(f"Phase 1: {len(episode_ids)} episodes, pipelines={pipelines}, conditions={conditions}")
    logger.info(f"Episodes: {episode_ids}")

    model_name = os.path.basename(os.path.normpath(args.model_path))
    logger.info(f"Loading model {args.model_path} ...")
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, model_name)
    model = model.cuda()

    metadata = {
        "experiment": "phase1_goal_only_search",
        "model": model_name,
        "split": args.split,
        "instructions_file": args.instructions,
        "episode_ids": episode_ids,
        "pipelines": pipelines,
        "conditions": conditions,
        "max_steps_override": args.max_steps,
    }

    # Resume support keyed by (pipeline, condition, episode_id).
    results = []
    done_keys = set()
    if os.path.exists(args.output):
        try:
            prior = json.load(open(args.output))
            results = list(prior.get("results", []))
            done_keys = {(r["pipeline"], r["condition"], r["episode_id"]) for r in results}
            logger.info(f"Resuming: loaded {len(results)} prior records from {args.output}")
        except Exception as e:  # noqa: BLE001
            logger.warn(f"Could not parse prior {args.output} ({e}); starting fresh.")
            results, done_keys = [], set()

    def flush_to_disk():
        tmp = args.output + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"metadata": metadata, "results": results}, f, indent=2)
        os.replace(tmp, args.output)

    base_video_dir = args.video_dir if args.video_dir else None

    for pipeline in pipelines:
        scaffold_fn = PIPELINES[pipeline]
        for condition in conditions:
            already = {eid for (p, c, eid) in done_keys if p == pipeline and c == condition}
            remaining = [e for e in episode_ids if e not in already]
            tag = f"{pipeline}/{condition}"
            if not remaining:
                logger.info(f"=== {tag} (already complete, skipping) ===")
                continue
            logger.info(f"=== {tag} — {len(remaining)}/{len(episode_ids)} to run ===")
            start = time.time()
            cell_video_dir = (
                os.path.join(base_video_dir, pipeline, condition) if base_video_dir else None
            )

            def on_episode_done(ep_id, cond_result, _p=pipeline, _c=condition):
                results.append(assemble_record(ep_id, _p, _c, cond_result))
                flush_to_disk()

            run_condition(
                model, tokenizer, image_processor, config, remaining,
                scaffold_fn=scaffold_fn,
                instruction_override=instr_map[condition],
                device=device,
                video_dir=cell_video_dir,
                max_steps=args.max_steps,
                tag=tag,
                on_episode_done=on_episode_done,
            )
            logger.info(f"{tag} done in {round(time.time() - start)}s ({len(results)} total records)")

    flush_to_disk()
    logger.info(f"Wrote {len(results)} records to {args.output}")


if __name__ == "__main__":
    main()
