import argparse
import logging
import multiprocessing as mp
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
import sys
from datetime import datetime
from PIL import Image
import pandas as pd

import gym
import cloudpickle

from copy_data import CopyData, DataStore
from policy import Policy, submitted_wrappers
from utils import load_config, merge_config, validate_config, write_output

sys.setrecursionlimit(10000)
logger = logging.getLogger(__file__)

OUT_FOLDER = "/home/yuant426/Desktop/SMARTS_track1/competition/track1/trainingData/20221026_2"

_EVALUATION_CONFIG_KEYS = {
    "phase",
    "eval_episodes",
    "seed",
    "scenarios",
    "bubble_env_evaluation_seeds",
}
_DEFAULT_EVALUATION_CONFIG = dict(
    phase="track1",
    eval_episodes=2,
    seed=42,
    scenarios=[
        "1_to_2lane_left_turn_c",
        "1_to_2lane_left_turn_t",
        # "3lane_merge_multi_agent",
        # "3lane_merge_single_agent",
        # "3lane_cruise_multi_agent",
        # "3lane_cruise_single_agent",
        # "3lane_cut_in",
        # "3lane_overtake",
    ],
    bubble_env_evaluation_seeds=[6],
)
_SUBMISSION_CONFIG_KEYS = {
    "img_meters",
    "img_pixels",
}
_DEFAULT_SUBMISSION_CONFIG = dict(
    img_meters=50,
    img_pixels=256,
)

def _make_env(
    env_type: str,
    scenario: Optional[str],
    shared_configs: Dict[str, Any],
    seed: int,
    wrapper_ctors,
):
    """Build env.

    Args:
        env_type (str): Env type.
        scenario (Optional[str]): Scenario name or path to scenario folder.
        shared_configs (Dict[str, A64 type is supplied.

    Returns:
        Tuple[gym.Env, DataStore]: Wrapped environment and the datastore storing the observations.
    """

    # Make env.
    if env_type == "smarts.env:multi-scenario-v0":
        env = gym.make(env_type, scenario=scenario, **shared_configs)
    elif env_type == "bubble_env_contrib:bubble_env-v0":
        env = gym.make(env_type, **shared_configs)
    else:
        raise ValueError("Unknown env type.")

    # Make datastore.
    datastore = DataStore()
    # Make a copy of original info.
    env = CopyData(env, list(env.agent_specs.keys()), datastore)
    # Disallow modification of attributes starting with "_" by external users.
    env = gym.Wrapper(env)

    # Wrap the environment.
    wrappers = wrapper_ctors()
    for wrapper in wrappers:
        env = wrapper(env)

    # Set seed.
    env.seed(seed)

    return env, datastore

def run(config):
    shared_configs = dict(
        action_space="TargetPose",
        img_meters=int(config["img_meters"]),
        img_pixels=int(config["img_pixels"]),
        sumo_headless=True,
    )
    env_ctors = {}
    for scenario in config["scenarios"]:
        env_ctors[f"{scenario}"] = partial(
            _make_env,
            env_type="smarts.env:multi-scenario-v0",
            scenario=scenario,
            shared_configs=shared_configs,
            seed=config["seed"],
            wrapper_ctors=submitted_wrappers,
        )
    for seed in config["bubble_env_evaluation_seeds"]:
        env_ctors[f"bubble_env_{seed}"] = partial(
            _make_env,
            env_type="bubble_env_contrib:bubble_env-v0",
            scenario=None,
            shared_configs=shared_configs,
            seed=seed + config["seed"],
            wrapper_ctors=submitted_wrappers,
        )

    # score = Score()
    forkserver_available = "forkserver" in mp.get_all_start_methods()
    start_method = "forkserver" if forkserver_available else "spawn"
    mp_context = mp.get_context(start_method)
    with ProcessPoolExecutor(max_workers=3, mp_context=mp_context) as pool:
        futures = [
            pool.submit(
                _worker, cloudpickle.dumps([env_name, env_ctor, Policy, config])
            )
            for env_name, env_ctor in env_ctors.items()
        ]
    # for env_name, env_ctor in env_ctors.items():
    #     _worker(cloudpickle.dumps([env_name, env_ctor, Policy, config]))


    # rank = score.compute()
    logger.info("\nFinished Running.\n")
    return None

def _worker(input: bytes) -> None:
    """Compute metrics of a given env.

    Args:
        input (bytes): cloudpickle of [env_name: str, env_ctor: Callable[[], gym.Env],
            policy_ctor: Callable[[], Policy], config: Dict[str, Any]]

    Returns:
        Tuple[Counts, Costs]: Count and cost metrics.
    """

    import cloudpickle

    config: Dict[str, Any]
    datastore: DataStore
    env: gym.Env
    env_ctor: Callable[[], gym.Env]
    env_name: str
    policy_ctor: Callable[[], Policy]

    env_name, env_ctor, policy_ctor, config = cloudpickle.loads(input)
    logger.info("\nStarted evaluating env %s.\n", env_name)
    env, datastore = env_ctor()
    policy = policy_ctor()

    eval_episodes = 1 if "naturalistic" in env_name else config["eval_episodes"]

    out_folder = os.path.join(OUT_FOLDER, env_name)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    counter = {
        "collision": 0,
        "off_road": 0,
        "on_shoulder": 0,
        "wrong_way": 0,
        "off_route": 0,
        "safe": 0
    }

    df_file = os.path.join(out_folder, "df_{}.pkl".format(env_name))
    if os.path.exists(df_file):
        df = pd.read_pickle(df_file)
    else:
        df = pd.DataFrame()

    # for _ in range(eval_episodes):
    while any(c < 100 for c in counter.values() ):
        observations = env.reset()
        dones = {"__all__": False}
        while not dones["__all__"]:
            old_observations = observations
            actions = policy.act(observations)
            observations, rewards, dones, infos = env.step(actions)
            counter = event_counter(counter, observations)
            df = save_data(action=actions, old_observation=old_observations, observation=observations, df=df, out_folder=out_folder, counter=counter)
            
        df.to_pickle(df_file)

    env.close()
    df.to_pickle(df_file)
    logger.info("\nFinished evaluating env %s.\n", env_name)
    
    return None

def event_counter(counter, observation):
    for agent_id, agent_obs in observation.items():
        if agent_obs["events"]["collisions"]:
            counter["collision"] += 1
        elif agent_obs["events"]["off_road"]:
            counter["off_road"] += 1
        elif agent_obs["events"]["on_shoulder"]:
            counter["on_shoulder"] += 1
        elif agent_obs["events"]["wrong_way"]:
            counter["wrong_way"] += 1
        elif agent_obs["events"]["off_route"]:
            counter["off_route"] += 1
        else:
            counter["safe"] += 1

    return counter

def save_data(action, old_observation, observation, df, out_folder, counter):
    
    for agent_id, agent_obs in observation.items():
        old_agent_obs = old_observation[agent_id]

        waypoints = old_agent_obs["waypoints"]["pos"][0][:5]
        waypoints[:, -1] = old_agent_obs["waypoints"]["heading"][0][:5]
        ego_pos = old_agent_obs["ego"]["pos"]
        ego_pos[-1] = old_agent_obs["ego"]["heading"]

        data = {
            "action":[action[agent_id][:3]],
            "ego_pos": [ego_pos],
            # "ego_pos": [old_agent_obs["ego"]["pos"]],
            # "egp_heading": [old_agent_obs["ego"]["heading"]],
            # "waypoints_pos": [old_agent_obs["waypoints"]["pos"][0][:10]],
            # "waypoints_heading": [old_agent_obs["waypoints"]["heading"][0][:10]],
            "waypoints": [waypoints.flatten()],
            "waypoints_lane_width": [old_agent_obs["waypoints"]["lane_width"][0][:5]]
        }

        # Skip when waypoints is less than 5
        mask = [i for i,v in enumerate(data["waypoints_lane_width"][0]) if v > 0.0]
        if mask[-1] < 4:
            break

        if agent_obs["events"]["collisions"]:
            data["event"] = "collision"
        elif agent_obs["events"]["off_road"]:
            data["event"] = "off_road"
        elif agent_obs["events"]["on_shoulder"]:
            data["event"] = "on_shoulder"
        elif agent_obs["events"]["wrong_way"]:
            data["event"] = "wrong_way"
        elif agent_obs["events"]["off_route"]:
            data["event"] = "off_route"
        else:
            if counter["safe"] > 300:
                break
            else:
                data["event"] = "safe"
        
        rgb = old_agent_obs["rgb"]
        image_file_name = save_image(rgb, agent_id, out_folder)
        data["image_file"] = [image_file_name]

        dfi = pd.DataFrame(data=data)
        if df.empty:
            df = dfi
        else:
            df = df.append(dfi)

    return df
    

def save_image(rgb, prex, out_folder):
    filename = generate_filename()
    filename += "_" + prex + ".png"
    img = Image.fromarray(rgb, 'RGB')
    filepath = os.path.join(out_folder, filename)
    img.save(filepath)
    return filename

def generate_filename(): 
    time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    return "image_"+time


if __name__ == "__main__":
    # Get config parameters.
    config = merge_config(
        self=_DEFAULT_EVALUATION_CONFIG,
        other=_DEFAULT_SUBMISSION_CONFIG,
    )
    
    run(config)