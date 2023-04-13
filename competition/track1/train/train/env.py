from typing import Any, Dict, List

import gym
import sys
from pathlib import Path
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from train.action import Action as DiscreteAction
from train.action import Continuous_Action as Continuous_Action
from train.observation import Concatenate, FilterObs, SaveObs
from train.info import Info
from train.reward import Reward
from train.history import HistoryStack

from smarts.core.controllers import ActionSpaceType
from smarts.env.wrappers.format_action import FormatAction
from smarts.env.wrappers.format_obs import FormatObs
from smarts.env.wrappers.frame_stack import FrameStack
from smarts.env.wrappers.single_agent import SingleAgent
from smarts.env.wrappers.recorder_wrapper import RecorderWrapper
from smarts.env.wrappers.parallel_env import ParallelEnv


def wrappers_baseline(config: Dict[str, Any]):
    # fmt: off
    wrappers = [
        # Used to format observation space such that it becomes gym-space compliant.
        FormatObs,
        # Used to format action space such that it becomes gym-space compliant.
        lambda env: FormatAction(env=env, space=ActionSpaceType["TargetPose"]),
        Info,
        # Used to shape rewards.
        lambda env: Reward(env=env, weights=config["weights"]), 
        # Used to save selected observation parameters for use in DiscreteAction wrapper.
        SaveObs,
        # Used to discretize action space for easier RL training.
        DiscreteAction ,#if config["action_wrapper"]=="discrete" else Continuous_Action,
        # Used to filter only the selected observation parameters.
        FilterObs,
        # Used to stack sequential observations to include temporal information. 
        lambda env: FrameStack(env=env, num_stack=config["num_stack"]),
        # Concatenates stacked dictionaries into numpy arrays.
        lambda env: Concatenate(env=env, channels_order="first"),
        # Modifies interface to a single agent interface, which is compatible with libraries such as gym, Stable Baselines3, TF-Agents, etc.
        SingleAgent,
        lambda env: DummyVecEnv([lambda: env]),
        lambda env: VecMonitor(venv=env, filename=str(config["logdir"]), info_keywords=("is_success",))
    ]
    # fmt: on

    return wrappers

def wrappers_vec(config: Dict[str, Any]):
    # fmt: off
    wrappers = [
        # Used to format observation space such that it becomes gym-space compliant.
        FormatObs,
        # Used to format action space such that it becomes gym-space compliant.
        lambda env: FormatAction(env=env, space=ActionSpaceType["TargetPose"]),

        lambda env: HistoryStack(env=env, num_stack=config["num_stack"]),
        Info,
        # Used to shape rewards.
        # Reward,
        lambda env: Reward(env=env, weights=[config["w"+str(i)] for i in range(6)]), 
        # Used to save selected observation parameters for use in DiscreteAction wrapper.
        SaveObs,
        # Used to discretize action space for easier RL training.
        DiscreteAction if config["action_wrapper"]=="discrete" else Continuous_Action,
        # Used to filter only the selected observation parameters.
        FilterObs,
        # Used to stack sequential observations to include temporal information. 
        lambda env: FrameStack(env=env, num_stack=config["num_stack"]),
        # Concatenates stacked dictionaries into numpy arrays.
        lambda env: Concatenate(env=env, channels_order="first"),
        # Modifies interface to a single agent interface, which is compatible with libraries such as gym, Stable Baselines3, TF-Agents, etc.
        SingleAgent,
        # lambda env: DummyVecEnv([lambda: env]),
        # lambda env: VecMonitor(venv=env, filename=str(config["logdir"]), info_keywords=("is_success",))
    ]
    # fmt: on

    return wrappers

def wrappers_eval(config: Dict[str, Any]):
    # fmt: off
    wrappers = [
        # Used to format observation space such that it becomes gym-space compliant.
        FormatObs,
        # Used to format action space such that it becomes gym-space compliant.
        lambda env: FormatAction(env=env, space=ActionSpaceType["TargetPose"]),

        lambda env: HistoryStack(env=env, num_stack=config["num_stack"]),
        Info,
        # Used to shape rewards.
        # Reward,
        lambda env: Reward(env=env, weights=[config["w"+str(i)] for i in range(6)]), 
        # Used to save selected observation parameters for use in DiscreteAction wrapper.
        SaveObs,
        # Used to discretize action space for easier RL training.
        DiscreteAction if config["action_wrapper"]=="discrete" else Continuous_Action,
        # Used to filter only the selected observation parameters.
        FilterObs,
        # Used to stack sequential observations to include temporal information. 
        lambda env: FrameStack(env=env, num_stack=config["num_stack"]),
        # Concatenates stacked dictionaries into numpy arrays.
        lambda env: Concatenate(env=env, channels_order="first"),
        # Modifies interface to a single agent interface, which is compatible with libraries such as gym, Stable Baselines3, TF-Agents, etc.
        SingleAgent,
        lambda env: RecorderWrapper(env=env, video_name="video", traffic_agent=config["traffic_agent"]),
        # lambda env: DummyVecEnv([lambda: env]),
        # lambda env: VecMonitor(venv=env, filename=str(config["logdir"]), info_keywords=("is_success",))
    ]
    # fmt: on

    return wrappers


def make(
    config: Dict[str, Any], scenario: str, wrappers: List[gym.Wrapper] = [], seed = 0
) -> gym.Env:
    """Make environment.

    Args:
        config (Dict[str, Any]): A dictionary of config parameters.
        scenario (str): Scenario
        wrappers (List[gym.Wrapper], optional): Sequence of gym environment wrappers.
            Defaults to empty list [].

    Returns:
        gym.Env: Environment corresponding to the `scenario`.
    """
    env = gym.make(
        "smarts.env:multi-scenario-v0",
        scenario=scenario,
        img_meters=config["img_meters"],
        img_pixels=config["img_pixels"],
        sumo_headless=not config["sumo_gui"],  # If False, enables sumo-gui display.
        headless=not config["head"],  # If False, enables Envision display.
    )
    
    env.seed(seed)

    # Wrap the environment
    for wrapper in wrappers:
        env = wrapper(env)

    return env
