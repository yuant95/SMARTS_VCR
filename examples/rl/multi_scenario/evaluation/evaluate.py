# Add submission directory to python search path
import os
import argparse
import sys
from pathlib import Path

print(f"Adding python search path: {Path(__file__).absolute().parents[1]}")
sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

from typing import Any, Dict, List

import gym
from evaluation.copy_data import CopyData, DataStore
from evaluation.metric import Metric
from evaluation.score import Score
from submission.policy import IMG_METERS, IMG_PIXELS, Policy, submitted_wrappers

_SCORES_FILENAME = "scores.txt"

def make_env(
    config: Dict[str, Any],
    scenario: str,
    datastore: DataStore,
    wrappers: List[gym.Wrapper] = [],
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

    # Create environment
    env = gym.make(
        "smarts.env:multi-scenario-v0",
        scenario=scenario,
        img_meters=config["img_meters"],
        img_pixels=config["img_pixels"],
        action_space="Continuous",
    )

    # Make a copy of original info.
    env = CopyData(env, datastore)
    # Disallow modification of attributes starting with "_" by external users.
    env = gym.Wrapper(env)

    # Wrap the environment
    for wrapper in wrappers:
        env = wrapper(env)

    return env


def evaluate():
    config = {
        "img_meters": IMG_METERS,
        "img_pixels": IMG_PIXELS,
        "eval_episodes": 2,
    }
    scenarios = [
        "1_to_2lane_left_turn_c",
        # "1_to_2lane_left_turn_t",
        # "3lane_merge_multi_agent",
        # "3lane_merge_single_agent",
        # "3lane_cruise_multi_agent",
        # "3lane_cruise_single_agent",
        # "3lane_cut_in",
        # "3lane_overtake",
    ]

    # Make evaluation environments.
    envs_eval = {}
    for scen in scenarios:
        datastore = DataStore()
        envs_eval[f"{scen}"] = (
            make_env(
                config=config,
                scenario=scen,
                datastore=datastore,
                wrappers=submitted_wrappers(),
            ),
            datastore,
        )

    # Instantiate submitted policy.
    policy = Policy()

    # Evaluate model for each scenario
    score = Score()
    for env_name, (env, datastore) in envs_eval.items():
        print(f"Evaluating env {env_name}.")
        res = run(
            env=env, datastore=datastore, name=env_name, policy=policy, config=config
        )

        score.add(res)

    rank = score.compute()
    print("Overall Rank:\n", rank)

    print("\nFinished evaluating.\n")

    # Close all environments
    for env, _ in envs_eval.values():
        env.close()

    return rank

def to_codalab_scores_string(self) -> str:
    """Convert the data in scores to a CodaLab-scores-compatible string."""
    # NOTE: The score string names must be the same as in the competition.yaml.
    return (
        f"score: 0\n"
        f"completion: {rank['completion']}\n"
        f"humanness: {rank['humanness']}\n"
        f"rules: {rank['rules']}\n"
        f"time: {rank['time']}\n"
    )

def run(env, datastore: DataStore, name, policy: Policy, config: Dict[str, Any]):
    # Instantiate metric for score calculation.
    metric = Metric(datastore.agent_names)

    for _ in range(config["eval_episodes"]):
        observations = env.reset()
        dones = {"__all__": False}
        while not dones["__all__"]:
            actions = policy.act(observations)
            observations, rewards, dones, infos = env.step(actions)

            import time

            time.sleep(0.5)

            metric.store(infos=datastore.data["infos"], dones=datastore.data["dones"])

    print("-----------------------")
    print(metric.results())

    return metric.results()


def write_scores(scores, output_dir):
    if output_dir:
        with open(os.path.join(output_dir, _SCORES_FILENAME), "w") as output_file:
            output_file.write(scores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="codalab-evaluation",
        description=(
            "Evaluation script run by CodaLab that outputs a contestant's leaderboard "
            "score by running their policy in simulation."
        ),
    )

    # Define arguments for evaluation through CodaLab.
    parser.add_argument(
        "--input-dir",
        help=(
            "The path to the directory containing the reference data and user "
            "submission data."
        ),
        default=None,
        type=str,
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Path to the directory where the submission's scores.txt file will be "
            "written to."
        ),
        default=None,
        type=str,
    )
    args = parser.parse_args()
    rank = evaluate()

    write_scores(to_codalab_scores_string(rank), args.output_dir)
