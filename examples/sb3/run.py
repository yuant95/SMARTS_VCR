import os

import tensorflow

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Silence the TF logs

import argparse
import pathlib
import warnings
from datetime import datetime
from typing import Any, Dict

import gym
from ruamel.yaml import YAML
from sb3 import action as sb3_action
from sb3 import observation as sb3_observation
from sb3 import reward as sb3_reward
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

warnings.simplefilter("ignore", category=DeprecationWarning)
yaml = YAML(typ="safe")


def main(args: argparse.Namespace):
    # Load config file.
    config_file = yaml.load(
        (pathlib.Path(__file__).absolute().parent / "config.yaml").read_text()
    )

    # Load env config.
    config_env = config_file["smarts"]
    config_env["mode"] = args.mode
    config_env["train_steps"] = args.train_steps

    # Create environment
    env = gym.make(
        "smarts.env:intersection-v0",
        headless=not args.head,  # If False, enables Envision display.
        visdom=config_env["visdom"],  # If True, enables Visdom display.
        sumo_headless=not config_env["sumo_gui"],  # If False, enables sumo-gui display.
    )

    # Wrap env with action, reward, and observation wrapper
    env = sb3_action.Action(env=env)
    env = sb3_reward.Reward(env=env)
    env = sb3_observation.Observation(env=env)

    # Setup train or evaluate.
    if config_env["mode"] == "train" and not args.logdir:
        # Begin training from scratch.
        time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        logdir = pathlib.Path(__file__).absolute().parents[0] / "logs" / time
    elif (config_env["mode"] == "train" and args.logdir) or (
        config_env["mode"] == "evaluate"
    ):
        # Begin training from a pretrained model.
        logdir = pathlib.Path(args.logdir)
    else:
        raise KeyError(
            f'Expected \'train\' or \'evaluate\', but got {config_env["mode"]}.'
        )
    logdir.mkdir(parents=True, exist_ok=True)
    print("Logdir:", logdir)

    # Check our custom environment compatibility with SB3
    env = Monitor(env=env, filename=str(logdir))
    check_env(env)

    # Run training or evaluation.
    run(env, config_env, logdir)
    env.close()


def run(env: gym.Env, config: Dict[str, Any], logdir: pathlib.Path):

    checkpoint_callback = CheckpointCallback(
        save_freq=config["checkpoint_freq"],
        save_path=logdir,
        name_prefix="rl_model",
    )

    if config["mode"] == "evaluate":
        print("Start evaluation.")
        model = PPO.load(logdir / "model.zip")
    elif config["mode"] == "train" and args.logdir:
        print("Start training from existing model.")
        model = PPO.load(logdir / "model.zip")
        model.set_env(env)
        model.learn(total_timesteps=config["train_steps"], callback=checkpoint_callback)
    else:
        print("Start training from scratch.")
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log=logdir / "tensorboard",
            use_sde=True,
        )
        model.learn(total_timesteps=config["train_steps"], callback=checkpoint_callback)

    # print("Evaluate policy")
    # evaluate_policy(model, env, n_eval_episodes=config["eval_eps"], deterministic=True)
    # print("Finished evaluating")

    if config["mode"] == "train":
        model.save(logdir / "model")


if __name__ == "__main__":
    program = pathlib.Path(__file__).stem
    parser = argparse.ArgumentParser(program)
    parser.add_argument(
        "--mode",
        help="`train` or `evaluate`. Default is `train`.",
        type=str,
        default="train",
    )
    parser.add_argument(
        "--logdir",
        help="Directory path to saved RL model. Required if `--mode=evaluate`, else optional.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--head", help="Run the simulation with display.", action="store_true"
    )

    args = parser.parse_args()

    if args.mode == "evaluate" and args.logdir is None:
        raise Exception("When --mode=evaluate, --logdir option must be specified.")

    main(args)
