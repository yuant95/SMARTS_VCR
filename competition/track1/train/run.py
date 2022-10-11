from multiprocessing import dummy
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import argparse
import warnings
from datetime import datetime
from itertools import cycle
from pathlib import Path
from typing import Any, Dict

import gym
import stable_baselines3 as sb3lib
import torch as th
from ruamel.yaml import YAML
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import dummy_vec_env, subproc_vec_env, VecMonitor 
from train import env as multi_scenario_env
import network

import wandb
from wandbCallback import CustomCallback 
from wandb.integration.sb3 import WandbCallback

print("\nTorch cuda is available: ", th.cuda.is_available(), "\n")
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ResourceWarning)
yaml = YAML(typ="safe")

def get_config(args: argparse.Namespace):
    # Load config file.
    config_file = yaml.load(
        (Path(__file__).absolute().parent / "config.yaml").read_text()
    )

    # Load env config.
    config = config_file["smarts"]
    config.update(vars(args))

    # Setup logdir.
    if not args.logdir:
        time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        logdir = Path(__file__).absolute().parents[0] / "logs" / time
    else:
        logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config["logdir"] = logdir
    print("\nLogdir:", logdir, "\n")

    wandb_run = wandb.init(
        project="SMARTS",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
        )
    config = wandb_run.config

    return config, wandb_run

def main(args: argparse.Namespace):
    config, wandb_run = get_config(args)

    # Setup model.
    if config["mode"] == "evaluate":
        # Begin evaluation.
        config["model"] = args.model
        print("\nModel:", config["model"], "\n")
    elif config["mode"] == "train" and not args.model:
        # Begin training.
        pass
    else:
        raise KeyError(f'Expected \'train\' or \'evaluate\', but got {config["mode"]}.')

    # Make training and evaluation environments.
    envs_train = {}
    envs_eval = {}
    wrappers = multi_scenario_env.wrappers_vec(config=config)

    envs_train = [multi_scenario_env.make(config=config, scenario=scen, wrappers=wrappers, seed=seed) 
                   for scen, seed in zip(config["scenarios"], range(len(config["scenarios"]))) ]
    envs_train = dummy_vec_env.DummyVecEnv([lambda i=i:envs_train[i] for i in range(len(envs_train))])
    envs_train = VecMonitor(venv=envs_train, filename=str(config["logdir"]), info_keywords=("is_success",))
    

    envs_eval = [multi_scenario_env.make(config=config, scenario=scen, wrappers=wrappers, seed=seed) 
                    for scen, seed in zip(config["scenarios"], range(len(config["scenarios"]))) ]   
    envs_eval = dummy_vec_env.DummyVecEnv([lambda i=i:envs_eval[i] for i in range(len(envs_eval))])
    envs_eval = VecMonitor(venv=envs_eval, filename=str(config["logdir"]), info_keywords=("is_success",))


    # Run training or evaluation.
    run(envs_train=envs_train, envs_eval=envs_eval, config=config, wandb_run = wandb_run)

    # Close all environments
    envs_train.close()
    envs_eval.close()


def run(
    envs_train: Dict[str, gym.Env],
    envs_eval: Dict[str, gym.Env],
    config: Dict[str, Any],
    wandb_run
):
    if config["mode"] == "train":
        print("\nStart training.\n")
        model = getattr(sb3lib, config["alg"])(
            env=envs_train, #[next(scenarios_iter)],
            verbose=1,
            tensorboard_log=config["logdir"] + "/tensorboard",
            **network.combined_extractor(config),
        )
        if config["baseline"]:
            model.load(config["baseline"])
        for index in range(config["epochs"]):
            checkpoint_callback = CheckpointCallback(
                save_freq=config["checkpoint_freq"],
                save_path=config["logdir"] + "/checkpoint",
                name_prefix=f"{config['alg']}_{index}",
            )
            custom_callback = CustomCallback(
                verbose = 1, 
                eval_env=envs_eval,
                n_eval_episodes = config["eval_eps"], 
                eval_freq=config["eval_freq"],
                log_freq=100, 
                deterministic=True,
                render=False, 
                model_name='sb3_model', 
                model_save_path= str(config["logdir"] + "/eval"),
                gradient_save_freq=0, 
                run_id=wandb_run.id
            )
            model.set_env(envs_train)
            model.learn(
                total_timesteps=config["train_steps"],
                callback=[custom_callback, checkpoint_callback],
            )

        # Save trained model.
        save_dir = config["logdir"] + "/train"
        save_dir.mkdir(parents=True, exist_ok=True)
        time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        model.save(save_dir + ("/model_" + time))
        print("\nSaved trained model.\n")

    if config["mode"] == "evaluate":
        print("\nEvaluate policy.\n")
        model = getattr(sb3lib, config["alg"]).load(
            config["model"], print_system_info=True
        )
        for env_name, env_eval in envs_eval.items():
            print(f"\nEvaluating env {env_name}.")
            mean_reward, std_reward = evaluate_policy(
                model, env_eval, n_eval_episodes=config["eval_eps"], deterministic=True
            )
            print(f"Mean reward:{mean_reward:.2f} +/- {std_reward:.2f}\n")
        print("\nFinished evaluating.\n")


if __name__ == "__main__":
    program = Path(__file__).stem
    parser = argparse.ArgumentParser(program)
    parser.add_argument(
        "--mode",
        help="`train` or `evaluate`. Default is `train`.",
        type=str,
        default="train",
    )
    parser.add_argument(
        "--logdir",
        help="Directory path for saving logs.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model",
        help="Directory path to saved RL model. Required if `--mode=evaluate`.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--epochs",
        help=" Number of training loops",
        type=int,
        default=5_000,
    )
    parser.add_argument(
        "--train_steps",
        help="Training per scenario",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--checkpoint_freq",
        help="Save a model every checkpoint_freq calls to env.step().",
        type=int,
        default=5_000 ,
    )
    parser.add_argument(
        "--eval_eps",
        help="Number of evaluation epsiodes.",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--eval_freq",
        help=" Evaluate the trained model every eval_freq steps and save the best model.",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--alg",
        help="Stable Baselines3 algorithm.",
        type=str,
        default="PPO",
    )
    parser.add_argument(
        "--action_wrapper",
        help="Choose from discrete and continous",
        type=str,
        default="discrete",
    )
    parser.add_argument(
        "--baseline",
        help="Will load the model given the path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--weights",
        help="The weights for reward category Complete, Humanness, Time, Rules, Goal, Distant.",
        type=float,
        nargs='+',
        default=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )

    args = parser.parse_args()

    if args.mode == "evaluate" and args.model is None:
        raise Exception("When --mode=evaluate, --model option must be specified.")

    main(args)