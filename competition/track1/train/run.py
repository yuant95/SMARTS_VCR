import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import warnings
import sys

from functools import partial
from datetime import datetime
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, List

import gym
import stable_baselines3 as sb3lib
import torch as th
import numpy as np

from ruamel.yaml import YAML
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy, evaluate_policy_details, evaluate_policy_visualization
from stable_baselines3.common.vec_env import dummy_vec_env, subproc_vec_env, VecMonitor 
from train import env as multi_scenario_env
from train import network
    
# To import submission folder
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submission"))

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
        (Path(__file__).resolve().parent / "config.yaml").read_text()
    )
    # Load env config.
    config = config_file["smarts"]
    config.update(vars(args))

    # Setup logdir.
    if not args.logdir:
        time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        logdir = Path(__file__).resolve().parents[0] / "logs" / time
    else:
        logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config["logdir"] = logdir
    print("\nLogdir:", logdir, "\n")

    # wandb.tensorboard.patch(root_logdir=str(logdir))
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
    elif config["mode"] == "train":
        if not args.model:
            # Begin training.
            pass
        else:
            raise KeyError(f'Expected no model, but got {args.model}.')

    else:
        raise KeyError(f'Expected \'train\' or \'evaluate\', but got {config["mode"]}.')

    # Make training and evaluation environments.
    envs_train = {}
    envs_eval = {}
    wrappers = multi_scenario_env.wrappers_vec(config=config)
    wrappers_eval = multi_scenario_env.wrappers_eval(config=config)
    if not config["eval_visualization"]:
        wrappers_eval = wrappers_eval[:-1]

    traffic_agent = config["traffic_agent"]
    if traffic_agent == "sumo":
        scenarios = config["sumo_scenarios"]
        scenarios_eval = scenarios
    elif traffic_agent == "zoo":
        scenarios = config["smarts_zoo_scenarios"]
        scenarios_eval = scenarios
    elif traffic_agent == "itra":
        scenarios = config["itra_scenarios"]
        scenarios_eval = scenarios #config["itra_evaluation_scenarios"]
    elif traffic_agent == "all":
        scenarios = []
        scenarios.extend(config["sumo_scenarios"])
        scenarios.extend(config["smarts_zoo_scenarios"])
        scenarios.extend(config["itra_scenarios"])
        if config["mode"] == "train":
            scenarios_eval = scenarios
        else:
            scenarios_eval = {
                "sumo":config["sumo_scenarios"],
                "zoo":config["smarts_zoo_scenarios"],
                "itra":config["itra_scenarios"],
            }
    else:
        raise RuntimeError("Traffic agent type {} is not supported.".format(traffic_agent))
                   
    # from smarts.env.multi_scenario_env import multi_scenario_v0_env
    # env_constructors = [
    #     partial(multi_scenario_v0_env, scenario=scen, 
    #     img_meters=config["img_meters"],
    #     img_pixels=config["img_pixels"],
    #     action_space="TargetPose",
    #     headless= True,
    #     visdom= False,
    #     sumo_headless = not config["sumo_gui"],
    #     envision_record_data_replay_path= None,
    #     wrappers=[]) 
    #     for scen, seed in zip(scenarios, range(len(scenarios)))
    # ]

    # # Create parallel environments
    # envs_train = [ParallelEnv(
    #     env_constructors=env_constructors,
    #     auto_reset=True,
    #     seed=42,
    # )]
    
    # envs_train = ParallelEnv(env_constructors=env_constructors,
    #                         auto_reset=True, 
    #                         seed=42)

    envs_train = [multi_scenario_env.make(config=config, scenario=scen, wrappers=wrappers, seed=seed) 
                   for scen, seed in zip(scenarios, range(len(scenarios))) ]
    n_envs_train = len(envs_train)
    envs_train = dummy_vec_env.DummyVecEnv([lambda i=i:envs_train[i] for i in range(len(envs_train))])
    envs_train = VecMonitor(venv=envs_train, filename=str(config["logdir"]), info_keywords=("is_success",))

    # Initialize evaluation with different random seed
    if type(scenarios_eval) == list:
        envs_eval = [multi_scenario_env.make(config=config, scenario=scen, wrappers=wrappers_eval, seed=seed*2) 
                        for scen, seed in zip(scenarios_eval, range(len(scenarios_eval))) ]
    else:
        envs_eval = {}   
        for key, value in scenarios_eval.items():
            envs_eval[key] = [multi_scenario_env.make(config=config, scenario=scen, wrappers=wrappers_eval, seed=seed*2) 
                        for scen, seed in zip(value, range(len(value))) ]
    # envs_eval = dummy_vec_env.DummyVecEnv([lambda i=i:envs_eval[i] for i in range(len(envs_eval))])
    # envs_eval = VecMonitor(venv=envs_eval, filename=str(config["logdir"]), info_keywords=("is_success",))

    CONST_N_ENVS = 6 
    train_steps = int (config["train_steps"] * CONST_N_ENVS / n_envs_train)
    config.update({"train_steps":train_steps}, allow_val_change=True)

    # Run training or evaluation.
    run(envs_train=envs_train, envs_eval=envs_eval, config=config, wandb_run = wandb_run)

    # Close all environments
    envs_train.close()
    # envs_eval.close()


def run(
    envs_train: Dict[str, gym.Env],
    envs_eval: Dict[str, gym.Env],
    config: Dict[str, Any],
    wandb_run
):
    if config["mode"] == "train":
        # envs_eval = dummy_vec_env.DummyVecEnv([lambda i=i:envs_eval[i] for i in range(len(envs_eval))])
        # envs_eval = VecMonitor(venv=envs_eval, filename=str(config["logdir"]), info_keywords=("is_success",))

        print("\nStart training.\n")
        if config["baseline"]:
            # model.load(config["baseline"])
            model = sb3lib.PPO.load(config["baseline"], 
                                    env=envs_train, verbose=1, 
                                    tensorboard_log=config["logdir"] + "/tensorboard")
            
        else:
            model = getattr(sb3lib, config["alg"])(
                env=envs_train, #[next(scenarios_iter)],
                verbose=1,
                tensorboard_log=config["logdir"] + "/tensorboard",
                **network.combined_extractor(config),
            )
        
        custom_callback = CustomCallback(
            verbose = 1, 
            eval_env=envs_eval,
            n_eval_episodes = config["eval_eps"], 
            eval_freq=config["eval_freq"],
            log_freq=100, 
            save_freq=config["checkpoint_freq"],
            deterministic=True,
            render=False, 
            model_name='sb3_model', 
            model_save_path= str(config["logdir"] + "/eval"),
            checkpoint_save_path=config["logdir"] + "/checkpoint",
            name_prefix=f"{config['alg']}",
            gradient_save_freq=0, 
            run_id=wandb_run.id
        )
        model.set_env(envs_train)
        model.learn(
            total_timesteps=config["train_steps"],
            callback=[custom_callback],
        )

        # Save trained model.
        save_dir = config["logdir"] + "/train"
        os.makedirs(save_dir, exist_ok=True)
        time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        model.save(save_dir + ("/model_" + time))
        wandb.save(save_dir + ("/model_" + time), base_path=save_dir)
        print("\nSaved trained model.\n")

    if config["mode"] == "evaluate":
        print("\nEvaluate policy.\n")
        model = getattr(sb3lib, config["alg"]).load(
            config["model"], print_system_info=True
        )
        columns = ["Traffic Agent", "Completion Rate", "Front Collision Rate", "Rear Collision Rate", "Other Collision Rate", "Collision Rate", "Off Road Rate"]
        result_table = wandb.Table(columns=columns)

        if config["traffic_agent"] == "all":
            for traffic_agent, envs in envs_eval.items():
                table_row = get_evaluation_results(traffic_agent, envs, model, config)
                result_table.add_data(*table_row)
        else:
            table_row = get_evaluation_results(config["traffic_agent"], envs_eval, model, config)
            result_table.add_data(*table_row)

        wandb.log({"Evaluation Table": result_table})

        print("\nFinished evaluating.\n")


def get_evaluation_results(traffic_agent: str, envs_eval: List[gym.Env], model, config):
    results = evaluate_policy_visualization(
                    model, 
                    envs_eval, 
                    n_eval_episodes=config["eval_eps"], 
                    deterministic=True, 
                    return_episode_rewards=False,
                    meta_data=config["eval_visualization"],
                    visualization=config["eval_visualization"],
                    return_detailed_results=True,
                    traffic_agent=traffic_agent
                )
    data = get_evaluation_table_row(traffic_agent, results)

    return data


def get_evaluation_table_row(traffic_agent:str, results:Dict):

    mean_collision_rate = np.mean([results[scen_name]["collision_rate"] for scen_name in list(results.keys())])
    mean_completion_rate = np.mean([results[scen_name]["completion_rate"] for scen_name in list(results.keys())])
    mean_off_road_rate = np.mean([results[scen_name]["off_road_rate"] for scen_name in list(results.keys())])

    total_collision = np.sum([results[scen_name]["collision"] for scen_name in list(results.keys())])
    total_front_collision = np.sum([results[scen_name]["collision_type"]["front"] for scen_name in list(results.keys())])
    total_rear_collision = np.sum([results[scen_name]["collision_type"]["rear"] for scen_name in list(results.keys())])
    total_other_collision = np.sum([results[scen_name]["collision_type"]["other"] for scen_name in list(results.keys())])

    mean_collision_front_rate = mean_collision_rate*total_front_collision/total_collision
    mean_collision_rear_rate = mean_collision_rate*total_rear_collision/total_collision
    mean_collision_other_rate = mean_collision_rate*total_other_collision/total_collision

    my_data = [traffic_agent, mean_completion_rate, mean_collision_front_rate, mean_collision_rear_rate, mean_collision_other_rate, mean_collision_rate, mean_off_road_rate]

    return my_data


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
        default="",
        # default="/home/yuant426/Downloads/PPO_990000_steps_smooth-rain-1288.zip",
        # default="/home/yuant426/Downloads/dandy-deluge-1257_PPO_450000_steps.zip",
        # default="/home/yuant426/Desktop/SMARTS_track1/competition/track1/train/logs/2023_06_06_14_17_37/checkpoint/PPO_1350000_steps.zip",
        # default="/home/yuant426/Downloads/fragrant-valley-1327_PPO_1680000_steps.zip"
        # default="/home/yuant426/Downloads/legendary-jazz-1422_PPO_1860000_steps.zip"
        # default="/home/yuant426/Downloads/driven-dew-1499_PPO_420000_steps.zip"
        # default="/home/yuant426/Downloads/PPO_720000_steps.zip"
    )
    # parser.add_argument(
    #     "--epochs",
    #     help=" Number of training loops",
    #     type=int,
    #     default=5_000,
    # )
    parser.add_argument(
        "--train_steps",
        help="Total training step",
        type=int,
        default=4_000_000,
    )
    parser.add_argument(
        "--checkpoint_freq",
        help="Save a model every checkpoint_freq calls to env.step().",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--eval_eps",
        help="Number of evaluation epsiodes.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--eval_freq",
        help=" Evaluate the trained model every eval_freq steps and save the best model.",
        type=int,
        default=5_000,
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
        default="discrete11",
    )
    parser.add_argument(
        "--eval_visualization",
        help="Whether include visualization upload in the evaluation process",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--meta_data",
        help="Whether include meta data including traffic density etc in the evaluation process",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--baseline",
        help="Will load the model given the path",
        type=str,
        default="",
        # default="/home/yuant426/Desktop/SMARTS_track1/competition/track1/train/logs/2023_03_30_00_58_00/checkpoint/PPO_640000_steps.zip"
        # default="/home/yuant426/Downloads/PPO_990000_steps_smooth-rain-1288.zip",
        # default="/home/yuant426/Desktop/SMARTS_track1/competition/track1/train/logs/2023_05_24_00_12_45/checkpoint/PPO_990000_steps.zip",
        # default="/home/yuant426/Downloads/super-frog-1366-PPO_990000_steps.zip"
        # default="/home/yuant426/Downloads/legendary-jazz-1422_PPO_1860000_steps.zip"
        # default="/home/yuant426/Downloads/PPO_1620000_steps.zip"
    )
    parser.add_argument(
        "--w0",
        help="Complete: -50 for collision.",
        type=float,
        default= 0.0
    )
    parser.add_argument(
        "--w1",
        help="Humanness: jerk angular + jerk linear + lane center offset",
        type=float,
        default= 0.2
    )
    parser.add_argument(
        "--w2",
        help="Time: -distance to goal",
        type=float,
        default= 0.5
    )
    parser.add_argument(
        "--w3",
        help="Rules: wrong way + speed limit.",
        type=float,
        default= 0.3
    )
    parser.add_argument(
        "--w4",
        help="Goal: reached goal reward + penalize off road/ off route/ on shoulder.",
        type=float,
        default= 0.0
    )
    parser.add_argument(
        "--w5",
        help="Traveled distance.",
        type=float,
        default= 0.0
    )
    parser.add_argument(
        "--traffic_agent",
        help="Pick traffic agent from sumo, smarts zoo, and itra",
        type=str,
        default= "itra"
    )


    args = parser.parse_args()

    if args.mode == "evaluate" and args.model is None:
        raise Exception("When --mode=evaluate, --model option must be specified.")

    main(args)