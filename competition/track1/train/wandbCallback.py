import torch
import numpy as np
import wandb
import gym
import os

# add the stable baselines implimentations
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import safe_mean

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(
        self, 
        eval_env,
        verbose: int = 1, 
        n_eval_episodes: int = 5, 
        eval_freq=10000, 
        log_freq=100, 
        save_freq = 1000,
        deterministic=True,
        render=False, 
        model_name='sb3_model', 
        model_save_path='./local_save/',
        checkpoint_save_path='./local_save/',
        name_prefix="",
        gradient_save_freq = 0, 
        run_id=''
    )-> None:

        super(CustomCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.checkpoint_save_path = checkpoint_save_path
        self.name_prefix = name_prefix
        self.deterministic = deterministic
        self.render = render
        self.eval_incr = 1
        self.model_name = model_name
        self.best_mean_reward = -np.Inf
        self.verbose = verbose
        self.model_save_path = model_save_path+str(run_id)
        self.gradient_save_freq = gradient_save_freq
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.checkpoint_save_path, exist_ok=True)
        self.path = os.path.join(self.model_save_path, model_name+".zip")
        self.current_mod = 1

    def _init_callback(self) -> None:
        d = {}
        if "algo" not in d:
            d["algo"] = type(self.model).__name__
        for key in self.model.__dict__:
            if key in wandb.config:
                continue
            if type(self.model.__dict__[key]) in [float, int, str]:
                d[key] = self.model.__dict__[key]
            else:
                d[key] = str(self.model.__dict__[key])
        if self.gradient_save_freq > 0:
            wandb.watch(self.model.policy, log_freq=self.gradient_save_freq, log="all")
        wandb.config.setdefaults(d)

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        # Evaluate the model
        # policy = lambda obs_: self.model.predict(obs_, deterministic=True)[0]
        # avg_return, avg_horizon, avg_wp = evaluate_policy(policy, self.eval_env, STATIC_ARGS['tracks_folder'])
        episode_rewards, episode_lengths, episode_infos = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            render=self.render,
            deterministic=self.deterministic,
            return_episode_rewards=True,
        )
    
        mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
        mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)

        # wandb.log({'det_avg_reward':mean_reward,
        #            'det_avg_ep_len':mean_ep_length,
        #         #    'stoch_avg_return': safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]),
        #         #    'stoch_avg_horizon': safe_mean([ep_info["l"] for ep_info in self.model.ep_info_buffer]),
        #            'time_steps': self.num_timesteps,
        #            'updates': self.model._n_updates})
        wandb.log(dict({'det_avg_reward':mean_reward,
            'det_avg_ep_len':mean_ep_length,
            'time_steps': self.num_timesteps,
            'updates': self.model._n_updates}, **episode_infos))

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        # just log stochastic info
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
            wandb.log({'stoch_avg_return': safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]),
                    'stoch_avg_horizon': safe_mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])})

        wandb.log({'time_steps': self.num_timesteps,
                    'updates': self.model._n_updates})

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.
        :return: (bool) If the callback returns False, training is aborted early.
        """
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
                wandb.log({'stoch_avg_return': safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]),
                       'stoch_avg_horizon': safe_mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])})
                       
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.checkpoint_save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.zip")
            self.model.save(path)
            wandb.save(path, base_path=self.checkpoint_save_path)
            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")

        # if we have hit conditions for full eval
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Evaluate the model
            # policy = lambda obs_: self.model.predict(obs_, deterministic=True)[0]
            # avg_return, avg_horizon, avg_wp = evaluate_policy(policy, self.eval_env, STATIC_ARGS['tracks_folder'])
            episode_rewards, episode_lengths, episode_infos = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
            )
        
            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)

            # wandb.log({'det_avg_reward':mean_reward,
            #            'det_avg_ep_len':mean_ep_length,
            #         #    'stoch_avg_return': safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]),
            #         #    'stoch_avg_horizon': safe_mean([ep_info["l"] for ep_info in self.model.ep_info_buffer]),
            #            'time_steps': self.num_timesteps,
            #            'updates': self.model._n_updates})
            wandb.log(dict({'det_avg_reward':mean_reward,
                'det_avg_ep_len':mean_ep_length,
                'time_steps': self.num_timesteps,
                'updates': self.model._n_updates}, **episode_infos))
            
            if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
                wandb.log({'stoch_avg_return': safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]),
                       'stoch_avg_horizon': safe_mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])})

            # New best model, you could save the agent here
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.save_model()

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        """
        This method is called before the first rollout starts.
        """
        # Evaluate the model
        # policy = lambda obs_: self.model.predict(obs_, deterministic=True)[0]
        # avg_return, avg_horizon, avg_wp = eval_policy(policy, self.eval_env, STATIC_ARGS['tracks_folder'])
        # episode_rewards, episode_lengths = evaluate_policy(
        #         self.model,
        #         self.eval_env,
        #         n_eval_episodes=self.n_eval_episodes,
        #         render=self.render,
        #         deterministic=self.deterministic,
        #         return_episode_rewards=True,
        #     )
        
        # mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
        # mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
        
        # self.eval_env.reset()
        # log to wandb
        # wandb.log({
        #             # 'det_avg_reward':mean_reward,
        #         #    'det_avg_ep_len':mean_ep_length,
        #            'stoch_avg_return': safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]),
        #            'stoch_avg_horizon': safe_mean([ep_info["l"] for ep_info in self.model.ep_info_buffer]),
        #            'time_steps': self.num_timesteps,
        #            'updates': self.model._n_updates})
        # self.save_model()
        # return None

        # Evaluate the model
        episode_rewards, episode_lengths, episode_infos = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
            )
        
        mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
        mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
        
        wandb.log(dict({'det_avg_reward':mean_reward,
            'det_avg_ep_len':mean_ep_length,
            'time_steps': self.num_timesteps,
            'updates': self.model._n_updates}, **episode_infos))
        

        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
            wandb.log({'stoch_avg_return': safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]),
                    'stoch_avg_horizon': safe_mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])})

        self.save_model()


    def save_model(self) -> None:
        self.model.save(self.path)
        wandb.save(self.path, base_path=self.model_save_path)