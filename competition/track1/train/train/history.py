# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import copy
from collections import defaultdict, deque
from typing import Any, Dict, Tuple

import gym


class HistoryStack(gym.Wrapper):
    """Wrapper stacks num_stack (default=3) consecutive frames, in a moving-window
    fashion, and returns the stacked_frames.

    Note:
        Wrapper returns a deepcopy of the stacked frames, which may be expensive
        for large frames and large num_stack values.
    """

    def __init__(self, env: gym.Env, num_stack: int = 3):
        """
        Args:
            env (gym.Env): Gym environment to be wrapped.
            num_stack (int, optional): Number of frames to be stacked. Defaults to 3.
        """
        assert num_stack > 1, f"Expected num_stack > 1, but got {num_stack}."
        super(HistoryStack, self).__init__(env)
        self._num_stack = num_stack
        self._frames = {
            key: deque(maxlen=self._num_stack) for key in self.env.agent_specs.keys()
        }

    def _get_observations(self, obs: Dict[str, Any], agent_actions:Dict[str, Any]=None) -> Dict[str, Tuple[Any]]:
        """Update and return frames stack with given latest single frame."""

        for agent_id, agent_obs in obs.items():
            
            obs_history = {"ego": agent_obs["ego"]}
            if agent_actions is not None:
                if agent_id in agent_actions:
                    obs_history["actions"] = agent_actions[agent_id]

            self._frames[agent_id].appendleft(obs_history)
            while len(self._frames[agent_id]) < self._num_stack:
                self._frames[agent_id].appendleft(obs_history)
                
            frames_seq = tuple(self._frames[agent_id])
            agent_obs["history"] = copy.deepcopy(frames_seq)

        return obs

    def step(
        self, agent_actions: Dict
    ) -> Tuple[
        Dict[str, Tuple[Any]],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        """Steps the environment by one step.

        Args:
            agent_actions (Dict): Actions for each agent.

        Returns:
            Tuple[ Dict[str, Tuple[Any]], Dict[str, float], Dict[str, bool], Dict[str, Dict[str, Any]] ]: Observation, reward, done, info, for each agent.
        """
        env_observations, rewards, dones, infos = super(HistoryStack, self).step(
            agent_actions
        )

        return self._get_observations(env_observations, agent_actions), rewards, dones, infos

    def reset(self) -> Dict[str, Tuple[Any]]:
        """Resets the environment.

        Returns:
            Dict[str, Tuple[Any]]: Observation upon reset for each agent.
        """
        env_observations = super(HistoryStack, self).reset()

        for agent_dequeue in self._frames.values():
            agent_dequeue.clear()

        return self._get_observations(env_observations)