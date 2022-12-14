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
from random import randint

import numpy as np
import pytest

from smarts.contrib.pymarl import PyMARLHiWayEnv
from smarts.contrib.pymarl.adapters.action_adapter import N_ACTIONS

N_AGENTS = 3


@pytest.fixture
def env():
    configs = {
        "scenarios": ["scenarios/sumo/figure_eight"],
        "n_agents": N_AGENTS,
        "headless": True,
        "episode_limit": 1000,
        "visdom": False,
        "fixed_timestep_sec": 0.1,
    }

    env = PyMARLHiWayEnv(configs)
    yield env
    env.close()


def test_pymarl_env(env):
    def act(observation):
        # TODO: Remove randomness
        return randint(0, N_ACTIONS - 1)

    step = 0
    max_steps_across_episodes = 50

    # TODO: Switch to `for episode in episodes(env, n=MAX_EPISODES):`
    while True:  # for each episode...
        dones_registered = 0
        observations = env.reset()

        while True:  # for each step...
            actions = []
            for agent_obs in observations:
                actions.append(act(agent_obs))

            _, dones, _ = env.step(actions)
            observations = env.get_obs()

            for done in dones:
                dones_registered += 1 if done else 0
            if dones_registered == N_AGENTS:
                break

            step += 1
            if step >= max_steps_across_episodes:
                break

        # Remove this, and replace above break w/ a return?
        if step >= max_steps_across_episodes:
            break

    assert step == (
        max_steps_across_episodes
    ), "Simulation must cycle through to the final step"
