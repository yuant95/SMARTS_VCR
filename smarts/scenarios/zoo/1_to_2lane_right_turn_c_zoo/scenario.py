# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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

import random
from itertools import combinations
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio.types import Flow, Mission, Route, Scenario, Traffic, TrafficActor
from smarts.sstudio import types as t

normal = TrafficActor(
    name="car",
)

vertical_routes = [
    ("E0", 0, "E3", 0),
    ("-E3", 0, "-E0", 0),
]

horizontal_routes = [
    ("E4", 0, "E1", 0),
    ("E4", 1, "E1", 1),
    ("-E1", 0, "-E4", 0),
    ("-E1", 1, "-E4", 1),
]

turn_left_routes = [
    ("E0", 0, "E1", 1),
    ("-E3", 0, "-E4", 1),
    ("-E1", 1, "E3", 0),
    ("E4", 1, "-E0", 0),
]

turn_right_routes = [
    ("E0", 0, "-E4", 0),
    ("-E3", 0, "E1", 0),
    ("-E1", 0, "-E0", 0),
    ("E4", 0, "E3", 0),
]

# Total route combinations = 14C1 + 14C2 + 14C3 + 14C4 = 1470
all_routes = vertical_routes + horizontal_routes + turn_left_routes + turn_right_routes
route_comb = [com for elems in range(1, 5) for com in combinations(all_routes, elems)]
traffic = {}
scale = 2
for name, routes in enumerate(route_comb):
    traffic[str(name)] = Traffic(
        flows=[
            Flow(
                route=Route(
                    begin=(start_edge, start_lane, 0),
                    end=(end_edge, end_lane, "max"),
                ),
                # Random flow rate, between x and y vehicles per minute.
                rate=60 * random.uniform(int(scale*5), int(scale*15)),
                # Random flow start time, between x and y seconds.
                begin=random.uniform(0, 3),
                # For an episode with maximum_episode_steps=3000 and step
                # time=0.1s, the maximum episode time=300s. Hence, traffic is
                # set to end at 900s, which is greater than maximum episode
                # time of 300s.
                end=60 * 15,
                actors={normal: 1},
            )
            for start_edge, start_lane, end_edge, end_lane in routes
        ]
    )

agent_prefabs = "smarts.scenarios.zoo.agent_prefabs"

invertedai_boid_agent = t.BoidAgentActor(
    name="invertedai-boid-agent",
    agent_locator=f"{agent_prefabs}:inverted-boid-agent-v0",
)

invertedai_agent_actor = t.SocialAgentActor(
    name="invertedai-agent",
    agent_locator=f"{agent_prefabs}:inverted-agent-v0",
)

zoo_agent_actor = t.SocialAgentActor(
    name="zoo-agent",
    agent_locator=f"{agent_prefabs}:zoo-agent-v0",
)

bubbles = [
    t.Bubble(
        zone=t.PositionalZone(pos=(50, 40), size=(120, 120)),
        margin=5,
        actor=zoo_agent_actor,
    ),
]


route = Route(begin=("E0", 0, 5), end=("-E4", 0, "max"))
ego_missions = [
    Mission(
        route=route,
        start_time=4,  # Delayed start, to ensure road has prior traffic.
    )
]

gen_scenario(
    scenario=Scenario(
        traffic=traffic,
        ego_missions=ego_missions,
        bubbles=bubbles,
    ),
    output_dir=Path(__file__).parent,
)
