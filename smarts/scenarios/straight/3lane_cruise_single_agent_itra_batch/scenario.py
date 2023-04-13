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

# flow_name = (start_lane, end_lane)
route_opt = [
    (0, 0),
    (1, 1),
    (2, 2),
]

# Traffic combinations = 3C2 + 3C3 = 3 + 1 = 4
# Repeated traffic combinations = 4 * 100 = 400
min_flows = 2
max_flows = 3
route_comb = [
    com
    for elems in range(min_flows, max_flows + 1)
    for com in combinations(route_opt, elems)
] * 100

traffic = {}
for name, routes in enumerate(route_comb):
    traffic[str(name)] = Traffic(
        flows=[
            Flow(
                route=Route(
                    begin=("gneE3", start_lane, 0),
                    end=("gneE3", end_lane, "max"),
                ),
                # Random flow rate, between x and y vehicles per minute.
                rate=60 * random.uniform(15, 25),
                # Random flow start time, between x and y seconds.
                begin=random.uniform(0, 5),
                # For an episode with maximum_episode_steps=3000 and step
                # time=0.1s, the maximum episode time=300s. Hence, traffic is
                # set to end at 900s, which is greater than maximum episode
                # time of 300s.
                end=60 * 15,
                actors={normal: 1},
                randomly_spaced=True,
            )
            for start_lane, end_lane in routes
        ]
    )


agent_prefabs = "smarts.scenarios.straight.3lane_cruise_single_agent_itra_batch.agent_prefabs"

# motion_planner_actor = t.SocialAgentActor(
#     name="motion-planner-agent",
#     agent_locator=f"{agent_prefabs}:motion-planner-agent-v0",
# )

# zoo_agent_actor = t.SocialAgentActor(
#     name="zoo-agent",
#     agent_locator=f"{agent_prefabs}:zoo-agent-v0",
# )

# invertedai_agent_actor = t.SocialAgentActor(
#     name="invertedai-agent",
#     agent_locator=f"{agent_prefabs}:inverted-agent-v0",
# )

invertedai_boid_agent = t.BoidAgentActor(
    name="invertedai-boid-agent",
    agent_locator=f"{agent_prefabs}:inverted-boid-agent-v0",
)

bubbles = [
    # t.Bubble(
    #     zone=t.MapZone(start=("gneE3", 0, 10), length=250, n_lanes=3),
    #     margin=2,
    #     actor=invertedai_boid_agent,
    #     keep_alive=True, 
    # ),
    t.Bubble(
        zone=t.PositionalZone(pos=(105, 0), size=(210, 20)),
        margin=5,
        actor=invertedai_boid_agent,
        keep_alive=True, 
    ),
]

social_agent_missions = {
    "all": (
        [
            t.SocialAgentActor(
                name="keep-lane-agent-v0",
                agent_locator="zoo.policies:keep-lane-agent-v0",
            ),
        ],
        [
            t.Mission(
                t.Route(begin=("gneE3", 0, 10), end=("gneE3", 0, "max"))
            )
        ],
    ),
}

route = Route(begin=("gneE3", 0, 10), end=("gneE3", 0, "max"))
ego_missions = [
    Mission(
        route=route,
        start_time=12,  # Delayed start, to ensure road has prior traffic.
    )
]

gen_scenario(
    scenario=Scenario(
        traffic=traffic,
        ego_missions=ego_missions,
        bubbles=bubbles,
        social_agent_missions=social_agent_missions,
    ),
    output_dir=Path(__file__).parent,
)