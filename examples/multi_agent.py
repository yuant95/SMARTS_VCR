import pathlib

import gym
from argument_parser import default_argument_parser

from smarts import sstudio
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.utils.episodes import episodes
from smarts.zoo.agent_spec import AgentSpec
from smarts.env.wrappers.format_obs import FormatObs
from smarts.core.controllers import ActionSpaceType
from invertedAIAgent import invertedAiAgent

N_AGENTS = 8
AGENT_IDS = ["Agent %i" % i for i in range(N_AGENTS)]
location = "smarts:3lane_cruise_single_agent"


class KeepLaneAgent(Agent):
    def act(self, obs):
        return "keep_lane"


def main(scenarios, headless, num_episodes, max_episode_steps=None):
    agent_interface = AgentInterface(
        max_episode_steps=1000,
        waypoints=True,
        neighborhood_vehicles=True,
        drivable_area_grid_map=True,
        ogm=True,
        rgb=True,
        lidar=False,
        action=ActionSpaceType.TargetPose,
    )

    agent_specs = {
        agent_id: AgentSpec(
            # interface=AgentInterface.from_type(
            #     AgentType.Laner, max_episode_steps=max_episode_steps
            # ),
            interface=agent_interface,
            agent_builder=invertedAiAgent,
        )
        for agent_id in AGENT_IDS
    }

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs=agent_specs,
        headless=headless,
        sumo_headless=False,
    )

    wrappers = [FormatObs]

    for wrapper in wrappers:
        env = wrapper(env)

    for episode in episodes(n=num_episodes):
        agents = {
            agent_id: agent_spec.build_agent()
            for agent_id, agent_spec in agent_specs.items()
        }
        observations = env.reset()
        episode.record_scenario(env.scenario_log)

        dones = {"__all__": False}
        i = 0
        while not dones["__all__"]:
            i += 1
            actions = {}
            for agent_id, agent_obs in observations.items():
                agent_obs=observations[agent_id]
                if agent_id not in dones:
                    actions[agent_id] = agents[agent_id].act(agent_obs)
                elif not dones[agent_id]:
                    actions[agent_id] = agents[agent_id].act(agent_obs)               

            observations, rewards, dones, infos = env.step(actions)
            episode.record_step(observations, rewards, dones, infos)

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("multi-agent-example")
    args = parser.parse_args()

    if not args.scenarios:
        args.scenarios = [
            str(
                pathlib.Path(__file__).absolute().parents[1]
                / "smarts"
                / "scenarios"
                / "straight"
                / "3lane_cruise_multi_agent"
            )
        ]

    sstudio.build_scenario(scenario=args.scenarios)

    main(
        scenarios=args.scenarios,
        headless=args.headless,
        num_episodes=args.episodes,
    )
