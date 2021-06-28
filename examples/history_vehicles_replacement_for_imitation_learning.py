from dataclasses import replace
import logging
import math
import pickle
import random
from typing import Sequence, Tuple, Union

from envision.client import Client as Envision
from examples.argument_parser import default_argument_parser
from smarts.core import seed as random_seed
from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.scenario import Mission, Scenario
from smarts.core.sensors import Observation
from smarts.core.smarts import SMARTS
from smarts.core.traffic_history_provider import TrafficHistoryProvider
from smarts.core.utils.math import rounder_for_dt

logging.basicConfig(level=logging.INFO)


class ReplayCheckerAgent(Agent):
    """This is just a place holder such that the example code here has a real Agent to work with.
    This agent checks that the action space is working 'as expected'.
    In actual use, this would be replaced by an agent based on a trained Imitation Learning model."""

    def __init__(self, timestep_sec: float):
        self._timestep_sec = timestep_sec
        self._rounder = rounder_for_dt(timestep_sec)

    def load_data_for_vehicle(self, vehicle_id: str, scenario: Scenario):
        self._vehicle_id = vehicle_id  # for debugging
        datafile = f"data_{scenario.name}_{scenario.traffic_history.name}_Agent-history-vehicle-{vehicle_id}.pkl"
        # We read actions from a datafile previously-generated by the
        # observation_collection_for_imitation_learning.py script.
        # This allows us to test the action space to ensure that it
        # can recreate the original behaviour.
        with open(datafile, "rb") as pf:
            self._data = pickle.load(pf)

    def act(self, obs: Observation, obs_time: float) -> Tuple[float, float]:
        # note: in a real agent, you would not pass "obs_time" to act().
        # we're just doing it here to support this fake agent checking itself.
        assert self._data

        # First, check the observations representing the current state
        # to see if it matches what we expected from the recorded data.
        exp = self._data.get(obs_time)
        if not exp:
            return (0.0, 0.0)
        cur_state = obs.ego_vehicle_state
        assert math.isclose(
            cur_state.heading, exp["heading"], abs_tol=1e-09
        ), f'vid={self._vehicle_id}: {cur_state.heading} != {exp["heading"]} @ {obs_time}'
        # Note: the other checks can't be as tight b/c we lose some accuracy (due to angular acceleration)
        # by converting the acceleration vector to a scalar in the observation script,
        # which compounds over time throughout the simulation.
        assert math.isclose(
            cur_state.speed, exp["speed"], abs_tol=0.1
        ), f'vid={self._vehicle_id}: {cur_state.speed} != {exp["speed"]} @ {obs_time}'
        assert math.isclose(
            cur_state.position[0], exp["ego_pos"][0], abs_tol=2
        ), f'vid={self._vehicle_id}: {cur_state.position[0]} != {exp["ego_pos"][0]} @ {obs_time}'
        assert math.isclose(
            cur_state.position[1], exp["ego_pos"][1], abs_tol=2
        ), f'vid={self._vehicle_id}: {cur_state.position[1]} != {exp["ego_pos"][1]} @ {obs_time}'

        # Then get and return the next set of control inputs
        atime = self._rounder(obs_time + self._timestep_sec)
        data = self._data.get(atime, {"acceleration": 0, "angular_velocity": 0})
        return (data["acceleration"], data["angular_velocity"])


def main(
    script: str,
    scenarios: Sequence[str],
    headless: bool,
    seed: int,
    vehicles_to_replace: int,
    episodes: int,
):
    assert vehicles_to_replace > 0
    assert episodes > 0
    logger = logging.getLogger(script)
    logger.setLevel(logging.INFO)

    logger.debug("initializing SMARTS")

    smarts = SMARTS(
        agent_interfaces={},
        traffic_sim=None,
        envision=None if headless else Envision(),
    )
    random_seed(seed)
    rounder = rounder_for_dt(smarts.timestep_sec)
    traffic_history_provider = smarts.get_provider_by_type(TrafficHistoryProvider)
    assert traffic_history_provider

    scenarios_iterator = Scenario.scenario_variations(scenarios, [])
    for scenario in scenarios_iterator:
        logger.debug("working on scenario {}".format(scenario.name))
        veh_missions = scenario.discover_missions_of_traffic_histories()
        if not veh_missions:
            logger.warning(
                "no vehicle missions found for scenario {}.".format(scenario.name)
            )
            continue
        veh_start_times = {
            v_id: mission.start_time for v_id, mission in veh_missions.items()
        }

        k = vehicles_to_replace
        if k > len(veh_missions):
            logger.warning(
                "vehicles_to_replace={} is greater than the number of vehicle missions ({}).".format(
                    vehicles_to_replace, len(veh_missions)
                )
            )
            k = len(veh_missions)

        # XXX replace with AgentSpec appropriate for IL model
        agent_spec = AgentSpec(
            interface=AgentInterface.from_type(AgentType.Imitation),
            agent_builder=ReplayCheckerAgent,
            agent_params=smarts.timestep_sec,
        )

        for episode in range(episodes):
            logger.info(f"starting episode {episode}...")

            # Pick k vehicle missions to hijack with agent
            # and figure out which one starts the earliest
            agentid_to_vehid = {}
            agent_interfaces = {}
            history_start_time = None
            sample = scenario.traffic_history.random_overlapping_sample(
                veh_start_times, k
            )
            if len(sample) < k:
                logger.warning(
                    f"Unable to choose {k} overlapping missions.  allowing non-overlapping."
                )
                leftover = set(veh_start_times.keys()) - sample
                sample.update(set(random.sample(leftover, k - len(sample))))
            logger.info(f"chose vehicles: {sample}")
            for veh_id in sample:
                agent_id = f"ego-agent-IL-{veh_id}"
                agentid_to_vehid[agent_id] = veh_id
                agent_interfaces[agent_id] = agent_spec.interface
                if (
                    not history_start_time
                    or veh_start_times[veh_id] < history_start_time
                ):
                    history_start_time = veh_start_times[veh_id]

            # Build the Agents for the to-be-hijacked vehicles
            # and gather their missions
            agents = {}
            dones = {}
            ego_missions = {}
            for agent_id in agent_interfaces.keys():
                agent = agent_spec.build_agent()
                veh_id = agentid_to_vehid[agent_id]
                agent.load_data_for_vehicle(veh_id, scenario)
                agents[agent_id] = agent
                dones[agent_id] = False
                mission = veh_missions[veh_id]
                ego_missions[agent_id] = replace(
                    mission, start_time=mission.start_time - history_start_time
                )

            # Tell the traffic history provider to start traffic
            # at the point when the earliest agent enters...
            traffic_history_provider.start_time = history_start_time
            # and all the other agents to offset their missions by this much too
            scenario.set_ego_missions(ego_missions)
            logger.info(f"offsetting sim_time by: {history_start_time}")

            # Take control of vehicles with corresponding agent_ids
            smarts.switch_ego_agents(agent_interfaces)

            # Finally start the simulation loop...
            logger.info(f"starting simulation loop...")
            observations = smarts.reset(scenario)
            while not all(done for done in dones.values()):
                obs_time = rounder(smarts.elapsed_sim_time + history_start_time)
                actions = {
                    agent_id: agents[agent_id].act(agent_obs, obs_time)
                    for agent_id, agent_obs in observations.items()
                }
                logger.debug(
                    "stepping @ sim_time={} for agents={}...".format(
                        smarts.elapsed_sim_time, list(observations.keys())
                    )
                )
                observations, rewards, dones, infos = smarts.step(actions)

                for agent_id in agents.keys():
                    if dones.get(agent_id, False):
                        if not observations[agent_id].events.reached_goal:
                            logger.warning(
                                "agent_id={} exited @ sim_time={}".format(
                                    agent_id, smarts.elapsed_sim_time
                                )
                            )
                            logger.warning(
                                "   ... with {}".format(observations[agent_id].events)
                            )
                        else:
                            logger.info(
                                "agent_id={} reached goal @ sim_time={}".format(
                                    agent_id, smarts.elapsed_sim_time
                                )
                            )
                            logger.debug(
                                "   ... with {}".format(observations[agent_id].events)
                            )
                        del observations[agent_id]

    smarts.destroy()


if __name__ == "__main__":
    parser = default_argument_parser("history-vehicles-replacement-example")
    parser.add_argument(
        "--replacements-per-episode",
        "-k",
        help="The number vehicles to randomly replace with agents per episode.",
        type=int,
        default=3,
    )
    args = parser.parse_args()

    main(
        script=parser.prog,
        scenarios=args.scenarios,
        headless=args.headless,
        seed=args.seed,
        vehicles_to_replace=args.replacements_per_episode,
        episodes=args.episodes,
    )
