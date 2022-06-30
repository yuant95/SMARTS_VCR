import re
from typing import Any, Dict

import numpy as np
from custom_dict import CustomDict

from smarts.core.sensors import Observation

_MAX_STEPS = 3000


class Metric:
    def __init__(self, agent_names):
        self._agent_names = agent_names
        self._num_agents = len(agent_names)
        self._cost_funcs = self._reinit()
        self._costs = {
            name: CustomDict(**{key: 0 for key in self._cost_funcs[name].keys()})
            for name in self._agent_names
        }

        self._episodes = 0
        """ Total number of episodes.
        """
        self._incomplete = 0
        """ Total number of episodes not completed by agents. Fractional `self._incomplete`
        occur when only some agents do not complete the episode in a multi-agent case.
        """
        self._steps = 0
        """ Total number of steps.
        """
        self._adjusted_steps = 0
        """ Total number of `self._adjusted_steps_per_episode`.
        """
        self._goals_unachieved = 0
        """ Total number of episodes, where at least one agent did not achieve their goal.
        """

        self._incomplete_per_episode = 0
        """ Number of agents which did not complete the episode. Episode is incomplete if 
        any agent becomes done due to traffic violation.
        """
        self._adjusted_steps_per_episode = 0
        """ Total `act` steps taken by all agents per episode. Maximum steps `_MAX_STEPS` 
        is assigned if the episode is incomplete.
        """
        self._goals_per_episode = 0
        """ Number of agents which achieved their goals per episode.         
        """

    def _reinit(self):
        return {
            agent_name: {
                cost_name: cost_func() for cost_name, cost_func in COST_FUNCS.items()
            }
            for agent_name in self._agent_names
        }

    def store(self, infos: Dict[str, Any], dones: Dict[str, bool]):
        # Compute all cost functions
        for agent_name, agent_info in infos.items():
            agent_obs = agent_info["env_obs"]
            results = map(
                lambda cost_func: cost_func(agent_obs),
                self._cost_funcs[agent_name].values(),
            )
            costs = {k: v for d in results for k, v in d.items()}
            self._costs[agent_name] += costs

        # Count only steps where an ego agent was present.
        if len(infos) > 0:
            self._steps += 1
            self._adjusted_steps_per_episode += 1

        # Count `self._incomplete_per_episode` and `self._goals_per_episode`.
        for agent_name, agent_info in infos.items():
            agent_done = dones[agent_name]
            agent_obs = agent_info["env_obs"]
            if agent_done and (
                len(agent_obs.events.collisions)
                > 0
                | agent_obs.events.off_road
                | agent_obs.events.off_route
                | agent_obs.events.on_shoulder
                | agent_obs.events.wrong_way
            ):
                self._incomplete_per_episode += 1
            elif agent_done and agent_obs.events.reached_goal:
                self._goals_per_episode += 1

        if dones["__all__"] == True:
            # Count number of episodes.
            self._episodes += 1
            # Check whether episode was complete.
            if self._incomplete_per_episode > 0:
                self._incomplete += self._incomplete_per_episode / self._num_agents
                self._adjusted_steps_per_episode == _MAX_STEPS  # Assign max steps if episode is incomplete.

            # Update running total number of steps.
            self._adjusted_steps += self._adjusted_steps_per_episode
            # Update running total number of episodes with goal unachieved.
            self._goals_unachieved += (
                1 if self._num_agents != self._goals_per_episode else 0
            )

            # Reset functions with memory.
            self._cost_funcs = self._reinit()
            # Reset per-episode counters.
            self._incomplete_per_episode = 0
            self._adjusted_steps_per_episode = 0
            self._goals_per_episode = 0

    def results(self):
        return {
            "costs": self._costs,
            "episodes": self._episodes,
            "incomplete": self._incomplete,
            "steps": self._steps,
            "adjusted_steps": self._adjusted_steps,
            "goals_unachieved": self._goals_unachieved,
        }


COST_FUNCS = {
    "collisions": lambda: _collisions,
    "dist_to_obstacles": lambda: _dist_to_obstacles,
    "jerk": lambda: _jerk,
    "lane_center_offset": lambda: _lane_center_offset,
    "off_road": lambda: _off_road,
    "off_route": lambda: _off_route,
    "on_shoulder": lambda: _on_shoulder,
    "steering_rate": lambda: _steering_rate(),
    "velocity_offset": lambda: _velocity_offset,
    "wrong_way": lambda: _wrong_way,
    "yaw_rate": lambda: _yaw_rate,
}


def _collisions(obs: Observation) -> Dict[str, int]:
    return {"collisions": len(obs.events.collisions)}


def _dist_to_obstacles(obs: Observation) -> Dict[str, float]:
    obstacle_dist_th = 50
    obstacle_angle_th = np.pi * 40 / 180
    w_dist = 0.05
    regexp_jn = re.compile(r":.*J")

    # Ego's position and heading with respect to the map's axes.
    # Note: All angles returned by smarts is with respect to the map's axes.
    #       On the map, angle is zero at positive y axis, and increases anti-clockwise.
    ego = obs.ego_vehicle_state
    ego_heading = (ego.heading + np.pi) % (2 * np.pi) - np.pi
    ego_pos = ego.position
    lane_ids = [wp.lane_id for path in obs.waypoint_paths for wp in path]
    lane_ids = set(lane_ids)
    ego_road_ids = [id.split("_")[0] for id in lane_ids]
    ego_road_ids = set(ego_road_ids)

    # Get neighbors.
    nghbs = obs.neighborhood_vehicle_states

    # Filter neighbors by road id.
    nghbs = [
        nghb
        for nghb in nghbs
        if (
            # Match neighbor and ego road id.
            nghb.road_id == ego.road_id
            # Match neighbor road id to ':.*J' pattern.
            or regexp_jn.search(nghb.road_id)
            # Match neighbor road id to any road id in ego path.
            or nghb.road_id in ego_road_ids
        )
    ]

    if len(nghbs) == 0:
        return {"dist_to_obstacles": 0}

    # Filter neighbors by distance.
    nghbs = [(nghb.position, np.linalg.norm(nghb.position - ego_pos)) for nghb in nghbs]
    nghbs = [nghb for nghb in nghbs if nghb[1] <= obstacle_dist_th]

    if len(nghbs) == 0:
        return {"dist_to_obstacles": 0}

    # Filter neighbors by angle.
    obstacles = []
    for pos, dist in nghbs:
        # Neighbors's angle with respect to the ego's position.
        # Note: In np.angle(), angle is zero at positive x axis, and increases anti-clockwise.
        #       Hence, map_angle = np.angle() - π/2
        rel_pos = pos - ego_pos
        obstacle_angle = np.angle(rel_pos[0] + 1j * rel_pos[1]) - np.pi / 2
        obstacle_angle = (obstacle_angle + np.pi) % (2 * np.pi) - np.pi
        # Obstacle heading is the angle correction required by ego agent to face the obstacle.
        obstacle_heading = obstacle_angle - ego_heading
        obstacle_heading = (obstacle_heading + np.pi) % (2 * np.pi) - np.pi
        if abs(obstacle_heading) <= obstacle_angle_th:
            obstacles.append((pos, dist, obstacle_heading))

    if len(obstacles) == 0:
        return {"dist_to_obstacles": 0}

    # J_D : Distance to obstacles cost
    _, di, _ = zip(*obstacles)
    di = np.array(di)
    j_d = np.amax(np.exp(-w_dist * di))

    return {"dist_to_obstacles": j_d}


def _jerk(obs: Observation) -> Dict[str, float]:
    w_jerk = [0.7, 0.3]

    lj_squared = np.sum(np.square(obs.ego_vehicle_state.linear_jerk))
    aj_squared = np.sum(np.square(obs.ego_vehicle_state.angular_jerk))

    jerk = np.dot(w_jerk, [lj_squared, aj_squared])

    return {"jerk": jerk}


def _lane_center_offset(obs: Observation) -> Dict[str, float]:
    # Nearest waypoints
    ego = obs.ego_vehicle_state
    waypoint_paths = obs.waypoint_paths
    wps = [path[0] for path in waypoint_paths]

    # Distance of vehicle from center of lane
    closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))
    signed_dist_from_center = closest_wp.signed_lateral_error(ego.position)
    lane_hwidth = closest_wp.lane_width * 0.5
    norm_dist_from_center = signed_dist_from_center / lane_hwidth

    # J_LC : Lane center offset
    jlc = np.float32(norm_dist_from_center ** 2)

    return {"lane_center_offset": jlc}


def _off_road(obs: Observation) -> Dict[str, int]:
    return {"off_road": obs.events.off_road}


def _off_route(obs: Observation) -> Dict[str, int]:
    return {"off_route": obs.events.off_route}


def _on_shoulder(obs: Observation) -> Dict[str, int]:
    return {"on_shoulder": int(obs.events.on_shoulder)}


def _steering_rate():
    _prev_steering = 0

    def func(obs: Observation) -> Dict[str, float]:
        nonlocal _prev_steering
        steering_velocity = (obs.ego_vehicle_state.steering - _prev_steering) / 0.1
        _prev_steering = obs.ego_vehicle_state.steering
        steering_velocity_squared = steering_velocity ** 2
        return {"steering_rate": steering_velocity_squared}

    return func


def _velocity_offset(obs: Observation) -> Dict[str, float]:
    # Nearest waypoints
    ego = obs.ego_vehicle_state
    waypoint_paths = obs.waypoint_paths
    wps = [path[0] for path in waypoint_paths]

    # Distance of vehicle from center of lane
    closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))
    speed_limit = closest_wp.speed_limit

    # Excess speed beyond speed limit
    excess = speed_limit - ego.speed
    excess = excess if excess > 0 else 0
    excess_squared = excess ** 2

    return {"velocity_offset": excess_squared}


def _wrong_way(obs: Observation) -> Dict[str, int]:
    return {"wrong_way": obs.events.wrong_way}


def _yaw_rate(obs: Observation) -> Dict[str, float]:
    yr_squared = obs.ego_vehicle_state.yaw_rate ** 2
    return {"yaw_rate": yr_squared}


# class _Overtake():
#     def __init__(self, agent_names):
#         self._traj = {name: [[],[]] for name in agent_names}

#     def reinit(self):
#         self._traj = {name: [[],[]] for name in self._overtake.keys()}

#     def __call__(self, obs: Observation, agent_name: str):
#         lane_index = obs.ego_vehicle_state.lane_index
#         lane_index = obs.ego_vehicle_state.lane_index

#         self._traj[agent_name].append(lane_index)

#     def check(agent_name: str):
#         overtake = 0

#         return {"overtake": overtake}
