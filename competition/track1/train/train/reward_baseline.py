from typing import Any, Dict

import gym
import numpy as np
from wandb import agent
from smarts.core.utils.math import signed_dist_to_line


class Reward(gym.Wrapper):
    def __init__(self, env: gym.Env, weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]):
        super().__init__(env)
        # TODO: debug code below, need to move weights to parameters
        self.weights = np.array(weights)
        if len(self.weights) != 6:
            raise Exception("The reward weights must have length of 6 rather than {}".format(len(self.weights)))
        self.reward = None
        self.weighted_reward = None

    def step(self, action):
        """Adapts the wrapped environment's step.

        Note: Users should not directly call this method.
        """
        obs, reward, done, info = self.env.step(action)
        wrapped_reward = self._reward(obs, reward)
        wrapped_info = self._info(obs, reward, info)

        for agent_id, agent_done in done.items():
            if agent_id != "__all__" and agent_done == True:
                if obs[agent_id]["events"]["reached_goal"]:
                    print(f"{agent_id}: Hooray! Reached goal.")
                elif obs[agent_id]["events"]["reached_max_episode_steps"]:
                    print(f"{agent_id}: Reached max episode steps.")
                elif (
                    obs[agent_id]["events"]["collisions"]
                    | obs[agent_id]["events"]["off_road"]
                    | obs[agent_id]["events"]["off_route"]
                    | obs[agent_id]["events"]["on_shoulder"]
                    | obs[agent_id]["events"]["wrong_way"]
                ):
                    pass
                else:
                    print("Events: ", obs[agent_id]["events"])
                    raise Exception("Episode ended for unknown reason.")

        return obs, wrapped_reward, done, wrapped_info

    def _info(self, obs: Dict[str, Dict[str, Any]], env_reward: Dict[str, np.float64], info: Dict[Any, Any]
    ) -> Dict[str, np.float64]:
        
        if (self.reward == None).any() or (self.weighted_reward == None).any():
            raise Exception("Tried to access reward and weighted reward before initialization.")

        for agent_id, agent_reward in env_reward.items():
            info[agent_id]["rewards"] = {}
            info[agent_id]["rewards"]["complete"] = self.reward[0]
            info[agent_id]["rewards"]["humanness"] = self.reward[1]
            info[agent_id]["rewards"]["time"] = self.reward[2]
            info[agent_id]["rewards"]["rules"] = self.reward[3]
            info[agent_id]["rewards"]["goal"] = self.reward[4]
            info[agent_id]["rewards"]["distant"] = self.reward[5]
            info[agent_id]["rewards"]["weighted_complete"] = self.weighted_reward[0]
            info[agent_id]["rewards"]["weighted_humanness"] = self.weighted_reward[1]
            info[agent_id]["rewards"]["weighted_time"] = self.weighted_reward[2]
            info[agent_id]["rewards"]["weighted_rules"] = self.weighted_reward[3]
            info[agent_id]["rewards"]["weighted_goal"] = self.weighted_reward[4]
            info[agent_id]["rewards"]["weighted_distant"] = self.weighted_reward[5]
    
        return info


    def _reward(
        self, obs: Dict[str, Dict[str, Any]], env_reward: Dict[str, np.float64]
    ) -> Dict[str, np.float64]:
        reward = {agent_id: np.float64(0) for agent_id in env_reward.keys()}

        for agent_id, agent_reward in env_reward.items():

            # These are from the evaluation metrics
            complete = self._completion(obs[agent_id])
            humanness = self._humanness(obs[agent_id])
            time = self._time(obs[agent_id])
            rules = self._rules(obs[agent_id])
            goal = self._goal(obs[agent_id])

            self.reward = np.array([complete, humanness, time, rules, goal, np.float64(agent_reward)])
            self.weighted_reward = np.multiply(self.reward, self.weights)
            reward[agent_id] += np.sum(self.weighted_reward)

        return reward

    def _completion(
        self, agent_obs: Dict[str, Dict[str, Any]]
    ) -> np.float64:
        if agent_obs["events"]["collisions"]:
            print(f"Collided.")
            return - np.float64(50)
        return np.float64(0.0)

    def _humanness(
        self, agent_obs: Dict[str, Dict[str, Any]]
    ) -> np.float64:
        # return min(np.float64(10.0), max( (self._dist_to_obstacles(agent_obs)
        #     - self._jerk_angular(agent_obs)
        #     - self._jerk_linear(agent_obs)
        #     - self._lane_center_offset(agent_obs)), np.float64(-10.0))
        # )
        return self._dist_to_obstacles(agent_obs) - self._jerk_angular(agent_obs) - self._jerk_linear(agent_obs) - self._lane_center_offset(agent_obs)
    
    def _rules(
        self, agent_obs: Dict[str, Dict[str, Any]]
    ) -> np.float64:
        score = np.float64(0.0)
        if agent_obs["events"]["wrong_way"]:
            print(f"Wrong way.")
            score -= np.float64(10)

        score -= self._speed_limit(agent_obs)

        # return min(np.float64(10.0), max(score, np.float64(-10.0))) 
        return score

    def _time(self, agent_obs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, np.float64]:

        # TODO: how to penalize the length of steps taken? (counts.steps_adjusted )
        # NOTE: This doesn't seem to make sense during training.
        # return min(np.float64(10.0), max(-self._dist_to_goal(agent_obs), np.float64(-10.0))) 
        # return -self._dist_to_goal(agent_obs)
        dist_to_goal = self._dist_to_goal(agent_obs)

        return np.exp(-abs(dist_to_goal)/ 1000)

    def _goal(self, agent_obs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, np.float64]:
        r = 0.0
        # Penalty for driving off road
        if agent_obs["events"]["off_road"]:
           print(f"Off road.")
           r -= np.float64(50)

        # Penalty for driving off route
        if agent_obs["events"]["off_route"]:
            print(f"Off route.")
            r -= np.float64(50)

        # Penalty for driving on road shoulder
        if agent_obs["events"]["on_shoulder"]:
            print(f"On shoulder")
            r -= np.float64(2)

        # Penalty for reach max episode steps
        if agent_obs["events"]["reached_max_episode_steps"]:
            r -= np.float64(0.5)
        else:
            r += np.float64(0.5)

        if agent_obs["events"]["not_moving"]:
            r -= np.float64(0.5)

        # Reward for reaching goal
        if agent_obs["events"]["reached_goal"]:
            r += np.float64(30)

        return np.float64(r)
        # return min(np.float64(10.0), max(r, np.float64(-10.0))) 

    def _dist_to_goal(self, obs: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        mission = obs["mission"]
        if "goal_pos" in mission:
            goal_position = mission["goal_pos"]
            rel = obs["ego"]["pos"][:2] - goal_position[:2]
            dist = sum(abs(rel))
        else:
            dist = 0

        return np.float64(dist)

    def _dist_to_obstacles(self, agent_obs: Dict[str, Dict[str, Any]]) -> np.float64:
        rel_angle_th = np.pi * 40 / 180
        rel_heading_th = np.pi * 179 / 180
        w_dist = 0.05

        # Ego's position and heading with respect to the map's coordinate system.
        # Note: All angles returned by smarts is with respect to the map's coordinate system.
        #       On the map, angle is zero at positive y axis, and increases anti-clockwise.
        ego = agent_obs["ego"]
        ego_heading = (ego["heading"] + np.pi) % (2 * np.pi) - np.pi
        ego_pos = ego["pos"]

        # Set obstacle distance threshold using 3-second rule
        obstacle_dist_th = ego["speed"] * 3
        if obstacle_dist_th == 0:
            return np.float64(0.0)

        # Get neighbors.
        nghbs = agent_obs["neighbors"]

        # Filter neighbors by distance.
        nghbs_state = [
            (nghb_idx, np.linalg.norm(nghbs["pos"][nghb_idx] - ego_pos)) for nghb_idx in range(len(nghbs["pos"]))
        ]
        nghbs_state = [
            nghb_state
            for nghb_state in nghbs_state
            if nghb_state[1] <= obstacle_dist_th
        ]
        if len(nghbs_state) == 0:
            return np.float64(0.0)

        # Filter neighbors within ego's visual field.
        obstacles = []
        for nghb_state in nghbs_state:
            # Neighbors's angle with respect to the ego's position.
            # Note: In np.angle(), angle is zero at positive x axis, and increases anti-clockwise.
            #       Hence, map_angle = np.angle() - π/2
            nghb_idx = nghb_state[0]
            rel_pos = np.array(nghbs["pos"][nghb_idx]) - ego_pos
            obstacle_angle = np.angle(rel_pos[0] + 1j * rel_pos[1]) - np.pi / 2
            obstacle_angle = (obstacle_angle + np.pi) % (2 * np.pi) - np.pi
            # Relative angle is the angle correction required by ego agent to face the obstacle.
            rel_angle = obstacle_angle - ego_heading
            rel_angle = (rel_angle + np.pi) % (2 * np.pi) - np.pi
            if abs(rel_angle) <= rel_angle_th:
                obstacles.append(nghb_state)
        nghbs_state = obstacles
        if len(nghbs_state) == 0:
            return np.float64(0.0)

        # Filter neighbors by their relative heading to that of ego's heading.
        nghbs_state = [
            nghb_state
            for nghb_state in nghbs_state
            #TODO: check whether we need clip here
            if abs(nghbs["heading"][nghb_state[0]] - (ego["heading"])) <= rel_heading_th
        ]
        if len(nghbs_state) == 0:
            return np.float64(0.0)

        # J_D : Distance to obstacles cost
        di = [nghb_state[1] for nghb_state in nghbs_state]
        di = np.array(di)
        j_d = np.amax(np.exp(-w_dist * di))

        return np.float64(j_d)

    def _jerk_angular(self, agent_obs: Dict[str, Dict[str, Any]]) -> np.float64:
        ja_squared = np.sum(np.square(agent_obs["ego"]["angular_jerk"]))
        return np.float64(ja_squared)

    def _jerk_linear(self, agent_obs: Dict[str, Dict[str, Any]]) -> np.float64:
        jl_squared = np.sum(np.square(agent_obs["ego"]["linear_jerk"]))

        return np.float64(jl_squared)

    def _lane_center_offset(self, agent_obs: Dict[str, Dict[str, Any]]) -> np.float64:

        # Nearest waypoints
        ego = agent_obs["ego"]
        waypoint_paths = agent_obs["waypoints"]
        # wps = [path[0] for path in waypoint_paths]
        wps = waypoint_paths["pos"][0]

        # Distance of vehicle from center of lane
        dist_wps = [np.linalg.norm(wp - ego["pos"]) for wp in wps]
        wp_index = np.argmin(dist_wps)
        closest_wp = wps[wp_index]

        # signed_dist_from_center = closest_wp.signed_lateral_error(ego.position)
        wp_heading = waypoint_paths["heading"][0][wp_index]
        angle = (wp_heading + np.pi * 0.5) % (2 * np.pi)
        heading_dir_vec =  np.array((np.cos(angle), np.sin(angle)))
        signed_dist_from_center = signed_dist_to_line(ego["pos"][:2], closest_wp[:2], heading_dir_vec)

        lane_width = waypoint_paths["lane_width"][0][wp_index] 
        lane_hwidth = lane_width * 0.5
        norm_dist_from_center = signed_dist_from_center / lane_hwidth

        # J_LC : Lane center offset
        j_lc = norm_dist_from_center**2

        return np.float64(j_lc)

    def _off_road(self, agent_obs: Dict[str, Dict[str, Any]]) -> np.float64:
        if agent_obs["events"]["off_road"]:
            return -np.float64(10)
        return np.float64(0.0)

    def _speed_limit(self, agent_obs: Dict[str, Dict[str, Any]]) -> np.float64:
        # Nearest waypoints.
        ego = agent_obs["ego"]
        waypoint_paths = agent_obs["waypoints"]
        # wps = [path[0] for path in waypoint_paths]
        wps = waypoint_paths["pos"][0]

        # Speed limit.
        dist_wps = [np.linalg.norm(wp - ego["pos"]) for wp in wps]
        wp_index = np.argmin(dist_wps)
        speed_limit = waypoint_paths["speed_limit"][0][wp_index]

        # Excess speed beyond speed limit.
        overspeed = ego["speed"] - speed_limit if ego["speed"] > speed_limit else 0
        j_v = overspeed**2

        return np.float64(j_v)