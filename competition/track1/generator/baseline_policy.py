from cmath import inf
from pathlib import Path
from typing import Any, Dict

from wandb import agent

class BasePolicy:
    def act(self, obs: Dict[str, Any]):
        """Act function to be implemented by user.

        Args:
            obs (Dict[str, Any]): A dictionary of observation for each ego agent step.

        Returns:
            Dict[str, Any]: A dictionary of actions for each ego agent.
        """
        raise NotImplementedError


def submitted_wrappers():
    """Return environment wrappers for wrapping the evaluation environment.
    Each wrapper is of the form: Callable[[env], env]. Use of wrappers is
    optional. If wrappers are not used, return empty list [].

    Returns:
        List[wrappers]: List of wrappers. Default is empty list [].
    """

    from action import Action as DiscreteAction
    from observation import Concatenate, FilterObs, SaveObs

    from smarts.core.controllers import ActionSpaceType
    from smarts.env.wrappers.format_action import FormatAction
    from smarts.env.wrappers.format_obs import FormatObs
    from smarts.env.wrappers.frame_stack import FrameStack

    # fmt: off
    wrappers = [
        FormatObs,
        lambda env: FormatAction(env=env, space=ActionSpaceType["TargetPose"]),
        # SaveObs,
        # DiscreteAction,
        # FilterObs,
        # lambda env: FrameStack(env=env, num_stack=3),
        # lambda env: Concatenate(env=env, channels_order="first"),
    ]
    # fmt: on

    return wrappers


class TrainPolicy(BasePolicy):
    """Policy class to be submitted by the user. This class will be loaded
    and tested during evaluation."""

    def __init__(self):
        """All policy initialization matters, including loading of model, is
        performed here. To be implemented by the user.
        """

        import stable_baselines3 as sb3lib
        import network

        model_path = Path(__file__).absolute().parents[0] / "best_model.zip"
        self.model = sb3lib.PPO.load(model_path)

    def act(self, obs: Dict[str, Any]):
        """Act function to be implemented by user.

        Args:
            obs (Dict[str, Any]): A dictionary of observation for each ego agent step.

        Returns:
            Dict[str, Any]: A dictionary of actions for each ego agent.
        """
        wrapped_act = {}
        for agent_id, agent_obs in obs.items():
            action, _ = self.model.predict(observation=agent_obs, deterministic=True)
            wrapped_act.update({agent_id: action})

        return wrapped_act


class RandomPolicy(BasePolicy):
    """A sample policy with random actions. Note that only the class named `Policy`
    will be tested during evaluation."""

    def __init__(self):
        """All policy initialization matters, including loading of model, is
        performed here. To be implemented by the user.
        """
        import gym

        self._action_space = gym.spaces.Discrete(4)

    def act(self, obs: Dict[str, Any]):
        """Act function to be implemented by user.

        Args:
            obs (Dict[str, Any]): A dictionary of observation for each ego agent step.

        Returns:
            Dict[str, Any]: A dictionary of actions for each ego agent.
        """
        wrapped_act = {}
        for agent_id, agent_obs in obs.items():
            action = self._action_space.sample()
            wrapped_act.update({agent_id: action})

        return wrapped_act

class Policy(BasePolicy):
    """A policy directly calculate action to the next waypoint."""

    def __init__(self):
        """All policy initialization matters, including loading of model, is
        performed here. To be implemented by the user.
        """
        import gym
        import numpy as np

        covar = 1.0
        # self._pos_space = gym.spaces.Box(low=np.array([-covar, -covar]), high=np.array([covar, covar]), dtype=np.float32)
        self._pos_space = gym.spaces.Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)

    def act(self, obs: Dict[str, Any]):
        """Act function to be implemented by user.

        Args:
            obs (Dict[str, Any]): A dictionary of observation for each ego agent step.

        Returns:
            Dict[str, Any]: A dictionary of actions for each ego agent.
        """
        import numpy as np
        time_delta = 0.1
        wrapped_act = {}
        for agent_id, agent_obs in obs.items():
            # action = self._action_space.sample()
            # wrapped_act.update({agent_id: action})
            next_pose, next_heading = self.get_next_goal_pos(agent_obs)
            action = np.array(
                        [next_pose[0], next_pose[1], next_heading, time_delta], dtype=np.float32
                    )
            wrapped_act.update({agent_id: action})

        return wrapped_act

    def get_closest_waypoint(self, agent_obs):
        # import numpy as np
        from smarts.core.utils.math import signed_dist_to_line  
        import numpy as np

        ego = agent_obs["ego"]
        waypoint_paths = agent_obs["waypoints"]
        # wps = [path[0] for path in waypoint_paths]
        wps = waypoint_paths["pos"][0]

        # Distance of vehicle from center of lane
        dist_wps = [np.linalg.norm(wp - ego["pos"]) for wp in wps]
        wp_index = np.argmin(dist_wps)
        closest_wp = wps[wp_index]

        wp_heading = waypoint_paths["heading"][0][wp_index]
        angle = (wp_heading + np.pi * 0.5) % (2 * np.pi)
        heading_dir_vec =  np.array((np.cos(angle), np.sin(angle)))
        signed_dist_from_center = signed_dist_to_line(ego["pos"][:2], closest_wp[:2], heading_dir_vec)

        if signed_dist_from_center > 0.5:
            return closest_wp, wp_index
        elif (wp_index + 1) < len(wps):
            return wps[wp_index+1], wp_index+1
        else:
            raise Exception("Running out of way points.")

    def get_next_goal_pos(self, agent_obs):
        # import numpy as np
        closest_wp, wp_index = self.get_closest_waypoint(agent_obs=agent_obs)
        speed_limit = agent_obs["waypoints"]["speed_limit"][0][wp_index]
        next_goal_pos, next_goal_heading = self.get_next_limited_action(agent_obs["ego"]["pos"], closest_wp, speed_limit)
        
        return next_goal_pos, next_goal_heading 

    def get_next_limited_action(self, ego_pos, pos, speed_limit):
        import numpy as np

        time_delta = 0.1
        #Check whether going to next waypoint exceed the speed limit
        goal_vec = pos - ego_pos
        goal_dist = np.linalg.norm(goal_vec)
        goal_speed = goal_dist / time_delta
        goal_dir = goal_vec/ goal_dist

        #Sample the distance
        # prop = self._pos_space.sample()
        prop = 1.0

        if goal_speed > speed_limit:
            next_goal_pos = ego_pos + speed_limit * goal_dir * time_delta * prop

        else: 
            next_goal_pos = ego_pos + goal_speed * goal_dir * time_delta * prop
        
        next_goal_heading = np.arctan2(-goal_dir[0], goal_dir[1])
        next_goal_heading = (next_goal_heading + np.pi) % (2 * np.pi) - np.pi
        

        return next_goal_pos, next_goal_heading

    def get_each_lane_last_waypoint(self, agent_obs):
        import numpy as np

        wps = agent_obs["waypoints"]
        last_waypoints = []
        for path in wps["lane_width"]:
            s = np.flatnonzero(path > 0)
            if s.size != 0:
                last_waypoints.append(s[-1])
            else:
                last_waypoints.append(-1)

        
        return last_waypoints

    def get_cloest_path_index_to_goal(self, agent_obs, last_waypoints):
        import numpy as np

        goal_pos = agent_obs["mission"]["goal_pos"]
        wps = agent_obs["waypoints"]

        last_waypoints_pos = [wps["pos"][index][point_index] for index,point_index in enumerate(last_waypoints) if point_index != -1]
        dist_to_goal = [np.linalg.norm(wp - goal_pos) for wp in last_waypoints_pos]
        path_index = np.argmin(dist_to_goal)

        return path_index





            