from cmath import inf
from pathlib import Path
from typing import Any, Dict
from torch import ne

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

class Policy(BasePolicy):
    """A policy directly calculate action to the next waypoint."""

    def __init__(self):
        """All policy initialization matters, including loading of model, is
        performed here. To be implemented by the user.
        """
        import gym
        import numpy as np
        import torch

        covar = 1.0
        # self._pos_space = gym.spaces.Box(low=np.array([-covar, -covar]), high=np.array([covar, covar]), dtype=np.float32)
        self._pos_space = gym.spaces.Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        model_path = Path(__file__).absolute().parents[0] / "model_2022_10_31_15_12_27"
        self.model = torch.load(model_path)
        self.model.eval()

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
            action = self.get_next_goal_pos(agent_obs)
            action = np.array(
                        [action[0], action[1], action[2], time_delta], dtype=np.float32
                    )
            wrapped_act.update({agent_id: action})

        return wrapped_act

    def get_current_waypoint_path_index(self, agent_obs):
        ego_lane = agent_obs["ego"]["lane_index"]

        waypoints_lane_indices = agent_obs["waypoints"]["lane_index"]

        for index, path in enumerate(waypoints_lane_indices):
            last_waypoint_index = self.get_last_waypoint_index(agent_obs["waypoints"]["lane_width"][index])

            if last_waypoint_index:
                if ego_lane in path[:last_waypoint_index+1]:
                    return index
        
        #If no way points lane index match ego, return the closest waypoint lane
        import numpy as np
        waypoints_pos = [pos[0] for pos in agent_obs["waypoints"]["pos"]]
        vec_wps = [wp - agent_obs["ego"]["pos"] for wp in waypoints_pos]
        dist_wps = [np.linalg.norm(vec_wp) for vec_wp in vec_wps]
        return np.argmin(dist_wps)
        

    def get_next_waypoint(self, agent_obs, wps_path_index):
        # import numpy as np
        from smarts.core.utils.math import signed_dist_to_line  
        import numpy as np

        ego = agent_obs["ego"]
        ego_head = ego["heading"]

        wps = agent_obs["waypoints"]["pos"][wps_path_index]

        # Distance of vehicle from way points
        vec_wps = [wp - ego["pos"] for wp in wps]
        dist_wps = [np.linalg.norm(vec_wp) for vec_wp in vec_wps]
        # wp_index = np.argmin(dist_wps)
        # closest_wp = wps[wp_index]

        # Heading angle of each waypoints
        dir_wps = [np.array(vec_wps[i]) / (dist_wps[i]+0.00001) for i in range(len(vec_wps))]
        head_wps = np.array([np.arctan2(-dir_wp[0], dir_wp[1]) - ego_head for dir_wp in dir_wps])
        head_wps = (head_wps + np.pi) % (2 * np.pi) - np.pi 
        
        # Find the next way points given that the heading is smaller than 45 degree
        max_angle = 35 / 180 * np.pi
        switch_lane_max_angle = 20 / 180 * np.pi

        last_waypoint_index = self.get_last_waypoint_index(agent_obs["waypoints"]["lane_width"][wps_path_index])

        for i in range(last_waypoint_index+1):

            wp_heading = agent_obs["waypoints"]["heading"][wps_path_index][i]
            angle = (wp_heading + np.pi * 0.5) % (2 * np.pi)
            heading_dir_vec =  np.array((np.cos(angle), np.sin(angle)))
            signed_dist_from_center = signed_dist_to_line(ego["pos"][:2], wps[i][:2], heading_dir_vec)

            # Switching lane behavior
            if signed_dist_from_center > 0.5:
                if abs(head_wps[i]) <= switch_lane_max_angle:
                    return wps[i], i

            else:
                if dist_wps[i] > 0.5:
                    if abs(head_wps[i]) <= max_angle:
                        return wps[i], i

                
        
        return wps[last_waypoint_index], last_waypoint_index

    def get_next_goal_pos(self, agent_obs):
        import numpy as np

        current_path_index = self.get_current_waypoint_path_index(agent_obs)
        goal_path_index = self.get_cloest_path_index_to_goal(agent_obs)

        # If the goal path index is 2 lane away, we only switch 1 lane at a time
        if abs(goal_path_index - current_path_index) > 1:
            next_path_index = current_path_index + np.sign(goal_path_index - current_path_index)
        else:
            next_path_index = goal_path_index

        # Get the next closest waypoints on the next path we decided
        closest_wp, wp_index = self.get_next_waypoint(agent_obs=agent_obs, wps_path_index=next_path_index)

        # TODO: check whether this closest waypoint is feasible
        # 1. The furthest it can get within speed limit
        # 2. Any potential collision? 
        #         - whether the trajectory will cross other neighbor's trajectory
        #         - whether the next loaction maintain the safe distance of the other car
        # 3. If collision, then cut the travel distance to half, and check again, recursively till the speed ~= 0


        speed_limit = agent_obs["waypoints"]["speed_limit"][next_path_index][wp_index]
        next_goal_pos, next_goal_heading = self.get_next_limited_action(agent_obs["ego"]["pos"], closest_wp, speed_limit)


        action = [next_goal_pos[0], next_goal_pos[1], next_goal_heading]

        # action_samples = self.get_action_samples(1, action, agent_obs["ego"]["pos"])

        scores = self.get_safe_scores(agent_obs, [action], next_path_index)

        if scores[0,0] > 0.8:
            goal_dir = action[:2] - agent_obs["ego"]["pos"][:2]
            action = agent_obs["ego"]["pos"][:2] + 0.01 * goal_dir[:2]
            action = [action[0], action[1], next_goal_heading]

        
        return action 

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
        wps = agent_obs["waypoints"]
        last_waypoints = []
        for path in wps["lane_width"]:
            last_waypoints.append(self.get_last_waypoint_index(path))
        
        return last_waypoints

    def get_last_waypoint_index(self, waypoint_lane_width):
        import numpy as np
        s = np.flatnonzero(waypoint_lane_width > 0)
        if s.size != 0:
            return s[-1]
        else:
            return -1

    def get_cloest_path_index_to_goal(self, agent_obs):
        import numpy as np

        goal_pos = agent_obs["mission"]["goal_pos"]
        wps = agent_obs["waypoints"]
        wps_lane_width = wps["lane_width"]
        s = [ np.flatnonzero(wps_lane_width[i] > 0.1) for i in range(len(wps_lane_width))]
        last_waypoints_index = [s[i][-1] if np.any(s[i]) else -1 for i in range(len(s))]

        last_waypoints_pos = [wps["pos"][index][point_index] for index,point_index in enumerate(last_waypoints_index) if point_index >= 0 ]
        dist_to_goal = [np.linalg.norm(wp - goal_pos) for wp in last_waypoints_pos]
        path_index = np.argmin(dist_to_goal)

        return path_index

    def get_action_samples(self, n_samples, action, current_pos):
        import numpy as np
        goal_dir = action[:2] - current_pos[:2]

        samples = []
        for i in range(n_samples):
            prop = self._pos_space.sample()
            sample_action = current_pos[:2] + prop * goal_dir[:2]
            samples.append([sample_action[0], sample_action[1], action[2]])

        return np.array(samples)

    def get_safe_scores(self, agent_obs, actions, path_index):
        import torch
        import torchvision.transforms as transforms
        import numpy as np

        inputs = self.get_model_input(agent_obs, actions, path_index)
        n_samples = inputs.shape[0]
        imgs = torch.permute(torch.from_numpy(agent_obs["rgb"]), (2, 0, 1)).unsqueeze(0).repeat(n_samples, 1, 1, 1)
    
        with torch.no_grad():
            outputs = self.model(imgs, inputs.float())
        sm = torch.nn.Softmax()
        prob = sm(outputs) 

        return prob
        # safe_choice_prob = sm(prob[:, 5])
        # indices = [i for i in range(len(actions))]
        # final_action_index = np.random.choice(indices, 1, p=safe_choice_prob.detach().numpy())

        # if prob[final_action_index[0], 0] > 0.8:
        #     return [agent_obs["ego"]["pos"][0], agent_obs["ego"]["pos"][1], agent_obs["ego"]["heading"]]

        # else:
        #     return actions[final_action_index[0]]

        # return prob

    def get_model_input(self, agent_obs, actions, path_index):
        import numpy as np
        import torch
        waypoints = agent_obs["waypoints"]["pos"][path_index][:5]
        waypoints[:, -1] = agent_obs["waypoints"]["heading"][path_index][:5]
        ego_pos = agent_obs["ego"]["pos"]
        ego_pos[-1] = agent_obs["ego"]["heading"]

        inputs = []
        for action in actions:
            input = []
            input.extend(action[:3])
            input.extend(ego_pos)
            input.extend(waypoints.flatten())
            inputs.append(input)

        inputs = torch.from_numpy(np.array(inputs))

        return inputs