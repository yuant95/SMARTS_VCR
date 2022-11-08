from cmath import inf
from pathlib import Path
from tracemalloc import start
from typing import Any, Dict
import numpy as np

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
        # model_path = "/home/yuant426/Desktop/SMARTS_track1/competition/track1/classifier/logs/2022_11_02_22_53_39/model_step10_epoch60_2022_11_03_09_25_34"
        self.model = torch.load(model_path)
        self.model.eval()
        self.smoothed_waypoints = {}
        self.waypoints_length = 18
        self.act_length = 8
        self.current_goal_path = {}
        self.score_history = {}
        self.goal_path_history = {}
        self.hist_len = 10

    def reset(self):
        self.smoothed_waypoints = {}
        # self.current_goal_path = {}
        self.score_history = {}
        self.goal_path_history = {}

    def get_smoothed_waypoints(self):
        ret ={}
        for agent_id, waypoints in self.smoothed_waypoints.items():
            ret[agent_id] = list(waypoints.queue)

        return ret

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
            action = self.get_next_goal_pos(agent_obs, agent_id)
            
            action = np.array(
                        [action[0], action[1], action[2], time_delta], dtype=np.float32
                    )
            wrapped_act.update({agent_id: action})

        return wrapped_act

    def get_current_waypoint_path_index(self, agent_obs):
        ego_pos = agent_obs["ego"]["pos"]
        waypoints_pos = agent_obs["waypoints"]["pos"]
        waypoints_lane_indices = agent_obs["waypoints"]["lane_width"]

        min_dist = np.inf
        min_index = -1
        for i in range(len(waypoints_pos)):
            if waypoints_lane_indices[i][0] > 0.0:
                dist = np.linalg.norm(waypoints_pos[i][0] - ego_pos)
                if dist <= min_dist:
                    min_dist = dist
                    min_index = i

        if min_index < 0:
            print("no way points found for ego pos at {}".format(ego_pos))
            print("waypoints pos: {}".format(str(waypoints_pos)))
            min_index = 0

        return min_index
        
    def get_waypoint_index_range(self, agent_obs, wps_path_index):
        # import numpy as np
        from smarts.core.utils.math import signed_dist_to_line  
        import numpy as np

        ego = agent_obs["ego"]
        ego_head = ego["heading"]
        ego_pos = ego["pos"]

        wps = agent_obs["waypoints"]["pos"][wps_path_index]

        # Distance of vehicle from way points
        vec_wps = [wp - ego_pos for wp in wps]
        dist_wps = [np.linalg.norm(vec_wp) for vec_wp in vec_wps]
        # wp_index = np.argmin(dist_wps)
        # closest_wp = wps[wp_index]

        # Heading angle of each waypoints
        dir_wps = [np.array(vec_wps[i]) / (dist_wps[i]+0.00001) for i in range(len(vec_wps))]
        head_wps = np.array([np.arctan2(-dir_wp[0], dir_wp[1]) - ego_head for dir_wp in dir_wps])
        head_wps = (head_wps + np.pi) % (2 * np.pi) - np.pi 
        
        # Find the next way points given that the heading is smaller than 45 degree
        max_angle = 40 / 180 * np.pi
        switch_lane_max_angle = 40 / 180 * np.pi

        last_waypoint_index = self.get_last_waypoint_index(agent_obs["waypoints"]["lane_width"][wps_path_index])

        for i in range(last_waypoint_index+1):

            wp_heading = agent_obs["waypoints"]["heading"][wps_path_index][i]
            angle = (wp_heading + np.pi * 0.5) % (2 * np.pi)
            heading_dir_vec =  np.array((np.cos(angle), np.sin(angle)))
            signed_dist_from_center = signed_dist_to_line(ego["pos"][:2], wps[i][:2], heading_dir_vec)

            # Switching lane behavior
            if signed_dist_from_center > 0.5:
                if abs(head_wps[i]) <= switch_lane_max_angle:
                    return i, last_waypoint_index

            else:
                if dist_wps[i] > 0.2:
                    if abs(head_wps[i]) <= max_angle:
                        return i, last_waypoint_index
        
        return last_waypoint_index, last_waypoint_index

    def get_next_goal_pos(self, agent_obs, agent_id):
        import numpy as np
        import queue

        if agent_id in self.smoothed_waypoints and self.smoothed_waypoints[agent_id].qsize() > (self.waypoints_length - self.act_length):
            return self.smoothed_waypoints[agent_id].get()
        
        else:
            if agent_id not in self.goal_path_history:
                self.goal_path_history[agent_id] = queue.Queue()

            # if agent_id not in self.score_history:
            #     self.score_history[agent_id] = queue.Queue()

            current_path_index = self.get_current_waypoint_path_index(agent_obs)
            next_path_index = current_path_index
            # goal_path_index = self.get_cloest_path_index_to_goal(agent_obs)
            goal_path_index = self.sample_path_index(agent_obs, n_sample=1)[0]

            # if agent_id not in self.current_goal_path:
            #     self.current_goal_path[agent_id] = goal_path_index

            if agent_id not in self.score_history:
                self.score_history[agent_id] = queue.Queue()
            
            self.goal_path_history[agent_id].put(goal_path_index)
            if self.goal_path_history[agent_id].qsize() > self.hist_len:
                self.goal_path_history[agent_id].get()

                if next_path_index!= goal_path_index:
                    next_path_index = goal_path_index

            # Get the next closest waypoints on the next path we decided
            wp_index, wp_last_index = self.get_waypoint_index_range(agent_obs=agent_obs, wps_path_index=next_path_index)

            # get the next speed limit
            speed_limit = agent_obs["waypoints"]["speed_limit"][next_path_index][wp_index]

            # Now get the 

            # TODO: check whether this closest waypoint is feasible
            # 1. The furthest it can get within speed limit
            # 2. Any potential collision? 
            #         - whether the trajectory will cross other neighbor's trajectory
            #         - whether the next loaction maintain the safe distance of the other car
            # 3. If collision, then cut the travel distance to half, and check again, recursively till the speed ~= 0
            waypoints_pos =  agent_obs["waypoints"]["pos"][next_path_index][wp_index:wp_last_index+1, :2]
            goal_pos = agent_obs["mission"]["goal_pos"][:2]
            ego_pos = agent_obs["ego"]["pos"][:2]

            # if (wp_last_index+1 - wp_index) < self.waypoints_length or (np.linalg.norm(ego_pos-goal_pos) < 10):
            #     waypoints_pos = np.append(waypoints_pos, [agent_obs["mission"]["goal_pos"][:2]], axis=0)
            #     r = np.ones(len(waypoints_pos))
            #     r[-1] = self.waypoints_length - len(waypoints_pos) + 1
            #     waypoints_pos = np.repeat(waypoints_pos, r.astype(int), axis=0)
            if len(waypoints_pos) < 1:
                waypoints_pos = np.append(waypoints_pos, [agent_obs["mission"]["goal_pos"][:2]], axis=0)

            # sampled_speed = self.get_speed_samples(n_sample=1)
            sampled_speed = [1.0]
            planned_waypoints = get_smoothed_future_waypoints(waypoints=waypoints_pos, 
                start_pos=agent_obs["ego"]["pos"][:2], 
                n_points=self.waypoints_length,
                speed=sampled_speed[0])
            
            if np.linalg.norm(np.array(planned_waypoints[2]) - np.array(planned_waypoints[1])) > 1.4:
                print("something wrong")


            import queue
            self.smoothed_waypoints[agent_id] = queue.Queue()
            for p in planned_waypoints:
                self.smoothed_waypoints[agent_id].put(p)

            

            next_waypoint = self.smoothed_waypoints[agent_id].get()
            next_goal_pos, _ = self.get_next_limited_action(agent_obs["ego"]["pos"][:2], next_waypoint[:2], speed_limit)
            action = [next_goal_pos[0], next_goal_pos[1], next_waypoint[2]]
            if sampled_speed[0] == 0.0:
                action[2] = agent_obs["ego"]["heading"]
            # action_samples, props = self.get_action_samples(1, action, agent_obs["ego"]["pos"])
            # action = action_samples[0]
            # prop = props[0]

            # #  update future waypoints based on given action
            # agent_obs_copy = {}
            # agent_obs_copy["waypoints"] = agent_obs["waypoints"]
            # agent_obs_copy["ego"]={}
            # agent_obs_copy["ego"]["pos"] = [ action[0], action[1], 0 ]
            # agent_obs_copy["ego"]["heading"] = action[2]
            # wp_index, wp_last_index = self.get_waypoint_index_range(agent_obs=agent_obs_copy, wps_path_index=next_path_index)
            # waypoints_pos =  agent_obs["waypoints"]["pos"][next_path_index][wp_index:wp_last_index+1, :2]
            # if (wp_last_index - wp_index) < 10:
            #     r = np.ones(len(waypoints_pos))
            #     r[-1] = 10 - len(waypoints_pos) + 1
            #     waypoints_pos = np.repeat(waypoints_pos, r.astype(int), axis=0)

            # planned_points = self.get_smoothed_future_points(
            #     agent_obs=agent_obs, 
            #     action=action,
            #     next_path_index=next_path_index,
            #     n_points= 10,
            #     speed = prop)

            # import queue
            # self.smoothed_waypoints[agent_id] = queue.Queue()
            # for p in planned_points:
            #     self.smoothed_waypoints[agent_id].put(p)

        # scores = self.get_safe_scores(agent_obs, [action], next_path_index)

        # if scores[0,0] > 0.8:
        #     goal_dir = action[:2] - agent_obs["ego"]["pos"][:2]
        #     action = agent_obs["ego"]["pos"][:2] + 0.01 * goal_dir[:2]
        #     action = [action[0], action[1], next_waypoint_heading[0]]
        
            return action
    
    def get_smoothed_future_points(self, agent_obs, action, next_path_index, n_points=5, speed=1.0):
        agent_obs_copy = {}
        agent_obs_copy["waypoints"] = agent_obs["waypoints"]
        agent_obs_copy["ego"]={}
        agent_obs_copy["ego"]["pos"] = [ action[0], action[1], 0 ]
        agent_obs_copy["ego"]["heading"] = action[2]
        wp_index, wp_last_index = self.get_waypoint_index_range(agent_obs=agent_obs_copy, wps_path_index=next_path_index)
        waypoints_pos =  agent_obs["waypoints"]["pos"][next_path_index][wp_index:wp_last_index+1, :2]
        if (wp_last_index - wp_index) < 5:
            r = np.ones(len(waypoints_pos))
            r[-1] = 5 - len(waypoints_pos) + 1
            waypoints_pos = np.repeat(waypoints_pos, r.astype(int), axis=0)

        return get_smoothed_future_waypoints(waypoints=waypoints_pos, 
            start_pos=action[:2], 
            n_points=n_points,
            speed=speed)


    def get_next_limited_action(self, ego_pos, pos, speed_limit):
        import numpy as np

        time_delta = 0.1
        #Check whether going to next waypoint exceed the speed limit
        goal_vec = pos - ego_pos
        goal_dist = np.linalg.norm(goal_vec)

        if goal_dist == 0.0:
            return pos, 0.0

        goal_speed = goal_dist / time_delta
        goal_dir = goal_vec/ goal_dist

        if goal_speed > speed_limit:
            next_goal_pos = ego_pos + speed_limit * goal_dir * time_delta

        else: 
            next_goal_pos = ego_pos + goal_speed * goal_dir * time_delta
        
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
        try:
            goal_pos = agent_obs["mission"]["goal_pos"]
            wps = agent_obs["waypoints"]
            wps_pos = wps["pos"]
            wps_lane_width = wps["lane_width"]
            s = [ np.flatnonzero(wps_lane_width[i] > 0.1) for i in range(len(wps_lane_width))]
            last_waypoints_index = [s[i][-1] if np.any(s[i]) else -1 for i in range(len(s))]
            waypoint_path_index_candidate = [i for i in range(len(wps_pos)) if last_waypoints_index[i] > 0]

            last_waypoints_pos = [wps["pos"][index][point_index] for index,point_index in enumerate(last_waypoints_index) if point_index >= 0 ]
            dist_to_goal = [np.linalg.norm(wp - goal_pos) for wp in last_waypoints_pos]
            index = np.argmin(dist_to_goal)

            return waypoint_path_index_candidate[index]
        except:
            print("failed to find the cloest path index to goal.")
            print("Mission = {}".format(str(agent_obs["mission"])))
            print("Waypoints = {}".format(str(wps)))

            return 0

    def sample_path_index(self, agent_obs, n_sample=1):
        wps_lane_width = agent_obs["waypoints"]["lane_width"]
        s = [ np.flatnonzero(wps_lane_width[i] > 0.1) for i in range(len(wps_lane_width))]
        last_waypoints_index = [s[i][-1] if np.any(s[i]) else -1 for i in range(len(s))]
        waypoint_path_index_candidate = [i for i in range(len(wps_lane_width)) if last_waypoints_index[i] > 0]

        try:
            samples = np.random.choice(waypoint_path_index_candidate, n_sample, p=[1.0/float(len(waypoint_path_index_candidate)) for i in range(len(waypoint_path_index_candidate))])
        except:
            return [0]
        return samples

    def get_speed_samples(self, n_sample):
        # from scipy.stats import truncnorm
        # samples = 1 - truncnorm.rvs(0.0, 1.0, size=n_sample)
        import numpy as np
        #For generator only control speed 0.0, 1.0 and 1.2
        speeds = [0.0, 1.0, 1.2]
        samples = np.random.choice(speeds, n_sample, p=[0.3, 0.4, 0.3])

        return samples

    def get_action_samples(self, n_samples, action, current_pos):
        import numpy as np
        from scipy.stats import truncnorm
        goal_dir = action[:2] - current_pos[:2]

        samples = []
        props = [] 
        for i in range(n_samples):
            # prop = self._pos_space.sample()
            prop = 1 - truncnorm.rvs(0.0, 1.0, size=1)[0]
            sample_action = current_pos[:2] + prop * goal_dir[:2]
            samples.append([sample_action[0], sample_action[1], action[2]])
            props.append(prop)

        return np.array(samples), props

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


def get_spline(waypoints_pos, ego_pos):
    from scipy.interpolate import CubicSpline
    xy_swap = False
    x = [ego_pos[0]]
    y = [ego_pos[1]]
    x.extend([pos[0] for pos in waypoints_pos])
    y.extend([pos[1] for pos in waypoints_pos])
    try:
        cb = CubicSpline(x, y)
    except ValueError:
        cb = CubicSpline(y, x)
        xy_swap = True

    return cb, xy_swap

def get_t(spline, t, start_pos, xy_swapped) -> np.array :

    current_t = 0.0 
    if (xy_swapped):
        x = start_pos[1]
    else:
        x = start_pos[0]

    inc = 0.1
    current_point = np.array([x, spline(x)])
    while current_t < t: 
        prev_point = current_point
        x += inc 
        current_point = np.array([x, spline(x)])
        dist = np.linalg.norm(current_point - prev_point)
        current_t += dist
    
    if xy_swapped:
        return np.array([current_point[1], current_point[0]])
    else:
        return current_point

# def set_smoothed_future_waypoints(self, waypoints, start_pos, n_points, agent_id):
#     import bezier

#     wps = []
#     wps_heading = []
#     curve = get_bezier_curve(waypoints, start_pos)
#     for i in range(n_points):
#         wp = curve.evaluate(1.0*(i+1)/curve.length)
#         dir_wp = curve.evaluate_hodograph(1.0/curve.length)
#         heading = np.arctan2(-dir_wp[0][0], dir_wp[1][0])
#         heading = (heading + np.pi) % (2 * np.pi) - np.pi 
#         wps_heading.append(heading)
#         wps.append([wp[0][0], wp[1][0]])

#     self.smoothed_waypoints[agent_id] = [[wp[0], wp[1], heading] for wp, heading in zip(wps, wps_heading)]
    
def get_spline_direction(spline, pos, xy_swapped):
    if xy_swapped:
        g = spline(x=pos[1], nu=1)
        return [0, g]
    else:
        g = spline(x=pos[0], nu=1)
        return [1, g]

def get_bezier_curve(waypoints_pos, ego_pos):
    import bezier

    xy_swap = False
    if len(ego_pos)==2:
        x = [ego_pos[0]]
        y = [ego_pos[1]]
    else:
        x = []
        y = []
    x.extend([pos[0] for pos in waypoints_pos])
    y.extend([pos[1] for pos in waypoints_pos])

    nodes = np.asfortranarray([x, y])
    return bezier.Curve(nodes, degree=nodes.shape[1]-1)

def get_smoothed_future_waypoints(waypoints, start_pos, n_points, speed=1.0):
    import bezier
    import numpy as np

    wps = []
    curve = get_bezier_curve(waypoints, start_pos)
    for i in range(n_points):
        s = np.min([speed * (i+1) /curve.length, 1.0])
        wp = curve.evaluate(s)
        dir_wp = curve.evaluate_hodograph(s)
        heading = np.arctan2(-dir_wp[0][0], dir_wp[1][0])
        heading = (heading + np.pi) % (2 * np.pi) - np.pi 
        wps.append([wp[0][0], wp[1][0], heading])

    return wps
    

            