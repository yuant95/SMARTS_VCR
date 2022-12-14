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
        # lambda env: HistoryStack(env=env, num_stack=5),
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

        self._pos_space = gym.spaces.Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        model_path = Path(__file__).absolute().parents[0] / "model_step10_epoch_15_2022_11_06_11_31_42"
        self.model = torch.load(model_path)
        self.model.eval()
        self.smoothed_waypoints = {}
        self.waypoints_length = 18
        self.act_length = 8
        self.hist_len = 10
        self.score_history = {}
        self.goal_path_history = {}
        self.axis_priority = -1

    def reset(self):
        self.smoothed_waypoints = {}

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
            try:
                action = self.get_next_goal_pos(agent_obs, agent_id)
            except Exception as e:
                print("Exception {}".format(str(e)))
                print("Using current pos as action. ")
                action = agent_obs["ego"]["pos"]
                action[2] = agent_obs["ego"]["heading"]
            
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

        # Heading angle of each waypoints
        dir_wps = [np.array(vec_wps[i]) / (dist_wps[i]+0.00001) for i in range(len(vec_wps))]
        head_wps = np.array([np.arctan2(-dir_wp[0], dir_wp[1]) - ego_head for dir_wp in dir_wps])
        head_wps = (head_wps + np.pi) % (2 * np.pi) - np.pi 
        
        # Find the next way points given that the heading is smaller than 35 degree
        max_angle = 35 / 180 * np.pi
        switch_lane_max_angle = 35 / 180 * np.pi

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

        # Only change lane every 5 steps. 
        current_path_index = self.get_current_waypoint_path_index(agent_obs)
        next_path_index = current_path_index
        goal_path_index = self.get_cloest_path_index_to_goal(agent_obs)

        if agent_id not in self.goal_path_history:
            self.goal_path_history[agent_id] = queue.Queue()

        if agent_id not in self.score_history:
            self.score_history[agent_id] = queue.Queue()
        
        self.goal_path_history[agent_id].put(goal_path_index)
        if self.goal_path_history[agent_id].qsize() > self.hist_len:
            self.goal_path_history[agent_id].get()

        if list(self.goal_path_history[agent_id].queue).count(self.goal_path_history[agent_id].queue[-1]) == self.hist_len:
            if current_path_index!= self.goal_path_history[agent_id].queue[-1]:
                next_path_index = self.goal_path_history[agent_id].queue[-1]
        
        wp_index, wp_last_index = self.get_waypoint_index_range(agent_obs=agent_obs, wps_path_index=next_path_index)

        # get the next speed limit
        speed_limit = agent_obs["waypoints"]["speed_limit"][next_path_index][wp_index]

        try:
            waypoints_pos =  agent_obs["waypoints"]["pos"][next_path_index][wp_index:wp_last_index+1, :2]
            waypoints_heading = agent_obs["waypoints"]["heading"][next_path_index][wp_index:wp_last_index+1]
        except:
            waypoints_pos = np.array([]) 
            waypoints_heading = np.array([])

        goal_pos = agent_obs["mission"]["goal_pos"][:2]
        ego_pos = agent_obs["ego"]["pos"][:2]

        planned_path = []
        if len(waypoints_pos) <= 1:
            waypoints_pos = np.append(waypoints_pos, [agent_obs["mission"]["goal_pos"][:2]], axis=0)
        
            # Regenerate the waypoints, if the dist between ego and mission is so far
            if np.linalg.norm(goal_pos - ego_pos) > 10:

                if self.axis_priority < 0:
                    if len(waypoints_pos) > 1:
                        current_heading = waypoints_heading[0]
                    else:
                        current_heading = agent_obs["ego"]["heading"]

                    heading_epslion = np.pi * 45 / 180.0

                    rel_y_pos_angle = current_heading - np.pi * 0.0
                    rel_y_neg_angle = current_heading - np.pi * 180.0 / 180.0
                    rel_x_pos_angle = current_heading - np.pi * 90.0 / 180.0
                    rel_x_neg_angle = current_heading - np.pi * -90.0 / 180.0

                    rel_y_pos_angle = (rel_y_pos_angle + np.pi) % (2 * np.pi) - np.pi
                    rel_y_neg_angle = (rel_y_neg_angle + np.pi) % (2 * np.pi) - np.pi
                    rel_x_pos_angle = (rel_x_pos_angle + np.pi) % (2 * np.pi) - np.pi
                    rel_x_neg_angle = (rel_x_neg_angle + np.pi) % (2 * np.pi) - np.pi

                    if abs(rel_y_pos_angle) <= heading_epslion or abs(rel_y_neg_angle) <= heading_epslion:
                        #along y axis first
                        self.axis_priority = 1

                    elif abs(rel_x_pos_angle) <= heading_epslion or abs(rel_x_neg_angle) <= heading_epslion:
                        self.axis_priority = 0

                planned_path = get_resampled_waypoints_pos(waypoints=waypoints_pos, start_pos=ego_pos, axis=self.axis_priority, n_points=self.act_length, speed=1.0)

        if not planned_path:
            planned_path = get_smoothed_future_waypoints(waypoints=waypoints_pos, 
                start_pos=agent_obs["ego"]["pos"][:2], 
                n_points=self.waypoints_length,
                speed=1.0)

        action = planned_path[0]
        scores = self.get_safe_scores(agent_obs, [action], [planned_path])
        self.score_history[agent_id].put(scores[0].numpy())
        past_scores_size = self.score_history[agent_id].qsize()
        if past_scores_size > self.hist_len:
            self.score_history[agent_id].get()

        past_scores_size = self.score_history[agent_id].qsize()

        weights = np.array([np.power(0.6, past_scores_size - i) for i in range(past_scores_size)])
        average_scores = np.average(np.array(list(self.score_history[agent_id].queue)), weights = weights, axis=0)

        # if average_scores[0] > 0.3:
        accelerate = self.accelerate(agent_obs=agent_obs)
        if accelerate < 0:
            prob_collision = average_scores[0]
            speed = 1/(1 + np.exp(-(1-prob_collision)*20 +14))
        else:
            speed = accelerate

        goal_dir = action[:2] - agent_obs["ego"]["pos"][:2]
        action[:2] = agent_obs["ego"]["pos"][:2] + speed * goal_dir[:2]

        return action

    def accelerate(self, agent_obs):
        obstacles = self.get_obstacles(agent_obs=agent_obs)
        if len(obstacles) == 0:
            return 1.1
        else:
            dist = np.array([state[1] for state in obstacles])
            front_obs = np.count_nonzero(dist > 0)
            back_obs = np.count_nonzero(dist <= 0)

            if front_obs:
                return -1
            elif back_obs:
                if np.count_nonzero(dist > -5):
                    return 2.5
                else:
                    return 1.1
            else: 
                return -1
    
    def get_obstacles(self, agent_obs: Dict[str, Dict[str, Any]]):
        rel_angle_th = np.pi * 60 / 180
        rel_heading_th = np.pi * 175 / 180

        # Ego's position and heading with respect to the map's coordinate system.
        # Note: All angles returned by smarts is with respect to the map's coordinate system.
        #       On the map, angle is zero at positive y axis, and increases anti-clockwise.
        ego = agent_obs["ego"]
        ego_heading = (ego["heading"] + np.pi) % (2 * np.pi) - np.pi
        ego_pos = ego["pos"]

        # Set obstacle distance threshold using 1-second rule
        obstacle_dist_th = 12 * 1.0

        # Get neighbors.
        nghbs = agent_obs["neighbors"]

        # Filter neighbors by distance.
        nghbs_state = [
            [nghb_idx, np.linalg.norm(nghbs["pos"][nghb_idx] - ego_pos)] for nghb_idx in range(len(nghbs["pos"]))
        ]
        nghbs_state = [
            nghb_state
            for nghb_state in nghbs_state
            if nghb_state[1] <= obstacle_dist_th
        ]
        if len(nghbs_state) == 0:
            return nghbs_state

        # Filter neighbors within ego's visual field.
        obstacles = []
        for nghb_state in nghbs_state:
            # Neighbors's angle with respect to the ego's position.
            # Note: In np.angle(), angle is zero at positive x axis, and increases anti-clockwise.
            #       Hence, map_angle = np.angle() - ??/2
            nghb_idx = nghb_state[0]
            rel_pos = np.array(nghbs["pos"][nghb_idx]) - ego_pos
            obstacle_angle = np.angle(rel_pos[0] + 1j * rel_pos[1]) - np.pi / 2
            obstacle_angle = (obstacle_angle + np.pi) % (2 * np.pi) - np.pi
            # Relative angle is the angle correction required by ego agent to face the obstacle.
            rel_angle = obstacle_angle - ego_heading
            rel_angle = (rel_angle + np.pi) % (2 * np.pi) - np.pi
            if abs(rel_angle) <= rel_angle_th:
                obstacles.append(nghb_state)
            if abs(rel_angle) >= np.pi - rel_angle_th:
                nghb_state[1] = -nghb_state[1]
                obstacles.append(nghb_state)

        nghbs_state = obstacles
        if len(nghbs_state) == 0:
            return nghbs_state

        # Filter neighbors by their relative heading to that of ego's heading.
        nghbs_state = [
            nghb_state
            for nghb_state in nghbs_state
            #TODO: check whether we need clip here
            if abs(nghbs["heading"][nghb_state[0]] - (ego["heading"])) <= rel_heading_th
        ]

        return nghbs_state
        
    
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
            last_waypoints_index = [s[i][-1] if s[i].size else -1 for i in range(len(s))]
            waypoint_path_index_candidate = [i for i in range(len(wps_pos)) if last_waypoints_index[i] >= 0]

            last_waypoints_pos = [wps["pos"][index][point_index] for index,point_index in enumerate(last_waypoints_index) if point_index >= 0 ]
            dist_to_goal = [np.linalg.norm(wp - goal_pos) for wp in last_waypoints_pos]
            index = np.argmin(dist_to_goal)

            return waypoint_path_index_candidate[index]
        except Exception as e:
            print("failed to find the cloest path index to goal due to {}".format(str(e)))
            print("Mission = {}".format(str(agent_obs["mission"])))
            print("Waypoints = {}".format(str(wps)))

            return 0

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

    def get_safe_scores(self, agent_obs, actions, sampled_waypoints):
        import torch
        import torchvision.transforms as transforms
        import numpy as np

        inputs = self.get_model_input(agent_obs, actions, sampled_waypoints)
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

    def get_model_input(self, agent_obs, actions, sampled_waypoints):
        import numpy as np
        import torch
        ego_pos = agent_obs["ego"]["pos"]
        ego_pos[-1] = agent_obs["ego"]["heading"]

        inputs = []
        for i in range(len(actions)):
            input = []
            input.extend(actions[i][:3])
            input.extend(ego_pos)
            input.extend(np.array(sampled_waypoints[i][:5]).flatten())
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
    

def get_resampled_waypoints_pos(waypoints, start_pos, axis, n_points, speed = 1.0):
    if len(waypoints) > 0:
        final_pos = waypoints[-1]
    else:
        raise Exception("Empty waypoints recieved from get_resampled_waypoints_pos()")
    
    ret_waypoints = []
    if axis == 0:
        # x first
        x_inc = final_pos[0] - start_pos[0]
        x_sign = np.sign(x_inc)
        current_pos = start_pos.copy()
        while (abs(final_pos[0] - current_pos[0]) > 3.0 * speed) and (len(ret_waypoints) < n_points):
            current_pos[0] += x_sign * speed
            ret_waypoints.append([current_pos[0], current_pos[1]])

    
        if len(ret_waypoints) < n_points:
            y_inc = final_pos[1] - current_pos[1]
            y_sign = np.sign(y_inc)
            current_pos[1] += y_sign * 3 * speed
            while (abs(final_pos[1] - current_pos[1]) > speed) and (len(ret_waypoints) < n_points):
                ret_waypoints.append([final_pos[0], current_pos[1]])
                current_pos[1] += y_sign * speed
                

    elif axis == 1:
        # y first
        y_inc = final_pos[1] - start_pos[1]
        y_sign = np.sign(y_inc)
        current_pos = start_pos.copy()
        while (abs(final_pos[1] - current_pos[1]) > 3.0 * speed) and (len(ret_waypoints) < n_points):
            current_pos[1] += y_sign * speed
            ret_waypoints.append([current_pos[0], current_pos[1]])

    
        if len(ret_waypoints) < n_points:
            x_inc = final_pos[0] - current_pos[0]
            x_sign = np.sign(x_inc)
            current_pos[0] += x_sign * 3.0 * speed
            while (abs(final_pos[0] - current_pos[0]) > speed) and (len(ret_waypoints) < n_points):
                ret_waypoints.append([current_pos[0], final_pos[1]])
                current_pos[0] += x_sign * speed

    
    return get_smoothed_future_waypoints(waypoints=ret_waypoints, start_pos=start_pos, n_points=n_points, speed=speed)




