from typing import Any, Callable, Dict, Tuple

import gym
import numpy as np


class Discrete_Action_4(gym.ActionWrapper):
    """Modifies the action space."""

    def __init__(self, env: gym.Env):
        """Sets identical action space, denoted by `space`, for all agents.

        Args:
            env (gym.Env): Gym env to be wrapped.
        """
        super().__init__(env)
        self._wrapper, action_space = _discrete()

        self.action_space = gym.spaces.Dict(
            {agent_id: action_space for agent_id in env.action_space.spaces.keys()}
        )

    def action(self, action):
        """Adapts the action input to the wrapped environment.

        `self.saved_obs` is retrieved from SaveObs wrapper. It contains previously
        saved observation parameters.

        Note: Users should not directly call this method.
        """
        wrapped_act = self._wrapper(action=action, saved_obs=self.saved_obs)

        return wrapped_act

def _discrete() -> Tuple[Callable[[Dict[str, int]], Dict[str, np.ndarray]], gym.Space]:
    space = gym.spaces.Discrete(n=4)

    time_delta = 0.1  # Time, in seconds, between steps.
    angle = 15 / 180 * np.pi  # Turning angle in radians
    speed = 50  # Speed in km/h
    dist = (
        speed * 1000 / 3600 * time_delta
    )  # Distance, in meter, travelled in time_delta seconds

    action_map = {
        # key: [magnitude, angle]
        0: [0, 0],  # slow_down
        1: [dist, 0],  # keep_direction
        2: [dist, angle],  # turn_left
        3: [dist, -angle],  # turn_right
    }

    def wrapper(
        action: Dict[str, int], saved_obs: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        wrapped_obs = {}
        for agent_id, agent_action in action.items():
            new_heading = saved_obs[agent_id]["heading"] + action_map[agent_action][1]
            new_heading = (new_heading + np.pi) % (2 * np.pi) - np.pi

            magnitude = action_map[agent_action][0]
            cur_coord = (
                saved_obs[agent_id]["pos"][0] + 1j * saved_obs[agent_id]["pos"][1]
            )
            # Note: On the map, angle is zero at positive y axis, and increases anti-clockwise.
            #       In np.exp(), angle is zero at positive x axis, and increases anti-clockwise.
            #       Hence, numpy_angle = map_angle + π/2
            new_pos = cur_coord + magnitude * np.exp(1j * (new_heading + np.pi / 2))
            x_coord = np.real(new_pos)
            y_coord = np.imag(new_pos)

            wrapped_obs.update(
                {
                    agent_id: np.array(
                        [x_coord, y_coord, new_heading, time_delta], dtype=np.float32
                    )
                }
            )

        return wrapped_obs

    return wrapper, space

class Continuous_Action(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._wrapper, action_space = _continuous()

        self.action_space = gym.spaces.Dict(
            {agent_id: action_space for agent_id in env.action_space.spaces.keys()}
        )

    def action(self, action):

        wrapped_act = self._wrapper(action=action, saved_obs=self.saved_obs)
        return wrapped_act


# def _continuous() -> Tuple[Callable[[np.array], np.array], gym.Space]:
#     space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

#     def wrapper(model_action):
#         throttle, brake, steering = model_action
#         throttle = (throttle + 1) / 2
#         brake = (brake + 1) / 2
#         return np.array([throttle, brake, steering], dtype=np.float32)

#     return wrapper, space

def _continuous() -> Tuple[Callable[[Dict[str, int]], Dict[str, np.ndarray]], gym.Space]:
    space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32) #Use angle and speed

    time_delta = 0.1  # Time, in seconds, between steps.
    max_angle = 90 / 180 * np.pi # Turning angle in radians
    max_speed = 28 # Speed in km/h

    def wrapper(
        action: Dict[str, int], saved_obs: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        wrapped_obs = {}
        for agent_id, agent_action in action.items():
            new_heading = saved_obs[agent_id]["heading"] + agent_action[0] * max_angle
            new_heading = (new_heading + np.pi) % (2 * np.pi) - np.pi

            magnitude = agent_action[1] * max_speed *time_delta
            cur_coord = (
                saved_obs[agent_id]["pos"][0] + 1j * saved_obs[agent_id]["pos"][1]
            )
            # Note: On the map, angle is zero at positive y axis, and increases anti-clockwise.
            #       In np.exp(), angle is zero at positive x axis, and increases anti-clockwise.
            #       Hence, numpy_angle = map_angle + π/2
            new_pos = cur_coord + magnitude * np.exp(1j * (new_heading + np.pi / 2))
            x_coord = np.real(new_pos)
            y_coord = np.imag(new_pos)

            wrapped_obs.update(
                {
                    agent_id: np.array(
                        [x_coord, y_coord, new_heading, time_delta], dtype=np.float32
                    )
                }
            )

        return wrapped_obs

    return wrapper, space


class Discrete_Action_11(gym.ActionWrapper):
    """Modifies the action space."""

    def __init__(self, env: gym.Env):
        """Sets identical action space, denoted by `space`, for all agents.

        Args:
            env (gym.Env): Gym env to be wrapped.
        """
        super().__init__(env)
        self._wrapper, action_space = _discrete_new()

        self.action_space = gym.spaces.Dict(
            {agent_id: action_space for agent_id in env.action_space.spaces.keys()}
        )

    def action(self, action):
        """Adapts the action input to the wrapped environment.

        `self.saved_obs` is retrieved from SaveObs wrapper. It contains previously
        saved observation parameters.

        Note: Users should not directly call this method.
        """
        wrapped_act = self._wrapper(action=action, saved_obs=self.saved_obs)

        return wrapped_act

def _discrete_new() -> Tuple[Callable[[Dict[str, int]], Dict[str, np.ndarray]], gym.Space]:
    time_delta = 0.1  # Time, in seconds, between steps.
    # Radians degree
    angles = [0, 2 / 180 * np.pi, 10 / 180 * np.pi, -2 / 180 * np.pi, -20 / 180 * np.pi]
    speeds = [30, 50] # Speed in km/h
    # Distance, in meter, travelled in time_delta seconds
    dists = [speed * 1000 / 3600 * time_delta for speed in speeds] 

    action_map = {}
    index = 0
    for dist in dists:
        for angle in angles:
            # disable the straight at 30 kmh/h speed
            if angle == 0 and dist == dists[0]:
                pass
            else: 
                action_map.update({index: [dist, angle]}) 
                index += 1

    #finally add the slow down action
    action_map.update({index: [0, 0]}) 

    space = gym.spaces.Discrete(n=len(action_map))

    def wrapper(
        action: Dict[str, int], saved_obs: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        wrapped_obs = {}
        for agent_id, agent_action in action.items():
            new_heading = saved_obs[agent_id]["heading"] + action_map[agent_action][1]
            new_heading = (new_heading + np.pi) % (2 * np.pi) - np.pi

            magnitude = action_map[agent_action][0]
            cur_coord = (
                saved_obs[agent_id]["pos"][0] + 1j * saved_obs[agent_id]["pos"][1]
            )
            # Note: On the map, angle is zero at positive y axis, and increases anti-clockwise.
            #       In np.exp(), angle is zero at positive x axis, and increases anti-clockwise.
            #       Hence, numpy_angle = map_angle + π/2
            new_pos = cur_coord + magnitude * np.exp(1j * (new_heading + np.pi / 2))
            x_coord = np.real(new_pos)
            y_coord = np.imag(new_pos)

            wrapped_obs.update(
                {
                    agent_id: np.array(
                        [x_coord, y_coord, new_heading, time_delta], dtype=np.float32
                    )
                }
            )

        return wrapped_obs

    return wrapper, space