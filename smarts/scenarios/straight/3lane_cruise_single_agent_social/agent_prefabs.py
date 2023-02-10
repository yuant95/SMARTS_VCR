import numpy as np

from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.controllers import ActionSpaceType
from smarts.zoo.agent_spec import AgentSpec
from smarts.zoo.registry import register

import invertedai as iai
from invertedai.common import RecurrentState

ITRA_MAP_LOCATION = "smarts:3lane_cruise_single_agent"

iai.add_apikey("JVzQDGMjeI7nMdZ0Ydl9G6yRD9NdxmPE1QCr0UGe")

class invertedAiAgent(Agent):
    def __init__(self):
        self.location = ITRA_MAP_LOCATION
        self.recurrent_states = []
        self.offset = [105, 0]
        super().__init__()

        
    def act(self, obs):
        agent_states, agent_attributes = self.get_iai_agents(obs)
        # This is hack to provide initial recurrent state to ITRA
        if len(self.recurrent_states) == 0:
            recurrent_states = [RecurrentState() for _ in range(len(agent_states))]
            for recurrent, st in zip(recurrent_states, agent_states):
                recurrent.packed[-4:] = [st.center.x, st.center.y, st.orientation, st.speed]

        else:
            recurrent_states = self.recurrent_states

        res = iai.api.drive(
            location=self.location, 
            agent_states=agent_states, 
            agent_attributes=agent_attributes, 
            recurrent_states=recurrent_states,
            get_birdview=False)
        
        # Code for export birdview for debugging

        # birdview = res.birdview.decode()
        # fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 5))
        # ax.set_axis_off(), ax.imshow(birdview)
        
        self.recurrent_states = res.recurrent_states

        return self.get_action(res)

    def get_iai_agents(self, obs):
        ego_center = iai.common.Point(x=float(obs.ego_vehicle_state.position[0]-self.offset[0]), y=float(obs.ego_vehicle_state.position[1]-self.offset[1]))
        ego_state = iai.common.AgentState(center=ego_center, orientation=float(obs.ego_vehicle_state.heading+np.pi/2.0), speed=float(obs.ego_vehicle_state.speed))
        agent_states = [ego_state]

        ego_length = float(obs.ego_vehicle_state.bounding_box.length)
        ego_width = float(obs.ego_vehicle_state.bounding_box.width)
        rear_axis_offset = ego_length * 0.4
        ego_attri = iai.common.AgentAttributes(length=ego_length, width=ego_width, rear_axis_offset=rear_axis_offset)
        
        agent_attributes = [ego_attri]

        
        if (obs.neighborhood_vehicle_states):
            neighbors = obs.neighborhood_vehicle_states
            for i in range(len(neighbors)):
                center = iai.common.Point(float(neighbors[i].position[0]-self.offset[0]), float(neighbors[i].position[1]-self.offset[1]))
                orientation = float(neighbors[i].heading+np.pi/2.0)
                speed = float(neighbors[i].speed)

                length = float(neighbors[i].bounding_box.length)
                width = float(neighbors[i].bounding_box.width)
                rear_axis_offset = length * 0.4

                state = iai.common.AgentState(center=center, orientation=orientation, speed=speed)
                attri = iai.common.AgentAttributes(length=length, width=width, rear_axis_offset=rear_axis_offset)
        
                agent_states.append(state)
                agent_attributes.append(attri)

        return agent_states, agent_attributes

    def get_action(self, res):

        # Currently only use the prediction of the ego car as action
        # Could controll all agents in the neighbors as well in the future. 
        action = [res.agent_states[0].center.x+self.offset[0], res.agent_states[0].center.y+self.offset[1], res.agent_states[0].orientation-np.pi/2.0]
        time_delta = 0.1
        action = np.array(
                        [action[0], action[1], action[2], time_delta], dtype=np.float32
                    )

        return action

class KeepLaneAgent(Agent):
    def act(self, obs):
        return "keep_lane"

class MotionPlannerAgent(Agent):
    def act(self, obs):
        wp = obs.waypoint_paths[0][:5][-1]
        dist_to_wp = np.linalg.norm(wp.pos - obs.ego_vehicle_state.position[:2])
        target_speed = 5  # m/s
        return np.array([*wp.pos, wp.heading, dist_to_wp / target_speed])


register(
    locator="zoo-agent-v0",
    entry_point=lambda **kwargs: AgentSpec(
        interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=20000),
        agent_builder=KeepLaneAgent,
    ),
)

register(
    locator="motion-planner-agent-v0",
    entry_point=lambda **kwargs: AgentSpec(
        interface=AgentInterface(waypoints=True, action=ActionSpaceType.TargetPose),
        agent_builder=MotionPlannerAgent,
    ),
)

register(
    locator="inverted-agent-v0",
    entry_point=lambda **kwargs: AgentSpec(
        interface=AgentInterface(waypoints=True, action=ActionSpaceType.TargetPose),
        agent_builder=invertedAiAgent,
    ),
)