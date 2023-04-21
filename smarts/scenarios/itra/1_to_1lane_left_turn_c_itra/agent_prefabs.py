import numpy as np

from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.controllers import ActionSpaceType
from smarts.zoo.agent_spec import AgentSpec
from smarts.zoo.registry import register

import invertedai as iai
from invertedai.common import RecurrentState

import time


ITRA_MAP_LOCATION = "smarts:1_to_1lane_left_turn_c_extended_no_stopline"

iai.add_apikey("JVzQDGMjeI7nMdZ0Ydl9G6yRD9NdxmPE1QCr0UGe")

class invertedAiBoidAgent(Agent):
    def __init__(self):
        self.location = ITRA_MAP_LOCATION
        self.recurrent_states = {}
        self.offset = [0, 0]
        # self.step_num = 0
        super().__init__()
        
    def act(self, obs):
        if len(obs) > 0 :
            agent_ids, agent_states, agent_attributes = self.get_iai_agents(obs)
            # This is hack to provide initial recurrent state to ITRA
            if len(self.recurrent_states) == 0:
                recurrent_states = [RecurrentState() for _ in range(len(agent_states))]
                for recurrent, st in zip(recurrent_states, agent_states):
                    recurrent.packed[-4:] = [st.center.x, st.center.y, st.orientation, st.speed]

            else:
                recurrent_states = []
                for index, agent_id in enumerate(agent_ids):
                    if agent_id in self.recurrent_states:
                        recurrent_states.append(self.recurrent_states[agent_id])
                    else:
                        recurrent_state = RecurrentState()
                        recurrent_state.packed[-4:] = [agent_states[index].center.x, agent_states[index].center.y, agent_states[index].orientation, agent_states[index].speed]
                        recurrent_states.append(recurrent_state)

            tries = 50
            for i in range(0,tries):
                try:
                    res = iai.api.drive(
                        location=self.location, 
                        agent_states=agent_states, 
                        agent_attributes=agent_attributes, 
                        recurrent_states=recurrent_states,
                        get_birdview=False)
                except Exception as e:
                    if i < tries - 1: # i is zero indexed
                        print("Exception raised from iai.api.drive： {}".format(str(e)))
                        print("Retrying sending the request {} times...".format(str(i)))
                        continue
                        
                    else:
                        raise e
                else:
                    break
            
            # Code for export birdview for debugging
            # image = res.birdview.decode()
            # folder = "/home/yuant426/miniconda3/envs/smartsEnvTest/lib/python3.8/site-packages/videos/iai"
            # time_stamp = int(time.time())
            # from moviepy.editor import ImageClip
            # with ImageClip(image) as image_clip:
            #     image_clip.save_frame(
            #         f"{folder}/video_{time_stamp}.jpeg"
            #     )
            
            # fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 5))
            # ax.set_axis_off(), ax.imshow(birdview)


            
            for index, agent_id in enumerate(obs):
                self.recurrent_states[agent_id] = res.recurrent_states[index]
            
            actions = {vehicle_id: self.get_action(res.agent_states[agent_ids.index(vehicle_id)]) for index, vehicle_id in enumerate(obs)}
            return actions
        else:
            return {}
            
    def get_iai_agents(self, obs):
        agent_states = []
        agent_attributes = []
        agent_ids = []
        for vehicle_id, obs_ in obs.items():
            agent_ids.append(vehicle_id)

            ego_center = iai.common.Point(x=float(obs_.ego_vehicle_state.position[0]-self.offset[0]), y=float(obs_.ego_vehicle_state.position[1]-self.offset[1]))
            ego_state = iai.common.AgentState(center=ego_center, orientation=float(obs_.ego_vehicle_state.heading+np.pi/2.0), speed=float(obs_.ego_vehicle_state.speed))
            agent_states.append(ego_state)

            ego_length = float(obs_.ego_vehicle_state.bounding_box.length)
            ego_width = float(obs_.ego_vehicle_state.bounding_box.width)
            rear_axis_offset = ego_length * 0.4
            ego_attri = iai.common.AgentAttributes(length=ego_length, width=ego_width, rear_axis_offset=rear_axis_offset)
            
            agent_attributes.append(ego_attri)

        for vehicle_id, obs_ in obs.items():
            if (obs_.neighborhood_vehicle_states):
                neighbors = obs_.neighborhood_vehicle_states
                for i in range(len(neighbors)):
                    if neighbors[i].id not in agent_ids:
                        agent_ids.append(neighbors[i].id)
                        
                        center = iai.common.Point(x=float(neighbors[i].position[0]-self.offset[0]), y=float(neighbors[i].position[1]-self.offset[1]))
                        orientation = float(neighbors[i].heading+np.pi/2.0)
                        speed = float(neighbors[i].speed)

                        length = float(neighbors[i].bounding_box.length)
                        width = float(neighbors[i].bounding_box.width)
                        rear_axis_offset = length * 0.4

                        state = iai.common.AgentState(center=center, orientation=orientation, speed=speed)
                        attri = iai.common.AgentAttributes(length=length, width=width, rear_axis_offset=rear_axis_offset)
                
                        agent_states.append(state)
                        agent_attributes.append(attri)

        return agent_ids, agent_states, agent_attributes

    def get_action(self, agent_state):

        # Currently only use the prediction of the ego car as action
        # Could controll all agents in the neighbors as well in the future. 
        action = [agent_state.center.x+self.offset[0], agent_state.center.y+self.offset[1], agent_state.orientation-np.pi/2.0]
        time_delta = 0.1
        action = np.array(
                        [action[0], action[1], action[2], time_delta], dtype=np.float32
                    )

        return action
        

class invertedAiAgent(Agent):
    def __init__(self):
        self.location = ITRA_MAP_LOCATION
        self.recurrent_states = {}
        self.offset = [50, 20]
        super().__init__()

        
    def act(self, obs):
        agent_ids, agent_states, agent_attributes, agent_recurrent_states = self.get_iai_agents(obs)
        # This is hack to provide initial recurrent state to ITRA

        res = iai.api.drive(
            location=self.location, 
            agent_states=agent_states, 
            agent_attributes=agent_attributes, 
            recurrent_states=agent_recurrent_states,
            get_birdview=False)
        
        # Code for export birdview for debugging

        # birdview = res.birdview.decode()
        # fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 5))
        # ax.set_axis_off(), ax.imshow(birdview)
        
        for index, id in enumerate(agent_ids):
            self.recurrent_states[id] = res.recurrent_states[index]

        return self.get_action(res)

    def get_iai_agents(self, obs):
        agent_ids  = []
        agent_recurrent_states = []

        agent_ids.append(obs.ego_vehicle_state.id)
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
                agent_ids.append(neighbors[i].id)
                
                center = iai.common.Point(x=float(neighbors[i].position[0]-self.offset[0]), y=float(neighbors[i].position[1]-self.offset[1]))
                orientation = float(neighbors[i].heading+np.pi/2.0)
                speed = float(neighbors[i].speed)

                length = float(neighbors[i].bounding_box.length)
                width = float(neighbors[i].bounding_box.width)
                rear_axis_offset = length * 0.4

                state = iai.common.AgentState(center=center, orientation=orientation, speed=speed)
                attri = iai.common.AgentAttributes(length=length, width=width, rear_axis_offset=rear_axis_offset)
        
                agent_states.append(state)
                agent_attributes.append(attri)

        # finally construct the recurrent states
        for index, id in enumerate(agent_ids):
            if id in self.recurrent_states:
                agent_recurrent_states.append(self.recurrent_states[id])
            else:
                tmp = RecurrentState()
                tmp.packed[-4:] = [agent_states[index].center.x, agent_states[index].center.y, agent_states[index].orientation, agent_states[index].speed]
                agent_recurrent_states.append(tmp)


        return agent_ids, agent_states, agent_attributes, agent_recurrent_states

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
        interface=AgentInterface(neighborhood_vehicles=True,waypoints=True, action=ActionSpaceType.TargetPose),
        agent_builder=invertedAiAgent,
    ),
)

register(
    locator="inverted-boid-agent-v0",
    entry_point=lambda **kwargs: AgentSpec(
        interface=AgentInterface(
            action=ActionSpaceType.TargetPose,
            waypoints=True,
            neighborhood_vehicles=True,
        ),
        agent_builder=invertedAiBoidAgent,
    ),
)