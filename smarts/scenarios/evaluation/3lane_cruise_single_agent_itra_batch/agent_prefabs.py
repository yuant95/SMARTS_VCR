import numpy as np

from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.controllers import ActionSpaceType
from smarts.zoo.agent_spec import AgentSpec
from smarts.zoo.registry import register

import invertedai as iai
from invertedai.common import RecurrentState

from pathlib import Path
import os
import time

SCENARIOS_NAME = "eval_3lane_cruise_single_agent_itra_batch"
ITRA_MAP_LOCATION = "smarts:3lane_cruise_single_agent_extended"

iai.add_apikey("JVzQDGMjeI7nMdZ0Ydl9G6yRD9NdxmPE1QCr0UGe")

class invertedAiBoidAgent(Agent):
    def __init__(self):
        self.location = ITRA_MAP_LOCATION
        self.recurrent_states = {}
        # Calculate the offset between iai map and smarts map
        # iai map is centered at map center while smarts map is centered at 
        
        self.offset = [-502163.15625, -99]
        # self.step_num = 0
        super().__init__()

        video_name = "video"
        root_path = Path(__file__).parents[4]  # smarts main repo path
        video_folder = os.path.join(
            root_path, "videos"
        )
        self.video_name_folder = os.path.join(
            video_folder, video_name
        )
        self.iai_frame_folder = self.frame_folder = (
            self.video_name_folder + "_" + SCENARIOS_NAME + "_iai_"
        )
        
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
            # time_stamp = int(time.time())
            # from moviepy.editor import ImageClip
            # with ImageClip(image) as image_clip:
            #     image_clip.save_frame(
            #         f"{self.iai_frame_folder}/video_{time_stamp}.jpeg"
            #     )
            
            # fig, ax = plt.subplots(constraineds_layout=True, figsize=(5, 5))
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
        

register(
    locator=f"inverted-boid-agent-{SCENARIOS_NAME}-v0",
    entry_point=lambda **kwargs: AgentSpec(
        interface=AgentInterface(
            action=ActionSpaceType.TargetPose,
            waypoints=True,
            neighborhood_vehicles=True
        ),
        agent_builder=invertedAiBoidAgent,
    ),
)