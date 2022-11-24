import matplotlib.pyplot as plt
import numpy as np
import cv2
import invertedai as iai
from invertedai.common import RecurrentState
from smarts.core.agent import Agent

# iai.add_apikey("RO7o44QnaE3bjNNIZZCgQ37IhXtUtRlCaqthMkJG")
iai.add_apikey("JVzQDGMjeI7nMdZ0Ydl9G6yRD9NdxmPE1QCr0UGe")

location = "smarts:3lane_cruise_single_agent"

class invertedAiAgent(Agent):
    def __init__(self):
        self.location = location
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
        ego_center = iai.common.Point(float(obs["ego"]["pos"][0]-self.offset[0]), float(obs["ego"]["pos"][1]-self.offset[1]))
        ego_state = iai.common.AgentState(center=ego_center, orientation=float(obs["ego"]["heading"]+np.pi/2.0), speed=float(obs["ego"]["speed"]))
        agent_states = [ego_state]

        ego_length = float(obs["ego"]["box"][0])
        ego_width = float(obs["ego"]["box"][1])
        rear_axis_offset = ego_length * 0.4
        ego_attri = iai.common.AgentAttributes(length=ego_length, width=ego_width, rear_axis_offset=rear_axis_offset)
        
        agent_attributes = [ego_attri]

        neighbors = obs["neighbors"]
        for i in range(len(neighbors["box"])):
            center = iai.common.Point(float(neighbors["pos"][i][0]-self.offset[0]), float(neighbors["pos"][i][1]-self.offset[1]))
            orientation = float(neighbors["heading"][i]+np.pi/2.0)
            speed = float(neighbors["speed"][i])

            length = float(neighbors["box"][i][0])
            width = float(neighbors["box"][i][1])
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