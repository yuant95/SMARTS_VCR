# MIT License
#
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import logging
import sys
import os

import gym
import numpy as np

try:
    from moviepy.editor import ImageClip, ImageSequenceClip
except (ImportError, ModuleNotFoundError):
    logging.warning(sys.exc_info())
    logging.warning(
        "You may not have installed the [gym] dependencies required to capture the video. Install them first with the `smarts[gym]` extras."
    )

    raise

import shutil
import time
from pathlib import Path

import wandb
import matplotlib.pyplot as plt
import imageio
import numpy as np
import invertedai as iai

from .iai_map_const import *

iai.add_apikey("JVzQDGMjeI7nMdZ0Ydl9G6yRD9NdxmPE1QCr0UGe")

class ScenePlotter:
    """
    Use images(rgb_array) to create a gif file.
    """

    def __init__(self, video_name_folder: str, traffic_agent: str, env: gym.Env):
        self.offset = [50, -60]
        self.env = env
        self.traffic_agent = traffic_agent
        self.scenarios_name = self.env.spec._kwargs["scenario"]

        self.map, self.offset = self.get_scenario_map_name_and_offset(self.scenarios_name)

        timestamp_str = time.strftime("%Y%m%d-%H%M%S")
        self.frame_folder = (
            video_name_folder + "_" + self.scenarios_name + "_" + timestamp_str
        )  # folder that uses to contain temporary frame images, will be deleted after the gif is created.# folder that uses to contain temporary frame images, will be deleted after the gif is created.

        self.iai_frame_folder = (
            video_name_folder + "_" + self.scenarios_name + "_iai_"
        )

        Path.mkdir(
            Path(self.frame_folder), exist_ok=True
        )  # create temporary frame images folder if not exists.

        Path.mkdir(
            Path(self.iai_frame_folder), exist_ok=True
        )

        self._video_root_path = str(
            Path(video_name_folder).parent
        )  # path of the video file
        self._video_name = str(Path(video_name_folder).name)  # name of the video

        self.count = 0

        self.plotter = None

    
    def get_scenario_map_name_and_offset(self, scenario_name):
        for key, value in IAI_MAP_NAME_MAPPING.items():
            if key in scenario_name:
                return value, IAI_MAP_OFFSET_MAPPING[key]

    def capture_frame(self, step_num: int, image: np.ndarray, infos):
        """
        Create image according to the rgb_array and store it with step number in the destinated folder
        """
        with ImageClip(image) as image_clip:
            image_clip.save_frame(
                f"{self.frame_folder}/{self._video_name}_{step_num}.jpeg"
            )

        if self.plotter == None:
            self.reset_scene_plotter(infos)
        else: 
            agent_ids, agent_states, agent_attributes = self.get_iai_agents(infos)
            if len(agent_states) < 20:
                diff = 20 - len(agent_states)
                agent_states_final = agent_states + [agent_states[-1]] * diff
                agent_states = agent_states_final
            self.plotter.record_step(agent_states)

    def generate_gif(self):
        """
        Use the images in the same folder to create a gif file.
        """
        video_paths = []
        try:
            fps=4
            timestamp_str = time.strftime("%Y%m%d-%H%M%S")
            caption = f"{self.scenarios_name}.gif"
            video_path = f"{self._video_root_path}/{self._video_name}_{self.scenarios_name}_{timestamp_str}.gif"
            with ImageSequenceClip(self.frame_folder, fps=fps) as clip:
                clip.write_gif(video_path)
            clip.close()
            video_paths.append(video_path)
            # wandb.log({caption:wandb.Video(video_path, fps=fps, format="gif", caption=timestamp_str)})

            #DEBUG for IAI
            # if self.traffic_agent == "itra":
            #     # caption = f"{self.scenarios_name}_iai.gif"
            #     video_path = f"{self._video_root_path}/{self._video_name}_{self.scenarios_name}_{timestamp_str}_iai.gif"
            #     with ImageSequenceClip(self.iai_frame_folder, fps=fps) as clip:
            #         clip.write_gif(video_path)
            #     clip.close()
            #     video_paths.append(video_path)
            #     # wandb.log({caption:wandb.Video(video_path, fps=fps, format="gif", caption=f"{timestamp_str}_iai")})

            # IAI visualization
            fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 10))
            video_path = f"{self._video_root_path}/{self._video_name}_{self.scenarios_name}_{timestamp_str}_iai.gif"
            self.plotter.animate_scene(output_name=video_path, ax=ax,
                      numbers=False, direction_vec=True, velocity_vec=False,
                      plot_frame_number=True)
            video_paths.append(video_path)
            
            self.count += 1
            
        except Exception as e:
            print("Something went wrong while generating gif: {}".format(str(e)))

        return video_paths, fps

    def close_recorder(self):
        """
        close the recorder by deleting the image folder.
        """
        shutil.rmtree(self.frame_folder, ignore_errors=True)

    def clear_content(self):
        # IAI debug code
        folder_list = [self.frame_folder]
        if self.traffic_agent == "itra":
            folder_list.append(self.iai_frame_folder)
        
        for folder in folder_list:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

        self.plotter = None

    def reset_scene_plotter(self, infos):
        location_info = iai.location_info(location=self.map)
        rendered_static_map = location_info.birdview_image.decode()
        self.plotter = iai.utils.ScenePlotter(rendered_static_map, location_info.map_fov, (location_info.map_center.x, location_info.map_center.y), location_info.static_actors)

        agent_ids, agent_states, agent_attributes = self.get_iai_agents(infos)
        agent_attributes = [agent_attributes[0]]*20
        if len(agent_states) < 20:
            diff = 20 - len(agent_states)
            agent_states_final = agent_states + [agent_states[-1]] * diff
            agent_states = agent_states_final
        self.plotter.initialize_recording(agent_states, agent_attributes=agent_attributes, conditional_agents = [0])

    def get_iai_agents(self, infos):
        agent_states = []
        agent_attributes = []
        agent_ids = []
        obs_ = infos["env_obs"]

        agent_ids.append(obs_.ego_vehicle_state.id)

        ego_center = iai.common.Point(x=float(obs_.ego_vehicle_state.position[0]-self.offset[0]), y=float(obs_.ego_vehicle_state.position[1]-self.offset[1]))
        ego_state = iai.common.AgentState(center=ego_center, orientation=float(obs_.ego_vehicle_state.heading+np.pi/2.0), speed=float(obs_.ego_vehicle_state.speed))
        agent_states.append(ego_state)

        ego_length = float(obs_.ego_vehicle_state.bounding_box.length)
        ego_width = float(obs_.ego_vehicle_state.bounding_box.width)
        rear_axis_offset = ego_length * 0.4
        ego_attri = iai.common.AgentAttributes(length=ego_length, width=ego_width, rear_axis_offset=rear_axis_offset)
        
        agent_attributes.append(ego_attri)

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
