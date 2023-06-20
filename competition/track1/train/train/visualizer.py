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
import os
import typing
from pathlib import Path

import gym
import gym.envs

# from smarts.env.wrappers.gif_recorder import GifRecorder
from .scene_plotter import ScenePlotter

import wandb


class RecorderWrapper(gym.Wrapper):
    """
    A Wrapper that interacts the gym environment with the GifRecorder to record video step by step.
    """

    def __init__(self, video_name: str, env: gym.Env, traffic_agent: str):
        root_path = Path(__file__).parents[3]  # smarts main repo path
        video_folder = os.path.join(
            root_path, "videos"
        )  # video folder for all video recording file (.gif)
        Path.mkdir(
            Path(video_folder), exist_ok=True
        )  # create video folder if not exist

        super().__init__(env)
        self.video_name_folder = os.path.join(
            video_folder, video_name
        )  # frames folder that uses to contain temporary frame images, will be created using video name and current time stamp in gif_recorder when recording starts
        self.scene_plotter = ScenePlotter(video_name_folder=self.video_name_folder, traffic_agent = traffic_agent, env=self.env)
        self.recording = False
        self.current_frame = -1

    def reset(self, **kwargs):
        """
        Reset the gym environment and restart recording.
        """
        observations = super().reset(**kwargs)
        if self.recording == False:
            self.scene_plotter.clear_content()
            self.start_recording()
        else:
            try:
                video_paths, fps = self.scene_plotter.generate_gif()
            except Exception as e:
                print("WARNING: failed to generate gif due to {}".format(str(e)))
            else:
                observations["video_paths"] = video_paths
                observations["fps"] = fps

            self.scene_plotter.clear_content()
            
        self.current_frame = 0
        return observations

    def start_recording(self):
        """
        Start the gif recorder and capture the first frame.
        """
        if self.scene_plotter is None:
            self.scene_plotter = ScenePlotter(self.video_name_folder, self.env)
        image = super().render(mode="rgb_array")
        # self.scene_plotter.capture_frame(self.next_frame_id(), image, infos)
        self.recording = True

    def stop_recording(self):
        """
        Stop recording.
        """
        self.recording = False

    def step(self, action):
        """
        Step the environment using the action and record the next frame.
        """
        observations, rewards, dones, infos = super().step(action)
        if self.recording == True:
            image = super().render(mode="rgb_array")
            self.scene_plotter.capture_frame(self.next_frame_id(), image, infos)

        observations["video_path"] = ""
        observations["fps"] = 0

        return observations, rewards, dones, infos

    def next_frame_id(self):
        """
        Get the id for next frame.
        """
        self.current_frame += 1
        return self.current_frame

    def close(self):
        """
        Close the recorder by deleting the image folder and generate the gif file.
        """
        if self.scene_plotter is not None:
            self.scene_plotter.generate_gif()
            self.scene_plotter.close_recorder()
            self.scene_plotter = None
            self.recording = False

    def __del__(self):
        self.close()
