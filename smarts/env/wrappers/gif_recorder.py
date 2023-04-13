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


class GifRecorder:
    """
    Use images(rgb_array) to create a gif file.
    """

    def __init__(self, video_name_folder: str, traffic_agent: str, env: gym.Env):
        self.env = env
        self.traffic_agent = traffic_agent
        self.scenarios_name = self.env.spec._kwargs["scenario"]

        timestamp_str = time.strftime("%Y%m%d-%H%M%S")
        self.frame_folder = (
            video_name_folder + "_" + self.scenarios_name + "_" + timestamp_str
        )  # folder that uses to contain temporary frame images, will be deleted after the gif is created.# folder that uses to contain temporary frame images, will be deleted after the gif is created.

        Path.mkdir(
            Path(self.frame_folder), exist_ok=True
        )  # create temporary frame images folder if not exists.

        self._video_root_path = str(
            Path(video_name_folder).parent
        )  # path of the video file
        self._video_name = str(Path(video_name_folder).name)  # name of the video

        self.count = 0

    def capture_frame(self, step_num: int, image: np.ndarray):
        """
        Create image according to the rgb_array and store it with step number in the destinated folder
        """
        with ImageClip(image) as image_clip:
            image_clip.save_frame(
                f"{self.frame_folder}/{self._video_name}_{step_num}.jpeg"
            )

    def generate_gif(self):
        """
        Use the images in the same folder to create a gif file.
        """
        timestamp_str = time.strftime("%Y%m%d-%H%M%S")
        video_path = f"{self._video_root_path}/{self._video_name}_{self.scenarios_name}_{timestamp_str}.gif"
        with ImageSequenceClip(self.frame_folder, fps=3) as clip:
            clip.write_gif(video_path)
        clip.close()
        wandb.log({"video":wandb.Video(video_path, fps=3, format="gif", caption=video_path)})

        #DEBUG for IAI
        if self.traffic_agent == "itra":
            iai_folder = "/home/yuant426/miniconda3/envs/smartsEnvTest/lib/python3.8/site-packages/videos/iai"
            video_path = f"{self._video_root_path}/{self._video_name}_{self.count}_iai.gif"
            with ImageSequenceClip(iai_folder, fps=10) as clip:
                clip.write_gif(video_path)
            clip.close()
            wandb.log({"video":wandb.Video(video_path, fps=10, format="gif", caption=f"{self.count}_iai")})
        
        self.count += 1

    def close_recorder(self):
        """
        close the recorder by deleting the image folder.
        """
        shutil.rmtree(self.frame_folder, ignore_errors=True)

    def clear_content(self):
        # IAI debug code
        folder_list = [self.frame_folder]
        if self.traffic_agent == "itra":
            iai_folder = "/home/yuant426/miniconda3/envs/smartsEnvTest/lib/python3.8/site-packages/videos/iai"
            folder_list.append(iai_folder)
        
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
