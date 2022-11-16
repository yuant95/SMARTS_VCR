import torch
import os
from custom_dataset import CustomImageDataset
from datetime import datetime
from pathlib import Path

# load dataset 
TRAINING_DATA = "/home/yuant426/Desktop/SMARTS_track1/competition/track1/trainingData/20221026_2/1_to_2lane_left_turn_t (copy)"
annotations_file = os.path.join(TRAINING_DATA, "df_1_to_2lane_left_turn_t.pkl")
img_dir = TRAINING_DATA

dataset = CustomImageDataset(annotations_file, img_dir)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"],
        shuffle=True, drop_last=True)

def get_data_loader(data_folder)