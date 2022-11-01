import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_pickle(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        labels_map = {
            "collisions": 0,
            "collision": 0,
            "off_road": 1,
            "on_shoulder": 2,
            "wrong_way": 3,
            "off_route": 4,
            "safe":5
        }
        try:
            img_path = os.path.join(self.img_dir, self.img_labels.loc[:, "image_file"].iloc[idx])
            image = read_image(img_path)
            label = labels_map[self.img_labels.loc[:, "event"].iloc[idx]]
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            features = self.img_labels.loc[:, 
                ["action", 
                "ego_pos",
                "waypoints",  
                ]].iloc[idx]
            return_features = np.array([])
            for element in features.ravel():
                return_features = np.concatenate((return_features, element.astype(np.float32)), axis=0)
            return image, return_features, label
        except:
            del self.img_labels[idx]
            return self.__getitem__(idx)

# TRAINING_DATA = "/home/yuant426/Desktop/SMARTS_track1/competition/track1/trainingData/20221026_2/1_to_2lane_left_turn_t"

# annotations_file = os.path.join(TRAINING_DATA, "df_1_to_2lane_left_turn_t.pkl")
# img_dir = TRAINING_DATA


# dataset = CustomImageDataset(annotations_file, img_dir)

# for imgs, features, label in dataset:
#     print(features)
#     print(label)