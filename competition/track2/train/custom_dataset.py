import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np

class track2Dataset(Dataset):
    def __init__(self, input_dir, transform=None, target_transform=None):
        # self.img_labels = pd.read_pickle(annotations_file)
        self.img_labels = pd.DataFrame()
        for dirpath, subdirs, files in os.walk(input_dir):
            for file in files:
                file_name, ext = os.path.splitext(file)
                if ext == ".pkl":   
                    vehicle_id = file_name
                    try:
                        relpath = os.path.relpath(dirpath, input_dir)
                        obs = pd.read_pickle(os.path.join(dirpath, file))
                        dfi = self.get_data(obs, relpath, vehicle_id)
                        if self.img_labels.empty:
                            self.img_labels = dfi
                        else:
                            self.img_labels = self.img_labels.append(dfi)
                    except Exception as e:
                        print("Failed to load data from {} due to {}".format(dir, str(e)))
        self.input_dir = input_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
    
    def get_data(self, observations, dirpath, vehicle_id):
        df = pd.DataFrame()
        times = list(observations.keys())
        for i in range(len(times)-1):
            t = times[i]
            next_t = times[i+1]

            obs = observations[t]
            next_obs = observations[next_t]

            action = next_obs.ego_vehicle_state.position
            action[2] = float(next_obs.ego_vehicle_state.heading)
            action[2] = (action[2] + np.pi) % (2 * np.pi) - np.pi 

            ego_pos = obs.ego_vehicle_state.position
            ego_pos[2] = float(obs.ego_vehicle_state.heading)
            ego_pos[2] = (ego_pos[2] + np.pi) % (2 * np.pi) - np.pi 

            waypoints = self.get_waypoints(obs)

            label = self.get_label(observations, i)
            print(i)

            image_file = os.path.join(dirpath, str(t) + "_" + vehicle_id + ".png")
            
            data = {
                "action":[action],
                "ego_pos": [ego_pos],
                "waypoints":[np.array(waypoints).flatten()],
                "event": label,
                "image_file": image_file
            }

            dfi =  pd.DataFrame(data=data)
            if df.empty:
                df = dfi
            else:
                df = df.append(dfi)

        return df
            
    def get_waypoints(self, obs):
        ret_waypoints = np.zeros((5, 3))
        path_index = self.get_current_waypoint_path_index(obs)
        if path_index > 0: 
            waypoints = obs.waypoint_paths[path_index]
            for i in range(min(5, len(waypoints))):
                pos = waypoints[i].pos
                heading = float(waypoints[i].heading)

                ret_waypoints[i][:2] = pos
                ret_waypoints[i][-1] = heading

        return ret_waypoints

    def get_current_waypoint_path_index(self, obs):
        ego_pos = obs.ego_vehicle_state.position
        waypoints = obs.waypoint_paths

        min_dist = np.inf
        min_index = -1
        for i in range(len(waypoints)):
            waypoint_pos = waypoints[i][0].pos
            dist = np.linalg.norm(waypoint_pos - ego_pos[2:])
            if dist <= min_dist:
                min_dist = dist
                min_index = i

        if min_index < 0:
            print("no way points found for ego pos at {}".format(ego_pos))

        return min_index

    def get_label(self, observations, current_index):
        times = list(observations.keys())
        label = ""
        for i in range(min(10, len(observations)-current_index-1)):
            index = current_index + i + 1  
            t = times[index]
            events = observations[t].events

            if events.collisions:
                label = "collisions"
            elif events.off_road:
                label = "off_road"
            elif events.off_route:
                label = "off_route"
            elif events.on_shoulder:
                label = "on_shoulder"
            elif events.wrong_way:
                label = "wrong_way"
        
        if not label:
            label = "safe"

        return label

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
            img_path = os.path.join(self.input_dir, self.img_labels.loc[:, "image_file"].iloc[idx])
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
        except Exception as e:
            print("Error loading item {} in the dataset with image file at {} : {}".format(str(idx), img_path,str(e)))
