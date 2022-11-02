import torch
import os
from custom_dataset import CustomImageDataset

TEST_DATA = "/home/yuant426/Desktop/SMARTS_track1/competition/track1/trainingData/20221029_2/3lane_merge_single_agent"
annotations_file = os.path.join(TEST_DATA, "df_3lane_merge_single_agent.pkl")
img_dir = TEST_DATA

model_path = "/home/yuant426/Desktop/SMARTS_track1/competition/track1/classifier/logs/2022_10_26_22_43_41/model_2022_10_27_22_04_42"

dataset = CustomImageDataset(annotations_file, img_dir)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=100,
        shuffle=True)


model = torch.load(model_path)
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for imgs, features, labels in data_loader:
        outputs = model(imgs, features.float())
        _, predicted = torch.max(outputs.data, 1)
        sm = torch.nn.Softmax()
        prob = sm(outputs)
        prob = sm(outputs) 
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        accuracy = correct/total
        print("Current Accuracy: {}".format(accuracy))

