import torch
import os
from custom_dataset import CustomImageDataset
from mnn import NeuralNetwork
import wandb
from datetime import datetime
from pathlib import Path

# load dataset 
TRAINING_DATA = "/home/yuant426/Desktop/SMARTS_track1/competition/track1/trainingData/20221026_2/1_to_2lane_left_turn_t (copy)"
annotations_file = os.path.join(TRAINING_DATA, "df_1_to_2lane_left_turn_t.pkl")
img_dir = TRAINING_DATA

time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
logdir = Path(__file__).absolute().parents[0] / "logs" / time
logdir.mkdir(parents=True, exist_ok=True)

config = {
    "training_data": TRAINING_DATA, 
    "learning_rate": 3e-4,
    "batch_size": 100,
    "n_epoch": 2000,
}

wandb_run = wandb.init(
    project="SMARTS_Classifier",
    config=config,
    save_code=True,  # optional
    )
config = wandb_run.config

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)  

dataset = CustomImageDataset(annotations_file, img_dir)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"],
        shuffle=True, drop_last=True)
model = NeuralNetwork()
model.to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

for epoch in range(1, config["n_epoch"]+1):
    loss_train = 0.0

    for imgs, features, label in data_loader:
        imgs = imgs.to(device)
        features = features.to(device)
        label = label.to(device)
        output = model(imgs, features.float())

        loss = loss_fn(output, label)

        l2_lambda=0.001
        l2_norm=sum(p.pow(2).sum() for p in model.parameters())
        loss += l2_lambda*l2_norm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_train += loss.item()

        wandb.log({'epoch': epoch, 'loss': loss_train})

    if epoch%200 == 0: 
        time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        path = str(logdir) + ("/model_" + time)
        torch.save(model, path)
        wandb.save(path, base_path=path)

time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
path = str(logdir) + ("/model_" + time)
torch.save(model, path)
wandb.save(path, base_path=path)