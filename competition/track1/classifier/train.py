import torch
import os
from custom_dataset import CustomImageDataset
from mnn import NeuralNetwork
import wandb

# load dataset 
TRAINING_DATA = "/home/yuant426/Desktop/SMARTS_track1/competition/track1/trainingData/20221026_2/1_to_2lane_left_turn_t (copy)"
annotations_file = os.path.join(TRAINING_DATA, "df_1_to_2lane_left_turn_t.pkl")
img_dir = TRAINING_DATA
config = {
    "training_data": TRAINING_DATA, 
    "learning_rate": 0.01,
    "batch_size": 100
}

wandb_run = wandb.init(
    project="SMARTS_Classifier",
    config=config,
    save_code=True,  # optional
    )
config = wandb_run.config

dataset = CustomImageDataset(annotations_file, img_dir)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"],
        shuffle=True, drop_last=True)
model = NeuralNetwork()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),
  lr=config["learning_rate"])

n_epochs = 1000

for epoch in range(1, n_epochs+1):
    loss_train = 0.0
    for imgs, features, label in data_loader:
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
