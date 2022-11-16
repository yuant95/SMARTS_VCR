# Add offline training code for Track-2 here.
# On completion of training, automatically save the trained model to `track2/submission` directory.

import argparse
import glob
import re
import numpy as np
import os
import pickle
import sys
import shutil
import yaml
from datetime import datetime
from pathlib import Path
from PIL import Image
from typing import Any, Dict, Optional

import torch
from custom_dataset import track2Dataset
from torch.utils.data.dataloader import default_collate
from mnn import NeuralNetwork

# To import submission folder
sys.path.insert(0, str(Path(__file__).parents[1]))

def my_collate(batch):
    batch = list(filter(lambda x : x is not None, batch))
    return default_collate(batch)
    
def load_config(path: Path) -> Optional[Dict[str, Any]]:
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config

def train(input_path, output_path):
    # Get config parameters.
    train_config = load_config(Path(__file__).absolute().parents[0] / "config.yaml")
    save_directory = Path(__file__).absolute().parents[0] / "torch_logs"
    if not os.path.isdir(save_directory):
        index = 0
        os.mkdir(save_directory)
    else:
        index = len(os.listdir(save_directory))

    train_config["input_path"] = input_path
    train_config["log_dir"] = save_directory

    # if torch.cuda.is_available(): 
    if False: 
        dev = "cuda:0" 
    else:  
        dev = "cpu"  
        device = torch.device(dev)  

    # dataset = CustomImageDataset(annotations_file, img_dir)
    dataset = track2Dataset(input_dir=train_config["input_path"])
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=train_config["n_batch"],
            shuffle=True, drop_last=True, collate_fn=my_collate)

    model = NeuralNetwork()
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config["learning_rate"])

    for epoch in range(1, train_config["n_epoch"]+1):
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
        
        if epoch%train_config["checkpoint_fq"] == 0: 
            time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            path = str(train_config["log_dir"]) + ("/model_checkpoint_epoch" + str(epoch) + "_" + time)
            torch.save(model, path)

    torch.save(model, os.path.join(output_path, "model"))
    shutil.rmtree(save_directory)

def main(args: argparse.Namespace):
    input_path = args.input_dir
    output_path = args.output_dir
    train(input_path, output_path)


if __name__ == "__main__":
    program = Path(__file__).stem
    parser = argparse.ArgumentParser(program)
    parser.add_argument(
        "--input_dir",
        help="The path to the directory containing the offline training data",
        type=str,
        default="/SMARTS/competition/offline_dataset/",
    )
    parser.add_argument(
        "--output_dir",
        help="The path to the directory storing the trained model",
        type=str,
        default="/SMARTS/competition/track2/submission/",
    )

    args = parser.parse_args()

    main(args)