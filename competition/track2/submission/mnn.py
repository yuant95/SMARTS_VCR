import torch.nn as nn
import torch

class NeuralNetwork(nn.Module):
    def __init__(self, img_shape=(3, 256, 256) ):
        super(NeuralNetwork, self).__init__()
        self.image_features_ = nn.Sequential(
            nn.Conv2d(img_shape[0], 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # action space + ego pos + 5 way points pos
        feature_dim = 3 + 3 + 5 * 3 
        self.numeric_features_ = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(64, 64*64),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self.combined_features_ = nn.Sequential(
            nn.Linear(64*64 + 32 * 15 * 15, 64*64*2*2),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(64*64*2*2, 64*64*2),
            nn.ReLU(inplace=True),
            nn.Linear(64*64*2, 64),
            nn.Linear(64, 6),
        )

    def forward(self, x, y):
        x = x / 255
        x = (x - 0.5) / 0.5
        x = self.image_features_(x)
        x = x.flatten(start_dim=1)
        y = self.numeric_features_(y)
        z = torch.cat((x,y), 1)
        z = self.combined_features_(z)
        return z

        


        