import torch.nn as nn

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
        feature_dim = 
        self.numeric_features_ = nn.Sequential(
            nn.Linear(feature_dim, repr_size),
            nn.ReLU(),
            nn.Linear(repr_size, repr_size),
        )

    def forward(self, x, y):
        x = x / 255
        x = (x - 0.5) / 0.5
        out = self.numeric_features_(x)
        out = flatten(out)
        return self.fc_layers(out)


        