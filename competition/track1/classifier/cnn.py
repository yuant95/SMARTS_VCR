import torch.nn as nn

import logging
logger = logging.getLogger(__name__)

class CNNEncoder(nn.Module):
    def __init__(self, repr_size, img_shape=(3, 256, 256)):
        super(CNNEncoder, self).__init__()

        self.conv_layers = nn.Sequential(
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

        feature_dim = conv_output_size(self.conv_layers, img_shape)
        logger.info("Representation size is {}".format(feature_dim))

        self.fc_layers = nn.Sequential(
            nn.Linear(feature_dim, repr_size),
            nn.ReLU(),
            nn.Linear(repr_size, repr_size),
        )

    def forward(self, x):
        # TODO: check what range our dataloader actually provides pixel values in
        x = x / 255
        x = (x - 0.5) / 0.5
        out = self.conv_layers(x)
        out = flatten(out)
        return self.fc_layers(out)