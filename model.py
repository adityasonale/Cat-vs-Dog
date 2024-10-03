import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNNetwork(nn.Module):
    def __init__(self, n_outputs) -> None:
        super(CNNNetwork, self).__init__()

        self.n_outputs = n_outputs

    # initialising layers:

        # Convolution Layers
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels= 32, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)

        # Pooling Layers:
        self.pool = nn.MaxPool2d(2,2)

        # Fully Connected Layers:
        self.fc_1 = nn.Linear(256 * 14 * 14, 512)
        self.fc_2 = nn.Linear(512, 256)
        self.fc_3 = nn.Linear(256,self.n_outputs)


    def forward(self, x):

        x = self.pool(F.relu(self.conv_1(x)))
        x = self.pool(F.relu(self.conv_2(x)))
        x = self.pool(F.relu(self.conv_3(x)))
        x = self.pool(F.relu(self.conv_4(x)))

        x = x.view(-1, 256 * 14 * 14)  # Flatten the tensor,

        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)

        return x