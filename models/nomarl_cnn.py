from torch import nn
import torch


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool3d(kernel_size=2, ceil_mode=False)

        self.conv1 = nn.Conv3d(in_channels=6, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=1)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=16, kernel_size=3)
        self.drop = nn.Dropout3d(0.2)

        self.fc1 = nn.Linear(432000, 256)
        self.fc2 = nn.Linear(256, 16)
        self.act = nn.Softmax(dim=1)
        self.act2 = nn.Sigmoid()

    def forward(self, x, y):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.max_pool(x)
        x = self.relu(self.conv3(x))
        x = x.view(x.size()[0], -1)
        x = self.drop(x)
        x = self.act(self.fc1(x))
        x = self.drop(x)
        x = self.act(self.fc2(x))
        output = torch.cat([x, y], dim=1)
        output = self.act2(nn.Linear(output.size()[1], 3)(output))
        return output


x = torch.rand([1, 6, 64, 64, 64])
y = torch.rand([1, 16])
net = CNN()
output = net(x, y)
print(output)
