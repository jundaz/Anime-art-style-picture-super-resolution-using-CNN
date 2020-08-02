import math
from torch import nn


class ACNet(nn.Module):
    def __init__(self, scale_factor):
        super(ACNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.prelu1 = nn.PReLU(8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=1)
        self.prelu2 = nn.PReLU(4)
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1)
        self.prelu3 = nn.PReLU(4)
        self.conv4 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1)
        self.prelu4 = nn.PReLU(4)
        self.conv5 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1)
        self.prelu5 = nn.PReLU(4)
        self.conv6 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1)
        self.prelu6 = nn.PReLU(4)
        self.conv7 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=1)
        self.prelu7 = nn.PReLU(8)
        self.convnet = nn.Sequential(self.conv1, self.prelu1, self.conv2, self.prelu2, self.conv3, self.prelu3,
                                     self.conv4, self.prelu4, self.conv5, self.prelu5, self.conv6, self.prelu6,
                                     self.conv7, self.prelu7)
        self.output = nn.ConvTranspose2d(8, 1, kernel_size=3, stride=scale_factor, padding=1,
                                         output_padding=scale_factor - 1)
        for i in self.convnet:
            if isinstance(i, nn.Conv2d):
                nn.init.normal_(i.weight.data, mean=0.0,
                                std=math.sqrt(2 / (i. out_channels * i. weight.data[0][0].numel())))
                nn.init.zeros_(i.bias.data)
        nn.init.normal_(self.output.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.output.bias.data)


    def forward(self, x):
        x = self.convnet(x)
        x = self.output(x)
        return x


