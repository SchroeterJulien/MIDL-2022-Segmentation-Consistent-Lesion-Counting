import torch.nn as nn


class Conv3d_LeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, avg_kernel=2, avg_stride=2, avg_pad=0):
        super(Conv3d_LeakyReLU, self).__init__()
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=1e-2)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        self.avg_pool = nn.AvgPool3d(kernel_size=avg_kernel, stride=avg_stride, padding=avg_pad)

    def forward(self, x):
        return self.avg_pool(self.relu(self.conv1(x)))


class Convolutions(nn.Module):
    def __init__(self, in_channels, out_channels, p=[]):
        super(Convolutions, self).__init__()
        self.conv1 = Conv3d_LeakyReLU(in_channels=in_channels, out_channels=out_channels[0], kernel_size=3,
                                      padding=0, stride=1)
        self.conv2 = Conv3d_LeakyReLU(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=(1, 3, 3),
                                      padding=0, stride=1, avg_kernel=(1, 2, 2), avg_stride=(1, 2, 2),
                                      padding2=(0, 1, 1))
        self.conv3 = nn.Conv3d(out_channels[1], out_channels[2], kernel_size=1, padding=0, stride=1)
        if len(p) != 0:
            self.dropout = nn.Dropout3d(p=p[0], inplace=True)
            self.conv1 = nn.Sequential(self.conv1, self.dropout)
            self.conv2 = nn.Sequential(self.conv2, self.dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class Bottleneck(nn.Module):
    def __init__(self, HP, i, k, num_classes=5, p=[]):
        super(Bottleneck, self).__init__()
        self.conv = Convolutions(i, [1 * k, 2 * k, num_classes], HP, p=p)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return x