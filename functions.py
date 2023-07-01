import numpy as np
import torch.nn as nn
import torch

torch.manual_seed(0)


#  Functions required for polynomial fitting
def polynomial(p, x):
    y = 0
    for coefficient in p:
        y = x * y + coefficient
    return y


# To create a regressor X of order n
def regressor(u, n):
    num_cols = u.shape
    X = np.zeros((num_cols[0], n+1))  # 20*
    for i in range(n+1):
        x = u ** i
        X[:, i] = x.transpose()
    return X


# To calculate least square solution
def least_sqr(X, Y, N):
    x_trans = X.transpose()
    a = np.linalg.inv(np.dot(x_trans, X))
    b = np.dot(x_trans, Y)
    theta = np.dot(a, b)
    Y_pred = np.dot(X, theta)
    error = Y - Y_pred
    nrmse = np.sqrt((1 / N) * (np.dot(error.transpose(), error)))
    return np.round_(theta[::-1], 1), nrmse
#######################################################################################

# Classes and functions required for image inpainting

# Decreases the shape of the input layer to half. Ex: from 256*256 to 128*128.
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, (4, 4), stride=2, padding=1, bias=False), nn.LeakyReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Doubles the shape of the input layer. Ex: from 128*128 to 256*256.
class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, (4, 4), stride=2, padding=1, bias=False),
            nn.ReLU(),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)

        return x


class Generator(nn.Module):
    def __init__(self, channels=1):  # Noise image is 2d so one channel
        super(Generator, self).__init__()

        self.down1 = UNetDown(channels, 64)  # output shape 128*128, if input shape is 256*256
        self.down2 = UNetDown(64, 128)   # output shape 64*64
        self.down3 = UNetDown(128, 256) # output shape 32*32
        self.down4 = UNetDown(256, 512) # output shape 16*16
        self.down5 = UNetDown(512, 512) # output shape 8*8
        self.down6 = UNetDown(512, 512) # output shape 4*4
        self.down7 = UNetDown(512, 512) # output shape 2*2

        self.up1 = UNetUp(512, 512)  # output shape 4*4
        self.up2 = UNetUp(512, 1024)  # output shape 8*8
        self.up3 = UNetUp(1024, 512)  # output shape 16*16
        self.up4 = UNetUp(512, 256)  # output shape 32*32
        self.up5 = UNetUp(256, 128)  # output shape 64*64
        self.up6 = UNetUp(128, 128)  # output shape 128*128

        self.final = nn.Sequential(nn.ConvTranspose2d(128, 3, (4, 4), stride=2, padding=1), nn.Tanh())

    def forward(self, x):
        #  Propagate noise through fc layer and reshape to img shape
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        u1 = self.up1(d7)
        u2 = self.up2(u1)
        u3 = self.up3(u2)
        u4 = self.up4(u3)
        u5 = self.up5(u4)
        u6 = self.up6(u5)

        return self.final(u6)
