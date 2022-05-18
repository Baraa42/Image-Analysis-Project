import torch.nn as nn


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.Linear)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super().__init__()
        # Input: N x channels_img x 64 x 64
        self.disc = nn.Sequential(
            nn.Conv2d(
                channels_img, features_d, kernel_size=5, stride=2, padding=2
            ),  # 32x32
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 5, 2, 2),  # 16x16
            self._block(features_d * 2, features_d * 4, 5, 2, 2),  # 8x8
            self._block(features_d * 4, features_d * 8, 5, 2, 2),  # 4x4
            nn.Flatten(),  # 1x1
            nn.Linear(features_d * 8 * 4 * 4, 1),
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,  # because we are using batch norm
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super().__init__()
        # Input: N x z_dim x 1 x 1
        # Reshape it in forward to N x _ x 4 x 4

        self.dense = nn.Linear(z_dim, features_g * 16 * 4 * 4)  #
        self.gen = nn.Sequential(
            self._block(features_g * 16, features_g * 8, 5, 2, 2, 1),  # 8x8
            self._block(features_g * 8, features_g * 4, 5, 2, 2, 1),  # 16x16
            self._block(features_g * 4, features_g * 2, 5, 2, 2, 1),  # 32x32
            nn.ConvTranspose2d(
                features_g * 2,
                channels_img,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),  # 64x64
            nn.Tanh(),  # [-1, 1]
        )

    def _block(
        self, in_channels, out_channels, kernel_size, stride, padding, output_padding
    ):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=False,  # because we are using batch norm
            ),
        )

    def forward(self, x):
        x = self.dense(x)
        x = x.view(x.size(0), -1, 4, 4)
        return self.gen(x)
