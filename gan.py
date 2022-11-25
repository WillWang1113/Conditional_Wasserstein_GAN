import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset


class Generator(nn.Module):

    def __init__(self, input_feature: int, output_feature: int):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(*block(input_feature, 128, normalize=True),
                                   *block(128, 256), *block(256, 512),
                                   *block(512, 1024),
                                   nn.Linear(1024, output_feature))

    def forward(self, z: torch.Tensor):
        out = self.model(z)
        return out


class Discriminator(nn.Module):

    def __init__(self, input_feature: int):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_feature, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor):
        x_flat = x.view(x.shape[0], -1)
        validity = self.model(x_flat)
        return validity


class MyDataset(Dataset):

    def __init__(self, data: np.ndarray, indice: list = None):
        data = data.astype(float)
        self.samples = torch.from_numpy(data).float().cuda()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.samples[idx]
