import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset


class Generator(nn.Module):

    def __init__(self, noise_len: int, condition_len: int,
                 output_feature: int):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # conditional gan
        input_feature = noise_len + condition_len
        self.model = nn.Sequential(*block(input_feature, 128, normalize=False),
                                   *block(128, 256), *block(256, 512),
                                   *block(512, 1024),
                                   nn.Linear(1024, output_feature))

    def forward(self, z: torch.Tensor, con: torch.Tensor):
        # Concatenate noise and condition to produce input
        x = torch.concat([z, con], dim=-1)
        out = self.model(x)
        return out


class Discriminator(nn.Module):

    def __init__(self, input_feature: int, condition_len: int):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_feature + condition_len, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, x: torch.Tensor, con: torch.Tensor):
        x_flat = torch.concat([x, con], dim=-1)
        validity = self.model(x_flat)
        return validity


class MyDataset(Dataset):

    def __init__(self, data: np.ndarray):
        data = data.astype(float)
        data = torch.from_numpy(data).float().cuda()
        self.samples = data[:, :-2]
        self.labels = data[:, -2:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]
