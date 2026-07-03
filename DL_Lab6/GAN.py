import torch
import torch.nn as nn

class ConditionEmbedding(nn.Module):
    def __init__(self, num_objects=24, embed_dim=128, image_size=64):
        super().__init__()
        self.image_size = image_size
        self.fc = nn.Sequential(
            nn.Linear(num_objects, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, image_size * image_size)
        )

    def forward(self, condition):
        batch_size = condition.shape[0]
        cond_map = self.fc(condition)
        cond_map = cond_map.view(batch_size, 1, self.image_size, self.image_size)
        return cond_map


class Generator(nn.Module):
    def __init__(self, noise_dim=128, num_objects=24, base_channels=64):
        super().__init__()
        self.noise_dim = noise_dim
        self.cond_emb = nn.Sequential(
            nn.Linear(num_objects, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(noise_dim + 128, base_channels * 8 * 4 * 4),
            nn.BatchNorm1d(base_channels * 8 * 4 * 4),
            nn.ReLU(inplace=True),
        )

        self.net = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_channels, base_channels // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels // 2, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, noise, condition):
        cond = self.cond_emb(condition)
        x = torch.cat([noise, cond], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), -1, 4, 4)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, num_objects=24, base_channels=64):
        super().__init__()
        self.cond_emb = ConditionEmbedding(num_objects=num_objects, image_size=64)

        self.net = nn.Sequential(
            nn.Conv2d(4, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 8, 1, 4, 1, 0),
        )

    def forward(self, image, condition):
        cond_map = self.cond_emb(condition)
        x = torch.cat([image, cond_map], dim=1)
        x = self.net(x)
        return x.view(-1, 1)