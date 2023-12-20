import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from numbers import Number

# class Leaf(nn.Module):
#     def __init__(self, image_size=84, num_classes=2):
#         super(Leaf, self).__init__()
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.MaxPool2d(2, 2),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.MaxPool2d(2, 2),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.MaxPool2d(2, 2),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.MaxPool2d(2, 2),
#             nn.ReLU()
#         )
#         self.fc_input_size = 32 * (image_size // 16) * (image_size // 16)
#         self.fc = nn.Linear(self.fc_input_size, num_classes)
#
#     def forward(self, x):
#         x = self.conv_layers(x)
#         x1 = x.view(-1, self.fc_input_size)
#         x = self.fc(x1)
#         return x, x1

class Leaf(nn.Module):
    def __init__(self, image_size=84, z_dim=256, probabilistic=False, num_classes=2):
        super(Leaf, self).__init__()
        self.z_dim = z_dim
        self.probabilistic = probabilistic
        self.num_classes = num_classes
        self.fc_input_size = 32 * (image_size // 16) * (image_size // 16)
        self.out_dim = 2 * self.z_dim if self.probabilistic else self.z_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.fc_input_size, self.out_dim)
        )
        self.decoder = nn.Linear(z_dim, num_classes)

    def featurize(self, x):
        if not self.probabilistic:
            return self.encoder(x)
        else:
            z_params = self.encoder(x)
            z_mu = z_params[:, :self.z_dim]
            z_sigma = F.softplus(z_params[:, self.z_dim:])
            z_dist = distributions.Independent(distributions.normal.Normal(z_mu, z_sigma), 1)
            z = z_dist.rsample([1]).view([-1, self.z_dim])
            return z, (z_mu, z_sigma)


    def forward(self, x):
        if not self.probabilistic:  # deterministic
            z = self.featurize(x=x)
            return self.decoder(z)
        else:
            if self.training:
                z, (z_mu, z_sigma) = self.featurize(x)
                return self.decoder(z), (z_mu, z_sigma)
            else:
                z, (z_mu, z_sigma) = self.featurize(x)
                return self.decoder(z_mu)

if __name__ == '__main__':
    model = Leaf(image_size=84, probabilistic=True)
    model.train()
    logits, (mu, sigma) = model(torch.randn((100, 3, 84, 84)))
    print(logits.shape, mu.shape, sigma.shape)
    model.eval()
    logits = model(torch.randn((100, 3, 84, 84)))
    print(logits.shape)
