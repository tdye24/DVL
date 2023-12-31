import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from torchvision import models

class ResNet(nn.Module):
    def __init__(self,
                 z_dim=256,
                 num_classes=2,
                 version=18,
                 probabilistic=True):
        super(ResNet, self).__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.out_dim = 2 * self.z_dim
        self.probabilistic = probabilistic
        if version == 18:
            encoder = models.resnet18(pretrained=False)
        else:
            encoder = models.resnet18(pretrained=False)

        num_features = encoder.fc.in_features
        encoder.fc = nn.Linear(num_features, self.out_dim)
        self.encoder = encoder
        self.decoder = nn.Linear(z_dim, num_classes)

    def forward(self, x):
        z_params = self.encoder(x)
        z_mu = z_params[:, :self.z_dim]
        z_sigma = F.softplus(z_params[:, self.z_dim:])
        if self.probabilistic:
            z_dist = distributions.Independent(distributions.normal.Normal(z_mu, z_sigma), 1)
            z = z_dist.rsample([1]).view([-1, self.z_dim])
            if self.training:
                return self.decoder(z), (z_mu, z_sigma)
            else:
                return self.decoder(z_mu)
        else:
            if self.training:
                return self.decoder(z_mu), (z_mu, z_sigma)
            else:
                return self.decoder(z_mu)

if __name__ == '__main__':
    model = ResNet(z_dim=256,
                   num_classes=2,
                   version=18)
    model.train()
    logits, (mu, sigma) = model(torch.randn((100, 3, 128, 128)))
    print(logits.shape, mu.shape, sigma.shape)
    model.eval()
    logits = model(torch.randn((100, 3, 128, 128)))
    print(logits.shape)
