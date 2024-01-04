import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from torchvision import models

class Vgg(nn.Module):
    def __init__(self,
                 z_dim=256,
                 num_classes=2,
                 version=11,
                 probabilistic=True):
        super(Vgg, self).__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.out_dim = 2 * self.z_dim
        self.probabilistic = probabilistic
        if version == 11:
            encoder = models.vgg11_bn(pretrained=False)
        else:
            encoder = models.vgg11_bn(pretrained=False)

        num_features = encoder.classifier[6].in_features
        encoder.classifier[6] = nn.Linear(num_features, self.out_dim)
        self.encoder = encoder
        self.decoder = nn.Linear(z_dim, num_classes)

    def forward(self, x):
        z_params = self.encoder(x)
        z_mu = z_params[:, :self.z_dim]
        z_sigma = F.softplus(z_params[:, self.z_dim:])
        z_dist = distributions.Independent(distributions.normal.Normal(z_mu, z_sigma), 1)
        z = z_dist.rsample([1]).view([-1, self.z_dim])
        if self.training:
            return self.decoder(z), (z_mu, z_sigma)
        else:
            return self.decoder(z_mu)

if __name__ == '__main__':
    model = Vgg(z_dim=256,
                   num_classes=2,
                   version=11)
    model.train()
    logits, (mu, sigma) = model(torch.randn((100, 3, 128, 128)))
    print(logits.shape, mu.shape, sigma.shape)
    model.eval()
    logits = model(torch.randn((100, 3, 128, 128)))
    print(logits.shape)
