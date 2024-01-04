import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from torchvision import models

class Model(nn.Module):
    def __init__(self,
                 z_dim=256,
                 num_classes=2,
                 probabilistic=True,
                 backbone='resnet'):
        super(Model, self).__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.out_dim = 2 * self.z_dim
        self.probabilistic = probabilistic

        if backbone == 'resnet18':
            encoder = models.resnet18(pretrained=False)
            encoder.fc = nn.Linear(encoder.fc.in_features, self.out_dim)
        elif backbone == 'resnet50':
            encoder = models.resnet50(pretrained=False)
            encoder.fc = nn.Linear(encoder.fc.in_features, self.out_dim)
        elif backbone == 'vgg11':
            encoder = models.vgg11_bn(pretrained=False)
            num_features = encoder.classifier[6].in_features
            encoder.classifier[6] = nn.Linear(num_features, self.out_dim)
        elif backbone == 'leaf':
            encoder = nn.Sequential(
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
                nn.Linear(800, self.out_dim)
            )
        else:
            raise NotImplementedError

        self.encoder = encoder
        self.decoder = nn.Linear(z_dim, num_classes)

    @staticmethod
    def re_parameterize(mu, std):
        z = torch.randn(mu.shape[0], mu.shape[1]).cuda()
        return mu + std * z

    def forward(self, x):
        z_params = self.encoder(x)
        z_mu = z_params[:, :self.z_dim]
        z_sigma = F.softplus(z_params[:, self.z_dim:])

        if self.probabilistic:
            if self.training:
                z = self.re_parameterize(z_mu, z_sigma)
                return (z_mu, z_sigma), z, self.decoder(z)
            else:
                return self.decoder(z_mu)
        else:
            if self.training:
                return (z_mu, z_sigma), z_mu, self.decoder(z_mu)
            else:
                return self.decoder(z_mu)

if __name__ == '__main__':
    model = Model(z_dim=256,
                   num_classes=2,
                   backbone='resnet18')
    model.cuda()
    model.train()
    inputs = torch.randn((100, 3, 128, 128)).cuda()
    logits, (mu, sigma) = model(inputs)
    print(logits.shape, mu.shape, sigma.shape)
    model.eval()
    logits = model(inputs)
    print(logits.shape)
