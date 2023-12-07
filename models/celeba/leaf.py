import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, image_size=84, num_classes=2):
        super(Model, self).__init__()
        self.conv_layers = nn.Sequential(
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
            nn.ReLU()
        )
        self.fc_input_size = 32 * (image_size // 16) * (image_size // 16)
        self.fc = nn.Linear(self.fc_input_size, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, self.fc_input_size)
        x = self.fc(x)
        return x


# if __name__ == '__main__':
#     model = Model(image_size=84)
#     out = model(torch.randn((100, 3, 84, 84)))
#     print(out.shape)
#     from torchsummary import summary
#     summary(model.cuda(), (3, 84, 84))
