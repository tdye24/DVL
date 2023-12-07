import torch
import torchvision.models as models

def get_resnet18(pretrained=True, num_classes=2):
    model = models.resnet18(pretrained=pretrained)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, num_classes)
    return model

# if __name__ == '__main__':
#     m = get_resnet18(pretrained=True)
#     print(m(torch.randn([1, 3, 84, 84])).shape)
#     from torchsummary import summary
#     summary(m.cuda(), (3, 84, 84))