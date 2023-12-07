import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


class Softmax(nn.Module):
    r"""Implement of Softmax (normal classification head):
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
        """
    def __init__(self, in_features=512, out_features=405):
        super(Softmax, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.clf = nn.Linear(in_features, out_features)
        self._initialize_weights()

    def forward(self, x):
        out = self.clf(x)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
