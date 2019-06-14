from __future__ import print_function

import torch.nn as nn


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)


class LinearClassifierAlexNet(nn.Module):
    def __init__(self, layer=5, n_label=1000, pool_type='max'):
        super(LinearClassifierAlexNet, self).__init__()
        if layer == 1:
            pool_size = 10
            nChannels = 96
        elif layer == 2:
            pool_size = 6
            nChannels = 256
        elif layer == 3:
            pool_size = 5
            nChannels = 384
        elif layer == 4:
            pool_size = 5
            nChannels = 384
        elif layer == 5:
            pool_size = 6
            nChannels = 256
        else:
            raise NotImplementedError('layer not supported: {}'.format(layer))

        self.classifier = nn.Sequential()

        if layer < 5:
            if pool_type == 'max':
                self.classifier.add_module('MaxPool', nn.AdaptiveMaxPool2d((pool_size, pool_size)))
            elif pool_type == 'avg':
                self.classifier.add_module('AvgPool', nn.AdaptiveAvgPool2d((pool_size, pool_size)))

        self.classifier.add_module('Flatten', Flatten())
        self.classifier.add_module('LinearClassifier', nn.Linear(nChannels*pool_size*pool_size, n_label))
        self.initilize()

    def initilize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.0)

    def forward(self, x):
        return self.classifier(x)
