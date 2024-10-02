import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AtariNetDuelingDQN(nn.Module):
    def __init__(self, num_classes=4, init_weights=True):
        super(AtariNetDuelingDQN, self).__init__()
        self.cnn = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4),
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                        nn.ReLU(True),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                        nn.ReLU(True)
                                        )
        self.extractor = nn.Sequential(nn.Linear(7*7*64, 512),
                                        nn.ReLU(True)
                                        )
        self.value_network = nn.Sequential(nn.Linear(512, 1))
        self.advantage_network = nn.Sequential(nn.Linear(512, num_classes))

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = x.float() / 255.
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.extractor(x)
        value = self.value_network(x)
        advantage = self.advantage_network(x)
        q_value = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_value

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)

