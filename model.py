# model.py

import torch.nn as nn
import torchvision.models as models

class SignLanguageModel(nn.Module):
    def __init__(self, num_classes):
        super(SignLanguageModel, self).__init__()
        weights = models.video.R3D_18_Weights.DEFAULT
        self.base_model = models.video.r3d_18(weights=weights)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.base_model(x)
        return x
