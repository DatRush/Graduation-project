# model.py

import torch.nn as nn
import torchvision.models as models

class SignLanguageModel(nn.Module):
    def __init__(self, num_classes):
        super(SignLanguageModel, self).__init__()
        
        # Загружаем предобученную модель R3D-18 с весами
        weights = models.video.R3D_18_Weights.DEFAULT
        self.base_model = models.video.r3d_18(weights=weights)

        # Замораживаем все параметры (градиенты не будут обновляться)
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Размораживаем последний блок (например, слой layer4)
        for param in self.base_model.layer4.parameters():
            param.requires_grad = True

        # Настраиваем финальный полносвязный слой на нужное число классов
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.5),  # Dropout для регуляризации
            nn.Linear(self.base_model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)
