# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import SignLanguageDataset
from model import SignLanguageModel
import os
from tqdm import tqdm
import pandas as pd

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def train():
    data_dir = 'data/train'  # Папка с видеофайлами для обучения
    annotations_file = 'data/annotations.csv'  # Путь к файлу аннотаций

    # Чтение файла аннотаций с указанием разделителя
    annotations = pd.read_csv(annotations_file, sep='\t', on_bad_lines='skip')
    # Фильтруем данные для обучения
    annotations = annotations[annotations['train'] == True]
    # Удаляем возможные пустые строки
    annotations.dropna(subset=['attachment_id', 'text'], inplace=True)

    # Проверяем наличие необходимых столбцов
    required_columns = ['attachment_id', 'text', 'begin', 'end']
    for col in required_columns:
        if col not in annotations.columns:
            print(f"Column '{col}' is missing in the annotations file.")
            return

    # Получаем список уникальных меток
    labels = sorted(annotations['text'].unique())
    labels_dict = {label: idx for idx, label in enumerate(labels)}
    print(f"Labels dictionary: {labels_dict}")

    dataset = SignLanguageDataset(data_dir, annotations_file, labels_dict, num_frames=16)

    print(f"Number of samples in dataset: {len(dataset)}")
    if len(dataset) == 0:
        print("Dataset is empty after processing. Please check your data and annotations.")
        return

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes = len(labels_dict)
    model = SignLanguageModel(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        # Сохранение модели
        if not os.path.exists('models'):
            os.makedirs('models')
        torch.save(model.state_dict(), f'models/sign_language_model_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    train()
