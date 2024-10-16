import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from datasets import SignLanguageDataset
from model import SignLanguageModel
from tqdm import tqdm
import pandas as pd
import os
import ssl

# Отключаем проверку SSL
ssl._create_default_https_context = ssl._create_unverified_context

def filter_empty_videos(dataset):
    """Фильтруем примеры с пустыми видеофайлами."""
    valid_samples = []
    for sample in dataset.samples:
        video = dataset._load_video(sample['video_path'], sample['begin_frame'], sample['end_frame'])
        if video is not None:
            valid_samples.append(sample)
        else:
            print(f"Skipping {sample['video_path']} due to empty video.")
    dataset.samples = valid_samples

def train():
    data_dir = 'data/train'
    annotations_file = 'data/annotations.csv'

    # Загружаем аннотации и создаем словарь меток
    annotations = pd.read_csv(annotations_file, sep='\t', on_bad_lines='skip')
    annotations = annotations[annotations['train'] == True]
    annotations.dropna(subset=['attachment_id', 'text'], inplace=True)

    # Создаем словарь меток
    labels = sorted(annotations['text'].unique())
    labels_dict = {label: idx for idx, label in enumerate(labels)}

    # Сохраняем словарь меток для инференса
    os.makedirs('models', exist_ok=True)
    with open('models/labels.json', 'w') as f:
        json.dump(labels_dict, f)

    # Создаем датасет и DataLoader
    dataset = SignLanguageDataset(data_dir, annotations_file, labels_dict, num_frames=16)

    print("Фильтрация пустых видео...")
    filter_empty_videos(dataset)

    if len(dataset) == 0:
        print("Dataset is empty after filtering empty videos.")
        return

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Инициализация модели
    model = SignLanguageModel(len(labels_dict)).to(device)

    # Определяем функцию потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

    num_epochs = 20
    best_loss = float('inf')

    try:
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            all_preds, all_labels = [], []

            # Тренировочный цикл
            for inputs, labels in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            # Метрики за эпоху
            epoch_loss = running_loss / len(dataset)
            accuracy = accuracy_score(all_labels, all_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

            print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}')
            print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')

            # Сохранение лучшей модели
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), 'models/best_model.pth')
                print(f"Best model saved with loss {best_loss:.4f}")

            # Обновление learning rate
            scheduler.step(epoch_loss)
            print(f'Current LR: {optimizer.param_groups[0]["lr"]:.8f}')

    except KeyboardInterrupt:
        print("Training interrupted. Saving the model...")
        torch.save(model.state_dict(), 'models/last_model.pth')
        print("Model saved successfully.")

if __name__ == '__main__':
    train()
