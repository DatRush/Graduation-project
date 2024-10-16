import cv2
import torch
import numpy as np
from model import SignLanguageModel
import torchvision.transforms as transforms
import json

def load_labels(label_file='models/labels.json'):
    """Загружает метки из файла JSON."""
    with open(label_file, 'r') as f:
        labels_dict = json.load(f)
    # Инвертируем словарь для поиска по индексу
    return {v: k for k, v in labels_dict.items()}

def preprocess_frame(frame):
    """
    Предобрабатывает кадр: изменяет размер и преобразует в тензор.
    """
    frame = cv2.resize(frame, (112, 112))  # Изменяем размер
    frame = torch.tensor(frame).permute(2, 0, 1).float() / 255.0  # Преобразуем в формат C, H, W
    return frame

def real_time_inference():
    """Выполняет инференс в реальном времени."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Загрузка меток
    labels_dict = load_labels()
    print(f"Loaded labels: {labels_dict}")

    # Инициализация модели и загрузка весов
    num_classes = len(labels_dict)
    model = SignLanguageModel(num_classes).to(device)
    model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
    model.eval()

    # Настраиваем камеру
    cap = cv2.VideoCapture(0)

    # Список для накопления кадров
    frames = []
    num_frames = 16  # Количество кадров для предсказания

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Ошибка захвата кадра.")
                break

            # Предобработка и накопление кадров
            preprocessed_frame = preprocess_frame(frame)
            frames.append(preprocessed_frame)

            # Выполняем инференс каждые 16 кадров
            if len(frames) == num_frames:
                input_frames = torch.stack(frames).unsqueeze(0).to(device)  # B, T, C, H, W
                frames = []  # Очищаем список кадров

                # Предсказание
                outputs = model(input_frames)
                _, pred = torch.max(outputs, 1)

                # Получаем предсказанный жест
                predicted_label = labels_dict[pred.item()]
                print(f"Predicted: {predicted_label}")

                # Выводим на экран предсказание
                cv2.putText(frame, f'Prediction: {predicted_label}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Отображаем кадр с предсказанием
            cv2.imshow('Real-Time Sign Language Recognition', frame)

            # Выход при нажатии 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    real_time_inference()
