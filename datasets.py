import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class SignLanguageDataset(Dataset):
    def __init__(self, data_dir, annotations_file, labels_dict=None, transform=None, num_frames=16):
        self.data_dir = data_dir
        self.annotations_file = annotations_file
        self.transform = transform
        self.labels_dict = labels_dict
        self.num_frames = num_frames  # Добавлено: сохранение параметра num_frames
        self.samples = self._load_samples()
    
    def _load_samples(self):
        samples = []
        # Читаем файл аннотаций с указанием разделителя
        annotations = pd.read_csv(self.annotations_file, sep='\t', on_bad_lines='skip')
        # Фильтруем данные для обучения
        annotations = annotations[annotations['train'] == True]
        # Удаляем возможные пустые строки
        annotations.dropna(subset=['attachment_id', 'text'], inplace=True)

        for idx, row in annotations.iterrows():
            attachment_id = row['attachment_id']
            label_name = row['text']
            video_file = f"{attachment_id}.mp4"  # Предполагаем, что расширение .mp4
            video_path = os.path.join(self.data_dir, video_file)
            if os.path.isfile(video_path):
                if self.labels_dict is not None:
                    label = self.labels_dict[label_name]
                else:
                    label = label_name
                begin_frame = int(row['begin'])
                end_frame = int(row['end'])
                samples.append({
                    'video_path': video_path,
                    'label': label,
                    'begin_frame': begin_frame,
                    'end_frame': end_frame
                })
            else:
                print(f"Warning: Video file {video_path} does not exist.")
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_video(self, video_path, begin_frame=None, end_frame=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Cannot open video file {video_path}")
            return np.array([])
        frames = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if begin_frame is not None and frame_idx < begin_frame:
                frame_idx += 1
                continue
            if end_frame is not None and frame_idx > end_frame:
                break
            frame = cv2.resize(frame, (112, 112))
            frames.append(frame)
            frame_idx += 1
        cap.release()
        frames = np.array(frames)

        # Обеспечиваем, что видео имеет нужное количество кадров
        frames = self._process_frames(frames)
        return frames

    def _process_frames(self, frames):
        total_frames = frames.shape[0]
        if total_frames > self.num_frames:
            # Выбираем кадры равномерно по всему видео
            indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
            frames = frames[indices]
        elif total_frames < self.num_frames:
            # Дополняем последним кадром
            pad_size = self.num_frames - total_frames
            pad_frames = np.tile(frames[-1:], (pad_size, 1, 1, 1))
            frames = np.concatenate((frames, pad_frames), axis=0)
        # Если total_frames == self.num_frames, ничего не делаем
        return frames

    def __getitem__(self, idx):
        sample = self.samples[idx]
        begin_frame = sample.get('begin_frame')
        end_frame = sample.get('end_frame')
        video = self._load_video(sample['video_path'], begin_frame, end_frame)
        if video.size == 0:
            # Если видео не удалось загрузить, пропускаем его
            idx = (idx + 1) % len(self.samples)
            return self.__getitem__(idx)
        label = sample['label']

        if self.transform:
            video = self.transform(video)

        video = torch.tensor(video).permute(3, 0, 1, 2) / 255.0  # C, T, H, W
        return video.float(), label
