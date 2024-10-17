import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

class SignLanguageDataset(Dataset):
    def __init__(self, data_dir, annotations_file, labels_dict=None, num_frames=16):
        self.data_dir = data_dir
        self.annotations_file = annotations_file
        self.labels_dict = labels_dict
        self.num_frames = num_frames

        # Трансформация для каждого кадра
        self.frame_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
        ])

        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        annotations = pd.read_csv(self.annotations_file, sep='\t', on_bad_lines='skip')
        annotations = annotations[annotations['train'] == True]
        annotations.dropna(subset=['attachment_id', 'text'], inplace=True)

        for _, row in annotations.iterrows():
            video_file = f"{row['attachment_id']}.mp4"
            video_path = os.path.join(self.data_dir, video_file)

            if os.path.isfile(video_path):
                label = self.labels_dict.get(row['text'])
                if label is not None:
                    samples.append({
                        'video_path': video_path,
                        'label': label,
                        'begin_frame': int(row['begin']),
                        'end_frame': int(row['end'])
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
            return None  # Возвращаем None вместо пустого массива

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

            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

            frame = self.frame_transform(frame)
            frames.append(frame)
            frame_idx += 1

        cap.release()

        if len(frames) == 0:
            print(f"Warning: No frames extracted from {video_path}")
            return None  # Возвращаем None, если кадры не были извлечены

        frames = torch.stack(frames)  # (Frames, Channels, Height, Width)

        if frames.size(0) < self.num_frames:
            pad_size = self.num_frames - frames.size(0)
            last_frame = frames[-1]
            padding = last_frame.unsqueeze(0).repeat(pad_size, 1, 1, 1)
            frames = torch.cat([frames, padding], dim=0)
        elif frames.size(0) > self.num_frames:
            indices = torch.linspace(0, frames.size(0) - 1, self.num_frames).long()
            frames = frames[indices]

        frames = frames.permute(1, 0, 2, 3)  # (Channels, Frames, Height, Width)
        return frames

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video = self._load_video(sample['video_path'], sample['begin_frame'], sample['end_frame'])

        if video is None:
            print(f"Skipping sample at index {idx} due to empty video.")
            idx = (idx + 1) % len(self.samples)
            return self.__getitem__(idx)  # Рекурсивно пробуем следующий элемент

        label = sample['label']
        return video, label
