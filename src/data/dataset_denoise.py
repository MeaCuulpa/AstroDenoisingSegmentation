import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset


class NoiseDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, img_size, transform=None):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.img_size = img_size
        self.file_list = os.listdir(noisy_dir)
        self.file_list_im = os.listdir(clean_dir)
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        noisy_path = os.path.join(self.noisy_dir, self.file_list[idx])
        clean_path = os.path.join(self.clean_dir, self.file_list_im[idx])

        # Загрузка изображений
        noisy_img = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)
        clean_img = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)

        # Преобразования
        noisy_img = cv2.resize(noisy_img, (self.img_size, self.img_size))
        clean_img = cv2.resize(clean_img, (self.img_size, self.img_size))

        # Нормализация и преобразование в тензоры
        noisy_img = torch.from_numpy(noisy_img.astype(np.float32) / 255.0).unsqueeze(0)
        clean_img = torch.from_numpy(clean_img.astype(np.float32) / 255.0).unsqueeze(0)

        if self.transform:
            noisy_img = self.transform(noisy_img)
            clean_img = self.transform(clean_img)

        return noisy_img, clean_img