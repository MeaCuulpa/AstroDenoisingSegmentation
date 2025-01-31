import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

# Кастомный датасет для загрузки реальных данных
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Загрузка изображения
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = Image.open(image_path).convert('L')
        mask = Image.open(mask_path)

        # Конвертация в numpy arrays
        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.uint8)

        # Преобразование маски в two-hot encoding (2 канала)
        mask_two_hot = np.zeros((2, 128, 128), dtype=np.float32)
        mask_two_hot[0, :, :] = (mask == 1).astype(np.float32)  # Звезды
        mask_two_hot[1, :, :] = (mask == 2).astype(np.float32)  # Треки

        return torch.tensor(image).unsqueeze(0), torch.tensor(mask_two_hot)