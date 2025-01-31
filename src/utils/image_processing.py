import os
import cv2
import torch
from PIL import Image


def load_image(image_path: str, device: str = "cuda"):
    # Проверка существования файла
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Для цветных: замените на cv2.IMREAD_COLOR
        original_size = img.shape[:2]

        # Предобработка
        img = cv2.resize(img, (128, 128))
        img_tensor = torch.from_numpy(img.astype('float32') / 255).unsqueeze(0).unsqueeze(0).to(device)

        return img_tensor, original_size

    except Exception as e:
        raise RuntimeError(f"Error loading image {image_path}: {str(e)}")


def save_image(tensor: torch.Tensor, original_size: tuple, output_path: str):
    try:
        # Конвертация тензора в numpy array
        if tensor.device != "cpu":
            tensor = tensor.cpu()

        result = tensor.squeeze(0).cpu().numpy()
        #result = cv2.resize(result, (original_size[1], original_size[0]))  # Восстановление исходного размера
        mask_stars = (result[0] * 255).astype('uint8')
        mask_tracks = (result[1] * 255).astype('uint8')

        Image.fromarray(mask_stars).save(output_path + r'/denoised_stars.jpg')
        Image.fromarray(mask_tracks).save(output_path + r'/denoised_tracks.jpg')

    except Exception as e:
        raise RuntimeError(f"Error saving image to {output_path}: {str(e)}")


