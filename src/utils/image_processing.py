import os
import cv2
import torch
from PIL import Image
from torchvision import transforms
from src.modules.denoise_model import Denoiser
from src.utils.metrics import calculate_metrics


def load_image(image_path: str, device: str = "cuda"):
    # Проверка существования файла
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Для цветных: замените на cv2.IMREAD_COLOR
        original_size = img.shape[:2]

        # Предобработка
        img = cv2.resize(img, (2000, 2000))
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
        print(tensor.shape)
        #result = cv2.resize(result, (original_size[1], original_size[0]))  # Восстановление исходного размера
        mask_stars = (result[0] >= 0.5).astype('uint8') * 255
        mask_tracks = (result[1] >= 0.5).astype('uint8') * 255

        Image.fromarray(mask_stars).save(output_path + r'/denoised_stars.jpg')
        Image.fromarray(mask_tracks).save(output_path + r'/denoised_tracks.jpg')

    except Exception as e:
        raise RuntimeError(f"Error saving image to {output_path}: {str(e)}")


def split_image(image, patch_size=256, stride=128):
    """Нарезает изображение на патчи с перекрытием"""
    width, height = image.size
    patches = []
    coordinates = []

    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            box = (x, y, x + patch_size, y + patch_size)
            patch = image.crop(box)
            patches.append(patch)
            coordinates.append((x, y))

    # Добавляем патчи для краев, если необходимо
    if (width - patch_size) % stride != 0:
        for y in range(0, height - patch_size + 1, stride):
            x = width - patch_size
            box = (x, y, x + patch_size, y + patch_size)
            patch = image.crop(box)
            patches.append(patch)
            coordinates.append((x, y))

    if (height - patch_size) % stride != 0:
        for x in range(0, width - patch_size + 1, stride):
            y = height - patch_size
            box = (x, y, x + patch_size, y + patch_size)
            patch = image.crop(box)
            patches.append(patch)
            coordinates.append((x, y))

    if (width - patch_size) % stride != 0 and (height - patch_size) % stride != 0:
        x = width - patch_size
        y = height - patch_size
        box = (x, y, x + patch_size, y + patch_size)
        patch = image.crop(box)
        patches.append(patch)
        coordinates.append((x, y))

    return patches, coordinates, width, height


def process_patches(model, patches, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Обрабатывает патчи через модель"""
    transform = transforms.ToTensor()
    processed_patches = []

    for patch in patches:
        patch_tensor = transform(patch).unsqueeze(0).to(device)  # [1, C, H, W]
        with torch.no_grad():
            output_tensor = model(patch_tensor)
        output_patch = transforms.ToPILImage()(output_tensor.squeeze(0).cpu())
        processed_patches.append(output_patch)

    return processed_patches


def merge_patches(processed_patches, coordinates, width, height):
    """Сшивает патчи обратно в изображение"""
    merged_image = Image.new('L', (width, height))  # Создаем пустое изображение
    for patch, (x, y) in zip(processed_patches, coordinates):
        merged_image.paste(patch, (x, y))
    return merged_image


def denoise_image(model_path, input_image_path, output_path, ground_truth_path,
                  patch_size=256, stride=128):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Используемое устройство: {device}")

    # Загрузка модели
    model = Denoiser()
    model.load_state_dict(torch.load(model_path))
    model.to('cuda')
    model.eval()  # Переводим модель в режим оценки

    # Загрузка изображений
    noisy_image = Image.open(input_image_path).convert('L')
    ground_truth = Image.open(ground_truth_path).convert('L')

    if noisy_image.size != ground_truth.size:
        raise ValueError("Размеры изображений не совпадают!")

    # Обработка изображения
    patches, coords, w, h = split_image(noisy_image, patch_size, stride)
    processed_patches = process_patches(model, patches, device)
    merged_image = merge_patches(processed_patches, coords, w, h)

    # Сохранение результата
    merged_image.save(output_path)

    # Вычисление метрик
    metrics = calculate_metrics(ground_truth, merged_image)

    print("\nНормированные метрики качества:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    return metrics
