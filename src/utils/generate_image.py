from src.utils.load_config import load_config
from src.utils.image_noising import ApplySpaceDefects
import shutil
import numpy as np
import cv2
import random
import os
from PIL import Image


# Функция для создания кругов (звезд)
def generate_static_stars(image, mask, count):
    for _ in range(count):
        while True:
            # Случайный центр и радиус
            x, y = random.randint(0, image.shape[1] - 1), random.randint(0, image.shape[0] - 1)
            radius = random.randint(5, 15)  # Диаметр от 10 до 20
            if 4 * np.pi * radius ** 2 / 4 >= 4:  # Проверка площади
                cv2.circle(image, (x, y), radius, 255, -1)  # Рисуем звезду
                cv2.circle(mask, (x, y), radius, 1, -1)  # Отмечаем на маске как класс 1
                break


# Функция для создания движущихся объектов (линий)
def generate_moving_objects(image, mask, count):
    for _ in range(count):
        while True:
            # Случайные точки для линии
            x1, y1 = random.randint(0, image.shape[1] - 1), random.randint(0, image.shape[0] - 1)
            x2, y2 = random.randint(0, image.shape[1] - 1), random.randint(0, image.shape[0] - 1)
            thickness = random.randint(2, 6)  # Толщина линии
            length = int(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
            if 20 <= length <= 50 and thickness * length >= 4:  # Проверка длины и площади
                cv2.line(image, (x1, y1), (x2, y2), 255, thickness)  # Рисуем линию
                cv2.line(mask, (x1, y1), (x2, y2), 2, thickness)  # Отмечаем на маске как класс 2
                break


# Функция для добавления свечения
def add_glow(image):
    # Применение размытия по Гауссу
    glow = cv2.GaussianBlur(image, (0, 0), sigmaX=10, sigmaY=10)
    # Нормализация свечения
    glow = (glow / glow.max() * 255).astype(np.uint8)
    # Объединение оригинального изображения и свечения
    combined = cv2.addWeighted(glow, 0.6, image, 1.0, 0)
    return combined


# Генерация изображения и маски
def generate_image_and_mask(shape=(128, 128)):
    image = np.zeros(shape, dtype=np.uint8)  # Черное изображение
    mask = np.zeros(shape, dtype=np.uint8)  # Маска (фон = 0)

    num_objects = random.randint(5, 10)  # Количество объектов
    num_static_stars = random.randint(1, num_objects - 1)  # Часть из них — звезды
    num_moving_objects = num_objects - num_static_stars

    # Генерация объектов
    generate_static_stars(image, mask, num_static_stars)
    generate_moving_objects(image, mask, num_moving_objects)

    # Добавление свечения
    image_with_glow = add_glow(image)

    return image_with_glow, mask


# Генерация датасета изображения и масок
def generate_test_dataset(img_size, dataset_size=1000):
    # Использование конфига
    config = load_config(r'../config/config.yaml')

    # Пути к данным
    MASKS_DIR = config['paths']['masks_data_dir']
    IMAGE_DIR = config['paths']['input_data_dir']
    NOISE_DIR = config['paths']['noised_data_dir']

    shutil.rmtree(MASKS_DIR)
    os.mkdir(MASKS_DIR)

    shutil.rmtree(IMAGE_DIR)
    os.mkdir(IMAGE_DIR)

    shutil.rmtree(NOISE_DIR)
    os.mkdir(NOISE_DIR)

    for i in range(dataset_size):  # Сгенерировать набор данных
        image, mask = generate_image_and_mask(img_size)
        transorm = ApplySpaceDefects()

        # Сохранение изображения и маски
        image_path = os.path.join(IMAGE_DIR, fr"image_{i}.tiff")
        mask_path = os.path.join(MASKS_DIR, fr"mask_{i}.tiff")
        noisy_image_path = os.path.join(NOISE_DIR, fr"def_{i}.tiff")

        cv2.imwrite(image_path, image)  # Сохраняем изображение
        cv2.imwrite(mask_path, mask)  # Сохраняем маску
        cv2.imwrite(noisy_image_path, transorm.apply_defects(image))


        print(f"Сохранены: {image_path}, {mask_path}, {noisy_image_path}")


def generate_coordinates(width, height, patch_size, stride):
    """Генерирует координаты для нарезки изображения с перекрытием"""
    x_coords = []
    y_coords = []

    # Генерация координат по X
    x = 0
    while x + patch_size <= width:
        x_coords.append(x)
        x += stride
    if x_coords and (x_coords[-1] + patch_size) < width:
        x_coords.append(max(0, width - patch_size))

    # Генерация координат по Y
    y = 0
    while y + patch_size <= height:
        y_coords.append(y)
        y += stride
    if y_coords and (y_coords[-1] + patch_size) < height:
        y_coords.append(max(0, height - patch_size))

    # Создаем все комбинации координат
    return [(x, y) for x in x_coords for y in y_coords]


def split_into_patches(raw_images_dir, noised_images_dir, patch_size=256, stride=128):
    """Основная функция для нарезки изображений на патчи"""

    # Создаем выходные директории
    output_raw = os.path.join(os.path.dirname(raw_images_dir), 'raw_patches')
    output_noised = os.path.join(os.path.dirname(noised_images_dir), 'noised_patches')
    os.makedirs(output_raw, exist_ok=True)
    os.makedirs(output_noised, exist_ok=True)

    # Обрабатываем 1000 файлов
    for i in range(0, 9):
        raw_file = f"image_{i}.tiff"
        noised_file = f"def_{i}.tiff"

        raw_path = os.path.join(raw_images_dir, raw_file)
        noised_path = os.path.join(noised_images_dir, noised_file)

        # Проверяем, существуют ли оба файла
        if not os.path.exists(raw_path) or not os.path.exists(noised_path):
            print(f"Файлы для {i} не найдены. Пропускаем.")
            continue

        try:
            with Image.open(raw_path) as raw_img, Image.open(noised_path) as noised_img:
                if raw_img.size != noised_img.size:
                    print(f"Размеры изображений для {i} не совпадают. Пропускаем.")
                    continue

                width, height = raw_img.size
                if width < patch_size or height < patch_size:
                    print(f"Изображение для {i} слишком маленькое. Пропускаем.")
                    continue

                # Генерируем координаты патчей
                coords = generate_coordinates(width, height, patch_size, stride)

                # Сохраняем патчи
                base_name = f"image_{i}"  # Используем имя из raw_images
                for x, y in coords:
                    box = (x, y, x + patch_size, y + patch_size)
                    raw_patch = raw_img.crop(box)
                    noised_patch = noised_img.crop(box)

                    patch_name = f"{base_name}_x{x}_y{y}.png"
                    raw_patch.save(os.path.join(output_raw, patch_name))
                    noised_patch.save(os.path.join(output_noised, patch_name))

                print(f"Обработан {i}: {len(coords)} патчей")

        except Exception as e:
            print(f"Ошибка обработки {i}: {str(e)}")

    print("Нарезка изображений завершена!")


# Пример использования
# split_into_patches(
#     raw_images_dir='E:\\AstroDenoisingSegmentation\\data\\raw_images',
#     noised_images_dir='E:\\AstroDenoisingSegmentation\\data\\noised_images',
#     patch_size=256,
#     stride=128
# )

#generate_test_dataset((256, 256))
