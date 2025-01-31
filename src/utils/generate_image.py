from src.utils.load_config import load_config
from src.utils.image_noising import ApplySpaceDefects
import shutil
import numpy as np
import cv2
import random
import os


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
def generate_image_and_mask():
    image = np.zeros((128, 128), dtype=np.uint8)  # Черное изображение
    mask = np.zeros((128, 128), dtype=np.uint8)  # Маска (фон = 0)

    num_objects = random.randint(2, 4)  # Количество объектов
    num_static_stars = random.randint(1, num_objects - 1)  # Часть из них — звезды
    num_moving_objects = num_objects - num_static_stars

    # Генерация объектов
    generate_static_stars(image, mask, num_static_stars)
    generate_moving_objects(image, mask, num_moving_objects)

    # Добавление свечения
    image_with_glow = add_glow(image)

    return image_with_glow, mask


# Генерация датасета изображения и масок
def generate_test_dataset(dataset_size=1000):
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
        image, mask = generate_image_and_mask()
        transorm = ApplySpaceDefects()

        # Сохранение изображения и маски
        image_path = os.path.join(IMAGE_DIR, fr"image_{i}.tiff")
        mask_path = os.path.join(MASKS_DIR, fr"mask_{i}.tiff")
        noisy_image_path = os.path.join(NOISE_DIR, fr"def_{i}.tiff")

        cv2.imwrite(image_path, image)  # Сохраняем изображение
        cv2.imwrite(mask_path, mask)  # Сохраняем маску
        cv2.imwrite(noisy_image_path, transorm.apply_defects(image))


        print(f"Сохранены: {image_path}, {mask_path}, {noisy_image_path}")
