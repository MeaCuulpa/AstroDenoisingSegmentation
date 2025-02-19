from src.utils.load_config import load_config
from src.utils.generate_image import generate_image_and_mask
from src.scripts.image_pipeline import AstroPipeline
from src.utils.image_noising import ApplySpaceDefects
import os
from skimage import exposure
import numpy as np
import cv2


def main(denoiser_path: str, segmentation_path: str, image_path: str, output_path: str):

    pipeline = AstroPipeline(denoiser_path, segmentation_path)
    pipeline.process(
        image_path,
        output_path,
    )


if __name__ == "__main__":

    # Использование конфига
    config = load_config(r'../config/config.yaml')

    # Пути к данным
    TEST_DIR = config['paths']['test_data_dir']
    PROCESSED_DIR = config['paths']['processed_data_dir']
    REAL_DIR = config['paths']['real_data_dir']
    denoiser_path = config['paths']['denoiser_model']
    segmentation_path = config['paths']['segmentation_model']

    # Заглушка для генерации изображения (заменяется на реальное)
    # Генерируем изображение
    #image, _ = generate_image_and_mask((2000, 2000))

    #image = cv2.imread(r"..\..\data\test_images\noised.png")

    #image = exposure.equalize_hist(image)
    #image = (image * 255).astype(np.uint8)


    # Накладываем на изображение дефекты
    #transform = ApplySpaceDefects()
    #image = transform.apply_defects(image)

    # Сохраняем изображение
    #image_path = os.path.join(TEST_DIR, fr"test_image.tiff")
    #cv2.imwrite(image_path, image)

    main(denoiser_path, segmentation_path, r"..\..\data\test_images\noised.png", PROCESSED_DIR)
