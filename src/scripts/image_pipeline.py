from src.modules.denoise_model import Denoiser
from src.modules.segmentation_model import TinySegmentation
from src.utils.image_processing import load_image, save_image
import torch


class AstroPipeline:
    def __init__(self, denoiser_path, segmentator_path, device='cuda'):
        self.device = device

        # Инициализация моделей
        self.denoiser = Denoiser().to(device)
        self.denoiser.load_state_dict(torch.load(denoiser_path, weights_only=False))
        self.denoiser.eval()

        self.segmentation = TinySegmentation().to(device)
        self.segmentation.load_state_dict(torch.load(segmentator_path, weights_only=False))
        self.segmentation.eval()

    def process(self, image_path, output_path):

        # Загрузка и предобработка
        image, original_size = load_image(image_path)
        image = image.to(self.device)

        # Этап 1: Удаление шумов
        with torch.no_grad():
            denoised = self.denoiser(image)

           # # Этап 2: Сегментация
        with torch.no_grad():
            segmentation_mask = self.segmentation(denoised)

        # Постобработка и сохранение
        save_image(segmentation_mask, original_size, output_path)

        return segmentation_mask