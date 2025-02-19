from src.modules.denoise_model import Denoiser
from src.modules.segmentation_model import TinySegmentation
from src.utils.image_processing import load_image, save_image
import torch
from PIL import Image
from torchvision import transforms


def split_image(image, patch_size=256, stride=128):
    """
    Нарезает изображение на патчи с перекрытием.
    Возвращает список патчей, координаты их расположения, а также ширину и высоту исходного изображения.
    """
    width, height = image.size
    patches = []
    coordinates = []

    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            box = (x, y, x + patch_size, y + patch_size)
            patch = image.crop(box)
            patches.append(patch)
            coordinates.append((x, y))

    # Добавляем патчи для краёв, если необходимо
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


def merge_patches(processed_patches, coordinates, width, height):
    """
    Сшивает патчи обратно в изображение.
    Для упрощения создаётся изображение в режиме 'L' (одноканальное).
    """
    merged_image = Image.new('L', (width, height))
    for patch, (x, y) in zip(processed_patches, coordinates):
        merged_image.paste(patch, (x, y))
    return merged_image


class AstroPipeline:
    def __init__(self, denoiser_path, segmentator_path, device='cuda',
                 use_denoising=True, use_segmentation=True, use_patching=False,
                 patch_size=256, stride=128):
        """
        Инициализация пайплайна.

        :param denoiser_path: путь к весам модели денойзера
        :param segmentator_path: путь к весам модели сегментации
        :param device: устройство для вычислений ('cuda' или 'cpu')
        :param use_denoising: флаг, указывающий, выполнять ли денойзинг
        :param use_segmentation: флаг, указывающий, выполнять ли сегментацию
        :param use_patching: флаг, указывающий, обрабатывать ли изображение патчами (для больших изображений)
        :param patch_size: размер патча
        :param stride: шаг при нарезке патчей
        """
        self.device = device
        self.use_denoising = use_denoising
        self.use_segmentation = use_segmentation
        self.use_patching = use_patching
        self.patch_size = patch_size
        self.stride = stride

        if self.use_denoising:
            self.denoiser = Denoiser().to(device)
            self.denoiser.load_state_dict(torch.load(denoiser_path, weights_only=False))
            self.denoiser.eval()

        if self.use_segmentation:
            self.segmentation = TinySegmentation().to(device)
            self.segmentation.load_state_dict(torch.load(segmentator_path, weights_only=False))
            self.segmentation.eval()

    def process(self, image_path, output_path):
        """
        Обрабатывает изображение с учетом установленных флагов.

        Если use_denoising=True, то сначала выполняется денойзинг, и затем сегментация (если активирована)
        выполняется на очищенном изображении.

        При use_patching=True изображение разбивается на патчи, обрабатывается и затем патчи сшиваются обратно.
        """
        transform = transforms.ToTensor()

        if self.use_patching:
            # Загружаем изображение через PIL для патчинга
            image_pil = Image.open(image_path).convert('L')
            patches, coordinates, width, height = split_image(image_pil, self.patch_size, self.stride)

            processed_patches = []
            # Если включен денойзинг, то обрабатываем патчи через денойзер
            if self.use_denoising:
                for patch in patches:
                    patch_tensor = transform(patch).unsqueeze(0).to(self.device)  # [1, C, H, W]
                    with torch.no_grad():
                        denoised_tensor = self.denoiser(patch_tensor)
                    processed_patches.append(denoised_tensor)
                # Сохраняем итоговое очищенное изображение
                denoised_pil_patches = [transforms.ToPILImage()(pt.squeeze(0).cpu()) for pt in processed_patches]
                denoised_image = merge_patches(denoised_pil_patches, coordinates, width, height)
                denoised_image.save(output_path + '/denoised.jpg')
            else:
                # Если денойзинг не используется, используем исходные патчи
                processed_patches = [transform(patch).unsqueeze(0).to(self.device) for patch in patches]

            # Сегментация проводится на изображениях после денойзинга, если она активна
            if self.use_segmentation:
                seg_patches_class0 = []
                seg_patches_class1 = []
                for patch_tensor in processed_patches:
                    with torch.no_grad():
                        seg_output = self.segmentation(patch_tensor)
                    # Предполагается, что модель сегментации возвращает тензор формы [1, 2, H, W]
                    out0 = seg_output[:, 0:1, :, :]
                    out1 = seg_output[:, 1:2, :, :]
                    out0_img = transforms.ToPILImage()(out0.squeeze(0).cpu())
                    out1_img = transforms.ToPILImage()(out1.squeeze(0).cpu())
                    seg_patches_class0.append(out0_img)
                    seg_patches_class1.append(out1_img)
                seg_image_class0 = merge_patches(seg_patches_class0, coordinates, width, height)
                seg_image_class1 = merge_patches(seg_patches_class1, coordinates, width, height)
                seg_image_class0.save(output_path + '/segmentation_class0.jpg')
                seg_image_class1.save(output_path + '/segmentation_class1.jpg')
        else:
            # Обработка целого изображения (без патчинга)
            image_tensor, original_size = load_image(image_path)
            image_tensor = image_tensor.to(self.device)

            # Если включён денойзинг, обрабатываем изображение и сохраняем результат
            if self.use_denoising:
                with torch.no_grad():
                    denoised = self.denoiser(image_tensor)
                denoised_cpu = denoised.cpu()
                result = denoised_cpu.squeeze(0).squeeze(0).numpy()
                result = (result * 255).astype('uint8')
                Image.fromarray(result).save(output_path + '/denoised.jpg')
                segmentation_input = denoised  # Сегментация проводится на очищенном изображении
            else:
                segmentation_input = image_tensor

            # Сегментация, если активирована
            if self.use_segmentation:
                with torch.no_grad():
                    segmentation_output = self.segmentation(segmentation_input)
                # segmentation_output имеет форму [1, 2, H, W]
                seg0 = segmentation_output[:, 0:1, :, :]
                seg1 = segmentation_output[:, 1:2, :, :]
                seg0_img = transforms.ToPILImage()(seg0.squeeze(0).cpu())
                seg1_img = transforms.ToPILImage()(seg1.squeeze(0).cpu())
                seg0_img.save(output_path + '/segmentation_class0.jpg')
                seg1_img.save(output_path + '/segmentation_class1.jpg')

        return
