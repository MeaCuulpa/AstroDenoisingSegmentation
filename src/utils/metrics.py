import math
import numpy as np
from piq import ssim
from scipy import ndimage
import torch
from torch import nn
import torch.nn.functional as F


def compute_psnr(output, target):
    mse = torch.mean((output - target) ** 2, dim=(1,2,3))
    psnr = 10 * torch.log10(1.0 / (mse + 1e-10))
    return torch.mean(psnr)

# Определяем класс для метрики SSIM
class SSIM(nn.Module):
    def __init__(self, window_size=11, channel=1):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = self.create_window(window_size, channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float()
        _2D_window = _2D_window.unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if self.window.dtype == img1.dtype and self.window.device == img1.device and channel == self.channel:
            window = self.window
        else:
            window = self.create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1 = (0.01 * 1.0)**2  # MAX=1 для нормализованных изображений
        C2 = (0.03 * 1.0)**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()


def calculate_fsim(original, processed):
    """Реализация Feature Similarity Index (FSIM) на основе статьи Zhou et al."""
    # Конвертируем в float32
    original = original.astype(np.float32)
    processed = processed.astype(np.float32)

    # Вычисляем градиенты с помощью фильтра Собеля
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    sobel_y = sobel_x.T

    grad_orig_x = ndimage.convolve(original, sobel_x)
    grad_orig_y = ndimage.convolve(original, sobel_y)
    grad_proc_x = ndimage.convolve(processed, sobel_x)
    grad_proc_y = ndimage.convolve(processed, sobel_y)

    # Вычисляем величину градиентов
    GM_orig = np.sqrt(grad_orig_x ** 2 + grad_orig_y ** 2)
    GM_proc = np.sqrt(grad_proc_x ** 2 + grad_proc_y ** 2)

    # Вычисляем фазовую согласованность
    PC_orig = (GM_orig - np.mean(GM_orig)) / (np.std(GM_orig) + 1e-8)
    PC_proc = (GM_proc - np.mean(GM_proc)) / (np.std(GM_proc) + 1e-8)

    # Вычисляем компоненты FSIM
    S_GM = (2 * GM_orig * GM_proc + 1e-8) / (GM_orig ** 2 + GM_proc ** 2 + 1e-8)
    S_PC = (2 * PC_orig * PC_proc + 1e-8) / (PC_orig ** 2 + PC_proc ** 2 + 1e-8)

    # Объединяем компоненты
    FSIM = np.mean(S_GM * S_PC)
    return FSIM

def normalize_metrics(metrics_dict, max_pixel=255):
    """Нормализует метрики в диапазон [0, 1]"""
    return {
        'SSIM': metrics_dict['SSIM'],
        'FSIM': metrics_dict['FSIM'],
        'PSNR': metrics_dict['PSNR'] / 60.0,
        'RMSE': 1.0 - (metrics_dict['RMSE'] / max_pixel),
        'MAE': 1.0 - (metrics_dict['MAE'] / max_pixel)
    }


def calculate_metrics(original, processed):
    """Вычисляет все метрики качества"""
    orig_np = np.array(original)
    proc_np = np.array(processed)

    metrics = {
        'SSIM': ssim(orig_np, proc_np, data_range=255),
        'FSIM': calculate_fsim(orig_np, proc_np),
        'PSNR': compute_psnr(orig_np, proc_np, data_range=255),
        'RMSE': np.sqrt(np.mean((orig_np - proc_np) ** 2)),
        'MAE': np.mean(np.abs(orig_np - proc_np))
    }

    return normalize_metrics(metrics)
