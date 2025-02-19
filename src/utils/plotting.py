import torch
import numpy as np
from matplotlib import pyplot as plt
from src.utils.load_config import load_config

config = load_config(r'../config/config.yaml')

DEVICE = config['hardware']['device']


# Визуализация результатов
def plot_results_noise(model, loader, device=DEVICE, num_samples=3):
    model.eval()
    with torch.no_grad():
        noisy, clean = next(iter(loader))
        noisy = noisy[:num_samples].to(device)
        clean = clean[:num_samples].to(device)
        outputs = model(noisy)

        plt.figure(figsize=(15, 5))
        for i in range(num_samples):
            plt.subplot(3, num_samples, i + 1)
            plt.imshow(noisy[i].squeeze().cpu().numpy(), cmap='gray')
            plt.title('Noisy')

            plt.subplot(3, num_samples, i + 1 + num_samples)
            plt.imshow(outputs[i].squeeze().cpu().numpy(), cmap='gray')
            plt.title('Denoised')

            plt.subplot(3, num_samples, i + 1 + 2 * num_samples)
            plt.imshow(clean[i].squeeze().cpu().numpy(), cmap='gray')
            plt.title('Original')
        plt.tight_layout()
        plt.show()


def plot_results_segmentation(model, train_losses, val_losses, precisions, recalls, f1_scores, ious, val_dataset,
                              device):
    """
    Визуализация результатов обучения модели сегментации с улучшенным оформлением.
    """
    # Настройка стилей
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'ggplot')
    plt.rcParams.update({
        'font.size': 12,
        'figure.figsize': (15, 8),
        'axes.titlesize': 14,
        'axes.titlepad': 20,
        'lines.linewidth': 2,
        'lines.markersize': 8
    })

    # График лосса (потерь)
    plt.figure(figsize=(15, 6))
    plt.plot(train_losses, label='Train Loss', color='#1f77b4', marker='o')
    plt.plot(val_losses, label='Val Loss', color='#ff7f0e', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Графики метрик (Precision, Recall, F1 Score, IoU)
    metrics_config = [
        (precisions, 'Precision', '#2ca02c'),
        (recalls, 'Recall', '#d62728'),
        (f1_scores, 'F1 Score', '#9467bd'),
        (ious, 'IoU', '#17becf')
    ]

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))  # 2x2 сетка для метрик
    axs = axs.ravel()  # Преобразуем в одномерный массив для удобства

    for i, (values, title, color) in enumerate(metrics_config):
        axs[i].plot(values, color=color, marker='s')
        axs[i].set_title(f'{title} Evolution')
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel(title)
        axs[i].set_ylim(0, 1.05)  # Добавляем небольшой отступ сверху
        axs[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Визуализация примеров сегментации
    model.eval()
    with torch.no_grad():
        sample_indices = [0, 5, 10]

        for idx in sample_indices:
            image, mask = val_dataset[idx]
            pred = model(image.unsqueeze(0).to(device)).cpu().squeeze()
            pred = (pred > 0.5).float()

            # Создаем фигуру для примера
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # Цветовая схема для масок
            cmap = plt.cm.viridis
            mask_kwargs = {
                'cmap': cmap,
                'alpha': 0.4,
                'vmin': 0,
                'vmax': 1 if mask.max() <= 1 else mask.max()
            }

            # Исходное изображение
            axes[0].imshow(image.squeeze(), cmap='gray')
            axes[0].set_title(f'Sample {idx}\nInput Image')
            axes[0].axis('off')

            # Истинная маска
            true_mask = mask.argmax(dim=0) if mask.ndim == 3 else mask.squeeze()
            im = axes[1].imshow(true_mask, **mask_kwargs)
            fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            axes[1].set_title('True Mask')
            axes[1].axis('off')

            # Предсказанная маска
            pred_mask = pred.argmax(dim=0) if pred.ndim == 3 else pred.squeeze()
            im = axes[2].imshow(pred_mask, **mask_kwargs)
            fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
            axes[2].set_title('Predicted Mask')
            axes[2].axis('off')

            plt.tight_layout()
            plt.show()

def plot_metrics_noisy(train_loss_history, val_loss_history, train_rmse_history, val_rmse_history,
                 train_psnr_history, val_psnr_history, train_ssim_history, val_ssim_history,
                 train_fsim_history, val_fsim_history):
    """
    Функция для построения графиков метрик и лосса.
    """
    epochs = range(1, len(train_loss_history) + 1)

    # График Loss
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_loss_history, label='Train Loss', marker='o')
    plt.plot(epochs, val_loss_history, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # График RMSE
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_rmse_history, label='Train RMSE', marker='o')
    plt.plot(epochs, val_rmse_history, label='Validation RMSE', marker='o')
    plt.title('Training and Validation RMSE')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    plt.show()

    # График PSNR
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_psnr_history, label='Train PSNR', marker='o')
    plt.plot(epochs, val_psnr_history, label='Validation PSNR', marker='o')
    plt.title('Training and Validation PSNR')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # График SSIM
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_ssim_history, label='Train SSIM', marker='o')
    plt.plot(epochs, val_ssim_history, label='Validation SSIM', marker='o')
    plt.title('Training and Validation SSIM')
    plt.xlabel('Epochs')
    plt.ylabel('SSIM')
    plt.legend()
    plt.grid(True)
    plt.show()

    # График FSIM
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_fsim_history, label='Train FSIM', marker='o')
    plt.plot(epochs, val_fsim_history, label='Validation FSIM', marker='o')
    plt.title('Training and Validation FSIM')
    plt.xlabel('Epochs')
    plt.ylabel('FSIM')
    plt.legend()
    plt.grid(True)
    plt.show()
