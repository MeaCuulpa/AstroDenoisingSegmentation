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


def plot_results_segmentation(model, train_losses, val_losses, precisions, recalls, f1_scores, val_dataset, device):
    # Визуализация метрик
    plt.figure(figsize=(15, 5))

    # График потерь
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Curves')
    plt.legend()

    # График метрик
    plt.subplot(1, 3, 2)
    plt.plot(precisions, label='Precision')
    plt.plot(recalls, label='Recall')
    plt.plot(f1_scores, label='F1 Score')
    plt.title('Metrics Evolution')
    plt.legend()

    # Визуализация примеров
    model.eval()
    with torch.no_grad():
        image, mask = val_dataset[0]
        pred = model(image.unsqueeze(0).to(device)).cpu().squeeze()
        pred = (pred > 0.5).float()

    plt.subplot(1, 3, 3)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title('Input Image')
    plt.show()

    # Визуализация масок
    plt.figure(figsize=(15, 5))

    # Исходные маски
    plt.subplot(1, 2, 1)
    plt.imshow(np.argmax(mask.numpy(), axis=0), cmap='jet')
    plt.title('True Mask')

    # Предсказанные маски
    plt.subplot(1, 2, 2)
    plt.imshow(np.argmax(pred.numpy(), axis=0), cmap='jet')
    plt.title('Predicted Mask')
    plt.show()

    # Выберем 3 примера с явными звездами
    sample_indices = [0, 5, 10]  # Можете поменять индексы

    plt.figure(figsize=(15, 10))

    for i, idx in enumerate(sample_indices):
        image, mask = val_dataset[idx]
        pred = model(image.unsqueeze(0).to(device)).cpu().squeeze()
        pred_stars = (pred[0] > 0.5).float().numpy()
        true_stars = mask[0].numpy()

        # Исходное изображение
        plt.subplot(3, 3, i * 3 + 1)
        plt.imshow(image.squeeze(), cmap='gray')
        plt.title(f'Sample {idx}\nInput Image')
        plt.axis('off')

        # Истинные звезды
        plt.subplot(3, 3, i * 3 + 2)
        plt.imshow(true_stars, cmap='viridis', vmin=0, vmax=1)
        plt.title('True Stars Mask')
        plt.axis('off')

        # Предсказанные звезды
        plt.subplot(3, 3, i * 3 + 3)
        plt.imshow(pred_stars, cmap='viridis', vmin=0, vmax=1)
        plt.title('Predicted Stars')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Визуализация с наложением масок
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(sample_indices[:2]):
        image, mask = val_dataset[idx]
        pred = model(image.unsqueeze(0).to(device)).cpu().squeeze()
        pred_stars = (pred[0] > 0.5).float().numpy()

        # Наложение предсказаний на изображение
        plt.subplot(1, 2, i + 1)
        plt.imshow(image.squeeze(), cmap='gray')
        plt.imshow(pred_stars, cmap='spring', alpha=0.3)
        plt.title(f'Stars Overlay (Sample {idx})')
        plt.axis('off')

    plt.tight_layout()
    plt.show()