from src.modules.denoise_model import Denoiser
from src.data.dataset_denoise import NoiseDataset
from src.utils.load_config import load_config
from src.utils.metrics import SSIM, compute_psnr
from src.utils.plotting import plot_results_noise, plot_metrics_noisy
import os
import torch
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from piq import fsim


# Использование конфига
config = load_config(r'../config/config.yaml')

# Конфигурация
IMG_SIZE = config['image']['img_size']
CHANNELS = config['image']['channels']
BATCH_SIZE = config['hardware']['batch_size']
DEVICE = torch.device(config['hardware']['device'])
EPOCHS = config['training']['epochs']
LR = config['training']['learning_rate']

# Пути к данным
NOISY_DIR = config['paths']['noised_data_dir']
CLEAN_DIR = config['paths']['input_data_dir']

# Проверка наличия данных
assert len(os.listdir(NOISY_DIR)) == len(os.listdir(CLEAN_DIR)), "Несоответствие количества файлов"

# Инициализация модели
model = Denoiser(in_channels=CHANNELS, out_channels=CHANNELS).to(DEVICE)

# Разделение данных
dataset = NoiseDataset(NOISY_DIR, CLEAN_DIR, IMG_SIZE)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Функция потерь и оптимизатор
def criterion(outputs, clean):
    mse_loss = F.mse_loss(outputs, clean)
    ssim_loss = 1 - ssim_criterion(outputs, clean)
    return mse_loss + 0.5*ssim_loss

ssim_criterion = SSIM(channel=CHANNELS).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

# Обучение
best_val_loss = float('inf')
early_stop_counter = 0

# История метрик
train_loss_history = []
val_loss_history = []
train_rmse_history = []
val_rmse_history = []
train_psnr_history = []
val_psnr_history = []
train_ssim_history = []
val_ssim_history = []
train_fsim_history = []
val_fsim_history = []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    train_rmse = 0.0
    train_psnr = 0.0
    train_ssim = 0.0
    train_fsim = 0.0

    for noisy, clean in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
        noisy = noisy.to(DEVICE)
        clean = clean.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(noisy)
        loss = criterion(outputs, clean)
        loss.backward()
        optimizer.step()

        # Расчет метрик
        with torch.no_grad():
            rmse = torch.sqrt(loss).item()
            psnr = compute_psnr(outputs, clean).item()
            ssim_val = ssim_criterion(outputs, clean).item()
            fsim_val = fsim(outputs, clean, data_range=1.0, chromatic=False).item()  # Расчет FSIM

        train_loss += loss.item() * noisy.size(0)
        train_rmse += rmse * noisy.size(0)
        train_psnr += psnr * noisy.size(0)
        train_ssim += ssim_val * noisy.size(0)
        train_fsim += fsim_val * noisy.size(0)

    # Валидация
    model.eval()
    val_loss = 0.0
    val_rmse = 0.0
    val_psnr = 0.0
    val_ssim = 0.0
    val_fsim = 0.0

    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy = noisy.to(DEVICE)
            clean = clean.to(DEVICE)

            outputs = model(noisy)
            loss = criterion(outputs, clean)

            rmse = torch.sqrt(loss).item()
            psnr = compute_psnr(outputs, clean).item()
            ssim_val = ssim_criterion(outputs, clean).item()
            fsim_val = fsim(outputs, clean, data_range=1.0, chromatic=False).item()  # Расчет FSIM

            val_loss += loss.item() * noisy.size(0)
            val_rmse += rmse * noisy.size(0)
            val_psnr += psnr * noisy.size(0)
            val_ssim += ssim_val * noisy.size(0)
            val_fsim += fsim_val * noisy.size(0)

    # Нормализация метрик
    train_loss /= len(train_loader.dataset)
    train_rmse /= len(train_loader.dataset)
    train_psnr /= len(train_loader.dataset)
    train_ssim /= len(train_loader.dataset)
    train_fsim /= len(train_loader.dataset)

    val_loss /= len(val_loader.dataset)
    val_rmse /= len(val_loader.dataset)
    val_psnr /= len(val_loader.dataset)
    val_ssim /= len(val_loader.dataset)
    val_fsim /= len(val_loader.dataset)

    # Сохранение истории
    train_loss_history.append(train_loss)
    train_rmse_history.append(train_rmse)
    train_psnr_history.append(train_psnr)
    train_ssim_history.append(train_ssim)
    train_fsim_history.append(train_fsim)

    val_loss_history.append(val_loss)
    val_rmse_history.append(val_rmse)
    val_psnr_history.append(val_psnr)
    val_ssim_history.append(val_ssim)
    val_fsim_history.append(val_fsim)

    # Вывод
    print(f'\nEpoch {epoch + 1}')
    print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
    print(f'Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f}')
    print(f'Train PSNR: {train_psnr:.2f} dB | Val PSNR: {val_psnr:.2f} dB')
    print(f'Train SSIM: {train_ssim:.4f} | Val SSIM: {val_ssim:.4f}')
    print(f'Train FSIM: {train_fsim:.4f} | Val FSIM: {val_fsim:.4f}')

    # Early stopping и сохранение модели
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')

        # Сохраняем метрики в текстовый файл
        metrics_content = f"""Best Model Metrics (Epoch {epoch + 1}):
        Validation Loss: {val_loss:.4f}
        Validation RMSE: {val_rmse:.4f}
        Validation PSNR: {val_psnr:.2f} dB
        Validation SSIM: {val_ssim:.4f}
        Validation FSIM: {val_fsim:.4f}
        Train Loss: {train_loss:.4f}
        Train RMSE: {train_rmse:.4f}
        Train PSNR: {train_psnr:.2f} dB
        Train SSIM: {train_ssim:.4f}
        Train FSIM: {train_fsim:.4f}"""

        with open(os.path.join(config["paths"]["processed_data_dir"], 'best_metrics.txt'), 'w') as f:
            f.write(metrics_content)
    else:
        early_stop_counter += 1
        if early_stop_counter >= 10:
            print("Early stopping!")
            break

plot_results_noise(model, val_loader, num_samples=3)
plot_metrics_noisy(train_loss_history, val_loss_history, train_rmse_history, val_rmse_history,
             train_psnr_history, val_psnr_history, train_ssim_history, val_ssim_history,
             train_fsim_history, val_fsim_history)