from src.modules.denoise_model import Denoiser
from src.data.dataset_denoise import NoiseDataset
from src.utils.load_config import load_config
from src.utils.plotting import plot_results_noise
import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

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
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

# Обучение
best_val_loss = float('inf')
early_stop_counter = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    # Тренировочная эпоха
    for noisy, clean in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
        noisy = noisy.to(DEVICE)
        clean = clean.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(noisy)
        loss = criterion(outputs, clean)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * noisy.size(0)

    # Валидация
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy = noisy.to(DEVICE)
            clean = clean.to(DEVICE)

            outputs = model(noisy)
            loss = criterion(outputs, clean)
            val_loss += loss.item() * noisy.size(0)

    # Статистика
    train_loss = train_loss / len(train_loader.dataset)
    val_loss = val_loss / len(val_loader.dataset)
    print(f'Epoch {epoch + 1} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}')

    # Callbacks
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= 10:
            print('Early stopping!')
            break

    plot_results_noise(model, val_loader, num_samples=3)