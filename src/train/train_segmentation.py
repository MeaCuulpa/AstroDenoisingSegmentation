from src.data.dataset_segm import SegmentationDataset
from src.utils.load_config import load_config
from src.modules.segmentation_model import TinySegmentation
from src.utils.plotting import plot_results_segmentation
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
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
MASKS_DIR = config['paths']['masks_data_dir']
IMAGE_DIR = config['paths']['input_data_dir']

dataset = SegmentationDataset(IMAGE_DIR, MASKS_DIR, IMG_SIZE)
train_size = int(0.8 * len(dataset))

val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Инициализация модели
device = torch.device(DEVICE)
model = TinySegmentation().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCELoss()

# Шедулер и ранняя остановка
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=2,
    verbose=True
)
early_stop_patience = 5
best_val_loss = float('inf')
patience_counter = 0

# Списки для хранения метрик
train_losses = []
val_losses = []
precisions = []
recalls = []
f1_scores = []
ious = []

# Обучение модели
for epoch in range(EPOCHS):
    # Тренировочная эпоха
    model.train()
    epoch_loss = 0
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Валидация
    model.eval()
    val_loss = 0
    all_preds = []
    all_masks = []
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            outputs = model(images)

            val_loss += criterion(outputs, masks.to(device)).item()

            preds = (outputs.cpu().numpy() > 0.5).astype(int)
            masks_np = masks.numpy()

            all_preds.append(preds.reshape(-1))
            all_masks.append(masks_np.reshape(-1))


    # Нормализация лоссов
    train_loss = epoch_loss / len(train_loader)
    val_loss = val_loss / len(val_loader)

    # Обновление шедулера
    scheduler.step(val_loss)

    # Расчет метрик
    all_preds = np.concatenate(all_preds)
    all_masks = np.concatenate(all_masks)

    iou = jaccard_score(all_masks, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_masks, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_masks, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_masks, all_preds, average='macro', zero_division=0)

    # Сохранение метрик
    train_losses.append(epoch_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    ious.append(iou)

    # Проверка на улучшение val_loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), '../../models/best_model_segm.pth')
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}!")
            break

    print(f'Epoch {epoch + 1}')
    print(f'Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}')
    print(f'Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | IoU: {iou:.4f}\n')

    # Загрузка лучшей модели
    model.load_state_dict(torch.load('../../models/best_model_segm.pth'))

plot_results_segmentation(model, train_losses, val_losses, precisions, recalls, f1_scores, ious, val_dataset, DEVICE)
