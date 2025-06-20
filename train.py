import os
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

from models.encoder import MiniUMTEncoder
from utils.transforms import transform_train, transform_test
from utils.data_loader import get_loaders

# ---- Device ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---- Paths ----
train_dir = "data/fer/train"
test_dir = "data/fer/test"

# ---- Loaders ----
train_loader, test_loader, class_names = get_loaders(train_dir, test_dir, transform_train, transform_test)

# ---- Model ----
model = MiniUMTEncoder(num_classes=len(class_names)).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=100)

# ---- Training ----
num_epochs = 75
train_losses, train_accuracies = [], []

for epoch in range(num_epochs):
    model.train()
    total, correct, loss_epoch = 0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs, _ = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()
        loss_epoch += loss.item()

    scheduler.step()
    acc = correct / total
    train_losses.append(loss_epoch)
    train_accuracies.append(acc)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss_epoch:.4f} - Acc: {acc:.2%}")

# ---- Save Model ----
torch.save(model.state_dict(), "model_weights.pth")

# ---- Plot ----
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
ax1.plot(train_losses, label='Loss')
ax1.set_title("Training Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()

ax2.plot(train_accuracies, label='Accuracy')
ax2.set_title("Training Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend()

plt.tight_layout()
plt.savefig("training_plot.png")
plt.show()
