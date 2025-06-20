import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from models.encoder import MiniUMTEncoder
from utils.data_loader import get_loaders
from utils.transforms import transform_train, transform_test
from utils.predict import predict_single_image

# ---- Device ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Paths ----
train_dir = "data/fer/train"
test_dir = "data/fer/test"
test_image_path = "path/to/your/test/image.png"  # CHANGE THIS

# ---- Loaders ----
_, test_loader, class_names = get_loaders(train_dir, test_dir, transform_train, transform_test)

# ---- Load Model ----
model = MiniUMTEncoder(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.eval()

# ---- Evaluation ----
y_true, y_pred = [], []
correct, total = 0, 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs, _ = model(images)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

test_acc = correct / total
print(f"Test Accuracy: {test_acc:.2%}")

# ---- Confusion Matrix ----
cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# ---- Predict Single Image ----
predict_single_image(test_image_path, model, transform_test, class_names, device)
