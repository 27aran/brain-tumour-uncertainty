from calibration import (
    collect_logits,
    compute_ece,
    fit_temperature,
    compute_nonconformity_scores,
    compute_quantile,
    build_prediction_sets)
from data_loader import test_data_loader
from src.model import model
import torch

from src.train import loss_metric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.load_state_dict(torch.load("../checkpoints/resnet18_brain_tumor.pth", map_location=device))

model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_data_loader:

        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.4f}")
