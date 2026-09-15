import torch
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from torchmetrics import CalibrationError
from src import data_loader
from src.model import model
from src.data_loader import cal_data_loader
from src.data_loader import test_data_loader
import numpy as np


def collect_logits(model, data_loader, device):
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            all_logits.append(output)
            all_labels.append(labels)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    return all_logits, all_labels

def compute_ece(logits, labels ,n_bins=10):
    metric = CalibrationError(task="multiclass", num_classes=4, n_bins=10, norm="l1")
    preds = torch.softmax(logits, dim=1)
    ece = metric(preds, labels)
    return ece.item()

def plot_reliability_diagram(
        logits,
        labels,
        n_bins= 10,
        title= "Reliability Diagram",
        save_path= "../results/figures/reliability.png"):

    probs = torch.softmax(logits, dim=1)
    confidences, predictions = torch.max(probs, 1)
    correct = (predictions == labels).int().cpu().numpy()
    confidences = confidences.detach().cpu().numpy()

    prob_true, prob_pred = calibration_curve(correct, confidences, n_bins=n_bins, strategy='quantile')

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfekte Kalibrierung')
    plt.plot(prob_pred, prob_true, marker='o', label='Modell')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(f"{title}\n(n={len(confidences)} Samples)")
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(save_path)
    plt.close()

def fit_temperature(logits, labels):
    temperature = torch.nn.Parameter(torch.ones(1) * 1.0)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
    criterion = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        scaled_logits = logits / temperature
        loss = criterion(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    print(f"Optimales T: {temperature.item():.4f}")
    return temperature.item()

def compute_nonconformity_scores(logits, labels):
    probs = torch.softmax(logits, dim=1)
    true_label_probs = probs[torch.arange(len(labels)), labels]
    scores = 1 - true_label_probs
    return scores

def compute_quantile(scores, alpha=0.01):
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_hat = torch.quantile(scores, q_level)
    return q_hat

def build_prediction_sets(logits, q_hat):
    probs = torch.softmax(logits, dim=1)
    prediction_sets = (1 - probs) <= q_hat
    top1 = torch.argmax(probs, dim=1)
    for i in range(len(top1)):
        prediction_sets[i, top1[i]] = True
    return prediction_sets

#What is needed for main method calls
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.load_state_dict(torch.load("../checkpoints/resnet18_brain_tumor.pth", map_location=device))
data_loader_cal = cal_data_loader
data_loader_test = test_data_loader

if __name__ == "__main__":
    #Before Temperature scaling
    all_logits, all_labels = collect_logits(model, data_loader_cal, device)

    ece = compute_ece(all_logits, all_labels, n_bins=10)
    print(f"ECE vor Kalibrierung: {ece: .4f}")

    plot_reliability_diagram(all_logits, all_labels, n_bins=10, title= "Reliability Diagram",
                             save_path= "../results/figures/reliability.png")
    optimal_temperature = fit_temperature(all_logits, all_labels)

    #After Temperature scaling
    scaled_logits = all_logits / optimal_temperature

    ece_after = compute_ece(scaled_logits, all_labels, n_bins=10)
    print(f"ECE nach Kalibrierung: {ece_after: .4f}")

    plot_reliability_diagram(scaled_logits, all_labels, n_bins=10, title="Reliability Diagram",
                             save_path="../results/figures/calibrated_reliability.png")

    #Conformative Prediction
    scores = compute_nonconformity_scores(scaled_logits, all_labels)
    q_hat = compute_quantile(scores, alpha=0.01)
    print(f"Q-Hat: {q_hat}")
    #Temperature Scaling on test data
    test_logits, test_labels = collect_logits(model, data_loader_test, device)
    scaled_test_logits = test_logits / optimal_temperature
    prediction_sets = build_prediction_sets(scaled_test_logits, q_hat)

    #What do the predictions look like?
    set_sizes = prediction_sets.sum(dim=1)  # wie viele Klassen pro Bild im Set
    print(f"Durchschnittliche Set-Größe: {set_sizes.float().mean().item():.2f}")
    print(f"Anteil Sets mit Größe 1: {(set_sizes == 1).float().mean().item():.2%}")
    print(f"Anteil Sets mit Größe 4 (alle Klassen): {(set_sizes == 4).float().mean().item():.2%}")

    # Check empiric coverage: is the true label actually in the predict set?
    true_label_in_set = prediction_sets[torch.arange(len(test_labels)), test_labels]
    empirical_coverage = true_label_in_set.float().mean().item()
    print(f"Empirische Coverage: {empirical_coverage:.2%} (Ziel: 99%)")