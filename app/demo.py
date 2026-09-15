import gradio as gr
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.model import model
from src.calibration import fit_temperature, build_prediction_sets

opt_T = 1.05
q_hat = 0.9734
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classes = ["glioma", "meningioma", "no tumor", "pituitary"]

def predict(img):
    trans_image = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = trans_image(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        scaled_outputs = outputs / opt_T
        probs = torch.softmax(scaled_outputs, 1)

        label_output = {classes[i]: float(probs[0][i]) for i in range(4)}

        pred_set_mask = build_prediction_sets(scaled_outputs, q_hat)
        set_classes = [classes[i] for i in range(4) if pred_set_mask[0][i]]
        set_text = f"Prediction Set (99% coverage target): {{{', '.join(set_classes)}}}"
    return label_output, set_text

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Label(num_top_classes=4, label="Calibrated Probabilities"), gr.Text(label="Conformal Prediction Set")],
    title="Brain Tumor MRI Classifier with Uncertainty Quantification",
    description="Upload an MRI scan. The model outputs calibrated class probabilities (via Temperature Scaling) and a Conformal Prediction Set with a statistical coverage guarantee."
)

if __name__ == "__main__":
    demo.launch()