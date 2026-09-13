# Brain Tumor MRI Classification with Uncertainty Quantification

A deep learning project for classifying brain tumors from MRI scans, with a focus on **calibrated, trustworthy uncertainty estimation** rather than pure accuracy optimization.

## Motivation

Most classification models output a prediction plus a softmax "confidence" score — but this value is often **not calibrated** in modern neural networks: a model that says "95% confident" frequently turns out to be right less often than 95% of the time (overconfidence). This is especially dangerous in a medical context, where clinicians might rely on this number without knowing it can be misleading.

This project investigates how well an image classifier for brain tumors is actually calibrated, and applies two standard techniques to check and improve this:

- **Temperature Scaling** – corrects a model's global over-/underconfidence
- **Conformal Prediction** – instead of a single predicted class, produces a *prediction set* with a statistically guaranteed coverage level (e.g. "the true label is contained in this set with ≥90% probability")

The approach builds on experience from an internship in uncertainty quantification (calibration methods, conformal prediction, Bayesian neural networks applied to field data from battery-electric vehicles) and transfers this methodology to a new problem: computer vision in a medical context.

## Dataset

[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Kaggle), 4 classes:
- Glioma
- Meningioma
- No tumor
- Pituitary

The dataset is a combination of three source datasets (figshare, SARTAJ, Br35H).

## ⚠️ Key Finding: Data Leakage in the Dataset

An initial training run reached a suspiciously high validation accuracy of ~99% after just a few epochs — a classic red flag. Investigation and a custom analysis revealed:

1. **Exact hashing (MD5)** found 187 byte-identical duplicates between the training and testing splits.
2. Since accuracy remained unusually high afterwards, **perceptual hashing** (`imagehash`, Hamming distance threshold ≤ 5) was applied as well. This found **2057 additional near-duplicates** (~29% of the dataset) — likely differently compressed/resized copies of identical or highly similar scans.
3. After cleanup (training data takes precedence on duplicates; affected images were removed from the testing split), test accuracy dropped to a considerably more realistic value.

Many publicly available analyses of this dataset (including numerous Kaggle notebooks) report accuracy values around 99% — consistent with the leakage issue found here, suggesting these numbers likely reflect the same data leakage effect rather than genuine generalization.

**Known remaining limitation:** The dataset does not provide patient IDs. It is therefore not possible to fully rule out that different MRI slices from the same patient are distributed across different splits ("patient-level leakage"). This risk cannot be eliminated with the available metadata.

## Methodology

**1. Data preparation**
- 4-way split: Train (70%) / Validation (15%) / Calibration (15%) / Test (untouched original test split)
- The calibration set is used exclusively for temperature scaling and conformal prediction, never for training
- Data augmentation (flip, rotation) applied only to the training split, to avoid leaking augmentation effects into val/cal/test

**2. Model**
- ResNet18, pretrained on ImageNet (transfer learning)
- Final fully connected layer replaced (4 instead of 1000 classes)
- Manually implemented PyTorch training loop (no high-level wrapper), Adam optimizer, cross-entropy loss

**3. Calibration — Temperature Scaling**
- A single scaling parameter T is fit on the calibration set to scale the logits before softmax: `softmax(logits / T)`
- Does not change the prediction itself (accuracy stays the same), only the reliability of the confidence values
- Evaluated via Expected Calibration Error (ECE) and reliability diagrams

**4. Conformal Prediction**
- Nonconformity score per image: `1 − P(true label)`
- A threshold (quantile) is computed on the calibration set, including a finite-sample correction
- For each test image, a *prediction set* is built: all classes whose score falls below the threshold (plus a safeguard so the top-1 class is never excluded)
- Result: instead of a single prediction, a set with a statistical coverage guarantee

## Results

| Metric | Value |
|---|---|
| Test accuracy (after leakage cleanup) | 90.7% |
| ECE before calibration | 0.0168 |
| ECE after temperature scaling | 0.0159 |
| Optimal T | 1.05 |
| Coverage at 90% target (conformal prediction) | 90.26% (empirical) |
| Coverage at 99% target (conformal prediction) | 94.75% (empirical) |
| Avg. prediction set size (99% target) | 1.28 |

**Interpretation:**

- The model was already reasonably well calibrated before explicit calibration (low ECE); temperature scaling only provided a marginal improvement. A possible explanation: transfer learning from an already well-calibrated backbone combined with moderate training duration without strong overfitting.
- At a 90% target coverage, conformal prediction nearly exactly honors its guarantee (90.26%).
- At a 99% target coverage, empirical coverage falls noticeably short of the target (94.75%). Reason: estimating such an extreme quantile relies on very few data points in the tail of the distribution when the calibration set only has ~620 images, making the estimate unstable. Reliable coverage close to 99% would require a considerably larger calibration set.
- At the 99% target, prediction sets show clear, adaptive variation: ~79% of images receive an unambiguous set (size 1), while the most uncertain cases retain all 4 classes in the set — exactly the behavior one would expect from conformal prediction.

## Reliability Diagrams

`results/figures/reliability.png` (before calibration)
`results/figures/calibrated_reliability.png` (after calibration)

## Setup

```bash
pip install -r requirements.txt
```

The dataset is downloaded automatically via `kagglehub` on first run.

**Run the pipeline:**
```bash
python scripts/dedup_dataset.py     # remove duplicates/near-duplicates
python src/train.py                 # train model, save checkpoint
python src/calibration.py           # temperature scaling + conformal prediction
python src/evaluate.py              # final test evaluation
```

## Project Structure

```
brain-tumor-uncertainty/
├── scripts/
│   └── dedup_dataset.py
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── calibration.py
│   └── evaluate.py
├── app/
│   └── demo.py
├── results/figures/
└── checkpoints/
```

## Limitations

- Patient-level leakage cannot be fully ruled out (no patient IDs available in the dataset)
- Conformal prediction's coverage guarantee is limited at very high target levels (e.g. 99%) due to the limited size of the calibration set
- Temperature scaling is a global, class-agnostic correction — it does not address class-specific calibration errors

## Tech Stack

Python, PyTorch, torchvision, torchmetrics, scikit-learn, imagehash, Kaggle Hub, Matplotlib