from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

PRED_FILE = Path("results/external_validation_predictions.csv")
OUT = Path("results")
OUT.mkdir(exist_ok=True)

df = pd.read_csv(PRED_FILE)

y = df["label"].astype(int)
pred = df["predicted_label"].astype(int)
prob = df["chf_probability"].astype(float)

# 1. Confusion matrix
cm = confusion_matrix(y, pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Healthy", "CHF"]
)

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax)
ax.set_title("External Validation Confusion Matrix")
fig.tight_layout()
fig.savefig(
    OUT / "external_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

# 2. ROC curve
fpr, tpr, _ = roc_curve(y, prob)
auc = roc_auc_score(y, prob)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}")
ax.plot([0, 1], [0, 1], linestyle="--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("External Validation ROC Curve")
ax.legend()
fig.tight_layout()
fig.savefig(
    OUT / "external_roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

# 3. Precision-recall curve
precision, recall, _ = precision_recall_curve(y, prob)
ap = average_precision_score(y, prob)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(recall, precision, label=f"Average Precision = {ap:.3f}")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("External Validation Precision-Recall Curve")
ax.legend()
fig.tight_layout()
fig.savefig(
    OUT / "external_precision_recall.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

# 4. Probability distribution
fig, ax = plt.subplots(figsize=(8, 6))

df[df["label"] == 0]["chf_probability"].hist(
    bins=40,
    alpha=0.6,
    ax=ax,
    label="Healthy"
)

df[df["label"] == 1]["chf_probability"].hist(
    bins=40,
    alpha=0.6,
    ax=ax,
    label="CHF"
)

ax.set_xlabel("Predicted CHF Probability")
ax.set_ylabel("Number of 5-minute Windows")
ax.set_title("External CHF Probability Distribution")
ax.legend()

fig.tight_layout()
fig.savefig(
    OUT / "external_probability_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

print("Created:")
print("results/external_confusion_matrix.png")
print("results/external_roc_curve.png")
print("results/external_precision_recall.png")
print("results/external_probability_distribution.png")