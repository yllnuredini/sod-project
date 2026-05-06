import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_loader import get_dataloaders
from sod_model import SODModel

# ─── Metrics ─────────────────────────────────────────────────
def compute_iou(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)

def compute_precision_recall_f1(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return precision.item(), recall.item(), f1.item()

def compute_mae(pred, target):
    return torch.abs(pred - target).mean().item()

# ─── Evaluate ────────────────────────────────────────────────
def evaluate(model, test_loader, device):
    model.eval()
    iou_scores, precisions, recalls, f1_scores, maes = [], [], [], [], []

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            for i in range(outputs.size(0)):
                pred = outputs[i]
                target = masks[i]
                iou_scores.append(compute_iou(pred, target).item())
                p, r, f1 = compute_precision_recall_f1(pred, target)
                precisions.append(p)
                recalls.append(r)
                f1_scores.append(f1)
                maes.append(compute_mae(pred, target))

    print("\n📊 Evaluation Results:")
    print(f"IoU:       {np.mean(iou_scores):.4f}")
    print(f"Precision: {np.mean(precisions):.4f}")
    print(f"Recall:    {np.mean(recalls):.4f}")
    print(f"F1-Score:  {np.mean(f1_scores):.4f}")
    print(f"MAE:       {np.mean(maes):.4f}")

    return {
        "iou": np.mean(iou_scores),
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "f1": np.mean(f1_scores),
        "mae": np.mean(maes)
    }

# ─── Visualize ───────────────────────────────────────────────
def visualize(model, test_loader, device, num_samples=4):
    model.eval()
    images, masks = next(iter(test_loader))
    images, masks = images.to(device), masks.to(device)

    with torch.no_grad():
        outputs = model(images)

    os.makedirs("results", exist_ok=True)

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, num_samples * 4))
    titles = ["Input Image", "Ground Truth", "Predicted Mask", "Overlay"]

    for i in range(num_samples):
        # Input image
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())

        # Ground truth
        gt = masks[i].cpu().squeeze().numpy()

        # Predicted mask
        pred = outputs[i].cpu().squeeze().numpy()

        # Overlay
        overlay = img.copy()
        overlay[:, :, 0] = np.clip(overlay[:, :, 0] + pred * 0.5, 0, 1)

        for j, (data, title) in enumerate(zip([img, gt, pred, overlay], titles)):
            axes[i, j].imshow(data, cmap="gray" if j in [1, 2] else None)
            axes[i, j].set_title(title)
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig("results/visualization.png", dpi=150)
    print("✅ Visualization saved to results/visualization.png")
    plt.show()

# ─── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_dataloaders("data/ECSSD", batch_size=16)

    model = SODModel().to(device)
    checkpoint = torch.load("checkpoints/best_model.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Model loaded from epoch {checkpoint['epoch']}")

    evaluate(model, test_loader, device)
    visualize(model, test_loader, device)