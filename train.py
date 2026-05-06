import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_loader import get_dataloaders
from sod_model import SODModel

# ─── Loss Function ───────────────────────────────────────────
def iou_loss(pred, target):
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return 1 - iou.mean()

def combined_loss(pred, target):
    bce = nn.BCELoss()(pred, target)
    iou = iou_loss(pred, target)
    return bce + 0.5 * iou

# ─── Train One Epoch ─────────────────────────────────────────
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for images, masks in tqdm(loader, desc="Training"):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = combined_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# ─── Validate One Epoch ──────────────────────────────────────
def val_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = combined_loss(outputs, masks)
            total_loss += loss.item()
    return total_loss / len(loader)

# ─── Training Loop ───────────────────────────────────────────
def train(num_epochs=20, batch_size=16, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, _ = get_dataloaders("data/ECSSD", batch_size)
    model = SODModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = val_epoch(model, val_loader, device)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, "checkpoints/best_model.pth")
            print(f"✅ Best model saved! Val Loss: {val_loss:.4f}")

        # Early stopping
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⛔ Early stopping at epoch {epoch}")
                break

    print("\n✅ Training complete!")

# ─── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    train()