import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.brain_tumor_segmentation.dataset import (
    load_segmentation_data,
    BrainTumorSegmentationDataset,
)

from src.brain_tumor_segmentation.model import build_unet


BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-4

MODEL_PATH = "models/brain_tumor_unet.pth"

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()

        dice = (
            (2.0 * intersection + self.smooth)
            / (probs.sum() + targets.sum() + self.smooth)
        )

        return 1.0 - dice


def dice_score(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()

    dice = (
        (2.0 * intersection + 1.0)
        / (preds.sum() + targets.sum() + 1.0)
    )

    return dice.item()


def main():
    train_df, val_df = load_segmentation_data()

    train_dataset = BrainTumorSegmentationDataset(train_df)
    val_dataset = BrainTumorSegmentationDataset(val_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = build_unet().to(DEVICE)

    pos_weight = torch.tensor([20.0]).to(DEVICE)

    bce_loss = nn.BCEWithLogitsLoss(
                pos_weight=pos_weight
            )

    dice_loss = DiceLoss()

    optimizer = torch.optim.Adam(
                model.parameters(),
                lr=LEARNING_RATE
            )

    best_dice = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for images, masks in train_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = bce_loss(outputs, masks) + dice_loss(outputs, masks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_dice = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                outputs = model(images)

                val_dice += dice_score(outputs, masks)

        val_dice = val_dice / len(val_loader)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Dice: {val_dice:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print(
                f"Best U-Net saved. Dice={val_dice:.4f}"
            )


if __name__ == "__main__":
    main()