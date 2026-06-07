import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from src.brain_tumor.dataset import (
    load_brain_tumor_data,
    BrainTumorDataset,
    get_val_transforms,
    CLASS_NAMES,
)
from src.brain_tumor.model import build_brain_tumor_model


MODEL_PATH = "models/brain_tumor_model.pth"

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


def evaluate():
    _, val_df = load_brain_tumor_data()

    val_dataset = BrainTumorDataset(
        val_df,
        transform=get_val_transforms()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False
    )

    model = build_brain_tumor_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)

            predictions.extend(predicted.cpu().numpy())
            targets.extend(labels.numpy())

    print("\nClassification Report:")
    print(
        classification_report(
            targets,
            predictions,
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(targets, predictions))


if __name__ == "__main__":
    evaluate()