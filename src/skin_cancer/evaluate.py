import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.skin_cancer.dataset import (
    load_skin_cancer_data,
    SkinCancerDataset,
    get_val_transforms,
    CLASS_NAMES,
)

from src.skin_cancer.model import build_skin_cancer_model


MODEL_PATH = "models/skin_cancer_model.pth"

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


def evaluate():
    _, val_df = load_skin_cancer_data()

    val_dataset = SkinCancerDataset(
        val_df,
        transform=get_val_transforms()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False
    )

    model = build_skin_cancer_model()
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)
    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predicted = torch.argmax(
                outputs,
                dim=1
            )

            predictions.extend(
                predicted.cpu().numpy()
            )

            targets.extend(
                labels.numpy()
            )

    accuracy = accuracy_score(
        targets,
        predictions
    )

    precision = precision_score(
        targets,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        targets,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        targets,
        predictions,
        average="weighted",
        zero_division=0
    )

    cm = confusion_matrix(
        targets,
        predictions
    )

    print("\n========== RESULTS ==========")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\n========== CONFUSION MATRIX ==========")
    print(cm)

    print("\n========== CLASSIFICATION REPORT ==========")

    print(
        classification_report(
            targets,
            predictions,
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )


if __name__ == "__main__":
    evaluate()