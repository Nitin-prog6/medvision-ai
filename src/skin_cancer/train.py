import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from src.skin_cancer.dataset import (
    load_skin_cancer_data,
    SkinCancerDataset,
    get_train_transforms,
    get_val_transforms,
)

from src.skin_cancer.model import build_skin_cancer_model


BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


def main():

    train_df, val_df = load_skin_cancer_data()

    train_dataset = SkinCancerDataset(
        train_df,
        transform=get_train_transforms()
    )

    val_dataset = SkinCancerDataset(
        val_df,
        transform=get_val_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = build_skin_cancer_model()
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_accuracy = 0

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

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

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Loss: {running_loss:.4f} | "
            f"Val Accuracy: {accuracy:.4f}"
        )

        if accuracy > best_accuracy:

            best_accuracy = accuracy

            torch.save(
                model.state_dict(),
                "models/skin_cancer_model.pth"
            )

            print(
                f"Best model saved. "
                f"Accuracy={accuracy:.4f}"
            )


if __name__ == "__main__":
    main()