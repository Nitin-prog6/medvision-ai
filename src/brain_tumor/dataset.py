import os
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset
from torchvision import transforms


CLASS_NAMES = [
    "No Tumor",
    "Tumor"
]


class BrainTumorDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        image = Image.open(
            row["image_path"]
        ).convert("RGB")

        label = row["label"]

        if self.transform:
            image = self.transform(image)

        return image, label


def load_brain_tumor_data(
    data_dir="data/brain_tumor",
    test_size=0.2
):

    records = []

    no_dir = os.path.join(data_dir, "no")
    yes_dir = os.path.join(data_dir, "yes")

    for file in os.listdir(no_dir):

        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            records.append({
                "image_path":
                    os.path.join(no_dir, file),
                "label": 0
            })

    for file in os.listdir(yes_dir):

        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            records.append({
                "image_path":
                    os.path.join(yes_dir, file),
                "label": 1
            })

    df = pd.DataFrame(records)

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df["label"]
    )

    return train_df, val_df


def get_train_transforms():

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])


def get_val_transforms():

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])