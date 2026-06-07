import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms


LABEL_MAP = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6,
}

CLASS_NAMES = [
    "Actinic Keratoses",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic Nevi",
    "Vascular Lesion",
]


class SkinCancerDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image = Image.open(row["image_path"]).convert("RGB")
        label = row["label"]

        if self.transform:
            image = self.transform(image)

        return image, label


def find_image_path(image_id, image_dir_1, image_dir_2):
    filename = image_id + ".jpg"

    path_1 = os.path.join(image_dir_1, filename)
    path_2 = os.path.join(image_dir_2, filename)

    if os.path.exists(path_1):
        return path_1

    if os.path.exists(path_2):
        return path_2

    raise FileNotFoundError(f"Image not found: {filename}")


def load_skin_cancer_data(data_dir="data/skin_cancer", test_size=0.2):
    csv_path = os.path.join(data_dir, "HAM10000_metadata.csv")
    image_dir_1 = os.path.join(data_dir, "HAM10000_images_part_1")
    image_dir_2 = os.path.join(data_dir, "HAM10000_images_part_2")

    df = pd.read_csv(csv_path)

    df["label"] = df["dx"].map(LABEL_MAP)
    df["image_path"] = df["image_id"].apply(
        lambda image_id: find_image_path(image_id, image_dir_1, image_dir_2)
    )

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df["label"],
    )

    return train_df, val_df


def get_train_transforms(image_size=224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_val_transforms(image_size=224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )