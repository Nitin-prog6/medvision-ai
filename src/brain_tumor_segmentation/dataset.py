import os
import cv2
import pandas as pd

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset


class BrainTumorSegmentationDataset(Dataset):

    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        image = cv2.imread(row["image_path"])
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = cv2.resize(
            image,
            (256, 256)
        )

        mask = cv2.imread(
            row["mask_path"],
            cv2.IMREAD_GRAYSCALE
        )

        mask = cv2.resize(
            mask,
            (256, 256)
        )

        image = image.astype("float32") / 255.0
        mask = mask.astype("float32") / 255.0

        image = torch.tensor(
            image.transpose(2, 0, 1)
        )

        mask = torch.tensor(
            mask
        ).unsqueeze(0)

        return image, mask


def load_segmentation_data(
    data_dir="data/brain_tumor_segmentation/kaggle_3m"
):

    records = []

    for root, _, files in os.walk(data_dir):

        for file in files:

            if (
                file.endswith(".tif")
                and "_mask" not in file
            ):

                image_path = os.path.join(
                    root,
                    file
                )

                mask_path = image_path.replace(
                    ".tif",
                    "_mask.tif"
                )

                if os.path.exists(mask_path):

                    records.append({
                        "image_path":
                            image_path,
                        "mask_path":
                            mask_path
                    })

    df = pd.DataFrame(records)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42
    )

    return train_df, val_df