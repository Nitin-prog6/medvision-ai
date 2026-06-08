import os
import cv2
import torch
import numpy as np

from src.brain_tumor_segmentation.model import build_unet


MODEL_PATH = "models/brain_tumor_unet.pth"

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


def predict_mask(
    image_path,
    output_dir="segmentation_outputs"
):
    os.makedirs(output_dir, exist_ok=True)

    model = build_unet()
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )
    model.to(DEVICE)
    model.eval()

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    resized = cv2.resize(
        image_rgb,
        (256, 256)
    )

    normalized = resized.astype("float32") / 255.0

    tensor = torch.tensor(
        normalized.transpose(2, 0, 1)
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)
        probability_map = torch.sigmoid(output)[0, 0].cpu().numpy()

        print("Min:", probability_map.min())
        print("Max:", probability_map.max())
        print("Mean:", probability_map.mean())

    max_prob = probability_map.max()

    if max_prob < 0.01:
        binary_mask = np.zeros_like(probability_map, dtype="uint8")
    else:
        threshold = max(0.3, max_prob * 0.5)
        max_prob = probability_map.max()

    if max_prob <= 0:
        binary_mask = np.zeros_like(probability_map, dtype="uint8")
    else:
        threshold = max_prob * 0.55
        binary_mask = (probability_map > threshold).astype("uint8") * 255

    heatmap = cv2.applyColorMap(
        binary_mask,
        cv2.COLORMAP_JET
    )

    heatmap_rgb = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = resized.copy()
    overlay[binary_mask > 0] = [255, 0, 0]

    blended = cv2.addWeighted(
        resized,
        0.75,
        overlay,
        0.25,
        0
    )

    original_path = os.path.join(
        output_dir,
        "original_mri.jpg"
    )

    mask_path = os.path.join(
        output_dir,
        "predicted_mask.jpg"
    )

    overlay_path = os.path.join(
        output_dir,
        "tumor_overlay.jpg"
    )

    cv2.imwrite(
        original_path,
        cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    )

    cv2.imwrite(
        mask_path,
        binary_mask
    )

    cv2.imwrite(
        overlay_path,
        cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
    )

    tumor_area_percentage = (
        np.sum(binary_mask > 0) / binary_mask.size
    ) * 100

    return {
        "original_path": original_path,
        "mask_path": mask_path,
        "overlay_path": overlay_path,
        "tumor_area_percentage": tumor_area_percentage,
    }


if __name__ == "__main__":
    image_path = (
        "data/brain_tumor_segmentation/"
        "kaggle_3m/"
        "TCGA_CS_6667_20011105/"
        "TCGA_CS_6667_20011105_9.tif"
    )

    result = predict_mask(image_path)

    print("Original:", result["original_path"])
    print("Mask:", result["mask_path"])
    print("Overlay:", result["overlay_path"])
    print(
        "Tumor Area:",
        round(result["tumor_area_percentage"], 2),
        "%"
    )