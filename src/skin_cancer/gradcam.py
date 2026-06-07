import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.skin_cancer.model import build_skin_cancer_model


MODEL_PATH = "models/skin_cancer_model.pth"

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


def get_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def generate_gradcam(image_path, output_path="gradcam_result.jpg"):

    model = build_skin_cancer_model()
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)
    model.eval()

    image = Image.open(image_path).convert("RGB")

    rgb_image = np.array(
        image.resize((224, 224))
    ).astype(np.float32) / 255.0

    input_tensor = get_transform()(image).unsqueeze(0).to(DEVICE)

    target_layers = [
        model.features[-1]
    ]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    grayscale_cam = cam(
        input_tensor=input_tensor
    )[0]

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    cv2.imwrite(
        output_path,
        cv2.cvtColor(
            visualization,
            cv2.COLOR_RGB2BGR
        )
    )

    print(
        f"Grad-CAM saved to {output_path}"
    )


if __name__ == "__main__":

    image_path = (
        "data/skin_cancer/"
        "HAM10000_images_part_1/"
        "ISIC_0027419.jpg"
    )

    generate_gradcam(image_path)
    