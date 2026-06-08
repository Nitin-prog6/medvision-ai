import torch
from PIL import Image
from torchvision import transforms

from src.skin_cancer.dataset import CLASS_NAMES
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


def predict_image(image_path):
    model = build_skin_cancer_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image_tensor = get_transform()(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_index = torch.argmax(probabilities).item()

    all_probabilities = {
        CLASS_NAMES[i]: float(probabilities[i])
        for i in range(len(CLASS_NAMES))
    }

    sorted_probabilities = dict(
        sorted(
            all_probabilities.items(),
            key=lambda item: item[1],
            reverse=True
        )
    )

    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "probabilities": sorted_probabilities,
    }


if __name__ == "__main__":
    test_image = "data/skin_cancer/HAM10000_images_part_1/ISIC_0027419.jpg"
    result = predict_image(test_image)

    print("Prediction:", result["predicted_class"])
    print("Confidence:", round(result["confidence"] * 100, 2), "%")

    print("\nClass Probabilities:")
    for class_name, probability in result["probabilities"].items():
        print(f"{class_name}: {probability * 100:.2f}%")