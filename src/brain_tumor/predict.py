import torch
from PIL import Image
from torchvision import transforms

from src.brain_tumor.dataset import CLASS_NAMES
from src.brain_tumor.model import build_brain_tumor_model


MODEL_PATH = "models/brain_tumor_model.pth"

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])


def predict_image(image_path):
    model = build_brain_tumor_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image_tensor = get_transform()(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_index = torch.argmax(probabilities).item()

    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": float(probabilities[predicted_index])
    }


if __name__ == "__main__":
    test_image = "data/brain_tumor/yes/Y1.jpg"

    result = predict_image(test_image)

    print("Prediction:", result["predicted_class"])
    print("Confidence:", round(result["confidence"] * 100, 2), "%")