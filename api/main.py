import os
import tempfile
from src.brain_tumor_segmentation.predict import predict_mask
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from src.brain_tumor.predict import predict_image as predict_brain_tumor_image
from src.skin_cancer.predict import predict_image
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="MedVision AI API",
    description="Backend API for medical image classification and segmentation",
    version="1.0.0"
)
app.mount(
    "/outputs",
    StaticFiles(directory="segmentation_outputs"),
    name="outputs"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "MedVision AI API is running",
        "status": "healthy"
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok"
    }


@app.post("/predict/skin-cancer")
async def predict_skin_cancer(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(await file.read())
        image_path = temp.name

    result = predict_image(image_path)

    os.remove(image_path)

    return result

@app.post("/predict/brain-tumor")
async def predict_brain_tumor(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(await file.read())
        image_path = temp.name

    result = predict_brain_tumor_image(image_path)

    os.remove(image_path)

    return result

@app.post("/predict/brain-segmentation")
async def predict_brain_segmentation(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(await file.read())
        image_path = temp.name

    result = predict_mask(
        image_path=image_path,
        output_dir="segmentation_outputs"
    )

    os.remove(image_path)

    return {
         "tumor_area_percentage": result["tumor_area_percentage"],
    "original_image": "/outputs/original_mri.jpg",
    "mask_image": "/outputs/predicted_mask.jpg",
    "overlay_image": "/outputs/tumor_overlay.jpg"
    }