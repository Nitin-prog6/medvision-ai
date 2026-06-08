import os
import sys
import tempfile
import streamlit as st

sys.path.append(os.path.abspath("."))

from src.skin_cancer.predict import predict_image
from src.skin_cancer.gradcam import generate_gradcam


DESCRIPTIONS = {
    "Melanoma": (
        "A serious form of skin cancer that develops from pigment-producing "
        "cells called melanocytes."
    ),
    "Melanocytic Nevi": (
        "Common benign moles formed by melanocyte cells."
    ),
    "Basal Cell Carcinoma": (
        "A common type of skin cancer that usually grows slowly."
    ),
    "Benign Keratosis": (
        "A non-cancerous skin growth often associated with aging."
    ),
    "Actinic Keratoses": (
        "Rough, scaly patches that can develop after long-term sun exposure."
    ),
    "Dermatofibroma": (
        "A usually benign skin nodule commonly found on the limbs."
    ),
    "Vascular Lesion": (
        "An abnormal growth or clustering of blood vessels in the skin."
    ),
}


CLINICAL_CATEGORY = {
    "Melanoma": "Cancerous",
    "Basal Cell Carcinoma": "Cancerous",
    "Actinic Keratoses": "Pre-cancerous",
    "Melanocytic Nevi": "Usually benign",
    "Benign Keratosis": "Benign",
    "Dermatofibroma": "Benign",
    "Vascular Lesion": "Usually benign",
}


def get_confidence_level(confidence):
    if confidence >= 90:
        return "High Confidence"
    if confidence >= 70:
        return "Moderate Confidence"
    return "Low Confidence"


def show_confidence_level(confidence_level):
    if confidence_level == "High Confidence":
        st.success("🟢 High Confidence")
    elif confidence_level == "Moderate Confidence":
        st.warning("🟡 Moderate Confidence")
    else:
        st.error("🔴 Low Confidence")


st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🩺",
    layout="wide"
)

st.title("Skin Cancer Classification")
st.write(
    "Upload a skin lesion image to classify it and generate Grad-CAM explainability."
)

uploaded_file = st.file_uploader(
    "Upload skin lesion image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        temp.write(uploaded_file.read())
        image_path = temp.name

    st.image(
        image_path,
        caption="Uploaded Image",
        width=350
    )

    if st.button("Analyze Image"):
        result = predict_image(image_path)

        top_predictions = list(result["probabilities"].items())

        primary_class, primary_probability = top_predictions[0]
        secondary_class, secondary_probability = top_predictions[1]

        confidence = primary_probability * 100
        confidence_level = get_confidence_level(confidence)

        category = CLINICAL_CATEGORY.get(primary_class, "Unknown")

        gradcam_path = "skin_gradcam_streamlit.jpg"
        generate_gradcam(image_path, gradcam_path)

        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Original Image")
            st.image(image_path)

        with col2:
            st.subheader("Grad-CAM")
            st.image(gradcam_path)
            st.caption(
                "Highlighted regions show areas that influenced the model prediction."
            )

        with col3:
            st.subheader("Prediction Details")

            st.markdown("### Primary Prediction")

            if primary_class == "Melanoma":
                st.error(f"⚠️ {primary_class}")
            elif confidence >= 80:
                st.success(primary_class)
            else:
                st.warning(primary_class)

            st.metric(
                "Primary Confidence",
                f"{confidence:.2f}%"
            )

            st.progress(primary_probability)

            st.markdown("### Confidence Level")
            show_confidence_level(confidence_level)

            if confidence < 70:
                st.warning(
                    "The model is uncertain. The top predictions are close, "
                    "so this result should be interpreted carefully."
                )

            st.markdown("### Alternative Prediction")
            st.info(
                f"{secondary_class}: {secondary_probability * 100:.2f}%"
            )

            st.markdown("### Clinical Category")

            if category == "Cancerous":
                st.error(category)
            elif category == "Pre-cancerous":
                st.warning(category)
            else:
                st.success(category)

            st.markdown("### Description")
            st.info(
                DESCRIPTIONS.get(
                    primary_class,
                    "No description available."
                )
            )

            st.markdown("### Model")
            st.write("EfficientNet-B0")

            st.markdown("### Dataset")
            st.write("HAM10000")

            st.markdown("### Performance")
            st.write("Accuracy: 85%")
            st.write("Weighted F1: 86%")

            st.markdown("### Top Class Probabilities")

            for class_name, probability in top_predictions[:3]:
                st.write(f"{class_name}: {probability * 100:.2f}%")
                st.progress(probability)

st.warning("Educational project only. Not for medical diagnosis.")