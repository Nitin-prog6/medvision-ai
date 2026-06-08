import os
import sys
import tempfile
import streamlit as st

sys.path.append(os.path.abspath("."))

from src.brain_tumor.predict import predict_image
from src.brain_tumor.gradcam import generate_gradcam


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
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

st.title("Brain Tumor Classification")
st.write(
    "Upload a brain MRI image to classify tumor presence and generate Grad-CAM explainability."
)

uploaded_file = st.file_uploader(
    "Upload brain MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        temp.write(uploaded_file.read())
        image_path = temp.name

    st.image(
        image_path,
        caption="Uploaded MRI",
        width=350
    )

    if st.button("Analyze MRI"):
        result = predict_image(image_path)

        predicted_class = result["predicted_class"]
        confidence = result["confidence"] * 100
        confidence_level = get_confidence_level(confidence)

        alternative_class = (
            "No Tumor"
            if predicted_class == "Tumor"
            else "Tumor"
        )

        alternative_confidence = 100 - confidence

        gradcam_path = "brain_gradcam_streamlit.jpg"
        generate_gradcam(image_path, gradcam_path)

        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Original MRI")
            st.image(image_path)

        with col2:
            st.subheader("Grad-CAM")
            st.image(gradcam_path)

            if predicted_class == "Tumor":
                st.caption(
                    "Highlighted regions show areas that influenced the tumor prediction."
                )
            else:
                st.caption(
                    "Highlighted regions show areas that influenced the no-tumor prediction. "
                    "They do not represent tumor location."
                )

        with col3:
            st.subheader("Prediction Details")

            st.markdown("### Predicted Result")

            if predicted_class == "Tumor":
                st.error("⚠️ Tumor Detected")
            else:
                st.success("No Tumor Detected")

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            st.progress(confidence / 100)

            st.markdown("### Confidence Level")
            show_confidence_level(confidence_level)

            if confidence < 70:
                st.warning(
                    "The model is uncertain. The result should be interpreted carefully."
                )

            st.markdown("### Alternative Prediction")
            st.info(
                f"{alternative_class}: {alternative_confidence:.2f}%"
            )

            st.markdown("### Model")
            st.write("EfficientNet-B0")

            st.markdown("### Dataset")
            st.write("Brain MRI Tumor Dataset")

            st.markdown("### Performance")
            st.write("Accuracy: 92%")
            st.write("Macro F1: 92%")
            st.write("Tumor Recall: 97%")

st.warning("Educational project only. Not for medical diagnosis.")