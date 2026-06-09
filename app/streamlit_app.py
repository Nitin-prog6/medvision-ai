import streamlit as st

st.set_page_config(
    page_title="MedVision AI",
    page_icon="🧠",
    layout="wide"
)

st.title("MedVision AI")
st.subheader("Multi-Disease Medical Imaging Platform")

st.write(
    """
    MedVision AI is an educational deep learning platform for medical image analysis.
    It combines image classification, explainable AI, and tumor segmentation into one interface.
    """
)

st.warning(
    "This project is for educational and research purposes only. "
    "It is not a medical diagnostic tool."
)

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("## 🩺 Skin Cancer")
    st.write("Classifies skin lesion images into 7 diagnostic categories.")
    st.metric("Accuracy", "85%")
    st.metric("Weighted F1", "86%")
    st.write("Features: EfficientNet-B0, Grad-CAM, class probabilities")

with col2:
    st.markdown("## 🧠 Brain Tumor")
    st.write("Classifies MRI images as tumor or no tumor.")
    st.metric("Accuracy", "92%")
    st.metric("Tumor Recall", "97%")
    st.write("Features: EfficientNet-B0, Grad-CAM, confidence scoring")

with col3:
    st.markdown("## 🎯 Tumor Segmentation")
    st.write("Generates predicted tumor masks from MRI slices.")
    st.metric("Dice Score", "0.685")
    st.metric("Model", "U-Net")
    st.write("Features: mask prediction, overlay visualization, tumor area estimate")

st.markdown("---")

st.markdown("## Project Highlights")

st.write(
    """
    - Multi-model medical imaging workflow
    - CNN-based image classification
    - U-Net tumor segmentation
    - Grad-CAM explainability
    - Confidence levels and alternative predictions
    - Streamlit-based interactive AI demo
    """
)

st.markdown("## Tech Stack")

st.write(
    """
    Python · PyTorch · Torchvision · OpenCV · Streamlit · Grad-CAM · Scikit-learn
    """
)

st.markdown("---")

st.info(
    "Use the sidebar to test Skin Cancer Classification, Brain Tumor Classification, "
    "or Brain Tumor Segmentation."
)