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
    MedVision AI is a deep learning medical imaging project that supports:

    - Skin cancer classification
    - Brain tumor classification
    - Brain tumor segmentation

    The system uses CNN-based models, U-Net segmentation, confidence scores,
    and explainable AI visualizations.
    """
)

st.warning(
    "This project is for educational and research purposes only. "
    "It is not a medical diagnostic tool."
)

st.markdown("### Current Model Performance")

st.table({
    "Model": [
        "Skin Cancer Classifier",
        "Brain Tumor Classifier",
        "Brain Tumor U-Net Segmentation"
    ],
    "Metric": [
        "85% Accuracy / 86% Weighted F1",
        "92% Accuracy / 97% Tumor Recall",
        "0.685 Dice Score"
    ]
})