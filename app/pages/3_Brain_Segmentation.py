import os
import sys
import tempfile
import streamlit as st

sys.path.append(os.path.abspath("."))

from src.brain_tumor_segmentation.predict import predict_mask


st.set_page_config(
    page_title="Brain Tumor Segmentation",
    page_icon="🎯",
    layout="wide"
)

st.title("Brain Tumor Segmentation")
st.write(
    "Upload a brain MRI image to generate a predicted tumor mask and overlay visualization."
)

uploaded_file = st.file_uploader(
    "Upload brain MRI image",
    type=["jpg", "jpeg", "png", "tif", "tiff"]
)

if uploaded_file:
    suffix = os.path.splitext(uploaded_file.name)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(uploaded_file.read())
        image_path = temp.name

    st.image(
        image_path,
        caption="Uploaded MRI",
        width=350
    )

    if st.button("Generate Segmentation"):
        result = predict_mask(
            image_path=image_path,
            output_dir="segmentation_outputs"
        )

        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Original MRI")
            st.image(result["original_path"])

        with col2:
            st.subheader("Predicted Tumor Mask")
            st.image(result["mask_path"])

        with col3:
            st.subheader("Tumor Overlay")
            st.image(result["overlay_path"])
            st.caption(
                "Red overlay indicates the region predicted by the model as tumor tissue."
            )

        st.markdown("---")

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            st.metric(
                "Estimated Tumor Area",
                f"{result['tumor_area_percentage']:.2f}%"
            )

        with metric_col2:
            st.metric(
                "Model",
                "U-Net"
            )

        with metric_col3:
            st.metric(
                "Dice Score",
                "0.685"
            )

        st.markdown("### Dataset")
        st.write("LGG MRI Segmentation Dataset")

        st.info(
            "The mask is generated using an adaptive visualization threshold over the U-Net probability map. "
"It highlights the strongest tumor-like regions predicted by the model and is not a clinical measurement."
        )

st.warning("Educational project only. Not for medical diagnosis.")