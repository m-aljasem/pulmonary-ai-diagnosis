"""Streamlit app for chest X-ray classification"""
import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path

from src.explainability import ModelExplainer
from src.model import build_chest_xray_model

CLASSES = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema',
           'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening',
           'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']

st.set_page_config(page_title="Chest X-ray Classification", page_icon="🫁")
st.title("🫁 Chest X-ray Multi-Classification")

MODELS_DIR = Path("models")
WEIGHTS_PATH = MODELS_DIR / "chest_xray_model.h5"

uploaded_file = st.file_uploader("Upload chest X-ray", type=['png', 'jpg'])
if uploaded_file:
    img = Image.open(uploaded_file).resize((224, 224))
    st.image(img)
    img_array = np.array(img) / 255.0
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    img_array = np.expand_dims(img_array, 0)
    
    if st.button("Classify"):
        # Ensure models directory exists
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        model = build_chest_xray_model()
        if WEIGHTS_PATH.exists():
            st.info(f"Loading trained weights from `{WEIGHTS_PATH}`")
            model.load_weights(str(WEIGHTS_PATH))
        else:
            st.warning(
                "⚠️ Trained weights not found. Using randomly initialized model "
                "(predictions will be unreliable). Train the model first."
            )

        pred = model.predict(img_array, verbose=0)[0]
        
        st.subheader("Detected Conditions:")
        detected = []
        for i, (cls, prob) in enumerate(zip(CLASSES, pred)):
            if prob > 0.5:
                st.warning(f"**{cls}**: {prob:.2%}")
                detected.append(i)
        
        # Explainability
        if detected:
            st.divider()
            st.subheader("🔍 Explainability")
            selected_class = st.selectbox("Explain condition:", [CLASSES[i] for i in detected], key="explain_class")
            if st.button("Explain Prediction (SHAP)"):
                try:
                    import matplotlib.pyplot as plt
                    import shap
                    class_idx = CLASSES.index(selected_class)
                    explainer = ModelExplainer(model, img_array[:1])
                    shap_values = explainer.explain_instance(img_array, plot=False, class_idx=class_idx)
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    axes[0].imshow(img_array[0])
                    axes[0].set_title('Original')
                    axes[0].axis('off')
                    shap_image = np.abs(shap_values[0]).sum(axis=2) if len(shap_values[0].shape) == 3 else np.abs(shap_values[0])
                    axes[1].imshow(shap_image, cmap='hot')
                    axes[1].set_title(f'SHAP for {selected_class}')
                    axes[1].axis('off')
                    axes[2].imshow(img_array[0])
                    axes[2].imshow(shap_image, cmap='hot', alpha=0.5)
                    axes[2].set_title('Overlay')
                    axes[2].axis('off')
                    plt.tight_layout()
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error: {str(e)}")

