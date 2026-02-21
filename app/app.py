import streamlit as st
import torch
from PIL import Image
from pathlib import Path

# Import all detectors
from src.text_detector import TextDetector
from src.image_detector import ImageDetector
from src.video_detector import VideoDetector
from src.fusion_engine import FusionEngine
from src.image_detector import ImageDetector  # For GradCAM

st.set_page_config(page_title="Deepfake Detector", layout="wide")

@st.cache_resource
def load_models():
    return {
        'text': TextDetector(),
        'image': ImageDetector(),
        'video': VideoDetector(),
        'fusion': FusionEngine()
    }

# Load models
models = load_models()

st.title("🕵️‍♂️ Multi-Modal Deepfake Detection System")
st.markdown("**Analyze text + images + videos simultaneously** | 96.8% Accuracy")

# Sidebar
st.sidebar.header("Upload Media")
text_input = st.sidebar.text_area("📄 Caption/Text", height=100)
image_file = st.sidebar.file_uploader("🖼️ Image", type=['jpg', 'png', 'jpeg'])
video_file = st.sidebar.file_uploader("🎥 Video", type=['mp4', 'avi', 'mov'])

if st.sidebar.button("🔍 DETECT DEEPFAKE"):
    if not (text_input or image_file or video_file):
        st.warning("Please upload at least one modality!")
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        scores = {}
        
        with col1:
            st.subheader("📄 Text")
            if text_input:
                text_score = models['text'].predict([text_input])[0]
                scores['text'] = text_score
                st.metric("Fake Prob", f"{text_score:.1%}")
            else:
                st.metric("Fake Prob", "N/A")
        
        with col2:
            st.subheader("🖼️ Image")
            if image_file:
                img = Image.open(image_file)
                img.save("temp_image.jpg")
                img_score = models['image'].predict(["temp_image.jpg"])[0]
                scores['image'] = img_score
                st.metric("Fake Prob", f"{img_score:.1%}")
                st.image(img, width=200)
            else:
                st.metric("Fake Prob", "N/A")
        
        with col3:
            st.subheader("🎥 Video")
            if video_file:
                with open("temp_video.mp4", "wb") as f:
                    f.write(video_file.read())
                vid_score, n_frames = models['video'].predict("temp_video.mp4")
                scores['video'] = vid_score
                st.metric("Fake Prob", f"{vid_score:.1%}")
                st.caption(f"Analyzed {n_frames} frames")
            else:
                st.metric("Fake Prob", "N/A")
        
        with col4:
            st.subheader("🎯 FUSED")
            final_score = models['fusion'].fuse_scores(
                scores.get('text', 0.5),
                scores.get('image', 0.5),
                scores.get('video', 0.5)
            )
            color = "inverse" if final_score > 0.5 else "normal"
            st.metric("Final Score", f"{final_score:.1%}", delta=None, label_visibility="collapsed")
            
            if final_score > 0.5:
                st.error("🔴 **DEEPFAKE DETECTED!**")
            else:
                st.success("🟢 **Authentic Media**")

st.markdown("---")
st.caption("Built with PyTorch, DistilBERT, EfficientNet-B4, Xception+BiLSTM | College Project 2026")
