#!/usr/bin/env python3
"""
Multi-Modal Deepfake Detection System - Main CLI
Usage: python main.py --text "fake text" --image fake.jpg --video fake.mp4
"""

import argparse
from pathlib import Path
from src.text_detector import TextDetector
from src.image_detector import ImageDetector
from src.video_detector import VideoDetector
from src.fusion_engine import FusionEngine
from src.explainability import ExplainabilityEngine

def main():
    parser = argparse.ArgumentParser(description="Multi-Modal Deepfake Detector")
    parser.add_argument('--text', type=str, help="Text/caption to analyze")
    parser.add_argument('--image', type=str, help="Image path")
    parser.add_argument('--video', type=str, help="Video path")
    parser.add_argument('--explain', action='store_true', help="Generate explanations")
    
    args = parser.parse_args()
    
    # Initialize detectors
    text_det = TextDetector()
    img_det = ImageDetector()
    vid_det = VideoDetector()
    fusion = FusionEngine()
    explainer = ExplainabilityEngine()
    
    scores = {}
    
    if args.text:
        scores['text'] = text_det.predict([args.text])[0]
        print(f"📄 Text Score: {scores['text']:.3f}")
    
    if args.image:
        scores['image'] = img_det.predict([args.image])[0]
        print(f"🖼️ Image Score: {scores['image']:.3f}")
    
    if args.video:
        scores['video'], n_frames = vid_det.predict(args.video)
        print(f"🎥 Video Score: {scores['video']:.3f} ({n_frames} frames)")
    
    # Fusion
    final_score = fusion.fuse_scores(
        scores.get('text', 0.5),
        scores.get('image', 0.5),
        scores.get('video', 0.5)
    )
    
    label = "🟢 REAL" if final_score < 0.5 else "🔴 FAKE"
    print(f"\n🎯 FINAL PREDICTION: {label}")
    print(f"   Fake Probability: {final_score:.3f}")
    
    if args.explain:
        explainer.generate_report(None, None, None, final_score)

if __name__ == "__main__":
    main()
