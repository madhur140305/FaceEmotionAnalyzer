import os
import torch
import cv2
import mediapipe as mp
import deepface

# --- Define folder structure ---
folders = [
    "data/fer2013",
    "data/utkface",
    "data/affectnet_subset",
    "models",
    "src",
    "notebooks",
    "outputs"
]

base_path = "D:\\FaceEmotionAnalyzer"
for folder in folders:
    path = os.path.join(base_path, folder)
    os.makedirs(path, exist_ok=True)

print("✅ Folder structure created successfully!")

# --- Verify key packages ---
print("\n🔍 Verifying your environment:")
print("OpenCV version:", cv2.__version__)
print("DeepFace version:", deepface.__version__)
print("MediaPipe version:", mp.__version__)

# --- Verify GPU availability ---
gpu_available = torch.cuda.is_available()
print("\n🚀 GPU Enabled:", gpu_available)
if gpu_available:
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("⚠️ GPU not detected in torch")

print("\n✅ Setup verification complete! You’re ready to start coding.")
