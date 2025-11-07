import streamlit as st
import cv2
import numpy as np
from fer import FER
from deepface import DeepFace
import mediapipe as mp
from PIL import Image

# =========================
# INITIALIZATION
# =========================
st.set_page_config(page_title="Real-Time Emotion Analyzer", layout="wide")

st.title("🎥 Real-Time Emotion, Age & Gender Analyzer")
st.write("This app uses **DeepFace**, **FER**, and **MediaPipe** for human emotion, age, and gender recognition.")

mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)
emotion_detector = FER(mtcnn=True)

# =========================
# IMAGE ANALYSIS FUNCTION
# =========================
def analyze_image(image):
    image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    results = mp_face.process(image_rgb)

    if not results.detections:
        st.warning("No face detected! Please try again.")
        return None

    preds = DeepFace.analyze(image_rgb, actions=['age', 'gender', 'emotion'], enforce_detection=False, silent=True)
    emotions = preds[0]['emotion']
    top_emotion = max(emotions, key=emotions.get)
    age = preds[0]['age']
    gender = max(preds[0]['gender'], key=preds[0]['gender'].get)

    st.success(f"**Emotion:** {top_emotion} | **Age:** {age} | **Gender:** {gender}")
    st.bar_chart(emotions)

    annotated_image = np.array(image)
    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = annotated_image.shape
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(annotated_image, f"{top_emotion} ({int(emotions[top_emotion])}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    st.image(annotated_image, channels="BGR", caption="Analyzed Frame", use_container_width=True)

# =========================
# STREAMLIT UI
# =========================
option = st.radio("Select Input Mode:", ["📸 Upload Image", "🎞 Webcam Snapshot"])

if option == "📸 Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        if st.button("🔍 Analyze Image"):
            analyze_image(image)

elif option == "🎞 Webcam Snapshot":
    img_file_buffer = st.camera_input("Capture a photo")
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        st.image(image, caption="Captured Frame", use_container_width=True)
        if st.button("🔍 Analyze Snapshot"):
            analyze_image(image)

st.markdown("---")
st.caption("Built by **Madhur Kumar** | Powered by DeepFace + MediaPipe + FER + PyTorch")
