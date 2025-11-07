# 🎥 Face Emotion Analyzer  

> **A Real-Time Emotion, Age, and Gender Detection System** powered by DeepFace, FER, MediaPipe, and Streamlit.  
> Built by **[Madhur Kumar](https://github.com/madhurkumar)** 💻  

---

## 🌟 Overview  

The **Face Emotion Analyzer** detects **emotions, age, and gender** from live webcam feeds or uploaded images in real time.  
It combines the strengths of multiple frameworks to deliver fast, accurate, and visually appealing results.

🧠 Powered by:
- **DeepFace** → for age and gender detection  
- **FER (Facial Emotion Recognition)** → for emotion classification  
- **MediaPipe** → for ultra-fast face tracking  
- **PyTorch** / **TensorFlow** → for backend inference  
- **Streamlit** → for visualization and deployment  

---

## 🚀 Demo  

Try the live app on **Streamlit Cloud** 🌐  
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

---

## 🧩 Features  

✅ Real-time **Emotion Detection** (Happy, Sad, Angry, Fear, Surprise, Neutral, etc.)  
✅ **Age & Gender Prediction** using pre-trained DeepFace models  
✅ Runs both on **GPU (CUDA)** and **CPU**  
✅ **Logging System** — tracks session FPS, average age, and dominant emotion  
✅ Fully **streamlit-based UI** for user-friendly interaction  
✅ Modular **src/** structure (easy to extend)  

---

## 🏗️ Project Structure  


FaceEmotionAnalyzer/
│
├── src/
│ ├── realtime_analyzer.py # Core emotion + age + gender analyzer
│ └── detectors/ # Modular face detector components
│ ├── age_gender_detector.py
│ └── emotion_detector.py
│
├── streamlit_app.py # Streamlit UI frontend
├── requirements.txt # Dependencies
├── README.md # Project documentation
├── .gitignore
└── output/
└── session_logs/ # Auto-generated emotion logs

---

## ⚙️ Installation & Local Setup  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/<your-username>/FaceEmotionAnalyzer.git
cd FaceEmotionAnalyzer

Create a Virtual Environment

python -m venv venv
venv\Scripts\activate     # (Windows)
# or
source venv/bin/activate  # (Linux/Mac)

Install Dependencies

pip install -r requirements.txt

Run the Streamlit App

streamlit run streamlit_app.py

Console Output:

[INFO] Using device: cuda
[EmotionDetector] Initialized ✅
[AgeGenderDetector] Using DeepFace backend ✅
🎥 Webcam started — Press 'q' to quit.
[INIT] First Prediction Loaded: Age 24 | Gender Man | Emotion happy
✅ Stream closed successfully.

Streamlit Interface:

📸 Live webcam view

📈 Real-time FPS counter

😊 Detected emotion with bounding boxes

🧾 Auto-saved session summary in output/session_logs/

| Category              | Tools                                 |
| --------------------- | ------------------------------------- |
| Core Frameworks       | PyTorch, TensorFlow                   |
| Face Detection        | MediaPipe                             |
| Emotion Analysis      | FER                                   |
| Age/Gender Estimation | DeepFace                              |
| Visualization         | Streamlit, OpenCV, Plotly, Matplotlib |
| Logging               | CSV, Rich                             |
| Optimization          | CUDA, Multi-threading                 |
