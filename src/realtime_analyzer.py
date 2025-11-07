import cv2
import torch
import time
import mediapipe as mp
from fer import FER
from deepface import DeepFace
import threading
import csv, os
from datetime import datetime
from collections import Counter

# =========================
# INITIALIZATION
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# Initialize detectors
mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)
emotion_detector = FER(mtcnn=True)
print("[EmotionDetector] Initialized ✅")
print("[AgeGenderDetector] Using DeepFace backend ✅")

# =========================
# SESSION LOGGER SETUP
# =========================
os.makedirs("output/session_logs", exist_ok=True)
session_start = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = f"output/session_logs/session_{session_start}.csv"

with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "emotion", "age", "gender", "fps"])

print(f"🧾 Logging session to: {log_path}")

fps_list, age_list, emotion_list = [], [], []
start_time = time.time()

# Global variables
last_results = {"emotion": "N/A", "age": "N/A", "gender": "N/A"}
frame_count = 0
lock = threading.Lock()

# =========================
# BACKGROUND ANALYSIS THREAD
# =========================
def analyze_face_thread(frame, force=False):
    global last_results
    try:
        preds = DeepFace.analyze(frame, actions=['age', 'gender'], enforce_detection=False, silent=True)
        emotions = emotion_detector.detect_emotions(frame)
        if emotions:
            top_emotion = max(emotions[0]["emotions"], key=emotions[0]["emotions"].get)
        else:
            top_emotion = "Neutral"

        with lock:
            last_results = {
                "emotion": top_emotion,
                "age": int(preds[0]['age']),
                "gender": max(preds[0]['gender'], key=preds[0]['gender'].get)
            }

        if force:
            print(f"[INIT] First Prediction Loaded: Age {last_results['age']} | Gender {last_results['gender']} | Emotion {last_results['emotion']}")

    except Exception as e:
        print("[WARN] DeepFace failed on a frame:", e)
        pass


# =========================
# MAIN LOOP
# =========================
cap = cv2.VideoCapture(0)
print("🎥 Webcam started — Press 'q' to quit.")

fps_start = time.time()
frame_counter = 0

# Warm-up: analyze one frame immediately
ret, init_frame = cap.read()
if ret:
    analyze_face_thread(init_frame, force=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_counter += 1
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face detection using MediaPipe
    results = mp_face.process(rgb_frame)
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Run heavy analysis every 10th frame in background
            if frame_counter % 10 == 0:
                threading.Thread(target=analyze_face_thread, args=(frame.copy(),), daemon=True).start()

            # Show cached predictions
            with lock:
                cv2.putText(frame, f"{last_results['emotion']}", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(frame, f"Age: {last_results['age']}, Gender: {last_results['gender']}",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # FPS Calculation
    if frame_counter % 30 == 0:
        fps_end = time.time()
        fps = 30 / (fps_end - fps_start)
        fps_start = fps_end
        cv2.putText(frame, f"FPS: {fps:.2f}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    # =========================
    # LOG DATA EACH FRAME
    # =========================
    if last_results["emotion"] != "Neutral" or last_results["age"] != "N/A":
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                last_results["emotion"],
                last_results["age"],
                last_results["gender"],
                round(fps, 2) if 'fps' in locals() else 0
            ])
        fps_list.append(fps if 'fps' in locals() else 0)
        age_list.append(last_results["age"])
        emotion_list.append(last_results["emotion"])

    # Display window
    cv2.imshow("Real-Time Emotion Analyzer", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# SESSION SUMMARY
# =========================
cap.release()
cv2.destroyAllWindows()

if fps_list:
    avg_fps = sum(fps_list) / len(fps_list)
    avg_age = sum(a for a in age_list if isinstance(a, (int, float))) / len(age_list)
    dominant_emotion = Counter(emotion_list).most_common(1)[0][0]
    duration = round(time.time() - start_time, 2)
    print("\n📊 Session Summary:")
    print(f" - Duration: {duration} sec")
    print(f" - Avg FPS: {avg_fps:.2f}")
    print(f" - Avg Age: {avg_age:.1f}")
    print(f" - Dominant Emotion: {dominant_emotion}")
    print(f" - Log saved to: {log_path}")

print("✅ Stream closed successfully.")
