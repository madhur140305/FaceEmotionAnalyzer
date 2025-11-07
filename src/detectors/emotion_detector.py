import cv2
from fer import FER
import torch

class EmotionDetector:
    def __init__(self, use_gpu=True):
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        self.detector = FER(mtcnn=True)
        print(f"[EmotionDetector] Initialized using {self.device.upper()}")

    def detect(self, frame):
        results = self.detector.detect_emotions(frame)
        emotions = []
        for face in results:
            (x, y, w, h) = face["box"]
            top_emotion, score = max(face["emotions"].items(), key=lambda x: x[1])
            emotions.append({
                "box": (x, y, w, h),
                "emotion": top_emotion,
                "confidence": score,
                "all_emotions": face["emotions"]
            })
        return emotions
