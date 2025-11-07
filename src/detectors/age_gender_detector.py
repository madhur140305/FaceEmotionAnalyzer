from deepface import DeepFace
import torch

class AgeGenderDetector:
    def __init__(self, use_gpu=True):
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        print(f"[AgeGenderDetector] Initialized using {self.device.upper()}")

    def analyze(self, frame):
        try:
            result = DeepFace.analyze(
                frame, actions=['age', 'gender'], enforce_detection=False, detector_backend='retinaface'
            )[0]
            return {
                "age": result.get('age', None),
                "gender": result.get('dominant_gender', None)
            }
        except Exception as e:
            print("[AgeGenderDetector] Error:", e)
            return {"age": None, "gender": None}
