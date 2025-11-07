import cv2
from fer import FER
import torch

print("✅ Using device:", "GPU" if torch.cuda.is_available() else "CPU")

# Initialize FER detector
detector = FER(mtcnn=True)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not access webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect emotions in the frame
    result = detector.detect_emotions(frame)

    # Draw results on frame
    for face in result:
        (x, y, w, h) = face["box"]
        emotion, score = max(face["emotions"].items(), key=lambda x: x[1])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{emotion} ({score:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Face Emotion Analyzer", frame)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
