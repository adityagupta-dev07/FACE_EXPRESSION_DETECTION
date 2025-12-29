import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# ==========================
# LOAD EMOTION MODEL
# ==========================
MODEL_PATH = "emotion_model.h5"
CASCADE_PATH = "haarcascade_frontalface_default.xml"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ emotion_model.h5 not found")

if not os.path.exists(CASCADE_PATH):
    raise FileNotFoundError("❌ haarcascade_frontalface_default.xml not found")

emotion_model = load_model(MODEL_PATH, compile=False)

# ==========================
# FACE DETECTOR
# ==========================
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# ==========================
# EMOTION LABELS (DO NOT CHANGE)
# ==========================
emotion_labels = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral"
]

# ==========================
# PREDICT FUNCTION
# ==========================
def predict_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    if len(faces) == 0:
        return "neutral", None

    # Take the largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    roi = gray[y:y+h, x:x+w]

    try:
        roi = cv2.resize(roi, (64, 64))
    except:
        return "neutral", None

    roi = roi.astype("float32") / 255.0
    roi = np.expand_dims(roi, axis=-1)
    roi = np.expand_dims(roi, axis=0)

    preds = emotion_model.predict(roi, verbose=0)
    emotion = emotion_labels[int(np.argmax(preds))]

    return emotion, (x, y, w, h)
