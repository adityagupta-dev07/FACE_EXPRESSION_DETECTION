import cv2
import mediapipe as mp
from collections import deque

emotion_buffer = deque(maxlen=7)

mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

def detect_face_emotion(lm):
    mouth_open = abs(lm[13].y - lm[14].y)
    mouth_width = abs(lm[61].x - lm[291].x)
    lip_drop = abs(lm[61].y - lm[291].y)
    brow_dist = lm[66].y - lm[105].y

    if mouth_open < 0.02 and mouth_width < 0.055:
        return "neutral"
    if mouth_open > 0.055:
        return "surprise"
    if mouth_width > 0.065:
        return "happy"
    if lip_drop > 0.024:
        return "sad"
    if brow_dist > 0.012:
        return "angry"
    return "neutral"

def predict_emotion(frame):
    global emotion_buffer

    frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    emotion = "neutral"
    lm = None

    face_result = face_mesh.process(rgb)
    hand_result = hands.process(rgb)

    if face_result.multi_face_landmarks:
        face = face_result.multi_face_landmarks[0]
        lm = face.landmark
        emotion = detect_face_emotion(lm)

        mp_draw.draw_landmarks(frame, face, mp_face.FACEMESH_TESSELATION)
        cv2.putText(frame, "FACE DETECTED", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    if hand_result.multi_hand_landmarks and lm:
        for hand in hand_result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, "HAND DETECTED", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,200,0), 2)

            index_tip = hand.landmark[8]
            index_pip = hand.landmark[6]

            upper_lip_y = lm[13].y
            mouth_open = abs(lm[13].y - lm[14].y)
            mouth_width = abs(lm[61].x - lm[291].x)

            if abs(index_tip.y - upper_lip_y) < 0.05 and mouth_open < 0.02:
                emotion = "thinking"
                break

            if index_tip.y < index_pip.y and mouth_width > 0.06:
                emotion = "idea"
                break

    emotion_buffer.append(emotion)
    emotion = max(set(emotion_buffer), key=emotion_buffer.count)

    return frame, emotion
