import cv2
import mediapipe as mp
from collections import deque

# ==========================
# STABILITY BUFFER
# ==========================
emotion_buffer = deque(maxlen=7)

# ==========================
# MEDIAPIPE SETUP
# ==========================
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================
# FACE EMOTION LOGIC
# ==========================
def detect_face_emotion(lm):
    # Mouth
    mouth_open = abs(lm[13].y - lm[14].y)
    mouth_width = abs(lm[61].x - lm[291].x)
    lip_drop = abs(lm[61].y - lm[291].y)

    # Eyes
    left_eye = abs(lm[159].y - lm[145].y)
    right_eye = abs(lm[386].y - lm[374].y)

    # Brows
    brow_dist = lm[66].y - lm[105].y

    # 🧘 NEUTRAL LOCK (VERY IMPORTANT)
    if (
        mouth_open < 0.02 and
        mouth_width < 0.055 and
        lip_drop < 0.02 and
        abs(brow_dist) < 0.006 and
        0.02 < left_eye < 0.03 and
        0.02 < right_eye < 0.03
    ):
        return "neutral"

    # 😲 SHOCKED (mouth wide open)
    if mouth_open > 0.055:
        return "surprise"

    # 😄 HAPPY (smile)
    if mouth_width > 0.065:
        return "happy"

    # 😢 CRY / SAD (STRICT: MULTIPLE SIGNS)
    if (
        lip_drop > 0.024 and
        left_eye < 0.02 and
        right_eye < 0.02
    ):
        return "sad"

    # 😡 ANGRY
    if brow_dist > 0.012 and mouth_open < 0.03:
        return "angry"

    # 😨 FEAR
    if mouth_open > 0.035 and brow_dist < -0.01:
        return "fear"

    # 🤔 THINKING (face-only fallback)
    if mouth_open < 0.02 and brow_dist > 0.008:
        return "thinking"

    return "neutral"

# ==========================
# MAIN FUNCTION
# ==========================
def predict_emotion(frame):
    global emotion_buffer

    # Light normalization
    frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    emotion = "neutral"

    face_result = face_mesh.process(rgb)
    hand_result = hands.process(rgb)

    lm = None

    # ---------- FACE ----------
    if face_result.multi_face_landmarks:
        face = face_result.multi_face_landmarks[0]
        lm = face.landmark
        emotion = detect_face_emotion(lm)

        mp_draw.draw_landmarks(
            frame,
            face,
            mp_face.FACEMESH_TESSELATION
        )

    # ---------- HAND + FACE COMBOS ----------
    if hand_result.multi_hand_landmarks and lm:
        for hand in hand_result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # Hand landmarks
            index_tip = hand.landmark[8]
            index_pip = hand.landmark[6]

            # Face landmarks
            upper_lip_y = lm[13].y
            mouth_open = abs(lm[13].y - lm[14].y)
            mouth_width = abs(lm[61].x - lm[291].x)

            # 🤔 THINKING (hand near lips)
            if abs(index_tip.y - upper_lip_y) < 0.05 and mouth_open < 0.02:
                emotion = "thinking"
                break

            # 💡 IDEA (index up + smile)
            index_up = index_tip.y < index_pip.y
            smiling = mouth_width > 0.06

            if index_up and smiling:
                emotion = "idea"
                break

    # ---------- STABILITY ----------
    emotion_buffer.append(emotion)
    emotion = max(set(emotion_buffer), key=emotion_buffer.count)

    return frame, emotion
