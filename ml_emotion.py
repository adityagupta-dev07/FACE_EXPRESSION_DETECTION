import cv2
import mediapipe as mp
import math

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
    max_num_hands=2,
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

    # Brows
    brow_raise = lm[105].y - lm[66].y
    brow_down = lm[66].y - lm[105].y

    # Eyes
    left_eye = abs(lm[159].y - lm[145].y)
    right_eye = abs(lm[386].y - lm[374].y)

    # Sad (one lip down)
    mouth_diff = abs(lm[61].y - lm[291].y)

    # 😄 HAPPY
    if mouth_width > 0.06:
        return "happy"

    # 😲 SURPRISE
    if mouth_open > 0.045:
        return "surprise"

    # 😡 ANGRY
    if brow_down > 0.008 and mouth_open < 0.03:
        return "angry"

    # 😢 SAD
    if mouth_diff > 0.015 or (left_eye < 0.02 and right_eye < 0.02):
        return "sad"

    # 😨 FEAR
    if brow_raise > 0.015:
        return "fear"

    return "neutral"

# ==========================
# MAIN FUNCTION (CALLED BY FLASK)
# ==========================
def predict_emotion(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_result = face_mesh.process(rgb)
    hand_result = hands.process(rgb)

    expression = "neutral"

    # ---------- FACE ----------
    if face_result.multi_face_landmarks:
        face_landmarks = face_result.multi_face_landmarks[0]
        lm = face_landmarks.landmark

        expression = detect_face_emotion(lm)

        mp_draw.draw_landmarks(
            frame,
            face_landmarks,
            mp_face.FACEMESH_TESSELATION,
            mp_draw.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
            mp_draw.DrawingSpec(color=(0,255,0), thickness=1)
        )

    # ---------- HAND GESTURES (OVERRIDE FACE) ----------
    if hand_result.multi_hand_landmarks:
        for hand_landmarks in hand_result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Landmarks
            thumb_tip = hand_landmarks.landmark[4]
            thumb_ip = hand_landmarks.landmark[3]
            thumb_mcp = hand_landmarks.landmark[2]

            finger_tips = [8, 12, 16, 20]
            finger_pips = [6, 10, 14, 18]

            # 👍 THUMBS UP (ROBUST)
            thumb_extended = (
                abs(thumb_tip.x - thumb_mcp.x) > 0.04 or
                abs(thumb_tip.y - thumb_mcp.y) > 0.04
            )

            fingers_folded = all(
                hand_landmarks.landmark[tip].y >
                hand_landmarks.landmark[pip].y
                for tip, pip in zip(finger_tips, finger_pips)
            )

            thumb_above_fingers = all(
                thumb_tip.y < hand_landmarks.landmark[tip].y
                for tip in finger_tips
            )

            if thumb_extended and fingers_folded and thumb_above_fingers:
                return "thumbs_up", None

            # 🤷 WHAT DID I DO (OPEN PALM)
            fingers_open = all(
                hand_landmarks.landmark[tip].y <
                hand_landmarks.landmark[pip].y
                for tip, pip in zip(finger_tips, finger_pips)
            )

            thumb_open = abs(thumb_tip.x - thumb_ip.x) > 0.04

            if fingers_open and thumb_open:
                return "what_did_i_do", None

    return expression, None
