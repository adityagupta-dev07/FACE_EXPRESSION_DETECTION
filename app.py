from flask import Flask, render_template, Response
import cv2

from ml_emotion import predict_emotion
from meme_loader import get_meme_by_expression

# ======================
# GLOBAL STATE
# ======================
current_expression = "neutral"

app = Flask(__name__)
camera = cv2.VideoCapture(0)

# ======================
# VIDEO STREAM
# ======================
def gen_frames():
    global current_expression

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        # 🎭 FACE EMOTION DETECTION
        emotion, face_box = predict_emotion(frame)

        # ✅ ALWAYS update expression (fixes meme mismatch)
        current_expression = emotion

        # Draw face box only if detected
        if face_box:
            x, y, w, h = face_box

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Emotion: {emotion}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )

# ======================
# ROUTES
# ======================
@app.route("/")
def index():
    meme = get_meme_by_expression(current_expression)

    if meme is None:
        meme = {
            "meme_name": "Default",
            "image": "neutral.gif",
            "default_caption": "Waiting for expression..."
        }

    return render_template(
        "index.html",
        expression=current_expression,
        meme_name=meme["meme_name"],
        caption=meme["default_caption"],
        image=meme["image"]
    )
@app.route("/video")
def video():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ======================
# RUN
# ======================
if __name__ == "__main__":
    app.run(debug=True)
