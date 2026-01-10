from flask import Flask, render_template, Response, jsonify
import cv2
from detector import predict_emotion

app = Flask(__name__)

camera = None
camera_active = False
latest_emotion = "neutral"


def gen_frames():
    global latest_emotion, camera
    while True:
        if not camera_active or camera is None:
            continue

        success, frame = camera.read()
        if not success:
            continue

        frame, latest_emotion = predict_emotion(frame)

        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start_camera")
def start_camera():
    global camera, camera_active
    if camera is None:
        camera = cv2.VideoCapture(0)
    camera_active = True
    return jsonify({"status": "started"})


@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/emotion")
def emotion():
    return jsonify({"emotion": latest_emotion})


if __name__ == "__main__":
    app.run(debug=False, threaded=True)
