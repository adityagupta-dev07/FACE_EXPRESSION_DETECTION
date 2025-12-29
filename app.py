from flask import Flask, render_template, Response, jsonify
import cv2
from detector import predict_emotion

app = Flask(__name__)

camera = cv2.VideoCapture(0)
latest_emotion = "neutral"

def gen_frames():
    global latest_emotion
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame, latest_emotion = predict_emotion(frame)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotion')
def emotion():
    return jsonify({"emotion": latest_emotion})

if __name__ == "__main__":
    app.run(debug=False, threaded=True)
