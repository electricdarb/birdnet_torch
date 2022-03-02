import cv2
from infer import ObjectDetector

from flask import Flask, render_template, Response

object_detector = ObjectDetector('yolov5n')
camera = cv2.VideoCapture(0)

LOGGER = print

def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            LOGGER('Camera read has failed')
            break
        detections = object_detector(frame)
        _, buffer = cv2.imencode('.jpg', detections)
        out = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + out + b'\r\n')

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

