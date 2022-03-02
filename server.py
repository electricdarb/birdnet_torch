import cv2
from cv2 import exp
from infer import ObjectDetector

from flask import Flask, render_template, Response
from threading import Lock

from openvino.inference_engine import IECore

app = Flask(__name__)

model_name = 'yolov5n'
device = 'MYRIAD'

# hack to not load ncs2 more than one
print('name:' ,__name__)
if __name__ != '__main__':
    print('Setting up network for NCS2')
    object_detector = ObjectDetector(model_name, device)
else:
    object_detector = None

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


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

