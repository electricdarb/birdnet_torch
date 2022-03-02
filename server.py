import cv2
from cv2 import exp
from infer import prep_cv2_img, parse_predictions, create_detector, draw_boxes, non_max_surpression, YOLOV5N_ANCHORS

from flask import Flask, render_template, Response
from threading import Lock

from openvino.inference_engine import IECore

app = Flask(__name__)

lock = Lock()

print('Setting up network for NCS2')

ie = IECore()

device = 'MYRIAD'
model_name = 'yolov5n'

net = ie.read_network(model = f'models/{model_name}.xml', weights = f'models/{model_name}.bin')
exec_net = ie.load_network(network = net, device_name = device, num_requests=2)


class ObjectDetector(): # for some reason this needs to be in here
    """
    A nice abstraction of the object detector
    
    inputs a cv2 image (bgr)
    outputs a cv2 image, same size as input (bgr)

    design thoughts:
    i guess this could be a function returning a function rather than a clas
    """
    def __init__(self, 
            model_name, 
            device = 'CPU',
            conf_threshold = .5, 
            iou_threshold = .5,
            num_classes = 80,
            anchors = YOLOV5N_ANCHORS,
            img_size = 640
            ):

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        self.detector = create_detector(num_classes = num_classes, anchors = anchors, img_size = img_size)
    
    def __call__(self, img):
        inputs = prep_cv2_img(img)
        with lock: # prevent multiple calls from happening at once
            outputs = exec_net.infer(inputs=inputs)

        outputs = [out for _, out in outputs.items()]

        preds = self.detector(outputs)

        objects = parse_predictions(preds, threshold = self.conf_threshold)
        objects = non_max_surpression(objects, threshold = self.iou_threshold)

        img_out = draw_boxes(img, objects)
        return img_out



object_detector = ObjectDetector(model_name, device)
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

