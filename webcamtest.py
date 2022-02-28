"""
Simply display the contents of the webcam with optional mirroring using OpenCV 
via the new Pythonic cv2 interface.  Press <esc> to quit.
"""

import cv2
from openvino.inference_engine import IECore
import numpy as np

def show_webcam(net):
    cam = cv2.VideoCapture(0)
    while True:
        _, img_orginal = cam.read()

        img = img_orginal[..., ::-1]
        img = cv2.resize(img, (640, 640))
        img = np.moveaxis(img, -1, 0) # put channels in side 1
        img = np.expand_dims(img, axis = 0)

        output = net.infer(inputs = {'images': img})
        print(output.shape)

        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()


def main():
    model_path = "./models/yolov5n"

    ie = IECore()
    device = 'MYRIAD'
    net = ie.read_network(model = f'{model_path}.xml', weights = f'{model_path}.bin')
    exec_net = ie.load_network(network = net, device_name = device, num_requests=2)

    show_webcam(exec_net)


if __name__ == '__main__':
    main()