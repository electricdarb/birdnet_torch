"""
 Copyright (C) 2018-2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import numpy as np
import cv2
from openvino.inference_engine import IECore

YOLOV5N_ANCHORS = [
    [10,13, 16,30, 33,23],  # P3/8
    [30,61, 62,45, 59,119],  # P4/16
    [116,90, 156,198, 373,326],  # P5/32 
    ]

def parse_predictions(preds, threshold = .25, min_edge = .05, img_size = 640):
    """
    parse the 
    """
    # prevent any hazards
    preds = np.copy(preds)

    # filter preds bellow the min confidence threshold
    preds = preds[(preds[..., 4] > threshold)]
    
    # convert xywh to xyxy
    xymin = preds[..., :2] - preds[..., 2:4] / 2 # min = center - width / 2
    xymax = preds[..., :2] + preds[..., 2:4] / 2 # max = center - width / 2

    # clip to 0, 1
    preds[..., 0:2] = np.clip(xymin, 0., 1.) 
    preds[..., 2:4] = np.clip(xymax, 0., 1.) 

    # filter boxes that are too small
    preds = preds[(preds[..., 2] > min_edge)]
    preds = preds[(preds[..., 3] > min_edge)]

    # get max class ids
    class_id = np.expand_dims(np.argmax(preds[..., 5:], axis = -1), axis = -1)

    # concat xmin, ymin, xmax, ymax
    preds = np.concatenate([preds[..., :5], class_id], axis = - 1)

    return preds

def intersection_of_union(box_1, box_2):
    """
    0: xmin, 1: ymin, 2: xmax, 3: ymax
    """
    width_overlap = min(box_1[2], box_2[2]) - max(box_1[0], box_2[0])
    height_overlap = min(box_1[3], box_2[3]) - max(box_1[1], box_2[1])

    if width_overlap <= 0 or height_overlap <= 0: return 0
    area_of_overlap = width_overlap * height_overlap

    box_1_area = (box_1[3] - box_1[1]) * (box_1[2] - box_1[0])
    box_2_area = (box_2[3] - box_2[1]) * (box_2[2] - box_2[0])

    area_of_union = box_1_area + box_2_area - area_of_overlap

    if area_of_union == 0: return 0
    return area_of_overlap / area_of_union

def non_max_surpression(objects, threshold = .5):
    """
    eliminates redundant boxes by checking the interection of union
     - right now this is O(n^2), thats sloopy and should be fixed
     - this could get some easy gains through vectorization
    """
    l = len(objects)
    for i in range(l): 
        # if confidence is too low, skip this iteration
        if objects[i][4] <= 0.0: continue
        # loop over boxes that havent been compared to current box
        for j in range(i + 1, l):
            if intersection_of_union(objects[i], objects[j]) > threshold:
                # set the object with less confidence to 0
                if objects[i][4] > objects[j][4]: 
                    objects[j][4] = 0.
                else: 
                    objects[i][4] = 0.
    # return objects that dont have confidence 0
    return objects[(objects[..., 4] > 0.)] 


def create_detector(num_classes = 80, anchors = YOLOV5N_ANCHORS, img_size = 640):
    num_outputs = num_classes + 5 # num outpus per anchor
    num_anchors = len(anchors[0]) // 2
    
    def make_grid(nx, ny, i):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx), indexing='xy')

        grid = np.stack((xv, yv), axis = 2)
        grid = np.broadcast_to(grid, (1, num_anchors, ny, nx, 2))

        anchor_grid = np.array(anchors[i], dtype=np.float32).reshape((1, num_anchors, 1, 1, 2))
        anchor_grid = np.broadcast_to(anchor_grid,(1, num_anchors, ny, nx, 2))

        return grid, anchor_grid

    def detect(predictions): # call on one image and output all detections
        """
        returns: output of the convnet to confidence ratio
        """
        z = []

        for i, pred in enumerate(predictions): # for each prediction layer
            batch_size, _, ny, nx = pred.shape
            
            stride = img_size // ny

            grid, anchor_grid = make_grid(nx, ny, i)

            x = np.reshape(pred, (batch_size, num_anchors, num_outputs, ny, nx))
            x = np.transpose(x, (0, 1, 3, 4, 2))

            y = 1 / (1 + np.exp(-x)) # sigmoid activation

            y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + grid) * stride / img_size  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid / img_size # wh

            y = np.reshape(y, (batch_size, -1, num_outputs))
            z.append(y)
            
        return np.concatenate(z, axis = 1)
        
    return detect

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def prep_cv2_img(img_in, size = 640):
    img = img_in.copy() # prevent hazards 
    img = cv2.resize(img, (size, size)) # resize image to correct size 
    img = img[..., ::-1] # gbr to rbg
    img = np.transpose(img, (2, 0, 1)) # swap dims to c, w, h
    img = np.expand_dims(img, axis = 0) # expand dims for batch size of 1
    return {'images': img} 

def draw_boxes(img, objects):
    """
    img: cv2/numpy in rgb order
    objects: numpy of shape (n, 6) where:
        0:4 are xyxy, 4 is confidence, 5 is class
    """

    height, width = img.shape[:2]
    result = img.copy()

    for obj in objects:
        color = (0, 0, 255)
        xmin, ymin, xmax, ymax = int(obj[0] * width), int(obj[1] * height), int(obj[2] * width), int(obj[3] * height)

        cv2.rectangle(result, (xmin, ymin), (xmax, ymax), color, 1)
        cv2.putText(result, f"class: {int(obj[5])} @ {obj[4]:.1%}",
            (xmin, ymin),
            cv2.FONT_HERSHEY_SIMPLEX,
            .5, color, 1, 2)

    return result

if __name__ == "__main__":
    iou_threshold = .5
    conf_threshold = .4

    model_name = 'yolov5n'
    ie = IECore()
    device = 'CPU'#'MYRIAD'

    net = ie.read_network(model = f'models\\{model_name}.xml', weights = f'models\\{model_name}.bin')
    exec_net = ie.load_network(network = net, device_name = device, num_requests=2)

    detector = create_detector()  

    cam = cv2.VideoCapture(0)

    while True:
        _, img = cam.read()

        inputs = prep_cv2_img(img)

        outputs = exec_net.infer(inputs=inputs)
        outputs = [out for _, out in outputs.items()]

        preds = detector(outputs)
        objects = parse_predictions(preds, conf_threshold)
        objects = non_max_surpression(objects, threshold = iou_threshold)
        
        img_out = draw_boxes(img, objects)
        
        cv2.imshow('my webcam', img_out)

        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()
    