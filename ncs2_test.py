from openvino.inference_engine import IECore
import numpy as np

model_path = "./models/yolov5n"

ie = IECore()
device = 'MYRIAD'

net = ie.read_network(model = f'{model_path}.xml', weights = f'{model_path}.bin')
exec_net = ie.load_network(network = net, device_name = device, num_requests=2)

img = np.zeros((1, 3, 640, 640))

sample_out = exec_net.infer(inputs = {'images': img})
ie = IECore()
device = 'MYRIAD'

net = ie.read_network(model = f'{model_path}.xml', weights = f'{model_path}.bin')
exec_net = ie.load_network(network = net, device_name = device, num_requests=2)