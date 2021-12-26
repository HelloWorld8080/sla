import json
import torch
import os
import cv2

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))
print("ROOT",ROOT)

from models.common import DetectMultiBackend
from utils.general import (check_img_size,non_max_suppression,scale_coords)
from utils.torch_utils import select_device
from utils.augmentations import letterbox

import numpy as np

weights = ROOT / 'yolov5s.pt'
device = ''
augment = False
dnn = False
visualize = False
conf_thres=0.25
iou_thres=0.45
classes=None
agnostic_nms=False
max_det=1000
imgsz=(640, 640)
def init():
    global augment,weights,device,dnn,visualize,conf_thres,iou_thres,classes,agnostic_nms,max_det,imgsz
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride= model.stride
    print(model.names)
    global imgsz
    imgsz = check_img_size(imgsz, s=stride)
    return model
def process_image(handle=None, input_image=None, args=None, **kwargs):
    global augment,weights,device,dnn,visualize,conf_thres,iou_thres,classes,agnostic_nms,max_det,imgsz
    handle = init()
    stride, names, pt= handle.stride, handle.names, handle.pt
    handle.model.float()
    input_image = cv2.imread('./data/images/zidane.jpg')  # BGR
    # Padded resize
    img = letterbox(input_image, imgsz, stride=stride, auto=pt)[0]
    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(img)
    im = torch.from_numpy(im).to(device)
    im = im.float()
    im/=255
    if len(im.shape) == 3:
        im = im[None]
    pred = handle(im, augment=augment, visualize=visualize)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    bs = 1  # batch_size
    names = handle.names;
    for i, det in enumerate(pred):  # per image
        s = ''
        im0 = input_image.copy()
        s += '%gx%g ' % im.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0.copy()   # for save_crop

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
            fake_result = {}
            fake_result["algorithm_data"] = {
                "is_alert": False,
                "target_count": 0,
                "target_info": []
            }
            fake_result["model_data"] = {"objects": []}
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                object ={
                    "x": int(xyxy[0]),
                    "y": int(xyxy[1]),
                    "height": int(xyxy[3]-xyxy[1]),
                    "width": int(xyxy[2]-xyxy[0]),
                    "confidence": float(conf),
                    "name": names[c]
                    }
                fake_result["algorithm_data"]["is_alert"] = True if names[c]== 'person' else False
                fake_result["model_data"]["objects"].append(object)
                fake_result["algorithm_data"]["target_info"].append(object)
                if names[c] == 'person':
                    fake_result["algorithm_data"]["target_count"]+=1
                    break
                print("xywh",xyxy,*xyxy)
            print("fake_result", fake_result)
            return json.dumps(fake_result, indent=4)
        print(s)
            # """Do inference to analysis input_image and get output
#     Attributes:
#         handle: algorithm handle returned by init()
#         input_image (numpy.ndarray): image to be process, format: (h, w, c), BGR
#     Returns: process result

#     """
# Process image here
#     fake_result = {}
#     fake_result["algorithm_data"]={
#            "is_alert": false,
#            "target_count": 0,
#            "target_info": []
#        }
#     fake_result["model_data"]={"objects": []}
#     return json.dumps (fake_result , indent = 4)
process_image()

# {
# "algorithm_data": {
# "is_alert": true,
# "target_count": 1,
# "target_info": [{
# "x": 397,
# "y": 397,
# "height": 488,
# "width": 215,
# "confidence": 0.978979,
# "name": "slagcar"
# }]
# },
# "model_data": {
# "objects": [{
# "x": 716,
# "y": 716,
# "height": 646,
# "width": 233,
# "confidence": 0.999660,
# "name": "non_slagcar"
# }, {
# "x": 397,
# "y": 397,
# "height": 488,
# "width": 215,
# "confidence": 0.978979,
# "name": "slagcar"
# }]
# }
# }