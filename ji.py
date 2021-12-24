import json
from torch import load

def init():

# """Initialize model
#     Returns: model
#     """

def process_image(handle=None, input_image=None, args=None, **kwargs):
    # model = init()
    # print(model)


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