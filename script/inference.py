# CACIOUS CODING
# Data     : 7/24/23  5:37 PM
# File name: inference
# Desc     :
from util.args_util import get_args
from util.pic_util import pic2tensor
from util.model_util import create_training_model
from util.weight_util import load_model
import torch

args = get_args()
args.device = torch.device("cpu")

img0 = pic2tensor(args.img0_path, args)
img1 = pic2tensor(args.img1_path, args)

training_model = create_training_model(args)
training_model = load_model(training_model, args.weight_path)
model = training_model.model
model.eval()

_input = torch.cat([img0, img1], 0)
_output = torch.nn.functional.normalize(model(_input))

img0_reid = _output[0]
img1_reid = _output[1]

cos_dis = torch.dot(img0_reid, img1_reid).item()
match_score = (cos_dis + 1) / 2
print(match_score)
