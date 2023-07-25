# CACIOUS CODING
# Data     : 7/25/23  2:32 PM
# File name: inference_class
# Desc     :
import time

from util.args_util import get_args
from util.model_util import create_training_model
from util.weight_util import load_model
from util.pic_util import pic2tensor
import torch


args = get_args()

device = torch.device('cpu')
training_model = create_training_model(args)
training_model = load_model(training_model, args.weight_path).to(device)
training_model.eval()
model = training_model.model

img0 = pic2tensor(args.img0_path, args).to(device)

start = time.time()
output = model(img0)
end = time.time()
print(end - start)
print(torch.argmax(output, dim=1))
