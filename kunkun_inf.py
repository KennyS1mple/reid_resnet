# CACIOUS CODING
# Data     : 7/27/23  5:05 PM
# File name: kunkun_inf
# Desc     :
import torch
from util.args_util import get_args
from util.model_util import create_training_model
from util.weight_util import load_model
from util.infer_util import get_reid_normed

args = get_args()
args.device = torch.device("cpu")
args.num_class = None

training_model = create_training_model(args)
training_model = load_model(training_model, args.weight_path)
model = training_model.model

reid_normed = get_reid_normed(model, args.inf_img_path, args)
print(reid_normed.detach().numpy().tolist())
