# CACIOUS CODING
# Data     : 7/24/23  5:37 PM
# File name: inference
# Desc     :
from util.args_util import get_args
from util.model_util import create_training_model
from util.infer_util import get_match_score
from util.weight_util import load_model
import torch

args = get_args()
args.device = torch.device("cpu")

training_model = create_training_model(args)
training_model = load_model(training_model, args.weight_path)
model = training_model.model

print(get_match_score(model, args.img0_path, args.img1_path, args))
