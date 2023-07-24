# CACIOUS CODING
# Data     : 7/24/23  7:37 PM
# File name: model_util
# Desc     : util 4 create model
from model.training_model import TrainingModel
from model.resnet import resnet


def create_training_model(args):
    reid_model = resnet(args.reid_dim)
    return TrainingModel(reid_model, args.reid_dim, args.num_class)
