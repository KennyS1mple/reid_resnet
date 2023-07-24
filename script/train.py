# CACIOUS CODING
# Data     : 7/24/23  5:37 PM
# File name: train
# Desc     :
from util.args_util import get_args
from util.model_util import create_training_model
from util.data_util import get_dataloader
from torch import nn
from util.train_util import train
import torch

args = get_args()

training_model = create_training_model(args)
dataloader = get_dataloader(args)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(training_model.parameters(), args.lr)

train(training_model, dataloader, loss_func, optimizer, args)
