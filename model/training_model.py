# CACIOUS CODING
# Data     : 7/24/23  5:40 PM
# File name: training_model
# Desc     :
from torch import nn


class TrainingModel(nn.Module):
    def __init__(self, model, reid_dim, num_class):
        super(TrainingModel, self).__init__()
        self.model = model
        self.classifier = nn.Linear(reid_dim, num_class)

    def forward(self, x):
        x = self.model(x)
        return self.classifier(x)

