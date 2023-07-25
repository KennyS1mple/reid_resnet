# CACIOUS CODING
# Data     : 7/24/23  5:36 PM
# File name: data_util
# Desc     : util 4 dataset and loader
from dataset.dataset import MyDataset
from torch.utils.data import DataLoader


def get_dataloader(args):
    dataset = MyDataset(args)
    return DataLoader(dataset, args.batch_size, True, num_workers=4)
