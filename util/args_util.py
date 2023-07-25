# CACIOUS CODING
# Data     : 7/24/23  7:49 PM
# File name: args
# Desc     : get arguments
import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--reid_dim', default=128, type=int, help='reid dimension')
    parser.add_argument('--img_size', default=128, type=int, help='reid dimension')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--flip_odds', default=0.5, type=float, help='odds of flip of augmentation')
    parser.add_argument('--sv_augmentation', default=True, type=bool, help='sv augmentation')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--num_class', default=67, type=int)
    parser.add_argument('--device', default='0', help='-1 for CPU')
    parser.add_argument('--weight_path', default='', help='path to trained model weight')
    parser.add_argument('--img0_path', default='', help='path for img0 waiting for inference')
    parser.add_argument('--img1_path', default='', help='path for img1 waiting for inference')
    parser.add_argument('--show_args', default=True, type=bool, help='print args to output')
    parser.add_argument('--dataset_path',
                        default="/media/cacious/share/luggage_detect/reid_dataset/labeled/reid_dataset_0724",
                        help='path to training dataset')

    args = parser.parse_args()
    if args.show_args:
        print("Model arguments:\n" + str(args))
    args.device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu")
                                if args.device != "-1" else "cpu")
    args.img_size = (args.img_size, args.img_size)
    return args
