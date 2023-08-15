# CACIOUS CODING
# Data     : 7/24/23  7:49 PM
# File name: args
# Desc     : get arguments
import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser()

    # bool参数只要添加了均为True
    parser.add_argument("--reid_dim", default=128, type=int, help="reid dimension")
    parser.add_argument("--img_size", default=128, type=int, help="image size")
    parser.add_argument("--res_depth", default=50, type=int, help="resnet depth")
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--flip_odds", default=0.5, type=float, help="odds of flip of augmentation")
    parser.add_argument("--rotate_odds", default=0.5, type=float, help="odds of rotate of augmentation")
    parser.add_argument("--sv_augmentation", default=True, type=bool, help="sv augmentation")
    parser.add_argument("--use_pretrained", default=False, type=bool, help="use weight trained on imagenet")
    parser.add_argument("--use_relu", default=False, type=bool, help="use relu in fc")
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--num_class", default=41, type=int)
    parser.add_argument("--device", default="0", help="-1 for CPU")
    parser.add_argument("--weight_path", default="", help="path to trained model weight")
    parser.add_argument("--weight_save_path", default="", help="path of saving weight")
    parser.add_argument("--inf_img_path", default="", help="kunkun inf only")
    parser.add_argument("--img0_path", default="", help="path for img0 waiting for inference")
    parser.add_argument("--img1_path", default="", help="path for img1 waiting for inference")
    parser.add_argument("--show_args", default="true", help="print args to output")
    parser.add_argument("--dataset_path",
                        default="/media/cacious/share/luggage_detect/reid_dataset/labeled/reid_dataset_0724",
                        help="path to training dataset")

    args = parser.parse_args()

    if args.show_args == "true":
        print("Model arguments:\n" + str(args))
    args.device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu")
                                if args.device != "-1" else "cpu")

    args.img_size = (args.img_size, args.img_size)
    assert args.res_depth in [18, 34, 50]

    return args
