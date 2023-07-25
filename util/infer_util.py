# CACIOUS CODING
# Data     : 7/25/23  8:59 PM
# File name: infer_util
# Desc     :
from util.pic_util import pic2tensor
import torch


def get_match_score(model, img0_path, img1_path, args):
    img0 = pic2tensor(img0_path, args)
    img1 = pic2tensor(img1_path, args)
    _input = torch.cat([img0, img1], 0).to(args.device)

    model.eval()
    _output = torch.nn.functional.normalize(model(_input))
    model.train()

    img0_reid = _output[0]
    img1_reid = _output[1]

    cos_dis = torch.dot(img0_reid, img1_reid).item()

    return (cos_dis + 1) / 2
