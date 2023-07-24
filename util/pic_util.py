# CACIOUS CODING
# Data     : 7/24/23  8:23 PM
# File name: pic_util
# Desc     :
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def pic2tensor(img_path, args):
    img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    h, w = img.shape[:2]
    ratio = min(args.img_size[0] / w, args.img_size[1] / h)
    new_h, new_w = int(h * ratio), int(w * ratio)
    img = cv2.resize(img, (new_w, new_h))

    grey_bg = np.full((args.img_size[1], args.img_size[0], 3), 255 // 2, dtype=np.uint8)
    grey_bg[int(grey_bg.shape[0] / 2 - img.shape[0] / 2):int(grey_bg.shape[0] / 2 + img.shape[0] / 2),
            int(grey_bg.shape[1] / 2 - img.shape[1] / 2):int(grey_bg.shape[1] / 2 + img.shape[1] / 2)] = img

    img = transform(grey_bg)
    return img.to(args.device).unsqueeze(0)
