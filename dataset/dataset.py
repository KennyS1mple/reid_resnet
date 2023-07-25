# CACIOUS CODING
# Data     : 6/1/23  7:59 PM
# File name: my_dataset
# Desc     :

from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import cv2
import random
import numpy as np

transform_pretrained = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform = transforms.ToTensor()


class MyDataset(Dataset):
    """
    注意各类别文件夹之间的文件名不可以重名！！！
    """

    def __init__(self, args):
        self.dataset_path = args.dataset_path
        self.flip_odds = args.flip_odds
        self.use_pretrained = args.use_pretrained
        self.sv_augmentation = args.sv_augmentation
        self.img_size = args.img_size
        self.category = os.listdir(self.dataset_path)
        self.imgs_path = []

        for category in self.category:
            for img_name in os.listdir(
                    os.path.join(self.dataset_path, category)):
                self.imgs_path.append(
                    os.path.join(self.dataset_path, category, img_name))

    def __getitem__(self, idx):

        img_path = self.imgs_path[idx]
        label = int(img_path.split('/')[-2])
        img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)

        """ 翻转 """
        if random.random() > self.flip_odds:
            img = cv2.flip(img, 0)

        """ 图片曝光增强 """
        if self.sv_augmentation:
            fraction = 0.30
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            S = img_hsv[:, :, 1].astype(np.float32)
            V = img_hsv[:, :, 2].astype(np.float32)

            a = (random.random() * 2 - 1) * fraction + 1
            S *= a
            if a > 1:
                np.clip(S, a_min=0, a_max=255, out=S)

            a = (random.random() * 2 - 1) * fraction + 1
            V *= a
            if a > 1:
                np.clip(V, a_min=0, a_max=255, out=V)

            img_hsv[:, :, 1] = S.astype(np.uint8)
            img_hsv[:, :, 2] = V.astype(np.uint8)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB, dst=img)

        """ 边缘填充 """
        h, w = img.shape[:2]
        ratio = min(self.img_size[0] / w, self.img_size[1] / h)
        new_h, new_w = int(h * ratio), int(w * ratio)
        img = cv2.resize(img, (new_w, new_h))

        grey_bg = np.full((self.img_size[1], self.img_size[0], 3), 255 // 2, dtype=np.uint8)
        grey_bg[int(grey_bg.shape[0] / 2 - img.shape[0] / 2):int(grey_bg.shape[0] / 2 + img.shape[0] / 2),
                int(grey_bg.shape[1] / 2 - img.shape[1] / 2):int(grey_bg.shape[1] / 2 + img.shape[1] / 2)] = img

        if self.use_pretrained:
            img = transform_pretrained(grey_bg)
        else:
            img = transform(grey_bg) * 2 - 1

        return img, label

    def __len__(self):
        return len(self.imgs_path)
