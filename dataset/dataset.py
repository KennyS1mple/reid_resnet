# CACIOUS CODING
# Data     : 6/1/23  7:59 PM
# File name: my_dataset
# Desc     :
import math

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
        self.rotate_odds = args.rotate_odds
        self.random_affine_odds = args.random_affine_odds
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

        """ 上下翻转 """
        if random.random() < self.flip_odds:
            img = cv2.flip(img, 0)

        """ 左右翻转 """
        if random.random() < self.flip_odds:
            img = cv2.flip(img, 1)

        """ 旋转90度 """
        if random.random() < self.rotate_odds:
            if random.random() > 0.5:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        """ 随机旋转随机角度 """
        if random.random() < self.random_affine_odds:
            img = self.random_affine(img)

        """ 图片曝光增强 """
        if self.sv_augmentation:
            img = self.sv_augmentation_func(img)

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

    @staticmethod
    def random_affine(src_img, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                      border_value=(127.5, 127.5, 127.5)):

        border = 0  # width of added border (optional)
        height = src_img.shape[0]
        width = src_img.shape[1]

        # Rotation and Scale
        R = np.eye(3)
        a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
        # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
        s = random.random() * (scale[1] - scale[0]) + scale[0]
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(src_img.shape[1] / 2, src_img.shape[0] / 2), scale=s)

        # Translation
        T = np.eye(3)
        T[0, 2] = (random.random() * 2 - 1) * translate[0] * src_img.shape[0] + border  # x translation (pixels)
        T[1, 2] = (random.random() * 2 - 1) * translate[1] * src_img.shape[1] + border  # y translation (pixels)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

        M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
        imw = cv2.warpPerspective(src_img, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                                  borderValue=border_value)  # BGR order border_value

        return imw

    @staticmethod
    def sv_augmentation_func(src_img):
        fraction = 0.30
        img_hsv = cv2.cvtColor(src_img, cv2.COLOR_RGB2HSV)
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
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB, dst=src_img)

        return src_img
