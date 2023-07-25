# CACIOUS CODING
# Data     : 7/24/23  8:23 PM
# File name: pic_util
# Desc     :
import numpy as np
from PIL import Image
import cv2
import os
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def pic2tensor(img_path, args):
    """
    边缘填充
    """
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


def get_obj_img(img_path, label_path, save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    img_total = []
    label_total = []
    img_file = os.listdir(img_path)
    label_file = os.listdir(label_path)

    for filename in img_file:
        name, _type = os.path.splitext(filename)
        if _type == ('.jpg' or '.png'):
            img_total.append(name)
    for filename in label_file:
        name, _type = os.path.splitext(filename)
        if _type == '.txt':
            label_total.append(name)

    for _img in img_total:
        if _img in label_total:
            filename_img = _img + '.jpg'
            path = os.path.join(img_path, filename_img)
            img = cv2.imread(path)  # 读取图片，结果为三维数组
            filename_label = _img + '.txt'
            w = img.shape[1]  # 图片宽度(像素)
            h = img.shape[0]  # 图片高度(像素)
            # 打开文件，编码格式'utf-8','r+'读写
            with open(os.path.join(label_path, filename_label), "r+", encoding='utf-8', errors="ignor") as f:
                for line in f:
                    msg = line.split(" ")  # 根据空格切割字符串，最后得到的是一个list
                    x1 = int((float(msg[1]) - float(msg[3]) / 2) * w)  # x_center - width/2
                    y1 = int((float(msg[2]) - float(msg[4]) / 2) * h)  # y_center - height/2
                    x2 = int((float(msg[1]) + float(msg[3]) / 2) * w)  # x_center + width/2
                    y2 = int((float(msg[2]) + float(msg[4]) / 2) * h)  # y_center + height/2
                    print(filename_img)
                    img_roi = img[y1:y2, x1:x2]  # 剪裁，roi:region of interest
                    cv2.imwrite(os.path.join(save_path, filename_img), img_roi)
        else:
            continue
