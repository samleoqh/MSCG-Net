import cv2
import numpy as np
import random

from lib.utils.funtions import image_enhance
from PIL import Image, ImageEnhance

from albumentations import (
    Compose,
    OneOf,
    Flip,
    PadIfNeeded,
    IAAAdditiveGaussianNoise,
    GaussNoise,
    MotionBlur,
    OpticalDistortion,
    RandomSizedCrop,
    HorizontalFlip,
    VerticalFlip,
    RandomRotate90,
    ShiftScaleRotate,
    CenterCrop,
    Transpose,
    GridDistortion,
    ElasticTransform,
    RandomGamma,
    RandomBrightnessContrast,
    RandomContrast,
    RandomBrightness,
    CLAHE,
    HueSaturationValue,
    Blur,
    MedianBlur,
    ChannelShuffle,
)


def scale(img, scale, interpolation=cv2.INTER_LINEAR):
    height, width = img.shape[:2]
    new_height, new_width = int(height * scale), int(width * scale)
    img = cv2.resize(img, (new_width, new_height), interpolation=interpolation)
    return img


def imload(filename, gray=False, scale_rate=1.0, enhance=False):
    if not gray:
        image = cv2.imread(filename)  # cv2 read color image as BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # (h, w, 3)
        if scale_rate != 1.0:
            image = scale(image, scale_rate)
        if enhance:
            image = Image.fromarray(np.asarray(image, dtype='uint8'))
            contrast = ImageEnhance.Contrast(image)
            image = contrast.enhance(1.55)
    else:
        image = cv2.imread(filename, -1)  # read gray image
        if scale_rate != 1.0:
            image = scale(image, scale_rate, interpolation=cv2.INTER_NEAREST)
        image = np.asarray(image, dtype='uint8')

    return image


def img_mask_crop(image, mask, size=(256, 256), limits=(224, 512)):
    rc = RandomSizedCrop(height=size[0], width=size[1], min_max_height=limits)
    crops = rc(image=image, mask=mask)
    return crops['image'], crops['mask']


def img_mask_pad(image, mask, target=(288, 288)):
    padding = PadIfNeeded(p=1.0, min_height=target[0], min_width=target[1])
    paded = padding(image=image, mask=mask)
    return paded['image'], paded['mask']


def composed_augmentation(image, mask):
    aug = Compose([
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        RandomRotate90(p=0.5),
        HueSaturationValue(hue_shift_limit=20,
                           sat_shift_limit=5,
                           val_shift_limit=15, p=0.5),

        OneOf([
            GridDistortion(p=0.5),
            Transpose(p=0.5)
        ], p=0.5),

        CLAHE(p=0.5)
    ])

    auged = aug(image=image, mask=mask)
    return auged['image'], auged['mask']


def get_random_pos(img, window_shape):
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2
