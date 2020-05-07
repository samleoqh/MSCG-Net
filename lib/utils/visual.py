import numpy as np
from PIL import Image
import torchvision.transforms as standard_transforms

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def get_visualize(args):
    visualize = standard_transforms.Compose([
        standard_transforms.Resize(300),
        standard_transforms.CenterCrop(300),
        standard_transforms.ToTensor()
    ])

    if args.pre_norm:
        restore = standard_transforms.Compose([
            DeNormalize(*mean_std),
            standard_transforms.ToPILImage(),
        ])
    else:
        restore = standard_transforms.Compose([
            standard_transforms.ToPILImage(),
        ])

    return visualize, restore


def setup_palette(palette):
    palette_rgb = []
    for _, color in palette.items():
        palette_rgb += color

    zero_pad = 256 * 3 - len(palette_rgb)

    for i in range(zero_pad):
        palette_rgb.append(0)

    return palette_rgb


def colorize_mask(mask, palette):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(setup_palette(palette))

    return new_mask


def convert_to_color(arr_2d, palette):
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d
