from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
from sklearn.model_selection import train_test_split, KFold

import cv2

# change DATASET ROOT to your dataset path
DATASET_ROOT = '/media/liu/diskb/data/Agriculture-Vision'

TRAIN_ROOT = os.path.join(DATASET_ROOT, 'train')
VAL_ROOT = os.path.join(DATASET_ROOT, 'val')
TEST_ROOT = os.path.join(DATASET_ROOT, 'test/images')


"""
In the loaded numpy array, only 0-6 integer labels are allowed, and they represent the annotations in the following way:

0 - background
1 - cloud_shadow
2 - double_plant
3 - planter_skip
4 - standing_water
5 - waterway
6 - weed_cluster

"""
palette_land = {
    0 : (0, 0, 0),        # background
    1 : (255, 255, 0),    # cloud_shadow
    2 : (255, 0, 255),    # double_plant
    3 : (0, 255, 0),      # planter_skip
    4 : (0, 0, 255),      # standing_water
    5 : (255, 255, 255),  # waterway
    6 : (0, 255, 255),    # weed_cluster
}

# customised palette for visualization, easier for reading in paper
palette_vsl = {
    0: (0, 0, 0),     # background
    1: (0, 255, 0),     # cloud_shadow
    2: (255, 0, 0),     # double_plant
    3: (0, 200, 200),   # planter_skip
    4: (255, 255, 255), # standing_water
    5: (128, 128, 0),   # waterway
    6: (0, 0, 255)        # weed_cluster
}

labels_folder = {
    'cloud_shadow': 1,
    'double_plant': 2,
    'planter_skip': 3,
    'standing_water': 4,
    'waterway': 5,
    'weed_cluster': 6
}

land_classes = ["background", "cloud_shadow", "double_plant", "planter_skip",
                "standing_water", "waterway", "weed_cluster"]


Data_Folder = {
    'Agriculture': {
        'ROOT': DATASET_ROOT,
        'RGB': 'images/rgb/{}.jpg',
        'NIR': 'images/nir/{}.jpg',
        'SHAPE': (512, 512),
        'GT': 'gt/{}.png',
    },
}

IMG = 'images' # RGB or IRRG, rgb/nir
GT = 'gt'
IDS = 'IDs'


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def img_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])


def is_image(filename):
    return any(filename.endswith(ext) for ext in ['.png','.jpg'])


def prepare_gt(root_folder = TRAIN_ROOT, out_path='gt'):
    if not os.path.exists(os.path.join(root_folder, out_path)):
        print('----------creating groundtruth data for training./.val---------------')
        check_mkdir(os.path.join(root_folder, out_path))
        basname = [img_basename(f) for f in os.listdir(os.path.join(root_folder,'images/rgb'))]
        gt = basname[0]+'.png'
        for fname in basname:
            gtz = np.zeros((512, 512), dtype=int)
            for key in labels_folder.keys():
                gt = fname + '.png'
                mask = np.array(cv2.imread(os.path.join(root_folder, 'labels', key, gt), -1)/255, dtype=int) * labels_folder[key]
                gtz[gtz < 1] = mask[gtz < 1]

            for key in ['boundaries', 'masks']:
                mask = np.array(cv2.imread(os.path.join(root_folder, key, gt), -1) / 255, dtype=int)
                gtz[mask == 0] = 255

            cv2.imwrite(os.path.join(root_folder, out_path, gt), gtz)


def get_training_list(root_folder = TRAIN_ROOT, count_label=True):
    dict_list = {}
    basname = [img_basename(f) for f in os.listdir(os.path.join(root_folder, 'images/nir'))]
    if count_label:
        for key in labels_folder.keys():
            no_zero_files=[]
            for fname in basname:
                gt = np.array(cv2.imread(os.path.join(root_folder, 'labels', key, fname+'.png'), -1))
                if np.count_nonzero(gt):
                    no_zero_files.append(fname)
                else:
                    continue
            dict_list[key] = no_zero_files
    return dict_list, basname
        # print(len(list[key]), list[key][0:5])


def split_train_val_test_sets(data_folder=Data_Folder, name='Agriculture', bands=['NIR','RGB'], KF=3, k=1, seeds=69278):

    train_id, t_list = get_training_list(root_folder=TRAIN_ROOT, count_label=False)
    val_id, v_list = get_training_list(root_folder=VAL_ROOT, count_label=False)

    if KF >=2:
        kf = KFold(n_splits=KF, shuffle=True, random_state=seeds)
        val_ids = np.array(v_list)
        idx = list(kf.split(np.array(val_ids)))
        if k >= KF:  # k should not be out of KF range, otherwise set k = 0
            k = 0
        t2_list, v_list = list(val_ids[idx[k][0]]), list(val_ids[idx[k][1]])
    else:
        t2_list=[]

    img_folders = [os.path.join(data_folder[name]['ROOT'], 'train', data_folder[name][band]) for band in bands]
    gt_folder = os.path.join(data_folder[name]['ROOT'], 'train', data_folder[name]['GT'])

    val_folders = [os.path.join(data_folder[name]['ROOT'], 'val', data_folder[name][band]) for band in bands]
    val_gt_folder = os.path.join(data_folder[name]['ROOT'], 'val', data_folder[name]['GT'])
    #                    {}
    train_dict = {
        IDS: train_id,
        IMG: [[img_folder.format(id) for img_folder in img_folders] for id in t_list] +
             [[val_folder.format(id) for val_folder in val_folders] for id in t2_list],
        GT: [gt_folder.format(id) for id in t_list] + [val_gt_folder.format(id) for id in t2_list],
        'all_files': t_list + t2_list
    }

    val_dict = {
        IDS: val_id,
        IMG: [[val_folder.format(id) for val_folder in val_folders] for id in v_list],
        GT: [val_gt_folder.format(id) for id in v_list],
        'all_files': v_list
    }

    test_dict = {
        IDS: val_id,
        IMG: [[val_folder.format(id) for val_folder in val_folders] for id in v_list],
        GT: [val_gt_folder.format(id) for id in v_list],
    }

    print('train set -------', len(train_dict[GT]))
    print('val set ---------', len(val_dict[GT]))
    return train_dict, val_dict, test_dict


def get_real_test_list(root_folder = TEST_ROOT, data_folder=Data_Folder, name='Agriculture', bands=['RGB']):
    dict_list = {}
    basname = [img_basename(f) for f in os.listdir(os.path.join(root_folder, 'nir'))]
    dict_list['all'] = basname

    test_dict = {
        IDS: dict_list,
        IMG: [os.path.join(data_folder[name]['ROOT'], 'test', data_folder[name][band]) for band in bands],
        # GT: os.path.join(data_folder[name]['ROOT'], 'val', data_folder[name]['GT'])
        # IMG: [[img_id.format(id) for img_id in img_ids] for id in test_ids]
    }
    return test_dict
