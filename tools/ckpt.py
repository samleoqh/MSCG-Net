from tools.model import *
from config.configs_kf import *

# score 0.547, no TTA
ckpt1 = {
    'net': 'MSCG-Rx50',
    'data': 'Agriculture',
    'bands': ['NIR','RGB'],
    'nodes': (32,32),
    'snapshot': '../ckpt/epoch_8_loss_0.99527_acc_0.82278_acc-cls_0.60967_'
                'mean-iu_0.48098_fwavacc_0.70248_f1_0.62839_lr_0.0000829109.pth'
}

# score 0.550 , no TTA
ckpt2 = {
    'net': 'MSCG-Rx101',
    'data': 'Agriculture',
    'bands': ['NIR','RGB'],
    'nodes': (32,32),
    'snapshot': '../ckpt/epoch_15_loss_1.03019_acc_0.83952_acc-cls_0.70245_'
                'mean-iu_0.54833_fwavacc_0.73482_f1_0.69034_lr_0.0001076031.pth'

}


ckpt3 = {
    'net': 'MSCG-Rx101',
    'data': 'Agriculture',
    'bands': ['NIR','RGB'],
    'nodes': (32,32),
    'snapshot': '../ckpt/epoch_15_loss_0.88412_acc_0.88690_acc-cls_0.78581_'
                'mean-iu_0.68205_fwavacc_0.80197_f1_0.80401_lr_0.0001075701.pth'
}

# ckpt1 + ckpt2, test score 0.599,
# ckpt1 + ckpt2 + ckpt3, test score 0.608


def get_net(ckpt=ckpt1):
    net = load_model(name=ckpt['net'],
                     classes=7,
                     node_size=ckpt['nodes'])

    net.load_state_dict(torch.load(ckpt['snapshot']))
    net.cuda()
    net.eval()
    return net


def loadtestimg(test_files):

    id_dict = test_files[IDS]
    image_files = test_files[IMG]
    # mask_files = test_files[GT]

    for key in id_dict.keys():
        for id in id_dict[key]:
            if len(image_files) > 1:
                imgs = []
                for i in range(len(image_files)):
                    filename = image_files[i].format(id)
                    path, _ = os.path.split(filename)
                    if path[-3:] == 'nir':
                        # img = imload(filename, gray=True)
                        img = np.asarray(Image.open(filename), dtype='uint8')
                        img = np.expand_dims(img, 2)

                        imgs.append(img)
                    else:
                        img = imload(filename)
                        imgs.append(img)
                image = np.concatenate(imgs, 2)
            else:
                filename = image_files[0].format(id)
                path, _ = os.path.split(filename)
                if path[-3:] == 'nir':
                    # image = imload(filename, gray=True)
                    image = np.asarray(Image.open(filename), dtype='uint8')
                    image = np.expand_dims(image, 2)
                else:
                    image = imload(filename)
            # label = np.asarray(Image.open(mask_files.format(id)), dtype='uint8')

            yield image


def loadids(test_files):
    id_dict = test_files[IDS]

    for key in id_dict.keys():
        for id in id_dict[key]:
            yield id


def loadgt(test_files):
    id_dict = test_files[IDS]
    mask_files = test_files[GT]
    for key in id_dict.keys():
        for id in id_dict[key]:
            label = np.asarray(Image.open(mask_files.format(id)), dtype='uint8')
            yield label