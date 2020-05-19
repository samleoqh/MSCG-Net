# MSCG-Net for Semantic Segmentation
## Introduce
This repository contains MSCG-Net models (MSCG-Net-50 and MSCG-Net-101) for semantic segmentation in [Agriculture-Vision Challenge and Workshop](https://github.com/SHI-Labs/Agriculture-Vision) (CVPR 2020), and the pipeline of training and testing models, implemented in PyTorch. 

## Code structure

```
├── config		# config code
├── data		# dataset loader and pre-processing code
├── tools		# train and test code, ckpt and model_load
├── lib			# model block, loss, utils code, etc
└── ckpt 		# output check point, trained weights, log files, etc

```

## Environments

- python 3.5+
- pytorch 1.4.0
- opencv 3.4+
- tensorboardx 1.9
- albumentations 0.4.0
- pretrainedmodels 0.7.4
- others (see requirements.txt)

## Dataset prepare

1. change DATASET_ROOT to your dataset path in ./data/AgricultureVision/pre_process.py
```
DATASET_ROOT = '/your/path/to/Agriculture-Vision'
```

2. keep the dataset structure as the same with the official structure shown as below
```
Agriculture-Vision
|-- train
|   |-- masks
|   |-- labels
|   |-- boundaries
|   |-- images
|   |   |-- nir
|   |   |-- rgb
|-- val
|   |-- masks
|   |-- labels
|   |-- boundaries
|   |-- images
|   |   |-- nir
|   |   |-- rgb
|-- test
|   |-- boundaries
|   |-- images
|   |   |-- nir
|   |   |-- rgb
|   |-- masks
```

## Train with a single GPU

```
CUDA_VISIBLE_DEVICES=0 python ./tools/train_R50.py  # trained weights ckpt1
# train_R101.py 								 # trained weights, ckpt2
# train_R101_k31.py 							  # trained weights, ckpt3
```

**Please note that:** we first train these models using Adam combined with Lookahead as the optimizer
for the first 10k iterations (around 7~10 epochs) and then change the optimizer to
SGD in the remaining iterations. So you will have to **manually change the code to switch the optimizer**
**to SGD** as follows:  

```
# Change line 48: Copy the file name ('----.pth') of the best ckpt trained with Adam
train_args.snapshot = '-------.pth'
...
# Comment line 92
# base_optimizer = optim.Adam(params, amsgrad=True)

# uncomment line 93
base_optimizer = optim.SGD(params, momentum=train_args.momentum, nesterov=True)
```

## Test with a single GPU

```
# To reproduce the leaderboard results (0.608), download the trained-weights ckpt1,2,3
# and save them with the original names into ./ckpt folder before run test_submission.py
CUDA_VISIBLE_DEVICES=0 python ./tools/test_submission.py
```

#### Trained weights for  3 models download (save them to ./ckpt before run test_submission)
[ckpt1](https://drive.google.com/open?id=1eVvUd4TVUtEe_aUgKamUrDdSlrIGHuH3),[ckpt2](https://drive.google.com/open?id=1vOlS4LfHGnWIUpqTFB2a07ndlpBxFmVE),[ckpt3](https://drive.google.com/open?id=1nEPjnTlcrzx0FOH__MbP3e_f9PlhjMa2)

## Results Summary

| Models                              | mIoU (%)        | Background      | Cloud shadow    | Double plant    | Planter skip    | Standing water  | Waterway        | Weed cluster    |
| ----------------------------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| MSCG-Net-50 (ckpt1)                 | 54.7            | 78.0            | 50.7            | 46.6            | 34.3            | 68.8            | 51.3            | 53.0            |
| ***MSCG-Net-101 (ckpt2)***          | ***55.0***      | ***79.8***      | ***44.8***      | ***55.0***      | ***30.5***      | ***65.4***      | ***59.2***      | ***50.6***      |
| MSCG-Net-101_k31 (ckpt3)            | 54.1            | 79.6            | 46.2            | 54.6            | 9.1             | 74.3            | 62.4            | 52.1            |
| Ensemble_TTA (ckpt1,2)              | 59.9            | 80.1            | 50.3            | 57.6            | 52.0            | 69.6            | 56.0            | 53.8            |
| <u>**Ensemble_TTA (ckpt1,2,3)**</u> | <u>**60.8**</u> | <u>**80.5**</u> | <u>**51.0**</u> | <u>**58.6**</u> | <u>**49.8**</u> | <u>**72.0**</u> | <u>**59.8**</u> | <u>**53.8**</u> |

Please note that all our single model's scores are computed with just single-scale (512x512) and single feed-forward inference without TTA. TTA denotes test time augmentation (e.g. flip and mirror). Ensemble_TTA (ckpt1,2) denotes two models (ckpt1, and ckpt2) ensemble with TTA, and (ckpt1, 2, 3) denotes three models ensemble. 

### Model Size

| Models           | Backbones           | Parameters | GFLOPs | Inference time <br />(CPU/GPU ) |
| ---------------- | ------------------- | ---------- | ------ | ------------------------------- |
| MSCG-Net-50      | Se_ResNext50_32x4d  | 9.59       | 18.21  | 522 / 26 ms                     |
| MSCG-Net-101     | Se_ResNext101_32x4d | 30.99      | 37.86  | 752 / 45 ms                     |
| MSCG-Net-101_k31 | Se_ResNext101_32x4d | 30.99      | 37.86  | 752 / 45 ms                     |

Please note that all backbones used pretrained weights on **ImageNet** that can be imported and downloaded from the [link](https://github.com/Cadene/pretrained-models.pytorch#senet). And MSCG-Net-101_k31 has exactly the same architecture wit MSCG-Net-101, while it is trained with extra 1/3 validation set (4,431) instead of just using the official training images (12,901). 

## Citation: 
Please consider citing our work if you find the code helps you

[Multi-view Self-Constructing Graph Convolutional Networks with Adaptive Class Weighting Loss for Semantic Segmentation](https://arxiv.org/pdf/2004.10327)

```
@inproceedings{liu2020CVPRW,
  title={Multi-view Self-Constructing Graph Convolutional Networks with Adaptive Class Weighting Loss for Semantic Segmentation},
  author={Qinghui Liu and Michael Kampffmeyer and Robert Jenssen and Arnt-Børre Salberg},
  booktitle={Proceedings of CVPRW 2020 on Agriculture-Vision},
  year={2020}
}
```
[Self-Constructing Graph Convolutional Networks for Semantic Labeling](https://arxiv.org/pdf/2003.06932)
```
@inproceedings{liu2020scg,
  title={Self-Constructing Graph Convolutional Networks for Semantic Labeling},
  author={Qinghui Liu and Michael Kampffmeyer and Robert Jenssen and Arnt-Børre Salberg},
  booktitle={Proceedings of IGARSS 2020 - 2020 IEEE International Geoscience and Remote Sensing Symposium},
  year={2020}
}
```
