# MSCG-Net for Semantic Segmentation
## Introduce
This repository contains MSCG-Net models (MSCG-Net-50 and MSCG-Net-101) for semantic segmentation in [Agriculture-Vision Challenge and Workshop](https://github.com/SHI-Labs/Agriculture-Vision) (CVPR 2020), and the pipeline of training and testing models, implemented in PyTorch. 

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

## Environments
- python 3.5+
- pytorch >= 1.2.0
- opencv 3.4+
- tensorboardx 1.9
- scikit-learn 
- numpy

## Code structure
```
├── config		# config code
├── data		# dataset loader and pre-processing code
├── tools		# train and test code
├── lib			# model, loss, utils code
├── submission	 # output test results for submission
└── ckpt 		 # output check point, trained weights, log files, etc

```

## Dataset prepare
1. change DATASET_ROOT to your dataset path in ./data/AgricultureVision/pre_process.py
```
DATASET_ROOT = '/your/path/to/Agriculture-Vision'
```

2. keep the dataset sturcture as the same with the official structure shown as below
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

### Train with a single GPU

```
CUDA_VISIBLE_DEVICES=0 python ./tools/train_R50.py
```

### Test  with a single GPU
```
CUDA_VISIBLE_DEVICES=0 python -u ./tools/test_submission.py
```
