## Iterative Transformer Network for 3D Point Cloud
#### [[paper]](https://arxiv.org/pdf/1811.11209.pdf) [[data]](https://drive.google.com/open?id=178Ykn-_GDAbfBlKDAs-9MBwRFI53bznF)

### Introduction
This repository contains the implementation of [Iterative Transformer Network](https://arxiv.org/abs/1811.11209) (IT-Net), a network module that predicts 3D rigid transformations from partial point clouds in an iterative fashion. IT-Net can be used independently for canonical pose estimation or jointly with downstream networks for shape classification and part segmentation. Please refer to our [paper](https://arxiv.org/abs/1811.11209) for more details.

### Citation
If you find our work useful, please consider citing our paper.
```
@article{yuan2018iterative,
  title   = {Iterative Transformer Network for 3D Point Cloud},
  author  = {Yuan, Wentao and Held, David and Mertz, Christoph and Hebert, Martial},
  journal = {arXiv preprint arXiv:1811.11209},
  year    = {2018}
}
```

### Usage
#### 1) Setup
1. Install dependencies by running `pip install -r requirments.txt`.
2. Download data and pre-trained models from [Google Drive](https://drive.google.com/open?id=178Ykn-_GDAbfBlKDAs-9MBwRFI53bznF).

This code is tested on Ubuntu 16.04 with CUDA 9.0 and python 3.6.

#### 2) Object Pose Estimation
Here is an example command to get results with pretrained model.
```
python test_pose.py --transformer it_net --n_iter 10 --lmdb data/shapenet_pose/car/test.lmdb --checkpoint data/trained_models/pose_estimation/car-5_iter --results results/pose_estimation/car
```
This will generate statistics as well as visualizations shown in the paper for car pose estimation. The default setting uses 5 unrolled iterations during training and 10 during testing (controlled by the `n_iter` option). Besides the car category, we also provide data and pre-trained model for the chair category.

We provide implementations for three different 3D transformer networks, including T-Net (baseline), IT-Net with PointNet backbone (evaluated in the paper) and IT-Net with DGCNN backbone (achieves slightly better results at the cost of more computation and memory). Use the `transformer` option to switch between different architectures.

#### 3) 3D Shape Classification
Here is an example command to get results with pretrained model.
```
python test_cls.py --classifier pointnet --transformer it_net --n_iter 2 --lmdb data/partial_modelnet40/test.lmdb --checkpoint data/trained_models/classification/pointnet-2_iter --results results/classification/pointnet
```
This will evaluate the 2-iteration IT-Net trained jointly with PointNet classifier on the partial ModelNet40 dataset. The `classifier` option selects the downstream classification network (PointNet or DGCNN). The `transformer` option selects the 3D transformer model. The `n_iter` options specifies the number of unrolled iterations for the 3D transformer. We also provide pre-trained 2-iteration IT-Net with DGCNN classifier.

#### 4) Object Part Segmentation
Here is an example command to get results with pretrained model.
```
python test_seg.py --classifier pointnet --segmenter it_net --n_iter 2 --lmdb data/shapenet_part/test.lmdb --checkpoint data/trained_models/segmentation/pointnet-2_iter --results results/segmentation/pointnet
```
The options are similar to classification, except that the downstream networks are point segmentation networks (controlled via the `segmenter` option) and the dataset is ShapeNet Part. We provide two pre-trained models: one with PointNet and one with DGCNN as the segmentation backbone.

#### 5) Data Generation
The `prepare_data` folder contains scripts to generate partial point clouds from CAD datasets. Feel free to use these scripts to generate your own partial point cloud data.

### License
This project code is released under the MIT License.
