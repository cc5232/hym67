Code for the Efficient Image Super-Resolution Reconstruction Network Based on Feature Modulation and Variance Attention
This code is built based on BasicSR. Before conducting training and testing operations, please ensure that the installation of the work and the preparation of the dataset have been completed correctly.
To keep the working environment tidy and simple, only three files, namely test.py, train.py and your_arch.py, are required here.

environment：

Python >= 3.8.0

Pyotch >= 1.8.1

torchvision >=0.16.1

basicsr = 1.4.2

dataset:

train_Data:

DIV2K(800 images)

Flicker2K(2650 images)

test_data:

Set5, Set14, BSDS100, Urban100, Manga109

All datasets could be found in https://paperswithcode.com/datasets.

More preparation for training datasets:

See https://github.com/XPixelGroup/BasicSR/tree/master/basicsr/data for more details

Training and testing:

For training:

you can run the testing demo with

CUDA_VISIBLE_DEVICES=0 python code/train.py -opt options/train/FMVAN_X2.yml

For testing:

you can run the testing demo with

CUDA_VISIBLE_DEVICES=0 python code/test.py -opt options/test/FMVAN_X2.yml
