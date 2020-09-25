# HUMBI_image_classification
This repository contains code for training and prediction for hand pose classification based on HUMBI dataset.
The work is based on tensorflow.

## HandPose labels
All labels of training and prediction labels are made using provided code.
There are total 18 categories as below.
[label](https://github.com/naoc-1861355/HUMBI_image_classification/blob/master/img/label.PNG)

## HUMBI dataset
The desired training format is `.tfrecord`, This repo provide code to build tfrecord files from scratch. <br>
If you have the corresponding files just ignore this section.<br>

To use them, use need to download HUMBi dataset first.<br>
`hand_label.py` is a script to label each frame directory
`generateTFRecords.py` is a script to construct .tfrecord files

## Usage 
`hand_label.py` is a script to label each frame directory <br>
`generateTFRecords.py` is a script to construct .tfrecord files<br>
`readTFRecords.py` provides functions to read .tfrecord files<br>
`train.py` is the main function of training<br>
`testModel.py` provides multple functions to visualize prodiction and confusion matrix<br>
`predict.py` provides files to make prediction per folder<br>

## Model
### RGB model
Base RGB model is based on MobileNet-v2. 

Best model `mobilenet_v2_modify_1234782.h5` contains in model folder, which achieve val acc of 69.22

### image(RGB) and jcd model
This model use image input and jcd input.
<img width="300" height="300" src="https://github.com/naoc-1861355/HUMBI_image_classification/blob/master/img/img_dist_model.png"/>

Best model `img_kps_dist_l2_moredata_3.h5` contains in model folder, which achieve val acc of 89.68


## Dev history
If you need more experiment details and development history, please visit this [site](http://note.youdao.com/s/3czYby59)
