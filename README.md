# HUMBI_image_classification
This repository contains code for training and prediction for hand pose classification based on HUMBI dataset

## HandPose labels
All labels of training and prediction labels are made using provided code.
There are total 18 categories as below.
[label](https://github.com/naoc-1861355/HUMBI_image_classification/blob/master/img/label.PNG)

## HUMBI dataset
The dired training format is `.tfrecord`, This repo provide code to build tfrecord files from scratch. <br>
If you have the corresponding files just ignore this section.<br>

To use them, use need to download HUMBi dataset first.
`hand_label` is a script to label each frame directory
`generateTFRecords` is a script to construct .tfrecord files

## Usage 
The desired the format is to use 
`hand_label`

## Model
### RGB model
Base RGB model is based on MobileNet-v2

### image(RGB) and jcd model
This model use image input and jcd input.
<img width="300" height="300" src="https://github.com/naoc-1861355/HUMBI_image_classification/blob/master/img/img_dist_model.png"/>


## Dev history
If you need more experiment details and development history, please visit [site](http://note.youdao.com/s/3czYby59)