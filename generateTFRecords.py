import glob
import os
import json
import tensorflow as tf
import numpy as np
from utils import generate_heatmap, read_json, generate_jcd


def generate_record_example(in_path, writer):
    train = glob.glob(in_path + '/*.png')
    label = in_path + '/label.txt'
    with open(label) as file:
        file_content = file.read()
        label = int(file_content)
    for image in train:
        img = open(image, 'rb').read()

        feature = {
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img]))
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example_proto.SerializeToString())


def generate_record_example_with_crop(in_path, info_path, writer):
    train = glob.glob(in_path + '/*.png')
    label = in_path + '/label.txt'
    with open(label) as file:
        file_content = file.read()
        label = int(file_content)
    for image in train:
        img = open(image, 'rb').read()
        json_file = open(info_path + image[-17:-3] + 'json', 'r')
        info = json.load(json_file)
        bbox = info['hand_bbox']
        feature = {
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
            'x_min': tf.train.Feature(float_list=tf.train.FloatList(value=[bbox[0]])),
            'x_max': tf.train.Feature(float_list=tf.train.FloatList(value=[bbox[1]])),
            'y_min': tf.train.Feature(float_list=tf.train.FloatList(value=[bbox[2]])),
            'y_max': tf.train.Feature(float_list=tf.train.FloatList(value=[bbox[3]]))
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example_proto.SerializeToString())


def generate_record_example_with_kp(in_path, info_path, writer):
    train = glob.glob(in_path + '/*.png')
    label = in_path + '/label.txt'
    with open(label) as file:
        file_content = file.read()
        label = int(file_content)
    for image in train:
        img = open(image, 'rb').read()
        json_file = open(info_path + image[-17:-3] + 'json', 'r')
        info = json.load(json_file)
        kpts = read_json(info)
        feature = {
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
            'kpts': tf.train.Feature(float_list=tf.train.FloatList(value=kpts))
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example_proto.SerializeToString())


def generate_record_example_with_heatmap(in_path, info_path, writer):
    train = glob.glob(in_path + '/*.png')
    label = in_path + '/label.txt'
    with open(label) as file:
        file_content = file.read()
        label = int(file_content)
    for image in train:
        img = open(image, 'rb').read()
        json_file = open(info_path + image[-17:-3] + 'json', 'r')
        # info = json.load(json_file)
        # joints = read_json(info)
        # heatmap = generate_heatmap(joints, 20, 8.0)
        json_name = info_path + image[-17:-3] + 'json'
        feature = {
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
            'json_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[json_name.encode()]))
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example_proto.SerializeToString())


def generate_record_from_dir(dir1, sub_list, record_path):
    for sub in os.listdir(dir1):
        if sub in sub_list:
            record_name = record_path + sub + '.tfrecord'
            sub = dir1 + sub
            with tf.io.TFRecordWriter(record_name) as writer:
                hand_dir = sub + '/hand/'
                for d in os.listdir(hand_dir):
                    print(d)
                    in_dir = hand_dir + d
                    if os.path.isdir(in_dir):
                        inpath_1 = os.path.join(in_dir, 'image_cropped', 'left')
                        inpath_2 = os.path.join(in_dir, 'image_cropped', 'right')
                        if os.path.exists(inpath_1 + '/label.txt'):
                            generate_record_example(inpath_1, writer)
                        if os.path.exists(inpath_2 + '/label.txt'):
                            generate_record_example(inpath_2, writer)


def generate_record_from_dir_crop(dir1, dir2, sub_list, record_path):
    for sub in os.listdir(dir1):
        if sub in sub_list:
            print(sub)
            record_name = record_path + sub + '_crop.tfrecord'
            info_dir = dir2 + sub
            sub = dir1 + sub
            with tf.io.TFRecordWriter(record_name) as writer:
                hand_dir = sub + '/hand/'
                info_dir_hand = info_dir + '/hand/'
                for d in os.listdir(hand_dir):
                    in_dir = hand_dir + d
                    indir_2 = info_dir_hand + d
                    if os.path.isdir(in_dir):
                        inpath_1 = os.path.join(in_dir, 'image_cropped', 'left')
                        info_path_1 = os.path.join(indir_2, 'left')
                        inpath_2 = os.path.join(in_dir, 'image_cropped', 'right')
                        info_path_2 = os.path.join(indir_2, 'right')
                        if os.path.exists(inpath_1 + '/label.txt'):
                            generate_record_example_with_crop(inpath_1, info_path_1, writer)
                        if os.path.exists(inpath_2 + '/label.txt'):
                            generate_record_example_with_crop(inpath_2, info_path_2, writer)


def generate_record_from_dir_kp(dir1, dir2, sub_list, record_path):
    for sub in os.listdir(dir1):
        if sub in sub_list:
            print(sub)
            record_name = record_path + sub + '_kps.tfrecord'
            info_dir = dir2 + sub
            sub = dir1 + sub
            with tf.io.TFRecordWriter(record_name) as writer:
                hand_dir = sub + '/hand/'
                info_dir_hand = info_dir + '/hand/'
                for d in os.listdir(hand_dir):
                    in_dir = hand_dir + d
                    indir_2 = info_dir_hand + d
                    if os.path.isdir(in_dir):
                        inpath_1 = os.path.join(in_dir, 'image_cropped', 'left')
                        info_path_1 = os.path.join(indir_2, 'left')
                        inpath_2 = os.path.join(in_dir, 'image_cropped', 'right')
                        info_path_2 = os.path.join(indir_2, 'right')
                        if os.path.exists(inpath_1 + '/label.txt'):
                            # generate_record_example_with_kp(inpath_1, info_path_1, writer)
                            generate_record_example_with_heatmap(inpath_1, info_path_1, writer)
                        if os.path.exists(inpath_2 + '/label.txt'):
                            # generate_record_example_with_kp(inpath_2, info_path_2, writer)
                            generate_record_example_with_heatmap(inpath_2, info_path_2, writer)


def main():
    dir1 = 'D:\Hand-data/HUMBI/Hand_1_80_updat/'
    dir2 = 'D:\Hand-data/HUMBI/data/Hand_1_80_updat/'
    # dir1 = 'D:\Hand-data/HUMBI/Hand_81_140_updat/'
    # dir2 = 'D:\Hand-data/HUMBI/data/Hand_81_140_updat/'
    # sublist = ['subject_82', 'subject_88', 'subject_115', 'subject_118']
    sublist = ['subject_1', 'subject_2', 'subject_3', 'subject_4', 'subject_7', 'subject_9','subject_11']
    record_path = 'D:\Hand-data/HUMBI/ds_heatmap/'
    generate_record_from_dir_kp(dir1, dir2, sublist, record_path)


if __name__ == '__main__':
    main()
