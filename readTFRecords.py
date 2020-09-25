import tensorflow as tf
import numpy as np
from utils import generate_heatmap, read_json
import json


def augment():
    def au(image, *args):
        image = tf.image.flip_left_right(image)
        result = list(args)
        result.insert(0, image)
        return tuple(result)

    return au


def read_and_decode_builder(img_size):
    def read_and_decode(filename):
        features = tf.io.parse_single_example(filename, features={
            "img_raw": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
        })
        image_raw = features["img_raw"]
        image = tf.image.decode_png(image_raw, channels=3)
        image = tf.image.resize(image, size=[img_size, img_size])

        # data preprocess
        image = image / 255
        label = features["label"]

        return image, label

    return read_and_decode


def crop_reader(img_size):
    def crop_read(filename):
        features = tf.io.parse_single_example(filename, features={
            "img_raw": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
            'x_min': tf.io.FixedLenFeature([], tf.float32),
            'x_max': tf.io.FixedLenFeature([], tf.float32),
            'y_min': tf.io.FixedLenFeature([], tf.float32),
            'y_max': tf.io.FixedLenFeature([], tf.float32)
        })
        image_raw = features["img_raw"]
        x_min = tf.math.maximum(int(features['x_min']) - 30, 0)
        y_min = tf.math.maximum(int(features['y_min']) - 30, 0)
        x_max = tf.math.minimum(int(features['x_max']) + 30, 250)
        y_max = tf.math.minimum(int(features['y_max']) + 30, 250)
        image = tf.image.decode_png(image_raw, channels=3)
        image = tf.image.crop_to_bounding_box(image, y_min, x_min, y_max - y_min,
                                              x_max - x_min)
        image = tf.image.resize_with_pad(image, img_size, img_size)

        # data preprocess
        image = image / 255
        label = features["label"]

        return image, label

    return crop_read


def kps_cord_reader(img_size):
    def kps_cord_read(filename):
        features = tf.io.parse_single_example(filename, features={
            "img_raw": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
            'kpts': tf.io.FixedLenFeature([42, ], tf.float32),
        })
        image_raw = features["img_raw"]

        image = tf.image.decode_png(image_raw, channels=3)
        image = tf.image.resize(image, size=[img_size, img_size])

        # data preprocess
        image = image / 255
        label = features["label"]
        kps = features['kpts']
        kps = tf.reshape(kps, [21, 2])
        mean = tf.math.reduce_mean(kps, 0)
        std = tf.math.reduce_std(kps, 0)
        kps = (kps - mean) / std
        kps = tf.reshape(kps, [42, ])
        return image, kps, label

    return kps_cord_read


def kps_reader(img_size):
    def kps_read(filename):
        features = tf.io.parse_single_example(filename, features={
            "img_raw": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
            'kpts': tf.io.FixedLenFeature([210, ], tf.float32),
        })
        image_raw = features["img_raw"]

        image = tf.image.decode_png(image_raw, channels=3)
        image = tf.image.resize(image, size=[img_size, img_size])

        # data preprocess
        image = image / 255
        label = features["label"]
        kps = features['kpts']
        return image, kps, label

    return kps_read


def heatmap_reader(img_size):
    def get_heatmap(json_name):
        with open(json_name, 'r') as file:
            info = json.load(file)
        joints = read_json(info)
        heatmap = generate_heatmap(joints, 20, 8.0)
        return heatmap

    def heatmap_read(filename):
        features = tf.io.parse_single_example(filename, features={
            "img_raw": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
            'json_name': tf.io.FixedLenFeature([], tf.string),
        })
        image_raw = features["img_raw"]
        image = tf.image.decode_png(image_raw, channels=3)
        image = tf.image.resize(image, size=[img_size, img_size])

        # data preprocess
        image = image / 255
        label = features["label"]
        json_name = features['json_name']
        heatmap = tf.numpy_function(get_heatmap, [json_name], tf.float32)
        heatmap.set_shape((250, 250, 21))
        return (image, heatmap), label

    return heatmap_read


def readTfRecords(filenames, batchsize, img_size, au_flag):
    """
    build a tf-dataset of train_tfrecord as being data input
    :param au_flag:
    :param num_classes:
    :param img_size:
    :param filenames: the train_tfrecord files as string list
    :param batchsize: batch
    :return: tf-dataset
    """
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(read_and_decode_builder(img_size))
    if au_flag:
        dataset_au = dataset.map(augment())
        dataset = dataset.concatenate(dataset_au)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batchsize)
    dataset = dataset.prefetch(buffer_size=2)
    return dataset


def readTfRecords_info(filenames, batchsize, img_size, info_flag, au_flag):
    """
    build a tf-dataset of train_tfrecord as being data input
    :param info_flag: 'crop' or 'kps' to indicate the additional info
    :param au_flag:
    :param img_size:
    :param filenames: the train_tfrecord files as string list
    :param batchsize: batch
    :return: tf-dataset
    """
    dataset = tf.data.TFRecordDataset(filenames)
    if info_flag == 'crop':
        dataset = dataset.map(crop_reader(img_size))
    elif info_flag == 'kps':
        dataset = dataset.map(kps_reader(img_size))
    elif info_flag == 'kps_cord':
        dataset = dataset.map(kps_cord_reader(img_size))
    elif info_flag == 'heat_map':
        dataset = dataset.map(heatmap_reader(img_size))
    if au_flag:
        if info_flag == 'heat_map':
            dataset_au = dataset.map(augment_heatmap)
            dataset = dataset.concatenate(dataset_au)
        else:
            dataset_au = dataset.map(augment())
            dataset = dataset.concatenate(dataset_au)
    dataset = dataset.shuffle(buffer_size=500)
    dataset = dataset.batch(batchsize)
    dataset = dataset.prefetch(buffer_size=2)
    return dataset


def augment_heatmap(arg1, label):
    image, heatmap = arg1
    image = tf.image.flip_left_right(image)
    heatmap = tf.image.flip_left_right(heatmap)
    return (image, heatmap), label
