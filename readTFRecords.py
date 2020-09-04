import tensorflow as tf


def augment():
    def au(image, label):
        image = tf.image.flip_left_right(image)
        return image, label

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
        x_min = tf.math.maximum(int(features['x_min'])-30, 0)
        y_min = tf.math.maximum(int(features['y_min'])-30, 0)
        x_max = tf.math.minimum(int(features['x_max'])+30, 250)
        y_max = tf.math.minimum(int(features['y_max'])+30, 250)
        image = tf.image.decode_png(image_raw, channels=3)
        image = tf.image.crop_to_bounding_box(image, y_min, x_min, y_max - y_min,
                                              x_max - x_min)
        image = tf.image.resize_with_pad(image, img_size, img_size)

        # data preprocess
        image = image / 255
        label = features["label"]

        return image, label

    return crop_read


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
    dataset = dataset.shuffle(buffer_size=3000)
    dataset = dataset.batch(batchsize)
    dataset = dataset.prefetch(buffer_size=2)
    return dataset


def readTfRecords_crop(filenames, batchsize, img_size, au_flag):
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
    dataset = dataset.map(crop_reader(img_size))
    if au_flag:
        dataset_au = dataset.map(augment())
        dataset = dataset.concatenate(dataset_au)
    dataset = dataset.shuffle(buffer_size=3000)
    dataset = dataset.batch(batchsize)
    dataset = dataset.prefetch(buffer_size=2)
    return dataset
