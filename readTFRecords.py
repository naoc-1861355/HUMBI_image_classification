import tensorflow as tf


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


def readTfRecords(filenames, batchsize, img_size):
    """
    build a tf-dataset of train_tfrecord as being data input
    :param num_classes:
    :param img_size:
    :param filenames: the train_tfrecord files as string list
    :param batchsize: batch
    :return: tf-dataset
    """
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(read_and_decode_builder(img_size))

    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batchsize)
    dataset = dataset.prefetch(buffer_size=2)
    return dataset
