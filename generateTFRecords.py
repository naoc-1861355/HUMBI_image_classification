import glob
import os

import tensorflow as tf


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


def generate_record_from_dir(dir1, sub_list, record_path):
    for sub in os.listdir(dir1):
        if sub in sub_list:
            record_name = record_path + sub + '.tfrecord'
            sub = dir1 + sub
            with tf.io.TFRecordWriter(record_name) as writer:
                hand_dir = sub + '/hand/'
                for d in os.listdir(hand_dir):
                    in_dir = hand_dir + d
                    if os.path.isdir(in_dir):
                        inpath_1 = os.path.join(in_dir, 'image_cropped', 'left')
                        inpath_2 = os.path.join(in_dir, 'image_cropped', 'right')
                        if os.path.exists(inpath_1 + '/label.txt'):
                            generate_record_example(inpath_1, writer)
                        if os.path.exists(inpath_2 + '/label.txt'):
                            generate_record_example(inpath_2, writer)


def main():
    dir1 = 'D:\Hand-data/HUMBI/Hand_1_80_updat/'
    sublist = ['subject_1', 'subject_2', 'subject_3']
    record_path = 'D:\Hand-data/HUMBI/train'
    generate_record_from_dir(dir1, sublist, record_path)


if __name__ == '__main__':
    main()
