import glob
import cv2
import numpy as np
import os


def load_data_from_path(path):
    train = glob.glob(path + '/*.png')
    label = path + '/label.txt'
    with open(label) as file:
        file_content = file.read()
        label = int(file_content)
        print(label)
    x_train = np.vstack([np.stack([cv2.imread(img), cv2.flip(cv2.imread(img), 1)]) for img in train]) / 255
    y_train = np.stack([label for _ in range(len(x_train))])
    print(x_train.shape)
    print(y_train.shape)
    return x_train, y_train


def load_data_from_dir(dir1, sub_list):
    for sub in os.listdir(dir1):
        if sub in sub_list:
            sub = dir1 + sub
            if os.path.isdir(sub):
                hand_dir = sub + '/hand/'
                for d in os.listdir(hand_dir):
                    in_dir = hand_dir + d
                    if os.path.isdir(in_dir) and int(d) == 1:
                        inpath = os.path.join(in_dir, 'image_cropped', 'left')
                        if os.path.exists(inpath + '/label.txt'):
                            print(inpath)
                            load_data_from_path(inpath)
