import glob
import cv2
import numpy as np


def load_data_from_path(path):
    train = glob.glob(path + '*.png')
    label = path + 'label.txt'
    with open(label) as file:
        file_content = file.read()
        label = int(file_content)
        print(label)
    x_train = np.vstack([np.stack([cv2.imread(img), cv2.flip(cv2.imread(img),1)]) for img in train]) / 255
    y_train = np.stack([label for _ in range(len(x_train))])
    print(x_train.shape)
    print(y_train.shape)
    return x_train, y_train
