from readTFRecords import readTfRecords
import numpy as np

def main():
    dir1 = 'D:\Hand-data/HUMBI/Hand_1_80_updat/'
    sublist = ['subject_1', 'subject_2', 'subject_3']
    record_path = 'D:\Hand-data/HUMBI/train_3subjects.tfrecord'
    #load_data_from_path('D:\Hand-data\HUMBI\Hand_1_80_updat\subject_1\hand/00000001\image_cropped/left/')
    #load_data_from_dir(dir1, sublist)
    ds = readTfRecords([record_path], 32, 250)
    for img, label in ds:
        img = img.numpy()
        shape = np.shape(img)
        print(shape)
        print(label)
        break


if __name__ == '__main__':
    main()
