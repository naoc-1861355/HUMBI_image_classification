import tensorflow as tf
import glob
import cv2
import numpy as np
import os
import json
from generateTFRecords import read_json


def predict_single_dir(path, path2, model, mode):
    if mode == 'normal':
        print(path)
        img_list = glob.glob(path + '/*.png')
        with open(path + '/label.txt') as file:
            file_content = file.read()
            label = int(file_content)
        input_list = []
        for img in img_list:
            im = cv2.imread(img)
            im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im_rgb = im_rgb.astype(float)
            im_rgb = tf.image.resize(im_rgb, [224, 224])
            im_rgb = im_rgb / 255
            input_list.append(im_rgb)
        input_img = np.array(input_list)
        result = model.predict(input_img)
    elif mode == 'kps_cord':
        print(path2)
        img_list = glob.glob(path2 + '/*.jpg')
        with open(path + '/label.txt') as file:
            file_content = file.read()
            label = int(file_content)
        input_list = []
        info_list = []
        for img in img_list:
            im = cv2.imread(img)
            im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im_rgb = im_rgb.astype(float)
            im_rgb = tf.image.resize(im_rgb, [224, 224])
            im_rgb = im_rgb / 255
            input_list.append(im_rgb)
            with open(img[:-3] + 'json', 'r') as json_file:
                info = json.load(json_file)
            kpts = read_json(info)
            mean = np.mean(kpts, axis=0)
            diff = mean - 125
            kpts = kpts - diff
            kpts = tf.reshape(kpts, [42, ])
            info_list.append(kpts)
        input_img = np.array(input_list)
        input_kps = np.array(info_list)
        result = model.predict((input_img, input_kps))
    else:  # mode = kps
        print(path2)
        img_list = glob.glob(path2 + '/*.jpg')
        with open(path + '/label.txt') as file:
            file_content = file.read()
            label = int(file_content)
        input_list = []
        info_list = []
        for img in img_list:
            im = cv2.imread(img)
            # cv2.imshow('1',im)
            # cv2.waitKey(50)
            im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im_rgb = im_rgb.astype(float)
            im_rgb = tf.image.resize(im_rgb, [224, 224])
            im_rgb = im_rgb / 255
            input_list.append(im_rgb)
            with open(img[:-3] + 'json', 'r') as json_file:
                info = json.load(json_file)
            dist_tri = read_json(info)
            # kpts = kpts.reshape((21, 2))
            # dist_tri = np.zeros(210)
            # for i in range(1, 21):
            #     for j in range(i):
            #         dist_tri[(i * (i - 1)) // 2 + j] = np.linalg.norm(kpts[i, :] - kpts[j, :])
            dist_tri = dist_tri / 100
            info_list.append(dist_tri)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        input_img = np.array(input_list)
        input_kps = np.array(info_list)
        result = model.predict((input_img, input_kps))
    result = np.argmax(result, axis=1)
    result_stat = np.zeros(18)
    for i in result:
        result_stat[i] += 1
    label_hat = np.argmax(result_stat)
    print(result_stat)
    print(label_hat)
    return label_hat, label


def predictor(in_dir, out_dir, model, sub_list, mode='normal'):
    """
    predictor is a function

    Args:
        :param sub_list: list of subjects
        :param model: model path
        :param mode: the mode of provided model. One of 'kps', 'normal'
        :param out_dir: the output dir contains label.txt (/)
        :param in_dir: the input dir contains both jpg and json files(/data/)

    Returns:
        None
    """
    if mode != 'normal' and mode != 'kps' and mode != 'kps_cord':
        raise NameError('mode not found: ' + mode)
    total = 0
    right = 0
    wrong_list = []
    model = tf.keras.models.load_model(model)
    for sub in os.listdir(in_dir):
        if sub in sub_list:
            out_dir_1 = os.path.join(out_dir, sub)
            sub_1 = os.path.join(in_dir, sub)
            if os.path.isdir(sub_1):
                hand_dir = os.path.join(sub_1, 'hand')
                hand_dir_out = os.path.join(out_dir_1, 'hand')
                for frame in os.listdir(hand_dir):
                    in_dir_2 = os.path.join(hand_dir, frame)
                    out_dir_2 = os.path.join(hand_dir_out, frame)
                    if os.path.isdir(in_dir_2):
                        outpath_l = os.path.join(out_dir_2, 'image_cropped', 'left')
                        inpath_l = os.path.join(in_dir_2, 'left')
                        outpath_r = os.path.join(out_dir_2, 'image_cropped', 'right')
                        inpath_r = os.path.join(in_dir_2, 'right')
                        if os.path.exists(outpath_l + '/label.txt'):
                            pred, label = predict_single_dir(outpath_l, inpath_l, model, mode)
                            if pred != label:
                                wrong_list.append({'name': outpath_l, 'pred': int(pred), 'label': int(label)})
                            else:
                                right += 1
                            total += 1
                        if os.path.exists(outpath_r + '/label.txt'):
                            pred, label = predict_single_dir(outpath_r, inpath_r, model, mode)
                            if pred != label:
                                wrong_list.append({'name': outpath_r, 'pred': int(pred), 'label': int(label)})
                            else:
                                right += 1
                            total += 1
    print('total prediction: ' + str(total))
    print('right prediction: ' + str(right))
    print(right / total)
    print(wrong_list)
    save = False
    if save:
        with open('error_info/img_kps_dist_labelsmooth0.1.json', 'w') as file:
            json.dump(wrong_list, file)


def load_info():
    with open('error_info/info_dist.json', 'r') as file:
        info_list = json.load(file)
    model = tf.keras.models.load_model('model/img_kps_dist.h5')
    for info in info_list:
        in_dir1 = info['name']
        in_dir2 = info['name'][0:19]+'data/'+info['name'][19:-19]+info['name'][-5:]
        predict_single_dir(in_dir1, in_dir2, model, 'kps')
        print(info['pred'])
        print(info['label'])


def main():
    in_dir = 'D:\Hand-data/HUMBI/data/Hand_81_140_updat'
    out_dir = 'D:\Hand-data/HUMBI/Hand_81_140_updat'
    subject_list = ['subject_88', 'subject_115', 'subject_118']
    # subject_list = ['subject_1', 'subject_2', 'subject_3', 'subject_4', 'subject_7']
    predictor(in_dir, out_dir, 'img_kps_dist_moredata_3.h5', subject_list, 'kps')
    # load_info()


if __name__ == '__main__':
    main()
