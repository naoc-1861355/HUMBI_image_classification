import tensorflow as tf
from readTFRecords import readTfRecords, readTfRecords_info
import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (20, 20)


def fine_tune(model):
    xception = model.get_layer('xception')
    xception.trainable = True
    model.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # Low learning rate
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['sparse_categorical_accuracy'])
    model.summary()


def map_test_dataset(ds):
    def split_img(image, label):
        return image

    def split_lb(image, label):
        return label

    return ds.map(split_img), ds.map(split_lb)


def map_crop(ds):
    def crop_img(image, label):
        image = image[:, 30:220, 30:220, :]
        return image, label

    return ds.map(crop_img)


def dist_tri(ds):
    def triangle(image, kps, label):
        kpts = np.zeros((32,231))
        for i in range(21):
            for j in range(i + 1):
                a = tf.norm(kps[:, i, :] - kps[:, j, :])
                print(a)
        print(kpts)
        return kps

    return ds.map(triangle)


def rotate(image, label):
    image = tf.image.rot90(image)
    return image, label


def viz_ds(ds):
    for img, label in ds:
        img = img.numpy()
        label = label.numpy()
        print(img.shape)
        print(label.shape)
        one = img[2, :] * 255
        print(label)
        cv2.imwrite('1.jpg', one)
        print(one)
        break


def viz_predict(test_ds, model):
    num = 0
    for img, label in test_ds:
        if num > 10:
            break
        label = label.numpy()
        result = model.predict(img)
        result = np.argmax(result, axis=1)
        img = img.numpy()
        img = img * 255
        for i in range(16):
            if label[i] != result[i]:
                cv2.putText(img[i, :], "p" + str(result[i]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(img[i, :], str(label[i]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.imwrite('pic/%02d_%02d.jpg' % (num, i), img[i, :])
        num += 1


def viz_predict_whole(test_ds, model):
    for img, label in test_ds:
        label = label.numpy()
        result = model.predict(img)
        result = np.argmax(result, axis=1)
        img = img.numpy()
        img = img * 255
        plt.clf()
        fig = plt.figure()
        plot_dict = {'ax%02d' % i: fig.add_subplot(4, 4, i + 1) for i in range(16)}
        for i in range(16):
            if label[i] != result[i]:
                plot_dict['ax%02d' % (i + 1)].imshow(img[i, :])
                plot_dict['ax%02d' % (i + 1)].text(20, 20, "l:" + str(label[i]))
                plot_dict['ax%02d' % (i + 1)].text(20, 40, "p:" + str(result[i]))
        plt.show()
        plt.waitforbuttonpress()
        plt.close(fig)
        break


def stat_predict(ds, model):
    stat = np.zeros((18, 18))
    for img, label in ds:
        label = label.numpy()
        result = model.predict(img)
        result = np.argmax(result, axis=1)
        for i in range(label.shape[0]):
            stat[label[i], result[i]] += 1
    return stat

def stat_predict_kps(ds, model):
    stat = np.zeros((18, 18))
    for img, kps, label in ds:
        label = label.numpy()
        result = model.predict((img,kps/100))
        result = np.argmax(result, axis=1)
        for i in range(label.shape[0]):
            stat[label[i], result[i]] += 1
    return stat

def viz_pred_kps(ds_kp, model):
    for img, kps, label in ds_kp:
        label = label.numpy()
        kps = kps.numpy()
        print(label)
        result = model.predict((img, kps))
        result = np.argmax(result, axis=1)
        img = img.numpy()
        img = img * 255
        for i in range(32):
            cv2.putText(img[i, :], "p" + str(result[i]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(img[i, :], str(label[i]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imwrite('%02d.jpg' % i, img[i, :])
        print(result)
        break


def main():
    model = tf.keras.models.load_model('img_kps_dist.h5')

    record_path = 'D:\Hand-data/HUMBI/ds/'

    ds_kp = readTfRecords_info([record_path + 'subject_1_kps.tfrecord', record_path + 'subject_2_kps.tfrecord',
                                record_path + 'subject_3_kps.tfrecord', record_path + 'subject_4_kps.tfrecord',
                                record_path + 'subject_7_kps.tfrecord', record_path + 'subject_82_kps.tfrecord'], 32,
                               224, 'kps', True)
    val_ds_kp = readTfRecords_info([record_path + 'subject_88_kps.tfrecord', record_path + 'subject_115_kps.tfrecord',
                                    record_path + 'subject_118_kps.tfrecord'], 16, 224, 'kps', False)

    stat_mat = stat_predict(val_ds_kp, model)
    print(stat_mat)
    num = np.sum(stat_mat, axis=1)
    confusion_matrix = stat_mat/num
    print(confusion_matrix)
    # viz_predict(val_ds,model)
    # print(stat_mat)
    # print(np.sum(stat_mat, axis=1))
    # mat = np.zeros((18))
    # for i in range(18):
    #     mat[i] = stat_mat[i,i]/(np.sum(stat_mat,axis=1)[i]+1)
    # print(mat)
    # viz_ds(ds2)
    #
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=['sparse_categorical_accuracy'])
    # model.fit(ds_kp, epochs=10, validation_data=val_ds_kp, callbacks=[
    #     tf.keras.callbacks.TensorBoard(log_dir='logs/img_kps_fintune',
    #                                    profile_batch=100000000)])
    # model.evaluate(ds_kp)
    # model.save('img_kps_finetune.h5', save_format='h5')


if __name__ == '__main__':
    main()
