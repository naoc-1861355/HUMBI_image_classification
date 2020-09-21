from readTFRecords import readTfRecords, readTfRecords_info
import tensorflow as tf
import os
import numpy as np
from testModel import map_crop


def decay(epoch, lr):
    if epoch < 10:
        return lr
    # elif epoch < 15:
    #     return 0.0005
    else:
        return lr * tf.math.exp(-0.1)
        # return lr * 0.95


def load_model_xception():
    base_model = tf.keras.applications.Xception(input_shape=(250, 250, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    x = base_model(base_model.input, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(1000, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(
        x)  # Regularize with dropout
    outputs = tf.keras.layers.Dense(18)(x)
    model = tf.keras.Model(base_model.input, outputs)
    # model = tf.keras.models.Sequential([base_model,
    #                                     tf.keras.layers.GlobalAveragePooling2D(),
    #                                     tf.keras.layers.Dense(1000, activation='relu'),
    #                                     tf.keras.layers.Dense(18)], name='Xception_clas')
    return model


def load_model_mobilenet():
    input = tf.keras.Input(shape=(224, 224, 3))
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = True
    x = base_model(input)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(18)(x)
    model = tf.keras.Model(input, outputs)
    return model


def load_model_mobilenet_kps():
    img_input = tf.keras.Input(shape=(224, 224, 3), name='img_input')
    kp_input = tf.keras.Input(shape=(210,), name='kp_input')
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = True
    kp_out = tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(1e-6))(kp_input)
    kp_out = tf.keras.layers.BatchNormalization()(kp_out)
    kp_out = tf.keras.layers.ReLU()(kp_out)
    kp_out = tf.keras.layers.Dropout(0.3)(kp_out)
    kp_out = tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(1e-6))(kp_out)
    kp_out = tf.keras.layers.BatchNormalization()(kp_out)
    kp_out = tf.keras.layers.ReLU()(kp_out)
    kp_out = tf.keras.layers.Dropout(0.3)(kp_out)
    kp_out = tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(1e-6))(kp_out)
    kp_out = tf.keras.layers.BatchNormalization()(kp_out)
    kp_out = tf.keras.layers.ReLU()(kp_out)
    img_out = base_model(img_input)
    img_out = tf.keras.layers.GlobalAveragePooling2D()(img_out)
    img_out = tf.keras.layers.Dropout(0.3)(img_out)
    img_out = tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(1e-6))(img_out)
    img_out = tf.keras.layers.BatchNormalization()(img_out)
    img_out = tf.keras.layers.ReLU()(img_out)

    output = tf.keras.layers.concatenate([img_out, kp_out])
    output = tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l2(1e-6))(output)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.ReLU()(output)
    output = tf.keras.layers.Dropout(0.3)(output)
    output = tf.keras.layers.Dense(18)(output)

    model = tf.keras.Model(inputs=[img_input, kp_input], outputs=output)
    return model

def load_model_kps():
    kp_input = tf.keras.Input(shape=(210,), name='kp_input')
    kp_out = tf.keras.layers.Dense(256, activation='relu')(kp_input)
    kp_out = tf.keras.layers.Dropout(0.5)(kp_out)
    kp_out = tf.keras.layers.Dense(256, activation='relu')(kp_out)
    kp_out = tf.keras.layers.Dropout(0.5)(kp_out)
    kp_out = tf.keras.layers.Dense(100, activation='relu')(kp_out)
    kp_out = tf.keras.layers.Dropout(0.5)(kp_out)
    output = tf.keras.layers.Dense(18)(kp_out)

    model = tf.keras.Model(inputs=kp_input, outputs=output)
    return model

def testmodel():
    record_path = 'D:\Hand-data/HUMBI/ds/'
    ds = readTfRecords_info([record_path + 'subject_1_kps.tfrecord'], 32, 224, 'kps', True)
    ds = ds.map(lambda image, kps, label: ((image, kps), label))



def rotate(image, label):
    image = tf.image.rot90(image)
    return image, label


def main():
    record_path = 'D:\Hand-data/HUMBI/ds_norm/'
    ds = readTfRecords_info([record_path + 'subject_1_kps.tfrecord', record_path + 'subject_2_kps.tfrecord',
                             record_path + 'subject_3_kps.tfrecord', record_path + 'subject_4_kps.tfrecord',
                             record_path + 'subject_7_kps.tfrecord', record_path + 'subject_82_kps.tfrecord',
                             record_path + 'subject_9_kps.tfrecord', record_path + 'subject_11_kps.tfrecord'], 32, 224,
                            'kps', True)
    val_ds = readTfRecords_info([record_path + 'subject_88_kps.tfrecord', record_path + 'subject_115_kps.tfrecord',
                                 record_path + 'subject_118_kps.tfrecord'], 16, 224, 'kps', False)
    ds = ds.map(lambda image, kps, label: ((image, kps), label))
    val_ds = val_ds.map(lambda image, kps, label: ((image, kps), label))
    # 32 batch_size without augment: 366 batches total

    model = load_model_mobilenet_kps()
    # model = load_model_kps()
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['sparse_categorical_accuracy'])
    learning_rate_decay = tf.keras.callbacks.LearningRateScheduler(decay, verbose=1)
    os.makedirs('logs/img_kps_dist_norm_2', exist_ok=True)
    checkpoint_filepath = 'tmp/checkpoint/'
    os.makedirs(checkpoint_filepath, exist_ok=True)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath+'model.hdf5',
        save_weights_only=False,
        monitor='val_acc',
        save_best_only=True)
    model.fit(ds, epochs=35, callbacks=[learning_rate_decay,
                                        tf.keras.callbacks.TensorBoard(log_dir='logs/img_kps_dist_norm_2',
                                                                       profile_batch=100000000),
                                        model_checkpoint_callback],
              validation_data=val_ds)
    model.evaluate(ds)
    model.evaluate(val_ds)
    model.save('img_kps_dist_norm_2.h5', save_format='h5')


if __name__ == '__main__':
    main()
