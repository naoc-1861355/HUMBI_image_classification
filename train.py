from readTFRecords import readTfRecords, readTfRecords_crop
import tensorflow as tf
import numpy as np
from testModel import map_crop


def decay(epoch, lr):
    if epoch < 5:
        return lr
    elif epoch < 15:
        return 0.0005
    else:
        return lr * tf.math.exp(-0.1)


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


def rotate(image, label):
    image = tf.image.rot90(image)
    return image, label


def main():
    record_path = 'D:\Hand-data/HUMBI/ds/'
    ds = readTfRecords_crop([record_path + 'subject_1_crop.tfrecord', record_path + 'subject_2_crop.tfrecord',
                             record_path + 'subject_3_crop.tfrecord', record_path + 'subject_4_crop.tfrecord',
                             record_path + 'subject_7_crop.tfrecord', record_path + 'subject_82_crop.tfrecord'], 32, 224, True)
    val_ds = readTfRecords_crop([record_path + 'subject_88_crop.tfrecord', record_path + 'subject_115_crop.tfrecord',
                                 record_path + 'subject_118_crop.tfrecord'], 16, 224, False)
    # 32 batch_size without augment: 366 batches total
    model = load_model_mobilenet()
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['sparse_categorical_accuracy'])
    learning_rate_decay = tf.keras.callbacks.LearningRateScheduler(decay, verbose=1)

    model.fit(ds, epochs=35, callbacks=[learning_rate_decay,
                                        tf.keras.callbacks.TensorBoard(log_dir='logs/mobile_modify_grayscale',
                                                                       profile_batch=100000000)],
              validation_data=val_ds)
    model.evaluate(ds)
    model.evaluate(val_ds)
    model.save('mobilenet_v2_modify_subj1234782_gray.h5', save_format='h5')


if __name__ == '__main__':
    main()
