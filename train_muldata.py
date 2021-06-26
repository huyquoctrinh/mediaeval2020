import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from data import load_data, tf_dataset
from model import build_model
from keras.preprocessing.image import ImageDataGenerator
# from keras import backend as K
def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

if __name__ == "__main__":
    ## Dataset
    path = "Kvasir-SEG"
    (train_x, train_y), (valid_x, valid_y),(test_x,test_y) = load_data(path)

    ## Hyperparameters
    batch = 1
    lr = 1e-4
    epochs = 20

    # train_dataset = tf_dataset(train_x, train_y, batch=batch)
    # valid_dataset = tf_dataset(valid_x, valid_y, batch=batch)
    aug_train = ImageDataGenerator(rescale=1./255, rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, 
                         zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
    aug_test= ImageDataGenerator(rescale=1./255)
    model = build_model()
    # print(train_dataset)
    opt = tf.keras.optimizers.Adam(lr)
    metrics = ["acc", tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=metrics)
    # aug_train = ImageDataGenerator(rescale=1./255, rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, 
    #                      zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
    # callbacks = [
    #     ModelCheckpoint("files/model.h5"),
    #     ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
    #     CSVLogger("files/data.csv"),
    #     TensorBoard(),
    #     EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
    # ]

    train_steps = len(train_x)//batch
    valid_steps = len(valid_x)//batch

    # if len(train_x) % batch != 0:
    #     train_steps += 1
    # if len(valid_x) % batch != 0:
    #     valid_steps += 1

    model.fit_generator(aug_train.flow(train_x, train_y, batch_size =2),
        validation_data=aug_test.flow(valid_x, valid_y, batch_size=2),
        epochs=epochs,
        steps_per_epoch=len(train_x)//32,
        validation_steps=len(valid_x)//32
        )
    # callbacks=callbacks
    model.save("model_aug.h5")