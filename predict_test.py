import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from tqdm import tqdm
from data import  tf_dataset
from train import iou
from sklearn.model_selection import train_test_split
from glob import glob
import time
def load_data(path, split=0.9):
    images = sorted(glob(os.path.join(path, "images/*")))
    masks = sorted(glob(os.path.join(path, "masks/*")))

    total_size = len(images)
    valid_size = int(split * total_size)
    test_size = int(split * total_size)

    train_x= images
    train_y = masks

    test_x = train_x
    test_y = train_y 
    return  (test_x,test_y)
def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x/255.0
    return x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = np.expand_dims(x, axis=-1)
    return x

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

if __name__ == "__main__":
    ## Dataset
    with CustomObjectScope({'iou': iou}):
        model = tf.keras.models.load_model("model_ver3_4_conv_relu.h5")
    time_taken = []
    path = "D:/Medico_automatic_polyp_segmentation_challenge_test_data"
    batch_size = 1
    (test_x, test_y) = load_data(path)

    test_dataset = tf_dataset(test_x, test_y, batch=batch_size)

    test_steps = (len(test_x)//batch_size)
    if len(test_x) % batch_size != 0:
        test_steps += 1


    with open("file1.txt","w") as fo:
        fo.write(str(iou))
    model.evaluate(test_dataset, steps=test_steps)
    start_time = time.time()
    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        x = read_image(x)
        # y = read_mask(y)
        y_pred = model.predict(np.expand_dims(x, axis=0))[0] > 0.5
        end_time = time.time() - start_time
        time_taken.append(end_time)
        h, w, _ = x.shape
        white_line = np.ones((h, 10, 3)) * 255.0

        all_images = [
             white_line,
            # mask_parse(y), white_line,
            mask_parse(y_pred) * 255.0
        ]
        image = np.concatenate(all_images, axis=1)
        cv2.imwrite("sub4/{i}.png", image)
        # x * 255.0,
    mean_time_taken = np.mean(time_taken)
    mean_fps = 1/mean_time_taken
    print("Mean FPS: ", mean_fps)