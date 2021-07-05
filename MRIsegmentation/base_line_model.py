import pandas as pd
import matplotlib.pyplot as plt

#from MRIsegmentation.data import get_data_from_drive
from MRIsegmentation.utils import process_path
from MRIsegmentation.utils import tversky
import tensorflow as tf


from MRIsegmentation.utils import tversky
from MRIsegmentation.utils import flatten_mask, normalize


class BaseLineModel:
    def __init__(
        self, red_threshold_coef=2.0, green_threshold_coef=1.6, blue_threshold_coef=2.0
    ):
        self.red_threshold_coef = red_threshold_coef
        self.green_threshold_coef = green_threshold_coef
        self.blue_threshold_coef = blue_threshold_coef

    def summary(self):
        return f"Baseline model"

    def score(self, ds):
        y_pred, y_true = self.predict(ds)
        IoU = tversky(
            # tf.cast(y_true, dtype=tf.float32), tf.cast(y_pred, dtype=tf.float32)
            y_true, y_pred
        )
        return IoU

    def predict(self, ds): # It takes the same kind of object of holdout output
        # Segmentate tumor based on color
        # Input: tensor (number_of_images, image_height, image_width, channel).
        # Output: calculated mask (number_of_images, image_height, image_width, channel)
        count = 0
        y_pred = None
        print(f"\n Number of images processed:")
        for img, mask in ds.map(process_path).map(flatten_mask).map(normalize):  
            count += 1
            if count%500 == 0:
                print(count, end=' ')
            img  = tf.cast(img,  dtype=tf.float32)
            mask = tf.cast(mask, dtype=tf.float32)
            red_threshold = tf.math.reduce_mean(
                img[:, :, 0]
            ) + self.red_threshold_coef * tf.math.reduce_std(
                img[:, :, 0]
            )
            green_threshold = tf.math.reduce_mean(
                img[:, :, 1]
            ) + self.green_threshold_coef * tf.math.reduce_std(
                img[:, :, 1]
            )
            blue_threshold = tf.math.reduce_mean(
                img[:, :, 2]
            ) + self.blue_threshold_coef * tf.math.reduce_std(
                img[:, :, 2]
            )

            red_pass   = tf.cast(img[:, :, 0] < red_threshold,   tf.float32)
            green_pass = tf.cast(img[:, :, 1] > green_threshold, tf.float32)
            blue_pass  = tf.cast(img[:, :, 2] < blue_threshold,  tf.float32)

            if y_pred == None:
                tmp = red_pass * green_pass * blue_pass
                y_pred = tf.expand_dims(tmp, axis=0)
                y_true = tf.expand_dims(mask[:,:], axis=0)
            else:
                tmp = tf.expand_dims(red_pass * green_pass * blue_pass, axis=0)
                y_pred = tf.concat([y_pred, tmp], axis=0)
                tmp = tf.expand_dims(mask[:,:], axis=0)
                y_true = tf.concat([y_true, tmp], axis=0)

        return y_pred, y_true
