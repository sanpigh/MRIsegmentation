import pandas as pd
import matplotlib.pyplot as plt

from MRIsegmentation.data import get_data_from_drive
from MRIsegmentation.utils import tversky
import tensorflow as tf



class BaseLineModel():
    def __init__(self, red_threshold_coef=1.5, green_threshold_coef=1.7, blue_threshold_coef=1.5):
        self.red_threshold_coef   = red_threshold_coef
        self.green_threshold_coef = green_threshold_coef
        self.blue_threshold_coef  = blue_threshold_coef

    def summary(self):
      return f'Baseline model'
  
    def score(self, X, y_true):
        y_pred = self.predict(X)
        IoU = tversky(tf.cast(y_true, dtype=tf.float32), tf.cast(y_pred, dtype=tf.float32))
        return IoU

    def predict(self, X):
    # Segmentate tumor based on color
    # Input: tensor (number_of_images, image_height, image_width, channel). 
    # Output: calculated mask (number_of_images, image_height, image_width, channel)
 
        mask = None
        print(f'Number of images processed:')
        for index, img in enumerate(X):
            if index%1000 == 0:
                print(f'{index} ', end='')
            red_threshold   = tf.math.reduce_mean(img[:, :, 0]).numpy() + self.red_threshold_coef   * tf.math.reduce_std(img[:, :, 0].numpy())
            green_threshold = tf.math.reduce_mean(img[:, :, 1]).numpy() + self.green_threshold_coef * tf.math.reduce_std(img[:, :, 1].numpy())
            blue_threshold  = tf.math.reduce_mean(img[:, :, 2]).numpy() + self.blue_threshold_coef  * tf.math.reduce_std(img[:, :, 2].numpy())

            red_pass   = tf.cast(img[:,:,0] < red_threshold, tf.int32)
            green_pass = tf.cast(img[:,:,1] > green_threshold, tf.int32)
            blue_pass  = tf.cast(img[:,:,2] < blue_threshold, tf.int32)

            if mask == None:
              mask = red_pass * green_pass * blue_pass
              y_pred = tf.expand_dims(mask, axis=0)

            else:
              mask = tf.expand_dims(red_pass * green_pass * blue_pass, axis=0)    
              y_pred = tf.concat([y_pred, mask], axis=0)

        return y_pred




def base_line_model_retrieve():
    # RETRIEVE DATA
    df = get_data_from_drive()
    # Create X_train and y_train from images and masks
    X = []
    y = []

    for index, row in df.iterrows():
        X.append(plt.imread(row['image_path']))
        y.append(plt.imread(row['mask_path']))

    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)

    return X_tensor, y_tensor
