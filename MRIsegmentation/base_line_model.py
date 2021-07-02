from google.colab import drive
import pandas as pd
import matplotlib.pyplot as plt

from MRIsegmentation.data import get_data_from_drive
from MRIsegmentation.utils import tversky
import tensorflow as tf
from os import walk # to navigate along directories



class BaseLineModel():

    def summary(self):
      return f'Baseline model'
  
    def score(self, X, y_true):
        y_pred = self.predict(X)
#        IoU = IoU_score(y_true, y_pred)
        IoU = tversky(tf.cast(y_true, dtype=tf.float32), tf.cast(y_pred, dtype=tf.float32))
        return IoU

    def predict(self, X, red_threshold_coef=1.5, green_threshold_coef=1.7, blue_threshold_coef=1.5):
    # Segmentate tumor based on color
    # Input: tensor (number_of_images, image_height, image_width, channel). 
    # Output: calculated mask (number_of_images, image_height, image_width, channel)
 
        mask = None
        print(f'Number of images already processed:')

        for index, img in enumerate(X):
            if index%30 == 0:
                print('\n')
            print(f'{index+1}  ', end='')
            red_threshold   = tf.math.reduce_mean(img[:, :, 0]).numpy() + red_threshold_coef   * tf.math.reduce_std(img[:, :, 0].numpy())
            green_threshold = tf.math.reduce_mean(img[:, :, 1]).numpy() + green_threshold_coef * tf.math.reduce_std(img[:, :, 1].numpy())
            blue_threshold  = tf.math.reduce_mean(img[:, :, 2]).numpy() + blue_threshold_coef  * tf.math.reduce_std(img[:, :, 2].numpy())

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
    drive.mount('/content/drive')
    
    # get data
    GDRIVE_DATA_PATH = '/content/drive/MyDrive/MRIsegmentation/'
    path = GDRIVE_DATA_PATH + 'kaggle_3m/'
#    path = 'raw_data/kaggle_3m/'    # to run local
    df = pd.read_csv(path+'data.csv')


    #### MY DICTIONARY WITH ALL THE IMAGES
    file_dic = {}
    for root, subdirs, files in walk(path):
        if root != path: # Eliminate the list of directories at the begining of the walk
            file_dic[root] = [file for file in files if 'mask' not in file]


    # RETRIEVE DATA

    # Create X_train and y_train from images and masks
    X_train   = []
    y_train   = []
    file_name = []

    for folder, images in file_dic.items():
        print(folder)
        for image in images:
            mask = image[:-4] + '_mask.tif'
            X_train.append(plt.imread(folder+'/'+image))
            y_train.append(plt.imread(folder+'/'+mask))


    X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)

    print(tf.shape(X_train_tensor))
    print(tf.shape(y_train_tensor))
    return X_train_tensor, y_train_tensor


def base_line_model_run(X_train_tensor, y_train_tensor):
    base = BaseLineModel()
    print('Calculating the baseline')
    score = base.score(X_train_tensor, y_train_tensor).numpy()
    return score