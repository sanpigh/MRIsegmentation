import cv2
import glob
import logging
import pandas as pd
import numpy as np
import os
import random

from google.cloud import storage
from http.client import LENGTH_REQUIRED
import tensorflow as tf

from MRIsegmentation.params import GDRIVE_DATA_PATH, BUCKET_NAME, BUCKET_DATA_PATH


def pos_neg_diagnosis(mask_path):
    value = np.max(cv2.imread(mask_path))
    if value > 0:
        return 1
    else:
        return 0


def get_data(nrows=1_000):
    '''returns a DataFrame with nrows from s3 bucket'''
    assert nrows <= 1_000
    blob_list = list_blobs(f"gs://{BUCKET_NAME}/{BUCKET_DATA_PATH}")

    #df = pd.read_csv(f"gs://{BUCKET_NAME}/{BUCKET_DATA_PATH}", nrows=nrows)
    print(blob_list)

    df = []

    return df


def get_data_from_drive():
    data = pd.read_csv(GDRIVE_DATA_PATH + '/kaggle_3m/data.csv')

    data_map = []

    for sub_dir_path in glob.glob(GDRIVE_DATA_PATH + "kaggle_3m/*"):
        #if os.path.isdir(sub_path_dir):
        try:
            dir_name = sub_dir_path.split('/')[-1]
            for filename in os.listdir(sub_dir_path):
                image_path = sub_dir_path + '/' + filename
                data_map.extend([dir_name, image_path])
        except Exception as e:
            logging.info(e)

    df = pd.DataFrame({"patient_id": data_map[::2], "path": data_map[1::2]})

    df_imgs = df[~df['path'].str.contains("mask")]  # if have not mask

    df_masks = df[df['path'].str.contains("mask")]  # if have mask

    # File path line length images for later sorting
    BASE_LEN = len(
        "/content/drive/MyDrive/MRIsegmentation/kaggle_3m/TCGA_DU_6404_19850629/TCGA_DU_6404_19850629_"
    )  # <-!!!43.tif)
    END_IMG_LEN = 4
    END_MASK_LEN = 9

    # Data sorting
    imgs = sorted(df_imgs["path"].values,
                  key=lambda x: int(x[BASE_LEN:-END_IMG_LEN]))
    masks = sorted(df_masks["path"].values,
                   key=lambda x: int(x[BASE_LEN:-END_MASK_LEN]))

    brain_df = pd.DataFrame({
        "patient_id": df_imgs.patient_id.values,
        "image_path": imgs,
        "mask_path": masks
    })

    brain_df['mask'] = brain_df['mask_path'].apply(
        lambda x: pos_neg_diagnosis(x))

    return brain_df


def list_blobs(bucket_name):
    """Lists all the blobs in the bucket."""

    storage_client = storage.Client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name)

    lb = []
    for blob in blobs:
        lb.append(blob.name)

    return lb


def holdout(df, train_ratio=0.8):

    img_paths = df['image_path'].values
    msk_paths = df['mmask_path'].values

    full_size = df.shape[0]

    test_size = int((1 - train_ratio) * 0.5)
    val_size = test_size
    train_size = full_size - val_size - test_size

    ds = tf.data.Dataset.from_tensor_slices((img_paths, msk_paths))
    ds = ds.shuffle(seed=42)
    ds_train = ds.keep(train_size)
    ds_test = ds.skip(train_size)
    ds_val = ds_test.skip(val_size)
    ds_test = ds_test.take(test_size)

    return ds_train, ds_val, ds_test
