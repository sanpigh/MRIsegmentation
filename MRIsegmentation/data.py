import cv2
import glob
import logging
import pandas as pd
import numpy as np
import os
import random

from google.cloud import storage
import tensorflow as tf

from MRIsegmentation.params import GDRIVE_DATA_PATH, BUCKET_NAME, BUCKET_DATA_PATH


def get_data(nrows=1_000):
    """returns a DataFrame with nrows from s3 bucket"""
    assert nrows <= 1_000
    blob_list = list_blobs(f"gs://{BUCKET_NAME}/{BUCKET_DATA_PATH}")

    # df = pd.read_csv(f"gs://{BUCKET_NAME}/{BUCKET_DATA_PATH}", nrows=nrows)
    print(blob_list)

    df = []

    return df


def get_data_from_drive():
    return pd.read_csv(GDRIVE_DATA_PATH + "/kaggle_3m/brain_df.csv")


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

    img_paths = df["image_path"].values
    msk_paths = df["mask_path"].values

    full_size = df.shape[0]

    test_size = int((1 - train_ratio) * 0.5)
    val_size = test_size
    train_size = full_size - val_size - test_size

    ds = tf.data.Dataset.from_tensor_slices((img_paths, msk_paths))
    ds = ds.shuffle(df.shape[0], seed=42)
    ds_train = ds.keep(train_size)
    ds_test = ds.skip(train_size)
    ds_val = ds_test.skip(val_size)
    ds_test = ds_test.take(test_size)

    return ds_train, ds_val, ds_test
