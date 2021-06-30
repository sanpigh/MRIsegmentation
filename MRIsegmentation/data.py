import cv2
import glob
import logging
import pandas as pd
import numpy as np
import os
import random

from sklearn.model_selection import train_test_split
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

    df_mask = df[df['mask'] == 1]

    df_train, df_val = train_test_split(df_mask, train_size=train_ratio)
    df_test, df_val = train_test_split(df_val, test_size=0.5)

    ds_train = tf.data.Dataset.from_tensor_slices(
        (df_train["image_path"].values, df_train["mask_path"].values))
    ds_val = tf.data.Dataset.from_tensor_slices(
        (df_val["image_path"].values, df_val["mask_path"].values))
    ds_test = tf.data.Dataset.from_tensor_slices(
        (df_test["image_path"].values, df_test["mask_path"].values))

    return ds_train, ds_val, ds_test
