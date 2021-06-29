from google.cloud import storage
import pandas as pd

from MRIsegmentation.params import GDRIVE_DATA_PATH, BUCKET_NAME, BUCKET_DATA_PATH


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

    return data


def list_blobs(bucket_name):
    """Lists all the blobs in the bucket."""
    # bucket_name = "your-bucket-name"

    storage_client = storage.Client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name)

    lb = []
    for blob in blobs:
        lb.append(blob.name)

    return lb


def holdout(df, test_size=0.2):
    return None, None, None, None
