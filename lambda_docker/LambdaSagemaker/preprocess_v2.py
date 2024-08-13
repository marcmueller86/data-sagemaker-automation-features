"""Feature engineers the abalone dataset."""
import argparse
import logging
import os
import pathlib
import requests
import tempfile
import json
import boto3
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

import sys
import subprocess

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

label_column = "target"
label_column_dtype = {"target": int}

def merge_two_dicts(x, y):
    """Merges two dicts, returning a new copy."""
    z = x.copy()
    z.update(y)
    return z

if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--test-data", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    train_data = args.train_data
    test_data = args.test_data
    logger.info(f"train_data: {train_data} train_data: {test_data}")

    bucket_train = train_data.split("/")[2]
    bucket_and_key_train = train_data.replace("s3://", "", 1)
    bucket_name_train, key_train = bucket_and_key_train.split('/', 1)
    logger.info("Downloading data from bucket: %s, key: %s, bucket_name: %s", bucket_train, key_train, bucket_name_train)
    fn_train = f"{base_dir}/data/train-dataset.csv"


    bucket_test = test_data.split("/")[2]
    bucket_and_key_test = test_data.replace("s3://", "", 1)
    bucket_name_test, key_test = bucket_and_key_test.split('/', 1)
    logger.info("Downloading data from bucket: %s, key: %s, bucket_name: %s", bucket_test, key_test, bucket_name_test)
    fn_test = f"{base_dir}/data/test-dataset.csv"


    
    s3 = boto3.resource("s3")
    s3.Bucket(bucket_test).download_file(key_train, fn_train)
    s3.Bucket(bucket_test).download_file(key_test, fn_test)
    logger.debug("Reading downloaded data.")
    df_train = pd.read_csv(fn_train)
    df_test = pd.read_csv(fn_test)


    os.unlink(fn_train)
    os.unlink(fn_test)



    logger.debug("Defining transformers.")
    logger.info("Applying transforms.")

    customer_ids = df_test['identifier']

    df_train = df_train.drop(['identifier'], axis=1)
    df_test = df_test.drop(['identifier'], axis=1)
    y_train = df_train.pop("target")
    y_test = df_test.pop("target")
    
    logger.info(f"df columns {df_train.columns}")
    logger.info(f"df columns {df_test.columns}")
    print(f"df columns {df_train.columns}")
    print(f"df columns {df_test.columns}")

    # Convert DataFrame to numpy array
    X_train = df_train.to_numpy()
    X_pre_test = df_test.to_numpy()
    y_train = y_train.to_numpy()
    y_pre_test = y_test.to_numpy()

    # Splitting the data into train, validation, and test sets
    logger.info("Splitting %d rows of data into train, validation, test datasets.", len(X_pre))
    # First split: 90% for training and 10% for the temporary set (which will be split into validation and test)
    # X_train, X_temp, y_train, y_temp = train_test_split(X_pre, y_pre, test_size=0.2, stratify=y_pre, random_state=42)

    # Second split: Split the temporary set into validation and test sets (50% each of the 20% temp set, thus 10% each of the original dataset)
    X_val, X_test, y_val, y_test = train_test_split(X_pre_test, y_pre_test, test_size=0.5, stratify=y_pre_test, random_state=42)

    # Concatenate the features and labels for each dataset
    train = np.concatenate((y_train.reshape(-1, 1), X_train), axis=1)
    validation = np.concatenate((y_val.reshape(-1, 1), X_val), axis=1)
    test = np.concatenate((y_test.reshape(-1, 1), X_test), axis=1)

    logger.info("Writing out datasets to %s.", base_dir)
    pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(f"{base_dir}/validation/validation.csv", header=False, index=False)
    pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)

    # Extract column names
    column_names = df_train.columns.tolist()

    # Convert to JSON
    json_data = json.dumps(column_names)

    # Directory path
    directory = f"{base_dir}/feature"

    # Create the directory if it does not exist
    os.makedirs(directory, exist_ok=True)


    # Create a mapping for the test set
    _, test_idx, _, _ = train_test_split(range(len(X_pre_test)), range(len(y_pre_test)), test_size=0.5, stratify=y_pre_test, random_state=42)

    test_customer_campaign_mapping = pd.DataFrame({'identifier': customer_ids[test_idx]})

    # You can now save this mapping or use it as needed
    # For example, to save:
    test_customer_campaign_mapping.to_csv(f"{base_dir}/data/test_customer_campaign_mapping.csv", index=False)
    # Write to a JSON file
    with open(f"{base_dir}/features/feature_names.json", 'w') as file:
        file.write(json_data)
