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
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    logger.info(f"input_data: {input_data}")

    bucket = input_data.split("/")[2]
    bucket_and_key = input_data.replace("s3://", "", 1)
    bucket_name, key = bucket_and_key.split('/', 1)
    logger.info("Downloading data from bucket: %s, key: %s, bucket_name: %s", bucket, key, bucket_name)
    fn = f"{base_dir}/data/train-dataset.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)

    logger.debug("Reading downloaded data.")
    df = pd.read_csv(fn)
    os.unlink(fn)

    logger.debug("Defining transformers.")
    logger.info("Applying transforms.")

    customer_ids = df['customer_key']
    campaign_ids = df['campaign_id']

    df = df.drop(['customer_key', 'campaign_id'], axis=1)
    y = df.pop("target")
    logger.info(f"df columns {df.columns}")
    print(f"df columns {df.columns}")

    # Convert DataFrame to numpy array
    X_pre = df.to_numpy()
    y_pre = y.to_numpy()

    # Splitting the data into train, validation, and test sets
    logger.info("Splitting %d rows of data into train, validation, test datasets.", len(X_pre))
    # First split: 90% for training and 10% for the temporary set (which will be split into validation and test)
    X_train, X_temp, y_train, y_temp = train_test_split(X_pre, y_pre, test_size=0.2, stratify=y_pre, random_state=42)

    # Second split: Split the temporary set into validation and test sets (50% each of the 20% temp set, thus 10% each of the original dataset)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # Concatenate the features and labels for each dataset
    train = np.concatenate((y_train.reshape(-1, 1), X_train), axis=1)
    validation = np.concatenate((y_val.reshape(-1, 1), X_val), axis=1)
    test = np.concatenate((y_test.reshape(-1, 1), X_test), axis=1)

    logger.info("Writing out datasets to %s.", base_dir)
    pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(f"{base_dir}/validation/validation.csv", header=False, index=False)
    pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)

    # Extract column names
    column_names = df.columns.tolist()

    # Convert to JSON
    json_data = json.dumps(column_names)

    # Directory path
    directory = f"{base_dir}/feature"

    # Create the directory if it does not exist
    os.makedirs(directory, exist_ok=True)

    # Write to a JSON file
    with open(f"{base_dir}/feature/feature_names.json", 'w') as file:
        file.write(json_data)

        # Create a mapping for the test set
    _, test_idx, _, _ = train_test_split(range(len(X_pre)), range(len(y_pre)), test_size=0.2, stratify=y_pre, random_state=42)
    _, test_idx_final, _, _ = train_test_split(test_idx, test_idx, test_size=0.5, stratify=y_temp, random_state=42)
    test_customer_campaign_mapping = pd.DataFrame({'customer_key': customer_ids[test_idx_final], 'campaign_id': campaign_ids[test_idx_final]})

    # You can now save this mapping or use it as needed
    # For example, to save:
    directory = f"{base_dir}/data"

    os.makedirs(directory, exist_ok=True)

    test_customer_campaign_mapping.to_csv(f"{base_dir}/data/test_customer_campaign_mapping.csv", index=False)
