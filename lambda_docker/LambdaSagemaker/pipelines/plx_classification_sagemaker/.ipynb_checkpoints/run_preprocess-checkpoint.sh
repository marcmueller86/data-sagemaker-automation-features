#!/bin/bash

cd /opt/ml/processing/input/code2/

# Exit on any error. SageMaker uses error code to mark failed job.
set -e

if [ -f 'requirements.txt' ]; then
    # Some py3 containers has typing, which may breaks pip install
    pip uninstall --yes typing

    pip install -r requirements.txt
fi

python preprocess.py "$@"