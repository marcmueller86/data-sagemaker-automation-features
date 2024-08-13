FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.11

# Copy requirements.txt
RUN yum -y install gcc gcc-c++ make
# Install the specified packages
RUN pip install --upgrade pip
RUN pip install sagemaker

# Copy function code
COPY  lambda_function.py ./
COPY  pipeline_inference.py ./
COPY  pipeline.py ./
COPY  run_preprocess.sh ./
COPY  run_evaluate.sh ./
COPY  run.sh ./
COPY  preprocess_v2.py ./
COPY  evaluate_v2.py ./

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD ["lambda_function.lambda_handler"]
