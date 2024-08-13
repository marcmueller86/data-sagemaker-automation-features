import boto3
import sagemaker
from datetime import datetime
from pipeline_inference import get_pipeline_inference
from pipeline import get_pipeline
import os
# Assuming 'base_job_prefix' and other necessary variables are set as environment variables
import os


def start_inference(use_cases, base_job_prefix, model_mapping, role, region):
    default_bucket = sagemaker.session.Session().default_bucket()

    for use_case in use_cases:
        
        pipeline_name = f"{base_job_prefix}-inference-{use_case}"
        print(pipeline_name)
        
        pipeline_inference = get_pipeline_inference(
            region=region,
            role=role,
            default_bucket=default_bucket,
            pipeline_name=pipeline_name,
            use_case=use_case,
            base_job_prefix=base_job_prefix
        )

        pipeline_inference.upsert(role_arn=role)
        latest_model_name = model_mapping[use_case]
        print("Starting USE CASE: ", use_case, " with model: ",
              latest_model_name, " at ", datetime.today().strftime('%Y-%m-%d'))

        pipeline_inference.start(parameters=dict(
            UseCase=use_case,
            BaseJobPrefix=base_job_prefix,
            ModelName=latest_model_name,
            Timestamp=datetime.today().strftime('%Y-%m-%d')
        ))
        print("Finished Started")


def start_training(use_cases, base_job_prefix, role, region):
    default_bucket = sagemaker.session.Session().default_bucket()

    for use_case in use_cases:

        pipeline_name = f"{base_job_prefix}-training-{use_case}"
        print(pipeline_name)
        pipeline = get_pipeline(
            region=region,
            role=role,
            default_bucket=default_bucket,
            pipeline_name=pipeline_name,
            use_case=use_case,
            base_job_prefix=base_job_prefix
        )
        
        pipeline.upsert(role_arn=role)

        execution = pipeline.start()



def lambda_handler(event, context):
    print(event)
    use_cases = event.get('use_cases', [])
    base_job_prefix_dict = event.get('base_job_prefix', {})
    region = boto3.Session().region_name
    # Set this in your Lambda environment variables
    role = os.environ['SAGEMAKER_EXECUTION_ROLE']
    # role = sagemaker.get_execution_role()

    base_job_prefix = base_job_prefix_dict['prefix']

    try:
        if event.get('type') == 'training':
            response = start_training(use_cases, base_job_prefix, role, region)
        elif event.get('type') == 'inference':
            model_mapping = event.get('model_mapping', {})

            response = start_inference(
                use_cases, base_job_prefix, model_mapping, role, region)
        else:
            raise ValueError("Invalid type")
    except Exception as e:
        print(e)
        return {
            'statusCode': 500,
            'body': f"Error: {e}"
        }
    return {
        'statusCode': 200,
        'body': f"Pipeline execution started for use cases: {use_cases} - {str(response)}"
    }


# Example event to test locally
if __name__ == "__main__":
    event = {
    "use_cases": [
        "SBVabo220240612",
        "MAVabo220240612"
    ],
    "base_job_prefix": {
        "prefix": "telesales"
    },
    "type": "training"
    }
    use_cases = event.get('use_cases', [])
    base_job_prefix_dict = event.get('base_job_prefix', {})
    region = boto3.Session().region_name
    # Set this in your Lambda environment variables
    role = os.environ['SAGEMAKER_EXECUTION_ROLE']
    base_job_prefix = base_job_prefix_dict['prefix']

    try:
        if event.get('type') == 'training':
            response = start_training(use_cases, base_job_prefix, role, region)
        elif event.get('type') == 'inference':
            model_mapping = event.get('model_mapping', {})

            response = start_inference(
                use_cases, base_job_prefix, model_mapping, role, region)
        else:
            raise ValueError("Invalid type")
    except Exception as e:
        print(e)
