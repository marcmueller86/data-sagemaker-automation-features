from aws_cdk import (
    Stack,
    aws_lambda as _lambda,
    aws_iam as iam,
    aws_s3 as s3,
    Duration
)
from constructs import Construct
# export AWS_ACCOUNT=812419731335
class DataSagemakerAutomationStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        sagemaker_inference_lambda = _lambda.DockerImageFunction(
            scope=self,
            id="LambdaSagemakerFeatures",
            function_name="LambdaImageSagemakerFeatures",
            # add enviroment variable : AmazonSageMaker-ExecutionRole-20240318T120107
            environment={
                "SAGEMAKER_EXECUTION_ROLE": "arn:aws:iam::812419731335:role/DataSagemakerAutomationSt-LambdaSagemakerServiceRol-hcAPvFVbvO0s"
            },
            code=_lambda.DockerImageCode.from_image_asset(
                directory="lambda_docker/LambdaSagemaker",
            ),
            timeout=Duration.seconds(120),
            memory_size=512,
            architecture=_lambda.Architecture.ARM_64,
            description="A lambda function that can start a sagemaker inference pipeline",
        )

        sagemaker_inference_lambda.add_to_role_policy(
            statement=iam.PolicyStatement(
                actions=[
                    "sagemaker:*",
                    "cloudwatch:PutMetricData",
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "iam:getRole",
                    "iam:PassRole",
                ],
                resources=["*"]
            )
        )
        # add to the role policy to allow the lambda to access the s3 bucket sagemaker-eu-central-1-812419731335
        sagemaker_inference_lambda.add_to_role_policy(
            statement=iam.PolicyStatement(
                actions=[
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:ListBucket",
                ],
                resources=[
                    "arn:aws:s3:::sagemaker-eu-central-1-812419731335",
                    "arn:aws:s3:::sagemaker-eu-central-1-812419731335/*",
                    "arn:aws:s3:::px-data-ml-ops-data-prod",
                    "arn:aws:s3:::px-data-ml-ops-data-prod/*",
                    "arn:aws:s3:::vnr-data-etl-ml-prod",
                    "arn:aws:s3:::vnr-data-etl-ml-prod/*"
                ]
            )
        )