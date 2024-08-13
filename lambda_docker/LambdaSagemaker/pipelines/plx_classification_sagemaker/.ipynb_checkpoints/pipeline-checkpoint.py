"""Example workflow pipeline script for abalone pipeline.

                                               . -ModelStep
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import os

import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionGreaterThan
from sagemaker.workflow.condition_step import (
    ConditionStep,
)
from sagemaker.workflow.functions import (
    JsonGet,
    Join
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    TuningStep
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.tuner import (
    ContinuousParameter,
    IntegerParameter,
    HyperparameterTuner,
    WarmStartConfig,
    WarmStartTypes,
)
from sagemaker.workflow.steps import CacheConfig

from sagemaker.workflow.lambda_step import LambdaStep
from sagemaker.lambda_helper import Lambda


cache_config = CacheConfig(enable_caching=True, expire_after="PT1H")


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
     """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
        """
     boto_session = boto3.Session(region_name=region)
     sagemaker_client = boto_session.client("sagemaker")
     return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_name=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.describe_project(ProjectName=sagemaker_project_name)
        sagemaker_project_arn = response["ProjectArn"]
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    sagemaker_project_name=None,
    role=None,
    default_bucket=None,
    # model_package_group_name="PLXClassificationPackageGroup",
    pipeline_name="PLXClassificationPipeline",
    base_job_prefix="PLXClassification",
    use_case="EXAMPLEMODEL",
    image_uri="492215442770.dkr.ecr.eu-central-1.amazonaws.com/sagemaker-xgboost:1.0-1-cpu-py3",
    processing_instance_type="ml.m5.xlarge",
    training_instance_type="ml.m5.2xlarge",
    script_code_preprocessing='s3://px-data-ml-ops-data/scripts/xgboost/preprocess.py',
    script_code_evaluation='s3://px-data-ml-ops-data/scripts/xgboost/evaluate.py',
    
):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = get_pipeline_session(region, default_bucket)

    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)

    
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )

    use_case = ParameterString(
        name="UseCase", default_value=use_case
    )

    base_job_prefix = ParameterString(
        name="BaseJobPrefix", default_value=base_job_prefix
    )

    image_uri = ParameterString(
        name="ImageUri", default_value=image_uri
    )
    
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.m5.xlarge"
    )
    
    
    # TODO this could be simplified if we add a folder for the "process file" and just name the file script.py, we would than just pass this as parameter.. this would help to just use one run.sh

    script_code_preprocessing = ParameterString(name="ScriptCodePreprocessing", default_value="s3://px-data-ml-ops-data/scripts/xgboost/preprocess.py")
    script_code_evaluation = ParameterString(name="ScriptCodeEvaluation", default_value="s3://px-data-ml-ops-data/scripts/xgboost/evaluate.py")


    sklearn_processor = SKLearnProcessor(
        role=role,
        framework_version="0.23-1",
        instance_type=processing_instance_type,
        instance_count=1,
        command=['sh'],
        base_job_name='machine-learning-plx-classification',
        sagemaker_session=pipeline_session,
    )

    step_args = sklearn_processor.run(
        inputs=[ProcessingInput(source=script_code_preprocessing, destination="/opt/ml/processing/input/code2")],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
            ProcessingOutput(output_name="feature_names", source="/opt/ml/processing/feature"),
            ProcessingOutput(output_name="test_customer_campaign_mapping", source="/opt/ml/processing/data")
        ],
        code=os.path.join(BASE_DIR, "run_preprocess.sh"),
        # you need to change the bucket !!!!
        arguments=["--input-data", Join(on='/', values=['s3://px-data-ml-ops-data', 'features', base_job_prefix, use_case, 'features.csv'])]
    )

    step_process = ProcessingStep(
        name="PreprocessClassificationData",
        step_args=step_args
    )

    # training step for generating model artifacts
    model_path = Join(on='/',values=['s3:/',sagemaker_session.default_bucket(),base_job_prefix,use_case,'train'])
    fixed_hyperparameters = {
        "scale_pos_weight": "131.25"  # Rounded value
    }


    xgb_train = Estimator(
        image_uri=image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        output_path=model_path,
        base_job_name="machine-learning-plx-classification",
        sagemaker_session=pipeline_session,
        hyperparameters=fixed_hyperparameters,
        role=role,
    )


    xgb_train.set_hyperparameters(
        eval_metric="f1",
        objective="binary:logistic",
        num_round=5,
        max_depth=2,
        eta=0.2,
        gamma=4,
        min_child_weight=6,
        subsample=0.7,
        silent=0
    )

    objective_metric_name = "validation:f1"
    
    
    # more robust 
    hyperparameter_ranges = {
    "eta": ContinuousParameter(0.15, 0.23, scaling_type="Linear"),  # Adjusted Learning rate range
    "max_depth": IntegerParameter(8, 12),  # Expanded Maximum depth of a tree range
    "min_child_weight": IntegerParameter(2, 6),  # Adjusted Minimum sum of instance weight range
    "subsample": ContinuousParameter(0.52, 0.6, scaling_type="Linear"),  # Adjusted Subsample ratio of the training instances range
    "gamma": ContinuousParameter(0.8, 1.4, scaling_type="Linear"),  # Adjusted Minimum loss reduction required range
    "alpha": ContinuousParameter(1.2, 2.0, scaling_type="Linear"),  # Adjusted L1 regularization term range
    "lambda": ContinuousParameter(1.5, 3.0, scaling_type="Logarithmic"),  # Adjusted L2 regularization term range
    "num_round": IntegerParameter(350, 550, scaling_type="Linear"),  # Expanded Number of boosting rounds range
    "colsample_bytree": ContinuousParameter(0.85, 0.95, scaling_type="Linear"),  # Adjusted Subsample ratio of columns when constructing each tree range
    "colsample_bylevel": ContinuousParameter(0.9, 1.0, scaling_type="Linear"),  # Adjusted Subsample ratio of columns for each level range
    "colsample_bynode": ContinuousParameter(0.8, 0.92, scaling_type="Linear"),  # Adjusted Subsample ratio of columns for each split range
    "max_delta_step": IntegerParameter(4, 8)  # Adjusted Maximum delta step range
}
    # more explorative
    hyperparameter_ranges = {
    "num_round": IntegerParameter(50, 500, scaling_type="Linear"),  # Expanded Number of boosting rounds range
    "alpha": ContinuousParameter(0.001, 1.0, scaling_type="Logarithmic"),
    "eta": ContinuousParameter(0.01, 0.3, scaling_type="Linear"),
    "gamma": ContinuousParameter(0, 10, scaling_type="Linear"),
    "max_depth": IntegerParameter(4, 12),
    "min_child_weight": ContinuousParameter(1, 10, scaling_type="Linear"),
    "lambda": ContinuousParameter(0.001, 1.0, scaling_type="Logarithmic"),
    "colsample_bytree": ContinuousParameter(0.3, 0.8, scaling_type="Linear"),
    "colsample_bylevel": ContinuousParameter(0.3, 0.8, scaling_type="Linear"),
    "subsample": ContinuousParameter(0.5, 1.0, scaling_type="Linear"),
    }

#### OLD RANGES ####
    # # Define the hyperparameter ranges
    # hyperparameter_ranges = {
    #     "eta": ContinuousParameter(0.01, 0.2, scaling_type="Linear"),  # Learning rate
    #     "max_depth": IntegerParameter(3, 10),  # Maximum depth of a tree
    #     "min_child_weight": IntegerParameter(1, 10),  # Minimum sum of instance weight (hessian) needed in a child
    #     "subsample": ContinuousParameter(0.5, 0.8, scaling_type="Linear"),  # Subsample ratio of the training instances
    #     "gamma": ContinuousParameter(0.1, 5, scaling_type="Linear"),  # Minimum loss reduction required to make a further partition on a leaf node
    #     "alpha": ContinuousParameter(0.001, 5, scaling_type="Linear"),  # L1 regularization term on weights, with a minimum value above zero
    #     "lambda": ContinuousParameter(1, 10, scaling_type="Logarithmic"),  # L2 regularization term on weights
    #     "num_round": IntegerParameter(50, 500, scaling_type="Linear"),  # Number of boosting rounds
    #     "colsample_bytree": ContinuousParameter(0.5, 1.0, scaling_type="Linear"),  # Subsample ratio of columns when constructing each tree
    #     "colsample_bylevel": ContinuousParameter(0.5, 1.0, scaling_type="Linear"),  # Subsample ratio of columns for each level
    #     "colsample_bynode": ContinuousParameter(0.5, 1.0, scaling_type="Linear"),  # Subsample ratio of columns for each split
    #     "max_delta_step": IntegerParameter(0, 10)  # Maximum delta step we allow each leaf output to be
    # }

# Name Type Value
#_tuning_objective_metric	FreeText	validation:f1
#alpha	Continuous	1.5909336778053522
#colsample_bylevel	Continuous	0.952407335639231
#colsample_bynode	Continuous	0.8604044575743939
#colsample_bytree	Continuous	0.8913410882978179
#eta	Continuous	0.19323512774553894
#eval_metric	FreeText	f1
#gamma	Continuous	1.1079841707865465
#lambda	Continuous	2.2691595537762375
#max_delta_step	Integer	6
#max_depth	Integer	10
#min_child_weight	Integer	4
#num_round	Integer	450
#objective	FreeText	binary:logistic
#silent	FreeText	0
#subsample	Continuous	0.55664145562483


    
    tuner_log = HyperparameterTuner(
        xgb_train,
        objective_metric_name,
        hyperparameter_ranges,
        max_jobs=75,
        max_parallel_jobs=1,
        strategy="Bayesian",
        objective_type="Maximize",
    )

    step_args = tuner_log.fit(
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        }
    )


    step_tuning = TuningStep(
        name="TuningPLXClassificationModel",
        step_args=step_args
    )

    # processing step for evaluation
    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["sh"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name="machine-learning-plx-classification",
        sagemaker_session=pipeline_session,
        role=role
    )
    
    step_args = script_eval.run(
        inputs=[
            ProcessingInput(
                source=step_tuning.get_top_model_s3_uri(
                top_k=0, 
                s3_bucket=default_bucket, 
                prefix=Join(on='/',values=[base_job_prefix,use_case,'train'])
            ),
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "feature_names"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/feature",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test_customer_campaign_mapping"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/data",
            ),
            ProcessingInput(source=script_code_evaluation, destination="/opt/ml/processing/input/code2"),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ],
        code=os.path.join(BASE_DIR, "run_evaluate.sh")
    )

    evaluation_report = PropertyFile(
        name="PLXClassificationEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    step_eval = ProcessingStep(
        name="EvaluatePLXClassificationModel",
        step_args=step_args,
        property_files=[evaluation_report],
    )

    # register model step that will be conditionally executed
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
            step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
        ),
            content_type="application/json"
        )
    )

    model = Model(
        image_uri=image_uri,
        model_data=step_tuning.get_top_model_s3_uri(
        top_k=0, s3_bucket=default_bucket, prefix=Join(on='/',values=[base_job_prefix,use_case,'train'])
    ),
        sagemaker_session=pipeline_session,
        role=role,
    )
    step_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=Join(on='-',values=[base_job_prefix, use_case]),
        description=Join(on='-',values=[base_job_prefix, use_case]),
        approval_status=model_approval_status,
        model_metrics=model_metrics
    )
    step_register = ModelStep(
        name="RegisterPLXClassificationModel",
        step_args=step_args,
    )
    
    # model_name = step_register.properties.ModelName
    
    # Lambda helper class can be used to create the Lambda function
    func = Lambda(
        function_name="StoreModelName",
        execution_role_arn="arn:aws:iam::423644542769:role/service-role/AmazonSageMakerServiceCatalogProductsLambdaRole",
        script="code/lambda_helper.py",
        handler="lambda_helper.lambda_handler",
    )

#    step_lambda = LambdaStep(
#        name="LambdaStep",
#        lambda_func=func,
#        inputs={
#            "model_name": step_register.properties.ModelPackageArn,
#            "base_job_prefix": base_job_prefix,
#            "use_case": use_case
#        },
#        depends_on=[step_register]
#    )

    # condition step for evaluating model quality and branching execution
    
    # f1 > 0.5
    cond_lte = ConditionGreaterThan(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="regression_metrics.f1"
        ),
        right=0.1,
    )
    step_cond = ConditionStep(
        name="Checkf1PLXClassificationEvaluation",
        conditions=[cond_lte],
        if_steps=[step_register],
        else_steps=[],
    )
    
    

    
    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status,
            image_uri,
            use_case,
            base_job_prefix,
            script_code_preprocessing,
            script_code_evaluation
        ],
        steps=[step_process, step_tuning, step_eval, step_cond],
        sagemaker_session=pipeline_session,
    )
    return pipeline
