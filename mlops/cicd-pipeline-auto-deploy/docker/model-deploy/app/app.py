import os
import numpy as np
import json
import boto3
from sagemaker.analytics import ExperimentAnalytics
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.pytorch.model import PyTorchModel


sagemaker_client = boto3.client('sagemaker')
s3_client = boto3.client('s3')
# config_name = 'flow.yml'


def endpoint_exists(endpoint_name):
    try:
        sagemaker_client.describe_endpoint(
            EndpointName=endpoint_name
        )
        return True
    except Exception:
        print('No endpoints.')
        # print(e)

    return False


def register_model(approval_comment, event, approved):

    print('approval_comment', approval_comment)

    model_data = event['model-data-path']
    inference_repository_uri = event['inf-image-uri']

    model = PyTorchModel(role=event['sagemaker-role'], model_data=model_data,
                         framework_version='1.9.1',
                         image_uri=inference_repository_uri,
                         entry_point="inference.py")

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=event['evaluate-result-path'],
            content_type="application/json",
        )
    )

    status = 'Rejected'
    if approved is True:
        status = 'Approved'

    inference_instance_type = "ml.m5.xlarge"
    model_package = model.register(
        model_package_group_name=event['model-package-group-arn'],
        inference_instances=[inference_instance_type],
        transform_instances=[inference_instance_type],
        content_types=["application/x-npy"],
        response_types=["application/x-npy"],
        description=approval_comment,
        model_metrics=model_metrics,
        approval_status=status,  # Approved/Rejected/PendingManualApproval (default: “PendingManualApproval”)
    )

    model_package_arn_v = model_package.model_package_arn
    print("Model Package ARN : ", model_package_arn_v)

    return model_package_arn_v


def handler(event, context):

    model_name = event['model-name']
    endpoint_config_name = event['endpoint-config-name']
    endpoint_name = event['endpoint-name']
    metric_threshold = event['metric-threshold']
    message = 'Deploy done'

    file_path = event['model-metrics-path']
    bucket = file_path.split('/')[2]
    key = file_path[6+len(bucket):]
    print(bucket, key)
    response = s3_client.get_object(Bucket=bucket, Key=key)
    data = json.loads(response['Body'].read())
    print(data)
    accuracy = data['custom_metrics']['accuracy']['value']

    if accuracy > metric_threshold:
        model_package_arn_v = register_model('Auto approved, accuracy:' +
                                             str(round(accuracy, 2)), event,
                                             True)

        # パラメタの詳細はこちら
        # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker.html#SageMaker.Client.create_model
        response = sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'ContainerHostname': 'PytorchContainer',
                'ModelPackageName': model_package_arn_v,
            },
            ExecutionRoleArn=event['sagemaker-role'],
            Tags=[
                {
                    'Key': 'owner',
                    'Value': event['user-name']
                },
            ],
            EnableNetworkIsolation=False
        )

        model_arn = response['ModelArn']
        print('model_arn:', model_arn)

        # パラメタの詳細はこちら
        # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker.html#SageMaker.Client.create_endpoint_config
        response = sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'Primary',
                    'ModelName': model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': 'ml.m5.xlarge',
                },
            ],
            Tags=[
                {
                    'Key': 'owner',
                    'Value': event['user-name']
                },
            ],
        )

        endpoint_config_arn = response['EndpointConfigArn']
        print('endpoint_config_arn:', endpoint_config_arn)

        if endpoint_exists(endpoint_name):
            # パラメタの詳細はこちら
            # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker.html#SageMaker.Client.update_endpoint
            response = sagemaker_client.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name,
                RetainAllVariantProperties=False,
                RetainDeploymentConfig=False
            )
        else:
            # パラメタの詳細はこちら
            # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker.html#SageMaker.Client.create_endpoint
            response = sagemaker_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name,
                Tags=[
                    {
                        'Key': 'owner',
                        'Value': event['user-name']
                    },
                ]
            )
    else:
        model_package_arn_v = register_model('Not approved, accuracy:' +
                                             str(round(accuracy, 2)), event,
                                             False)
        message = 'Model not deployed. accuracy: ' + str(accuracy)

    print(message)

    return {
        'statusCode': 200,
        'result': message
    }
