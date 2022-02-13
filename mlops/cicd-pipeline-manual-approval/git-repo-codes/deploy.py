import boto3
import os
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.pytorch.model import PyTorchModel
import yaml

sagemaker_client = boto3.client('sagemaker')
codepipeline_client = boto3.client('codepipeline')
config_name = 'flow.yml'


def get_parameters():
    params = {}
    with open(config_name) as file:
        config = yaml.safe_load(file)
        params['user-name'] = config['config']['user-name']
        params['sagemaker-role'] = config['config']['sagemaker-role']
        params['model-package-group-arn'] = config['config']['model-package-group-arn']
        params['inf-image-uri'] = os.environ['INFERENCE_IMAGE_URI']
        params['model-name'] = config['config']['job-name-prefix']
        params['model-data-path'] = os.environ['TRAINED_MODEL_S3']
        params['timestamp'] = os.environ['TIMESTAMP']
        params['codepipeline-name'] = config['config']['codepipeline-name']
        params['codepipeline-exec-id'] = os.environ['EXEC_ID']
        params['evaluate-result-path'] = os.environ['EVAL_RESULT_PATH']
        print('------------------')
        print(params)
    return params


def endpoint_exists(endpoint_name):
    next_token = ''
    while True:
        if next_token == '':
            response = sagemaker_client.list_endpoints(
                SortBy='Name',
                SortOrder='Ascending',
                MaxResults=100,
                NameContains=endpoint_name,
            )
        else:
            response = sagemaker_client.list_endpoints(
                SortBy='Name',
                SortOrder='Ascending',
                NextToken=next_token,
                MaxResults=100,
                NameContains=endpoint_name,
            )
        for content in response['Endpoints']:
            if endpoint_name == content['EndpointName']:
                return True
        if 'NextToken' in response:
            next_token = response['NextToken']
        else:
            break
    return False


def register_model(approval_comment):
    
    print('approval_comment', approval_comment)

    model_data = params['model-data-path']
    inference_repository_uri = params['inf-image-uri']

    model = PyTorchModel(role=params['sagemaker-role'], model_data=model_data,
                         framework_version='1.9.1', image_uri=inference_repository_uri, entry_point="inference.py")

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=params['evaluate-result-path'],
            content_type="application/json",
        )
    )

    inference_instance_type = "ml.m5.xlarge"
    model_package = model.register(
        model_package_group_name=params['model-package-group-arn'],
        inference_instances=[inference_instance_type],
        transform_instances=[inference_instance_type],
        content_types=["application/x-npy"],
        response_types=["application/x-npy"],
        description=approval_comment,
        model_metrics=model_metrics,
        approval_status="Approved",  # Approved/Rejected/PendingManualApproval (default: “PendingManualApproval”)
    )

    model_package_arn_v = model_package.model_package_arn
    print("Model Package ARN : ", model_package_arn_v)

    return model_package_arn_v


def get_approval_comment():
    action_list = []
    next_token = ''
    while True:
        if next_token == '':
            response = codepipeline_client.list_action_executions(
                pipelineName=params['codepipeline-name'],
                filter={
                    'pipelineExecutionId': params['codepipeline-exec-id']
                },
            )
        else:
            response = codepipeline_client.list_action_executions(
                pipelineName=params['codepipeline-name'],
                filter={
                    'pipelineExecutionId': params['codepipeline-exec-id']
                },
                nextToken=next_token
            )
        for r in response['actionExecutionDetails']:
            if r['pipelineExecutionId'] == params['codepipeline-exec-id']:
                action_list = response['actionExecutionDetails']
                break
        if 'NextToken' in response:
            next_token = response['NextToken']
        else:
            break

#     response = codepipeline_client.list_action_executions(
#         pipelineName=params['codepipeline-name'],
#         filter={
#             'pipelineExecutionId': params['codepipeline-exec-id']
#         },
#     #     nextToken='string'
#     )
    for r in action_list:
        if r['actionName'] == 'ManualApproval':
            return r['output']['executionResult']['externalExecutionSummary']

    return ''


if __name__ == '__main__':

    params = get_parameters()

    model_name = params['model-name'] + '-' + params['timestamp']
    endpoint_config_name = params['model-name'] + '-' + params['timestamp']
    endpoint_name = params['model-name']

    model_package_arn_v = register_model(get_approval_comment())

    # パラメタの詳細はこちら
    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker.html#SageMaker.Client.create_model
    response = sagemaker_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            'ContainerHostname': 'PytorchContainer',
#             'Image': params['inf-image-uri'],
#             'ModelDataUrl': params['model-data-path'],
            'ModelPackageName': model_package_arn_v,
#             'Environment': {
#                 "SAGEMAKER_PROGRAM": "inference.py"
#             },
        },
        ExecutionRoleArn=params['sagemaker-role'],
        Tags=[
            {
                'Key': 'owner',
                'Value': params['user-name']
            },
        ],
        EnableNetworkIsolation=False
    )

    model_arn = response['ModelArn']

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
                'Value': params['user-name']
            },
        ],
    )

    endpoint_config_arn = response['EndpointConfigArn']

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
                    'Value': params['user-name']
                },
            ]
        )

    