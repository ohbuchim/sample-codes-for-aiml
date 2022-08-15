import boto3
import json
from time import sleep
from stepfunctions.workflow import Workflow


def get_policy_arn(policy_name):
    iam_client = boto3.client('iam')
    marker = ''
    while True:
        if marker == '':
            response = iam_client.list_policies(Scope='Local')
        else:
            response = iam_client.list_policies(Scope='Local', Marker=marker)
        for content in response['Policies']:
            if policy_name == content['PolicyName']:
                return content['Arn']
        if 'Marker' in response:
            marker = response['Marker']
        else:
            break

    return ''


def detach_role_policies(role_name):
    iam_client = boto3.client('iam')
    try:
        response = iam_client.list_attached_role_policies(
            RoleName=role_name,
        )
    except Exception as ex:
        print(ex)
    policies = response['AttachedPolicies']

    for p in policies:
        response = iam_client.detach_role_policy(
            RoleName=role_name,
            PolicyArn=p['PolicyArn']
        )

            
def create_role(role_name, assume_role_policy):
    iam_client = boto3.client('iam')
    try:
        response = iam_client.create_role(
            Path='/service-role/',
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(assume_role_policy),
            MaxSessionDuration=3600*12  # 12 hours
        )
        role_arn = response['Role']['Arn']
    except Exception as ex:
        if "EntityAlreadyExists" in str(ex):
            detach_role_policies(role_name)
            response = iam_client.delete_role(
                RoleName=role_name,
            )
            response = iam_client.create_role(
                Path='/service-role/',
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(assume_role_policy),
                MaxSessionDuration=3600*12  # 12 hours
            )
            role_arn = response['Role']['Arn']
        else:
            print(ex)
    sleep(10)
    return role_arn


def create_policy(policy_name, policy_json_name):
    iam_client = boto3.client('iam')
    with open('policy/' + policy_json_name, 'r') as f:
        policy_json = json.load(f)
    try:
        response = iam_client.create_policy(
            PolicyName=policy_name,
            PolicyDocument=json.dumps(policy_json),
        )
        policy_arn = response['Policy']['Arn']
    except Exception as ex:
        if "EntityAlreadyExists" in str(ex):
            response = iam_client.delete_policy(
                PolicyArn=get_policy_arn(policy_name)
            )
            response = iam_client.create_policy(
                PolicyName=policy_name,
                PolicyDocument=json.dumps(policy_json),
            )
            policy_arn = response['Policy']['Arn']
#     policy_arn_list.append(policy_arn)

    sleep(10)
    return policy_arn


def create_policy_role(policy_name, policy_json_name, role_name,
                       assume_role_policy,
                       role_name_list, policy_arn_list):
    iam_client = boto3.client('iam')
    role_arn = create_role(role_name, assume_role_policy)
    policy_arn = create_policy(policy_name, policy_json_name)

    sleep(5)
    response = iam_client.attach_role_policy(
        RoleName=role_name,
        PolicyArn=policy_arn
    )

    role_name_list.append(role_name)
    policy_arn_list.append(policy_arn)
    sleep(10)
    return role_arn


def create_bucket(bucket_name, region, account_id):
    s3_client = boto3.client('s3', region_name=region)
    if region == 'us-east-1':
        response = s3_client.create_bucket(Bucket=bucket_name)
    else:
        location = {'LocationConstraint': region}
        response = s3_client.create_bucket(Bucket=bucket_name,
                                           CreateBucketConfiguration=location)
    sleep(10)
    response = s3_client.put_bucket_encryption(
        Bucket=bucket_name,
        ServerSideEncryptionConfiguration={
            'Rules': [
                {
                    'ApplyServerSideEncryptionByDefault': {
                        'SSEAlgorithm': 'AES256',
                    },
                },
            ]
        },
    )

    response = s3_client.put_public_access_block(
        Bucket=bucket_name,
        PublicAccessBlockConfiguration={
            'BlockPublicAcls': True,
            'IgnorePublicAcls': True,
            'BlockPublicPolicy': True,
            'RestrictPublicBuckets': True
        },
        ExpectedBucketOwner=account_id
    )


def create_lambda_function(function_name, file_name, role_arn, handler_name,
                           lambda_function_list,
                           envs={}, py_version='python3.9'):
    lambda_client = boto3.client('lambda')

    with open(file_name+'.zip', 'rb') as f:
        zip_data = f.read()

    if function_exists(function_name):

        response = lambda_client.update_function_configuration(
            FunctionName=function_name,
            Environment={
                'Variables': envs
            },
        )
        sleep(10)
        response = lambda_client.update_function_code(
            FunctionName=function_name,
            ZipFile=zip_data,
            Publish=True,
        )

    else:
        response = lambda_client.create_function(
            FunctionName=function_name,
            Role=role_arn,
            Handler=handler_name+'.lambda_handler',
            Runtime=py_version,
            Code={
                'ZipFile':zip_data
            },
            Environment={
                'Variables': envs
            },
            Timeout=60*5,  # 5 minutes
            MemorySize=128,  # 128 MB
            Publish=True,
            PackageType='Zip',
        )
    lambda_function_list.append(function_name)
    return response['FunctionArn']


def function_exists(function_name):
    lambda_client = boto3.client('lambda')
    try:
        lambda_client.get_function(
            FunctionName=function_name,
        )
        return True
    except Exception as e:
        return False


def delete_role_policy(role_name_list, policy_arn_list):
    iam_client = boto3.client('iam')
    for r in role_name_list:
        try:
            detach_role_policies(r)
            iam_client.delete_role(RoleName=r)
            print('IAM Role 削除完了:', r)
        except Exception as e:
            print(e)
            pass

    for p in policy_arn_list:
        try:
            iam_client.delete_policy(PolicyArn=p)
            print('IAM Policy 削除完了:', p)
        except Exception as e:
            print(e)


def get_sfn_workflow_arn(workflow_name):
    workflow_list = Workflow.list_workflows()
    workflow_arn = [d['stateMachineArn'] for d in workflow_list  if d['name']==workflow_name][0]
    return workflow_arn