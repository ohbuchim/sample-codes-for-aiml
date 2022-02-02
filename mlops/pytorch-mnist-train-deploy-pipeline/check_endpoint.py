import boto3
import json

client = boto3.client('sagemaker')

def lambda_handler(event, context):
    endpoint_name = event['endpoint-name']
    next_token = ''
    result = False
    while True:
        if next_token == '':
            response = client.list_endpoints(
                SortBy='Name',
                SortOrder='Ascending',
                MaxResults=100,
                NameContains=endpoint_name,
            )
        else:
            response = client.list_endpoints(
                SortBy='Name',
                SortOrder='Ascending',
                NextToken=next_token,
                MaxResults=100,
                NameContains=endpoint_name,
            )
        for content in response['Endpoints']:
            if endpoint_name == content['EndpointName']:
                result = True
                break
        if 'NextToken' in response:
            next_token = response['NextToken']
        else:
            break

    return {
        'statusCode': 200,
        'result': result
    }
