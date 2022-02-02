import boto3
import json
import os

s3_client = boto3.client('s3')

def lambda_handler(event, context):
    # TODO implement
    print(event)
    file_path = os.path.join(event['model-metrics-path'], 'evaluation.json')
    bucket = file_path.split('/')[2]
    key = file_path[6+len(bucket):]
    print(bucket)
    print(key)
    response = s3_client.get_object(Bucket=bucket, Key=key)
    data = json.loads(response['Body'].read())
    print(data)
    
    return {
        'statusCode': 200,
        'accuracy': data['accuracy']
    }
