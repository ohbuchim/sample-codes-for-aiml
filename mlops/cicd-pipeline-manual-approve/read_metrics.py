import ast
import boto3
import json
import os

codepipeline_client = boto3.client('codepipeline')
s3_client = boto3.client('s3',)

def lambda_handler(event, context):
    jobId = event['CodePipeline.job']['id']
    
    s3path = ast.literal_eval(event['CodePipeline.job']['data']
                                   ['actionConfiguration']['configuration']
                                   ['UserParameters'])['s3path']
    print(s3path)
    bucket = s3path.split('/')[2]
    key = s3path[6+len(bucket):]
    print(bucket)
    print(key)
    response = s3_client.get_object(Bucket=bucket, Key=key)
    data = json.loads(response['Body'].read())
    print(data)
    print('---------')
    # for k, v in d.items():
    result = {}
    for k, v in data['custom_metrics'].items():
        print(k, v['value'])
        result |= {k: str(round(v['value'], 2))}
    print(result)
    
    response = codepipeline_client.put_job_success_result(
        jobId=jobId,
        outputVariables=result
    )
    print(event)
    return {
        'statusCode': 200
    }
