
import sys
import time
import os
import glob
import numpy as np
import json
import boto3
from sagemaker.analytics import ExperimentAnalytics

def handler(event, context):

    experiment_name = event['experiment-name']
    job_name = event['evaluation-job-name']
    
    print('job_name: ', job_name)
    
    search_expression = {
        "Filters":[
            {
                "Name": "TrialComponentName",
                "Operator": "Contains",
                "Value": job_name,
            }
        ],
    }

    trial_component_analytics = ExperimentAnalytics(
        experiment_name=experiment_name,
        search_expression=search_expression,
    )
    
    df = trial_component_analytics.dataframe()
    print('is_best: ', str(df['is_best']))

    result = False
    if int(df['is_best']) > 0:
        print('This model is the best ever!')
        result = True
    else:
        print('This model is not so good!')
    
    return {
        'statusCode'        : 200,
        'result':result
    }
