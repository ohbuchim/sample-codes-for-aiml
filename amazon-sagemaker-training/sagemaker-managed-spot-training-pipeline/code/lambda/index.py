import json

def lambda_handler(event, context):
    print(event)
    print('------------')
    
    loop_counter = event['counter']
    
    if 'info' in event:
        if 'Cause' in event['info']:
            print("TrainingJobName:", event['info']['Cause']['TrainingJobName'])
            print("TrainingJobStatus:", event['info']['Cause']['TrainingJobStatus'])
            print("SecondaryStatus:", event['info']['Cause']['SecondaryStatus'])
    
    if loop_counter == len(event['input']['InstanceList']):
        isSpot = 'Fail'
        instance_type = 'None'

    else:
        instance_type = event['input']['InstanceList'][loop_counter][0]
        training_type = event['input']['InstanceList'][loop_counter][1]
        
        if training_type == 'spot':
            isSpot = 'true'
            stopping_condition = {
                'MaxRuntimeInSeconds':event['input']['SpotMaxRuntimeInSeconds'],
                'MaxWaitTimeInSeconds':event['input']['SpotMaxWaitTimeInSeconds']
            }
        else:
            isSpot = 'false'
            stopping_condition = {
                'MaxRuntimeInSeconds':event['input']['OndemandMaxRuntimeInSeconds']
            }
    return {
        'isSpot': isSpot,
        'instanceType': instance_type,
        'instanceList': event['input']['InstanceList'],
        'counter': loop_counter+1,
        'StoppingCondition': stopping_condition
    }
