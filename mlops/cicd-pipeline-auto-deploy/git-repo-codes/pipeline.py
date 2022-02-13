import boto3
import json
import logging
import os
import time
import yaml

import sagemaker
from sagemaker import get_execution_role
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.processing import Processor
from sagemaker.processing import ProcessingInput, ProcessingOutput

import stepfunctions
from stepfunctions.inputs import ExecutionInput
from stepfunctions.workflow import Workflow
from stepfunctions.steps.states import (
    Catch,
    Choice,
    Fail,
    Retry
)
from stepfunctions.steps import (
    Chain,
    ProcessingStep,
    TrainingStep,
    ModelStep,
    EndpointConfigStep,
    EndpointStep
)


stepfunctions.set_stream_logger(level=logging.INFO)
config_name = 'flow.yml'

REGION = os.environ['REGION']
ACCOUNT_ID = os.environ['ACCOUNT_ID']
EXEC_ID = os.environ['EXEC_ID']


def get_parameters():
    params = {}
    with open(config_name) as file:
        config = yaml.safe_load(file)
        params['sagemaker-role'] = config['config']['sagemaker-role']
        params['user-name'] = config['config']['user-name']
        params['sagemaker-experiment-name'] = config['config']['sagemaker-experiment-name']
        params['sfn-workflow-name'] = config['config']['sfn-workflow-name']
        params['sfn-role-arn'] = config['config']['sfn-role-arn']
        params['codepipeline-name'] = config['config']['codepipeline-name']
        params['model-package-group-arn'] = config['config']['model-package-group-arn']
        params['job-name-prefix'] = config['config']['job-name-prefix']
        params['prep-job-name'] = os.environ['PREP_JOB_NAME']
        params['prep-image-uri'] = os.environ['PREPRO_IMAGE_URI']
        params['prep-input-path'] = config['preprocess']['input-data-path']
        params['prep-output-path'] = config['preprocess']['output-data-path']
        params['train-job-name'] = os.environ['TRAIN_JOB_NAME']
        params['train-image-uri'] = os.environ['TRAIN_IMAGE_URI']
        params['train-output-path'] = config['train']['output-path']
        params['trained-model-s3'] = os.environ['TRAINED_MODEL_S3']
        params['hyperparameters'] = {}
        params['hyperparameters']['batch-size'] = config['train']['hyperparameters']['batch-size']
        params['hyperparameters']['epoch'] = config['train']['hyperparameters']['epoch']
        params['eval-job-name'] = os.environ['EVAL_JOB_NAME']
        params['eval-image-uri'] = os.environ['EVALUATE_IMAGE_URI']
        params['eval-data-path'] = config['evaluate']['data-path']
        params['eval-result-path'] = config['evaluate']['result-path']
        params['model-name'] = os.environ['MODEL_NAME']
        params['model-name-prefix'] = config['deploy']['model-name-prefix']
        params['metric-threshold'] = config['deploy']['metric-threshold']
        params['endpoint-name'] = config['deploy']['endpoint-name']
        params['deploy-lambda-func-name'] = config['deploy']['lambda-func-name']
        params['inference-image-uri'] = os.environ['INFERENCE_IMAGE_URI']
        params['eval-result-file-path'] = os.environ['EVAL_RESULT_PATH']
        print('------------------')
        print(params)
    return params


def create_prepro_processing(params, sagemaker_role):
    prepro_repository_uri = params['prep-image-uri']

    pre_processor = Processor(
        role=sagemaker_role,
        image_uri=prepro_repository_uri,
        instance_count=1, 
        instance_type="ml.m5.xlarge",
        volume_size_in_gb=16,
        volume_kms_key=None,
        output_kms_key=None,
        max_runtime_in_seconds=86400,  # default is 24 hours(60*60*24)
        sagemaker_session=None,
        env=None,
        network_config=None
    )
    return pre_processor


def create_prepro_step(params, pre_processor, execution_input):
    prepro_input_data = params['prep-input-path']
    prepro_output_data = params['prep-output-path']
    input_dir = '/opt/ml/processing/input'
    output_dir = '/opt/ml/processing/output'

    prepro_inputs = [
        ProcessingInput(
            source=prepro_input_data,
            destination=input_dir,
            input_name="input-data"
        )
    ]

    prepro_outputs = [
        ProcessingOutput(
            source=output_dir,
            destination=prepro_output_data,
            output_name="prepared-data",
        )
    ]

    processing_step = ProcessingStep(
        "SageMaker Pre-processing Step",
        processor=pre_processor,
        job_name=execution_input["PreprocessingJobName"],
        inputs=prepro_inputs,
        outputs=prepro_outputs,
        container_arguments=["--input-dir", input_dir,
                             "--output-dir", output_dir],
        tags={'EXEC_ID': EXEC_ID}
    )
    return processing_step


def create_estimator(params, sagemaker_role):
    train_repository_uri = params['train-image-uri']
    instance_type = 'ml.p3.2xlarge'

    metric_definitions = [{'Name': 'average test loss',
                           'Regex': 'Test set: Average loss: ([0-9\\.]+)'}]
    
    estimator = Estimator(
        image_uri=train_repository_uri,
        role=sagemaker_role,
        metric_definitions=metric_definitions,
        instance_count=1,
        instance_type=instance_type,
        enable_sagemaker_metrics=True,
        hyperparameters={
            'batch-size': params['hyperparameters']['batch-size'],
            'test-batch-size': 4,
            'lr': 0.01,
            'epochs': params['hyperparameters']['epoch']
        },
        output_path=params['train-output-path'],
        )

    return estimator


def create_training_step(params, estimator, execution_input):
    prepro_output_data = params['prep-output-path']
    training_input = TrainingInput(s3_data=prepro_output_data,
                                   input_mode='FastFile')

    training_step = TrainingStep(
        "SageMaker Training Step",
        estimator=estimator,
        data={"training": training_input},
        job_name=execution_input["TrainingJobName"],
        wait_for_completion=True,
        tags={'EXEC_ID': EXEC_ID}
    )

    return training_step


def create_evaluation_processor(params, sagemaker_role):
    evaluation_repository_uri = params['eval-image-uri']
    model_evaluation_processor = Processor(
        image_uri=evaluation_repository_uri,
        role=sagemaker_role,
        instance_count=1,
        instance_type='ml.m5.xlarge',
        max_runtime_in_seconds=1200,
    )
    return model_evaluation_processor


def create_evaluation_step(params, model_evaluation_processor,
                           execution_input, job_name, train_job_name):
    evaluation_output_destination = os.path.join(
        params['eval-result-path'], job_name)
    prepro_output_data = params['prep-output-path']
    trained_model_data = params['trained-model-s3']
    model_dir = '/opt/ml/processing/model'
    data_dir = '/opt/ml/processing/test'
    output_dir = '/opt/ml/processing/evaluation'

    inputs_evaluation = [
        # data path for model evaluation
        ProcessingInput(
            source=prepro_output_data,
            destination=data_dir,
            input_name="data-dir",
        ),
        # model path
        ProcessingInput(
            source=trained_model_data,
            destination=model_dir,
            input_name="model-dir",
        ),
    ]

    outputs_evaluation = [
        ProcessingOutput(
            source=output_dir,
            destination=evaluation_output_destination,
            output_name="output-dir",
        ),
    ]

    evaluation_step = ProcessingStep(
        "SageMaker Evaluation Step",
        processor=model_evaluation_processor,
        job_name=execution_input["EvaluationJobName"],
        inputs=inputs_evaluation,
        outputs=outputs_evaluation,
        experiment_config={
             "ExperimentName": params['sagemaker-experiment-name']},
        container_arguments=[
                "--data-dir", data_dir, "--model-dir", model_dir,
                "--output-dir", output_dir,
                "--experiment-name", params['sagemaker-experiment-name']],
        tags={'EXEC_ID': EXEC_ID}
    )

    return evaluation_step


def create_deploy_step(lambda_function_name):

    deploy_lambda_step = stepfunctions.steps.compute.LambdaStep(
        "Model Deploy Step",
        parameters={
            "FunctionName": lambda_function_name,
            "Payload": {
                "model-name": params['model-name'],
                "endpoint-config-name": params['model-name'],
                "endpoint-name": params['endpoint-name'],
                "model-data-path": params['trained-model-s3'],
                "inf-image-uri": params['inference-image-uri'],
                "sagemaker-role": params['sagemaker-role'],
                "evaluate-result-path": params['eval-result-file-path'],
                "model-package-group-arn": params['model-package-group-arn'],
                "user-name": params['user-name'],
                "model-metrics-path": params['eval-result-file-path'],
                'metric-threshold': params['metric-threshold'],
            },
        },
    )
    deploy_lambda_step.add_retry(
        Retry(error_equals=["States.TaskFailed"], interval_seconds=15,
              max_attempts=2, backoff_rate=4.0)
    )
    return deploy_lambda_step


# def create_pass_data_step(lambda_function_name):

#     pass_data_lambda_step = stepfunctions.steps.compute.LambdaStep(
#         "Pass Evaluation Results",
#         parameters={
#             "FunctionName": lambda_function_name,
#             "Payload": {
#                 "model-metrics-path": params['eval-result-file-path'],
#             },
#         },
#     )
#     pass_data_lambda_step.add_retry(
#         Retry(error_equals=["States.TaskFailed"], interval_seconds=15,
#               max_attempts=2, backoff_rate=4.0)
#     )

#     return pass_data_lambda_step


# def create_check_endpoint_step(lambda_function_name, endpoint_name):

#     check_ep_lambda_step = stepfunctions.steps.compute.LambdaStep(
#         "Check Endpoint Existance",
#         parameters={
#             "FunctionName": lambda_function_name,
#             "Payload": {
#                 "endpoint-name": endpoint_name,
#             },
#         },
#     )
#     check_ep_lambda_step.add_retry(
#         Retry(error_equals=["States.TaskFailed"], interval_seconds=15,
#               max_attempts=2, backoff_rate=4.0)
#     )

#     return check_ep_lambda_step


# def create_model_step(model_name, inference_image_uri):
#     model_step = ModelStep(
#         "Save Model",
#         model=training_step.get_expected_model(),
#         model_name=model_name,
#         parameters={
#             "PrimaryContainer": { 
#                 "ContainerHostname": "Container1",
#                 "Environment": { 
#                     "SAGEMAKER_PROGRAM": "inference.py"
#                 },
#                 "Image": inference_image_uri,
#             },
#         },
#         result_path="$.ModelStepResults",
#     )
#     return model_step


# def create_endpoint_config_step(model_name, endpoint_config_name):
#     endpoint_config_step = EndpointConfigStep(
#         "Create Model Endpoint Config",
#         endpoint_config_name=endpoint_config_name,
#         model_name=model_name,
#         initial_instance_count=1,
#         instance_type="ml.m5.xlarge",
#     )
#     return endpoint_config_step


# def create_create_endpoint_step(endpoint_name, endpoint_config_name):
#     create_endpoint_step = EndpointStep(
#         "Create Endpoint",
#         endpoint_name=endpoint_name,
#         endpoint_config_name=endpoint_config_name,
#         update=False,
#     )
#     return create_endpoint_step


# def create_update_endpoint_step(endpoint_name, endpoint_config_name):
#     update_endpoint_step = EndpointStep(
#         "Update Endpoint",
#         endpoint_name=endpoint_name,
#         endpoint_config_name=endpoint_config_name,
#         update=True,
#     )
#     return update_endpoint_step


# def create_judge_metrics_step(pass_data_step, endpoint_config_step):
#     choice_greater_than = stepfunctions.steps.choice_rule.Rule(
#         variable=pass_data_step.output()['Payload']['accuracy'],
#         operator='NumericGreaterThan',
#         value=90)
#     choice_step = Choice(
#                         state_id='Model Deploy Choice')
#     choice_step.add_choice(choice_greater_than, endpoint_config_step)

#     return choice_step


# def create_choose_api_step(check_endpoint_step,
#                            update_endpoint_step,
#                            create_endpoint_step):
#     choice_endpoint_exists = stepfunctions.steps.choice_rule.ChoiceRule.BooleanEquals(
#         variable=check_endpoint_step.output()['Payload']['result'],
#         value=True)
#     endpoint_choice_step = Choice(state_id='Endpoint Choice')
#     endpoint_choice_step.add_choice(choice_endpoint_exists, update_endpoint_step)
#     endpoint_choice_step.default_choice(create_endpoint_step)

#     return endpoint_choice_step


# def create_error_handling():
#     failed_state_sagemaker_processing_failure = Fail(
#         "ML Workflow failed", cause="SageMakerProcessingJobFailed"
#     )
#     catch_state_processing = Catch(
#         error_equals=["States.TaskFailed"],
#         next_step=failed_state_sagemaker_processing_failure,
#     )

#     judge_metrics_step.default_choice(failed_state_sagemaker_processing_failure)
#     processing_step.add_catch(catch_state_processing)
#     evaluation_step.add_catch(catch_state_processing)
#     training_step.add_catch(catch_state_processing)
#     check_endpoint_step.add_catch(catch_state_processing)
#     pass_data_step.add_catch(catch_state_processing)
#     model_step.add_catch(catch_state_processing)
#     endpoint_config_step.add_catch(catch_state_processing)
#     create_endpoint_step.add_catch(catch_state_processing)
#     update_endpoint_step.add_catch(catch_state_processing)


def create_sfn_workflow(params, steps):
    sfn_workflow_name = params['sfn-workflow-name']
    workflow_execution_role = params['sfn-role-arn']

    workflow_graph = Chain(steps)

    branching_workflow = Workflow(
        name=sfn_workflow_name,
        definition=workflow_graph,
        role=workflow_execution_role,
    )

    branching_workflow.create()
    branching_workflow.update(workflow_graph)

    time.sleep(5)

    return branching_workflow


if __name__ == '__main__':
    params = get_parameters()

    sagemaker_role = params['sagemaker-role']
    prepro_job_name = params['prep-job-name']
    train_job_name = params['train-job-name']
    eval_job_name = params['eval-job-name']
    # pass_lambda_function_name = params['pass-data-lambda-func-name']
    # check_endpoint_lambda_function_name = params['check-endpoint-lambda-func-name']
    deploy_lambda_function_name = params['deploy-lambda-func-name']
    model_name = params['model-name']
    endpoint_config_name = model_name
    inference_image_uri = params['inference-image-uri']
    endpoint_name = params['job-name-prefix']

    execution_input = ExecutionInput(
        schema={
            "PreprocessingJobName": str,
            "TrainingJobName": str,
            "EvaluationJobName": str,
        }
    )

    pre_processor = create_prepro_processing(params,
                                             sagemaker_role)
    processing_step = create_prepro_step(params,
                                         pre_processor, execution_input)

    estimator = create_estimator(params, sagemaker_role)
    training_step = create_training_step(params, estimator, execution_input)

    model_evaluation_processor = create_evaluation_processor(params,
                                                             sagemaker_role)
    evaluation_step = create_evaluation_step(
        params, model_evaluation_processor,
        execution_input, eval_job_name, train_job_name)

    deploy_step = create_deploy_step(deploy_lambda_function_name)

    # pass_data_step = create_pass_data_step(pass_lambda_function_name)
    # check_endpoint_step = create_check_endpoint_step(
    #             check_endpoint_lambda_function_name, endpoint_name)

    # model_step = create_model_step(
    #             model_name, inference_image_uri)
    # endpoint_config_step = create_endpoint_config_step(
    #             model_name, endpoint_config_name)
    # create_endpoint_step = create_create_endpoint_step(
    #             endpoint_name, endpoint_config_name)
    # update_endpoint_step = create_update_endpoint_step(
    #             endpoint_name, endpoint_config_name)

    # judge_metrics_step = create_judge_metrics_step(
    #                                 pass_data_step,
    #                                 endpoint_config_step)
    # choose_api_step = create_choose_api_step(
    #                         check_endpoint_step,
    #                         update_endpoint_step,
    #                         create_endpoint_step)
    # choice_endpoint_exists = stepfunctions.steps.choice_rule.ChoiceRule.BooleanEquals(
    #     variable=check_endpoint_step.output()['Payload']['result'],
    #     value=True)

    # endpoint_config_step.next(check_endpoint_step)
    # check_endpoint_step.next(choose_api_step)

    # create_error_handling()

    branching_workflow = create_sfn_workflow(
        params, [processing_step, training_step,
                 evaluation_step, deploy_step])

#     branching_workflow = create_sfn_workflow(
#         params, [processing_step, training_step, model_step,
#                  evaluation_step,
#                  pass_data_step, judge_metrics_step])
