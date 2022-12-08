# Image Classification using AWS SageMaker

Useing AWS Sagemaker and PyTorch to train a RESNET18 pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. 

## Project Set Up and Installation
- hpo.py: this file contains pytorch training script used for sagemaker hyperparameter tuner estimator to find best hyperparameters
- train_model.py: this file contains pytorch training and testing scripts and model architechture to be used with a sagemaker estimator to fine tune a model with best hyperparameters found and makes use of sagemaker debugger and profiler to track how well model is doing
- train_and_deploy.ipynb: in this notebook the data is downloaded from source and stored in S3 Bucket to be used with sagemaker, builds the sagemaker estimators, Deploys the model to an EndPoint and display debugger and profiler reports.

## Dataset
The dataset is the dogbreed classification dataset which contains 133 different classes of dogs breads
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
A ResNET18 pretrained model was used for finetuning on this task and the following hyperparameters values were tuned:
- batch-size: [32,64,128,256,512]
- learning-rate: range(0.001,0.1)
- epochs: [2,4,6]

hyperparameter tuning job:
![]('imgs/tuning-job.png')

Training job:
![]('imgs/training-job.png')

## Debugging and Profiling
**Debugger Configs**: the following sagemaker smdebug rules are hooked to the training pytorch scripts and the sagemaker estimator in the notebook to track the following metrics during training
- vanishing gradient
- overfitting
- poor weight initialization
- overtraining

**Profiler Configs**: the following rules applied to training pytorch scripts and sagemaker estimator to track system metrics during training such as:
- LowGPUUtilization
- Profiler report is provided in the next Results section which includes various system metrics

### Results
**Debugger**: Revealed PoorWeightInitialization issue.

**Profiler** Revealed Low GPU Utilization

profiler Full HTML report included [Report](/profiler_report.html)


## Model Deployment
- The Sagemaker estimator model is deployed to a sagemaker maker endpoint on machine of instance-type: ml.m4.xlarge 
- the endpoint can be queried by send requests to the endpoint through a sagemaker predictor instance
- the data should be in json format.

Deployed Model Endpoint in Service:
![]('imgs/model-endpoint.png')
