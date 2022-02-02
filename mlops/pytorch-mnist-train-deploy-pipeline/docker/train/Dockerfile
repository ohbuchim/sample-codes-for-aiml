FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.9.1-gpu-py38-cu111-ubuntu20.04
 
COPY requirements.txt ./
RUN pip install --no-cache-dir -U -r requirements.txt

ENV SAGEMAKER_PROGRAM train.py
