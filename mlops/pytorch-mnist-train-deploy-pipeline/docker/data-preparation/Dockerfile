FROM 420964472730.dkr.ecr.us-east-1.amazonaws.com/pytorch-mnist-base-sample2:latest
    
ENV AWS_DEFAULT_REGION us-east-1

COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -qU -r requirements.txt

ENTRYPOINT ["python3", "/opt/ml/processing/input/code/preprocessing.py"]
