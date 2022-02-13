FROM 021269156472.dkr.ecr.us-east-1.amazonaws.com/mlops-autodeploy-pipeline-base-sample:latest

# Include global arg in this stage of the build
ARG FUNCTION_DIR="/function"

COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -qU -r requirements.txt

# Copy function code
RUN mkdir -p $FUNCTION_DIR
COPY app/ $FUNCTION_DIR/

# Set working directory to function root directory
WORKDIR $FUNCTION_DIR


ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]
CMD [ "app.handler" ]
