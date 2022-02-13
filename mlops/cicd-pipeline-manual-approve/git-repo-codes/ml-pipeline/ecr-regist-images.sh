REGION=${AWS_DEFAULT_REGION}
REGISTRY_URL="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com" 
# IMAGE_TAG="$(git rev-parse HEAD)"
# IMAGE_TAG="$(git rev-parse --short HEAD)"

echo "========"
echo EXEC_ID: ${EXEC_ID}
echo "========"
echo TRAIN_JOB_NAME: ${TRAIN_JOB_NAME}
echo "========"
echo TRAINED_MODEL_S3: ${TRAINED_MODEL_S3}
echo "========"
echo REGION: ${REGION}

IMAGE_TAG=${EXEC_ID}

# prepro
ECR_REPOGITORY=${PREP_REPO_NAME}
IMAGE_URI="${REGISTRY_URL}/${ECR_REPOGITORY}"
PREPRO_IMAGE_URI=$IMAGE_URI:$IMAGE_TAG

aws ecr get-login-password | docker login --username AWS --password-stdin $REGISTRY_URL
aws ecr create-repository --repository-name $ECR_REPOGITORY

docker build -t $ECR_REPOGITORY ml-pipeline/data-preparation/
docker tag ${ECR_REPOGITORY} $IMAGE_URI:${IMAGE_TAG}
docker push $IMAGE_URI:${IMAGE_TAG}
docker tag ${ECR_REPOGITORY} "$IMAGE_URI:latest"
docker push "$IMAGE_URI:latest"

echo "Container registered. URI:${IMAGE_URI}"

# train
ECR_REPOGITORY=${TRAIN_REPO_NAME}
IMAGE_URI="${REGISTRY_URL}/${ECR_REPOGITORY}"
TRAIN_IMAGE_URI=$IMAGE_URI:$IMAGE_TAG

aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin 763104351884.dkr.ecr.${REGION}.amazonaws.com
aws ecr get-login-password | docker login --username AWS --password-stdin $REGISTRY_URL
aws ecr create-repository --repository-name $ECR_REPOGITORY

docker build -t $ECR_REPOGITORY ml-pipeline/train/
docker tag ${ECR_REPOGITORY} $IMAGE_URI:${IMAGE_TAG}
docker push $IMAGE_URI:${IMAGE_TAG}
docker tag ${ECR_REPOGITORY} "$IMAGE_URI:latest"
docker push "$IMAGE_URI:latest"

echo "Container registered. URI:${IMAGE_URI}"

# evaluate
ECR_REPOGITORY=${EVAL_REPO_NAME}
IMAGE_URI="${REGISTRY_URL}/${ECR_REPOGITORY}"
EVALUATE_IMAGE_URI=$IMAGE_URI:$IMAGE_TAG

aws ecr get-login-password | docker login --username AWS --password-stdin $REGISTRY_URL
aws ecr create-repository --repository-name $ECR_REPOGITORY

docker build -t $ECR_REPOGITORY ml-pipeline/model-evaluation/
docker tag ${ECR_REPOGITORY} $IMAGE_URI:${IMAGE_TAG}
docker push $IMAGE_URI:${IMAGE_TAG}
docker tag ${ECR_REPOGITORY} "$IMAGE_URI:latest"
docker push "$IMAGE_URI:latest"

echo "Container registered. URI:${IMAGE_URI}"

# inference
ECR_REPOGITORY=${INF_REPO_NAME}
IMAGE_URI="${REGISTRY_URL}/${ECR_REPOGITORY}"
INFERENCE_IMAGE_URI=$IMAGE_URI:$IMAGE_TAG

aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin 763104351884.dkr.ecr.${REGION}.amazonaws.com
aws ecr get-login-password | docker login --username AWS --password-stdin $REGISTRY_URL
aws ecr create-repository --repository-name $ECR_REPOGITORY

docker build -t $ECR_REPOGITORY ml-pipeline/inference/
docker tag ${ECR_REPOGITORY} $IMAGE_URI:${IMAGE_TAG}
docker push $IMAGE_URI:${IMAGE_TAG}
docker tag ${ECR_REPOGITORY} "$IMAGE_URI:latest"
docker push "$IMAGE_URI:latest"

echo "Container registered. URI:${IMAGE_URI}"
