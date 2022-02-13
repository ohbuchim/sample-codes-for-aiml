ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text)
REGION=$(python setenv.py config region)
TIMESTAMP=$(TZ='Asia/Tokyo' date '+%Y%m%d%H%M')
EXEC_ID=${EXEC_ID}
ECR_REPOGITORY_PREFIX=$(python setenv.py config image-name-prefix)
PREP_REPO_NAME=$(python setenv.py preprocess image-repo-name)
TRAIN_REPO_NAME=$(python setenv.py train image-repo-name)
EVAL_REPO_NAME=$(python setenv.py evaluate image-repo-name)
INF_REPO_NAME=$(python setenv.py inference image-repo-name)
TRAIN_JOB_NAME=$(python setenv.py config job-name-prefix)-train-${TIMESTAMP}
PREP_JOB_NAME=$(python setenv.py config job-name-prefix)-prep-${TIMESTAMP}
EVAL_JOB_NAME=$(python setenv.py config job-name-prefix)-eval-${TIMESTAMP}
TRAINED_MODEL_S3=$(python setenv.py train output-path)/${TRAIN_JOB_NAME}/output/model.tar.gz
EVAL_RESULT_PATH=$(python setenv.py evaluate result-path)/${EVAL_JOB_NAME}/evaluation.json
MODEL_NAME=$(python setenv.py deploy model-name-prefix)-${TIMESTAMP}