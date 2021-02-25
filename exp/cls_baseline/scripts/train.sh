DATA_SPLIT=$1
DISTURL=$2
EXP_NAME="cls_baseline_${DATA_SPLIT}"
LOCAL_EXP_DIR="/home/tanmayg/Data/gpv/coco_exp/${EXP_NAME}"
AWS_EXP_DIR="s3://ai2-prior-gpv/paper_exps/${EXP_NAME}"

python -m exp.cls_baseline.train \
    exp_name=$EXP_NAME \
    dist_url="tcp://localhost:${DISTURL}" \
    task_configs.data_split=$DATA_SPLIT