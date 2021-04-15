EXP_NAME=$1
LOCAL_EVAL_DIR="/home/tanmayg/Data/gpv/coco_exp/${EXP_NAME}/eval"
AWS_EVAL_DIR="s3://ai2-prior-gpv/paper_exps_detr_wo_resizing/${EXP_NAME}/eval"
aws s3 cp $LOCAL_EVAL_DIR $AWS_EVAL_DIR --recursive