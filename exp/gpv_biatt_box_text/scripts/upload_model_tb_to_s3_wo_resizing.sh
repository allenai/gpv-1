EXP_NAME=$1
LOCAL_EXP_DIR="/home/tanmayg/Data/gpv/coco_exp/${EXP_NAME}"
AWS_EXP_DIR="s3://ai2-prior-gpv/paper_exps_detr_wo_resizing/${EXP_NAME}"
ckpt="${LOCAL_EXP_DIR}/ckpts/model.pth"
aws s3 cp $ckpt "${AWS_EXP_DIR}/ckpts/model.pth"
tb_logs="${LOCAL_EXP_DIR}/tb_logs"
aws s3 sync $tb_logs "${AWS_EXP_DIR}/tb_logs"