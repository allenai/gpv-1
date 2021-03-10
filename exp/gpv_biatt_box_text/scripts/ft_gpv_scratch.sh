LEARNING_DATASETS="refcocop"
NGPUS=$1
DISTURL=$2
TRAIN_PERCENT=$3
EXP_NAME="gpv_${LEARNING_DATASETS}_perc_${TRAIN_PERCENT}_ft_from_scratch"
LOCAL_EXP_DIR="/home/tanmayg/Data/gpv/coco_exp/${EXP_NAME}"
AWS_EXP_DIR="s3://ai2-prior-gpv/paper_exps_detr_wo_resizing/refcocop/${EXP_NAME}"

DETR_CKPT="/home/tanmayg/Data/gpv/detr/detr_gpv_coco_256_1e-4_bs_16.pth"

python -m exp.gpv_biatt_box_text.finetune_distr \
    exp_name=$EXP_NAME \
    ngpus_per_node=$NGPUS \
    multiprocessing_distributed=True \
    dist_url="tcp://localhost:${DISTURL}" \
    learning_datasets=$LEARNING_DATASETS \
    task_configs.refcocop.train_percent=$TRAIN_PERCENT \
    training.ckpt=null \
    training.freeze=False \
    model.pretr_detr=$DETR_CKPT \
    model.roi_head=True \
    model.detr_joiner.detr_dim=2304

ckpt="${LOCAL_EXP_DIR}/ckpts/model.pth"
aws s3 cp $ckpt "${AWS_EXP_DIR}/ckpts/model.pth"

tb_logs="${LOCAL_EXP_DIR}/tb_logs"
aws s3 cp $tb_logs "${AWS_EXP_DIR}/tb_logs" --recursive

bash exp/gpv_biatt_box_text/scripts/eval_w_refexp.sh $EXP_NAME

eval_dir="${LOCAL_EXP_DIR}/eval"
aws s3 cp $eval_dir "${AWS_EXP_DIR}/eval" --recursive