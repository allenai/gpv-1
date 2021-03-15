LEARNING_DATASETS="refcocop"
NGPUS=$1
DISTURL=$2
PRETR_EXP_NAME=$3
TRAIN_PERCENT=$4
EXP_NAME="gpv_${LEARNING_DATASETS}_perc_${TRAIN_PERCENT}_ft_from_${PRETR_EXP_NAME}"
LOCAL_EXP_DIR="/home/tanmayg/Data/gpv/coco_exp/${EXP_NAME}"
AWS_EXP_DIR="s3://ai2-prior-gpv/paper_exps_detr_wo_resizing/refcocop/${EXP_NAME}"
PRETR_EXP_DIR="/home/tanmayg/Data/gpv/coco_exp/${PRETR_EXP_NAME}"

DETR_CKPT="/home/tanmayg/Data/gpv/detr/detr_gpv_coco_256_1e-4_bs_16.pth"
if [[ $DATA_SPLIT == "original_split" ]]
then 
    DETR_CKPT="/home/tanmayg/Data/gpv/detr/detr-r50-e632da11.pth"
fi

ckpt="${PRETR_EXP_DIR}/ckpts/model.pth"


python -m exp.gpv_biatt_box_text.finetune_distr \
    exp_name=$EXP_NAME \
    ngpus_per_node=$NGPUS \
    multiprocessing_distributed=True \
    dist_url="tcp://localhost:${DISTURL}" \
    learning_datasets=$LEARNING_DATASETS \
    task_configs.refcocop.train_percent=$TRAIN_PERCENT \
    training.ckpt=$ckpt \
    training.freeze=False \
    training.num_epochs=200 \
    training.eval_every=5 \
    model.pretr_detr=$DETR_CKPT \
    model.roi_head=True \
    model.detr_joiner.detr_dim=2304

ckpt="${LOCAL_EXP_DIR}/ckpts/model.pth"
aws s3 cp $ckpt "${AWS_EXP_DIR}/ckpts/model.pth"

tb_logs="${LOCAL_EXP_DIR}/tb_logs"
aws s3 cp $tb_logs "${AWS_EXP_DIR}/tb_logs" --recursive

bash exp/gpv_biatt_box_text/scripts/eval_w_refexp_test.sh $EXP_NAME

eval_dir="${LOCAL_EXP_DIR}/eval"
aws s3 cp $eval_dir "${AWS_EXP_DIR}/eval" --recursive