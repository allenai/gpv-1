LEARNING_DATASETS="refcocop"
DATA_SPLIT=$1
NGPUS=$2
DISTURL=$3
PRETR_EXP_NAME=$4
TRAIN_PERCENT=$5
EXP_NAME="gpv_${LEARNING_DATASETS}_${DATA_SPLIT}_ft_refcocop_${TRAIN_PERCENT}"
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
    task_configs.data_split=$DATA_SPLIT \
    task_configs.refcocop.train_percent=$TRAIN_PERCENT \
    training.ckpt=$ckpt \
    training.freeze=False \
    model.pretr_detr=$DETR_CKPT \
    model.roi_head=True \
    model.detr_joiner.detr_dim=2304

ckpt="${LOCAL_EXP_DIR}/ckpts/model.pth"
aws s3 cp $ckpt "${AWS_EXP_DIR}/ckpts/model.pth"
