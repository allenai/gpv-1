LEARNING_DATASETS=$1
DATA_SPLIT=$2
NGPUS=$3
DISTURL=$4
EXP_NAME="gpv_wo_ft_${LEARNING_DATASETS}_${DATA_SPLIT}"
LOCAL_EXP_DIR="/home/tanmayg/Data/gpv/coco_exp/${EXP_NAME}"
AWS_EXP_DIR="s3://ai2-prior-gpv/paper_exps/${EXP_NAME}"

DETR_CKPT="/home/tanmayg/Data/gpv/detr/detr_gpv_coco_256_1e-4_bs_16.pth"
if [[ $DATA_SPLIT == "original_split" ]]
then 
    DETR_CKPT="/home/tanmayg/Data/gpv/detr/detr-r50-e632da11.pth"
fi

python -m exp.gpv_biatt_box_text.train_distr \
    exp_name=$EXP_NAME \
    ngpus_per_node=$NGPUS \
    multiprocessing_distributed=True \
    dist_url="tcp://localhost:${DISTURL}" \
    learning_datasets=$LEARNING_DATASETS \
    task_configs.data_split=$DATA_SPLIT \
    training.ckpt=null \
    training.freeze=True \
    model.pretr_detr=$DETR_CKPT \
    model.roi_head=True \
    model.detr_joiner.detr_dim=2304

ckpt="${LOCAL_EXP_DIR}/ckpts/model.pth"
aws s3 cp $ckpt "${AWS_EXP_DIR}/ckpts/model.pth"

tb_logs="${LOCAL_EXP_DIR}/tb_logs"
aws s3 cp $tb_logs "${AWS_EXP_DIR}/tb_logs" --recursive