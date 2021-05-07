LEARNING_DATASETS=$1
DATA_SPLIT=$2
EXP_NAME=$3
OUTPUT_DIR=$4
DATA_DIR=$5

DETR_CKPT="${DATA_DIR}/detr/detr_coco_sce.pth"
if [[ $DATA_SPLIT == "original_split" ]]
then 
    DETR_CKPT="${DATA_DIR}/detr/detr_coco.pth"
fi

# DETR components are frozen and rest of the model weights are finetuned
python -m exp.gpv.train_distr \
    exp_name=$EXP_NAME \
    output_dir=$OUTPUT_DIR \
    data_dir=$DATA_DIR \
    learning_datasets=$LEARNING_DATASETS \
    task_configs.data_split=$DATA_SPLIT \
    model.pretr_detr=$DETR_CKPT \
    training.freeze=True

# Path to the checkpoint saved from the previous step
CKPT="${OUTPUT_DIR}/${EXP_NAME}/ckpts/model.pth"

# Finetune entire model including DETR weights
python -m exp.gpv.train_distr \
    exp_name=$EXP_NAME \
    output_dir=$OUTPUT_DIR \
    data_dir=$DATA_DIR \
    learning_datasets=$LEARNING_DATASETS \
    task_configs.data_split=$DATA_SPLIT \
    training.ckpt=$CKPT \
    training.freeze=False