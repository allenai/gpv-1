PRETR_CKPT=$1
TRAIN_PERCENT=$2
OUTPUT_DIR=$3
DATA_DIR=$4
EXP_NAME="gpv_refcocop_perc_${TRAIN_PERCENT}"

python -m exp.gpv.finetune_distr \
    exp_name=$EXP_NAME \
    output_dir=$OUTPUT_DIR \
    data_dir=$DATA_DIR \
    learning_datasets=refcocop \
    task_configs.refcocop.train_percent=$TRAIN_PERCENT \
    training.ckpt=$PRETR_CKPT \
    training.freeze=False \
    training.num_epochs=200 \
    training.eval_every=5