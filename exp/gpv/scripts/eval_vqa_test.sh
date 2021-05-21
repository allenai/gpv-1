EXP_NAME=$1
SUBSET=$2   # possible values: {test, testdev}
OUTPUT_DIR=$3
DATA_DIR=$4

echo "********************************************"
echo "exp: ${EXP_NAME}"
echo "split: coco (original_split)"
echo "task: CocoVqa"
echo "subset: ${SUBSET}"
echo "********************************************"
python -m exp.gpv.compute_vqa_test_predictions \
    exp_name=$EXP_NAME \
    output_dir=$OUTPUT_DIR \
    data_dir=$DATA_DIR \
    task_configs.data_split=original_split \
    model.max_text_len=5 \
    eval.subset=$SUBSET \
    eval.task=CocoVqa