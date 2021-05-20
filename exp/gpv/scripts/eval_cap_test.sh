EXP_NAME=$1
SUBSET=$2   # possible values: {val, test}
OUTPUT_DIR=$3
DATA_DIR=$4

echo "********************************************"
echo "exp: ${EXP_NAME}"
echo "split: coco (original_split)"
echo "task: CocoCaptioning"
echo "subset: ${SUBSET}"
echo "********************************************"
python -m exp.gpv.compute_cap_test_predictions \
    exp_name=$EXP_NAME \
    output_dir=$OUTPUT_DIR \
    data_dir=$DATA_DIR \
    task_configs.data_split=original_split \
    eval.subset=$SUBSET \
    eval.task=CocoCaptioning
