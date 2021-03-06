EXP_NAME=$1
TASK=$2     # possible values: { all, all_but_refexp, <task_name>}
SUBSET=$3   # possible values: { val_test, <subset_name>}
SPLIT=$4    # possible values: { original_split, gpv_split}
OUTPUT_DIR=$5
DATA_DIR=$6

if [[ $TASK == "all" ]]
then
    declare -a TASK_LIST=("RefCocop" "CocoClassification" "CocoVqa" "CocoDetection" "CocoCaptioning")
elif [[ $TASK == "all_but_refexp" ]]
then
    declare -a TASK_LIST=("CocoClassification" "CocoVqa" "CocoDetection" "CocoCaptioning")
else
    declare -a TASK_LIST=($TASK)
fi

if [[ $SUBSET == "val_test" ]]
then
    declare -a SUBSET_LIST=("val" "test")
else
    declare -a SUBSET_LIST=($SUBSET)
fi

for task in ${TASK_LIST[@]}
do
    for subset in ${SUBSET_LIST[@]}
    do
        if [[ $task == "CocoDetection" ]]
        then
            max_text_len=20
            learning_datasets="det"
        elif [[ $task == "CocoVqa" ]]
        then
            max_text_len=5
            learning_datasets="vqa"
        elif [[ $task == "CocoCaptioning" ]]
        then
            max_text_len=20
            learning_datasets="cap"
        elif [[ $task == "CocoClassification" ]]
        then
            max_text_len=5
            learning_datasets="cls"
        elif [[ $task == "RefCocop" ]]
        then
            max_text_len=5
            learning_datasets="refcocop"
        else
            echo "learning dataset ${task} not found"
            exit 1
        fi

        echo "********************************************"
        echo "task: ${task}"
        echo "learning_datasets: ${learning_datasets}"
        echo "subset: ${subset}"
        echo "max_text_len: ${max_text_len}"
        python -m exp.gpv.compute_predictions \
            exp_name=$EXP_NAME \
            output_dir=$OUTPUT_DIR \
            data_dir=$DATA_DIR \
            learning_datasets=$learning_datasets \
            task_configs.data_split=$SPLIT \
            model.max_text_len=$max_text_len \
            eval.task=$task \
            eval.subset=$subset
    done
done