EXP_NAME=$1

# eval det
for task in "CocoVqa" "CocoDetection" "CocoCaptioning" 
do
    for subset in "val" "test"
    do
        max_text_len=5
        if [[ $task == "CocoDetection" ]]
        then
            learning_datasets="det"
        elif [[ $task == "CocoVqa" ]]
        then
            learning_datasets="vqa"
        elif [[ $task == "CocoCaptioning" ]]
        then
            max_text_len=20
            learning_datasets="cap"
        else
            echo "learning dataset ${task} not found"
            exit 1
        fi

        echo "********************************************"
        echo "task: ${task}"
        echo "learning_datasets: ${learning_datasets}"
        echo "subset: ${subset}"
        echo "max_text_len: ${max_text_len}"
        python -m exp.gpv_biatt_box_text.compute_predictions \
            exp_name=$EXP_NAME \
            learning_datasets=$learning_datasets \
            model.max_text_len=$max_text_len \
            eval.task=$task \
            eval.subset=$subset \
            eval.predict=True \
            eval.num_eval_batches=null \
            model.roi_head=True \
            model.detr_joiner.detr_dim=2304
    done
done