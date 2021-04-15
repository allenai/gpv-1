EXP_NAME=$1
SUBSET=$2   # possible values: {val, test}

echo "********************************************"
echo "exp: ${EXP_NAME}"
echo "split: coco (original_split)"
echo "subset: ${SUBSET}"
echo "********************************************"
python -m exp.gpv_biatt_box_text.compute_cap_test_predictions \
    exp_name=$EXP_NAME \
    task_configs.data_split=original_split \
    model.max_text_len=20 \
    eval.subset=$SUBSET \
    eval.predict=True \
    eval.task=CocoCaptioning \
    eval.num_eval_batches=null \
    model.roi_head=True \
    model.detr_joiner.detr_dim=2304
