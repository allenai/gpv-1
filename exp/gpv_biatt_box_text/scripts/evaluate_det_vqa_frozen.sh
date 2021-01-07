SUBSET=$1
TASK=$2
python -m exp.gpv_biatt_box_text.compute_predictions \
    exp_name=gpv_biatt_det_vqa_freeze \
    learning_datasets=det \
    eval.task=$TASK \
    eval.subset=$SUBSET \
    eval.predict=True \
    eval.num_eval_batches=null