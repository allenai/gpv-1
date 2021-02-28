EXP_NAME=$1

for subset in "val" "test"
do
    python -m exp.cls_baseline.compute_predictions \
        exp_name=$EXP_NAME \
        learning_datasets=cls \
        eval.task=CocoClassification \
        eval.subset=$subset \
        eval.predict=True \
        eval.num_eval_batches=null
done
