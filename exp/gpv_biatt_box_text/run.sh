python -m exp.gpv_biatt_box_text.train_distr \
    exp_name=gpv_biatt_vqa \
    ngpus_per_node=8 \
    multiprocessing_distributed=True \
    learning_datasets=vqa \
    training.ckpt=null \
    training.freeze=False \
    training.frozen_batch_size=120 \
    training.batch_size=120