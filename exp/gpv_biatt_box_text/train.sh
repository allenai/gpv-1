python -m exp.gpv_biatt_box_text.train_distr \
    exp_name=gpv_biatt_vqa_freeze \
    ngpus_per_node=4 \
    multiprocessing_distributed=True \
    dist_url='tcp://localhost:10002' \
    learning_datasets=vqa \
    training.ckpt=null \
    training.freeze=True \
    training.frozen_epochs=40 \
    training.frozen_batch_size=120 \
    training.batch_size=120 \
    training.num_val_samples.coco_vqa=20000