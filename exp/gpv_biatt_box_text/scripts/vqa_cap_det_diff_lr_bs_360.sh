python -m exp.gpv_biatt_box_text.train_distr \
    exp_name=gpv_biatt_det_vqa_cap_wt_cap_5e-2_bs_360 \
    ngpus_per_node=2 \
    multiprocessing_distributed=True \
    dist_url='tcp://localhost:10001' \
    learning_datasets=det_vqa_cap \
    training.ckpt=null \
    training.freeze=True \
    training.frozen_epochs=40 \
    training.frozen_batch_size=360 \
    training.batch_size=120 \
    training.num_workers=40 \
    losses.CaptionLoss.loss_wts.loss_caption=5e-2 \
    losses.VqaLoss.loss_wts.loss_vqa=1