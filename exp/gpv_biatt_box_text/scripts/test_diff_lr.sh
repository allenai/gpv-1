python -m exp.gpv_biatt_box_text.train_distr \
    exp_name=gpv_biatt_det_freeze_diff_lr_trial \
    ngpus_per_node=4 \
    multiprocessing_distributed=True \
    dist_url='tcp://localhost:10002' \
    learning_datasets=det \
    training.ckpt=null \
    training.freeze=True \
    training.frozen_epochs=40 \
    training.frozen_batch_size=120 \
    training.batch_size=120 \
    losses.CaptionLoss.loss_wts.loss_caption=5e-2 \
    losses.VqaLoss.loss_wts.loss_vqa=1