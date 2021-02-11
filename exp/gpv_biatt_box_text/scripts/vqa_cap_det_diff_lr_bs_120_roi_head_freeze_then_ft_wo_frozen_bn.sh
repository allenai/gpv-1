EXP_NAME=gpv_biatt_det_vqa_cap_wt_cap_5e-2_bs_120_roi_head_concat_roi_and_detr_hs_freeze_then_ft_wo_frozen_bn
python -m exp.gpv_biatt_box_text.train_distr \
    exp_name=$EXP_NAME \
    ngpus_per_node=2 \
    multiprocessing_distributed=True \
    dist_url='tcp://localhost:10002' \
    learning_datasets=det_vqa_cap \
    training.ckpt=null \
    training.freeze=True \
    training.frozen_epochs=10 \
    training.frozen_batch_size=120 \
    losses.CaptionLoss.loss_wts.loss_caption=5e-2 \
    losses.VqaLoss.loss_wts.loss_vqa=1 \
    model.roi_head=True \
    model.detr_joiner.detr_dim=2304 \
    model.detr.frozenbatchnorm=False

ckpt="/home/tanmayg/Data/gpv/coco_exp/${EXP_NAME}/ckpts/model.pth"
aws s3 cp $ckpt "s3://ai2-prior-gpv/pretrained_models/${EXP_NAME}/ckpts/frozen_model.pth"

python -m exp.gpv_biatt_box_text.train_distr \
    exp_name=$EXP_NAME \
    ngpus_per_node=2 \
    multiprocessing_distributed=True \
    dist_url='tcp://localhost:10002' \
    learning_datasets=det_vqa_cap \
    training.ckpt=$ckpt \
    training.freeze=False \
    training.batch_size=120 \
    losses.CaptionLoss.loss_wts.loss_caption=5e-2 \
    losses.VqaLoss.loss_wts.loss_vqa=1 \
    model.roi_head=True \
    model.detr_joiner.detr_dim=2304 \
    model.detr.frozenbatchnorm=False

aws s3 cp $ckpt "s3://ai2-prior-gpv/pretrained_models/${EXP_NAME}/ckpts/model.pth"