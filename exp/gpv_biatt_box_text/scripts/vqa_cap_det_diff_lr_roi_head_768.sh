EXP_NAME=gpv_biatt_det_vqa_cap_wt_cap_5e-2_bs_120_roi_head_concat_roi_and_detr_hs_768
ckpt="/home/tanmayg/Data/gpv/coco_exp/${EXP_NAME}/ckpts/model.pth"
python -m exp.gpv_biatt_box_text.train_distr \
    exp_name=$EXP_NAME \
    ngpus_per_node=2 \
    multiprocessing_distributed=True \
    dist_url='tcp://localhost:10001' \
    learning_datasets=det_vqa_cap \
    training.ckpt=$ckpt \
    training.freeze=True \
    training.frozen_epochs=40 \
    training.frozen_batch_size=120 \
    training.batch_size=120 \
    losses.CaptionLoss.loss_wts.loss_caption=5e-2 \
    losses.VqaLoss.loss_wts.loss_vqa=1 \
    model.roi_head=True \
    model.detr.hidden_dim=768 \
    model.detr_joiner.detr_dim=2816 \
    model.pretr_detr=/home/tanmayg/Data/gpv/detr/detr_gpv_coco_768_5e-5_bs_32.pth

# ckpt="/home/tanmayg/Data/gpv/coco_exp/${EXP_NAME}/ckpts/model.pth"
aws s3 cp $ckpt "s3://ai2-prior-gpv/pretrained_models/${EXP_NAME}/ckpts/model.pth"