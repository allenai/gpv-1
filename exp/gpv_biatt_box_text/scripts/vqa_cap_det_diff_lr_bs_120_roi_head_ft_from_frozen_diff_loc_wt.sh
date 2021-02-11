disturl=$1
EXP_NAME="gpv_biatt_det_vqa_cap_wt_cap_5e-2_bs_120_roi_head_concat_roi_and_detr_hs_ft_from_frozen_all_loc_wt_x10"
ckpts_dir="/home/tanmayg/Data/gpv/coco_exp/${EXP_NAME}/ckpts"
mkdir -p $ckpts_dir
ckpt="${ckpts_dir}/model.pth"
aws s3 cp "s3://ai2-prior-gpv/pretrained_models/gpv_biatt_det_vqa_cap_wt_cap_5e-2_bs_120_roi_head_concat_roi_and_detr_hs_freeze_then_ft/ckpts/frozen_model.pth" $ckpt

python -m exp.gpv_biatt_box_text.train_distr \
    exp_name=$EXP_NAME \
    ngpus_per_node=2 \
    multiprocessing_distributed=True \
    dist_url="tcp://localhost:${disturl}" \
    learning_datasets=det_vqa_cap \
    training.ckpt=$ckpt \
    training.freeze=False \
    training.batch_size=120 \
    losses.CaptionLoss.loss_wts.loss_caption=5e-2 \
    losses.VqaLoss.loss_wts.loss_vqa=1 \
    losses.Localization.loss_wts.loss_ce=10 \
    losses.Localization.loss_wts.loss_bbox=50 \
    losses.Localization.loss_wts.loss_giou=20 \
    model.roi_head=True \
    model.detr_joiner.detr_dim=2304

aws s3 cp $ckpt "s3://ai2-prior-gpv/pretrained_models/${EXP_NAME}/ckpts/model.pth"