EXP_NAME=gpv_cls_roi_head
python -m exp.gpv_biatt_box_text.train_distr \
    exp_name=$EXP_NAME \
    ngpus_per_node=4 \
    multiprocessing_distributed=True \
    dist_url='tcp://localhost:10002' \
    learning_datasets=cls \
    training.ckpt=null \
    training.freeze=True \
    model.roi_head=True \
    model.detr_joiner.detr_dim=2304

ckpt="/home/tanmayg/Data/gpv/coco_exp/${EXP_NAME}/ckpts/model.pth"
aws s3 cp $ckpt "s3://ai2-prior-gpv/pretrained_models/${EXP_NAME}/ckpts/frozen_model.pth"

python -m exp.gpv_biatt_box_text.train_distr \
    exp_name=$EXP_NAME \
    ngpus_per_node=4 \
    multiprocessing_distributed=True \
    dist_url='tcp://localhost:10002' \
    learning_datasets=cls \
    training.ckpt=$ckpt \
    training.freeze=False \
    model.roi_head=True \
    model.detr_joiner.detr_dim=2304

aws s3 cp $ckpt "s3://ai2-prior-gpv/pretrained_models/${EXP_NAME}/ckpts/model.pth"