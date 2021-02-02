BUCKET="s3://ai2-prior-gpv"
GPV_DATA=$HOME/Data/gpv
mkdir -p $GPV_DATA
mkdir $GPV_DATA/learning_phase_data
aws s3 cp $BUCKET/splits.zip $GPV_DATA/learning_phase_data/
unzip $GPV_DATA/learning_phase_data/splits.zip -d $GPV_DATA/learning_phase_data/
mkdir $GPV_DATA/detr
DETR_CKPT="detr_gpv_checkpoint0299.pth" #detr_gpv_coco_256_1e-4_bs_16.pth or detr_gpv_coco_768_5e-5_bs_32.pth
aws s3 cp $BUCKET/detr/$DETR_CKPT $GPV_DATA/detr/$DETR_CKPT
python -m data.coco.download download_coco_images_only=True