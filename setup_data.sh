BUCKET="s3://ai2-prior-gpv"
GPV_DATA=$HOME/Data/gpv
mkdir -p $GPV_DATA
mkdir $GPV_DATA/learning_phase_data
aws s3 cp $BUCKET/splits.zip $GPV_DATA/learning_phase_data/
unzip $GPV_DATA/learning_phase_data/original_and_gpv_splits.zip -d $GPV_DATA/learning_phase_data/
mkdir $GPV_DATA/detr
aws s3 cp $BUCKET/detr/detr-r50-e632da11.pth $GPV_DATA/detr/
aws s3 cp $BUCKET/detr/detr_gpv_coco_256_1e-4_bs_16.pth $GPV_DATA/detr/
python -m data.coco.download download_coco_images_only=True