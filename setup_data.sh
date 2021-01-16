BUCKET="s3://ai2-prior-gpv"
GPV_DATA=$HOME/Data/gpv
mkdir -p $GPV_DATA
mkdir $GPV_DATA/learning_phase_data
aws s3 cp $BUCKET/splits.zip $GPV_DATA/learning_phase_data/
unzip $GPV_DATA/learning_phase_data/splits.zip -d $GPV_DATA/learning_phase_data/
mkdir $GPV_DATA/detr
aws s3 cp $BUCKET/detr/detr_gpv_checkpoint0299.pth $GPV_DATA/detr/detr_gpv_checkpoint0299.pth
python -m data.coco.download download_coco_images_only=True