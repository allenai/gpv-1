## Download data
# python -m data.coco.download

## Split images and categories
# python -m data.coco.split_categories
# python -m data.coco.split_images subset=train
# python -m data.coco.split_images subset=val


## Create original train and val split
# python -m data.vqa.preprocess_vqa subset=train
# python -m data.vqa.preprocess_vqa subset=val

# python -m data.coco.preprocess_coco_captioning subset=train
# python -m data.coco.preprocess_coco_captioning subset=val

# python -m data.coco.preprocess_coco_detection subset=train
# python -m data.coco.preprocess_coco_detection subset=val

# python -m data.coco.preprocess_coco_classification subset=train
# python -m data.coco.preprocess_coco_classification subset=val

## Create gpv splits
for dataset in coco_classification coco_detection vqa coco_captions 
do
    echo $dataset
    for data_subset in 'train' 'val' 'test'
    do
        echo $data_subset
        python -m data.split_data_by_categories \
            dataset_name=$dataset \
            subset=$data_subset \
            stats_only=False
    done
done