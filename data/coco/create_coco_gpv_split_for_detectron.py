from tqdm import tqdm

import utils.io as io


def keep_anno(anno,held_category_ids,subset_image_ids):
    image_id = anno['image_id']
    cat_id = anno['category_id']
    if image_id in subset_image_ids:
        if cat_id not in held_category_ids:
            return True
    
    return False


def keep_img(img,subset_image_ids):
    image_id = img['id']
    if image_id in subset_image_ids:
        return True
    
    return False


def main(subset):
    anno_json = f"/home/tanmayg/Data/gpv/learning_phase_data/coco/anno/annotations/instances_train2014.json"
    category_split_json = "/home/tanmayg/Data/gpv/learning_phase_data/split_coco_categories/category_split.json"
    image_ids_json = f"/home/tanmayg/Data/gpv/learning_phase_data/split_coco_images/{subset}_images.json"
    held_categories = io.load_json_object(category_split_json)['held_from_det']
    held_category_ids = set([c['id'] for c in held_categories])
    subset_image_ids = set(io.load_json_object(image_ids_json)['image_ids'])
    annos = io.load_json_object(anno_json)
    instance_annos = annos['annotations']
    filtered_annos = []
    for anno in tqdm(instance_annos):
        if keep_anno(anno,held_category_ids,subset_image_ids):
            filtered_annos.append(anno)
    print('Instances before filtering:',len(instance_annos))
    print('Instances after filtering:',len(filtered_annos))

    images = annos['images']
    filtered_images = []
    for img in tqdm(images):
        if keep_img(img,subset_image_ids):
            filtered_images.append(img)
    print('Images before filtering:',len(images))
    print('Images after filtering:',len(filtered_images))

    annos['annotations'] = filtered_annos
    annos['images'] = filtered_images
    filename = f"/home/tanmayg/Data/gpv/learning_phase_data/gpv_instances_{subset}2014.json"
    io.dump_json_object(annos,filename)


if __name__=='__main__':
    main('train')
    main('val')
