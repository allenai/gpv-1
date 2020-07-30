import os
import json
import skimage.io as skio
import skimage.measure as skmeasure
import numpy as np


def get_color_mask(seg,color):
    # get a boolean mask for pixels that have the provided color
    mask = np.ones(seg.shape[:2],dtype=np.bool)
    for i in range(len(color)):
        tmp_mask = seg[:,:,i]==color[i]
        mask = mask*tmp_mask
    
    return mask


def get_labels(seg):
    unique_colors = np.unique(seg.reshape(-1, seg.shape[2]), axis=0)

    masks = []
    for color in unique_colors:
        # remove background [64,64,64]
        if np.all(color==64):
            continue

        mask = get_color_mask(seg,color)
        masks.append(mask)
        

    
def main():
    clevr_dir = '/home/tanmayg/Data/gpv/clevr_min_objects_1_max_objects_3/test_task/test/'
    scenes_path = os.path.join(clevr_dir,f'scenes/CLEVR_new_000002.json')
    seg_path = os.path.join(clevr_dir,'images/CLEVR_new_000002_mask.png')
    seg = skio.imread(seg_path)[:,:,:3]
    #num_groups = skmeasure.label(img,neighbors=4,background=-1,return_num=True,connectivity=1)
    #boxes = json.load(open(scenes_path))
    get_labels(seg)

    import pdb; pdb.set_trace()


if __name__=='__main__':
    main()