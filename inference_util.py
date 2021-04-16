import cv2
import IPython
import skimage.io as skio
from skimage.transform import resize
import numpy as np

from exp.gpv.vis import add_box

def read_image(img_path,imh=480,imw=640,resize_image=True):
    img = skio.imread(img_path)
    if len(img.shape)==2:
        img = np.tile(np.expand_dims(img,2),(1,1,3))
    else:
        img = img[:,:,:3]
    
    original_image_size = img.shape[:2] # HxW 
    
    if resize_image is False:
        imh = original_image_size[0]
        imw = original_image_size[1]
        
    resized_image = resize(img,(imh,imw),anti_aliasing=True)
    
    return resized_image, original_image_size


def imshow(img):
    _,ret = cv2.imencode('.png', img) 
    i = IPython.display.Image(data=ret)
    IPython.display.display(i)


def vis_sample(img_path,output,num_boxes):
    img = cv2.imread(img_path)
    boxes = output['boxes']
    relevance = output['relevance']
    for b in range(min(boxes.shape[0],num_boxes)):
        add_box(img,boxes[b],relevance[b])
    
    return img