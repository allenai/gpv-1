import numpy as np
import skimage.draw as skdraw

def vis_bbox(bbox,img,color=(255,0,0),modify=False,alpha=0.2,fmt='ncxcywh'):
    im_h,im_w = img.shape[0:2]
    if fmt=='cxcywh':
        bbox = cxcywh_to_xyxy(bbox)
    elif fmt=='ncxcywh':
        bbox = cxcywh_to_xyxy(bbox,im_h,im_w)
    elif fmt=='xyxy':
        pass
    else:
        raise NotImplementedError(f'fmt={fmt} not implemented')

    x1,y1,x2,y2 = bbox
    x1 = max(0,min(x1,im_w-1))
    x2 = max(x1,min(x2,im_w-1))
    y1 = max(0,min(y1,im_h-1))
    y2 = max(y1,min(y2,im_h-1))
    r = [y1,y1,y2,y2]
    c = [x1,x2,x2,x1]

    if modify==True:
        img_ = img
    else:
        img_ = np.copy(img)

    if len(img.shape)==2:
        color = (color[0],)

    rr,cc = skdraw.polygon(r,c,img.shape[:2])
    skdraw.set_color(img_,(rr,cc),color,alpha=alpha)

    rr,cc = skdraw.polygon_perimeter(r,c,img.shape[:2])
    
    if len(img.shape)==3:
        for k in range(3):
            img_[rr,cc,k] = color[k]
    elif len(img.shape)==2:
        img_[rr,cc]=color[0]

    return img_


def compute_iou(bbox1,bbox2,fmt='cxcywh',verbose=False):
    if fmt in ['cxcywh' or 'ncxcywh']:
        bbox1 = cxcywh_to_xyxy(bbox1)
        bbox2 = cxcywh_to_xyxy(bbox2)
    elif fmt=='xyxy':
        pass
    else:
        raise NotImplementedError(f'fmt={fmt} not implemented')

    x1,y1,x2,y2 = bbox1
    x1_,y1_,x2_,y2_ = bbox2
    
    x1_in = max(x1,x1_)
    y1_in = max(y1,y1_)
    x2_in = min(x2,x2_)
    y2_in = min(y2,y2_)

    intersection = compute_area(bbox=[x1_in,y1_in,x2_in,y2_in],invalid=0.0)
    area1 = compute_area(bbox1)
    area2 = compute_area(bbox2)
    union = area1 + area2 - intersection
    iou = intersection / (union + 1e-6)

    if verbose:
        return iou, intersection, union

    return iou 


def compute_area(bbox,invalid=None):
    x1,y1,x2,y2 = bbox

    if (x2 <= x1) or (y2 <= y1):
        area = invalid
    else:
        area = (x2 - x1) * (y2 - y1)

    return area


def compute_center(bbox):
    x1,y1,x2,y2 = bbox
    xc = 0.5*(x1+x2)
    yc = 0.5*(y1+y2)
    return (xc,yc)


def cxcywh_to_xyxy(bbox,im_h=1,im_w=1):
    cx,cy,w,h = bbox
    cx,cy,w,h = cx*im_w,cy*im_h,w*im_w,h*im_h
    x1 = cx - 0.5*w
    y1 = cy - 0.5*h
    x2 = cx + 0.5*w
    y2 = cy + 0.5*h
    return (x1,y1,x2,y2)