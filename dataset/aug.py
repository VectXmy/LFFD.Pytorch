import albumentations as A
import cv2
import random
import numpy as np


BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)
def visualize_bbox(img, bbox, color=BOX_COLOR, thickness=2, **kwargs):
    #height, width = img.shape[:2]
    x_min, y_min, x_max, y_max = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    return img

def visualize_titles(img, bbox, title, color=BOX_COLOR, thickness=2, font_thickness = 2, font_scale=0.35, **kwargs):
    #height, width = img.shape[:2]
    x_min, y_min, x_max, y_max = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
    
    ((text_width, text_height), _) = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(img, title, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, TEXT_COLOR,
                font_thickness, lineType=cv2.LINE_AA)
    return img

def augment_and_show(aug, image, mask=None, bboxes=[],
                     categories=[], category_id_to_name=[],
                     font_scale_orig=0.35, font_scale_aug=0.35, 
                     draw=False, **kwargs):

    augmented = aug(image=image, mask=mask, bboxes=bboxes, category_id=categories)

    
    if draw:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_aug = cv2.cvtColor(augmented['image'], cv2.COLOR_BGR2RGB)
        for bbox in bboxes:
            visualize_bbox(image, bbox, **kwargs)

        for bbox in augmented['bboxes']:
            visualize_bbox(image_aug, bbox, **kwargs)

        for bbox,cat_id in zip(bboxes, categories):
            visualize_titles(image, bbox, category_id_to_name[cat_id], font_scale=font_scale_orig, **kwargs)
        for bbox,cat_id in zip(augmented['bboxes'], augmented['category_id']):
            visualize_titles(image_aug, bbox, category_id_to_name[cat_id], font_scale=font_scale_aug, **kwargs)
        cv2.imshow("aug",np.hstack((image,image_aug)))
        cv2.waitKey(0)
    if mask is not None:
        return augmented['image'], augmented['mask'], augmented['bboxes']
    else:
        return augmented['image'], augmented['bboxes'],augmented["category_id"]

color_aug=A.Compose([
    A.RandomBrightness(limit=0.1,p=0.5),
    A.RandomContrast(limit=0.1,p=0.5),
    A.GaussNoise(p=0.5),
    A.HueSaturationValue(p=0.5),
])

pix_aug=A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=2,p=0.5),
    # A.RandomCropNearBBox(p=0.5),
],bbox_params={'format':'pascal_voc','min_area':10,'label_fields': ['category_id']})


def lffd_random_resize(img,boxes,classes,sizes=[(10,15),(15,20),(20,40),(40,70),(70,110),(110,250),(250,400),(400,560)],final_size=(640,640)):
    o_h,o_w=img.shape[:2]
    d_h,d_w=final_size# 目标尺寸

    box_num=len(boxes)
    size_num=len(sizes)
    random_box_ind=random.randint(0,box_num-1)
    random_size_ind=random.randint(0,size_num-1)

    selected_box=boxes[random_box_ind]#随机选择的box
    selected_size=random.randint(sizes[random_size_ind][0],sizes[random_size_ind][1])#随机选一个scale
    
    s_b_h,s_b_w=selected_box[3]-selected_box[1],selected_box[2]-selected_box[0]
    
    s_b_size=(s_b_h+s_b_w)/2
    scale=selected_size/s_b_size
    nw, nh  = int(scale * o_w), int(scale * o_h)
    try:
        img_resized = cv2.resize(img, (nw, nh))
    except:
        print("oh! too large!!")
        return img,boxes,classes
    
    boxes=boxes*scale
    s_b_center_point=(int((selected_box[3]+selected_box[1])*scale/2),int((selected_box[2]+selected_box[0])*scale/2))#(y,x) format
    y_range=(int(max(s_b_center_point[0]-d_h/2,0)),int(min(d_h/2+s_b_center_point[0],nh)))
    x_range=(int(max(s_b_center_point[1]-d_w/2,0)),int(min(d_w/2+s_b_center_point[1],nw)))
    # print(x_range,y_range,scale,(nh,nw))
    

    img_center_croped=img_resized[y_range[0]:y_range[1],x_range[0]:x_range[1],:].copy()
    c_h,c_w=img_center_croped.shape[:2]
    
    _img=np.zeros([d_h,d_w,3],dtype=np.uint8)
    _img[:c_h,:c_w]=img_center_croped

    off_y=y_range[0]
    off_x=x_range[0]
    boxes[:,0::2]-=off_x# scaled boxes加上相对偏移
    boxes[:,1::2]-=off_y 
    _boxes=[]
    _classes=[]
    for b,c in zip(boxes,classes):
        x1,y1,x2,y2=b
        cond1= ((0<=x1<c_w) and (0<=y1<c_h)) 
        cond2= ((0<=x2<c_w) and (0<=y2<c_h))
        if not (cond1 or cond2):#超出crop的范围
            continue
        
        elif cond1 and (not cond2):
            if (0<=x2<c_w):
                y2=c_h-1
            elif (0<=y2<c_h):
                x2=c_w-1
            else:
                x2=c_w-1
                y2=c_h-1
        elif cond2 and (not cond1):
            if (0<=x1<c_w):
                y1=0
            elif (0<=y1<c_h):
                x1=0
            else:
                x1=0
                y1=0
        else:
            pass    
        b_w=(x2-x1)
        b_h=(y2-y1)
        if b_h/max(b_w,1) >=4 or b_w/max(b_h,1)>=4:
            continue
        if max(b_w,b_h)<sizes[0][0]:#小于最小尺寸
            continue
        _boxes.append([x1,y1,x2,y2])
        _classes.append(c)
    _boxes=np.array(_boxes,dtype=np.float32)

    return _img,_boxes,_classes



if __name__ == "__main__":
    '''
    import cv2
    import numpy as np
    from vocdataset import VOCDataset

    # cv2.namedWindow("test",cv2.WINDOW_NORMAL)
    dataset=VOCDataset("/home/xht/dataset/VOC2028",split="test")
    print("INFO===>dataset has %d imgs"%len(dataset))

    n=0
    for i,(img,boxes,classes,_,_) in enumerate(dataset):
        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        aug_img,aug_boxes,aug_labels=augment_and_show(color_aug,img.astype(np.uint8),bboxes=boxes,categories=classes,draw=False)
        aug_img,aug_boxes,aug_labels=augment_and_show(pix_aug,aug_img,draw=False,bboxes=boxes,categories=classes,category_id_to_name={0:'bg',1:"hat",2:"person"})

        for box,label in zip(aug_boxes,aug_labels):
            pt1=(int(box[0]),int(box[1]))
            pt2=(int(box[2]),int(box[3]))
            # print(pt1,pt2)
            if label==1:
                aug_img=cv2.rectangle(aug_img,pt1,pt2,(0,255,0),1)
            elif label==2:
                aug_img=cv2.rectangle(aug_img,pt1,pt2,(0,0,255),1)
        cv2.imwrite("test_%d.jpg"%i,aug_img)
        n+=1
        if n>=10:
            break
        # if cv2.waitKey(0)==27:
        #     break
        '''
