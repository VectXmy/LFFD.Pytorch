import albumentations as A


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
    A.HueSaturationValue(p=0.5)
])

pix_aug=A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=5,p=0.5),
    # A.RandomCropNearBBox(p=0.5),
],bbox_params={'format':'pascal_voc','min_area':10,'label_fields': ['category_id']})

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
