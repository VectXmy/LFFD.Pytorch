'''
@Author: xxxmy
@Github: github.com/VectXmy
@Date: 2019-10-06
@Email: xxxmy@foxmail.com
'''

import torch
import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np
from torchvision import transforms
from .aug import augment_and_show as augment_apply

class VOCDataset(torch.utils.data.Dataset):
    CLASSES_NAME = (
        "__background__ ",
        "hat",
        "person"
    )
    def __init__(self,root_dir,resize_size=[640,640],split='trainval',use_difficult=True,augmentator=[]):
        self.root=root_dir
        self.use_difficult=use_difficult
        self.imgset=split

        self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
        self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")

        with open(self._imgsetpath%self.imgset) as f:
            self.img_ids=f.readlines()
        self.img_ids=[x.strip() for x in self.img_ids]
        self.name2id=dict(zip(VOCDataset.CLASSES_NAME,range(len(VOCDataset.CLASSES_NAME))))
        self.resize_size=resize_size
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
        self.augmentator=augmentator
        print("INFO=====>voc dataset init finished  ! !")
        if len(self.augmentator)>0:
            print("INFO====>add augmentation")
        else:
            print("INFO====>no augmentation")

    def __len__(self):
        return len(self.img_ids)

    def _read_img_rgb(self,path):
        try:
            return cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
        except:
            print("Failed read %s "%path)

    def __getitem__(self,index):
        
        img_id=self.img_ids[index]
        img=self._read_img_rgb(self._imgpath%img_id)
        orig_size=img.shape[:2]

        anno=ET.parse(self._annopath%img_id).getroot()
        boxes=[]
        classes=[]
        for obj in anno.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.use_difficult and difficult:
                continue
            _box=obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box=[
                _box.find("xmin").text, 
                _box.find("ymin").text, 
                _box.find("xmax").text, 
                _box.find("ymax").text,
            ]
            TO_REMOVE=1
            box = tuple(
                map(lambda x: max(x - TO_REMOVE,0), list(map(float, box)))
            )
            boxes.append(box)

            name=obj.find("name").text.lower().strip()
            if name not in self.CLASSES_NAME:
                raise TypeError("%shas not exit label:%s"%(img_id,name))
            classes.append(self.name2id[name])

        
        ############## augmentation ###################
        if len(self.augmentator)>0:
            for t in self.augmentator:
                img,boxes,classes=augment_apply(t,img,bboxes=boxes,categories=classes)
            boxes=np.array(boxes)
                
            try:
                x1x2=boxes[:,[0,2]]
                y1y2=boxes[:,[1,3]]
                x1x2[x1x2>=orig_size[1]]=orig_size[1]-1
                y1y2[y1y2>=orig_size[0]]=orig_size[0]-1
                boxes=np.stack([x1x2[:,0],y1y2[:,0],x1x2[:,1],y1y2[:,1]],axis=-1)
            except Exception as e:
                print("-----------------",boxes,img_id)
                print(e)
        ################################################
        boxes=np.array(boxes,dtype=np.float32)
        img,boxes,scale=self.preprocess_img_boxes(img,boxes,self.resize_size)
        

        img=transforms.ToTensor()(img)#!!!!!!!!!!!!!! ToTensor() only scales narray of np.uint8!!!!!!!!!!
        boxes=torch.tensor(boxes,dtype=torch.float32)
        classes=torch.tensor(classes,dtype=torch.int64)

        return img,boxes,classes,scale,orig_size
        

    def preprocess_img_boxes(self,image,boxes,input_ksize):
        '''
        resize image and bboxes 
        Returns
        image_paded: input_ksize  
        bboxes: [None,4]
        '''
        min_side, max_side    = input_ksize
        h,  w, _  = image.shape

        smallest_side = min(w,h)
        largest_side=max(w,h)
        scale=min_side/smallest_side
        if largest_side*scale>max_side:
            scale=max_side/largest_side
        nw, nh  = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        # pad_w=32-nw%32
        # pad_h=32-nh%32

        # image_paded = np.zeros(shape=[nh+pad_h, nw+pad_w, 3],dtype=np.uint8)
        image_paded = np.zeros(shape=[min_side, max_side, 3],dtype=np.uint8)
        image_paded[:nh, :nw, :] = image_resized

        if boxes is None:
            return image_paded
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale 
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale 
            return image_paded, boxes,scale

    def collate_fn(self,data):
        imgs_list,boxes_list,classes_list,scale_list,orig_size_list=zip(*data)
        assert len(imgs_list)==len(boxes_list)==len(classes_list)==len(scale_list)
        batch_size=len(boxes_list)
        pad_imgs_list=[]
        pad_boxes_list=[]
        pad_classes_list=[]

        h_list = [int(s.shape[1]) for s in imgs_list]
        w_list = [int(s.shape[2]) for s in imgs_list]
        max_h = np.array(h_list).max()
        max_w = np.array(w_list).max()
        for i in range(batch_size):
            img=imgs_list[i]
            pad_imgs_list.append(transforms.Normalize(self.mean, self.std,inplace=True)(torch.nn.functional.pad(img,(0,int(max_w-img.shape[2]),0,int(max_h-img.shape[1])),value=0.)))


        max_num=0
        for i in range(batch_size):
            n=boxes_list[i].shape[0]
            if n>max_num:max_num=n   
        for i in range(batch_size):
            pad_boxes_list.append(torch.nn.functional.pad(boxes_list[i],(0,0,0,max_num-boxes_list[i].shape[0]),value=-1))
            pad_classes_list.append(torch.nn.functional.pad(classes_list[i],(0,max_num-classes_list[i].shape[0]),value=-1))
        

        batch_boxes=torch.stack(pad_boxes_list)
        batch_classes=torch.stack(pad_classes_list)
        batch_imgs=torch.stack(pad_imgs_list)
        

        return batch_imgs,batch_boxes,batch_classes,scale_list,orig_size_list

if __name__ == "__main__":
    from aug import color_aug,pix_aug
    dataset=VOCDataset("/home/xht/dataset/VOC2028",augmentator=[color_aug,pix_aug])
    print(len(dataset))
    imgs,boxes,classes,scales,orig_sizes=dataset.collate_fn([dataset[100],dataset[101],dataset[200],dataset[500]])
    print(boxes,classes,scales,orig_sizes,"\n",imgs.shape,boxes.shape,classes.shape,imgs.dtype,boxes.dtype,classes.dtype)
    for index,i in enumerate(imgs):
        i=i.numpy().astype(np.uint8)
        i=np.transpose(i,(1,2,0))
        i=cv2.cvtColor(i,cv2.COLOR_RGB2BGR)
        print(i.shape,type(i))
        for box,label in zip(boxes[index],classes[index]):
            if box[0]<0:
                continue
            pt1=(int(box[0]),int(box[1]))
            pt2=(int(box[2]),int(box[3]))
            if label==1:
                cv2.rectangle(i,pt1,pt2,[0,255,0],2)
            if label==2:
                cv2.rectangle(i,pt1,pt2,[200,255,0],2)
        cv2.imwrite(str(index)+".jpg",i)