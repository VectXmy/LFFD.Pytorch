'''
@Author: xxxmy
@Github: github.com/VectXmy
@Date: 2019-12-17
@Email: xxxmy@foxmail.com
'''

import torch
import torch.nn as nn
import math


def coords_fmap2orig(size,stride):
    '''
    transfor one fmap coords to orig coords
    Args
    size [2,] tuple
    stride int
    Returns 
    coords [n,2]
    '''
    h,w=size
    shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32)
    shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = torch.reshape(shift_x, [-1])
    shift_y = torch.reshape(shift_y, [-1])
    coords = torch.stack([shift_x, shift_y], -1) + stride // 2
    return coords

class BranchTarget():
    '''
    gen target for certain branch
    '''
    def __init__(self,size,stride,limit_size,RF_size,mode="area"):
        '''
        size tuple [2,]
        stride int
        limit_size tuple [2,]
        '''
        self.size=size
        self.stride=stride
        self.limit_size=limit_size
        self.RF_size=RF_size
        self.mode=mode
        self.ignore_lower_area=(0.9*self.limit_size[0])**2
        self.ignore_upper_area=(1.1*self.limit_size[1])**2
        self.ignore_lower_length=0.9*self.limit_size[0]
        self.ignore_upper_length=1.1*self.limit_size[1]
        
    def __call__(self,gt_boxes,labels):
        '''
        Args:
        gt_boxes   [b,m,4]
        labels     [b,m]
        Return:
        reg_target [b,h*w,4]
        cls_target [b,h*w]
        '''
        batch_size=gt_boxes.shape[0]
        coords=coords_fmap2orig(self.size,self.stride).to(gt_boxes.device)#[h*w,2]
        x=coords[:,0]#[h*w,]
        y=coords[:,1]
        l_off=x[None,:,None]-gt_boxes[...,0][:,None,:]#[1,h*w,1]-[b,1,m]-->[b,h*w,m]
        t_off=y[None,:,None]-gt_boxes[...,1][:,None,:]
        r_off=gt_boxes[...,2][:,None,:]-x[None,:,None]
        b_off=gt_boxes[...,3][:,None,:]-y[None,:,None]
        ltrb_off=torch.stack([l_off,t_off,r_off,b_off],dim=-1)#[b,h*w,m,4]
        norm_constant=self.RF_size/2.0
        reg_target=ltrb_off/norm_constant#[b,h*w,m,4]
        
        tolerance=1.
        off_min=torch.min(ltrb_off,dim=-1)[0]#[b,h*w,m]
        in_gt_mask=off_min>tolerance#[b,h*w,m]

        if self.mode=="area":
            # in_single_gt_mask=(torch.sum((off_min>0).long(),dim=-1)==1)#[b,h*w]
            areas=(ltrb_off[...,0]+ltrb_off[...,2])*(ltrb_off[...,1]+ltrb_off[...,3])#[b,h*w,m]
            # areas=torch.where(in_gt_mask,areas,torch.full_like(areas,math.inf,dtype=areas.dtype,device=areas.device))
            areas[~in_gt_mask]=math.inf
            areas_min,areas_min_inds=torch.min(areas,dim=-1)#[b,h*w]
            area_lower=self.limit_size[0]**2
            area_upper=self.limit_size[1]**2
            in_size_mask=(areas_min>area_lower)&(areas_min<=area_upper)#[b,h*w]
            
            # pos_mask=in_single_gt_mask&in_size_mask#[b,h*w]
            pos_mask=in_size_mask#[b,h*w]
            ignore_mask=((areas_min>=self.ignore_lower_area)&(areas_min<=area_lower))|\
                        ((areas_min<=self.ignore_upper_area)&(areas_min>area_upper))
        elif self.mode=="edge":
            # areas=(ltrb_off[...,0]+ltrb_off[...,2])*(ltrb_off[...,1]+ltrb_off[...,3])#[b,h*w,m]
            # areas[~in_gt_mask]=math.inf
            # areas_min,areas_min_inds=torch.min(areas,dim=-1)#[b,h*w]
            
            gt_wh=torch.stack([(ltrb_off[...,0]+ltrb_off[...,2]),(ltrb_off[...,1]+ltrb_off[...,3])],dim=3)#[b,h*w,m,2]
            wh_max=torch.max(gt_wh,dim=-1)[0]#[b,h*w,m]
            wh_max[~in_gt_mask]=math.inf
            edges_min,edges_min_inds=torch.min(wh_max,dim=-1)

            length_lower=self.limit_size[0]
            length_upper=self.limit_size[1]
            in_size_mask=(edges_min>length_lower)&(edges_min<=length_upper)#[b,h*w]
            pos_mask=in_size_mask#[b,h*w]
            ignore_mask=((edges_min>=self.ignore_lower_length)&(edges_min<=length_lower))|\
                        ((edges_min<=self.ignore_upper_length)&(edges_min>length_upper))
            
            areas_min_inds=edges_min_inds
            areas=wh_max

        # print(in_single_gt_mask)
        # print(in_size_mask)
        # print(pos_mask)
        # print(ignore_mask)
        area_min_inds_mask=torch.zeros_like(areas,dtype=torch.bool).scatter_(-1,areas_min_inds.unsqueeze(dim=-1),1)#[b,h*w,m]
        reg_target=reg_target[area_min_inds_mask].reshape((batch_size,-1,4))#[b,h*w,4]

        labels=torch.broadcast_tensors(labels[:,None,:],areas.long())[0]#[b,h*w,m]
        cls_targets=labels[area_min_inds_mask]
        cls_targets=torch.reshape(cls_targets,(batch_size,-1))#[b,h*w]

        # obj_targets=torch.zeros_like(cls_targets,dtype=torch.float32,device=cls_targets.device)

        reg_target[~pos_mask]=-1
        cls_targets[~pos_mask]=0
        cls_targets[ignore_mask]=-1
        # obj_targets[pos_mask]=1
        # obj_targets[ignore_mask]=-1

        return reg_target,cls_targets#,obj_targets

def reg_loss(preds,targets,mode="smooth_l1"):
    '''
    preds [b,4,h,w]
    targets [b,h*w,4]
    '''
    b=preds.shape[0]
    preds=preds.permute(0,2,3,1)
    pos_mask=(targets.reshape((-1,4)).sum(dim=1)>0.)#[b*h*w]
    assert preds.shape[3]==4
    preds=preds.reshape((b,-1,4)).reshape((-1,4))#[b*h*w,4]
    targets=targets.reshape((-1,4))
    assert preds.shape==targets.shape
    preds=preds[pos_mask]
    targets=targets[pos_mask]
    pos_num=float(targets.shape[0])
    if mode=="l2":
        if pos_num>0:
            loss=(targets-preds).pow(2).sum()/pos_num
        else:
            loss=(targets-preds).pow(2).sum()
    elif mode=="smooth_l1":
        if pos_num>0:
            loss=nn.functional.smooth_l1_loss(preds,targets)
        else:
            loss=(targets-preds).abs().sum()
    else:
        raise NotImplementedError("only implement [\"smooth_l1\",\"l2\"]")
    return loss
  
def cls_loss(logits,targets):
    '''
    logits [b,class_num+1,h,w]
    targets [b,h*w]
    '''
    b=logits.shape[0]
    c=logits.shape[1]
    logits=logits.permute(0,2,3,1)
    logits=logits.reshape((b,-1,c)).reshape((-1,c))#[b*h*w,class_num+1]
    targets=targets.reshape((-1,))#[b*h*w,]
    return nn.functional.cross_entropy(logits,targets,ignore_index=-1)

def obj_loss(logits,targets):
    '''
    logits [b,1,h,w]
    targets [b,h*w]
    '''
    b=logits.shape[0]
    c=logits.shape[1]
    logits=logits.permute(0,2,3,1)
    logits=logits.reshape((b,-1,c)).reshape((-1,c)).squeeze(dim=1)#[b*h*w,]
    targets=targets.reshape((-1,))#[b*h*w,]
    mask=targets.long()>=0
    logits=logits[mask]
    targets=targets[mask]
    return nn.functional.binary_cross_entropy_with_logits(logits,targets)
    
    

if __name__ == "__main__":
    import cv2
    import numpy as np  
    np.set_printoptions(threshold=np.inf)  

    size=(39,39)
    stride=16
    coords=coords_fmap2orig(size,stride)
    target_gen=BranchTarget(size,stride,(70,110),233,mode="edge")
    boxes=[[[10.,10.,90.,90.],[50.,50.,121.,121.],[100.,100.,176.,176.],[200.,200.,309.,309.],[400.,400.,520.,520.],[312.,312.,361.,361.],[-1,-1,-1,-1]]]
    gt_boxes=torch.tensor(boxes,dtype=torch.float32)
    labels=torch.tensor([[1,2,1,2,1,2,-1]],dtype=torch.int64)
    reg_target,cls_targets=target_gen(gt_boxes,labels)
    print("reg_targets shape: ",reg_target.shape)
    print("cls_targets shape: ",cls_targets.shape)
    print(reg_target[reg_target.sum(dim=-1)>0])
    print(cls_targets[cls_targets>0])

    draw=np.zeros((640,640,3),dtype=np.uint8)
    pos_mask=reg_target.sum(dim=-1)>0#[b,h*w]
    ignore_mask=(cls_targets==-1)#[b,h*w]
    pos_coords=coords[pos_mask.squeeze(dim=0)].numpy()
    pos_labels=cls_targets[pos_mask].numpy()
    ignore_coords=coords[ignore_mask.squeeze(dim=0)].numpy()
    print(pos_coords.shape,pos_labels.shape)
    print(ignore_coords.shape)

    for box in boxes[0]:
        pt1=(int(box[0]),int(box[1]))
        pt2=(int(box[2]),int(box[3]))
        cv2.rectangle(draw,pt1,pt2,[0,255,0],3)
    for pt,label in zip(pos_coords,pos_labels):
        pt=(int(pt[0]),int(pt[1]))
        if label==1:
            cv2.circle(draw,pt,2,[200,255,0],-1)
        elif label==2:
            cv2.circle(draw,pt,2,[0,0,255],-1)
    cv2.imwrite("./target_test.jpg",draw)




        
        

