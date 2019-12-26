'''
@Author: xxxmy
@Github: github.com/VectXmy
@Date: 2019-12-17
@Email: xxxmy@foxmail.com
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss import BranchTarget,reg_loss,cls_loss,coords_fmap2orig

'''
| Branch | RF_size | face_size | 
|    1   |   55    |   10-15   |
|    2   |   71    |   15-20   |
|    3   |   111   |   20-40   |
|    4   |   143   |   40-70   |
|    5   |   223   |   70-110  |
|    6   |   383   |   110-250 |
|    7   |   511   |   250-400 |
|    8   |   639   |   400-560 |
'''

class ResV1Block(nn.Module):
    '''
    basic residual block v1
    '''
    def __init__(self,in_channels,out_channels,stride=1,padding=1,BN=True):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=padding)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.BN=BN
        if self.BN:
            self.bn1=nn.BatchNorm2d(out_channels)
            self.bn2=nn.BatchNorm2d(out_channels)
    def forward(self,x):
        out=self.conv1(x)
        if self.BN:
            out=self.bn1(out)
        out=F.relu(out)
        out=self.conv2(out)
        if self.BN:
            out=self.bn2(out)
        out+=x
        out=F.relu(out)
        return out

class ResV2Block(nn.Module):
    '''
    basic residual block v2
    '''
    def __init__(self,in_channels,out_channels,stride=1,padding=1,BN=False):
        super().__init__()
        self.BN=BN
        if BN:
            self.bn1=nn.BatchNorm2d(in_channels)
            self.bn2=nn.BatchNorm2d(out_channels)
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=padding)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        
    def forward(self,x):
        if self.BN:
            x=self.bn1(x)
        out=F.relu(x)
        out=self.conv1(x)
        if self.BN:
            out=self.bn2(out)
        out=F.relu(out)
        out=self.conv2(out)
        out+=x
        return out

class Head(nn.Module):
    '''
    prediction branches
    '''
    def __init__(self,in_channels,out_channels,class_num):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=1)
        self.conv_cls=nn.Conv2d(out_channels,out_channels,kernel_size=1)
        self.cls_logits=nn.Conv2d(out_channels,class_num+1,kernel_size=1)
        self.conv_reg=nn.Conv2d(out_channels,out_channels,kernel_size=1)
        self.reg_pred=nn.Conv2d(out_channels,4,kernel_size=1)
    def forward(self,x):
        out=self.conv1(x)
        out=F.relu(out)

        cls_logits=self.conv_cls(out)
        cls_logits=F.relu(cls_logits)
        cls_logits=self.cls_logits(cls_logits)

        reg_preds=self.conv_reg(out)
        reg_preds=F.relu(reg_preds)
        reg_preds=self.reg_pred(reg_preds)

        return cls_logits,reg_preds

class LFFD(nn.Module):
    def __init__(self,arch_name='LFFD_resv2_25',
                    basic_block=ResV2Block,
                    chs=[64,64,128,128],
                    head_chs=128,
                    class_num=2,
                    BN=False):
        super().__init__()
        self.arch_name=arch_name
        self.basic_block=basic_block
        self.chs=chs
        self.head_chs=head_chs
        self.class_num=class_num
        self.BN=BN
        
        if self.arch_name=="LFFD_resv2_25":
            if self.basic_block==ResV2Block:
                self.stem=nn.Conv2d(3,self.chs[0],kernel_size=3,stride=2)
                if self.BN:
                    self.bn_stem=nn.BatchNorm2d(chs[0])

                self.downsample_conv1=nn.Conv2d(self.chs[0],self.chs[0],kernel_size=3,stride=2)
                self.stage1_1=nn.Sequential(
                                            self.basic_block(self.chs[0],self.chs[0],BN=self.BN),
                                            self.basic_block(self.chs[0],self.chs[0],BN=self.BN),
                                            self.basic_block(self.chs[0],self.chs[0],BN=self.BN)
                                            )
                self.relu1_1=nn.ReLU()
                self.branch1=Head(self.chs[0],self.head_chs,self.class_num)
                self.stage1_2=nn.Sequential(self.basic_block(self.chs[0],chs[0],BN=self.BN))
                self.relu1_2=nn.ReLU()
                self.branch2=Head(self.chs[0],self.head_chs,self.class_num)

                self.downsample_conv2=nn.Conv2d(self.chs[0],self.chs[1],kernel_size=3,stride=2)
                self.stage2_1=nn.Sequential(self.basic_block(self.chs[1],self.chs[1],BN=self.BN))
                self.relu2_1=nn.ReLU()
                self.branch3=Head(self.chs[1],self.head_chs,self.class_num)
                self.stage2_2=nn.Sequential(self.basic_block(self.chs[1],self.chs[1],BN=self.BN))
                self.relu2_2=nn.ReLU()
                self.branch4=Head(self.chs[1],self.head_chs,self.class_num)

                self.downsample_conv3=nn.Conv2d(self.chs[1],self.chs[2],kernel_size=3,stride=2)
                self.stage3_1=nn.Sequential(self.basic_block(self.chs[2],self.chs[2],BN=self.BN))
                self.relu3_1=nn.ReLU()
                self.branch5=Head(self.chs[2],self.head_chs,self.class_num)
                
                self.downsample_conv4=nn.Conv2d(self.chs[2],self.chs[3],kernel_size=3,stride=2)
                self.stage4_1=nn.Sequential(self.basic_block(self.chs[3],self.chs[3],BN=self.BN))
                self.relu4_1=nn.ReLU()
                self.branch6=Head(self.chs[2],self.head_chs,self.class_num)
                self.stage4_2=nn.Sequential(self.basic_block(self.chs[3],self.chs[3],BN=self.BN))
                self.relu4_2=nn.ReLU()
                self.branch7=Head(self.chs[2],self.head_chs,self.class_num)
                self.stage4_3=nn.Sequential(self.basic_block(self.chs[3],self.chs[3],BN=self.BN))
                self.relu4_3=nn.ReLU()
                self.branch8=Head(self.chs[2],self.head_chs,self.class_num)

                if self.BN:
                    self.bn1_1=nn.BatchNorm2d(chs[0])
                    self.bn1_2=nn.BatchNorm2d(chs[0])
                    self.bn2_1=nn.BatchNorm2d(chs[1])
                    self.bn2_2=nn.BatchNorm2d(chs[1])
                    self.bn3_1=nn.BatchNorm2d(chs[2])
                    self.bn4_1=nn.BatchNorm2d(chs[3])
                    self.bn4_2=nn.BatchNorm2d(chs[3])
                    self.bn4_3=nn.BatchNorm2d(chs[3])

        self.initialize_weights()
    
    def initialize_weights(self): 
        for m in self.modules(): 
            if isinstance(m, nn.Conv2d): 
                torch.nn.init.xavier_normal_(m.weight.data) 
                if m.bias is not None: 
                    m.bias.data.zero_() 
    
    def forward(self,x):
        cls_logits=[]
        reg_preds=[]
        if self.arch_name=="LFFD_resv2_25":
            if self.basic_block==ResV2Block:
                x=self.stem(x)
                if self.BN:
                    x=self.bn_stem(x)
                x=F.relu(x)
                x=self.downsample_conv1(x)

                x=self.stage1_1(x)
                if self.BN:
                    x=self.bn1_1(x)
                x=self.relu1_1(x)
                cls_1,reg_1=self.branch1(x)
                cls_logits.append(cls_1)
                reg_preds.append(reg_1)

                x=self.stage1_2(x)
                if self.BN:
                    x=self.bn1_2(x)
                x=self.relu1_2(x)
                cls_2,reg_2=self.branch2(x)
                cls_logits.append(cls_2)
                reg_preds.append(reg_2)

                x=self.downsample_conv2(x)

                x=self.stage2_1(x)
                if self.BN:
                    x=self.bn2_1(x)
                x=self.relu2_1(x)
                cls_3,reg_3=self.branch3(x)
                cls_logits.append(cls_3)
                reg_preds.append(reg_3)

                x=self.stage2_2(x)
                if self.BN:
                    x=self.bn2_2(x)
                x=self.relu2_2(x)
                cls_4,reg_4=self.branch4(x)
                cls_logits.append(cls_4)
                reg_preds.append(reg_4)

                x=self.downsample_conv3(x)

                x=self.stage3_1(x)
                if self.BN:
                    x=self.bn3_1(x)
                x=self.relu3_1(x)
                cls_5,reg_5=self.branch5(x)
                cls_logits.append(cls_5)
                reg_preds.append(reg_5)

                x=self.downsample_conv4(x)

                x=self.stage4_1(x)
                if self.BN:
                    x=self.bn4_1(x)
                x=self.relu4_1(x)
                cls_6,reg_6=self.branch6(x)
                cls_logits.append(cls_6)
                reg_preds.append(reg_6)

                x=self.stage4_2(x)
                if self.BN:
                    x=self.bn4_2(x)
                x=self.relu4_2(x)
                cls_7,reg_7=self.branch7(x)
                cls_logits.append(cls_7)
                reg_preds.append(reg_7)

                x=self.stage4_3(x)
                if self.BN:
                    x=self.bn4_3(x)
                x=self.relu4_3(x)
                cls_8,reg_8=self.branch8(x)
                cls_logits.append(cls_8)
                reg_preds.append(reg_8)

                return cls_logits,reg_preds

class lffd_config:
    class_num=2
    BN=True

    sizes=[(159,159),(159,159),(79,79),(79,79),(39,39),(19,19),(19,19),(19,19)]
    strides=[4,4,8,8,16,32,32,32]
    limit_sizes=[(10,15),(15,20),(20,40),(40,70),(70,110),(110,250),(250,400),(400,560)]
    RF_sizes=[55,71,111,143,223,383,511,639]

    score_threshold=0.5
    nms_iou_threshold=0.2
    max_detection_boxes_num=150

class TrainLFFD(nn.Module):
    def __init__(self,
                config=lffd_config):
        super().__init__()
        self.sizes=config.sizes
        self.strides=config.strides
        self.limit_sizes=config.limit_sizes
        self.RF_sizes=config.RF_sizes
        self.body=LFFD(BN=config.BN,class_num=config.class_num)
        self.target_gen=[BranchTarget(size,stride,limit_size,RF_size) for size,stride,limit_size,RF_size in zip(self.sizes,self.strides,self.limit_sizes,self.RF_sizes)]
    
    def forward(self,x):
        '''
        x list include [batch_imgs,batch_boxes,batch_classes]
        '''
        b_imgs=x[0]#[b,3,img_h,img_w]
        b_boxes=x[1]#[b,m,4]
        b_labels=x[2]#[b,m]
        cls_logits,reg_preds=self.body(b_imgs)
        reg_targets=[]
        cls_targets=[]
        for t in self.target_gen:
            r,c=t(b_boxes,b_labels)
            reg_targets.append(r)
            cls_targets.append(c)
        reg_losses=[]
        cls_losses=[]
        cls_loss_sum=0.
        reg_loss_sum=0.
        assert len(reg_targets)==len(reg_preds)
        assert len(cls_targets)==len(cls_logits)
        for r_p,r_t in zip(reg_preds,reg_targets):
            l=reg_loss(r_p,r_t)
            reg_losses.append(l)
            reg_loss_sum+=l
        for c_p,c_t in zip(cls_logits,cls_targets):
            l=cls_loss(c_p,c_t)
            cls_losses.append(l)
            cls_loss_sum+=l
        return cls_losses,reg_losses,cls_loss_sum,reg_loss_sum

class DetectionHead(nn.Module):
    def __init__(self,score_thr,nms_iou_thr,max_boxes,fmap_sizes,fmap_strides,RF_sizes):
        super().__init__()
        self.score_thr=score_thr
        self.nms_iou_thr=nms_iou_thr
        self.max_boxes=max_boxes
        self.sizes=fmap_sizes
        self.strides=fmap_strides
        self.RF_sizes=RF_sizes

    def forward(self,x):
        '''
        x list contain [cls_logits,reg_preds]
        cls_logits list contain all branch outputs
        reg_preds
        '''
        cls_logits,reg_preds=x[0],x[1]
        cls_logits,coords=self._reshape_cat_out(cls_logits)
        coords=coords.to(cls_logits.device)
        reg_preds,_=self._reshape_cat_out(reg_preds)#[b,?,4]
        cls_preds=cls_logits.softmax(dim=2)#[b,?,class_num+1]
        cls_scores,cls_classes=torch.max(cls_preds,dim=2)#[b,?]
        pred_boxes=self._coords2boxes(coords,reg_preds)#[b,?,4]

        # remove bg boxes
        batch_size=cls_classes.shape[0]
        fg_mask=cls_classes>0#[b,?]
        cls_classes_fg=[]
        cls_scores_fg=[]
        pred_boxes_fg=[]
        for b in range(batch_size):
            cls_scores_fg.append(cls_scores[b][fg_mask[b]])
            pred_boxes_fg.append(pred_boxes[b][fg_mask[b]])
            cls_classes_fg.append(cls_classes[b][fg_mask[b]])
        return self._post_process(cls_scores_fg,cls_classes_fg,pred_boxes_fg)
    
    def _post_process(self,cls_scores_fg,cls_classes,pred_boxes_fg):
        '''
        list 
        '''
        _cls_scores=[]
        _cls_classes=[]
        _boxes=[]
        for cls_scores_b,cls_classes_b,pred_boxes_b in zip(cls_scores_fg,cls_classes,pred_boxes_fg):
            max_num=min(self.max_boxes,len(cls_scores_b))
            topk_ind=torch.topk(cls_scores_b,max_num,dim=-1,largest=True,sorted=True)[1]#[?,]
            cls_scores_topk=cls_scores_b[topk_ind]#[?,]
            cls_classes_topk=cls_classes_b[topk_ind]#[?,]
            pred_boxes_topk=pred_boxes_b[topk_ind]#[?,4]
            
            score_thr_mask=cls_scores_topk>=self.score_thr
            cls_scores_thred=cls_scores_topk[score_thr_mask]#[??,]
            cls_classes_thred=cls_classes_topk[score_thr_mask]#[??,]
            pred_boxes_thred=pred_boxes_topk[score_thr_mask]#[??,4]
            nms_inds=self.box_nms(pred_boxes_thred,cls_scores_thred,self.nms_iou_thr)
            _cls_scores.append(cls_scores_thred[nms_inds])
            _cls_classes.append(cls_classes_thred[nms_inds])
            _boxes.append(pred_boxes_thred[nms_inds])
        return _cls_scores,_cls_classes,_boxes
    
    def _reshape_cat_out(self,x):
        '''
        Args
        x list contain 8 branch out [[b,c,h,w],...]
        Returns
        out [b,sum(_h*_w),c]
        coords [sum(_h*_w),2]
        '''
        b,c=x[0].shape[:2]
        out=[]
        coords=[]
        for i,x_branch in enumerate(x):
            x_branch=x_branch.permute(0,2,3,1)#[b,h,w,c]
            x_branch=x_branch.reshape((b,-1,c))#[b,h*w,c]
            coords_branch=coords_fmap2orig(self.sizes[i],self.strides[i])#[h*w,2]
            out.append(x_branch)
            coords.append(coords_branch)
        return torch.cat(out,dim=1),torch.cat(coords,dim=0)
    
    def _coords2boxes(self,coords,reg_preds):
        '''
        Args
        coords [sum(_h*_w),2]
        reg_preds [b,sum(_h*_w),4] 
        '''
        branch_num=[x[0]*x[1] for x in self.sizes]
        for i,n in enumerate(branch_num):
            reg_preds[:,sum(branch_num[:i]):sum(branch_num[:(i+1)]),:]*=(self.RF_sizes[i]/2.)
        x1y1=coords[None,:,:]-reg_preds[...,:2]
        x2y2=coords[None,:,:]+reg_preds[...,2:]#[b,sum(_h*_w),2]
        boxes=torch.cat([x1y1,x2y2],dim=2)#[batch_size,sum(_h*_w),4]
        return boxes

    def box_nms(self,boxes,scores,thr):
        '''
        boxes: [?,4]
        scores: [?]
        '''
        if boxes.shape[0]==0:
            return torch.zeros(0,device=boxes.device).long()
        assert boxes.shape[-1]==4
        x1,y1,x2,y2=boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
        areas=(x2-x1+1)*(y2-y1+1)
        order=scores.sort(0,descending=True)[1]
        keep=[]
        while order.numel()>0:
            if order.numel()==1:
                i=order.item()
                keep.append(i)
                break
            else:
                i=order[0].item()
                keep.append(i)
            
            xmin=x1[order[1:]].clamp(min=float(x1[i]))
            ymin=y1[order[1:]].clamp(min=float(y1[i]))
            xmax=x2[order[1:]].clamp(max=float(x2[i]))
            ymax=y2[order[1:]].clamp(max=float(y2[i]))
            inter=(xmax-xmin).clamp(min=0)*(ymax-ymin).clamp(min=0)
            iou=inter/(areas[i]+areas[order[1:]]-inter)
            idx=(iou<=thr).nonzero().squeeze()
            if idx.numel()==0:
                break
            order=order[idx+1]
        return torch.LongTensor(keep)

class ClipBoxes(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,batch_imgs,batch_boxes,orig_sizes):
        cliped_boxes=[]
        for b,boxes in enumerate(batch_boxes):
            boxes=boxes.clamp_(min=0)
            # h,w=batch_imgs.shape[2:]
            h,w=orig_sizes[b]#(2,)
            boxes[...,[0,2]]=boxes[...,[0,2]].clamp_(max=w-1)
            boxes[...,[1,3]]=boxes[...,[1,3]].clamp_(max=h-1)
            cliped_boxes.append(boxes)
        return cliped_boxes

class TestLFFD(nn.Module):
    def __init__(self,config=lffd_config):
        super().__init__()
        self.score_threshold=config.score_threshold
        self.nms_iou_threshold=config.nms_iou_threshold
        self.max_detection_boxes_num=config.max_detection_boxes_num
        self.sizes=config.sizes
        self.strides=config.strides
        self.RF_sizes=config.RF_sizes
        self.body=LFFD(BN=config.BN,class_num=config.class_num)
        self.detection_head=DetectionHead(self.score_threshold,
                                self.nms_iou_threshold,
                                self.max_detection_boxes_num,
                                self.sizes,
                                self.strides,
                                self.RF_sizes)
        self.clip_boxes=ClipBoxes()

    def forward(self,x,orig_sizes):
        out=self.body(x)
        scores,classes,boxes=self.detection_head(out)
        boxes=self.clip_boxes(x,boxes,orig_sizes)
        return scores,classes,boxes

if __name__ == "__main__":
    import time

    net=LFFD().cuda()
    x = torch.randn(1, 3, 640, 640).cuda()
    print(x.shape)
    with torch.no_grad():
        s_t=time.time()
        out=net(x)
        e_t=time.time()
        t=(e_t-s_t)*1000
        print("cost time ====> %.4f ms"%t)
        s_t=time.time()
        out=net(x)
        e_t=time.time()
        t=(e_t-s_t)*1000
        print("cost time ====> %.4f ms"%t)
        s_t=time.time()
        out=net(x)
        e_t=time.time()
        t=(e_t-s_t)*1000
        print("cost time ====> %.4f ms"%t)
    for i in out:
        for j in i:
            print(j.shape)

    '''
    torch.Size([1, class_num+1, 159, 159])
    torch.Size([1, class_num+1, 159, 159])
    torch.Size([1, class_num+1, 79, 79])
    torch.Size([1, class_num+1, 79, 79])
    torch.Size([1, class_num+1, 39, 39])
    torch.Size([1, class_num+1, 19, 19])
    torch.Size([1, class_num+1, 19, 19])
    torch.Size([1, class_num+1, 19, 19])
    torch.Size([1, 4, 159, 159])
    torch.Size([1, 4, 159, 159])
    torch.Size([1, 4, 79, 79])
    torch.Size([1, 4, 79, 79])
    torch.Size([1, 4, 39, 39])
    torch.Size([1, 4, 19, 19])
    torch.Size([1, 4, 19, 19])
    torch.Size([1, 4, 19, 19])
    '''



        






    