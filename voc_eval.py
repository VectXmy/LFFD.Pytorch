import torch
from tqdm import tqdm
import numpy as np


def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []

    pred_boxes = outputs[2]
    pred_scores = outputs[0]
    pred_labels = outputs[1]

    true_positives = np.zeros(pred_boxes.shape[0])

    target_labels = targets[0] if len(targets[0]) else []
    target_boxes=targets[1]

    if len(target_labels):
        detected_boxes = []

        for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
            # # If targets are found break
            # if len(detected_boxes) == len(target_labels):
            #     break
            # Ignore if label is not one of the target labels
            if pred_label not in target_labels:
                continue
            iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
            if iou >= iou_threshold and box_index not in detected_boxes:
                true_positives[pred_i] = 1
                detected_boxes += [box_index]
    batch_metrics.append([true_positives, pred_scores.cpu().numpy(), pred_labels.cpu().numpy()])
    return batch_metrics

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def evaluate(model, dataloader, iou_thres):
    model.eval()

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (img,boxes,classes,scale,orig_size) in enumerate(tqdm(dataloader, desc="Detecting objects")):

        with torch.no_grad():
            scale=scale[0]
            orig_size=[np.array(orig_size[0])*scale,]
            out=model(img.cuda(),orig_size)

        outputs=[out[0][0],out[1][0],out[2][0]/scale]#[scores,classes,boxes]
        targets=[classes[0].to(outputs[0].device),boxes[0].to(outputs[0].device)/scale]#[classes,boxes]
        labels+=classes[0].numpy().tolist()

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class

if __name__ == "__main__":
    from model.baseline import TestLFFD
    from dataset.vocdataset import VOCDataset
    import argparse,os

    parser=argparse.ArgumentParser()
    parser.add_argument("--weights",required=True,type=str)
    args=parser.parse_args()
    weights_path=args.weights

    if not os.path.exists(weights_path):
        raise TypeError("not exit %s"%weights_path)

    class lffd_config:        
        class_num=2
        BN=True

        sizes=[(159,159),(159,159),(79,79),(79,79),(39,39),(19,19),(19,19),(19,19)]
        strides=[4,4,8,8,16,32,32,32]
        limit_sizes=[(10,15),(15,20),(20,40),(40,70),(70,110),(110,250),(250,400),(400,560)]
        RF_sizes=[55,71,111,143,223,383,511,639]

        score_threshold=0.05
        nms_iou_threshold=0.2
        max_detection_boxes_num=150

    eval_dataset=VOCDataset("/home/xht/dataset/VOC2028",split="test")
    print("INFO===>eval dataset has %d imgs"%len(eval_dataset))
    eval_loader=torch.utils.data.DataLoader(eval_dataset,batch_size=1,shuffle=False,collate_fn=eval_dataset.collate_fn)
    class_names=eval_dataset.CLASSES_NAME
    model=TestLFFD(config=lffd_config)
    model.load_state_dict(torch.load(weights_path,map_location=torch.device('cpu')))
    model=model.cuda().eval()
    print("===>success loading model")

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        eval_loader,
        iou_thres=0.5,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")