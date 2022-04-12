from __future__ import print_function

import numpy as np
import json
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Inmport superpoint
from retinanet.networks.superpoint_pytorch import SuperPointFrontend
from retinanet.networks.superpoint_utils import post_processing_superpoint_detector, plot_superpoint_keypoints
from retinanet.losses import *




def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def _compute_ap(recall, precision):
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
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(dataset, retinanet, superpoint, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the retinanet using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
    # Arguments
        dataset         : The generator used to run images through the retinanet.
        retinanet           : The retinanet to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(dataset.num_classes())] for j in range(len(dataset))]

    retinanet.eval()
    
    with torch.no_grad():
    
        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']
            # run network
            if torch.cuda.is_available():
                scores, labels, boxes, output = retinanet([data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0), data['annot'].unsqueeze(dim=0)])
            else:
                scores, labels, boxes, output = retinanet([data['img'].permute(2, 0, 1).float().unsqueeze(dim=0), data['annot'].unsqueeze(dim=0)])
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            boxes  = boxes.cpu().numpy()

            

            # SuperPoint output Tensors
            output_desc = output['desc'].type(torch.FloatTensor)#.to(device)
            output_semi = output['semi'].type(torch.FloatTensor)#.to(device)
            
            # Post-processing superpoint detector
            img_original = data['img_gray'].numpy()*255
            H, W = img_original.shape[0], img_original.shape[1]
            superpoint_keypoints,_ = post_processing_superpoint_detector(F.softmax(output_semi.squeeze(), dim=1), H, W)
            #plot_superpoint_keypoints(img_original, pts, title='student')

            # SuperPoint label with teacher model
            output_superpoint = superpoint.run(data['img_gray'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
            desc_teacher = torch.from_numpy(output_superpoint['local_descriptor_map']).type(torch.FloatTensor)#.to(device)
            dect_teacher = torch.from_numpy(output_superpoint['dense_scores']).type(torch.FloatTensor)#.to(device)
            
            # Teacher Model - Post-processing superpoint detector
            #pts,_ = post_processing_superpoint_detector(dect_teacher, H, W)
            #plot_superpoint_keypoints(img_original, pts, title='teacher')

            # Compute SuperPoint Losses
            desc_l = descriptor_local_loss(output_desc, desc_teacher)
            detc_l = detector_loss(output_semi, dect_teacher)
            
            # Retina Losses
            classification_loss, regression_loss = output['focalLoss'][0], output['focalLoss'][1] 
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            
            # Compute total loss
            loss_retina = classification_loss + regression_loss
            loss_superpoint = desc_l + detc_l
            loss = loss_retina + loss_superpoint

            losses = {'loss_superpoint': loss_superpoint, 'loss_retina': loss_retina, 'total_loss': loss}

            # correct boxes for image scale
            boxes /= scale

            # select indices which have a score above the threshold
            indices = np.where(scores > score_threshold)[0]
            #print('indices = ', indices.shape)
            if indices.shape[0] > 0:
                # select those scores
                scores = scores[indices]

                # find the order with which to sort the scores
                scores_sort = np.argsort(-scores)[:max_detections]

                # select detections
                image_boxes      = boxes[indices[scores_sort], :]
                image_scores     = scores[scores_sort]
                image_labels     = labels[indices[scores_sort]]
                image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = image_detections[image_detections[:, -1] == label, :-1]
            else:
                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = np.zeros((0, 5))

            print('{}/{}'.format(index + 1, len(dataset)), end='\r')

    return all_detections, losses, superpoint_keypoints


def _get_annotations(generator, return_img_names = False):
    """ Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]
    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(len(generator))]

    for i in range(len(generator)):
        # load the annotations
        annotations = generator.load_annotations(i) 

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label] = annotations[annotations[:, -1] == label, :-1].copy()

        print('{}/{}'.format(i + 1, len(generator)), end='\r')

    if return_img_names:
        return all_annotations, generator.image_names

    return all_annotations


def layer_to_img(semi):
    semi = semi.squeeze()[1:].unsqueeze(0)
    pixel_shuffle = torch.nn.PixelShuffle(8)
    output = pixel_shuffle(semi)
    return output.squeeze()

import cv2
def draw_keypoints(img, corners, color):
    keypoints = [cv2.KeyPoint(c[1], c[0], 1) for c in np.stack(corners).T]
    return cv2.drawKeypoints(img.astype(np.uint8), keypoints, None, color=color)

def evaluate(
    generator,
    retinanet,
    superpoint,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    save_path=None
):
    """ Evaluate a given dataset using a given retinanet.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        retinanet           : The retinanet to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save precision recall curve of each label.
    # Returns
        A dict mapping class names to mAP scores.
    """



    # gather all detections and annotations

    all_detections, losses, superpoint_keypoints     = _get_detections(generator, retinanet, superpoint, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
    all_annotations    = _get_annotations(generator)

    average_precisions = {}

    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0
        for i in range(len(generator)):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []
            #print('len d =',len(detections))
            for d in detections:
                
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        #print('TP = ',true_positives)
        #print('n_anno = ', num_annotations)
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision  = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations


    print('\nmAP:')
    for label in range(generator.num_classes()):
        label_name = generator.label_to_name(label)
        print('{} ({} instances): {}'.format(label_name, int(average_precisions[label][1]), average_precisions[label][0])) 
        #print("Precision: ",precision[-1])
        #print("Recall: ",recall[-1])
        #print("Precision: ",precision[-1])
        #print("Recall: ",recall[-1])
        
        if save_path!=None:
            plt.plot(recall,precision)
            plt.xlabel('Recall') 
            plt.ylabel('Precision') 
            plt.title('Precision Recall curve') 
            plt.savefig(save_path+'/'+label_name+'_precision_recall.jpg')

    #saving_path = '/home/baudoin/pytorch-retinanet-pipeline/results/retinanet_eval_data/'
    #np.save(saving_path + 'model_mix_tuned',average_precisions)

    return average_precisions, losses


"""
def post_processing_superpoint_detector(semi, H, W):
    
    cell = 8
    border_remove = 4
    conf_thresh = 0.015

    # --- Process points.
    dense = semi.squeeze().numpy()
    
    #dense = np.exp(semi) # Softmax.
    dense = dense / (np.sum(dense, axis=0)+.00001) # Should sum to 1.
    # Remove dustbin.
    nodust = dense[:-1, :, :]
    # Reshape to get full resolution heatmap.
    Hc = int(H / cell)
    Wc = int(W / cell)
    nodust = nodust.transpose(1, 2, 0)
    heatmap = np.reshape(nodust, [Hc, Wc, cell, cell])
    heatmap = np.transpose(heatmap, [0, 2, 1, 3])
    heatmap = np.reshape(heatmap, [Hc*cell, Wc*cell])
    xs, ys = np.where(heatmap >= conf_thresh) # Confidence threshold.
    if len(xs) == 0:
      return np.zeros((3, 0))
    pts = np.zeros((3, len(xs))) # Populate point data sized 3xN.
    pts[0, :] = ys
    pts[1, :] = xs
    pts[2, :] = heatmap[xs, ys]
    #pts, _ = nms_fast(pts, H, W, dist_thresh=nms_dist) # Apply NMS.
    inds = np.argsort(pts[2,:])
    pts = pts[:,inds[::-1]] # Sort by confidence.
    # Remove points along border.
    bord = border_remove
    toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W-bord))
    toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H-bord))
    toremove = np.logical_or(toremoveW, toremoveH)
    pts = pts[:, ~toremove]

    return pts


def plot_superpoint_keypoints(img_gray, pts, title='keypoints'):
    out2 = (np.dstack((img_gray, img_gray, img_gray)) * 255.).astype('uint8')
    for pt in pts.T:
        pt1 = (int(round(pt[0])), int(round(pt[1])))
        cv2.circle(out2, pt1, 3, (0, 255, 0), -1, lineType=16)
        
    plt.figure(figsize=(16,8))
    plt.imshow(out2)
    plt.savefig(title + '.jpeg')
    plt.show()
"""
