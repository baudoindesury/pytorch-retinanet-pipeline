#from inspect import get_annotations
from pickle import FALSE
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from os import listdir
from os.path import isfile, join
import os
import shutil
from tqdm import tqdm
from PIL import Image
import cv2


import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, \
    AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from retinanet import model


local_dataset_path = '/home/baudoin/data/zurich_data/annotated_images/images/'#'/home/baudoin/data/VisDrone/VisDrone2019-DET-val/images/'  #'/home/baudoin/data/VisDrone/VisDrone2019-DET-val/images/' #'/home/baudoin/data/zurich_data/annotated_images/images/' # #
cropped_images_dir = '/home/baudoin/data/zurich_data/cropped_images/'
output_viz_dir = '/home/baudoin/pytorch-retinanet-pipeline/pytorch-retinanet/output_images/'
classes_path = '/home/baudoin/pytorch-retinanet-pipeline/annotations/visdrone_anno/class_labels.csv'
model_path = '/home/baudoin/pytorch-retinanet-pipeline/results/training_results/20220209-193813/checkpoints/best_model_retinanet_csv.pt'
output_annotation_file = '/home/baudoin/pytorch-retinanet-pipeline/annotations/zurich_anno/anno.csv'#154.csv'
gt_annotations_file  = '/home/baudoin/pytorch-retinanet-pipeline/annotations/zurich_anno/original_anno.csv'#'/home/baudoin/data/zurich_data/annotated_images/retina_anno.csv' # ##img_154_anno.csv'

window_sizes = [600]#[500,600,700,800,900,1000,1100,1200]
ratio = [1]#[0.7,0.8,0.9,1,1.1,1.2,1.3]

KEEP_TILES = False
EVAL = True
COMPUTE_mAP = True
VISUALISATION = True
nb_tiles = 3
MinConfidence = 0.05
MaxDetections = 100
IOU_THRESHOLD = 0.5

def save_tiles_to_local_dir(tiles_data, folder_path, anno=False):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)
    tiles_paths = []
    for idx, w in enumerate(tiles_data):
        img_to_save_path = folder_path + '/' + 'tile_' + str(idx) + '.jpg'
        img_to_save = Image.fromarray(w[2])
        img_to_save.save(img_to_save_path)
        # break
        if anno:
            tiles_paths.append(img_to_save_path)

    if anno:
        with open(folder_path + '/' + 'anno.csv', 'w') as csv_file:
            for tile_path in tiles_paths:
                csv_file.write(tile_path + ',' +
                               ','.join(['', '', '', '']) + ',' + '' + '\n')


def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize[0]):
        for x in range(0, image.shape[1], stepSize[1]):
            #print(image.shape)
            #print(windowSize)
            #print(image[y:y + windowSize[0], x:x + windowSize[1]].shape)
            yield (x, y, image[y:y + windowSize[0], x:x + windowSize[1]])


def get_tiles_metadata(tiles_data):
    metadata = []
    for w in tiles_data:
        metadata.append([w[0], w[1], w[2].shape[0], w[2].shape[1]])
    return metadata


def detect_lables_retinanet_superpoint(tiles_anno_path, classes_path, model_path, MinConfidence=0.5, max_detections = 100):

    tiles_data = CSVDataset(train_file=tiles_anno_path, class_list=classes_path,
                            transform=transforms.Compose([Normalizer(), Resizer()]))


    #temp = tiles_data[0]
    #print(temp['annot'])
    #dataloader_val = tiles_data
    dataloader_val = DataLoader(
        tiles_data, num_workers=1, collate_fn=collater)#, batch_sampler=sampler_val)

    retinanet = model.resnet50(
        num_classes=tiles_data.num_classes(), pretrained=True)
    retinanet = torch.nn.DataParallel(retinanet)

    checkpoint = torch.load(model_path)
    retinanet.load_state_dict(checkpoint['model'])

        
    if torch.cuda.is_available():
        retinanet = retinanet.cuda()
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)
    
    retinanet.training = False
    retinanet.eval()

    unnormalize = UnNormalizer()
    detected_objects_per_image = []
    for idx, data in enumerate(dataloader_val):
        with torch.no_grad():
            if torch.cuda.is_available():
                scores, classification, transformed_anchors, output = retinanet([data['img'].cuda().float(), data['annot']])
            else:
                scores, classification, transformed_anchors, output = retinanet([data['img'].float(), data['annot']])
            
            detected_objects_per_tile = []
            idxs = np.where(scores.cpu() > MinConfidence)[0]
            if idxs.shape[0] > 0:
                # select those scores
                scores = scores.cpu()[idxs]
                # find the order with which to sort the scores
                scores_sort = np.argsort(-scores)[:max_detections]
                # select detections
                image_boxes      = transformed_anchors.cpu()[idxs[scores_sort], :]
                image_scores     = scores.cpu()[scores_sort]
                image_labels     = classification.cpu()[idxs[scores_sort]]

                if len(image_scores) > 1:
                    for j in range(len(image_scores)):
                        bbox = image_boxes[j]
                        obj = {}
                        obj['Label'] = tiles_data.labels[int(image_labels.numpy()[j])]
                        obj['x1'] = int(bbox[0]*1/data['scale'][0])
                        obj['y1'] = int(bbox[1]*1/data['scale'][0])
                        obj['x2'] = int(bbox[2]*1/data['scale'][0])
                        obj['y2'] = int(bbox[3]*1/data['scale'][0])
                        obj['score'] = image_scores[j].item()
                        detected_objects_per_tile.append(obj)
                else:
                    bbox = image_boxes
                    obj = {}
                    obj['Label'] = tiles_data.labels[int(image_labels.numpy())]
                    obj['x1'] = int(bbox[0]*1/data['scale'][0])
                    obj['y1'] = int(bbox[1]*1/data['scale'][0])
                    obj['x2'] = int(bbox[2]*1/data['scale'][0])
                    obj['y2'] = int(bbox[3]*1/data['scale'][0])
                    obj['score'] = image_scores[0].item()
                    detected_objects_per_tile.append(obj)            
            """
            for j in range(idxs.shape[0]):
                bbox = transformed_anchors[idxs[j], :]
                obj = {}
                obj['Label'] = tiles_data.labels[int(
                    classification[idxs[j]])]
                obj['x1'] = int(bbox[0]*1/data['scale'][0])
                obj['y1'] = int(bbox[1]*1/data['scale'][0])
                obj['x2'] = int(bbox[2]*1/data['scale'][0])
                obj['y2'] = int(bbox[3]*1/data['scale'][0])
                obj['score'] = scores.cpu()[idxs[j]].item()
                detected_objects_per_tile.append(obj)"""
        detected_objects_per_image.append(detected_objects_per_tile)

    return detected_objects_per_image


def convert_bbox_format(detected_objects_per_image, tiles_metadata):
    new_objects = []
    idx = 0
    for objects_per_tile in detected_objects_per_image:
        for obj in objects_per_tile:
            new_obj = {}
            new_obj['x1'] = tiles_metadata[idx][0] + int(obj['x1'])
            new_obj['y1'] = tiles_metadata[idx][1] + int(obj['y1'])
            new_obj['x2'] = tiles_metadata[idx][0] + int(obj['x2'])
            new_obj['y2'] = tiles_metadata[idx][1] + int(obj['y2'])
            new_obj['Label'] = obj['Label']
            new_obj['score'] = obj['score']
            new_objects.append(new_obj)
        idx += 1
    return new_objects


def save_annotations(detected_objects, img_path, output_annotation_file, multi_resolution_classes=None, save_score=False):
    with open(output_annotation_file, 'a') as csv_file:
        for obj in detected_objects:
            label = obj['Label']
            x1 = obj['x1']
            y1 = obj['y1']
            x2 = obj['x2']
            y2 = obj['y2']
            score = obj['score']
            
            if multi_resolution_classes is not None and label not in multi_resolution_classes:
                continue
            
            if save_score:
                if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
                    csv_file.write(img_path + ',' +
                                ','.join(['', '', '', '']) + ',' + '' + ',' + '' + '\n')
                else:
                    csv_file.write(
                        img_path + ',' + ','.join([str(x1), str(y1), str(x2), str(y2)]) + ',' + str(score) + ',' + label + '\n')
            else:
                if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
                    csv_file.write(img_path + ',' +
                                ','.join(['', '', '', '']) + ',' + '' + '\n')
                else:
                    csv_file.write(
                        img_path + ',' + ','.join([str(x1), str(y1), str(x2), str(y2)]) + ',' + label  + '\n')


def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10),
                cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 4)

def visualisation(img, detected_objects, output_viz_path):

    for obj in detected_objects:
        x1 = obj['x1']
        y1 = obj['y1']
        x2 = obj['x2']
        y2 = obj['y2']
        caption = obj['Label']
        draw_caption(img, (x1, y1, x2, y2), caption)
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1)

    cv2.imwrite(output_viz_path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def visualisation_gt_and_pred(gt_annotations_file, pred_annotation_file):
    from retinanet.csv_eval import _get_annotations, compute_overlap, _compute_ap

    gt_data = CSVDataset(train_file=gt_annotations_file, class_list=classes_path,
                            transform=transforms.Compose([Normalizer(), Resizer()]))
    pred_data = CSVDataset(train_file=pred_annotation_file, class_list=classes_path,
                            transform=transforms.Compose([Normalizer(), Resizer()]), read_score=True)
    gt, gt_img_names = _get_annotations(gt_data, return_img_names=True)
    pred, pred_img_names = _get_annotations(pred_data, return_img_names=True)

    index = [gt_img_names.index(i) for i in pred_img_names]
    gt = [gt[i] for i in index]

    for i, img_path in enumerate(pred_img_names):
        image = Image.open(img_path)
        img = np.asarray(image)
        detections_per_label = pred[i]
        annotations_per_label = gt[i]
        
        for label_idx, annotations in enumerate(annotations_per_label):
            for a in annotations:
                #draw_caption(img, (d[0], d[1], d[2], d[3]), gt_data.label_to_name(label_idx))
                cv2.rectangle(img, (int(a[0]), int(a[1])), (int(a[2]), int(a[3])), color=(0, 255, 0), thickness=1)
        for label_idx, detections in enumerate(detections_per_label):
            for d in detections:
                #draw_caption(img, (d[0], d[1], d[2], d[3]), pred_data.label_to_name(label_idx))
                if d[4] > 0.5:
                    cv2.rectangle(img, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), color=(255, 0, 0), thickness=2)

        output_viz_path = output_viz_dir + 'img_gt_pred_' + str(i) + '.jpg'
        cv2.imwrite(output_viz_path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def compute_mAP(gt_anno, pred_anno, classes_path, iou_threshold=0.5, verbose=1):
    from retinanet.csv_eval import _get_annotations, compute_overlap, _compute_ap

    gt_data = CSVDataset(train_file=gt_anno, class_list=classes_path,
                            transform=transforms.Compose([Normalizer(), Resizer()]))
    pred_data = CSVDataset(train_file=pred_anno, class_list=classes_path,
                            transform=transforms.Compose([Normalizer(), Resizer()]), read_score=True)
    gt, gt_img_names = _get_annotations(gt_data, return_img_names=True)
    pred, pred_img_names = _get_annotations(pred_data, return_img_names=True)



    index = [gt_img_names.index(name) for name in pred_img_names]
    gt = [gt[i] for i in index]


    #index = [pred_img_names.index(i) for i in gt_img_names]
    #pred = [pred[i] for i in index]
    
    average_precisions = {}
    for label in range(pred_data.num_classes()):
        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        for i in range(len(pred_data)):
            detections           = pred[i][label]
            annotations          = gt[i][label]
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

    mAP = {}
    for label in range(gt_data.num_classes()):
        label_name = gt_data.label_to_name(label)
        mAP[label_name] = average_precisions[label][0]
        if verbose:
            print('{}: {}'.format(label_name, average_precisions[label][0]))

    average_precisions = np.array([list(ele) for ele in average_precisions.values()])
    mAP = np.sum(average_precisions[:,0] * average_precisions[:,1]) / np.sum(average_precisions[:,1])
    print('average mAP = ', mAP)
    
    return mAP

def tile_detection_strategy(local_dataset_path, output_annotation_file, window_size=(600,600), step_size=(600,600), multi_resolution_classes = None):
    images_path = [local_dataset_path + f for f in listdir(
        local_dataset_path) if isfile(join(local_dataset_path, f))]
    
    if not multi_resolution_classes and os.path.exists(output_annotation_file):
        os.remove(output_annotation_file)

    
    for img_path in tqdm(images_path):
        image = Image.open(img_path)
        img = np.asarray(image)

        #window_H = int(img.shape[0]/nb_tiles)
        #window_W = int(img.shape[1]/nb_tiles)
        #step_size = int(0.9*min(window_H, window_W))

        window_H = window_size[0]
        window_W = window_size[1]
        #window_H = img.shape[0]
        #window_W = img.shape[1]
        #step_size = (img.shape[0], img.shape[1])

        tiles_data = sliding_window(img, step_size, (window_H, window_W))
        tiles_metadata = get_tiles_metadata(
            sliding_window(img, step_size, (window_H, window_W)))
        #print(tiles_metadata)

        full_img_name = Path(img_path).stem
        folder_path = cropped_images_dir + full_img_name

        save_tiles_to_local_dir(tiles_data, folder_path, anno=True)

        tiles_anno_path = folder_path + '/' + 'anno.csv'
        detected_objects_per_image = detect_lables_retinanet_superpoint(
            tiles_anno_path, classes_path, model_path, MinConfidence=MinConfidence, max_detections = MaxDetections)

        detected_objects = convert_bbox_format(
            detected_objects_per_image, tiles_metadata)

        save_annotations(detected_objects, img_path, output_annotation_file, multi_resolution_classes, save_score=True)

        #output_viz_path = output_viz_dir + 'detection_' + full_img_name + '.jpg'
        #visualisation(img, detected_objects, output_viz_path)

        if os.path.exists(folder_path) and not KEEP_TILES:
            shutil.rmtree(folder_path)



def main(args=None):
    mAP_car = []
    mAP_pedestrian = []
    mAP_truck = []
    mAP_van = []
    mAP_motor = []
    r=1
    for w in window_sizes:
        if EVAL:
            tile_detection_strategy(local_dataset_path, output_annotation_file, window_size=(w,int(r*w)), step_size=(w,int(r*w)))
        if COMPUTE_mAP:
            print('Compute mean Average Precision')
            mAP = compute_mAP(gt_annotations_file, output_annotation_file, classes_path, iou_threshold=IOU_THRESHOLD)
            mAP_car.append(mAP['car'])
            mAP_pedestrian.append(mAP['pedestrian'])
            mAP_truck.append(mAP['truck'])
            mAP_van.append(mAP['van'])
            mAP_motor.append(mAP['motor'])
        if VISUALISATION:
            visualisation_gt_and_pred(gt_annotations_file, output_annotation_file)
    
    if False:
        plt.figure()
        plt.plot(ratio, mAP_car, label='car')
        plt.plot(ratio, mAP_pedestrian, label='pedestrian')
        plt.plot(ratio, mAP_truck, label='truck')
        plt.plot(ratio, mAP_van, label='van')
        plt.plot(ratio, mAP_motor, label='motor')
        plt.legend()
        plt.ylim((0,1))
        plt.ylabel('mAP')
        plt.xlabel('window size')
        plt.savefig(output_viz_dir + 'ratio.jpg')

if __name__ == '__main__':
    main()
