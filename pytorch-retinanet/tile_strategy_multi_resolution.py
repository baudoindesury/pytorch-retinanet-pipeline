from tile_strategy import tile_detection_strategy, compute_mAP, visualisation_gt_and_pred
import matplotlib.pyplot as plt
import os

local_dataset_path = '/home/baudoin/data/VisDrone/VisDrone2019-DET-val/images/'  #'/home/baudoin/data/VisDrone/VisDrone2019-DET-val/images/' #'/home/baudoin/data/zurich_data/annotated_images/images/' # #
cropped_images_dir = '/home/baudoin/data/zurich_data/cropped_images/'
output_viz_dir = '/home/baudoin/pytorch-retinanet-pipeline/pytorch-retinanet/output_images/'
classes_path = '/home/baudoin/pytorch-retinanet-pipeline/annotations/visdrone_anno/class_labels.csv'
model_path = '/home/baudoin/pytorch-retinanet-pipeline/results/training_results/20220209-193813/checkpoints/best_model_retinanet_csv.pt'
output_annotation_file = '/home/baudoin/pytorch-retinanet-pipeline/annotations/visdrone_anno/tiles_multi_anno_dev.csv'#154.csv'
gt_annotations_file  = '/home/baudoin/pytorch-retinanet-pipeline/annotations/visdrone_anno/annotations_dev.csv'#'/home/baudoin/data/zurich_data/annotated_images/retina_anno.csv' # ##img_154_anno.csv'

"""
local_dataset_path = '/home/baudoin/data/zurich_data/annotated_images/images/'#'/home/baudoin/data/VisDrone/VisDrone2019-DET-val/images/'  #'/home/baudoin/data/VisDrone/VisDrone2019-DET-val/images/' #'/home/baudoin/data/zurich_data/annotated_images/images/' # #
cropped_images_dir = '/home/baudoin/data/zurich_data/cropped_images/'
output_viz_dir = '/home/baudoin/pytorch-retinanet-pipeline/pytorch-retinanet/output_images/'
classes_path = '/home/baudoin/pytorch-retinanet-pipeline/annotations/visdrone_anno/class_labels.csv'
model_path = '/home/baudoin/pytorch-retinanet-pipeline/results/training_results/20220209-193813/checkpoints/best_model_retinanet_csv.pt'
output_annotation_file = '/home/baudoin/pytorch-retinanet-pipeline/annotations/zurich_anno/anno.csv'#154.csv'
gt_annotations_file  = '/home/baudoin/pytorch-retinanet-pipeline/annotations/zurich_anno/original_anno.csv'#'/home/baudoin/data/zurich_data/annotated_images/retina_anno.csv' # ##img_154_anno.csv'
"""

KEEP_TILES = False
EVAL = True
COMPUTE_mAP = True
VISUALISATION = True
nb_tiles = 3
MinConfidence = 0.05
MaxDetections = 100
IOU_THRESHOLD = 0.5

# Visdrone
window_sizes = {400: ['car', 'pedestrian', 'people'], 600: ['van', 'motor'], 800: ['truck', 'bus', 'bicycle', 'tricycle']}
# Zurich
#window_sizes = {400: ['pedestrian', 'people'], 600: ['car', 'motor', 'awning-tricycle'], 800: ['truck', 'bus', 'bicycle','van'], 1000:['tricycle', 'bus', 'motor']}


def main(args=None):
    r=1
    if EVAL:
        if os.path.exists(output_annotation_file):
            os.remove(output_annotation_file)
        for w in window_sizes.keys():
            tile_detection_strategy(local_dataset_path, output_annotation_file, window_size=(w,int(r*w)), step_size=(w,int(r*w)), multi_resolution_classes = window_sizes[w])
        
    if COMPUTE_mAP:
        print('Compute mean Average Precision')
        mAP = compute_mAP(gt_annotations_file, output_annotation_file, classes_path, iou_threshold=IOU_THRESHOLD)
    if VISUALISATION:
        visualisation_gt_and_pred(gt_annotations_file, output_annotation_file)

    if False:
        plt.figure(figsize=(16,8))
        plt.plot(window_sizes, mAP_car, label='car', marker='o')
        plt.plot(window_sizes, mAP_pedestrian, label='pedestrian', marker='o')
        plt.plot(window_sizes, mAP_truck, label='truck', marker='o')
        plt.plot(window_sizes, mAP_van, label='van', marker='o')
        plt.plot(window_sizes, mAP_motor, label='motor', marker='o')
        plt.plot(window_sizes, mAP_people, label='people', marker='o')
        plt.plot(window_sizes, mAP_bicycle, label='bicycle', marker='o')
        plt.plot(window_sizes, mAP_tricycle, label='tricycle', marker='o')
        plt.plot(window_sizes, mAP_bus, label='bus', marker='o')
        plt.legend()
        plt.ylim((0,0.8))
        plt.ylabel('mAP')
        plt.xlabel('window size')
        plt.savefig(output_viz_dir + 'mAP_windowSize.jpg')

if __name__ == '__main__':
    main()
