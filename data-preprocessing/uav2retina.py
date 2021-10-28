# code adapted from mmtracking convert datasets tools

import argparse
import os
import os.path as osp
from collections import defaultdict
import re


import mmcv
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description='UAV123 dataset to retinanet Video format')
    parser.add_argument(
        '-i',
        '--input',
        help='root directory of UAV123 dataset',
    )
    parser.add_argument(
        '-o',
        '--output',
        help='directory to save retinanet formatted label file',
    )
    return parser.parse_args()

def convert_uav123(uav123, ann_dir, save_dir):
    """Convert trackingnet dataset to COCO style.
    Args:
        uav123 (dict): The converted retinanet style annotations.
        ann_dir (str): The path of trackingnet test dataset
        save_dir (str): The path to save `uav123`.
    """
    # The format of each line in "uav_info123.txt" is
    # "anno_name,anno_path,video_path,start_frame,end_frame"
    info_path = osp.join(os.path.abspath(''), 'UAV123/uav123_info.txt')
    uav_info = mmcv.list_from_file(info_path)[1:]

    records = dict(vid_id=1, img_id=1, ann_id=1, global_instance_id=1)

    # Define labels _file.of UAV123 dataset
    labels = {'bike':1, 'bird':2, 'boat':3, 'building':4, 'car':5 , 'group':6, 'person':7, 'truck':8, 'uav':9, 'wakeboard':10}
    labels_cond = ''
    for label in list(labels.keys()):
        labels_cond += label + '|' 
    labels_cond = labels_cond[:-1]

    print(len(uav_info))

    with open(save_dir, 'w') as csv_file:
        for info in tqdm(uav_info):
            anno_name, anno_path, video_path, start_frame, end_frame = info.split(
                ',')
            start_frame = int(start_frame)
            end_frame = int(end_frame)
            # video_name is not the same as anno_name since one video may have
            # several fragments.
            # Example: video_name: "bird"   anno_name: "bird_1"
            video_name = video_path.split('/')[-1]
            video = dict(id=records['vid_id'], name=video_name)
            uav123['videos'].append(video)

            gt_bboxes = mmcv.list_from_file(osp.join(ann_dir, anno_path))
            assert len(gt_bboxes) == end_frame - start_frame + 1

            img = mmcv.imread(
                osp.join(ann_dir, video_path, '%06d.jpg' % (start_frame)))
            
            for frame_id, src_frame_id in enumerate(
                    range(start_frame, end_frame + 1)):
                file_name = osp.join(video_name, '%06d.jpg' % (src_frame_id))

                if 'NaN' in gt_bboxes[frame_id]:
                    x1 = y1 = x2 = y2 = 0
                    label = ''
                else:
                    x1, y1, w, h = gt_bboxes[frame_id].split(',')
                    x2 = int(x1) + int(w)
                    y2 = int(y1) + int(h)
                    label = re.findall(labels_cond, video_name)[0]
                
                csv_file.write(file_name + ',' + ','.join([str(x1),str(y1),str(x2), str(y2)]) + ',' + label + '\n')

                break


    print('New RetinaNet annotations saved at :' + save_dir)

def main():
    args = parse_args()
    uav123 = defaultdict(list)
    convert_uav123(uav123, args.input, args.output)

if __name__ == '__main__':
    main()

#input = '/media/baudoin/SSDUbuntu/data/UAV123'
#output = '/home/baudoin/new_uav123.csv'