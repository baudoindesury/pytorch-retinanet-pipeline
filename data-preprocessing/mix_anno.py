
import os
import random
import argparse

from parso import parse


def mix_annotations(dataset_1_path, dataset_2_path, saving_path):
    #synthetic_anno_path = '/home/baudoin/pytorch-superpoint/anno.csv'
    #dataset_anno_path = '/home/baudoin/pytorch-superpoint/anno.csv'

    file_1 = open(dataset_1_path, "r")
    lines_1 = file_1.readlines()
    file_1.close()

    file_2 = open(dataset_2_path, "r")
    lines_2 = file_2.readlines()
    file_2.close()

    #n = min(len(lines_1), len(lines_2))
    li = lines_1 + lines_2

    random.shuffle(li)

    fid = open(saving_path, "w")
    fid.writelines(li)
    fid.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--anno_1', help='Path of first dataset annotations')
    parser.add_argument('--anno_2', help='Path of second dataset annotations')
    parser.add_argument('--output_dir', help='output directory where to save mixed annotations', default = 'anno_mix.csv')
    parser = parser.parse_args()
    #synthetic_anno_path = '/home/baudoin/pytorch-superpoint/anno.csv'
    #dataset_anno_path = '/home/baudoin/pytorch-superpoint/anno.csv'
    #saving_path = '/home/baudoin/pytorch-retinanet-pipeline/synthetic_shapes_anno/anno_mix_'+ set +'.csv'
    mix_annotations(parser.anno_1, parser.anno_2, parser.output_dir)