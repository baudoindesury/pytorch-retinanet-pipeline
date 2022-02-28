import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse

import sys
import cv2

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from retinanet.networks.superpoint_pytorch import SuperPointFrontend


from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
	UnNormalizer, Normalizer
from retinanet.networks.superpoint_utils import post_processing_superpoint, plot_superpoint_keypoints, plot_superpoint_keypoints_gt_and_pred, draw_matches


# load checkpoint (to remove)
from retinanet import model

# GPU use
import GPUtilext

import matplotlib.pyplot as plt



assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
	parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

	parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
	parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

	parser.add_argument('--model', help='Path to model (.pt) file.')
	
	parser.add_argument('--ground_truth', dest='ground_truth', action='store_true')
	parser.add_argument('--no-ground_truth', dest='ground_truth', action='store_false')
	parser.set_defaults(ground_truth=False)

	parser.add_argument('--superpoint', help ='If flag set, save superpoint visualisations', default= False)
	parser.add_argument('--output_path', help = 'Output path to save visualisations', default = 'output_images/')

	parser = parser.parse_args(args)

	dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))

	sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
	dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)
	
	retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)
	retinanet = torch.nn.DataParallel(retinanet)
    
    # Modify load checkpoints
	checkpoint = torch.load(parser.model)
	retinanet.load_state_dict(checkpoint['model'])
	#retinanet = torch.load(parser.model)

	use_gpu = True

	if use_gpu:
		if torch.cuda.is_available():
			retinanet = retinanet.cuda()

	if torch.cuda.is_available():
		retinanet = torch.nn.DataParallel(retinanet).cuda()
	else:
		retinanet = torch.nn.DataParallel(retinanet)

	retinanet.eval()
	superpoint = SuperPointFrontend(project_root='') #Modify project_root
	unnormalize = UnNormalizer()
	threashold_student = 0.15
	threashold_teacher = 0.015
	for idx, data in enumerate(tqdm(dataloader_val)):
		with torch.no_grad():
			if torch.cuda.is_available():
				scores, classification, transformed_anchors, output = retinanet([data['img'].cuda().float(), data['annot']])
			else:
				scores, classification, transformed_anchors, output = retinanet([data['img'].float(), data['annot']])

			img_gray = np.squeeze(data['img_gray'].numpy()*255)
			H, W = img_gray.shape[0], img_gray.shape[1]

			# SuperPoint label with teacher model
			output_superpoint = superpoint.run(data['img_gray'].float())
			dect_teacher = torch.from_numpy(output_superpoint['dense_scores']).type(torch.FloatTensor)
			desc_teacher = torch.from_numpy(output_superpoint['local_descriptor_map']).type(torch.FloatTensor)
			pts_teacher, desc_teacher, heatmap  = post_processing_superpoint(dect_teacher, desc_teacher, H, W, conf_thresh=threashold_teacher, border_remove=50)
			keypoints_teacher, desc_teacher = extract_superpoint_keypoints_and_descriptors(pts_teacher, desc_teacher)
			#print('shape keypoints_teacher = ' ,np.shape(keypoints_teacher))
			#print('shape desc_teacher = ',np.shape(desc_teacher))
			# SuperPoint output Tensors
			output_desc = output['desc'].type(torch.FloatTensor)
			output_semi = output['semi'].type(torch.FloatTensor)
				
			pts_student, desc_student, heatmap= post_processing_superpoint(F.softmax(output_semi.squeeze(), dim=1), output_desc, H, W, conf_thresh=threashold_student, border_remove=50)
			keypoints_student, desc_student = extract_superpoint_keypoints_and_descriptors(pts_student, desc_student)
			#print(keypoints_student)
			#print('shape keypoints_student = ' ,np.shape(keypoints_student))
			#print('shape desc_student = ',np.shape(desc_student))

			m_kp1, m_kp2, matches = match_descriptors(keypoints_teacher, desc_teacher, keypoints_student, desc_student)

			H, inliers = compute_homography(m_kp1, m_kp2)
			#print('H = ', H)
			#print('inliers = ', inliers)

			# Draw SuperPoint matches
			matches = np.array(matches)[inliers.astype(bool)].tolist()
			img = (np.dstack((img_gray, img_gray, img_gray)) * 255.).astype('uint8')
			matched_img = cv2.drawMatches(img, keypoints_teacher, img, keypoints_student, matches,
										None, matchColor=(0, 255, 0),
										singlePointColor=(0, 0, 255))
			path = '/home/baudoin/pytorch-retinanet-pipeline/pytorch-retinanet/output_images/match'
			cv2.imwrite(path + '.jpeg', matched_img)

			break


def extract_superpoint_keypoints_and_descriptors(keypoint_map, descriptor_map,
													keep_k_points=1000):

	def select_k_best(points, k):
		""" Select the k most probable points (and strip their proba).
		points has shape (num_points, 3) where the last coordinate is the proba. """
		sorted_prob = points[points[:, 2].argsort(), :2]
		start = min(k, points.shape[0])
		return sorted_prob[-start:, :]

	keypoints = keypoint_map.T
	
	keypoints = select_k_best(keypoints, keep_k_points)
	keypoints = keypoints.astype(int)
	print(keypoints)
	
	
	# Get descriptors for keypoints
	desc = descriptor_map.T

	# Convert from just pts to cv2.KeyPoints
	keypoints = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in keypoints]

	return keypoints, desc

def match_descriptors(kp1, desc1, kp2, desc2):
	# Match the keypoints with the warped_keypoints with nearest neighbor search
	bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
	matches = bf.match(desc1, desc2)
	matches_idx = np.array([m.queryIdx for m in matches])
	m_kp1 = [kp1[idx] for idx in matches_idx]
	matches_idx = np.array([m.trainIdx for m in matches])
	m_kp2 = [kp2[idx] for idx in matches_idx]
	return m_kp1, m_kp2, matches

def compute_homography(matched_kp1, matched_kp2):
    matched_pts1 = cv2.KeyPoint_convert(matched_kp1)
    matched_pts2 = cv2.KeyPoint_convert(matched_kp2)

    # Estimate the homography between the matches using RANSAC
    H, inliers = cv2.findHomography(matched_pts1[:, [1, 0]],
                                    matched_pts2[:, [1, 0]],
                                    cv2.RANSAC)
    inliers = inliers.flatten()
    return H, inliers


if __name__ == '__main__':
 main()