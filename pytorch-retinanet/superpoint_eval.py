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
from retinanet.networks.superpoint_utils import post_processing_superpoint_detector, plot_superpoint_keypoints, plot_superpoint_keypoints_gt_and_pred, draw_matches


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

	#threshold_list = np.linspace(0.1,0.3,9)
	threshold_list = [0.15]
	print('Threasholds = ', threshold_list)

	tp, fp, prob, matches, n_gt = [], [], [], [], 0
	localization_error = []
	k = 100
	best_k_pts = False
	threashold_teacher = [0.015] 

	for t_teacher in threashold_teacher:
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

				pts_teacher, heatmap  = post_processing_superpoint_detector(dect_teacher, H, W, conf_thresh=t_teacher, border_remove=50)
				if best_k_pts :
					pts_teacher = select_k_best(pts_teacher, k)

				# SuperPoint output Tensors
				output_desc = output['desc'].type(torch.FloatTensor)
				output_semi = output['semi'].type(torch.FloatTensor)

				for thresh in threshold_list:
					
					pts_student, heatmap= post_processing_superpoint_detector(F.softmax(output_semi.squeeze(), dim=1), H, W, conf_thresh=thresh, border_remove=50)
					if best_k_pts :
						pts_student = select_k_best(pts_student, k)
					t, f, p, n, m = compute_tp_fp(pts_teacher, pts_student, distance_thresh=2, simplified=True)
					tp.append(t)
					fp.append(f)
					prob.append(p)
					matches.append(m)
					n_gt+=n
					#print('\nThresh = ', thresh)
					#print('n_gt = ' + str(n) + ' | n_pred = ' + str(len(pts_student[0])))
					#print('TP = ' + str(np.sum(t)) + ' \t | FP = ' + str(np.sum(f)))

					correct_dist = loc_error_per_image(pts_teacher, pts_student, distance_thresh=2)
					localization_error.append(correct_dist)
					#if np.abs(thresh - 0.15) < 0.001:
					path = '/home/baudoin/pytorch-retinanet-pipeline/pytorch-retinanet/output_images/superpoint'
					plot_superpoint_keypoints_gt_and_pred(img_gray, pts_teacher, pts_student, title=path + str(idx))

					#print('correct_dist = ', correct_dist)
				#print('loc_error = ', np.mean(np.concatenate(localization_error)))

				path = '/home/baudoin/pytorch-retinanet-pipeline/pytorch-retinanet/output_images/superpoint'
				#plot_superpoint_keypoints_gt_and_pred(img_gray, pts_teacher, pts_student, title=path + str(idx))
				draw_matches(img_gray, m, lw = 0.2, if_fig=False, filename= path + '_matches_' + str(idx))
		
		# Compute precision and Recall (To verify)
		precision, recall, _ = compute_pr(tp, fp, prob, n_gt)
		#np.save('/home/baudoin/pytorch-retinanet-pipeline/PR_data/precision_uav_' + str(t_teacher) , precision)
		#np.save('/home/baudoin/pytorch-retinanet-pipeline/PR_data/recall_uav_'  + str(t_teacher), recall)

		plt.figure()
		plt.plot(recall, precision)
		plt.xlabel('recall')
		plt.ylabel('precision')
		plt.savefig('/home/baudoin/pytorch-retinanet-pipeline/PR.png')


def compute_tp_fp(pts_gt, pts_pred, remove_zero=1e-4, distance_thresh=2, simplified=False):
	"""
	Compute the true and false positive rates.
	"""
	gt = pts_gt[0:2].T
	n_gt = len(gt)
	prob = pts_pred[2]
	pred = pts_pred[0:2].T

	# When several detections match the same ground truth point, only pick
	# the one with the highest score  (the others are false positive)
	sort_idx = np.argsort(prob)[::-1]
	prob = prob[sort_idx]
	pred = pred[sort_idx]

	
	diff = np.expand_dims(pred, axis=1) - np.expand_dims(gt, axis=0)
	dist = np.linalg.norm(diff, axis=-1)
	matches = np.less_equal(dist, distance_thresh)
	
	idx_matches = matches.nonzero()
	key_1 = pred[idx_matches[0]]
	key_2 = gt[idx_matches[1]]
	match_pairs = np.c_[key_1,key_2]
	
	tp = []
	matched = np.zeros(len(gt))
	for m in matches:
		correct = np.any(m)
		if correct:
			gt_idx = np.argmax(m)
			tp.append(not matched[gt_idx])
			matched[gt_idx] = 1
		else:
			tp.append(False)
	tp = np.array(tp, bool)
	if simplified:
		tp = np.any(matches, axis=1)  # keeps multiple matches for the same gt point
		n_gt = np.sum(np.minimum(np.sum(matches, axis=0), 1))  # buggy
	fp = np.logical_not(tp)
	return tp, fp, prob, n_gt, match_pairs
			
def compute_pr(tp, fp, prob, n_gt):
	"""
	Compute precision and recall.
	"""

	tp = np.concatenate(tp)
	fp = np.concatenate(fp)
	prob = np.concatenate(prob)

	# Sort in descending order of confidence
	sort_idx = np.argsort(prob)[::-1]
	tp = tp[sort_idx]
	fp = fp[sort_idx]
	prob = prob[sort_idx]

	# Cumulative
	tp_cum = np.cumsum(tp)
	print(np.shape(tp_cum))
	fp_cum = np.cumsum(fp)
	recall = div0(tp_cum, n_gt)
	precision = div0(tp_cum, tp_cum + fp_cum)
	recall = np.concatenate([[0], recall, [1]])
	precision = np.concatenate([[0], precision, [0]])
	precision = np.maximum.accumulate(precision[::-1])[::-1]
	return precision, recall, prob

def div0(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        idx = ~np.isfinite(c)
        c[idx] = np.where(a[idx] == 0, 1, 0)  # -inf inf NaN
    return c

def loc_error_per_image(pts_gt, pts_pred, distance_thresh=2):
	# Read data
	gt = pts_gt[0:2].T
	n_gt = len(gt)
	prob = pts_pred[2]
	pred = pts_pred[0:2].T

	if not len(gt) or not len(pred):
		return []

	diff = np.expand_dims(pred, axis=1) - np.expand_dims(gt, axis=0)
	dist = np.linalg.norm(diff, axis=-1)
	dist = np.min(dist, axis=1)
	correct_dist = dist[np.less_equal(dist, distance_thresh)]
	return correct_dist

def select_k_best(points, k):
	""" Select the k most probable points (and strip their proba).
	points has shape (num_points, 3) where the last coordinate is the proba. """
	points = points.T
	sorted_prob = points
	if points.shape[1] > 2:
		sorted_prob = points[points[:, 2].argsort()]
		start = min(k, points.shape[0])
		sorted_prob = sorted_prob[-start:, :]
	return sorted_prob.T

if __name__ == '__main__':
 main()