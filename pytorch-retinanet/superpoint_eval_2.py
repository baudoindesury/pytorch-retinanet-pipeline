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

    parser.add_argument('--superpoint', dest='ground_truth', action='store_true')
    parser.set_defaults(superpoint=False)
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

    avg_repeatibility_list = []
    avg_loc_error_list = []
    tp, fp, prob, matches, n_gt = [], [], [], [], 0

    COMPUTE_PR = False
    COMPUTE_REP_LE = True
    PLOT_KEYPOINTS = True
    SAVE_DATA = False
    suffix = '_synthetic_teacher_old_model'

    threashold_student = [0.16]#[0.05, 0.06, 0.07, 0.08, 0.1, 0.12,0.14, 0.15, 0.16, 0.17, 0.18, 0.19]#[0.11]#[0.08, 0.09, 0.1, 0.105, 0.11, 0.115, 0.12, 0.13, 0.14, 0.15, 0.16]
    threashold_teacher = [0.015] #np.linspace(0.005, 0.05,15) #[0.015, 0.016]

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

            repeatibility_list = []
            loc_error_list = []

            for t_teacher in threashold_teacher:
                pts_teacher, heatmap_teacher  = post_processing_superpoint_detector(dect_teacher, H, W, conf_thresh=t_teacher, border_remove=50)
        
                # Student SuperPoint output Tensors
                output_desc = output['desc'].type(torch.FloatTensor)
                output_semi = output['semi'].type(torch.FloatTensor)

                for t in threashold_student:
                    pts_student, heatmap_student = post_processing_superpoint_detector(F.softmax(output_semi.squeeze(), dim=1), H, W, conf_thresh=t, border_remove=50)
                    #pts_student = select_k_best(pts_student, pts_teacher.shape[1])

                    data_keypoints = {}
                    data_keypoints['prob'] = pts_teacher.T[:,[1,0,2]]
                    data_keypoints['warped_prob'] = pts_student.T[:,[1,0,2]]#select_k_best(pts_student.T[:,[1,0,2]], pts_teacher.shape[1])
                    data_keypoints['image'] = img_gray
                    data_keypoints['homography'] = np.identity(3)

                    if COMPUTE_PR:
                        t, f, p, n = compute_tp_fp(data_keypoints, remove_zero=1e-4, distance_thresh=10, simplified=True)
                        tp.append(t)
                        fp.append(f)
                        prob.append(p)
                        n_gt+=n
                    
                    if COMPUTE_REP_LE:
                        repeatability, localization_err = compute_repeatability(data_keypoints, distance_thresh=3, keep_k_points=1000)
                        repeatibility_list.append(repeatability)
                        loc_error_list.append(localization_err)

                    if PLOT_KEYPOINTS :
                        img = img_gray
                        suffix = 'superpoint'
                        save_path = '/home/baudoin/pytorch-retinanet-pipeline/pytorch-retinanet/output_images/img_' + suffix + '_' + str(idx) + '.jpeg'
                        plot_images(img, img, data_keypoints, save_path)

                if COMPUTE_REP_LE:
                    avg_repeatibility_list.append(repeatibility_list)
                    avg_loc_error_list.append(loc_error_list)
    if COMPUTE_PR:
        precision, recall, _ = compute_pr(tp, fp, prob, n_gt)

    if COMPUTE_REP_LE:
        print('Avg Repeatibility = ', np.mean(avg_repeatibility_list, axis = 0))
        print('Avg Localization_err = ', np.mean(avg_loc_error_list, axis=0))  
    
    path_data = '/home/baudoin/pytorch-retinanet-pipeline/results/superpoint_eval_data/'

    if COMPUTE_PR and SAVE_DATA:
        np.save(path_data + 'precision' + suffix, precision)
        np.save(path_data + 'recall' + suffix, recall)

    if COMPUTE_REP_LE and SAVE_DATA:
        np.save(path_data + 'rep' + suffix, avg_repeatibility_list)
        np.save(path_data + 'LE' + suffix, avg_loc_error_list)

        path = '/home/baudoin/pytorch-retinanet-pipeline/results/plots/'
        plt.figure(0)
        plt.plot(threashold_student,np.mean(avg_repeatibility_list, axis = 0))
        plt.xlabel('teacher threashold')
        plt.ylabel('Avg Repeatability')
        plt.savefig(path + 'rep' + suffix)

        plt.figure(1)
        plt.plot(threashold_student,np.mean(avg_loc_error_list, axis = 0))
        plt.xlabel('teacher threashold')
        plt.ylabel('Avg Loc Error')
        plt.savefig(path + 'LE' + suffix)



def post_processing_superpoint_detector(semi, H, W ,conf_thresh=0.015, cell=8, border_remove=8):
    # --- Process points.
    dense = semi.squeeze().numpy()
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
      return np.zeros((3, 0)), heatmap
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
    return pts, heatmap

def warp_keypoints(keypoints, H):
    """
    :param keypoints:
    points:
        numpy (N, (x,y))
    :param H:
    :return:
    """
    num_points = keypoints.shape[0]
    homogeneous_points = np.concatenate([keypoints, np.ones((num_points, 1))],
                                        axis=1)
    warped_points = np.dot(homogeneous_points, np.transpose(H))
    return warped_points[:, :2] / warped_points[:, 2:]


def select_k_best(points, k):
    """ Select the k most probable points.
    points has shape (num_points, 3) where the last coordinate is the proba. """
    sorted_prob = points
    if points.shape[1] > 2:
        sorted_prob = points[points[:, 2].argsort(), :3]
        start = min(k, points.shape[0])
        sorted_prob = sorted_prob[-start:, :]
    return sorted_prob


def compute_repeatability(data, keep_k_points=300,
                          distance_thresh=3, verbose=False):
    """
    Compute the repeatability. The experiment must contain in its output the prediction
    on 2 images, an original image and a warped version of it, plus the homography
    linking the 2 images.
    """

    def filter_keypoints(points, shape):
        """ Keep only the points whose coordinates are
        inside the dimensions of shape. """
        """
        points:
            numpy (N, (x,y))
        shape:
            (y, x)
        """
        mask = (points[:, 0] >= 0) & (points[:, 0] < shape[1]) &\
               (points[:, 1] >= 0) & (points[:, 1] < shape[0])
        return points[mask, :]

    def select_k_best(points, k):
        """ Select the k most probable points (and strip their proba).
        points has shape (num_points, 3) where the last coordinate is the proba. """
        sorted_prob = points
        if points.shape[1] > 2:
            sorted_prob = points[points[:, 2].argsort(), :2]
            start = min(k, points.shape[0])
            sorted_prob = sorted_prob[-start:, :]
        return sorted_prob

    def keep_true_keypoints(points, H, shape):
        """ Keep only the points whose warped coordinates by H
        are still inside shape. """
        """
        input:
            points: numpy (N, (x,y))
            shape: (y, x)
        return:
            points: numpy (N, (x,y))
        """
        # warped_points = warp_keypoints(points[:, [1, 0]], H)
        warped_points = warp_keypoints(points[:, [0, 1]], H)
        # warped_points[:, [0, 1]] = warped_points[:, [1, 0]]
        mask = (warped_points[:, 0] >= 0) & (warped_points[:, 0] < shape[1]) &\
               (warped_points[:, 1] >= 0) & (warped_points[:, 1] < shape[0])
        return points[mask, :]

    # paths = get_paths(exper_name)
    localization_err = -1
    repeatability = []
    N1s = []
    N2s = []
    # for path in paths:
    # data = np.load(path)
    shape = data['image'].shape
    #print('img shape = ', shape)
    H = data['homography']

    # Filter out predictions
    # keypoints = np.where(data['prob'] > 0)
    # prob = data['prob'][keypoints[0], keypoints[1]]
    # keypoints = np.stack([keypoints[0], keypoints[1]], axis=-1)
    # warped_keypoints = np.where(data['warped_prob'] > 0)
    # warped_prob = data['warped_prob'][warped_keypoints[0], warped_keypoints[1]]
    # warped_keypoints = np.stack([warped_keypoints[0],
    #                              warped_keypoints[1],
    #                              warped_prob], axis=-1)
    # keypoints = data['prob'][:, :2]
    keypoints = data['prob']
    #print('prob/teacher shape = ', np.shape(keypoints))
    #print(keypoints)
    # warped_keypoints = data['warped_prob'][:, :2]
    warped_keypoints = data['warped_prob']
    #print('warped_prob/student shape = ', np.shape(warped_keypoints))
    #print(warped_keypoints)
    
    warped_keypoints = keep_true_keypoints(warped_keypoints, np.linalg.inv(H),
                                           data['image'].shape)

    # Warp the original keypoints with the true homography
    true_warped_keypoints = keypoints
    #true_warped_keypoints[:,:2] = warp_keypoints(keypoints[:, [1, 0]], H)
    true_warped_keypoints[:,:2] = warp_keypoints(keypoints[:, :2], H) # make sure the input fits the (x,y)
    # true_warped_keypoints = np.stack([true_warped_keypoints[:, 1],
    #                                   true_warped_keypoints[:, 0],
    #                                   prob], axis=-1)
    true_warped_keypoints = filter_keypoints(true_warped_keypoints, shape)

    # Keep only the keep_k_points best predictions

    warped_keypoints = select_k_best(warped_keypoints, keep_k_points)
    true_warped_keypoints = select_k_best(true_warped_keypoints, keep_k_points)

    # Compute the repeatability
    N1 = true_warped_keypoints.shape[0]
    #print('true_warped_keypoints shape =', true_warped_keypoints.shape)
    #print('true_warped_keypoints: ', true_warped_keypoints[:2,:])
    N2 = warped_keypoints.shape[0]
    #print('warped_keypoints shape =', warped_keypoints.shape)
    #print('warped_keypoints: ', warped_keypoints[:2,:])
    N1s.append(N1)
    N2s.append(N2)
    true_warped_keypoints = np.expand_dims(true_warped_keypoints, 1)
    warped_keypoints = np.expand_dims(warped_keypoints, 0)
    # shapes are broadcasted to N1 x N2 x 2:
    norm = np.linalg.norm(true_warped_keypoints - warped_keypoints,
                          ord=None, axis=2)
    count1 = 0
    count2 = 0
    local_err1, local_err2 = None, None
    if N2 != 0:
        min1 = np.min(norm, axis=1)
        count1 = np.sum(min1 <= distance_thresh)
        # print("count1: ", count1)
        local_err1 = min1[min1 <= distance_thresh]
        # print("local_err1: ", local_err1)
    if N1 != 0:
        min2 = np.min(norm, axis=0)
        count2 = np.sum(min2 <= distance_thresh)
        local_err2 = min2[min2 <= distance_thresh]

    if N1 + N2 > 0:
        # repeatability.append((count1 + count2) / (N1 + N2))
        repeatability = (count1 + count2) / (N1 + N2)
    if count1 + count2 > 0:
        localization_err = 0
        if local_err1 is not None:
            localization_err += (local_err1.sum())/ (count1 + count2)
        if local_err2 is not None:
            localization_err += (local_err2.sum())/ (count1 + count2)
    else:
        repeatability = 0
    if verbose:
        print("Average number of points in the first image: " + str(np.mean(N1s)))
        print("Average number of points in the second image: " + str(np.mean(N2s)))
    # return np.mean(repeatability)
    return repeatability, localization_err

def compute_tp_fp(data, remove_zero=1e-4, distance_thresh=2, simplified=False):
    """
    Compute the true and false positive rates.
    """
    # Read data
    #gt = np.where(data['keypoint_map'])
    #gt = np.stack([gt[0], gt[1]], axis=-1)
    gt = data['prob'][:,0:2]
    n_gt = len(gt)
    prob = data['warped_prob'][:,2]
    pred = data['warped_prob'][:,0:2]
    
    # Filter out predictions with near-zero probability
    mask = np.where(prob > remove_zero)
    prob = prob[mask]
    pred = np.array(mask).T

    # When several detections match the same ground truth point, only pick
    # the one with the highest score  (the others are false positive)
    sort_idx = np.argsort(prob)[::-1]
    prob = prob[sort_idx]
    pred = pred[sort_idx]

    diff = np.expand_dims(pred, axis=1) - np.expand_dims(gt, axis=0)
    dist = np.linalg.norm(diff, axis=-1)
    matches = np.less_equal(dist, distance_thresh)

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
    return tp, fp, prob, n_gt

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


def plot_images(image, warped_image, data, save_path):
    # img = to3dim(image)
    img = image
    pts = data['prob'][:,[1,0,2]]
    img1 = draw_keypoints(img*255, pts.transpose())

    # img = to3dim(warped_image)
    img = warped_image
    pts = data['warped_prob'][:,[1,0,2]]
    img2 = draw_keypoints(img*255, pts.transpose(), color = (0,0,255))

    plot_imgs([img1.astype(np.uint8), img2.astype(np.uint8)], titles=['Teacher', 'Student'], dpi=200)
    #plt.title("rep: " + str(repeatability[-1]))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

def draw_keypoints(img, corners, color=(0, 255, 0), radius=5, s=3):
    '''
    :param img:
        image:
        numpy [H, W]
    :param corners:
        Points
        numpy [N, 2]
    :param color:
    :param radius:
    :param s:
    :return:
        overlaying image
        numpy [H, W]
    '''
    img = np.repeat(cv2.resize(img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)
    for c in np.stack(corners).T:
        # cv2.circle(img, tuple(s * np.flip(c, 0)), radius, color, thickness=-1)
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, color, thickness=-1)
    return img


def plot_imgs(imgs, titles=None, cmap='brg', ylabel='', normalize=False, ax=None, dpi=100):
    n = len(imgs)
    if not isinstance(cmap, list):
        cmap = [cmap]*n
    if ax is None:
        fig, ax = plt.subplots(1, n, figsize=(6*n, 6), dpi=dpi)
        if n == 1:
            ax = [ax]
    else:
        if not isinstance(ax, list):
            ax = [ax]
        assert len(ax) == len(imgs)
    for i in range(n):
        if imgs[i].shape[-1] == 3:
            imgs[i] = imgs[i][..., ::-1]  # BGR to RGB
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmap[i]),
                     vmin=None if normalize else 0,
                     vmax=None if normalize else 1)
        if titles:
            ax[i].set_title(titles[i])
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    ax[0].set_ylabel(ylabel)
    plt.tight_layout()




if __name__ == '__main__':
 main()