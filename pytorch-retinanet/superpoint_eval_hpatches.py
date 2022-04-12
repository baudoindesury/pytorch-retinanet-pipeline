import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageOps




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

from descriptor_evaluation import homography_estimation


# load checkpoint (to remove)
from retinanet import model
import matplotlib.pyplot as plt



assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))




def get_superpoint_output(scene_paths, classes_path, model_path):
    num_classes=10
    retinanet = model.resnet50(num_classes=num_classes, pretrained=True)
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
    superpoint = SuperPointFrontend(project_root='') #Modify project_root
    t_teacher = 0.015
    t_student = 0.15
    i_img = 0


    repeatability_teacher_list = []
    repeatability_student_list = []

    for scene_path in tqdm(scene_paths):

        i_img = 0
        data_keypoints_teacher_paths = []
        data_keypoints_student_paths = []
        scene_name = Path(scene_path).stem
        if scene_name[0] == 'i':
            continue

        pairs_name = [[1,1], [1,2], [1,3], [1,4], [1,5], [1,6]]
        
        for pair_name in pairs_name:
            pts_student = []
            heatmap_student = []
            desc_student = []
            pts_teacher = []
            heatmap_teacher = []
            desc_teacher = []
            imgs_gray = []
            img_1 = scene_path + '/' + str(pair_name[0]) + '.ppm'
            img_2 = scene_path + '/' + str(pair_name[1]) + '.ppm'
            H_1_2 = scene_path + '/H_'+str(pair_name[0])+'_'+str(pair_name[1])
            img_init_shape = np.asarray(Image.open(img_1)).shape[:2]

            save_empty_anno(scene_path, [img_1, img_2])
            data = CSVDataset(train_file=scene_path + '/anno.csv', class_list=classes_path,transform=transforms.Compose([Normalizer(), Resizer()]))

            dataloader_val = DataLoader(data, num_workers=1, collate_fn=collater)

            for idx, data in enumerate(dataloader_val):
                with torch.no_grad():
                    
                    if torch.cuda.is_available():
                        scores, classification, transformed_anchors, output = retinanet([data['img'].cuda().float(), data['annot']])
                    else:
                        scores, classification, transformed_anchors, output = retinanet([data['img'].float(), data['annot']])
                    
                    img_gray = np.squeeze(data['img_gray'].numpy()*255)
                    H, W = img_gray.shape[0], img_gray.shape[1]
                    imgs_gray.append(img_gray[:-32,:-32])

                    S = np.identity(3)
                    S_inv = np.identity(3)
                    np.fill_diagonal(S, [(H-32)/img_init_shape[0],(W-32)/img_init_shape[1],1])
                    S_inv[:2,:2] = np.linalg.inv(S[:2,:2])
                    S_inv[2,2] = 1
                    
                    # Student SuperPoint output Tensors
                    output_desc = output['desc'].type(torch.FloatTensor)
                    output_semi = output['semi'].type(torch.FloatTensor)
                    pts, heatmap = post_processing_superpoint_detector(F.softmax(output_semi.squeeze() ,dim=1),output_desc , H, W, conf_thresh=t_student, border_remove=5)
                    pts_student.append(pts)
                    heatmap_student.append(heatmap)
                    
                    # SuperPoint label with teacher model
                    output_superpoint = superpoint.run(data['img_gray'].float())
                    out_dect_teacher = torch.from_numpy(output_superpoint['dense_scores']).type(torch.FloatTensor)
                    out_desc_teacher = torch.from_numpy(output_superpoint['local_descriptor_map']).type(torch.FloatTensor)
                    pts, heatmap  = post_processing_superpoint_detector(out_dect_teacher, out_desc_teacher ,H, W, conf_thresh=t_teacher, border_remove=5)
                    pts_teacher.append(pts)
                    heatmap_teacher.append(heatmap)

            data_keypoints_teacher = {}
            data_keypoints_teacher['prob'] = heatmap_teacher[0][:-32,:-32]
            data_keypoints_teacher['image'] = imgs_gray[0][:,:, np.newaxis]#[:-32,:-32, np.newaxis]
            
            if len(heatmap_teacher) == 1:
                data_keypoints_teacher['warped_prob'] = heatmap_teacher[0][:-32,:-32]
                data_keypoints_teacher['homography'] = np.identity(3)
                imgs_gray.append(img_gray[:-32,:-32])
                pts_teacher.append(pts)
            else:
                data_keypoints_teacher['warped_prob'] = heatmap_teacher[1][:-32,:-32]
                data_keypoints_teacher['homography'] = np.dot(np.dot(S, np.loadtxt(H_1_2)),S_inv)
            
            data_keypoints_student = {}
            data_keypoints_student['prob'] = heatmap_student[0][:-32,:-32]
            data_keypoints_student['image'] = imgs_gray[0][:,:, np.newaxis]#[:-32,:-32, np.newaxis]
            
            if len(heatmap_student) == 1:
                data_keypoints_student['warped_prob'] = heatmap_student[0][:-32,:-32]
                data_keypoints_student['homography'] = np.identity(3)
                pts_student.append(pts)
            else:
                data_keypoints_student['warped_prob'] = heatmap_student[1][:-32,:-32]
                data_keypoints_student['homography'] = np.dot(np.dot(S, np.loadtxt(H_1_2)),S_inv)
            
            saving_path_teacher_keypoints_data = '/home/baudoin/pytorch-retinanet-pipeline/data_keypoints/teacher_' + str(i_img) + '.npz'
            np.savez(saving_path_teacher_keypoints_data, **{'0':data_keypoints_teacher})
            data_keypoints_teacher_paths.append(saving_path_teacher_keypoints_data)

            saving_path_student_keypoints_data = '/home/baudoin/pytorch-retinanet-pipeline/data_keypoints/student_' + str(i_img) + '.npz'
            np.savez(saving_path_student_keypoints_data, **{'0':data_keypoints_student})
            data_keypoints_student_paths.append(saving_path_student_keypoints_data)
            i_img+=1
            if True:
                save_path = '/home/baudoin/pytorch-retinanet-pipeline/pytorch-retinanet/output_images/img_teacher_'+ str(scene_name) + str(i_img) + '.jpeg'
                plot_images(imgs_gray[0], imgs_gray[1], pts_teacher[0].T, pts_teacher[1].T, save_path)
                save_path = '/home/baudoin/pytorch-retinanet-pipeline/pytorch-retinanet/output_images/img_student_'+ str(scene_name) + str(i_img) + '.jpeg'
                plot_images(imgs_gray[0], imgs_gray[1], pts_student[0].T, pts_student[1].T, save_path)    

        repeatability_teacher_list.append(compute_repeatability(data_keypoints_teacher_paths, distance_thresh=8, keep_k_points=300, verbose=False))
        repeatability_student_list.append(compute_repeatability(data_keypoints_student_paths, distance_thresh=8, keep_k_points=300, verbose=False))

    print('repeatability teacher = ', np.mean([i for l in repeatability_teacher_list for i in l]))
    print('repeatability student = ', np.mean([i for l in repeatability_student_list for i in l]))




def save_empty_anno(output_path, img_paths):
    with open(output_path + '/anno.csv', 'w') as csv_file:
        for tile_path in img_paths:
            csv_file.write(tile_path + ',' +
                            ','.join(['', '', '', '']) + ',' + '' + '\n')




def post_processing_superpoint_detector(semi, coarse_desc, H, W ,conf_thresh=0.015, cell=8, border_remove=8, nms_dist=3):
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
    pts, _ = nms_fast(pts, H, W, dist_thresh=nms_dist) # Apply NMS.
    inds = np.argsort(pts[2,:])
    pts = pts[:,inds[::-1]] # Sort by confidence.
    # Remove points along border.
    bord = border_remove
    toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W-bord))
    toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H-bord))
    toremove = np.logical_or(toremoveW, toremoveH)
    pts = pts[:, ~toremove]

    return pts, heatmap


def select_k_best(points, k):
    """ Select the k most probable points.
    points has shape (num_points, 3) where the last coordinate is the proba. """
    sorted_prob = points
    if points.shape[1] > 2:
        sorted_prob = points[points[:, 2].argsort(), :3]
        start = min(k, points.shape[0])
        sorted_prob = sorted_prob[-start:, :]
    return sorted_prob


def compute_repeatability(paths, keep_k_points=300, distance_thresh=3, verbose=False):
    """
    Compute the repeatability. The experiment must contain in its output the prediction
    on 2 images, an original image and a warped version of it, plus the homography
    linking the 2 images.
    """
    def warp_keypoints(keypoints, H):
        num_points = keypoints.shape[0]
        homogeneous_points = np.concatenate([keypoints, np.ones((num_points, 1))],
                                            axis=1)
        warped_points = np.dot(homogeneous_points, np.transpose(H))
        return warped_points[:, :2] / warped_points[:, 2:]

    def filter_keypoints(points, shape):
        """ Keep only the points whose coordinates are
        inside the dimensions of shape. """
        mask = (points[:, 0] >= 0) & (points[:, 0] < shape[0]) &\
               (points[:, 1] >= 0) & (points[:, 1] < shape[1])
        return points[mask, :]

    def keep_true_keypoints(points, H, shape):
        """ Keep only the points whose warped coordinates by H
        are still inside shape. """
        warped_points = warp_keypoints(points[:, [1, 0]], H)
        warped_points[:, [0, 1]] = warped_points[:, [1, 0]]
        mask = (warped_points[:, 0] >= 0) & (warped_points[:, 0] < shape[0]) &\
               (warped_points[:, 1] >= 0) & (warped_points[:, 1] < shape[1])
        return points[mask, :]

    def select_k_best(points, k):
        """ Select the k most probable points (and strip their proba).
        points has shape (num_points, 3) where the last coordinate is the proba. """
        sorted_prob = points[points[:, 2].argsort(), :2]
        start = min(k, points.shape[0])
        return sorted_prob[-start:, :]

    
    repeatability = []
    N1s = []
    N2s = []
    for path in paths:
        
        data = np.load(path, allow_pickle=True)['0'].item()
        shape = data['warped_prob'].shape
        H = data['homography']

        # Filter out predictions
        keypoints = np.where(data['prob'] > 0)
        prob = data['prob'][keypoints[0], keypoints[1]]
        keypoints = np.stack([keypoints[0], keypoints[1]], axis=-1)
        warped_keypoints = np.where(data['warped_prob'] > 0)
        warped_prob = data['warped_prob'][warped_keypoints[0], warped_keypoints[1]]
        warped_keypoints = np.stack([warped_keypoints[0],
                                     warped_keypoints[1],
                                     warped_prob], axis=-1)
        warped_keypoints = keep_true_keypoints(warped_keypoints, np.linalg.inv(H),
                                               data['prob'].shape)

        # Warp the original keypoints with the true homography
        true_warped_keypoints = warp_keypoints(keypoints[:, [1, 0]], H)
        true_warped_keypoints = np.stack([true_warped_keypoints[:, 1],
                                          true_warped_keypoints[:, 0],
                                          prob], axis=-1)
        true_warped_keypoints = filter_keypoints(true_warped_keypoints, shape)

        # Keep only the keep_k_points best predictions
        warped_keypoints = select_k_best(warped_keypoints, keep_k_points)
        true_warped_keypoints = select_k_best(true_warped_keypoints, keep_k_points)

        # Compute the repeatability
        N1 = true_warped_keypoints.shape[0]
        N2 = warped_keypoints.shape[0]
        N1s.append(N1)
        N2s.append(N2)
        true_warped_keypoints = np.expand_dims(true_warped_keypoints, 1)
        warped_keypoints = np.expand_dims(warped_keypoints, 0)
        # shapes are broadcasted to N1 x N2 x 2:
        norm = np.linalg.norm(true_warped_keypoints - warped_keypoints,
                              ord=None, axis=2)
        count1 = 0
        count2 = 0
        if N2 != 0:
            min1 = np.min(norm, axis=1)
            count1 = np.sum(min1 <= distance_thresh)
        if N1 != 0:
            min2 = np.min(norm, axis=0)
            count2 = np.sum(min2 <= distance_thresh)
        if N1 + N2 > 0:
            repeatability.append((count1 + count2) / (N1 + N2))

    if verbose:
        print("Average number of points in the first image: " + str(np.mean(N1s)))
        print("Average number of points in the second image: " + str(np.mean(N2s)))
    #return np.mean(repeatability)
    return repeatability


def img_to_numpy_grayscale(path, H=None, W=None, resize=False):
    im = Image.open(path)
    if resize:
        im = im.resize((H,W),Image.ANTIALIAS)
    im = ImageOps.grayscale(im)
    im = np.array(im)
    return im

def plot_images(image, warped_image, pts, pts_warped, save_path):
    img = image
    img1 = draw_keypoints(img*255, pts.transpose())

    img = warped_image
    pts = pts_warped
    img2 = draw_keypoints(img*255, pts.transpose(), color = (0,0,255))

    plot_imgs([img1.astype(np.uint8), img2.astype(np.uint8)], titles=['Img 1', 'Img 2'], dpi=200)
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




def nms_fast(in_corners, H, W, dist_thresh):
    """
    Run a faster approximate Non-Max-Suppression on numpy corners shaped:
      3xN [x_i,y_i,conf_i]^T
  
    Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
    are zeros. Iterate through all the 1's and convert them either to -1 or 0.
    Suppress points by setting nearby values to 0.
  
    Grid Value Legend:
    -1 : Kept.
     0 : Empty or suppressed.
     1 : To be processed (converted to either kept or supressed).
  
    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundaries.
  
    Inputs
      in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      H - Image height.
      W - Image width.
      dist_thresh - Distance to suppress, measured as an infinty norm distance.
    Returns
      nmsed_corners - 3xN numpy matrix with surviving corners.
      nmsed_inds - N length numpy vector with surviving corner indices.
    """
    grid = np.zeros((H, W)).astype(int) # Track NMS data.
    inds = np.zeros((H, W)).astype(int) # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2,:])
    corners = in_corners[:,inds1]
    rcorners = corners[:2,:].round().astype(int) # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
      return np.zeros((3,0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
      out = np.vstack((rcorners, in_corners[2])).reshape(3,1)
      return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
      grid[rcorners[1,i], rcorners[0,i]] = 1
      inds[rcorners[1,i], rcorners[0,i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad,pad), (pad,pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
      # Account for top and left padding.
      pt = (rc[0]+pad, rc[1]+pad)
      if grid[pt[1], pt[0]] == 1: # If not yet suppressed.
        grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
        grid[pt[1], pt[0]] = -1
        count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid==-1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds

def main():
    classes_path = '/home/baudoin/pytorch-retinanet-pipeline/annotations/visdrone_anno/class_labels.csv'
    model_path = '/home/baudoin/pytorch-retinanet-pipeline/results/training_results/20220209-193813/checkpoints/best_model_retinanet_csv.pt'

    hpatches_path = '/home/baudoin/data/hpatches-sequences-release/'
    scene_paths = [x[0] for x in os.walk(hpatches_path)][1:]

    get_superpoint_output(scene_paths, classes_path, model_path)

if __name__ == '__main__':
    main()