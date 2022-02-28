import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from torch import nn
import os.path as osp
from torch.nn import functional as F


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

    """
    # --- Process descriptor.
    D = coarse_desc.shape[1]
    if pts.shape[1] == 0:
      desc = np.zeros((D, 0))
    else:
      # Interpolate into descriptor map using 2D point locations.
      samp_pts = torch.from_numpy(pts[:2, :].copy())
      samp_pts[0, :] = (samp_pts[0, :] / (float(W)/2.)) - 1.
      samp_pts[1, :] = (samp_pts[1, :] / (float(H)/2.)) - 1.
      samp_pts = samp_pts.transpose(0, 1).contiguous()
      samp_pts = samp_pts.view(1, 1, -1, 2)
      samp_pts = samp_pts.float()
      if self.cuda:
        samp_pts = samp_pts.cuda()
      desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts)
      desc = desc.data.cpu().numpy().reshape(D, -1)
      desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]"""

    return pts, heatmap


def post_processing_superpoint(semi, coarse_desc, H, W ,conf_thresh=0.015, cell=8, border_remove=8):
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


    # --- Process descriptor.
    D = coarse_desc.shape[1]
    if pts.shape[1] == 0:
      desc = np.zeros((D, 0))
    else:
      # Interpolate into descriptor map using 2D point locations.
      samp_pts = torch.from_numpy(pts[:2, :].copy())
      samp_pts[0, :] = (samp_pts[0, :] / (float(W)/2.)) - 1.
      samp_pts[1, :] = (samp_pts[1, :] / (float(H)/2.)) - 1.
      samp_pts = samp_pts.transpose(0, 1).contiguous()
      samp_pts = samp_pts.view(1, 1, -1, 2)
      samp_pts = samp_pts.float()
      desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts, align_corners=True)
      desc = desc.data.cpu().numpy().reshape(D, -1)
      desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]

    return pts, desc, heatmap

def plot_superpoint_keypoints(img_gray, pts, title='keypoints'):
    out2 = (np.dstack((img_gray, img_gray, img_gray)) * 255.).astype('uint8')
    for pt in pts.T:
        pt1 = (int(round(pt[0])), int(round(pt[1])))
        cv2.circle(out2, pt1, 3, (0, 255, 0), -1, lineType=16)
        
    plt.figure(figsize=(16,8))
    plt.imshow(out2)
    plt.savefig(title + '.jpeg')
    plt.show()


def plot_superpoint_keypoints_gt_and_pred(img_gray, pts_gt, pts_pred, title='keypoints'):
    out2 = (np.dstack((img_gray, img_gray, img_gray)) * 255.).astype('uint8')
    for pt in pts_gt.T:
        pt1 = (int(round(pt[0])), int(round(pt[1])))
        cv2.circle(out2, pt1, 5, (0, 255, 0), 1, lineType=16)
    for pt in pts_pred.T:
        pt1 = (int(round(pt[0])), int(round(pt[1])))
        cv2.circle(out2, pt1, 5, (0, 0, 255), 1, lineType=16)

    
    cv2.imwrite(title + '.jpeg', out2)

def draw_matches(img_gray, match_pairs, lw = 0.5, color='g', if_fig=True,
                filename='matches.png', show=False):
    '''
    :param rgb1:
        image1
        numpy (H, W)
    :param rgb2:
        image2
        numpy (H, W)
    :param match_pairs:
        numpy (keypoiny1 x, keypoint1 y, keypoint2 x, keypoint 2 y)
    :return:
        None
    '''
    from matplotlib import pyplot as plt
    rgb1 = (np.dstack((img_gray, img_gray, img_gray)) * 255.).astype('uint8')


    h1, w1 = rgb1.shape[:2]
    h2, w2 = rgb1.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=rgb1.dtype)
    canvas[:h1, :w1] = rgb1[:,:,np.newaxis].squeeze()
    canvas[:h2, w1:] = rgb1[:,:,np.newaxis].squeeze()
    # fig = plt.figure(frameon=False)
    if if_fig:
        fig = plt.figure(figsize=(15,5))
    plt.axis("off")
    plt.imshow(canvas, zorder=1)

    xs = match_pairs[:, [0, 2]]
    xs[:, 1] += w1
    ys = match_pairs[:, [1, 3]]

    alpha = 1
    sf = 5
    # lw = 0.5
    # markersize = 1
    markersize = 2

    plt.plot(
        xs.T, ys.T,
        alpha=alpha,
        linestyle="-",
        linewidth=lw,
        aa=False,
        marker='o',
        markersize=markersize,
        fillstyle='none',
        color=color,
        zorder=2,)

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    #print('#Matches = {}'.format(len(match_pairs)))
    if show:
        plt.show()







