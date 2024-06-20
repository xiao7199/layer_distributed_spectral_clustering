from skimage import measure
from sklearn.cluster import AgglomerativeClustering
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from pycocotools import mask as maskUtils
import cv2

def torch_silhouette_score(feat, label):
    # label: N
    # feat : N * C
    device = feat.device
    N,C = feat.shape
    assert label.shape[0] == N
    label = label.to(device)
    max_label = label.max()
    # N * C_label
    label_one_hot = F.one_hot(label, max_label + 1).float()

    # C * C_label
    feat_center = (feat.T @ label_one_hot) / (label_one_hot.sum(dim = 0, keepdims = True))
    # N * N
    pairdist = torch.cdist(feat, feat)
    # N
    intra_avg_distance = ((pairdist @ label_one_hot) * label_one_hot).sum(dim = -1) / (label_one_hot @ label_one_hot.sum(dim = 0) - 1  + 1e-10)
    raw_inter_avg_distance = ((pairdist @ label_one_hot) * (1 - label_one_hot)) / (label_one_hot.sum(dim = 0)  + 1e-10)[None,:]
    inter_avg_distance = (raw_inter_avg_distance * (1 - label_one_hot) + (raw_inter_avg_distance.max() + 1) *  label_one_hot).min(dim = -1)[0]

    score = (inter_avg_distance - intra_avg_distance) / torch.stack([inter_avg_distance, intra_avg_distance], dim = -1).max(dim = -1)[0]
    score_zero_mask = (label_one_hot @ (label_one_hot.sum(dim = 0) == 1).float())
    score[score_zero_mask == 1] = 0
    return score.mean()

@torch.no_grad()
def auto_kmeans_feat(feat, max_n_clusters=8):
    from sklearn.cluster import KMeans
    c, h, w = feat.shape
    coeffs = []
    label_list = []
    for n_clusters in range(2, max_n_clusters+1):
        new_feat = feat
        X = new_feat.flatten(1).cpu().numpy().T
        kmeans = KMeans(n_clusters=n_clusters).fit(X)
        coeff = torch_silhouette_score(torch.from_numpy(X).cuda(), torch.from_numpy(kmeans.labels_).cuda().long())
        coeff = coeff.item()
        #coeff = silhouette_score(X, kmeans.labels_, metric='euclidean')
        coeffs.append(coeff)
        label_list.append([kmeans.labels_, coeff, n_clusters])
    best_n = np.argmax(coeffs)
    kmeans = KMeans(n_clusters=best_n+2).fit(X)
    best_kmeans= torch.from_numpy(kmeans.labels_).reshape(h, w)
    return best_kmeans, best_n + 2, label_list

def pca_feat(feat, pca_dim = 3, normalize_as_img = False):
    device = feat.device
    N,C,H,W = feat.shape
    feat = feat.reshape(N,C,-1).permute(0,2,1)
    pca_dim = min(min(pca_dim, feat.shape[1]), feat.shape[2])
    [u,s,v] = torch.pca_lowrank(feat, pca_dim, niter = 2)
    #v = multi_grid_low_rank(feat, pca_dim, niter = 1)
    feat = torch.matmul(feat, v)
    feat = feat.reshape(N,H,W,pca_dim).permute(0,3,1,2).contiguous()
    if normalize_as_img:
        feat_min = feat.reshape(feat.shape[0], pca_dim, -1).min(dim = -1)[0]
        feat_max = feat.reshape(feat.shape[0], pca_dim, -1).max(dim = -1)[0]
        feat = 255 * (feat - feat_min.reshape(-1,pca_dim,1,1)) / (feat_max - feat_min).reshape(-1,pca_dim,1,1)
        feat = feat.cpu().data.numpy().transpose(0,2,3,1).astype(np.uint8)
    return feat

def mask_nms(mask1, mask2, score2, iou_th = 0.7):
    intersect = (mask1[:,:,None] * mask2).sum(dim = [0,1])
    union = (((mask1[:,:,None] + mask2) > 0).float()).sum(dim = [0,1])
    iou = (intersect / union)
    mask2 = mask2[:,:,iou < iou_th]
    score2 = score2[iou < iou_th]
    return mask2, score2

def compute_binary_silhouette_score(feat, label, feat_h, feat_w):
    feat = feat.reshape(feat.shape[0],-1).T
    label = label.reshape(feat.shape[0], -1)
    # label: N * K
    # feat : N * C
    device = feat.device
    N,C = feat.shape
    K = label.shape[1]
    label = label.to(device)
    max_label = label.max()
    # N * K * 2
    label_one_hot = F.one_hot(label.long(), 2).float()
    # N * Nb
    pairdist = torch.cdist(feat, feat)
    bg_mask = find_comp_mask(pairdist, label)
    bg_mask = bg_mask * (1 - label)
    bg_label_one_hot = F.one_hot(bg_mask.long(), 2).float()
    # N * K
    intra_avg_distance = ((pairdist @ label_one_hot.flatten(1,-1)).reshape(N,K,2) * label_one_hot).sum(dim = -1) / (label_one_hot[:,:,None] @ label_one_hot.sum(dim = 0, keepdims = True)[:,:,:,None] - 1  + 1e-10)[:,:,0,0]
    inter_avg_distance = ((pairdist @ bg_label_one_hot.flatten(1,-1)).reshape(N,K,2) * label_one_hot).sum(dim = -1) / (label_one_hot[:,:,None] @ bg_label_one_hot.sum(dim = 0, keepdims = True)[:,:,:,None] + 1e-10)[:,:,0,0]
    score = (inter_avg_distance - intra_avg_distance) / torch.stack([inter_avg_distance, intra_avg_distance], dim = -1).max(dim = -1)[0]
    full_mask = (bg_mask + label)
    score = (score * full_mask).sum(dim = 0) / full_mask.sum(dim = 0)
    return score

def compute_segment_center(feat, label):
    feat = feat.flatten(1,-1).T
    label = label.reshape(-1)
    # feat: N * C
    # label_one_hot: N * K
    label_one_hot = F.one_hot(label.long(), label.max() + 1).float()
    feat_center = label_one_hot.T @ feat
    return feat_center / (label_one_hot.sum(dim = 0, keepdims = True).T + 1e-10)

def find_label_adj_pair(label):
    max_label = label.max()
    label = label.long()
    label = label + 1
    pair_label = torch.cat([torch.nn.Unfold(kernel_size = 3, padding = 1)((label).float()[None,None,...]), label.reshape(1,1,-1).expand(-1, 9, -1)], dim = 0).reshape(2,-1).T
    pair_label = pair_label[(pair_label != 0).float().sum(dim = -1) == 2] -1
    unique_pair = torch.unique(pair_label, dim = 0)
    filtered_unique_pair = [unique_pair[i] for i in range(unique_pair.shape[0]) if unique_pair[i,0] != unique_pair[i,1]]
    pair = torch.stack(filtered_unique_pair).long()
    adj_graph = torch.zeros(max_label + 1, max_label + 1)
    adj_graph[pair[:,0], pair[:,1]] = 1
    adj_graph[pair[:,1], pair[:,0]] = 1
    return adj_graph

def find_comp_mask(pairdist, fg_mask):
    # Resampled balanced bg_mask and only preserve the pixels that are
    # close to fg_mask

    # paridist : N * N
    # fg_mask : N * k
    paridist = pairdist.clone()
    max_dist = pairdist.max().abs() * 2
    # N * N * K
    pair_fg_mask = fg_mask[:,None] * (1 - fg_mask[None,:])
    new_pairdist = pairdist[:,:,None] * (pair_fg_mask) + max_dist * torch.ones_like(pairdist)[:,:,None] * (1 - pair_fg_mask)
    new_pairdist_sort_val, new_pairdist_sort_id = new_pairdist.sort(dim = 1)

    fb_mask_sum = fg_mask.sum(dim = 0).long()
    for idx in range(fb_mask_sum.shape[0]):
        new_pairdist_sort_val[:,fb_mask_sum[idx]:, idx] = max_dist

    top_new_pairdist = new_pairdist_sort_val.gather(dim = 1, index = new_pairdist_sort_id.argsort(1))
    topk_dist, topk_ind = top_new_pairdist.min(dim = 0)[0].sort(dim = 0)
    bg_mask = torch.zeros_like(fg_mask)
    for idx in range(fb_mask_sum.shape[0]):
        ind = topk_ind[:fb_mask_sum[idx],idx]
        bg_mask[ind, idx] = 1
    return bg_mask

def estimate_connected_components(connectivity):
    # Make the connectivity matrix symmetric:
    connectivity = connectivity + connectivity.T
    # Convert connectivity matrix to LIL
    if not sparse.isspmatrix_lil(connectivity):
        if not sparse.isspmatrix(connectivity):
            connectivity = sparse.lil_matrix(connectivity)
        else:
            connectivity = connectivity.tolil()

    # Compute the number of nodes
    n_connected_components, labels = connected_components(connectivity)
    return n_connected_components

def remove_bg_mask(mask, edge_count):
    edge = (mask[:,0,:].sum(dim = 0) > 0).float() + \
           (mask[:,-1,:].sum(dim = 0) > 0).float() + \
           (mask[0,:,:].sum(dim = 0) > 0).float() + \
           (mask[-1,:,:].sum(dim = 0) > 0).float()
    bg = (edge >= edge_count)
    mask = mask[:,:,~bg]
    return mask


def hierarchical_clustering(init_seg, feat):
    init_seg = torch.from_numpy(init_seg).to(feat.device)
    adj_graph = find_label_adj_pair(init_seg)
    feat_center = compute_segment_center(feat, init_seg)
    adj_graph = adj_graph.cpu().data.numpy()
    connected_components = estimate_connected_components(adj_graph)
    cluster = AgglomerativeClustering(compute_full_tree = False, n_clusters = connected_components, connectivity = adj_graph)
    cluster.fit(feat_center.cpu().data.numpy())

    max_child = cluster.children_.max()
    segment = torch.zeros(init_seg.shape[0], init_seg.shape[1], init_seg.max() + 1 + cluster.children_.shape[0]).to(feat.device)
    segment[:,:,:init_seg.max() + 1] = F.one_hot(init_seg.long(), init_seg.max() + 1)
    for merge_id in range(cluster.children_.shape[0]):
        new_segment = segment[:,:,cluster.children_[merge_id]+1].sum(dim = -1)
        segment[:,:,init_seg.max() + 1 + merge_id] = new_segment

    new_segment_list = []
    for seg_id in range(1, segment.shape[-1]):
        new_segment = segment[:,:,seg_id]
        all_labels = measure.label(new_segment.cpu())
        unique_labels = np.unique(all_labels)
        if np.unique(all_labels).shape[0] == 2:
            new_segment_list.append(new_segment[:,:,None].cuda())
        else:
            for unique_label in unique_labels:
                this_segment = torch.from_numpy(all_labels == unique_label).float()
                new_segment_list.append(this_segment[:,:,None].cuda())
    segment = torch.cat(new_segment_list, dim = -1)
    return segment
@torch.no_grad()
def compute_instance_segmentation(feat, cluster_list,
            rg_pixel_upper_bound = 0.9, rg_pixel_lower_bound = 0.01):
    _, num_of_eig, feat_h, feat_w = feat.shape
    feat_map = F.normalize(feat.reshape(num_of_eig, -1).cuda(), dim = -1).reshape(num_of_eig, feat_h, feat_w)
    feat_vect = feat_map.flatten(1,-1)
    feat_vect = (feat_vect - feat_vect.min(dim = -1, keepdims =True)[0])/(feat_vect.max(dim = -1, keepdims =True)[0] - feat_vect.min(dim = -1, keepdims =True)[0])
    feat_map = feat_vect.reshape(feat_map.shape)
    feat_map = feat_map * 2 -1

    feat = feat_map
    c,h,w = feat.shape
    feat_vect = feat.reshape(c, h * w).T
    segment_list = []
    for k in cluster_list:
        kmeans = KMeans(n_clusters=k).fit(feat_vect.cpu().data.numpy())
        # (h * w)
        kmeans_label = kmeans.labels_.reshape(h,w)
        all_labels = measure.label(kmeans_label)
        segment_list.append(hierarchical_clustering(all_labels, feat))
    num_pixels = feat_h * feat_w
    regions = torch.cat(segment_list,dim = -1)
    regions = regions[:,:,regions.sum(dim = [0,1]) >= rg_pixel_lower_bound * num_pixels]
    regions = regions[:,:,regions.sum(dim = [0,1]) <= rg_pixel_upper_bound * num_pixels]
    new_regions = remove_bg_mask(regions, 2)
    if new_regions.shape[-1] == 0:
        new_regions = remove_bg_mask(regions, 3)
    if new_regions.shape[-1] == 0:
        new_regions = regions
    regions = new_regions

    num_pixels = feat_h * feat_w
    score_list = []
    mini_batch = 30
    for idx in range((regions.shape[-1] + mini_batch - 1) // mini_batch):
        score = compute_binary_silhouette_score(feat, regions[...,mini_batch * idx: mini_batch * (idx + 1)], h, w)
        score_list.append(score)
    score = torch.cat(score_list)
    idx = score.argsort(descending = True)
    score = score[idx]
    regions = regions[:,:,idx]
    final_mask = []
    final_score = []
    for i in range(100):
        final_mask.append(regions[:,:,0].clone())
        final_score.append(score[0].item())
        new_regions, new_score = mask_nms(regions[:,:,0], regions[:,:,1:], score[1:], iou_th = 0.7)
        regions = new_regions
        score = new_score
        if regions.shape[-1] ==0:
            break
    return final_mask, final_score

def draw_inst_seg(img, mask_array, score_array = None, color_ratio = 0.4):
    pred_img = img.copy()
    for mid, mask in enumerate(mask_array):
        if score_array is not None and score_array[mid] < 0:
            continue
        color = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]
        mask = F.interpolate(mask[None,None], size = (img.shape[0], img.shape[1]), mode = 'nearest').cpu().data.numpy()[0,0]
        binary_mask_encoded = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
        area = maskUtils.area(binary_mask_encoded)
        if area == 0:
            continue
        bounding_box = maskUtils.toBbox(binary_mask_encoded)
        bbox = [int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[0] + bounding_box[2]), int(bounding_box[1] + bounding_box[3])]
        mask = mask > 0
        pred_img[mask] = pred_img[mask] * color_ratio + np.array(color, dtype=np.uint8) * (1- color_ratio)
        # Draw rectangles around all found objects
        pred_img = cv2.rectangle(pred_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    return pred_img
