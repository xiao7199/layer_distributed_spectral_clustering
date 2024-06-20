import numpy as np
import torch
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import torch,pdb
import torch.nn.functional as F
import torchvision.transforms.functional as VF


def upsample_img_vect(vect, img_res, final_res):
    n,c = vect.shape
    num_of_img = n // (img_res * img_res)
    vect = vect.reshape(num_of_img, img_res, img_res, c).permute(0,3,1,2)
    vect = F.interpolate(vect, size = (final_res, final_res), mode = 'bilinear')
    return vect.permute(0, 2,3,1).reshape(-1, c)

@torch.no_grad()
def eval_semantic_segmentation(dataloader, model, device = 0, epoch = 0, use_crf = False):
    num_of_gt_label = dataloader.dataset.n_classes
    color_palette = color_map()
    prob = model(torch.arange(len(model.image_feat_lookup_table)).to(device)).softmax(dim = 1)
    num_of_feat, num_of_channel, feat_h, feat_w = prob.shape

    out = torch.svd_lowrank(prob.permute(0,2,3,1).reshape(-1,prob.shape[1]), q = min(100, prob.shape[1]), niter = 10)
    data_for_cluster = out[0] * out[1][None,:]
    data_for_cluster = data_for_cluster[:, :num_of_gt_label].cpu().data.numpy()

    kmeans_label = KMeans(n_clusters=num_of_gt_label, random_state=0).fit(data_for_cluster).labels_.reshape(prob.shape[0], feat_h, feat_w)
    kmeans_label = torch.from_numpy(kmeans_label).to(device)
    torch.distributed.broadcast(kmeans_label, src = 0)
    kmeans_label = kmeans_label.cpu().data.numpy()
    label_matching = torch.zeros(num_of_gt_label, num_of_gt_label + 1).to(device)
    for (img_index, img_diffusion, label, _) in dataloader:
        gt_label = label + 1
        up_kmeans_label = F.interpolate(torch.from_numpy(kmeans_label[img_index.cpu().long()]).to(device).float()[:,None], size = (label.shape[2], label.shape[3]), mode = 'nearest').long()
        hist = torch.bincount(((up_kmeans_label.to(device)) * (num_of_gt_label + 1) + gt_label.to(device)).reshape(-1).long(), minlength = (num_of_gt_label * (num_of_gt_label + 1)))
        label_matching += hist.reshape(label_matching.shape)
    torch.distributed.all_reduce(label_matching)
    for matching_mode in ['hungarian', 'greedy']:
        if matching_mode == 'hungarian':
            pred_label = torch.from_numpy(linear_sum_assignment(label_matching[:,1:].cpu().data.numpy(), maximize = True)[1][kmeans_label])
        else:
            pred_label = label_matching[:,1:].argmax(dim = -1)[kmeans_label]
        hist = torch.zeros((num_of_gt_label) ** 2).to(device)
        for (img_index, img_diffusion, label, _) in dataloader:
            if use_crf:
                pred = batch_crf(img_diffusion, pred_label[img_index], num_of_gt = 27)[:,None].long().to(device)
            else:
                pred = F.interpolate(pred_label[img_index][:,None].float(), size = (label.shape[2], label.shape[3]), mode = 'nearest').long().to(device)
            mask = (label >= 0).to(device)
            hist += torch.bincount(((pred[mask]) * (num_of_gt_label) + label.to(device)[mask]).reshape(-1).long(), minlength = (num_of_gt_label) ** 2)
        hist = hist.reshape(num_of_gt_label, num_of_gt_label).to(device)
        torch.distributed.all_reduce(hist)
        tp = torch.diag(hist)
        fp = torch.sum(hist, dim=0) - tp
        fn = torch.sum(hist, dim=1) - tp
        iou = (tp / (fp + fn + tp + 1e-10))
        print(f'matching_mode: {matching_mode}, epoch: {epoch}, miou: {iou.mean()}')

#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

POS_W = 3
POS_XY_STD = 1
Bi_W = 4
Bi_XY_STD = 67
Bi_RGB_STD = 3
BGR_MEAN = np.array([104.008, 116.669, 122.675])


def dense_crf(image_tensor: torch.FloatTensor, output_logits: torch.FloatTensor, MAX_ITER=10):
    image = np.array(VF.to_pil_image(image_tensor))[:, :, ::-1]
    H, W = image.shape[:2]
    image = np.ascontiguousarray(image)

    output_logits = F.interpolate(output_logits.unsqueeze(0), size=(H, W), mode="bilinear",
                                  align_corners=False).squeeze()
    output_probs = F.softmax(output_logits, dim=0).cpu().numpy()

    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=image, compat=Bi_W)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))
    return Q

def batch_crf(img, pred, num_of_gt):
    #pred: N * C * H * W
    img_h, img_w = img.shape[2:]
    pred_one_hot = F.one_hot(pred, num_of_gt).permute(0,3,1,2).float()
    pred_map = F.interpolate(pred_one_hot, size = (img_h, img_w), mode = 'bilinear')
    out = []
    for i in range(img.shape[0]):
        out.append(dense_crf(img[i].cpu(), pred_map[i].cpu().float()).argmax(axis = 0))
    return torch.from_numpy(np.stack(out)).to(img.device)

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap
