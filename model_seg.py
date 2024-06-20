import torch, pdb
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import matplotlib.pyplot as plt
import math
import numpy as np
from time import time

@torch.no_grad()
def topk_memory_efficient(query, key, batch_size = 1024, remove_diag = False, intra_mask = None, num_of_inter_pixel = -1, num_of_intra_pixel = -1):
    query_shape = query.shape[:-1]
    q_size = query_shape.numel()
    k_size = key.shape[0]
    iter = (q_size + batch_size - 1) //  batch_size
    topk_sim_list = []
    topk_id_list = []
    for i in range(iter):
        query_batch = query[batch_size * i :batch_size * (i + 1)]
        sim = query_batch @ key.T
        if remove_diag:
            id = batch_size * i + torch.arange(query_batch.shape[0])
            sim[torch.arange(query_batch.shape[0]), id] = -1e5
        if intra_mask is not None:
            #intra_mask[i, j] = 1 means i,j are intra pixels
            # query_batch_size * k_size
            mask_batch = intra_mask[batch_size * i :batch_size * (i + 1)]
            topk_sim_intra, topk_id_intra =  (sim * mask_batch).topk(num_of_intra_pixel, dim = -1, largest = True)
            topk_sim_inter, topk_id_inter =  (sim * (~mask_batch)).topk(num_of_inter_pixel, dim = -1, largest = True)
            topk_id = torch.cat([topk_id_inter, topk_id_intra], dim = -1)
            topk_sim = torch.cat([topk_sim_inter, topk_sim_intra], dim = -1)
        else:
            topk_sim, topk_id =  sim.topk(num_of_inter_pixel + num_of_intra_pixel, dim = -1, largest = True)
        topk_sim_list.append(topk_sim)
        topk_id_list.append(topk_id)
    return torch.cat(topk_sim_list).reshape(query_shape + (-1,)).float(), torch.cat(topk_id_list).reshape(query_shape + (-1,))

@torch.no_grad()
def build_knn_graph(query_feat, key_feat, value_feat, num_of_inter_pixel, num_of_intra_pixel, num_of_sample_per_img, use_intra_inter_split, vv_graph=False, sparse_graph = True):
    N = query_feat.shape[0]
    #topk = min(topk, N)
    num_of_intra_pixel = min(num_of_intra_pixel, N)
    num_of_inter_pixel = min(num_of_inter_pixel, N)
    if vv_graph:
        knn_q = F.normalize(value_feat, dim = -1)
        knn_k = knn_q
    else:
        knn_q = F.normalize(query_feat, dim = -1)
        knn_k = F.normalize(key_feat, dim = -1)
    if use_intra_inter_split:
        #### Compute mask
        pixel_num = num_of_sample_per_img
        batch_size = N // pixel_num
        intra_mask = torch.zeros(N, N, dtype = torch.bool).to(query_feat.device)
        intra_mask = intra_mask.reshape(batch_size, pixel_num, batch_size, pixel_num)
        intra_mask[torch.arange(batch_size), :, torch.arange(batch_size), :] = 1
        intra_mask = intra_mask.reshape(-1, pixel_num * batch_size)
        ####
    else:
        intra_mask = None

    topk_sim, topk_id = topk_memory_efficient(knn_q, knn_k, remove_diag = True, intra_mask = intra_mask,
                        num_of_inter_pixel = num_of_inter_pixel, num_of_intra_pixel = num_of_intra_pixel)
    topk_sim[topk_sim <= 0.1] = 0
    row_id = torch.arange(topk_id.shape[0], device = query_feat.device)[:,None].expand(-1, topk_id.shape[1])
    affinity = torch.sparse_coo_tensor(torch.stack([row_id, topk_id], dim = -1).reshape(-1, 2).T, topk_sim.reshape(-1), (N, N),device = query_feat.device)
    if sparse_graph:
        affinity = 0.5 * (affinity.T + affinity)
        D = 1.0 / affinity.sum(dim = -1, keepdim = True).to_dense().reshape(-1) + 1e-6
    else:
        affinity = affinity.to_dense()
        affinity = 0.5 * (affinity.T + affinity)
        D = 1.0 / affinity.sum(dim = -1, keepdim = True).reshape(-1) + 1e-6
    A = affinity * D.reshape(-1, 1)
    return A

class SegNet(nn.Module):
    def __init__(self, args, num_of_layer):
        super().__init__()
        self.args = args
        self.image_feat_lookup_table = nn.ParameterList([nn.Parameter(torch.randn(num_of_layer, 32, 32).float()) for i in range(args.num_of_image_feat_table)])

    def build_KNN_graph(self, query_list, key_list, value_list):
        A_list = []
        logit_list = []
        for fid, (query, key, value) in enumerate(zip(query_list, key_list, value_list)):
            num_of_sample_per_img = query.shape[2]
            num_of_head = query.shape[1]
            query = query.permute(1,0,2,3).flatten(1,2)
            key = key.permute(1,0,2,3).flatten(1,2)
            value = value.permute(1,0,2,3).flatten(1,2)
            for hid in range(num_of_head):
                A = build_knn_graph(query[hid], key[hid],
                                    value[hid],
                                    num_of_inter_pixel = self.args.num_of_inter_pixel,
                                    num_of_intra_pixel = self.args.num_of_intra_pixel,
                                    num_of_sample_per_img = num_of_sample_per_img,
                                    vv_graph=self.args.vv_graph,
                                    use_intra_inter_split = self.args.use_intra_inter_split,
                                    sparse_graph = self.args.sparse_graph)
                A_list.append(A)

        return A_list

    def forward(self, img_index):
        return torch.stack([self.image_feat_lookup_table[i] for i in img_index])
