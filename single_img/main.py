import torch, pdb, time,os
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from model.diffusion_extractor import StableDiffusion
from model.vit_extractor import MAEFeatureExtractor, DINOFeatureExtractor
from eval_feat import compute_instance_segmentation, draw_inst_seg, pca_feat, auto_kmeans_feat
from train_utils import ncut_loss, ToyCNN, SimpleDataset
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2)
parser.add_argument('--train_steps', type=int, default=1300, help='number of optimization steps')
parser.add_argument('--gpu', type=int, default=0, help='gpu id')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate for optimization')
parser.add_argument('--accum_grads', type=int, default=20, help='number of steps to accumulate gradients for')
parser.add_argument('--model', type=str, default='stable_diffusion', choices = ['stable_diffusion', 'mae', 'dino'], help='model to extract attention')
parser.add_argument('--huggingface_token', type=str, default='', help='huggingface API token to load stable diffusion')
parser.add_argument('--img_path', type=str, default='', help='the path of input image')

# model args
parser.add_argument('--symmetric_matrix', type=int, default=0, choices=[0,1], help='1 symmetrical normalized randomwalk matrix,\
        0 for conventional attention matrix')
parser.add_argument('--vit_patch_size', type=int, default=8, choices=[8,16], help = 'the patch size in DINO VIT')
parser.add_argument('--vit_model_arch', type=str, default='vit_base', choices=['vit_base', 'vit_small'], help = 'the model arch in DINO VIT')
parser.add_argument('--vit_resize_img_size', type = int, default = 480, help = 'the smaller edge of input image for VIT')
# t sampling
parser.add_argument('--noise_schedule', type=str, default='random', choices=['random', 'constant', 'increasing', 'decreasing', 'cyclic'], help='noise schedule to use for training')
parser.add_argument('--noise_min_t', type=int, default=0, help='minimum t to use in diffusion model')
parser.add_argument('--noise_max_t', type=int, default=999, help='maximum t to use in diffusion model')
parser.add_argument('--noise_periods', type=float, default=1, help='periods for cyclic noise schedule')
parser.add_argument('--noise_sampling', action='store_true', default=True, help='if true, sample noise random uniformly from below maximum value defined by schedule')
# temp scheduling
parser.add_argument('--num_of_eig', type=int, default=15, help='number of eigenvector to compute')
parser.add_argument('--eig_loss_weight', type=float, default=2, help='weight ratio for primary objective')
parser.add_argument('--ortho_loss_weight', type=float, default=2, help='weight ratio for orthogonal regularization')
# attn buffer
parser.add_argument('--use_buffer_prob', type=float, default=None, help='chance to use buffer')
parser.add_argument('--attn_buffer_size', type=int, default=5, help='attn buffer size')
args = parser.parse_args()

def copy_param_to_ema_model(model, ema_model):
    for param_q, param_k in zip(
        model.parameters(), ema_model.parameters()
    ):
        param_k.data.copy_(param_q.data)  # initialize
        param_k.requires_grad = False  # not update by gradient

def update_ema_model(model, ema_model, m):
    for param_q, param_k in zip(
        model.parameters(), ema_model.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1.0 - m)

def main():
    # initialize and config the model
    if args.model == 'stable_diffusion':
        attention_extractor = StableDiffusion(args)
    elif args.model == 'mae':
        attention_extractor = MAEFeatureExtractor(args)
    elif args.model == 'dino':
        attention_extractor = DINOFeatureExtractor(args)

    rgb_img = np.array(Image.open(args.img_path))
    img = (torch.from_numpy(rgb_img).float() / 255)[None,...].permute(0,3,1,2)
    img = img.to(args.gpu)
    input_feats = attention_extractor.process_input(img)
    toy_cnn = ToyCNN(attention_extractor.feat_h, attention_extractor.feat_w, int(args.num_of_eig))
    toy_cnn.to(args.gpu)
    optim = torch.optim.Adam(toy_cnn.parameters(), lr=args.learning_rate)

    # main optimization loop
    for i in tqdm(range(args.train_steps + 1)):
        ortho_list = []
        eigval_list = []
        attns = attention_extractor.collect_attention(input_feats, i)
        dense_feat = toy_cnn().softmax(dim = 1)
        for attn in attns:
            eigval, ortho = ncut_loss(attn, dense_feat, args.symmetric_matrix)
            eigval_list.append(eigval)
            ortho_list.append(ortho)
        eigval = torch.cat(eigval_list).reshape(-1)
        ortho = torch.cat(ortho_list).reshape(-1)
        loss = args.eig_loss_weight * (eigval - (1 - args.symmetric_matrix)).abs().mean() + args.ortho_loss_weight * ortho.pow_(2).mean()
        loss.backward()
        if i % args.accum_grads == 0 and i != 0:
            optim.step()
            optim.zero_grad()
    # release the attention hook
    attention_extractor.clear_after_loop()

    # Collect final dense feature
    dense_feat = toy_cnn().softmax(dim = 1)
    u,s,v = F.normalize(dense_feat.flatten(2,-1), dim = -1)[0].T.svd()
    reorth_dense_feat = torch.mm(u, torch.diag(s)).T
    dense_feat = reorth_dense_feat.reshape(dense_feat.shape)
    eigenvector = u.T.reshape(dense_feat.shape)

    ## compute the instance segmentation
    inst_mask, inst_score = compute_instance_segmentation(dense_feat, np.arange(2,10))
    ## visualize instsance segmentation over image
    img_with_inst = draw_inst_seg(rgb_img, inst_mask, inst_score, color_ratio = 0.4)

    # project feature map onto 3 dimensional features, by excluding the first eigenvector, for visualization
    pca_img = pca_feat(dense_feat, 4, normalize_as_img = True)[0,:,:,1:]
    fig, ax = plt.subplots(2,2)

    # Kmeans segmentation with K automatically determined by silhouette scores
    auto_kmeans = auto_kmeans_feat(dense_feat[0], max_n_clusters = 10)[0]

    #visualization
    ax = ax.reshape(-1)
    ax[0].imshow(rgb_img)
    ax[0].set_title('input img')
    ax[0].set_axis_off()
    ax[1].imshow(pca_img)
    ax[1].set_title('PCA')
    ax[1].set_axis_off()
    ax[2].imshow(img_with_inst)
    ax[2].set_title('Instance Segmentation')
    ax[2].set_axis_off()
    ax[3].imshow(auto_kmeans, cmap = 'Set1')
    ax[3].set_title('Image Segmentation')
    ax[3].set_axis_off()
    plt.tight_layout()
    plt.savefig('output.png')
    plt.close()

    eig_to_show_row = 3
    eig_to_show_col = 4
    fig, ax = plt.subplots(eig_to_show_row,eig_to_show_col)
    ax = ax.reshape(-1)
    for i in range(eig_to_show_col * eig_to_show_row):
        ax[i].imshow(eigenvector[0,i].cpu().data.numpy(), cmap = 'gray')
        ax[i].set_title(f'Eigevec:{i}')
        ax[i].set_axis_off()
    plt.savefig('eigevector.png')
    plt.close()

if __name__ == '__main__':
    main()
