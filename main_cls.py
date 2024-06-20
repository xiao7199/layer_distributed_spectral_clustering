import argparse
import os, json
import random
import shutil
import time, glob, copy
import os
import time
import torch
import socket
import argparse
import subprocess
import math
from tqdm import tqdm
import warnings
import numpy as np
import torch, pdb
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import argparse, pdb
import numpy as np
from torch import autograd
from torch.optim import Adam, SGD, AdamW
from typing import Union, List, Optional, Sequence, Dict, Iterator, Tuple, Callable
import pdb
from utils import AverageMeter,ProgressMeter
from model_seg import SegNet
from dataloader import SimpleDataLoader
import builtins
import torchvision.utils as vutils
from PIL import Image
from diffusion_extractor import StableDiffusion
from PIL import ImageFilter, ImageOps
from train_utils import subsample_feat_and_solve_ncut_semantic, save_checkpoint

#from sklearn.metrics import average_precision_score
warnings.filterwarnings("ignore")

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--img_size', default=256, type=int,
                    help='img size')
parser.add_argument('--batch_iter', default=48, type=int,
                    help='img size')
parser.add_argument('--output_dim', default=10, type=int,
                    help='output dimensionality of features that solve ncut loss')
parser.add_argument('--temp1', default=1.0, type=float,
                    help='parameter_1')
parser.add_argument('--temp2', default=1.0, type=float,
                    help='parameter_2')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save_freq', default=5, type=int,
                     help='print frequency (default: 10)')
parser.add_argument('--resume', action = 'store_true',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--setting', default='0_0_0', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='local rank for distributed training')
parser.add_argument('--local-rank', default=-1, type=int,
                    help='local rank for distributed training')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# ae params
parser.add_argument('--huggingface_token', type=str, default='', help='huggingface API token to load stable diffusion')
parser.add_argument('--train_steps', type=int, default=1300, help='number of optimization steps')
parser.add_argument('--avg_steps', type=int, default=4, help='number of optimization steps')
# model args
parser.add_argument('--symmetric_matrix', type=int, default=0, choices=[0,1], help='1 symmetrical normalized randomwalk matrix,\
        0 for conventional attention matrix')
parser.add_argument('--vit_patch_size', type=int, default=8, choices=[8,16], help = 'the patch size in DINO VIT')
parser.add_argument('--vit_model_arch', type=str, default='vit_base', choices=['vit_base', 'vit_small'], help = 'the model arch in DINO VIT')
parser.add_argument('--vit_resize_img_size', type = int, default = 480, help = 'the smaller edge of input image for VIT')
# t sampling
parser.add_argument('--noise_schedule', type=str, default='random', choices=['random', 'constant', 'increasing', 'decreasing', 'cyclic'], help='noise schedule to use for training')
parser.add_argument('--noise_min_t', type=int, default=0, help='minimum t to use in diffusion model')
parser.add_argument('--noise_max_t', type=int, default=300, help='maximum t to use in diffusion model')
parser.add_argument('--noise_periods', type=float, default=1, help='periods for cyclic noise schedule')
parser.add_argument('--noise_sampling', action='store_true', default=True, help='if true, sample noise random uniformly from below maximum value defined by schedule')
# temp scheduling
parser.add_argument('--num_of_eig', type=int, default=5, help='number of eigenvector to compute')
parser.add_argument('--eig_loss_weight', type=float, default=2, help='weight ratio for primary objective')
parser.add_argument('--ortho_loss_weight', type=float, default=2, help='weight ratio for orthogonal regularization')
# attn buffer
parser.add_argument('--use_buffer_prob', type=float, default=None, help='chance to use buffer')
parser.add_argument('--attn_buffer_size', type=int, default=5, help='attn buffer size')
parser.add_argument('--num_of_image_feat_table', type=int, default=200, help='attn buffer size')
# optimizer param
parser.add_argument("--data_type", type=str)  # adam/lamb
parser.add_argument("--load_checkpoint", type=str)  # adam/lamb
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--solver_iters", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--vv_graph", action = 'store_true', default = False)
parser.add_argument("--sparse_graph", action = 'store_true', default = False)
parser.add_argument("--use_intra_inter_split", action = 'store_true', default = False)
parser.add_argument("--num_of_intra_pixel", type=int, default=1)
parser.add_argument("--num_of_inter_pixel", type=int, default=1)
parser.add_argument("--weight_decay", type=float, default=0)

# training params
parser.add_argument("--gpus", type=int, default=1)
parser.add_argument("--subsample_pixel_num", type=int, default=1e10)

# datamodule params
parser.add_argument("--data_path", type=str, default=".")
parser.add_argument("--stable_diffusion_cache_path", type=str, default=".")
parser.add_argument("--dataset", type=str, default="cifar10")  # cifar10, stl10, imagenet

# transforms param

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

best_acc1 = 0
EPS = 1e-20
def pca_feat(feat, pca_dim = 3):
    device = feat.device
    N,C,H,W = feat.shape
    feat = feat.reshape(N,C,-1).permute(0,2,1)
    pca_dim = min(min(pca_dim, feat.shape[1]), feat.shape[2])
    feat = feat.reshape(1, -1, C)
    [u,s,v] = torch.pca_lowrank(feat, pca_dim, niter = 2)
    u = u.reshape(N, H * W, -1)
    v = v.reshape(1, C, -1)
    feat = feat.reshape(N, -1, C)
    #v = multi_grid_low_rank(feat, pca_dim, niter = 1)
    feat = torch.matmul(feat, v)
    feat = feat.reshape(N,H,W,pca_dim).permute(0,3,1,2).contiguous()
    feat_min = feat.reshape(feat.shape[0], pca_dim, -1).min(dim = -1)[0]
    feat_max = feat.reshape(feat.shape[0], pca_dim, -1).max(dim = -1)[0]
    feat = 255 * (feat - feat_min.reshape(-1,pca_dim,1,1)) / (feat_max - feat_min).reshape(-1,pca_dim,1,1)
    return feat

def main():
    import os
    #torch.backends.cudnn.benchmark=False
    cudnn.deterministic = True
    args = parser.parse_args()
    #assert args.batch_size % args.batch_iter == 0
    if not os.path.exists('checkpoint'):
        os.system('mkdir checkpoint')
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size >= 1
    ngpus_per_node = torch.cuda.device_count()

    print('start')
    if args.distributed:
        if args.local_rank != -1: # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    args.gpu = args.gpu % torch.cuda.device_count()
    print('world_size', args.world_size)
    print('rank', args.rank)
    # suppress printing if not on master gpu
    if args.rank!=0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    args.distributed = args.world_size >= 1 or args.multiprocessing_distributed
    main_worker(args.gpu, ngpus_per_node, args)

def visualize_eigenvector(img, feat, num_of_row, num_of_col, vis_res, num_of_eig, img_path):
    feat = F.interpolate(feat, size = (vis_res, vis_res), mode = 'bilinear', antialias = True)
    img= F.interpolate(img, size = (vis_res, vis_res), mode = 'bilinear', antialias = True)
    num_of_eig = min(feat.shape[1], num_of_eig)
    n,_,h,w = feat.shape
    assert num_of_row * num_of_col == n
    feat = F.normalize(feat.permute(0,2,3,1).flatten(0,2), dim = -1)
    u,s,v = torch.svd(feat)
    for i in range(num_of_eig // 3):
        subfeat = u[:,i * 3:(i+1) * 3]
        feat_min = subfeat.min(dim = 0)[0]
        feat_max = subfeat.max(dim = 0)[0]
        if subfeat.shape[-1] != 3:
            break
        subfeat = subfeat.reshape(n, h, w, 3).permute(0,3,1,2)
        feat = (subfeat - feat_min.reshape(-1,3,1,1)) / (feat_max - feat_min).reshape(1,3,1,1)
        feat = feat.reshape(num_of_row, num_of_col, 3, vis_res, vis_res).permute(0,3, 1, 4, 2).reshape(num_of_row * vis_res, num_of_row * vis_res, 3)
        feat = (feat * 255).cpu().data.numpy().astype(np.uint8)
        Image.fromarray(feat).save(img_path + f'/eig_{3 * i}-{3 * (i + 1)}.png')

    img = img.reshape(num_of_row, num_of_col, 3, vis_res, vis_res).permute(0,3, 1, 4, 2).reshape(num_of_row * vis_res, num_of_row * vis_res, 3)
    Image.fromarray((img * 255).cpu().data.numpy().astype(np.uint8)).save(img_path + f'/img.png')

def main_worker(gpu, ngpus_per_node, args):
    args = copy.deepcopy(args)
    args.cos = True
    save_folder_path = '''checkpoint/lr_{}_si_{}_batch_size_{}_weight_decay_{}_out_dim_{}_vv{}_iisplit{}_{}'''.replace('\n',' ').replace(' ','').format(
                        args.learning_rate, args.solver_iters, args.batch_size, args.weight_decay, args.output_dim, args.vv_graph, args.use_intra_inter_split,
                        args.num_of_inter_pixel)
    args.save_folder_path = save_folder_path
    args.is_master = args.rank == 0
    diffusion = StableDiffusion(args, ncut_weight = 1, orth_weight = 1)
    diffusion.collect_attention_and_feat(torch.randn(1, 3, 512, 512).float().to(args.gpu))
    diffusion.attn_buffer.clear()

    image_transform = transforms.Compose([
        transforms.Resize(256),
        torchvision.transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    train_dataset = SimpleDataLoader(args.data_path, image_transform)
    args.num_of_image_feat_table = len(train_dataset)

    model = SegNet(args, args.output_dim)
    model.cuda(args.gpu)

    ema_model = SegNet(args, args.output_dim)
    ema_model.cuda(args.gpu)


    scaler = torch.cuda.amp.GradScaler()

    torch.cuda.set_device(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True,
    broadcast_buffers=False)
    copy_param_to_ema_model(model, ema_model)

    optimizer = AdamW(model.parameters(),
                lr=  args.learning_rate, weight_decay = args.weight_decay)
    args.global_step_counter = 0
    args.start_epoch = 0
    if args.resume or len(args.load_checkpoint) > 0:
        if args.load_checkpoint is not None and len(args.load_checkpoint) > 0:
            args.resume = args.load_checkpoint
        else:
            args.resume = '{}/last.pth.{}_{}.tar'.format(save_folder_path, args.temp1, args.temp2)
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            ema_model.load_state_dict(checkpoint['ema_state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.global_step_counter = checkpoint['global_step_counter']
            print("=> loaded checkpoint '{}' (iter {})"
                  .format(args.resume, checkpoint['global_step_counter']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    print('NUM of training images: {}'.format(len(train_dataset)))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle = True, drop_last = True)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last = True, persistent_workers = True)

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            and args.is_master):
        if not os.path.exists(save_folder_path):
            os.system('mkdir -p {}'.format(save_folder_path))
    iter_per_epoch = len(train_loader)
    total_iter = (args.epochs * iter_per_epoch)
    print('iter per epoch:{}, total_iter:{}'.format(iter_per_epoch, total_iter))

    while True:
        epoch = args.global_step_counter // iter_per_epoch
        if args.distributed:
            train_sampler.set_epoch(epoch)
        input_scaler = None
        train(diffusion, train_loader, model, ema_model, optimizer, epoch, args, ngpus_per_node, input_scaler)
        if args.global_step_counter >= total_iter:
            break
        if epoch >= 190:
            break

    raw_feat = ema_model(torch.arange(100).to(args.gpu)).softmax(dim = 1)
    raw_img = torch.stack([train_loader.dataset[i][1] for i in range(100)])
    visualize_eigenvector(raw_img, raw_feat, 10, 10, 64, 15, f'{args.save_folder_path}/img_out')

def train(diffusion, train_loader, model, ema_model, optimizer, epoch, args, ngpus_per_node, scaler = None):
    loss_name = [
                'ortho', 'eig_val',
                'GPU Mem', 'Time', 'current_iter']
    moco_loss_meter = [AverageMeter(name, ':6.3f') for name in loss_name]
    progress = ProgressMeter(
        len(train_loader),
        moco_loss_meter,
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    t0 = time.time()


    for img_index, img_diffusion in tqdm(train_loader):
        args.global_step_counter += 1
        img_diffusion = img_diffusion.cuda(args.gpu, non_blocking = True)
        query_list, key_list, value_list = diffusion.collect_attention_and_feat(img_diffusion)
        raw_feat = model(img_index)
        A_list = model.module.build_KNN_graph(query_list, key_list, value_list)
        target, eigval, ortho, input_eig, input_ortho = subsample_feat_and_solve_ncut_semantic(model, img_index, A_list, optimizer,
        diffusion.feat_res_list, iteration = args.solver_iters)
        t1 = time.time()
        for val_id, val in enumerate([ortho.abs().mean(), eigval.mean(),
                    torch.cuda.max_memory_allocated() / (1024.0 * 1024.0), t1 - t0,  args.global_step_counter
                    ]):
            if not isinstance(val, float) and not isinstance(val, int):
                val = val.item()
            moco_loss_meter[val_id].update(val)
        progress.display(args.global_step_counter)
        update_ema_model(model, ema_model, 0.99)
        if args.global_step_counter % 20 == 0 and args.gpu == 0:
            num_img_to_display = 18
            raw_feat = ema_model(img_index[:num_img_to_display].to(args.gpu))
            pred_seg_rgb = F.interpolate(pca_feat(raw_feat.softmax(dim = 1), 4)[:,1:], size = (256, 256), mode = 'bilinear').permute(0,2,3,1).cpu().data.numpy().astype(np.uint8)
            img_norm = (((img_diffusion[:num_img_to_display].reshape(-1, 3, 256, 256).cpu().data.numpy().transpose(0,2,3,1).reshape(-1, 256, 3))).clip(0, 1) * 255).astype(np.uint8)
            img_and_pca = np.concatenate([img_norm, pred_seg_rgb[:args.batch_size].reshape(-1,256,3)], axis = 1)
            img_and_pca = img_and_pca.reshape(6, 3, 256, 512, 3).transpose(0,2, 1, 3, 4).reshape(6 * 256, 6 * 256, 3)
            _path = f'{args.save_folder_path}/img_out/pca_{args.global_step_counter:06d}.png'
            os.makedirs(os.path.dirname(_path), exist_ok=True)
            Image.fromarray(img_and_pca).save(_path)
        model.train()

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            and args.is_master):
        if epoch % 10 == 0 or epoch == 190:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'global_step_counter': args.global_step_counter,
            }, False, filename = '{}/last.pth.{}_{}.tar'.format(args.save_folder_path, args.temp1, args.temp2))
    # to syncrhonize the process across GPUs
    torch.distributed.barrier()
    torch.cuda.reset_max_memory_allocated()

if __name__ == '__main__':
    main()
