import torch.nn as nn
import torch, pdb
import torch.distributed as dist
import numpy as np
import math
import torch.nn.functional as F
def gaussian(x, sigma=1.0):
    return np.exp(-(x**2) / (2*(sigma**2)))


def build_gauss_kernel(
        size=5, sigma=1.0, n_channels=1, device=None):
    """Construct the convolution kernel for a gaussian blur
    See https://en.wikipedia.org/wiki/Gaussian_blur for a definition.
    Overall I first generate a NxNx2 matrix of indices, and then use those to
    calculate the gaussian function on each element. The two dimensional
    Gaussian function is then the product along axis=2.
    Also, in_channels == out_channels == n_channels
    """
    if size % 2 != 1:
        raise ValueError("kernel size must be uneven")
    grid = np.mgrid[range(size), range(size)] - size//2
    kernel = np.prod(gaussian(grid, sigma), axis=0)
    # kernel = np.sum(gaussian(grid, sigma), axis=0)
    kernel /= np.sum(kernel)

    # repeat same kernel for all pictures and all channels
    # Also, conv weight should be (out_channels, in_channels/groups, h, w)
    kernel = np.tile(kernel, (n_channels, 1, 1, 1))
    kernel = torch.from_numpy(kernel).to(torch.float).to(device)
    return kernel


def blur_images(images, kernel):
    """Convolve the gaussian kernel with the given stack of images"""
    _, n_channels, _, _ = images.shape
    _, _, kw, kh = kernel.shape
    imgs_padded = F.pad(images, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
    return F.conv2d(imgs_padded, kernel, groups=n_channels)


def laplacian_pyramid(images, kernel, max_levels=5):
    """Laplacian pyramid of each image
    https://en.wikipedia.org/wiki/Pyramid_(image_processing)#Laplacian_pyramid
    """
    current = images
    pyramid = []

    for level in range(max_levels):
        filtered = blur_images(current, kernel)
        diff = current - filtered
        pyramid.append(diff)
        current = F.avg_pool2d(filtered, 2)
    pyramid.append(current)
    return pyramid


class LapLoss(nn.Module):
    def __init__(self, max_levels=5, kernel_size=5, sigma=1.0):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.kernel_size = kernel_size
        self.sigma = sigma
        self._gauss_kernel = None

    def forward(self, output, target):
        if (self._gauss_kernel is None
                or self._gauss_kernel.shape[1] != output.shape[1]):
            self._gauss_kernel = build_gauss_kernel(
                n_channels=output.shape[1],
                device=output.device)
        output_pyramid = laplacian_pyramid(
            output, self._gauss_kernel, max_levels=self.max_levels)
        target_pyramid = laplacian_pyramid(
            target, self._gauss_kernel, max_levels=self.max_levels)
        diff_levels = [F.l1_loss(o, t)
                        for o, t in zip(output_pyramid, target_pyramid)]
        loss = sum([2**(-2*j) * diff_levels[j]
                    for j in range(self.max_levels)])
        return loss

def batch_norm_without_running_stats(module, flag):
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.track_running_stats = flag

def replace_all_batch_norm_modules(root, flag):
    """
    In place updates :attr:`root` by setting the ``running_mean`` and ``running_var`` to be None and
    setting track_running_stats to be False for any nn.BatchNorm module in :attr:`root`
    """
    # base case
    batch_norm_without_running_stats(root, flag)

    for obj in root.modules():
        batch_norm_without_running_stats(obj, flag)
    return root

def process_sn(model, action = None):
    assert action in ['collect_sn', 'turn_on_sn', 'turn_off_sn']
    if action == 'collect_sn':
        singular_value = []
        for module in model.modules():
          if (isinstance(module, nn.Conv2d)
              or isinstance(module, nn.Linear)
              or isinstance(module, nn.Embedding)):
            if hasattr(module, "svs_list"):
              singular_value.append(module.svs_list[0])
        return singular_value
    if action == 'turn_on_sn':
        for module in model.modules():
          if (isinstance(module, nn.Conv2d)
              or isinstance(module, nn.Linear)
              or isinstance(module, nn.Embedding)):
            if hasattr(module, "svs_list"):
                module.sn_reg = True
                module.use_sn = True
    if action == 'turn_off_sn':
        for module in model.modules():
          if (isinstance(module, nn.Conv2d)
              or isinstance(module, nn.Linear)
              or isinstance(module, nn.Embedding)):
            if hasattr(module, "svs_list"):
                module.sn_reg = False
                module.use_sn = False
                module.svs_list = None

def process_mbn(model):
    bn_scaler= []
    for module in model.modules():
      if isinstance(module, MovingBatchNorm):
        if hasattr(module, "weight"):
          bn_scaler.append(module.weight.squeeze())
    return bn_scaler

class MaximalCodingRateReduction(torch.nn.Module):
    def __init__(self, eps=0.5, gamma=1):
        super(MaximalCodingRateReduction, self).__init__()
        self.eps = eps
        self.gamma = gamma

    def compute_discrimn_loss(self, W):
        """Discriminative Loss."""
        p, m = W.shape
        I = torch.eye(p,device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.

    def compute_compress_loss(self, W, Pi):
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p,device=W.device).expand((k,p,p))
        trPi = Pi.sum(2) + 1e-8
        scale = (p/(trPi*self.eps)).view(k,1,1)

        W = W.view((1,p,m))
        log_det = torch.logdet(I + scale*W.mul(Pi).matmul(W.transpose(1,2)))
        compress_loss = (trPi.squeeze()*log_det/(2*m)).sum()
        return compress_loss, log_det

    def forward(self, X, Y, num_classes=None):
        #This function support Y as label integer or membership probablity.
        if len(Y.shape)==1:
            #if Y is a label vector
            if num_classes is None:
                num_classes = Y.max() + 1
            Pi = torch.zeros((num_classes,1,Y.shape[0]),device=Y.device)
            for indx, label in enumerate(Y):
                Pi[label,0,indx] = 1
        else:
            #if Y is a probility matrix
            if num_classes is None:
                num_classes = Y.shape[1]
            Pi = Y.T.reshape((num_classes,1,-1))

        W = X.T
        discrimn_loss = self.compute_discrimn_loss(W)
        compress_loss, log_det = self.compute_compress_loss(W, Pi)

        total_loss = - discrimn_loss + self.gamma*compress_loss
        return total_loss, discrimn_loss * 2,  log_det[0], log_det[1]

def cov_reg(logdet, weight, hinge):
    return weight * F.relu(hinge - logdet)

def compute_mb_dist(args, feat1, feat2, eps = 0.5):
    m, p = feat1.shape
    mu1 = F.normalize(feat1.mean(dim = 0)[None,:], dim = -1)[0]
    mu2 = F.normalize(feat2.mean(dim = 0)[None,:], dim = -1)[0]
    feat = torch.cat([feat1, feat2], dim = 0)
    mu = F.normalize(feat.mean(dim = 0)[None,:], dim = -1)[0]
    diff = (mu1 - mu2)
    eye = torch.eye(p, device = feat1.device)
    cov1 = eye + p / eps  * (feat1 -  mu1[None,:]).T @ (feat1 - mu1[None,:]) / (m - 1)
    cov2 = eye +  p / eps * (feat2 - mu2[None,:]).T @ (feat2 - mu2[None,:]) / (m - 1)
    cov = eye + p / eps * (feat - mu[None,:]).T @ (feat - mu[None,:]) / (2 * m - 1)
    mdis_dist = diff @ torch.linalg.inv(cov) @ diff.T
    first_term = mdis_dist
    sum_detcov = torch.logdet(cov)
    detcov1 = torch.logdet(cov1)
    detcov2 = torch.logdet(cov2)
    second_term = sum_detcov - 0.5 * (detcov1 +  detcov2)
    mb = 1.0 / 8 * first_term + 0.5 * second_term
    return -1 * mb, sum_detcov, detcov1, detcov2

def mcr_loss(feat, rec_feat, args):
    total_feat = torch.cat([feat, rec_feat], dim = 0)
    label = total_feat.new_zeros((total_feat.shape[0], 2))
    label[:feat.shape[0], 0] = 1
    label[feat.shape[0]:, 1] = 1
    D_loss = MaximalCodingRateReduction()(total_feat, label, args)
    return D_loss

class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.raw_val = []

    def reset(self):
        self.raw_val = []

    def update(self, val):
        self.val = val
        self.raw_val.append(val)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        data = np.array(self.raw_val)
        return fmtstr.format(name = self.name, val = self.val, avg = data.mean())


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.learning_rate * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
