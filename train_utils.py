import numpy as np
import torch,pdb
import torch.nn.functional as F

def off_diagonal(x):
    b, n, m = x.shape
    assert n == m
    return x.flatten(1,-1)[:, :-1].view(b, n - 1, n + 1)[:, :, 1:]

def print_gradients(model, print_str):
    norms = []
    max_grad = 0
    max_norm = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            max_grad = max(max_grad, p.grad.norm())
            max_norm = max(max_norm, p.data.norm(2))
    return max_grad, max_norm

def subsample_feat(pred_logit, ds_size, sample_index):
    ds_pred_logit = F.adaptive_avg_pool2d(pred_logit, ds_size)
    sample_vect = torch.gather(ds_pred_logit.flatten(2,-1), index = sample_index[:,None,:].expand(-1, ds_pred_logit.shape[1], -1), dim = -1)
    return sample_vect.permute(0,2,1).flatten(0,1)

def subsample_feat_and_solve_ncut_semantic(model, image_index, A_list, optimizer, res_list, lr = 1e-2, iteration = 1000):
    input_eig = 0
    input_ortho = 0
    num_of_head = len(A_list) // len(res_list)
    for idx in range(iteration):
        eigval_list = []
        ortho_list = []
        weight_list = []
        ortho_weight_list = []
        feat_param = model(image_index)
        feat_prob = feat_param.softmax(dim = 1)
        def downsample_feat(pred_logit, ds_size):
            ds_pred_logit = F.adaptive_avg_pool2d(pred_logit, ds_size)
            return ds_pred_logit.permute(0,2,3,1).flatten(0,2)[:,None].expand(-1, num_of_head, -1)
        downsample_feat_list = []
        for res in res_list:
            downsample_feat_list += downsample_feat(feat_prob, res).chunk(num_of_head, dim = 1)

        # subsample heads
        subsample_heads = False
        if subsample_heads:
            head_ids = torch.randint(0, num_of_atten_head, size=(len(res_list),)).tolist()
            head_id_ix = 0

        for A_id, (prob, A) in enumerate(zip(downsample_feat_list, A_list)):
            if subsample_heads:
                # skip graph if not sampled
                if A_id % num_of_atten_head != head_ids[head_id_ix]:
                    if A_id % num_of_atten_head == num_of_atten_head - 1:
                        # increment on last head of layer
                        head_id_ix += 1
                    continue
                if A_id % num_of_atten_head == num_of_atten_head - 1:
                    # increment on last head of layer
                    head_id_ix += 1
            norm_prob = F.normalize(prob, dim = 0).squeeze()
            sigval = ((A @ norm_prob) * norm_prob).sum(dim = 0)
            offdiagval = off_diagonal((norm_prob.T @ norm_prob)[None,...])
            eigval_list.append(sigval)
            ortho_list.append(offdiagval)
            weight_list.append(torch.ones_like(sigval) * 32 / res_list[A_id])
            ortho_weight_list.append(torch.ones_like(offdiagval) * 32 / res_list[A_id])
        eigval = torch.cat(eigval_list).reshape(-1)
        ortho = torch.cat(ortho_list).reshape(-1)
        weight = torch.cat(weight_list).reshape(-1)
        ortho_weight = torch.cat(ortho_weight_list).reshape(-1)
        loss = (weight * (eigval - 1)).abs().mean() + (ortho_weight * (ortho).pow(2)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print(eigval.mean().abs(), ortho.pow(2).mean().abs())
        if idx == 0:
            input_eig = eigval
            input_ortho = ortho
    return feat_param, eigval, ortho, input_eig, input_ortho
# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
