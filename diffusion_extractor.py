import torch, math, random
from tqdm import tqdm
import pdb,os
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F

from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, DDPMScheduler
from diffusers.models.attention import CrossAttention
from typing import List, Optional, Tuple, Union
import numpy as np
import time

import importlib
import inspect
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
# assumes timesteps is sorted in ascending order
def get_t_schedule(timesteps, max_iter, mode='cyclic', min_t=0.0, max_t=1.0, n_periods=1, sampling=False):
    def _t_schedule(t):
        if mode == 'cyclic':
            # cos curve wih n_periods between min_t and max_t starting
            # at min t # normal period is 2pi,
            # so cos(max_iter / (n_periods * 2 * pi)) gives
            # periods length max_iter / n_periods
            schedule_max_t = (
                -1
                * ((max_t - min_t) / 2)
                * np.cos((n_periods * 2 * np.pi / max_iter) * t)
                + ((max_t - min_t) / 2)
                + min_t
            )
        elif mode == 'random':
            schedule_max_t = max_t
        elif mode == 'increasing':
            schedule_max_t = ((max_t - min_t) / max_iter) * t + min_t
        elif mode == 'decreasing':
            schedule_max_t = ((min_t - max_t) / max_iter) * t + max_t
        elif mode == 'constant':
            schedule_max_t = max_t
        else:
            raise ValueError(
                f'Mode {mode} must be one of [cyclic, random, constant, increasing, decreasing]'
            )

        # find closest step in timesteps list
        max_ix = np.argmin([abs(schedule_max_t - step) for step in timesteps])
        min_ix = np.argmin([abs(min_t - step) for step in timesteps])
        if sampling or mode == 'random':
            ix = random.randint(min_ix, max_ix)
        else:
            ix = max_ix
        step = timesteps[ix]
        return step

    return _t_schedule

def get_attn_fn(num_sample_pixel = 64, reshape_by_num_of_head = False):
    attns = []
    query_feat_list = []
    key_feat_list = []
    value_feat_list = []
    res_list = []
    @torch.no_grad()
    def attn_hook_fn(module, args, kwargs, output):

        # don't compute text cross attn to save memory
        if 'encoder_hidden_states' in kwargs and kwargs['encoder_hidden_states'] is not None:
            return

        attn = module
        attention_mask = None
        encoder_hidden_states = None
        hidden_states = args[0].detach()

        batch_size, sequence_length, _ = hidden_states.shape
        if sequence_length == 4096:
            return
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.cross_attention_norm:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        batch_size = output.shape[0]
        num_of_pixel = output.shape[1]
        if reshape_by_num_of_head:
            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
        num_of_head = query.shape[0] // batch_size
        npixel = query.shape[1]
        batch_size = encoder_hidden_states.shape[0]
        res_list.append(npixel)
        query_feat_list.append(query.reshape(batch_size, num_of_head, num_of_pixel, -1))
        key_feat_list.append(key.reshape(batch_size, num_of_head, num_of_pixel, -1))
        value_feat_list.append(value.reshape(batch_size, num_of_head, num_of_pixel, -1))
    return attns, query_feat_list, key_feat_list, value_feat_list, res_list, attn_hook_fn

class shift_PNDMScheduler(PNDMScheduler):
    def __init__(self, *args, **kwargs):
        super(shift_PNDMScheduler, self).__init__(*args, **kwargs)

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None, shift: int = 0):
        """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.
        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """

        self.num_inference_steps = num_inference_steps
        step_ratio = self.config.num_train_timesteps // self.num_inference_steps
        # creates integer timesteps by multiplying by ratio
        # casting to int to avoid issues when num_inference_step is power of 3
        self._timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()
        # Shift the timesteps
        if shift > 0:
            self._timesteps = self._timesteps[:self._timesteps.tolist().index(shift) + 1]
        self._timesteps += self.config.steps_offset

        if self.config.skip_prk_steps:
            # for some models like stable diffusion the prk steps can/should be skipped to
            # produce better results. When using PNDM with `self.config.skip_prk_steps` the implementation
            # is based on crowsonkb's PLMS sampler implementation: https://github.com/CompVis/latent-diffusion/pull/51
            self.prk_timesteps = np.array([])
            self.plms_timesteps = np.concatenate([self._timesteps[:-1], self._timesteps[-2:-1], self._timesteps[-1:]])[
                ::-1
            ].copy()
        else:
            prk_timesteps = np.array(self._timesteps[-self.pndm_order :]).repeat(2) + np.tile(
                np.array([0, self.config.num_train_timesteps // num_inference_steps // 2]), self.pndm_order
            )
            self.prk_timesteps  = (prk_timesteps[:-1].repeat(2)[1:-1])[::-1].copy()
            self.plms_timesteps = self._timesteps[:-3][
                ::-1
            ].copy()  # we copy to avoid having negative strides which are not supported by torch.from_numpy

        timesteps = np.concatenate([self.prk_timesteps, self.plms_timesteps]).astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps).to(device)
        self.ets = []
        self.counter = 0

class StableDiffusion(nn.Module):
    def __init__(self, args, cache='./cache', ncut_weight = 1, orth_weight = 1):
        super().__init__()
        cache = args.stable_diffusion_cache_path
        self.args = args
        self.token = args.huggingface_token
        self.device = args.gpu
        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)

        print(f'[INFO] loading stable diffusion...')

        # 1. Load the autoencoder model which will be used to decode the latents into image space.
        self.cache = cache
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_auth_token=self.token, cache_dir = self.cache).to(self.device)

        # 2. Load the tokenizer and text encoder to tokenize and encode the text.
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", cache_dir = self.cache)
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir = self.cache).to(self.device)

        # 3. The UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", use_auth_token=self.token, cache_dir = self.cache).to(self.device)
        # 4. Create a scheduler for inference
        self.scheduler = shift_PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=self.num_train_timesteps)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience
        self.attn_buffer = []
        print(f'[INFO] loaded stable diffusion!')

        self.feat_h = 64
        self.feat_w = 64

        self.attn_buffer = []
        self.attn_buffer_size = 10
        self.use_buffer_prob = 0
        self.attns, self.query_list, self.key_list, self.value_list, self.res_list, attn_hook_fn = get_attn_fn()
        self.attn_hooks = []
        for module in self.unet.mid_block.modules():
            if isinstance(module, CrossAttention):
                self.attn_hooks.append(module.register_forward_hook(attn_hook_fn, with_kwargs=True))
        for module in self.unet.up_blocks.modules():
            if isinstance(module, CrossAttention):
                self.attn_hooks.append(module.register_forward_hook(attn_hook_fn, with_kwargs=True))
        self.feat_dim_list = []
        self.feat_res_list = []

    def get_text_embeds(self, prompt = [None]):
        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer([''] * len(prompt), padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        return uncond_embeddings

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215
        return latents

    def process_input(self, img, slice_size = 8):
        img_512 = F.interpolate(img, (512, 512), mode='bilinear', align_corners=False)
        text_embeddings = self.get_text_embeds().detach()
        iter_num = (img.shape[0] + slice_size - 1) // slice_size
        latent_list = []
        for i in range(iter_num):
            img_latents = self.encode_imgs(img_512[i * slice_size: (i + 1) * slice_size]).detach()
            latent_list.append(img_latents)
        return torch.cat(latent_list), text_embeddings

    @torch.no_grad()
    def collect_attention_and_feat(self, img):
        img_latents, text_embeddings = self.process_input(img)
        t = (torch.rand(img_latents.shape[0]).to(img_latents.device) * (self.args.noise_max_t - self.args.noise_min_t) + \
            self.args.noise_min_t).long()
        noise = torch.randn_like(img_latents)
        latents_noisy = self.scheduler.add_noise(img_latents, noise, t)
        self.attns.clear()
        self.query_list.clear()
        self.key_list.clear()
        self.value_list.clear()
        self.unet(latents_noisy, t, encoder_hidden_states=text_embeddings.expand(img_latents.shape[0], -1,-1).to(dtype = latents_noisy.dtype)).sample

        # record the feature channel and resolution
        if len(self.feat_dim_list) == 0:
            for feat, res in zip(self.query_list, self.res_list):
                subsample_feat_shape = torch.Size((feat.shape[0], feat.shape[1], min(self.args.subsample_pixel_num, feat.shape[2]), feat.shape[3]))
                self.feat_dim_list.append(subsample_feat_shape)
                self.feat_res_list.append(int(np.sqrt(res)))

        num_of_feat = len(self.feat_dim_list)
        query_list = self.query_list.copy()
        key_list = self.key_list.copy()
        value_list = self.value_list.copy()
        self.attns.clear()
        self.res_list.clear()
        self.query_list.clear()
        self.key_list.clear()
        self.value_list.clear()
        return query_list, key_list, value_list

        #return attn_list

    def clear_after_loop(self):
        self.attns.clear()
        self.attn_buffer.clear()
