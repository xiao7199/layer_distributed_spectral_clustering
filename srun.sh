#!/bin/bash

export NCCL_DEBUG=info
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL



while
  port=$(shuf -n 1 -i 49152-65535)
  netstat -atun | grep -q "$port"
do
  continue
done

echo "$port"
### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gn  oded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

data_path='dataset'
sd_cache='sd_cache'
HUGGIN_TOKEN=

python -m torch.distributed.launch \
--nproc_per_node=1 --master_port=${port} main_cls.py \
--data_path ${data_path} \
--stable_diffusion_cache_path ${sd_cache} \
--huggingface_token $HUGGIN_TOKEN \
--num_of_inter_pixel 50 \
--num_of_intra_pixel 50 \
--sparse_graph \
--output_dim 100 \
--epochs 2000 \
--avg_steps 1 \
--batch_size 40 \
--num_of_image_feat_table 2000 \
--seed 12345 \
--learning_rate 1e-2 \
--weight_decay 0 \
--use_intra_inter_split \
--vv_graph \
--resume
