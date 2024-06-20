HUGGIN_TOKEN=
python main.py --huggingface_token $HUGGIN_TOKEN \
    --img_path img.jpg \
    --model stable_diffusion \
    --noise_schedule=random --noise_sampling --noise_min_t=0 --noise_max_t=250 \
    --learning_rate=2e-3 --train_steps=2000 \
    --use_buffer_prob=0.75 --attn_buffer_size=7 \
    --num_of_eig 20
