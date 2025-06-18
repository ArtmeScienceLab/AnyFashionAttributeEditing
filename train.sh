python -W ignore -m scripts.train_twinnet --dataset_type='fs_men_encode'  --start_from_latent_avg \
--id_lambda=0.1  --val_interval=1000 --save_interval=1000 --max_steps=200000  --stylegan_size=1024 --is_train=True \
--distortion_scale=0.15 --aug_rate=0.9 --res_lambda=0.1  \
--checkpoint_path='/path_to/e4e_s_print.pt' \
--workers=8  --batch_size=8  --test_batch_size=2 --test_workers=4 --exp_dir='./experiment/s_print/' 