#!/bin/bash

for ((i=1; i<=100; i++)); do
	CUDA_VISIBLE_DEVICES=2 python hydra_main.py xp=cdorff/xp_BBP_rand10 entrypoint=test file_paths=BBP entrypoint.ckpt_path='version_1/checkpoints/modelCalSLAInterpGF-epoch\=97-val_loss\=25.6458.ckpt'
	python rand10_post_proc.py
        mv "./dashboard/xp_BBP_dt7/lightning_logs/version_0/test0.nc" "/DATASET/turbidity_new/BBP443_rand10/run_$i.nc"
	rm -rf "./dashboard/xp_BBP_dt7/lightning_logs"
done
