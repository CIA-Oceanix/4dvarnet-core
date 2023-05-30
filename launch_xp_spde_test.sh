#!/bin/bash

# xp gp
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python hydra_main.py xp=mbeaucha/xp_spde/gp_diff_pow_2/oi_gp_spde_wlp_mlloss file_paths=mee_a100_gp entrypoint=test +entrypoint.ckpt_path=../../model_oi_gp_spde.ckpt &
# xp ssh
CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python hydra_main.py xp=mbeaucha/xp_spde/ssh/oi_osse_spde file_paths=mee_a100_ssh entrypoint=test +entrypoint.ckpt_path=../../model_oi_osse_spde.ckpt &
# xp ssh_qg
CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python hydra_main.py xp=mbeaucha/xp_spde/ssh/oi_osse_spde_qg file_paths=mee_a100_ssh entrypoint=test +entrypoint.ckpt_path=../../model_oi_osse_spde_qg.ckpt &

sleep 15m

N=`ls dashboard/oi_gp_spde_wlp_ml_loss/lightning_logs/ | tail -n 1 | cut -f2 -d'_'`
cp -rf dashboard/oi_gp_spde_wlp_ml_loss/lightning_logs/version_${N}/maps.nc $HOME/maps_gp_4DvarNet_SPDE.nc
N=`ls dashboard/oi_osse_spde/lightning_logs/ | tail -n 1 | cut -f2 -d'_'`
cp -rf dashboard/oi_osse_spde/lightning_logs/version_${N}/maps.nc $HOME/maps_osse_4DvarNet_SPDE.nc
N=`ls dashboard/oi_osse_spde_qg/lightning_logs/ | tail -n 1 | cut -f2 -d'_'`
cp -rf dashboard/oi_osse_spde_qg/lightning_logs/version_${N}/maps.nc $HOME/maps_osse_4DvarNet_SPDE_qg.nc
