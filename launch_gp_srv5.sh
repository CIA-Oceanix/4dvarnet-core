#!/bin/bash


# OI SPDE with learning parameters
#CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python hydra_main.py xp=mbeaucha/xp_spde/${gp_dataset}/gp_diff_pow_2/oi_osse_spde_learn_params file_paths=srv5_osse entrypoint=run entrypoint.max_epochs=1
#CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python hydra_main.py xp=mbeaucha/xp_spde/${gp_dataset}/gp_diff_pow_2/oi_gp_spde_wlp_oiloss file_paths=srv5_gp entrypoint=run entrypoint.max_epochs=1
#CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python hydra_main.py xp=mbeaucha/xp_spde/${gp_dataset}/gp_diff_pow_2/oi_gp_spde_wlp_mseloss file_paths=srv5_gp entrypoint=run entrypoint.max_epochs=1
#CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python hydra_main.py xp=mbeaucha/xp_spde/${gp_dataset}/gp_diff_pow_2/oi_gp_spde_wlp_mlloss file_paths=srv5_gp entrypoint=run entrypoint.max_epochs=1
#CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python hydra_main.py xp=mbeaucha/xp_spde/${gp_dataset}/gp_diff_pow_2/oi_gp_spde_wlp_diffloss file_paths=srv5_gp entrypoint=run entrypoint.max_epochs=3 

##############################
##############################

#list_gp_dataset=('gp_diff_pow_1' 'gp_diff_pow_2' 'gp_iso_pow_1' 'gp_iso_pow_2')
list_gp_dataset=('gp_iso_pow_2' 'gp_diff_pow_2' 'gp_iso_pow_1')
#list_gp_dataset=('gp_diff_pow_2')

for gp_dataset in ${list_gp_dataset[*]}; do

  mkdir -p  oi/logs/${gp_dataset}

  # OI Covariance
  CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python hydra_main.py xp=mbeaucha/xp_spde/${gp_dataset}/oi_gp_ref file_paths=srv5_gp entrypoint=run entrypoint.max_epochs=1 | tee oi/logs/${gp_dataset}/log_oi_gp_ref

  # OI-GD Covariance
  CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python hydra_main.py xp=mbeaucha/xp_spde/${gp_dataset}/oi_gp_ref_gd file_paths=srv5_gp entrypoint=run entrypoint.max_epochs=1 | tee oi/logs/${gp_dataset}/log_oi_gp_ref_gd

  # UNet (MSE loss) 
  CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python hydra_main.py xp=mbeaucha/xp_spde/${gp_dataset}/oi_gp_unet_mseloss file_paths=srv5_gp entrypoint=run entrypoint.max_epochs=20 | tee oi/logs/${gp_dataset}/log_oi_gp_unet_mseloss
  # UNet (OI loss) 
  CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python hydra_main.py xp=mbeaucha/xp_spde/${gp_dataset}/oi_gp_unet_oiloss file_paths=srv5_gp entrypoint=run entrypoint.max_epochs=20 | tee oi/logs/${gp_dataset}/log_oi_gp_unet_oiloss

  # 4DVarNet-FP Covariance (nothing to learn): 1-S = 1-LT  
  # CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python hydra_main.py xp=mbeaucha/xp_spde/${gp_dataset}/oi_gp_spde_fp_wolp file_paths=srv5_gp entrypoint=run entrypoint.max_epochs=1 | tee oi/logs/${gp_dataset}/log_oi_gp_spde_fp_wolp
  # 4DVarNet-FP UNet (MSE loss) 
  CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python hydra_main.py xp=mbeaucha/xp_spde/${gp_dataset}/oi_gp_4dvarnet_fp_mseloss file_paths=srv5_gp entrypoint=run entrypoint.max_epochs=20 | tee oi/logs/${gp_dataset}/log_oi_gp_4dvarnet_fp_mseloss
  # 4DVarNet-FP UNet (OI loss)
  CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python hydra_main.py xp=mbeaucha/xp_spde/${gp_dataset}/oi_gp_4dvarnet_fp_oiloss file_paths=srv5_gp entrypoint=run entrypoint.max_epochs=20 | tee oi/logs/${gp_dataset}/log_oi_gp_4dvarnet_fp_oiloss

  # 4DVarNet-GD Covariance (MSE loss)
  CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python hydra_main.py xp=mbeaucha/xp_spde/${gp_dataset}/oi_gp_spde_wolp_mseloss file_paths=srv5_gp entrypoint=run entrypoint.max_epochs=20 | tee oi/logs/${gp_dataset}/log_oi_gp_spde_wolp_mseloss
  # 4DVarNet-GD Covariance (OI loss)
  CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python hydra_main.py xp=mbeaucha/xp_spde/${gp_dataset}/oi_gp_spde_wolp_oiloss file_paths=srv5_gp entrypoint=run entrypoint.max_epochs=20 | tee oi/logs/${gp_dataset}/log_oi_gp_spde_wolp_oiloss 
  # 4DVarNet-GD UNet (MSE loss)
  CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python hydra_main.py xp=mbeaucha/xp_spde/${gp_dataset}/oi_gp_4dvarnet_mseloss file_paths=srv5_gp entrypoint=run entrypoint.max_epochs=20 | tee oi/logs/${gp_dataset}/log_oi_gp_4dvarnet_mseloss
  # 4DVarNet-GD UNet (OI loss)
  CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python hydra_main.py xp=mbeaucha/xp_spde/${gp_dataset}/oi_gp_4dvarnet_oiloss file_paths=srv5_gp entrypoint=run entrypoint.max_epochs=20 | tee oi/logs/${gp_dataset}/log_oi_gp_4dvarnet_oiloss
  # 4DVarNet-GD Linear Phi (MSE loss)
  CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python hydra_main.py xp=mbeaucha/xp_spde/${gp_dataset}/oi_gp_4dvarnet_wphi_linear_mseloss file_paths=srv5_gp entrypoint=run entrypoint.max_epochs=20 | tee oi/logs/${gp_dataset}/log_oi_gp_4dvarnet_mseloss
  # 4DVarNet-GD Linear Phi (OI loss)
  CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python hydra_main.py xp=mbeaucha/xp_spde/${gp_dataset}/oi_gp_4dvarnet_wphi_linear_oiloss file_paths=srv5_gp entrypoint=run entrypoint.max_epochs=20 | tee oi/logs/${gp_dataset}/log_oi_gp_4dvarnet_oiloss
  
done

#######################

# extra
#CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python hydra_main.py xp=mbeaucha/xp_spde/${gp_dataset}/oi_gp_4dvarnet_mseloss_test_grad_lstm file_paths=srv5_gp entrypoint=run entrypoint.max_epochs=1 | tee oi/logs/log_oi_gp_${gp_dataset}_4dvarnet_mseloss_test_grad_lstm



