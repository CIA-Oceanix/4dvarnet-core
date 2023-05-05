#!/bin/bash

# Unet w/wosst

#  OSSE OI wosst
CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python hydra_main.py xp=mbeaucha/xp_ose/duacs_free_ose_gf_wosst file_paths=srv5_ose entrypoint=test +entrypoint.ckpt_path="../../ose/ckpt/lx691_lt406_mu959.ckpt"
#  OSSE OI wsst
CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python hydra_main.py xp=mbeaucha/xp_ose/duacs_free_ose_gf_wsst file_paths=srv5_ose entrypoint=test +entrypoint.ckpt_path="../../ose/ckpt/lx691_lt406_mu959.ckpt"
#  OSSE wDUACS wosst
CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python hydra_main.py xp=mbeaucha/xp_ose/duacs_ose_gf_wosst file_paths=srv5_ose entrypoint=test +entrypoint.ckpt_path="../../ose/ckpt/lx691_lt406_mu959.ckpt"
#  OSSE wDUACS wsst
CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python hydra_main.py xp=mbeaucha/xp_ose/duacs_ose_gf_wsst file_paths=srv5_ose entrypoint=test +entrypoint.ckpt_path="../../ose/ckpt/lx691_lt406_mu959.ckpt"

# fine-tuning OSSE OI wosst
CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python hydra_main.py xp=mbeaucha/xp_ose/duacs_free_ose_gf_wosst file_paths=srv5_ose entrypoint=run entrypoint.max_epochs=30 +entrypoint.ckpt_path="../../ose/ckpt/lx691_lt406_mu959.ckpt"
# fine-tuning OSSE OI wsst
CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python hydra_main.py xp=mbeaucha/xp_ose/duacs_free_ose_gf_wsst file_paths=srv5_ose entrypoint=run entrypoint.max_epochs=30 +entrypoint.ckpt_path="../../ose/ckpt/lx691_lt406_mu959.ckpt"
# fine-tuning OSSE wDUACS wosst
CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python hydra_main.py xp=mbeaucha/xp_ose/duacs_ose_gf_wosst file_paths=srv5_ose entrypoint=run entrypoint.max_epochs=30 +entrypoint.ckpt_path="../../ose/ckpt/lx691_lt406_mu959.ckpt"
# fine-tuning OSSE wDUACS wsst
CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python hydra_main.py xp=mbeaucha/xp_ose/duacs_ose_gf_wsst file_paths=srv5_ose entrypoint=run entrypoint.max_epochs=30 +entrypoint.ckpt_path="../../ose/ckpt/lx691_lt406_mu959.ckpt"

