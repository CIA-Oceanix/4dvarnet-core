python hydra_main xp=cdorff/xp_turbidity file_paths=turbidity_jz entrypoint=train +backend=slurm_1x4 -m
