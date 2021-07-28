# Commands to manage slurm jobs

# SWOT with ROLL and NADIRS  : config_q/roll.py
## start training
sbatch --parsable scripts/run.slurm q.roll >> scripts/roll.jobs
## See logs
tail -f *$(tail -n1 scripts/roll.jobs)*
## Test latest ckpt
sbatch --parsable scripts/test.slurm $(tail -n1 scripts/roll.jobs) q.roll >> scripts/roll.test_jobs
## tb latest test ckpt
tensorboard --logdir=lightning_logs/version_$(tail -n1 scripts/roll.test_jobs) --bind_all
