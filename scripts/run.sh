
### Cancel all runs
squeue -u $USER -h -o %i | xargs scancel


# NADIRS + SWOT : config_q/swot.py
## start training
echo (sbatch --parsable scripts/run.slurm swot)  >> scripts/swot.jobs
## See logs
tail -f *(tail -n1 scripts/swot.jobs)*
## Test latest ckpt
sbatch scripts/test.slurm $(tail -n1 scripts/swot.jobs) swot >> scripts/swot.test_jobs
## tb latest test ckpt
tensorboard --logdir=lightning_logs/version_(tail -n1 scripts/swot.test_jobs) --bind_all


# NADIRS ONLY : config_q/nad.py
## start training
echo (sbatch --parsable scripts/run.slurm nad) >> scripts/nad.jobs
## See logs
tail -f *(tail -n1 scripts/nad.jobs)*
## Test latest ckpt
sbatch scripts/test.slurm $(tail -n1 scripts/nad.jobs) nad >> scripts/nad.test_jobs
## tb latest test ckpt
tensorboard --logdir=lightning_logs/version_(tail -n1 scripts/nad.test_jobs) --bind_all


# SWOT with ROLL and NADIRS  : config_q/roll.py
## start training
echo (sbatch --parsable scripts/run.slurm roll) >> scripts/roll.jobs
## See logs
tail -f *(tail -n1 scripts/roll.jobs)*
## Test latest ckpt
sbatch scripts/test.slurm $(tail -n1 scripts/roll.jobs) roll >> scripts/roll.test_jobs
## tb latest test ckpt
tensorboard --logdir=lightning_logs/version_(tail -n1 scripts/roll.test_jobs) --bind_all


# NADIRS with SWOT +ROLL as "sst" : config_q/nad_roll.py
## start training
echo (sbatch --parsable scripts/run.slurm nad_roll) >> scripts/nad_roll.jobs
## See logs
tail -f *(tail -n1 scripts/nad_roll.jobs)*
## Test latest ckpt
sbatch scripts/test.slurm $(tail -n1 scripts/nad_roll.jobs) nad_roll >> scripts/nad_roll.test_jobs
## tb latest test ckpt
tensorboard --logdir=lightning_logs/version_(tail -n1 scripts/nad_roll.test_jobs) --bind_all

# NADIRS with SST as "sst": config_q/nad_sst.py
# test job already running
squeue -u $USER -h  | grep (tail -n1 scripts/nad_sst.jobs)
## start training
echo (sbatch --parsable scripts/run.slurm nad_sst) >> scripts/nad_sst.jobs
## See logs
tail -f *(tail -n1 scripts/nad_sst.jobs)*
## Test latest ckpt
sbatch scripts/test.slurm $(tail -n1 scripts/nad_sst.jobs) nad_sst >> scripts/nad_sst.test_jobs
## tb latest test ckpt
tensorboard --logdir=lightning_logs/version_(tail -n1 scripts/nad_sst.test_jobs) --bind_all


# NADIRS+SWOT with SST as "sst": config_q/roll_sst.py
# test job already running
squeue -u $USER -h  | grep (tail -n1 scripts/roll_sst.jobs)

## start training
echo (sbatch --parsable scripts/run.slurm roll_sst) >> scripts/roll_sst.jobs
## See logs
tail -f *(tail -n1 scripts/roll_sst.jobs)*
## Test latest ckpt
sbatch scripts/test.slurm $(tail -n1 scripts/roll_sst.jobs) roll_sst >> scripts/roll_sst.test_jobs
## tb latest test ckpt
tensorboard --logdir=lightning_logs/version_(tail -n1 scripts/roll_sst.test_jobs) --bind_all
