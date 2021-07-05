# Fish commands to manage xps


# NADIRS + SWOT : config_q/swot.py
# Cancel running job
scancel (tail -n1 scripts/swot.jobs)
# start training
echo (sbatch --parsable scripts/run.slurm q.swot)  >> scripts/swot.jobs
## See logs
tail -f *(tail -n1 scripts/swot.jobs)*
## Test latest ckpt
sbatch --parsable scripts/test.slurm (tail -n1 scripts/swot.jobs) q.swot >> scripts/swot.test_jobs
## tb latest test ckpt
tensorboard --logdir=lightning_logs/version_(tail -n1 scripts/swot.test_jobs) --bind_all


# NADIRS ONLY : config_q/nad.py
## start training
echo (sbatch --parsable scripts/run.slurm q.nad) >> scripts/nad.jobs
## See logs
tail -f *(tail -n1 scripts/nad.jobs)*
## Test latest ckpt
sbatch --parsable scripts/test.slurm (tail -n1 scripts/nad.jobs) q.nad >> scripts/nad.test_jobs
## tb latest test ckpt
tensorboard --logdir=lightning_logs/version_(tail -n1 scripts/nad.test_jobs) --bind_all


# SWOT with ROLL and NADIRS  : config_q/roll.py
## start training
echo (sbatch --parsable scripts/run.slurm q.roll) >> scripts/roll.jobs
## See logs
tail -f *(tail -n1 scripts/roll.jobs)*
## Test latest ckpt
sbatch --parsable scripts/test.slurm (tail -n1 scripts/roll.jobs) q.roll >> scripts/roll.test_jobs
## tb latest test ckpt
tensorboard --logdir=lightning_logs/version_(tail -n1 scripts/roll.test_jobs) --bind_all


# NADIRS with SWOT +ROLL as "sst" : config_q/nad_roll.py
## start training
echo (sbatch --parsable scripts/run.slurm q.nad_roll) >> scripts/nad_roll.jobs
## See logs
tail -f *(tail -n1 scripts/nad_roll.jobs)*
## Test latest ckpt
sbatch --parsable scripts/test.slurm (tail -n1 scripts/nad_roll.jobs) q.nad_roll >> scripts/nad_roll.test_jobs
## tb latest test ckpt
tensorboard --logdir=lightning_logs/version_(tail -n1 scripts/nad_roll.test_jobs) --bind_all

# NADIRS with SST as "sst": config_q/nad_sst.py
# test job already running
squeue -u $USER -h  | grep (tail -n1 scripts/nad_sst.jobs)
## start training
echo (sbatch --parsable scripts/run.slurm q.nad_sst) >> scripts/nad_sst.jobs
## See logs
tail -f *(tail -n1 scripts/nad_sst.jobs)*
## Test latest ckpt
sbatch --parsable scripts/test.slurm (tail -n1 scripts/nad_sst.jobs) q.nad_sst >> scripts/nad_sst.test_jobs
## tb latest test ckpt
tensorboard --logdir=lightning_logs/version_(tail -n1 scripts/nad_sst.test_jobs) --bind_all


# NADIRS+SWOT+ROLL with SST as "sst": config_q/roll_sst.py
# test job already running
squeue -u $USER -h  | grep (tail -n1 scripts/roll_sst.jobs)
## start training
echo (sbatch --parsable scripts/run.slurm q.roll_sst) >> scripts/roll_sst.jobs
## See logs
tail -f *(tail -n1 scripts/roll_sst.jobs)*
## Test latest ckpt
sbatch --parsable scripts/test.slurm (tail -n1 scripts/roll_sst.jobs) q.roll_sst >> scripts/roll_sst.test_jobs
## tb latest test ckpt
tensorboard --logdir=lightning_logs/version_(tail -n1 scripts/roll_sst.test_jobs) --bind_all


# NADIRS+SWOT with SST as "sst": config_q/swot_sst.py
# test job already running
squeue -u $USER -h  | grep (tail -n1 scripts/swot_sst.jobs)
## start training
echo (sbatch --parsable scripts/run.slurm q.swot_sst) >> scripts/swot_sst.jobs
## See logs
tail -f *(tail -n1 scripts/swot_sst.jobs)*
## Test latest ckpt
sbatch --parsable scripts/test.slurm (tail -n1 scripts/swot_sst.jobs) q.swot_sst >> scripts/swot_sst.test_jobs
## tb latest test ckpt
tensorboard --logdir=lightning_logs/version_(tail -n1 scripts/swot_sst.test_jobs) --bind_all



function tbdash
	for xp in swot swot_sst roll_sst nad_sst nad_roll roll nad
		ln -sf ../../lightning_logs/version_(tail -n1 scripts/$xp.jobs | string match -r '[0-9]+$')  dashboard/train/$xp
		ln -sf ../../lightning_logs/version_(tail -n1 scripts/$xp.test_jobs | string match -r '[0-9]+$')  dashboard/test/$xp
	end
end
# Next step tb dashboard:
# sym link all last test jobs with nice dir name in test directory 

# Refacto:
# move commands to functions 
