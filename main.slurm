#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=2
#SBATCH --account=nlu@gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --time=02:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH -C v100-16g
#SBATCH --qos=qos_gpu-dev

# activate conda env
source activate $1

# run script from above
srun python main.py
