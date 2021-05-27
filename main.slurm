#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=2
#SBATCH --account=nlu@gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --reservation=hackathon_idr27
#SBATCH -C v100-16g
#SBATCH --qos=qos_gpu-dev
#SBATCH --output=log/%j.out
#SBATCH --error=log/%j.err

# activate conda env
source activate $1

# run script from above
srun python main.py --dataloading=new train --progress_bar_refresh_rate=5
