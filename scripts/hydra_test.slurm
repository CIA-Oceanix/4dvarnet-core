#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --account=yrf@gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH -C v100-16g
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=log%j.out
#SBATCH --error=log%j.err

# activate conda env
#source activate $1

module purge

eval "$(conda shell.bash hook)"
conda activate --stack 4dvarnet
# export PYTHONPATH=${SCRATCH}/james:${PYTHONPATH}
echo $(pwd)
# cd $SCRATCH/james
mr_ls() {
	ls -t dashboard/current/train/$1/checkpoints | head -n1
}

test_most_recent_ckpt () {
	ckpt_path="dashboard/current/train/$1/checkpoints/$(mr_ls $1 | sed 's/=/\\=/g')"
	echo $ckpt_path
	python hydra_main.py xp=$1 entrypoint=test entrypoint.ckpt_path="$ckpt_path"
}

test_most_recent_ckpt $1


