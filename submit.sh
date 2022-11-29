#!/bin/bash
#PBS -N 4DVarNet
#PBS -q qgpgpu
#PBS -l select=1:ncpus=4:mem=92G:ngpus=1  
#PBS -l walltime=11:00:00

# deplacement dans le repertoire local de travail
cd /home/qt/trebouta/4dvarnet-main/

#Chargement des environnements
module load conda
conda activate /work/scratch/buschec/shared_env/4dvarnet

# lancement du programme GPU
python hydra_main_AT.py xp=AT/xp_oi_MODIFs file_paths=hal_AT  entrypoint=train

