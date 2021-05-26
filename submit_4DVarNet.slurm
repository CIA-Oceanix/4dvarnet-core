#!/bin/bash
#SBATCH -A nlu@gpu                  # nom du compte
#SBATCH --job-name=gpu_mono         # nom du job
#SBATCH --ntasks=1                  # nombre de tâche (un unique processus ici)
#SBATCH --gres=gpu:1                # nombre de GPU à réserver (un unique GPU ici)
#SBATCH --cpus-per-task=40          # nombre de coeurs à réserver (un quart du noeud)
#SBATCH -C v100-32g                 # partition
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread        # on réserve des coeurs physiques et non logiques
#SBATCH --time=02:00:00             # temps exécution maximum demande (HH:MM:SS)
#SBATCH --output=FourDVarNet%j.out     # nom du fichier de sortie
#SBATCH --error=FourDVarNet%j.err      # nom du fichier erreur (ici commun avec la sortie)
#SBATCH --reservation=hackathon_idr26
 
# nettoyage des modules charges en interactif et hérités par défaut
module purge
 
# chargement des modules
eval "$(conda shell.bash hook)"
conda activate 4dvarnet
export PYTHONPATH=${WORK}/4dvarnet-core:${PYTHONPATH}
 
# echo des commandes lancées
set -x

# copie vers le WORK pour execution
JobID=$RANDOM
rsync -avrz --exclude=figs ${WORK}/4dvarnet-core/ ${WORK}/4dvarnet-core_${JobID}

# exécution du code
python -u ${WORK}/4dvarnet-core_${JobID}/main.py --dataloading=new --max_epochs=1 run

# remove copy
rm -rf ${WORK}/4dvarnet-core_${JobID}
