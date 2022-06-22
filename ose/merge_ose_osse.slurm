#!/bin/bash
#SBATCH -A yrf@cpu                  # nom du compte
#SBATCH --job-name=build_dataset    # nom du job
#SBATCH --ntasks=1                  # nombre de tâche (un unique processus ici)
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread        # on réserve des coeurs physiques et non logiques
#SBATCH --time=01:00:00             # temps exécution maximum demande (HH:MM:SS)
#SBATCH --output=merge%j.out     # nom du fichier de sortie
#SBATCH --error=merge%j.err      # nom du fichier erreur (ici commun avec la sortie)

# nettoyage des modules charges en interactif et hérités par défaut
module purge

# chargement des modules
module load tensorflow-gpu/py3/1.14-openmpi
module load geos/3.7.3
module load netcdf-fortran/4.5.3-mpi-cuda netcdf-c/4.7.4-mpi-cuda parallel-netcdf/1.12.1-mpi-cuda

export PYTHONPATH=${HOME}/NADIR_TOOLS/:${HOME}/DINAE_keras/:${HOME}/PB_ANDA:${PYTHONPATH}:${HOME}/4DVARNN-DinAE:${HOME}/NATL60:/gpfswork/rech/yrf/uba22to/esmf/esmpy/lib/python3.7/site-packages:/gpfswork/rech/yrf/uba22to/esmf/xesmf/lib/python3.7/site-packages
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfswork/rech/yrf/uba22to/esmf/lib/libO/Linux.intel.64.openmpi.default

# echo des commandes lancées
set -x

# exécution du code
cd /gpfswork/rech/yrf/uba22to/4dvarnet-core/ose
python -u merge_ose_osse.py

