# 4dvarnet-core
Core implementation of the 4DVarNet architecture 

## Principles and on-going developments
Discussed in this [HackMd file](https://hackmd.io/@maxbeauchamp/ryVfI3rdu)

## Installation
### Prerequisite
- git
- conda

### Instructions
- Clone repo:
`git clone https://github.com/CIA-Oceanix/4dvarnet-core.git`

- Install environment
```
conda create -n 4dvarnet mamba -c conda-forge
conda activate 4dvarnet
mamba env update -f environment.yaml
```

### Download the data
The data is available at
```

#oi
https://s3.eu-central-1.wasabisys.com/melody/NATL/oi/ssh_NATL60_4nadir.nc
https://s3.eu-central-1.wasabisys.com/melody/NATL/oi/ssh_NATL60_swot_4nadir.nc

#ref
https://s3.eu-central-1.wasabisys.com/melody/NATL/ref/NATL60-CJM165_NATL_ssh_y2013.1y.nc

# obs and noise 
https://s3.eu-central-1.wasabisys.com/melody/quentin_cloud/icassp_data/cal_data_karin_noise_only.nc
https://s3.eu-central-1.wasabisys.com/melody/quentin_cloud/icassp_data/full_cal_obs.nc

# raw sensor data
https://s3.eu-central-1.wasabisys.com/melody/quentin_cloud/icassp_data/sensor_zarr.tar

```


### ICASSP2022 experiments:

#### Training commands:

```
python main.py --max_epochs=250 --config=q.xp_icassp.direct_phi --progress_bar_refresh_rate=5 train
python main.py --max_epochs=250 --config=q.xp_icassp.direct_vit --progress_bar_refresh_rate=5 train
python main.py --max_epochs=250 --config=q.xp_icassp.fourdvarnet_calmap --progress_bar_refresh_rate=5 train
python main.py --max_epochs=250 --config=q.xp_icassp.fourdvarnet_calmapgrad --progress_bar_refresh_rate=5 train
python main.py --max_epochs=250 --config=q.xp_icassp.fourdvarnet_map --progress_bar_refresh_rate=5 train
```

#### Results generation

### Run
```
python main.py
```

## Contribution workflow
- [Install the project](#installation)
- create a feature branch:
`git checkout -b <my-feature-branch>`
- Code your contribution ...
- Review and commmit your contributions 
```
git add -p
git commit -m "<A clear message>"
```
- Check that you are up to date with the common version 
```
git pull --rebase origin main
```

- Push your branch
```
git push origin <my-feature-branch>
```
- Create a merge request of your branch to main https://github.com/CIA-Oceanix/4dvarnet-core/pulls 
- Ask a 4dvarnet team member to review and merge your code

## Preprints and Software License
Associated preprints: https://arxiv.org/abs/2006.03653
License: CECILL-C license

Copyright IMT Atlantique/OceaniX, contributor(s) : M.M. Amar, M. Beauchamp, R. Fablet, Q. Febvre (IMT Atlantique), B. Carpentier (CLS) 21/03/2020

Contact person: ronan.fablet@imt-atlantique.fr
This software is a computer program whose purpose is to apply deep learning
schemes to dynamical systems and ocean remote sensing data.
This software is governed by the CeCILL-C license under French law and
abiding by the rules of distribution of free software.  You can  use,
modify and/ or redistribute the software under the terms of the CeCILL-C
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info".
As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability.
In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or
data to be ensured and,  more generally, to use and operate it in the
same conditions as regards security.
The fact that you are presently reading this means that you have had
knowledge of the CeCILL-C license and that you accept its terms.

