## 4DVarNet code for the reconstruction of SSH fields from SST-SSH synergies
Associated preprints: [https://arxiv.org/abs/2006.03653](https://arxiv.org/abs/2207.01372)

## Installation and prerequisite:
See instructions available at [https://github.com/CIA-Oceanix/4dvarnet-core/Readme.md](https://github.com/CIA-Oceanix/4dvarnet-core/Readme.md)

## Trained models and associated SSH reconstruction results:
See [https://doi.org/10.5281/zenodo.7429391](https://doi.org/10.5281/zenodo.7429391)

## Visualisation of the results
See notebook_visu_4DvarNet_SLANATL60_SST.ipynb

## OSSE data
The data is available at
```
# obs
https://s3.eu-central-1.wasabisys.com/melody/NATL/data/gridded_data_swot_wocorr/dataset_nadir_0d_swot.nc
https://s3.eu-central-1.wasabisys.com/melody/NATL/data/gridded_data_swot_wocorr/dataset_nadir_0d.nc
https://s3.eu-central-1.wasabisys.com/melody/NATL/data/gridded_data_swot_wocorr/dataset_swot.nc

#oi
https://s3.eu-central-1.wasabisys.com/melody/NATL/oi/ssh_NATL60_4nadir.nc
https://s3.eu-central-1.wasabisys.com/melody/NATL/oi/ssh_NATL60_swot_4nadir.nc
https://s3.eu-central-1.wasabisys.com/melody/NATL/oi/ssh_NATL60_swot.nc

#ref
https://s3.eu-central-1.wasabisys.com/melody/NATL/ref/NATL60-CJM165_NATL_ssh_y2013.1y.nc
https://s3.eu-central-1.wasabisys.com/melody/NATL/ref/NATL60-CJM165_NATL_sst_y2013.1y.nc
```

## Running 4dvarnet code
See [README.md](https://github.com/CIA-Oceanix/4dvarnet-core/blob/4dvarnet-mm-tgrs/hydra_config/README.md)

## Preprints and Software License
License: CECILL-C license

Copyright IMT Atlantique/OceaniX, contributor(s) : R. Fablet, Q. Febvre (IMT Atlantique) 01/12/2022

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
