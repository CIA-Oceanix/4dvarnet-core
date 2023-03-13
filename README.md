# Scale-aware neural calibration for wide swath altimetry observations
This repo contains the code asociated with the paper [https://doi.org/10.48550/arXiv.2302.04497](https://doi.org/10.48550/arXiv.2302.04497)

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
conda create -n 4dvarnet mamba pytorch=1.11 torchvision cudatoolkit=11.3 -c conda-forge -c pytorch
conda activate 4dvarnet
mamba install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
mamba env update -f environment.yaml
```

### Download the data
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


### Training scripts
See the README.md in [hydra\_config](https://github.com/CIA-Oceanix/4dvarnet-core/tree/main/hydra_config)

The 4dvarnets models have been trained with the commands:

```
python hydra_main.py xp=qxp20_5nad_no_sst file_paths=<XXX>
python hydra_main.py xp=qxp20_5nad_sst file_paths=<XXX>
```

The CalCNN models training code is in `scratches/qf_swath_calib_tgrs.py`
