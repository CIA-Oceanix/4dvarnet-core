# Hydra usage

## Intro
More info [on the website](https://hydra.cc/)

In short hydra allows for splitting config into groups and composing them back.
It also provide helpers to specify python objects thourgh the config and instantiating them.


## In short:
### The experiment entrypoint is hydra\_main.py
### The final config needs to contain:
- **The datamodule object**
e.g. :
```yaml
datamodule:
  _target_: new_dataloading.FourDVarNetDataModule
  kwarg1: value1
  ....
```

- **The Lightning module class**

e.g. :
```yaml
lit_mod_cls: lit_model.LitModel
```

- **the parameters of the experiment**
e.g. :
```yaml
params:
  nb_grad_update: [...]
  ...

```
- **The entrypoint of the experiment**

```yaml
entrypoint: 
  _target_: main.FourDVarNetRunner.run
  max_epochs: 200
  progress_bar_refresh_rate: 5
```

### The Workflow is:

- You should not change the main.yaml
- You can add files in the subdirectories of hydra\_config
- You should not change existing files in the subdirectories except for files managed by you in the xp subdirectory


### Example commands:
- Run the experiment specified in the  `xp/sla_gf.yaml` files:
```
python hydra_main.py xp=sla_gf entrypoint=run
```

- Run the experiment in a slurm job (1 node 4 gpus) on jz specified in the  `xp/sla_gf.yaml` files:
```
python hydra_main.py xp=sla_gf entrypoint=run +backend=slurm_1x4 -m
```

- Run the two experiments specified in the  `xp/sla_gf.yaml`, `xp/sla_natl.yaml`files:
```
python hydra_main.py --multirun xp=sla_gf,sla_natl entrypoint=run 
```

- Run the experiment specified in the  `xp/sla_gf.yaml` for a test run (only on train batch, val batch test\_batch) :
```
python hydra_main.py  xp=sla_gf entrypoint=run entrypoint.fast_dev_run=True
```

- Run the experiment specified in the  `xp/sla_gf.yaml` and modify a parameter:

```
python hydra_main.py  xp=sla_gf params.n_grad=15 entrypoint=run 
```

- (To be tested) Test a trained model on a different domain :
```
python hydra_main.py  xp=sla_gf entrypoint=test entrypoint.ckpt_path=<path_withescaped_equal_signs_\=>  /domain@datamodule.dim_range: natl
```


- print the hydra help (you can set up autocomplete): 
```
python hydra_main.py --hydra-help 
```

