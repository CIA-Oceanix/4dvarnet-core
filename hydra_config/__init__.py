#import hydra_config.quentins_xps
from omegaconf import OmegaConf
import pandas as pd

OmegaConf.register_new_resolver("mul", lambda x,y: int(x)*y, replace=True)
OmegaConf.register_new_resolver("div", lambda x,y: x//int(y), replace=True)
OmegaConf.register_new_resolver("adjust_testslices", lambda x,y:
        [str((pd.to_datetime(x[0]) - pd.to_timedelta(f'{int(y)//2}D')).date()),
         str((pd.to_datetime(x[1]) + pd.to_timedelta(f'{1+int(y)//2}D')).date())],
        replace=True)