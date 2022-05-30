import hydra_config.quentins_xps
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("mul", lambda x,y: int(x)*y, replace=True)
OmegaConf.register_new_resolver("div", lambda x,y: x//int(y), replace=True)
