from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import importlib

LEGACY_CONFIGS = {
        'fourdvarnet_map':  'config_q.xp_icassp.fourdvarnet_map',
}
cs = ConfigStore.instance()
for name, cfg in LEGACY_CONFIGS.items():
    config = importlib.import_module(str(cfg))
    cs.store(name=name, node=config.params, group='legacy', package='params')

