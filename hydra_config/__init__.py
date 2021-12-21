# from hydra.core.config_store import ConfigStore
# from omegaconf import OmegaConf
# import importlib

# LEGACY_CONFIGS = {
#     # 'direct_phi':  'config_q.xp_icassp.direct_phi',
#     # 'direct_vit':  'config_q.xp_icassp.direct_vit',
#     # 'fourdvarnet_calmap':  'config_q.xp_icassp.fourdvarnet_calmap',
#     # 'nad_swot':  'config_q.nad_swot',
#     # 'icassp_fourdvarnet_calmap':  'icassp_code_bis.config_q.xp_icassp.fourdvarnet_calmap',
#     # 'fourdvarnet_calmapgrad':  'config_q.xp_icassp.fourdvarnet_calmapgrad',
#     # 'fourdvarnet_map':  'config_q.xp_icassp.fourdvarnet_map',
# }

# cs = ConfigStore.instance()
# for name, cfg in LEGACY_CONFIGS.items():
#     config = importlib.import_module(str(cfg))
#     cs.store(name=name, node=config.params, group='legacy', package='params')

