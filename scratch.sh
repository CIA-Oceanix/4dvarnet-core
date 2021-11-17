# Should be all icassp
python -m icassp_code_bis.main --config=icassp_code_bis.config_q.xp_icassp.fourdvarnet_calmap train


# Hydra change only main
python hydra_main.py  xp=calmap_gf entrypoint=train
