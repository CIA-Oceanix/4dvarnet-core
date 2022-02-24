import inspect
import hydra
import hydra_config
import shlex
from omegaconf import OmegaConf
from pathlib import Path
import subprocess

"""
- use relative paths (additional resolver)
- add possible overrides during exp run in configs
- Keep thinking of visidata workflow to facilitate runs etc... 
- Sort out glob targets etc
- Think about xp name overwrite etc
- Complete outs, params etc...
- check queued tmps etc
"""

XP_FILE_NAME = "xp_config"

class DvcStageBuilder:
    def __init__(self):
        self.options = []
        self.base_cmd = []
        self.app_cmd = []
        self.init_attrs()

    def init_attrs(self):
        self.options = []
        self.base_cmd = ["dvc", "stage", "add", "--force"]
        self.app_cmd = [
            "python",
            "-c",
            "import sys;sys.path.append('../..');import dvc_main as dm; dm.dvc_execute()",
        ]

    def add_opt(self, value, opt):
        if value is not None:
            self.options.append((opt, value))
        else: 
            self.options.append((opt,))
        return value

    def cmd(self):
        parts = self.base_cmd + [tt for t in set(self.options) for tt in t] + self.app_cmd
        print(parts)
        cmd = shlex.join(parts)
        self.options = []
        return cmd

def register_resolvers(stage_builder):
        OmegaConf.register_new_resolver(
            "aprl", stage_builder.add_opt, replace=True
        )
        OmegaConf.register_new_resolver(
            f"aprl_cmd", stage_builder.cmd, replace=True
        )

@hydra.main(config_path='hydra_config', config_name='dvc_main')
def dvc_dump(cfg):
    sb = DvcStageBuilder()
    register_resolvers(sb)
    OmegaConf.resolve(cfg)
    Path(f'{ XP_FILE_NAME }.yaml').write_text(OmegaConf.to_yaml(cfg))
    print(cfg.dvc.cmd)
    ret_code = subprocess.call(shlex.split(cfg.dvc.cmd))
    print(ret_code)


def dvc_execute():
    import hydra
    import os
    print(os.getcwd())
    with hydra.initialize_config_dir(config_dir=str(Path('.').absolute())):
        cfg = hydra.compose(config_name=XP_FILE_NAME)

    print(OmegaConf.to_yaml(cfg))
    # hydra.utils.call(cfg.dvc.entrypoint, cfg=cfg)

if __name__ == '__main__':
    dvc_dump()
