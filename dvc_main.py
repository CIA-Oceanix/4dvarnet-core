import inspect
import traceback
import hydra
import hydra_config
import shlex
from omegaconf import OmegaConf
from pathlib import Path
import subprocess

"""
- [x] check queued tmps etc
- [x] use relative paths (additional resolver)
- [ ] test entrypoint
- [ ] check queued tmps etc
- [ ] add possible overrides during exp run in configs
- [ ] Sort out glob targets etc
- [ ] Think about xp name overwrite etc
- [ ] Complete outs, params etc...
- [ ] Keep thinking of visidata workflow to facilitate runs etc... 
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


def rel_with_backward_search(path_a, path_b=None):
    if path_b is None:
        path_b = '.'
    abs_a = Path(hydra.utils.to_absolute_path(path_a))
    abs_b = Path(path_b).absolute()
    common_parent = max(set(abs_a.parents) & set(abs_b.parents), key=lambda p: len(str(p)))

    return str(
            (
                Path(path_b) /
                '/'.join(['..'] * len(list(abs_b.relative_to(common_parent).parents))) /
                abs_a.relative_to(common_parent)
            ).relative_to(path_b)
    )

def register_resolvers(stage_builder):

        OmegaConf.register_new_resolver(
            "rel_path", rel_with_backward_search, replace=True
        )
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
    print(cfg.dvc.create_stage_cmd)
    # with open(cfg.dvc.get('log_file', 'dvc.log'), 'w') as f:
    ret_code_dump= subprocess.call(shlex.split(cfg.dvc.create_stage_cmd))
    print(ret_code_dump)
    if ret_code_dump != 0:
        raise Exception('Create stage Failed')

    print(cfg.dvc.run_cmd)
    ret_code_run= subprocess.check_call(shlex.split(cfg.dvc.run_cmd))
    print(ret_code_run)
    if ret_code_run != 0:
        raise Exception('Create stage Failed')

def test_xp(cfg):
    print("Should be here")
    import hashlib
    out_path = 'test_xp_out.txt'
    out_s = ""
    dep1  = hashlib.md5(Path(cfg.dvc.deps[0]).read_bytes())

    out_s += f"{dep1}\n"
    out_s += f"{cfg.params.loss_loc}\n"
    Path(out_path).write_text(out_s)

def dvc_execute():
    try:
        with hydra.initialize_config_dir(config_dir=str(Path('.').absolute())):
            cfg = hydra.compose(config_name=XP_FILE_NAME)

        print(OmegaConf.to_yaml(cfg))
        print(f'calling {cfg.dvc.entrypoint}')
        hydra.utils.call(cfg.dvc.entrypoint, cfg=cfg, _recursive_=False)
    except:
        print(traceback.format_exc()) 

if __name__ == '__main__':
    dvc_dump()
