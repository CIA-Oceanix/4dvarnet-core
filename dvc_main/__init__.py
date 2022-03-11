import platform
from datetime import date

from pytorch_lightning import Callback
from git.repo import Repo as GitRepo
from git import IndexFile
import os
import traceback
import hydra
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
# HOME_SYMLINK = "__home"

class DvcStageBuilder:
    def __init__(self):
        self.options = []
        self.base_cmd = []
        self.app_cmd = []
        self.init_attrs()

    def init_attrs(self):
        self.options = []
        self.base_cmd = ["dvc", "stage", "add", "--force"]
        self.app_cmd =  ["PYTHONPATH=${original_cwd:} python -m dvc_main.run"]
        # self.app_cmd =  [f"PYTHONPATH=__home python -m dvc_main.run"]
        # self.app_cmd =  [f"python dvc_main/run.py"]


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



def commit_cwd(branch, message, repo=None):
    # TODO : full test suite with fixtures
    # TODO : test with Keyboard interrupt
    repo = repo or GitRepo('.')
    index = IndexFile(repo)
    log_branch = getattr(repo.heads, branch, False) or repo.create_head(branch)

    to_add = (
                     set(repo.untracked_files)
                     | set([d.a_path or d.b_path for d in index.diff(None) if d.change_type != 'D'])
                     | set([d.a_path or d.b_path for d in index.diff(repo.head.commit) if d.change_type != 'A'])
             ) - set([d.a_path or d.b_path for d in index.diff(None) if d.change_type == 'D'])

    index.add(
        to_add,
        force=False,
        write=False,
        write_extension_data=True,
    )

    log_commit = index.commit(message, parent_commits=[log_branch.commit], head=False)
    log_branch.commit = log_commit
    return log_commit

class VersioningCallback(Callback):
    def __init__(self, repo_path='.', branch='run_log', push=False):
        self.git_repo = GitRepo(repo_path)
        self.setup_hash = None
        self.date = str(date.today())
        self.branch = f'{branch}-{self.date}-{platform.node()}'
        self.push = push

    def setup(self, trainer, pl_module, stage):
        msg = ''
        if trainer.logger is not None:
            msg = trainer.logger.log_dir
        commit = commit_cwd(
            self.branch,
            f'Setup {stage} {msg}',
            repo=self.git_repo
        )
        self.setup_hash = str(commit)


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

# def rel_with_symlink(path_a):
#     abs_a = Path(hydra.utils.to_absolute_path(path_a))
#     abs_b = Path(hydra.utils.to_absolute_path('.'))
#     common_parent = max(set(abs_a.parents) & set(abs_b.parents), key=lambda p: len(str(p)))

#     return str(
#              Path(HOME_SYMLINK) / (
#                  abs_b /
#                 '/'.join(['..'] * len(list(abs_b.relative_to(common_parent).parents))) /
#                 abs_a.relative_to(common_parent)
#             ).relative_to(abs_b)
#     )
def register_resolvers(stage_builder):
        OmegaConf.register_new_resolver(
            "original_cwd", hydra.utils.get_original_cwd, replace=True
        )

        OmegaConf.register_new_resolver(
            "cwd", os.getcwd, replace=True
        )
        OmegaConf.register_new_resolver(
            "rel_path", rel_with_backward_search, replace=True
        )
        # OmegaConf.register_new_resolver(
        #     "rel_sl_path", rel_with_symlink, replace=True
        # )
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
    OmegaConf.resolve(cfg)
    Path(f'{ XP_FILE_NAME }.yaml').write_text(OmegaConf.to_yaml(cfg))
    print(cfg.dvc.create_stage_cmd)
    # with open(cfg.dvc.get('log_file', 'dvc.log'), 'w') as f:
    ret_code_dump= subprocess.run(shlex.split(cfg.dvc.create_stage_cmd))
    ret_code_dump.check_returncode()

    # if cfg.dvc.mk_home_symlink:
    #     print("Making home symlink")
    #     Path(HOME_SYMLINK).unlink(missing_ok=True)
    #     os.symlink(hydra.utils.get_original_cwd(),HOME_SYMLINK, target_is_directory=True)

    print(cfg.dvc.run_cmd)
    # ret_code_run= subprocess.check_call(shlex.split(cfg.dvc.run_cmd))
    # ret_code_run= subprocess.run(shlex.split(cfg.dvc.run_cmd), shell=True)
    ret_code_run= subprocess.run(cfg.dvc.run_cmd, shell=True)
    # ret_code_run.check_returncode()
    print('I am out')


def dvc_execute():
    try:
        with hydra.initialize_config_dir(config_dir=str(Path('.').absolute())):
            cfg = hydra.compose(config_name=XP_FILE_NAME)

        print(OmegaConf.to_yaml(cfg))
        print(f'calling {cfg.dvc.entrypoint}')
        hydra.utils.call(cfg.dvc.entrypoint, cfg=cfg, _recursive_=False)
    except:
        print('Failed')
        print(traceback.format_exc()) 

