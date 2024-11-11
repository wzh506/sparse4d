from __future__ import division
import sys
import os

print(sys.executable, os.path.abspath(__file__))
sys.path.append('/home/wzh/study/github/3D object detection/Sparse4D')#额外配置路径，否则无法找到mmdet3d
# import init_paths # for conda pkgs submitting method
import argparse
import copy
import mmcv
import time
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from os import path as osp

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version
from torch import distributed as dist
from train import parse_args
import cv2

cv2.setNumThreads(8)

import os
print('------------------------------------------------------------------------')
print('os',os.getcwd())



# python tools/wzh.py projects/configs/sparse4d_r101_H1.py --work-dir=work_dirs/sparse4d_r101_H1
# 目标：创建一个model以及一个dataset，然后forward，其他的不管
if __name__ == "__main__":
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
        print('ok2-----------------------------------')
    # import modules from string list.
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        print('ok1---------------------------------')
        import_modules_from_strings(**cfg["custom_imports"])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, "plugin"):
        if cfg.plugin:
            import importlib

            if hasattr(cfg, "plugin_dir"):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split("/")
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                print('_module_path1:',_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split("/")
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                print('_module_path1',_module_path)
                plg_lib = importlib.import_module(_module_path)
            from projects.mmdet3d_plugin.apis.train import custom_train_model

    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer["lr"] = cfg.optimizer["lr"] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    elif args.launcher == "mpi_nccl":
        distributed = True

        import mpi4py.MPI as MPI

        comm = MPI.COMM_WORLD
        mpi_local_rank = comm.Get_rank()
        mpi_world_size = comm.Get_size()
        print(
            "MPI local_rank=%d, world_size=%d"
            % (mpi_local_rank, mpi_world_size)
        )

        # num_gpus = torch.cuda.device_count()
        device_ids_on_machines = list(range(args.gpus_per_machine))
        str_ids = list(map(str, device_ids_on_machines))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str_ids)
        torch.cuda.set_device(mpi_local_rank % args.gpus_per_machine)

        dist.init_process_group(
            backend="nccl",
            init_method=args.dist_url,
            world_size=mpi_world_size,
            rank=mpi_local_rank,
        )

        cfg.gpu_ids = range(mpi_world_size)
        print("cfg.gpu_ids:", cfg.gpu_ids)
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    # mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    # print('cfg.work_dir',cfg.work_dir)

    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    if cfg.model.type in ["EncoderDecoder3D"]:
        logger_name = "mmseg"
    else:
        logger_name = "mmdet"
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name
    )

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    logger.info(
        "Environment info:\n" + dash_line + env_info + "\n" + dash_line
    )
    meta["env_info"] = env_info
    meta["config"] = cfg.pretty_text

    # log some basic info
    logger.info(f"Distributed training: {distributed}")
    logger.info(f"Config:\n{cfg.pretty_text}")

    # set random seeds
    if args.seed is not None:
        logger.info(
            f"Set random seed to {args.seed}, "
            f"deterministic: {args.deterministic}"
        )
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta["seed"] = args.seed
    meta["exp_name"] = osp.basename(args.config)

    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
