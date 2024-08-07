# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, ConfigDict, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet3d.utils import replace_ceph_backend


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet3D test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--ceph', action='store_true', help='Use ceph as data storage backend')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--score-thr', type=float, default=0.1, help='bbox score threshold')
    parser.add_argument(
        '--task',
        type=str,
        choices=[
            'mono_det', 'multi-view_det', 'lidar_det', 'lidar_seg',
            'multi-modality_det'
        ],
        help='Determine the visualization method depending on the task.')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/test.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument(
        '--test_c', action='store_true', help='test corruption')
    parser.add_argument(
        '--log', default='log', type=str, help='save result')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualization_hook['test_out_dir'] = args.show_dir
        all_task_choices = [
            'mono_det', 'multi-view_det', 'lidar_det', 'lidar_seg',
            'multi-modality_det'
        ]
        assert args.task in all_task_choices, 'You must set '\
            f"'--task' in {all_task_choices} in the command " \
            'if you want to use visualization hook'
        visualization_hook['vis_task'] = args.task
        visualization_hook['score_thr'] = args.score_thr
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)

    # TODO: We will unify the ceph support approach with other OpenMMLab repos
    if args.ceph:
        cfg = replace_ceph_backend(cfg)

    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:
        # Currently, we only support tta for 3D segmentation
        # TODO: Support tta for 3D detection
        assert 'tta_model' in cfg, 'Cannot find ``tta_model`` in config.'
        assert 'tta_pipeline' in cfg, 'Cannot find ``tta_pipeline`` in config.'
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)

    if args.test_c:
        corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                       'defocus_blur', 'glass_blur', 'motion_blur',  'zoom_blur',
                       'snow', 'frost', 'fog', 'brightness',
                       'contrast',  'elastic_transform', 'pixelate', 'jpeg_compression']
        total_dict = dict()
        for corruption in corruptions:
            # for severity in range(5):
            severity = 2
            cfg.test_dataloader.dataset.data_prefix['img'] = f'val_c/{corruption}/{severity+1}'
            if corruption in ['fog', 'snow', 'motion_blur']:
                cfg.test_dataloader.dataset.data_prefix['pts'] = f'val_c/{corruption}/moderate/velodyne'

            # build the runner from config
            if 'runner_type' not in cfg:
                # build the default runner
                runner = Runner.from_cfg(cfg)
            else:
                # build customized runner from the registry
                # if 'runner_type' is set in the cfg
                runner = RUNNERS.build(cfg)

            # start testing
            metric_dict = runner.test()
            print(metric_dict)

            # key = f'{corruption}/AP40_Moderate_0.7_3D_Car'
            # total_dict[key] = metric_dict['Kitti metric/pred_instances_3d/KITTI/Car_3D_AP40_moderate_strict']

            for key in metric_dict.keys():
                if 'Car_3D_AP40_moderate_strict' in key:
                    new_key = f"{corruption}/{key.split('/')[-1]}"
                    total_dict[new_key] = metric_dict[key]
                elif 'Pedestrian_3D_AP40_moderate_loose' in key:
                    new_key = f"{corruption}/{key.split('/')[-1]}"
                    total_dict[new_key] = metric_dict[key]
                elif 'Cyclist_3D_AP40_moderate_loose' in key:
                    new_key = f"{corruption}/{key.split('/')[-1]}"
                    total_dict[new_key] = metric_dict[key]
                elif 'Car_2D_AP40_moderate_strict' in key:
                    new_key = f"{corruption}/{key.split('/')[-1]}"
                    total_dict[new_key] = metric_dict[key]
                elif 'Pedestrian_2D_AP40_moderate_loose' in key:
                    new_key = f"{corruption}/{key.split('/')[-1]}"
                    total_dict[new_key] = metric_dict[key]
                elif 'Cyclist_2D_AP40_moderate_loose' in key:
                    new_key = f"{corruption}/{key.split('/')[-1]}"
                    total_dict[new_key] = metric_dict[key]

        for key, value in total_dict.items():
            print(f'{key}: {value}')

        with open(f'/ws/external/work_dirs/{args.log}.txt', 'w') as f:
            for key, value in total_dict.items():
                f.write(f'{key}: {value}\n')

if __name__ == '__main__':
    main()
