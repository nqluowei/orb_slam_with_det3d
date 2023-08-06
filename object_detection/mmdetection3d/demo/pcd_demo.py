# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmdet3d.apis import inference_detector, init_model, show_result_meshlab


#              0       1        2       3        4        5       6              7            8            9
cls_names = ['bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser', 'night_stand', 'bookshelf', 'bathtub']


import torch
import numpy as np
# 关闭numpy中的科学计数输出
np.set_printoptions(precision=4, suppress=True)
# 关闭pytorch中的科学计数输出
torch.set_printoptions(precision=4, sci_mode=False)


def main():
    parser = ArgumentParser()
    parser.add_argument('pcd', help='Point cloud file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.0, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='demo', help='dir to save results')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show online visualization results')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visualization results')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    print(model)

    # test a single image
    result, data = inference_detector(model, args.pcd)


    print("data=", data)
    print("result=",result)

    # show the results
    show_result_meshlab(
        data,
        result,
        args.out_dir,
        args.score_thr,
        show=args.show,
        snapshot=args.snapshot,
        task='det')


if __name__ == '__main__':
    main()


# data= {'img_metas': [[{'flip': False, 'pcd_horizontal_flip': False, 'pcd_vertical_flip': False,
#                        'box_mode_3d': <Box3DMode.DEPTH: 2>,
#                       'box_type_3d': <class 'mmdet3d.core.bbox.structures.depth_box3d.DepthInstance3DBoxes'>,
# 'pcd_trans': array([0., 0., 0.]),
# 'pcd_scale_factor': 1.0,
# 'pcd_rotation': tensor([[1., 0., 0.],
#         [-0., 1., 0.],
#         [0., 0., 1.]]), 'pcd_rotation_angle': 0.0,
# 'pts_filename': '/home/luowei/ai/object_detection/multi_view_3d_detection/output/450.bin',
# 'transformation_3d_flow': ['R', 'S', 'T']}]],
#
#
# 'points': [[tensor([[  3.7324,  -1.3174,   0.3442,   2.0000,   2.0000,   4.0000],
#         [  3.2382,  -1.4039,   1.3150, 202.0000, 193.0000, 198.0000],
#         [  1.6631,  -0.7850,   0.2554,  30.0000,  28.0000,  39.0000],
#         ...,
#         [  2.6186,  -0.2481,   0.4062,  20.0000,   4.0000,  14.0000],
#         [  2.6186,  -1.2983,   1.0156,  98.0000,  89.0000,  94.0000],
#         [  3.6541,  -1.0957,   0.4041,   0.0000,   0.0000,   0.0000]],
#        device='cuda:0')]]}
