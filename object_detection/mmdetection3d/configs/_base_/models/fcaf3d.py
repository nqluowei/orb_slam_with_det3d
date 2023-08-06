model = dict(
    type='MinkSingleStage3DDetector',
    voxel_size=.01,
    backbone=dict(type='MinkResNet', in_channels=3, depth=34), #dept是网络深度
    head=dict(
        type='FCAF3DHead',
        in_channels=(64, 128, 256, 512),
        out_channels=128,
        voxel_size=.01,
        pts_prune_threshold=100000,
        pts_assign_threshold=27,
        pts_center_threshold=18,
        n_classes=18,
        n_reg_outs=6),
    train_cfg=dict(),
    #test_cfg=dict(nms_pre=1000, iou_thr=.5, score_thr=.01))
    test_cfg=dict(nms_pre=100, iou_thr=.1, score_thr=.15)) #nms_pre是计算分数的topk个数，可看作是最大目标个数，IOU越大重合越厉害，在做nms之前会通过score_thr过滤一遍
