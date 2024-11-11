checkpoint_config = dict(interval=6)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/sparse4d_r101_H1'
load_from = 'fcos3d.pth'
resume_from = None
workflow = [('train', 1)]
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
fp16 = dict(loss_scale=32.0)
optimizer = dict(
    type='AdamW',
    lr=0.0002,
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.1))),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    min_lr_ratio=0.001)
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
num_classes = 10
embed_dims = 256
num_groups = 8
num_decoder = 6
model = dict(
    type='Sparse4D',
    use_grid_mask=True,
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        frozen_stages=1,
        norm_eval=True,
        style='caffe',
        with_cp=True,
        out_indices=(0, 1, 2, 3),
        stage_with_dcn=(False, False, True, True),
        norm_cfg=dict(type='BN2d', requires_grad=False),
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False)),
    img_neck=dict(
        type='FPN',
        num_outs=4,
        start_level=1,
        out_channels=256,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
        in_channels=[256, 512, 1024, 2048]),
    head=dict(
        type='Sparse4DHead',
        cls_threshold_to_reg=0.05,
        num_decoder=6,
        instance_bank=dict(
            type='InstanceBank',
            num_anchor=900,
            embed_dims=256,
            anchor='nuscenes_kmeans900.npy',
            anchor_handler=dict(type='SparseBox3DKeyPointsGenerator')),
        anchor_encoder=dict(
            type='SparseBox3DEncoder', embed_dims=256, vel_dims=3),
        graph_model=dict(
            type='MultiheadAttention',
            embed_dims=256,
            num_heads=8,
            batch_first=True,
            dropout=0.1),
        norm_layer=dict(type='LN', normalized_shape=256),
        ffn=dict(
            type='FFN',
            embed_dims=256,
            feedforward_channels=512,
            num_fcs=2,
            ffn_drop=0.1,
            act_cfg=dict(type='ReLU', inplace=True)),
        deformable_model=dict(
            type='DeformableFeatureAggregation',
            embed_dims=256,
            num_groups=8,
            num_levels=4,
            num_cams=6,
            proj_drop=0.1,
            kps_generator=dict(
                type='SparseBox3DKeyPointsGenerator',
                num_learnable_pts=6,
                fix_scale=[[0, 0, 0], [0.45, 0, 0], [-0.45, 0, 0],
                           [0, 0.45, 0], [0, -0.45, 0], [0, 0, 0.45],
                           [0, 0, -0.45]])),
        refine_layer=dict(
            type='SparseBox3DRefinementModule', embed_dims=256, num_cls=10),
        sampler=dict(
            type='SparseBox3DTarget',
            cls_weight=2.0,
            box_weight=0.25,
            reg_weights=[2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            cls_wise_reg_weights=dict(
                {9: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]})),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_reg=dict(type='L1Loss', loss_weight=0.25),
        gt_cls_key='gt_labels_3d',
        gt_reg_key='gt_bboxes_3d',
        decoder=dict(type='SparseBox3DDecoder'),
        reg_weights=[2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        kps_generator=dict(
            type='SparseBox3DKeyPointsGenerator',
            fix_scale=[[0, 0, 0], [0.45, 0, 0], [-0.45, 0, 0], [0, 0.45, 0],
                       [0, -0.45, 0], [0, 0, 0.45], [0, 0, -0.45]]),
        depth_module=dict(type='DepthReweightModule', embed_dims=256)))
dataset_type = 'NuScenes3DDetTrackDataset'
data_root = 'data/nuscenes/'
anno_root = 'data/nuscenes_cam/'
file_client_args = dict(backend='disk')
img_crop_range = [260, 900, 0, 1600]
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='CustomCropMultiViewImage', crop_range=[260, 900, 0, 1600]),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(
        type='NormalizeMultiviewImage',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(
        type='CircleObjectRangeFilter',
        class_dist_thred=[55, 55, 55, 55, 55, 55, 55, 55, 55, 55]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(type='NuScenesSparse4DAdaptor'),
    dict(
        type='Collect3D',
        keys=[
            'gt_bboxes_3d', 'gt_labels_3d', 'img', 'timestamp',
            'projection_mat', 'image_wh'
        ],
        meta_keys=['timestamp', 'T_global', 'T_global_inv'])
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='CustomCropMultiViewImage', crop_range=[260, 900, 0, 1600]),
    dict(
        type='NormalizeMultiviewImage',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        with_label=False),
    dict(type='NuScenesSparse4DAdaptor'),
    dict(
        type='Collect3D',
        keys=['img', 'timestamp', 'projection_mat', 'image_wh'],
        meta_keys=['timestamp', 'T_global', 'T_global_inv'])
]
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
data_basic_config = dict(
    type='NuScenes3DDetTrackDataset',
    data_root='data/nuscenes/',
    classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    modality=dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False),
    box_type_3d='LiDAR',
    version='v1.0-trainval')
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='NuScenes3DDetTrackDataset',
        data_root='data/nuscenes/',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        box_type_3d='LiDAR',
        version='v1.0-trainval',
        ann_file='data/nuscenes_cam/nuscenes_infos_train.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='CustomCropMultiViewImage',
                crop_range=[260, 900, 0, 1600]),
            dict(type='PhotoMetricDistortionMultiViewImage'),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_attr_label=False),
            dict(
                type='CircleObjectRangeFilter',
                class_dist_thred=[55, 55, 55, 55, 55, 55, 55, 55, 55, 55]),
            dict(
                type='ObjectNameFilter',
                classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ]),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ]),
            dict(type='NuScenesSparse4DAdaptor'),
            dict(
                type='Collect3D',
                keys=[
                    'gt_bboxes_3d', 'gt_labels_3d', 'img', 'timestamp',
                    'projection_mat', 'image_wh'
                ],
                meta_keys=['timestamp', 'T_global', 'T_global_inv'])
        ],
        test_mode=False),
    val=dict(
        type='NuScenes3DDetTrackDataset',
        data_root='data/nuscenes/',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        box_type_3d='LiDAR',
        version='v1.0-trainval',
        ann_file='data/nuscenes_cam/nuscenes_infos_val.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='CustomCropMultiViewImage',
                crop_range=[260, 900, 0, 1600]),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(type='NuScenesSparse4DAdaptor'),
            dict(
                type='Collect3D',
                keys=['img', 'timestamp', 'projection_mat', 'image_wh'],
                meta_keys=['timestamp', 'T_global', 'T_global_inv'])
        ],
        test_mode=True),
    test=dict(
        type='NuScenes3DDetTrackDataset',
        data_root='data/nuscenes/',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        box_type_3d='LiDAR',
        version='v1.0-trainval',
        ann_file='data/nuscenes_cam/nuscenes_infos_val.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='CustomCropMultiViewImage',
                crop_range=[260, 900, 0, 1600]),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(type='NuScenesSparse4DAdaptor'),
            dict(
                type='Collect3D',
                keys=['img', 'timestamp', 'projection_mat', 'image_wh'],
                meta_keys=['timestamp', 'T_global', 'T_global_inv'])
        ],
        test_mode=True))
vis_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['img'], meta_keys=['timestamp', 'lidar2img'])
]
total_epochs = 24
evaluation = dict(
    interval=24,
    pipeline=[
        dict(type='LoadMultiViewImageFromFiles', to_float32=True),
        dict(
            type='DefaultFormatBundle3D',
            class_names=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            with_label=False),
        dict(
            type='Collect3D',
            keys=['img'],
            meta_keys=['timestamp', 'lidar2img'])
    ])
runner = dict(type='EpochBasedRunner', max_epochs=24)
gpu_ids = range(0, 1)
