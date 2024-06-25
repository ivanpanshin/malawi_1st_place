_base_ = [
    'co_dino_5scale_r50_1x_coco.py'
]

#load_from = './pretrained/co_dino_5scale_swin_large_16e_o365tococo.pth'
load_from = './work_dirs/exp_007/best_bbox_mAP_epoch_11.pth'
pretrained = None
# model settings
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformerV1',
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        out_indices=(0, 1, 2, 3),
        window_size=12,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=True,
        pretrained=pretrained),
    neck=dict(in_channels=[192, 192*2, 192*4, 192*8]),
    query_head=dict(
        dn_cfg=dict(
            type='CdnQueryGenerator',
            noise_scale=dict(label=0.5, box=0.4),
            group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=500)),
        transformer=dict(
            encoder=dict(
                # number of layers that use checkpoint
                with_cp=6))))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

image_size = 1536

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=[0.5, 0.5], direction=['horizontal', 'vertical']),
    dict(
        type='AutoAugment',
        policies=[
            [
                dict(
                    type='Resize',
                    img_scale=[(480, image_size), (512, image_size), (544, image_size), (576, image_size),
                               (608, image_size), (640, image_size), (672, image_size), (704, image_size),
                               (736, image_size), (768, image_size), (800, image_size), (832, image_size),
                               (864, image_size), (896, image_size), (928, image_size), (960, image_size),
                               (992, image_size), (1024, image_size), (1056, image_size), (1088, image_size),
                               (1120, image_size), (1152, image_size), (1184, image_size), (1216, image_size),
                               (1248, image_size), (1280, image_size), (1312, image_size), (1344, image_size),
                               (1376, image_size), (1408, image_size), (1440, image_size), (1472, image_size),
                               (1504, image_size), (1536, image_size)],
                    multiscale_mode='value',
                    keep_ratio=True)
            ],
            [
                dict(
                    type='Resize',
                    # The radio of all image in train dataset < 7
                    # follow the original impl
                    img_scale=[(400, 4200), (500, 4200), (600, 4200)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='Resize',
                    img_scale=[(480, image_size), (512, image_size), (544, image_size), (576, image_size),
                               (608, image_size), (640, image_size), (672, image_size), (704, image_size),
                               (736, image_size), (768, image_size), (800, image_size), (832, image_size),
                               (864, image_size), (896, image_size), (928, image_size), (960, image_size),
                               (992, image_size), (1024, image_size), (1056, image_size), (1088, image_size),
                               (1120, image_size), (1152, image_size), (1184, image_size), (1216, image_size),
                               (1248, image_size), (1280, image_size), (1312, image_size), (1344, image_size),
                               (1376, image_size), (1408, image_size), (1440, image_size), (1472, image_size),
                               (1504, image_size), (1536, image_size)],
                    multiscale_mode='value',
                    override=True,
                    keep_ratio=True)
            ]
        ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1280),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(filter_empty_gt=False, pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

# optimizer
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0.0001,
    # custom_keys of sampling_offsets and reference_points in DeformDETR
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))

optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))

# learning policy
lr_config = dict(policy='step', step=[8])
runner = dict(type='EpochBasedRunner', max_epochs=16)
# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (16 GPUs) x (1 samples per GPU)
auto_scale_lr = dict(base_batch_size=4)

#load_from = './pretrained/co_dino_5scale_swin_large_16e_o365tococo.pth'