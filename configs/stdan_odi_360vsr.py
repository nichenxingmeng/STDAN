# STDAN: Spatio-Temporal Distortion Aware Omnidirectional Video SR (AAAI 2026).
# The generator (BasicVSRPlusPlus) contains the STCA (OPE-conditioned spatial DCN)
# and IMFR (interlaced multi-frame reconstruction) modules; the CharbonnierLoss here
# is the LSA (latitude-saliency adaptive) loss. Edit the `data/360Video/...` paths
# and the `saliency_folder` field below to point at your dataset.
#
# NOTE: the backbone reads a 4th LQ channel as the OPE positional cue
# (`condition = lqs[:, :, 3]`). It is loaded by `LoadImageFromFileList_ope` from
# precomputed OPE maps under `ope_folder` (see the pipelines below).
exp_name = 'stdan_odi_360vsr'

# model settings
model = dict(
    type='BasicVSR',
    generator=dict(
        type='BasicVSRPlusPlus',
        mid_channels=96,
        num_blocks=7,
        is_low_res_input=True,
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
        'basicvsr/spynet_20210409-c6c1bd09.pth'),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'))
# model training and testing settings
train_cfg = dict(fix_iter=5000)
test_cfg = dict(metrics=['PSNR'], crop_border=0)

# dataset settings
train_dataset_type = 'SRODIMultipleGTDataset'
val_dataset_type = 'SRODIMultipleGTDataset'
test_dataset_type = 'SRODIMultipleGTDataset'

train_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(
        # loads the LR frame + the precomputed OPE map as a 4th channel
        # (the backbone reads it as `condition = lqs[:, :, 3]` for STCA).
        # The OPE map is looked up at `{ope_folder}/{path_after 'LR_BIx4'}`.
        type='LoadImageFromFileList_ope',
        ope_folder='data/360Video/ope',
        path_split_token='LR_BIx4',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList_saliency',
        # Root folder of the precomputed saliency maps. The map for each
        # frame is looked up at `{saliency_folder}/{path_after 'HR'}`.
        saliency_folder='data/360Video/saliency',
        path_split_token='HR',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='PairedRandomCrop', gt_patch_size=256),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]

test_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(
        type='LoadImageFromFileList_ope',
        ope_folder='data/360Video/ope',
        path_split_token='LR',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'key'])
]

demo_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(
        type='LoadImageFromFileList_ope',
        ope_folder='data/360Video/ope',
        path_split_token='LR',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(type='FramesToTensor', keys=['lq']),
    dict(type='Collect', keys=['lq'], meta_keys=['lq_path', 'key'])
]

data = dict(
    workers_per_gpu=6,
    train_dataloader=dict(samples_per_gpu=2, drop_last=True, pin_memory=False),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='data/360Video/training/LR_BIx4',
            gt_folder='data/360Video/training/HR',
            num_input_frames=20,
            pipeline=train_pipeline,
            scale=4,
            test_mode=False)),
    # val
    val=dict(
        type=val_dataset_type,
        lq_folder='data/360Video/validation/LR',
        gt_folder='data/360Video/validation/HR',
        num_input_frames=100,
        pipeline=test_pipeline,
        scale=4,
        repeat=2,
        test_mode=True),
    # test
    test=dict(
        type=test_dataset_type,
        lq_folder='data/360Video/testing/LR',
        gt_folder='data/360Video/testing/HR',
        pipeline=test_pipeline,
        scale=1,
        test_mode=True),
)

# optimizer
optimizers = dict(
    generator=dict(
        type='Adam',
        lr=1e-4,
        betas=(0.9, 0.99),
        paramwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.25)})))

# learning policy
total_iters = 250000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[250000],
    restart_weights=[1],
    min_lr=1e-7)

checkpoint_config = dict(interval=1000, save_optimizer=True, by_epoch=False)
# remove gpu_collect=True in non distributed training
evaluation = dict(interval=260000, save_image=False, gpu_collect=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook'),
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
