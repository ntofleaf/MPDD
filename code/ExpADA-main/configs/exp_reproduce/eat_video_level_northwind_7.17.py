_base_ = [
    '../_base_/datasets/avec_video_anchor.py',
    '../_base_/models/video_anchor.py',
    '../_base_/schedules/video_anchor.py',
    '../_base_/default_runtime.py',
]

data_root = 'data/avec_2014_northwind/extracted_features'
train_pipeline = [
    # dict(type='LoadImageFromFile'),
    # dict(type='ResizeEdge', scale=224, edge='short'),
    # dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackFeatInputs', algorithm_keys=["subj_id",]),
]

test_pipeline = [
    # dict(type='LoadImageFromFile'),
    # dict(type='ResizeEdge', scale=224, edge='short'),
    dict(type='PackFeatInputs', algorithm_keys=["subj_id",]),
]

train_dataloader = dict(dataset=dict(data_root=data_root, pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(data_root=data_root, pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(data_root=data_root,pipeline=test_pipeline))

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=1359908369, deterministic=True)