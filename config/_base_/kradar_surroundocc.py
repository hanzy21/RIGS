import os

# ================== data ========================
# KRadar dataset configuration
# data_root points to the parent directory containing all sequence directories.
data_root = os.environ.get("KRADAR_DATA_ROOT", "data/K-Radar")
anno_root = os.environ.get("RIGS_INDEX_ROOT", "data/index/")
occ_path = data_root
input_shape = (720, 1280)  # 根据KRadar图像尺寸调整
batch_size = 1

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

# Training pipeline for K-Radar semantic occupancy labels.
train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True, crop_size=input_shape),
    dict(type="LoadOccupancyKRadar", occ_path=occ_path, semantic=True, use_ego=False, 
         use_sweeps=False, use_3class=True, cc_refine=False),
    dict(type="ResizeCropFlipImage"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="KRadarAdaptor", use_ego=False, num_cams=1),
]

# Deterministic pipeline used for validation and inference.
inference_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True, crop_size=input_shape),
    dict(type="LoadOccupancyKRadar", occ_path=occ_path, semantic=True, use_ego=False, 
         use_sweeps=False, use_3class=True, cc_refine=False),
    dict(type="ResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="KRadarAdaptor", use_ego=False, num_cams=1),
]

data_aug_conf = {
    "resize_lim": (0.40, 0.47),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (-5.4, 5.4),
    "H": 720,  # KRadar图像高度
    "W": 2560,  # KRadar图像宽度
    "rand_flip": True,
}

train_dataset_config = dict(
    type='KRadarDataset',
    data_root=data_root,
    imageset=anno_root + "kradar_infos_train.pkl",
    data_aug_conf=data_aug_conf,
    pipeline=train_pipeline,
    phase='train'
)

val_dataset_config = dict(
    type='KRadarDataset',
    data_root=data_root,
    imageset=anno_root + "kradar_infos_val.pkl",
    data_aug_conf=data_aug_conf,
    pipeline=inference_pipeline,
    phase='val'
)

train_loader = dict(
    batch_size=batch_size,
    num_workers=2,
    shuffle=True
)

val_loader = dict(
    batch_size=batch_size,
    num_workers=2
)

# Inference split configuration.
inference_dataset_config = dict(
    type='KRadarDataset',
    data_root=data_root,
    imageset=anno_root + "kradar_infos_test.pkl",
    data_aug_conf=data_aug_conf,
    pipeline=inference_pipeline,
    phase='test'
)
