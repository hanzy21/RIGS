from mmengine.registry import Registry
OPENOCC_DATASET = Registry('openocc_dataset')
OPENOCC_DATAWRAPPER = Registry('openocc_datawrapper')
OPENOCC_TRANSFORMS = Registry('openocc_transforms')

from .kradar_dataset import KRadarDataset
from .transform_3d import *
from .sampler import CustomDistributedSampler
from .utils import custom_collate_fn_temporal

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader


def get_dataloader(
    train_dataset_config, 
    val_dataset_config, 
    train_loader, 
    val_loader, 
    dist=False,
    iter_resume=False,
    train_sampler_config=dict(
        shuffle=True,
        drop_last=True),
    val_sampler_config=dict(
        shuffle=False,
        drop_last=False),
    val_only=False,
    use_temporal_init=False,
):
    if val_only:
        val_wrapper = OPENOCC_DATASET.build(
            val_dataset_config)
                
        val_sampler = None
        if dist:
            val_sampler = DistributedSampler(val_wrapper, **val_sampler_config)

        val_dataset_loader = DataLoader(
            dataset=val_wrapper,
            batch_size=val_loader["batch_size"],
            collate_fn=custom_collate_fn_temporal,
            shuffle=False,
            sampler=val_sampler,
            num_workers=val_loader["num_workers"],
            pin_memory=True)

        return None, val_dataset_loader

    train_wrapper = OPENOCC_DATASET.build(
        train_dataset_config)
    val_wrapper = OPENOCC_DATASET.build(
        val_dataset_config)
    
    # 如果使用时序初始化，强制shuffle=False
    if use_temporal_init:
        train_sampler_config = dict(
            shuffle=False,  # 时序初始化必须顺序读取
            drop_last=train_sampler_config.get('drop_last', True)
        )
        print('[时序初始化] 训练数据将按顺序读取（shuffle=False）')
        
    train_sampler = val_sampler = None

    if dist:
        if iter_resume:
            train_sampler = CustomDistributedSampler(train_wrapper, **train_sampler_config)
        else:
            train_sampler = DistributedSampler(train_wrapper, **train_sampler_config)
        val_sampler = DistributedSampler(val_wrapper, **val_sampler_config)

    effective_shuffle = train_loader["shuffle"]
    if use_temporal_init:
        effective_shuffle = False

    train_dataset_loader = DataLoader(
        dataset=train_wrapper,
        batch_size=train_loader["batch_size"],
        collate_fn=custom_collate_fn_temporal,
        shuffle=False if (dist or train_sampler is not None) else effective_shuffle,
        sampler=train_sampler,
        num_workers=train_loader["num_workers"],
        pin_memory=True)
    val_dataset_loader = DataLoader(
        dataset=val_wrapper,
        batch_size=val_loader["batch_size"],
        collate_fn=custom_collate_fn_temporal,
        shuffle=False,
        sampler=val_sampler,
        num_workers=val_loader["num_workers"],
        pin_memory=True)

    return train_dataset_loader, val_dataset_loader
