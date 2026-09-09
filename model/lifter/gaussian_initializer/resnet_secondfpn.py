from mmengine.model import BaseModule
from mmengine.registry import MODELS
from mmseg.models import builder
from mmdet3d.registry import MODELS as mmdet3dMODELS
import torch


@MODELS.register_module()
class ResNetSecondFPN(BaseModule):
    def __init__(
        self, 
        img_backbone_config, 
        neck_config,
        img_backbone_out_indices,
        pretrained_path=None
    ):

        super().__init__()

        self.img_backbone = builder.build_backbone(img_backbone_config)
        self.img_neck = mmdet3dMODELS.build(neck_config)
        self.img_backbone_out_indices = img_backbone_out_indices
        if pretrained_path is not None:
            ckpt = torch.load(pretrained_path, map_location='cpu')
            ckpt = ckpt.get("state_dict", ckpt)
            print(self.load_state_dict(ckpt, strict=False))
            print("ResNetSecondFPN Weight Loaded Successfully.")

    def forward(self, imgs):
        img_feats_backbone = self.img_backbone(imgs)
        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())
        img_feats = []
        for idx in self.img_backbone_out_indices:
            img_feats.append(img_feats_backbone[idx])
        
        # FPN上采样后拼接时可能出现尺寸不匹配（如期望90但得到92）
        # 问题的根本原因：Stage 3输出22，x4上采样后=88，无法精确对齐到90
        # 解决方案：在deconv上采样后，使用interpolate精确对齐到目标尺寸（stage 1的尺寸）
        target_size = img_feats[1].shape[2:]  # (H, W) - stage 1的尺寸作为目标（通常是90x160）
        
        # 手动处理：通过每个deblock，然后对齐尺寸
        processed_feats = []
        for feat, deblock in zip(img_feats, self.img_neck.deblocks):
            upsampled = deblock(feat)
            # 对齐到目标尺寸（解决deconv上采样无法精确匹配的问题）
            if upsampled.shape[2:] != target_size:
                upsampled = torch.nn.functional.interpolate(
                    upsampled, size=target_size, mode='bilinear', align_corners=False
                )
            processed_feats.append(upsampled)
        
        # 拼接所有特征图
        secondfpn_out = torch.cat(processed_feats, dim=1)
        return secondfpn_out
