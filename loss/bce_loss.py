import torch, torch.nn as nn
import torch.nn.functional as F

from . import OPENOCC_LOSS
from .base_loss import BaseLoss


@OPENOCC_LOSS.register_module()
class BinaryCrossEntropyLoss(BaseLoss):

    def __init__(
        self,
        weight=1.0,
        empty_label=0,
        class_weights=[1.0, 1.0],
        input_dict=None
    ):
        
        super().__init__()

        self.weight = weight
        if input_dict is None:
            self.input_dict = {
                'bin_logits': 'bin_logits',
                'sampled_label': 'sampled_label',
                'occ_mask': 'occ_mask'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.loss_voxel

        self.empty_label = empty_label
        self.class_weights = torch.tensor(class_weights)
        self.class_weights = 2 * F.normalize(self.class_weights, 1, -1)
        print(self.__class__, self.class_weights)

    def loss_voxel(self, bin_logits, sampled_label, occ_mask=None):

        tot_loss = 0.
        sampled_label = sampled_label != self.empty_label
        sample_weight = torch.ones_like(sampled_label)
        sample_weight[sampled_label] = self.class_weights[1]
        sample_weight[~sampled_label] = self.class_weights[0]

        if occ_mask is not None:
            occ_mask = occ_mask.flatten(1)
            sampled_label = sampled_label[occ_mask][None]
            sample_weight = sample_weight[occ_mask][None]
        
        for semantics in bin_logits: # b, n
            if occ_mask is not None:
                semantics = semantics[occ_mask][None] # 1, n
            semantics = torch.clamp(semantics.float(), 1e-6, 1 - 1e-6)
            # BCE 在 AMP 下被禁止，需在 autocast 外执行
            with torch.amp.autocast(device_type='cuda', enabled=False):
                loss = nn.functional.binary_cross_entropy(semantics, sampled_label.float(), sample_weight.float())
            tot_loss = tot_loss + loss
        return tot_loss
    

@OPENOCC_LOSS.register_module()
class PixelDistributionLoss(BaseLoss):

    def __init__(
        self,
        weight=1.0,
        use_sigmoid=True,
        input_dict=None
    ):
        
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'pixel_logits': 'pixel_logits',
                'pixel_gt': 'pixel_gt',
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.loss_voxel
        self.use_sigmoid = use_sigmoid

    def loss_voxel(self, pixel_logits, pixel_gt):
        pixel_gt = pixel_gt.float()
        if self.use_sigmoid:
            # 使用 BCEWithLogits（logits 输入），AMP 安全且数值更稳定
            loss = nn.functional.binary_cross_entropy_with_logits(pixel_logits, pixel_gt)
        else:
            pixel_logits = torch.softmax(pixel_logits, dim=-1)
            # BCE 在 AMP 下被禁止，需在 autocast 外执行
            with torch.amp.autocast(device_type='cuda', enabled=False):
                loss = nn.functional.binary_cross_entropy(pixel_logits.float(), pixel_gt)
        return loss
    
@OPENOCC_LOSS.register_module()
class OccDepthLoss(BaseLoss):

    def __init__(
        self,
        weight=1.0,
        input_dict=None
    ):
        
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'pixel_logits': 'pixel_logits',
                'pixel_gt': 'pixel_gt',
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.loss_voxel

    def loss_voxel(self, pixel_logits, pixel_gt):
        pixel_logits = pixel_logits.permute(0, 4, 1, 2, 3)
        ### get depth gt from occ
        occ_depth = pixel_gt.float().argmax(dim=-1) # b, n, h, w
        loss = nn.functional.cross_entropy(pixel_logits, occ_depth)
        return loss
