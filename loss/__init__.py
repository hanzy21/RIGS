from mmengine.registry import Registry
OPENOCC_LOSS = Registry('openocc_loss')

from .multi_loss import MultiLoss
from .occupancy_loss import OccupancyLoss
from .bce_loss import BinaryCrossEntropyLoss, PixelDistributionLoss
from .evidential_loss import EvidentialLoss
from .velocity_loss import VelocityLoss
from .radar_depth_loss import RadarDepthLoss
