from ..registry import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module
class FasterRCNN(TwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        super(FasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

@DETECTORS.register_module
class FasterRCNN_IoU(TwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 IoU_roi_extractor=None,
                 IoU_head=None,
                 reg_roi_extractor=None,
                 reg_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        super(FasterRCNN_IoU, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            IoU_roi_extractor=IoU_roi_extractor,
            IoU_head=IoU_head,
            reg_roi_extractor=reg_roi_extractor,
            reg_head=reg_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)