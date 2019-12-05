import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from ..registry import HEADS
from ..utils import ConvModule
from .bbox_head import BBoxHead
from mmdet.core import (auto_fp16, bbox_target, delta2bbox, bbox2delta, force_fp32,
                        multiclass_nms, bbox_target_IoU, bbox_overlaps)
from ..builder import build_loss
from ..losses import accuracy

@HEADS.register_module
class ConvFCBBoxHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

                                /-> cls convs -> cls fcs -> cls
    shared convs -> shared fcs
                                \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 *args,
                 **kwargs):
        super(ConvFCBBoxHead, self).__init__(*args, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(ConvFCBBoxHead, self).init_weights()
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.view(x_cls.size(0), -1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.view(x_reg.size(0), -1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred

@HEADS.register_module
class IoU_head(nn.Module):
    def __init__(self,
                 num_IoU_convs=0,
                 num_IoU_fcs=0,
                 roi_feat_size=7,
                 in_channels=256,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 num_classes=81,
                 class_agnostic=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_IoU=dict(type='SmoothL1Loss', beta=0.5, loss_weight=1.0)):
        super(IoU_head, self).__init__()
        assert (num_IoU_convs + num_IoU_fcs > 0)

        self.num_IoU_convs = num_IoU_convs
        self.num_IoU_fcs = num_IoU_fcs
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.loss_IoU = build_loss(loss_IoU)
        # add IoU specific branch
        self.IoU_convs, self.IoU_fcs, self.IoU_last_dim = \
            self._add_conv_fc_branch(
                self.num_IoU_convs, self.num_IoU_fcs, self.in_channels)

        self.relu = nn.ReLU(inplace=True)

        out_dim_IoU = (1 if self.class_agnostic else self.num_classes)
        self.fc_IoU = nn.Linear(self.IoU_last_dim, out_dim_IoU)


    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):

        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        for module_list in [self.IoU_convs, self.IoU_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # shared part
        x_IoU = x.view(x.size(0), -1)

        # IoU branch
        for conv in self.IoU_convs:
            x_IoU = conv(x_IoU)
        if x_IoU.dim() > 2:
            x_IoU = x_IoU.view(x_IoU.size(0), -1)
        for fc in self.IoU_fcs:
            x_IoU = self.relu(fc(x_IoU))

        IoU_pred  = self.fc_IoU(x_IoU)

        return IoU_pred

    @force_fp32(apply_to=('IoU_pred'))
    def loss(self,
             IoU_pred,
             IoU_targets,
             gt_indexes,
             reduction_override=None):
        losses = dict()
        if IoU_pred is not None:
            IoU_weights = IoU_pred.new_full((IoU_pred.shape[0],), 1)

            if self.class_agnostic:
                IoU_pred = IoU_pred.view(-1)
            else:
                IoU_pred = IoU_pred.view(IoU_pred.size(0), -1)

            losses['loss_IoU'] = self.loss_IoU(
                IoU_pred,
                IoU_targets,
                reduction_override=reduction_override)
        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'IoU_pred'))
    def get_det_bboxes(self,
                       rois,
                       cls_score,
                       bbox_pred,
                       IoU_pred,
                       img_shape,
                       scale_factor,
                       rescale=False,
                       cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        IoU_pred = IoU_pred.view(-1)
        if bbox_pred is not None:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1] - 1)
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0] - 1)

        if rescale:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                bboxes /= torch.from_numpy(scale_factor).to(bboxes.device)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img, score_factors=IoU_pred)

            return det_bboxes, det_labels

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'IoU_pred'))
    def get_first_det_bboxes(self,
                       rois,
                       bbox_pred,
                       img_shape):
        if bbox_pred is not None:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1] - 1)
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0] - 1)

        return bboxes

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'IoU_pred'))
    def get_final_det_bboxes(self,
                       bboxes,
                       cls_score,
                       IoU_pred,
                       img_shape,
                       scale_factor,
                       rescale=False,
                       cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        IoU_pred = IoU_pred.view(-1)
        bboxes = bboxes[:, 1:]
        if rescale:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                bboxes /= torch.from_numpy(scale_factor).to(bboxes.device)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img, score_factors=IoU_pred)

            return det_bboxes, det_labels

@HEADS.register_module
class reg_head(nn.Module):
    def __init__(self,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 roi_feat_size=7,
                 in_channels=256,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 num_classes=81,
                 class_agnostic=False,
                 target_means=[0., 0., 0., 0.],
                 target_stds=[0.1, 0.1, 0.2, 0.2],
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_reg=dict(type='SmoothL1Loss', beta=0.5, loss_weight=1.0)):
        super(reg_head, self).__init__()
        assert (num_reg_convs + num_reg_fcs > 0)

        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.target_means = target_means
        self.target_stds = target_stds
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.loss_reg = build_loss(loss_reg)
        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.in_channels)

        self.relu = nn.ReLU(inplace=True)

        out_dim_reg = (4 if self.class_agnostic else self.num_classes*4)
        self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)


    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):

        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        for module_list in [self.reg_convs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # shared part
        x_reg = x.view(x.size(0), -1)

        # reg branch
        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            x_reg = x_reg.view(x_reg.size(0), -1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        reg_pred  = self.fc_reg(x_reg)

        return reg_pred

    @force_fp32(apply_to=('reg_pred'))
    def loss(self,
             bbox_pred,
             bbox_targets,
             IoU_targets,
             reduction_override=None):
        losses = dict()

        losses['loss_reg'] = self.loss_reg(
            bbox_pred,
            bbox_targets,
            reduction_override=reduction_override)

        losses['cnt_pos'] = torch.tensor(IoU_targets.shape[0]).to(IoU_targets.device).float()
        interval = ((IoU_targets - 0.00001) / 0.1).int()
        reg_loss = self.loss_reg(
            bbox_pred,
            bbox_targets,
            reduction_override='none')
        reg_loss = reg_loss.mean(dim=1)
        losses['cnt_5'] = (interval == 5).sum().float()
        losses['reg_5'] = reg_loss[interval == 5].sum()

        losses['cnt_6'] = (interval == 6).sum().float()
        losses['reg_6'] = reg_loss[interval == 6].sum()

        losses['cnt_7'] = (interval == 7).sum().float()
        losses['reg_7'] = reg_loss[interval == 7].sum()

        losses['cnt_8'] = ((interval == 8).sum() + (interval == 9).sum()).float()
        losses['reg_8'] = reg_loss[interval == 8].sum() + reg_loss[interval == 9].sum()

        return losses

    def get_target(self, rois, gt_bboxes):
        bbox_targets = bbox2delta(rois, gt_bboxes, self.target_means, self.target_stds)
        return bbox_targets

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'IoU_pred'))
    def get_det_bboxes(self,
                       rois,
                       cls_score,
                       bbox_pred,
                       IoU_pred,
                       img_shape,
                       scale_factor,
                       rescale=False,
                       cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        IoU_pred = IoU_pred.view(-1)
        if bbox_pred is not None:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1] - 1)
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0] - 1)

        if rescale:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                bboxes /= torch.from_numpy(scale_factor).to(bboxes.device)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img, score_factors=IoU_pred)

            return det_bboxes, det_labels

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'IoU_pred'))
    def get_first_det_bboxes(self,
                       rois,
                       bbox_pred,
                       img_shape):

        if bbox_pred is not None:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1] - 1)
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0] - 1)

        return bboxes

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'IoU_pred'))
    def get_final_det_bboxes(self,
                       bboxes,
                       cls_score,
                       IoU_pred,
                       img_shape,
                       scale_factor,
                       rescale=False,
                       cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        IoU_pred = IoU_pred.view(-1)
        bboxes = bboxes[:, 1:]
        if rescale:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                bboxes /= torch.from_numpy(scale_factor).to(bboxes.device)

        if cfg is None:
            return bboxes, scores
        else:

            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img, score_factors=IoU_pred)

            return det_bboxes, det_labels

@HEADS.register_module
class IoU_reg_head(nn.Module):
    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_IoU_convs=0,
                 num_IoU_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 roi_feat_size=7,
                 in_channels=256,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 num_classes=81,
                 target_means=[0., 0., 0., 0.],
                 target_stds=[0.1, 0.1, 0.2, 0.2],
                 class_agnostic=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_reg=dict(type='SmoothL1Loss', beta=1, loss_weight=1.0),
                 loss_IoU=dict(type='SmoothL1Loss', beta=0.5, loss_weight=1.0)):
        super(IoU_reg_head, self).__init__()
        assert (num_shared_convs+num_shared_fcs+num_IoU_convs + num_IoU_fcs+num_reg_convs+num_reg_fcs > 0)
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_IoU_convs = num_IoU_convs
        self.num_IoU_fcs = num_IoU_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.num_classes = num_classes
        self.target_means = target_means
        self.target_stds = target_stds
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.loss_IoU = build_loss(loss_IoU)
        self.loss_reg = build_loss(loss_reg)
        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim
        # add IoU specific branch
        self.IoU_convs, self.IoU_fcs, self.IoU_last_dim = \
            self._add_conv_fc_branch(
                self.num_IoU_convs, self.num_IoU_fcs, self.shared_out_channels)
        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        self.relu = nn.ReLU(inplace=True)

        out_dim_IoU = (1 if self.class_agnostic else self.num_classes)
        self.fc_IoU = nn.Linear(self.IoU_last_dim, out_dim_IoU)

        out_dim_reg = (4 if self.class_agnostic else self.num_classes*4)
        self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)


    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):

        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        for module_list in [self.shared_fcs, self.IoU_convs, self.IoU_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)
        if self.num_shared_fcs > 0:
            x = x.view(x.size(0), -1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_reg = x
        x_IoU = x

        # reg branch
        for conv in self.IoU_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            x_reg = x_reg.view(x_reg.size(0), -1)
        for fc in self.IoU_fcs:
            x_reg = self.relu(fc(x_reg))

        # IoU branch
        for conv in self.IoU_convs:
            x_IoU = conv(x_IoU)
        if x_IoU.dim() > 2:
            x_IoU = x_IoU.view(x_IoU.size(0), -1)
        for fc in self.IoU_fcs:
            x_IoU = self.relu(fc(x_IoU))

        IoU_pred  = self.fc_IoU(x_IoU)
        bbox_pred = self.fc_reg(x_reg)

        return IoU_pred, bbox_pred

    @force_fp32(apply_to=('IoU_pred'))
    def loss(self,
             bbox_pred,
             bbox_targets,
             IoU_pred,
             IoU_targets,
             gt_indexes,
             reduction_override=None):
        losses = dict()
        if IoU_pred is not None:
            IoU_weights = IoU_pred.new_full((IoU_pred.shape[0],), 1)

            if self.class_agnostic:
                IoU_pred = IoU_pred.view(-1)
            else:
                IoU_pred = IoU_pred.view(IoU_pred.size(0), -1)

            losses['loss_IoU'] = self.loss_IoU(
                IoU_pred,
                IoU_targets,
                reduction_override=reduction_override)

            losses['cnt_pos'] = torch.tensor(IoU_targets.shape[0]).to(IoU_targets.device).float()
            interval = ((IoU_targets - 0.00001) / 0.1).int()
            reg_weights = bbox_pred.new_full((bbox_pred.shape[0], 4), 1)
            reg_weights[interval == 6, :] = 1.5
            reg_weights[interval == 7, :] = 2.0
            reg_weights[interval == 8, :] = 3.0
            reg_weights[interval == 9, :] = 3.0

            losses['loss_reg'] = self.loss_reg(
                bbox_pred,
                bbox_targets,
                weight=reg_weights, reduction_override=reduction_override)

            reg_loss = self.loss_reg(
                bbox_pred,
                bbox_targets,
                weight=reg_weights,
                reduction_override='none')
            reg_loss = reg_loss.mean(dim=1)
            losses['cnt_5'] = (interval == 5).sum().float()
            losses['reg_5'] = reg_loss[interval == 5].sum()

            losses['cnt_6'] = (interval == 6).sum().float()
            losses['reg_6'] = reg_loss[interval == 6].sum()

            losses['cnt_7'] = (interval == 7).sum().float()
            losses['reg_7'] = reg_loss[interval == 7].sum()

            losses['cnt_8'] = ((interval == 8).sum() + (interval == 9).sum()).float()
            losses['reg_8'] = reg_loss[interval == 8].sum() + reg_loss[interval == 9].sum()

        return losses

    def get_target(self, rois, gt_bboxes):
        bbox_targets = bbox2delta(rois, gt_bboxes, self.target_means, self.target_stds)
        return bbox_targets

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'IoU_pred'))
    def get_det_bboxes(self,
                       rois,
                       cls_score,
                       bbox_pred,
                       IoU_pred,
                       img_shape,
                       scale_factor,
                       rescale=False,
                       cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        IoU_pred = IoU_pred.view(-1)
        if bbox_pred is not None:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1] - 1)
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0] - 1)

        if rescale:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
                rois[:, 1:] /= scale_factor
            else:
                bboxes /= torch.from_numpy(scale_factor).to(bboxes.device)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img, score_factors=IoU_pred)

            return det_bboxes, det_labels

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'IoU_pred'))
    def get_first_det_bboxes(self,
                       rois,
                       bbox_pred,
                       img_shape):

        if bbox_pred is not None:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1] - 1)
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0] - 1)

        return bboxes

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'IoU_pred'))
    def get_final_det_bboxes(self,
                       bboxes,
                       cls_score,
                       IoU_pred,
                       img_shape,
                       scale_factor,
                       rescale=False,
                       cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        IoU_pred = IoU_pred.view(-1)
        bboxes = bboxes[:, 1:]
        if rescale:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                bboxes /= torch.from_numpy(scale_factor).to(bboxes.device)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img, score_factors=IoU_pred)

            return det_bboxes, det_labels

@HEADS.register_module
class SharedFCBBoxHead(ConvFCBBoxHead):

    def __init__(self, num_fcs=2, fc_out_channels=1024, *args, **kwargs):
        assert num_fcs >= 1
        super(SharedFCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=num_fcs,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
