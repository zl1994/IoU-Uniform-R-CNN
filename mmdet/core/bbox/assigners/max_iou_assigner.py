import torch
import numpy as np
from ..geometry import bbox_overlaps
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


class MaxIoUAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, `0`, or a positive integer
    indicating the ground truth index.

    - -1: don't care
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
        gpu_assign_thr (int): The upper bound of the number of GT for GPU
            assign. When the number of gt is above this threshold, will assign
            on CPU device. Negative values mean not assign on CPU.
    """

    def __init__(self,
                 pos_iou_thr,
                 neg_iou_thr,
                 min_pos_iou=.0,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 gpu_assign_thr=-1):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, 0, or a positive number. -1 means don't care,
        0 means negative sample, positive number is the index (1-based) of
        assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every bbox to -1
        2. assign proposals whose iou with all gts < neg_iou_thr to 0
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr,
           assign it to that bbox
        4. for each gt bbox, assign its nearest proposals (may be more than
           one) to itself

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        if bboxes.shape[0] == 0 or gt_bboxes.shape[0] == 0:
            raise ValueError('No gt or bboxes')
        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
            gt_bboxes.shape[0] > self.gpu_assign_thr) else False
        # compute overlap and assign gt on CPU when number of GT is large
        if assign_on_cpu:
            device = bboxes.device
            bboxes = bboxes.cpu()
            gt_bboxes = gt_bboxes.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()
            if gt_labels is not None:
                gt_labels = gt_labels.cpu()
        bboxes = bboxes[:, :4]
        overlaps = bbox_overlaps(gt_bboxes, bboxes)

        if (self.ignore_iof_thr > 0) and (gt_bboxes_ignore is not None) and (
                gt_bboxes_ignore.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = bbox_overlaps(
                    bboxes, gt_bboxes_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = bbox_overlaps(
                    gt_bboxes_ignore, bboxes, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)
        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        return assign_result

    def assign_IoU(self, gt_bboxes, img_meta, reg_sample=False, num_sample=64, fraction=0.25):
        new_bboxes_list = []
        iou_targets_list = []
        bbox_targets_list = []
        gt_indexes_list = []
        num_random = 500
        count = int(num_sample * fraction)
        for i in range(gt_bboxes.shape[0]):
            cnt = 0
            gt_bbox = gt_bboxes[i].unsqueeze(0)
            max_shape = img_meta['img_shape']
            # before jittering
            cxcy = (gt_bbox[:, 2:4] + gt_bbox[:, :2]) / 2
            wh = (gt_bbox[:, 2:4] - gt_bbox[:, :2]).abs()

            # generate samples for 0.5-0.6
            offset_scope = 0.75
            wh_scope_bottom = 0.5
            wh_scope_top = 2
            random_offsets = gt_bbox.new_empty(num_random, 2).uniform_(-offset_scope, offset_scope)
            random_wh = gt_bbox.new_empty(num_random, 2).uniform_(wh_scope_bottom, wh_scope_top)

            # after jittering
            new_cxcy = cxcy + wh * random_offsets
            new_wh = wh * random_wh
            # xywh to xyxy
            new_x1y1 = (new_cxcy - new_wh / 2)
            new_x2y2 = (new_cxcy + new_wh / 2)
            new_bbox_all = torch.cat([new_x1y1, new_x2y2], dim=1)
            # clip bboxes
            if max_shape is not None:
                new_bbox_all[:, 0::2].clamp_(min=0, max=max_shape[1] - 1)
                new_bbox_all[:, 1::2].clamp_(min=0, max=max_shape[0] - 1)
            iou = bbox_overlaps(new_bbox_all, gt_bbox)
            inds_5 = (iou[:, 0] > 0.5) & (iou[:, 0] <= 0.6)

            iou_all = bbox_overlaps(new_bbox_all, gt_bboxes)
            _, index = iou_all.max(dim=1)
            inds_5 = (index == i) & inds_5

            new_bbox = new_bbox_all[inds_5, :]
            iou_targets = iou[inds_5, 0]
            if new_bbox.shape[0] > count:
                cands = np.arange(new_bbox.shape[0])
                np.random.shuffle(cands)
                rand_inds = cands[:count]
                rand_inds = torch.from_numpy(rand_inds).long().to(new_bbox.device)
                new_bbox = new_bbox[rand_inds, :]
                iou_targets = iou_targets[rand_inds]
            new_bboxes_list.append(new_bbox)
            iou_targets_list.append(iou_targets)
            cnt += new_bbox.shape[0]

            # generate samples for 0.6-0.7
            offset_scope = 0.5
            wh_scope_bottom = 0.6
            wh_scope_top = 1.6
            random_offsets = gt_bbox.new_empty(num_random, 2).uniform_(-offset_scope, offset_scope)
            random_wh = gt_bbox.new_empty(num_random, 2).uniform_(wh_scope_bottom, wh_scope_top)

            # after jittering
            new_cxcy = cxcy + wh * random_offsets
            new_wh = wh * random_wh
            # xywh to xyxy
            new_x1y1 = (new_cxcy - new_wh / 2)
            new_x2y2 = (new_cxcy + new_wh / 2)
            new_bbox_all = torch.cat([new_x1y1, new_x2y2], dim=1)
            # clip bboxes
            if max_shape is not None:
                new_bbox_all[:, 0::2].clamp_(min=0, max=max_shape[1] - 1)
                new_bbox_all[:, 1::2].clamp_(min=0, max=max_shape[0] - 1)
            iou = bbox_overlaps(new_bbox_all, gt_bbox)
            inds_6 = (iou[:, 0] > 0.6) & (iou[:, 0] <= 0.7)

            iou_all = bbox_overlaps(new_bbox_all, gt_bboxes)
            _, index = iou_all.max(dim=1)
            inds_6 = (index == i) & inds_6

            new_bbox = new_bbox_all[inds_6, :]
            iou_targets = iou[inds_6, 0]
            if new_bbox.shape[0] > count:
                cands = np.arange(new_bbox.shape[0])
                np.random.shuffle(cands)
                rand_inds = cands[:count]
                rand_inds = torch.from_numpy(rand_inds).long().to(new_bbox.device)
                new_bbox = new_bbox[rand_inds, :]
                iou_targets = iou_targets[rand_inds]
            new_bboxes_list.append(new_bbox)
            iou_targets_list.append(iou_targets)
            cnt += new_bbox.shape[0]

            # generate samples for 0.7-0.8
            offset_scope = 0.3
            wh_scope_bottom = 0.7
            wh_scope_top = 1.4
            random_offsets = gt_bbox.new_empty(num_random, 2).uniform_(-offset_scope, offset_scope)
            random_wh = gt_bbox.new_empty(num_random, 2).uniform_(wh_scope_bottom, wh_scope_top)

            # after jittering
            new_cxcy = cxcy + wh * random_offsets
            new_wh = wh * random_wh
            # xywh to xyxy
            new_x1y1 = (new_cxcy - new_wh / 2)
            new_x2y2 = (new_cxcy + new_wh / 2)
            new_bbox_all = torch.cat([new_x1y1, new_x2y2], dim=1)
            # clip bboxes
            if max_shape is not None:
                new_bbox_all[:, 0::2].clamp_(min=0, max=max_shape[1] - 1)
                new_bbox_all[:, 1::2].clamp_(min=0, max=max_shape[0] - 1)
            iou = bbox_overlaps(new_bbox_all, gt_bbox)
            inds_7 = (iou[:, 0] > 0.7) & (iou[:, 0] <= 0.8)

            iou_all = bbox_overlaps(new_bbox_all, gt_bboxes)
            _, index = iou_all.max(dim=1)
            inds_7 = (index == i) & inds_7

            new_bbox = new_bbox_all[inds_7, :]
            iou_targets = iou[inds_7, 0]
            if new_bbox.shape[0] > count:
                cands = np.arange(new_bbox.shape[0])
                np.random.shuffle(cands)

                rand_inds = cands[:count]
                rand_inds = torch.from_numpy(rand_inds).long().to(new_bbox.device)
                new_bbox = new_bbox[rand_inds, :]
                iou_targets = iou_targets[rand_inds]
            new_bboxes_list.append(new_bbox)
            iou_targets_list.append(iou_targets)
            cnt += new_bbox.shape[0]

            # generate samples for 0.8-1.0
            offset_scope = 0.125
            wh_scope_bottom = 0.8
            wh_scope_top = 1.25
            random_offsets = gt_bbox.new_empty(num_random, 2).uniform_(-offset_scope, offset_scope)
            random_wh = gt_bbox.new_empty(num_random, 2).uniform_(wh_scope_bottom, wh_scope_top)

            # after jittering
            new_cxcy = cxcy + wh * random_offsets
            new_wh = wh * random_wh
            # xywh to xyxy
            new_x1y1 = (new_cxcy - new_wh / 2)
            new_x2y2 = (new_cxcy + new_wh / 2)
            new_bbox_all = torch.cat([new_x1y1, new_x2y2], dim=1)
            # clip bboxes
            if max_shape is not None:
                new_bbox_all[:, 0::2].clamp_(min=0, max=max_shape[1] - 1)
                new_bbox_all[:, 1::2].clamp_(min=0, max=max_shape[0] - 1)
            iou = bbox_overlaps(new_bbox_all, gt_bbox)
            inds_8 = iou[:, 0] > 0.8

            iou_all = bbox_overlaps(new_bbox_all, gt_bboxes)
            _, index = iou_all.max(dim=1)
            inds_8 = (index == i) & inds_8

            new_bbox = new_bbox_all[inds_8, :]
            iou_targets = iou[inds_8, 0]
            if new_bbox.shape[0] > count:
                cands = np.arange(new_bbox.shape[0])
                np.random.shuffle(cands)

                rand_inds = cands[:count]
                rand_inds = torch.from_numpy(rand_inds).long().to(new_bbox.device)
                new_bbox = new_bbox[rand_inds, :]
                iou_targets = iou_targets[rand_inds]
            new_bboxes_list.append(new_bbox)
            iou_targets_list.append(iou_targets)
            cnt += new_bbox.shape[0]
            if reg_sample:
                bbox_targets = gt_bbox.expand(cnt, 4)
                bbox_targets_list.append(bbox_targets)
            gt_indexes = gt_bbox.new_full((cnt,), i)
            gt_indexes_list.append(gt_indexes)

        new_bboxes = torch.cat(new_bboxes_list, 0)
        iou_targets = torch.cat(iou_targets_list, 0)
        gt_indexes = torch.cat(gt_indexes_list, 0)
        if reg_sample:
            bbox_targets = torch.cat(bbox_targets_list, 0)
            return new_bboxes, iou_targets, gt_indexes, bbox_targets
        else:
            return new_bboxes, iou_targets, gt_indexes

    def assign_IoU_2(self, gt_bboxes, img_meta, reg_sample=False, num_sample=64, fraction=0.25):
        new_bboxes_list = []
        iou_targets_list = []
        bbox_targets_list = []
        gt_indexes_list = []
        num_random = 500
        count = int(num_sample * fraction)
        for i in range(gt_bboxes.shape[0]):
            cnt = 0
            gt_bbox = gt_bboxes[i].unsqueeze(0)
            max_shape = img_meta['img_shape']
            # before jittering
            cxcy = (gt_bbox[:, 2:4] + gt_bbox[:, :2]) / 2
            wh = (gt_bbox[:, 2:4] - gt_bbox[:, :2]).abs()

            # generate samples for 0.6-0.7
            offset_scope = 0.5
            wh_scope_bottom = 0.6
            wh_scope_top = 1.6
            random_offsets = gt_bbox.new_empty(num_random, 2).uniform_(-offset_scope, offset_scope)
            random_wh = gt_bbox.new_empty(num_random, 2).uniform_(wh_scope_bottom, wh_scope_top)

            # after jittering
            new_cxcy = cxcy + wh * random_offsets
            new_wh = wh * random_wh
            # xywh to xyxy
            new_x1y1 = (new_cxcy - new_wh / 2)
            new_x2y2 = (new_cxcy + new_wh / 2)
            new_bbox_all = torch.cat([new_x1y1, new_x2y2], dim=1)
            # clip bboxes
            if max_shape is not None:
                new_bbox_all[:, 0::2].clamp_(min=0, max=max_shape[1] - 1)
                new_bbox_all[:, 1::2].clamp_(min=0, max=max_shape[0] - 1)
            iou = bbox_overlaps(new_bbox_all, gt_bbox)
            inds_6 = (iou[:, 0] > 0.6) & (iou[:, 0] <= 0.7)
            new_bbox = new_bbox_all[inds_6, :]
            iou_targets = iou[inds_6, 0]
            if new_bbox.shape[0] > count:
                cands = np.arange(new_bbox.shape[0])
                np.random.shuffle(cands)
                rand_inds = cands[:count]
                rand_inds = torch.from_numpy(rand_inds).long().to(new_bbox.device)
                new_bbox = new_bbox[rand_inds, :]
                iou_targets = iou_targets[rand_inds]
            new_bboxes_list.append(new_bbox)
            iou_targets_list.append(iou_targets)
            cnt += new_bbox.shape[0]

            # generate samples for 0.7-0.8
            offset_scope = 0.3
            wh_scope_bottom = 0.7
            wh_scope_top = 1.4
            random_offsets = gt_bbox.new_empty(num_random, 2).uniform_(-offset_scope, offset_scope)
            random_wh = gt_bbox.new_empty(num_random, 2).uniform_(wh_scope_bottom, wh_scope_top)

            # after jittering
            new_cxcy = cxcy + wh * random_offsets
            new_wh = wh * random_wh
            # xywh to xyxy
            new_x1y1 = (new_cxcy - new_wh / 2)
            new_x2y2 = (new_cxcy + new_wh / 2)
            new_bbox_all = torch.cat([new_x1y1, new_x2y2], dim=1)
            # clip bboxes
            if max_shape is not None:
                new_bbox_all[:, 0::2].clamp_(min=0, max=max_shape[1] - 1)
                new_bbox_all[:, 1::2].clamp_(min=0, max=max_shape[0] - 1)
            iou = bbox_overlaps(new_bbox_all, gt_bbox)
            inds_7 = (iou[:, 0] > 0.7) & (iou[:, 0] <= 0.8)
            new_bbox = new_bbox_all[inds_7, :]
            iou_targets = iou[inds_7, 0]
            if new_bbox.shape[0] > count:
                cands = np.arange(new_bbox.shape[0])
                np.random.shuffle(cands)

                rand_inds = cands[:count]
                rand_inds = torch.from_numpy(rand_inds).long().to(new_bbox.device)
                new_bbox = new_bbox[rand_inds, :]
                iou_targets = iou_targets[rand_inds]
            new_bboxes_list.append(new_bbox)
            iou_targets_list.append(iou_targets)
            cnt += new_bbox.shape[0]

            # generate samples for 0.8-0.9
            offset_scope = 0.125
            wh_scope_bottom = 0.8
            wh_scope_top = 1.25
            random_offsets = gt_bbox.new_empty(num_random, 2).uniform_(-offset_scope, offset_scope)
            random_wh = gt_bbox.new_empty(num_random, 2).uniform_(wh_scope_bottom, wh_scope_top)

            # after jittering
            new_cxcy = cxcy + wh * random_offsets
            new_wh = wh * random_wh
            # xywh to xyxy
            new_x1y1 = (new_cxcy - new_wh / 2)
            new_x2y2 = (new_cxcy + new_wh / 2)
            new_bbox_all = torch.cat([new_x1y1, new_x2y2], dim=1)
            # clip bboxes
            if max_shape is not None:
                new_bbox_all[:, 0::2].clamp_(min=0, max=max_shape[1] - 1)
                new_bbox_all[:, 1::2].clamp_(min=0, max=max_shape[0] - 1)
            iou = bbox_overlaps(new_bbox_all, gt_bbox)
            inds_8 = iou[:, 0] > 0.8
            new_bbox = new_bbox_all[inds_8, :]
            iou_targets = iou[inds_8, 0]
            if new_bbox.shape[0] > count:
                cands = np.arange(new_bbox.shape[0])
                np.random.shuffle(cands)

                rand_inds = cands[:count]
                rand_inds = torch.from_numpy(rand_inds).long().to(new_bbox.device)
                new_bbox = new_bbox[rand_inds, :]
                iou_targets = iou_targets[rand_inds]
            new_bboxes_list.append(new_bbox)
            iou_targets_list.append(iou_targets)
            cnt += new_bbox.shape[0]

            # generate samples for 0.9-1.0
            offset_scope = 0.125
            wh_scope_bottom = 0.9
            wh_scope_top = 1.25
            random_offsets = gt_bbox.new_empty(num_random, 2).uniform_(-offset_scope, offset_scope)
            random_wh = gt_bbox.new_empty(num_random, 2).uniform_(wh_scope_bottom, wh_scope_top)

            # after jittering
            new_cxcy = cxcy + wh * random_offsets
            new_wh = wh * random_wh
            # xywh to xyxy
            new_x1y1 = (new_cxcy - new_wh / 2)
            new_x2y2 = (new_cxcy + new_wh / 2)
            new_bbox_all = torch.cat([new_x1y1, new_x2y2], dim=1)
            # clip bboxes
            if max_shape is not None:
                new_bbox_all[:, 0::2].clamp_(min=0, max=max_shape[1] - 1)
                new_bbox_all[:, 1::2].clamp_(min=0, max=max_shape[0] - 1)
            iou = bbox_overlaps(new_bbox_all, gt_bbox)
            inds_9 = iou[:, 0] > 0.9
            new_bbox = new_bbox_all[inds_9, :]
            iou_targets = iou[inds_9, 0]
            if new_bbox.shape[0] > count:
                cands = np.arange(new_bbox.shape[0])
                np.random.shuffle(cands)

                rand_inds = cands[:count]
                rand_inds = torch.from_numpy(rand_inds).long().to(new_bbox.device)
                new_bbox = new_bbox[rand_inds, :]
                iou_targets = iou_targets[rand_inds]
            new_bboxes_list.append(new_bbox)
            iou_targets_list.append(iou_targets)
            cnt += new_bbox.shape[0]

            if reg_sample:
                bbox_targets = gt_bbox.expand(cnt, 4)
                bbox_targets_list.append(bbox_targets)
            gt_indexes = gt_bbox.new_full((cnt,), i)
            gt_indexes_list.append(gt_indexes)

        new_bboxes = torch.cat(new_bboxes_list, 0)
        iou_targets = torch.cat(iou_targets_list, 0)
        gt_indexes = torch.cat(gt_indexes_list, 0)
        if reg_sample:
            bbox_targets = torch.cat(bbox_targets_list, 0)
            return new_bboxes, iou_targets, gt_indexes, bbox_targets
        else:
            return new_bboxes, iou_targets, gt_indexes

    def assign_IoU_3(self, gt_bboxes, img_meta, reg_sample=False, num_sample=60, fraction=1 / 3):
        new_bboxes_list = []
        iou_targets_list = []
        bbox_targets_list = []
        gt_indexes_list = []
        num_random = 500
        count = int(num_sample * fraction)
        for i in range(gt_bboxes.shape[0]):
            cnt = 0
            gt_bbox = gt_bboxes[i].unsqueeze(0)
            max_shape = img_meta['img_shape']
            # before jittering
            cxcy = (gt_bbox[:, 2:4] + gt_bbox[:, :2]) / 2
            wh = (gt_bbox[:, 2:4] - gt_bbox[:, :2]).abs()

            # generate samples for 0.7-0.8
            offset_scope = 0.3
            wh_scope_bottom = 0.7
            wh_scope_top = 1.4
            random_offsets = gt_bbox.new_empty(num_random, 2).uniform_(-offset_scope, offset_scope)
            random_wh = gt_bbox.new_empty(num_random, 2).uniform_(wh_scope_bottom, wh_scope_top)

            # after jittering
            new_cxcy = cxcy + wh * random_offsets
            new_wh = wh * random_wh
            # xywh to xyxy
            new_x1y1 = (new_cxcy - new_wh / 2)
            new_x2y2 = (new_cxcy + new_wh / 2)
            new_bbox_all = torch.cat([new_x1y1, new_x2y2], dim=1)
            # clip bboxes
            if max_shape is not None:
                new_bbox_all[:, 0::2].clamp_(min=0, max=max_shape[1] - 1)
                new_bbox_all[:, 1::2].clamp_(min=0, max=max_shape[0] - 1)
            iou = bbox_overlaps(new_bbox_all, gt_bbox)
            inds_7 = (iou[:, 0] > 0.7) & (iou[:, 0] <= 0.8)
            new_bbox = new_bbox_all[inds_7, :]
            iou_targets = iou[inds_7, 0]
            if new_bbox.shape[0] > count:
                cands = np.arange(new_bbox.shape[0])
                np.random.shuffle(cands)

                rand_inds = cands[:count]
                rand_inds = torch.from_numpy(rand_inds).long().to(new_bbox.device)
                new_bbox = new_bbox[rand_inds, :]
                iou_targets = iou_targets[rand_inds]
            new_bboxes_list.append(new_bbox)
            iou_targets_list.append(iou_targets)
            cnt += new_bbox.shape[0]

            # generate samples for 0.8-0.9
            offset_scope = 0.125
            wh_scope_bottom = 0.8
            wh_scope_top = 1.25
            random_offsets = gt_bbox.new_empty(num_random, 2).uniform_(-offset_scope, offset_scope)
            random_wh = gt_bbox.new_empty(num_random, 2).uniform_(wh_scope_bottom, wh_scope_top)

            # after jittering
            new_cxcy = cxcy + wh * random_offsets
            new_wh = wh * random_wh
            # xywh to xyxy
            new_x1y1 = (new_cxcy - new_wh / 2)
            new_x2y2 = (new_cxcy + new_wh / 2)
            new_bbox_all = torch.cat([new_x1y1, new_x2y2], dim=1)
            # clip bboxes
            if max_shape is not None:
                new_bbox_all[:, 0::2].clamp_(min=0, max=max_shape[1] - 1)
                new_bbox_all[:, 1::2].clamp_(min=0, max=max_shape[0] - 1)
            iou = bbox_overlaps(new_bbox_all, gt_bbox)
            inds_8 = iou[:, 0] > 0.8
            new_bbox = new_bbox_all[inds_8, :]
            iou_targets = iou[inds_8, 0]
            if new_bbox.shape[0] > count:
                cands = np.arange(new_bbox.shape[0])
                np.random.shuffle(cands)

                rand_inds = cands[:count]
                rand_inds = torch.from_numpy(rand_inds).long().to(new_bbox.device)
                new_bbox = new_bbox[rand_inds, :]
                iou_targets = iou_targets[rand_inds]
            new_bboxes_list.append(new_bbox)
            iou_targets_list.append(iou_targets)
            cnt += new_bbox.shape[0]

            # generate samples for 0.9-1.0
            offset_scope = 0.125
            wh_scope_bottom = 0.9
            wh_scope_top = 1.25
            random_offsets = gt_bbox.new_empty(num_random, 2).uniform_(-offset_scope, offset_scope)
            random_wh = gt_bbox.new_empty(num_random, 2).uniform_(wh_scope_bottom, wh_scope_top)

            # after jittering
            new_cxcy = cxcy + wh * random_offsets
            new_wh = wh * random_wh
            # xywh to xyxy
            new_x1y1 = (new_cxcy - new_wh / 2)
            new_x2y2 = (new_cxcy + new_wh / 2)
            new_bbox_all = torch.cat([new_x1y1, new_x2y2], dim=1)
            # clip bboxes
            if max_shape is not None:
                new_bbox_all[:, 0::2].clamp_(min=0, max=max_shape[1] - 1)
                new_bbox_all[:, 1::2].clamp_(min=0, max=max_shape[0] - 1)
            iou = bbox_overlaps(new_bbox_all, gt_bbox)
            inds_9 = iou[:, 0] > 0.9
            new_bbox = new_bbox_all[inds_9, :]
            iou_targets = iou[inds_9, 0]
            if new_bbox.shape[0] > count:
                cands = np.arange(new_bbox.shape[0])
                np.random.shuffle(cands)

                rand_inds = cands[:count]
                rand_inds = torch.from_numpy(rand_inds).long().to(new_bbox.device)
                new_bbox = new_bbox[rand_inds, :]
                iou_targets = iou_targets[rand_inds]
            new_bboxes_list.append(new_bbox)
            iou_targets_list.append(iou_targets)
            cnt += new_bbox.shape[0]

            if reg_sample:
                bbox_targets = gt_bbox.expand(cnt, 4)
                bbox_targets_list.append(bbox_targets)
            gt_indexes = gt_bbox.new_full((cnt,), i)
            gt_indexes_list.append(gt_indexes)

        new_bboxes = torch.cat(new_bboxes_list, 0)
        iou_targets = torch.cat(iou_targets_list, 0)
        gt_indexes = torch.cat(gt_indexes_list, 0)
        if reg_sample:
            bbox_targets = torch.cat(bbox_targets_list, 0)
            return new_bboxes, iou_targets, gt_indexes, bbox_targets
        else:
            return new_bboxes, iou_targets, gt_indexes

    def assign_wrt_overlaps(self, overlaps, gt_labels=None):
        """Assign w.r.t. the overlaps of bboxes with gts.

        Args:
            overlaps (Tensor): Overlaps between k gt_bboxes and n bboxes,
                shape(k, n).
            gt_labels (Tensor, optional): Labels of k gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        if overlaps.numel() == 0:
            raise ValueError('No gt or proposals')

        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

        # 2. assign negative: below
        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, tuple):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0])
                             & (max_overlaps < self.neg_iou_thr[1])] = 0

        # 3. assign positive: above positive IoU threshold
        pos_inds = max_overlaps >= self.pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        # 4. assign fg: for each gt, proposals with highest IoU
        for i in range(num_gts):
            if gt_max_overlaps[i] >= self.min_pos_iou:
                if self.gt_max_assign_all:
                    max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                    assigned_gt_inds[max_iou_inds] = i + 1
                else:
                    assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_zeros((num_bboxes, ))
            pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)
