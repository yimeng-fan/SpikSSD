from typing import Dict, List

import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torchvision
import torchvision.models.detection._utils as det_utils
import torchvision.ops.boxes as box_ops
from torchvision.ops import complete_box_iou_loss
import pytorch_lightning as pl

from spikingjelly.activation_based import neuron
import spikingjelly
from spikingjelly.activation_based.monitor import OutputMonitor

from models.detection_backbone import DetectionBackbone
from models.detection_neck import DetectionNeck
from models.detection_head import SSDHead
from models.SSD_utils import GridSizeDefaultBoxGenerator, filter_boxes, bbox_iou, soft_nms

from prophesee_utils.metrics.coco_utils import coco_eval


def cal_firing_rate(s_seq: torch.Tensor):
    # s_seq.shape = [T, N, *]
    return s_seq.sum() / torch.numel(s_seq)


class DetectionLitModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters('args')
        self.args = args
        self.lr = args.lr
        self.batch_size = args.b
        self.fusion = args.fusion

        self.backbone = DetectionBackbone(args)  # backbone

        out_channels = self.backbone.out_channels

        if self.fusion:
            self.neck = DetectionNeck(self.backbone.out_channels, channels=args.channels,
                                      attention=args.attention, args=args)  # neck

            out_channels = self.neck.out_channels

        self.anchor_generator = GridSizeDefaultBoxGenerator(
            args.aspect_ratios, args.min_ratio, args.max_ratio)  # anchor_generator

        print(out_channels)
        assert len(out_channels) == len(self.anchor_generator.aspect_ratios)

        num_anchors = self.anchor_generator.num_anchors_per_location()
        self.head = SSDHead(out_channels, num_anchors, args.num_classes, bn=args.bn, args=args)  # SSD Head

        self.box_coder = det_utils.BoxCoder(weights=args.box_coder_weights)
        self.proposal_matcher = det_utils.SSDMatcher(args.iou_threshold)

        # Sparsity initialization
        self.backbone.add_hooks(neuron.BaseNode)
        if self.fusion:
            self.neck.add_hooks(neuron.BaseNode)
        self.head.add_hooks(neuron.BaseNode)
        self.all_nnz, self.all_nnumel = 0, 0

    def forward(self, events):
        events = events.permute(1, 0, 2, 3, 4)  # x.shape = [T, N, C, H, W]

        assert events.shape[0] == self.args.T

        features = self.backbone(events)
        if self.fusion:
            features = self.neck(features)
            assert len(features[-1].shape) == 4

        head_outputs = self.head(features)

        return features, head_outputs  # N HWA K

    def on_train_epoch_start(self):
        self.train_detections, self.train_targets = [], []

    def on_validation_epoch_start(self):
        self.test_detections, self.test_targets = [], []

    def on_test_epoch_start(self):
        self.test_detections, self.test_targets = [], []

    def step(self, batch, batch_idx, mode):
        events, targets = batch  # event: N T C W H  targets(list): N {'boxes': torch_boxes, 'labels': torch_labels}

        features, head_outputs = self(events)  # (list)N C H W; N HWA K
        anchors = self.anchor_generator(features, self.args.image_shape)  # N_batch(list) (N_anchors 4) (real size)

        # match targets with anchors
        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if targets_per_image["boxes"].numel() == 0:
                matched_idxs.append(
                    torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device)
                )
                continue

            match_quality_matrix = box_ops.distance_box_iou(targets_per_image["boxes"],
                                                            anchors_per_image)  # N_target M_anchors
            matched_idxs.append(self.proposal_matcher(match_quality_matrix))  # N_anchors

        # Loss computation
        loss = None
        if mode != "test":
            losses = self.compute_loss(targets, head_outputs, anchors, matched_idxs)

            bbox_loss = losses['bbox_regression']
            cls_loss = losses['classification']

            self.log(f'{mode}_loss_bbox', bbox_loss.detach(), on_step=True, on_epoch=True, prog_bar=True,
                     sync_dist=True, batch_size=self.batch_size)
            self.log(f'{mode}_loss_classif', cls_loss.detach(), on_step=True, on_epoch=True, prog_bar=True,
                     sync_dist=True, batch_size=self.batch_size)

            loss = bbox_loss + cls_loss # + iou_loss
            self.log(f'{mode}_loss', loss.detach(), on_step=True, on_epoch=True,
                     prog_bar=True, sync_dist=True, batch_size=self.batch_size)

        # Postprocessing for mAP computation
        if mode == "test":
            detections = self.postprocess_detections(head_outputs,
                                                     anchors)  # (N, A, K) list((Anchors 4(xyxy)))
            detections = list(map(filter_boxes, detections))
            targets = list(
                map(filter_boxes, targets))  # targets(list): N {'boxes': torch_boxes, 'labels': torch_labels}

            getattr(self, f"{mode}_detections").extend(
                [{k: v.cpu().detach() for k, v in d.items()} for d in detections])
            getattr(self, f"{mode}_targets").extend([{k: v.cpu().detach() for k, v in t.items()} for t in targets])

        if mode != "train":
            # Sparsity measurement
            self.process_nz(self.backbone.get_nz_numel())
            self.backbone.reset_nz_numel()
            if self.fusion:
                self.process_nz(self.neck.get_nz_numel())
                self.neck.reset_nz_numel()

        spikingjelly.activation_based.functional.reset_net(self.backbone)
        if self.fusion:
            spikingjelly.activation_based.functional.reset_net(self.neck)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode="val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode="test")

    def on_mode_epoch_end(self, mode):
        print()

        if mode == "test":
            print(f"[{self.current_epoch}] {mode} results:")

            targets = getattr(self, f"{mode}_targets")  # N {'boxes': torch_boxes, 'labels': torch_labels}
            detections = getattr(self, f"{mode}_detections")  # N {boxes scores labels}

            if detections == []:
                print("No detections")
                return

            h, w = self.args.image_shape
            stats = coco_eval(
                targets,
                detections,
                height=h, width=w,
                labelmap=("car", "pedestrian"))

            keys = [
                'val_AP_IoU=.5:.05:.95', 'val_AP_IoU=.5', 'val_AP_IoU=.75',
                'val_AP_small', 'val_AP_medium', 'val_AP_large',
                'val_AR_det=1', 'val_AR_det=10', 'val_AR_det=100',
                'val_AR_small', 'val_AR_medium', 'val_AR_large',
            ]
            stats_dict = {k: v for k, v in zip(keys, stats)}
            self.log_dict(stats_dict, sync_dist=True)

        if mode != "train":
            # Output sparsity
            print(
                f"{mode}: Total sparsity: {self.all_nnz} / {self.all_nnumel} ({100 * self.all_nnz / (self.all_nnumel + 1e-3):.2f}%)")
            self.all_nnz, self.all_nnumel = 0, 0

    def on_train_epoch_end(self):
        self.on_mode_epoch_end(mode="train")

    def on_validation_epoch_end(self):
        self.on_mode_epoch_end(mode="val")

    def on_test_epoch_end(self):
        self.on_mode_epoch_end(mode="test")

    def compute_loss(self, targets: List[Dict[str, Tensor]],
                     head_outputs: Dict[str, Tensor], anchors: List[Tensor],
                     matched_idxs: List[Tensor]) -> Dict[str, Tensor]:
        bbox_regression = head_outputs["bbox_regression"]
        cls_logits = head_outputs["cls_logits"]  # Size=(N, A, K)

        num_foreground_reg = 0
        num_foreground_cls = 0
        bbox_loss, cls_loss = [], []

        # Match original targets with default boxes
        for (targets_per_image,  # {'boxes': torch_boxes, 'labels': torch_labels}
             bbox_regression_per_image,  # A 4
             cls_logits_per_image,  # A Classes
             # iou_logits_per_image,  # A 1
             anchors_per_image,  # A, 4(xyxy)(real size)
             matched_idxs_per_image  # N_anchors
             ) in zip(targets, bbox_regression, cls_logits, anchors, matched_idxs):
            # produce the matching between boxes and targets
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[
                0]  # N_matched(idx_anchor)
            foreground_matched_idxs_per_image = matched_idxs_per_image[
                foreground_idxs_per_image]  # N_matched idx
            num_foreground_reg += foreground_idxs_per_image.numel()

            # Compute regression loss
            matched_gt_boxes_per_image = targets_per_image["boxes"][
                foreground_matched_idxs_per_image]  # N_matched (target)
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]  # N_matched (model)
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]  # N_matched
            decoded_bbox_per_image = self.box_coder.decode_single(bbox_regression_per_image,
                                                                  anchors_per_image)  # N_matched 4(xyxy) (anchor)
            decoded_bbox_per_image = box_ops.clip_boxes_to_image(decoded_bbox_per_image, self.args.image_shape)
            iou = bbox_iou(decoded_bbox_per_image, matched_gt_boxes_per_image, x1y1x2y2=True,
                           CIoU=True)  # iou(prediction and target) N_matched

            bbox_loss.append(
                (1.0 - iou).sum()  # IOU loss
            )

            # Compute classification loss (focal loss)
            foreground_idxs_per_image = matched_idxs_per_image >= 0  # N_matched
            num_foreground_cls += foreground_idxs_per_image.sum()
            gt_classes_target = torch.zeros_like(cls_logits_per_image)  # A, K

            gt_classes_target[
                foreground_idxs_per_image,
                targets_per_image["labels"][foreground_matched_idxs_per_image],
            ] = 1.0

            cls_loss.append(
                0.05 * torchvision.ops.focal_loss.sigmoid_focal_loss(
                    cls_logits_per_image,
                    gt_classes_target,
                    reduction="sum",
                )
            )

        bbox_loss = torch.stack(bbox_loss)
        cls_loss = torch.stack(cls_loss)

        return {
            "bbox_regression": bbox_loss.sum() / max(1, num_foreground_reg),
            "classification": cls_loss.sum() / max(1, num_foreground_cls),
        }

    def postprocess_detections(
            self, head_outputs: Dict[str, Tensor], image_anchors: List[Tensor]
    ) -> List[Dict[str, Tensor]]:
        bbox_regression = head_outputs["bbox_regression"]
        pred_logits = head_outputs["cls_logits"] #  head_outputs["iou_logits"]

        detections = []

        for boxes, logits, anchors in zip(bbox_regression, pred_logits, image_anchors):
            boxes = self.box_coder.decode_single(boxes, anchors)  # N_anchors 4(xyxy)
            boxes = box_ops.clip_boxes_to_image(boxes, self.args.image_shape)

            image_boxes, image_scores, image_labels = [], [], []
            for label in range(self.args.num_classes):
                logits_per_class = logits[:, label]
                score = torch.sigmoid(logits_per_class).flatten()

                # remove low scoring boxes
                keep_idxs = score > self.args.score_thresh
                score = score[keep_idxs]
                box = boxes[keep_idxs]

                # keep only topk scoring predictions
                num_topk = min(self.args.topk_candidates, score.size(0))
                score, idxs = score.topk(num_topk)
                box = box[idxs]

                image_boxes.append(box)
                image_scores.append(score)
                image_labels.append(torch.full_like(score, fill_value=label, dtype=torch.int64))

            image_boxes = torch.cat(image_boxes, dim=0)  # N_anchors(match) 4
            image_scores = torch.cat(image_scores, dim=0)  # N_anchors(match),
            image_labels = torch.cat(image_labels, dim=0)  # # N_anchors(match),

            # non-maximum suppression
            # keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.args.nms_thresh)
            keep = soft_nms(image_boxes, image_scores, iou_thresh=self.args.nms_thresh,
                            score_threshold=self.args.score_thresh)
            keep = keep[:self.args.detections_per_img]

            detections.append(
                {
                    "boxes": image_boxes[keep],  # N_keep 4
                    "scores": image_scores[keep],  # N_keep
                    "labels": image_labels[keep],  # N_keep
                }
            )
        return detections

    def process_nz(self, nz_numel):
        nz, numel = nz_numel
        total_nnz, total_nnumel = 0, 0

        for module, nnz in nz.items():
            nnumel = numel[module]
            if nnumel != 0:
                total_nnz += nnz
                total_nnumel += nnumel
        if total_nnumel != 0:
            self.all_nnz += total_nnz
            self.all_nnumel += total_nnumel

    def configure_optimizers(self):
        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('Number of parameters:', n_parameters)

        g0, g1, g2 = [], [], []  # optimizer parameter groups
        for v in self.backbone.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
                g2.append(v.bias)
            if isinstance(v, nn.BatchNorm3d):  # weight (no decay)
                g0.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                g1.append(v.weight)
        for v in self.head.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
                g2.append(v.bias)
            if isinstance(v, nn.BatchNorm3d):  # weight (no decay)
                g0.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                g1.append(v.weight)
        if self.fusion:
            for v in self.neck.modules():
                if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
                    g2.append(v.bias)
                if isinstance(v, nn.BatchNorm3d):  # weight (no decay)
                    g0.append(v.weight)
                elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                    g1.append(v.weight)

        optimizer = torch.optim.AdamW(
            g0,
            lr=self.lr,
            betas=(0.937, 0.999),
        )

        optimizer.add_param_group({'params': g1, 'weight_decay': 0.0005})  # add g1 with weight_decay
        optimizer.add_param_group({'params': g2})  # add g2 (biases)

        del g0, g1, g2

        scheduler = combined_lr_scheduler(optimizer, self.args.epochs, self.args.warmup_epochs, self.args.linear_lr,
                                          self.args.warmup_lrf, self.args.lrf)
        return [optimizer], [scheduler]


def combined_lr_scheduler(optimizer, epochs, warmup_epochs, linear_lr, warmup_lrf, lrf):
    def lr_lambda(current_epoch):
        if current_epoch <= warmup_epochs:
            return np.interp(current_epoch, [0, warmup_epochs], [warmup_lrf, 1])
        else:
            if linear_lr:
                return (1 - (current_epoch - warmup_epochs) / (epochs - 1 - warmup_epochs)) * (1.0 - lrf) + lrf
            else:
                return ((1 - math.cos((current_epoch - warmup_epochs) * math.pi / (epochs - warmup_epochs))) / 2) * (
                        lrf - 1) + 1

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler
