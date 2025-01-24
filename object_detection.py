from os.path import join
import sys
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from data import GEN1DetectionDataset
from object_detection_module import DetectionLitModule


def augmentation_collate_fn(batch):
    samples = torch.stack([item[i] for item in batch for i in (0, 2) if len(item) > i and item[i] is not None], 0)

    targets = [item[i] for item in batch for i in (1, 3) if len(item) > i and item[i] is not None]

    return [samples, targets]


def augmentation_collate_fn2(batch):
    samples = torch.stack([item[i] for item in batch for i in (0, 2, 4) if len(item) > i and item[i] is not None], 0)

    targets = [item[i] for item in batch for i in (1, 3, 5) if len(item) > i and item[i] is not None]

    return [samples, targets]


def collate_fn(batch):
    samples = [item[0] for item in batch]
    samples = torch.stack(samples, 0)

    targets = [item[1] for item in batch]
    return [samples, targets]


def main():
    parser = argparse.ArgumentParser(description='Classify event dataset')
    # Dataset
    parser.add_argument('-dataset', default='gen1', type=str, help='dataset used {GEN1}')
    parser.add_argument('-path',
                        default='/Gen1/detection_dataset_duration_60s_ratio_1.0/',
                        type=str,
                        help='path to dataset location')
    parser.add_argument('-num_classes', default=2, type=int, help='number of classes')
    parser.add_argument('-devices', default=[0], type=int, nargs='+', help='number of devices')

    # Data
    parser.add_argument('-b', default=32, type=int, help='batch size')
    parser.add_argument('-sample_size', default=100000, type=int, help='duration of a sample in µs')
    parser.add_argument('-T', default=5, type=int, help='simulating time-steps')
    parser.add_argument('-tbin', default=2, type=int, help='number of micro time bins')
    parser.add_argument('-image_shape', default=(240, 304), type=tuple, help='spatial resolution of events')
    parser.add_argument('-zoom_in', default=0.5, type=float, help='zoom_in weight')
    parser.add_argument('-zoom_in_factor', default=0.5, type=float, help='zoom_in_factor')
    parser.add_argument('-zoom_out_factor', default=0.2, type=float, help='zoom_out_factor')
    parser.add_argument('-zoom_prob', default=0.5, type=float, help='zoom probability')
    parser.add_argument('-flip_prob', default=0.5, type=float, help='filp probability')

    # Training
    parser.add_argument('-epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('-warmup_epochs', default=5, type=int, help='number of warm epochs to run')
    parser.add_argument('-linear_lr', action='store_true', help='saves checkpoints')
    parser.add_argument('-lrf', default=0.01, type=float, help='learning rate factor')
    parser.add_argument('-warmup_lrf', default=0.01, type=float, help='warmup learning rate factor')
    parser.add_argument('-lr', default=1e-3, type=float, help='learning rate used')
    parser.add_argument('-wd', default=1e-4, type=float, help='weight decay used')
    parser.add_argument('-limit_train_batches', default=1., type=float, help='train batches limit')
    parser.add_argument('-limit_val_batches', default=1., type=float, help='val batches limit')
    parser.add_argument('-limit_test_batches', default=1., type=float, help='test batches limit')
    parser.add_argument('-num_workers', default=4, type=int, help='number of workers for dataloaders')
    parser.add_argument('-no_train', action='store_false', help='whether to train the model', dest='train')
    parser.add_argument('-test', action='store_true', help='whether to test the model')
    parser.add_argument('-precision', default='16-mixed', type=str, help='whether to use AMP {16, 32, 64}')
    parser.add_argument('-save_ckpt', action='store_true', help='saves checkpoints')
    parser.add_argument('-early_stopping', action='store_true', help='early stopping')
    parser.add_argument('-augmentation', action='store_true', help='augment image')

    # Backbone
    parser.add_argument('-backbone', default='mdsresnet-18', type=str,
                        help='model used {model-v}', dest='model')
    parser.add_argument('-no_bn', action='store_false', help='don\'t use BatchNorm2d', dest='bn')
    parser.add_argument('-pretrained_backbone', default=None, type=str, help='path to pretrained backbone model')
    parser.add_argument('-pretrained', default=None, type=str, help='path to pretrained model')
    parser.add_argument('-decode', default='spiking', type=str, help='decoding method')

    # neck
    parser.add_argument('-extras', type=int, default=[512, 256], nargs=2,
                        help='number of channels for extra layers after the backbone')
    parser.add_argument('-channels', default=128, type=int, help='number of feature channels')
    parser.add_argument('-fusion', action='store_true', help='if to fusion the features')
    parser.add_argument('-attention', action='store_true', help='if to attention the features')
    parser.add_argument('-connect_f', default='ADD', type=str, help='connect mode {ADD, AND, IAND}')
    parser.add_argument('-recover', action='store_true', help='if to recover the output dimension')

    # Priors
    parser.add_argument('-min_ratio', default=0.05, type=float, help='min ratio for priors\' box generation')
    parser.add_argument('-max_ratio', default=0.80, type=float, help='max ratio for priors\' box generation')
    parser.add_argument('-aspect_ratios', default=[[2], [2, 3], [2, 3], [2, 3], [2]], type=int,
                        help='aspect ratios for priors\' box generation')

    # Loss parameters
    parser.add_argument('-box_coder_weights', default=[10.0, 10.0, 5.0, 5.0], type=float, nargs=4,
                        help='weights for the BoxCoder class')
    parser.add_argument('-iou_threshold', default=0.50, type=float,
                        help='intersection over union threshold for the SSDMatcher class')
    parser.add_argument('-score_thresh', default=0.01, type=float,
                        help='score threshold used for postprocessing the detections')
    parser.add_argument('-nms_thresh', default=0.45, type=float,
                        help='NMS threshold used for postprocessing the detections')
    parser.add_argument('-topk_candidates', default=200, type=int, help='number of best detections to keep before NMS')
    parser.add_argument('-detections_per_img', default=100, type=int,
                        help='number of best detections to keep after NMS')

    args = parser.parse_args()
    print(args)

    torch.set_float32_matmul_precision('medium')

    # Random seed
    SEED = 445
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    if args.dataset == "gen1":
        dataset = GEN1DetectionDataset
        torch.multiprocessing.set_sharing_strategy('file_system')
    else:
        sys.exit(f"{args.dataset} is not a supported dataset.")

    module = DetectionLitModule(args)

    # LOAD PRETRAINED MODEL
    if args.pretrained is not None:
        ckpt_path = join(f"aug-ckpt-od-{args.dataset}-{args.model}-val", args.pretrained)
        module = DetectionLitModule.load_from_checkpoint(ckpt_path, strict=False)

    callbacks = []
    if args.save_ckpt:
        ckpt_callback_val = ModelCheckpoint(
            monitor='val_loss',
            dirpath=f"aug-ckpt-od-{args.dataset}-{args.model}-val/",
            filename=f"{args.dataset}" + "-{epoch:02d}-{val_loss:.4f}",
            save_top_k=5,
            mode='min',
        )
        ckpt_callback_train = ModelCheckpoint(
            monitor='train_loss',
            dirpath=f"aug-ckpt-od-{args.dataset}-{args.model}-train/",
            filename=f"{args.dataset}" + "-{epoch:02d}-{train_loss:.4f}",
            save_top_k=3,
            mode='min',
        )
        callbacks.append(ckpt_callback_val)
        callbacks.append(ckpt_callback_train)

    if args.early_stopping:
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min',
        )
        callbacks.append(early_stopping)

    trainer = pl.Trainer(
        devices=args.devices, accelerator="gpu",  # args.devices
        gradient_clip_val=1., max_epochs=args.epochs,
        limit_train_batches=args.limit_train_batches, limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
        check_val_every_n_epoch=1,
        deterministic=False,
        precision=args.precision,
        callbacks=callbacks,
        strategy='ddp',
    )

    if args.train:
        train_dataset = dataset(args, mode="train")
        val_dataset = dataset(args, mode="val")

        train_dataloader = DataLoader(train_dataset, batch_size=args.b, collate_fn=augmentation_collate_fn,
                                      num_workers=args.num_workers, shuffle=True, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.b, collate_fn=collate_fn, num_workers=args.num_workers)

        trainer.fit(module, train_dataloader, val_dataloader)
    if args.test:
        test_dataset = dataset(args, mode="test")
        test_dataloader = DataLoader(test_dataset, batch_size=args.b, collate_fn=collate_fn,
                                     num_workers=args.num_workers)

        trainer.test(module, test_dataloader)


if __name__ == '__main__':
    main()
