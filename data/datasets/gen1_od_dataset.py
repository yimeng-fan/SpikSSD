import copy
import os
import random

import torchvision
import tqdm

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torch.nn.functional as F

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured

from models.SSD_utils import filter_boxes
from prophesee_utils.io.psee_loader import PSEELoader


# modified from https://github.com/loiccordone/object-detection-with-spiking-neural-networks/blob/main/datasets/gen1_od_dataset.py

class GEN1DetectionDataset(Dataset):
    def __init__(self, args, mode="train"):
        self.mode = mode
        self.tbin = args.tbin
        self.C, self.T = 2 * args.tbin, args.T
        self.sample_size = args.sample_size  # duration of a sample in µs
        self.quantization_size = [args.sample_size // args.T, 1, 1]  # T，y，x
        self.h, self.w = args.image_shape
        self.quantized_w = self.w // self.quantization_size[1]
        self.quantized_h = self.h // self.quantization_size[2]
        self.augmentation = args.augmentation
        self.zoom_augmentor = ZoomAugmentor(zoom_in_factor=args.zoom_in_factor, zoom_out_factor=args.zoom_out_factor,
                                            zoom_in_weight=args.zoom_in)
        self.zoom_prob = args.zoom_prob
        self.flip_prob = args.flip_prob

        save_file_name = \
            f"gen1_{mode}_{self.sample_size // 1000}_{self.quantization_size[0] / 1000}ms_{self.tbin}tbin.pt"
        save_file = os.path.join(args.path, save_file_name)
        if os.path.isfile(save_file):
            self.samples = torch.load(save_file)
            print("File loaded.")
        else:
            data_dir = os.path.join(args.path, mode)
            self.samples = self.build_dataset(data_dir, save_file)
            torch.save(self.samples, save_file)
            print(f"Done! File saved as {save_file}")

    def __getitem__(self, index):
        (coords, feats), target = self.samples[index]

        sample = torch.sparse_coo_tensor(
            coords.t(),
            feats.to(torch.float32),
            size=(self.T, self.quantized_h, self.quantized_w, self.C)
        )
        sample = sample.coalesce().to_dense().permute(0, 3, 1, 2)

        # augmentations
        if (self.mode == 'train' or self.mode == 'trainval') and self.augmentation:
            if target['labels'].sum() > 0:
                aug_sample1, aug_target1 = self._augmentation_func(sample.clone(), copy.deepcopy(target))
                # aug_sample2, aug_target2 = self._augmentation_func(sample.clone(), copy.deepcopy(target))

                # augmentations
                sample, target = self._augmentation_func(sample, target)

                return sample, target, aug_sample1, aug_target1  # , aug_sample2, aug_target2
            else:
                sample, target = self._augmentation_func(sample, target)

                return sample, target, None, None

        return sample, target

    def _augmentation_func(self, sample, target):
        if random.random() < self.zoom_prob:
            sample, target = self.zoom_augmentor(sample, target)

        if random.random() < self.flip_prob:
            sample, target = horizontal_flip(sample, target)

        '''if random.random() < self.translate_prob:
            sample, target = translate_image(sample, target)'''

        return sample, target

    def _up_augmentation_func(self, sample, target):
        if random.random() < self.zoom_prob:
            sample, target = self.zoom_augmentor(sample, target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def build_dataset(self, path, save_file):
        # Remove duplicates (.npy and .dat)
        files = [os.path.join(path, time_seq_name[:-9]) for time_seq_name in os.listdir(path)
                 if time_seq_name[-3:] == 'npy']

        print('Building the Dataset')
        pbar = tqdm.tqdm(total=len(files), unit='File', unit_scale=True)
        samples = []
        for file_name in files:
            print(f"Processing {file_name}...")
            events_file = file_name + '_td.dat'
            video = PSEELoader(events_file)

            boxes_file = file_name + '_bbox.npy'
            boxes = np.load(boxes_file)
            # Rename 'ts' in 't' if needed (Prophesee GEN1)
            boxes.dtype.names = [dtype if dtype != "ts" else "t" for dtype in boxes.dtype.names]

            boxes_per_ts = np.split(boxes, np.unique(boxes['t'], return_index=True)[1][1:])

            samples.extend([sample for b in boxes_per_ts if (sample := self.create_sample(video, b)) is not None])
            pbar.update(1)

        pbar.close()
        return samples

    def create_sample(self, video, boxes):
        ts = boxes['t'][0]
        video.seek_time(ts - self.sample_size)
        events = video.load_delta_t(self.sample_size)

        targets = self.create_targets(boxes)

        if targets['boxes'].shape[0] == 0:
            print(f"No boxes at {ts}")
            return None
        elif events.size == 0:
            print(f"No events at {ts}")
            return None
        else:
            return (self.create_data(events), targets)

    def create_targets(self, boxes):
        torch_boxes = torch.from_numpy(structured_to_unstructured(boxes[['x', 'y', 'w', 'h']], dtype=np.float32))

        # keep only last instance of every object per target
        _, unique_indices = np.unique(np.flip(boxes['track_id']), return_index=True)  # keep last unique objects
        unique_indices = np.flip(-(unique_indices + 1))
        torch_boxes = torch_boxes[[*unique_indices]]

        torch_boxes[:, 2:] += torch_boxes[:, :2]  # implicit conversion to xyxy
        torch_boxes[:, 0::2].clamp_(min=0, max=self.w)
        torch_boxes[:, 1::2].clamp_(min=0, max=self.h)

        # valid idx = width and height of GT bbox aren't 0
        valid_idx = (torch_boxes[:, 2] - torch_boxes[:, 0] != 0) & (torch_boxes[:, 3] - torch_boxes[:, 1] != 0)
        torch_boxes = torch_boxes[valid_idx, :]

        torch_labels = torch.from_numpy(boxes['class_id']).to(torch.long)
        torch_labels = torch_labels[[*unique_indices]]
        torch_labels = torch_labels[valid_idx]

        return {'boxes': torch_boxes, 'labels': torch_labels}

    def create_data(self, events):
        events['t'] -= events['t'][0]
        feats = torch.nn.functional.one_hot(torch.from_numpy(events['p']).to(torch.long), self.C)

        coords = torch.from_numpy(
            structured_to_unstructured(events[['t', 'y', 'x']], dtype=np.int32))

        # Bin the events on T timesteps
        coords = torch.floor(coords / torch.tensor(self.quantization_size))
        coords[:, 1].clamp_(min=0, max=self.quantized_h - 1)
        coords[:, 2].clamp_(min=0, max=self.quantized_w - 1)

        # TBIN computations
        tbin_size = self.quantization_size[0] / self.tbin

        # get for each ts the corresponding tbin index
        tbin_coords = (events['t'] % self.quantization_size[0]) // tbin_size
        # tbin_index * polarity produces the real tbin index according to polarity (range 0-(tbin*2))
        # tbin_feats = ((events['p'] + 1) * (tbin_coords + 1)) - 1 bug
        polarity = events['p'].copy().astype(np.int8)
        polarity[events['p'] == 0] = -1
        tbin_feats = (polarity * (tbin_coords + 1))
        tbin_feats[tbin_feats > 0] -= 1
        tbin_feats += (tbin_coords + 1).max()

        feats = torch.nn.functional.one_hot(torch.from_numpy(tbin_feats).to(torch.long), 2 * self.tbin).to(bool)

        return coords.to(torch.int16), feats


class ZoomAugmentor:
    def __init__(self, zoom_in_factor=0.2, zoom_out_factor=0.2, zoom_in_weight=0.5):
        self.zoom_in_factor = zoom_in_factor
        self.zoom_out_factor = zoom_out_factor
        self.zoom_in_weight = zoom_in_weight

    def __call__(self, image, targets):
        if random.random() < self.zoom_in_weight:
            zoom_factor = random.uniform(1., 1. + self.zoom_in_factor)
            return zoom_in(image, targets, zoom_factor)
        else:
            zoom_factor = random.uniform(1. - self.zoom_out_factor, 1.)
            return zoom_out(image, targets, zoom_factor)


def horizontal_flip_boxes(boxes, width):
    """
    Perform a horizontal flip on the bounding box

    :param boxes: Boundary box, shaped as [number_boxes, 4] ([xmin, ymin, xmax, ymax])
    :param width: The width of the image
    :return: Flipped bounding box
    """
    boxes_flipped = boxes.clone()
    boxes_flipped[:, 0] = width - boxes[:, 2]
    boxes_flipped[:, 2] = width - boxes[:, 0]
    return boxes_flipped


def horizontal_flip(image, targets):
    num_steps, channels, height, width = image.size()

    transform_t = torch.eye(3)  # 3x3 identity matrix
    transform_t[0, 0] *= -1  # flip horizontal
    transform_t_single = transform_t[:2, :].unsqueeze(0).repeat(num_steps, 1, 1).to(torch.float32)
    affine_t = F.affine_grid(transform_t_single.view(-1, 2, 3), [num_steps, channels, height, width],
                             align_corners=False)

    image_augmented = F.grid_sample(image, affine_t, padding_mode='border', align_corners=False)

    targets['boxes'] = horizontal_flip_boxes(targets['boxes'], width)

    return image_augmented, targets


def translate_image(image, targets, translate=0.1):
    """
    Translate the input image and corresponding bounding boxes.

    Args:
        image (torch.Tensor): Input image tensor of shape (N, C, H, W).
        targets (torch.Tensor): Bounding boxes tensor of shape (N, 4) in format [xmin, ymin, xmax, ymax].
        translate (float): Translation factor as a fraction of image dimensions.

    Returns:
        torch.Tensor: Translated image.
        torch.Tensor: Translated bounding boxes.
    """
    N, C, H, W = image.shape  # Get image dimensions

    # Calculate translation distances
    translate_x = torch.tensor(random.uniform(-translate * W, translate * W), device=image.device)
    translate_y = torch.tensor(random.uniform(-translate * H, translate * H), device=image.device)

    # Perform image translation
    translated_image = torch.zeros_like(image)

    # Calculate translated image boundaries
    left_bound = max(0, int(translate_x))
    right_bound = min(W, W + int(translate_x))
    top_bound = max(0, int(translate_y))
    bottom_bound = min(H, H + int(translate_y))

    # Calculate source image boundaries
    left_src = max(0, -int(translate_x))
    right_src = min(W, W - int(translate_x))
    top_src = max(0, -int(translate_y))
    bottom_src = min(H, H - int(translate_y))

    # Perform image translation with boundary clipping
    translated_image[:, :, top_bound:bottom_bound, left_bound:right_bound] = image[:, :, top_src:bottom_src,
                                                                             left_src:right_src]

    boxes = targets['boxes']

    # Perform bounding boxes translation
    if boxes.numel() > 0:
        boxes[:, 0] += translate_x  # xmin
        boxes[:, 1] += translate_y  # ymin
        boxes[:, 2] += translate_x  # xmax
        boxes[:, 3] += translate_y  # ymax

        # Clamp bounding boxes to image boundaries
        boxes[:, 0].clamp_(min=0, max=W)  # Clamp xmin
        boxes[:, 1].clamp_(min=0, max=H)  # Clamp ymin
        boxes[:, 2].clamp_(min=0, max=W)  # Clamp xmax
        boxes[:, 3].clamp_(min=0, max=H)  # Clamp ymax

        # Compute width and height of boxes
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]

        # Filter out boxes with width or height equal to 0
        keep_mask = (widths > 0) & (heights > 0)
        targets['boxes'] = boxes[keep_mask]
        targets['labels'] = targets['labels'][keep_mask]

    return translated_image, targets
