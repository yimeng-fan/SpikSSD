import torch
import torchvision.transforms.functional as F


def zoom_in_augmentation(images, boxes, final_timestep, default_resolution):
    # Find a crop that contains at least one full bounding box
    for box in boxes[final_timestep]:
        xmin, ymin, xmax, ymax = box
        if xmin >= 0 and ymin >= 0 and xmax <= images.size(2) and ymax <= images.size(3):
            crop = torch.tensor([xmin, ymin, xmax, ymax])
            break
    else:
        # If no suitable crop is found, return the original images
        return images

    # Apply the crop to the entire sequence
    cropped_images = []
    for image in images:
        cropped_image = F.crop(image, crop[1], crop[0], crop[3] - crop[1], crop[2] - crop[0])
        cropped_images.append(cropped_image)

    # Rescale the cropped images to the default resolution
    rescaled_images = []
    for image in cropped_images:
        rescaled_image = F.resize(image, default_resolution)
        rescaled_images.append(rescaled_image)

    return torch.stack(rescaled_images)


def crop_boxes(boxes, crop):
    """
    :param boxes:  [number_boxes, 4] ([xmin, ymin, xmax, ymax])
    :param crop:  [4] ([xmin, ymin, xmax, ymax])
    :return: [number_boxes, 4] ([xmin, ymin, xmax, ymax])
    """
    boxes_cropped = boxes.clone()
    boxes_cropped[:, :2] = boxes[:, :2] - crop[:2]  # Update xmin and ymin
    boxes_cropped[:, 2:] = boxes[:, 2:] - crop[:2]  # Update xmax and ymax

    # Ensure the coordinates are not less than 0 and not greater than the size of the crop
    boxes_cropped[:, :2] = torch.clamp(boxes_cropped[:, :2], min=0)
    boxes_cropped[:, 2:] = torch.clamp(boxes_cropped[:, 2:], max=crop[2:] - crop[:2])

    return boxes_cropped
