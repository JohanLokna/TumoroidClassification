"""
Utility functions and classes for training
"""

from typing import Callable
import torch


def collate_fn(data: tuple, transform_img : Callable, transform_labels : Callable):
    """
    Define how the DataLoaders should batch the data

    Parameters:
        data:
            tuple such that data[0] is the data and data[1] is the label
    
    Returns:
        a dict of the input 'x' with shape (1, *x.shape), the label 'y' 
    """

    # Get properties from data
    device = data[0][0].device
    dtype = data[0][0].dtype

    # Get padding sizes
    img_x = max([d[0].shape[0] for d in data])
    img_y = max([d[0].shape[1] for d in data])

    # Create image batch
    imgs = [torch.zeros(img_x, img_y, dtype=dtype, device=device) for _ in data]
    for i, d in enumerate(data):
        d_x = d[0].shape[0]
        d_y = d[0].shape[1]
        img_transformed = transform_img(d[0].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        img_transformed[d == 0] = 0

        imgs[i][:d_x, :d_y] = img_transformed
    x = torch.stack(imgs, axis=0).unsqueeze(-3)

    y = transform_labels(torch.stack([torch.tensor(d[1]) for d in data]))
    return {'x': x, 'y': y}