import numpy as np
from PIL import Image
from typing import List, Tuple
import hydra
from omegaconf import DictConfig
import pandas as pd
import torch

def tensor_to_array(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

def collect_outputs(outputs, keys=['predictions', 'targets', 'attention', 'masks']):
    '''
    Collects outputs from the lightning validation loop, concatenating tensors/floats and 
    converting the 'metadata' field to a DataFrame.
    
    Args:
        outputs: List of dictionaries with tensors, floats, and a 'metadata' dict
        keys: List of keys to collect from the 'metadata' dict
        
    Returns:
        Dictionary with concatenated tensors and metadata DataFrame
    '''
    if not outputs:
        return {}
    
    metadata = [{k: tensor_to_array(v) for k, v in batch['metadata'].items()} for batch in outputs]
    metadata = pd.concat([pd.DataFrame(x) for x in metadata])
    
    # Process each key
    tensors = {}
    for key in keys:
        tensors[key] = torch.cat([x[key] for x in outputs]).squeeze().cpu()
    
    # Combine results
    return {**tensors, 'metadata': metadata}


def instantiate_modules(module_cfgs: DictConfig) -> List:
    '''Instantiate all modules with _target_ defined in config.'''
    modules = []
    for _, module in module_cfgs.items():
        if '_target_' in module:
            modules.append(hydra.utils.instantiate(module))
    return modules


def place_shapes(shape_imgs: List[np.ndarray],
                canvas_size: Tuple[int, int] = (256, 256),
                img_size: int = 40) -> Tuple[Image.Image, np.ndarray]:
    '''
    Place multiple shapes on a canvas at non-overlapping positions.
    
    Args:
        shape_imgs: List of shape images to place
        canvas_size: Size of the canvas (height, width)
        img_size: Size of each shape image
        
    Returns:
        Tuple of (canvas image, array of positions)
    '''
    # Create blank canvas
    canvas = np.ones((3, canvas_size[0], canvas_size[1]), dtype=np.uint8) * 255
    canvas = np.transpose(canvas, (1, 2, 0))  # Transpose to (HxWx3) for PIL
    canvas_img = Image.fromarray(canvas)
    
    # Initialize positions array
    n_shapes = len(shape_imgs)
    positions = np.zeros([n_shapes, 2])
    sizes = np.full(n_shapes, img_size)
    
    # Place each shape
    for i, img in enumerate(shape_imgs):
        positions = paste_shape(img, positions, sizes, canvas_img, i, img_size=img_size)
        
    return canvas_img, positions

def paste_shape(shape: np.ndarray,
               positions: np.ndarray, 
               sizes: np.ndarray,
               canvas_img: Image.Image,
               i: int,
               img_size: int = 40,
               max_attempts: int = 5000,
               padding: int = 20) -> np.ndarray:
   '''
   Paste a shape onto a canvas image at a non-overlapping position.
   Raises ValueError if unable to find valid position after max_attempts tries.
   '''
   assert len(positions) == len(sizes), 'positions and sizes must have same length'
   img = Image.fromarray(np.transpose(shape.astype(np.uint8), (1, 2, 0)))
   canvas_width, canvas_height = canvas_img.size
   max_pos_x = canvas_width - img_size  # Adjust for shape size to keep within canvas bounds
   max_pos_y = canvas_height - img_size  # Adjust for shape size to keep within canvas bounds
   position = np.random.randint(0, [max_pos_x, max_pos_y], size=2).reshape(1, -1)
   
   attempts = 0
   while attempts < max_attempts:
       # Skip overlap check for first shape
       if i == 0:
           break
           
       # Calculate minimum distances needed to avoid overlap
       min_distances = (sizes[:i] + img_size) / 2 + padding
       distances = np.linalg.norm(positions[:i] - position, axis=1)
       if np.all(distances >= min_distances):
           break
           
       position = np.random.randint(0, [max_pos_x, max_pos_y], size=2).reshape(1, -1)
       attempts += 1
   
   if attempts == max_attempts:
       raise ValueError(f'Failed to find non-overlapping position after {max_attempts} attempts')
       
   canvas_img.paste(img, tuple(position.squeeze()))
   positions[i] = position
   return positions


def color_shape(img: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    '''
    Colors a grayscale numpy array based on pixel intensities and target RGB color.
    
    Args:
        img: Grayscale image as numpy array of shape (H, W)
        rgb: RGB color values as numpy array of shape (3,)
        
    Returns:
        Colored image as numpy array of shape (3, H, W)
    '''
    rgb = rgb.astype(np.float32) / 255
    img = img.astype(np.float32) / 255
    
    # Linear interpolation between target color and white based on intensity
    colored = rgb.reshape(3, 1, 1) + (1 - rgb.reshape(3, 1, 1)) * img
    return (255 * colored).astype(np.uint8)


def get_attr_or_item(obj, attr):
    '''
    Access nested attributes or indexed items in a PyTorch module flexibly.
    Allows for unified access to both object attributes and ModuleList indices
    using a single interface.

    Args:
        obj: The object to access (typically a PyTorch module or submodule)
        attr (str): The attribute name or index to access. If the string represents 
                   a number (e.g., '0', '1', '28'), it will be used as an index.
                   Otherwise, it will be used as an attribute name.

    Returns:
        The accessed attribute or indexed item
    '''
    try:
        index = int(attr)  # Try to convert the attribute to an integer
        return obj[index]  # If successful, use it as an index
    except ValueError:
        # If conversion fails, treat it as a regular attribute name
        return getattr(obj, attr)