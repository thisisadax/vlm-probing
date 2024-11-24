import numpy as np
import numpy as np
from PIL import Image
from typing import List
import hydra
from omegaconf import DictConfig


def instantiate_modules(module_cfgs: DictConfig) -> List:
    '''Instantiate all modules with _target_ defined in config.'''
    modules = []
    for _, module in module_cfgs.items():
        if '_target_' in module:
            modules.append(hydra.utils.instantiate(module))
    return modules


def paste_shape(shape: np.ndarray,
               positions: np.ndarray, 
               sizes: np.ndarray,
               canvas_img: Image.Image,
               i: int,
               img_size: int = 40,
               max_attempts: int = 1000) -> np.ndarray:
   '''
   Paste a shape onto a canvas image at a non-overlapping position.
   Raises ValueError if unable to find valid position after max_attempts tries.
   '''
   assert len(positions) == len(sizes), 'positions and sizes must have same length'
   img = Image.fromarray(np.transpose(shape, (1, 2, 0)))
   max_pos = 256 - img_size  # Adjust for shape size to keep within canvas bounds
   position = np.random.randint(0, max_pos, size=2).reshape(1, -1)
   
   attempts = 0
   while attempts < max_attempts:
       # Skip overlap check for first shape
       if i == 0:
           break
           
       # Calculate minimum distances needed to avoid overlap
       min_distances = (sizes[:i] + img_size) / 2
       distances = np.linalg.norm(positions[:i] - position, axis=1)
       if np.all(distances >= min_distances):
           break
           
       position = np.random.randint(0, max_pos, size=2).reshape(1, -1)
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