from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Tuple, Dict
import matplotlib.colors as mcolors
import itertools

from tasks.task import Task
from utils import paste_shape, color_shape, place_shapes

class SceneDescription(Task):
    """Task class for generating scenes with multiple objects of varying colors and shapes"""
    
    def __init__(
        self,
        min_objects: int,
        max_objects: int,
        n_trials: int,
        size: int,
        colors: List[str],
        shapes: List[str],
        shape_inds: List[int],
        canvas_size: Tuple[int, int] = (256, 256),
        **kwargs
    ):
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.n_trials = n_trials
        self.size = size
        self.shapes = shapes
        self.shape_inds = shape_inds
        self.canvas_size = canvas_size
        
        # Convert color names to RGB values
        self.colors = {color: np.array(mcolors.to_rgb(color)) * 255 
                      for color in colors}
        
        # Generate all possible feature combinations
        self.feature_combinations = list(itertools.product(
            self.shapes, self.colors.keys()
        ))
        
        super().__init__(**kwargs)
        
        # Load shape images
        self.shape_imgs = np.load(Path(self.data_dir) / 'imgs.npy')[self.shape_inds]
        self.shape_map = {shape: idx for idx, shape in enumerate(self.shapes)}

    def generate_full_dataset(self) -> pd.DataFrame:
        """Generate dataset of images with varying numbers of objects"""
        img_path = Path(self.data_dir) / self.task_name / 'images'
        img_path.mkdir(parents=True, exist_ok=True)
        
        metadata = []
        objects_range = range(self.min_objects, self.max_objects + 1)
        
        for n_objects in objects_range:
            for trial in range(self.n_trials):
                # Sample n_objects unique feature combinations
                selected_features = self._sample_features(n_objects)
                if selected_features is None:
                    continue
                    
                # Prepare all colored shapes
                colored_shapes = []
                for shape_name, color_name in selected_features:
                    shape_idx = self.shape_map[shape_name]
                    shape_img = self.shape_imgs[shape_idx]
                    colored_shape = color_shape(shape_img, self.colors[color_name])
                    colored_shapes.append(colored_shape)
                
                # Place all shapes at once
                try:
                    canvas, positions = place_shapes(
                        colored_shapes,
                        canvas_size=self.canvas_size,
                        img_size=self.size
                    )
                except ValueError:
                    print(f"Warning: Failed to place objects in trial {trial}")
                    continue
                
                # Save image and metadata
                filename = f'objects={n_objects}_trial={trial}.png'
                save_path = img_path / filename
                canvas.save(save_path)
                
                # Record features and metadata
                features_dict = {
                    f"object_{i}": {
                        "shape": shape,
                        "color": color,
                        "position": positions[i].tolist()
                    }
                    for i, (shape, color) in enumerate(selected_features)
                }
                
                unique_shapes = len(set(shape for shape, _ in selected_features))
                unique_colors = len(set(color for _, color in selected_features))
                
                metadata.append({
                    'path': str(save_path),
                    'n_objects': n_objects,
                    'trial': trial,
                    'features': features_dict,
                    'unique_shapes': unique_shapes,
                    'unique_colors': unique_colors
                })
        
        return pd.DataFrame(metadata)
    
    def _sample_features(self, n_objects: int) -> List[Tuple[str, str]]:
        """Sample n_objects unique feature combinations"""
        if n_objects > len(self.feature_combinations):
            print(f"Warning: Requested {n_objects} objects but only {len(self.feature_combinations)} combinations available")
            return None
            
        return list(np.random.choice(
            self.feature_combinations, 
            size=n_objects, 
            replace=False
        ))
