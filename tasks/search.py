from enum import Enum
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from tasks.task_utils import Task
from utils import paste_shape

class SearchType(Enum):
    CONJUNCTIVE = 'conjunctive'
    DISJUNCTIVE = 'disjunctive'

@dataclass
class SearchObject:
    x: int
    y: int
    size: int
    color: str
    shape: str
    is_target: bool


class SearchTrial:
    """Represents a single search trial"""
    def __init__(
        self,
        search_type: SearchType,
        n_objects: int,
        trial_num: int,
        colors: List[str],
        shapes: List[str],
        size: int,
        canvas_size: Tuple[int, int]
    ):
        self.search_type = search_type
        self.n_objects = n_objects
        self.trial_num = trial_num
        self.size = size
        
        # Select target properties
        self.target_color = np.random.choice(colors)
        self.target_shape = np.random.choice(shapes)
        
        # Create and place objects
        self.objects = self._create_objects(colors, shapes)
        
    def _create_objects(
        self,
        colors: List[str],
        shapes: List[str]
    ) -> List[SearchObject]:
        """Create objects for the trial with positions determined later"""
        objects = []
        
        # Create target object (position will be set later)
        objects.append(SearchObject(0, 0, self.size, self.target_color, self.target_shape, True))
        
        # Create distractors
        for _ in range(self.n_objects - 1):
            if self.search_type == SearchType.CONJUNCTIVE:
                # Share one feature with target
                if np.random.random() < 0.5:
                    color = self.target_color
                    shape = np.random.choice([s for s in shapes if s != self.target_shape])
                else:
                    color = np.random.choice([c for c in colors if c != self.target_color])
                    shape = self.target_shape
            else:  # DISJUNCTIVE
                # Share no features with target
                color = np.random.choice([c for c in colors if c != self.target_color])
                shape = np.random.choice([s for s in shapes if s != self.target_shape])
                
            objects.append(SearchObject(0, 0, self.size, color, shape, False))
        
        # Place all objects at once using place_shapes
        try:
            _, positions = place_shapes(
                [np.zeros((3, self.size, self.size))] * len(objects),
                canvas_size=self.canvas_size,
                img_size=self.size
            )
            # Update object positions
            for obj, pos in zip(objects, positions):
                obj.x, obj.y = pos
        except ValueError:
            raise RuntimeError("Failed to place objects")
            
        return objects
    
    def to_metadata(self, image_path: str) -> Dict:
        """Convert trial data to metadata dictionary"""
        return {
            'path': str(image_path),
            'n_objects': self.n_objects,
            'search_type': self.search_type.value,
            'trial': self.trial_num,
            'target_color': self.target_color,
            'target_shape': self.target_shape,
            'objects_data': [vars(obj) for obj in self.objects]
        }

class SearchTask(Task):
    def __init__(
        self,
        min_objects: int,
        max_objects: int,
        n_trials: int,
        size: int,
        colors: List[str],
        shapes: List[str],
        shape_inds: List[int],
        canvas_size: Tuple[int, int] = (512, 512),
        **kwargs
    ):
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.n_trials = n_trials
        self.size = size
        self.colors = colors
        self.shapes = shapes
        self.shape_inds = shape_inds
        self.canvas_size = canvas_size
        
        super().__init__(**kwargs)

    def render_trial(self, trial: SearchTrial) -> Image.Image:
        """Create image for a trial"""
        canvas = Image.new('RGB', self.canvas_size, 'white')
        
        # Prepare all shapes at once
        shape_indices = [self.shape_inds[self.shapes.index(obj.shape)] for obj in trial.objects]
        positions = np.array([[obj.x, obj.y] for obj in trial.objects])
        sizes = np.array([obj.size for obj in trial.objects])
        
        # Place all shapes in one go
        for i, shape_idx in enumerate(shape_indices):
            paste_shape(
                shape=np.array([shape_idx]),
                positions=positions[i:i+1],
                sizes=sizes[i:i+1],
                canvas_img=canvas,
                i=0,
                img_size=trial.objects[i].size
            )
        return canvas

    def generate_full_dataset(self) -> pd.DataFrame:
        """Generate dataset of images with both conjunctive and disjunctive search trials."""
        img_path = Path(self.data_dir) / self.task_name / 'images'
        img_path.mkdir(parents=True, exist_ok=True)
        
        metadata = []
        for n_objects in range(self.min_objects, self.max_objects + 1):
            for search_type in SearchType:
                for trial_num in range(self.n_trials):
                    # Generate and render trial
                    trial = SearchTrial(
                        search_type=search_type,
                        n_objects=n_objects,
                        trial_num=trial_num,
                        colors=self.colors,
                        shapes=self.shapes,
                        size=self.size,
                        canvas_size=self.canvas_size
                    )
                    img = self.render_trial(trial)
                    
                    # Save image
                    filename = f'n={n_objects}_type={search_type.value}_trial={trial_num}.png'
                    save_path = img_path / filename
                    img.save(save_path)
                    
                    # Collect metadata
                    metadata.append(trial.to_metadata(save_path))
        
        return pd.DataFrame(metadata)
