from enum import Enum
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Tuple, Dict
from dataclasses import dataclass
import random

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
        self.canvas_size = canvas_size
        self.colors = colors
        self.shapes = shapes
        
        # Randomly select target features
        self.target_color = random.choice(colors)
        self.target_shape = random.choice(shapes)
        
        # Create and place objects
        self.objects = self._create_objects()

    def _create_objects(self) -> List[SearchObject]:
        """Create objects for the trial with random positions"""
        
        # Calculate valid position range
        margin = self.size // 2
        min_x = margin
        max_x = self.canvas_size[0] - margin
        min_y = margin
        max_y = self.canvas_size[1] - margin
        
        # Pre-allocate positions array
        positions = np.zeros((self.n_objects, 2))
        
        # Generate valid positions
        for i in range(self.n_objects):
            while True:
                pos = np.random.randint([min_x, min_y], [max_x, max_y], size=2)
                if i == 0:  # First object can go anywhere
                    positions[i] = pos
                    break
                    
                # Check distance from all previous objects
                distances = np.linalg.norm(positions[:i] - pos, axis=1)
                if np.all(distances >= self.size):
                    positions[i] = pos
                    break
        
        objects = []
        
        # Create target object first
        objects.append(SearchObject(
            x=positions[0,0],
            y=positions[0,1],
            size=self.size,
            color=self.target_color,
            shape=self.target_shape,
            is_target=True
        ))
        
        # Create distractors
        for i in range(1, self.n_objects):
            if self.search_type == SearchType.CONJUNCTIVE:
                # Share one feature with target
                if random.random() < 0.5:
                    color = self.target_color
                    shape = random.choice([s for s in self.shapes if s != self.target_shape])
                else:
                    color = random.choice([c for c in self.colors if c != self.target_color])
                    shape = self.target_shape
            else:  # DISJUNCTIVE
                # Share no features with target
                color = random.choice([c for c in self.colors if c != self.target_color])
                shape = random.choice([s for s in self.shapes if s != self.target_shape])
            
            objects.append(SearchObject(
                x=positions[i,0],
                y=positions[i,1],
                size=self.size,
                color=color,
                shape=shape,
                is_target=False
            ))
            
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
        self.shape_inds = np.array(shape_inds)
        self.canvas_size = canvas_size
        
        # Load shape images
        self.shape_imgs = np.load('data/imgs.npy')[self.shape_inds]
        self.shape_map = {shape: idx for idx, shape in enumerate(self.shapes)}
        
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
        trial_counter = 0
        
        for n_objects in range(self.min_objects, self.max_objects + 1):
            for search_type in SearchType:
                for _ in range(self.n_trials):
                    trial = SearchTrial(
                        search_type=search_type,
                        n_objects=n_objects,
                        trial_num=trial_counter,
                        colors=self.colors,
                        shapes=self.shapes,
                        size=self.size,
                        canvas_size=self.canvas_size
                    )
                    
                    img = self.render_trial(trial)
                    
                    # Save image
                    filename = f'n={n_objects}_type={search_type.value}_trial={trial_counter}.png'
                    save_path = img_path / filename
                    img.save(save_path)
                    
                    # Collect metadata
                    metadata.append(trial.to_metadata(str(save_path)))
                    trial_counter += 1
        
        return pd.DataFrame(metadata)
