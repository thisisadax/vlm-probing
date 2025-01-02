from enum import Enum
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Tuple, Dict
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

    def _get_valid_position(self, positions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Get a valid non-overlapping position for a new object."""
        while True:
            x = np.random.randint(0, self.canvas_size[0])
            y = np.random.randint(0, self.canvas_size[1])
            if not positions or all((x-px)**2 + (y-py)**2 > (self.size*2)**2 for px, py in positions):
                return x, y

    def make_conjunctive_trial(self, n_objects: int) -> List[SearchObject]:
        """Generate objects for a conjunctive search trial."""
        target_color = np.random.choice(self.colors)
        target_shape = np.random.choice(self.shapes)
        
        objects = []
        positions = []
        
        # Add target
        x, y = self._get_valid_position(positions)
        objects.append(SearchObject(x, y, self.size, target_color, target_shape, True))
        positions.append((x, y))
        
        # Add distractors that share one feature with target
        for _ in range(n_objects - 1):
            x, y = self._get_valid_position(positions)
            
            if np.random.random() < 0.5:
                # Share color, different shape
                color = target_color
                shape = np.random.choice([s for s in self.shapes if s != target_shape])
            else:
                # Share shape, different color
                color = np.random.choice([c for c in self.colors if c != target_color])
                shape = target_shape
                
            objects.append(SearchObject(x, y, self.size, color, shape, False))
            positions.append((x, y))
            
        return objects

    def make_disjunctive_trial(self, n_objects: int) -> List[SearchObject]:
        """Generate objects for a disjunctive search trial."""
        target_color = np.random.choice(self.colors)
        target_shape = np.random.choice(self.shapes)
        
        objects = []
        positions = []
        
        # Add target
        x, y = self._get_valid_position(positions)
        objects.append(SearchObject(x, y, self.size, target_color, target_shape, True))
        positions.append((x, y))
        
        # Add distractors with no shared features
        for _ in range(n_objects - 1):
            x, y = self._get_valid_position(positions)
            color = np.random.choice([c for c in self.colors if c != target_color])
            shape = np.random.choice([s for s in self.shapes if s != target_shape])
            
            objects.append(SearchObject(x, y, self.size, color, shape, False))
            positions.append((x, y))
            
        return objects

    def generate_full_dataset(self) -> pd.DataFrame:
        """Generate dataset of images with both conjunctive and disjunctive search trials."""
        img_path = Path(self.data_dir) / self.task_name / 'images'
        img_path.mkdir(parents=True, exist_ok=True)
        
        metadata = []
        for n_objects in range(self.min_objects, self.max_objects + 1):
            for search_type in SearchType:
                for trial in range(self.n_trials):
                    # Generate objects based on search type
                    if search_type == SearchType.CONJUNCTIVE:
                        objects = self.make_conjunctive_trial(n_objects)
                    else:
                        objects = self.make_disjunctive_trial(n_objects)
                    
                    # Create image
                    img = Image.new('RGB', self.canvas_size, 'white')
                    for obj in objects:
                        shape_idx = self.shape_inds[self.shapes.index(obj.shape)]
                        paste_shape(
                            shape=np.array([shape_idx]),
                            positions=np.array([[obj.x, obj.y]]),
                            sizes=np.array([obj.size]),
                            canvas_img=img,
                            i=0,
                            img_size=obj.size
                        )
                    
                    # Save image
                    filename = f'n={n_objects}_type={search_type.value}_trial={trial}.png'
                    save_path = img_path / filename
                    img.save(save_path)
                    
                    # Get target properties from first object (which is always the target)
                    target = objects[0]
                    metadata.append({
                        'path': str(save_path),
                        'n_objects': n_objects,
                        'search_type': search_type.value,
                        'trial': trial,
                        'target_color': target.color,
                        'target_shape': target.shape,
                        'objects_data': [vars(obj) for obj in objects]
                    })
        
        return pd.DataFrame(metadata)
