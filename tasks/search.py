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

class ObjectPlacer:
    """Handles object placement with collision detection"""
    def __init__(self, canvas_size: Tuple[int, int], min_spacing: int):
        self.canvas_size = canvas_size
        self.min_spacing = min_spacing
        self.positions: List[Tuple[int, int]] = []
        
    def place_object(self, size: int, max_attempts: int = 1000) -> Optional[Tuple[int, int]]:
        """Get valid non-overlapping position or None if placement fails"""
        for _ in range(max_attempts):
            x = np.random.randint(size, self.canvas_size[0] - size)
            y = np.random.randint(size, self.canvas_size[1] - size)
            
            if not self.positions or all(
                (x-px)**2 + (y-py)**2 > self.min_spacing**2 
                for px, py in self.positions
            ):
                self.positions.append((x, y))
                return x, y
        return None

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
        
        # Place objects
        placer = ObjectPlacer(canvas_size, min_spacing=size*2)
        self.objects = self._create_objects(placer, colors, shapes)
        
    def _create_objects(
        self, 
        placer: ObjectPlacer,
        colors: List[str],
        shapes: List[str]
    ) -> List[SearchObject]:
        """Create and place all objects for the trial"""
        objects = []
        
        # Place target
        if (pos := placer.place_object(self.size)) is None:
            raise RuntimeError("Failed to place target")
        objects.append(SearchObject(*pos, self.size, self.target_color, self.target_shape, True))
        
        # Place distractors
        for _ in range(self.n_objects - 1):
            if (pos := placer.place_object(self.size)) is None:
                raise RuntimeError("Failed to place distractor")
                
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
                
            objects.append(SearchObject(*pos, self.size, color, shape, False))
            
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
        img = Image.new('RGB', self.canvas_size, 'white')
        for obj in trial.objects:
            shape_idx = self.shape_inds[self.shapes.index(obj.shape)]
            paste_shape(
                shape=np.array([shape_idx]),
                positions=np.array([[obj.x, obj.y]]),
                sizes=np.array([obj.size]),
                canvas_img=img,
                i=0,
                img_size=obj.size
            )
        return img

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
