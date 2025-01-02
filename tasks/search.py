from enum import Enum
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Tuple, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod

from tasks.task_utils import Task
from utils import paste_shape

@dataclass
class SearchObject:
    x: int
    y: int
    size: int
    color: str
    shape: str
    is_target: bool

class SearchStrategy(ABC):
    """Base class for different search task strategies"""
    def __init__(self, colors: List[str], shapes: List[str]):
        self.colors = colors
        self.shapes = shapes
    
    @abstractmethod
    def generate_distractors(self, target_color: str, target_shape: str, n_distractors: int) -> List[Tuple[str, str]]:
        """Generate color-shape pairs for distractor objects"""
        pass

class ConjunctiveSearch(SearchStrategy):
    def generate_distractors(self, target_color: str, target_shape: str, n_distractors: int) -> List[Tuple[str, str]]:
        distractors = []
        for _ in range(n_distractors):
            if np.random.random() < 0.5:
                # Share color, different shape
                color = target_color
                shape = np.random.choice([s for s in self.shapes if s != target_shape])
            else:
                # Share shape, different color
                color = np.random.choice([c for c in self.colors if c != target_color])
                shape = target_shape
            distractors.append((color, shape))
        return distractors

class DisjunctiveSearch(SearchStrategy):
    def generate_distractors(self, target_color: str, target_shape: str, n_distractors: int) -> List[Tuple[str, str]]:
        distractors = []
        for _ in range(n_distractors):
            color = np.random.choice([c for c in self.colors if c != target_color])
            shape = np.random.choice([s for s in self.shapes if s != target_shape])
            distractors.append((color, shape))
        return distractors

class Trial:
    """Represents a single search trial with its objects and metadata"""
    def __init__(self, objects: List[SearchObject], search_type: str, n_objects: int, trial_num: int):
        self.objects = objects
        self.search_type = search_type
        self.n_objects = n_objects
        self.trial_num = trial_num
        self.target = objects[0]  # First object is always target
    
    def to_metadata(self, image_path: str) -> Dict:
        return {
            'path': str(image_path),
            'n_objects': self.n_objects,
            'search_type': self.search_type,
            'trial': self.trial_num,
            'target_color': self.target.color,
            'target_shape': self.target.shape,
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
        
        self.strategies = {
            'conjunctive': ConjunctiveSearch(colors, shapes),
            'disjunctive': DisjunctiveSearch(colors, shapes)
        }
        
        super().__init__(**kwargs)

    def _get_valid_position(self, positions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Get a valid non-overlapping position for a new object."""
        while True:
            x = np.random.randint(0, self.canvas_size[0])
            y = np.random.randint(0, self.canvas_size[1])
            if not positions or all((x-px)**2 + (y-py)**2 > (self.size*2)**2 for px, py in positions):
                return x, y

    def generate_trial(self, n_objects: int, search_type: str, trial_num: int) -> Trial:
        """Generate a single trial with the specified parameters"""
        positions = []
        objects = []
        
        # Create target
        target_color = np.random.choice(self.colors)
        target_shape = np.random.choice(self.shapes)
        x, y = self._get_valid_position(positions)
        objects.append(SearchObject(x, y, self.size, target_color, target_shape, True))
        positions.append((x, y))
        
        # Generate distractors using appropriate strategy
        strategy = self.strategies[search_type]
        distractors = strategy.generate_distractors(target_color, target_shape, n_objects - 1)
        
        # Create distractor objects
        for color, shape in distractors:
            x, y = self._get_valid_position(positions)
            objects.append(SearchObject(x, y, self.size, color, shape, False))
            positions.append((x, y))
            
        return Trial(objects, search_type, n_objects, trial_num)

    def render_trial(self, trial: Trial) -> Image.Image:
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
            for search_type in ['conjunctive', 'disjunctive']:
                for trial_num in range(self.n_trials):
                    # Generate and render trial
                    trial = self.generate_trial(n_objects, search_type, trial_num)
                    img = self.render_trial(trial)
                    
                    # Save image
                    filename = f'n={n_objects}_type={search_type}_trial={trial_num}.png'
                    save_path = img_path / filename
                    img.save(save_path)
                    
                    # Collect metadata
                    metadata.append(trial.to_metadata(save_path))
        
        return pd.DataFrame(metadata)
