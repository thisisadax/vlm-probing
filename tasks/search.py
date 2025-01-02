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
        canvas_size: Tuple[int, int],
        target_present: bool = True
    ):
        self.search_type = search_type
        self.n_objects = n_objects
        self.trial_num = trial_num
        self.size = size
        self.canvas_size = canvas_size
        self.colors = colors
        self.shapes = shapes
        self.target_present = target_present
        
        # Randomly select target features
        self.target_color = random.choice(colors)
        self.target_shape = random.choice(shapes)
        
        # Select alternative features for distractors
        self.alt_color = random.choice([c for c in colors if c != self.target_color])
        self.alt_shape = random.choice([s for s in shapes if s != self.target_shape])
        
        # Create and place objects
        self.objects = self._create_objects()

    def _generate_positions(self) -> np.ndarray:
        """Generate non-overlapping positions for all objects"""
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
        
        return positions

    def _create_objects(self) -> List[SearchObject]:
        """Create objects for the trial with random positions"""
        positions = self._generate_positions()
        
        # Define possible feature combinations for distractors
        if self.search_type == SearchType.CONJUNCTIVE:
            # Share exactly one feature with target
            distractor_features = [
                (self.target_color, self.alt_shape),
                (self.alt_color, self.target_shape)
            ]
        else:  # DISJUNCTIVE
            # Share no features with target
            distractor_features = [(self.alt_color, self.alt_shape)]
            
        # Prepare features for all objects
        features = []
        is_targets = []
        
        # Add target if present
        if self.target_present:
            features.append((self.target_color, self.target_shape))
            is_targets.append(True)
        
        # Add distractors
        n_distractors = self.n_objects - (1 if self.target_present else 0)
        distractor_idxs = np.random.choice(len(distractor_features), size=n_distractors)
        features.extend([distractor_features[idx] for idx in distractor_idxs])
        is_targets.extend([False] * n_distractors)
            
        # Create all objects using list comprehension
        return [
            SearchObject(
                x=int(pos[0]),
                y=int(pos[1]),
                size=self.size,
                color=color,
                shape=shape,
                is_target=is_target
            )
            for pos, (color, shape), is_target in zip(positions, features, is_targets)
        ]
    
    def to_metadata(self, image_path: str) -> Dict:
        """Convert trial data to metadata dictionary"""
        return {
            'path': str(image_path),
            'n_objects': self.n_objects,
            'search_type': self.search_type.value,
            'trial': self.trial_num,
            'target_color': self.target_color,
            'target_shape': self.target_shape,
            'target_present': self.target_present,
            'objects_data': [vars(obj) for obj in self.objects]
        }


class Search(Task):
    def __init__(
        self,
        n_objects: List[int],
        n_trials: int,
        size: int,
        colors: List[str],
        shapes: List[str],
        shape_inds: List[int],
        canvas_size: Tuple[int, int] = (512, 512),
        **kwargs
    ):
        self.n_objects = n_objects
        self.n_trials = n_trials
        self.size = size
        self.colors = colors
        self.shapes = shapes
        self.shape_inds = np.array(shape_inds)
        self.canvas_size = tuple(canvas_size)
        
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
            shape_img = self.shape_imgs[shape_idx]  # Get the actual shape image
            paste_shape(
                shape=shape_img,
                positions=positions[i:i+1],
                sizes=sizes[i:i+1],
                canvas_img=canvas,
                i=0,
                img_size=trial.objects[i].size
            )
        return canvas

    def get_prompt(self, row: pd.Series) -> str:
        """Format the prompt template with the target color and shape for this trial."""
        return self.prompt.format(
            target_color=row['target_color'],
            target_shape=row['target_shape']
        )
        
    def generate_full_dataset(self) -> pd.DataFrame:
        """Generate dataset of images with both conjunctive and disjunctive search trials."""
        img_path = Path(self.data_dir) / self.task_name / 'images'
        img_path.mkdir(parents=True, exist_ok=True)
        
        metadata = []
        trial_counter = 0
        
        for n_objects in self.n_objects:
            for search_type in SearchType:
                for _ in range(self.n_trials):
                    # Create one target-present and one target-absent trial
                    for target_present in [True, False]:
                        trial = SearchTrial(
                            search_type=search_type,
                            n_objects=n_objects,
                            trial_num=trial_counter,
                            colors=self.colors,
                            shapes=self.shapes,
                            size=self.size,
                            canvas_size=self.canvas_size,
                            target_present=target_present
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
