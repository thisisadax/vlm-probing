from enum import Enum
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Tuple, Dict
from dataclasses import dataclass
import random
import numpy as np
from matplotlib import colors as mcolors

from tasks.task_utils import Task
from utils import color_shape


class SearchCondition(Enum):
    SHARED_FEATURE = 'shared_feature'  # Distractors share one feature with target
    NO_SHARED_FEATURE = 'no_shared_feature'  # Distractors share no features with target

@dataclass
class SearchObject:
    x: int
    y: int
    size: int
    color: str
    shape: str
    is_target: bool

class SearchTrial:
    """Represents a single search trial with controlled distractor properties"""
    def __init__(
        self,
        search_condition: SearchCondition,
        n_objects: int,
        trial_num: int,
        colors: List[str],
        shapes: List[str],
        size: int,
        canvas_size: Tuple[int, int],
        target_present: bool = True
    ):
        self.search_condition = search_condition
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
        self.alt_colors = [c for c in colors if c != self.target_color]
        self.alt_shapes = [s for s in shapes if s != self.target_shape]
        
        # Create and place objects
        self.objects = self._create_objects()

    def _generate_positions(self, buffer=5) -> np.ndarray:
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
                if np.all(distances >= self.size+buffer):
                    positions[i] = pos
                    break
        
        # Shuffle positions before returning
        np.random.shuffle(positions)
        return positions

    def _create_objects(self) -> List[SearchObject]:
        """Create objects for the trial with controlled distractor features"""
        positions = self._generate_positions()
        
        # Prepare features for all objects
        features = []
        is_targets = []
        
        # Add target if present
        if self.target_present:
            features.append((self.target_color, self.target_shape))
            is_targets.append(True)
        
        # Add distractors based on condition
        n_distractors = self.n_objects - (1 if self.target_present else 0)
        
        if self.search_condition == SearchCondition.SHARED_FEATURE:
            # Create distractors that share exactly one feature with the target
            for _ in range(n_distractors):
                # Randomly decide which feature to share (color or shape)
                if random.choice([True, False]):
                    # Share color, different shape
                    features.append((self.target_color, random.choice(self.alt_shapes)))
                else:
                    # Share shape, different color
                    features.append((random.choice(self.alt_colors), self.target_shape))
                is_targets.append(False)
                
        else:  # NO_SHARED_FEATURE
            # Create distractors that share no features with the target
            for _ in range(n_distractors):
                features.append((
                    random.choice(self.alt_colors),
                    random.choice(self.alt_shapes)
                ))
                is_targets.append(False)
            
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
            'search_condition': self.search_condition.value,
            'trial': self.trial_num,
            'target_color': self.target_color,
            'target_shape': self.target_shape,
            'target_present': self.target_present,
            'features': {f'object_{i}': vars(obj) for i, obj in enumerate(self.objects)}
        }


class SearchControlled(Task):
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
        
        # Place each object at its predetermined position
        for obj in trial.objects:
            # Get shape image and color it
            shape_idx = self.shape_map[obj.shape]
            shape_img = self.shape_imgs[shape_idx]  # This is a (32, 32) grayscale image
            rgb_color = np.array([int(255 * x) for x in mcolors.to_rgb(obj.color)])
            colored_shape = color_shape(shape_img, rgb_color)  # Returns (3, H, W)
            
            # Convert to PIL Image
            shape_pil = Image.fromarray(colored_shape.transpose(1, 2, 0))
            
            # Resize if needed
            if shape_pil.size != (obj.size, obj.size):
                shape_pil = shape_pil.resize((obj.size, obj.size))
            
            # Calculate top-left position for pasting
            paste_x = obj.x - obj.size // 2
            paste_y = obj.y - obj.size // 2
            
            # Paste the shape
            canvas.paste(shape_pil, (paste_x, paste_y))
            
        return canvas

    def get_prompt(self, row: pd.Series) -> str:
        """Format the prompt template with the target color and shape for this trial."""
        try:
            return self.prompt.format(
                target_color=row.target_color,
                target_shape=row.target_shape
            )
        except KeyError as e:
            print(f"Error formatting prompt. Row contents: {row.to_dict()}")
            print(f"Prompt template: {self.prompt}")
            raise
        
    def generate_full_dataset(self) -> pd.DataFrame:
        """Generate dataset of images with controlled search conditions."""
        img_path = Path(self.data_dir) / self.task_name / 'images'
        img_path.mkdir(parents=True, exist_ok=True)
        
        metadata = []
        trial_counter = 0
        
        for n_objects in self.n_objects:
            for search_condition in SearchCondition:
                for _ in range(self.n_trials):
                    # Create one target-present and one target-absent trial
                    for target_present in [True, False]:
                        trial = SearchTrial(
                            search_condition=search_condition,
                            n_objects=n_objects,
                            trial_num=trial_counter,
                            colors=self.colors,
                            shapes=self.shapes,
                            size=self.size,
                            canvas_size=self.canvas_size,
                            target_present=target_present
                        )
                    
                        img = self.render_trial(trial)
                        
                        # Save image with condition and target presence info in filename
                        filename = f'n={n_objects}_cond={search_condition.value}_trial={trial_counter}_target={target_present}.png'
                        save_path = img_path / filename
                        img.save(save_path)
                        
                        # Collect metadata
                        metadata.append(trial.to_metadata(str(save_path)))
                        trial_counter += 1
        
        return pd.DataFrame(metadata)
