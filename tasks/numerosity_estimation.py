from enum import Enum
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from typing import List, Tuple, Optional
from dataclasses import dataclass

from tasks.task_utils import Task

class TrialType(Enum):
    CONSTANT_AREA = 'constant_area'
    RANDOM_SIZE = 'random_size'

@dataclass
class Dot:
    x: int
    y: int
    radius: int

class DotSizeStrategy:
    '''Base class for different dot size generation strategies'''
    def get_radii(self, n_dots: int) -> List[int]:
        raise NotImplementedError

class ConstantAreaStrategy(DotSizeStrategy):
    def __init__(self, total_area: int):
        self.total_area = total_area
    def get_radii(self, n_dots: int) -> List[int]:
        area_per_dot = self.total_area / n_dots
        radius = int(np.sqrt(area_per_dot / np.pi))
        return [max(radius, 1)] * n_dots

class RandomSizeStrategy(DotSizeStrategy):
    def __init__(self, total_area: int, max_dots: int, min_radius: int = 10):
        self.total_area = total_area
        self.min_radius = min_radius
        
        # Calculate max_radius to achieve roughly total_area with max_dots
        # total_area ≈ max_dots * π * ((min_r + max_r)/2)²
        target_avg_area = total_area / max_dots
        target_avg_radius = np.sqrt(target_avg_area / np.pi)
        self.max_radius = int(2 * target_avg_radius - min_radius)
        
    def get_radii(self, n_dots: int) -> List[int]:
        return [np.random.randint(self.min_radius, self.max_radius + 1) 
                for _ in range(n_dots)]

class NumerosityTask(Task):
    def __init__(
        self,
        min_dots: int,
        max_dots: int,
        n_trials: int,
        total_area: int = 40000,  # Default total area in pixels
        min_radius: int = 10,
        canvas_size: Tuple[int, int] = (512, 512),
        **kwargs
    ):
        self.min_dots = min_dots
        self.max_dots = max_dots
        self.n_trials = n_trials
        self.total_area = total_area
        self.canvas_size = canvas_size
        self.dots_range = range(min_dots, max_dots + 1)
        
        # Initialize size strategies
        self.size_strategies = {
            TrialType.CONSTANT_AREA: ConstantAreaStrategy(total_area),
            TrialType.RANDOM_SIZE: RandomSizeStrategy(total_area, max_dots, min_radius)
        }
        
        super().__init__(**kwargs)

    def generate_full_dataset(self) -> pd.DataFrame:
        '''Generate dataset of images with both constant area and random size trials.'''
        img_path = Path(self.data_dir) / self.task_name / 'images'
        img_path.mkdir(parents=True, exist_ok=True)
        
        metadata = []
        for n_dots in self.dots_range:
            for trial_type in TrialType:
                radii = self.size_strategies[trial_type].get_radii(n_dots)
                
                for trial in range(self.n_trials):
                    dots = self._generate_dots(n_dots, radii)
                    if dots is None:  # Failed to place dots
                        continue
                        
                    img = self._create_image(dots)
                    
                    filename = f'dots={n_dots}_type={trial_type.value}_trial={trial}.png'
                    save_path = img_path / filename
                    img.save(save_path)
                    
                    total_area = sum(np.pi * dot.radius ** 2 for dot in dots)
                    metadata.append({
                        'path': str(save_path),
                        'n_dots': n_dots,
                        'trial_type': trial_type.value,
                        'trial': trial,
                        'radii': radii,
                        'actual_total_area': total_area,
                        'dots_data': [vars(dot) for dot in dots]
                    })
        
        return pd.DataFrame(metadata)

    def _generate_dots(self, n_dots: int, radii: List[int], max_attempts: int = 1000) -> Optional[List[Dot]]:
        '''Generate list of non-overlapping dots with specified radii.'''
        dots = []
        overall_attempts = 0
        
        for radius in radii:
            attempts = 0
            placed = False
            
            while attempts < max_attempts and not placed:
                x = np.random.randint(radius, self.canvas_size[0] - radius)
                y = np.random.randint(radius, self.canvas_size[1] - radius)
                
                new_dot = Dot(x, y, radius)
                
                if not self._check_overlap(new_dot, dots):
                    dots.append(new_dot)
                    placed = True
                
                attempts += 1
                overall_attempts += 1
                
                if overall_attempts >= max_attempts:
                    print(f"Warning: Failed to place all dots after {max_attempts} total attempts")
                    return None
                    
            if not placed:
                print(f"Warning: Could not place dot with radius {radius}")
                return None
                
        return dots

    def _check_overlap(self, new_dot: Dot, existing_dots: List[Dot]) -> bool:
        '''Check if new dot overlaps with any existing dots.'''
        for dot in existing_dots:
            distance = np.sqrt((new_dot.x - dot.x)**2 + (new_dot.y - dot.y)**2)
            if distance < (new_dot.radius + dot.radius):
                return True
        return False

    def _create_image(self, dots: List[Dot]) -> Image.Image:
        '''Create image with specified dots.'''
        img = Image.new('RGB', self.canvas_size, 'white')
        draw = ImageDraw.Draw(img)
        
        for dot in dots:
            bbox = [
                (dot.x - dot.radius, dot.y - dot.radius),
                (dot.x + dot.radius, dot.y + dot.radius)
            ]
            draw.ellipse(bbox, fill='black')
            
        return img