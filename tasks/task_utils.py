import os
from typing import List
from pathlib import Path
import pandas as pd

class Task:
    '''Base class for ML evaluation tasks that manages data loading and results tracking'''
    
    def __init__(self,
                 task_name=None,
                 model_name=None, 
                 root_dir=None,
                 output_dir=None,
                 data_dir=None,
                 metadata_file=None,
                 prompt_path=None):
        
        # Initialize task configuration
        self.task_name = task_name
        self.model_name = model_name
        self.run_id = self.model_name + '_' + self.task_name
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.data_dir = data_dir
        self.metadata_file = metadata_file
        self.prompt = Path(prompt_path).read_text()

        # Set up output directory structure
        outpath = os.path.join(self.output_dir, self.task_name, self.model_name)
        os.makedirs(outpath, exist_ok=True)
        self.results_path = os.path.join(outpath, 'model_responses.csv')

        # Load or generate task dataset
        task_path = os.path.join(self.data_dir, self.task_name, self.metadata_file)
        if os.path.exists(self.results_path):
            print(f'Loading task metadata from {self.results_path}...')
            self.results_df = pd.read_csv(self.results_path)
        elif os.path.exists(task_path):
            print(f'Loading task metadata from {task_path}...')
            self.results_df = pd.read_csv(task_path)
        else:
            print('Generating full dataset...')
            metadata_path = os.path.join(self.data_dir, self.task_name)
            os.makedirs(metadata_path, exist_ok=True)
            print(metadata_path)
            self.results_df = self.generate_full_dataset()
            self.results_df.to_csv(task_path, index=False)
        return None

    def generate_full_dataset(self):
        """Generate the complete dataset for this task. Must be implemented by subclasses."""
        raise NotImplementedError

    def get_prompt(self, row: pd.Series) -> str:
        """Get the formatted prompt for a specific trial. Must be implemented by subclasses."""
        raise NotImplementedError

    def num_remaining_trials(self):
        """Calculate number of remaining evaluation trials. Must be implemented by subclasses."""
        pass
