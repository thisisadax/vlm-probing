import pandas as pd
from tasks.task_utils import Task
from os.path import join

class PixmoNumerosityTask(Task):
    '''Task class for the Pixmo numerosity estimation dataset'''
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def get_prompt(self, row: pd.Series) -> str:
        '''Format the prompt template with the object name for this trial.'''
        try:
            return self.prompt.format(object=row.object)
        except KeyError as e:
            print(f'Error formatting prompt. Row contents: {row.to_dict()}')
            print(f'Prompt template: {self.prompt}')
            raise

    def generate_full_dataset(self) -> pd.DataFrame:
        '''Load the existing metadata CSV file.'''
        return pd.read_csv(join(self.output_dir, self.task_name, self.metadata_file))
