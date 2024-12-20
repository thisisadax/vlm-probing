import time
from tasks.task_utils import Task


class Model():

    def __init__(
            self,
            task: Task
    ):
        self.task = task

    def run(self):
        print('Need to specify a particular model class.')
        raise NotImplementedError
    
    def save_results(self, results_file: str=None):
        if results_file:
            self.task.results_df.to_csv(results_file, index=False)
        else:
            filename = f'results_{time.time()}.csv'
            self.task.results_df.to_csv(filename, index=False)