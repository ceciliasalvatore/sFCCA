import os.path

class Config:
    def __init__(self):
        self.data_base_dir = "datasets"  # Directory for datasets
        self.results_base_dir = "results"  # Directory for results

        self.timelimit = 60

        self.seed = 100
        self.k = 3

    def set_name(self, name):
        self.name = name
        self.data_dir = f"{self.data_base_dir}/{self.name}.csv"
        self.results_dir = f"{self.results_base_dir}/{self.name}"
        if os.path.exists(self.results_dir)==False:
            os.makedirs(self.results_dir)

    def get_filename(self, type_, ext='csv'):
        return f"{self.results_dir}/{cfg.seed}_{type_}.{ext}"

cfg = Config()