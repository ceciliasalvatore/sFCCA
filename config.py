import os.path

class Config:
    def __init__(self):
        self.data_base_dir = "datasets"  # Directory for datasets
        self.results_base_dir = "results"  # Directory for results

        self.timelimit = 1*60

        self.seed = 100
        self.k = 5
        self.gosdt_timelimit = 10*60

        self.logger = True

        self.target_depth = 4
        self.target_nestimators = 100

        self.dt_depth = 3
        self.regularization_factor = 1
        self.train_size = None
        self.Q = 0.6

        self.p1 = 1
        self.p2 = 1
        self.lambda0 = 0.1
        self.lambda1 = 1

        self.q_list = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99]

        self.fold = ""
        self.load_thresholds = True

    def set_name(self, name):
        self.name = name
        self.data_dir = f"{self.data_base_dir}/{self.name}.csv"
        self.results_dir = f"{self.results_base_dir}/{self.name}"
        if os.path.exists(self.results_dir)==False:
            os.makedirs(self.results_dir)

        if self.name == 'magic' or self.name == 'particle' or self.name == 'vehicle':
            self.train_size = 5000
            self.p1 = 0.7

    def get_filename(self, type_, ext='txt'):
        return f"{self.results_dir}/{cfg.seed}_{type_}.{ext}"

    def get_filename_fold(self, type_, ext='txt'):
        return f"{self.results_dir}/{cfg.seed}_{type_}_{self.fold}.{ext}"

cfg = Config()