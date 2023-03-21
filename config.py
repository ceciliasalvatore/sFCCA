import os.path

class Config:
    def __init__(self):
        self.data_base_dir = "datasets"  # Directory for datasets
        self.results_base_dir = "results"  # Directory for results

        self.timelimit = 60

        self.seed = 100
        self.k = 5
        self.gosdt_timelimit = 10*60

        self.logger = True

        self.rf_depth = 6
        self.rf_nestimators = 50
        self.dt_depth = 3
        self.p1 = 1
        self.p2 = 1
        self.lambda0 = 0.1
        self.lambda1 = 1

        self.fold = ""
        self.load_thresholds = False

    def set_name(self, name):
        self.name = name
        self.data_dir = f"{self.data_base_dir}/{self.name}.csv"
        self.results_dir = f"{self.results_base_dir}/{self.name}"
        if os.path.exists(self.results_dir)==False:
            os.makedirs(self.results_dir)

    def get_filename(self, type_, ext='txt'):
        return f"{self.results_dir}/{cfg.seed}_{type_}.{ext}"

    def get_filename_fold(self, type_, ext='txt'):
        return f"{self.results_dir}/{cfg.seed}_{type_}_{self.fold}.{ext}"

cfg = Config()