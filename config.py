import os.path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC

class Config:
    def __init__(self):
        self.data_base_dir = "datasets"  # Directory for datasets

        self.seed = 100
        self.k = 5
        self.gosdt_timelimit = 10*60

        self.logger = True

        self.target_model = GradientBoostingClassifier
        #self.target_model = RandomForestClassifier
        #self.target_model = LinearSVC
        self.target_cross_validation = False
        if self.target_model == RandomForestClassifier or self.target_model == LinearSVC:
            self.target_cross_validation = True

        self.target_depth = 1
        self.target_nestimators = 100
        if self.target_model == RandomForestClassifier:
            self.target_nestimators = 50
        self.target_C = 1

        self.dt_depth = 5
        self.regularization_factor = 1
        self.train_size = None

        self.p1 = 1
        self.p2 = 1

        self.lambda0 = 0.1
        self.lambda1 = 1
        self.lambda2 = 0.0

        self.results_base_dir = f"results_{self.dt_depth}"  # Directory for results

        self.fold = ""

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