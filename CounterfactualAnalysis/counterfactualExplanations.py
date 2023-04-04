import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC, SVC

from CounterfactualAnalysis.RandomForestSolver import CESolver_RandomForest
from CounterfactualAnalysis.SVCSolver import CESolver_SVC
from CounterfactualAnalysis.GradientBoostingSolver import CESolver_GradientBoosting
from config import cfg

class CounterfactualExplanation():
    def __init__(self, estimator, lambda0, lambda1, lambda2, eps):
        self.estimator = estimator
        self.eps = eps

        self.counterfactual_labels = {0:1, 1:0}

        if isinstance(self.estimator, RandomForestClassifier):
            self.solver = CESolver_RandomForest(self.estimator, lambda0, lambda1, lambda2, eps)
        elif isinstance(self.estimator, GradientBoostingClassifier):
            self.solver = CESolver_GradientBoosting(self.estimator, lambda0, lambda1, lambda2, eps)
        elif isinstance(self.estimator, LinearSVC) or isinstance(self.estimator, SVC):
            self.solver = CESolver_SVC(self.estimator, lambda0, lambda1, lambda2, eps)
        else:
            raise ModuleNotFoundError(f"CounterfactualExplanation solver not implemented for estimator {self.estimator.__class__}")

        self.solver.initialize_model()

    def compute(self, x0, y0):
        yCE = y0.copy()
        xCE = x0.copy()
        t0 = time.time()
        for i in range(len(x0)):
            #print(f"Computing Counterfactual Explanation for point {i} out of {len(x0)}")
            t_i = time.time()
            yCE_i = self.counterfactual_labels[y0.iloc[i]]
            self.solver.build(x0.iloc[i], yCE_i)
            xCE_i = self.solver.solve()
            xCE.iloc[i] = xCE_i
            yCE.iloc[i] = yCE_i
            if cfg.logger:
                print(f"{np.max(self.estimator.predict_proba(x0.iloc[i:i+1]))}, {time.time()-t_i}", file=open(cfg.get_filename('counterfactuals_time'),mode='a'))

        if cfg.logger:
            print(f"{len(x0)} Counterfactual Explanations computed in {time.time()-t0} s", file=open(cfg.get_filename('logger'),mode='a'))

        return xCE, yCE




