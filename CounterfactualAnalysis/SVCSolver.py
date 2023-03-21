from sklearn.svm import LinearSVC, SVC
from CounterfactualAnalysis.gurobiSolver import CESolver

class CESolver_SVC(CESolver):
    def __init__(self, estimator, lambda0, lambda1, lambda2, eps):
        if isinstance(estimator,SVC) and estimator.kernel!='linear':
            raise ModuleNotFoundError(f"CounterfactualExplanation solver not implemented for estimator {estimator.__class__} with non-linear kernel")
        super().__init__(estimator, lambda0, lambda1, lambda2, eps)


    def initialize_model(self):
        super().initialize_model()
        raise NotImplementedError


    def build(self, x0, yCE):
        return super().build(x0, yCE)
        raise NotImplementedError