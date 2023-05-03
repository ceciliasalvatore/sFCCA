from sklearn.svm import SVC
from gurobipy import quicksum
from CounterfactualAnalysis.gurobiSolver import CESolver

class CESolver_SVC(CESolver):
    def __init__(self, estimator, lambda0, lambda1, lambda2, eps, timelimit):
        if isinstance(estimator,SVC) and estimator.kernel!='linear':
            raise ModuleNotFoundError(f"CounterfactualExplanation solver not implemented for estimator {estimator.__class__} with non-linear kernel")
        super().__init__(estimator, lambda0, lambda1, lambda2, eps, timelimit)


    def initialize_model(self):
        super().initialize_model()


    def build(self, x0, yCE):

        super().build(x0, yCE)
        lhs = quicksum(self.getW()[i]*self.xCE[i] for i in range(len(self.xCE))) + self.getB()
        if yCE == 0:
            self.class_assignment = self.model.addConstr(lhs<=-1)
        elif yCE == 1:
            self.class_assignment = self.model.addConstr(lhs>=1)
        else:
            raise ValueError(f'Unknown Counterfactual Label {yCE}')
        self.reset.append(self.class_assignment)

    def getW(self):
        return self.estimator.coef_[0,:]

    def getB(self):
        if self.estimator.fit_intercept:
            return self.estimator.intercept_[0]
        else:
            return 0.0