from gurobipy import quicksum

from CounterfactualAnalysis.TreeEnsembleSolver import CESolver_TreeEnsemble

class CESolver_GradientBoosting(CESolver_TreeEnsemble):
    def __init__(self, estimator, lambda0, lambda1, lambda2, eps):
        super().__init__(estimator, lambda0, lambda1, lambda2, eps)
        self.T = self.estimator.n_estimators
        self.M1 = 1
        self.M2 = 1

    def build(self, x0, yCE):
        super().build(x0, yCE)
        lhs = self.getInitialPrediction(x0)+quicksum(self.getLearningRate()*self.getPrediction(t,l)*self.z[t,l] for t in range(self.T) for l in self.getLeaves(t))
        if yCE == 1:
            self.class_assignment = self.model.addConstr(lhs >= 0+1.e-4)
        elif yCE == 0 or yCE == -1:
            self.class_assignment = self.model.addConstr(lhs <= 0-1.e-4)
        else:
            raise ValueError(f'Unknown value yCE = {yCE}')
        self.reset.append(self.class_assignment)

    def getTree(self, t):
        return self.estimator.estimators_[t,0].tree_

    def getPrediction(self, t, l):
        return self.getTree(t).value[l,0,0]

    def getInitialPrediction(self, x0):
        return self.estimator._raw_predict_init(x0.to_numpy().reshape(1,-1))[0, 0]

    def getLearningRate(self):
        return self.estimator.learning_rate