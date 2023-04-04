import numpy as np
from sklearn import tree
from matplotlib import pyplot as plt
from gurobipy import GRB, quicksum

from CounterfactualAnalysis.TreeEnsembleSolver import CESolver_TreeEnsemble

class CESolver_RandomForest(CESolver_TreeEnsemble):
    def __init__(self, estimator, lambda0, lambda1, lambda2, eps):
        super().__init__(estimator, lambda0, lambda1, lambda2, eps)
        self.T = self.estimator.n_estimators
        self.M1 = 1
        self.M2 = 1

    def build(self, x0, yCE):
        super().build(x0, yCE)
        self.class_assignment = self.model.addConstrs((quicksum(self.getWeight(t,l,yCE)*self.z[t,l] for t in range(self.T) for l in self.getLeaves(t)) >= 1.e-4 + quicksum(self.getWeight(t,l,k)*self.z[t,l] for t in range(self.T) for l in self.getLeaves(t)) for k in self.K if k!=yCE))
        self.reset.append(self.class_assignment)

    def getWeight(self, t, l, k):
        value = self.getTree(t).value[l,0,:]
        return (1/self.T*(value[np.where(self.K==k)[0]]/np.sum(value)))[0]
