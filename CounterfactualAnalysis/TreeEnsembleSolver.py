import numpy as np
from sklearn import tree
from matplotlib import pyplot as plt
from gurobipy import GRB

from CounterfactualAnalysis.gurobiSolver import CESolver

class CESolver_TreeEnsemble(CESolver):
    def __init__(self, estimator, lambda0, lambda1, lambda2, eps):
        super().__init__(estimator, lambda0, lambda1, lambda2, eps)
        self.T = self.estimator.n_estimators
        self.M1 = 1
        self.M2 = 1

    def initialize_model(self):
        super().initialize_model()
        self.z = self.model.addVars([(t, l) for t in range(self.T) for l in self.getLeaves(t)], vtype=GRB.BINARY)
        self.model.addConstrs((self.xCE[self.getFeature(t, s)] - self.M1*(1-self.z[t,l]) + self.eps[self.features[self.getFeature(t, s)]] <= self.getBias(t, s) for t in range(self.T) for l in self.getLeaves(t) for s in self.getLeftAncestors(t, l)))
        self.model.addConstrs((self.xCE[self.getFeature(t, s)] + self.M2*(1-self.z[t,l]) - self.eps[self.features[self.getFeature(t, s)]] >= self.getBias(t, s) for t in range(self.T) for l in self.getLeaves(t) for s in self.getRightAncestors(t, l)))
        self.model.addConstrs((self.z.sum(t,'*')==1 for t in range(self.T)))

    def getTree(self, t):
        return self.estimator.estimators_[t].tree_

    def getLeaves(self, t):
        return np.where(self.getTree(t).feature<0)[0]

    def getInternalNodes(self, t):
        return np.where(self.getTree(t).feature>=0)[0]

    def getFeature(self, t, s):
        return self.getTree(t).feature[s]

    def getBias(self, t, s):
        return self.getTree(t).threshold[s]

    def getLeftAncestors(self, t, l):
        return self.getPath(t, l)[0]

    def getRightAncestors(self, t, l):
        return self.getPath(t, l)[1]

    def getPath(self, t, l):
        left_path = []
        right_path = []
        node = l
        while node>0:
            left_father = np.where(self.getTree(t).children_left==node)[0]
            right_father = np.where(self.getTree(t).children_right==node)[0]
            if len(left_father)>0:
                left_path.append(left_father[0])
                node = left_father
            if len(right_father)>0:
                right_path.append(right_father[0])
                node = right_father
        return left_path, right_path

    def plot_tree(self, t):
        plt.figure()
        tree.plot_tree(self.getTree(t))
        plt.show()
