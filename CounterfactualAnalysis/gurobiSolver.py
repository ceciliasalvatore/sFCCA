import time
import warnings
import gurobipy as grb
from gurobipy import GRB, quicksum

from config import cfg

class CESolver():
    def __init__(self, estimator, lambda0, lambda1, lambda2, eps):
        self.estimator = estimator
        try:
            self.F = self.estimator.n_features_in_
            self.features = eps.index#self.estimator.feature_names_in_
            self.K = self.estimator.classes_
        except:
            raise Exception('Unfitted estimator')
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.eps = eps
        self.M0 = 1

        self.model = grb.Model()
        self.model.setParam(GRB.Param.TimeLimit, cfg.timelimit)
        self.model.Params.LogToConsole = 0
        self.reset = []

    def initialize_model(self):
        self.xCE = self.model.addVars(self.F, vtype=GRB.CONTINUOUS, lb=0, ub=1)
        self.x_l0 = self.model.addVars(self.F, vtype=GRB.BINARY)
        self.x_l1 = self.model.addVars(self.F, vtype=GRB.CONTINUOUS, lb=0, ub=1)

    def build(self, x0, yCE):
        self.model.reset()
        for c in self.reset:
            for j in c.keys():
                self.model.remove(c[j])
        self.reset = []

        self.model.update()

        self.l0_constraint_a = self.model.addConstrs((x0[j]-self.xCE[j] >= -self.M0*self.x_l0[j] for j in range(self.F)))
        self.l0_constraint_b = self.model.addConstrs((x0[j]-self.xCE[j] <= self.M0*self.x_l0[j] for j in range(self.F)))

        self.l1_constraint_a = self.model.addConstrs((self.x_l1[j] >= (x0[j]-self.xCE[j]) for j in range(self.F)))
        self.l1_constraint_b = self.model.addConstrs((self.x_l1[j] >= -(x0[j]-self.xCE[j]) for j in range(self.F)))

        self.reset.append(self.l0_constraint_a)
        self.reset.append(self.l0_constraint_b)
        self.reset.append(self.l1_constraint_a)
        self.reset.append(self.l1_constraint_b)

        self.model.setObjective((self.lambda0*self.x_l0.sum() + self.lambda1*self.x_l1.sum() + self.lambda2*quicksum((x0[j]-self.xCE[j])**2 for j in range(self.F))), sense=GRB.MINIMIZE)


    def solve(self):
        t0 = time.time()
        self.model.optimize()
        if self.model.getAttr('Status')!=GRB.Status.OPTIMAL:
            warnings.warn(f'Gurobi Optimization completed not to optimality, exit status {self.model.getAttr("Status")}')
        #print(f"Gurobi Optimal Solution {self.model.getAttr('ObjVal')} found in {time.time()-t0} s")
        return self.model.getAttr('X',self.xCE).values()

