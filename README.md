# FCCA

This repository contains the experimental code for the paper <a href="https://arxiv.org/abs/2211.09894">Supervised Feature Compression based on Counterfactual Analysis</a>

## Installation

* The MILP problem for computing the Counterfactual Explanation for a point is implemented in <a href="https://www.gurobi.com/solutions/gurobi-optimizer/?campaignid=18262689303&adgroupid=138243449982&creative=620260718865&keyword=gurobi&matchtype=e&gclid=Cj0KCQiA4OybBhCzARIsAIcfn9mYA1eyslmYMVKkmSzUWuZeLKwpNXdPrcIoKLnEr60zcnHFDSpc5j8aAgzgEALw_wcB">Gurobi</a>.
An active Gurobi Licence is needed to run the code.
* The testing phase also relies on the code for GOSDT trees from the paper <a href='https://proceedings.mlr.press/v119/lin20g.html'>Generalized and Scalable Optimal Sparse Decision Trees</a> and the relative guessing thresholds strategy from the paper <a href='https://ojs.aaai.org/index.php/AAAI/article/view/21194'>Fast Sparse Decision Tree Optimization via Reference Ensembles
</a>. The code can be found at the following link <a href='https://github.com/ubc-systopia/gosdt-guesses'>https://github.com/ubc-systopia/gosdt-guesses</a>.

## Hyperparameters
The implementation of the FCCA procedure can be found in the file *discretize.py* that contains the Python class *FCCA* which takes the following parameters:
* **estimator**: an unfitted binary classifier from the <a href='https://scikit-learn.org/stable/'>sklearn</a> package. It can be one of the following: RandomForestClassifier, GradientBoosting, LinearSVC, SVC(kernel='linear'). It is also possible to take in input GridSearchCV to choose in cross validation the parameters of the estimator;
* **p0**, **p1**: lower and upper bound for the classification probability of points for which computing the Counterfactual Explanation; 
* **lambda0**, **lambda1*, **lambda2**: hyperparameters for the Counterfactual Explanation problem that represents respectively the weights for the l0-, l1- and l2- norm;
* **compress**: boolean that is set to True to merge thresholds whose absolute difference is smaller than 0.01;
* **Q**: value of Q for the FCCA procedure;
* **timelimit**: time limit in seconds for solving the Counterfactual Explanations problem.

The FCCA class offers the following methods:
* **fit**: method for fitting the FCCA procedure;
* **transform**: method for discretizing a dataset by using the set of thresholds previously computed via the **fit** method;
* **fit_transform**: method for applying in sequence the **fit** and **transform** methods;
* **selectThresholds**: method for setting a different value of Q after the **fit** has been called; this method allows to subsample the set of thresholds in a fast way without recomputing the FCCA procedure.

## Execution

The code for reproducing the experiments of the paper can be run through the file *main.py*.

We report an example on how to use the FCCA procedure on new data. The example can also be found in the file *example.py*

```
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from discretize import FCCA

# Reading the dataset
data = pd.read_csv('datasets/boston.csv')
label_column = data.columns[-1]
feature_columns = data.columns[:-1]

# Scaling the features between 0 and 1
scaler = MinMaxScaler()
data[feature_columns] = scaler.fit_transform(data[feature_columns])

# Train - test split
data_ts = data.sample(n=int(0.3*len(data)))
data_tr = data.drop(index=data_ts.index)

x_tr, y_tr = data_tr[feature_columns], data_tr[label_column]
x_ts, y_ts = data_ts[feature_columns], data_ts[label_column]

# Target model
target = GradientBoostingClassifier(max_depth=1, n_estimators=100,learning_rate=0.1)

# Hyperparameters for the discretization - default values
discretizer = FCCA(target, p0=0.5, p1=1, lambda0=0.1, lambda1=1, lambda2=0)

# Discretization
x_tr_discr, y_tr_discr = discretizer.fit_transform(x_tr, y_tr)
x_ts_discr, y_ts_discr = discretizer.transform(x_ts, y_ts)

# Compression - inconsistency rate
print(f'Compression rate: {discretizer.compression_rate(x_ts, y_ts)}')
print(f'Inconsistency rate: {discretizer.inconsistency_rate(x_ts, y_ts)}')

print('Setting Q to 0.7')
# Increasing the value of Q
tao_q = discretizer.selectThresholds(0.7)

# Discretization
x_tr_discr, y_tr_discr = discretizer.transform(x_tr, y_tr, tao_q)
x_ts_discr, y_ts_discr = discretizer.transform(x_ts, y_ts, tao_q)

# Compression - inconsistency rate
print(f'Compression rate: {discretizer.compression_rate(x_ts, y_ts, tao_q)}')
print(f'Inconsistency rate: {discretizer.inconsistency_rate(x_ts, y_ts, tao_q)}')

```
