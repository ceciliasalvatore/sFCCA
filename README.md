# FCCA

This repository contains the experimental code for the paper <a href="https://arxiv.org/abs/2211.09894">Supervised Feature Compression based on Counterfactual Analysis</a>

## Installation

* The MILP problem for computing the Counterfactual Explanation for a point is implemented in <a href="https://www.gurobi.com/solutions/gurobi-optimizer/?campaignid=18262689303&adgroupid=138243449982&creative=620260718865&keyword=gurobi&matchtype=e&gclid=Cj0KCQiA4OybBhCzARIsAIcfn9mYA1eyslmYMVKkmSzUWuZeLKwpNXdPrcIoKLnEr60zcnHFDSpc5j8aAgzgEALw_wcB">Gurobi</a>.
An active Gurobi Licence is needed to run the code.
* The testing phase also relies on the code for GOSDT trees from the paper <a href='https://proceedings.mlr.press/v119/lin20g.html'>Generalized and Scalable Optimal Sparse Decision Trees</a> and the relative guessing thresholds strategy from the paper <a href='https://ojs.aaai.org/index.php/AAAI/article/view/21194'>Fast Sparse Decision Tree Optimization via Reference Ensembles
</a>. The code can be found at the following link <a href='https://github.com/ubc-systopia/gosdt-guesses'>https://github.com/ubc-systopia/gosdt-guesses</a>.

## Execution

The code for reproducing the experiments of the paper can be run through the file main.py.

For using the FCCA procedure on new data, the following snippet of code can be useful:

```
from discretize import FCCA


```
