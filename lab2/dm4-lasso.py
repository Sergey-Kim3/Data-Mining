import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd

A = pd.read_csv("lasso_example.txt",header=None)

x=A.values[:,1]
X=A.values[:,0:4]
y=A.values[:,4]

lasmodel = linear_model.Lasso(alpha=1e-5,fit_intercept=False)
lasmodel.fit(X,y)

print(X.head())
print(y.head())

print(lasmodel.coef_)

### 
