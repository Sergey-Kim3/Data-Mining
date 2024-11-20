import numpy as np
import pandas as pd

tt = pd.read_csv('bin-classifier-2.txt', header=None)
X = tt.values[:, 0:2]
y = tt.values[:, 2]

I = y == 1
# list comprehension
J = [not x for x in I]

# selection with array of booleans
print(X[I,:])
print(X[J,:])

# computing the mean vectors
m1 = np.mean(X[I,:],axis=0)
m2 = np.mean(X[J,:],axis=0)

# find a linear classifier 'w' according
# to the LDA recipe from the slides
# Also, find the best value for the threshold
