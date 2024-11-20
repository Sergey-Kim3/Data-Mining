import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

A = pd.read_csv("bin-classifier-2.txt", header=None)
X = A.values[:, 0:2]
y = A.values[:, 2]

I = y == 1
# list comprehension
J = [not x for x in I]

clf = LinearDiscriminantAnalysis()

clf.fit(X,y)

print(np.vstack((clf.predict(X), y)).T)

plt.plot(X[I,0],X[I,1],'.')
plt.plot(X[J,0],X[J,1],'.')
plt.grid()
plt.show()

# compute the accuracy, precision, and recall
# for this classifier on the training set
