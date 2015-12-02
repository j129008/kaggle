import csv
import numpy as np
# from numpy import array
#from sklearn import tree
#from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC

f = open('./train.csv','r')
X = []
y = []

for row in csv.reader(f):
    tmp = map(int, row)
    y.append(tmp.pop(0))
    X.append(tmp)
f.close()

X = np.array(X)
y = np.array(y)

print SVC().fit(X, y).score(X, y)
