import csv
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

f = open('./train.csv','r')
X = []
y = []

f.readline()
for row in csv.reader(f):
    tmp = map(int, row)
    y.append(tmp.pop(0))
    X.append(tmp)
f.close()

X = np.array(X)
y = np.array(y)

tuned_parameters = [{'kernel' : ['rbf'], 'gamma': [0.1, 1e-2, 1e-3], 'C': [10, 100, 1000]},
                    {'kernel' : ['poly'], 'degree' : [5, 9], 'C' : [1, 10]}]

clf = GridSearchCV( SVC(), tuned_parameters, cv=3, verbose=2, n_jobs=2 ).fit(X, y)

t = open('./test.csv','r')
tX = []
t.readline()
for row in csv.reader(t):
    tX.append(map(int, row)[1:])
t.close()

tX = np.array(tX)
ans = clf.predict(tX)

out = open('ans2.csv','w')
linum = 0
for ele in ans:
    linum += 1
    out.write( str(linum )+ "," + str(ele)+"\n" )
