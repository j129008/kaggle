import csv
import numpy as np
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

clf = SVC().fit(X, y)

t = open('./test.csv','r')
tX = []
t.readline()
for row in csv.reader(t):
    tX.append(map(int, row)[1:])
t.close()

tX = np.array(tX)
ans = clf.predict(tX)

linum = 0
for ele in ans:
    linum += 1
    print str(linum )+ "," + str(ele)
