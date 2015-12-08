import csv
import numpy as np
from sklearn import tree

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

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

t = open('./test.csv','r')
tX = []
t.readline()
for row in csv.reader(t):
    tX.append(map(int, row)[1:])
t.close()

tX = np.array(tX)
ans = clf.predict(tX)

out = open('ans2.csv','w')
out.write("Id,Action\n")
linum = 0
for ele in ans:
    linum += 1
    out.write( str(linum )+ "," + str(ele)+"\n" )
