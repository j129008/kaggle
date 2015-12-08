import csv
import numpy as np
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot

dot_data = StringIO()
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

tree.export_graphviz(clf, out_file=dot_data,
                     feature_names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                     class_names=["0", "1"],
                     filled=True, rounded=True,
                     special_characters=True)

t = open('./test.csv','r')
tX = []
t.readline()
for row in csv.reader(t):
    tX.append(map(int, row)[1:])
t.close()

tX = np.array(tX)
ans = clf.predict(tX)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_png("dm.png")

out = open('ans2.csv','w')
out.write("Id,Action\n")
linum = 0
for ele in ans:
    linum += 1
    out.write( str(linum )+ "," + str(ele)+"\n" )
