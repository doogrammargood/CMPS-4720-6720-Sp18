import scipy
import csv
import numpy as np
from sklearn import svm

with open('iris.csv', 'r') as f:
    data = list(csv.reader(f))
data = np.random.permutation(data) #The data comes sorted by petal type.
train_data = data[0:100]
test_data = data[100:]

train_features = [t[0:4] for t in train_data]
train_labels = [t[4] for t in train_data]

clf = svm.SVC()
clf.fit(train_features, train_labels)


predictions = clf.predict([t[0:4] for t in test_data])
outcomes = [t[4] for t in test_data]

correct_answers = reduce(lambda x, y: x+y, map(lambda x: 1 if x[0] == x[1] else 0, zip(predictions, outcomes)))
total_answers = len(test_data)
print float(correct_answers)/total_answers
