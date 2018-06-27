import csv
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# load data
def read_data(path):
    return csv.reader(open(path, 'r'), delimiter=',')

# preprocessing data
train_data = read_data("csv/train.csv")
test_data = read_data("csv/test.csv")

X_train = []
Y_train = []
X_test = []
Y_test = []
flag = 1
for i in train_data:
    if (flag == 1):
        flag = 0
        continue
    temp = map(float,i.split(','))
    X_train.append(temp[:len(temp)-1])	
    Y_train.append(temp[len(temp)-1])
    
for i in test_data:
    if (flag == 0):
        flag = 1
        continue
    X_test.append(map(float,i.split(',')))

# implementing naive bayes model
model = GaussianNB()
model_fit = model.fit(X_train, Y_train)
Y_pred = model_fit.predict(X_train)
Y_test = mode_fit.predict(X_test)

# Generate CSV
train_data = read_data("csv/train.csv")
test_data = read_data("csv/test.csv")

solution = []
solution.append(['PERID', 'Criminal'])

with open("result.csv", "w") as fobj:
    writer = csv.writer(fobj, delimiter=',')
    temp_id = []
    temp_class  = []
    flag = 0
    for i in test_data:
        if (flag == 0):
            flag=1
            continue
        temp_id.append(i[0])
    for i in Y_test:
        temp_class.append(str(i))
    for i in range(len(temp_id)):
        temp = [str(temp_id[i]), str(temp_class[i])]
        solution.append(temp)
    for i in solution:
        writer.writerow(i)