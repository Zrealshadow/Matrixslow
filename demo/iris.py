import numpy as np
import pandas as pd
import matrixslow as ms
from sklearn.preprocessing import LabelEncoder , OneHotEncoder

import os
pwd = os.getcwd()

iris_path = os.path.join(pwd, 'data', 'iris.csv')

iris_data = pd.read_csv(iris_path)
data = iris_data.sample(len(iris_data), replace=False)

le = LabelEncoder()
number_label = le.fit_transform(data["species"])
oh = OneHotEncoder(sparse=False)
label = np.mat(oh.fit_transform(number_label.reshape(-1,1))).astype(np.float)

feature = np.mat(np.array(data)[:,:-1]).astype(np.float)

x = ms.core.Variable((4,1), init=False, trainable=False)
y = ms.core.Variable((3,1), init=False, trainable=False)
w = ms.core.Variable((3,4), init=True, trainable=True)
b = ms.core.Variable((3,1), init=True, trainable=True)
linear = ms.ops.Add(ms.ops.MatMul(w,x), b)
predict = ms.ops.SoftMax(linear)
loss = ms.loss.CrossEntropyWithSoftMax(linear, y)



learning_rate = 0.02
optim = ms.optimizer.Adam(ms.default_graph, loss, learning_rate=learning_rate)
batch_size = 16

for epoch in range(200):
    batch_count = 0
    for i in range(len(feature)):
        f = feature[i,:].T
        l = label[i,:].T
        
        x.set_value(f)
        y.set_value(l)
        # import pdb;pdb.set_trace()
        optim.one_step()
        batch_count += 1

        if batch_count >= batch_size:
            optim.update()
            batch_count = 0
    pred = []

    for i in range(len(feature)):
        f = feature[i,:].T
        x.set_value(f)
        predict.forward()
        pred.append(predict.value.A1)
    # import pdb;pdb.set_trace()
    pred = np.array(pred).argmax(axis = 1)
    accuracy = (number_label == pred).sum()/ len(data)

    print("epoch:{:d} accuracy: {:3f}".format(epoch + 1, accuracy))