
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import OneHotEncoder
from matrixslow.layer import fc
import matrixslow as ms
import pandas as pd
import os
pwd = os.getcwd()
data_path = os.path.join(pwd, 'data', 'mnist_784.csv')
mnist_data = np.array(pd.read_csv(data_path))

X = mnist_data[:,:-1]
label = mnist_data[:,-1]
X = X[:5000] / 255
label = label[:5000].astype(np.int)

oh = OneHotEncoder(sparse=False)
one_hot_label = oh.fit_transform(label.reshape(-1, 1))
x = ms.core.Variable(dim= (784, 1), init=False, trainable=False)
one_hot = ms.core.Variable(dim = (10 , 1), init=False, trainable=False)

hidden_layer_size_1 = 100
hidden_layer_size_2 = 20
hidden_1 = fc(x, x.dim[0] ,hidden_layer_size_1, "ReLU")
hidden_2 = fc(hidden_1, hidden_layer_size_1, hidden_layer_size_2, "ReLU")
output = fc(hidden_2, hidden_layer_size_2, one_hot_label.shape[-1], None)
predict = ms.ops.SoftMax(output)
loss = ms.ops.loss.CrossEntropyWithSoftMax(output, one_hot)

optim = ms.optimizer.Adam(ms.default_graph, loss, learning_rate=1e-4)
batch_size = 64
# ms.default_graph.draw(filepath=".")
for epoch in range(30):

    batch_count = 0

    for i in range(len(X)):
        feature = np.mat(X[i]).T
        y = np.mat(one_hot_label[0]).T
        x.set_value(feature)
        # import pdb;pdb.set_trace()
        one_hot.set_value(y)

        optim.one_step()
        batch_count += 1

        if batch_count >= batch_size:
            # import pdb;pdb.set_trace()
            print("epoch :{:d} , iteration: {:d}, loss:{:.3f}".format(
                epoch + 1, i + 1, loss.value[0,0]
            ))
            optim.update()
            batch_count = 0
    pred = []

    for i in range(len(X)):
        feature = np.mat(X[i]).T
        x.set_value(feature)
        predict.forward()
        pred.append(predict.value.ravel())
    pred = np.array(pred).argmax(axis = 1)
    accuracy = (y == pred).sum() * 1.00 / len(x)
    print("epoch : {:d}, accuracy: {:3f}".format(epoch+1, accuracy))