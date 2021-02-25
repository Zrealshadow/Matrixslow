'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2021-02-25 15:01:47
 * @desc 
'''


import matrixslow as ms
from matrixslow.core import Variable
import numpy as np
import matrixslow.optimizer as optim
male_heights = np.random.normal(171, 6, 500)
female_heights = np.random.normal(158, 5, 500)

male_weights = np.random.normal(70, 10, 500)
female_weights = np.random.normal(57, 8, 500)

male_bfrs = np.random.normal(16, 2, 500)
female_bfrs = np.random.normal(22, 2, 500)

male_labels = [1] * 500
female_labels = [-1] * 500

train_set = np.array(
    [
        np.concatenate([male_heights, female_heights]),
        np.concatenate([male_weights, female_weights]),
        np.concatenate([male_bfrs, female_bfrs]),
        np.concatenate([male_labels, female_labels])
    ]
).T

np.random.shuffle(train_set)

x = Variable(dim = (3, 1), init=False, trainable=False)

label = Variable(dim = (1, 1), init=False, trainable=False)

w = Variable(dim = (1, 3), init=True, trainable=True)
b = Variable(dim = (1, 1), init=True, trainable=True)

output = ms.ops.Add(ms.ops.MatMul(w,x), b)
predict = ms.ops.Sigmoid(output)

loss = ms.loss.LogLoss(ms.ops.MatMul(predict,label))

learning_rate = 1e-4

opt = optim.Adam(ms.default_graph, loss, learning_rate=learning_rate)

batch_size = 16

for epoch in range(200):

    batch_count = 0

    for i in range(len(train_set)):

        features = np.mat(train_set[i,:-1]).T

        l = np.mat(train_set[i,-1])

        x.set_value(features)
        label.set_value(l)

        opt.one_step()

        batch_count += 1

        if batch_count == batch_size:
            batch_count = 0
            opt.update()
    
    pred = []

    for i in range(len(train_set)):
        features = np.mat(train_set[i, :-1]).T
        x.set_value(features)
        predict.forward()
        pred.append(predict.value[0,0])
    
    pred = np.where(np.array(pred) > 0.5, 1, -1)
    y = train_set[:,-1]
    accuracy = np.where(y * pred > 0.0 , 1.0, 0.0)
    accuracy = np.sum(accuracy) * 1.00 / accuracy.size

    print("epoch :{:d}, accuracy :{:.3f}".format(epoch+1, accuracy))

