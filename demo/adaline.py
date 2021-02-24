'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2021-02-23 16:17:13
 * @desc 
'''

import matrixslow as ms
from matrixslow.core import Variable
import numpy as np

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
# import pdb;pdb.set_trace()
x = Variable(dim = (3, 1), init=False, trainable=False)

label = Variable(dim = (1,1), init=False, trainable=False)

w = Variable(dim = (1, 3), init=True, trainable=True)
b = Variable(dim = (1, 1), init=True, trainable=True)

output1 = ms.ops.MatMul(w,x)
output2 = ms.ops.Add(output1,b)
output3 = ms.ops.MatMul(output2,label)
loss = ms.ops.PerceptionLoss(output3)

learning_rate = 1e-4
# ms.default_graph.draw(filepath='.')
for epoch in range(50):

    for i in range(len(train_set)):

        features = np.mat(train_set[i,:-1]).T 

        l = np.mat(train_set[i, -1])

        x.set_value(features)
        label.set_value(l)

        loss.forward()

        w.backward(loss)
        b.backward(loss)

        """update the parameter's value 
        Using Gradient Descent
        """
        # import pdb;pdb.set_trace()
        w.set_value(w.value - learning_rate * w.jacobi.T.reshape(w.shape()))
        b.set_value(b.value - learning_rate * b.jacobi.T.reshape(b.shape()))

        ms.default_graph.clear_jacobi()

    
    # evaluate the accuracy

    pred = []

    for i in range(len(train_set)):
        features = np.mat(train_set[i, :-1]).T
        x.set_value(features)
        output2.forward()
        pred.append(output2.value[0,0])
    
    pred = np.array(pred)
    y = train_set[:,-1]
    accuracy = np.where(y * pred > 0.0 , 1.0, 0.0)
    accuracy = np.sum(accuracy)* 1.00 / accuracy.size

    print("epoch :{:d}, accuracy :{:.3f}".format(epoch+1, accuracy))

