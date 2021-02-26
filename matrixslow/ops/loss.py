'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2021-02-22 09:59:10
 * @desc 
'''
import numpy as np
from numpy.matrixlib.defmatrix import matrix as npmat
from matrixslow.core.node import Node

class LossFunction(Node):
    pass


class PerceptionLoss(LossFunction):
    """Perception Loss Function
    if x > 0 else loss += 0 
    if x <= 0  else loss += -x
    """

    def compute(self):
        assert len(self.parents) == 1, "Error in the number of parents"
        p = self.parents[0]
        self.value = np.mat(np.where(
            p.value >= 0.0 , 0.0, - p.value
        ))
        return super().compute()

    def get_jacobi(self, parent: 'Node'):
        diag = np.where(parent.value >= 0.0, 0.0, -1)
        return np.mat(np.diag(diag.ravel()))


class LogLoss(LossFunction):
    def compute(self):
        """
        loss = log(1 + e^{-x})
        """
        assert len(self.parents) == 1, "only allow one parents"
        v = self.parents[0].value
        self.value = np.log(1 + np.exp(-1.0 * v))
    
    def get_jacobi(self, parent: 'Node') -> npmat:
        v = self.parents[0].value
        diag = - 1.0 / (1 + np.exp(v))
        return np.mat(np.diag(diag.ravel()))




class CrossEntropy(LossFunction):
    def compute(self):
        """
        CrossEntropy = sum { label_i * log(p_i) }
        the first parent node is predicted result
        the second parent node is true labels
        """
        assert len(self.parents) == 2, "Error in the number of parents"
        assert self.parents[0].shape ==  self.parents[1].shape , "Error in parents'node value shape"
        p = self.parents[0].value
        l = self.parents[1].value
        self.value = np.mat(np.sum(np.multiply(l,np.log(p + 1e-10))))
        
    def get_jacobi(self, parent: 'Node') -> npmat:
        p = self.parents[0].value
        l = self.parents[1].value
        if parent == self.parents[0]:
            j = -1 * np.multiply(l, 1.0 / p).A1
            return np.mat(j)
        else:
            # there is no need to calculate 
            raise np.mat((-np.log(p)).T)

class CrossEntropyWithSoftMax(LossFunction):
    """Merge CrossEntropy and SoftMax into one operation node
    """
    def softmax(self) -> npmat:
        a = self.parents[0].value
        MAX_VALUE = 1e2
        max_x = np.max(a)
        if max_x > MAX_VALUE:
            a = a / max_x * MAX_VALUE
        s = np.sum(np.exp(a))
        p = np.exp(a) / s

        return p
    def compute(self):
        """
        """
        assert len(self.parents) == 2
        # first parents is predicted node
        # second parents is true label
        l = self.parents[1].value
        p = self.softmax()
        self.value = np.mat(np.sum(np.multiply(l,np.log(p))))
            
    def get_jacobi(self, parent: 'Node') -> npmat:
        p = self.softmax()
        l = self.parents[1].value
        jacobi = np.mat(np.zeros((self.dimension(), parent.dimension())))

        if parent is self.parents[0]:
            return (p - l).T
        else:
            return (-np.log(p)).T



