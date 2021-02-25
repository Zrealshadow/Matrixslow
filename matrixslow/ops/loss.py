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







