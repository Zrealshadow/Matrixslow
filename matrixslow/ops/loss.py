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
    """
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

    