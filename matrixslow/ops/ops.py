from matrixslow.core.node import Node
import numpy as np
from numpy.matrixlib.defmatrix import matrix as npmat
from typing import Optional, Union

class Operator(Node):
    """ReName the Base Class for Identification"""
    pass


def fill_diagonal(to_be_filled:npmat, filter:Union[npmat,int]):
    assert to_be_filled.shape[0] / filter.shape[0] == to_be_filled.shape[1] / filter.shape[1] ,\
        "Error in size of two matrix"
    n = int(to_be_filled.shape[0] / filter.shape[0])
    r, c =filter.shape
    for i in range(n):
        to_be_filled[i * r : (i+1) * r, i * c : (i+1) * c] = filter
    
        

class MatMul(Operator):
    """ Multiplication of Matrix"""

    def compute(self) -> None:
        """the multiplication of two matrix"""
        assert len(self.parents) == 2 , "Parents node is not 2, Error in Computing Graph,"
        assert self.parents[0].shape()[-1] == self.parents[-1].shape()[0] , "the shape is error in matrix multiplication"
        self.value = self.parents[0].value * self.parents[1].value

    def get_jacobi(self, parent: 'Node') -> npmat:
        zeros = np.mat(np.zeros((self.dimension(), parent.dimension())))
        # for the first element in multiplication of matrix
        if parent == self.parent[0]:
            return fill_diagonal(zeros, self.parents[1].value.T)
        else:
            pass
            
        return super().get_jacobi(parent)
    
    
