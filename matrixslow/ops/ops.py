from matrixslow.core.node import Node
import numpy as np
from numpy.matrixlib.defmatrix import matrix as npmat
from typing import Optional, Union

class Operator(Node):
    """ReName the Base Class for Identification"""
    pass


def fill_diagonal(to_be_filled:npmat, filter:npmat) -> npmat:
    """Fill one matrix with a filter matrix in diagonal way

    Parameters
    ----------
    to_be_filled : npmat
        filled matrix
    filter : npmat
        filter matrix

    Returns
    -------
    npmat
        new filled matrix
    """    
    assert to_be_filled.shape[0] / filter.shape[0] == to_be_filled.shape[1] / filter.shape[1] ,\
        "Error in size of two matrix"
    n = int(to_be_filled.shape[0] / filter.shape[0])
    r, c =filter.shape
    for i in range(n):
        to_be_filled[i * r : (i+1) * r, i * c : (i+1) * c] = filter
    return to_be_filled
        

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
            jacobi = fill_diagonal(zeros, self.parents[0].value)
            row_sort = np.arange(self.dimension()).reshape(self.shape()[::-1]).T.ravel()
            col_sort = np.arange(parent.dimension()).reshape(parent.shape()[::-1]).Y.ravel()
            return jacobi[row_sort:][:col_sort]
        
        """
        A = [
            [a11, a12, a13, a14]
            [a21, a22, a23, a24]
            [a31, a32, a33, a34]
            ]
        A * B = C   A 3*4 B 4*3 C 3*3  jocabi 9 * 12
        jacobi = 
        [
            a11, a12, a13, a14, |0  , 0  , 0  , 0  ,|0  , 0  , 0  , 0  
            a21, a22, a23, a24, |0  , 0  , 0  , 0  ,|0  , 0  , 0  , 0  
            a31, a32, a33, a34, |0  , 0  , 0  , 0  ,|0  , 0  , 0  , 0  
            0  , 0  , 0  , 0  , |a11, a12, a13, a14,|0  , 0  , 0  , 0  
            0  , 0  , 0  , 0  , |a21, a22, a23, a24,|0  , 0  , 0  , 0  
            0  , 0  , 0  , 0  , |a31, a32, a33, a34,|0  , 0  , 0  , 0 
            0  , 0  , 0  , 0  , |0  , 0  , 0  , 0  ,|a11, a12, a13, a14 
            0  , 0  , 0  , 0  , |0  , 0  , 0  , 0  ,|a21, a22, a23, a24
            0  , 0  , 0  , 0  , |0  , 0  , 0  , 0  ,|a31, a32, a33, a34 
        ]
        change the sort of column and row
        row_sort = [0, 3, 6, 1, 4, 7, 2, 5, 8]
        jacobi = 
        [
            a11, a12, a13, a14, |0  , 0  , 0  , 0  ,|0  , 0  , 0  , 0 
            0  , 0  , 0  , 0  , |a11, a12, a13, a14,|0  , 0  , 0  , 0
            0  , 0  , 0  , 0  , |0  , 0  , 0  , 0  ,|a11, a12, a13, a14 
            a21, a22, a23, a24, |0  , 0  , 0  , 0  ,|0  , 0  , 0  , 0 
            0  , 0  , 0  , 0  , |a21, a22, a23, a24,|0  , 0  , 0  , 0
            0  , 0  , 0  , 0  , |0  , 0  , 0  , 0  ,|a21, a22, a23, a24
            a31, a32, a33, a34, |0  , 0  , 0  , 0  ,|0  , 0  , 0  , 0 
            0  , 0  , 0  , 0  , |a31, a32, a33, a34,|0  , 0  , 0  , 0 
            0  , 0  , 0  , 0  , |0  , 0  , 0  , 0  ,|a31, a32, a33, a34
        ]

        col_sort = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
        jacobi = [
            a11 ,  0  ,  0  , a12 ,  0  ,  0  , a13 ,  0  ,  0  , a14 ,  0  ,  0
             0  , a11 ,  0  ,  0  , a12 ,  0  ,  0  , a13 ,  0  ,  0  , a14 ,  0
             0  ,  0  , a11 ,  0  ,  0  , a12 ,  0  ,  0  , a13 ,  0  ,  0  , a14
            a21 ,  0  ,  0  , a22 ,  0  ,  0  , a23 ,  0  ,  0  , a24 ,  0  ,  0
             0  , a21 ,  0  ,  0  , a22 ,  0  ,  0  , a23 ,  0  ,  0  , a24 ,  0
             0  ,  0  , a21 ,  0  ,  0  , a22 ,  0  ,  0  , a23 ,  0  ,  0  , a24
            a31 ,  0  ,  0  , a32 ,  0  ,  0  , a33 ,  0  ,  0  , a34 ,  0  ,  0
             0  , a31 ,  0  ,  0  , a32 ,  0  ,  0  , a33 ,  0  ,  0  , a34 ,  0
             0  ,  0  , a31 ,  0  ,  0  , a32 ,  0  ,  0  , a33 ,  0  ,  0  , a34
        ]
        """
    
    
class Add(Operator):
    """ Add operation of a list of matrix
    """

    def compute(self):
        self.value = np.mat(np.zeros(self.parents[0].shape()))
        for parent in self.parents:
            self.value += parent.value

    def get_jacobi(self, parent: 'Node'):
        return np.mat(np.eye(self.dimension()))