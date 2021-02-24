'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2021-02-08 17:15:16
 * @desc 
'''


import numpy as np
from abc import abstractmethod
from  numpy.matrixlib.defmatrix import matrix as npmat
from matrixslow.core.graph import default_graph

class Node(object):
    """Node in computing graph
    It is base class 
    """

    def __init__(self, *parents:'Node', **kwargs):
        """[summary]
        """ 
        self.graph = kwargs.get('graph', default_graph) 
        self.need_save = kwargs.get('need_save', True)
        self.generate_node_name(**kwargs)

        self.parents = list(parents)
        self.children = []
        self.value = None
        self.jacobi = None

        for parent in self.parents:
            parent.children.append(self)
        self.graph.add_node(self) 

    def forward(self):
        """Forward Propagation Value
        If some parent nodes' value is None, recursive to call parent nodes' forward method
        """        
        for node in self.parents:
            if node.value is None:
                node.forward()
        
        self.compute()

    def backward(self,result:'Node') -> npmat:
        """Backward Propagation Diff, Calculate Jacobi Matrix

        Parameters
        ----------
        result : [Node]
             Final result of the computing graph
        """
        if self.jacobi is None:
            if self is result:
                self.jacobi = np.mat(np.eye(self.dimension()))
            else:
                self.jacobi = np.mat(
                    np.zeros((result.dimension(), self.dimension())))
                #sum
                for child in self.get_children():
                    if child.value is not None:
                        self.jacobi += child.backward(result) * child.get_jacobi(self) 

        return self.jacobi

    @abstractmethod
    def compute(self):
        """Compute the operate of this step
        
        """
    
    @abstractmethod
    def get_jacobi(self,parent:'Node') -> npmat:
        """jacobi matrix for the parent node

        Parameters
        ----------
        parent : [Node]
            [description]

        Returns
        -------
        np.matrix
            return the jacobi matrix for certain parent
        """        
    
    def get_children(self):
        return self.children

    def get_parents(self):
        return self.parents
    
    def generate_node_name(self, **kwargs) -> None:
        """Generate node name
        """        
        self.name = kwargs.get('name', '{self.__class__.__name__}:{self.graph.node_count()}')
        if self.graph.name_scope:
            self.name = '{self.graph.name_scope} {self.name}'
    
    def clear_jacobi(self) -> None:
        """clear jacobi value
        """        
        self.jacobi = None
    
    def dimension(self) -> int:
        """[summary]

        Returns
        -------
        [int]
            the size of 2-D matrix
        """        
        return self.value.shape[0] * self.value.shape[1]
    
    def shape(self) -> tuple:
        """[summary]

        Returns
        -------
        tuple
            Tuple(x,y) x,y is the line and column of 2D matrix
        """          
        return self.value.shape

    def reset_value(self,recursive:bool = True) -> None:
        """[summary]

        Parameters
        ----------
        recursive : bool, optional
            Whether recursive to clear all children node, by default True
        """     
        self.value = None
        if recursive:
            for child in self.get_children():
                if child.value is not None:
                    child.reset_value()


    
class Variable(Node):
    """Variable Node in computing graph
    """

    def __init__(self, dim:int, init:bool = False, trainable:bool = True, **kwargs):
        super().__init__(**kwargs)
        # Node.__init__(self,**kwargs)
        self.dim = dim

        if init:
            self.value = np.mat(np.random.normal(0,0.001,self.dim))
        self.trainable = trainable

    def set_value(self, value:npmat):
        """Set the Varaible Value

        Parameters
        ----------
        value : numpy.matrixlib.defmatrix.matrix
            Variable Value
        """        
        assert value.shape == self.dim, "the shape is not equal to the setting"
        # shape is equal

        self.reset_value()
        self.value = value