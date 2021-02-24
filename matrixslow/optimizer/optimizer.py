'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2021-02-24 14:59:40
 * @desc 
'''

from typing import Dict,Optional
from matrixslow.core.graph import Graph
from matrixslow.core.node import Node, Variable
from abc import abstractmethod
from numpy.matrixlib.defmatrix import matrix as npmat


class Optimizer(object):
    """Base Abstract Optimizer
    """
    
    def __init__(self, graph:Graph, target:Node, learning_rate:float = 1e-4):
        super(Optimizer, self).__init__()
        
        self.graph = graph
        self.target = target

        self.learning_rate = learning_rate

        self.acc_gradient = {}
        # {node : gradient}
        self.acc_no = 0

    def one_step(self) -> None:
        """ forward propagation one time
        """
        self.forward_backward()
        self.acc_no += 1
        

    def get_gradient(self, node:Node) -> npmat:
        """Get Average Gradient

        Parameters
        ----------
        node : Node
            Given index node in computing graph

        Returns
        -------
        npmat
            average gradient for given node 
        """
        assert node in self.acc_gradient
        return self.acc_gradient[node] / self.acc_no
    

    def apply_gradients(self, node_gradient_dict:Dict, summarize:bool = False, acc_no:Optional[int] = None) -> None:
        """It is useless temporarily"""
        pass
    
    @abstractmethod
    def _update(self):
        """the implementation of optimizer
        """
    
    def update(self, var_gradients:Optional[Dict] = None):
        """Updating new value for variable node

        Parameters
        ----------
        var_gradients : Optional[Dict], optional
            Some variable graidient is given , using in distributed training, by default None
        """
        if var_gradients is not None:
            self.apply_gradients(var_gradients)
        
        self._update()

        self.acc_gradient.clear()
        self.acc_no = 0
    
    def forward_backward(self):
        """for given target node, forward propagating value
        and pass all varaible node, record their jacobi matrix
        """
        self.graph.clear_jacobi()

        self.target.forward()

        # pass all node in computing graph 
        # update every trainable node's gradient
        for node in self.graph.nodes:

            if isinstance(node, Variable) and node.trainable:
                node.backward(self.target)
                gradient = node.jacobi.T.reshape(node.shape())
                
                if node in self.acc_gradient:
                    self.acc_gradient[node] += gradient
                else:
                    self.acc_gradient[node] = gradient
