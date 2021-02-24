'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2021-02-24 14:59:40
 * @desc 
'''

from typing import Dict,Optional

from numpy.lib.function_base import gradient
from matrixslow.core.graph import Graph
from matrixslow.core.node import Node, Variable
from abc import abstractmethod
from numpy.matrixlib.defmatrix import matrix as npmat
import numpy as np

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



class SGD(Optimizer):
    """Simple Gradient Descent Optimizer
    """

    def __init__(self, graph: Graph, target: Node, learning_rate: float):
        super().__init__(graph, target, learning_rate=learning_rate)
    
    def _update(self):
        """
        g = jacobi(node)
        v = - rate * g
        w_{new} = w_{old} + v
        """
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                gradient = self.get_gradient(node)
                v = - self.learning_rate * gradient
                node.set_value(node.value + v)
        

class MomentumGD(Optimizer):
    """Add momentum based on simple gradient descent optimizer
    """

    def __init__(self, graph: Graph, target: Node, learning_rate: float, momentum: float = 0.9):
        super().__init__(graph, target, learning_rate=learning_rate)
        self.beta = momentum
        assert momentum < 1.0 and momentum > 0.0
        self.vlist = []

    def _update(self):
        """
        g = jacobi(node)
        v_{k+1} = beta * v_{k} - rate * g
        w_{new} = w_{old} + v
        """
        
        for node in self.graph.nodes:

            if isinstance(node, Variable) and node.trainable:
                gradient = self.get_gradient(node)
                
                if node not in self.v:
                    self.v[node] = - self.learning_rate * gradient
                else:
                    self.v[node] = self.beta * self.v[node] - self.learning_rate * gradient
                
                node.set_value(node.value + self.v[node])


class AdaGD(Optimizer):
    """AdaGrad
    """

    def __init__(self, graph: Graph, target: Node, learning_rate: float):
        super().__init__(graph, target, learning_rate=learning_rate)
        self.e = 1e-10
        self.s = {}
    
    def _update(self):
        """
        g = gradient
        s_new = s_old + np.multiply(gradient, gradient)
        v = - learning_rate / sqrt(s_new + e) * gradient
        w_new = w_old + v
        """
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                gradient = self.get_gradient
                #np matrix

                if node not in self.s:
                    self.s[node] = np.power(gradient, 2)
                else:
                    self.s[node] = self.s[node] + np.power(gradient, 2)
                
                rate = self.learning_rate / np.sqrt(self.s[node] + self.e) 
                # np.mat
                v = - np.multiply(rate, gradient)
                node.set_value(node.value + v)


class RMSProp(Optimizer):
    """
    slide window in second derivative of gradient
    """

    def __init__(self, graph: Graph, target: Node, learning_rate: float, beta:float = 0.9):
        super().__init__(graph, target, learning_rate=learning_rate)
        self.beta = beta
        assert beta < 1.0 and beta > 0
        self.s = {}
        self.e = 1e-10
        
    def _update(self):
        """
        g = gradient
        s_new = beta * s_old + (1 - beta) * np.multiply(gradient, gradient)
        v = - learning_rate / sqrt(s_new + e) * gradient
        w_new = w_old + v
        """
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                gradient =  self.get_gradient(node)

                if node not in self.s:
                    self.s[node] = np.power(gradient, 2)
                else:
                    self.s[node] = self.s[node] * self.beta + (1 - self.beta) * np.power(gradient, 2)
                
                v = - self.learning_rate * gradient * np.sqrt(self.s[node] + self.e)
               
                node.set_value(node.value + v)

class Adam(Optimizer):
    """
    Adaptive Moment Estimation
    """

    def __init__(self, graph: Graph, target: Node, learning_rate: float, beta1:float = 0.9, beta2:float = 0.99):
        super().__init__(graph, target, learning_rate=learning_rate)

        assert beta1 < 1.0  and beta1 > 0.0 and beta2 < 1.0 and beta2 > 0.0

        self.beta1 = beta1
        self.beta2 = beta2
        self.e = 1e-10

        self.v = {}
        self.s = {}

    def _update(self):
        """
        g = gradient
        v_new = beta1 * v_old + (1 - beta1) * gradient
        s_new = beta2 * s_old + (1 - beta2) * np.multiply(gradient, gradient)
        tmp = - learning_rate / sqrt(s_new + e) * v_new
        w_new = w_old + tmp
        """
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:

                gradient =  self.get_gradient(node)

                if node not in self.s:
                    self.s[node] = np.power(gradient, 2)
                    self.v[node] = gradient
                else:
                    self.v[node] = self.beta1 * self.v[node] + (1 - self.beta1) * gradient
                    self.s[node] = self.beta2 * self.s[node] + (1 - self.beta2) * np.power(gradient, 2)
                
                tmp = - self.learning_rate * self.v[node] / np.sqrt(self.s[node] + self.e)

                node.set_value(node.value + tmp)
