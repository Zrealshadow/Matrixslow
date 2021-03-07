from matrixslow.ops.ops import Sigmoid
from matrixslow.core.node import Node, Variable
from matrixslow.ops import Add,MatMul,ReLU

def fc(input:Variable, input_size:int, output_size:int, activation:str = None) -> Node:
    """Wrapper of a full connected layer

    Parameters
    ----------
    input : Variable
        input data
    output_size : int
        second layer size
    activation : str, optional
        activation function, by default None

    Raises
    ------
    KeyError
        the error key of activation function
    """
    weight = Variable((output_size, input_size), trainable=True, init=True)
    bias = Variable((output_size, 1), trainable=True, init=True)
    result = Add(MatMul(weight, input), bias)

    if activation == "ReLU":
        return ReLU(result)
    elif activation == "Sigmoid":
        return Sigmoid(result)
    elif activation == None:
        return result
    else:
        raise KeyError("No Specific Activation")

    