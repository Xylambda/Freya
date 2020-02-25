import numpy as np

class Layer:
    """Layer abstract class"""
    def __init__(self):
        pass
    
    def __len__(self):
        pass
    
    def __str__(self):
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
    def optimize(self):
        pass


class Linear(Layer):
    """Linear.
    
    A linear layer. Equivalent to Dense in Keras and to torch.nn.Linear
    in torch.
    
    Parameters
    ----------
    input_dim : int
        Number of input features of this layer.
    output_dim : int
        Number of output features of this layer.
    
    """
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.rand(output_dim, input_dim)
        self.biases = np.random.rand(output_dim, 1)
        self.units = output_dim
        self.type = 'Linear'

    def _len_(self):
        return self.units

    def __str__(self):
        return f"{self.type} Layer"
        
    def forward(self, input_val):
        """Forward.
        
        Performs forward propagation of this layer.
        
        Parameters
        ----------
        input_val : numpy.Array
            Forward propagation of the previous layer.
            
        Returns
        -------
        activation : numpy.Array
            Forward propagation operation of the linear layer.
        
        """
        self._prev_acti = input_val
        return np.matmul(self.weights, input_val) + self.biases
    
    def backward(self, dA):
        """Backward.
        
        Performs backward propagation of this layer.
        
        Parameters
        ----------
        dA : numpy.Array
            Gradient of the next layer.
            
        Returns
        -------
        delta : numpy.Array
            Upcoming gradient, usually from an activation function.
        dW : numpy.Array
            Weights gradient of this layer.
        dB : numpy.Array
            Biases gradient of this layer.

        References
        ----------
        [1] Justin Johnson - Backpropagation for a Linear Layer:
        http://cs231n.stanford.edu/handouts/linear-backprop.pdf

        [2] Pedro Almagro Blanco - Algoritmo de Retropropagación:
        http://www.cs.us.es/~fsancho/ficheros/IAML/2016/Sesion04/
        capitulo_BP.pdf

        [3] Raúl Rojas - Neural Networks: A Systematic Introduction:
        https://page.mi.fu-berlin.de/rojas/neural/chapter/K7.pdf
        
        """
        dW = np.dot(dA, self._prev_acti.T)
        dB = dA.mean(axis=1, keepdims=True)
        
        delta = np.dot(self.weights.T, dA)
        
        return delta, dW, dB
    
    def optimize(self, dW, dB, rate):
        """Optimizes.
        
        Performs the optimization of the parameters. For now, 
        optimization can only be performed using gradient descent.
        
        Parameters
        ----------
        dW : numpy.Array
            Weights gradient.
        dB : numpy.Array
            Biases gradient.
        rate: float
            Learning rate of the gradient descent.
        
        """
        self.weights = self.weights - rate * dW
        self.biases = self.biases - rate * dB


class ReLU(Layer):
    """ReLU.
    
    Rectified linear unit layer.
    
    Parameters
    ----------
    output_dim : int
        Number of neurons in this layer.
    
    """
    
    def __init__(self, output_dim):
        self.units = output_dim
        self.type = 'ReLU'

    def _len_(self):
        return self.units        

    def __str__(self):
        return f"{self.type} Layer"        
        
    def forward(self, input_val):
        """Forward.
        
        Computes forward propagation pass of this layer.
        
        Parameters
        ----------
        input_val : numpy.Array
            Forward propagation of the previous layer.
        
        Returns
        -------
        _prev_acti : numpy.Array
            Forward propagation of this layer.
        
        """
        self._prev_acti = np.maximum(0, input_val)
        return self._prev_acti
    
    def backward(self, dJ):
        return dJ * np.heaviside(self._prev_acti, 0)


class Sigmoid(Layer):
    """Sigmoid.

    Sigmoid layer.

    Parameters
    ----------
    output_dim : int
        Number of neurons in this layers.
    
    """
    def __init__(self, output_dim):
        self.units = output_dim
        self.type = 'Sigmoid'

    def _len_(self):
        return self.units        

    def __str__(self):
        return f"{self.type} Layer"        
        
    def forward(self, input_val):
        """Forward.

        Computes forward propagation pass of this layer.

        Parameters
        ----------
        input_val : numpy.Array
            Forward propagation of the previous layer.
        
        Returns
        -------
        _prev_acti : numpy.Array
            Forward propagation of this layer.
        
        """
        self._prev_acti = 1 / (1 + np.exp(-input_val))
        return self._prev_acti
    
    def backward(self, dJ):
        """Backward.

        Computes backward propagation pass of this layer.

        Returns
        -------
        dJ : numpy.Array
            Gradient of this layer.

        """
        sig = self._prev_acti
        return dJ * sig * (1 - sig)


class Tanh(Layer):
    """Tanh.

    Hyperbolic tangent layer.

    Parameters
    ----------
    output_dim : int
        Number of neurons in this layers.

    References
    ----------
    [1] Wolfram Alpha - Hyperbolic tangent:
    http://mathworld.wolfram.com/HyperbolicTangent.html

    [2] Brendan O'Connor - tanh is a rescaled logistic sigmoid function:
    https://brenocon.com/blog/2013/10/tanh-is-a-rescaled-logistic-sigmoid-
    function/

    """
    def __init__(self, output_dim):
        self.units = output_dim
        self.type = 'Tanh'

    def _len_(self):
        return self.units        

    def __str__(self):
        return f"{self.type} Layer"        

    def forward(self, input_val):
        """Forward.

        Computes forward propagation pass of this layer.

        Parameters
        ----------
        input_val : numpy.Array
            Forward propagation of the previous layer.
        
        Returns
        -------
        _prev_acti : numpy.Array
            Forward propagation of this layer.
        
        """        
        self._prev_acti = np.tanh(input_val)
        return self._prev_acti

    def backward(self, dJ):
        """Backward.

        Computes backward propagation pass of this layer.

        Returns
        -------
        dJ : numpy.Array
            Gradient of this layer.

        """
        return dJ * (1 - np.square(self._prev_acti))


class Swish(Layer):
    """Swish.

    Swish layer. Swish is a self-gated activation function discovered by 
    researchers at Google.

    References
    ----------
    [1] Prajit Ramachandran, Barret Zoph, Quoc V. Le - Swish: A self-gated 
    activation function:
    https://arxiv.org/pdf/1710.05941v1.pdf

    """
    def __init__(self, output_dim):
        self.units = output_dim
        self.type = 'Swish'

    def _len_(self):
        return self.units        

    def __str__(self):
        return f"{self.type} Layer"

    def forward(self, input_val):
        """Forward.

        Computes forward propagation pass of this layer.

        Parameters
        ----------
        input_val : numpy.Array
            Forward propagation of the previous layer.
        
        Returns
        -------
        _prev_acti : numpy.Array
            Forward propagation of this layer.
        
        """        
        self._prev_acti = (1 / (1 + np.exp(-input_val))) * input_val
        return self._prev_acti

    def backward(self, dJ):
        """Backward.

        Computes backward propagation pass of this layer.

        Returns
        -------
        dJ : numpy.Array
            Gradient of this layer.

        """
        swish = self._prev_acti
        sig = 1 / (1 + np.exp(-swish))
        
        return dJ * (1 * sig + swish * (1 - swish))