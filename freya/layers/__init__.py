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
        dZ : numpy.Array
            Gradient of this layer.

        """
        sig = self._prev_acti
        return dJ * sig * (1 - sig)


class Tanh(Layer):
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

    def backward(self, dJ):
        """Backward.

        Computes backward propagation pass of this layer.

        Returns
        -------
        dZ : numpy.Array
            Gradient of this layer.

        """
        return dJ * (1 - np.square(self._prev_acti))
