import numpy as np
from freya.layers import Layer

class MeanSquaredError(Layer):
    """Mean squared error.

    A layer to compute the mean squared error of the neural network. Freya
    assumes the loss layer is the last of the network; hence, it does not need
    the error of the following layer.

    Parameters
    ----------
    predicted : torch.Tensor
        A torch tensor containing the data of predicted values.
    real : torch.Tensor
        A torch tensor containing the data of real values (target).

    """
    def __init__(self, predicted, real):
        self.predicted = predicted
        self.real = real
        self.type = 'Mean Squared Error'
    
    def forward(self):
        """Forward pass.

        Computes forward propagation by using the Mean Squared Error formula.

        Returns
        -------
        loss : float
            A scalar value representing the loss
        
        """
        return np.power(self.predicted - self.real, 2).mean()

    def backward(self):
        """Backward pass.

        Computes backward propagation by using the Mean Squared Error 
        derivative formula.

        Freya assumes that the loss layer is the last layer of the net. Hence,
        it does not need the error of the following layer.

        Returns
        -------
        dJ : float
            A scalar value representing the loss gradient.

        """
        return 2 * (self.predicted - self.real).mean()


class BinaryCrossEntropy(Layer):
    """Binary Cross-Entropy.
    
    Binary Cross-Entropy cost function layer. Freya assumes the loss layer is 
    the last of the network; hence, it does not need the error of the following 
    layer.
    
    Note that we want to compare probabilities with labels.
    
    Parameters
    ----------
    real : numpy.Array
        Real Y labels.
    predicted : numpy.Array
        Predicted probabilities.
    
    """
    def __init__(self, predicted, real):
        self.real = real
        self.predicted = predicted
        self.type = 'Binary Cross-Entropy'
    
    def forward(self):
        """Forward.
        
        Computes the forward pass of BinaryCrossEntropy layer.
        
        Returns
        -------
        loss : float
            A scalar value representing the loss
        
        """
        n = len(self.real)
        loss = np.nansum(-self.real * np.log(self.predicted) - (1 - self.real) * np.log(1 - self.predicted)) / n
        
        return np.squeeze(loss)
    
    def backward(self):
        """Backward
        
        Computes the backward pass of BinaryCrossEntropy layer.
        
        Returns
        -------
        dJ : float
            A scalar value representing the loss gradient.
        
        """
        n = len(self.real)
        return (-(self.real / self.predicted) + ((1 - self.real) / (1 - self.predicted))) / n


