import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # TODO: Implement forward pass
        # Compute the softmax in a numerically stable way
        # Apply it to the dimension specified by the `dim` parameter
        self.A = np.exp(Z)
        self.A /= np.sum(self.A, axis=self.dim, keepdims=True)
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass
        
        # Get the shape of the input
        shape = self.A.shape
        
        # Find the dimension along which softmax was applied
        C = shape[self.dim]

        # Reshape input to 2D
        if len(shape) > 2:
            self.A = np.moveaxis(self.A, self.dim, -1)
            dLdA = np.moveaxis(dLdA, self.dim, -1)
            self.A = self.A.reshape(-1, C)
            dLdA = dLdA.reshape(-1, C)
        
        dLdZ = np.zeros_like(self.A)
        for i in range(dLdA.shape[0]):
            J = np.zeros((C, C))

            for m in range(C):
                for n in range(C):
                    J[m, n] = np.where(m==n, self.A[i, m]*(1-self.A[i, n]), -self.A[i, m]*self.A[i, n])
            dLdZ[i, :] = np.dot(dLdA[i, :], J)

        # Reshape back to original dimensions if necessary
        if len(shape) > 2:
            # Restore shapes to original
            self.A = self.A.reshape(shape)
            dLdZ = dLdZ.reshape(shape)
            
        return dLdZ
    