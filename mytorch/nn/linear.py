import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch
        """
        # TODO: Implement forward pass
        Z = A @ self.W.T + self.b
        
        # Store input for backward pass
        self.A = A
        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # TODO: Implement backward pass

        input_shape = self.A.shape
        batch_size = np.prod(dLdZ.shape[:-1])
        dLdZ_2d = dLdZ.reshape(batch_size, -1)
        A = self.A.reshape(batch_size, -1)

        # Compute gradients
        self.dLdA = dLdZ_2d @ self.W
        self.dLdW = dLdZ_2d.T @ A
        self.dLdb = np.sum(dLdZ_2d, axis=0)

        # Reshape gradient wrt input to original shape
        dLdA_reshaped = self.dLdA.reshape(input_shape)
        
        return dLdA_reshaped
