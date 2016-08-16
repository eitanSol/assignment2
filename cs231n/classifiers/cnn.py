import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #

    ############################################################################
    C, H, W = input_dim
    size_after_pull = (H/2) * (W/2)* num_filters
    self.params = {
      'W1': weight_scale * np.random.randn(num_filters, C, filter_size, filter_size),
      'W2': weight_scale * np.random.randn(size_after_pull, hidden_dim),
      'W3': weight_scale * np.random.randn(hidden_dim, num_classes),
      'b1': np.zeros(num_filters), 
      'b2': np.zeros(hidden_dim),
      'b3': np.zeros(num_classes),
    }
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    x1_conv, conv1_cache = conv_forward_fast(X, W1, b1, conv_param)
    x1_relu, relu1_cache = relu_forward(x1_conv)
    x1_pull, pull1_cache = max_pool_forward_fast(x1_relu, pool_param)
    x2_affi, affi2_cache = affine_forward(x1_pull, W2, b2)
    x2_relu, relu2_cache = relu_forward(x2_affi)
    x3_affi, affi3_cache = affine_forward(x2_relu, W3, b3)
    scores = x3_affi

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dout = softmax_loss(scores, y) 
    dx3_affi = dout
    dx2_relu, dW3, db3 = affine_backward(dx3_affi, affi3_cache)
    dx2_affi = relu_backward(dx2_relu, relu2_cache)
    dx1_pull, dW2, db2 = affine_backward(dx2_affi, affi2_cache)
    dx1_relu = max_pool_backward_fast(dx1_pull, pull1_cache)
    dx1_conv = relu_backward(dx1_relu, relu1_cache)
    dx, dW1, db1 = conv_backward_fast(dx1_conv, conv1_cache)

    # loss += 0.5 * self.reg * np.sum(W1**2 + W2**2 + W3**2)
    grads = {
    "W1": dW1 + self.reg * self.params['W1'],
    "W2": dW2 + self.reg * self.params['W2'],
    "W3": dW3 + self.reg * self.params['W3'],
    "b1": db1,
    "b2": db2,
    "b3": db3
    }


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
