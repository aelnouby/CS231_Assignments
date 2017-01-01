import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in range(num_train):
      f = np.dot(X[i], W)
      f -= np.max(f)
      normalizer = np.sum(np.exp(f))
      p = np.exp(f[y[i]])/normalizer
      loss += -np.log(p)
      for k in range(num_classes):
          dW[:, k] += X[i] * (np.exp(f[k])/normalizer - (k == y[i]))


  dW /= num_train
  dW += reg * W
  loss /= num_train
  loss += 0.5 * reg * np.sum(W**2)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]

  scores = np.dot(X, W)
  f_i = np.max(scores, axis=1)
  scores -= np.matrix(f_i).T
  softmax = np.exp(scores[np.arange(num_train), y])/np.sum(np.exp(scores), axis=1)
  loss = -np.sum(np.log(softmax))
  loss /= num_train
  loss += 0.5 * reg* np.sum(W**2)

  sftmx = np.exp(scores)/np.matrix(np.sum(np.exp(scores), axis=1)).T
  mask = np.zeros(sftmx.shape)
  mask[np.arange(num_train), y] = 1
  dW = np.dot(X.T, sftmx - mask)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
