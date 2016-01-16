import numpy as np
from random import shuffle
from math import log, exp

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[1]
  num_classes = W.shape[0]
  for i in xrange(num_train):
    scores = W.dot(X[:, i])
    scores -= np.amax(scores)
    correct_class_score = scores[y[i]]
    denom = np.sum(np.exp(scores))
    loss += -correct_class_score + log(denom)
    for j in xrange(num_classes):
        dW[j, :] += (exp(scores[j])/denom)*X[:, i].T
        if (j == y[i]):
          dW[j,:] -= X[:, i].T

  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)
  dW /= num_train
  dW += reg*W
    

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  num_train = X.shape[1]
  loss = 0.0
  dW = np.zeros_like(W)
  scores = W.dot(X) # C * N vector
  scores -= np.max(scores, axis = 0)
  true_scores = scores[y, np.arange(num_train)]  # N
  scores = np.exp(scores)
  loss_mat = np.sum(scores, axis = 0) # size = N
  loss = np.sum(-1*true_scores + np.log(loss_mat))/num_train
  loss += 0.5 * reg * np.sum(W*W)
  scores /= loss_mat
  scores[y, np.arange(num_train)] -= 1
  dW = np.dot(scores, X.T)
  dW /= num_train
  dW += reg* W
  

  return loss, dW
