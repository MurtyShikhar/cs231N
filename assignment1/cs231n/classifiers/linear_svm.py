import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  for i in xrange(num_train):
    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[j, :] += X[:, i].T   
        dW[y[i], :] -= X[:, i].T

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  
  dW += reg* W



  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  num_train = X.shape[1]
  dW = np.zeros(W.shape) # initialize the gradient as zero
  scores = W.dot(X) # C * N vector 
  true_scores = scores[y, np.arange(num_train)] # true_scores[i] contains
  mat = scores - true_scores + 1
  mat[y, np.arange(num_train)] = 0
  mat = np.maximum(np.zeros(scores.shape), mat)
  loss = np.sum(mat)/num_train
  loss += 0.5 * reg * np.sum(W*W)

  
  bin = mat
  bin[mat > 0] = 1 # bin is of size C * N
  col_sum = np.sum(bin, axis = 0) # mat is of size 1 * N the number of times > 0 for a data point for all classes
  bin[y, np.arange(num_train)] = -col_sum[np.arange(num_train)]
  dW = np.dot(bin, X.T)  # dW is a C * D
  dW /= num_train
  
  dW += reg* W



  return loss, dW
