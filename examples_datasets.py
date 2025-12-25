# library for working with arrays
import numpy as np 

# sigmoid function formula
def sigmoid(x): 
  return 1 / (1 + np.exp(-x)) 

# training model
# backpropagation method
def train(X, y, epochs=20000):
    # set the initial seed for the random number generator
    np.random.seed(1)
    # initialize random weights for the synapses
    weights = 2 * np.random.random((3,1)) - 1

    # training loop (by default 20 000 iterations)
    for _ in range(epochs):
        # compute the output using the sigmoid activation function
        output = sigmoid(np.dot(X, weights))
        
        # subtract the activation function output from the expected results
        err = y - output
        
        # calculate the weight adjustment using the formula
        weights += np.dot(X.T, err * (output * (1 - output)))
      
    return weights

# datsets
X1 = np.array([
    [1,0,0],
    [0,1,1],
    [1,1,1],
    [1,0,0]
])
y1 = np.array([[1,0,1,1]]).T

X2 = np.array([
    [0,0,1],
    [1,1,1],
    [1,0,0],
    [0,1,1]
])
y2 = np.array([[0,1,0,1]]).T

X3 = np.array([
    [0,0,0],
    [1,1,1],
    [1,1,0],
    [0,0,1]
])
y3 = np.array([[0,1,1,0]]).T

w1 = train(X1, y1)
w2 = train(X2, y2)
w3 = train(X3, y3)

print("Result Dataset 1:")
print(sigmoid(np.dot([0,0,0], w1)))

print("Result Dataset 2:")
print(sigmoid(np.dot([1,0,0], w2)))

print("Result Dataset 3:")
print(sigmoid(np.dot([1,0,1], w3)))

